"""
관리자 에이전트 Tool Selector 체인.

설계서: docs/관리자_AI에이전트_설계서.md §3.2 (tool_selector), §10 (프롬프트)

역할:
- Intent 가 stats/query/action 등으로 분류된 뒤, admin_role 에 허용된 Tool 중 **가장 적합한
  하나** 를 Solar Pro 의 `bind_tools()` 로 선택받는다.
- Step 2 범위: Tool RAG 도입 전이라 admin_role 필터된 전체 tool (최대 8~10개) 을 프롬프트에
  주입. 이후 Step 3 에서 Qdrant 기반 top-5 retrieval 로 확장될 예정.

반환:
- `SelectedTool(name, arguments, rationale)` — LLM 이 tool_call 을 내뱉은 경우
- `None` — 적절한 tool 이 없거나 호출 실패. 상위에서 narrator 가 "적절한 도구 없음" 안내.

주의:
- LangChain `bind_tools()` 는 `BaseTool` 인스턴스를 원한다. 레지스트리의 ToolSpec 을
  `StructuredTool.from_function` 으로 감싸 넘긴다. `coroutine` 은 no-op 더미를 준다 — 실제
  실행은 `tool_executor` 노드의 레지스트리 조회로 수행한다 (LLM 이 직접 실행하지 않음).
- `chain = prompt | llm` 대신 prompt 를 미리 messages 로 렌더한 뒤 llm.ainvoke 에 직접
  넘긴다. MagicMock 호환 (question_chain/admin_intent_chain 과 동일 이유).
"""

from __future__ import annotations

import time
import traceback
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from monglepick.llm.factory import get_solar_api_llm, guarded_ainvoke
from monglepick.tools.admin_tools import ToolSpec, list_tools_for_role

logger = structlog.get_logger()


# ============================================================
# 가상 tool — finish_task
# ============================================================
# LLM 이 "더 이상 tool 이 필요 없다, 수집된 데이터로 답변 가능" 이라고 판단할 때 호출한다.
# 실제 Backend 호출이 없는 시그널 전용 tool.  observation 노드가 이 이름을 감지하면
# narrator 로 직행한다 (설계서 §4.4).

class FinishTaskArgs(BaseModel):
    """finish_task 가상 tool 인자 스키마."""

    reason: str = Field(
        default="",
        description="왜 더 이상 tool 이 필요 없는지 한 줄 설명 (디버깅/로깅용).",
    )


# ============================================================
# 반환 모델
# ============================================================

class SelectedTool(BaseModel):
    """tool_selector 가 LLM 응답에서 추출한 단일 tool-call."""

    name: str = Field(..., description="선택된 tool 의 레지스트리 이름")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="LLM 이 제안한 인자 (args_schema 검증 전)",
    )
    rationale: str = Field(default="", description="선택 이유 (디버깅용, 옵셔널)")


# ============================================================
# 프롬프트
# ============================================================

_SELECTOR_SYSTEM_PROMPT = """당신은 몽글픽 관리자 데이터 조회와 안내를 돕는 어시스턴트예요.

**중요한 제약**:
- 당신은 데이터를 **생성·수정·삭제할 수 없어요.**
- 생성/수정/삭제가 필요한 요청은 `*_draft` 도구로 폼 내용을 채워주거나,
  `goto_*` 도구로 해당 관리 화면으로 안내해 주세요.
- 금전·회원 제재 같은 위험한 요청은 반드시 `goto_*` 로 화면 링크만 제공해요.
  실제 실행은 관리자가 직접 버튼을 눌러야 해요.

**여러 단계가 필요하면**:
- 먼저 필요한 데이터를 read 도구로 수집하세요.
- 정보가 충분하면 `*_draft` 또는 `goto_*` 로 종결하세요.
- 더 이상 도구가 필요 없으면 `finish_task` 를 호출해 자연어로 답변하세요.

**보고서/요약(report) 의도일 때**:
- "최근 ~ 요약", "주간 운영 리포트", "이번 달 ~ 보고서" 같은 발화는 보통 여러 도구의
  결과를 **종합** 해야 답할 수 있어요.
- 흐름 예시:
  1) 관련 통계 도구(예: `stats_community_overview`) 1회
  2) 도메인 목록 조회 도구(예: `reports_list`, `posts_list`) 1~2회 — 상위 N건/대기 건수
  3) 더 깊이 보고 싶은 항목이 있으면 `goto_report_detail` 또는 `goto_*` 로 화면 이동을 제안
  4) 충분하다고 판단되면 `finish_task` 로 종결 — narrator 가 섹션별로 정리해서 답해요.
- read 도구의 페이지 크기는 기본값(`size=20`) 그대로 두세요. 표는 클라이언트가 알아서 잘라요.
- 같은 도구를 반복 호출하지 말고, 수치가 부족하면 **다른 카테고리** 도구로 보강하세요.

# 적극적 매칭 규칙
1. 발화 어휘가 tool 설명과 **정확히 일치하지 않아도, 의미가 겹치면 매칭한다**.
   - 예: "환불된 결제 보여줘" → `orders_list` 선택 + `status="REFUNDED"` 주입.
   - 예: "AI 잘 쓰이고 있어?" → `stats_ai_service_overview` 선택.
   - 예: "커뮤니티 신고 건수 많아?" → `stats_community_overview` 또는 `reports_list`.
   - 예: "가장 많이 정지당한 유저" → `users_list(status="SUSPENDED")`.
   - 예: "chulsoo 환불해줘" → `users_list` 로 먼저 사용자 검색, 이후 `goto_order_refund`.
   - 예: "공지 등록해줘" → `notice_draft` 로 폼 채움 (Backend 직접 호출 X).
   - 질문이 모호하면 일단 **가장 근접한 tool 을 골라 호출**하고, 결과를 돌려준 뒤 관리자에게
     더 구체화를 유도하는 것이 "적합한 도구 없음" 이라고 포기하는 것보다 낫다.

# Tool 호출이 정말 어려운 경우만 None
2. 다음 경우에만 tool 을 호출하지 말고 빈 응답을 준다:
   - 질문이 관리 업무와 전혀 관련 없음 (예: "오늘 날씨 어때?")
   - 관리자 권한 {admin_role} 에 허용된 tool 목록에 의미상 근접한 후보가 전혀 없음
   - 이 경우에는 상위 노드(smart_fallback_responder) 가 관리자에게 "이런 표현으로 바꾸면
     답할 수 있어요" 라는 역제안을 생성한다 — 여기서는 도구 호출만 생략하면 된다.

# 인자 채우기
3. 발화에 명시된 값만 채우고, 나머지는 **tool 기본값** 에 맡긴다 (예: `period="7d"`,
   `page=0`, `size=20`).
4. 빈 문자열 `""` 이 기본값인 필터(keyword/status/category 등)는 발화에 명시되지 않으면
   그대로 `""` 로 두어 전체 조회를 유도한다.

# 기타
5. 여러 tool 이 비슷하게 맞을 때는 **더 구체적인 쪽** 을 선택한다
   (예: `orders_list` vs `stats_revenue` 중 "환불 주문 목록" 은 orders_list).
6. 발화가 여러 작업을 요구해도 **가장 핵심인 하나** 만 호출한다.
7. 수치를 만들어내지 않는다. tool 호출만 수행하고 결과 해석은 이 체인의 역할이 아니다.

이전 호출 결과 요약: {tool_history_summary}
현재 hop: {hop_count} / {max_hops}
"""


_SELECTOR_HUMAN_PROMPT = """관리자 역할: {admin_role}
분류된 의도: {intent_kind}
발화: {user_message}

위 발화에 가장 적합한 tool 을 정확히 하나 호출하라. 적절한 tool 이 없으면 호출하지 말 것.
더 이상 tool 이 필요 없다고 판단되면 finish_task 를 호출하라.
"""


_selector_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SELECTOR_SYSTEM_PROMPT),
        ("human", _SELECTOR_HUMAN_PROMPT),
    ]
)

# 프롬프트 변수 목록:
#   system: admin_role, tool_history_summary, hop_count, max_hops
#   human:  admin_role, intent_kind, user_message


# ============================================================
# ToolSpec → LangChain StructuredTool 변환
# ============================================================

async def _selector_noop(**kwargs: Any) -> str:
    """
    LLM 이 tool_call 을 내뱉기만 하면 되므로 실제 실행은 이 no-op 이 받아 폐기한다.

    tool_executor 노드가 ADMIN_TOOL_REGISTRY 에서 원본 handler 를 다시 조회해 실행한다.
    이 분리 덕분에 LLM 이 직접 Backend 를 호출하지 않고, 실행은 에이전트 본체에 남는다.
    """
    return ""


def _to_structured_tool(spec: ToolSpec) -> StructuredTool:
    """
    ToolSpec → LangChain StructuredTool 변환.

    - name / description: ToolSpec 값 그대로
    - args_schema: ToolSpec.args_schema (Pydantic BaseModel)
    - coroutine: 더미 no-op (LLM 은 선택만 하고 실제 실행은 tool_executor 가 한다)
    """
    return StructuredTool.from_function(
        name=spec.name,
        description=spec.description,
        args_schema=spec.args_schema,
        coroutine=_selector_noop,
    )


# ============================================================
# 핵심 함수
# ============================================================

async def select_admin_tool(
    user_message: str,
    admin_role: str,
    intent_kind: str,
    request_id: str = "",
    tool_history_summary: str = "",
    hop_count: int = 0,
    max_hops: int = 5,
    allowed_tool_names: list[str] | None = None,
) -> SelectedTool | None:
    """
    관리자 발화 + intent 를 보고 최적 Tool 하나를 선택한다 (v3 Phase D).

    v3 변경점:
    - bind_tools 목록에 가상 tool `finish_task` 추가. LLM 이 "더 이상 tool 불필요"라고
      판단하면 이 이름으로 tool_call 을 내뱉는다. observation 노드가 감지해 narrator 로 직행.
    - tool_history_summary / hop_count / max_hops 를 프롬프트에 주입해 ReAct 문맥 제공.
    - finish_task 는 실제 레지스트리에 등록되지 않으므로 허용 목록 재검증에서 예외 처리.

    Step 7a 변경점 (Tool RAG):
    - allowed_tool_names: Tool RAG 가 선별한 top-k tool 이름 목록.
      None (기본): 기존처럼 list_tools_for_role(admin_role) 전체를 bind → 하위호환.
      list 전달: role 필터 결과와 교집합을 취해 그 tool 만 bind.
      finish_task 가상 tool 은 allowed_tool_names 결과와 무관하게 항상 bind 된다.

    Args:
        user_message: 관리자 자연어 입력
        admin_role: 정규화된 AdminRoleEnum — 레지스트리 필터 기준
        intent_kind: AdminIntent.kind (query/action/stats/report/sql/smalltalk)
        request_id: 동시성 슬롯 / 로그 식별자
        tool_history_summary: 이전 hop 결과 축약 문자열 (observation 노드가 생성)
        hop_count: 현재까지 실행된 hop 수 (0부터 시작)
        max_hops: 최대 허용 hop 수 (기본 5)
        allowed_tool_names: Tool RAG top-k 이름 목록. None 이면 전체 role 필터 사용.

    Returns:
        SelectedTool: LLM 이 tool 을 선택한 경우 (finish_task 포함)
        None: tool 없음 / 권한 없음 / LLM 에러 (어떤 경우든 graceful)
    """
    start = time.perf_counter()

    # 1) admin_role 필터 — 하나도 허용 없으면 조기 반환
    allowed = list_tools_for_role(admin_role)
    if not allowed:
        logger.info(
            "admin_tool_selector_no_allowed_tools",
            admin_role=admin_role or "(blank)",
        )
        return None

    # 1-b) Tool RAG top-k 필터 적용 (allowed_tool_names 가 주어진 경우)
    #
    # allowed_tool_names 가 None 이면 role 필터된 전체 tool 을 bind (기존 동작, 하위호환).
    # list 가 주어지면 "role 허용 ∩ RAG top-k" 교집합만 bind 한다.
    #
    # 교집합이 비어있는 경우는 Role 불일치(RAG 가 다른 role tool 을 반환한 경우) 이므로
    # role 필터 원본(allowed) 을 그대로 사용해 안전하게 폴백한다.
    if allowed_tool_names is not None:
        rag_name_set = set(allowed_tool_names)
        filtered = [s for s in allowed if s.name in rag_name_set]
        if filtered:
            allowed = filtered
            logger.info(
                "admin_tool_selector_rag_filtered",
                before=len(list_tools_for_role(admin_role)),
                after=len(allowed),
                top_k_names=allowed_tool_names[:10],  # 로그는 최대 10개만
            )
        else:
            # 교집합 빈 경우 — role 필터 전체로 안전 폴백
            logger.warning(
                "admin_tool_selector_rag_empty_intersection_fallback",
                rag_names=allowed_tool_names,
                admin_role=admin_role,
            )

    # 2) Solar API LLM + bind_tools (실제 tool + finish_task 가상 tool)
    try:
        llm = get_solar_api_llm(temperature=0.1)
        tools = [_to_structured_tool(s) for s in allowed]

        # finish_task 가상 tool — LLM 이 "이상 tool 불필요" 시 호출하는 종결 시그널.
        # 실제 handler 없음. observation 노드가 이 이름을 감지해 narrator 로 보낸다.
        finish_tool = StructuredTool.from_function(
            name="finish_task",
            description=(
                "더 이상 도구 호출이 필요 없을 때 선택하세요. "
                "수집된 데이터만으로 관리자 질문에 충분히 답변할 수 있을 때 사용합니다."
            ),
            args_schema=FinishTaskArgs,
            coroutine=_selector_noop,
        )
        tools_with_finish = tools + [finish_tool]

        # tool_choice="auto": LLM 이 자율 판단 (tool 없으면 text 반환 가능)
        llm_with_tools = llm.bind_tools(tools_with_finish, tool_choice="auto")

        # 이전 hop 요약이 없으면 "없음" 표기
        history_summary = tool_history_summary.strip() or "없음 (첫 번째 hop)"

        # `to_messages()` 대신 ChatPromptValue 자체를 넘긴다 — admin_intent_chain.py 와 동일.
        prompt_value = _selector_prompt.format_prompt(
            user_message=user_message.strip(),
            admin_role=admin_role or "UNKNOWN",
            intent_kind=intent_kind or "unknown",
            tool_history_summary=history_summary,
            hop_count=hop_count,
            max_hops=max_hops,
        )
        response = await guarded_ainvoke(
            llm_with_tools,
            prompt_value,
            model="solar_api",
            request_id=request_id or "admin_tool_selector",
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "admin_tool_selector_failed",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )
        return None

    # 3) tool_calls 추출 — LangChain 0.3 규약: response.tool_calls 는 list[dict]
    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        elapsed_ms = (time.perf_counter() - start) * 1000
        content_preview = (getattr(response, "content", "") or "")[:120]
        logger.info(
            "admin_tool_selector_no_tool_call",
            content_preview=content_preview,
            elapsed_ms=round(elapsed_ms, 1),
        )
        return None

    first = tool_calls[0]
    name = first.get("name") or ""
    arguments = first.get("args") or {}

    # 4) 허용 tool 집합 내부인지 재검증 (프롬프트 지시 위반 방어).
    #    finish_task 는 레지스트리 미등록 가상 tool 이므로 예외 허용.
    allowed_names = {s.name for s in allowed} | {"finish_task"}
    if name not in allowed_names:
        logger.warning(
            "admin_tool_selector_disallowed_tool_rejected",
            tool_name=name,
            allowed=sorted(allowed_names),
        )
        return None

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "admin_tool_selected",
        tool_name=name,
        args_keys=sorted(arguments.keys()),
        hop_count=hop_count,
        elapsed_ms=round(elapsed_ms, 1),
    )
    # RAG 필터 적용 여부를 rationale 에 기록 (디버깅용)
    rag_note = f" rag_top_k={len(allowed_tool_names)}" if allowed_tool_names is not None else ""
    return SelectedTool(
        name=name,
        arguments=arguments,
        rationale=f"bind_tools[{len(allowed)}+finish_task] auto-selected hop={hop_count}{rag_note}",
    )
