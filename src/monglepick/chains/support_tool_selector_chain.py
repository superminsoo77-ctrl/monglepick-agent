"""
고객센터 AI 에이전트 v4 Tool Selector 체인.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §3.1 (tool_selector 확장)

역할:
- SupportIntent 와 이전 hop 이력을 바탕으로 Solar Pro 의 `bind_tools()` 로
  SUPPORT_TOOL_REGISTRY 중 최적 tool 하나를 선택한다.
- 더 이상 tool 이 필요 없으면 가상 `finish_task` tool 을 선택해 narrator 로 직행.
- 게스트(is_guest=True) 는 requires_login=True tool 바인드에서 제외.

참조 패턴:
- `chains/admin_tool_selector_chain.py` — Solar bind_tools + tool_call_history 압축 + finish_task

반환:
- `SupportSelectedTool(name, arguments, rationale)` — LLM 이 tool_call 을 내뱉은 경우
- `None` — 적절한 tool 이 없거나 호출 실패 (graceful fallback).

주의:
- LangChain `bind_tools()` 는 `StructuredTool` 인스턴스를 원한다.
  `SupportToolSpec` 을 `StructuredTool.from_function` 으로 감싸 넘기며,
  coroutine 은 no-op 더미를 사용한다 — 실제 실행은 `tool_executor` 노드가 한다.
- `finish_task` 는 레지스트리 미등록 가상 tool. observation 노드가 이름을 감지해 narrator 로 직행.
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
from monglepick.tools.support_tools import SUPPORT_TOOL_REGISTRY, SupportToolSpec

logger = structlog.get_logger()


# ============================================================
# 가상 tool — finish_task
# ============================================================
# LLM 이 "더 이상 tool 호출이 필요 없다" 고 판단할 때 이 이름으로 tool_call 을 내뱉는다.
# 실제 Backend 호출 없는 시그널 전용 tool.
# observation 노드가 이 이름을 감지하면 narrator 로 직행한다.

class FinishTaskArgs(BaseModel):
    """finish_task 가상 tool 인자 스키마."""

    reason: str = Field(
        default="",
        description=(
            "더 이상 tool 이 필요 없는 이유 (디버깅/로깅용). "
            "예: '조회 완료 — AI 쿼터와 구독 정보가 충분히 수집됐음'"
        ),
    )


# ============================================================
# 반환 모델
# ============================================================

class SupportSelectedTool(BaseModel):
    """support_tool_selector 가 LLM 응답에서 추출한 단일 tool-call."""

    name: str = Field(..., description="선택된 tool 의 레지스트리 이름 (또는 'finish_task')")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="LLM 이 제안한 인자 (args_schema 검증 전)",
    )
    rationale: str = Field(default="", description="선택 이유 (디버깅용, 옵션)")


# ============================================================
# 프롬프트
# ============================================================

from monglepick.prompts.support_assistant import (
    SUPPORT_TOOL_SELECTOR_SYSTEM_PROMPT,
    SUPPORT_TOOL_SELECTOR_HUMAN_PROMPT,
)

_selector_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SUPPORT_TOOL_SELECTOR_SYSTEM_PROMPT),
        ("human", SUPPORT_TOOL_SELECTOR_HUMAN_PROMPT),
    ]
)

# 프롬프트 변수:
#   system: tool_history_summary, hop_count, max_hops
#   human:  user_message, intent_kind, intent_confidence, is_guest


# ============================================================
# ToolSpec → LangChain StructuredTool 변환
# ============================================================

async def _noop(**kwargs: Any) -> str:
    """
    no-op 더미 coroutine.

    LLM 이 tool_call 을 내뱉기만 하면 되므로 실제 실행은 tool_executor 노드가
    SUPPORT_TOOL_REGISTRY 에서 원본 handler 를 조회해 처리한다.
    이 분리 덕분에 LLM 이 직접 Backend 를 호출하지 않는다.
    """
    return ""


def _to_structured_tool(spec: SupportToolSpec) -> StructuredTool:
    """
    SupportToolSpec → LangChain StructuredTool 변환.

    - name / description: SupportToolSpec 값 그대로
    - args_schema: SupportToolSpec.args_schema (Pydantic BaseModel)
    - coroutine: no-op 더미 (선택만 하고 실행은 tool_executor 가 한다)
    """
    return StructuredTool.from_function(
        name=spec.name,
        description=spec.description,
        args_schema=spec.args_schema,
        coroutine=_noop,
    )


# ============================================================
# 압축 유틸 — tool_results_cache → 요약 문자열
# ============================================================

def compact_tool_results_summary(
    tool_call_history: list[dict],
    tool_results_cache: dict,
) -> str:
    """
    tool_call_history 와 tool_results_cache 를 token-saving 형태로 압축해 문자열로 반환한다.

    LLM 에 매 hop 마다 full result 를 넣으면 컨텍스트 비대화가 발생하므로,
    tool_name / ok / row_count(있으면) / 30자 요약만 추출한다.

    형식:
        1. lookup_my_ai_quota → ok=True (dailyAiUsed=2, dailyAiLimit=3)
        2. lookup_policy → ok=True (3개 정책 청크)

    Args:
        tool_call_history: observation 노드가 append 한 이력 리스트.
            각 항목: {"hop": int, "tool_name": str, "ok": bool, "error": str|None}
        tool_results_cache: tool_executor 가 저장한 ref_id → result dict.

    Returns:
        압축된 요약 문자열. 이력이 없으면 "없음 (첫 번째 hop)".
    """
    if not tool_call_history:
        return "없음 (첫 번째 hop)"

    lines: list[str] = []
    for idx, entry in enumerate(tool_call_history, start=1):
        tool_name = entry.get("tool_name", "?")
        ok = entry.get("ok", False)
        hop = entry.get("hop", idx)

        # tool_results_cache 에서 ref_id 매핑 (tool_name_{hop-1} 인덱싱)
        ref_id = f"{tool_name}_{hop - 1}"
        result = tool_results_cache.get(ref_id, {})

        # 30자 이내 요약 생성
        summary = _summarize_result(tool_name, result)
        ok_str = "ok=True" if ok else f"ok=False"
        lines.append(f"{idx}. {tool_name} → {ok_str} ({summary})")

    return "\n".join(lines)


def _summarize_result(tool_name: str, result: dict) -> str:
    """
    tool 결과 dict 에서 30자 이내의 핵심 요약을 추출한다.

    tool_name 에 따라 특화된 요약을 생성한다:
    - lookup_my_ai_quota: dailyAiUsed/dailyAiLimit 쌍
    - lookup_my_point_history: N건 이력
    - lookup_policy: N개 청크
    - 기타: row_count 또는 data 존재 여부
    """
    if not result.get("ok"):
        error = result.get("error", "unknown_error")
        return f"error={error}"[:30]

    data = result.get("data", {})
    if not data:
        return "data={}"

    # tool 별 특화 요약
    if tool_name == "lookup_my_ai_quota":
        daily_used = data.get("dailyAiUsed", "?")
        daily_limit = data.get("dailyAiLimit", "?")
        bonus = data.get("remainingAiBonus", "?")
        return f"used={daily_used}/{daily_limit} bonus={bonus}"[:40]

    if tool_name == "lookup_my_point_history":
        items = data if isinstance(data, list) else data.get("items", [])
        count = len(items) if isinstance(items, list) else "?"
        return f"{count}건 이력"

    if tool_name == "lookup_policy":
        chunks = data.get("chunks", [])
        return f"{len(chunks)}개 청크"

    if tool_name == "lookup_my_subscription":
        sub_data = data
        plan = sub_data.get("planId", "?") if isinstance(sub_data, dict) else "?"
        return f"plan={plan}"[:30]

    if tool_name == "lookup_my_grade":
        grade = data.get("gradeId", "?") if isinstance(data, dict) else "?"
        return f"grade={grade}"[:30]

    if tool_name == "lookup_my_attendance":
        streak = data.get("currentStreak", "?") if isinstance(data, dict) else "?"
        return f"streak={streak}일"[:30]

    if tool_name == "lookup_my_orders":
        items = data if isinstance(data, list) else data.get("items", [])
        count = len(items) if isinstance(items, list) else "?"
        return f"{count}건 주문"

    if tool_name == "lookup_my_tickets":
        items = data if isinstance(data, list) else data.get("items", [])
        count = len(items) if isinstance(items, list) else "?"
        return f"{count}건 티켓"

    if tool_name == "lookup_my_recent_activity":
        return "조회됨"

    # 기본 요약
    return "조회됨"


# ============================================================
# 의도 → 바인딩할 tool 목록 결정
# ============================================================

def _get_bindable_tools(
    intent_kind: str,
    is_guest: bool,
) -> list[SupportToolSpec]:
    """
    의도 종류와 게스트 여부에 따라 LLM 에 bind 할 tool 목록을 결정한다.

    ### 바인딩 정책
    - personal_data (로그인): Read tool 8개 + lookup_policy + finish_task
    - personal_data (게스트): lookup_policy + finish_task (본인 데이터 tool 차단)
    - policy:                 lookup_policy + finish_task
    - faq:                    lookup_faq 는 레지스트리 외부 ES 경로이므로 finish_task 만
                              → 이 경로는 tool_selector 가 직접 lookup_faq 를 매핑하므로
                                bind_tools 는 호출되지 않는다.
    - redirect / complaint / smalltalk: tool_selector 에 진입하지 않으므로 공집합.

    Returns:
        바인딩할 SupportToolSpec 리스트 (finish_task 제외, 별도 추가됨)
    """
    all_specs = list(SUPPORT_TOOL_REGISTRY.values())

    if intent_kind == "personal_data":
        if is_guest:
            # 게스트: 본인 데이터 tool 전부 차단 — 정책 RAG 만
            return [s for s in all_specs if not s.requires_login]
        else:
            # 로그인: 전체 레지스트리 (Read 8 + lookup_policy)
            return all_specs

    if intent_kind == "policy":
        # 정책 질문: lookup_policy 만
        return [s for s in all_specs if s.name == "lookup_policy"]

    # 그 외 (faq 는 ES 직접 경로) — 정책 보조로 lookup_policy 만 바인딩
    return [s for s in all_specs if s.name == "lookup_policy"]


# ============================================================
# 핵심 함수
# ============================================================

async def select_support_tool(
    user_message: str,
    intent_kind: str,
    intent_confidence: float,
    is_guest: bool,
    tool_call_history: list[dict],
    tool_results_cache: dict,
    hop_count: int = 0,
    max_hops: int = 3,
    request_id: str = "",
) -> SupportSelectedTool | None:
    """
    고객센터 에이전트 발화 + intent 를 보고 최적 Tool 하나를 선택한다 (v4 ReAct).

    admin_tool_selector_chain.select_admin_tool 의 고객센터 버전.

    v4 변경점:
    - bind_tools 목록에 가상 tool `finish_task` 추가.
      LLM 이 "충분한 정보를 수집했다" 고 판단하면 이 이름으로 tool_call 을 내뱉는다.
    - tool_call_history + tool_results_cache 에서 압축 요약을 생성해 프롬프트에 주입.
    - is_guest=True 이면 requires_login=True tool 을 bind 목록에서 제외.
    - finish_task 는 실제 레지스트리에 없으므로 허용 목록 재검증 예외 처리.

    Args:
        user_message:        사용자 자연어 발화
        intent_kind:         SupportIntent.kind (faq/personal_data/policy/redirect/complaint/smalltalk)
        intent_confidence:   SupportIntent.confidence (0.0~1.0)
        is_guest:            비로그인 게스트 여부
        tool_call_history:   observation 노드가 누적한 이전 hop 이력
        tool_results_cache:  tool_executor 가 저장한 ref_id → result dict
        hop_count:           현재까지 실행된 hop 수 (0 부터 시작)
        max_hops:            최대 허용 hop 수 (기본 3, SUPPORT_MAX_HOPS 오버라이드)
        request_id:          로그 식별자

    Returns:
        SupportSelectedTool: LLM 이 tool 을 선택한 경우 (finish_task 포함)
        None: 적절한 tool 없음 / LLM 에러 (어떤 경우든 graceful)
    """
    start = time.perf_counter()

    # 1) 바인딩할 tool 목록 결정 (의도 + 게스트 여부)
    bindable = _get_bindable_tools(intent_kind, is_guest)

    # 2) Solar API LLM + bind_tools (실제 tool + finish_task 가상 tool)
    try:
        llm = get_solar_api_llm(temperature=0.1)
        tools = [_to_structured_tool(s) for s in bindable]

        # finish_task 가상 tool — "더 이상 tool 불필요" 시그널 전용
        finish_tool = StructuredTool.from_function(
            name="finish_task",
            description=(
                "더 이상 tool 호출이 필요 없을 때 선택하세요. "
                "수집된 데이터만으로 사용자 문의에 충분히 답변할 수 있을 때 사용합니다. "
                "게스트가 본인 데이터를 요청한 경우에도 이걸 선택해 로그인 안내를 제공하세요."
            ),
            args_schema=FinishTaskArgs,
            coroutine=_noop,
        )
        tools_with_finish = tools + [finish_tool]

        # tool_choice="auto": LLM 이 자율 판단 (tool 없으면 text 반환 가능)
        llm_with_tools = llm.bind_tools(tools_with_finish, tool_choice="auto")

        # 이전 hop 요약 생성 (토큰 절약)
        history_summary = compact_tool_results_summary(
            tool_call_history, tool_results_cache
        )

        # 프롬프트 렌더링 — ChatPromptValue 형태로 렌더 후 ainvoke 에 직접 넘김
        # (admin_tool_selector_chain.py 와 동일 패턴 — MagicMock 테스트 호환)
        prompt_value = _selector_prompt.format_prompt(
            user_message=user_message.strip(),
            intent_kind=intent_kind or "unknown",
            intent_confidence=float(intent_confidence),
            is_guest=is_guest,
            tool_history_summary=history_summary,
            hop_count=hop_count,
            max_hops=max_hops,
        )
        response = await guarded_ainvoke(
            llm_with_tools,
            prompt_value,
            model="solar_api",
            request_id=request_id or "support_tool_selector",
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "support_tool_selector_failed",
            error=str(exc),
            error_type=type(exc).__name__,
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
            "support_tool_selector_no_tool_call",
            content_preview=content_preview,
            hop_count=hop_count,
            elapsed_ms=round(elapsed_ms, 1),
        )
        return None

    first = tool_calls[0]
    name = first.get("name") or ""
    arguments = first.get("args") or {}

    # 4) 허용 tool 집합 재검증 (프롬프트 지시 위반 방어)
    #    finish_task 는 레지스트리 미등록 가상 tool 이므로 예외 허용.
    allowed_names = {s.name for s in bindable} | {"finish_task"}
    if name not in allowed_names:
        logger.warning(
            "support_tool_selector_disallowed_tool_rejected",
            tool_name=name,
            allowed=sorted(allowed_names),
            hop_count=hop_count,
        )
        return None

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "support_tool_selected",
        tool_name=name,
        args_keys=sorted(arguments.keys()),
        hop_count=hop_count,
        bindable_count=len(bindable),
        elapsed_ms=round(elapsed_ms, 1),
    )
    return SupportSelectedTool(
        name=name,
        arguments=arguments,
        rationale=f"bind_tools[{len(bindable)}+finish_task] hop={hop_count}",
    )
