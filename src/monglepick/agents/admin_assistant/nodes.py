"""
관리자 AI 에이전트 LangGraph 노드.

설계서: docs/관리자_AI에이전트_설계서.md §3.2 LangGraph 노드

Step 1 (2026-04-23):
- context_loader / intent_classifier / smalltalk_responder / response_formatter

Step 2 (2026-04-23, 추가):
- tool_selector   : Solar bind_tools 로 단일 tool_call 결정 (admin_role matrix 필터)
- tool_executor   : 레지스트리 handler 실행 → tool_results_cache 저장
- narrator        : Solar 가 tool_result 축약본을 한국어로 서술 (수치 생성 금지)

후속 Step 에서 추가될 노드:
- risk_gate       (Tier ≥ 2 → LangGraph interrupt)
- data_analyzer   (루프 종료 판단, pandas aggregate 필요 여부)
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
import uuid
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.types import interrupt

from monglepick.agents.admin_assistant.models import (
    AdminAssistantState,
    AdminIntent,
    ConfirmationPayload,
    ToolCall,
    ensure_intent,
    ensure_tool_call,
    normalize_admin_role,
)
from monglepick.api.admin_backend_client import AdminApiResult, summarize_for_llm
from monglepick.chains.admin_intent_chain import classify_admin_intent
from monglepick.chains.admin_tool_selector_chain import select_admin_tool
from monglepick.llm.factory import (
    get_conversation_llm,
    get_solar_api_llm,
    guarded_ainvoke,
)
from monglepick.prompts.admin_assistant import (
    NARRATOR_HUMAN_PROMPT,
    NARRATOR_SYSTEM_PROMPT,
    SMALLTALK_SYSTEM_PROMPT,
    SMART_FALLBACK_HUMAN_PROMPT,
    SMART_FALLBACK_SYSTEM_PROMPT,
)
from monglepick.tools.admin_tools import (
    ADMIN_TOOL_REGISTRY,
    ToolContext,
    list_tools_for_role,
)

logger = structlog.get_logger()

# ============================================================
# ReAct 루프 상한 (Phase D v3)
# ============================================================
# 환경변수로 override 가능. 기본 5 (토큰 비용 + 무한 루프 방어).
MAX_HOPS: int = int(os.getenv("ADMIN_ASSISTANT_MAX_HOPS", "5"))


# ============================================================
# 관리자 아닌 사용자 진입 차단 메시지
# ============================================================

_NOT_ADMIN_MESSAGE = (
    "관리자 권한이 필요한 기능이에요. 관리자 계정으로 로그인해주세요."
)


# ============================================================
# intent 별 placeholder 응답 (Step 1: tool 실행 미구현)
# ============================================================

# ============================================================
# Narrator 출력 후처리 — LLM 이 프롬프트 규칙을 어기고 메타/검증 체크리스트를
# 응답에 섞어 내는 케이스 2차 방어 (2026-04-23 운영 발견 이슈).
# ============================================================

# 아래 패턴이 감지되면 그 지점 이후를 모두 잘라낸다. 실제 관리자에게 보여줄
# "본문 + [출처: ...]" 는 이 패턴보다 앞에 오는 것이 정상이므로 공격적 truncate OK.
_NARRATOR_META_CUT_PATTERNS: list[re.Pattern] = [
    # "---" 단독 구분선 이후 전체 (LLM 이 --- 뒤에 검증 체크리스트를 붙이는 빈도 높음)
    re.compile(r"\n\s*-{3,}\s*\n.*$", re.DOTALL),
    # "**검증 사항**" / "검증 사항:" / "## 검증 사항" 같은 메타 헤더 이후 전체
    re.compile(r"\n\s*\**#*\s*검증\s*사항\**.*$", re.DOTALL),
    # "사고 과정" / "체크리스트" / "규칙 준수" 같은 유사 메타 헤더
    re.compile(r"\n\s*\**#*\s*(사고\s*과정|체크리스트|규칙\s*준수)\**.*$", re.DOTALL),
]

# 위 truncation 과 별도로 "(※ 실제 응답 시 ~)" 같은 자기-인용 괄호 문단은 본문
# 사이에도 끼어들 수 있어 문자열 전역에서 제거한다.
_NARRATOR_INLINE_META_PATTERNS: list[re.Pattern] = [
    re.compile(r"\n*\(※[^)]{0,400}\)\n*", re.DOTALL),
    re.compile(r"\n*\(\s*실제\s*응답\s*시[^)]{0,400}\)\n*", re.DOTALL),
]


def _sanitize_narrator_output(text: str) -> str:
    """
    narrator LLM 응답에서 메타/검증 체크리스트 누수를 제거한다.

    전략:
    - 길이 기반 truncation: "---" 또는 "**검증 사항**" 같은 시그니처 이후를 싹 자름.
    - 인라인 제거: "(※ 실제 응답 시 ...)" 같은 괄호 자기-인용 문단을 전역 삭제.
    - `[출처: ...]` 라인은 보존 (사용자에게 필요한 근거 표기).
    - 최종 strip 으로 trailing 공백/개행 정리.

    주의: 이 함수는 LLM 출력에 대한 **방어적 후처리** 일 뿐이며, 프롬프트(NARRATOR_SYSTEM_PROMPT)
    에서 "메타 금지" 를 명시적으로 지시하는 것이 1차 방어다.
    """
    if not text:
        return text
    cleaned = text
    # 1단계: 뒤쪽 메타 블록 잘라내기 (가장 먼저 매치되는 지점까지)
    for pat in _NARRATOR_META_CUT_PATTERNS:
        cleaned = pat.sub("", cleaned)
    # 2단계: 본문 중간에 낀 자기-인용 괄호 제거
    for pat in _NARRATOR_INLINE_META_PATTERNS:
        cleaned = pat.sub("\n", cleaned)
    # 마지막 공백 정리 + 빈 줄 3개 이상 연속 → 2줄로 축약
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# ============================================================
# Audit target 추론 (Step 6b)
# ============================================================

# tool 이름 → (Backend AdminAuditService 의 TARGET_* 상수 문자열, arguments key 후보 순서)
# Agent 쪽에서는 Backend 의 TARGET_USER / TARGET_PAYMENT / TARGET_SUBSCRIPTION 등을
# 하드코딩 상수로 맞춰둔다(AdminAuditService.java 의 값과 동일해야 한다).
_AUDIT_TARGET_HINTS: dict[str, tuple[str, tuple[str, ...]]] = {
    # users_write (Step 6b)
    "user_suspend": ("USER", ("userId",)),
    "user_unsuspend": ("USER", ("userId",)),  # Step 6c 예정
    "user_role_update": ("USER", ("userId",)),
    # points_write (Step 6b)
    "points_manual_adjust": ("USER", ("userId",)),
    # payment_write (Step 6c 예정)
    "payment_refund": ("PAYMENT", ("orderId",)),
    "ai_token_grant": ("USER", ("userId",)),
    # support_write / settings_write (Step 5a)
    "faq_create": ("FAQ", ("faqId",)),
    "banner_create": ("BANNER", ("bannerId",)),
}


def _infer_audit_target(tool_name: str, arguments: dict) -> tuple[str | None, str | None]:
    """
    tool 이름 + arguments 에서 감사 로그의 (targetType, targetId) 를 추론한다.

    매핑 규칙(§7.2):
    - 레지스트리에 hint 가 있으면 그걸 사용. arguments 에서 후보 키를 순서대로 검사해 첫
      non-empty 문자열을 targetId 로 반환.
    - hint 가 없으면 둘 다 None — Backend 가 targetType/targetId null 로 저장.
    """
    hint = _AUDIT_TARGET_HINTS.get(tool_name)
    if not hint:
        return (None, None)
    target_type, keys = hint
    for k in keys:
        value = arguments.get(k) if isinstance(arguments, dict) else None
        if isinstance(value, (str, int)) and str(value).strip():
            return (target_type, str(value))
    return (target_type, None)


_PLACEHOLDER_MESSAGES: dict[str, str] = {
    # Step 4 부터 query 도 tool_selector 경로를 경유한다. 이 메시지는 tool_selector 가
    # "적합한 도구를 못 찾은" 경우의 fallback 으로만 쓰인다. "개발 중" 문구는 제거.
    "query": (
        "요청하신 조회에 적합한 도구를 찾지 못했어요. "
        "사용자/결제/게시글/리뷰/티켓 등의 조회는 지원되지만, 구체 대상(userId·orderId 등)이 "
        "발화에 포함되어야 하는 경우가 많아요. 질문을 조금 더 구체적으로 말씀해 주세요."
    ),
    "action": (
        "🛠️ 쓰기 작업(공지·배너·FAQ·퀴즈 CRUD, 계정 정지/환불 등) 은 "
        "아직 구현되지 않았어요. 관리자 승인 플로우(HITL) 와 함께 다음 단계에서 추가될 예정입니다."
    ),
    # Step 2 에서 stats 는 실제 tool 경로로 분기. 이 placeholder 는
    # "stats intent 이지만 적합한 tool 이 없거나 tool 실행 실패" fallback 으로 쓰인다.
    "stats": (
        "요청하신 통계에 적합한 도구를 찾지 못했어요. "
        "다음 Step 에서 더 많은 통계 도구(추천 성능·포인트 경제·참여도 등) 가 추가됩니다."
    ),
    "report": (
        "🛠️ 보고서 생성은 Phase 4 예정 기능이에요. "
        "지금은 통계 조회가 연결된 후 주간/월간 템플릿으로 확장됩니다."
    ),
    "sql": (
        "이 에이전트는 자유 SQL 실행을 지원하지 않아요. "
        "기존 통계 화면이나 미리 준비된 조회를 이용해 주세요."
    ),
}


# ============================================================
# Node 1 — context_loader
# ============================================================

async def context_loader(state: AdminAssistantState) -> dict:
    """
    요청 수명 시작점. admin_id / admin_role 검증 + 빈 필드 기본값 채움.

    - admin_role 이 비어있으면 (정규화 실패) 이후 노드가 placeholder 로 우회해
      안내 메시지만 내려가도록 한다.
    - history 는 Step 1 에서 매 요청 빈 배열로 초기화 (세션 저장소 미도입).
    - tool_results_cache / analysis_outputs / chart_payloads / table_payloads
      모두 빈 기본값으로 초기화 해 downstream 노드가 접근해도 KeyError 가 나지 않게.
    """
    start = time.perf_counter()
    admin_id = state.get("admin_id", "") or ""
    admin_role = normalize_admin_role(state.get("admin_role", ""))

    if not admin_id:
        logger.warning("admin_assistant_missing_admin_id")
    if not admin_role:
        logger.warning(
            "admin_assistant_missing_or_invalid_role",
            raw_role=state.get("admin_role"),
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "admin_context_loaded",
        admin_id=admin_id or "(unknown)",
        admin_role=admin_role or "(blank)",
        session_id=state.get("session_id", ""),
        elapsed_ms=round(elapsed_ms, 1),
    )
    return {
        "admin_id": admin_id,
        "admin_role": admin_role,
        "history": state.get("history", []) or [],
        "candidate_tools": state.get("candidate_tools", []) or [],
        "tool_results_cache": state.get("tool_results_cache", {}) or {},
        "analysis_outputs": state.get("analysis_outputs", []) or [],
        "chart_payloads": state.get("chart_payloads", []) or [],
        "table_payloads": state.get("table_payloads", []) or [],
        "awaiting_confirmation": False,
        "iteration_count": 0,
        "error": None,
    }


# ============================================================
# Node 2 — intent_classifier
# ============================================================

async def intent_classifier(state: AdminAssistantState) -> dict:
    """
    발화를 AdminIntent(kind/confidence/reason) 로 분류한다.

    관리자 권한이 없는 상태(admin_role="") 로 진입하면 LLM 호출을 건너뛰고
    smalltalk 로 고정해 불필요한 Solar API 비용을 절감한다.
    """
    admin_role = state.get("admin_role", "") or ""
    admin_id = state.get("admin_id", "") or ""
    user_message = state.get("user_message", "") or ""

    # 비관리자 → 분류 생략, 이후 response_formatter 가 _NOT_ADMIN_MESSAGE 로 응답
    if not admin_role:
        return {
            "intent": AdminIntent(
                kind="smalltalk",
                confidence=0.0,
                reason="no_admin_role",
            ),
        }

    # 빈 발화 방어 — 체인을 태우지 않고 smalltalk 로 처리
    if not user_message.strip():
        return {
            "intent": AdminIntent(
                kind="smalltalk",
                confidence=0.0,
                reason="empty_user_message",
            ),
        }

    intent = await classify_admin_intent(
        user_message=user_message,
        admin_role=admin_role,
        request_id=f"admin:{admin_id[:8]}" if admin_id else "admin:anon",
    )
    return {"intent": intent}


# ============================================================
# Node 3 — smalltalk_responder
# ============================================================

async def smalltalk_responder(state: AdminAssistantState) -> dict:
    """
    smalltalk intent 응답 생성.

    hybrid 모드: vLLM EXAONE 1.2B 또는 Ollama 몽글이(빠른 응답 체인).
    api_only: Solar API.
    local_only: Ollama EXAONE 32B.

    get_conversation_llm() 이 LLM_MODE 를 내부 분기한다.
    수치/유저 정보는 이 응답에서 만들지 않도록 프롬프트로 강제한다.
    """
    start = time.perf_counter()
    user_message = state.get("user_message", "") or ""
    admin_id = state.get("admin_id", "")

    try:
        llm = get_conversation_llm()
        messages = [
            SystemMessage(content=SMALLTALK_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response_obj = await guarded_ainvoke(
            llm,
            messages,
            model="admin_smalltalk",
            request_id=f"admin:{admin_id[:8]}" if admin_id else "admin:anon",
        )
        text = getattr(response_obj, "content", None) or str(response_obj)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "admin_smalltalk_generated",
            length=len(text),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"response_text": text.strip()}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "admin_smalltalk_failed_fallback",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )
        # 에러 전파 금지 — 고정 안내로 폴백
        return {
            "response_text": (
                "안녕하세요! 관리자 어시스턴트예요. "
                "통계 조회·리소스 조회·배너/공지/FAQ 등록 같은 걸 자연어로 요청해주시면 도와드릴게요."
            ),
        }


# ============================================================
# Node 4 — response_formatter
# ============================================================

async def response_formatter(state: AdminAssistantState) -> dict:
    """
    최종 응답 조립.

    우선순위:
    1) admin_role 이 비어있으면 → _NOT_ADMIN_MESSAGE
    2) smalltalk_responder 가 채운 response_text 가 있으면 → 그대로 사용
    3) 그 외 intent 는 Step 1 에서 placeholder 안내

    이 단계는 수치를 생성/가공하지 않는다 (§6.1 "LLM 은 숫자를 만들지 않는다").
    """
    admin_role = state.get("admin_role", "") or ""
    # MemorySaver 복원으로 dict 가 된 경우도 AdminIntent 로 되살림 (Step 6b 후속 방어)
    intent = ensure_intent(state.get("intent"))
    already_composed = state.get("response_text", "") or ""

    # 1) 비관리자 차단
    if not admin_role:
        return {"response_text": _NOT_ADMIN_MESSAGE}

    # 2) smalltalk 응답이 이미 채워진 경우
    if already_composed:
        return {"response_text": already_composed}

    # 3) 나머지 intent 는 Step 1 placeholder
    intent_kind = intent.kind if intent is not None else "smalltalk"
    placeholder = _PLACEHOLDER_MESSAGES.get(
        intent_kind,
        "요청을 처리하지 못했어요. 다시 한 번 말씀해주시겠어요?",
    )
    return {"response_text": placeholder}


# ============================================================
# Step 2 Node 5 — tool_selector
# ============================================================

async def tool_selector(state: AdminAssistantState) -> dict:
    """
    stats/query/action intent 에서 Solar bind_tools 로 단일 tool-call 을 선택한다 (v3 Phase D).

    v3 변경점:
    - state 에서 iteration_count / tool_call_history / tool_results_history 를 읽어
      tool_history_summary 문자열을 생성한 뒤 select_admin_tool 에 전달.
    - finish_task 가 선택된 경우: pending_tool_call 에 finish_task ToolCall 을 담아 반환.
      route_after_tool_select 가 이를 감지해 tool_executor 를 건너뛰고 narrator 로 직행.
    - admin_role 이 없거나 레지스트리 필터 결과가 비어있으면 pending_tool_call=None.
    - LLM 응답에 tool_call 이 없으면 pending_tool_call=None → smart_fallback_responder 로 직행.
    - 성공 시 ToolCall 객체를 state.pending_tool_call 에 저장. tier 는 레지스트리에서 주입.
    """
    admin_role = state.get("admin_role", "") or ""
    admin_id = state.get("admin_id", "") or ""
    user_message = state.get("user_message", "") or ""
    intent = ensure_intent(state.get("intent"))
    intent_kind = intent.kind if intent is not None else "unknown"

    # 현재 hop 카운트 (iteration_count 재사용)
    hop_count: int = state.get("iteration_count") or 0

    # 이전 hop 결과 이력에서 tool_history_summary 생성
    # 형식: "1. tool_name → ok=True row_count=5\n2. ..."
    results_history: list[dict[str, Any]] = list(state.get("tool_results_history") or [])
    summary_lines: list[str] = []
    for idx, entry in enumerate(results_history, start=1):
        tool_name = entry.get("tool_name", "?")
        ok = entry.get("ok", False)
        row_count = entry.get("row_count")
        row_str = f" row_count={row_count}" if row_count is not None else ""
        summary_lines.append(f"{idx}. {tool_name} → ok={ok}{row_str}")
    tool_history_summary: str = "\n".join(summary_lines) if summary_lines else ""

    if not admin_role or not user_message.strip():
        return {"pending_tool_call": None}

    # ── Step 7a/7b: Tool 후보 빌더 — 카테고리 필터 + (옵션) RAG 의미 유사도 머지 ──
    # 1) tool_filter.shortlist_tools_by_category — Qdrant 0 의존, 빠른 1차 후보 (intent_kind 기반)
    # 2) ADMIN_TOOL_RAG_ENABLED=true 면 search_similar_tools 추가 호출 → 의미 유사도 보강
    # 두 결과를 머지(filter 우선)해 LLM bind 전에 최대 ADMIN_TOOL_FILTER_MAX(기본 30) 개로 절단.
    # RAG 장애 시 filter 결과만 반환되므로 페일 세이프.
    filter_max: int = int(os.getenv("ADMIN_TOOL_FILTER_MAX", "30"))

    allowed_tool_names: list[str] | None = None
    if admin_role:
        try:
            from monglepick.tools.admin_tools.tool_rag import build_admin_tool_candidates_async
            allowed_tool_names = await build_admin_tool_candidates_async(
                user_message=user_message,
                admin_role=admin_role,
                intent_kind=intent_kind,
                max_tools=filter_max,
            )
            logger.info(
                "admin_tool_filter_candidates",
                count=len(allowed_tool_names),
                top=allowed_tool_names[:5],
                hop_count=hop_count,
            )
        except Exception as filter_err:
            # 후보 빌더 자체가 실패하면 None → select_admin_tool 이 role 기반 전체 bind
            logger.warning(
                "admin_tool_filter_failed_fallback_all",
                error=str(filter_err),
                error_type=type(filter_err).__name__,
            )
            allowed_tool_names = None

    selected = await select_admin_tool(
        user_message=user_message,
        admin_role=admin_role,
        intent_kind=intent_kind,
        request_id=f"admin:{admin_id[:8]}" if admin_id else "admin:anon",
        tool_history_summary=tool_history_summary,
        hop_count=hop_count,
        max_hops=MAX_HOPS,
        allowed_tool_names=allowed_tool_names,
    )

    if selected is None:
        return {"pending_tool_call": None}

    # finish_task 가상 tool: 레지스트리에 없지만 tier=0 으로 처리.
    # route_after_tool_select 에서 이 이름을 감지해 narrator 로 직행.
    if selected.name == "finish_task":
        call = ToolCall(
            tool_name="finish_task",
            arguments=selected.arguments,
            tier=0,
            rationale=selected.rationale,
        )
        logger.info(
            "admin_tool_selector_finish_task",
            hop_count=hop_count,
            reason=selected.arguments.get("reason", ""),
        )
        return {"pending_tool_call": call}

    # 레지스트리에서 tier 주입 (selector 체인은 tier 를 반환하지 않음)
    spec = ADMIN_TOOL_REGISTRY.get(selected.name)
    tier = spec.tier if spec is not None else 0

    call = ToolCall(
        tool_name=selected.name,
        arguments=selected.arguments,
        tier=tier,
        rationale=selected.rationale,
    )
    return {"pending_tool_call": call}


# ============================================================
# Step 2 Node 6 — tool_executor
# ============================================================

async def tool_executor(state: AdminAssistantState) -> dict:
    """
    pending_tool_call 을 실제로 실행한다.

    - 레지스트리에서 ToolSpec 조회 후 args_schema 로 arguments 검증.
    - admin_role 이 허용 목록에 포함되지 않으면 실행 거부 (설계서 §4.2 Role matrix).
    - Tier 2/3 는 Step 2 범위 밖 — 현재는 Tier 0/1 만 실행되도록 가드.
    - 결과는 tool_results_cache[ref_id] 에 저장하고, ref_id 를 state.latest_tool_ref_id 에 기록.
    - 실패(ok=False) 결과도 그대로 캐시한다 — narrator 가 error 메시지를 정확히 서술하도록.
    """
    # MemorySaver 직렬화로 dict 복원된 경우도 ToolCall 로 되살림 (Step 6b 후속 방어)
    call = ensure_tool_call(state.get("pending_tool_call"))
    admin_role = state.get("admin_role", "") or ""
    cache = dict(state.get("tool_results_cache", {}) or {})

    if call is None:
        return {"tool_results_cache": cache, "latest_tool_ref_id": ""}

    spec = ADMIN_TOOL_REGISTRY.get(call.tool_name)
    if spec is None:
        logger.warning("admin_tool_executor_unknown_tool", tool_name=call.tool_name)
        return {"tool_results_cache": cache, "latest_tool_ref_id": ""}

    # Step 5a: Tier 2/3 하드 가드는 제거됐다 (risk_gate 에서 사용자 승인을 받은 뒤에만
    # tool_executor 가 호출되는 구조). 다만 Tier 4(SQL 샌드박스) 는 설계상 영구 미지원이라
    # 여전히 차단한다.
    if spec.tier >= 4:
        logger.info(
            "admin_tool_executor_tier4_blocked",
            tool_name=call.tool_name,
            tier=spec.tier,
            reason="tier=4 (SQL 샌드박스) 는 v2 영구 미지원",
        )
        return {"tool_results_cache": cache, "latest_tool_ref_id": ""}

    # Role matrix 재검증 (selector 에서 한 번 걸러도 실행 직전 이중 방어)
    allowed = {s.name for s in list_tools_for_role(admin_role)}
    if call.tool_name not in allowed:
        logger.warning(
            "admin_tool_executor_role_denied",
            tool_name=call.tool_name,
            admin_role=admin_role,
        )
        return {"tool_results_cache": cache, "latest_tool_ref_id": ""}

    # args 검증 — Pydantic 으로 타입/기본값 적용
    try:
        validated = spec.args_schema.model_validate(call.arguments)
        args_dict = validated.model_dump()
    except Exception as e:
        logger.warning(
            "admin_tool_executor_args_validation_failed",
            tool_name=call.tool_name,
            arguments=call.arguments,
            error=str(e),
        )
        failed = AdminApiResult(
            ok=False, status_code=0, error=f"args_validation:{type(e).__name__}",
        )
        ref_id = f"tr_{uuid.uuid4().hex[:10]}"
        cache[ref_id] = failed
        return {"tool_results_cache": cache, "latest_tool_ref_id": ref_id}

    # 실행
    ctx = ToolContext(
        admin_jwt=state.get("admin_jwt", "") or "",
        admin_role=admin_role,
        admin_id=state.get("admin_id", "") or "",
        session_id=state.get("session_id", "") or "",
        invocation_id=f"admin:{state.get('session_id', '')[:12]}",
    )
    start = time.perf_counter()
    try:
        result = await spec.handler(ctx=ctx, **args_dict)
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.warning(
            "admin_tool_executor_handler_crashed",
            tool_name=call.tool_name,
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=elapsed_ms,
            stack_trace=traceback.format_exc(),
        )
        result = AdminApiResult(
            ok=False, status_code=0,
            error=f"handler_crash:{type(e).__name__}",
            latency_ms=elapsed_ms,
        )

    ref_id = f"tr_{uuid.uuid4().hex[:10]}"
    cache[ref_id] = result
    logger.info(
        "admin_tool_executed",
        tool_name=call.tool_name,
        tier=spec.tier,
        ok=result.ok,
        status_code=result.status_code,
        latency_ms=result.latency_ms,
        ref_id=ref_id,
    )

    # ── Step 6a/6b: Tier≥2 쓰기 실행은 감사 로그에 자동 기록 ──
    # Tier 0/1 읽기는 볼륨 폭증 방지를 위해 감사 미기록(§7.2). Tier 2/3 은 성공/실패 모두
    # 한 건 기록 — "AGENT_EXECUTED" actionType 으로, 실제 쓰기가 발생시킨 도메인 감사 로그
    # (POINT_MANUAL 등) 와 별도 레코드로 남아 양방향 추적 가능(§7.1). 감사 기록 실패는
    # graceful — 원 작업 응답은 그대로 narrator 로 흘러간다.
    #
    # Step 6b 추가: Tier 3 tool 이 `AdminApiResult.before_data`/`after_data` 에 담아 올린
    # 리소스 스냅샷을 audit 의 beforeData/afterData 필드로 그대로 전달한다. 또한
    # arguments 의 `userId`/`orderId` 를 target_id 로 유추해 감사 조회 시 특정 리소스로
    # 필터링하기 쉽게 한다(targetType 은 tool 이름에서 간단 매핑).
    if spec.tier >= 2:
        from monglepick.api.admin_audit_client import log_agent_action
        target_type, target_id = _infer_audit_target(call.tool_name, call.arguments)
        try:
            await log_agent_action(
                admin_jwt=state.get("admin_jwt", "") or "",
                tool_name=call.tool_name,
                arguments=call.arguments,
                ok=result.ok,
                user_prompt=state.get("user_message", "") or "",
                target_type=target_type,
                target_id=target_id,
                before_data=result.before_data,
                after_data=result.after_data,
                error=result.error if not result.ok else "",
                invocation_id=ctx.invocation_id,
            )
        except Exception as audit_err:
            # log_agent_action 내부에서 이미 graceful 처리하지만 이중 안전망.
            logger.warning(
                "admin_tool_audit_outer_error",
                tool_name=call.tool_name,
                error=str(audit_err),
            )

    return {"tool_results_cache": cache, "latest_tool_ref_id": ref_id}


# ============================================================
# Step 2 Node 7 — narrator
# ============================================================

async def narrator(state: AdminAssistantState) -> dict:
    """
    tool_result 축약본을 Solar Pro 로 한국어 서술.

    - latest_tool_ref_id 가 없으면 아무것도 하지 않음 (response_formatter 가 placeholder 사용).
    - 축약본은 summarize_for_llm 으로 생성 — raw rows 가 LLM 컨텍스트에 들어가지 않음 (§6.1).
    - 수치 생성 금지 규칙은 프롬프트로 강제.
    - LLM 실패 시 "조회는 됐지만 해석이 실패했다" 로 안내 + tool 원시값을 축약해 인용.
    """
    ref_id = state.get("latest_tool_ref_id", "") or ""
    cache = state.get("tool_results_cache", {}) or {}
    call = ensure_tool_call(state.get("pending_tool_call"))

    if not ref_id or ref_id not in cache:
        # tool_selector 가 None 을 낸 경우 — response_formatter 에서 placeholder 처리
        return {}

    result = cache[ref_id]
    # result 는 AdminApiResult 인스턴스 또는 dict (테스트용). 둘 다 허용.
    if isinstance(result, AdminApiResult):
        summarized = summarize_for_llm(result)
    elif isinstance(result, dict):
        summarized = result
    else:
        summarized = {"ok": False, "error": f"unexpected_result_type:{type(result).__name__}"}

    tool_name = call.tool_name if call else "(unknown)"
    args_repr = json.dumps(call.arguments, ensure_ascii=False) if call else "{}"
    result_json = json.dumps(summarized, ensure_ascii=False, default=str)

    start = time.perf_counter()
    try:
        llm = get_solar_api_llm(temperature=0.2)
        system = SystemMessage(content=NARRATOR_SYSTEM_PROMPT)
        human = HumanMessage(content=NARRATOR_HUMAN_PROMPT.format(
            user_message=state.get("user_message", ""),
            tool_name=tool_name,
            tool_args=args_repr,
            tool_result_json=result_json,
        ))
        response = await guarded_ainvoke(
            llm, [system, human],
            model="solar_api",
            request_id=f"admin_narrator:{ref_id}",
        )
        raw_text = getattr(response, "content", None) or str(response)
        # 2차 방어: LLM 이 프롬프트 규칙을 어기고 "검증 사항" / "---" / "(※ ~)" 같은
        # 메타 텍스트를 섞어 내는 케이스를 후처리로 제거 (2026-04-23 운영 발견 이슈).
        text = _sanitize_narrator_output(raw_text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "admin_narrator_generated",
            tool_name=tool_name,
            raw_length=len(raw_text),
            sanitized_length=len(text),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"response_text": text}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "admin_narrator_failed",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed_ms, 1),
        )
        # narrator 실패 fallback — 축약본 원문을 그대로 인용
        if isinstance(result, AdminApiResult) and result.ok:
            return {
                "response_text": (
                    f"{tool_name} 호출은 성공했지만 결과 해석에 실패했어요. "
                    f"원시 데이터를 그대로 전달드려요:\n\n```\n{result_json[:1500]}\n```\n\n"
                    f"[출처: {tool_name}]"
                ),
            }
        return {
            "response_text": (
                f"요청을 처리하는 중 문제가 발생했어요 ({type(e).__name__}). 잠시 후 다시 시도해주세요."
            ),
        }


# ============================================================
# Phase D v3 Node — observation
# ============================================================

async def observation(state: AdminAssistantState) -> dict:
    """
    tool_executor 결과를 tool_call_history / tool_results_history 에 누적한다 (v3 Phase D).

    역할:
    - pending_tool_call 을 tool_call_history 에 append.
    - tool_results_cache[latest_tool_ref_id] 에서 결과를 꺼내 축약본을 tool_results_history 에 append.
    - iteration_count(= hop_count) 를 1 증가.
    - route_after_observation 이 이 노드 이후 경로를 결정한다:
        * 마지막 tool 이 finish_task → narrator 직행
        * 마지막 tool 이 *_draft → draft_emitter
        * 마지막 tool 이 goto_* → navigator
        * 그 외 read tool → tool_selector (다음 hop)
        * hop_count >= MAX_HOPS → narrator 강제 종결

    에러 전파 금지 — tool_results 파싱 실패 시 ok=False 축약본을 기록하고 계속 진행.
    """
    call = ensure_tool_call(state.get("pending_tool_call"))
    cache: dict[str, Any] = state.get("tool_results_cache", {}) or {}
    ref_id: str = state.get("latest_tool_ref_id", "") or ""
    result = cache.get(ref_id) if ref_id else None

    # tool_call_history 누적 (ToolCall 을 그대로 append — MemorySaver 직렬화로 dict 화됨)
    history: list = list(state.get("tool_call_history") or [])
    results_history: list[dict[str, Any]] = list(state.get("tool_results_history") or [])

    if call is not None:
        history.append(call)

    # 결과 축약본 생성
    if result is not None:
        # AdminApiResult 인스턴스와 dict 양쪽 처리 (테스트 환경은 dict 를 쓰기도 함)
        if isinstance(result, AdminApiResult):
            ok = result.ok
            row_count = result.row_count
        elif isinstance(result, dict):
            ok = bool(result.get("ok", False))
            row_count = result.get("row_count")
        else:
            ok = False
            row_count = None

        results_history.append({
            "tool_name": call.tool_name if call else "",
            "ok": ok,
            "row_count": row_count,
            "summary": "",  # 필요 시 narrator 가 full result 에서 생성
        })
        logger.info(
            "admin_observation_recorded",
            tool_name=call.tool_name if call else "(none)",
            ok=ok,
            row_count=row_count,
            hop_after=(state.get("iteration_count") or 0) + 1,
        )
    else:
        logger.debug(
            "admin_observation_no_result",
            ref_id=ref_id or "(empty)",
            tool_name=call.tool_name if call else "(none)",
        )

    return {
        "tool_call_history": history,
        "tool_results_history": results_history,
        "iteration_count": (state.get("iteration_count") or 0) + 1,
    }


# ============================================================
# Phase D v3 Node — draft_emitter
# ============================================================

async def draft_emitter(state: AdminAssistantState) -> dict:
    """
    *_draft tool 실행 직후 결과를 state.form_prefill 로 확정한다 (v3 Phase D).

    - tool_results_cache[latest_tool_ref_id].data 가 dict 이면 form_prefill 로 저장.
    - graph.py 의 SSE 루프가 이 노드 완료 시 form_prefill 이벤트를 발행한다.
    - Backend 는 호출하지 않는다 — Draft tool 자체가 Backend 미호출이다.
    - 에러 전파 금지: data 가 없거나 dict 가 아니면 form_prefill=None 으로 처리.
    """
    cache: dict[str, Any] = state.get("tool_results_cache", {}) or {}
    ref_id: str = state.get("latest_tool_ref_id", "") or ""
    result = cache.get(ref_id) if ref_id else None

    data: Any = None
    if isinstance(result, AdminApiResult):
        data = result.data
    elif isinstance(result, dict):
        data = result.get("data")

    if not isinstance(data, dict):
        logger.debug(
            "admin_draft_emitter_no_data",
            ref_id=ref_id or "(empty)",
            data_type=type(data).__name__,
        )
        return {"form_prefill": None}

    logger.info(
        "admin_draft_emitter_set",
        target_path=data.get("target_path", ""),
        tool_name=data.get("tool_name", ""),
    )
    return {"form_prefill": data}


# ============================================================
# Phase D v3 Node — navigator
# ============================================================

async def navigator(state: AdminAssistantState) -> dict:
    """
    goto_* tool 실행 직후 결과를 state.navigation 으로 확정한다 (v3 Phase D).

    - tool_results_cache[latest_tool_ref_id].data 가 dict 이면 navigation 으로 저장.
    - graph.py 의 SSE 루프가 이 노드 완료 시 navigation 이벤트를 발행한다.
    - Backend 는 호출하지 않는다 — Navigate tool 이 내부적으로 read 를 한 번 호출해 이미 결과를 담아 온다.
    - candidates 가 여러 개인 경우에도 navigation.candidates 배열로 그대로 전달한다.
    - 에러 전파 금지: data 가 없거나 dict 가 아니면 navigation=None 으로 처리.
    """
    cache: dict[str, Any] = state.get("tool_results_cache", {}) or {}
    ref_id: str = state.get("latest_tool_ref_id", "") or ""
    result = cache.get(ref_id) if ref_id else None

    data: Any = None
    if isinstance(result, AdminApiResult):
        data = result.data
    elif isinstance(result, dict):
        data = result.get("data")

    if not isinstance(data, dict):
        logger.debug(
            "admin_navigator_no_data",
            ref_id=ref_id or "(empty)",
            data_type=type(data).__name__,
        )
        return {"navigation": None}

    logger.info(
        "admin_navigator_set",
        target_path=data.get("target_path"),
        label=data.get("label", ""),
        candidates_count=len(data.get("candidates", [])),
    )
    return {"navigation": data}


# ============================================================
# Step 5a Node — risk_gate (HITL 승인 게이트) — v3 에서 비활성화
# ============================================================
# Phase D v3 재설계에서 risk_gate 는 그래프에서 제거되었다.
# 실제 쓰기 tool 이 없어져 HITL interrupt 가 불필요하기 때문이다.
# 함수는 _deprecated_risk_gate 로 이름 변경해 dead code 로 보존 (revert 대비).

async def _deprecated_risk_gate(state: AdminAssistantState) -> dict:  # noqa: D401
    """DEPRECATED — v3 Phase D 에서 그래프에서 제거됨. 아래 구현은 revert 용 보존."""
    return await _impl_deprecated_risk_gate(state)


# v2 하위호환 — `from ... import risk_gate` 로 참조하는 레거시 테스트·모듈이 남아있음.
# Phase E 에서 테스트 정리 후 제거 예정. 지금은 동일 deprecated 함수를 alias 로 노출.
risk_gate = _deprecated_risk_gate


async def _impl_deprecated_risk_gate(state: AdminAssistantState) -> dict:
    """
    Tier≥2 쓰기 작업이 실행되기 전 사용자 승인을 받는 관문.

    흐름:
    - pending_tool_call.tier < 2 → 통과 (아무 것도 안 하고 return {}). tool_executor 로 직행.
    - pending_tool_call.tier >= 2 → LangGraph `interrupt(payload)` 호출.
      · 최초 실행 시: 그래프가 여기서 멈추고, astream 루프가 종료된다. SSE 레이어가
        state.confirmation_payload 를 보고 `confirmation_required` 이벤트를 발행한다.
      · Client 가 `/resume` 으로 ConfirmationDecision 을 보내면 LangGraph 가 이 지점부터
        재개하는데, interrupt() 는 이번엔 decision 값을 반환한다.
    - decision.decision == 'approve' → tool_executor 진행 (state 그대로 유지).
    - decision.decision == 'reject' → pending_tool_call=None, response_text 에 거절 안내를
      직접 써서 response_formatter 가 그대로 내보내게 한다.

    주의: Tier 4(SQL) 는 레지스트리에 등록이 금지되어 있지만, 혹시 등록돼도 tool_executor 가
    재차 차단한다.
    """
    # MemorySaver 직렬화로 dict 복원된 경우도 ToolCall 로 되살림 (Step 6b 후속 방어)
    call = ensure_tool_call(state.get("pending_tool_call"))
    if call is None:
        return {}

    # Tier 0/1 은 HITL 불필요 — 바로 통과
    if call.tier < 2:
        return {}

    # 레지스트리에서 rationale/설명/confirm_keyword 를 끌어와 plan_summary 를 구성
    spec = ADMIN_TOOL_REGISTRY.get(call.tool_name)
    tool_desc = spec.description[:120] if spec else call.tool_name
    # Step 6b: Tier 3 tool 은 spec.confirm_keyword 를 payload.required_keyword 로 전달.
    # Admin UI 가 이 키워드를 사용자에게 타이핑시켜 오조작 2중 방어.
    keyword = (spec.confirm_keyword if spec and spec.confirm_keyword else "") or ""
    payload = ConfirmationPayload(
        tool_name=call.tool_name,
        arguments=call.arguments,
        tier=call.tier,
        plan_summary=tool_desc,
        rationale=call.rationale,
        required_keyword=keyword,
    )

    logger.info(
        "admin_risk_gate_interrupt",
        tool_name=call.tool_name,
        tier=call.tier,
        session_id=state.get("session_id", ""),
    )

    # interrupt() 는 최초 호출 시 그래프를 여기서 멈추고, resume 이 들어오면 그 값을 반환한다.
    decision_raw = interrupt(payload.model_dump())

    # ── 재개 경로: decision 값 파싱 ──
    # decision_raw 는 ConfirmationDecision.model_dump() 형태로 오는 걸 기대하지만,
    # 방어적으로 dict / 문자열 / None 모든 경우를 처리한다.
    if isinstance(decision_raw, dict):
        decision = str(decision_raw.get("decision", "")).lower()
        comment = str(decision_raw.get("comment", ""))
    elif isinstance(decision_raw, str):
        decision = decision_raw.lower()
        comment = ""
    else:
        decision = ""
        comment = ""

    if decision == "approve":
        logger.info(
            "admin_risk_gate_approved",
            tool_name=call.tool_name,
            session_id=state.get("session_id", ""),
        )
        return {
            "awaiting_confirmation": False,
            "confirmation_decision": {"decision": "approve", "comment": comment},
        }

    # reject 또는 알 수 없는 응답 → 안전하게 거절 처리
    logger.info(
        "admin_risk_gate_rejected",
        tool_name=call.tool_name,
        raw_decision=str(decision_raw)[:100],
        session_id=state.get("session_id", ""),
    )
    reject_msg = (
        "요청하신 쓰기 작업을 실행하지 않았어요. "
        f"(거부된 작업: `{call.tool_name}`)"
    )
    if comment:
        reject_msg += f" 메모: {comment}"
    return {
        "awaiting_confirmation": False,
        "confirmation_decision": {"decision": "reject", "comment": comment},
        # pending_tool_call 을 None 으로 비워 route_after_risk_gate 가 실행 경로 차단 판단
        "pending_tool_call": None,
        "response_text": reject_msg,
    }


# ============================================================
# Smart Fallback Node (2026-04-23) — tool 매칭 실패 시 LLM 역제안
# ============================================================

async def smart_fallback_responder(state: AdminAssistantState) -> dict:
    """
    tool_selector 가 pending_tool_call=None 을 낸 경우(= 현재 tool 목록으로 답할 수
    없는 질문) 고정 placeholder 대신 Solar 가 **사용 가능한 tool 목록을 컨텍스트로**
    "이런 표현으로 바꾸면 답할 수 있어요" 역제안을 생성한다.

    사용자 체감상 기존의 "적합한 도구를 찾지 못했어요" 고정 메시지 → "아 이렇게 물어보면
    되는구나" 가 되도록 자연어 가이드를 돌려준다. 에이전트가 "LLM 이 없는 것처럼 뻣뻣하다"
    는 피드백의 해소책.

    실패 시 graceful — Solar 예외가 나도 response_text 에 짧은 안내를 넣는다.
    """
    admin_role = state.get("admin_role", "") or ""
    user_message = state.get("user_message", "") or ""

    if not admin_role or not user_message.strip():
        # 관리자 아님 혹은 빈 발화 — response_formatter 가 기본 placeholder 로 처리하도록 pass
        return {}

    # 허용된 tool 카탈로그 (이름 + 설명 요약 100자) 를 프롬프트에 주입
    allowed = list_tools_for_role(admin_role)
    if not allowed:
        # 권한이 전혀 없는 상태 — response_formatter 의 query placeholder 에 위임
        return {}

    catalog_lines = []
    for spec in allowed:
        # 설명은 150자로 컷 + tool 이름 (LLM 이 한국어로 의역하도록 지시하되 내부적으론 참고용)
        desc = (spec.description or "").strip().replace("\n", " ")
        if len(desc) > 150:
            desc = desc[:150] + "..."
        catalog_lines.append(f"- {spec.name} — {desc}")
    tool_catalog = "\n".join(catalog_lines)

    start = time.perf_counter()
    try:
        llm = get_solar_api_llm(temperature=0.4)
        system = SystemMessage(content=SMART_FALLBACK_SYSTEM_PROMPT)
        human = HumanMessage(content=SMART_FALLBACK_HUMAN_PROMPT.format(
            admin_role=admin_role,
            user_message=user_message,
            tool_catalog=tool_catalog,
        ))
        response = await guarded_ainvoke(
            llm, [system, human],
            model="solar_api",
            request_id="admin_smart_fallback",
        )
        raw_text = getattr(response, "content", None) or str(response)
        # narrator 와 동일한 sanitize 재사용 — 메타/구분선/자기-인용 제거
        text = _sanitize_narrator_output(raw_text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "admin_smart_fallback_generated",
            length=len(text),
            catalog_size=len(allowed),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"response_text": text or _PLACEHOLDER_MESSAGES.get("query", "")}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "admin_smart_fallback_failed",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed_ms, 1),
        )
        # 실패 시 기존 query placeholder 로 폴백
        return {"response_text": _PLACEHOLDER_MESSAGES.get("query", "")}
