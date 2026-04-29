"""
관리자 AI 에이전트 LangGraph StateGraph 구성 + SSE 실행 인터페이스.

설계서: docs/관리자_AI에이전트_v3_재설계.md §2 (ReAct 그래프), §3 (SSE 이벤트)

Phase D v3 범위 (2026-04-23):
    START → context_loader → intent_classifier ──┐
                                                  │
    ┌────────── smalltalk ─────────────────┤
    ▼                                       │ stats/query/action
    smalltalk_responder                     ▼
    │                                tool_selector ◀─────────────┐
    │                                       │                     │ continue
    │                         pending=None  │ pending_tool_call   │
    │                                ▼      ▼                     │
    │                  smart_fallback  tool_executor              │
    │                  _responder           ▼                     │
    │                       │           observation ──────────────┘
    │                       │               │
    │                       │               ├─ *_draft  → draft_emitter
    │                       │               ├─ goto_*   → navigator
    │                       │               └─ finish/max_hops → narrator
    │                       │               │
    │                       │               ▼
    │                       └──────→ narrator → response_formatter → END
    │                                                ▲
    └────────────────────────────────────────────────┘

변경 이력 (v2 → v3):
- risk_gate 노드 제거 (실제 쓰기 tool 없음, HITL 불필요)
- observation / draft_emitter / navigator 신규 노드 추가
- ReAct 루프: tool_executor → observation → (tool_selector | draft_emitter | navigator | narrator)
- SSE 이벤트 2종 신규: form_prefill, navigation
- HITL interrupt 감지 블록 보존 (v3 에서 발동 안 함 — risk_gate 제거로 snapshot.next 항상 빔)

SSE 이벤트 (v3 발행 목록):
- session, status, tool_call (매 hop), tool_result (매 hop), token, done, error
- form_prefill (draft_emitter 완료 시)
- navigation (navigator 완료 시)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from monglepick.config import settings

from monglepick.agents.admin_assistant.models import (
    AdminAssistantState,
    AdminIntent,
    ToolCall,
    ensure_intent,
    ensure_tool_call,
)
from monglepick.agents.admin_assistant.nodes import (
    MAX_HOPS,
    context_loader,
    draft_emitter,
    intent_classifier,
    narrator,
    navigator,
    observation,
    response_formatter,
    smalltalk_responder,
    smart_fallback_responder,
    tool_executor,
    tool_selector,
)
from monglepick.api.admin_backend_client import AdminApiResult

logger = structlog.get_logger()


# ============================================================
# 라우팅 — Step 1 에서는 smalltalk vs 그 외 2분기
# ============================================================

def route_after_intent(state: AdminAssistantState) -> str:
    """
    Intent 분류 이후 분기 (Step 2 확장).

    - admin_role 이 비어있으면 → response_formatter 직행 (차단 메시지).
    - smalltalk → smalltalk_responder.
    - **stats** → tool_selector (실제 Admin Stats API 호출 경로).
    - query/action/report/sql → response_formatter (현재 placeholder; Step 3+ 에서 확장).

    v2 설계에서 sql 은 영구 미지원이라 placeholder. query/action/report 는 Tier 1/2/3
    tool 추가 후 순차적으로 tool_selector 에 붙는다.
    """
    admin_role = state.get("admin_role", "") or ""
    if not admin_role:
        logger.info("route_after_intent_blocked", reason="no_admin_role")
        return "response_formatter"

    # MemorySaver 복원 시 dict 로 변환된 경우도 AdminIntent 로 되살린다.
    intent = ensure_intent(state.get("intent"))
    kind = intent.kind if intent is not None else "smalltalk"

    if kind == "smalltalk":
        return "smalltalk_responder"
    # Step 2: stats → tool_selector
    # Step 4(2026-04-23): query 도 tool_selector.
    # Step 5a(2026-04-23): action 도 tool_selector — HITL(risk_gate) 게이트가 있어 안전.
    # Phase 4(2026-04-27): report 도 tool_selector. 여러 read tool 을 ReAct 루프로 묶어
    #   종합 요약(narrator) + 표(table_data) + 화면 이동(navigation) 으로 답한다.
    #   기존 "Phase 4 예정" placeholder 응답 제거.
    if kind in ("stats", "query", "action", "report"):
        return "tool_selector"
    # sql 만 영구 미지원 placeholder.
    return "response_formatter"


def route_after_tool_select(state: AdminAssistantState) -> str:
    """
    tool_selector 이후 분기 (v3 Phase D).

    v3 변경점:
    - risk_gate 제거 — tool_executor 로 직행.
    - finish_task 가 선택된 경우 tool_executor 를 건너뛰고 narrator 로 직행.
      (finish_task 는 가상 tool 로 실제 Backend 호출이 없으므로 executor 불필요)
    - pending_tool_call 이 None 이면 smart_fallback_responder.

    흐름:
      pending=finish_task → narrator
      pending=실제 tool → tool_executor → observation → ...
      pending=None → smart_fallback_responder
    """
    call = ensure_tool_call(state.get("pending_tool_call"))
    if call is None:
        logger.info("route_after_tool_select_no_tool_to_fallback")
        return "smart_fallback_responder"

    if call.tool_name == "finish_task":
        # 가상 tool — executor 건너뛰고 narrator 로 직행
        logger.info(
            "route_after_tool_select_finish_task",
            reason=call.arguments.get("reason", ""),
        )
        return "narrator"

    logger.info(
        "route_after_tool_select",
        tool_name=call.tool_name,
        tier=call.tier,
    )
    return "tool_executor"


def route_after_observation(state: AdminAssistantState) -> str:
    """
    observation 이후 분기 (v3 Phase D 신규).

    ReAct 루프의 핵심 분기점. tool_executor 결과를 observation 이 기록한 뒤,
    다음 hop 을 계속할지 종결 경로로 나갈지 결정한다.

    우선순위:
    1. iteration_count >= MAX_HOPS → narrator (강제 종결, 토큰 비용/무한 루프 방어)
    2. tool_call_history 가 비어있음 → narrator (방어 코드, 정상 흐름에서는 발생 안 함)
    3. 마지막 tool 이 finish_task → narrator (LLM 이 "충분하다" 고 판단)
    4. 마지막 tool 이 *_draft → draft_emitter (form_prefill SSE 발행 후 narrator)
    5. 마지막 tool 이 goto_* → navigator (navigation SSE 발행 후 narrator)
    6. 그 외 read tool → tool_selector (다음 hop 계속)
    """
    hop_count: int = state.get("iteration_count") or 0

    # 1) MAX_HOPS 도달 → 강제 종결
    if hop_count >= MAX_HOPS:
        logger.info(
            "route_after_observation_max_hops",
            hop_count=hop_count,
            max_hops=MAX_HOPS,
        )
        return "narrator"

    # tool_call_history 에서 마지막 항목 꺼내기
    history = state.get("tool_call_history") or []
    if not history:
        logger.debug("route_after_observation_empty_history")
        return "narrator"

    last_raw = history[-1]
    # MemorySaver 직렬화로 dict 화된 경우도 처리
    last_call = ensure_tool_call(last_raw)
    last_name: str = last_call.tool_name if last_call else (
        last_raw.get("tool_name", "") if isinstance(last_raw, dict) else ""
    )

    # 2) finish_task → narrator
    if last_name == "finish_task":
        logger.info("route_after_observation_finish_task")
        return "narrator"

    # 3) *_draft → draft_emitter
    if last_name.endswith("_draft"):
        logger.info("route_after_observation_draft", tool_name=last_name)
        return "draft_emitter"

    # 4) goto_* → navigator
    if last_name.startswith("goto_"):
        logger.info("route_after_observation_navigate", tool_name=last_name)
        return "navigator"

    # 5) 그 외 read tool → tool_selector (다음 hop)
    logger.info(
        "route_after_observation_continue",
        tool_name=last_name,
        hop_count=hop_count,
    )
    return "tool_selector"


# ============================================================
# 그래프 빌드
# ============================================================

def build_admin_assistant_graph():
    """
    Admin Assistant StateGraph 구성 + 컴파일 (v3 Phase D).

    v3 변경점:
    - risk_gate 노드 제거. tool_selector → tool_executor 직행.
    - observation / draft_emitter / navigator 신규 노드 추가.
    - ReAct 루프: tool_executor → observation → route_after_observation →
        (tool_selector | draft_emitter | navigator | narrator)
    - draft_emitter / navigator → narrator → response_formatter → END
    - MemorySaver checkpointer 유지 (v3 에서 interrupt 발동 안 하지만 세션 유지 용도 보존).

    노드 수: 11개 (context_loader, intent_classifier, smalltalk_responder, tool_selector,
             tool_executor, observation, draft_emitter, navigator, narrator,
             smart_fallback_responder, response_formatter)
    """
    graph = StateGraph(AdminAssistantState)

    # ── 기존 노드 ──
    graph.add_node("context_loader", context_loader)
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("smalltalk_responder", smalltalk_responder)
    graph.add_node("tool_selector", tool_selector)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("narrator", narrator)
    graph.add_node("smart_fallback_responder", smart_fallback_responder)
    graph.add_node("response_formatter", response_formatter)

    # ── v3 Phase D 신규 노드 ──
    graph.add_node("observation", observation)        # tool_executor 결과 누적
    graph.add_node("draft_emitter", draft_emitter)   # *_draft tool 결과 → form_prefill
    graph.add_node("navigator", navigator)            # goto_* tool 결과 → navigation

    # ── 고정 엣지 ──
    graph.add_edge(START, "context_loader")
    graph.add_edge("context_loader", "intent_classifier")

    # intent_classifier → (smalltalk_responder | tool_selector | response_formatter)
    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "smalltalk_responder": "smalltalk_responder",
            "tool_selector": "tool_selector",
            "response_formatter": "response_formatter",
        },
    )
    graph.add_edge("smalltalk_responder", "response_formatter")

    # tool_selector → (tool_executor | narrator | smart_fallback_responder)
    # finish_task 선택 시 narrator 직행, 일반 tool 은 tool_executor, 매칭 실패는 fallback
    graph.add_conditional_edges(
        "tool_selector",
        route_after_tool_select,
        {
            "tool_executor": "tool_executor",
            "narrator": "narrator",
            "smart_fallback_responder": "smart_fallback_responder",
        },
    )
    graph.add_edge("smart_fallback_responder", "response_formatter")

    # tool_executor → observation (항상)
    graph.add_edge("tool_executor", "observation")

    # observation → (tool_selector | draft_emitter | navigator | narrator)
    graph.add_conditional_edges(
        "observation",
        route_after_observation,
        {
            "tool_selector": "tool_selector",
            "draft_emitter": "draft_emitter",
            "navigator": "navigator",
            "narrator": "narrator",
        },
    )

    # draft_emitter / navigator → narrator (form_prefill/navigation 세팅 후 자연어 안내)
    graph.add_edge("draft_emitter", "narrator")
    graph.add_edge("navigator", "narrator")

    # narrator → response_formatter → END
    graph.add_edge("narrator", "response_formatter")
    graph.add_edge("response_formatter", END)

    # Step 7c (2026-04-27): Checkpointer — env ADMIN_REDIS_CHECKPOINTER_ENABLED=true 면 RedisSaver,
    # 그 외에는 기존 MemorySaver 유지 (단일 프로세스 인스턴스 / 테스트 환경).
    # RedisSaver 는 다중 Agent 인스턴스(부하 분산) 간 세션 공유 + 영속성 제공.
    # asetup() 은 main.py lifespan 의 setup_admin_assistant_checkpointer() 에서 1회 호출.
    checkpointer, kind = _make_admin_checkpointer()

    # 모듈 변수에 저장 — startup hook 이 같은 인스턴스에 asetup() 호출하기 위함.
    global _admin_assistant_saver
    _admin_assistant_saver = checkpointer

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(
        "admin_assistant_graph_compiled",
        node_count=11,
        checkpointer=kind,
        version="v3_phase_d",
    )
    return compiled


# ============================================================
# Step 7c — Checkpointer 팩토리 + lifespan setup
# ============================================================

#: 모듈 레벨 saver 인스턴스 보관 — `setup_admin_assistant_checkpointer()` 가 같은 인스턴스에
#: asetup() 호출하도록. None 이면 그래프가 아직 컴파일되지 않은 상태.
_admin_assistant_saver: Any | None = None


def _is_redis_checkpointer_enabled() -> bool:
    """ADMIN_REDIS_CHECKPOINTER_ENABLED 환경변수 — true/1/yes 외에는 비활성."""
    return os.getenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", "false").lower() in ("true", "1", "yes")


def _make_admin_checkpointer() -> tuple[Any, str]:
    """
    환경변수에 따라 RedisSaver 또는 MemorySaver 반환.

    RedisSaver 키 prefix 는 admin_assistant 전용 네임스페이스로 격리해 다른 Agent 의
    체크포인트와 충돌하지 않게 한다. asetup() (Redis Search 인덱스 생성) 은 별도 함수
    `setup_admin_assistant_checkpointer()` 가 FastAPI lifespan 에서 호출.

    Redis 패키지 import 실패 시(개발 환경에서 패키지 미설치) MemorySaver 로 안전 폴백.

    Returns:
        (saver_instance, kind_label) — kind 는 로그/메트릭 용 ("memory" | "redis").
    """
    if not _is_redis_checkpointer_enabled():
        return MemorySaver(), "memory"

    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    except ImportError as e:
        logger.warning(
            "admin_redis_checkpointer_import_failed_fallback_memory",
            error=str(e),
        )
        return MemorySaver(), "memory"

    try:
        saver = AsyncRedisSaver(
            redis_url=settings.REDIS_URL,
            checkpoint_prefix="admin_assistant:checkpoint",
            checkpoint_blob_prefix="admin_assistant:cp_blob",
            checkpoint_write_prefix="admin_assistant:cp_write",
        )
    except Exception as e:
        logger.warning(
            "admin_redis_checkpointer_init_failed_fallback_memory",
            error=str(e),
            error_type=type(e).__name__,
        )
        return MemorySaver(), "memory"

    logger.info("admin_redis_checkpointer_initialized", redis_url=settings.REDIS_URL)
    return saver, "redis"


async def setup_admin_assistant_checkpointer() -> None:
    """
    FastAPI lifespan 에서 1회 호출 — RedisSaver 의 Redis Search 인덱스 생성.

    MemorySaver 인 경우 no-op. RedisSaver `asetup()` 는 idempotent — 이미 인덱스가
    있어도 안전하게 통과. 실패 시 경고만 남기고 앱 기동 차단하지 않음 (체크포인트 없이도
    Agent 는 동작 — interrupt 미사용 v3 그래프).
    """
    saver = _admin_assistant_saver
    if saver is None:
        logger.warning("admin_checkpointer_not_initialized")
        return

    asetup = getattr(saver, "asetup", None)
    if asetup is None:
        # MemorySaver — setup 불필요
        logger.info("admin_checkpointer_setup_skipped", kind=type(saver).__name__)
        return

    try:
        await asetup()
        logger.info("admin_checkpointer_setup_done", kind=type(saver).__name__)
    except Exception as e:
        # 실패 시 운영자가 인지하도록 ERROR 레벨 + 앱은 계속 기동 (체크포인트 미사용 동작)
        logger.error(
            "admin_checkpointer_setup_failed",
            kind=type(saver).__name__,
            error=str(e),
            error_type=type(e).__name__,
        )


# 모듈 레벨 싱글턴 — 컴파일 1회
admin_assistant_graph = build_admin_assistant_graph()


# ============================================================
# 노드 → 한국어 status 메시지
# ============================================================

_NODE_STATUS_MESSAGES: dict[str, str] = {
    "context_loader": "관리자 정보를 확인하고 있어요...",
    "intent_classifier": "요청 의도를 분석하고 있어요...",
    "smalltalk_responder": "답변을 준비하고 있어요...",
    "tool_selector": "적합한 도구를 고르고 있어요...",
    "tool_executor": "관리자 API를 호출하고 있어요...",
    # v3 Phase D 신규 노드
    "observation": "결과를 검토하고 있어요...",
    "draft_emitter": "폼 내용을 정리하고 있어요...",
    "navigator": "관리 화면 링크를 준비하고 있어요...",
    "narrator": "결과를 정리해 설명하고 있어요...",
    "smart_fallback_responder": "답변 방향을 고민하고 있어요...",
    "response_formatter": "응답을 정리하고 있어요...",
    # v3 에서 제거된 노드 — 하위 호환 메시지 보존 (SSE status 가 이 키를 참조하는 경우 대비)
    "risk_gate": "실행 전 안전 점검 중이에요...",  # v3 미사용
}


# ============================================================
# SSE 유틸
# ============================================================

_KEEPALIVE_INTERVAL_SEC = 15
_SENTINEL = object()


def _format_sse_event(event_type: str, data: dict) -> dict:
    """
    sse_starlette 호환 dict 포맷.

    Chat Agent graph.py 의 _format_sse_event 와 동일 규약.
    EventSourceResponse 가 {"event": ..., "data": ...} dict 를 받으면
    "event: {type}\\ndata: {json}\\n\\n" 로 직렬화한다.
    """
    return {"event": event_type, "data": json.dumps(data, ensure_ascii=False)}


def state_snapshot_tool_call(merged_state: dict) -> ToolCall | None:
    """
    merged state 에서 현재 `pending_tool_call` 을 안전히 꺼낸다.

    SSE 발행 시점에는 `updates` 에만 최신 값이 있고 `final_state` 는 누적본이라,
    tool_executor 완료 이벤트 쪽에서 tool 이름을 참조하려면 이 헬퍼로 꺼낸다.
    MemorySaver 직렬화로 dict 화된 경우도 ensure_tool_call 로 복원한다.
    """
    return ensure_tool_call(merged_state.get("pending_tool_call"))


# ============================================================
# Phase 4 (2026-04-27) — table_data SSE 빌더
# ============================================================
# 행 갯수가 일정 임계치 이상인 read 결과는 narrator 자연어 외에 **표** 로도 보여준다.
# Client 의 TableDataCard 가 이 payload 를 카드 형태로 렌더하고, navigate_path 가 있으면
# "전체 보기" 버튼으로 해당 관리 페이지로 이동시킨다.

# 발행 임계치 — row_count 가 이 값 이상일 때만 table_data 발행 (소량은 narrator 자연어로 충분).
_TABLE_DATA_MIN_ROWS = 3
# 카드에 표시할 최대 행 수 (Client 카드의 시각적 부담 + payload 크기 균형).
_TABLE_DATA_SAMPLE_ROWS = 10
# 카드에 표시할 최대 컬럼 수 (가로 스크롤 회피 + payload 크기 균형).
_TABLE_DATA_MAX_COLS = 6
# 셀 내 문자열 최대 길이 — 긴 본문은 잘라 줄바꿈 방지.
_TABLE_DATA_MAX_CELL_LEN = 80

# tool 이름 → "전체 보기" 이동 경로 매핑. read tool 만 등록.
# Page 응답을 돌려주는 list 류 tool 위주. 매핑 없는 tool 은 "전체 보기" 버튼 미렌더.
#
# **2026-04-27 (Phase 4 후속) 매핑 정합성 정정:**
# Admin Client 의 실제 tab id 와 어긋나 있던 매핑 10건을 페이지 진실 원본에 맞춰 수정.
# (검증 출처: SettingsPage TABS=[terms/banners/admins], AiOpsPage TABS=[trigger/history/
# chatlog/review-verify/chat-suggestions], PaymentPage TABS=[orders/orders_sub/orders_point/
# subscription/point/items/point_pack/reward_policy], SupportPage TABS=[notice/faq/help/
# ticket], ContentEventsPage SUB_TABS=[roadmap_course/quiz/worldcup_candidate/ocr_event],
# BoardPage TABS=[moderation/reports/toxicity/posts/reviews/categories])
#
# 변경 요약:
#   subscriptions_list  : tab=subscriptions   → tab=subscription (no s)
#   point_histories     : tab=point-history   → tab=point
#   point_items         : tab=point-items     → tab=items
#   tickets_list        : tab=tickets         → tab=ticket (no s)
#   quizzes_list        : /admin/ai?tab=quiz  → /admin/content-events?tab=quiz (위치 자체 이동)
#   review_verifications_list : tab=review-verifications → tab=review-verify
#   chatbot_sessions_list     : tab=chatbot-sessions     → tab=chatlog
#   banners_list        : /admin/content-events?tab=banner → /admin/settings?tab=banners
#   chat_suggestions_list : /admin/settings?tab=chat-sugg → /admin/ai?tab=chat-suggestions
#   audit_logs_list     : /admin/settings?tab=audit → 매핑 제거 (audit 탭 없음. None 으로 fallback)
_TABLE_NAVIGATE_PATHS: dict[str, str] = {
    # 콘텐츠 모더레이션 (BoardPage)
    "reports_list": "/admin/board?tab=reports",
    "toxicity_list": "/admin/board?tab=toxicity",
    "posts_list": "/admin/board?tab=posts",
    "reviews_list": "/admin/board?tab=reviews",
    # 사용자 (UsersPage — base path 진입 시 기본 탭=list)
    "users_list": "/admin/users",
    # 결제 (PaymentPage — tab id 가 단수 형태)
    "orders_list": "/admin/payment?tab=orders",
    "subscriptions_list": "/admin/payment?tab=subscription",
    "point_histories": "/admin/payment?tab=point",
    "point_items": "/admin/payment?tab=items",
    # 고객센터 (SupportPage — ticket 단수)
    "tickets_list": "/admin/support?tab=ticket",
    "faqs_list": "/admin/support?tab=faq",
    "help_articles_list": "/admin/support?tab=help",
    "notices_list": "/admin/support?tab=notice",
    # AI 운영 (AiOpsPage / ContentEventsPage 분산)
    # quiz 는 ContentEventsPage 소속이라 다른 페이지로 이동.
    "quizzes_list": "/admin/content-events?tab=quiz",
    "review_verifications_list": "/admin/ai?tab=review-verify",
    "chatbot_sessions_list": "/admin/ai?tab=chatlog",
    # 설정/감사 (SettingsPage)
    # audit_logs_list 는 settings 에 audit 탭이 없어 매핑 제거. navigate_path=None 폴백
    # → TableDataCard 가 "전체 보기" 버튼을 미렌더 (혼란 방지).
    "admins_list": "/admin/settings?tab=admins",
    "terms_list": "/admin/settings?tab=terms",
    # banner CRUD 는 SettingsPage 의 banners 탭이 담당 (BannerTab.jsx 위치도 settings 산하).
    "banners_list": "/admin/settings?tab=banners",
    # chat_suggestions 는 AiOpsPage 의 chat-suggestions 탭 (ChatSuggestionTab.jsx 위치 ai 산하).
    "chat_suggestions_list": "/admin/ai?tab=chat-suggestions",
}


def _coerce_cell_value(value: Any) -> Any:
    """
    표 셀에 들어갈 단일 값을 직렬화 가능한 형태로 정규화한다.

    - 스칼라(str/int/float/bool/None)는 그대로 유지하되 너무 긴 문자열은 자른다.
    - dict/list 는 짧은 JSON 문자열로 직렬화 후 자른다 (객체 미리보기용).
    - 그 외(객체)는 str() 캐스트 후 자른다.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > _TABLE_DATA_MAX_CELL_LEN:
            return value[:_TABLE_DATA_MAX_CELL_LEN] + "…"
        return value
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        encoded = str(value)
    if len(encoded) > _TABLE_DATA_MAX_CELL_LEN:
        return encoded[:_TABLE_DATA_MAX_CELL_LEN] + "…"
    return encoded


# ============================================================
# Phase 4 후속 (2026-04-28) — chart_data SSE 빌더
# ============================================================
# 시계열 통계 tool 의 결과를 차트로 렌더하기 위한 chart_data SSE payload 빌더.
# table_data 와 동시에 발행 가능 (보고서 본문 + 차트 + 표 3종 병행).
#
# 발행 전제:
# - tool 이름이 _CHART_TOOL_SPECS 에 등록되어 있을 것 (whitelisted).
# - 응답 data 가 dict 이고 명시된 data_key 안에 list[dict] 가 있을 것.
# - 시계열 list 가 _CHART_DATA_MIN_POINTS 이상 (작은 시계열은 narrator 본문으로 충분).
#
# 매핑 외 tool 은 자동 스킵 (None 반환). table_data 와 달리 chart 는 필드 의미 추론이
# 어려워 보수적으로 등록된 시계열만 처리한다.

# 발행 임계치 — 시계열 데이터포인트 수가 이 값 이상일 때만 차트 발행.
_CHART_DATA_MIN_POINTS = 3
# 차트 카드에 표시할 최대 데이터포인트 수 (페이로드 크기 + 시각적 부담).
_CHART_DATA_MAX_POINTS = 90

#: tool 이름 → 차트 메타. 키:
#:   - data_key      : 시계열 list 가 들어있는 응답 dict 키 (예: "trends", "dailyRevenue").
#:                     단계적 접근(`data["trends"]`, `data["a"]["b"]`) 은 미지원 (단일 키).
#:   - x_key         : 데이터포인트 dict 에서 X축 라벨로 쓸 키 (예: "date").
#:   - series        : [(label, key), ...] — 차트에 그릴 시리즈 정의. label 은 한국어 표시명,
#:                     key 는 데이터포인트 dict 의 숫자 필드 이름.
#:   - title         : 카드 헤더 라벨.
#:   - chart_type    : "line" | "bar" — Client recharts 컴포넌트 분기 힌트.
#:   - unit          : (선택) Y축 값 단위 — Client tooltip 표시 ("명", "원", "회" 등).
#:   - navigate_path : (선택) "전체 화면에서 보기" 버튼 경로 (Admin Client 통계 탭).
_CHART_TOOL_SPECS: dict[str, dict] = {
    # 통계 - 일별 추이 (DAU/신규/리뷰/게시글 4시리즈)
    "stats_trends": {
        "data_key": "trends",
        "x_key": "date",
        "series": [
            ("DAU", "dau"),
            ("신규 가입", "newUsers"),
            ("리뷰", "reviews"),
            ("게시글", "posts"),
        ],
        "title": "일별 추이",
        "chart_type": "line",
        "unit": "건",
        "navigate_path": "/admin/stats?tab=overview",
    },
    # 대시보드 - 일별 추이 (신규 가입/결제액/AI 채팅 3시리즈, 결제액 단위 다름 주의)
    "dashboard_trends": {
        "data_key": "trends",
        "x_key": "date",
        "series": [
            ("신규 가입", "newUsers"),
            ("결제액(원)", "paymentAmount"),
            ("AI 채팅 요청", "chatRequests"),
        ],
        "title": "대시보드 일별 추이",
        "chart_type": "line",
        "unit": "혼합",
        "navigate_path": "/admin/dashboard",
    },
    # 매출 - 일별 매출 (단일 시리즈)
    "stats_revenue": {
        "data_key": "dailyRevenue",
        "x_key": "date",
        "series": [
            ("매출", "amount"),
        ],
        "title": "일별 매출 추이",
        "chart_type": "bar",
        "unit": "원",
        "navigate_path": "/admin/stats?tab=revenue",
    },
    # 2026-04-28 후속 — 분포 시각화 추가 (pie / bar 확장):
    # 구독 플랜 분포 — pie 차트. SubscriptionStatsResponse.plans 의 PlanDistribution 4종.
    # x_key 는 슬라이스 라벨(planName), 단일 series 의 key 는 슬라이스 값(count).
    # _build_chart_payload 가 동일 schema 로 처리해 ChartDataCard 의 chart_type 분기로 렌더.
    "stats_subscription": {
        "data_key": "plans",
        "x_key": "planName",
        "series": [
            ("활성 구독", "count"),
        ],
        "title": "구독 플랜별 분포",
        "chart_type": "pie",
        "unit": "건",
        "navigate_path": "/admin/payment?tab=subscription",
    },
    # 인기 검색어 Top-N — bar 차트. KeywordItem 의 keyword/searchCount.
    # 분포보다 랭킹 시각화가 직관적이라 bar 로 처리. limit 인자(default 20) 결과 그대로 사용.
    "stats_search_popular": {
        "data_key": "keywords",
        "x_key": "keyword",
        "series": [
            ("검색 수", "searchCount"),
        ],
        "title": "인기 검색어 Top",
        "chart_type": "bar",
        "unit": "회",
        "navigate_path": "/admin/stats?tab=search",
    },
    # 2026-04-28 후속2 — 분포 도구 3종 pie 차트 매핑:
    # 추천 장르 분포 — DistributionResponse{genres: [{genre, count, percentage}]}.
    # 한국어 장르명(genre) 을 슬라이스 라벨, 추천 건수(count) 를 값으로.
    "stats_recommendation_distribution": {
        "data_key": "genres",
        "x_key": "genre",
        "series": [
            ("추천 건수", "count"),
        ],
        "title": "추천 장르 분포",
        "chart_type": "pie",
        "unit": "건",
        "navigate_path": "/admin/stats?tab=recommendation",
    },
    # 포인트 유형별 분포 — PointTypeDistributionResponse{distribution: [{pointType, label, count, totalAmount, percentage}]}.
    # 한국어 라벨(label, 예: "활동 리워드", "AI 추천 사용") 을 슬라이스 라벨로 사용 — pointType
    # 코드(earn/spend 등) 는 운영자 친화도 낮음. 값은 거래 건수(count).
    "stats_point_distribution": {
        "data_key": "distribution",
        "x_key": "label",
        "series": [
            ("거래 건수", "count"),
        ],
        "title": "포인트 유형별 분포",
        "chart_type": "pie",
        "unit": "건",
        "navigate_path": "/admin/stats?tab=point-economy",
    },
    # 등급별 사용자 분포 — GradeDistributionResponse{grades: [{gradeCode, gradeName, count, percentage}]}.
    # 한국어 등급명(gradeName, 예: "팝콘", "다이아몬드") 을 슬라이스 라벨, 사용자 수(count) 를 값으로.
    "stats_grade_distribution": {
        "data_key": "grades",
        "x_key": "gradeName",
        "series": [
            ("사용자 수", "count"),
        ],
        "title": "등급별 사용자 분포",
        "chart_type": "pie",
        "unit": "명",
        "navigate_path": "/admin/stats?tab=point-economy",
    },
    # 2026-04-28 후속2 (시계열 line 추가) — 일별 추이 3종.
    # 포인트 발행/소비/순유입 — PointTrendsResponse.trends[{date, issued, spent, netFlow}].
    # netFlow 가 음수일 수 있어 단위 "P" 표기 (Client tooltip 에서 표시).
    "stats_point_trends": {
        "data_key": "trends",
        "x_key": "date",
        "series": [
            ("발행", "issued"),
            ("소비", "spent"),
            ("순유입", "netFlow"),
        ],
        "title": "일별 포인트 흐름",
        "chart_type": "line",
        "unit": "P",
        "navigate_path": "/admin/stats?tab=point-economy",
    },
    # AI 세션/턴 추이 — AiSessionTrendsResponse.trends[{date, sessions, turns}].
    "stats_ai_session_trends": {
        "data_key": "trends",
        "x_key": "date",
        "series": [
            ("세션", "sessions"),
            ("턴 수", "turns"),
        ],
        "title": "AI 세션·턴 추이",
        "chart_type": "line",
        "unit": "건",
        "navigate_path": "/admin/stats?tab=ai-service",
    },
    # 커뮤니티 게시글/댓글/신고 추이 — CommunityTrendsResponse.trends[{date, posts, comments, reports}].
    # reports 시리즈가 다른 두 시리즈와 스케일이 차이 날 수 있지만 차트 자체는 정상 렌더.
    "stats_community_trends": {
        "data_key": "trends",
        "x_key": "date",
        "series": [
            ("게시글", "posts"),
            ("댓글", "comments"),
            ("신고", "reports"),
        ],
        "title": "커뮤니티 일별 추이",
        "chart_type": "line",
        "unit": "건",
        "navigate_path": "/admin/stats?tab=community",
    },
}


# 2026-04-28 (길 A v3 보강) — table_data / chart_data SSE 카드 dedup
# Client(TableDataCard / ChartDataCard) 가 카드 배열을 push 만 하면 같은 통계가
# 여러 번 호출됐을 때 화면에 카드가 누적된다. payload 에 dedup_key 를 실어 Client 가
# Map<dedup_key, card> 로 upsert 하도록 한다. 같은 tool 을 같은 인자로 다시 호출 시
# 동일 키 → 카드 1장으로 유지.

def _make_dedup_key(tool_name: str, arguments: dict | None) -> str:
    """
    tool_name + 정규화된 arguments 로 결정적 dedup_key 를 만든다.

    arguments 가 같은 dict 라도 키 순서가 다르면 다른 키가 되지 않게 sort_keys.
    값은 본문성 필드는 길이만 사용 (긴 문자열이 다를 수 있어도 같은 카드로 묶음).
    """
    if not arguments:
        return tool_name
    # 본문성 필드는 hash 부담 + 노출 위험으로 길이만 사용
    _heavy = {"content", "answer", "explanation", "body"}
    norm: dict = {}
    for k, v in arguments.items():
        if v is None:
            continue
        if k in _heavy and isinstance(v, str):
            norm[k] = f"len={len(v)}"
        else:
            norm[k] = v
    try:
        canonical = json.dumps(norm, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canonical = str(sorted(norm.items()))
    return f"{tool_name}:{canonical}"


def _build_chart_payload(
    tool_name: str,
    result: AdminApiResult,
    arguments: dict | None = None,
) -> dict | None:
    """
    AdminApiResult 의 등록된 시계열 응답에서 SSE chart_data payload 를 빌드한다.

    반환 None 의 경우:
    - tool 이 _CHART_TOOL_SPECS 에 없음 (whitelisted only)
    - ok=False
    - data 가 dict 가 아니거나 data_key 위치에 list 가 없음
    - 데이터포인트 수가 _CHART_DATA_MIN_POINTS 미만
    - 첫 데이터포인트가 dict 가 아니거나 x_key 가 없음

    payload 스키마:
        {
          "tool_name": str,
          "title": str,
          "chart_type": "line" | "bar",
          "unit": str,
          "x_axis": {"key": str, "values": list[str]},
          "series": list[{"name": str, "key": str, "data": list[number|None]}],
          "total_points": int,
          "truncated": bool,
          "navigate_path": str | None,
        }

    series.data 는 x_axis.values 와 길이·인덱스 1:1. 누락 값은 None 으로 채워 Client 가
    안전하게 라인 보간 처리할 수 있게 한다.
    """
    spec = _CHART_TOOL_SPECS.get(tool_name)
    if spec is None:
        return None
    if not isinstance(result, AdminApiResult) or not result.ok:
        return None
    data = result.data
    if not isinstance(data, dict):
        return None

    points = data.get(spec["data_key"])
    if not isinstance(points, list) or len(points) < _CHART_DATA_MIN_POINTS:
        return None
    if not isinstance(points[0], dict):
        return None
    x_key = spec["x_key"]
    if x_key not in points[0]:
        return None

    capped = points[:_CHART_DATA_MAX_POINTS]
    x_values: list[str] = [str(p.get(x_key, "")) for p in capped if isinstance(p, dict)]

    series_payload: list[dict] = []
    for label, key in spec["series"]:
        series_data: list = []
        for p in capped:
            if not isinstance(p, dict):
                series_data.append(None)
                continue
            v = p.get(key)
            # 숫자/None 만 통과. 문자열 등은 None 으로 정규화 (Client 보간 처리).
            if v is None or isinstance(v, (int, float)):
                series_data.append(v)
            else:
                series_data.append(None)
        # 시리즈 전체가 None 만 있으면 제외 (백엔드 응답에 키 자체가 빠진 경우).
        if any(v is not None for v in series_data):
            series_payload.append({
                "name": label,
                "key": key,
                "data": series_data,
            })

    if not series_payload:
        # 매핑된 모든 series 키가 응답에 없으면 차트 의미 없음 → 스킵.
        return None

    return {
        "tool_name": tool_name,
        "dedup_key": _make_dedup_key(tool_name, arguments),
        "title": spec["title"],
        "chart_type": spec["chart_type"],
        "unit": spec.get("unit", ""),
        "x_axis": {"key": x_key, "values": x_values},
        "series": series_payload,
        "total_points": len(points),
        "truncated": len(points) > len(capped),
        "navigate_path": spec.get("navigate_path"),
    }


def _build_table_payload(
    tool_name: str,
    result: AdminApiResult,
    arguments: dict | None = None,
) -> dict | None:
    """
    AdminApiResult 의 list/Page 응답에서 SSE table_data payload 를 빌드한다.

    반환 None 의 경우:
    - ok=False
    - data 가 list 도 Page(content list 를 가진 dict) 도 아님
    - row 수가 _TABLE_DATA_MIN_ROWS 미만 (narrator 자연어로 충분)
    - 첫 행이 dict 가 아니라 컬럼 추출 불가 (스칼라 list 등)

    payload 스키마:
        {
          "tool_name": str,
          "title": str,                          # 카드 헤더 ("리뷰 목록", "신고 목록" 등)
          "columns": list[str],                  # 첫 행 키 기반, 최대 _TABLE_DATA_MAX_COLS 개
          "rows": list[dict],                    # 최대 _TABLE_DATA_SAMPLE_ROWS 행
          "total_rows": int,
          "truncated": bool,
          "navigate_path": str | None,           # "전체 보기" 버튼 링크 (등록된 tool 만)
        }
    """
    if not isinstance(result, AdminApiResult) or not result.ok:
        return None

    data: Any = result.data
    rows_source: list[Any] | None = None
    total_rows: int | None = None

    if isinstance(data, list):
        rows_source = data
        total_rows = len(data)
    elif isinstance(data, dict):
        # Spring Data Page 응답 — content/totalElements 사용
        content = data.get("content")
        if isinstance(content, list):
            rows_source = content
            te = data.get("totalElements")
            total_rows = te if isinstance(te, int) else len(content)

    if rows_source is None:
        return None
    if total_rows is None:
        total_rows = len(rows_source)
    if total_rows < _TABLE_DATA_MIN_ROWS:
        return None
    if not rows_source or not isinstance(rows_source[0], dict):
        # 스칼라 list 같은 케이스는 표로 의미가 없어 스킵
        return None

    # 첫 행 키를 컬럼 후보로. 식별성 있는 키(id/createdAt 등) 가 있으면 앞으로 정렬해도
    # 좋지만 단순화 위해 첫 행 자연 순서 그대로 _TABLE_DATA_MAX_COLS 개만 자른다.
    first_row = rows_source[0]
    columns = list(first_row.keys())[:_TABLE_DATA_MAX_COLS]

    sample_rows: list[dict] = []
    for row in rows_source[:_TABLE_DATA_SAMPLE_ROWS]:
        if not isinstance(row, dict):
            continue
        sample_rows.append({c: _coerce_cell_value(row.get(c)) for c in columns})

    truncated = total_rows > len(sample_rows)

    return {
        "tool_name": tool_name,
        "dedup_key": _make_dedup_key(tool_name, arguments),
        "title": tool_name.replace("_", " ").strip().title(),
        "columns": columns,
        "rows": sample_rows,
        "total_rows": total_rows,
        "truncated": truncated,
        "navigate_path": _TABLE_NAVIGATE_PATHS.get(tool_name),
    }


# ============================================================
# SSE 스트리밍 실행
# ============================================================

async def run_admin_assistant(
    admin_id: str,
    admin_role: str,
    admin_jwt: str,
    session_id: str,
    user_message: str,
    resume_payload: dict | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Admin Assistant 를 SSE 스트리밍 모드로 실행한다.

    Chat Agent run_chat_agent() 와 동일한 asyncio.Queue + keepalive 패턴.
    - 15초 동안 이벤트가 없으면 keepalive status 발행 (SSE 연결 유지)
    - 그래프 완료 시 done / 에러 시 error → done 순 발행
    - Step 5a: HITL interrupt 발생 시 `confirmation_required` 발행 후 done 없이 스트림 종료

    Args:
        admin_id: JWT sub (관리자 user_id)
        admin_role: 정규화된 AdminRoleEnum 문자열 (빈 문자열이면 차단 메시지)
        admin_jwt: Backend forwarding 용 JWT 원문 (tool_executor 에서 사용)
        session_id: 세션 ID (빈 문자열이면 자동 생성). HITL 재개 시에는 반드시 기존 세션 ID 필수.
        user_message: 관리자 발화 (resume 모드에서는 빈 문자열/무시됨)
        resume_payload: 재개용 `Command(resume=...)` 에 들어갈 dict.
            None 이면 신규 대화, dict 면 `/resume` 경로로 기존 thread_id 에서 재개한다.
            기대 shape: {"decision": "approve"|"reject", "comment": str}

    Yields:
        sse_starlette 호환 dict. {"event": ..., "data": json_string}
    """
    graph_start = time.perf_counter()

    # 세션 ID 자동 생성 (resume 모드에서는 호출 측이 기존 session_id 전달해야 함)
    if not session_id:
        session_id = str(uuid.uuid4())

    # LangGraph checkpointer 구분 키 — 세션 단위로 체크포인트 네임스페이스 분리
    graph_config = {"configurable": {"thread_id": session_id}}

    # resume 이면 initial_state 대신 Command(resume=...) 를 astream 에 전달.
    # 신규 대화는 기존처럼 state dict.
    is_resume = resume_payload is not None
    initial_state: AdminAssistantState | None
    if is_resume:
        initial_state = None  # 실행 입력은 Command 로 대체
    else:
        initial_state = {
            "admin_id": admin_id,
            "admin_role": admin_role,
            "admin_jwt": admin_jwt,
            "session_id": session_id,
            "user_message": user_message,
            "history": [],
        }

    logger.info(
        "admin_assistant_stream_start",
        admin_id=admin_id or "(anonymous)",
        admin_role=admin_role or "(blank)",
        session_id=session_id,
        is_resume=is_resume,
        message_preview=("(resume)" if is_resume else user_message[:100]),
    )

    # session 이벤트 발행
    yield _format_sse_event("session", {"session_id": session_id})

    queue: asyncio.Queue = asyncio.Queue()
    # resume 모드면 첫 노드는 risk_gate (interrupt 직후 지점) — 그 외 신규는 context_loader
    current_phase = "risk_gate" if is_resume else "context_loader"
    current_message = _NODE_STATUS_MESSAGES[current_phase]
    final_state: dict = {}

    async def _run_graph_to_queue():
        """LangGraph astream → Queue → SSE 소비 패턴."""
        try:
            # 신규 대화: initial_state dict / 재개: Command(resume=...).
            graph_input = (
                Command(resume=resume_payload) if is_resume else initial_state
            )
            async for event in admin_assistant_graph.astream(
                graph_input,
                config=graph_config,
                stream_mode="updates",
            ):
                await queue.put(event)
            await queue.put(_SENTINEL)
        except Exception as e:
            await queue.put(e)

    graph_task = asyncio.create_task(_run_graph_to_queue())

    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=_KEEPALIVE_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                yield _format_sse_event(
                    "status",
                    {"phase": current_phase, "message": current_message, "keepalive": True},
                )
                continue

            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item

            # {"node_name": {updates}} — 단, LangGraph 1.0 은 내부 특수 노드
            # (`__start__`, `__interrupt__`, `__end__`) 이벤트에서 value 로 None 또는
            # non-dict(tuple of Interrupt) 을 실어 보내기도 한다. 이런 이벤트는 final_state
            # 에 merge 할 대상이 아니라 스킵 + SSE 이벤트도 발행하지 않는다.
            for node_name, updates in item.items():
                if updates is None or not isinstance(updates, dict):
                    logger.debug(
                        "admin_assistant_stream_skip_special_event",
                        node_name=node_name,
                        updates_type=type(updates).__name__,
                    )
                    continue
                final_state.update(updates)

                # 노드 완료 status
                completed_msg = _NODE_STATUS_MESSAGES.get(
                    node_name, f"{node_name} 처리 중...",
                )
                yield _format_sse_event(
                    "status", {"phase": node_name, "message": completed_msg},
                )

                # 다음 노드 예측해서 keepalive 메시지 갱신
                next_phase, next_msg = _predict_next_node(
                    node_name, {**initial_state, **final_state},
                )
                if next_msg:
                    current_phase = next_phase
                    current_message = next_msg

                # tool_selector 완료 시 tool_call SSE 이벤트 (투명성 — 사용자에게 "무엇을
                # 하려는지" 노출). pending_tool_call 이 None 이거나 finish_task 이면 발행 스킵.
                if node_name == "tool_selector":
                    call = ensure_tool_call(updates.get("pending_tool_call"))
                    if call is not None and call.tool_name != "finish_task":
                        yield _format_sse_event(
                            "tool_call",
                            {
                                "tool_name": call.tool_name,
                                "arguments": call.arguments,
                                "tier": call.tier,
                            },
                        )

                # tool_executor 완료 시 tool_result SSE 이벤트 — 축약된 메타 정보만.
                # raw 데이터는 프런트에 보내지 않는다 (narrator 가 자연어로 서술).
                if node_name == "tool_executor":
                    ref_id = updates.get("latest_tool_ref_id", "")
                    cache = updates.get("tool_results_cache", {}) or {}
                    call = state_snapshot_tool_call(final_state)
                    result = cache.get(ref_id) if ref_id else None
                    if isinstance(result, AdminApiResult):
                        yield _format_sse_event(
                            "tool_result",
                            {
                                "tool_name": call.tool_name if call else "",
                                "ok": result.ok,
                                "status_code": result.status_code,
                                "latency_ms": result.latency_ms,
                                "row_count": result.row_count,
                                "ref_id": ref_id,
                                "error": result.error if not result.ok else "",
                            },
                        )
                        # Phase 4 (2026-04-27): list/Page 응답이면 table_data SSE 도 함께 발행.
                        # _build_table_payload 가 임계치(<3행) 이하 / 비list / 실패 케이스는
                        # None 을 돌려 자동 스킵된다. Client TableDataCard 가 이 payload 를
                        # 카드 형태로 렌더하고 navigate_path 가 있으면 "전체 보기" 버튼 노출.
                        # 2026-04-28 (길 A v3): payload 에 dedup_key 포함 → Client 가 같은
                        # 도구 재호출 시 카드 1장으로 upsert 하도록.
                        table_payload = _build_table_payload(
                            call.tool_name if call else "",
                            result,
                            arguments=call.arguments if call else None,
                        )
                        if table_payload is not None:
                            yield _format_sse_event("table_data", table_payload)

                        # Phase 4 후속 (2026-04-28): 등록된 시계열 stats tool 이면
                        # chart_data SSE 도 함께 발행. _build_chart_payload 가 whitelist 외
                        # tool / 임계 미달 / 응답 형태 불일치 케이스는 None 을 돌려 자동 스킵.
                        # Client ChartDataCard 가 recharts 로 라인/막대 차트 렌더.
                        chart_payload = _build_chart_payload(
                            call.tool_name if call else "",
                            result,
                            arguments=call.arguments if call else None,
                        )
                        if chart_payload is not None:
                            yield _format_sse_event("chart_data", chart_payload)

                # v3 Phase D: draft_emitter 완료 시 form_prefill SSE 이벤트 발행.
                # Client 가 이 이벤트를 받으면 FormPrefillCard 를 렌더하고
                # "[action_label]" 버튼으로 navigate(target_path, {state: {draft: ...}}) 제공.
                if node_name == "draft_emitter":
                    prefill = updates.get("form_prefill")
                    if prefill and isinstance(prefill, dict):
                        yield _format_sse_event("form_prefill", prefill)

                # v3 Phase D: navigator 완료 시 navigation SSE 이벤트 발행.
                # Client 가 이 이벤트를 받으면 NavigationCard 를 렌더하고
                # 단건이면 "이동" 버튼, 다건이면 candidates 리스트 + 각각의 "이동" 버튼 제공.
                if node_name == "navigator":
                    nav = updates.get("navigation")
                    if nav and isinstance(nav, dict):
                        yield _format_sse_event("navigation", nav)

                # response_formatter 완료 시 최종 응답 텍스트를 token 으로 발행
                if node_name == "response_formatter":
                    response_text = updates.get("response_text", "")
                    if response_text:
                        yield _format_sse_event(
                            "token", {"delta": response_text},
                        )

        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        _final_intent = ensure_intent(final_state.get("intent"))
        intent_kind = _final_intent.kind if _final_intent is not None else "unknown"

        # ── HITL interrupt 감지 ──
        # v3 에서 발동 안 함 — risk_gate 노드 제거로 interrupt() 호출 지점이 사라졌다.
        # snapshot.next 는 항상 빈 tuple 이므로 is_interrupted=False 로 처리된다.
        # 블록 자체는 하위 호환 및 향후 Phase E+ HITL 재도입 대비로 보존.
        snapshot = await admin_assistant_graph.aget_state(graph_config)
        is_interrupted = bool(snapshot.next)

        if is_interrupted:
            # tasks 목록 중 interrupt payload 가 담긴 첫 값 꺼내기
            confirmation_value: dict | None = None
            for task in (snapshot.tasks or []):
                interrupts = getattr(task, "interrupts", None) or []
                for intr in interrupts:
                    val = getattr(intr, "value", None)
                    if isinstance(val, dict):
                        confirmation_value = val
                        break
                if confirmation_value is not None:
                    break

            if confirmation_value is not None:
                logger.info(
                    "admin_assistant_interrupt_emitted",
                    session_id=session_id,
                    tool_name=confirmation_value.get("tool_name", ""),
                    tier=confirmation_value.get("tier", -1),
                    elapsed_ms=round(graph_elapsed_ms, 1),
                )
                yield _format_sse_event("confirmation_required", confirmation_value)
                # done 을 발행하지 않는다 — Client 는 '승인 대기' 상태로 대기.
                return

            # payload 를 꺼내지 못한 예외 상황: 에러 이벤트로 대체
            logger.warning(
                "admin_assistant_interrupt_without_payload",
                session_id=session_id,
                snapshot_next=snapshot.next,
            )
            yield _format_sse_event("error", {
                "message": "승인 요청을 준비하지 못했어요. 잠시 후 다시 시도해주세요.",
            })
            yield _format_sse_event("done", {})
            return

        logger.info(
            "admin_assistant_stream_done",
            session_id=session_id,
            intent=intent_kind,
            elapsed_ms=round(graph_elapsed_ms, 1),
        )
        yield _format_sse_event("done", {})

    except Exception as e:
        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        # Step 6b 후속 디버깅(2026-04-23): 실전에서 `'NoneType' object is not iterable`
        # 같은 에러가 원인 지점 불명으로 올라왔다. 스택 트레이스를 로그에 남겨 다음
        # 재현에서 정확한 프레임을 특정한다.
        logger.error(
            "admin_assistant_stream_error",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(graph_elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )
        yield _format_sse_event("error", {"message": str(e)})
        yield _format_sse_event("done", {})

    finally:
        if not graph_task.done():
            graph_task.cancel()
            try:
                await graph_task
            except (asyncio.CancelledError, Exception):
                pass


def _predict_next_node(completed_node: str, merged_state: dict) -> tuple[str, str]:
    """
    방금 완료된 노드 이후 실행될 다음 노드의 (phase, message) 예측 (v3 Phase D).

    keepalive status 메시지를 정확히 갱신하기 위함. 라우팅 함수를 직접 호출해 예측.
    예외 발생 시 ("", "") 반환 — keepalive 메시지가 갱신 안 될 뿐 흐름에 영향 없음.

    v3 변경점:
    - risk_gate 예측 제거
    - tool_executor → observation 예측 추가
    - observation → route_after_observation 호출로 예측
    - draft_emitter / navigator → narrator 예측 추가

    Returns:
        (phase, message) — 예측 불가면 ("", "")
    """
    try:
        if completed_node == "context_loader":
            return ("intent_classifier", _NODE_STATUS_MESSAGES["intent_classifier"])
        if completed_node == "intent_classifier":
            next_node = route_after_intent(merged_state)
            msg = _NODE_STATUS_MESSAGES.get(next_node, "")
            return (next_node, msg)
        if completed_node == "smalltalk_responder":
            return ("response_formatter", _NODE_STATUS_MESSAGES["response_formatter"])
        if completed_node == "tool_selector":
            next_node = route_after_tool_select(merged_state)
            msg = _NODE_STATUS_MESSAGES.get(next_node, "")
            return (next_node, msg)
        if completed_node == "tool_executor":
            # v3: tool_executor 는 항상 observation 으로 이동
            return ("observation", _NODE_STATUS_MESSAGES["observation"])
        if completed_node == "observation":
            next_node = route_after_observation(merged_state)
            msg = _NODE_STATUS_MESSAGES.get(next_node, "")
            return (next_node, msg)
        if completed_node == "draft_emitter":
            return ("narrator", _NODE_STATUS_MESSAGES["narrator"])
        if completed_node == "navigator":
            return ("narrator", _NODE_STATUS_MESSAGES["narrator"])
        if completed_node == "narrator":
            return ("response_formatter", _NODE_STATUS_MESSAGES["response_formatter"])
        if completed_node == "smart_fallback_responder":
            return ("response_formatter", _NODE_STATUS_MESSAGES["response_formatter"])
    except Exception:
        pass
    return ("", "")


# ============================================================
# 동기 실행 (테스트/디버그 용)
# ============================================================

async def run_admin_assistant_sync(
    admin_id: str,
    admin_role: str,
    admin_jwt: str,
    session_id: str,
    user_message: str,
) -> AdminAssistantState:
    """
    동기 모드 실행 — 최종 state 반환 (SSE 이벤트 수집 없이 디버그용).
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    initial_state: AdminAssistantState = {
        "admin_id": admin_id,
        "admin_role": admin_role,
        "admin_jwt": admin_jwt,
        "session_id": session_id,
        "user_message": user_message,
        "history": [],
    }
    # Step 5a: checkpointer 활성화로 thread_id 필수. 세션 ID 를 그대로 재사용.
    return await admin_assistant_graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
    )
