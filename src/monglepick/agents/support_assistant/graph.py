"""
support_assistant LangGraph StateGraph + SSE 스트리머 (v4 Phase 2 — 다중 hop ReAct).

### v4 Phase 2 변경점 (2026-04-28)
Phase 1 의 단일 hop 골격을 다중 hop ReAct 루프로 확장한다.

### 그래프 구조 (v4 Phase 2)

START → context_loader → intent_classifier → route_after_intent
  ├─ smalltalk  → smalltalk_responder  → response_formatter → END
  ├─ complaint  → response_formatter              → END
  ├─ redirect   → narrator             → response_formatter → END
  ├─ faq / policy / personal_data
  │     → tool_selector → route_after_tool_select
  │         ├─ finish_task  → narrator → response_formatter → END   ← v4 Phase 2 신규
  │         ├─ 정상 tool    → tool_executor → observation → route_after_observation
  │         │                   ├─ hop < MAX_HOPS + 미완료 → tool_selector (ReAct 루프)
  │         │                   ├─ finish_task / hop >= MAX_HOPS → narrator
  │         │                   └─ navigate tool → narrator
  │         └─ 실패    → smart_fallback → response_formatter → END

### SSE 이벤트 (9종)
  session / status / matched_faq / token / needs_human / done / error
  policy_chunk  : lookup_policy 결과 (narrator 완료 후, rag_chunks 있을 때)
  navigation    : redirect 의도 시 (narrator 완료 후, navigation 있을 때)

### status 이벤트 phase 명
context_loader / intent_classifier / smalltalk_responder / tool_selector /
tool_executor / observation / narrator / smart_fallback / response_formatter

### MAX_HOPS
nodes.MAX_HOPS (기본 3, SUPPORT_MAX_HOPS 환경변수로 오버라이드).
route_after_observation 에서 hop_count >= MAX_HOPS 이면 narrator 로 강제 분기.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §3 (그래프) §10 (SSE) §11 (회귀)
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

from monglepick.agents.support_assistant.models import (
    MatchedFaq,
    SupportAssistantState,
)
from monglepick.agents.support_assistant.nodes import (
    # v4 노드
    context_loader,
    response_formatter,
    intent_classifier,
    smalltalk_responder,
    tool_selector,
    tool_executor,
    observation,
    narrator,
    smart_fallback,
    # Phase 2: MAX_HOPS 상수 (route_after_observation 에서 사용)
    MAX_HOPS,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# v4 라우팅 함수
# =============================================================================


def route_after_intent(state: SupportAssistantState) -> str:
    """
    intent_classifier 완료 후 분기.

    intent.kind 에 따라 다음 노드를 결정한다.

    반환값:
      "smalltalk"        → smalltalk_responder
      "complaint"        → response_formatter  (고정 템플릿 직행)
      "redirect"         → narrator            (고정 메시지 + navigation 설정)
      "faq"              → tool_selector
      "policy"           → tool_selector
      "personal_data"    → tool_selector
      기타 / None        → tool_selector       (안전 기본값 — faq 폴백)
    """
    intent = state.get("intent")
    kind = getattr(intent, "kind", None) if intent is not None else None

    if kind == "smalltalk":
        return "smalltalk_responder"
    if kind == "complaint":
        return "response_formatter"
    if kind == "redirect":
        return "narrator"
    # faq / policy / personal_data → tool 경로
    return "tool_selector"


def route_after_tool_select(state: SupportAssistantState) -> str:
    """
    tool_selector 완료 후 분기.

    ### v4 Phase 2 변경점
    - finish_task 가상 tool 이 선택된 경우: tool_executor 를 건너뛰고 narrator 로 직행.
      Solar 가 "이미 충분한 데이터가 있다" 고 판단한 시그널이다.
    - pending_tool_call 이 None 이거나 error 가 있으면: smart_fallback.
    - 정상 tool 이면: tool_executor.

    반환값:
      "narrator"      → finish_task 선택 (tool_executor 스킵)
      "tool_executor" → 정상 tool 실행
      "smart_fallback"→ tool 없음 / 에러
    """
    pending = state.get("pending_tool_call")
    error = state.get("error")

    if pending is None or error:
        return "smart_fallback"

    # finish_task 가상 tool: tool_executor 건너뛰고 narrator 로 직행
    tool_name = pending.get("tool_name", "")
    if tool_name == "finish_task":
        logger.debug(
            "route_after_tool_select_finish_task",
            hop_count=state.get("hop_count", 0),
        )
        return "narrator"

    return "tool_executor"


def route_after_observation(state: SupportAssistantState) -> str:
    """
    observation 완료 후 분기 (v4 Phase 2 — 다중 hop ReAct).

    ### 분기 우선순위
    1. hop_count >= MAX_HOPS → "narrator"  (상한 도달, 부분 결과로 답변)
    2. 마지막 tool 이 "finish_task" → "narrator"  (Solar 종결 시그널)
    3. 그 외 read tool 완료 → "tool_selector"  (다음 hop 재진입)

    ### 왜 observation 이 아니라 이 함수에서 라우팅하는가?
    LangGraph 의 add_conditional_edges 는 state 를 읽기만 하므로,
    observation 노드가 hop_count 를 갱신한 직후의 state 를 이 함수가 받아
    다음 경로를 결정한다. 노드는 state 갱신만 담당하고 라우팅은 분리한다.

    Args:
        state: observation 노드 실행 완료 후의 최신 SupportAssistantState.

    Returns:
        "narrator"      : 루프 종료 후 답변 생성
        "tool_selector" : 다음 hop 재진입 (ReAct 루프 계속)
    """
    hop_count: int = state.get("hop_count") or 0
    history: list[dict] = state.get("tool_call_history") or []

    # 1) MAX_HOPS 상한 도달 → 부분 결과로 narrator 강제 분기
    if hop_count >= MAX_HOPS:
        logger.info(
            "route_after_observation_max_hops_reached",
            hop_count=hop_count,
            max_hops=MAX_HOPS,
        )
        return "narrator"

    # 2) 마지막 tool 이 finish_task → narrator (Solar 가 종결 판단)
    if history:
        last_call = history[-1]
        last_tool_name = last_call.get("tool_name", "")
        if last_tool_name == "finish_task":
            logger.info(
                "route_after_observation_finish_task",
                hop_count=hop_count,
            )
            return "narrator"

    # 3) 그 외: 다음 hop 을 위해 tool_selector 재진입
    logger.debug(
        "route_after_observation_continue_react",
        hop_count=hop_count,
        max_hops=MAX_HOPS,
        last_tool=history[-1].get("tool_name", "?") if history else "none",
    )
    return "tool_selector"


# =============================================================================
# v4 그래프 빌드
# =============================================================================


def build_support_assistant_graph():
    """
    support_assistant StateGraph v4 Phase 2 — 9노드 + 다중 hop ReAct 루프.

    노드 목록:
      context_loader / intent_classifier / smalltalk_responder /
      tool_selector / tool_executor / observation / narrator /
      smart_fallback / response_formatter

    조건부 엣지:
      intent_classifier  → route_after_intent       (4가지 목적지)
      tool_selector      → route_after_tool_select  (3가지 목적지: tool_executor / narrator / smart_fallback)
      observation        → route_after_observation  (2가지 목적지: tool_selector(ReAct 루프) / narrator)

    ### v4 Phase 2 변경점
    - tool_selector → route_after_tool_select 에 "narrator" 목적지 추가
      (finish_task 선택 시 tool_executor 스킵)
    - observation → route_after_observation 에 "tool_selector" 목적지 추가
      (hop_count < MAX_HOPS + 미완료 → ReAct 루프 재진입)
    """
    graph = StateGraph(SupportAssistantState)

    # ── 노드 등록 ──
    graph.add_node("context_loader", context_loader)
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("smalltalk_responder", smalltalk_responder)
    graph.add_node("tool_selector", tool_selector)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("observation", observation)
    graph.add_node("narrator", narrator)
    graph.add_node("smart_fallback", smart_fallback)
    graph.add_node("response_formatter", response_formatter)

    # ── 고정 엣지 ──
    graph.add_edge(START, "context_loader")
    graph.add_edge("context_loader", "intent_classifier")

    # intent_classifier → 조건부 분기 (4가지 목적지)
    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "smalltalk_responder": "smalltalk_responder",
            "response_formatter": "response_formatter",
            "narrator": "narrator",
            "tool_selector": "tool_selector",
        },
    )

    # tool_selector → 조건부 분기 (Phase 2: 3가지 목적지)
    # "narrator" 목적지 추가 — finish_task 선택 시 tool_executor 스킵
    graph.add_conditional_edges(
        "tool_selector",
        route_after_tool_select,
        {
            "tool_executor": "tool_executor",
            "narrator": "narrator",
            "smart_fallback": "smart_fallback",
        },
    )

    # tool_executor → observation (고정)
    graph.add_edge("tool_executor", "observation")

    # observation → 조건부 분기 (Phase 2: 2가지 목적지)
    # "tool_selector" 목적지 추가 — hop < MAX_HOPS 이면 ReAct 루프 재진입
    graph.add_conditional_edges(
        "observation",
        route_after_observation,
        {
            "narrator": "narrator",
            "tool_selector": "tool_selector",
        },
    )

    # 나머지 고정 엣지 → response_formatter → END
    graph.add_edge("smalltalk_responder", "response_formatter")
    graph.add_edge("narrator", "response_formatter")
    graph.add_edge("smart_fallback", "response_formatter")
    graph.add_edge("response_formatter", END)

    # ── Checkpointer 분기 ──
    # SUPPORT_REDIS_CHECKPOINTER_ENABLED=true 면 Redis 기반 AsyncRedisSaver 로 멀티턴
    # 세션 상태를 영속한다. 기본값 false — MemorySaver (단일 프로세스 인스턴스 내 유지).
    #
    # 키 prefix 는 admin_assistant (admin_assistant:checkpoint*) 와 격리하기 위해
    # support_assistant:* 네임스페이스를 사용한다.
    #
    # asetup() (Redis Search 인덱스 생성) 은 main.py lifespan 의
    # setup_support_assistant_checkpointer() 에서 1회 호출.
    checkpointer, kind = _make_support_checkpointer()

    # 모듈 변수에 저장 — startup hook 이 같은 인스턴스에 asetup() 호출하기 위함.
    global _support_assistant_saver
    _support_assistant_saver = checkpointer

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(
        "support_assistant_graph_v4_compiled",
        node_count=9,
        max_hops=MAX_HOPS,
        phase="2_multi_hop_react",
        checkpointer=kind,
    )
    return compiled


# =============================================================================
# Phase 3 — Checkpointer 팩토리 + lifespan setup
# =============================================================================

#: 모듈 레벨 saver 인스턴스 보관 — `setup_support_assistant_checkpointer()` 가
#: 같은 인스턴스에 asetup() 호출하도록. None 이면 그래프가 아직 컴파일되지 않은 상태.
_support_assistant_saver: Any | None = None


def _is_support_redis_checkpointer_enabled() -> bool:
    """
    SUPPORT_REDIS_CHECKPOINTER_ENABLED 환경변수 — true/1/yes 외에는 비활성.

    기본값은 false (MemorySaver). 운영 환경에서 멀티 인스턴스 배포 시 true 로 전환.
    """
    return os.getenv("SUPPORT_REDIS_CHECKPOINTER_ENABLED", "false").lower() in (
        "true", "1", "yes"
    )


def _make_support_checkpointer() -> tuple[Any, str]:
    """
    환경변수에 따라 AsyncRedisSaver 또는 MemorySaver 반환.

    admin_assistant._make_admin_checkpointer() 와 동일한 패턴.
    키 prefix 는 support_assistant 전용 네임스페이스로 격리해 admin_assistant 와
    충돌하지 않게 한다.

    Redis 패키지 import 실패 시(개발 환경 패키지 미설치) MemorySaver 로 안전 폴백.
    AsyncRedisSaver 초기화 예외 시도 MemorySaver 로 폴백.

    Returns:
        (saver_instance, kind_label) — kind 는 로그/메트릭 용 ("memory" | "redis").
    """
    if not _is_support_redis_checkpointer_enabled():
        logger.info(
            "support_memory_checkpointer_selected",
            reason="SUPPORT_REDIS_CHECKPOINTER_ENABLED 미설정 또는 false",
        )
        return MemorySaver(), "memory"

    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    except ImportError as exc:
        logger.warning(
            "support_redis_checkpointer_import_failed_fallback_memory",
            error=str(exc),
        )
        return MemorySaver(), "memory"

    try:
        from monglepick.config import settings

        saver = AsyncRedisSaver(
            redis_url=settings.REDIS_URL,
            # support_assistant 전용 prefix — admin_assistant:* 와 키 충돌 없음
            checkpoint_prefix="support_assistant:checkpoint",
            checkpoint_blob_prefix="support_assistant:cp_blob",
            checkpoint_write_prefix="support_assistant:cp_write",
        )
    except Exception as exc:
        logger.warning(
            "support_redis_checkpointer_init_failed_fallback_memory",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return MemorySaver(), "memory"

    logger.info(
        "support_redis_checkpointer_initialized",
        redis_url=settings.REDIS_URL,
        checkpoint_prefix="support_assistant:checkpoint",
    )
    return saver, "redis"


async def setup_support_assistant_checkpointer() -> None:
    """
    FastAPI lifespan 에서 1회 호출 — AsyncRedisSaver 의 Redis Search 인덱스 생성.

    admin_assistant.setup_admin_assistant_checkpointer() 와 동일한 패턴.
    MemorySaver 인 경우 no-op. asetup() 은 idempotent — 이미 인덱스가 있어도
    안전하게 통과. 실패 시 경고만 남기고 앱 기동 차단하지 않음.

    main.py lifespan 에서 setup_admin_assistant_checkpointer() 호출 직후에 추가 등록.
    """
    saver = _support_assistant_saver
    if saver is None:
        logger.warning("support_checkpointer_not_initialized")
        return

    asetup = getattr(saver, "asetup", None)
    if asetup is None:
        # MemorySaver — Redis Search 인덱스 생성 불필요
        logger.info(
            "support_checkpointer_setup_skipped",
            kind=type(saver).__name__,
        )
        return

    try:
        await asetup()
        logger.info(
            "support_checkpointer_setup_done",
            kind=type(saver).__name__,
        )
    except Exception as exc:
        # 실패 시 운영자가 인지하도록 ERROR 레벨 + 앱은 계속 기동
        # (체크포인트 없이도 support_assistant 는 동작 가능)
        logger.error(
            "support_checkpointer_setup_failed",
            kind=type(saver).__name__,
            error=str(exc),
            error_type=type(exc).__name__,
        )


# =============================================================================
# 모듈 레벨 싱글턴 — v4 그래프
# =============================================================================

# v4 그래프 — run_support_assistant / run_support_assistant_sync 사용
support_assistant_graph = build_support_assistant_graph()


# =============================================================================
# SSE 유틸
# =============================================================================

_KEEPALIVE_INTERVAL_SEC = 15
_SENTINEL = object()

# v4 노드별 status 메시지 (한국어)
# Phase 2: tool_selector / observation 메시지를 hop 진행 맥락에 맞게 업데이트.
_NODE_STATUS_MESSAGES: dict[str, str] = {
    "context_loader": "고객센터 정보를 확인하고 있어요...",
    "intent_classifier": "질문 의도를 파악 중이에요...",
    "smalltalk_responder": "답변을 준비하고 있어요...",
    "tool_selector": "필요한 정보를 조회하고 있어요...",
    "tool_executor": "계정 정보와 정책 자료를 가져오고 있어요...",
    "observation": "조회 결과를 분석하고 있어요...",
    "narrator": "진단 답변을 작성하고 있어요...",
    "smart_fallback": "안내 메시지를 준비하고 있어요...",
    "response_formatter": "답변을 마무리하고 있어요...",
}

# v4 노드 완료 후 다음 예측 노드 매핑 (keepalive 메시지 정확도용)
_NEXT_NODE_MAP: dict[str, str] = {
    "context_loader": "intent_classifier",
    "intent_classifier": "tool_selector",      # 대표 경로 (faq/policy 기준)
    "tool_selector": "tool_executor",
    "tool_executor": "observation",
    "observation": "narrator",
    "narrator": "response_formatter",
    "smalltalk_responder": "response_formatter",
    "smart_fallback": "response_formatter",
}


def _format_sse_event(event_type: str, data: dict) -> dict:
    """sse_starlette EventSourceResponse 호환 dict."""
    return {"event": event_type, "data": json.dumps(data, ensure_ascii=False)}


def _serialize_matched_faqs(value) -> list[dict]:
    """
    matched_faqs 를 SSE JSON 직렬화 가능 형태로 변환.

    `question` 이 비어 있는 id-only MatchedFaq 는 SSE 페이로드에서 제외한다.
    `_select_matched_faqs`(nodes.py) 가 ES candidate 로 검증된 ID 를 Backend
    FAQ 캐시에서 못 찾았을 때 kind 강등 방어 목적으로 question="" 인 축약본을
    state 에 남기지만, UI 의 FaqMatchCard 는 question 텍스트를 본문으로 렌더하므로
    그대로 보내면 빈 박스가 노출된다 (QA 2026-04-28).
    """
    if not value:
        return []
    out: list[dict] = []
    for item in value:
        if isinstance(item, MatchedFaq):
            question = (item.question or "").strip()
            if not question:
                continue
            out.append(
                {
                    "faq_id": item.faq_id,
                    "category": item.category,
                    "question": question,
                }
            )
        elif isinstance(item, dict):
            question = (item.get("question") or "").strip()
            if not question:
                continue
            out.append(
                {
                    "faq_id": item.get("faq_id"),
                    "category": item.get("category"),
                    "question": question,
                }
            )
    return out


def _predict_next_node(completed_node: str) -> tuple[str, str]:
    """
    방금 완료된 노드 다음의 (phase, message) 를 돌려 keepalive 메시지 정확도를 유지.
    """
    next_node = _NEXT_NODE_MAP.get(completed_node, "")
    next_msg = _NODE_STATUS_MESSAGES.get(next_node, "") if next_node else ""
    return next_node, next_msg


# =============================================================================
# SSE 스트리밍 실행 (v4)
# =============================================================================


async def run_support_assistant(
    user_id: str,
    session_id: str,
    user_message: str,
) -> AsyncGenerator[dict, None]:
    """
    support_assistant v4 를 SSE 스트리밍 모드로 실행한다.

    v3 호환 SSE 이벤트 (7종) + v4 신규 이벤트 (2종) 발행.

    Args:
        user_id:      JWT 에서 추출한 사용자 ID (비로그인이면 빈 문자열).
        session_id:   세션 ID (빈 문자열이면 자동 생성).
        user_message: 현재 턴 사용자 발화.

    Yields:
        sse_starlette 호환 dict. {"event": ..., "data": json_string}

    SSE 이벤트 발행 규칙:
      session       : 항상 첫 번째 이벤트
      status        : 각 노드 완료 시 (phase=노드명, message=한국어)
      matched_faq   : tool_executor 완료 후 matched_faqs 있을 때 (faq 경로)
      policy_chunk  : narrator 완료 후 rag_chunks 있을 때 (policy/personal_data 경로)
      navigation    : narrator 완료 후 navigation 있을 때 (redirect 경로)
      token         : response_formatter 완료 후 response_text
      needs_human   : response_formatter 완료 후 needs_human_agent 값
      done          : 항상 마지막
      error         : 예외 발생 시 done 직전
    """
    graph_start = time.perf_counter()

    if not session_id:
        session_id = str(uuid.uuid4())

    # 멀티턴 컨텍스트 보존: history 는 초기 state 에 포함하지 않는다.
    # LangGraph checkpointer(MemorySaver / AsyncRedisSaver) 가 thread_id=session_id
    # 단위로 직전 턴 state 를 보존하므로, history 필드를 비우면 이전 대화가
    # 매 턴 초기화되는 회귀가 발생한다 (2026-04-28 사용자 보고).
    # context_loader 가 None/미존재인 경우 [] 로 초기화한다.
    initial_state: SupportAssistantState = {
        "user_id": user_id or "",
        "session_id": session_id,
        "user_message": user_message,
    }

    logger.info(
        "support_assistant_stream_start",
        session_id=session_id,
        user_id=user_id or "(guest)",
        message_preview=user_message[:100],
    )

    yield _format_sse_event("session", {"session_id": session_id})

    queue: asyncio.Queue = asyncio.Queue()
    current_phase = "context_loader"
    current_message = _NODE_STATUS_MESSAGES[current_phase]
    final_state: dict = {}

    async def _run_graph_to_queue():
        """LangGraph astream 의 이벤트를 큐에 흘려넣는다."""
        try:
            async for event in support_assistant_graph.astream(
                initial_state,
                # Phase 3 RedisSaver: session_id 를 thread_id 로 사용해
                # checkpointer 가 멀티턴 상태를 session 단위로 격리·저장한다.
                # MemorySaver 모드에서도 thread_id 를 넘겨야 단일 프로세스 내에서
                # session 간 state 충돌을 방지할 수 있다.
                config={"configurable": {"thread_id": session_id}},
                stream_mode="updates",
            ):
                await queue.put(event)
            await queue.put(_SENTINEL)
        except Exception as exc:  # noqa: BLE001
            await queue.put(exc)

    graph_task = asyncio.create_task(_run_graph_to_queue())

    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=_KEEPALIVE_INTERVAL_SEC
                )
            except asyncio.TimeoutError:
                yield _format_sse_event(
                    "status",
                    {
                        "phase": current_phase,
                        "message": current_message,
                        "keepalive": True,
                    },
                )
                continue

            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item

            # item = {"node_name": {updates}} — LangGraph 특수 이벤트(None/tuple) 방어
            for node_name, updates in item.items():
                if updates is None or not isinstance(updates, dict):
                    logger.debug(
                        "support_stream_skip_special_event",
                        node_name=node_name,
                        updates_type=type(updates).__name__,
                    )
                    continue
                final_state.update(updates)

                # 노드 완료 status 이벤트
                completed_msg = _NODE_STATUS_MESSAGES.get(
                    node_name, f"{node_name} 처리 중..."
                )
                yield _format_sse_event(
                    "status", {"phase": node_name, "message": completed_msg}
                )

                # 다음 phase 예측 — keepalive 메시지 업데이트
                next_phase, next_msg = _predict_next_node(node_name)
                if next_msg:
                    current_phase = next_phase
                    current_message = next_msg

                # ── v4: tool_executor 완료 → matched_faq 이벤트 (faq 경로) ──
                if node_name == "tool_executor":
                    faqs = _serialize_matched_faqs(updates.get("matched_faqs"))
                    if faqs:
                        yield _format_sse_event("matched_faq", {"items": faqs})

                # ── v4 신규: narrator 완료 → policy_chunk / navigation 이벤트 ──
                if node_name == "narrator":
                    # policy_chunk: rag_chunks 가 있을 때
                    rag_chunks: list[dict] = final_state.get("rag_chunks") or []
                    if rag_chunks:
                        # 상위 3건만 SSE 로 전달 (Client PolicyChunkCard 소재)
                        yield _format_sse_event(
                            "policy_chunk",
                            {
                                "items": [
                                    {
                                        "doc_id": c.get("doc_id", ""),
                                        "section": c.get("section", ""),
                                        "policy_topic": c.get("policy_topic", ""),
                                        "text": c.get("text", "")[:300],  # 미리보기 300자
                                        "score": c.get("score", 0.0),
                                    }
                                    for c in rag_chunks[:3]
                                ]
                            },
                        )

                    # navigation: redirect 의도 시
                    navigation = final_state.get("navigation")
                    if navigation:
                        yield _format_sse_event("navigation", navigation)

                # ── response_formatter 완료 → token + needs_human ──
                if node_name == "response_formatter":
                    response_text = updates.get("response_text", "")
                    needs_human = bool(
                        final_state.get("needs_human_agent", False)
                    )
                    if response_text:
                        yield _format_sse_event(
                            "token", {"delta": response_text}
                        )
                    yield _format_sse_event(
                        "needs_human", {"value": needs_human}
                    )

        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000

        # 로그용 intent kind 추출
        intent = final_state.get("intent")
        kind = getattr(intent, "kind", "unknown") if intent is not None else "unknown"

        logger.info(
            "support_assistant_stream_done",
            session_id=session_id,
            kind=kind,
            matched_count=len(final_state.get("matched_faqs") or []),
            needs_human=bool(final_state.get("needs_human_agent", False)),
            hop_count=final_state.get("hop_count", 0),
            elapsed_ms=round(graph_elapsed_ms, 1),
        )
        yield _format_sse_event("done", {})

    except Exception as exc:  # noqa: BLE001
        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        logger.error(
            "support_assistant_stream_error",
            error=str(exc),
            error_type=type(exc).__name__,
            elapsed_ms=round(graph_elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )
        yield _format_sse_event("error", {"message": str(exc)})
        yield _format_sse_event("done", {})

    finally:
        if not graph_task.done():
            graph_task.cancel()
            try:
                await graph_task
            except (asyncio.CancelledError, Exception):
                pass


# =============================================================================
# 동기 실행 (디버그/테스트용)
# =============================================================================


async def run_support_assistant_sync(
    user_id: str,
    session_id: str,
    user_message: str,
) -> SupportAssistantState:
    """
    SSE 없이 v4 그래프를 1회 실행하고 최종 state 를 반환한다.

    테스트 코드 및 동기 API 엔드포인트(/support/chat/sync)에서 사용.
    v3 테스트(test_support_assistant_v3.py)도 이 함수를 통해 v4 라우터에서 통과하는지 검증.
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    # 멀티턴 컨텍스트 보존: history 는 초기 state 에 포함하지 않는다 (run_support_assistant 와 동일).
    initial_state: SupportAssistantState = {
        "user_id": user_id or "",
        "session_id": session_id,
        "user_message": user_message,
    }
    # Phase 3 RedisSaver: ainvoke 에도 thread_id 주입 — SSE 경로와 동일한 세션 격리.
    # 테스트에서 MemorySaver 모드로 동작하더라도 thread_id 로 세션 간 state 충돌 방지.
    return await support_assistant_graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
    )
