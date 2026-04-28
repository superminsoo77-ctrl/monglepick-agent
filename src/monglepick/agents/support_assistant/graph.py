"""
support_assistant LangGraph StateGraph + SSE 스트리머 (v4 Phase 1 — 9노드).

### v4 변경점 (2026-04-28)
기존 v3 3노드 그래프를 9노드로 확장한다.
v3 SSE 이벤트 7종 완전 호환 유지 + v4 신규 2종 추가.

v3 그래프 (build_support_assistant_graph_v3) 는 완전히 보존되며
기존 테스트(test_support_assistant_v3.py)는 이 함수를 통해 회귀 검증한다.

### 그래프 구조 (v4 Phase 1)

START → context_loader → intent_classifier → route_after_intent
  ├─ smalltalk  → smalltalk_responder  → response_formatter → END
  ├─ complaint  → response_formatter              → END
  ├─ redirect   → narrator             → response_formatter → END
  ├─ faq / policy / personal_data
  │     → tool_selector → route_after_tool_select
  │         ├─ 정상    → tool_executor → observation → narrator → response_formatter → END
  │         └─ 실패    → smart_fallback              → response_formatter → END
  └─ (예비 — v3 호환 경로)
        → support_agent → response_formatter → END

### SSE 이벤트
v3 호환 (7종):
  session / status / matched_faq / token / needs_human / done / error

v4 신규 (2종):
  policy_chunk  : lookup_policy 결과 (narrator 완료 후, rag_chunks 있을 때)
  navigation    : redirect 의도 시 (narrator 완료 후, navigation 있을 때)

tool_call / tool_result 는 Phase 2 에서 도입. Phase 1 미발행.

### status 이벤트 phase 명
context_loader / intent_classifier / smalltalk_responder / tool_selector /
tool_executor / narrator / response_formatter

설계서: docs/고객센터_AI에이전트_v4_재설계.md §3 (그래프) §10 (SSE) §11 (회귀)
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
import uuid
from collections.abc import AsyncGenerator

import structlog
from langgraph.graph import END, START, StateGraph

from monglepick.agents.support_assistant.models import (
    MatchedFaq,
    SupportAssistantState,
    ensure_reply,
)
from monglepick.agents.support_assistant.nodes import (
    # v3 원본 보존 노드
    context_loader,
    support_agent,
    response_formatter,
    # v4 신규 노드
    intent_classifier,
    smalltalk_responder,
    tool_selector,
    tool_executor,
    observation,
    narrator,
    smart_fallback,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# v3 그래프 (완전 보존 — 기존 테스트용)
# =============================================================================


def build_support_assistant_graph_v3():
    """
    v3 3노드 그래프 — 기존 테스트 및 v3 회귀 보존용.

    START → context_loader → support_agent → response_formatter → END

    test_support_assistant_v3.py 가 이 함수를 통해 컴파일된 그래프를 사용한다.
    v4 마이그레이션이 완전히 완료된 이후에도 회귀 보존을 위해 유지한다.
    """
    graph = StateGraph(SupportAssistantState)

    graph.add_node("context_loader", context_loader)
    graph.add_node("support_agent", support_agent)
    graph.add_node("response_formatter", response_formatter)

    graph.add_edge(START, "context_loader")
    graph.add_edge("context_loader", "support_agent")
    graph.add_edge("support_agent", "response_formatter")
    graph.add_edge("response_formatter", END)

    compiled = graph.compile()
    logger.info("support_assistant_graph_v3_compiled", node_count=3)
    return compiled


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

    pending_tool_call 이 None 이거나 error 가 있으면 smart_fallback.
    정상이면 tool_executor.
    """
    pending = state.get("pending_tool_call")
    error = state.get("error")
    if pending is None or error:
        return "smart_fallback"
    return "tool_executor"


def route_after_observation(state: SupportAssistantState) -> str:
    """
    observation 완료 후 분기.

    Phase 1: 항상 narrator (단일 hop).
    Phase 2 에서 hop_count >= MAX_HOPS 시 narrator 강제 분기 추가 예정.
    """
    # Phase 1 — 단일 hop, 즉시 narrator 진행
    return "narrator"


# =============================================================================
# v4 그래프 빌드
# =============================================================================


def build_support_assistant_graph():
    """
    support_assistant StateGraph v4 Phase 1 — 9노드 + 조건부 분기.

    노드 목록:
      context_loader / intent_classifier / smalltalk_responder /
      tool_selector / tool_executor / observation / narrator /
      smart_fallback / response_formatter

    조건부 엣지:
      intent_classifier  → route_after_intent   (4가지 목적지)
      tool_selector      → route_after_tool_select (2가지 목적지)
      observation        → route_after_observation (Phase 1: narrator 고정)
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

    # intent_classifier → 조건부 분기
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

    # tool_selector → 조건부 분기
    graph.add_conditional_edges(
        "tool_selector",
        route_after_tool_select,
        {
            "tool_executor": "tool_executor",
            "smart_fallback": "smart_fallback",
        },
    )

    # tool_executor → observation → 조건부 분기 (Phase 1: 항상 narrator)
    graph.add_edge("tool_executor", "observation")
    graph.add_conditional_edges(
        "observation",
        route_after_observation,
        {
            "narrator": "narrator",
        },
    )

    # 나머지 고정 엣지 → response_formatter → END
    graph.add_edge("smalltalk_responder", "response_formatter")
    graph.add_edge("narrator", "response_formatter")
    graph.add_edge("smart_fallback", "response_formatter")
    graph.add_edge("response_formatter", END)

    compiled = graph.compile()
    logger.info("support_assistant_graph_v4_compiled", node_count=9)
    return compiled


# =============================================================================
# 모듈 레벨 싱글턴 — v4 (주 그래프) + v3 (회귀 보존)
# =============================================================================

# v4 기본 그래프 — run_support_assistant / run_support_assistant_sync 사용
support_assistant_graph = build_support_assistant_graph()

# v3 보존 그래프 — test_support_assistant_v3.py 가 직접 import 해 사용
support_assistant_graph_v3 = build_support_assistant_graph_v3()


# =============================================================================
# SSE 유틸
# =============================================================================

_KEEPALIVE_INTERVAL_SEC = 15
_SENTINEL = object()

# v4 노드별 status 메시지 (한국어)
_NODE_STATUS_MESSAGES: dict[str, str] = {
    "context_loader": "고객센터 정보를 확인하고 있어요...",
    "intent_classifier": "질문 의도를 파악 중이에요...",
    "smalltalk_responder": "답변을 준비하고 있어요...",
    "tool_selector": "필요한 정보를 찾고 있어요...",
    "tool_executor": "정책 자료를 찾고 있어요...",
    "observation": "검색 결과를 확인하고 있어요...",
    "narrator": "답변을 작성하고 있어요...",
    "smart_fallback": "안내 메시지를 준비하고 있어요...",
    "response_formatter": "답변을 마무리하고 있어요...",
    # v3 호환 — support_agent 는 v3 그래프 전용이지만 _predict_next_node 에서 참조
    "support_agent": "질문 내용을 정리하고 있어요...",
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
    # v3
    "support_agent": "response_formatter",
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

    initial_state: SupportAssistantState = {
        "user_id": user_id or "",
        "session_id": session_id,
        "user_message": user_message,
        "history": [],
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

                # ── v3 호환: support_agent 완료 → matched_faq 이벤트 ──
                if node_name == "support_agent":
                    faqs = _serialize_matched_faqs(updates.get("matched_faqs"))
                    if faqs:
                        yield _format_sse_event("matched_faq", {"items": faqs})

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
        intent_kind = getattr(intent, "kind", "unknown") if intent is not None else "unknown"
        # v3 호환: reply 도 확인
        reply = ensure_reply(final_state.get("reply"))
        kind = reply.kind if reply is not None else intent_kind

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
    initial_state: SupportAssistantState = {
        "user_id": user_id or "",
        "session_id": session_id,
        "user_message": user_message,
        "history": [],
    }
    return await support_assistant_graph.ainvoke(initial_state)
