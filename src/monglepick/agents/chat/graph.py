"""
Chat Agent LangGraph StateGraph 구성 (§6-2).

15노드 + 4개 조건부 라우팅 함수로 구성된 Chat Agent 그래프.
SSE 스트리밍과 동기 실행 인터페이스를 제공한다.
Redis 세션 저장소를 통해 멀티턴 대화 상태를 영속화한다.

그래프 흐름:
    START → context_loader → route_has_image (조건부 분기)
          │
          ├─ 이미지 있음 → image_analyzer → intent_emotion_classifier
          └─ 이미지 없음 → intent_emotion_classifier
          │
          → route_after_intent (조건부 분기)
          │
          ├─ recommend/search → preference_refiner
          │   → route_after_preference (조건부 분기)
          │       ├─ needs_clarification=True  → question_generator → response_formatter → END
          │       └─ needs_clarification=False → query_builder → rag_retriever → retrieval_quality_checker
          │            → route_after_retrieval (조건부 분기)
          │                ├─ 품질 OK → llm_reranker → recommendation_ranker → explanation_generator → response_formatter → END
          │                └─ 품질 미달 → question_generator → response_formatter → END
          │
          ├─ general → general_responder → response_formatter → END
          │
          └─ info/theater/booking → tool_executor_node → response_formatter → END

세션 영속화 흐름:
    1. 요청 수신 → session_id 자동 생성 (빈 문자열이면)
    2. Redis에서 기존 세션 로드 (messages, preferences, emotion, turn_count, ...)
    3. 초기 State에 세션 데이터 병합
    4. 그래프 실행
    5. 실행 완료 후 세션 저장 (영속 필드만 선택적 저장)

공개 인터페이스:
- run_chat_agent(user_id, session_id, message, image_data) → AsyncGenerator[str, None]
- run_chat_agent_sync(user_id, session_id, message, image_data) → ChatAgentState
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator

import structlog
from langgraph.graph import END, START, StateGraph

from monglepick.agents.chat.models import (
    RETRIEVAL_MIN_CANDIDATES,
    RETRIEVAL_MIN_TOP_SCORE,
    RETRIEVAL_QUALITY_MIN_AVG,
    TURN_COUNT_OVERRIDE,
    ChatAgentState,
)
from monglepick.agents.chat.nodes import (
    context_loader,
    error_handler,
    explanation_generator,
    general_responder,
    graph_traversal_node,
    image_analyzer,
    intent_emotion_classifier,
    llm_reranker,
    preference_refiner,
    query_builder,
    question_generator,
    rag_retriever,
    recommendation_ranker,
    response_formatter,
    retrieval_quality_checker,
    tool_executor_node,
)
from monglepick.memory.session_store import load_session, save_session

logger = structlog.get_logger()


# ============================================================
# 라우팅 함수
# ============================================================

def route_has_image(state: ChatAgentState) -> str:
    """
    context_loader 이후 이미지 존재 여부에 따른 분기 결정.

    - image_data가 있으면 → image_analyzer (VLM 분석)
    - image_data가 없으면 → intent_emotion_classifier (이미지 분석 스킵)

    Args:
        state: ChatAgentState (image_data 확인)

    Returns:
        다음 노드 이름 문자열
    """
    has_image = bool(state.get("image_data"))
    next_node = "image_analyzer" if has_image else "intent_emotion_classifier"

    logger.info(
        "route_has_image",
        has_image=has_image,
        route=next_node,
    )
    return next_node


def route_after_intent(state: ChatAgentState) -> str:
    """
    intent_emotion_classifier 이후 분기 결정.

    - recommend/search → preference_refiner (추천 흐름, emotion_analyzer 스킵)
    - general → general_responder (일반 대화)
    - info/theater/booking → tool_executor_node (도구 실행)
    - unknown/None → error_handler (에러 처리)

    Args:
        state: ChatAgentState (intent 필요)

    Returns:
        다음 노드 이름 문자열
    """
    intent = state.get("intent")

    if intent is None:
        logger.info("route_after_intent", route="error_handler", reason="intent_is_none")
        return "error_handler"

    intent_type = intent.intent

    if intent_type in ("recommend", "search"):
        # 통합 노드에서 이미 감정 분석 완료 → preference_refiner 직행
        next_node = "preference_refiner"
    elif intent_type == "general":
        next_node = "general_responder"
    elif intent_type == "relation":
        # 관계 기반 탐색: Neo4j 멀티홉 그래프 탐색 (§관계_대사_검색_설계서)
        # "봉준호 감독 스릴러 배우들이 찍은 영화" 등 인물 관계 기반 질의
        next_node = "graph_traversal_node"
    elif intent_type in ("info", "theater", "booking"):
        next_node = "tool_executor_node"
    else:
        next_node = "error_handler"

    logger.info(
        "route_after_intent",
        intent=intent_type,
        confidence=intent.confidence,
        route=next_node,
    )
    return next_node


def route_after_preference(state: ChatAgentState) -> str:
    """
    preference_refiner 이후 분기 결정.

    - needs_clarification=True → question_generator (후속 질문)
    - needs_clarification=False → query_builder (추천 진행)

    Args:
        state: ChatAgentState (needs_clarification 필요)

    Returns:
        다음 노드 이름 문자열
    """
    needs_clarification = state.get("needs_clarification", True)

    if needs_clarification:
        next_node = "question_generator"
    else:
        next_node = "query_builder"

    logger.info(
        "route_after_preference",
        needs_clarification=needs_clarification,
        route=next_node,
    )
    return next_node


def route_after_retrieval(state: ChatAgentState) -> str:
    """
    rag_retriever 이후 검색 품질에 따른 분기 결정.

    품질 기준 (모두 충족해야 PASS):
    1. 후보 수 ≥ RETRIEVAL_MIN_CANDIDATES (3개)
    2. Top-1 RRF 점수 ≥ RETRIEVAL_MIN_TOP_SCORE (0.015)
    3. 상위 5개 평균 ≥ RETRIEVAL_QUALITY_MIN_AVG (0.01)

    품질 미달 + turn_count < TURN_COUNT_OVERRIDE(3) → question_generator (추가 질문)
    품질 미달 + turn_count ≥ TURN_COUNT_OVERRIDE(3) → recommendation_ranker (있는 결과로 진행)
    품질 충족 → recommendation_ranker

    Args:
        state: ChatAgentState (candidate_movies, turn_count 필요)

    Returns:
        다음 노드 이름 문자열
    """
    candidates = state.get("candidate_movies", [])
    turn_count = state.get("turn_count", 0)

    # 검색 품질 판정
    num_candidates = len(candidates)
    top_score = candidates[0].rrf_score if candidates else 0.0
    avg_score = (
        sum(c.rrf_score for c in candidates[:5]) / min(len(candidates), 5)
        if candidates else 0.0
    )

    quality_passed = (
        num_candidates >= RETRIEVAL_MIN_CANDIDATES
        and top_score >= RETRIEVAL_MIN_TOP_SCORE
        and avg_score >= RETRIEVAL_QUALITY_MIN_AVG
    )

    logger.info(
        "route_after_retrieval",
        num_candidates=num_candidates,
        top_score=round(top_score, 6),
        avg_score=round(avg_score, 6),
        quality_passed=quality_passed,
        turn_count=turn_count,
    )

    if quality_passed or turn_count >= TURN_COUNT_OVERRIDE:
        # 품질 통과 또는 TURN_COUNT_OVERRIDE(3)턴 이상이면 LLM 재랭킹 → 추천 진행
        return "llm_reranker"
    else:
        # 품질 미달: state에 피드백 메시지 설정 (question_generator에서 활용)
        # retrieval_quality_checker 노드가 retrieval_feedback을 설정한 후
        # 이 라우터가 호출된다. quality_ok/quality_low 분기를 결정한다. (W-3)
        return "question_generator"


# ============================================================
# 그래프 빌드
# ============================================================

def build_chat_graph() -> StateGraph:
    """
    Chat Agent StateGraph를 구성하고 컴파일한다.

    14개 노드와 4개 조건부 분기를 등록하여 영화 추천 대화 흐름을 정의한다.
    retrieval_quality_checker는 rag_retriever와 route_after_retrieval 사이에서
    검색 품질을 판정하여 state에 기록하는 독립 노드이다.

    Returns:
        컴파일된 StateGraph (CompiledGraph)
    """
    # StateGraph 생성 (ChatAgentState TypedDict 기반)
    graph = StateGraph(ChatAgentState)

    # ── 노드 등록 (15개) ──
    graph.add_node("context_loader", context_loader)
    graph.add_node("image_analyzer", image_analyzer)
    graph.add_node("intent_emotion_classifier", intent_emotion_classifier)
    graph.add_node("preference_refiner", preference_refiner)
    graph.add_node("question_generator", question_generator)
    graph.add_node("query_builder", query_builder)
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("retrieval_quality_checker", retrieval_quality_checker)
    graph.add_node("llm_reranker", llm_reranker)
    graph.add_node("recommendation_ranker", recommendation_ranker)
    graph.add_node("explanation_generator", explanation_generator)
    graph.add_node("response_formatter", response_formatter)
    graph.add_node("error_handler", error_handler)
    graph.add_node("general_responder", general_responder)
    graph.add_node("tool_executor_node", tool_executor_node)
    # relation Intent 전용: Neo4j 멀티홉 탐색 (§관계_대사_검색_설계서.md §5.6)
    graph.add_node("graph_traversal_node", graph_traversal_node)

    # ── 엣지 정의 ──

    # START → context_loader → route_has_image (이미지 유무에 따라 분기)
    #   ├─ 이미지 있음 → image_analyzer → intent_emotion_classifier
    #   └─ 이미지 없음 → intent_emotion_classifier (image_analyzer 스킵)
    graph.add_edge(START, "context_loader")
    graph.add_conditional_edges(
        "context_loader",
        route_has_image,
        {
            "image_analyzer": "image_analyzer",
            "intent_emotion_classifier": "intent_emotion_classifier",
        },
    )
    graph.add_edge("image_analyzer", "intent_emotion_classifier")

    # intent_emotion_classifier → 조건부 분기 (route_after_intent)
    graph.add_conditional_edges(
        "intent_emotion_classifier",
        route_after_intent,
        {
            "preference_refiner": "preference_refiner",
            "general_responder": "general_responder",
            "graph_traversal_node": "graph_traversal_node",  # relation Intent
            "tool_executor_node": "tool_executor_node",
            "error_handler": "error_handler",
        },
    )

    # 추천 흐름: preference_refiner → 조건부 분기
    graph.add_conditional_edges(
        "preference_refiner",
        route_after_preference,
        {
            "question_generator": "question_generator",
            "query_builder": "query_builder",
        },
    )

    # 후속 질문 흐름: question_generator → response_formatter → END
    graph.add_edge("question_generator", "response_formatter")

    # 추천 진행 흐름: query_builder → rag_retriever → retrieval_quality_checker → 조건부 분기
    graph.add_edge("query_builder", "rag_retriever")
    graph.add_edge("rag_retriever", "retrieval_quality_checker")
    graph.add_conditional_edges(
        "retrieval_quality_checker",
        route_after_retrieval,
        {
            "llm_reranker": "llm_reranker",
            "question_generator": "question_generator",
        },
    )

    # LLM 재랭킹 → 추천 엔진 서브그래프 → 설명 생성
    graph.add_edge("llm_reranker", "recommendation_ranker")

    # recommendation_ranker → explanation_generator → response_formatter
    graph.add_edge("recommendation_ranker", "explanation_generator")
    graph.add_edge("explanation_generator", "response_formatter")

    # 일반 대화: general_responder → response_formatter
    graph.add_edge("general_responder", "response_formatter")

    # 도구 실행: tool_executor_node → response_formatter
    graph.add_edge("tool_executor_node", "response_formatter")

    # 관계 탐색: graph_traversal_node → recommendation_ranker
    # 탐색 결과(candidate_movies)를 recommendation_ranker에서 CF+CBF와 결합하여 최종 순위 결정
    # final_answer가 있으면(결과 없음 폴백) recommendation_ranker가 스킵하고 response_formatter로 직행
    graph.add_edge("graph_traversal_node", "recommendation_ranker")

    # 에러 처리: error_handler → response_formatter
    graph.add_edge("error_handler", "response_formatter")

    # response_formatter → END
    graph.add_edge("response_formatter", END)

    # 그래프 컴파일
    compiled = graph.compile()
    logger.info("chat_graph_compiled", node_count=16)
    return compiled


# ── 모듈 레벨 싱글턴: 컴파일 1회 ──
chat_graph = build_chat_graph()


# ============================================================
# 노드 이름 → 한국어 상태 메시지 매핑 (SSE status 이벤트용)
# ============================================================

NODE_STATUS_MESSAGES: dict[str, str] = {
    "context_loader": "사용자 정보를 불러오고 있어요...",
    "image_analyzer": "이미지를 분석하고 있어요... 🖼️",
    "intent_emotion_classifier": "말씀을 이해하고 감정을 분석하고 있어요...",
    "preference_refiner": "취향을 파악하고 있어요...",
    "question_generator": "질문을 준비하고 있어요...",
    "query_builder": "검색 조건을 구성하고 있어요...",
    "rag_retriever": "영화를 검색하고 있어요... 🔍",
    "retrieval_quality_checker": "검색 결과 품질을 확인하고 있어요...",
    "llm_reranker": "사용자님의 요청에 맞는 영화인지 검증하고 있어요...",
    "recommendation_ranker": "최적의 영화를 고르고 있어요...",
    "explanation_generator": "추천 이유를 작성하고 있어요...",
    "response_formatter": "응답을 정리하고 있어요...",
    "error_handler": "문제를 처리하고 있어요...",
    "general_responder": "답변을 준비하고 있어요...",
    "tool_executor_node": "기능을 확인하고 있어요...",
    "graph_traversal_node": "관계 그래프를 탐색하고 있어요... 🕸️",
}


# ============================================================
# 다음 노드 예측 (SSE 상태 메시지 지연 방지)
# ============================================================


def _predict_next_node(completed_node: str, state: dict) -> tuple[str, str] | None:
    """
    완료된 노드 이후 실행될 다음 노드를 예측하여 (phase, message) 튜플을 반환한다.

    노드 완료 시점에 다음 노드의 "진행 중" 상태 메시지를 즉시 발행하기 위해 사용한다.
    기존 라우팅 함수(route_has_image, route_after_intent 등)를 호출하여
    정확한 다음 노드를 결정한다.

    이 함수가 없으면 context_loader(~60ms) 완료 후 intent_emotion_classifier(50~70초)가
    실행되는 동안 "사용자 정보를 불러오고 있어요..." 메시지가 keepalive로 반복 표시된다.
    이 함수를 통해 다음 노드 메시지("말씀을 이해하고 감정을 분석하고 있어요...")를
    즉시 발행하여 사용자에게 정확한 진행 상태를 안내한다.

    Args:
        completed_node: 방금 완료된 노드 이름
        state: 현재까지 누적된 state dict (라우팅 함수에 전달)

    Returns:
        (다음 노드 이름, 한국어 상태 메시지) 튜플. 예측 불가 시 None.
    """
    try:
        # 조건부 분기가 있는 노드: 라우팅 함수를 호출하여 다음 노드 결정
        if completed_node == "context_loader":
            next_node = route_has_image(state)
        elif completed_node == "image_analyzer":
            # image_analyzer → intent_emotion_classifier (고정 엣지)
            next_node = "intent_emotion_classifier"
        elif completed_node == "intent_emotion_classifier":
            next_node = route_after_intent(state)
        elif completed_node == "preference_refiner":
            next_node = route_after_preference(state)
        elif completed_node == "query_builder":
            # query_builder → rag_retriever (고정 엣지)
            next_node = "rag_retriever"
        elif completed_node == "rag_retriever":
            # rag_retriever → retrieval_quality_checker (고정 엣지)
            next_node = "retrieval_quality_checker"
        elif completed_node == "retrieval_quality_checker":
            next_node = route_after_retrieval(state)
        elif completed_node == "llm_reranker":
            # llm_reranker → recommendation_ranker (고정 엣지)
            next_node = "recommendation_ranker"
        elif completed_node == "recommendation_ranker":
            # recommendation_ranker → explanation_generator (고정 엣지)
            next_node = "explanation_generator"
        elif completed_node == "explanation_generator":
            # explanation_generator → response_formatter (고정 엣지)
            next_node = "response_formatter"
        elif completed_node in (
            "general_responder",
            "tool_executor_node",
            "error_handler",
            "question_generator",
        ):
            # 모두 response_formatter로 이동 (고정 엣지)
            next_node = "response_formatter"
        else:
            return None

        # 예측된 노드의 상태 메시지가 있으면 반환
        next_message = NODE_STATUS_MESSAGES.get(next_node)
        if next_message:
            return (next_node, next_message)
    except Exception:
        # 예측 실패 시 무시 — 다음 노드 완료 시 자연스럽게 갱신됨
        pass
    return None


# ============================================================
# SSE 스트리밍 인터페이스
# ============================================================

# SSE keepalive 간격 (초). 이 시간마다 status 이벤트를 발행하여 연결 유지
_KEEPALIVE_INTERVAL_SEC = 15

# 그래프 완료를 알리는 센티넬 객체
_SENTINEL = object()


async def run_chat_agent(
    user_id: str,
    session_id: str,
    message: str,
    image_data: str | None = None,
    effective_cost: int = 0,
) -> AsyncGenerator[str, None]:
    """
    Chat Agent를 SSE 스트리밍 모드로 실행한다.

    asyncio.Queue 기반으로 그래프 이벤트를 수집하고,
    _KEEPALIVE_INTERVAL_SEC(15초)마다 keepalive status 이벤트를 발행하여
    VLM 등 장시간 노드 실행 중에도 SSE 연결이 끊기지 않도록 한다.

    세션 영속화 흐름:
    1. session_id가 비어있으면 자동 생성
    2. Redis에서 기존 세션 로드 → 초기 State에 병합
    3. asyncio.Task로 그래프 실행, Queue를 통해 이벤트 전달
    4. Queue에서 이벤트를 꺼내 yield (15초 타임아웃 시 keepalive 발행)
    5. 실행 완료 후 세션 저장

    SSE 이벤트 형식:
    - {"event": "session", "data": {"session_id": "uuid-..."}}  — 세션 ID 전달
    - {"event": "status", "data": {"phase": "노드명", "message": "한국어 상태"}}
    - {"event": "status", "data": {"phase": "...", "message": "...", "keepalive": true}}
    - {"event": "movie_card", "data": {RankedMovie JSON}}
    - {"event": "token", "data": {"delta": "응답 텍스트"}}
    - {"event": "clarification", "data": {ClarificationResponse JSON}}
    - {"event": "done", "data": {}}
    - {"event": "error", "data": {"message": "에러 메시지"}}

    Args:
        user_id: 사용자 ID (빈 문자열이면 익명)
        session_id: 세션 ID (빈 문자열이면 자동 생성)
        message: 사용자 입력 메시지
        image_data: base64 인코딩된 이미지 데이터 (None이면 이미지 없음)
        effective_cost: 실제 차감 포인트 (chat.py에서 쿼터 체크 후 전달).
            무료 잔여가 있으면 0이 전달되어 deduct 호출을 스킵한다.

    Yields:
        SSE 이벤트 JSON 문자열 (줄바꿈 포함)
    """
    # 노드 실행 타이밍 측정 시작
    graph_start = time.perf_counter()

    # 1. 세션 ID가 비어있으면 자동 생성
    if not session_id:
        session_id = str(uuid.uuid4())

    # 2. MySQL에서 기존 세션 로드 (Backend API 경유)
    session_data = await load_session(user_id, session_id)

    # 3. 초기 State 구성 (세션 데이터 병합)
    initial_state: ChatAgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "current_input": message,
        "image_data": image_data,
        # 세션에서 복원 (없으면 빈 기본값)
        "messages": session_data["messages"] if session_data else [],
        "preferences": session_data["preferences"] if session_data else None,
        "emotion": session_data["emotion"] if session_data else None,
        "turn_count": session_data["turn_count"] if session_data else 0,
        "user_profile": session_data["user_profile"] if session_data else {},
        "watch_history": session_data["watch_history"] if session_data else [],
    }

    logger.info(
        "chat_agent_stream_start",
        user_id=user_id,
        session_id=session_id,
        message_preview=message[:100],
        has_image=bool(image_data),
        session_restored=session_data is not None,
        restored_turn_count=initial_state.get("turn_count", 0),
    )

    # 4. session 이벤트 발행 (클라이언트에 session_id 전달)
    yield _format_sse_event("session", {"session_id": session_id})

    # ── Queue 기반 그래프 실행 + keepalive ──
    queue: asyncio.Queue = asyncio.Queue()
    # 현재 처리 중인 노드 이름/메시지 (keepalive에서 사용)
    current_phase = "context_loader"
    current_message = NODE_STATUS_MESSAGES.get("context_loader", "처리 중...")

    # 최종 State를 추적하기 위한 누적 dict
    final_state: dict = {}

    async def _run_graph_to_queue():
        """
        LangGraph astream을 실행하고, 각 노드 완료 이벤트를 Queue에 넣는다.

        그래프 완료 시 _SENTINEL, 에러 시 Exception 객체를 Queue에 넣어
        소비자 루프에 종료를 알린다.
        """
        try:
            async for event in chat_graph.astream(
                initial_state,
                stream_mode="updates",
            ):
                await queue.put(event)
            # 그래프 정상 완료 → 센티넬 삽입
            await queue.put(_SENTINEL)
        except Exception as e:
            # 그래프 실행 에러 → Exception 객체 삽입
            await queue.put(e)

    # 그래프를 백그라운드 Task로 실행
    graph_task = asyncio.create_task(_run_graph_to_queue())

    try:
        while True:
            try:
                # 15초 동안 Queue에서 이벤트를 기다림
                item = await asyncio.wait_for(
                    queue.get(),
                    timeout=_KEEPALIVE_INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                # 15초 동안 이벤트 없음 → keepalive status 발행 (연결 유지)
                yield _format_sse_event("status", {
                    "phase": current_phase,
                    "message": current_message,
                    "keepalive": True,
                })
                continue

            # 센티넬: 그래프 정상 완료
            if item is _SENTINEL:
                break

            # Exception 객체: 그래프 실행 에러
            if isinstance(item, Exception):
                raise item

            # 정상 이벤트: {"node_name": {updates_dict}}
            for node_name, updates in item.items():
                # 최종 state 누적 (세션 저장용)
                final_state.update(updates)

                # 완료된 노드의 상태 메시지 발행
                completed_message = NODE_STATUS_MESSAGES.get(node_name, f"{node_name} 처리 중...")
                yield _format_sse_event("status", {
                    "phase": node_name,
                    "message": completed_message,
                })

                # 다음 노드의 "진행 중" 상태 메시지를 즉시 발행하여
                # keepalive가 정확한 메시지를 표시하도록 한다.
                # 예: context_loader 완료 → "말씀을 이해하고 감정을 분석하고 있어요..." 즉시 표시
                merged = {**initial_state, **final_state}
                predicted = _predict_next_node(node_name, merged)
                if predicted:
                    current_phase, current_message = predicted
                    yield _format_sse_event("status", {
                        "phase": current_phase,
                        "message": current_message,
                    })
                else:
                    # 예측 불가 시 완료된 노드의 메시지 유지
                    current_phase = node_name
                    current_message = completed_message

                # question_generator 완료 시 clarification 이벤트 발행
                if node_name == "question_generator":
                    clarification = updates.get("clarification")
                    if clarification and hasattr(clarification, "model_dump"):
                        yield _format_sse_event("clarification", clarification.model_dump())

                # response_formatter 완료 시 결과 발행
                if node_name == "response_formatter":
                    # token 이벤트: 응답 텍스트 발행 (MVP: 전체 텍스트를 단일 이벤트로)
                    response_text = updates.get("response", "")
                    if response_text:
                        yield _format_sse_event("token", {"delta": response_text})

                # recommendation_ranker 완료 시: 포인트 차감 → movie_card 이벤트 발행
                # 과금 단위: "추천 완료" (movie_card 발행 시점에만 1회 차감)
                # AI의 후속 질문(clarification) 턴은 과금하지 않음
                # effective_cost=0이면 무료 잔여로 커버된 요청이므로 차감 스킵
                if node_name == "recommendation_ranker":
                    ranked_movies = updates.get("ranked_movies", [])
                    if ranked_movies:
                        # ── 포인트 차감 (추천 결과가 있고, effective_cost > 0일 때만) ──
                        from monglepick.config import settings
                        if settings.POINT_CHECK_ENABLED and user_id and effective_cost > 0:
                            from monglepick.api.point_client import deduct_point
                            deduct_result = await deduct_point(
                                user_id=user_id,
                                session_id=session_id,
                                amount=effective_cost,
                                description="AI 추천 사용",
                            )
                            if deduct_result.success:
                                # 차감 성공: 잔여 포인트 정보를 SSE로 전달
                                yield _format_sse_event("point_update", {
                                    "balance": deduct_result.balance_after,
                                    "deducted": effective_cost,
                                })
                                logger.info(
                                    "point_deducted",
                                    user_id=user_id,
                                    session_id=session_id,
                                    deducted=effective_cost,
                                    balance_after=deduct_result.balance_after,
                                )
                            else:
                                # 차감 실패: error_code를 파싱하여 세분화된 SSE error 이벤트 발행.
                                # INSUFFICIENT_POINT / DAILY_LIMIT_EXCEEDED / MONTHLY_LIMIT_EXCEEDED
                                # 각각의 error_code는 프론트엔드에서 적절한 안내 UI를 표시하는 데 사용된다.
                                # 단, 추천 결과(movie_card)는 이미 생성된 상태이므로
                                # 그대로 전달하는 것이 UX상 더 낫다 (graceful degradation).
                                # → error 이벤트는 포인트 관련 알림용이며, 추천 차단이 아님.
                                deduct_error_code = deduct_result.error_code or "INSUFFICIENT_POINT"
                                deduct_error_msg = deduct_result.error_message or "포인트 차감에 실패했습니다."
                                logger.warning(
                                    "point_deduct_failed_graceful",
                                    user_id=user_id,
                                    session_id=session_id,
                                    effective_cost=effective_cost,
                                    error_code=deduct_error_code,
                                )
                                # SSE error 이벤트: 프론트엔드가 포인트/쿼터 상태를 파악할 수 있도록 전달
                                yield _format_sse_event("error", {
                                    "message": deduct_error_msg,
                                    "error_code": deduct_error_code,
                                    "balance": deduct_result.balance_after,
                                    # 쿼터 초과 여부: 프론트엔드가 구매 유도 UI를 선택적으로 표시
                                    "needs_purchase": deduct_error_code == "INSUFFICIENT_POINT",
                                })
                        elif settings.POINT_CHECK_ENABLED and user_id and effective_cost == 0:
                            # 무료 잔여로 커버된 요청 → 차감 없이 point_update 발행
                            logger.info(
                                "point_free_usage",
                                user_id=user_id,
                                session_id=session_id,
                            )
                            yield _format_sse_event("point_update", {
                                "balance": -1,  # 무료 사용 시 잔액 미변동 (-1은 미조회)
                                "deducted": 0,
                                "free_usage": True,
                            })

                        # movie_card 이벤트 발행
                        for movie in ranked_movies:
                            movie_data = movie.model_dump() if hasattr(movie, "model_dump") else movie
                            yield _format_sse_event("movie_card", movie_data)

        # 5. 그래프 완료 후 세션 저장 (Backend API → MySQL)
        merged_state = {**initial_state, **final_state}
        await save_session(user_id, session_id, merged_state)

        # 완료 이벤트
        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        logger.info("chat_agent_stream_done", session_id=session_id, elapsed_ms=round(graph_elapsed_ms, 1))
        yield _format_sse_event("done", {})

    except Exception as e:
        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        logger.error("chat_agent_stream_error", error=str(e), error_type=type(e).__name__, elapsed_ms=round(graph_elapsed_ms, 1))
        yield _format_sse_event("error", {"message": str(e)})
        yield _format_sse_event("done", {})

    finally:
        # 그래프 Task가 아직 실행 중이면 정리 (에러 발생 시)
        if not graph_task.done():
            graph_task.cancel()
            try:
                await graph_task
            except (asyncio.CancelledError, Exception):
                pass


def _format_sse_event(event_type: str, data: dict) -> dict:
    """
    SSE 이벤트를 sse_starlette가 인식하는 dict로 포맷한다.

    sse_starlette.EventSourceResponse는 dict를 yield하면
    ServerSentEvent(event=..., data=...)로 변환하여
    "event: {type}\ndata: {json}\n\n" 형식으로 전송한다.

    Args:
        event_type: 이벤트 타입 (status, movie_card, token, clarification, done, error)
        data: 이벤트 데이터 dict

    Returns:
        sse_starlette 호환 dict ({"event": type, "data": json_string})
    """
    return {"event": event_type, "data": json.dumps(data, ensure_ascii=False)}


# ============================================================
# 동기 실행 인터페이스 (테스트용)
# ============================================================

async def run_chat_agent_sync(
    user_id: str,
    session_id: str,
    message: str,
    image_data: str | None = None,
    effective_cost: int = 0,
) -> ChatAgentState:
    """
    Chat Agent를 동기 모드로 실행하여 최종 State를 반환한다 (테스트/디버그용).

    세션 영속화 흐름은 run_chat_agent()와 동일:
    1. session_id 자동 생성 → 2. 세션 로드 → 3. 그래프 실행 → 4. 세션 저장

    Args:
        user_id: 사용자 ID
        session_id: 세션 ID (빈 문자열이면 자동 생성)
        message: 사용자 입력 메시지
        image_data: base64 인코딩된 이미지 데이터 (None이면 이미지 없음)
        effective_cost: 실제 차감 포인트 (동기 엔드포인트는 디버그용이므로 기본값 0)

    Returns:
        실행 완료된 ChatAgentState (session_id 포함)
    """
    # 노드 실행 타이밍 측정 시작
    graph_start = time.perf_counter()

    # 1. 세션 ID가 비어있으면 자동 생성
    if not session_id:
        session_id = str(uuid.uuid4())

    # 2. MySQL에서 기존 세션 로드 (Backend API 경유)
    session_data = await load_session(user_id, session_id)

    # 3. 초기 State 구성 (세션 데이터 병합)
    initial_state: ChatAgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "current_input": message,
        "image_data": image_data,
        # 세션에서 복원 (없으면 빈 기본값)
        "messages": session_data["messages"] if session_data else [],
        "preferences": session_data["preferences"] if session_data else None,
        "emotion": session_data["emotion"] if session_data else None,
        "turn_count": session_data["turn_count"] if session_data else 0,
        "user_profile": session_data["user_profile"] if session_data else {},
        "watch_history": session_data["watch_history"] if session_data else [],
    }

    logger.info(
        "chat_agent_sync_start",
        user_id=user_id,
        session_id=session_id,
        message_preview=message[:100],
        has_image=bool(image_data),
        session_restored=session_data is not None,
        restored_turn_count=initial_state.get("turn_count", 0),
    )

    result = await chat_graph.ainvoke(initial_state)

    # 4. 그래프 완료 후 세션 저장 (Backend API → MySQL)
    await save_session(user_id, session_id, result)

    graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
    logger.info(
        "chat_agent_sync_done",
        session_id=session_id,
        has_response=bool(result.get("response")),
        response_preview=str(result.get("response", ""))[:100],
        ranked_count=len(result.get("ranked_movies", [])),
        elapsed_ms=round(graph_elapsed_ms, 1),
    )
    return result
