"""
Chat Agent LangGraph StateGraph 구성 (§6-2).

13노드 + 4개 조건부 라우팅 함수로 구성된 Chat Agent 그래프.
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
          │       └─ needs_clarification=False → query_builder → rag_retriever
          │            → route_after_retrieval (조건부 분기)
          │                ├─ 품질 OK → recommendation_ranker → explanation_generator → response_formatter → END
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
    ChatAgentState,
)
from monglepick.memory.session_store import load_session, save_session
from monglepick.agents.chat.nodes import (
    context_loader,
    error_handler,
    explanation_generator,
    general_responder,
    image_analyzer,
    intent_emotion_classifier,
    preference_refiner,
    query_builder,
    question_generator,
    rag_retriever,
    recommendation_ranker,
    response_formatter,
    tool_executor_node,
)

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
    2. Top-1 RRF 점수 ≥ RETRIEVAL_MIN_TOP_SCORE (0.02)
    3. 상위 5개 평균 ≥ RETRIEVAL_QUALITY_MIN_AVG (0.015)

    품질 미달 + turn_count < 3 → question_generator (추가 질문)
    품질 미달 + turn_count ≥ 3 → recommendation_ranker (있는 결과로 진행)
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

    if quality_passed or turn_count >= 3:
        # 품질 통과 또는 3턴 이상이면 추천 진행 (무한 루프 방지)
        return "recommendation_ranker"
    else:
        # 품질 미달: state에 피드백 메시지 설정 (question_generator에서 활용)
        # Note: LangGraph에서 라우터는 state를 수정할 수 없으므로
        # rag_retriever에서 미리 설정하거나, 별도 노드가 필요하다.
        # 여기서는 state.retrieval_feedback이 이미 rag_retriever에서 설정된다고 가정.
        return "question_generator"


# ============================================================
# 그래프 빌드
# ============================================================

def build_chat_graph() -> StateGraph:
    """
    Chat Agent StateGraph를 구성하고 컴파일한다.

    13개 노드와 4개 조건부 분기를 등록하여 영화 추천 대화 흐름을 정의한다.

    Returns:
        컴파일된 StateGraph (CompiledGraph)
    """
    # StateGraph 생성 (ChatAgentState TypedDict 기반)
    graph = StateGraph(ChatAgentState)

    # ── 노드 등록 (13개) ──
    graph.add_node("context_loader", context_loader)
    graph.add_node("image_analyzer", image_analyzer)
    graph.add_node("intent_emotion_classifier", intent_emotion_classifier)
    graph.add_node("preference_refiner", preference_refiner)
    graph.add_node("question_generator", question_generator)
    graph.add_node("query_builder", query_builder)
    graph.add_node("rag_retriever", _rag_retriever_with_quality_check)
    graph.add_node("recommendation_ranker", recommendation_ranker)
    graph.add_node("explanation_generator", explanation_generator)
    graph.add_node("response_formatter", response_formatter)
    graph.add_node("error_handler", error_handler)
    graph.add_node("general_responder", general_responder)
    graph.add_node("tool_executor_node", tool_executor_node)

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

    # 추천 진행 흐름: query_builder → rag_retriever → route_after_retrieval (조건부)
    graph.add_edge("query_builder", "rag_retriever")
    graph.add_conditional_edges(
        "rag_retriever",
        route_after_retrieval,
        {
            "recommendation_ranker": "recommendation_ranker",
            "question_generator": "question_generator",
        },
    )

    # recommendation_ranker → explanation_generator → response_formatter
    graph.add_edge("recommendation_ranker", "explanation_generator")
    graph.add_edge("explanation_generator", "response_formatter")

    # 일반 대화: general_responder → response_formatter
    graph.add_edge("general_responder", "response_formatter")

    # 도구 실행: tool_executor_node → response_formatter
    graph.add_edge("tool_executor_node", "response_formatter")

    # 에러 처리: error_handler → response_formatter
    graph.add_edge("error_handler", "response_formatter")

    # response_formatter → END
    graph.add_edge("response_formatter", END)

    # 그래프 컴파일
    compiled = graph.compile()
    logger.info("chat_graph_compiled", node_count=13)
    return compiled


async def _rag_retriever_with_quality_check(state: ChatAgentState) -> dict:
    """
    rag_retriever 래퍼: 검색 실행 후 품질 판정 결과를 state에 기록한다.

    route_after_retrieval 라우터가 state를 수정할 수 없으므로,
    검색 결과를 기반으로 retrieval_quality_passed와 retrieval_feedback을
    이 노드에서 미리 설정한다.

    Args:
        state: ChatAgentState

    Returns:
        dict: candidate_movies, retrieval_quality_passed, retrieval_feedback 업데이트
    """
    # 실제 rag_retriever 노드 호출
    result = await rag_retriever(state)
    candidates = result.get("candidate_movies", [])

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

    # 품질 미달 시 피드백 메시지 생성
    feedback = ""
    if not quality_passed:
        if num_candidates == 0:
            feedback = "조건에 맞는 영화를 찾지 못했어요."
        elif top_score < RETRIEVAL_MIN_TOP_SCORE:
            feedback = "조건과 딱 맞는 영화를 찾기 어려웠어요."
        else:
            feedback = "검색 결과가 충분하지 않아요."

    result["retrieval_quality_passed"] = quality_passed
    result["retrieval_feedback"] = feedback
    return result


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
    "recommendation_ranker": "최적의 영화를 고르고 있어요...",
    "explanation_generator": "추천 이유를 작성하고 있어요...",
    "response_formatter": "응답을 정리하고 있어요...",
    "error_handler": "문제를 처리하고 있어요...",
    "general_responder": "답변을 준비하고 있어요...",
    "tool_executor_node": "기능을 확인하고 있어요...",
}


# ============================================================
# SSE 스트리밍 인터페이스
# ============================================================

async def run_chat_agent(
    user_id: str,
    session_id: str,
    message: str,
    image_data: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Chat Agent를 SSE 스트리밍 모드로 실행한다.

    세션 영속화 흐름:
    1. session_id가 비어있으면 자동 생성
    2. Redis에서 기존 세션 로드 → 초기 State에 병합
    3. LangGraph astream으로 그래프 실행
    4. 실행 완료 후 세션 저장

    SSE 이벤트 형식:
    - {"event": "session", "data": {"session_id": "uuid-..."}}  — 세션 ID 전달
    - {"event": "status", "data": {"phase": "노드명", "message": "한국어 상태"}}
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

    Yields:
        SSE 이벤트 JSON 문자열 (줄바꿈 포함)
    """
    # 노드 실행 타이밍 측정 시작
    graph_start = time.perf_counter()

    # 1. 세션 ID가 비어있으면 자동 생성
    if not session_id:
        session_id = str(uuid.uuid4())

    # 2. Redis에서 기존 세션 로드
    session_data = await load_session(session_id)

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

    # 4-1. 이미지가 있으면 곧바로 "이미지 분석 중" status 발행 (VLM 호출이 1~2분 걸릴 수 있어 무한 로딩처럼 보이는 것 방지)
    if image_data:
        yield _format_sse_event("status", {
            "phase": "image_analysis",
            "message": "이미지를 분석하고 있어요... 🖼️ (1~2분 걸릴 수 있어요)",
        })

    # 최종 State를 추적하기 위한 누적 dict
    final_state: dict = {}

    try:
        # LangGraph astream: 각 노드 완료 시 업데이트 수신
        async for event in chat_graph.astream(
            initial_state,
            stream_mode="updates",
        ):
            # event 형식: {"node_name": {updates_dict}}
            for node_name, updates in event.items():
                # 최종 state 누적 (세션 저장용)
                final_state.update(updates)

                # status 이벤트 발행
                status_msg = NODE_STATUS_MESSAGES.get(node_name, f"{node_name} 처리 중...")
                yield _format_sse_event("status", {
                    "phase": node_name,
                    "message": status_msg,
                })

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

                # recommendation_ranker 완료 시 movie_card 이벤트 발행
                if node_name == "recommendation_ranker":
                    ranked_movies = updates.get("ranked_movies", [])
                    for movie in ranked_movies:
                        movie_data = movie.model_dump() if hasattr(movie, "model_dump") else movie
                        yield _format_sse_event("movie_card", movie_data)

        # 5. 그래프 완료 후 세션 저장
        merged_state = {**initial_state, **final_state}
        await save_session(session_id, merged_state)

        # 완료 이벤트
        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        logger.info("chat_agent_stream_done", session_id=session_id, elapsed_ms=round(graph_elapsed_ms, 1))
        yield _format_sse_event("done", {})

    except Exception as e:
        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        logger.error("chat_agent_stream_error", error=str(e), error_type=type(e).__name__, elapsed_ms=round(graph_elapsed_ms, 1))
        yield _format_sse_event("error", {"message": str(e)})
        yield _format_sse_event("done", {})


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

    Returns:
        실행 완료된 ChatAgentState (session_id 포함)
    """
    # 노드 실행 타이밍 측정 시작
    graph_start = time.perf_counter()

    # 1. 세션 ID가 비어있으면 자동 생성
    if not session_id:
        session_id = str(uuid.uuid4())

    # 2. Redis에서 기존 세션 로드
    session_data = await load_session(session_id)

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

    # 4. 그래프 완료 후 세션 저장
    await save_session(session_id, result)

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
