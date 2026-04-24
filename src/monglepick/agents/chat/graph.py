"""
Chat Agent LangGraph StateGraph 구성 (§6-2).

17노드 + 4개 조건부 라우팅 함수로 구성된 Chat Agent 그래프.
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
          │                ├─ 후보 0건 + 최신 시그널 → external_search_node (DuckDuckGo 웹 검색) → response_formatter → END
          │                │   (2026-04-23 추가: "2026년 영화" 같은 DB 밖 신작 질의 fallback)
          │                ├─ 후보 0건 (시그널 없음) → question_generator → response_formatter → END
          │                └─ 후보 있음 but 품질 낮음 → similar_fallback_search → llm_reranker → ...
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
    RETRIEVAL_SOFT_AMBIGUOUS_TOP_SCORE,
    TURN_COUNT_OVERRIDE,
    ChatAgentState,
    Location,
)
from monglepick.agents.chat.nodes import (
    context_loader,
    error_handler,
    explanation_generator,
    external_search_node,
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
    similar_fallback_search,
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


def _has_recency_signal(state: ChatAgentState) -> bool:
    """
    "최신/최근/올해" 같은 시기 시그널이 preferences 에 담겨 있는지 판정한다.

    external_search_node 진입 조건으로 사용된다 — DB 에서 후보가 0 건이더라도
    사용자가 "명시적 시기" 를 요구하지 않았다면 외부 검색으로 분기하지 않는다.
    (DuckDuckGo 호출은 레이턴시 + 외부 의존성 비용이 있으므로 보수적으로 판단.)

    판정 기준 (OR):
    1) preferences.dynamic_filters 에 release_year >= N 이 있고, N >= (current_year - 1)
       → "올해" / "작년 이후" / "내년" 같은 명시적 최신 필터가 추출된 경우.
    2) current_input 에 "최신/최근/올해/신작/{current_year}" 키워드가 포함된 경우
       (preference 추출이 실패했어도 원문 키워드가 있으면 신호로 인정).

    Args:
        state: ChatAgentState

    Returns:
        True 이면 external_search_node 로 분기
    """
    from datetime import datetime

    current_year = datetime.now().year

    # 1) dynamic_filters 에서 release_year 하한값 확인
    preferences = state.get("preferences")
    if preferences is not None:
        for fc in getattr(preferences, "dynamic_filters", []):
            if fc.field == "release_year" and fc.operator == "gte":
                try:
                    year_gte = int(fc.value)
                    if year_gte >= current_year - 1:
                        return True
                except (TypeError, ValueError):
                    continue

    # 2) 원문 키워드 탐지 (preference 추출이 실패한 경우의 안전망)
    current_input = (state.get("current_input") or "").lower()
    recency_keywords = ["최신", "최근", "올해", "신작", "요즘", "이번 년", "내년"]
    if any(kw in current_input for kw in recency_keywords):
        return True
    # 미래 연도 숫자 직접 언급 (예: "2026 영화")
    if any(str(y) in current_input for y in (current_year, current_year + 1)):
        return True

    return False


def route_after_retrieval(state: ChatAgentState) -> str:
    """
    rag_retriever 이후 검색 품질에 따른 분기 결정.

    품질 기준 (모두 충족해야 PASS):
    1. 후보 수 ≥ RETRIEVAL_MIN_CANDIDATES (3개)
    2. Top-1 RRF 점수 ≥ RETRIEVAL_MIN_TOP_SCORE (기본 0.010)
    3. 상위 5개 평균 ≥ RETRIEVAL_QUALITY_MIN_AVG (기본 0.008)

    분기 우선순위 (2026-04-15 개정 — "애매하면 재질문" 정책 반영):
    1) 품질 통과 또는 turn_count ≥ TURN_COUNT_OVERRIDE → llm_reranker
    2) 후보 0개 → question_generator
    3) 후보 있지만 top_score < RETRIEVAL_SOFT_AMBIGUOUS_TOP_SCORE 이고 초기 턴
       (turn_count < TURN_COUNT_OVERRIDE) → question_generator (soft-ambiguous)
       사용자 의도가 모호해 점수가 낮게 나왔을 가능성이 높다고 보고, 확장 검색 전에
       AI 가 생성한 제안 카드로 의도 확인을 먼저 수행한다.
    4) 그 외 (후보는 있고 점수도 중간 이상) → similar_fallback_search (기존 Phase Q-3)

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
    # "애매 구간" 판정: 후보는 있지만 top_score 가 soft 임계값 미만
    is_soft_ambiguous = (
        num_candidates > 0
        and top_score < RETRIEVAL_SOFT_AMBIGUOUS_TOP_SCORE
        and turn_count < TURN_COUNT_OVERRIDE
    )

    logger.info(
        "route_after_retrieval",
        num_candidates=num_candidates,
        top_score=round(top_score, 6),
        avg_score=round(avg_score, 6),
        quality_passed=quality_passed,
        is_soft_ambiguous=is_soft_ambiguous,
        turn_count=turn_count,
    )

    # ── 라우팅 우선순위 (2026-04-23 재조정) ──
    # 1-a) 후보 0개 + 최신 영화 시그널 존재 → external_search_node.
    #      "2026년 개봉", "올해 영화" 같은 명시적 시기 요구가 있는데 DB 에 후보가
    #      0 건이면 DB 가 신작을 미수집한 상황이 가장 흔하다. 이전에는 question_generator
    #      로만 보내 재질문만 반복했으나, DuckDuckGo 웹 검색으로 Wikipedia/나무위키의
    #      개봉작 정보를 끌어와 "DB 외" 카드로 응답한다. (설계: §외부검색_아키텍처)
    # 1-b) 후보 0개 + 최신 시그널 없음 → 기존대로 question_generator.
    #      과거에는 "3턴째이면 어떻게든 추천" 의도로 override 를 먼저 평가했으나, 후보가
    #      0 개라면 llm_reranker → recommendation_ranker_no_candidates → response_formatter
    #      의 빈 응답("무엇을 도와드릴까요?" 32 자 fallback) 으로 흘러가 사용자가 대화 흐름을
    #      잃어버리는 실제 장애를 재현한다 (2026-04-15 로그 확인). 후보 0 이면 항상 재질문.
    if num_candidates == 0:
        if _has_recency_signal(state):
            logger.info(
                "route_to_external_search",
                turn_count=turn_count,
                reason="num_candidates==0 AND recency_signal",
            )
            return "external_search_node"
        logger.info(
            "route_to_question_empty_candidates",
            turn_count=turn_count,
            reason="num_candidates==0",
        )
        return "question_generator"

    # 2) 품질 통과 또는 턴 오버라이드 → 추천 진행 (후보가 있을 때만)
    if quality_passed or turn_count >= TURN_COUNT_OVERRIDE:
        return "llm_reranker"

    # 3) soft-ambiguous: 의도 확인 재질문으로 보내면서 retrieval_feedback 힌트 주입
    if is_soft_ambiguous:
        logger.info(
            "route_to_question_soft_ambiguous",
            num_candidates=num_candidates,
            top_score=round(top_score, 6),
            reason="ambiguous_intent_low_score",
        )
        return "question_generator"

    # 4) 기존 Phase Q-3: 후보 있고 점수 중간 → 유사 영화 확장 검색
    logger.info(
        "route_to_similar_fallback",
        num_candidates=num_candidates,
        reason="quality_low_but_has_candidates",
    )
    return "similar_fallback_search"


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

    # ── 노드 등록 (16개) ──
    graph.add_node("context_loader", context_loader)
    graph.add_node("image_analyzer", image_analyzer)
    graph.add_node("intent_emotion_classifier", intent_emotion_classifier)
    graph.add_node("preference_refiner", preference_refiner)
    graph.add_node("question_generator", question_generator)
    graph.add_node("query_builder", query_builder)
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("retrieval_quality_checker", retrieval_quality_checker)
    graph.add_node("similar_fallback_search", similar_fallback_search)
    # 2026-04-23 추가: DB 후보 0건 + 최신 영화 시그널 시 DuckDuckGo 웹 검색으로 fallback
    graph.add_node("external_search_node", external_search_node)
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
            "similar_fallback_search": "similar_fallback_search",
            "question_generator": "question_generator",
            "external_search_node": "external_search_node",
        },
    )

    # Phase Q-3: 비슷한 영화 확장 검색 완료 후 → LLM 재랭킹으로 진행
    graph.add_edge("similar_fallback_search", "llm_reranker")

    # 2026-04-23 추가: external_search_node → response_formatter 직결.
    # 외부 웹 검색 결과는 이미 RankedMovie 스텁으로 ranked_movies 에 담겼고
    # CF/CBF 점수 계산 대상이 아니므로 recommendation_ranker 와 explanation_generator 를
    # 건너뛴다 (explanation_generator 의 enrich_movies_batch() 는 여기서 불필요 — 이미
    # DuckDuckGo 로 overview 를 채웠음).
    graph.add_edge("external_search_node", "response_formatter")

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
    logger.info("chat_graph_compiled", node_count=17)
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
    "similar_fallback_search": "비슷한 영화를 추가로 찾고 있어요...",
    "external_search_node": "최신 영화 정보를 웹에서 찾아보고 있어요... 🌐",
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
        elif completed_node == "similar_fallback_search":
            # similar_fallback_search → llm_reranker (고정 엣지)
            next_node = "llm_reranker"
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
            # 2026-04-23 추가: external_search_node 완료 후 recommendation_ranker /
            # explanation_generator 를 건너뛰고 response_formatter 로 직행
            "external_search_node",
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
    guest_id: str = "",
    client_ip: str = "",
    location: Location | None = None,
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
        guest_id: 비로그인 게스트 쿠키 UUID (로그인 유저는 빈 문자열).
            recommendation_ranker 완료 시점에 Backend 의 Redis 키를 소비한다.
        client_ip: 실제 클라이언트 IP. 게스트 쿠키 삭제 후 재진입 방어선.

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
        # 사용자 위치 — Client 가 보낸 좌표(있으면). 미제공 시 tool_executor_node 가
        # 메시지에서 지명을 추출해 geocoding 으로 사후 채워 넣는다.
        "location": location,
        # 세션에서 복원 (없으면 빈 기본값)
        "messages": session_data["messages"] if session_data else [],
        "preferences": session_data["preferences"] if session_data else None,
        "emotion": session_data["emotion"] if session_data else None,
        "turn_count": session_data["turn_count"] if session_data else 0,
        "user_profile": session_data["user_profile"] if session_data else {},
        "watch_history": session_data["watch_history"] if session_data else [],
        # 세션 내 최근 추천된 영화 ID (중복 추천 방지 롤링 윈도우 — 2026-04-24)
        "recent_recommended_ids": (
            session_data.get("recent_recommended_ids", []) if session_data else []
        ),
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
                # 페이로드: { question, hints[], primary_field, suggestions[], allow_custom }
                # suggestions 는 2026-04-15 추가된 AI 생성 제안 카드 (Claude Code 스타일).
                # ClarificationResponse.model_dump() 가 Pydantic 전체 필드를 자동 직렬화한다.
                if node_name == "question_generator":
                    clarification = updates.get("clarification")
                    if clarification and hasattr(clarification, "model_dump"):
                        yield _format_sse_event("clarification", clarification.model_dump())

                # response_formatter 완료 시 결과 발행
                if node_name == "response_formatter":
                    # token 이벤트: 응답 텍스트 발행 (MVP: 전체 텍스트를 단일 이벤트로)
                    # 2026-04-15 중복 답변 제거:
                    # 이 턴에서 `clarification` 이벤트가 이미 question 텍스트를 전달했다면
                    # 동일 텍스트의 `token` 이벤트 emit 을 스킵한다. 클라이언트는 clarification
                    # 이벤트 → ClarificationOptions 컴포넌트에서 question 을 단일 렌더한다.
                    # (nodes.py response_formatter 는 clarification 동반 시 vLLM 재작성을 스킵하고
                    # question 텍스트를 그대로 response 에 담아두므로 chat history 재현에도 문제 없음.)
                    response_text = updates.get("response", "")
                    already_emitted_as_clarification = bool(final_state.get("clarification"))
                    if response_text and not already_emitted_as_clarification:
                        yield _format_sse_event("token", {"delta": response_text})

                # recommendation_ranker 완료 시: AI 쿼터 1회 차감 → movie_card 이벤트 발행
                # 과금 단위: "추천 완료" (movie_card 발행 시점에만 1회 차감)
                # AI의 후속 질문(clarification) 턴은 과금하지 않음
                #
                # 2026-04-15 변경: check/consume 분리 정책 반영.
                # 이전까지는 chat.py 진입 시 check_point 가 쿼터까지 즉시 차감했으나,
                # "추천 완료(movie_card 발행) 전에 쿼터가 소진"되는 정책 버그가 있었다.
                # 이제 check_point 는 순수 조회로 남고, consume_point 가 이 자리에서
                # 유일한 쓰기 경로를 담당한다. 포인트 차감(deduct_point)은 v3.0 AI 무과금
                # 정책상 effective_cost=0 이 기본값이라 거의 호출되지 않으며, 기존 경로는
                # 유료 번들(effective_cost > 0) 호환성 위해 그대로 유지한다.
                if node_name == "recommendation_ranker":
                    ranked_movies = updates.get("ranked_movies", [])
                    if ranked_movies:
                        from monglepick.config import settings

                        # ── AI 쿼터 차감 (movie_card 발행 직전, 로그인 사용자 한정) ──
                        if settings.POINT_CHECK_ENABLED and user_id:
                            from monglepick.api.point_client import consume_point
                            consume_result = await consume_point(user_id=user_id)
                            logger.info(
                                "ai_quota_consumed",
                                user_id=user_id,
                                session_id=session_id,
                                source=consume_result.source,
                                daily_used=consume_result.daily_used,
                                daily_limit=consume_result.daily_limit,
                                sub_bonus_remaining=consume_result.sub_bonus_remaining,
                                purchased_remaining=consume_result.purchased_remaining,
                                allowed=consume_result.allowed,
                            )
                            # point_update 이벤트: UI 배너가 "오늘 N/M" · 이용권 잔량 등 표시
                            yield _format_sse_event("point_update", {
                                "balance": consume_result.balance,
                                "deducted": 0,  # v3.0 AI 무과금
                                "source": consume_result.source,
                                "daily_used": consume_result.daily_used,
                                "daily_limit": consume_result.daily_limit,
                                "sub_bonus_remaining": consume_result.sub_bonus_remaining,
                                "purchased_remaining": consume_result.purchased_remaining,
                                "message": consume_result.message,
                                "free_usage": consume_result.source == "GRADE_FREE",
                            })
                            if not consume_result.allowed:
                                # check 와 consume 사이 레이스로 BLOCKED 로 전환된 경우 —
                                # movie_card 는 이미 생성된 상태라 graceful 유지 (에러 안내만)
                                logger.warning(
                                    "ai_quota_consume_blocked_graceful",
                                    user_id=user_id,
                                    session_id=session_id,
                                    message=consume_result.message,
                                )

                        # ── 게스트(비로그인) 쿼터 차감 (평생 1회, movie_card 발행 직전) ──
                        # 로그인 유저의 consume_point 와 동일 위치. 쿠키+IP 양쪽 Redis 키에 SETNX.
                        # guest_id 가 비어있으면 (쿠키 미발급 상태) 소비 스킵 — 다음 진입에서 차단됨.
                        elif not user_id and guest_id and client_ip:
                            from monglepick.api.guest_quota_client import consume_quota
                            guest_consume = await consume_quota(guest_id, client_ip)
                            logger.info(
                                "guest_quota_consumed",
                                guest_id=guest_id,
                                client_ip=client_ip,
                                session_id=session_id,
                                success=guest_consume.success,
                                reason=guest_consume.reason,
                            )
                            # 게스트 UI 배너용 point_update — source=GUEST_FREE 로 구분.
                            # Client 는 이 이벤트를 받으면 "무료 체험 종료 예정" 안내 UI 를 낼 수 있다.
                            yield _format_sse_event("point_update", {
                                "source": "GUEST_FREE",
                                "guest_used": True,
                                "deducted": 0,
                                "message": "무료 체험 1회를 사용하셨어요. 다음 추천부터는 로그인이 필요해요.",
                            })

                        # ── 유료 포인트 차감 (effective_cost > 0 일 때만; v3.0 기본 0) ──
                        if settings.POINT_CHECK_ENABLED and user_id and effective_cost > 0:
                            from monglepick.api.point_client import deduct_point
                            deduct_result = await deduct_point(
                                user_id=user_id,
                                session_id=session_id,
                                amount=effective_cost,
                                description="AI 추천 사용",
                            )
                            if deduct_result.success:
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
                                # 포인트 잔액 부족 등 — graceful: movie_card 는 유지
                                deduct_error_code = deduct_result.error_code or "INSUFFICIENT_POINT"
                                deduct_error_msg = deduct_result.error_message or "포인트 차감에 실패했습니다."
                                logger.warning(
                                    "point_deduct_failed_graceful",
                                    user_id=user_id,
                                    session_id=session_id,
                                    effective_cost=effective_cost,
                                    error_code=deduct_error_code,
                                )
                                yield _format_sse_event("error", {
                                    "message": deduct_error_msg,
                                    "error_code": deduct_error_code,
                                    "balance": deduct_result.balance_after,
                                    "needs_purchase": deduct_error_code == "INSUFFICIENT_POINT",
                                })

                        # ── 추천 로그 배치 저장 (movie_card 발행 직전, 로그인 사용자 한정) ──
                        # 2026-04-15 신규: 마이픽 추천 내역 + 관리자 AI 추천 분석 둘 다 이 로그를
                        # DB 원천으로 사용한다. 이전까지는 저장 경로 자체가 없어 양쪽 모두 빈 화면.
                        # 저장된 recommendation_log_id 를 RankedMovie 에 주입 → SSE movie_card
                        # payload 에 실려 Client 피드백(관심없음/좋아요) 버튼이 FK 로 사용 가능.
                        if user_id and ranked_movies:
                            from monglepick.api.recommendation_client import save_recommendation_logs
                            prefs = final_state.get("preferences")
                            emotion_result = final_state.get("emotion")
                            user_intent_str = (
                                getattr(prefs, "user_intent", None)
                                if prefs is not None else None
                            ) or ""
                            emotion_str = (
                                getattr(emotion_result, "emotion", None)
                                if emotion_result is not None else None
                            ) or ""
                            mood_tags_list = (
                                getattr(emotion_result, "mood_tags", None)
                                if emotion_result is not None else None
                            ) or []
                            elapsed_ms_int = int((time.perf_counter() - graph_start) * 1000)
                            log_ids = await save_recommendation_logs(
                                user_id=user_id,
                                session_id=session_id,
                                user_intent=user_intent_str,
                                emotion=emotion_str,
                                mood_tags=mood_tags_list,
                                response_time_ms=elapsed_ms_int,
                                model_version="chat-v3.4",
                                movies=ranked_movies,
                            )
                            # movies 와 log_ids 순서 일치 (Backend saveAll 순서 보존 + skip 은 None 채움)
                            # 길이 mismatch 시 zip 은 짧은 쪽 기준으로 수렴하여 graceful
                            for movie, lid in zip(ranked_movies, log_ids):
                                if hasattr(movie, "recommendation_log_id"):
                                    movie.recommendation_log_id = lid

                        # movie_card 이벤트 발행
                        for movie in ranked_movies:
                            movie_data = movie.model_dump() if hasattr(movie, "model_dump") else movie
                            yield _format_sse_event("movie_card", movie_data)

                # 2026-04-23 추가: external_search_node 완료 시 movie_card 이벤트 발행.
                # recommendation_ranker 를 건너뛰므로 별도 처리가 필요하다.
                # - AI 쿼터는 "실제 추천을 받았을 때" 차감하는 정책이라 여기서도 consume_point 수행
                # - 단, recommendation_log 저장은 건너뜀 (external_X id 는 Movie FK 에 없음 → 무의미)
                if node_name == "external_search_node":
                    ranked_movies = updates.get("ranked_movies", [])
                    if ranked_movies:
                        from monglepick.config import settings

                        # AI 쿼터 차감 (로그인 사용자 한정, graceful — 실패해도 카드는 발행)
                        if settings.POINT_CHECK_ENABLED and user_id:
                            try:
                                from monglepick.api.point_client import consume_point
                                consume_result = await consume_point(user_id=user_id)
                                yield _format_sse_event("point_update", {
                                    "balance": consume_result.balance,
                                    "deducted": 0,
                                    "source": consume_result.source,
                                    "daily_used": consume_result.daily_used,
                                    "daily_limit": consume_result.daily_limit,
                                    "sub_bonus_remaining": consume_result.sub_bonus_remaining,
                                    "purchased_remaining": consume_result.purchased_remaining,
                                    "message": consume_result.message,
                                    "free_usage": consume_result.source == "GRADE_FREE",
                                })
                                logger.info(
                                    "ai_quota_consumed_external",
                                    user_id=user_id,
                                    session_id=session_id,
                                    source=consume_result.source,
                                )
                            except Exception as consume_err:
                                # 포인트 서비스 장애와 무관하게 외부 검색 카드는 유지
                                logger.warning(
                                    "ai_quota_consume_external_failed_graceful",
                                    user_id=user_id,
                                    error=str(consume_err),
                                )

                        # 게스트(비로그인) — external_search_node 에서도 동일하게 쿼터 소비.
                        # 외부 검색 결과도 "추천을 받았다" 로 간주하므로 평생 1회가 차감된다.
                        elif not user_id and guest_id and client_ip:
                            try:
                                from monglepick.api.guest_quota_client import consume_quota
                                guest_consume = await consume_quota(guest_id, client_ip)
                                logger.info(
                                    "guest_quota_consumed_external",
                                    guest_id=guest_id,
                                    client_ip=client_ip,
                                    session_id=session_id,
                                    success=guest_consume.success,
                                    reason=guest_consume.reason,
                                )
                                yield _format_sse_event("point_update", {
                                    "source": "GUEST_FREE",
                                    "guest_used": True,
                                    "deducted": 0,
                                    "message": "무료 체험 1회를 사용하셨어요. 다음 추천부터는 로그인이 필요해요.",
                                })
                            except Exception as guest_err:
                                logger.warning(
                                    "guest_quota_consume_external_failed_graceful",
                                    guest_id=guest_id,
                                    error=str(guest_err),
                                )

                        # movie_card 이벤트 발행 (recommendation_log_id 는 None 유지)
                        for movie in ranked_movies:
                            movie_data = movie.model_dump() if hasattr(movie, "model_dump") else movie
                            yield _format_sse_event("movie_card", movie_data)

                # 2026-04-23 추가: tool_executor_node 완료 시 외부 지도 결과 카드 발행.
                # theater 의도: theater_search 결과 → theater_card 이벤트 N개
                #               kobis_now_showing 결과 → now_showing 이벤트 1개 (Top-N 묶음)
                # info 의도:    movie_detail / ott / similar 는 token 이벤트(response 텍스트)로 충분.
                # 이 분기는 AI 쿼터를 차감하지 않는다 — info/theater/booking 은 추천 흐름이 아니므로.
                if node_name == "tool_executor_node":
                    tool_results = updates.get("tool_results", {}) or {}
                    theaters = tool_results.get("theater_search")
                    if isinstance(theaters, list):
                        for t in theaters:
                            yield _format_sse_event("theater_card", t)
                    now_showing = tool_results.get("kobis_now_showing")
                    if isinstance(now_showing, list) and now_showing:
                        yield _format_sse_event("now_showing", {"movies": now_showing})

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

        # [FIX] 그래프 에러 시에도 현재까지의 대화 내용을 세션에 저장한다.
        # 기존에는 except 블록에서 save_session이 호출되지 않아,
        # 그래프 중간에 에러가 발생하면 세션이 MySQL에 전혀 저장되지 않았음.
        # 사용자는 채팅 응답을 일부 받았지만 이력에는 남지 않는 문제의 원인.
        try:
            error_merged_state = {**initial_state, **final_state}
            await save_session(user_id, session_id, error_merged_state)
        except Exception as save_err:
            logger.error("chat_agent_error_save_also_failed",
                         session_id=session_id, save_error=str(save_err))

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
    location: Location | None = None,
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
        location: 사용자 위치 (theater/booking 의도용, 외부 지도 연동)

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
        # 사용자 위치 — Client 가 보낸 좌표(있으면). 미제공 시 tool_executor_node 가
        # 메시지에서 지명을 추출해 geocoding 으로 사후 채워 넣는다.
        "location": location,
        # 세션에서 복원 (없으면 빈 기본값)
        "messages": session_data["messages"] if session_data else [],
        "preferences": session_data["preferences"] if session_data else None,
        "emotion": session_data["emotion"] if session_data else None,
        "turn_count": session_data["turn_count"] if session_data else 0,
        "user_profile": session_data["user_profile"] if session_data else {},
        "watch_history": session_data["watch_history"] if session_data else [],
        # 세션 내 최근 추천된 영화 ID (중복 추천 방지 롤링 윈도우 — 2026-04-24)
        "recent_recommended_ids": (
            session_data.get("recent_recommended_ids", []) if session_data else []
        ),
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
