"""
Movie Match Agent LangGraph StateGraph 구성 (§21-4).

6노드 + 1개 조건부 라우팅 함수로 구성된 Movie Match 그래프.
SSE 스트리밍과 동기 실행 인터페이스를 제공한다.

그래프 흐름:
    START → movie_loader → route_after_load
          │
          ├─ error → END  (영화 미발견 시 즉시 종료)
          └─ success → feature_extractor → query_builder → rag_retriever
                     → match_scorer → explanation_generator → END

SSE 이벤트 스트림 (5종):
    - status          : 각 노드 진입 시 {phase, message}
    - shared_features : feature_extractor 완료 시 {SharedFeatures JSON}
    - match_result    : explanation_generator 완료 시 {movies: [MatchedMovie]}
    - error           : 에러 발생 시 {error_code, message}
    - done            : 모든 처리 완료 시 {}

공개 인터페이스:
    - build_match_graph()           → CompiledGraph (모듈 레벨 싱글턴 초기화용)
    - run_match_agent(...)          → AsyncGenerator[dict, None] (SSE 이벤트 생성기)
    - run_match_agent_sync(...)     → MovieMatchState (동기 디버그용)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator

import structlog
from langgraph.graph import END, START, StateGraph

from monglepick.agents.match.models import (
    MatchedMovie,
    MovieMatchState,
    SharedFeatures,
)
from monglepick.agents.match.nodes import (
    explanation_generator,
    feature_extractor,
    llm_reranker,
    match_scorer,
    movie_loader,
    query_builder,
    rag_retriever,
)
from monglepick.metrics import match_duration_seconds, match_requests_total

logger = structlog.get_logger()


# ============================================================
# 노드별 SSE 상태 메시지 (§21-6 노드별 상태 메시지)
# ============================================================

NODE_STATUS_MESSAGES: dict[str, dict[str, str]] = {
    # 노드 이름 → {phase: str, message: str}
    "movie_loader": {
        "phase": "loading",
        "message": "선택한 영화 정보를 불러오고 있어요...",
    },
    "feature_extractor": {
        "phase": "analyzing",
        "message": "두 영화의 공통점을 분석하고 있어요...",
    },
    "query_builder": {
        "phase": "building",
        "message": "비슷한 영화를 찾을 조건을 만들고 있어요...",
    },
    "rag_retriever": {
        "phase": "searching",
        "message": "15만 편의 영화에서 검색하고 있어요...",
    },
    # Match v3: llm_reranker 노드의 SSE 상태 메시지
    "llm_reranker": {
        "phase": "reranking",
        "message": "AI가 두 분의 취향에 맞는 영화를 고르고 있어요...",
    },
    "match_scorer": {
        "phase": "scoring",
        "message": "두 분 모두 좋아할 영화를 선별하고 있어요...",
    },
    "explanation_generator": {
        "phase": "explaining",
        "message": "추천 이유를 정리하고 있어요...",
    },
}

# SSE keepalive 간격 (초) — 장시간 노드 실행 중 연결 유지
_KEEPALIVE_INTERVAL_SEC = 15

# 그래프 완료를 알리는 센티넬 객체
_SENTINEL = object()


# ============================================================
# 라우팅 함수
# ============================================================

def route_after_load(state: MovieMatchState) -> str:
    """
    movie_loader 이후 분기 결정.

    - error 필드가 존재하면 → END (영화 미발견 에러)
    - error 필드가 없으면 → feature_extractor (정상 진행)

    Args:
        state: MovieMatchState (error 필드 존재 여부 확인)

    Returns:
        다음 노드 이름 또는 END 상수
    """
    has_error = bool(state.get("error"))

    logger.info(
        "route_after_load",
        has_error=has_error,
        route=END if has_error else "feature_extractor",
        error=state.get("error", ""),
    )

    if has_error:
        return END
    return "feature_extractor"


# ============================================================
# 그래프 빌드
# ============================================================

def build_match_graph():
    """
    Movie Match Agent StateGraph를 구성하고 컴파일한다.

    Match v3 (2026-04-14): 7개 노드로 확장
    - rag_retriever 와 match_scorer 사이에 llm_reranker(Solar LLM) 노드 추가
    - "두 영화를 모두 좋아할 사용자 관점" 을 Chat Agent 수준의 LLM 리랭커로 판단

    Returns:
        컴파일된 StateGraph (CompiledGraph)
    """
    # MovieMatchState TypedDict 기반 StateGraph 생성
    graph = StateGraph(MovieMatchState)

    # ── 노드 등록 (7개, Match v3) ──
    graph.add_node("movie_loader", movie_loader)
    graph.add_node("feature_extractor", feature_extractor)
    graph.add_node("query_builder", query_builder)
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("llm_reranker", llm_reranker)            # Match v3 신규
    graph.add_node("match_scorer", match_scorer)
    graph.add_node("explanation_generator", explanation_generator)

    # ── 엣지 정의 ──

    # START → movie_loader (진입점)
    graph.add_edge(START, "movie_loader")

    # movie_loader → route_after_load → feature_extractor 또는 END
    graph.add_conditional_edges(
        "movie_loader",
        route_after_load,
        {
            "feature_extractor": "feature_extractor",
            END: END,
        },
    )

    # 이후는 선형 파이프라인 (조건부 분기 없음)
    # Match v3: rag_retriever → llm_reranker → match_scorer 순서로 LLM 삽입
    graph.add_edge("feature_extractor", "query_builder")
    graph.add_edge("query_builder", "rag_retriever")
    graph.add_edge("rag_retriever", "llm_reranker")
    graph.add_edge("llm_reranker", "match_scorer")
    graph.add_edge("match_scorer", "explanation_generator")
    graph.add_edge("explanation_generator", END)

    # 그래프 컴파일
    compiled = graph.compile()
    logger.info("match_graph_compiled", node_count=7)
    return compiled


# ── 모듈 레벨 싱글턴: 컴파일 1회 ──
# FastAPI 시작 시 1회만 컴파일되어 모든 요청이 재사용한다.
match_graph = build_match_graph()


# ============================================================
# SSE 포맷 헬퍼
# ============================================================

def _format_sse_event(event_type: str, data: dict) -> dict:
    """
    SSE 이벤트를 sse_starlette가 인식하는 dict로 포맷한다.

    sse_starlette.EventSourceResponse는 dict를 yield하면
    "event: {type}\\ndata: {json}\\n\\n" 형식으로 전송한다.

    Args:
        event_type: 이벤트 타입 (status, shared_features, match_result, error, done)
        data      : 이벤트 데이터 dict

    Returns:
        sse_starlette 호환 dict {"event": type, "data": json_string}
    """
    return {"event": event_type, "data": json.dumps(data, ensure_ascii=False)}


# ============================================================
# SSE 스트리밍 인터페이스
# ============================================================

async def run_match_agent(
    movie_id_1: str,
    movie_id_2: str,
    user_id: str = "",
) -> AsyncGenerator[dict, None]:
    """
    Movie Match Agent를 SSE 스트리밍 모드로 실행한다.

    asyncio.Queue 기반으로 그래프 이벤트를 수집하고,
    _KEEPALIVE_INTERVAL_SEC(15초)마다 keepalive status 이벤트를 발행하여
    장시간 노드 실행 중에도 SSE 연결이 끊기지 않도록 한다.

    SSE 이벤트 발행 시점:
    - movie_loader 완료 → status(loading) 발행
    - feature_extractor 완료 → status(analyzing) + shared_features 이벤트 발행
    - llm_reranker 완료 → status(reranking) 발행
    - match_scorer 완료 → status(scoring) 발행
    - explanation_generator 완료 → match_result 이벤트 발행
    - 모든 완료 → done 이벤트 발행
    - 에러 발생 → error 이벤트 + done 이벤트 발행

    Args:
        movie_id_1: 첫 번째 선택 영화 ID
        movie_id_2: 두 번째 선택 영화 ID
        user_id   : 요청 사용자 ID (선택, 로그용)

    Yields:
        SSE 이벤트 dict {"event": str, "data": str}
    """
    graph_start = time.perf_counter()

    # Prometheus outcome 추적 — finally 블록에서 최종 라벨 확정하여 기록.
    outcome_label = "success"

    # 초기 State 구성
    initial_state: MovieMatchState = {
        "movie_id_1": movie_id_1,
        "movie_id_2": movie_id_2,
        "user_id": user_id,
    }

    logger.info(
        "match_agent_stream_start",
        movie_id_1=movie_id_1,
        movie_id_2=movie_id_2,
        user_id=user_id or "anonymous",
    )

    # ── Queue 기반 비동기 이벤트 처리 ──
    queue: asyncio.Queue = asyncio.Queue()

    # keepalive에서 현재 진행 중인 단계 메시지를 유지하기 위한 상태
    current_phase = "loading"
    current_message = NODE_STATUS_MESSAGES["movie_loader"]["message"]

    async def _run_graph_to_queue():
        """
        LangGraph astream을 실행하고 각 노드 완료 이벤트를 Queue에 넣는다.

        그래프 완료 시 _SENTINEL, 에러 시 Exception 객체를 삽입하여
        소비자 루프에 종료를 알린다.
        """
        try:
            async for event in match_graph.astream(
                initial_state,
                stream_mode="updates",   # 각 노드 완료 시 updates dict를 emit
            ):
                await queue.put(event)
            # 정상 완료 → 센티넬 삽입
            await queue.put(_SENTINEL)
        except Exception as e:
            # 그래프 실행 에러 → Exception 객체 삽입
            await queue.put(e)

    # 그래프를 백그라운드 Task로 실행 (소비자 루프와 병렬 실행)
    graph_task = asyncio.create_task(_run_graph_to_queue())

    # 최종 State를 추적하기 위한 누적 dict (이벤트 발행 시점에 사용)
    final_state: dict = {}

    try:
        # ── 첫 번째 status 이벤트 즉시 발행 (movie_loader 시작 알림) ──
        node_info = NODE_STATUS_MESSAGES["movie_loader"]
        yield _format_sse_event("status", {
            "phase": node_info["phase"],
            "message": node_info["message"],
        })

        while True:
            try:
                # keepalive 타임아웃으로 이벤트 대기
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
                # 최종 State 누적
                final_state.update(updates)

                # 완료된 노드의 status 이벤트 발행
                node_info = NODE_STATUS_MESSAGES.get(node_name)
                if node_info:
                    current_phase = node_info["phase"]
                    current_message = node_info["message"]
                    yield _format_sse_event("status", {
                        "phase": current_phase,
                        "message": current_message,
                    })

                    # 다음 노드 진행 상태를 keepalive용 변수에만 반영 (이벤트 중복 발행 방지)
                    # 장시간 노드 실행 중 keepalive가 정확한 다음 단계 메시지를 표시한다.
                    next_node = _predict_next_node(node_name)
                    if next_node and next_node in NODE_STATUS_MESSAGES:
                        next_info = NODE_STATUS_MESSAGES[next_node]
                        current_phase = next_info["phase"]
                        current_message = next_info["message"]

                # ── 특수 이벤트: feature_extractor 완료 → shared_features 발행 ──
                # 프론트엔드에서 공통 특성 배지를 즉시 표시하기 위해 이른 시점에 발행
                if node_name == "feature_extractor":
                    shared: SharedFeatures | None = updates.get("shared_features")
                    if shared is not None:
                        shared_data = shared.model_dump() if hasattr(shared, "model_dump") else {}
                        yield _format_sse_event("shared_features", shared_data)

                # ── 특수 이벤트: explanation_generator 완료 → match_result 발행 ──
                # 설명이 채워진 최종 추천 결과를 한 번에 발행
                if node_name == "explanation_generator":
                    ranked_movies: list[MatchedMovie] = updates.get("ranked_movies", [])
                    if ranked_movies:
                        movies_data = [
                            m.model_dump() if hasattr(m, "model_dump") else m
                            for m in ranked_movies
                        ]
                        # 3편 미만이면 부분 결과 경고를 함께 전달 (설계서 §21-7 기준)
                        partial = len(ranked_movies) < 3
                        yield _format_sse_event("match_result", {
                            "movies": movies_data,
                            "partial": partial,
                            **({"warning": "조건에 정확히 맞는 영화가 적어 일부만 추천해요."} if partial else {}),
                        })
                    else:
                        # 추천 결과가 없음 (후보 부족 등)
                        outcome_label = "no_results"
                        yield _format_sse_event("error", {
                            "error_code": "NO_RESULTS",
                            "message": "조건에 맞는 영화를 찾지 못했어요. 다른 영화 조합을 시도해보세요.",
                        })

                # ── 에러 이벤트: movie_loader가 error 필드를 설정한 경우 ──
                # route_after_load에서 END로 분기하므로 그래프는 계속 실행되지 않음
                if node_name == "movie_loader" and updates.get("error"):
                    error_msg = updates["error"]
                    # 에러 코드 파싱: "MOVIE_NOT_FOUND:id1,id2" 또는 "SERVICE_UNAVAILABLE:..." 형식
                    if ":" in error_msg:
                        error_code, detail = error_msg.split(":", 1)
                    else:
                        error_code, detail = error_msg, ""

                    # Prometheus outcome: movie_loader 단계 에러 → "error" 로 분류
                    outcome_label = "error"

                    # 인프라 장애와 영화 미발견을 구분하여 사용자 메시지 분리
                    if error_code == "SERVICE_UNAVAILABLE":
                        yield _format_sse_event("error", {
                            "error_code": error_code,
                            "message": "서버 연결에 문제가 있어요. 잠시 후 다시 시도해주세요.",
                        })
                    else:
                        yield _format_sse_event("error", {
                            "error_code": error_code,
                            "message": f"선택한 영화를 찾을 수 없어요. (ID: {detail})",
                        })

        # 완료 이벤트
        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        logger.info(
            "match_agent_stream_done",
            movie_id_1=movie_id_1,
            movie_id_2=movie_id_2,
            elapsed_ms=round(graph_elapsed_ms, 1),
        )
        yield _format_sse_event("done", {})

    except Exception as e:
        graph_elapsed_ms = (time.perf_counter() - graph_start) * 1000
        outcome_label = "error"
        logger.error(
            "match_agent_stream_error",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(graph_elapsed_ms, 1),
        )
        yield _format_sse_event("error", {
            "error_code": "SEARCH_FAILED",
            "message": "처리 중 오류가 발생했어요. 잠시 후 다시 시도해주세요.",
        })
        yield _format_sse_event("done", {})

    finally:
        # 그래프 Task가 아직 실행 중이면 정리 (에러 발생 시 누수 방지)
        if not graph_task.done():
            graph_task.cancel()
            try:
                await graph_task
            except (asyncio.CancelledError, Exception):
                pass

        # ── Prometheus 메트릭 기록 (항상 실행, 클라이언트 조기 종료 시에도 반영) ──
        # outcome_label 은 실행 경로상 가장 마지막으로 확정된 값을 기록한다.
        # - success    : explanation_generator 가 ranked_movies 를 정상 반환
        # - no_results : 후보가 비어 NO_RESULTS 에러 발행
        # - error      : movie_loader 실패 / 기타 예외
        try:
            match_requests_total.labels(outcome=outcome_label).inc()
            match_duration_seconds.labels(outcome=outcome_label).observe(
                time.perf_counter() - graph_start,
            )
        except Exception as metric_err:
            # 메트릭 기록 실패는 서비스 중단 사유가 아니므로 경고 로그만
            logger.warning("match_metric_record_error", error=str(metric_err))


def _predict_next_node(completed_node: str) -> str | None:
    """
    완료된 노드 이후 실행될 다음 노드 이름을 반환한다.

    Movie Match 그래프는 선형 파이프라인이므로 정적 매핑으로 충분하다.
    keepalive 메시지가 정확한 다음 단계를 표시하도록 하기 위해 사용한다.

    Args:
        completed_node: 방금 완료된 노드 이름

    Returns:
        다음 노드 이름 (없으면 None)
    """
    # Movie Match 그래프의 고정 선형 순서 (Match v3: llm_reranker 포함)
    _NEXT_NODE: dict[str, str] = {
        "movie_loader": "feature_extractor",
        "feature_extractor": "query_builder",
        "query_builder": "rag_retriever",
        "rag_retriever": "llm_reranker",
        "llm_reranker": "match_scorer",
        "match_scorer": "explanation_generator",
    }
    return _NEXT_NODE.get(completed_node)


# ============================================================
# 동기 실행 인터페이스 (테스트/디버그용)
# ============================================================

async def run_match_agent_sync(
    movie_id_1: str,
    movie_id_2: str,
    user_id: str = "",
) -> MovieMatchState:
    """
    Movie Match Agent를 동기 모드로 실행하여 최종 State를 반환한다 (테스트/디버그용).

    LangGraph ainvoke()로 그래프를 실행하고 최종 State dict를 반환한다.
    SSE 이벤트를 발행하지 않으므로 빠르게 전체 결과를 확인할 수 있다.

    Args:
        movie_id_1: 첫 번째 선택 영화 ID
        movie_id_2: 두 번째 선택 영화 ID
        user_id   : 요청 사용자 ID (선택, 로그용)

    Returns:
        최종 MovieMatchState dict (LangGraph 출력)
    """
    sync_start = time.perf_counter()
    outcome_label = "success"

    initial_state: MovieMatchState = {
        "movie_id_1": movie_id_1,
        "movie_id_2": movie_id_2,
        "user_id": user_id,
    }

    logger.info(
        "match_agent_sync_start",
        movie_id_1=movie_id_1,
        movie_id_2=movie_id_2,
        user_id=user_id or "anonymous",
    )

    try:
        # ainvoke: 그래프를 끝까지 실행하고 최종 State 반환
        final_state: MovieMatchState = await match_graph.ainvoke(initial_state)

        elapsed_ms = (time.perf_counter() - sync_start) * 1000
        ranked_movies = final_state.get("ranked_movies", [])

        # 동기 모드도 outcome 을 세분화하여 Prometheus 에 기록
        if final_state.get("error"):
            outcome_label = "error"
        elif not ranked_movies:
            outcome_label = "no_results"
        else:
            outcome_label = "success"

        logger.info(
            "match_agent_sync_done",
            movie_id_1=movie_id_1,
            movie_id_2=movie_id_2,
            ranked_count=len(ranked_movies),
            elapsed_ms=round(elapsed_ms, 1),
        )

        try:
            match_requests_total.labels(outcome=outcome_label).inc()
            match_duration_seconds.labels(outcome=outcome_label).observe(
                time.perf_counter() - sync_start,
            )
        except Exception as metric_err:
            logger.warning("match_metric_record_error", error=str(metric_err))

        return final_state

    except Exception as e:
        elapsed_ms = (time.perf_counter() - sync_start) * 1000
        outcome_label = "error"
        try:
            match_requests_total.labels(outcome=outcome_label).inc()
            match_duration_seconds.labels(outcome=outcome_label).observe(
                time.perf_counter() - sync_start,
            )
        except Exception:
            pass
        logger.error(
            "match_agent_sync_error",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed_ms, 1),
        )
        # 에러 시 error 필드가 있는 최소 State 반환
        return {
            "movie_id_1": movie_id_1,
            "movie_id_2": movie_id_2,
            "user_id": user_id,
            "error": f"SEARCH_FAILED:{str(e)}",
            "ranked_movies": [],
        }
