"""
Recommend FastAPI — Co-watched CF 클라이언트.

Movie Match 의 rag_retriever 가 RRF 병합용 후보 소스로 호출한다.
- 엔드포인트: POST {RECOMMEND_BASE_URL}/api/v2/match/co-watched
- 타임아웃: settings.MATCH_COWATCH_TIMEOUT (기본 3초)
- 실패/타임아웃/빈 응답 모두 graceful fallback → 빈 리스트 반환 (에이전트 보호)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog

from monglepick.config import settings
from monglepick.metrics import (
    match_cowatch_duration_seconds,
    match_cowatch_request_total,
)
from monglepick.rag.hybrid_search import SearchResult

logger = structlog.get_logger(__name__)

# httpx.AsyncClient 싱글턴 — point_client 와 동일한 패턴으로 커넥션 풀 재사용
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_http_client() -> httpx.AsyncClient:
    """Recommend FastAPI 전용 httpx 클라이언트 싱글턴."""
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is None:
            _client = httpx.AsyncClient(
                base_url=settings.RECOMMEND_BASE_URL,
                timeout=httpx.Timeout(
                    settings.MATCH_COWATCH_TIMEOUT,
                    connect=min(2.0, settings.MATCH_COWATCH_TIMEOUT),
                ),
            )
    return _client


async def close_client() -> None:
    """앱 종료 시 httpx 클라이언트 정리."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def fetch_cowatched_candidates(
    movie_id_1: str,
    movie_id_2: str,
    top_k: int | None = None,
    rating_threshold: float = 3.5,
) -> list[SearchResult]:
    """
    두 영화 모두 높게 평가한 사용자의 다른 영화 목록을 Recommend API 로 조회한다.

    Returns:
        SearchResult 리스트 (source="cf", score=cf_score). 실패 시 빈 리스트.
        결과 형식은 hybrid_search 의 다른 소스와 동일하므로 RRF 에 바로 투입 가능.
    """
    # 입력 검증 — 동일 영화면 CF 의미 없음
    if not movie_id_1 or not movie_id_2 or movie_id_1 == movie_id_2:
        return []

    effective_top_k = top_k or settings.MATCH_COWATCH_TOP_K

    # Prometheus: 호출 시작 시각 기록 → outcome 확정 시점에 duration 기록.
    # outcome 은 return 지점별로 분기되므로 별도 변수에 유지한다.
    start = time.perf_counter()
    outcome = "exception"

    try:
        client = await _get_http_client()
        resp = await client.post(
            "/api/v2/match/co-watched",
            json={
                "movie_id_1": movie_id_1,
                "movie_id_2": movie_id_2,
                "top_k": effective_top_k,
                "rating_threshold": rating_threshold,
            },
        )
    except httpx.TimeoutException:
        # 타임아웃은 경고 로그만 남기고 빈 결과 (CF 는 보조 소스이므로 필수 아님)
        logger.warning(
            "match_cowatch_timeout",
            movie_id_1=movie_id_1,
            movie_id_2=movie_id_2,
            timeout=settings.MATCH_COWATCH_TIMEOUT,
        )
        outcome = "timeout"
        match_cowatch_request_total.labels(outcome=outcome).inc()
        match_cowatch_duration_seconds.observe(time.perf_counter() - start)
        return []
    except httpx.HTTPError as e:
        # 네트워크 장애 등 HTTP 클라이언트 오류
        logger.warning(
            "match_cowatch_http_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        outcome = "http_error"
        match_cowatch_request_total.labels(outcome=outcome).inc()
        match_cowatch_duration_seconds.observe(time.perf_counter() - start)
        return []
    except Exception as e:
        # 예측 못 한 오류 — 상위로 전파하지 않고 빈 결과 반환
        logger.error(
            "match_cowatch_unexpected_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        outcome = "exception"
        match_cowatch_request_total.labels(outcome=outcome).inc()
        match_cowatch_duration_seconds.observe(time.perf_counter() - start)
        return []

    # 2xx 가 아니면 실패 간주
    if resp.status_code != 200:
        logger.warning(
            "match_cowatch_non_200",
            status=resp.status_code,
            body_preview=resp.text[:200],
        )
        outcome = "non_200"
        match_cowatch_request_total.labels(outcome=outcome).inc()
        match_cowatch_duration_seconds.observe(time.perf_counter() - start)
        return []

    # ── 응답 파싱 및 SearchResult 변환 ──
    try:
        data: dict[str, Any] = resp.json()
    except Exception as e:
        logger.warning("match_cowatch_json_decode_error", error=str(e))
        outcome = "exception"
        match_cowatch_request_total.labels(outcome=outcome).inc()
        match_cowatch_duration_seconds.observe(time.perf_counter() - start)
        return []

    movies = data.get("movies") or []
    if not isinstance(movies, list):
        outcome = "empty"
        match_cowatch_request_total.labels(outcome=outcome).inc()
        match_cowatch_duration_seconds.observe(time.perf_counter() - start)
        return []

    results: list[SearchResult] = []
    for item in movies:
        if not isinstance(item, dict):
            continue
        mid = item.get("movie_id")
        if not mid:
            continue
        # cf_score 를 SearchResult.score 에 담아 RRF 에서 순위 계산에 활용
        results.append(
            SearchResult(
                movie_id=str(mid),
                title="",  # 제목은 RRF 병합 후 Qdrant retrieve 에서 보강
                score=float(item.get("cf_score", 0.0)),
                source="cf",
                metadata={
                    "co_user_count": int(item.get("co_user_count", 0)),
                    "avg_rating": float(item.get("avg_rating", 0.0)),
                    "cf_score": float(item.get("cf_score", 0.0)),
                },
            )
        )

    logger.info(
        "match_cowatch_fetched",
        count=len(results),
        movie_id_1=movie_id_1,
        movie_id_2=movie_id_2,
        top_preview=[
            {"id": r.movie_id, "cf_score": round(r.score, 4)}
            for r in results[:5]
        ],
    )
    # 성공 outcome 분기: 빈 결과는 "empty" 로 구분 (네트워크 성공 + 데이터 없음)
    outcome = "ok" if results else "empty"
    match_cowatch_request_total.labels(outcome=outcome).inc()
    match_cowatch_duration_seconds.observe(time.perf_counter() - start)
    return results
