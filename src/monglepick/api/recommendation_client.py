"""
Backend 추천 로그 API 비동기 클라이언트 (2026-04-15 신규).

Agent 가 `recommendation_ranker` 완료 후 `movie_card` SSE 를 발행하는 시점에
추천된 영화 N 개를 한 번에 Backend `POST /api/v1/recommendations/internal/batch` 로
전송하여 MySQL `recommendation_log` 테이블에 기록한다.

이 저장이 없으면 유저 대면 "마이픽 > 추천 내역" + 관리자 "AI 추천 분석" 탭이
모두 빈 화면으로 나온다. 본 클라이언트가 유일한 쓰기 경로다.

Backend 통신: httpx.AsyncClient 싱글턴 (point_client 의 _get_http_client 재사용)
인증: X-Service-Key 헤더 (settings.SERVICE_API_KEY)
"""

from __future__ import annotations

from typing import Any, Iterable

import structlog

# point_client 의 싱글턴 httpx 클라이언트를 공유한다
# (같은 BACKEND_BASE_URL + X-Service-Key 헤더 + 타임아웃 설정).
from monglepick.api.point_client import _get_http_client

logger = structlog.get_logger(__name__)


def _movie_to_item(movie: Any) -> dict[str, Any]:
    """
    RankedMovie (또는 dict) 를 Backend Item DTO 에 맞는 dict 로 직렬화한다.

    - RankedMovie 는 Pydantic 모델. score_detail 하위의 cf_score/cbf_score/
      hybrid_score/genre_match/mood_match 를 평평하게 올려 보낸다.
    - explanation 필드 → Backend `reason` 컬럼 (NOT NULL). 빈 문자열이면 " " 로.
    - Backend DTO 의 카멜케이스 필드명을 정확히 따른다.
    """
    # Pydantic v2 기준 dump. dict 이면 그대로 사용.
    if hasattr(movie, "model_dump"):
        d = movie.model_dump()
    elif isinstance(movie, dict):
        d = movie
    else:
        d = dict(movie)  # fallback

    score_detail = d.get("score_detail") or {}
    explanation = d.get("explanation") or ""

    # Entity 의 `score` 가 NOT NULL 이므로 최소값 보장 (ScoreDetail.hybrid_score 또는 0.0)
    final_score = (
        score_detail.get("hybrid_score")
        if score_detail.get("hybrid_score") is not None
        else 0.0
    )

    return {
        "movieId": d.get("id", ""),
        "rankPosition": d.get("rank", 0),
        "reason": explanation if explanation else " ",
        "score": final_score,
        "cfScore": score_detail.get("cf_score"),
        "cbfScore": score_detail.get("cbf_score"),
        "hybridScore": score_detail.get("hybrid_score"),
        "genreMatch": score_detail.get("genre_match"),
        "moodMatch": score_detail.get("mood_match"),
    }


async def save_recommendation_logs(
    user_id: str,
    session_id: str,
    user_intent: str,
    emotion: str,
    mood_tags: Iterable[str],
    response_time_ms: int | None,
    model_version: str,
    movies: list,
) -> list[int | None]:
    """
    추천 로그 N 개를 Backend 에 배치 저장한다.

    호출 시점: `recommendation_ranker` 완료 → `movie_card` yield 직전.

    Args:
        user_id: 사용자 ID (JWT 검증 통과한 실제 userId)
        session_id: 채팅 세션 UUID (chat_session_archive.session_id 와 동일)
        user_intent: Intent-First 아키텍처에서 LLM 이 추출한 사용자 의도 요약
        emotion: 감정 라벨 — 현재 Entity 컬럼 없어 메타로만 전달
        mood_tags: 무드 태그 목록 — 현재 Entity 컬럼 없어 메타로만 전달
        response_time_ms: 그래프 전체 소요 시간 (ms, nullable)
        model_version: "chat-v3.4" 등 식별자
        movies: RankedMovie 리스트 (순서 보존 필수)

    Returns:
        저장된 recommendation_log_id 리스트 (movies 와 동일 길이).
        일부 영화의 movieId 가 Backend movies 테이블에 없으면 해당 자리는 None.
        전체 요청 실패 (네트워크/5xx) 시 빈 리스트 반환 (graceful: Agent 는
        movie_card 를 계속 yield 하되 recommendation_log_id 는 None 으로 나감).
    """
    if not movies:
        return []

    items = [_movie_to_item(m) for m in movies]
    payload = {
        "userId": user_id,
        "sessionId": session_id,
        "userIntent": user_intent or "",
        "emotion": emotion or "",
        "moodTags": list(mood_tags) if mood_tags else [],
        "responseTimeMs": response_time_ms,
        "modelVersion": model_version,
        "items": items,
    }

    try:
        client = await _get_http_client()
        resp = await client.post(
            "/api/v1/recommendations/internal/batch",
            json=payload,
        )

        if resp.status_code == 200:
            data = resp.json() or {}
            log_ids = data.get("recommendationLogIds", []) or []
            saved_count = sum(1 for lid in log_ids if lid is not None)
            logger.info(
                "recommendation_log_batch_saved",
                user_id=user_id,
                session_id=session_id,
                requested=len(items),
                saved=saved_count,
                skipped=len(items) - saved_count,
                log_ids=log_ids,
            )
            return log_ids

        # Backend 실패 — graceful: 빈 리스트 반환 (Agent 는 movie_card 노출 유지)
        logger.error(
            "recommendation_log_batch_failed",
            user_id=user_id,
            session_id=session_id,
            status_code=resp.status_code,
            body=resp.text[:300],
        )
        return []

    except Exception as e:
        # 네트워크/타임아웃 등 — graceful
        logger.error(
            "recommendation_log_batch_error",
            user_id=user_id,
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return []
