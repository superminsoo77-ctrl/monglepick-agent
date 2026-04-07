"""
Qdrant 코사인 유사도 기반 유사 영화 검색 도구 (Phase 6 Tool 5).

기준 영화의 임베딩 벡터를 Qdrant에서 조회한 뒤,
코사인 유사도로 가장 가까운 영화들을 반환한다.

"이 영화랑 비슷한 영화 추천해줘" 등 유사 영화 탐색에 활용된다.
Neo4j의 SIMILAR_TO 관계와 달리 임베딩 공간에서의 의미적 유사도를 측정한다.

Qdrant API:
- query_points: 벡터 기반 ANN 검색 (approximate nearest neighbors)
- recommend: 특정 포인트 ID 기반 유사 포인트 검색 (임베딩 조회 없이 사용 가능)
"""

from __future__ import annotations

import asyncio

import structlog
from langchain_core.tools import tool
from qdrant_client.models import Filter, FieldCondition, MatchExcept

from monglepick.config import settings
from monglepick.db.clients import get_qdrant

logger = structlog.get_logger()

# Qdrant 검색 타임아웃 (초)
_QDRANT_TIMEOUT_SEC = 5.0

# Qdrant 컬렉션명
_COLLECTION = settings.QDRANT_COLLECTION  # "movies"


@tool
async def similar_movies(
    movie_id: str,
    limit: int = 5,
) -> list[dict]:
    """
    Qdrant 코사인 유사도 기반으로 기준 영화와 가장 유사한 영화를 검색한다.

    "이 영화랑 비슷한 영화 추천해줘"에 사용한다.
    Qdrant recommend API를 활용해 기준 영화 벡터를 별도 임베딩 없이 재사용한다.

    Args:
        movie_id: 기준 영화 ID (Qdrant 포인트 ID, 예: "157336")
        limit: 반환할 유사 영화 수 (기본 5, 최대 20)

    Returns:
        유사 영화 정보 dict 목록 (유사도 내림차순):
        [
            {
                "id": str,            # 영화 ID
                "title": str,         # 영화 제목
                "similarity_score": float,  # 코사인 유사도 (0.0~1.0)
                "genres": list[str],  # 장르 목록
                "director": str,      # 감독명
                "rating": float,      # TMDB 평점
                "release_year": int,  # 개봉 연도
                "poster_path": str,   # 포스터 경로
                "mood_tags": list[str], # 무드 태그
            }
        ]
        기준 영화 미존재 또는 에러 시: 빈 리스트 반환 (에러 전파 금지).
    """
    # limit 범위 보정 (최소 1, 최대 20)
    safe_limit = max(1, min(int(limit), 20))

    try:
        client = await get_qdrant()

        logger.info(
            "similar_movies_tool_start",
            movie_id=movie_id,
            limit=safe_limit,
        )

        # Qdrant recommend API:
        # - positive: 기준 포인트 ID(들) — 이 벡터 방향으로 검색
        # - negative: 제외 포인트 ID(들) — 기준 영화 자신 제외
        # - query_filter: 기준 영화 자신을 결과에서 제외
        # - with_payload: True → 메타데이터 반환
        #
        # recommend는 포인트 ID를 직접 받으므로 임베딩 API 호출이 불필요하다.
        # 기준 영화 벡터가 Qdrant에 없으면 빈 결과가 반환된다.
        #
        # [주의] Qdrant v1.7+에서는 query_points에 RecommendStrategy를 쓰는 방식도 가능하지만
        #        qdrant-client SDK의 recommend() 메서드가 더 직관적이다.

        # 기준 영화 자신을 결과에서 제외하는 필터
        # movie_id가 payload의 "id" 필드에 저장되어 있어야 정확히 필터링된다.
        # Qdrant 포인트 ID(정수형)와 payload의 문자열 ID가 다를 수 있으므로
        # 포인트 ID 기준으로 자신을 negative에 추가한다.
        response = await asyncio.wait_for(
            client.recommend(
                collection_name=_COLLECTION,
                positive=[movie_id],          # 기준 포인트 ID (문자열/정수 모두 지원)
                negative=[],                   # 음성 예시 없음
                limit=safe_limit + 1,          # 기준 영화 자신이 포함될 수 있으므로 +1
                with_payload=True,
                score_threshold=0.5,           # 유사도 0.5 미만은 무관한 영화로 제외
            ),
            timeout=_QDRANT_TIMEOUT_SEC,
        )

        # 결과 파싱 — 기준 영화 자신 제외
        results: list[dict] = []
        for hit in response:
            hit_id = str(hit.id)
            # 기준 영화 자신 제외
            if hit_id == str(movie_id):
                continue
            if len(results) >= safe_limit:
                break

            payload = hit.payload or {}
            results.append({
                "id": hit_id,
                "title": payload.get("title", ""),
                "similarity_score": round(float(hit.score), 4),
                "genres": payload.get("genres", []),
                "director": payload.get("director", ""),
                "rating": round(float(payload.get("rating", 0.0)), 1),
                "release_year": payload.get("release_year", 0),
                "poster_path": payload.get("poster_path", ""),
                "mood_tags": payload.get("mood_tags", []),
                "overview": payload.get("overview", ""),
                "ott_platforms": payload.get("ott_platforms", []),
            })

        logger.info(
            "similar_movies_tool_done",
            movie_id=movie_id,
            result_count=len(results),
            top_titles=[r.get("title", "") for r in results[:3]],
        )
        return results

    except asyncio.TimeoutError:
        logger.error(
            "similar_movies_tool_timeout",
            movie_id=movie_id,
            timeout_sec=_QDRANT_TIMEOUT_SEC,
        )
        return []

    except Exception as e:
        # Qdrant 연결 실패, 포인트 미존재 등 모든 예외 처리 (에러 전파 금지)
        logger.error(
            "similar_movies_tool_error",
            movie_id=movie_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return []
