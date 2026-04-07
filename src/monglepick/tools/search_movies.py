"""
내부 RAG 기반 영화 검색 도구 (Phase 6 Tool 1).

Qdrant 벡터 검색 + Elasticsearch BM25 + Neo4j 그래프 검색을
RRF(Reciprocal Rank Fusion, k=60)로 합산하여 최적 영화 목록을 반환한다.

기존 hybrid_search() 함수를 LangChain @tool 인터페이스로 래핑한다.
info 의도(영화 정보 조회) 및 search 의도 보조에 사용된다.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from langchain_core.tools import tool

from monglepick.rag.hybrid_search import SearchResult, hybrid_search

logger = structlog.get_logger()

# 도구 실행 타임아웃 (초) — Upstage 임베딩 API 지연 고려
_SEARCH_TIMEOUT_SEC = 15.0


@tool
async def search_movies(
    query: str,
    filters: dict[str, Any] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    내부 RAG(Qdrant + Elasticsearch + Neo4j)를 활용해 영화를 검색한다.

    사용자의 자연어 쿼리를 의미적으로 해석하여 관련 영화를 반환한다.
    장르, 연도, OTT 플랫폼 등 선택적 필터를 추가로 적용할 수 있다.

    Args:
        query: 검색할 자연어 쿼리 (예: "크리스토퍼 놀란의 SF 영화")
        filters: 추가 필터 조건 dict (아래 키 지원):
            - genre_filter: list[str] — 장르 목록 (예: ["SF", "액션"])
            - mood_tags: list[str] — 무드 태그 목록 (예: ["긴장감 있는"])
            - ott_filter: list[str] — OTT 플랫폼 (예: ["Netflix"])
            - min_rating: float — 최소 평점 (예: 7.5)
            - year_from: int — 최소 개봉 연도 (예: 2010)
            - year_to: int — 최대 개봉 연도 (예: 2024)
            - director: str — 감독명 (예: "봉준호")
            - exclude_ids: list[str] — 제외할 영화 ID 목록
        limit: 반환할 영화 수 (기본 5, 최대 15)

    Returns:
        영화 정보 dict 목록. 각 항목:
        {
            "id": str,           # 영화 ID
            "title": str,        # 한국어 제목
            "score": float,      # RRF 합산 점수
            "source": str,       # 검색 엔진 (qdrant/es/neo4j/rrf)
            "genres": list[str], # 장르 목록
            "director": str,     # 감독명
            "rating": float,     # TMDB 평점
            "release_year": int, # 개봉 연도
            "poster_path": str,  # 포스터 경로
            "ott_platforms": list[str], # 시청 가능 OTT 플랫폼
        }
        에러 발생 시 빈 리스트 반환 (에러 전파 금지).
    """
    try:
        # 필터 조건 파싱 — 없으면 빈 dict 처리
        f = filters or {}

        # year_from/year_to → year_range 튜플 변환
        year_range: tuple[int, int] | None = None
        year_from = f.get("year_from")
        year_to = f.get("year_to")
        if year_from or year_to:
            year_range = (
                int(year_from) if year_from else 1900,
                int(year_to) if year_to else 2030,
            )

        # limit 범위 보정 (최소 1, 최대 15)
        safe_limit = max(1, min(int(limit), 15))

        logger.info(
            "search_movies_tool_start",
            query_preview=query[:80],
            filter_keys=list(f.keys()),
            limit=safe_limit,
        )

        # 하이브리드 검색 실행 (타임아웃 적용)
        results: list[SearchResult] = await asyncio.wait_for(
            hybrid_search(
                query=query,
                top_k=safe_limit,
                genre_filter=f.get("genre_filter"),
                mood_tags=f.get("mood_tags"),
                ott_filter=f.get("ott_filter"),
                min_rating=f.get("min_rating"),
                year_range=year_range,
                director=f.get("director"),
                exclude_ids=f.get("exclude_ids"),
            ),
            timeout=_SEARCH_TIMEOUT_SEC,
        )

        # SearchResult → dict 변환 (tool 반환값은 JSON 직렬화 가능해야 함)
        output: list[dict[str, Any]] = []
        for r in results[:safe_limit]:
            meta = r.metadata or {}
            output.append({
                "id": r.movie_id,
                "title": r.title,
                "score": round(r.score, 6),
                "source": r.source,
                # metadata에서 추가 필드 추출 (hybrid_search가 payload를 metadata에 포함)
                "genres": meta.get("genres", []),
                "director": meta.get("director", ""),
                "rating": meta.get("rating", 0.0),
                "release_year": meta.get("release_year", 0),
                "poster_path": meta.get("poster_path", ""),
                "ott_platforms": meta.get("ott_platforms", []),
                "overview": meta.get("overview", ""),
                "mood_tags": meta.get("mood_tags", []),
            })

        logger.info(
            "search_movies_tool_done",
            result_count=len(output),
            top_results=[r.get("title", "") for r in output[:3]],
        )
        return output

    except asyncio.TimeoutError:
        # 타임아웃: 빈 결과 반환 (에러 전파 금지)
        logger.error(
            "search_movies_tool_timeout",
            query_preview=query[:80],
            timeout_sec=_SEARCH_TIMEOUT_SEC,
        )
        return []

    except Exception as e:
        # 예상치 못한 에러: 빈 결과 반환 (에러 전파 금지)
        logger.error(
            "search_movies_tool_error",
            error=str(e),
            error_type=type(e).__name__,
            query_preview=query[:80],
        )
        return []
