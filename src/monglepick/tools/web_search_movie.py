"""
DuckDuckGo 기반 영화 정보 외부 검색 도구 (Phase 6 Tool 8).

내부 DB(TMDB/Qdrant/ES)에 줄거리나 상세 정보가 부족한 영화에 대해
DuckDuckGo 웹 검색을 통해 Wikipedia/나무위키 등에서 실제 정보를 수집한다.

사용 케이스:
- info 의도에서 영화 상세 정보 보강
- explanation_generator에서 overview 부족 영화 보강 (enrich_movies_batch 경유)
- 최신 개봉작 등 TMDB에 아직 반영되지 않은 정보 수집
"""

from __future__ import annotations

import structlog
from langchain_core.tools import tool

from monglepick.utils.movie_info_enricher import enrich_movie_overview

logger = structlog.get_logger()


@tool
async def web_search_movie(title: str, title_en: str = "", director: str = "",
                           release_year: str = "", overview: str = "") -> dict | str:
    """
    DuckDuckGo 웹 검색으로 영화의 부족한 정보를 보강한다.

    영화 제목으로 Wikipedia, 나무위키 등을 검색하여
    줄거리, 연출 스타일, 출연진 정보 등을 수집한다.
    내부 DB에 줄거리가 없거나 50자 미만일 때 유용하다.

    Args:
        title: 영화 한국어 제목 (필수, 예: "올드보이")
        title_en: 영화 영문 제목 (예: "Oldboy")
        director: 감독명 (예: "박찬욱")
        release_year: 개봉연도 (예: "2003")
        overview: 기존 줄거리 (부족하면 외부 검색으로 보강)

    Returns:
        성공 시 dict:
        {
            "title": str,              # 영화 제목
            "overview": str,           # 보강된 줄거리
            "enriched": bool,          # 외부 검색으로 보강되었는지 여부
            "original_overview": str,  # 원본 줄거리
        }
        에러 시: "영화 정보를 외부에서 찾을 수 없어요"
    """
    if not title:
        return "검색할 영화 제목이 필요해요"

    try:
        # 기존 정보를 dict로 구성하여 enrich_movie_overview에 전달
        movie_dict = {
            "title": title,
            "title_en": title_en,
            "director": director,
            "release_year": release_year,
            "overview": overview,
        }

        # 외부 검색 보강 실행
        enriched = await enrich_movie_overview(movie_dict)

        result = {
            "title": title,
            "overview": enriched.get("overview", overview),
            "enriched": enriched.get("_enriched", False),
            "original_overview": overview,
        }

        logger.info(
            "web_search_movie_tool_done",
            title=title,
            enriched=result["enriched"],
            overview_len=len(result["overview"]),
        )
        return result

    except Exception as e:
        # 에러 전파 금지 — 모든 예외를 안내 문자열로 반환
        logger.error(
            "web_search_movie_tool_error",
            title=title,
            error=str(e),
            error_type=type(e).__name__,
        )
        return "영화 정보를 외부에서 찾을 수 없어요"
