"""
TMDB API 기반 영화 상세 정보 조회 도구 (Phase 6 Tool 2).

TMDB(The Movie Database) REST API를 통해 영화 ID로 상세 정보를 조회한다.
info 의도 처리 시 tool_executor_node에서 호출된다.

TMDB API v3 엔드포인트:
- 영화 상세: GET /movie/{movie_id}?language=ko-KR
- Credits:   GET /movie/{movie_id}/credits?language=ko-KR
- Providers: GET /movie/{movie_id}/watch/providers
"""

from __future__ import annotations

import os

import httpx
import structlog
from langchain_core.tools import tool

from monglepick.config import settings

logger = structlog.get_logger()

# TMDB API 설정
_TMDB_BASE_URL = settings.TMDB_BASE_URL  # "https://api.themoviedb.org/3"
_TMDB_API_KEY = settings.TMDB_API_KEY    # .env에서 로드
_TMDB_TIMEOUT_SEC = 5.0                  # TMDB API 응답 타임아웃 (초)

# 한국 OTT 플랫폼 provider_id → 한국어 이름 매핑 (TMDB Watch Providers 기준)
_KR_PROVIDER_NAMES: dict[int, str] = {
    8:    "Netflix",
    337:  "Disney+",
    356:  "웨이브",
    412:  "왓챠",
    97:   "애플TV+",
    567:  "티빙",
    619:  "쿠팡플레이",
    350:  "애플TV",
    11:   "Mubi",
}


@tool
async def movie_detail(movie_id: str) -> dict | str:
    """
    TMDB API로 영화 상세 정보를 조회한다.

    영화 ID로 제목, 줄거리, 감독, 출연진, 장르, OTT 제공 플랫폼 등
    상세 정보를 한국어로 반환한다.

    Args:
        movie_id: TMDB 영화 ID (예: "tt0816692" 또는 "157336")

    Returns:
        성공 시 dict:
        {
            "id": str,               # TMDB 영화 ID
            "title": str,            # 한국어 제목
            "title_en": str,         # 영문 원제
            "release_date": str,     # 개봉일 (YYYY-MM-DD)
            "runtime": int,          # 상영시간 (분)
            "genres": list[str],     # 장르 목록 (한국어)
            "overview": str,         # 줄거리
            "director": str,         # 감독명
            "cast": list[str],       # 주연 배우 (상위 5명)
            "ott_providers": list[str], # 한국 OTT 플랫폼 목록
            "poster_path": str,      # 포스터 경로 (/xxx.jpg)
            "rating": float,         # TMDB 평점
            "vote_count": int,       # 투표 수
        }
        404 시: "영화 정보를 찾을 수 없어요"
        에러 시: "영화 정보를 잠시 불러올 수 없어요"
    """
    # API 키 누락 시 조기 반환
    if not _TMDB_API_KEY:
        logger.warning("movie_detail_tool_no_api_key")
        return "영화 정보를 불러오려면 TMDB API 키가 필요해요"

    try:
        # 공통 쿼리 파라미터 (한국어 응답 요청)
        base_params = {
            "api_key": _TMDB_API_KEY,
            "language": "ko-KR",
        }

        async with httpx.AsyncClient(timeout=_TMDB_TIMEOUT_SEC) as client:
            # ① 영화 기본 정보 조회
            detail_url = f"{_TMDB_BASE_URL}/movie/{movie_id}"
            detail_resp = await client.get(detail_url, params=base_params)

            # 404: 영화 없음
            if detail_resp.status_code == 404:
                logger.warning(
                    "movie_detail_tool_not_found",
                    movie_id=movie_id,
                    status=detail_resp.status_code,
                )
                return "영화 정보를 찾을 수 없어요"

            # 기타 HTTP 에러
            detail_resp.raise_for_status()
            detail_data = detail_resp.json()

            # ② Credits (감독/출연진) 조회
            credits_url = f"{_TMDB_BASE_URL}/movie/{movie_id}/credits"
            credits_resp = await client.get(credits_url, params=base_params)
            credits_data = credits_resp.json() if credits_resp.is_success else {}

            # ③ Watch Providers (OTT 제공) 조회
            providers_url = f"{_TMDB_BASE_URL}/movie/{movie_id}/watch/providers"
            providers_resp = await client.get(
                providers_url, params={"api_key": _TMDB_API_KEY}
            )
            providers_data = providers_resp.json() if providers_resp.is_success else {}

        # ── 장르 파싱 (한국어) ──
        genres: list[str] = [
            g.get("name", "") for g in detail_data.get("genres", []) if g.get("name")
        ]

        # ── 감독 파싱 ──
        crew = credits_data.get("crew", [])
        directors = [
            p.get("name", "") for p in crew
            if p.get("job") == "Director" and p.get("name")
        ]
        director = directors[0] if directors else ""

        # ── 주연 배우 (상위 5명) ──
        cast_list = [
            p.get("name", "") for p in credits_data.get("cast", [])[:5]
            if p.get("name")
        ]

        # ── 한국 OTT 플랫폼 파싱 ──
        # TMDB Watch Providers: results.KR.flatrate (스트리밍 서비스)
        kr_results = providers_data.get("results", {}).get("KR", {})
        flatrate = kr_results.get("flatrate", [])  # 구독형 스트리밍
        rent = kr_results.get("rent", [])           # 렌탈
        buy = kr_results.get("buy", [])             # 구매

        # 스트리밍 우선, 중복 제거
        ott_ids_seen: set[int] = set()
        ott_providers: list[str] = []
        for provider in flatrate + rent + buy:
            pid = provider.get("provider_id")
            if pid and pid not in ott_ids_seen:
                ott_ids_seen.add(pid)
                name = _KR_PROVIDER_NAMES.get(pid, provider.get("provider_name", ""))
                if name:
                    ott_providers.append(name)

        result = {
            "id": str(detail_data.get("id", movie_id)),
            "title": detail_data.get("title", ""),
            "title_en": detail_data.get("original_title", ""),
            "release_date": detail_data.get("release_date", ""),
            "runtime": detail_data.get("runtime") or 0,
            "genres": genres,
            "overview": detail_data.get("overview", ""),
            "director": director,
            "cast": cast_list,
            "ott_providers": ott_providers,
            "poster_path": detail_data.get("poster_path", ""),
            "rating": round(float(detail_data.get("vote_average", 0.0)), 1),
            "vote_count": detail_data.get("vote_count", 0),
            "tagline": detail_data.get("tagline", ""),
            "status": detail_data.get("status", ""),
        }

        logger.info(
            "movie_detail_tool_done",
            movie_id=movie_id,
            title=result["title"],
            ott_count=len(ott_providers),
        )
        return result

    except httpx.TimeoutException:
        # TMDB API 응답 타임아웃
        logger.error(
            "movie_detail_tool_timeout",
            movie_id=movie_id,
            timeout_sec=_TMDB_TIMEOUT_SEC,
        )
        return "영화 정보를 잠시 불러올 수 없어요"

    except httpx.HTTPStatusError as e:
        # HTTP 오류 (5xx 등)
        logger.error(
            "movie_detail_tool_http_error",
            movie_id=movie_id,
            status=e.response.status_code,
        )
        return "영화 정보를 잠시 불러올 수 없어요"

    except Exception as e:
        # 예상치 못한 에러 (에러 전파 금지)
        logger.error(
            "movie_detail_tool_error",
            movie_id=movie_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return "영화 정보를 잠시 불러올 수 없어요"
