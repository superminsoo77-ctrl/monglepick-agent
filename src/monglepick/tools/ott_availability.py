"""
TMDB Watch Providers API 기반 OTT 시청 가능 여부 조회 도구 (Phase 6 Tool 4).

TMDB Watch Providers API로 특정 영화의 한국 OTT 스트리밍 서비스 목록을 반환한다.
info 의도 처리 시 "어디서 볼 수 있나요?" 질문에 응답하는 용도로 사용된다.

TMDB Watch Providers API:
- GET /movie/{movie_id}/watch/providers
- 응답: {results: {KR: {flatrate: [...], rent: [...], buy: [...]}}}
"""

from __future__ import annotations

import httpx
import structlog
from langchain_core.tools import tool

from monglepick.config import settings

logger = structlog.get_logger()

# TMDB API 설정
_TMDB_BASE_URL = settings.TMDB_BASE_URL  # "https://api.themoviedb.org/3"
_TMDB_API_KEY = settings.TMDB_API_KEY
_TMDB_TIMEOUT_SEC = 5.0

# TMDB provider_id → 한국어 서비스명 매핑
# 출처: TMDB Watch Provider 공식 목록 (KR 기준)
_PROVIDER_ID_TO_NAME: dict[int, str] = {
    8:    "Netflix",
    337:  "Disney+",
    356:  "웨이브",
    412:  "왓챠",
    97:   "애플TV+",
    350:  "애플TV",
    567:  "티빙",
    619:  "쿠팡플레이",
    11:   "Mubi",
    192:  "YouTube Premium",
    100:  "Amazon Prime Video",
}

# 시청 유형 우선순위 (flatrate가 가장 좋은 방법 — 추가 비용 없음)
_WATCH_TYPE_PRIORITY = ["flatrate", "rent", "buy"]
_WATCH_TYPE_LABEL = {
    "flatrate": "구독",
    "rent": "렌탈",
    "buy": "구매",
}


@tool
async def ott_availability(
    movie_id: str,
    region: str = "KR",
) -> list[str]:
    """
    TMDB Watch Providers API로 특정 영화의 한국 OTT 스트리밍 서비스를 조회한다.

    "이 영화 어디서 볼 수 있어요?"와 같은 질문에 응답할 때 사용한다.

    Args:
        movie_id: TMDB 영화 ID (예: "157336")
        region: 국가 코드 ISO 3166-1 (기본 "KR" — 한국)

    Returns:
        시청 가능한 OTT 서비스 이름 목록 (구독 → 렌탈 → 구매 순서로 중복 제거).
        예: ["Netflix", "티빙", "웨이브"]
        시청 불가 또는 에러 시: 빈 리스트 반환 (에러 전파 금지).
    """
    # API 키 누락 시 조기 반환
    if not _TMDB_API_KEY:
        logger.warning("ott_availability_tool_no_api_key", movie_id=movie_id)
        return []

    try:
        url = f"{_TMDB_BASE_URL}/movie/{movie_id}/watch/providers"
        params = {"api_key": _TMDB_API_KEY}

        async with httpx.AsyncClient(timeout=_TMDB_TIMEOUT_SEC) as client:
            resp = await client.get(url, params=params)

            # 404: 영화 없음 → 빈 리스트
            if resp.status_code == 404:
                logger.warning(
                    "ott_availability_tool_not_found",
                    movie_id=movie_id,
                )
                return []

            resp.raise_for_status()
            data = resp.json()

        # 요청 국가 코드의 결과 추출
        region_data: dict = data.get("results", {}).get(region, {})

        if not region_data:
            # 해당 지역에서 시청 불가
            logger.info(
                "ott_availability_tool_no_providers",
                movie_id=movie_id,
                region=region,
            )
            return []

        # 시청 유형 순서대로 서비스명 수집 (구독 → 렌탈 → 구매, 중복 제거)
        seen_ids: set[int] = set()
        providers: list[str] = []

        for watch_type in _WATCH_TYPE_PRIORITY:
            for provider in region_data.get(watch_type, []):
                pid = provider.get("provider_id")
                if pid is None or pid in seen_ids:
                    continue
                seen_ids.add(pid)
                # 매핑 테이블 우선, 없으면 TMDB 반환값 사용
                name = _PROVIDER_ID_TO_NAME.get(pid, provider.get("provider_name", ""))
                if name:
                    providers.append(name)

        logger.info(
            "ott_availability_tool_done",
            movie_id=movie_id,
            region=region,
            providers=providers,
        )
        return providers

    except httpx.TimeoutException:
        logger.error(
            "ott_availability_tool_timeout",
            movie_id=movie_id,
            timeout_sec=_TMDB_TIMEOUT_SEC,
        )
        return []

    except httpx.HTTPStatusError as e:
        logger.error(
            "ott_availability_tool_http_error",
            movie_id=movie_id,
            status=e.response.status_code,
        )
        return []

    except Exception as e:
        # 예상치 못한 에러 (에러 전파 금지)
        logger.error(
            "ott_availability_tool_error",
            movie_id=movie_id,
            region=region,
            error=str(e),
            error_type=type(e).__name__,
        )
        return []
