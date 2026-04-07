"""
카카오맵 API 기반 근처 영화관 검색 도구 (Phase 6 Tool 3).

카카오 로컬 API(키워드 검색)를 사용해 사용자 위치 기반 영화관을 검색한다.
theater 의도 처리 시 tool_executor_node에서 호출된다.

카카오 로컬 API 문서:
- 키워드로 장소 검색: GET https://dapi.kakao.com/v2/local/search/keyword.json
- Authorization: KakaoAK {REST_API_KEY}
"""

from __future__ import annotations

import asyncio

import httpx
import structlog
from langchain_core.tools import tool

from monglepick.config import settings

logger = structlog.get_logger()

# 카카오 로컬 API 설정
_KAKAO_API_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
_KAKAO_API_KEY = settings.KAKAO_API_KEY   # .env에서 로드
_KAKAO_TIMEOUT_SEC = 5.0                  # 카카오 API 응답 타임아웃 (초)

# 영화관 체인명 키워드 목록 — 카카오 키워드 검색에 순차적으로 시도
# 카카오 API는 단일 키워드만 지원하므로 여러 번 호출 후 병합한다
_THEATER_KEYWORDS = ["CGV", "롯데시네마", "메가박스"]

# 검색 반경 최대값 (미터) — 카카오 API 최대 20,000m
_MAX_RADIUS_M = 20_000


@tool
async def theater_search(
    latitude: float,
    longitude: float,
    radius: int = 5000,
) -> list[dict] | str:
    """
    카카오맵 API로 사용자 위치 기반 근처 영화관을 검색한다.

    CGV, 롯데시네마, 메가박스 3대 체인을 동시 검색하여 거리순으로 정렬한다.

    Args:
        latitude: 사용자 위도 (예: 37.5665)
        longitude: 사용자 경도 (예: 126.9780)
        radius: 검색 반경 (미터, 기본 5,000 / 최대 20,000)

    Returns:
        성공 시 영화관 정보 dict 목록 (거리 오름차순):
        [
            {
                "theater_id": str,   # 카카오 장소 ID
                "name": str,         # 영화관명 (예: "CGV 강남")
                "address": str,      # 도로명 주소
                "phone": str,        # 전화번호
                "latitude": float,   # 위도
                "longitude": float,  # 경도
                "distance_m": int,   # 검색 위치로부터 거리 (미터)
                "place_url": str,    # 카카오맵 상세 URL
            }
        ]
        API 키 누락 또는 에러 시: "영화관 검색이 잠시 안 돼요" 문자열 반환
    """
    # API 키 누락 시 조기 반환
    if not _KAKAO_API_KEY:
        logger.warning("theater_search_tool_no_api_key")
        return "영화관 검색이 잠시 안 돼요"

    # 반경 범위 보정 (최소 100m, 최대 20,000m)
    safe_radius = max(100, min(int(radius), _MAX_RADIUS_M))

    try:
        # 공통 요청 헤더 (카카오 REST API 인증)
        headers = {"Authorization": f"KakaoAK {_KAKAO_API_KEY}"}

        async with httpx.AsyncClient(timeout=_KAKAO_TIMEOUT_SEC) as client:
            # 3개 체인 키워드 병렬 검색 — asyncio.gather로 동시 요청
            tasks = [
                _search_keyword(client, headers, keyword, latitude, longitude, safe_radius)
                for keyword in _THEATER_KEYWORDS
            ]
            results_per_keyword = await asyncio.gather(*tasks, return_exceptions=True)

        # 체인별 결과 병합 + 중복 제거 (theater_id 기준)
        seen_ids: set[str] = set()
        merged: list[dict] = []

        for result in results_per_keyword:
            # gather 중 예외 발생 시 해당 체인 결과 건너뜀
            if isinstance(result, Exception):
                logger.warning(
                    "theater_search_keyword_error",
                    error=str(result),
                )
                continue
            for theater in result:
                tid = theater.get("theater_id", "")
                if tid and tid not in seen_ids:
                    seen_ids.add(tid)
                    merged.append(theater)

        # 거리 오름차순 정렬
        merged.sort(key=lambda x: x.get("distance_m", 999_999))

        logger.info(
            "theater_search_tool_done",
            latitude=latitude,
            longitude=longitude,
            radius_m=safe_radius,
            result_count=len(merged),
            top_names=[t.get("name", "") for t in merged[:3]],
        )
        return merged

    except httpx.TimeoutException:
        logger.error(
            "theater_search_tool_timeout",
            latitude=latitude,
            longitude=longitude,
            timeout_sec=_KAKAO_TIMEOUT_SEC,
        )
        return "영화관 검색이 잠시 안 돼요"

    except Exception as e:
        # 예상치 못한 에러 (에러 전파 금지)
        logger.error(
            "theater_search_tool_error",
            error=str(e),
            error_type=type(e).__name__,
            latitude=latitude,
            longitude=longitude,
        )
        return "영화관 검색이 잠시 안 돼요"


async def _search_keyword(
    client: httpx.AsyncClient,
    headers: dict,
    keyword: str,
    latitude: float,
    longitude: float,
    radius: int,
) -> list[dict]:
    """
    카카오 로컬 키워드 검색 API 단일 호출 헬퍼.

    하나의 영화관 체인 키워드(CGV 등)로 검색 후 파싱 결과를 반환한다.
    에러 시 빈 리스트를 반환하여 gather 전체를 깨뜨리지 않는다.

    Args:
        client: 재사용 httpx.AsyncClient 인스턴스
        headers: Authorization 헤더
        keyword: 검색 키워드 (예: "CGV")
        latitude: 위도 (y 파라미터)
        longitude: 경도 (x 파라미터)
        radius: 검색 반경 (미터)

    Returns:
        영화관 dict 목록 (파싱 완료)
    """
    try:
        resp = await client.get(
            _KAKAO_API_URL,
            headers=headers,
            params={
                "query": keyword,
                "x": str(longitude),       # 경도 (카카오: x=경도)
                "y": str(latitude),        # 위도 (카카오: y=위도)
                "radius": str(radius),
                "sort": "distance",        # 거리 오름차순 정렬
                "size": 5,                 # 체인당 최대 5개
                "category_group_code": "CT1",  # CT1: 문화시설 (영화관 포함)
            },
        )
        resp.raise_for_status()
        data = resp.json()

        theaters: list[dict] = []
        for doc in data.get("documents", []):
            # 카카오 응답 필드: id, place_name, road_address_name, phone, x(경도), y(위도), distance
            theaters.append({
                "theater_id": doc.get("id", ""),
                "name": doc.get("place_name", ""),
                "address": doc.get("road_address_name") or doc.get("address_name", ""),
                "phone": doc.get("phone", ""),
                "latitude": float(doc.get("y", 0)),
                "longitude": float(doc.get("x", 0)),
                # distance: 카카오 API가 문자열로 반환 ("1234")
                "distance_m": int(doc.get("distance", 0) or 0),
                "place_url": doc.get("place_url", ""),
            })
        return theaters

    except Exception as e:
        logger.warning(
            "theater_search_keyword_request_error",
            keyword=keyword,
            error=str(e),
        )
        return []
