"""
KMDb (한국영화 데이터베이스) API 수집기.

한국영상자료원이 운영하는 KMDb Open API를 통해 한국영화 데이터를 수집한다.

API 사양:
- 엔드포인트: http://api.koreafilm.or.kr/openapi-data2/wisenut/search_api/search_json2.jsp
- 인증: ServiceKey (Query Parameter)
- 필수 파라미터: collection=kmdb_new2, ServiceKey
- 최대 listCount: 500
- 일일 호출 제한: 1,000건 (개발계정)

특이사항:
- title 필드에 !HS/!HE 하이라이트 마크업 포함 → 제거 필요
- posters/stlls 필드가 파이프(|) 구분 문자열 → 리스트 변환 필요
- detail=Y 설정 시 감독/배우/줄거리/스태프 등 상세 정보 포함
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from monglepick.config import settings
from monglepick.data_pipeline.models import KMDbRawMovie

logger = structlog.get_logger()

# KMDb API 동시 요청 제한 (일일 1,000건 제한이므로 보수적으로 운영)
_semaphore = asyncio.Semaphore(5)

# !HS / !HE 하이라이트 마크업 제거용 정규식
_HIGHLIGHT_RE = re.compile(r"!HS|!HE")


def _clean_title(raw_title: str) -> str:
    """
    KMDb title 필드의 !HS/!HE 하이라이트 마크업을 제거하고 양쪽 공백을 정리한다.

    예: " !HS기생충!HE " → "기생충"
    """
    return _HIGHLIGHT_RE.sub("", raw_title).strip()


def _split_pipe_urls(pipe_str: str) -> list[str]:
    """
    KMDb posters/stlls 필드의 파이프(|) 구분 URL 문자열을 리스트로 변환한다.

    빈 문자열이나 공백만 있는 항목은 제외한다.
    예: "http://a.jpg|http://b.jpg|" → ["http://a.jpg", "http://b.jpg"]
    """
    if not pipe_str or not pipe_str.strip():
        return []
    return [url.strip() for url in pipe_str.split("|") if url.strip()]


def _parse_movie(raw: dict) -> KMDbRawMovie:
    """
    KMDb API 응답의 단일 영화 객체(Result[i])를 KMDbRawMovie로 변환한다.

    KMDb API는 중첩 구조(directors.director, actors.actor 등)를 사용하며,
    단일 결과일 때는 dict, 복수 결과일 때는 list를 반환하는 비일관적인 형태이므로
    모든 경우를 list로 통일한다.

    Args:
        raw: KMDb API 응답의 단일 영화 dict (Data[0].Result[i])

    Returns:
        KMDbRawMovie: 파싱된 영화 데이터 모델
    """
    # KMDb API는 단일 항목이면 dict, 복수면 list로 반환 → list로 통일
    directors_data = raw.get("directors", {})
    directors = directors_data.get("director", []) if isinstance(directors_data, dict) else []
    if isinstance(directors, dict):
        directors = [directors]

    actors_data = raw.get("actors", {})
    actors = actors_data.get("actor", []) if isinstance(actors_data, dict) else []
    if isinstance(actors, dict):
        actors = [actors]

    plots_data = raw.get("plots", {})
    plots = plots_data.get("plot", []) if isinstance(plots_data, dict) else []
    if isinstance(plots, dict):
        plots = [plots]

    staffs_data = raw.get("staffs", {})
    staffs = staffs_data.get("staff", []) if isinstance(staffs_data, dict) else []
    if isinstance(staffs, dict):
        staffs = [staffs]

    vods_data = raw.get("vods", {})
    vods = vods_data.get("vod", []) if isinstance(vods_data, dict) else []
    if isinstance(vods, dict):
        vods = [vods]

    codes_data = raw.get("Codes", {})
    codes = codes_data.get("Code", []) if isinstance(codes_data, dict) else []
    if isinstance(codes, dict):
        codes = [codes]

    # 포스터/스틸컷 파이프 분리
    posters = _split_pipe_urls(raw.get("posters", ""))
    stills = _split_pipe_urls(raw.get("stlls", ""))

    return KMDbRawMovie(
        doc_id=raw.get("DOCID", ""),
        movie_id=raw.get("movieId", ""),
        movie_seq=raw.get("movieSeq", ""),
        title=_clean_title(raw.get("title", "")),
        title_eng=raw.get("titleEng", "").strip(),
        title_org=raw.get("titleOrg", "").strip(),
        prod_year=raw.get("prodYear", "").strip(),
        nation=raw.get("nation", "").strip(),
        company=raw.get("company", "").strip(),
        runtime=raw.get("runtime", "").strip(),
        genre=raw.get("genre", "").strip(),
        rating=raw.get("rating", "").strip(),
        type_name=raw.get("type", "").strip(),
        use=raw.get("use", "").strip(),
        keywords=raw.get("keywords", "").strip(),
        release_date=raw.get("repRlsDate", "").strip() or raw.get("releaseDate", "").strip(),
        rep_rls_date=raw.get("repRlsDate", "").strip(),
        plots=plots,
        directors=directors,
        actors=actors,
        staffs=staffs,
        posters=posters,
        stills=stills,
        vods=vods,
        awards1=raw.get("Awards1", "").strip(),
        awards2=raw.get("Awards2", "").strip(),
        sales_acc=raw.get("salesAcc", "").strip(),
        audi_acc=raw.get("audiAcc", "").strip(),
        f_location=raw.get("fLocation", "").strip(),
        theme_song=raw.get("themeSong", "").strip(),
        soundtrack_field=raw.get("soundtrack", "").strip(),
        kmdb_url=raw.get("kmdbUrl", "").strip(),
        codes=codes,
    )


class KMDbCollector:
    """
    KMDb API 비동기 수집기.

    사용 예:
        async with KMDbCollector() as collector:
            # 전체 한국영화 수집 (연도별 페이지네이션)
            movies = await collector.collect_all_movies()

            # 특정 제목 검색
            results = await collector.search_by_title("기생충")

            # 연도 범위 수집
            movies_2020s = await collector.collect_by_year_range(2020, 2026)
    """

    def __init__(self) -> None:
        """KMDb 수집기를 초기화한다. 반드시 async with 문으로 사용해야 한다."""
        self._client: httpx.AsyncClient | None = None
        self._base_url = settings.KMDB_BASE_URL
        self._api_key = settings.KMDB_API_KEY
        self._request_count = 0  # 일일 요청 카운터 (1,000건 한도 추적)

    async def __aenter__(self) -> KMDbCollector:
        """HTTP 클라이언트 초기화."""
        self._client = httpx.AsyncClient(timeout=60.0)
        logger.info("kmdb_collector_initialized", base_url=self._base_url)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """HTTP 클라이언트 종료 및 요청 카운트 로깅."""
        if self._client:
            await self._client.aclose()
        logger.info("kmdb_collector_closed", total_requests=self._request_count)

    # ── 내부 HTTP 호출 ──

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=30))
    async def _get(self, params: dict) -> dict:
        """
        KMDb API GET 요청 (Rate Limit + 재시도 적용).

        필수 파라미터(collection, ServiceKey)는 자동 추가된다.
        최대 3회 재시도 (지수 백오프: 2초, 4초, 8초... 최대 30초).

        Returns:
            API 응답 JSON (전체)

        Raises:
            httpx.HTTPStatusError: 4xx/5xx 응답 시
            ValueError: API 에러 코드 반환 시
        """
        async with _semaphore:  # Semaphore(5)로 동시 요청 5개 제한
            # collection, ServiceKey는 모든 요청에 필수이므로 자동 추가
            full_params = {
                "collection": "kmdb_new2",
                "ServiceKey": self._api_key,
                **params,
            }

            assert self._client is not None, "KMDbCollector must be used as async context manager"
            resp = await self._client.get(self._base_url, params=full_params)
            resp.raise_for_status()

            self._request_count += 1

            # KMDb API 응답에 유효하지 않은 제어 문자(0x00~0x1F)가 포함되어
            # JSON 파싱이 실패할 수 있음. 이 경우 제어 문자를 제거한 뒤 재파싱한다.
            import json as _json
            import re as _re

            try:
                data = resp.json()
            except _json.JSONDecodeError:
                # 탭(\t=0x09), LF(\n=0x0a), CR(\r=0x0d)은 유지하고 나머지 제어 문자 제거
                cleaned = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', resp.text)
                data = _json.loads(cleaned)

            return data

    # ── 검색 메서드 ──

    async def search(
        self,
        list_count: int = 500,
        start_count: int = 0,
        **search_params: str,
    ) -> tuple[list[KMDbRawMovie], int]:
        """
        KMDb 영화 검색 (범용 검색 메서드).

        Args:
            list_count: 한 페이지 결과 수 (최대 500)
            start_count: 시작 위치 (0부터)
            **search_params: 검색 파라미터 (title, director, actor, genre,
                            createDts, createDte, releaseDts, releaseDte 등)

        Returns:
            (영화 목록, 총 검색 결과 수) 튜플
        """
        params = {
            "detail": "Y",
            "listCount": str(list_count),
            "startCount": str(start_count),
            **search_params,
        }

        data = await self._get(params)

        # KMDb 응답 구조: { "Data": [{ "TotalCount": N, "Result": [...] }] }
        data_list = data.get("Data", [])
        if not data_list:
            return [], 0

        collection_data = data_list[0]
        total_count = int(collection_data.get("TotalCount", 0))
        results = collection_data.get("Result", [])

        # TotalCount > 0이지만 Result가 None인 경우 방어
        if not results:
            return [], total_count

        movies = [_parse_movie(r) for r in results]

        logger.info(
            "kmdb_search_complete",
            returned=len(movies),
            total_count=total_count,
            start_count=start_count,
            request_num=self._request_count,
        )

        return movies, total_count

    async def search_by_title(self, title: str) -> list[KMDbRawMovie]:
        """
        제목으로 영화를 검색한다.

        KMDb API의 title 파라미터를 사용한다.
        주의: 공백이 포함된 제목은 검색 결과가 부정확할 수 있다.

        Args:
            title: 검색할 영화 제목 (한국어)

        Returns:
            검색된 영화 목록
        """
        # KMDb API는 공백 포함 제목 검색 시 결과가 부정확하므로 공백 제거
        clean_title = title.replace(" ", "")
        movies, _ = await self.search(title=clean_title)
        return movies

    async def collect_by_year_range(
        self,
        start_year: int,
        end_year: int,
        list_count: int = 500,
    ) -> list[KMDbRawMovie]:
        """
        제작연도 범위로 영화를 수집한다.

        KMDb의 createDts/createDte 파라미터를 사용하여 연도별로 페이지네이션한다.
        한 연도에 500건을 초과하는 경우 자동으로 다음 페이지를 요청한다.

        Args:
            start_year: 시작 연도 (예: 1960)
            end_year: 종료 연도 (예: 2026)
            list_count: 페이지당 결과 수 (최대 500)

        Returns:
            수집된 전체 영화 목록
        """
        all_movies: list[KMDbRawMovie] = []

        for year in range(start_year, end_year + 1):
            start_count = 0

            while True:
                movies, total_count = await self.search(
                    list_count=list_count,
                    start_count=start_count,
                    createDts=str(year),
                    createDte=str(year),
                )

                if not movies:
                    break

                all_movies.extend(movies)
                start_count += len(movies)

                logger.info(
                    "kmdb_year_progress",
                    year=year,
                    collected=start_count,
                    total_in_year=total_count,
                )

                # 모든 결과를 가져왔으면 다음 연도로
                if start_count >= total_count:
                    break

                # 일일 한도 1,000건 초과 방지 (안전 임계값 950건에서 중단)
                if self._request_count >= 950:
                    logger.warning(
                        "kmdb_daily_limit_approaching",
                        request_count=self._request_count,
                        stopping_at_year=year,
                    )
                    return all_movies

        logger.info(
            "kmdb_year_range_complete",
            start_year=start_year,
            end_year=end_year,
            total_collected=len(all_movies),
            total_requests=self._request_count,
        )

        return all_movies

    async def collect_all_movies(
        self,
        start_year: int = 1960,
        end_year: int | None = None,
        list_count: int = 500,
    ) -> list[KMDbRawMovie]:
        """
        KMDb의 전체 한국영화를 수집한다.

        연도별로 페이지네이션하여 수집하며, 일일 요청 한도(1,000건)를 준수한다.
        listCount=500으로 설정하면 약 52회 요청으로 26,000+건 수집 가능.

        Args:
            start_year: 시작 연도 (기본 1960, 이전 영화는 소수)
            end_year: 종료 연도 (기본 현재 연도)
            list_count: 페이지당 결과 수 (기본 500, 최대값)

        Returns:
            수집된 전체 영화 목록
        """
        from datetime import datetime

        if end_year is None:
            end_year = datetime.now().year

        logger.info(
            "kmdb_collect_all_start",
            start_year=start_year,
            end_year=end_year,
        )

        return await self.collect_by_year_range(start_year, end_year, list_count)

    async def collect_paginated(
        self,
        list_count: int = 500,
        max_pages: int | None = None,
        **search_params: str,
    ) -> list[KMDbRawMovie]:
        """
        검색 조건에 대해 전체 결과를 페이지네이션하여 수집한다.

        Args:
            list_count: 페이지당 결과 수 (최대 500)
            max_pages: 최대 페이지 수 (None이면 전체)
            **search_params: 검색 파라미터

        Returns:
            수집된 전체 영화 목록
        """
        all_movies: list[KMDbRawMovie] = []
        start_count = 0
        page = 0

        while True:
            movies, total_count = await self.search(
                list_count=list_count,
                start_count=start_count,
                **search_params,
            )

            if not movies:
                break

            all_movies.extend(movies)
            start_count += len(movies)
            page += 1

            # 모든 결과를 가져왔으면 종료
            if start_count >= total_count:
                break

            # 최대 페이지 제한
            if max_pages and page >= max_pages:
                logger.info("kmdb_max_pages_reached", max_pages=max_pages, collected=len(all_movies))
                break

        return all_movies
