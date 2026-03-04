"""
KOBIS (영화진흥위원회) 영화 데이터 수집기.

§11-5-1 KOBIS 수집기:
- searchMovieList: 영화 목록 검색 (페이지네이션, 100건/페이지)
- searchMovieInfo: 영화 상세정보 (배우, 스태프, 관람등급, 상영시간)
- searchDailyBoxOfficeList: 일별 박스오피스 Top-10 (관객수, 매출액)
- Rate Limit: ~3,000 calls/day, asyncio.Semaphore(2) + 0.5초 간격
- 재시도: 3회 (지수 백오프)

수집 전략:
  [1] 영화 목록 전체 수집 (페이지네이션) → 로컬 캐시
  [2] 기존 DB 영화와 매칭 (제목 + 연도)
  [3] 매칭된 영화의 상세정보 수집
  [4] 박스오피스 히스토리 수집 (최근 N일)
"""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from monglepick.config import settings
from monglepick.data_pipeline.models import KOBISRawMovie, KOBISBoxOffice

logger = structlog.get_logger()

# KOBIS API Rate Limit: ~3,000 calls/day → 동시 2개 요청 + 0.5초 간격
_semaphore = asyncio.Semaphore(2)


class KOBISAPIError(Exception):
    """KOBIS API에서 반환하는 비즈니스 에러 (faultInfo). 재시도 불필요."""
    pass


def _normalize_title(title: str) -> str:
    """
    제목 정규화 (매칭용).

    1. 유니코드 정규화 (NFC)
    2. 소문자 변환
    3. 공백/특수문자 제거
    4. 한국어 조사/접미사 등은 유지 (형태소 분석 없이 단순 정규화)
    """
    if not title:
        return ""
    # 유니코드 NFC 정규화
    title = unicodedata.normalize("NFC", title)
    # 소문자 변환
    title = title.lower()
    # 공백, 특수문자 제거 (한글, 영문, 숫자만 유지)
    title = re.sub(r"[^\w가-힣a-z0-9]", "", title)
    return title


class KOBISCollector:
    """
    KOBIS Open API 비동기 수집기.

    사용 예:
        async with KOBISCollector() as collector:
            # 영화 목록 전체 수집
            movies = await collector.collect_all_movie_list()

            # 상세정보 수집
            detail = await collector.collect_movie_detail("20190009")

            # 일별 박스오피스
            box_office = await collector.collect_daily_boxoffice("20260225")
    """

    def __init__(self) -> None:
        """KOBIS 수집기를 초기화한다. 반드시 async with 문으로 사용해야 한다."""
        self._client: httpx.AsyncClient | None = None
        self._base_url = settings.KOBIS_BASE_URL
        self._api_key = settings.KOBIS_API_KEY
        self._call_count = 0  # API 호출 카운트 (일일 ~3,000건 한도 추적용)

    async def __aenter__(self) -> KOBISCollector:
        """비동기 HTTP 클라이언트를 초기화한다."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """HTTP 클라이언트를 안전하게 종료한다."""
        if self._client:
            await self._client.aclose()

    # ── 내부 HTTP 호출 (Rate Limit 적용) ──

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=8),
        # KOBIS 비즈니스 에러(KOBISAPIError)는 재시도하지 않음 (검증 에러 등 영구적)
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def _get(self, endpoint: str, params: dict) -> dict:
        """
        KOBIS API GET 요청 (Rate Limit 준수).

        Args:
            endpoint: API 경로 (예: 'movie/searchMovieList')
            params: 쿼리 파라미터 (key는 자동 추가)

        Returns:
            dict: JSON 응답

        Raises:
            Exception: KOBIS API 에러 (faultInfo) 발생 시
        """
        async with _semaphore:  # Semaphore(2)로 동시 요청 2개 제한
            params["key"] = self._api_key
            url = f"/{endpoint}.json"

            resp = await self._client.get(url, params=params)
            resp.raise_for_status()

            self._call_count += 1
            data = resp.json()

            # Rate Limit 준수: 요청 간 최소 0.5초 대기
            await asyncio.sleep(0.5)

            # KOBIS API는 HTTP 200으로 비즈니스 에러를 반환하므로 별도 체크 필요
            if "faultInfo" in data:
                error_msg = data["faultInfo"].get("message", "Unknown error")
                error_code = data["faultInfo"].get("errorCode", "")
                raise KOBISAPIError(
                    f"KOBIS API error [{error_code}]: {error_msg}"
                )

            return data

    @property
    def call_count(self) -> int:
        """현재 세션의 API 호출 횟수."""
        return self._call_count

    # ── 영화 목록 검색 ──

    async def search_movie_list(
        self,
        page: int = 1,
        item_per_page: int = 100,
        movie_nm: str = "",
        open_start_dt: str = "",
        open_end_dt: str = "",
        prdt_start_year: str = "",
        prdt_end_year: str = "",
        rep_nation_cd: str = "",
    ) -> tuple[list[dict], int]:
        """
        KOBIS 영화 목록을 검색한다.

        Args:
            page: 페이지 번호 (1부터)
            item_per_page: 페이지당 결과 수 (최대 100)
            movie_nm: 영화명 검색어 (부분 매칭)
            open_start_dt: 개봉일 시작 (YYYYMMDD)
            open_end_dt: 개봉일 종료 (YYYYMMDD)
            prdt_start_year: 제작년도 시작 (YYYY)
            prdt_end_year: 제작년도 종료 (YYYY)
            rep_nation_cd: 대표 국가 코드 ('K'=한국, 'F'=외국)

        Returns:
            (영화목록, 전체건수) 튜플
        """
        params: dict[str, Any] = {
            "curPage": str(page),
            "itemPerPage": str(item_per_page),
        }
        if movie_nm:
            params["movieNm"] = movie_nm
        if open_start_dt:
            params["openStartDt"] = open_start_dt
        if open_end_dt:
            params["openEndDt"] = open_end_dt
        if prdt_start_year:
            params["prdtStartYear"] = prdt_start_year
        if prdt_end_year:
            params["prdtEndYear"] = prdt_end_year
        if rep_nation_cd:
            params["repNationCd"] = rep_nation_cd

        data = await self._get("movie/searchMovieList", params)

        result = data.get("movieListResult", {})
        total_count = int(result.get("totCnt", 0))
        movie_list = result.get("movieList", [])

        return movie_list, total_count

    async def collect_all_movie_list(
        self,
        rep_nation_cd: str = "",
        open_start_dt: str = "",
        open_end_dt: str = "",
        max_pages: int = 0,
    ) -> list[dict]:
        """
        KOBIS 영화 목록을 전체 페이지 수집한다.

        전체 영화 ~117K건 (한국 영화만: ~50K건).
        100건/페이지 × ~1,200페이지 = ~1,200 API 호출.

        Args:
            rep_nation_cd: 'K'=한국만, 'F'=외국만, ''=전체
            open_start_dt: 개봉일 시작 (YYYYMMDD)
            open_end_dt: 개봉일 종료 (YYYYMMDD)
            max_pages: 최대 수집 페이지 (0=제한없음)

        Returns:
            list[dict]: 전체 영화 목록 (KOBIS searchMovieList 응답 형태)
        """
        all_movies: list[dict] = []

        # 첫 페이지 호출: 전체 건수 확인
        first_page, total_count = await self.search_movie_list(
            page=1,
            rep_nation_cd=rep_nation_cd,
            open_start_dt=open_start_dt,
            open_end_dt=open_end_dt,
        )
        all_movies.extend(first_page)

        # 전체 페이지 수 계산 (올림 나눗셈)
        total_pages = (total_count + 99) // 100
        if max_pages > 0:
            total_pages = min(total_pages, max_pages)

        logger.info(
            "kobis_movie_list_started",
            total_count=total_count,
            total_pages=total_pages,
            nation_filter=rep_nation_cd or "ALL",
        )

        # 나머지 페이지 순차 수집 (API 호출 한도 준수)
        for page in range(2, total_pages + 1):
            try:
                movies, _ = await self.search_movie_list(
                    page=page,
                    rep_nation_cd=rep_nation_cd,
                    open_start_dt=open_start_dt,
                    open_end_dt=open_end_dt,
                )
                all_movies.extend(movies)

                # 500건마다 진행률 로깅
                if len(all_movies) % 500 == 0 or page == total_pages:
                    logger.info(
                        "kobis_movie_list_progress",
                        page=page,
                        total_pages=total_pages,
                        collected=len(all_movies),
                    )
            except Exception as e:
                logger.warning(
                    "kobis_movie_list_page_error",
                    page=page,
                    error=str(e),
                )
                # 에러 발생 시 다음 페이지 계속 진행
                continue

        logger.info(
            "kobis_movie_list_complete",
            total_collected=len(all_movies),
            api_calls=self._call_count,
        )

        return all_movies

    # ── 영화 상세정보 ──

    async def collect_movie_detail(self, movie_cd: str) -> KOBISRawMovie:
        """
        KOBIS 영화 상세정보를 수집한다.

        searchMovieInfo API를 호출하여 배우, 스태프, 관람등급, 상영시간 등을 가져온다.

        Args:
            movie_cd: KOBIS 영화 코드

        Returns:
            KOBISRawMovie: 파싱된 영화 상세정보
        """
        data = await self._get(
            "movie/searchMovieInfo",
            {"movieCd": movie_cd},
        )

        info = data.get("movieInfoResult", {}).get("movieInfo", {})

        # 관람등급 추출 (audits 배열에서 첫 번째)
        watch_grade = ""
        audits = info.get("audits", [])
        if audits:
            watch_grade = audits[0].get("watchGradeNm", "")

        return KOBISRawMovie(
            movie_cd=info.get("movieCd", movie_cd),
            movie_nm=info.get("movieNm", ""),
            movie_nm_en=info.get("movieNmEn", ""),
            movie_nm_og=info.get("movieNmOg", ""),
            open_dt=info.get("openDt", "").replace("-", ""),  # YYYY-MM-DD → YYYYMMDD
            prdt_year=info.get("prdtYear", ""),
            show_tm=info.get("showTm", ""),
            type_nm=info.get("typeNm", ""),
            prdt_stat_nm=info.get("prdtStatNm", ""),
            nations=info.get("nations", []),
            genres=info.get("genres", []),
            directors=info.get("directors", []),
            actors=info.get("actors", []),
            audits=audits,
            companys=info.get("companys", []),
            staffs=info.get("staffs", []),
            show_types=info.get("showTypes", []),
            watch_grade_nm=watch_grade,
            detail_fetched=True,
        )

    async def collect_movie_details_batch(
        self,
        movie_cds: list[str],
    ) -> list[KOBISRawMovie]:
        """
        여러 영화의 상세정보를 순차 수집한다.

        Rate limit 준수를 위해 순차 처리하며, 100건마다 진행률을 로깅한다.

        Args:
            movie_cds: KOBIS 영화 코드 리스트

        Returns:
            list[KOBISRawMovie]: 수집 성공한 영화 상세정보 리스트
        """
        results: list[KOBISRawMovie] = []
        failed = 0

        for i, movie_cd in enumerate(movie_cds):
            try:
                detail = await self.collect_movie_detail(movie_cd)
                results.append(detail)
            except Exception as e:
                failed += 1
                logger.warning(
                    "kobis_detail_error",
                    movie_cd=movie_cd,
                    error=str(e),
                )

            # 100건마다 진행률 로깅
            if (i + 1) % 100 == 0 or (i + 1) == len(movie_cds):
                logger.info(
                    "kobis_detail_progress",
                    completed=i + 1,
                    total=len(movie_cds),
                    success=len(results),
                    failed=failed,
                )

        return results

    # ── 박스오피스 ──

    async def collect_daily_boxoffice(self, target_date: str) -> list[KOBISBoxOffice]:
        """
        특정 날짜의 일별 박스오피스 Top-10을 수집한다.

        Args:
            target_date: 조회 대상 날짜 (YYYYMMDD 형식)

        Returns:
            list[KOBISBoxOffice]: 박스오피스 Top-10 리스트
        """
        data = await self._get(
            "boxoffice/searchDailyBoxOfficeList",
            {"targetDt": target_date},
        )

        result = data.get("boxOfficeResult", {})
        box_office_list = result.get("dailyBoxOfficeList", [])

        return [
            KOBISBoxOffice(
                movie_cd=item.get("movieCd", ""),
                movie_nm=item.get("movieNm", ""),
                rank=int(item.get("rank", 0)),
                rank_inten=int(item.get("rankInten", 0)),
                rank_old_and_new=item.get("rankOldAndNew", "OLD"),
                audi_cnt=int(item.get("audiCnt", 0)),
                audi_acc=int(item.get("audiAcc", 0)),
                sales_amt=int(item.get("salesAmt", 0)),
                sales_acc=int(item.get("salesAcc", 0)),
                scrn_cnt=int(item.get("scrnCnt", 0)),
                show_cnt=int(item.get("showCnt", 0)),
                open_dt=item.get("openDt", "").replace("-", ""),
            )
            for item in box_office_list
        ]

    async def collect_boxoffice_history(
        self,
        days: int = 365,
        end_date: str = "",
    ) -> dict[str, KOBISBoxOffice]:
        """
        최근 N일간의 박스오피스 히스토리를 수집하고 영화별 최대 누적 데이터를 반환한다.

        각 영화(movieCd)에 대해 가장 최근(=최대 누적치) 박스오피스 데이터를 보존한다.
        365일 수집 시 ~365 API 호출 (일일 한도 내).

        Args:
            days: 수집할 일수 (기본 365일)
            end_date: 종료 날짜 (YYYYMMDD, 기본 어제)

        Returns:
            dict[str, KOBISBoxOffice]: movieCd → 최대 누적 데이터 매핑
        """
        if not end_date:
            # 당일 박스오피스는 집계 전일 수 있으므로 어제를 기본값으로 사용
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

        # 영화별 최대 누적 데이터를 보존하는 딕셔너리
        # 같은 영화가 여러 날짜에 등장하면 누적 관객수가 가장 큰 것을 유지
        movie_boxoffice: dict[str, KOBISBoxOffice] = {}

        end_dt = datetime.strptime(end_date, "%Y%m%d")

        # 종료일부터 역순으로 N일간 박스오피스 데이터 수집
        for day_offset in range(days):
            target_dt = end_dt - timedelta(days=day_offset)
            target_date = target_dt.strftime("%Y%m%d")

            try:
                daily = await self.collect_daily_boxoffice(target_date)

                for item in daily:
                    mc = item.movie_cd
                    if mc not in movie_boxoffice or item.audi_acc > movie_boxoffice[mc].audi_acc:
                        movie_boxoffice[mc] = item
            except Exception as e:
                logger.warning(
                    "kobis_boxoffice_date_error",
                    date=target_date,
                    error=str(e),
                )

            # 30일마다 진행률 로깅
            if (day_offset + 1) % 30 == 0 or (day_offset + 1) == days:
                logger.info(
                    "kobis_boxoffice_progress",
                    days_processed=day_offset + 1,
                    total_days=days,
                    unique_movies=len(movie_boxoffice),
                    api_calls=self._call_count,
                )

        logger.info(
            "kobis_boxoffice_history_complete",
            total_days=days,
            unique_movies=len(movie_boxoffice),
        )

        return movie_boxoffice


# ── 매칭 유틸리티 ──


def match_kobis_to_db(
    kobis_movies: list[dict],
    db_movies: list[dict],
) -> list[tuple[dict, dict]]:
    """
    KOBIS 영화 목록과 DB 영화를 매칭한다.

    매칭 기준 (§11-5-1):
    1. 정규화된 한국어 제목 + 개봉 연도 일치 → 동일 영화
    2. 정규화된 영문 제목 + 개봉 연도 일치 → 동일 영화 (보조)

    Args:
        kobis_movies: KOBIS searchMovieList 응답 리스트
            각 항목: {'movieCd', 'movieNm', 'movieNmEn', 'openDt', 'prdtYear', ...}
        db_movies: 우리 DB 영화 리스트
            각 항목: {'id', 'title', 'title_en', 'release_year'}

    Returns:
        list[tuple[dict, dict]]: (kobis_movie, db_movie) 매칭 쌍 리스트
    """
    # DB 영화 인덱스 구축: (정규화된_제목, 연도) → db_movie
    # O(1) 검색을 위해 해시 테이블로 구성
    db_index_kr: dict[tuple[str, int], dict] = {}
    db_index_en: dict[tuple[str, int], dict] = {}

    for movie in db_movies:
        year = movie.get("release_year", 0)
        title_kr = _normalize_title(movie.get("title", ""))
        if title_kr and year:
            db_index_kr[(title_kr, year)] = movie
        title_en = _normalize_title(movie.get("title_en", ""))
        if title_en and year:
            db_index_en[(title_en, year)] = movie

    matched: list[tuple[dict, dict]] = []
    matched_db_ids: set[str] = set()  # 하나의 DB 영화가 여러 KOBIS 영화에 매칭되는 것 방지

    for kobis in kobis_movies:
        # KOBIS 개봉 연도 추출 (openDt 우선, prdtYear 보조)
        open_dt = kobis.get("openDt", "")
        prdt_year = kobis.get("prdtYear", "")
        year = 0
        if open_dt and len(open_dt) >= 4:
            try:
                year = int(open_dt[:4])
            except ValueError:
                pass
        if not year and prdt_year:
            try:
                year = int(prdt_year)
            except ValueError:
                pass
        if not year:
            continue

        # 1차 매칭: 한국어 제목 + 연도 (±1년 허용, 개봉일/제작년도 차이 보정)
        kobis_title_kr = _normalize_title(kobis.get("movieNm", ""))
        if kobis_title_kr:
            for y_offset in [0, -1, 1]:
                key = (kobis_title_kr, year + y_offset)
                db_movie = db_index_kr.get(key)
                if db_movie and db_movie["id"] not in matched_db_ids:
                    matched.append((kobis, db_movie))
                    matched_db_ids.add(db_movie["id"])
                    break
            else:
                # for-else: 한국어 제목으로 매칭 실패 시 영문 제목으로 2차 매칭
                kobis_title_en = _normalize_title(kobis.get("movieNmEn", ""))
                if kobis_title_en:
                    for y_offset in [0, -1, 1]:
                        key = (kobis_title_en, year + y_offset)
                        db_movie = db_index_en.get(key)
                        if db_movie and db_movie["id"] not in matched_db_ids:
                            matched.append((kobis, db_movie))
                            matched_db_ids.add(db_movie["id"])
                            break

    logger.info(
        "kobis_matching_complete",
        kobis_total=len(kobis_movies),
        db_total=len(db_movies),
        matched=len(matched),
    )

    return matched


# ── 캐시 유틸리티 ──


def save_kobis_cache(movies: list[dict], cache_path: str) -> None:
    """KOBIS 영화 목록을 JSON 캐시 파일로 저장한다."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)
    logger.info("kobis_cache_saved", path=str(path), count=len(movies))


def load_kobis_cache(cache_path: str) -> list[dict] | None:
    """JSON 캐시 파일에서 KOBIS 영화 목록을 로드한다. 파일 없으면 None."""
    path = Path(cache_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        movies = json.load(f)
    logger.info("kobis_cache_loaded", path=str(path), count=len(movies))
    return movies
