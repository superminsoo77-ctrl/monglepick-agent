"""
외부 영화 검색 v2 — 메타데이터 우선 Fan-out 모듈.

기존 DuckDuckGo 단일 경로의 결함 (비영화 페이지를 영화로 오인) 을 근본 해결하기 위해
TMDB Discover / KOBIS 박스오피스 / KMDb 한국영화 DB 세 개의 신뢰 가능한 API 를
fan-out 으로 조회하고, overview 가 빈 경우에만 Wikipedia/나무위키 DDG 검색으로 보강한다.

호출 흐름:
  1. is_korean_focus 에 따라 소스 우선순위 결정
     - 한국 영화: KOBIS 박스오피스 + KMDb → TMDB 보조
     - 글로벌:    TMDB Discover 우선 → KOBIS 보조
  2. 모든 소스를 asyncio.gather 로 병렬 조회 (부분 실패 허용)
  3. 제목+연도 기준 중복 제거 후 popularity 정렬, max_movies×2 풀 구성
  4. 상위 max_movies 편에 대해서만 Wikipedia 보강 (overview 가 빈 경우)
  5. 표준 dict 스키마로 정규화 반환 → 호출부(external_search_node)가 RankedMovie 로 변환

타임아웃:
  - per-source: 5초
  - 전체 함수: 25초

API 키 미설정 시 해당 소스를 조용히 스킵 (graceful degrade).
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any

import httpx
import structlog

from monglepick.config import settings

logger = structlog.get_logger()

# ── 타임아웃 상수 ──
_PER_SOURCE_TIMEOUT_SEC = 5.0   # 소스별 최대 대기 시간
_TOTAL_TIMEOUT_SEC = 25.0       # 전체 함수 최대 실행 시간

# ── TMDB 이미지 기본 URL ──
_TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# ── 한국어 장르명 → TMDB genre_id 매핑 ──
# TMDB /genre/movie/list 의 19개 공식 장르 + 한국어 동의어/관용 표현 포함.
# list 형식: 첫 원소는 TMDB ID, 나머지는 대소문자/공백 무관하게 매핑될 한국어 키워드.
# _resolve_genre_ids() 함수에서 소문자 변환 + 공백 제거 후 매핑한다.
_KOREAN_GENRE_MAP: list[tuple[int, list[str]]] = [
    # TMDB ID  한국어 키워드 목록 (소문자 공백제거 매칭)
    (28,    ["액션", "action"]),
    (12,    ["어드벤처", "모험", "adventure"]),
    (16,    ["애니메이션", "animation", "anime", "아니메"]),
    (35,    ["코미디", "comedy", "코메디"]),
    (80,    ["범죄", "crime"]),
    (99,    ["다큐멘터리", "documentary", "다큐"]),
    (18,    ["드라마", "drama"]),
    (10751, ["가족", "family"]),
    (14,    ["판타지", "fantasy"]),
    (36,    ["역사", "history", "시대극", "사극"]),
    (27,    ["공포", "호러", "horror"]),
    (10402, ["음악", "music", "뮤지컬", "musical"]),
    (9648,  ["미스터리", "mystery"]),
    (10749, ["로맨스", "romance", "멜로", "로맨틱"]),
    (878,   ["sf", "사이언스픽션", "공상과학", "science fiction", "사이언스 픽션", "공상과학영화"]),
    (10770, ["tv영화", "tvmovie"]),
    (53,    ["스릴러", "thriller"]),
    (10752, ["전쟁", "war"]),
    (37,    ["서부", "western"]),
]

# 역색인: 정규화된 키워드 → TMDB genre_id (빠른 조회용)
_KEYWORD_TO_GENRE_ID: dict[str, int] = {}
for _genre_id, _keywords in _KOREAN_GENRE_MAP:
    for _kw in _keywords:
        # 소문자 변환 + 공백/특수문자 제거로 정규화
        _normalized = re.sub(r"[\s/·,]+", "", _kw.lower())
        _KEYWORD_TO_GENRE_ID[_normalized] = _genre_id


def _resolve_genre_ids(genres: list[str] | None) -> list[int]:
    """
    한국어 장르 문자열 리스트를 TMDB genre_id 리스트로 변환한다.

    매핑 과정:
    1. 각 장르 문자열을 소문자 변환 + 공백/특수문자 제거로 정규화
    2. _KEYWORD_TO_GENRE_ID 역색인에서 조회
    3. 미매핑 항목은 조용히 드롭 (경고 로그)
    4. 중복 genre_id 제거 (set → list)

    Args:
        genres: 한국어 장르 문자열 목록 (예: ["SF", "스릴러"])

    Returns:
        TMDB genre_id 정수 목록 (예: [878, 53])
    """
    if not genres:
        return []

    result_ids: list[int] = []
    for genre_str in genres:
        # 장르 문자열을 콤마/슬래시로 분할 (복합 장르 처리: "SF/스릴러" → ["SF", "스릴러"])
        parts = re.split(r"[,/\s·]+", genre_str.strip())
        for part in parts:
            normalized = re.sub(r"[\s/·,]+", "", part.lower())
            if not normalized:
                continue
            genre_id = _KEYWORD_TO_GENRE_ID.get(normalized)
            if genre_id is not None:
                if genre_id not in result_ids:
                    result_ids.append(genre_id)
            else:
                logger.debug("genre_not_mapped", genre_input=part, normalized=normalized)

    return result_ids


def _is_korean_signal(text: str) -> bool:
    """
    텍스트에 한국 영화 포커스 시그널이 있는지 판단한다.

    "한국", "국내", "한국 영화", "한국 신작" 등의 키워드를 탐지한다.

    Args:
        text: 사용자 입력 또는 의도 텍스트

    Returns:
        True이면 한국 영화 우선 검색 필요
    """
    korean_signals = ["한국", "국내", "국산", "한국영화", "k-movie", "kmovie"]
    text_lower = text.lower().replace(" ", "")
    return any(sig in text_lower for sig in korean_signals)


def _normalize_title_for_dedup(title: str) -> str:
    """
    중복 제거를 위한 제목 정규화.

    대소문자, 공백, 특수문자, 유니코드 합성 차이를 무시하고 동일 제목으로 처리한다.

    Args:
        title: 원본 영화 제목

    Returns:
        정규화된 제목 문자열
    """
    # NFC 정규화
    normalized = unicodedata.normalize("NFC", title)
    # 소문자 변환 + 공백/특수문자 제거
    normalized = re.sub(r"[^\w가-힣]", "", normalized.lower())
    return normalized


async def _fetch_tmdb_discover(
    year_gte: int | None,
    genre_ids: list[int],
    region: str = "KR",
    language: str = "ko-KR",
    page: int = 1,
) -> list[dict[str, Any]]:
    """
    TMDB /discover/movie 엔드포인트를 직접 호출해 영화 후보를 가져온다.

    TMDBCollector 에 discover 메서드가 없으므로 httpx 로 직접 호출한다.
    API 키 미설정 시 빈 리스트 반환 (graceful degrade).

    Args:
        year_gte: 개봉연도 하한 (primary_release_year 또는 release_date.gte)
        genre_ids: TMDB genre_id 목록 (AND 조건). 비면 장르 필터 없음.
        region: 박스오피스 집계 지역 (기본 KR)
        language: 응답 언어 (기본 ko-KR)
        page: 페이지 번호 (기본 1)

    Returns:
        정규화된 영화 dict 목록:
        {source, external_id, title, original_title, release_year, overview, poster_url, extra}
    """
    api_key = settings.TMDB_API_KEY
    if not api_key:
        logger.debug("tmdb_api_key_not_set_skip")
        return []

    base_url = settings.TMDB_BASE_URL

    # 쿼리 파라미터 구성
    params: dict[str, Any] = {
        "api_key": api_key,
        "language": language,
        "region": region,
        "sort_by": "popularity.desc",
        "page": page,
        "include_adult": "false",
    }

    # 개봉연도 필터: primary_release_year (단일 연도) 또는 release_date.gte (범위)
    if year_gte:
        # 현재 연도와 같거나 미래면 release_date.gte 사용 (미개봉 포함)
        current_year = datetime.now().year
        if year_gte >= current_year:
            params["primary_release_date.gte"] = f"{year_gte}-01-01"
        else:
            params["primary_release_date.gte"] = f"{year_gte}-01-01"
            params["primary_release_date.lte"] = f"{year_gte + 2}-12-31"

    # 장르 필터: OR 조건 (TMDB는 | 로 구분하면 OR, 콤마로 구분하면 AND)
    if genre_ids:
        # OR 조건으로 검색 (더 넓은 후보 풀)
        params["with_genres"] = "|".join(str(gid) for gid in genre_ids)

    try:
        async with httpx.AsyncClient(timeout=_PER_SOURCE_TIMEOUT_SEC) as client:
            resp = await client.get(f"{base_url}/discover/movie", params=params)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        movies: list[dict[str, Any]] = []

        for item in results:
            # 개봉년도 파싱 (release_date: "2026-03-15" 형식)
            release_date_str = item.get("release_date", "")
            release_year = 0
            if release_date_str and len(release_date_str) >= 4:
                try:
                    release_year = int(release_date_str[:4])
                except ValueError:
                    pass

            # TMDB 포스터 URL 완성
            poster_path = item.get("poster_path") or ""
            poster_url = f"{_TMDB_IMAGE_BASE}{poster_path}" if poster_path else None

            movies.append({
                "source": "tmdb",
                "external_id": f"external_tmdb_{item.get('id', '')}",
                "title": item.get("title", "") or item.get("original_title", ""),
                "original_title": item.get("original_title", ""),
                "release_year": release_year,
                "overview": item.get("overview", "") or "",
                "poster_url": poster_url,
                "extra": {
                    "genres_kr": [],          # 추후 genre_id → 한국어 역변환 가능
                    "popularity": item.get("popularity", 0.0),
                    "vote_average": item.get("vote_average", 0.0),
                    "vote_count": item.get("vote_count", 0),
                    "tmdb_id": item.get("id"),
                },
            })

        logger.info(
            "tmdb_discover_fetched",
            year_gte=year_gte,
            genre_ids=genre_ids,
            count=len(movies),
        )
        return movies

    except asyncio.TimeoutError:
        logger.warning("tmdb_discover_timeout", year_gte=year_gte, genre_ids=genre_ids)
        return []
    except Exception as e:
        logger.warning("tmdb_discover_error", error=str(e), error_type=type(e).__name__)
        return []


async def _fetch_kobis_recent_boxoffice(days: int = 14) -> list[dict[str, Any]]:
    """
    최근 N일간 KOBIS 일별 박스오피스를 합집합 후 누적관객수 정렬로 반환한다.

    KOBISCollector.collect_daily_boxoffice 를 날짜별로 반복 호출하고
    같은 movie_cd 의 중복을 제거 (누적관객수 최대값 유지) 한다.
    API 키 미설정 시 빈 리스트 반환 (graceful degrade).

    Args:
        days: 조회할 일수 (기본 14일)

    Returns:
        정규화된 영화 dict 목록:
        {source, external_id, title, original_title, release_year, overview, poster_url, extra}
    """
    api_key = settings.KOBIS_API_KEY
    if not api_key:
        logger.debug("kobis_api_key_not_set_skip")
        return []

    # KOBISCollector 는 async context manager 이므로 직접 인스턴스화
    from monglepick.data_pipeline.kobis_collector import KOBISCollector

    # 날짜별 수집 (당일은 집계 중이므로 어제부터)
    end_date = datetime.now() - timedelta(days=1)

    # movie_cd → 최대 누적관객 데이터 보존 딕셔너리
    best_by_movie_cd: dict[str, Any] = {}

    try:
        async with KOBISCollector() as collector:
            for i in range(days):
                target_date = (end_date - timedelta(days=i)).strftime("%Y%m%d")
                try:
                    box_offices = await asyncio.wait_for(
                        collector.collect_daily_boxoffice(target_date),
                        timeout=_PER_SOURCE_TIMEOUT_SEC,
                    )
                    for bo in box_offices:
                        existing = best_by_movie_cd.get(bo.movie_cd)
                        # 같은 영화가 여러 날짜에 등장 시 누적관객 최대값 유지
                        if existing is None or bo.audi_acc > existing.audi_acc:
                            best_by_movie_cd[bo.movie_cd] = bo
                except asyncio.TimeoutError:
                    logger.warning("kobis_single_date_timeout", target_date=target_date)
                    break  # 타임아웃 시 남은 날짜 처리 중단 (부분 결과 반환)
                except Exception as date_err:
                    logger.warning(
                        "kobis_single_date_error",
                        target_date=target_date,
                        error=str(date_err),
                    )
                    break

    except Exception as e:
        logger.warning("kobis_collector_error", error=str(e), error_type=type(e).__name__)
        return []

    if not best_by_movie_cd:
        return []

    # 누적관객수 내림차순 정렬
    sorted_bos = sorted(
        best_by_movie_cd.values(),
        key=lambda bo: bo.audi_acc,
        reverse=True,
    )

    # 정규화된 dict 변환
    movies: list[dict[str, Any]] = []
    for bo in sorted_bos:
        # 개봉년도 파싱 (open_dt: "20260315" 또는 "" 형식)
        release_year = 0
        if bo.open_dt and len(bo.open_dt) >= 4:
            try:
                release_year = int(bo.open_dt[:4])
            except ValueError:
                pass

        movies.append({
            "source": "kobis",
            "external_id": f"external_kobis_{bo.movie_cd}",
            "title": bo.movie_nm,
            "original_title": bo.movie_nm,   # KOBIS 는 원제 별도 제공 안 함
            "release_year": release_year,
            "overview": "",    # KOBIS 박스오피스에는 줄거리 없음 → Wikipedia 보강 대상
            "poster_url": None,
            "extra": {
                "genres_kr": [],
                "popularity": float(bo.audi_acc),  # 누적관객수를 popularity 대역으로 활용
                "audi_acc": bo.audi_acc,
                "rank": bo.rank,
                "movie_cd": bo.movie_cd,
            },
        })

    logger.info("kobis_boxoffice_fetched", days=days, count=len(movies))
    return movies


async def _fetch_kmdb_search(
    query: str,
    year_gte: int | None,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """
    KMDb API 를 이용해 한국 영화를 검색한다.

    KMDbCollector.search() 를 연도 필터와 함께 호출하고
    개봉년도 내림차순으로 정렬해 최신 한국 영화 우선 반환한다.
    API 키 미설정 시 빈 리스트 반환 (graceful degrade).

    Args:
        query: 검색 키워드 (사용자 의도 요약)
        year_gte: 개봉연도 하한 (createDts 파라미터)
        max_results: 최대 반환 수

    Returns:
        정규화된 영화 dict 목록
    """
    api_key = settings.KMDB_API_KEY
    if not api_key:
        logger.debug("kmdb_api_key_not_set_skip")
        return []

    from monglepick.data_pipeline.kmdb_collector import KMDbCollector

    # 검색 파라미터 구성
    search_params: dict[str, str] = {}

    # 제목 검색어 — 사용자 의도에서 한국어 키워드 추출 (첫 단어만 사용해 넓게 검색)
    if query:
        # "SF 신작 추천" → "SF" 첫 단어 사용 (빈 값이면 전체 최신 목록)
        keywords = re.split(r"[\s,]+", query.strip())
        # 2글자 이상 한국어/영어 키워드만 검색어로 활용
        valid_kws = [kw for kw in keywords if len(kw) >= 2 and re.search(r"[가-힣A-Za-z]", kw)]
        if valid_kws:
            search_params["title"] = valid_kws[0]

    # 개봉연도 필터 (createDts: YYYYMMDD 형식)
    if year_gte:
        search_params["releaseDts"] = f"{year_gte}0101"

    try:
        async with KMDbCollector() as collector:
            movies_raw, _ = await asyncio.wait_for(
                collector.search(list_count=max_results, **search_params),
                timeout=_PER_SOURCE_TIMEOUT_SEC,
            )
    except asyncio.TimeoutError:
        logger.warning("kmdb_search_timeout", query=query)
        return []
    except Exception as e:
        logger.warning("kmdb_search_error", error=str(e), error_type=type(e).__name__)
        return []

    if not movies_raw:
        return []

    movies: list[dict[str, Any]] = []
    for raw in movies_raw:
        # KMDbRawMovie 필드에서 정보 추출
        release_year = 0
        if raw.release_date:
            # release_date: "2026" 또는 "2026-03-15" 형식
            try:
                release_year = int(str(raw.release_date)[:4])
            except (ValueError, TypeError):
                pass

        # KMDb 제목에서 하이라이트 마크업 제거는 KMDbCollector 가 이미 처리
        title = raw.title or ""

        # 포스터 URL (리스트 중 첫 번째)
        poster_url: str | None = None
        if raw.posters:
            poster_url = raw.posters[0]

        # overview: plot 필드 사용 (overview 보다 상세한 경우 많음)
        overview = ""
        if raw.plot:
            overview = raw.plot[:500]   # 길이 제한 (너무 길면 truncate)

        movies.append({
            "source": "kmdb",
            "external_id": f"external_kmdb_{raw.kmdb_id or title}",
            "title": title,
            "original_title": raw.title_en or title,
            "release_year": release_year,
            "overview": overview,
            "poster_url": poster_url,
            "extra": {
                "genres_kr": [raw.genre] if raw.genre else [],
                "popularity": float(release_year),   # 연도가 클수록 최신 우선
                "director": raw.director or "",
                "kmdb_id": raw.kmdb_id,
            },
        })

    # 연도 내림차순 (최신 우선)
    movies.sort(key=lambda m: m["release_year"], reverse=True)

    logger.info("kmdb_search_fetched", query=query, count=len(movies))
    return movies


def _dedupe_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    제목+연도 기반 중복 제거.

    같은 영화가 TMDB/KOBIS/KMDb 모두에 있으면 소스 우선순위에 따라 하나만 유지한다.
    우선순위: tmdb > kobis > kmdb (tmdb 가 overview 품질 가장 우수)

    Args:
        items: 소스별 영화 dict 목록 (혼합)

    Returns:
        중복 제거된 영화 dict 목록
    """
    # 소스 우선순위 (낮은 값이 우선)
    _source_priority = {"tmdb": 0, "kobis": 1, "kmdb": 2}

    # 정규화된 (제목, 연도) → 대표 항목 딕셔너리
    seen: dict[tuple[str, int], dict[str, Any]] = {}

    for item in items:
        norm_title = _normalize_title_for_dedup(item.get("title", ""))
        release_year = item.get("release_year", 0)
        key = (norm_title, release_year)

        existing = seen.get(key)
        if existing is None:
            seen[key] = item
        else:
            # 더 높은 우선순위 소스로 교체
            existing_priority = _source_priority.get(existing.get("source", ""), 99)
            current_priority = _source_priority.get(item.get("source", ""), 99)
            if current_priority < existing_priority:
                seen[key] = item

    return list(seen.values())


async def _enrich_with_wikipedia(candidate: dict[str, Any]) -> dict[str, Any]:
    """
    overview 가 비어있는 영화 1편에 대해 Wikipedia/나무위키 DDG 검색으로 보강한다.

    movie_info_enricher.fetch_wikipedia_summary 를 호출하며,
    실패해도 원본 dict 그대로 반환 (에러 전파 금지).

    Args:
        candidate: 영화 dict (overview 빈 경우 보강 대상)

    Returns:
        overview 가 채워진 영화 dict (보강 실패 시 원본 반환)
    """
    # overview 가 충분히 있으면 보강 불필요 (150자 기준)
    existing_overview = candidate.get("overview", "")
    if existing_overview and len(existing_overview.strip()) >= 150:
        return candidate

    title = candidate.get("title", "")
    release_year = candidate.get("release_year") or None

    if not title:
        return candidate

    try:
        from monglepick.utils.movie_info_enricher import fetch_wikipedia_summary

        snippet = await fetch_wikipedia_summary(title=title, year=release_year)
        if snippet:
            candidate = dict(candidate)   # 원본 수정 방지 (shallow copy)
            candidate["overview"] = snippet

    except Exception as e:
        logger.debug(
            "wikipedia_enrich_skip",
            title=title,
            release_year=release_year,
            error=str(e),
        )

    return candidate


async def search_external_movies_v2(
    user_intent: str,
    current_input: str,
    release_year_gte: int | None,
    genres: list[str] | None,
    is_korean_focus: bool,
    max_movies: int = 5,
) -> list[dict[str, Any]]:
    """
    메타데이터 우선 fan-out 외부 영화 검색 (v2).

    TMDB Discover / KOBIS 박스오피스 / KMDb 한국영화 DB 세 소스를 병렬 조회하고
    후보 풀을 구성한 뒤, overview 가 빈 경우에만 Wikipedia/나무위키 보강을 수행한다.

    기존 search_external_movies(DDG 단일 경로) 의 근본 결함:
    - DuckDuckGo 가 "2026 달력/공휴일" 같은 비영화 페이지를 영화로 오인하는 문제
    - 이를 구조적으로 해결: API 기반 신뢰 가능한 소스를 우선, DDG 는 보조 보강에만

    소스 우선순위:
    - is_korean_focus=True: KOBIS + KMDb 우선, TMDB 보조
    - is_korean_focus=False: TMDB 우선, KOBIS 보조 (KMDb 는 한국영화 전문이므로 제외)

    Args:
        user_intent: preference_refiner 가 요약한 사용자 의도
        current_input: 사용자 원문 입력
        release_year_gte: 개봉연도 하한 (dynamic_filters 에서 추출)
        genres: 한국어 장르 목록 (예: ["SF", "스릴러"])
        is_korean_focus: 한국 영화 우선 검색 여부
        max_movies: 최종 반환 최대 편수 (기본 5)

    Returns:
        정규화된 영화 dict 목록 (빈 리스트이면 호출부가 question_generator 폴백 처리)
        각 dict 스키마:
        {
            "source": "tmdb" | "kobis" | "kmdb",
            "external_id": "external_tmdb_12345" | "external_kobis_20231234" | "external_kmdb_K-12345",
            "title": "...",
            "original_title": "...",
            "release_year": 2026,
            "overview": "줄거리...",
            "poster_url": "https://..." | None,
            "extra": {"genres_kr": [...], "popularity": ..., "audi_acc": ...},
        }
    """
    # ── 1. 입력 처리 ──
    # 한국 영화 포커스 판정: 파라미터 우선, 없으면 텍스트 시그널로 보강
    if not is_korean_focus:
        is_korean_focus = _is_korean_signal(user_intent) or _is_korean_signal(current_input)

    # 장르 → TMDB genre_id 변환
    genre_ids = _resolve_genre_ids(genres)

    logger.info(
        "external_search_v2_start",
        user_intent=user_intent[:80],
        release_year_gte=release_year_gte,
        genres=genres,
        genre_ids=genre_ids,
        is_korean_focus=is_korean_focus,
        max_movies=max_movies,
    )

    # ── 2. 소스별 병렬 조회 ──
    # asyncio.gather 로 모든 소스를 동시에 호출 (return_exceptions=True: 부분 실패 허용)
    gather_start = asyncio.get_event_loop().time()

    # KOBIS 박스오피스 (최근 14일 → 너무 많으면 5일로 단축)
    kobis_days = 7 if is_korean_focus else 5

    # KMDb 검색 쿼리 구성
    # user_intent 에서 장르 키워드 + 사용자 원문 활용
    kmdb_query = user_intent or current_input

    if is_korean_focus:
        # 한국 영화 우선: KOBIS + KMDb 를 먼저 채우고 TMDB 보조
        sources_coros = [
            _fetch_kobis_recent_boxoffice(days=kobis_days),
            _fetch_kmdb_search(kmdb_query, year_gte=release_year_gte, max_results=max_movies * 2),
            _fetch_tmdb_discover(
                year_gte=release_year_gte,
                genre_ids=genre_ids,
                region="KR",
                language="ko-KR",
            ),
        ]
    else:
        # 글로벌 우선: TMDB Discover + KOBIS 보조
        sources_coros = [
            _fetch_tmdb_discover(
                year_gte=release_year_gte,
                genre_ids=genre_ids,
                region="KR",
                language="ko-KR",
            ),
            _fetch_kobis_recent_boxoffice(days=kobis_days),
        ]

    try:
        results_raw = await asyncio.wait_for(
            asyncio.gather(*sources_coros, return_exceptions=True),
            timeout=_TOTAL_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "external_search_v2_total_timeout",
            timeout_sec=_TOTAL_TIMEOUT_SEC,
            user_intent=user_intent[:80],
        )
        return []

    # 부분 실패 처리: Exception 이면 빈 리스트 대체
    all_candidates: list[dict[str, Any]] = []
    for source_result in results_raw:
        if isinstance(source_result, Exception):
            logger.warning("source_gather_exception", error=str(source_result))
        elif isinstance(source_result, list):
            all_candidates.extend(source_result)

    gather_elapsed = asyncio.get_event_loop().time() - gather_start
    logger.info(
        "external_search_v2_gathered",
        total_raw=len(all_candidates),
        elapsed_sec=round(gather_elapsed, 2),
    )

    if not all_candidates:
        logger.info("external_search_v2_no_candidates")
        return []

    # ── 3. 중복 제거 ──
    deduped = _dedupe_candidates(all_candidates)

    # ── 4. 정렬 (popularity 내림차순) + 풀 제한 ──
    # popularity 필드: TMDB=popularity 점수, KOBIS=누적관객수, KMDb=연도
    deduped.sort(
        key=lambda m: float(m.get("extra", {}).get("popularity", 0)),
        reverse=True,
    )

    # 풀 크기 제한: Wikipedia 보강 대상을 max_movies×2 까지만 유지
    pool = deduped[: max_movies * 2]

    # ── 5. Wikipedia 보강 (overview 가 비어있거나 짧은 영화만) ──
    # 병렬로 보강하되 개별 실패는 무시
    enrich_coros = [_enrich_with_wikipedia(m) for m in pool]
    enriched_results = await asyncio.gather(*enrich_coros, return_exceptions=True)

    enriched: list[dict[str, Any]] = []
    for item in enriched_results:
        if isinstance(item, Exception):
            logger.debug("wikipedia_enrich_exception", error=str(item))
        elif isinstance(item, dict):
            enriched.append(item)

    # 보강 후 popularity 재정렬 (enrichment 순서가 흐트러질 수 있음)
    enriched.sort(
        key=lambda m: float(m.get("extra", {}).get("popularity", 0)),
        reverse=True,
    )

    # 최종 편수 제한
    final = enriched[:max_movies]

    logger.info(
        "external_search_v2_done",
        deduped_count=len(deduped),
        pool_size=len(pool),
        enriched_count=len(enriched),
        final_count=len(final),
        sources=[m.get("source") for m in final],
    )

    return final
