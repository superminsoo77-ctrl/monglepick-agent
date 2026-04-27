"""
영화 정보 외부 검색 보강 모듈.

내부 DB(Qdrant/ES/TMDB)에 줄거리(overview)나 연출 정보가 부족한 영화에 대해
DuckDuckGo 웹 검색을 통해 Wikipedia/나무위키 등에서 실제 정보를 수집하여 보강한다.

사용 시점:
- explanation_generator 노드에서 추천 이유 생성 전
- overview가 없거나 250자 미만인 영화에 대해 자동 호출

설계 원칙:
- 에러 전파 금지: 외부 검색 실패 시 기존 데이터 그대로 반환
- 타임아웃 보호: 개별 검색 5초 제한, 전체 배치 8초 제한
- 캐싱: 동일 세션 내 중복 검색 방지 (인메모리 LRU)
- 비동기: 모든 함수 async def
- 패키지: `ddgs` (구 `duckduckgo-search`). 8.x 부터 ddgs 로 rename + primp(Rust) 백엔드.
  과거 httpx 기반 8.1.1 은 rate limit 시 thread hang → 워커 사망 패턴이 있었으므로
  primp 기반 9.x 로 고정 (2026-04-27).
- Kill switch: 환경변수 `MOVIE_ENRICHMENT_ENABLED=false` 면 외부 검색 일체 skip.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from functools import lru_cache
from typing import Any

import structlog

logger = structlog.get_logger()

# ── 보강이 필요한 최소 overview 길이 (자 수) ──
# 이 값 미만이면 외부 검색을 시도한다.
_MIN_OVERVIEW_LENGTH = 250

# ── DuckDuckGo 검색 타임아웃 (초) ──
_SEARCH_TIMEOUT_SEC = 5.0

# ── 배치 전체 타임아웃 (초) ──
# 2026-04-27: 15초 → 8초 단축. 외부 검색은 best-effort 이며, 추천 이유 생성에 더 오래
# 잡히면 SSE 사용자 체감 응답이 늦어진다. 보강 실패 시 원본 overview 로 fallback.
_BATCH_TIMEOUT_SEC = 8.0

# ── 검색 결과 최대 수 ──
_MAX_SEARCH_RESULTS = 5

# ── Kill switch: 외부 검색 보강 활성화 여부 ──
# 운영에서 ddgs/네트워크 이슈로 추천이 멈추는 사고가 재발하면 즉시 false 로 끌 수 있도록
# 환경변수로 제어. 기본 true (보강 수행). docker-compose 의 .env 에서 토글 가능.
_ENRICHMENT_ENABLED = os.getenv("MOVIE_ENRICHMENT_ENABLED", "true").lower() not in {
    "false", "0", "no", "off",
}

# ── 인메모리 캐시 (영화 제목 → 보강된 overview) ──
# 동일 세션 내 같은 영화에 대한 중복 검색 방지
_enrichment_cache: dict[str, str] = {}

# ── 신뢰 도메인 우선순위 (낮은 인덱스일수록 우선) ──
# 2026-04-23 후속 과제 (P2): 한국 영화 매칭 정확도를 높이기 위해 한국영화DB(KMDB)
# 와 영화진흥위원회(KOBIS) 를 Wikipedia/나무위키보다 상위로 승격.
# KMDB 는 한국영상자료원이 운영하는 공식 영화 정보 DB 로 배우·감독·OTT·등급 등
# 구조화된 메타데이터를 제공하고, KOBIS 는 공식 박스오피스/개봉정보를 제공한다.
# Wikipedia 는 영문 우선이라 한국 영화 문서가 누락되거나 번역 지연이 있을 수 있고,
# 나무위키는 주관적 서술/팬덤 편집이 섞일 수 있으므로 2·3 순위로 유지.
# 이 리스트는 _merge_search_results() 와 _extract_movie_candidates() 두 곳에서
# 참조된다 — 단일 진실 원본으로 유지하기 위해 모듈 상수로 추출.
_PRIORITY_DOMAINS: list[str] = [
    "kmdb.or.kr",       # 1순위: 한국영상자료원 공식 영화 DB (한국 영화 최우선)
    "kobis.or.kr",      # 2순위: 영화진흥위원회 박스오피스 (개봉·스크린·관객)
    "wikipedia.org",    # 3순위: 한/영 위키피디아 (일반 영화 광범위 커버)
    "namu.wiki",        # 4순위: 나무위키 (한국 팝컬처 상세, 주관적 서술 주의)
    "imdb.com",         # 5순위: IMDB (해외 영화 메타데이터)
]


def _needs_enrichment(overview: str | None) -> bool:
    """
    overview가 보강이 필요한지 판정한다.

    Args:
        overview: 영화 줄거리 문자열 (None 가능)

    Returns:
        True이면 외부 검색 보강 필요
    """
    if not overview:
        return True
    return len(overview.strip()) < _MIN_OVERVIEW_LENGTH


def _extract_useful_text(body: str) -> str:
    """
    검색 결과 본문에서 유용한 텍스트를 추출/정제한다.

    HTML 태그, 불필요한 공백, 광고 문구 등을 제거하고
    영화 줄거리/설명에 해당하는 텍스트만 추출한다.

    Args:
        body: 검색 결과 원본 텍스트

    Returns:
        정제된 텍스트 문자열
    """
    if not body:
        return ""

    # HTML 태그 제거
    text = re.sub(r"<[^>]+>", "", body)
    # 연속 공백/줄바꿈 정리
    text = re.sub(r"\s+", " ", text).strip()
    # 너무 짧은 결과 무시
    if len(text) < 20:
        return ""
    return text


def _build_search_query(title: str, title_en: str | None = None,
                         director: str | None = None,
                         release_year: int | str | None = None) -> str:
    """
    영화 정보로부터 DuckDuckGo 검색 쿼리를 구성한다.

    한국어 제목 + 영문 제목(있으면) + "영화 줄거리" 키워드를 조합하여
    줄거리/시놉시스 관련 검색 결과가 상위에 오도록 한다.

    Args:
        title: 영화 한국어 제목
        title_en: 영화 영문 제목 (None이면 생략)
        director: 감독명 (None이면 생략)
        release_year: 개봉연도 (None이면 생략)

    Returns:
        DuckDuckGo 검색 쿼리 문자열
    """
    parts = [title]

    # 영문 제목이 한국어 제목과 다르면 추가 (동명 영화 구분용)
    if title_en and title_en != title:
        parts.append(title_en)

    # 개봉연도 추가 (동명 영화 구분 + 검색 정확도 향상)
    if release_year:
        parts.append(str(release_year))

    # "영화 줄거리" 키워드로 줄거리/시놉시스 결과 유도
    parts.append("영화 줄거리 시놉시스")

    return " ".join(parts)


async def _search_duckduckgo(query: str) -> list[dict[str, str]]:
    """
    DuckDuckGo 웹 검색을 비동기로 실행한다.

    duckduckgo-search 라이브러리는 동기 API이므로
    asyncio.to_thread()로 별도 스레드에서 실행한다.

    Args:
        query: 검색 쿼리 문자열

    Returns:
        검색 결과 리스트 [{"title": ..., "body": ..., "href": ...}, ...]
        에러 시 빈 리스트 반환
    """
    try:
        # 2026-04-27: duckduckgo_search → ddgs 마이그레이션. 9.x 는 primp(Rust) 백엔드를
        # 쓰므로 httpx 기반 hang/thread leak 문제가 사라진다. text() 시그니처는 호환.
        from ddgs import DDGS

        def _sync_search() -> list[dict[str, str]]:
            """동기 DuckDuckGo 검색 (스레드풀에서 실행)."""
            with DDGS() as ddgs_client:
                results = list(ddgs_client.text(
                    query,
                    region="kr-kr",  # 한국 지역 우선
                    max_results=_MAX_SEARCH_RESULTS,
                ))
            return results

        # 동기 검색을 별도 스레드에서 실행 + 타임아웃 보호
        results = await asyncio.wait_for(
            asyncio.to_thread(_sync_search),
            timeout=_SEARCH_TIMEOUT_SEC,
        )
        return results

    except asyncio.TimeoutError:
        logger.warning("duckduckgo_search_timeout", query=query, timeout_sec=_SEARCH_TIMEOUT_SEC)
        return []
    except ImportError:
        # ddgs 패키지 미설치 시 graceful 처리
        logger.warning("duckduckgo_search_not_installed", query=query)
        return []
    except Exception as e:
        # 네트워크 오류, rate limit 등 모든 에러를 포착 (에러 전파 금지)
        logger.warning("duckduckgo_search_error", query=query, error=str(e), error_type=type(e).__name__)
        return []


def _merge_search_results(results: list[dict[str, str]]) -> str:
    """
    여러 검색 결과를 하나의 보강 텍스트로 병합한다.

    Wikipedia/나무위키 결과를 우선하고, 중복/광고성 내용을 필터링하여
    영화 줄거리와 연출 정보를 추출한다.

    Args:
        results: DuckDuckGo 검색 결과 리스트

    Returns:
        병합된 보강 텍스트 (최대 500자)
    """
    if not results:
        return ""

    # 신뢰 도메인 우선 정렬 (모듈 상수 _PRIORITY_DOMAINS 참조)
    def _source_priority(result: dict) -> int:
        """소스 도메인에 따른 우선순위 (낮을수록 우선)."""
        href = result.get("href", "")
        for i, domain in enumerate(_PRIORITY_DOMAINS):
            if domain in href:
                return i
        return len(_PRIORITY_DOMAINS)  # 기타 소스는 최하위

    # 우선순위순 정렬
    sorted_results = sorted(results, key=_source_priority)

    # 각 결과에서 유용한 텍스트 추출
    useful_texts: list[str] = []
    seen_fragments: set[str] = set()  # 중복 문장 필터링용

    for result in sorted_results:
        body = result.get("body", "")
        text = _extract_useful_text(body)
        if not text:
            continue

        # 중복 필터: 앞 30자가 동일하면 건너뛰기
        fragment = text[:30]
        if fragment in seen_fragments:
            continue
        seen_fragments.add(fragment)

        useful_texts.append(text)

    # 텍스트 병합 (최대 500자)
    merged = " ".join(useful_texts)
    if len(merged) > 500:
        # 500자 근처 마침표에서 자르기 (문장 중간 끊김 방지)
        cut_point = merged.rfind(".", 0, 500)
        if cut_point > 200:
            merged = merged[:cut_point + 1]
        else:
            merged = merged[:500]

    return merged.strip()


async def enrich_movie_overview(movie: dict[str, Any]) -> dict[str, Any]:
    """
    단일 영화의 overview가 부족하면 DuckDuckGo 검색으로 보강한다.

    overview가 충분하면(50자 이상) 원본 그대로 반환한다.
    검색 실패 시에도 원본 그대로 반환한다 (에러 전파 금지).

    보강된 텍스트는 기존 overview에 추가(append)하며,
    '[외부 정보]' 접두사를 붙여 LLM이 내부/외부 소스를 구분할 수 있게 한다.

    Args:
        movie: 영화 정보 dict (title, overview, title_en, director, release_year 등)

    Returns:
        overview가 보강된 영화 정보 dict (원본 변경 없이 새 dict 반환)
    """
    title = movie.get("title", "")
    overview = movie.get("overview", "") or ""

    # overview가 충분하면 보강 불필요
    if not _needs_enrichment(overview):
        return movie

    if not title:
        logger.debug("enrich_skip_no_title", movie_id=movie.get("id", ""))
        return movie

    # Kill switch (배치 함수와 동일): 운영 사고 시 외부 검색 일체 차단
    if not _ENRICHMENT_ENABLED:
        logger.debug("enrich_disabled_by_env", title=title)
        return movie

    # 캐시 확인 (동일 제목의 이전 검색 결과 재사용)
    cache_key = f"{title}_{movie.get('release_year', '')}"
    if cache_key in _enrichment_cache:
        cached = _enrichment_cache[cache_key]
        if cached:
            enriched = dict(movie)
            enriched["overview"] = _combine_overview(overview, cached)
            enriched["_enriched"] = True  # 보강 여부 플래그
            logger.debug("enrich_cache_hit", title=title, cache_key=cache_key)
            return enriched
        # 캐시에 빈 문자열이 있으면 이전 검색 실패 → 재시도하지 않음
        return movie

    # DuckDuckGo 검색 실행
    search_query = _build_search_query(
        title=title,
        title_en=movie.get("title_en"),
        director=movie.get("director"),
        release_year=movie.get("release_year"),
    )

    logger.info("enrich_search_start", title=title, query=search_query)
    search_start = time.perf_counter()

    results = await _search_duckduckgo(search_query)
    enriched_text = _merge_search_results(results)

    elapsed_ms = (time.perf_counter() - search_start) * 1000

    # 캐시에 저장 (실패 시 빈 문자열 저장 → 재시도 방지)
    _enrichment_cache[cache_key] = enriched_text

    if not enriched_text:
        logger.info("enrich_no_results", title=title, elapsed_ms=round(elapsed_ms, 1))
        return movie

    # overview 보강
    enriched = dict(movie)
    enriched["overview"] = _combine_overview(overview, enriched_text)
    enriched["_enriched"] = True

    logger.info(
        "enrich_success",
        title=title,
        original_len=len(overview),
        enriched_len=len(enriched["overview"]),
        elapsed_ms=round(elapsed_ms, 1),
        sources=len(results),
    )
    return enriched


def _combine_overview(original: str, enriched: str) -> str:
    """
    기존 overview와 외부 검색 보강 텍스트를 결합한다.

    기존 overview가 있으면 그 뒤에 외부 정보를 추가하고,
    없으면 외부 정보만 사용한다.

    Args:
        original: 기존 overview (빈 문자열 가능)
        enriched: 외부 검색에서 수집한 보강 텍스트

    Returns:
        결합된 overview 문자열
    """
    original = original.strip() if original else ""
    enriched = enriched.strip() if enriched else ""

    if original and enriched:
        return f"{original}\n[외부 정보] {enriched}"
    elif enriched:
        return f"[외부 정보] {enriched}"
    return original


async def enrich_movies_batch(
    movies: list[dict[str, Any]],
    max_concurrent: int = 3,
) -> list[dict[str, Any]]:
    """
    여러 영화의 overview를 일괄 보강한다.

    overview가 부족한 영화만 선별하여 외부 검색을 실행한다.
    동시 검색 수를 제한하여 DuckDuckGo rate limit을 방지한다.

    Args:
        movies: 영화 정보 dict 리스트
        max_concurrent: 동시 검색 최대 수 (기본 3)

    Returns:
        overview가 보강된 영화 정보 dict 리스트 (순서 유지)
    """
    if not movies:
        return movies

    # Kill switch: 운영 사고 시 즉시 외부 검색 차단 (MOVIE_ENRICHMENT_ENABLED=false)
    if not _ENRICHMENT_ENABLED:
        logger.debug("enrich_batch_disabled_by_env", movie_count=len(movies))
        return movies

    # 보강이 필요한 영화 인덱스 식별
    needs_enrichment_indices = [
        i for i, m in enumerate(movies)
        if _needs_enrichment(m.get("overview", ""))
    ]

    if not needs_enrichment_indices:
        logger.debug("enrich_batch_all_sufficient", movie_count=len(movies))
        return movies

    logger.info(
        "enrich_batch_start",
        total=len(movies),
        needs_enrichment=len(needs_enrichment_indices),
    )

    batch_start = time.perf_counter()

    # 세마포어로 동시 검색 수 제한 (DuckDuckGo rate limit 보호)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _enriched_with_semaphore(movie: dict) -> dict:
        async with semaphore:
            return await enrich_movie_overview(movie)

    # 보강 필요한 영화만 비동기 병렬 검색 (배치 타임아웃 보호)
    enriched_movies = list(movies)  # 원본 복사
    try:
        tasks = [
            _enriched_with_semaphore(movies[i])
            for i in needs_enrichment_indices
        ]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=_BATCH_TIMEOUT_SEC,
        )

        # 결과 반영 (에러 발생한 영화는 원본 유지)
        for idx, result in zip(needs_enrichment_indices, results):
            if isinstance(result, Exception):
                logger.warning(
                    "enrich_batch_item_error",
                    title=movies[idx].get("title", ""),
                    error=str(result),
                )
                continue
            enriched_movies[idx] = result

    except asyncio.TimeoutError:
        logger.warning(
            "enrich_batch_timeout",
            timeout_sec=_BATCH_TIMEOUT_SEC,
            attempted=len(needs_enrichment_indices),
        )
        # 타임아웃 시 이미 완료된 결과만 사용, 나머지는 원본 유지

    batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000
    enriched_count = sum(1 for m in enriched_movies if m.get("_enriched"))

    logger.info(
        "enrich_batch_done",
        total=len(movies),
        attempted=len(needs_enrichment_indices),
        enriched=enriched_count,
        elapsed_ms=round(batch_elapsed_ms, 1),
    )

    return enriched_movies


def clear_enrichment_cache() -> None:
    """
    인메모리 보강 캐시를 초기화한다.

    테스트 또는 메모리 관리용.
    """
    _enrichment_cache.clear()
    logger.debug("enrichment_cache_cleared")


# ============================================================
# 외부 검색 전용 유틸 — "DB 에 없는 신작" fallback 용
# ============================================================
#
# 이 구간은 DB 에서 추천 후보가 0 건이지만 사용자가 "최신 영화" 같은
# 시기 시그널을 명시한 경우에 external_search_node 가 호출한다.
# 기존 enrich_movie_overview() 와 달리 "영화 자체를 외부에서 찾아오는"
# 용도이므로, 결과 텍스트에서 영화 제목/연도를 추출하고 별도의 스텁
# dict 를 생성해 응답 레이어로 넘길 수 있게 한다.
# ============================================================

# 검색 결과에서 영화 제목을 뽑아내기 위한 정규식.
# 예) "인터스텔라(2014) 영화 정보", "영화 '괴물' (2006) 줄거리" 등
#     - 『영화명』 / 「영화명」 / '영화명' / "영화명" 4종 + 앞뒤 연도 괄호
_TITLE_QUOTED_PATTERN = re.compile(
    r"[『「\"']([^』」\"']{2,40})[』」\"']\s*(?:\((\d{4})\))?"
)
# "제목 (2024)" 패턴 (따옴표 없는 일반 형태). 보조 fallback.
_TITLE_WITH_YEAR_PATTERN = re.compile(
    r"([가-힣A-Za-z0-9:\-\s]{2,40})\s*\((\d{4})\)"
)

# ── 영화 제목 유효성 검증용 블랙리스트 (2026-04-23 후속 P1) ──
# 정규식 패턴이 영화가 아닌 일반명사/제네릭 단어까지 잡는 false-positive 를 차단한다.
# 배경: DuckDuckGo 결과에는 "「영화」" / "「최신 영화 추천」" 같이 일반 명사를 따옴표로
# 감싼 표현이 흔히 포함돼 제목으로 오인식되는 경우가 있었다. 이 블랙리스트에 걸리면
# 해당 결과는 드롭하고 다음 결과로 넘어간다.
# 대소문자/공백 제거 후 키로 매칭한다.
_GENERIC_TITLE_BLACKLIST: set[str] = {
    "영화",
    "최신영화",
    "최근영화",
    "올해영화",
    "추천",
    "영화추천",
    "신작영화",
    "개봉영화",
    "ott",
    "넷플릭스",
    "상영중",
    "박스오피스",
    "영화관",
    "예고편",
    "트레일러",
    "movie",
    "film",
    "trailer",
    "latest",
    "new",
    "release",
}

# ── 연도 일치 허용 오차 (2026-04-23 후속 P1) ──
# 사용자가 release_year_gte=2026 을 지정했는데 추출된 영화 연도가 2020 이면
# "원하지 않는 영화를 보여주게 된다" → 오차 3 년 이내만 허용.
# 보수적으로 낮게 잡으면 검색 결과가 과소되므로 경험적으로 ±3 년.
_YEAR_MATCH_TOLERANCE = 3


def _is_valid_movie_title(title: str) -> bool:
    """
    정규식으로 뽑은 제목이 실제 영화 제목일 가능성이 있는지 검증한다.

    드롭 규칙:
     1. 공백 제거 후 2 자 미만
     2. `_GENERIC_TITLE_BLACKLIST` 에 포함 (대소문자/공백 제거 후)
     3. 숫자/특수문자만 있는 경우 (예: "2026", "!!!")
     4. 한글/영문자가 하나도 없는 경우

    Args:
        title: _extract_movie_candidates 가 추출한 후보 제목

    Returns:
        True 이면 유효한 영화 제목 후보
    """
    if not title:
        return False

    normalized = title.lower().replace(" ", "")
    if len(normalized) < 2:
        return False

    if normalized in _GENERIC_TITLE_BLACKLIST:
        return False

    # 한글(가-힣) 또는 영문자(A-Za-z) 중 하나라도 있어야 영화 제목으로 간주
    if not re.search(r"[가-힣A-Za-z]", title):
        return False

    return True


def _is_year_compatible(
    extracted_year: int,
    release_year_gte: int | None,
) -> bool:
    """
    추출된 영화의 연도가 사용자가 요청한 시기와 맞는지 판정한다.

    release_year_gte 가 None 이면 판정 스킵(허용). extracted_year 가 0 이면
    정규식이 연도를 잡지 못한 경우인데, 이 경우에도 드롭하지 않는다(제목은 유효할 수 있음).
    둘 다 양의 정수일 때만 비교하며, `extracted_year >= release_year_gte - tolerance` 면 허용.

    예) release_year_gte=2026, _YEAR_MATCH_TOLERANCE=3
        extracted_year 2023 → 통과 (2026-3=2023)
        extracted_year 2020 → 드롭
        extracted_year 2028 → 통과 (하한만 검사, 미래는 허용)
        extracted_year 0    → 통과 (연도 미상)

    Args:
        extracted_year: _extract_movie_candidates 가 뽑은 영화 개봉연도
        release_year_gte: 사용자 요청 하한

    Returns:
        True 이면 release_year_gte 와 호환
    """
    if release_year_gte is None or release_year_gte <= 0:
        return True
    if extracted_year <= 0:
        return True  # 연도 미상 → 제목 기반으로만 판단
    return extracted_year >= release_year_gte - _YEAR_MATCH_TOLERANCE


def _build_external_query(
    user_intent: str,
    current_input: str,
    release_year_gte: int | None = None,
) -> str:
    """
    .. deprecated::
        2026-04-27 이후 external_search_node 는 tools/external_movie_search.py 의
        search_external_movies_v2 를 사용한다. 이 함수는 web_search_movie.py 등
        다른 호출부가 있을 수 있으므로 삭제하지 않고 유지한다.

    external_search_node 가 사용하는 DuckDuckGo 쿼리를 구성한다.

    사용자 의도(user_intent) 가 있으면 그것을 중심으로,
    없으면 원문 입력을 fallback 으로 사용한다.
    release_year_gte 가 지정되면 연도 키워드를 명시적으로 추가해
    "최신 영화 2026 개봉 추천" 같은 형태로 질의 정확도를 올린다.

    Args:
        user_intent: preference_refiner 가 요약한 사용자 의도 문자열
        current_input: 사용자가 실제로 입력한 원문
        release_year_gte: dynamic_filters 에서 추출한 개봉연도 하한

    Returns:
        DuckDuckGo 검색 쿼리 문자열
    """
    base = (user_intent or current_input or "최신 영화 추천").strip()
    parts = [base] if base else []

    # 시기 신호가 있으면 연도를 쿼리에 직접 포함해 검색 정확도를 높인다.
    if release_year_gte and release_year_gte > 0:
        parts.append(f"{release_year_gte}년")

    # DuckDuckGo 가 Wikipedia/나무위키 개봉작 목록 페이지를 우선 노출하도록 유도
    parts.append("개봉 영화 추천 목록")

    return " ".join(parts)


def _extract_movie_candidates(
    results: list[dict[str, str]],
    max_movies: int = 5,
    release_year_gte: int | None = None,
) -> list[dict[str, Any]]:
    """
    DuckDuckGo 검색 결과 리스트에서 영화 제목·연도·overview 를 추출한다.

    RankedMovie 로 변환 가능한 "스텁 dict" 리스트를 반환한다.
    신뢰 도메인(KMDB/KOBIS/Wikipedia/namu.wiki) 결과에서 제목을 우선 뽑고,
    동일 제목은 중복 제거한다.

    실제 내부 DB 의 movie_id 를 부여할 수 없으므로 id 는 `external_{i}`
    형태의 합성 ID 를 사용한다. Client/Backend 저장 경로에서 이 접두사로
    "외부 소스" 여부를 구분할 수 있다.

    2026-04-23 후속 P1: 오보강 방지를 위해 제목 유효성(_is_valid_movie_title)과
    연도 호환성(_is_year_compatible) 검증을 추가. 일반명사/숫자만 있는 "영화" 같은
    제네릭 제목과 사용자 요청 시기와 동떨어진 연도의 영화는 드롭된다.

    Args:
        results: _search_duckduckgo() 반환값
        max_movies: 최대 추출 영화 수
        release_year_gte: 사용자 요청 개봉연도 하한 (None 이면 연도 검증 스킵)

    Returns:
        영화 스텁 dict 리스트. 예:
        [{"id": "external_0", "title": "인터스텔라", "release_year": 2014,
          "overview": "...", "source_url": "...", "_external": True}, ...]
    """
    if not results:
        return []

    # 신뢰 도메인 우선 정렬 (모듈 상수 _PRIORITY_DOMAINS 재사용 — 단일 진실 원본)
    def _priority(r: dict) -> int:
        href = r.get("href", "")
        for i, d in enumerate(_PRIORITY_DOMAINS):
            if d in href:
                return i
        return len(_PRIORITY_DOMAINS)

    sorted_results = sorted(results, key=_priority)

    extracted: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for r in sorted_results:
        title_field = r.get("title", "") or ""
        body_field = r.get("body", "") or ""
        href = r.get("href", "") or ""

        # 1차: 검색결과 title 에서 따옴표로 감싼 영화 제목 추출
        match = _TITLE_QUOTED_PATTERN.search(title_field) or \
                _TITLE_QUOTED_PATTERN.search(body_field)

        # 2차 fallback: "제목 (YYYY)" 형태
        if not match:
            match = _TITLE_WITH_YEAR_PATTERN.search(title_field) or \
                    _TITLE_WITH_YEAR_PATTERN.search(body_field)

        if match:
            title = match.group(1).strip()
            year_str = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            release_year = int(year_str) if year_str and year_str.isdigit() else 0
        else:
            # 제목 추출 실패 → 검색결과 title 을 그대로 쓰되 너무 긴 경우 컷
            title = title_field.split("-")[0].split("|")[0].strip()[:40]
            release_year = 0

        # ── 제목 유효성 검증 (P1) ──
        # 블랙리스트/길이/문자종 검증 — 일반명사 false-positive 드롭
        if not _is_valid_movie_title(title):
            logger.debug("external_candidate_reject_invalid_title", title=title, href=href)
            continue

        # ── 연도 호환성 검증 (P1) ──
        # release_year_gte 요청이 있는데 추출 연도가 3 년 이상 차이나면 드롭
        if not _is_year_compatible(release_year, release_year_gte):
            logger.debug(
                "external_candidate_reject_year_mismatch",
                title=title,
                extracted_year=release_year,
                requested_gte=release_year_gte,
            )
            continue

        # 중복 제거 (동일 제목)
        title_key = title.lower().replace(" ", "")
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)

        # overview: body 에서 HTML 제거 + 300자 컷
        overview = _extract_useful_text(body_field)[:300]

        extracted.append({
            "id": f"external_{len(extracted)}",
            "title": title,
            "release_year": release_year,
            "overview": overview,
            "source_url": href,
            "_external": True,
        })

        if len(extracted) >= max_movies:
            break

    return extracted


async def search_external_movies(
    user_intent: str,
    current_input: str,
    release_year_gte: int | None = None,
    max_movies: int = 5,
) -> list[dict[str, Any]]:
    """
    .. deprecated::
        2026-04-27 이후 external_search_node 는 tools/external_movie_search.py 의
        search_external_movies_v2 를 사용한다. 이 함수(DuckDuckGo 단일 경로)는
        "2026 달력/공휴일" 같은 비영화 페이지를 영화로 오인하는 구조적 결함이 있다.

        web_search_movie.py 등 다른 호출부가 있을 수 있으므로 삭제하지 않고
        deprecated 상태로 유지한다. 직접 호출 금지.

    DuckDuckGo 로 "DB 밖 신작 영화" 를 검색해 스텁 영화 dict 리스트를 반환한다.

    external_search_node 에서 사용하는 최상위 유틸. 에러 시 빈 리스트 반환.

    Args:
        user_intent: 사용자 의도 요약 (preferences.user_intent)
        current_input: 원문 입력
        release_year_gte: 개봉연도 하한 (dynamic_filters 에서 추출)
        max_movies: 최대 영화 수

    Returns:
        영화 스텁 dict 리스트 (RankedMovie 로 변환 가능)
    """
    query = _build_external_query(user_intent, current_input, release_year_gte)
    logger.info(
        "external_movie_search_start",
        query=query,
        user_intent=(user_intent or "")[:80],
        release_year_gte=release_year_gte,
    )

    start = time.perf_counter()
    results = await _search_duckduckgo(query)
    # 연도 하한을 _extract_movie_candidates 로 전파하여 연도 미스매치 자동 드롭 (P1)
    candidates = _extract_movie_candidates(
        results,
        max_movies=max_movies,
        release_year_gte=release_year_gte,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "external_movie_search_done",
        query=query,
        raw_results=len(results),
        extracted=len(candidates),
        elapsed_ms=round(elapsed_ms, 1),
    )
    return candidates


# ============================================================
# 공개 유틸: Wikipedia/나무위키 snippet 단건 조회
# ============================================================

async def fetch_wikipedia_summary(title: str, year: int | None) -> str | None:
    """
    영화 제목에 대해 Wikipedia/나무위키 한정 DuckDuckGo 검색을 수행하고
    첫 번째 결과의 snippet 텍스트를 반환한다.

    tools/external_movie_search.py 의 _enrich_with_wikipedia 가 이 함수를 호출한다.
    DDG 검색 실패/타임아웃/결과 없음이면 None 을 반환 (에러 전파 금지).

    검색 전략:
    - 쿼리: "site:ko.wikipedia.org OR site:namu.wiki {title} {year} 영화 줄거리"
    - 한국어 지역 설정 (region="kr-kr")
    - 결과 첫 번째 body 텍스트를 정제해 최대 400자 반환

    Args:
        title: 영화 제목 (한국어 또는 영어)
        year: 개봉연도 (None 이면 생략)

    Returns:
        정제된 snippet 텍스트 (최대 400자), 실패 시 None
    """
    if not title:
        return None

    # Wikipedia/나무위키 한정 쿼리 구성
    year_part = f" {year}" if year else ""
    query = f"site:ko.wikipedia.org OR site:namu.wiki {title}{year_part} 영화 줄거리"

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_sync_ddg_search, query),
            timeout=_SEARCH_TIMEOUT_SEC,
        )

        if not results:
            return None

        # 첫 번째 결과의 body 텍스트 정제
        first_body = results[0].get("body", "") or ""
        snippet = _extract_useful_text(first_body)[:400].strip()

        return snippet if snippet else None

    except asyncio.TimeoutError:
        logger.debug(
            "fetch_wikipedia_summary_timeout",
            title=title,
            year=year,
            timeout_sec=_SEARCH_TIMEOUT_SEC,
        )
        return None
    except Exception as e:
        logger.debug(
            "fetch_wikipedia_summary_error",
            title=title,
            year=year,
            error=str(e),
        )
        return None


def _sync_ddg_search(query: str) -> list[dict[str, str]]:
    """
    DuckDuckGo 동기 검색 (to_thread 에서 실행용).

    fetch_wikipedia_summary 와 기존 _search_duckduckgo 가 공통으로 사용하는
    동기 검색 내부 구현체.

    Args:
        query: 검색 쿼리

    Returns:
        검색 결과 리스트, 실패 시 빈 리스트
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(
                query,
                region="kr-kr",
                max_results=_MAX_SEARCH_RESULTS,
            ))
    except Exception:
        return []
