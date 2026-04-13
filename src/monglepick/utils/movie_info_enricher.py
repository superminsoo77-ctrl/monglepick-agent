"""
영화 정보 외부 검색 보강 모듈.

내부 DB(Qdrant/ES/TMDB)에 줄거리(overview)나 연출 정보가 부족한 영화에 대해
DuckDuckGo 웹 검색을 통해 Wikipedia/나무위키 등에서 실제 정보를 수집하여 보강한다.

사용 시점:
- explanation_generator 노드에서 추천 이유 생성 전
- overview가 없거나 250자 미만인 영화에 대해 자동 호출

설계 원칙:
- 에러 전파 금지: 외부 검색 실패 시 기존 데이터 그대로 반환
- 타임아웃 보호: 개별 검색 5초 제한, 전체 배치 15초 제한
- 캐싱: 동일 세션 내 중복 검색 방지 (인메모리 LRU)
- 비동기: 모든 함수 async def
"""

from __future__ import annotations

import asyncio
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
_BATCH_TIMEOUT_SEC = 15.0

# ── 검색 결과 최대 수 ──
_MAX_SEARCH_RESULTS = 5

# ── 인메모리 캐시 (영화 제목 → 보강된 overview) ──
# 동일 세션 내 같은 영화에 대한 중복 검색 방지
_enrichment_cache: dict[str, str] = {}


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
        from duckduckgo_search import DDGS

        def _sync_search() -> list[dict[str, str]]:
            """동기 DuckDuckGo 검색 (스레드풀에서 실행)."""
            with DDGS() as ddgs:
                results = list(ddgs.text(
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
        # duckduckgo-search 패키지 미설치 시 graceful 처리
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

    # Wikipedia/나무위키 등 신뢰할 수 있는 소스 우선 정렬
    _priority_domains = ["wikipedia.org", "namu.wiki", "kmdb.or.kr", "kobis.or.kr", "imdb.com"]

    def _source_priority(result: dict) -> int:
        """소스 도메인에 따른 우선순위 (낮을수록 우선)."""
        href = result.get("href", "")
        for i, domain in enumerate(_priority_domains):
            if domain in href:
                return i
        return len(_priority_domains)  # 기타 소스는 최하위

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
