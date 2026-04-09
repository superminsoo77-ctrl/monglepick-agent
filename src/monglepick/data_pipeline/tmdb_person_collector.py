"""
TMDB Person API 수집기 (Phase ML §9.5 Phase 1).

Neo4j Person 노드(572K+)에 대해 TMDB Person API 를 호출하여
biography, 필모그래피, 외부 ID, 다중 이미지를 수집한다.

참조 설계: docs/데이터_적재_프로세스_전체분석_및_개선계획.md §9.5 Phase 1

핵심 정책:
    - 기존 `tmdb_collector.py` 는 수정하지 않음 (Task #5 영향 방지를 위해 독립 모듈)
    - 동일 API 키 사용 (.env TMDB_API_KEY)
    - Rate limit: 40 req/sec (TMDB 기본)
    - max_workers 비동기 워커 (기본 10, run_tmdb_full_collection 보다 보수적)
    - append_to_response 로 5개 sub-resource 1회 호출
    - 타임아웃/네트워크 에러 재시도 3회 + 지수 백오프

수집 데이터 (TMDB Person API + sub-resources):
    - 기본: id, name, also_known_as, biography, birthday, deathday,
            place_of_birth, gender, popularity, profile_path,
            known_for_department, homepage, imdb_id
    - movie_credits: cast (배우 출연작) + crew (감독/각본/제작 등)
    - external_ids: imdb_id, facebook_id, instagram_id, twitter_id, tiktok_id, youtube_id
    - images: profiles[]
    - translations: 다국어 biography (한국어/일본어 등)
    - tagged_images: 영화별 인물 태그 이미지 (선택, popularity 보조 지표)

사용처:
    `scripts/run_tmdb_persons_collect.py` 에서 사용
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import structlog

from monglepick.config import settings

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# TMDB Person API 상수
# ══════════════════════════════════════════════════════════════

TMDB_BASE_URL = "https://api.themoviedb.org/3"

# 인물 상세 + 5개 sub-resource (1회 호출)
PERSON_APPEND_RESOURCES = (
    "movie_credits,"     # 출연/감독/각본 영화 전체
    "external_ids,"      # IMDb / FB / IG / Twitter / TikTok / YouTube
    "images,"            # profile 이미지 다중
    "translations"       # 다국어 biography (한국어 포함)
)

# Rate limit (TMDB 공식 한도: 50 req/sec, 안전 마진 80%)
DEFAULT_RATE_LIMIT_RPS = 35
DEFAULT_MAX_WORKERS = 10
DEFAULT_REQUEST_TIMEOUT = 15.0
DEFAULT_MAX_RETRIES = 3


# ══════════════════════════════════════════════════════════════
# Rate Limiter (token bucket 단순 구현)
# ══════════════════════════════════════════════════════════════


class _TokenBucketRateLimiter:
    """
    초당 request 수를 제한하는 단순 토큰 버킷 limiter.

    TMDB 50 req/sec 한도 내에서 안전 마진을 두고 사용.
    `acquire()` 호출 시 지난 1초 내 요청 수가 한도 이상이면 대기한다.
    """

    def __init__(self, rps: int):
        self.rps = rps
        self.timestamps: list[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            now = time.monotonic()
            # 1초 이전 타임스탬프 제거
            self.timestamps = [t for t in self.timestamps if now - t < 1.0]
            if len(self.timestamps) >= self.rps:
                wait = 1.0 - (now - self.timestamps[0])
                if wait > 0:
                    await asyncio.sleep(wait)
                now = time.monotonic()
                self.timestamps = [t for t in self.timestamps if now - t < 1.0]
            self.timestamps.append(time.monotonic())


# ══════════════════════════════════════════════════════════════
# TMDB Person Collector
# ══════════════════════════════════════════════════════════════


class TMDBPersonCollector:
    """
    TMDB Person API 비동기 수집기.

    사용 패턴 (async context manager):
        async with TMDBPersonCollector(api_key=...) as collector:
            person_data = await collector.collect_person_full(person_id=525)
    """

    def __init__(
        self,
        api_key: str | None = None,
        rps: int = DEFAULT_RATE_LIMIT_RPS,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.api_key = api_key or settings.TMDB_API_KEY
        if not self.api_key:
            raise ValueError("TMDB_API_KEY 가 설정되지 않았습니다.")
        self.rps = rps
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limiter = _TokenBucketRateLimiter(rps)
        self.client: httpx.AsyncClient | None = None
        self.call_count = 0

    async def __aenter__(self) -> "TMDBPersonCollector":
        self.client = httpx.AsyncClient(
            base_url=TMDB_BASE_URL,
            timeout=self.timeout,
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self.client:
            await self.client.aclose()

    async def _get(self, path: str, params: dict | None = None) -> dict:
        """
        TMDB API GET 요청 (rate limit + 재시도 + 백오프).

        429 (rate limit) → 30s 대기 후 재시도.
        5xx 또는 네트워크 에러 → 지수 백오프 후 재시도.
        4xx (404 등) → 즉시 빈 dict 반환.
        """
        if self.client is None:
            raise RuntimeError("TMDBPersonCollector must be used as async context manager")

        params = dict(params or {})
        params["api_key"] = self.api_key
        params.setdefault("language", "ko-KR")

        for attempt in range(self.max_retries):
            await self.rate_limiter.acquire()
            try:
                response = await self.client.get(path, params=params)
                self.call_count += 1
            except httpx.HTTPError as e:
                logger.warning(
                    "tmdb_person_network_error",
                    path=path,
                    attempt=attempt + 1,
                    error=str(e)[:200],
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except Exception as e:
                    logger.warning("tmdb_person_json_decode_error", path=path, error=str(e))
                    return {}

            if response.status_code == 429:
                # rate limit hit
                wait = 30
                logger.warning("tmdb_person_rate_limit", attempt=attempt + 1, wait=wait)
                await asyncio.sleep(wait)
                continue

            if 400 <= response.status_code < 500:
                # 404 등은 영구 에러 — 즉시 포기
                logger.debug(
                    "tmdb_person_client_error",
                    path=path,
                    status=response.status_code,
                )
                return {}

            # 5xx — 재시도
            logger.warning(
                "tmdb_person_server_error",
                path=path,
                status=response.status_code,
                attempt=attempt + 1,
            )
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        return {}

    async def collect_person_full(self, person_id: int) -> dict | None:
        """
        Person 단일 수집: 기본 + movie_credits + external_ids + images + translations.

        Args:
            person_id: TMDB person ID

        Returns:
            dict | None — 응답이 비어있으면 None
        """
        data = await self._get(
            f"/person/{person_id}",
            params={"append_to_response": PERSON_APPEND_RESOURCES},
        )
        if not data or not data.get("id"):
            return None
        return data

    async def collect_persons_batch(
        self,
        person_ids: list[int],
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> list[dict]:
        """
        여러 person_id 를 비동기 배치 수집한다.

        Args:
            person_ids: TMDB person ID 리스트
            max_workers: 동시 실행 워커 수 (기본 10)

        Returns:
            성공한 person 데이터 리스트 (None/실패 항목 제외)
        """
        sem = asyncio.Semaphore(max_workers)
        results: list[dict] = []

        async def _worker(pid: int) -> None:
            async with sem:
                try:
                    data = await self.collect_person_full(pid)
                    if data:
                        results.append(data)
                except Exception as e:
                    logger.warning("person_collect_failed", person_id=pid, error=str(e)[:200])

        await asyncio.gather(*(_worker(pid) for pid in person_ids))
        return results
