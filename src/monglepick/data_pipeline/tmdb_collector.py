"""
TMDB API 영화 데이터 수집기.

§11-5 TMDB 수집기 상세:
- collect_popular_movies, collect_top_rated_movies, collect_now_playing, collect_korean_movies
- collect_movie_details (상세 + 14개 서브리소스 통합)
- Rate Limiting: asyncio.Semaphore(35) — TMDB 10초당 40회 제한에 여유분 확보

Phase D: 전체 TMDB 데이터 수집 (Daily Export 기반)
- download_daily_export(): TMDB Daily Export 파일에서 전체 영화 ID 풀 (~1M+) 다운로드
- collect_movie_details_full(): 14개 서브리소스를 단 1회 API 호출로 수집
- collect_full_details_with_checkpoint(): 체크포인트 기반 대량 수집 (며칠에 걸쳐 재개 가능)
- 예상 소요: ~1M건 / ~35 req/sec = ~8시간 연속 수집
"""

from __future__ import annotations

import asyncio
import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from monglepick.config import settings
from monglepick.data_pipeline.models import TMDBRawMovie

logger = structlog.get_logger()

# TMDB API Rate Limit: 10초당 40회 → 여유있게 35로 제한
_semaphore = asyncio.Semaphore(35)

# ── Phase D: 전체 수집 관련 상수 ──
# Daily Export 파일 URL 템플릿 (TMDB가 매일 08:00 UTC에 생성)
_DAILY_EXPORT_URL = "http://files.tmdb.org/p/exports/movie_ids_{date}.json.gz"
# 전체 수집 데이터 저장 디렉토리
_FULL_DATA_DIR = Path("data/tmdb_full")
# 체크포인트 파일 (수집 진행 상태 기록)
_CHECKPOINT_FILE = _FULL_DATA_DIR / "checkpoint.json"
# 수집된 영화 데이터 JSONL 파일 (한 줄에 하나의 JSON)
_MOVIES_JSONL = _FULL_DATA_DIR / "tmdb_full_movies.jsonl"
# Daily Export 캐시 파일
_EXPORT_CACHE = _FULL_DATA_DIR / "daily_export_ids.json"

# append_to_response에 포함할 14개 서브리소스 (TMDB movie 엔드포인트 전체)
# 1회 API 호출로 기본 상세 + 14개 서브리소스를 모두 수집한다.
_ALL_SUB_RESOURCES = (
    "alternative_titles,"   # 대체 제목 (검색 recall 개선)
    "changes,"              # 최근 변경 이력 (데이터 신선도 추적)
    "credits,"              # 출연진/제작진 전체
    "external_ids,"         # IMDb, Facebook, Instagram, Twitter, Wikidata ID
    "images,"               # 다중 포스터/배경/로고 이미지
    "keywords,"             # 키워드 목록
    "lists,"                # 이 영화가 포함된 사용자 리스트
    "recommendations,"      # TMDB 추천 영화
    "release_dates,"        # 국가별 개봉일 + 관람등급
    "reviews,"              # 사용자 리뷰
    "similar,"              # TMDB 유사 영화
    "translations,"         # 다국어 번역 (overview 빈값 보강용)
    "videos,"               # 트레일러/비하인드 영상
    "watch/providers"       # OTT 제공 정보 (별도 API 호출 불필요)
)


class TMDBCollector:
    """
    TMDB API 비동기 수집기.

    사용 예:
        async with TMDBCollector() as collector:
            movies = await collector.collect_popular_movies(pages=50)
            for movie in movies:
                detail = await collector.collect_movie_details(movie["id"])
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url = settings.TMDB_BASE_URL
        self._api_key = settings.TMDB_API_KEY

    async def __aenter__(self) -> TMDBCollector:
        """비동기 컨텍스트 매니저 진입: httpx 클라이언트를 초기화한다."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            params={"api_key": self._api_key, "language": "ko-KR"},
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        """비동기 컨텍스트 매니저 종료: httpx 클라이언트를 정리한다."""
        if self._client:
            await self._client.aclose()

    # ── 내부 HTTP 호출 (Rate Limit 적용) ──

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
    async def _get(self, path: str, params: dict | None = None) -> dict:
        """Rate-limited GET 요청. 최대 3회 재시도 (지수 백오프)."""
        async with _semaphore:
            assert self._client is not None, "TMDBCollector must be used as async context manager"
            resp = await self._client.get(path, params=params or {})
            resp.raise_for_status()
            return resp.json()

    # ── 목록 수집 메서드 ──

    async def _collect_list(self, endpoint: str, pages: int, extra_params: dict | None = None) -> list[dict]:
        """
        페이지네이션 목록 API에서 영화 ID 목록을 수집한다.

        Args:
            endpoint: TMDB API 엔드포인트 (예: "/movie/popular")
            pages: 최대 수집할 페이지 수
            extra_params: 추가 쿼리 파라미터 (예: region, sort_by)

        Returns:
            영화 요약 정보 딕셔너리 리스트
        """
        results: list[dict] = []
        for page in range(1, pages + 1):
            params = {"page": page, **(extra_params or {})}
            data = await self._get(endpoint, params)
            results.extend(data.get("results", []))
            # TMDB 응답의 total_pages를 확인하여 마지막 페이지면 조기 종료
            total_pages = data.get("total_pages", 1)
            if page >= total_pages:
                break
        logger.info("tmdb_list_collected", endpoint=endpoint, count=len(results))
        return results

    async def collect_popular_movies(self, pages: int = 50) -> list[dict]:
        """인기 영화 수집 (~1,000편). §11-5: /movie/popular"""
        return await self._collect_list("/movie/popular", pages)

    async def collect_top_rated_movies(self, pages: int = 50) -> list[dict]:
        """높은 평점 영화 수집 (~1,000편). §11-5: /movie/top_rated"""
        return await self._collect_list("/movie/top_rated", pages)

    async def collect_now_playing(self) -> list[dict]:
        """현재 상영 중 영화 수집 (~40편). §11-5: /movie/now_playing"""
        return await self._collect_list("/movie/now_playing", pages=2)

    async def collect_korean_movies(self, pages: int = 100) -> list[dict]:
        """한국 영화 수집 (~2,000편). §11-5: /discover/movie?region=KR"""
        return await self._collect_list(
            "/discover/movie",
            pages,
            extra_params={"region": "KR", "with_original_language": "ko", "sort_by": "popularity.desc"},
        )

    # ── 상세 수집 메서드 ──

    async def collect_movie_details(self, movie_id: int) -> TMDBRawMovie:
        """
        영화 상세정보 수집 (기존 9개 서브리소스, 하위 호환용).

        기존 파이프라인(run_pipeline.py)에서 사용하는 메서드.
        전체 수집은 collect_movie_details_full()을 사용한다.
        """
        return await self._collect_details_with_append(
            movie_id,
            append_to_response=(
                "credits,keywords,reviews,videos,similar_movies,"
                "release_dates,images,alternative_titles,recommendations"
            ),
        )

    async def collect_movie_details_full(self, movie_id: int) -> TMDBRawMovie:
        """
        영화 상세정보 수집 (14개 서브리소스 전체 포함).

        Phase D 전체 수집: append_to_response에 14개 서브리소스를 포함하여
        단 1회의 API 호출로 TMDB가 제공하는 모든 데이터를 수집한다.

        포함되는 서브리소스 (14개):
        - alternative_titles: 대체 제목 (검색 recall 개선)
        - changes: 최근 변경 이력 (데이터 신선도 추적)
        - credits: 출연진/제작진 전체 (id, name, character, profile_path 등)
        - external_ids: IMDb, Facebook, Instagram, Twitter, Wikidata ID
        - images: 다중 포스터/배경/로고 이미지
        - keywords: 키워드 목록
        - lists: 이 영화가 포함된 TMDB 사용자 리스트
        - recommendations: TMDB 추천 영화 (similar와 다른 알고리즘)
        - release_dates: 국가별 개봉일 + 관람등급
        - reviews: 사용자 리뷰
        - similar: TMDB 유사 영화
        - translations: 다국어 번역 (overview 빈값 보강용 핵심)
        - videos: 트레일러/비하인드 영상
        - watch/providers: OTT 제공 정보 (별도 API 호출 불필요)
        """
        return await self._collect_details_with_append(
            movie_id,
            append_to_response=_ALL_SUB_RESOURCES,
        )

    async def _collect_details_with_append(
        self,
        movie_id: int,
        append_to_response: str,
    ) -> TMDBRawMovie:
        """
        append_to_response 파라미터로 영화 상세 + 서브리소스를 수집하는 내부 메서드.

        collect_movie_details()와 collect_movie_details_full()이 공유한다.
        서브리소스 응답을 파싱하여 TMDBRawMovie로 변환한다.

        Args:
            movie_id: TMDB 영화 ID
            append_to_response: 쉼표 구분 서브리소스 문자열

        Returns:
            TMDBRawMovie: 파싱된 영화 데이터
        """
        data = await self._get(
            f"/movie/{movie_id}",
            params={"append_to_response": append_to_response},
        )

        # ── reviews.results 파싱 (Phase D: 원본 dict 전체 저장) ──
        # 기존: author/content/rating 3개 필드만 추출
        # 변경: created_at, updated_at, url, id, author_details 등 전체 보존
        raw_reviews = data.get("reviews", {}).get("results", [])
        reviews = list(raw_reviews)  # 원본 전체 저장

        # ── videos.results 파싱 (Phase D: 원본 dict 전체 저장) ──
        # 기존: 9개 필드를 수동 선택 → 변경: 원본 전체 저장으로 TMDB id 등 누락 방지
        raw_videos = data.get("videos", {}).get("results", [])
        videos = list(raw_videos)  # 원본 전체 저장

        # ── similar movies 파싱 (Phase D: 전체 dict 저장 + 하위 호환 ID 리스트) ──
        # 기존: ID만 추출 → 변경: title, overview, poster_path 등 전체 메타데이터 보존
        raw_similar = (
            data.get("similar_movies", {}).get("results", [])
            or data.get("similar", {}).get("results", [])
        )
        similar_movie_ids = [m.get("id") for m in raw_similar if m.get("id")]

        # ── release_dates 파싱 ──
        release_dates = data.get("release_dates", {}).get("results", [])

        # ── alternative_titles 파싱 ──
        raw_alt_titles = data.get("alternative_titles", {}).get("titles", [])
        alternative_titles = [
            {
                "iso_3166_1": t.get("iso_3166_1", ""),
                "title": t.get("title", ""),
                "type": t.get("type", ""),
            }
            for t in raw_alt_titles
            if t.get("title")
        ]

        # ── recommendations 파싱 (Phase D: 전체 dict 저장 + 하위 호환 ID 리스트) ──
        # 기존: ID만 추출 → 변경: title, overview, poster_path 등 전체 메타데이터 보존
        raw_recommendations = data.get("recommendations", {}).get("results", [])
        recommendations = list(raw_recommendations)  # 전체 dict 저장

        # ── images 파싱 (Phase D: 절단 제거, 전체 이미지 저장) ──
        # 기존: posters[:10], backdrops[:10], logos[:5] 절단
        # 변경: 제한 없이 전체 이미지 경로 저장
        raw_images = data.get("images", {})
        images = {
            "posters": [
                img.get("file_path", "")
                for img in raw_images.get("posters", [])
                if img.get("file_path")
            ],
            "backdrops": [
                img.get("file_path", "")
                for img in raw_images.get("backdrops", [])
                if img.get("file_path")
            ],
            "logos": [
                img.get("file_path", "")
                for img in raw_images.get("logos", [])
                if img.get("file_path")
            ],
        }

        # ── Phase D: translations 파싱 (다국어 번역, data dict 전체 저장) ──
        # 기존: data에서 title/overview/tagline/homepage/runtime 5개만 추출
        # 변경: data dict 전체를 보존하여 누락 필드 방지
        raw_translations = data.get("translations", {}).get("translations", [])
        translations = [
            {
                "iso_3166_1": t.get("iso_3166_1", ""),
                "iso_639_1": t.get("iso_639_1", ""),
                "name": t.get("name", ""),
                "english_name": t.get("english_name", ""),
                "data": t.get("data", {}),  # data dict 전체 보존
            }
            for t in raw_translations
        ]

        # ── Phase D: external_ids 파싱 (raw dict 전체 저장) ──
        # 기존: 5개 ID만 추출 (imdb, facebook, instagram, twitter, wikidata)
        # 변경: TMDB 응답 전체를 보존하여 새로운 외부 ID가 추가되어도 누락 방지
        raw_external = data.get("external_ids", {})
        # id 키는 영화 자체 ID이므로 제거 (중복)
        external_ids = {k: v for k, v in raw_external.items() if k != "id"}

        # ── Phase D: lists 파싱 (절단 제거 + id 없는 항목 필터링) ──
        # 기존: results[:20] 절단, id 기본값 없음
        # 변경: 제한 제거 (전체 저장), id 없는 항목은 필터링
        raw_lists = data.get("lists", {})
        lists_data = {
            "total_results": raw_lists.get("total_results", 0),
            "results": [
                {"id": lst.get("id"), "name": lst.get("name", "")}
                for lst in raw_lists.get("results", [])
                if lst.get("id") is not None  # id 없는 항목 필터링 (버그 수정)
            ],
        }

        # ── Phase D: watch/providers 파싱 (OTT 제공 정보, 별도 API 호출 불필요) ──
        # append_to_response로 포함 시 응답 키가 "watch/providers"
        watch_providers = data.get("watch/providers", {}).get("results", {})

        return TMDBRawMovie(
            id=data.get("id", movie_id),
            title=data.get("title", ""),
            original_title=data.get("original_title", ""),
            overview=data.get("overview", ""),
            release_date=data.get("release_date", ""),
            vote_average=data.get("vote_average", 0.0),
            vote_count=data.get("vote_count", 0),
            popularity=data.get("popularity", 0.0),
            poster_path=data.get("poster_path"),
            runtime=data.get("runtime"),
            # Phase D: video 플래그 추가
            video=data.get("video", False),
            genres=data.get("genres", []),
            credits=data.get("credits", {}),
            keywords=data.get("keywords", {}),
            reviews=reviews,
            videos=videos,
            similar_movie_ids=similar_movie_ids,
            release_dates=release_dates,
            watch_providers=watch_providers,
            # Phase B: TMDB 추가 필드
            budget=data.get("budget", 0) or 0,
            revenue=data.get("revenue", 0) or 0,
            tagline=data.get("tagline", "") or "",
            homepage=data.get("homepage", "") or "",
            belongs_to_collection=data.get("belongs_to_collection"),
            production_companies=data.get("production_companies", []),
            production_countries=data.get("production_countries", []),
            original_language=data.get("original_language", "") or "",
            spoken_languages=data.get("spoken_languages", []),
            imdb_id=data.get("imdb_id", "") or "",
            backdrop_path=data.get("backdrop_path"),
            adult=data.get("adult", False),
            status=data.get("status", "") or "",
            # Phase C: 완전 데이터 추출
            origin_country=data.get("origin_country", []),
            alternative_titles=alternative_titles,
            recommendations=recommendations,
            images=images,
            # Phase D: 전체 데이터 수집
            translations=translations,
            external_ids=external_ids,
            lists=lists_data,
        )

    async def collect_watch_providers(self, movie_id: int) -> dict:
        """
        OTT 제공 정보 수집 (별도 API 호출, 기존 호환용).

        Phase D부터는 append_to_response에 watch/providers가 포함되므로
        별도 호출이 불필요하다. 기존 collect_full_details()에서만 사용.
        """
        data = await self._get(f"/movie/{movie_id}/watch/providers")
        return data.get("results", {})

    # ── 기존 대량 수집 (하위 호환) ──

    async def collect_all_movie_ids(self) -> list[int]:
        """
        4개 목록 API에서 중복 제거된 전체 영화 ID를 수집한다.

        §11-5: popular + top_rated + now_playing + korean → ~3,600편 (중복 제거 후)
        전체 TMDB 수집은 download_daily_export()를 사용한다.
        """
        # 4개 목록 동시 수집
        popular, top_rated, now_playing, korean = await asyncio.gather(
            self.collect_popular_movies(pages=50),
            self.collect_top_rated_movies(pages=50),
            self.collect_now_playing(),
            self.collect_korean_movies(pages=100),
        )

        # 중복 제거 (TMDB ID 기준)
        seen: set[int] = set()
        unique_ids: list[int] = []
        for movie in popular + top_rated + now_playing + korean:
            mid = movie.get("id")
            if mid and mid not in seen:
                seen.add(mid)
                unique_ids.append(mid)

        logger.info("tmdb_all_ids_collected", total=len(unique_ids))
        return unique_ids

    async def collect_full_details(self, movie_ids: list[int]) -> list[TMDBRawMovie]:
        """
        영화 ID 목록에 대해 상세정보 + OTT 정보를 수집한다 (기존 호환).

        기존 파이프라인용. 영화당 2회 API 호출 (상세 + OTT).
        전체 수집은 collect_full_details_with_checkpoint()를 사용한다.
        """
        results: list[TMDBRawMovie] = []

        for i, mid in enumerate(movie_ids):
            try:
                movie = await self.collect_movie_details(mid)

                # OTT 정보는 별도 API로 수집 후 병합
                providers = await self.collect_watch_providers(mid)
                movie.watch_providers = providers

                results.append(movie)

                if (i + 1) % 500 == 0:
                    logger.info("tmdb_detail_progress", completed=i + 1, total=len(movie_ids))

            except Exception as e:
                logger.warning("tmdb_detail_failed", movie_id=mid, error=str(e))
                continue

        logger.info("tmdb_full_details_collected", success=len(results), total=len(movie_ids))
        return results

    # ══════════════════════════════════════════════════════════════
    # Phase D: 전체 TMDB 데이터 수집 (Daily Export 기반, 1M+ 영화)
    # ══════════════════════════════════════════════════════════════

    async def download_daily_export(
        self,
        min_popularity: float = 0.0,
        exclude_adult: bool = True,
        use_cache: bool = True,
    ) -> list[int]:
        """
        TMDB Daily Export 파일에서 전체 영화 ID 목록을 다운로드한다.

        TMDB는 매일 08:00 UTC에 전체 영화 ID를 gzip JSON Lines 형식으로 제공한다.
        URL: http://files.tmdb.org/p/exports/movie_ids_MM_DD_YYYY.json.gz
        각 라인: {"adult":false,"id":550,"original_title":"Fight Club","popularity":45.12,"video":false}

        Args:
            min_popularity: 최소 인기도 필터 (0.0이면 전체 수집, 기본값)
            exclude_adult: 성인물 제외 여부 (기본 True)
            use_cache: True이면 이전 다운로드 캐시 사용 (기본 True)

        Returns:
            정렬된 영화 ID 리스트 (인기도 내림차순)
        """
        _FULL_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 캐시 파일이 존재하고 24시간 이내면 재사용
        if use_cache and _EXPORT_CACHE.exists():
            cache_age = datetime.now().timestamp() - _EXPORT_CACHE.stat().st_mtime
            if cache_age < 86400:  # 24시간
                cached = json.loads(_EXPORT_CACHE.read_text())
                logger.info("daily_export_cache_loaded", count=len(cached))
                return cached

        # Daily Export 파일 URL 생성 (오늘 또는 어제 날짜)
        # TMDB는 08:00 UTC에 생성하므로, 당일 파일이 없을 수 있어 어제 날짜도 시도
        today = datetime.utcnow()
        dates_to_try = [
            today.strftime("%m_%d_%Y"),
            (today - timedelta(days=1)).strftime("%m_%d_%Y"),
            (today - timedelta(days=2)).strftime("%m_%d_%Y"),
        ]

        gz_data: bytes | None = None
        used_date = ""
        for date_str in dates_to_try:
            url = _DAILY_EXPORT_URL.format(date=date_str)
            try:
                logger.info("daily_export_downloading", url=url)
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        gz_data = resp.content
                        used_date = date_str
                        break
                    else:
                        logger.debug("daily_export_not_found", date=date_str, status=resp.status_code)
            except Exception as e:
                logger.debug("daily_export_download_error", date=date_str, error=str(e))

        if gz_data is None:
            logger.error("daily_export_download_failed", tried_dates=dates_to_try)
            raise RuntimeError("TMDB Daily Export 다운로드 실패. 네트워크 연결을 확인하세요.")

        # gzip 해제 후 JSON Lines 파싱 (Phase D: 에러 처리 추가)
        try:
            decompressed = gzip.decompress(gz_data)
        except (gzip.BadGzipFile, OSError) as e:
            logger.error("daily_export_gzip_decompress_failed", error=str(e))
            raise RuntimeError(f"TMDB Daily Export gzip 해제 실패: {e}") from e
        lines = decompressed.decode("utf-8").strip().split("\n")

        # 각 라인을 파싱하여 영화 ID + 인기도 추출, 필터링 적용
        movie_entries: list[tuple[int, float]] = []
        parse_errors = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                movie_id = entry.get("id")
                popularity = entry.get("popularity", 0.0)
                is_adult = entry.get("adult", False)

                if not movie_id:
                    continue
                if exclude_adult and is_adult:
                    continue
                if popularity < min_popularity:
                    continue

                movie_entries.append((movie_id, popularity))
            except json.JSONDecodeError:
                parse_errors += 1

        # 인기도 내림차순 정렬 (인기 영화 먼저 수집)
        movie_entries.sort(key=lambda x: x[1], reverse=True)
        movie_ids = [mid for mid, _ in movie_entries]

        # 캐시 저장
        _EXPORT_CACHE.write_text(json.dumps(movie_ids))

        logger.info(
            "daily_export_parsed",
            date=used_date,
            total_lines=len(lines),
            filtered_count=len(movie_ids),
            parse_errors=parse_errors,
            min_popularity=min_popularity,
            exclude_adult=exclude_adult,
        )
        return movie_ids

    async def collect_full_details_with_checkpoint(
        self,
        movie_ids: list[int],
        batch_size: int = 1000,
        save_interval: int = 100,
    ) -> dict:
        """
        체크포인트 기반으로 전체 영화 상세정보를 수집한다.

        며칠에 걸쳐 수집을 중단/재개할 수 있도록 체크포인트 파일에 진행 상태를 기록한다.
        수집된 데이터는 JSONL 형식으로 저장되어 메모리 부담 없이 대량 데이터를 처리한다.

        동작 흐름:
        1. 체크포인트 파일에서 마지막 수집 위치를 로드
        2. 남은 ID 목록에서 순차적으로 collect_movie_details_full() 호출
        3. save_interval마다 JSONL에 flush + 체크포인트 갱신
        4. 개별 영화 실패 시 failed_ids에 기록하고 계속 진행
        5. 모든 ID 처리 완료 또는 KeyboardInterrupt 시 최종 체크포인트 저장

        Args:
            movie_ids: 수집할 전체 영화 ID 리스트
            batch_size: 로그 출력 간격 (기본 1000)
            save_interval: JSONL flush + 체크포인트 저장 간격 (기본 100)

        Returns:
            수집 결과 요약 dict:
            {
                "total": 전체 ID 수,
                "collected": 성공 수집 수,
                "failed": 실패 수,
                "skipped": 이미 수집된 수 (체크포인트 재개 시),
                "output_file": JSONL 파일 경로,
            }
        """
        _FULL_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # ── 체크포인트 로드: 이전 수집 상태 복원 ──
        checkpoint = self._load_checkpoint()
        collected_ids: set[int] = set(checkpoint.get("collected_ids", []))
        failed_ids: set[int] = set(checkpoint.get("failed_ids", []))
        total_collected = len(collected_ids)

        # 이미 수집된 ID를 제외한 남은 ID 목록
        remaining_ids = [mid for mid in movie_ids if mid not in collected_ids and mid not in failed_ids]
        skipped = len(movie_ids) - len(remaining_ids)

        logger.info(
            "full_collection_start",
            total_ids=len(movie_ids),
            remaining=len(remaining_ids),
            already_collected=len(collected_ids),
            already_failed=len(failed_ids),
        )

        if not remaining_ids:
            logger.info("full_collection_already_complete")
            return {
                "total": len(movie_ids),
                "collected": len(collected_ids),
                "failed": len(failed_ids),
                "skipped": skipped,
                "output_file": str(_MOVIES_JSONL),
            }

        # ── JSONL 파일에 append 모드로 수집 데이터 기록 ──
        new_collected = 0
        new_failed = 0

        try:
            with open(_MOVIES_JSONL, "a", encoding="utf-8") as jsonl_file:
                for i, mid in enumerate(remaining_ids):
                    try:
                        # 14개 서브리소스를 포함한 전체 상세 수집 (1회 API 호출)
                        movie = await self.collect_movie_details_full(mid)

                        # JSONL에 한 줄로 기록 (Pydantic model_dump → JSON)
                        json_line = json.dumps(movie.model_dump(), ensure_ascii=False)
                        jsonl_file.write(json_line + "\n")

                        collected_ids.add(mid)
                        new_collected += 1

                    except httpx.HTTPStatusError as e:
                        # 404: 삭제된 영화, 기타 HTTP 에러
                        if e.response.status_code == 404:
                            logger.debug("movie_not_found", movie_id=mid)
                        else:
                            logger.warning(
                                "movie_detail_http_error",
                                movie_id=mid,
                                status=e.response.status_code,
                            )
                        failed_ids.add(mid)
                        new_failed += 1

                    except Exception as e:
                        logger.warning("movie_detail_error", movie_id=mid, error=str(e))
                        failed_ids.add(mid)
                        new_failed += 1

                    # ── 주기적 체크포인트 저장 + 로그 ──
                    if (i + 1) % save_interval == 0:
                        jsonl_file.flush()
                        self._save_checkpoint(collected_ids, failed_ids)

                    if (i + 1) % batch_size == 0:
                        total_now = len(collected_ids)
                        elapsed_pct = (i + 1) / len(remaining_ids) * 100
                        logger.info(
                            "full_collection_progress",
                            progress=f"{i + 1}/{len(remaining_ids)}",
                            percent=f"{elapsed_pct:.1f}%",
                            total_collected=total_now,
                            new_failed=new_failed,
                        )

        except KeyboardInterrupt:
            # Ctrl+C로 중단 시에도 체크포인트를 저장하여 다음 실행에서 이어갈 수 있도록
            logger.warning("full_collection_interrupted", saving_checkpoint=True)
        finally:
            # 최종 체크포인트 저장
            self._save_checkpoint(collected_ids, failed_ids)

        result = {
            "total": len(movie_ids),
            "collected": len(collected_ids),
            "failed": len(failed_ids),
            "skipped": skipped,
            "new_collected": new_collected,
            "new_failed": new_failed,
            "output_file": str(_MOVIES_JSONL),
        }
        logger.info("full_collection_complete", **result)
        return result

    # ── 체크포인트 관리 ──

    @staticmethod
    def _load_checkpoint() -> dict:
        """
        체크포인트 파일에서 수집 진행 상태를 로드한다.

        체크포인트 형식:
        {
            "collected_ids": [int, ...],  # 성공 수집된 영화 ID 목록
            "failed_ids": [int, ...],     # 실패한 영화 ID 목록
            "last_updated": "ISO 8601",   # 마지막 저장 시각
        }

        Returns:
            체크포인트 dict (파일 없으면 빈 상태)
        """
        if _CHECKPOINT_FILE.exists():
            try:
                data = json.loads(_CHECKPOINT_FILE.read_text())
                logger.info(
                    "checkpoint_loaded",
                    collected=len(data.get("collected_ids", [])),
                    failed=len(data.get("failed_ids", [])),
                    last_updated=data.get("last_updated", ""),
                )
                return data
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("checkpoint_load_failed", error=str(e))
        return {"collected_ids": [], "failed_ids": []}

    @staticmethod
    def _save_checkpoint(collected_ids: set[int], failed_ids: set[int]) -> None:
        """
        수집 진행 상태를 체크포인트 파일에 저장한다.

        save_interval마다 호출되며, KeyboardInterrupt 시에도 호출된다.
        다음 실행 시 이 파일에서 상태를 복원하여 수집을 이어간다.

        Args:
            collected_ids: 성공 수집된 영화 ID set
            failed_ids: 실패한 영화 ID set
        """
        _FULL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "collected_ids": sorted(collected_ids),
            "failed_ids": sorted(failed_ids),
            "last_updated": datetime.now().isoformat(),
            "total_collected": len(collected_ids),
            "total_failed": len(failed_ids),
        }
        _CHECKPOINT_FILE.write_text(json.dumps(checkpoint, ensure_ascii=False))
        logger.debug(
            "checkpoint_saved",
            collected=len(collected_ids),
            failed=len(failed_ids),
        )
