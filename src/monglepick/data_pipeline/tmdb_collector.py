"""
TMDB API 영화 데이터 수집기.

§11-5 TMDB 수집기 상세:
- collect_popular_movies, collect_top_rated_movies, collect_now_playing, collect_korean_movies
- collect_movie_details (상세 + 크레딧 + 키워드)
- collect_watch_providers (OTT 정보)
- Rate Limiting: asyncio.Semaphore(35) — TMDB 10초당 40회 제한에 여유분 확보
- 10,000편 수집 시 약 1~1.5시간 소요
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from monglepick.config import settings
from monglepick.data_pipeline.models import TMDBRawMovie

logger = structlog.get_logger()

# TMDB API Rate Limit: 10초당 40회 → 여유있게 35로 제한
_semaphore = asyncio.Semaphore(35)


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
        영화 상세정보 수집 (모든 서브리소스 포함).

        Phase C 완전 데이터: append_to_response에 9개 서브리소스를 포함하여
        단 1회의 API 호출로 모든 데이터를 수집한다.
        - credits: 출연진/제작진 (id, name, character, profile_path, gender, popularity, original_name 등)
        - keywords: 키워드 목록
        - reviews: 사용자 리뷰
        - videos: 트레일러/비하인드 영상 (language, official 포함)
        - similar_movies: TMDB 유사 영화
        - release_dates: 국가별 개봉일 + 관람등급
        - images: 다중 포스터/배경 이미지
        - alternative_titles: 대체 제목 (검색 개선용)
        - recommendations: TMDB 추천 영화 (similar와 다른 알고리즘)
        """
        data = await self._get(
            f"/movie/{movie_id}",
            params={
                "append_to_response": (
                    "credits,keywords,reviews,videos,similar_movies,"
                    "release_dates,images,alternative_titles,recommendations"
                ),
            },
        )

        # Phase A: reviews.results에서 author, content, rating 추출
        raw_reviews = data.get("reviews", {}).get("results", [])
        reviews = [
            {
                "author": r.get("author", ""),
                "content": r.get("content", ""),
                "rating": r.get("author_details", {}).get("rating"),
            }
            for r in raw_reviews
        ]

        # Phase C: videos.results에서 모든 유용한 필드 추출 (language/official/published_at 포함)
        raw_videos = data.get("videos", {}).get("results", [])
        videos = [
            {
                "key": v.get("key", ""),
                "type": v.get("type", ""),
                "site": v.get("site", ""),
                "name": v.get("name", ""),
                "iso_639_1": v.get("iso_639_1", ""),  # Phase C: 비디오 언어
                "iso_3166_1": v.get("iso_3166_1", ""),  # Phase C: 비디오 국가
                "official": v.get("official", False),  # Phase C: 공식 여부
                "published_at": v.get("published_at", ""),  # Phase C: 게시 날짜
                "size": v.get("size", 0),  # Phase C: 해상도
            }
            for v in raw_videos
        ]

        # Phase A: similar_movies.results에서 id만 추출
        raw_similar = data.get("similar_movies", {}).get("results", [])
        similar_movie_ids = [m.get("id") for m in raw_similar if m.get("id")]

        # Phase A: release_dates.results (국가별 개봉일 + 관람등급)
        release_dates = data.get("release_dates", {}).get("results", [])

        # Phase C: alternative_titles 추출
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

        # Phase C: recommendations.results에서 id 추출 (similar와 다른 알고리즘)
        raw_recommendations = data.get("recommendations", {}).get("results", [])
        recommendations = [m.get("id") for m in raw_recommendations if m.get("id")]

        # Phase C: images (다중 포스터/배경/로고 이미지 경로)
        raw_images = data.get("images", {})
        images = {
            "posters": [
                img.get("file_path", "")
                for img in raw_images.get("posters", [])[:10]  # 최대 10개
                if img.get("file_path")
            ],
            "backdrops": [
                img.get("file_path", "")
                for img in raw_images.get("backdrops", [])[:10]  # 최대 10개
                if img.get("file_path")
            ],
        }

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
            genres=data.get("genres", []),
            credits=data.get("credits", {}),
            keywords=data.get("keywords", {}),
            reviews=reviews,
            videos=videos,
            similar_movie_ids=similar_movie_ids,
            release_dates=release_dates,
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
        )

    async def collect_watch_providers(self, movie_id: int) -> dict:
        """
        OTT 제공 정보 수집.

        §11-5: /movie/{id}/watch/providers → 한국(KR) flatrate 기준
        """
        data = await self._get(f"/movie/{movie_id}/watch/providers")
        return data.get("results", {})

    # ── 대량 수집 오케스트레이션 ──

    async def collect_all_movie_ids(self) -> list[int]:
        """
        4개 목록 API에서 중복 제거된 전체 영화 ID를 수집한다.

        §11-5: popular + top_rated + now_playing + korean → ~10,000편 (중복 제거 후)
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
        영화 ID 목록에 대해 상세정보 + OTT 정보를 수집한다.

        §11-5: 영화당 상세 API 1회 + OTT API 1회 = 총 2회 호출.
        개별 실패 시 로그 기록 후 건너뛴다 (§11-9: 부분 실패 허용).

        Args:
            movie_ids: 수집할 TMDB 영화 ID 리스트

        Returns:
            성공적으로 수집된 TMDBRawMovie 리스트 (실패 건은 제외)
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
                # 개별 영화 실패는 전체 파이프라인을 중단하지 않음
                logger.warning("tmdb_detail_failed", movie_id=mid, error=str(e))
                continue

        logger.info("tmdb_full_details_collected", success=len(results), total=len(movie_ids))
        return results
