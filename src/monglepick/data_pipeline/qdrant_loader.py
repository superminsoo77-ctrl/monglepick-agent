"""
Qdrant 벡터 DB 적재기.

§11-7-1 Qdrant 적재 명세:
- upsert 배치 크기: 100 points
- 병렬 요청 수: 4 (asyncio.Semaphore(4))
- 재시도: 3회 (지수 백오프: 1s, 2s, 4s)
- wait 옵션: True (upsert 완료 확인)

적재 흐름: MovieDocument[] → 임베딩 배치 생성 → PointStruct 생성 → 배치 upsert
"""

from __future__ import annotations

import asyncio
import uuid

import numpy as np
import structlog
from qdrant_client.models import PointStruct
from tenacity import retry, stop_after_attempt, wait_exponential

from monglepick.config import settings
from monglepick.data_pipeline.models import MovieDocument
from monglepick.db.clients import get_qdrant

logger = structlog.get_logger()


def _to_point_id(doc_id: str) -> int | str:
    """
    MovieDocument ID를 Qdrant PointStruct ID로 변환한다.

    - 순수 숫자 ID (TMDB/Kaggle): int() 변환 (기존 호환)
    - 알파벳 포함 ID (KOBIS 코드 '2026A342' 등): uuid5() 변환

    Qdrant PointStruct ID는 int 또는 UUID(str) 형식만 허용한다.
    KOBIS 코드에 영문자가 포함될 수 있어 int() 변환이 실패하므로,
    uuid5(NAMESPACE_URL, doc_id)를 사용하여 결정적 UUID를 생성한다.
    """
    if doc_id.isdigit():
        return int(doc_id)
    # 알파벳 포함 ID → 결정적 UUID5 변환 (동일 입력 → 동일 UUID)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"kobis:{doc_id}"))

# §11-7-1: 병렬 upsert 제한
_upsert_semaphore = asyncio.Semaphore(4)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=4))
async def _upsert_batch(points: list[PointStruct]) -> None:
    """
    단일 배치를 Qdrant에 upsert한다.

    Semaphore(4)로 동시 upsert를 제한하고, 실패 시 지수 백오프로 최대 3회 재시도한다.
    wait=True 옵션으로 서버 측 인덱싱 완료를 보장한다.

    Args:
        points: upsert할 PointStruct 리스트 (최대 100건)
    """
    async with _upsert_semaphore:
        client = await get_qdrant()
        await client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
            wait=True,  # §11-7-1: 완료 확인 후 다음 배치
        )


def _movie_to_point(doc: MovieDocument, vector: np.ndarray) -> PointStruct:
    """
    MovieDocument + 임베딩 벡터를 Qdrant PointStruct로 변환한다.

    payload에 모든 메타데이터 필드를 포함하여 Qdrant에서 직접 필터/조회가 가능하도록 한다.
    Qdrant payload 인덱스 대상 필드는 db/clients.py에서 설정한다.

    Args:
        doc: 영화 문서 모델
        vector: 임베딩 벡터 (4096차원)

    Returns:
        Qdrant에 적재할 PointStruct
    """
    return PointStruct(
        id=_to_point_id(doc.id),  # TMDB ID → int, KOBIS 코드 → uuid5
        vector=vector.tolist(),
        payload={
            # ── 기본 메타데이터 ──
            "title": doc.title,
            "title_en": doc.title_en,
            "genres": doc.genres,
            "director": doc.director,
            "cast": doc.cast,
            "mood_tags": doc.mood_tags,
            "ott_platforms": doc.ott_platforms,
            "release_year": doc.release_year,
            "rating": doc.rating,
            "popularity_score": doc.popularity_score,
            "overview": doc.overview,
            "keywords": doc.keywords,
            "poster_path": doc.poster_path,
            "runtime": doc.runtime,
            # ── Phase A: TMDB 보강 필드 ──
            "certification": doc.certification,
            "trailer_url": doc.trailer_url,
            "vote_count": doc.vote_count,
            "reviews": doc.reviews,
            "behind_the_scenes": doc.behind_the_scenes,
            "similar_movie_ids": doc.similar_movie_ids,
            # ── Phase B: 재무/흥행/텍스트 ──
            "budget": doc.budget,
            "revenue": doc.revenue,
            "tagline": doc.tagline,
            "homepage": doc.homepage,
            # ── Phase B: 컬렉션/프랜차이즈 ──
            "collection_id": doc.collection_id,
            "collection_name": doc.collection_name,
            # ── Phase B: 제작사 (전체 정보 저장) ──
            "production_companies": [
                {"id": c.get("id", 0), "name": c.get("name", ""),
                 "logo_path": c.get("logo_path", ""), "origin_country": c.get("origin_country", "")}
                for c in doc.production_companies
            ],
            # ── Phase B: 국가/언어 ──
            "production_countries": doc.production_countries,
            "original_language": doc.original_language,
            "spoken_languages": doc.spoken_languages,
            # ── Phase B: 외부 ID 및 미디어 ──
            "imdb_id": doc.imdb_id,
            "backdrop_path": doc.backdrop_path,
            "adult": doc.adult,
            "status": doc.status,
            # ── Phase B: 확장 크레딧 ──
            "cast_characters": doc.cast_characters,
            "cinematographer": doc.cinematographer,
            "composer": doc.composer,
            "screenwriters": doc.screenwriters,
            "producers": doc.producers,
            "editor": doc.editor,
            # ── Phase C: 완전 데이터 추출 ──
            "origin_country": doc.origin_country,
            "director_id": doc.director_id,
            "director_profile_path": doc.director_profile_path,
            "director_original_name": doc.director_original_name,
            "alternative_titles": doc.alternative_titles,
            "recommendation_ids": doc.recommendation_ids,
            "images_posters": doc.images_posters,
            "images_backdrops": doc.images_backdrops,
            "collection_poster_path": doc.collection_poster_path,
            "collection_backdrop_path": doc.collection_backdrop_path,
            "kr_release_date": doc.kr_release_date,
            "video_flag": doc.video_flag,
            "executive_producers": doc.executive_producers,
            "production_designer": doc.production_designer,
            "costume_designer": doc.costume_designer,
            "source_author": doc.source_author,
            "production_country_names": doc.production_country_names,
            "spoken_language_names": doc.spoken_language_names,
            # ── 임베딩 텍스트 (감사 추적용) ──
            "embedding_text": doc.embedding_text,
            # ── KMDb 보강 필드 ──
            "kmdb_id": doc.kmdb_id,
            "awards": doc.awards,
            "audience_count": doc.audience_count,
            "filming_location": doc.filming_location,
            "stills": doc.stills,
            "theme_song": doc.theme_song,
            "soundtrack": doc.soundtrack,
            # ── KOBIS 보강 필드 ──
            "kobis_movie_cd": doc.kobis_movie_cd,
            "sales_acc": doc.sales_acc,
            "screen_count": doc.screen_count,
            "kobis_genres": doc.kobis_genres,
            "kobis_directors": doc.kobis_directors,
            "kobis_actors": doc.kobis_actors,
            "kobis_companies": doc.kobis_companies,
            "kobis_staffs": doc.kobis_staffs,
            "kobis_nation": doc.kobis_nation,
            "kobis_watch_grade": doc.kobis_watch_grade,
            "kobis_open_dt": doc.kobis_open_dt,
            "kobis_type_nm": doc.kobis_type_nm,
            # ── 데이터 출처 추적 ──
            "source": doc.source,
        },
    )


async def load_to_qdrant(
    documents: list[MovieDocument],
    embeddings: np.ndarray,
    batch_size: int = 100,
) -> int:
    """
    MovieDocument 리스트와 임베딩을 Qdrant에 적재한다.

    §11-7-1 적재 흐름:
    [1] PointStruct 생성 → [2] 100건씩 배치 → [3] upsert (4 병렬)

    Args:
        documents: 적재할 MovieDocument 리스트
        embeddings: 임베딩 벡터 배열 (shape: len(documents) × 1024)
        batch_size: upsert 배치 크기 (기본 100)

    Returns:
        int: 적재 완료된 포인트 수
    """
    # PointStruct 변환
    points = [
        _movie_to_point(doc, embeddings[i])
        for i, doc in enumerate(documents)
    ]

    # 배치 분할 후 Semaphore(4)로 병렬 upsert 실행
    tasks = []
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        tasks.append(_upsert_batch(batch))

    await asyncio.gather(*tasks)

    # 적재 검증 (§11-7-1 [4])
    client = await get_qdrant()
    info = await client.get_collection(settings.QDRANT_COLLECTION)
    logger.info(
        "qdrant_load_complete",
        loaded=len(points),
        total_in_collection=info.points_count,
    )

    return len(points)
