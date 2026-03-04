"""
데이터 파이프라인 오케스트레이터.

§11-1 전체 흐름: 수집 → 전처리 → 임베딩 → 적재

§11-9 배치 실패 복구:
- pipeline_state.json에 진행 상태 저장
- 중단점 재개: 마지막 성공 지점부터 재시작
- 부분 실패: 개별 영화 실패 시 로그 기록 후 건너뜀

§11-11 초기 데이터 적재 예상 시간:
- TMDB 수집: ~1.5시간 (API Rate Limit)
- 무드태그 생성: ~30분 (GPT-4o-mini 배치)
- 임베딩 (CPU): ~1.5시간
- Qdrant 적재: ~5분
- Neo4j 구축: ~15분
- CF 매트릭스: ~20분
- 총합: ~4시간 
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import structlog

from monglepick.data_pipeline.cf_builder import build_cf_matrix, cache_cf_to_redis
from monglepick.data_pipeline.embedder import embed_texts
from monglepick.data_pipeline.es_loader import load_to_elasticsearch
from monglepick.data_pipeline.kaggle_loader import KaggleLoader
from monglepick.data_pipeline.models import MovieDocument, PipelineState, TMDBRawMovie
from monglepick.data_pipeline.neo4j_loader import load_to_neo4j
from monglepick.data_pipeline.preprocessor import process_raw_movie
from monglepick.data_pipeline.qdrant_loader import load_to_qdrant
from monglepick.data_pipeline.tmdb_collector import TMDBCollector
from monglepick.db.clients import init_all_clients, close_all_clients

logger = structlog.get_logger()

# 파이프라인 상태 파일 경로 (§11-9)
STATE_FILE = Path("data/pipeline_state.json")

# TMDB Raw 데이터 캐시 경로 — API 재호출 없이 재처리 가능
TMDB_CACHE_DIR = Path("data/tmdb")
TMDB_RAW_CACHE = TMDB_CACHE_DIR / "tmdb_raw_movies.json"


def _load_state() -> PipelineState:
    """
    파이프라인 진행 상태를 JSON 파일에서 로드한다.

    파일이 없으면 초기 상태(PipelineState)를 반환한다.
    중단된 파이프라인을 이어서 실행할 때 마지막 성공 단계를 파악하는 데 사용한다.

    Returns:
        PipelineState: 복원된 상태 또는 초기 상태
    """
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        return PipelineState(**data)
    return PipelineState()


def _save_state(state: PipelineState) -> None:
    """
    파이프라인 진행 상태를 JSON 파일에 저장한다.

    각 단계(collect, preprocess, embed, load) 완료 시 호출되어
    중단 시 마지막 성공 지점을 기록한다.

    Args:
        state: 현재 파이프라인 상태 (current_step, failed_ids 등)
    """
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state.timestamp = datetime.now().isoformat()
    STATE_FILE.write_text(state.model_dump_json(indent=2))


# ── TMDB Raw 데이터 캐시 함수 ──
# API 호출 비용(~30분, Rate Limit)을 절약하기 위해 수집 결과를 JSON으로 캐싱한다.
# --use-cache 플래그로 캐시에서 로드하여 재처리 시 API 재호출을 방지한다.


def save_tmdb_cache(raw_movies: list[TMDBRawMovie]) -> None:
    """
    TMDB API 수집 결과를 JSON 파일로 캐시한다.

    TMDBRawMovie의 Pydantic model_dump()를 사용하여 직렬화한다.
    캐시 파일 크기: ~3,600건 기준 약 100~200MB.

    Args:
        raw_movies: TMDBCollector에서 수집한 원본 영화 데이터 리스트
    """
    TMDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = [movie.model_dump() for movie in raw_movies]
    with open(TMDB_RAW_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    logger.info(
        "tmdb_cache_saved",
        path=str(TMDB_RAW_CACHE),
        count=len(raw_movies),
    )


def load_tmdb_cache() -> list[TMDBRawMovie] | None:
    """
    JSON 캐시 파일에서 TMDB 원본 영화 데이터를 로드한다.

    캐시 파일이 없거나 파싱 실패 시 None을 반환한다 (API 수집으로 fallback).

    Returns:
        TMDBRawMovie 리스트 또는 None (캐시 없음/오류 시)
    """
    if not TMDB_RAW_CACHE.exists():
        logger.warning("tmdb_cache_not_found", path=str(TMDB_RAW_CACHE))
        return None

    try:
        with open(TMDB_RAW_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_movies = [TMDBRawMovie(**item) for item in data]
        logger.info(
            "tmdb_cache_loaded",
            path=str(TMDB_RAW_CACHE),
            count=len(raw_movies),
        )
        return raw_movies
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("tmdb_cache_load_failed", error=str(e))
        return None


async def run_full_pipeline(
    skip_collect: bool = False,
    skip_mood_generation: bool = False,
    kaggle_data_dir: str = "data/kaggle_movies",
    embedding_batch_size: int = 32,
    use_cache: bool = False,
) -> None:
    """
    전체 데이터 파이프라인을 실행한다.

    §11-1 전체 흐름:
    1. DB 클라이언트 초기화 (Qdrant 컬렉션, Neo4j 인덱스, ES 인덱스)
    2. TMDB 수집 (10,000편 영화 ID + 상세/OTT) — 캐시 사용 가능
    3. 전처리 (장르 변환, 무드태그 생성, 임베딩 텍스트 구성)
    4. 임베딩 (multilingual-e5-large, 1024차원)
    5. 적재 (Qdrant, Neo4j, Elasticsearch 동시)
    6. CF 매트릭스 구축 (Kaggle ratings → Redis 캐시)

    Args:
        skip_collect: True이면 TMDB 수집 건너뜀 (기존 데이터 재적재)
        skip_mood_generation: True이면 GPT 무드태그 생성 건너뜀 (fallback 사용)
        kaggle_data_dir: Kaggle 데이터 디렉토리 경로
        embedding_batch_size: 임베딩 배치 크기 (CPU: 32, GPU: 128)
        use_cache: True이면 TMDB API 대신 캐시 파일에서 로드 (data/tmdb/tmdb_raw_movies.json)
    """
    state = _load_state()

    # ── Step 0: DB 클라이언트 초기화 ──
    logger.info("pipeline_step_0_init_clients")
    await init_all_clients()

    try:
        # ── Step 1: TMDB 수집 (캐시 지원) ──
        documents: list[MovieDocument] = []
        raw_movies: list[TMDBRawMovie] | None = None

        if not skip_collect:
            # use_cache=True: 캐시 파일에서 로드 시도 → 실패 시 API 수집 fallback
            if use_cache:
                logger.info("pipeline_step_1_tmdb_load_from_cache")
                raw_movies = load_tmdb_cache()

            # 캐시 로드 실패 또는 use_cache=False → API 수집
            if raw_movies is None:
                logger.info("pipeline_step_1_tmdb_collect")
                state.current_step = "collect"
                _save_state(state)

                async with TMDBCollector() as collector:
                    # 모든 영화 ID 수집
                    movie_ids = await collector.collect_all_movie_ids()
                    state.total_collected = len(movie_ids)

                    # 상세 정보 수집
                    raw_movies = await collector.collect_full_details(movie_ids)

                # API 수집 완료 후 캐시 파일로 저장 (재처리 시 API 재호출 방지)
                save_tmdb_cache(raw_movies)
            else:
                state.total_collected = len(raw_movies)

            # ── Step 2: 전처리 ──
            logger.info("pipeline_step_2_preprocess", count=len(raw_movies))
            state.current_step = "preprocess"
            _save_state(state)

            for i, raw in enumerate(raw_movies):
                try:
                    doc = await process_raw_movie(
                        raw,
                        generate_mood=not skip_mood_generation,
                    )
                    if doc:
                        documents.append(doc)
                    else:
                        state.failed_ids.append(str(raw.id))
                except Exception as e:
                    logger.warning("preprocess_failed", movie_id=raw.id, error=str(e))
                    state.failed_ids.append(str(raw.id))

                # 진행률 (1000건마다)
                if (i + 1) % 1000 == 0:
                    logger.info("preprocess_progress", completed=i + 1, total=len(raw_movies))

            state.total_processed = len(documents)
            _save_state(state)

        logger.info("pipeline_documents_ready", count=len(documents))

        if not documents:
            logger.warning("pipeline_no_documents_to_load")
            return

        # ── Step 3: 임베딩 ──
        logger.info("pipeline_step_3_embedding", count=len(documents))
        state.current_step = "embed"
        _save_state(state)

        # Upstage Solar embedding-passage 모델로 4096차원 벡터 생성
        texts = [doc.embedding_text for doc in documents]
        embeddings = embed_texts(texts, batch_size=embedding_batch_size)

        # ── Step 4: 3개 DB 동시 적재 (Qdrant 벡터 + Neo4j 그래프 + ES BM25) ──
        logger.info("pipeline_step_4_load", count=len(documents))
        state.current_step = "load"
        _save_state(state)

        qdrant_count = await load_to_qdrant(documents, embeddings)
        await load_to_neo4j(documents)
        es_count = await load_to_elasticsearch(documents)

        state.total_loaded = len(documents)
        _save_state(state)

        logger.info(
            "pipeline_load_complete",
            qdrant=qdrant_count,
            neo4j=len(documents),
            elasticsearch=es_count,
        )

        # ── Step 5: CF 매트릭스 구축 (Kaggle ratings → Redis 캐시) ──
        # links.csv로 movieLens ID → TMDB ID 매핑 후,
        # ratings.csv 26M건으로 유저 유사도 매트릭스를 구축하여 Redis에 캐싱한다.
        logger.info("pipeline_step_5_cf_matrix")
        kaggle = KaggleLoader(kaggle_data_dir)

        kaggle_data_path = Path(kaggle_data_dir)
        if (kaggle_data_path / "ratings.csv").exists():
            id_map = kaggle.load_links()
            ratings_df = kaggle.load_ratings(id_map)

            similar_users, user_ratings, movie_avg = await build_cf_matrix(ratings_df)
            await cache_cf_to_redis(similar_users, user_ratings, movie_avg)
        else:
            logger.warning("kaggle_ratings_not_found", path=kaggle_data_dir)

        # ── 완료 ──
        state.current_step = "complete"
        _save_state(state)

        logger.info(
            "pipeline_complete",
            total_processed=state.total_processed,
            total_loaded=state.total_loaded,
            failed=len(state.failed_ids),
        )

    finally:
        await close_all_clients()
