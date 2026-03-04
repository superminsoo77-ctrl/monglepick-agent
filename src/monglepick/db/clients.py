"""
DB 클라이언트 초기화 및 컬렉션/인덱스 설정.

§10-2-1 Qdrant 컬렉션, §10-8 Elasticsearch 인덱스 설정을 포함한다.
모든 클라이언트는 싱글턴 패턴으로 관리하며, FastAPI lifespan에서 초기화/종료한다.
"""

from __future__ import annotations

import time
import traceback

import aiomysql
import structlog
from elasticsearch import AsyncElasticsearch
from neo4j import AsyncGraphDatabase, AsyncDriver
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    VectorParams,
)
from redis.asyncio import Redis

from monglepick.config import settings

logger = structlog.get_logger()

# ── 싱글턴 클라이언트 인스턴스 ──
_qdrant_client: AsyncQdrantClient | None = None
_neo4j_driver: AsyncDriver | None = None
_redis_client: Redis | None = None
_es_client: AsyncElasticsearch | None = None
_mysql_pool: aiomysql.Pool | None = None


# ============================================================
# Qdrant
# ============================================================

async def get_qdrant() -> AsyncQdrantClient:
    """Qdrant 비동기 클라이언트를 반환한다 (싱글턴)."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            check_compatibility=False,  # 클라이언트/서버 마이너 버전 차이 허용
        )
        logger.info("qdrant_client_initialized", url=settings.QDRANT_URL)
    return _qdrant_client


async def ensure_qdrant_collection() -> None:
    """
    Qdrant 'movies' 컬렉션이 없으면 생성하고 payload 인덱스를 설정한다.

    §10-2-1 설정:
    - 벡터 크기: 1024 (multilingual-e5-large)
    - 거리 메트릭: Cosine
    - HNSW: M=16, ef_construct=100
    - Payload 인덱스: genres, director, mood_tags, release_year, rating, popularity_score, ott_platforms, title
    """
    client = await get_qdrant()
    collection_name = settings.QDRANT_COLLECTION

    # 컬렉션 존재 여부 확인
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]

    if collection_name not in existing_names:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,  # 1024
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )
        logger.info("qdrant_collection_created", name=collection_name)

    # Payload 인덱스 설정 (§10-2-1: 필터 필드)
    # Phase C: 모든 필터 가능 필드에 인덱스 설정 (검색 성능 최적화)
    keyword_fields = [
        "title", "genres", "director", "mood_tags", "ott_platforms", "certification",
        "original_language", "production_countries", "status", "collection_name",
        # Phase C 추가: cast, keywords, origin_country, imdb_id
        "cast", "keywords", "origin_country", "imdb_id", "source",
        # KOBIS 보강: 필터링 가능 필드
        "kobis_movie_cd", "kobis_genres", "kobis_nation", "kobis_watch_grade", "kobis_type_nm",
    ]
    for field in keyword_fields:
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )

    # Phase C: 모든 정수 필터 필드
    integer_fields = [
        "release_year", "collection_id", "budget", "revenue", "vote_count",
        "runtime", "audience_count", "director_id",
        # KOBIS 보강: 매출/스크린 수
        "sales_acc", "screen_count",
    ]
    for field in integer_fields:
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.INTEGER,
        )

    float_fields = ["rating", "popularity_score"]
    for field in float_fields:
        await client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.FLOAT,
        )

    logger.info("qdrant_payload_indexes_configured", collection=collection_name)


# ============================================================
# Neo4j
# ============================================================

async def get_neo4j() -> AsyncDriver:
    """Neo4j 비동기 드라이버를 반환한다 (싱글턴)."""
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        logger.info("neo4j_driver_initialized", uri=settings.NEO4J_URI)
    return _neo4j_driver


async def ensure_neo4j_indexes() -> None:
    """
    Neo4j 인덱스 및 제약조건을 생성한다.

    §10-3-1: 각 노드 라벨에 유니크 제약조건 + 인덱스 설정
    """
    driver = await get_neo4j()
    async with driver.session() as session:
        # 유니크 제약조건 (MERGE 성능 보장)
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (mt:MoodTag) REQUIRE mt.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:OTTPlatform) REQUIRE o.name IS UNIQUE",
            # Phase B: 새 노드 타입 제약조건
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Studio) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Collection) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (co:Country) REQUIRE co.iso_code IS UNIQUE",
        ]
        for cypher in constraints:
            await session.run(cypher)

        logger.info("neo4j_indexes_configured")


# ============================================================
# Redis
# ============================================================

async def get_redis() -> Redis:
    """Redis 비동기 클라이언트를 반환한다 (싱글턴)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
        logger.info("redis_client_initialized", url=settings.REDIS_URL)
    return _redis_client


# ============================================================
# MySQL
# ============================================================

async def get_mysql() -> aiomysql.Pool:
    """
    MySQL 비동기 커넥션 풀을 반환한다 (싱글턴).

    aiomysql.create_pool()로 커넥션 풀을 생성하며, 최소 1개 / 최대 10개 연결을 유지한다.
    charset은 utf8mb4로 설정하여 한국어/이모지를 지원한다.
    """
    global _mysql_pool
    if _mysql_pool is None:
        _mysql_pool = await aiomysql.create_pool(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            db=settings.MYSQL_DATABASE,
            charset="utf8mb4",
            minsize=1,
            maxsize=10,
            autocommit=True,
        )
        logger.info(
            "mysql_pool_initialized",
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            db=settings.MYSQL_DATABASE,
        )
    return _mysql_pool


# ============================================================
# Elasticsearch
# ============================================================

# §10-8 Nori 한국어 분석기 + movies_bm25 인덱스 설정
ES_INDEX_NAME = "movies_bm25"

ES_INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "30s",
        "analysis": {
            "tokenizer": {
                "nori_tokenizer": {
                    "type": "nori_tokenizer",
                    "decompound_mode": "mixed",
                }
            },
            "analyzer": {
                "korean_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": [
                        "nori_readingform",
                        "nori_part_of_speech",
                        "lowercase",
                    ],
                }
            },
            "filter": {
                "nori_part_of_speech": {
                    "type": "nori_part_of_speech",
                    "stoptags": [
                        "E", "IC", "J", "MAG", "MAJ", "MM",
                        "SP", "SSC", "SSO", "SC", "SE",
                        "XPN", "XSA", "XSN", "XSV",
                        "UNA", "NA", "VSV",
                    ],
                }
            },
        },
    },
    # ES 8.x에서 boost는 매핑이 아닌 쿼리 시 적용 (§10-8)
    # 검색 쿼리에서 title^3, director^2.5, title_en^2, cast^2, keywords^1.5 적용
    "mappings": {
        "properties": {
            # ── 기본 메타데이터 (text: 검색 대상) ──
            "id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "korean_analyzer"},
            "title_en": {"type": "text", "analyzer": "standard"},
            "director": {"type": "text", "analyzer": "korean_analyzer"},
            "overview": {"type": "text", "analyzer": "korean_analyzer"},
            "cast": {"type": "text", "analyzer": "korean_analyzer"},
            "keywords": {"type": "text", "analyzer": "korean_analyzer"},
            "genres": {"type": "keyword"},
            "mood_tags": {"type": "keyword"},
            "ott_platforms": {"type": "keyword"},
            "release_year": {"type": "integer"},
            "rating": {"type": "float"},
            "popularity_score": {"type": "float"},
            "runtime": {"type": "integer"},
            "poster_path": {"type": "keyword"},
            # ── Phase A: 리뷰/트레일러/관람등급 ──
            "reviews": {"type": "text", "analyzer": "korean_analyzer"},
            "certification": {"type": "keyword"},
            "trailer_url": {"type": "keyword"},
            "vote_count": {"type": "integer"},
            "behind_the_scenes": {"type": "keyword"},  # URL 목록
            "similar_movie_ids": {"type": "keyword"},  # ID 목록
            # ── KMDb 보강 필드 ──
            "awards": {"type": "text", "analyzer": "korean_analyzer"},
            "filming_location": {"type": "text", "analyzer": "korean_analyzer"},
            "audience_count": {"type": "long"},
            # ── Phase B: 텍스트 검색 필드 ──
            "tagline": {"type": "text", "analyzer": "korean_analyzer"},
            "collection_name": {"type": "text", "analyzer": "korean_analyzer"},
            "production_companies": {"type": "text", "analyzer": "standard"},
            "screenwriters": {"type": "text", "analyzer": "korean_analyzer"},
            "cinematographer": {"type": "text", "analyzer": "korean_analyzer"},
            "composer": {"type": "text", "analyzer": "korean_analyzer"},
            "producers": {"type": "text", "analyzer": "korean_analyzer"},
            "editor": {"type": "text", "analyzer": "korean_analyzer"},
            # ── Phase B: 필터링 필드 ──
            "original_language": {"type": "keyword"},
            "production_countries": {"type": "keyword"},
            "budget": {"type": "long"},
            "revenue": {"type": "long"},
            "adult": {"type": "boolean"},
            "status": {"type": "keyword"},
            "collection_id": {"type": "integer"},
            "imdb_id": {"type": "keyword"},
            "spoken_languages": {"type": "keyword"},
            "backdrop_path": {"type": "keyword"},
            "homepage": {"type": "keyword"},
            # ── Phase C: 완전 데이터 추출 ──
            "origin_country": {"type": "keyword"},
            "director_id": {"type": "integer"},
            "alternative_titles": {"type": "text", "analyzer": "korean_analyzer"},
            "recommendation_ids": {"type": "keyword"},
            "kr_release_date": {"type": "keyword"},
            "video_flag": {"type": "boolean"},
            "executive_producers": {"type": "text", "analyzer": "korean_analyzer"},
            "production_designer": {"type": "text", "analyzer": "korean_analyzer"},
            "costume_designer": {"type": "text", "analyzer": "korean_analyzer"},
            "source_author": {"type": "text", "analyzer": "korean_analyzer"},
            "production_country_names": {"type": "keyword"},
            "spoken_language_names": {"type": "keyword"},
            "cast_characters": {"type": "text", "analyzer": "korean_analyzer"},
            "embedding_text": {"type": "text", "analyzer": "korean_analyzer"},
            # ── KOBIS 보강 필드 ──
            "kobis_movie_cd": {"type": "keyword"},
            "sales_acc": {"type": "long"},
            "screen_count": {"type": "integer"},
            "kobis_genres": {"type": "keyword"},
            "kobis_nation": {"type": "keyword"},
            "kobis_watch_grade": {"type": "keyword"},
            "kobis_open_dt": {"type": "keyword"},
            "kobis_type_nm": {"type": "keyword"},
            "kobis_directors": {"type": "text", "analyzer": "korean_analyzer"},
            "kobis_actors": {"type": "text", "analyzer": "korean_analyzer"},
            "kobis_companies": {"type": "text", "analyzer": "korean_analyzer"},
            "kobis_staffs": {"type": "text", "analyzer": "korean_analyzer"},
            # ── 데이터 출처 ──
            "source": {"type": "keyword"},
        }
    },
}


async def get_elasticsearch() -> AsyncElasticsearch:
    """Elasticsearch 비동기 클라이언트를 반환한다 (싱글턴)."""
    global _es_client
    if _es_client is None:
        _es_client = AsyncElasticsearch(settings.ELASTICSEARCH_URL)
        logger.info("elasticsearch_client_initialized", url=settings.ELASTICSEARCH_URL)
    return _es_client


async def ensure_es_index() -> None:
    """
    Elasticsearch 'movies_bm25' 인덱스가 없으면 생성한다.

    §10-8: Nori 한국어 분석기 + 12개 필드 매핑
    """
    client = await get_elasticsearch()

    if not await client.indices.exists(index=ES_INDEX_NAME):
        await client.indices.create(
            index=ES_INDEX_NAME,
            body=ES_INDEX_SETTINGS,
        )
        logger.info("elasticsearch_index_created", index=ES_INDEX_NAME)


# ============================================================
# 전체 초기화 / 종료
# ============================================================

async def init_all_clients() -> None:
    """
    모든 DB 클라이언트를 초기화하고 컬렉션/인덱스를 설정한다.

    5개 DB를 개별 try/except로 격리하여, 하나의 DB 실패가 다른 DB 초기화를 막지 않도록 한다.
    각 DB별 초기화 소요 시간을 개별 측정하여 성능 병목을 파악할 수 있게 한다.
    """
    init_start = time.perf_counter()
    failed_dbs: list[str] = []

    # ── Qdrant 초기화 + 컬렉션/인덱스 설정 ──
    try:
        db_start = time.perf_counter()
        await get_qdrant()
        await ensure_qdrant_collection()
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        logger.info("qdrant_init_done", elapsed_ms=round(db_elapsed_ms, 1))
    except Exception as e:
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        failed_dbs.append("qdrant")
        logger.error(
            "qdrant_init_error", error=str(e), error_type=type(e).__name__,
            stack_trace=traceback.format_exc(), elapsed_ms=round(db_elapsed_ms, 1),
        )

    # ── Neo4j 초기화 + 인덱스/제약조건 설정 ──
    try:
        db_start = time.perf_counter()
        await get_neo4j()
        await ensure_neo4j_indexes()
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        logger.info("neo4j_init_done", elapsed_ms=round(db_elapsed_ms, 1))
    except Exception as e:
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        failed_dbs.append("neo4j")
        logger.error(
            "neo4j_init_error", error=str(e), error_type=type(e).__name__,
            stack_trace=traceback.format_exc(), elapsed_ms=round(db_elapsed_ms, 1),
        )

    # ── Redis 초기화 ──
    try:
        db_start = time.perf_counter()
        await get_redis()
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        logger.info("redis_init_done", elapsed_ms=round(db_elapsed_ms, 1))
    except Exception as e:
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        failed_dbs.append("redis")
        logger.error(
            "redis_init_error", error=str(e), error_type=type(e).__name__,
            stack_trace=traceback.format_exc(), elapsed_ms=round(db_elapsed_ms, 1),
        )

    # ── Elasticsearch 초기화 + 인덱스 설정 ──
    try:
        db_start = time.perf_counter()
        await get_elasticsearch()
        await ensure_es_index()
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        logger.info("elasticsearch_init_done", elapsed_ms=round(db_elapsed_ms, 1))
    except Exception as e:
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        failed_dbs.append("elasticsearch")
        logger.error(
            "elasticsearch_init_error", error=str(e), error_type=type(e).__name__,
            stack_trace=traceback.format_exc(), elapsed_ms=round(db_elapsed_ms, 1),
        )

    # ── MySQL 초기화 ──
    try:
        db_start = time.perf_counter()
        await get_mysql()
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        logger.info("mysql_init_done", elapsed_ms=round(db_elapsed_ms, 1))
    except Exception as e:
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        failed_dbs.append("mysql")
        logger.error(
            "mysql_init_error", error=str(e), error_type=type(e).__name__,
            stack_trace=traceback.format_exc(), elapsed_ms=round(db_elapsed_ms, 1),
        )

    # ── 전체 초기화 완료 요약 ──
    total_elapsed_ms = (time.perf_counter() - init_start) * 1000
    logger.info(
        "all_db_clients_initialized",
        total_elapsed_ms=round(total_elapsed_ms, 1),
        failed_count=len(failed_dbs),
        failed_dbs=failed_dbs if failed_dbs else None,
    )


async def close_all_clients() -> None:
    """
    모든 DB 클라이언트 연결을 정리한다.

    개별 DB 종료 실패가 다른 DB 종료를 막지 않도록 각각 try/except로 격리한다.
    """
    global _qdrant_client, _neo4j_driver, _redis_client, _es_client, _mysql_pool
    close_start = time.perf_counter()

    if _qdrant_client is not None:
        try:
            await _qdrant_client.close()
            logger.info("qdrant_client_closed")
        except Exception as e:
            logger.error("qdrant_close_error", error=str(e), error_type=type(e).__name__)
        _qdrant_client = None

    if _neo4j_driver is not None:
        try:
            await _neo4j_driver.close()
            logger.info("neo4j_driver_closed")
        except Exception as e:
            logger.error("neo4j_close_error", error=str(e), error_type=type(e).__name__)
        _neo4j_driver = None

    if _redis_client is not None:
        try:
            await _redis_client.close()
            logger.info("redis_client_closed")
        except Exception as e:
            logger.error("redis_close_error", error=str(e), error_type=type(e).__name__)
        _redis_client = None

    if _es_client is not None:
        try:
            await _es_client.close()
            logger.info("elasticsearch_client_closed")
        except Exception as e:
            logger.error("elasticsearch_close_error", error=str(e), error_type=type(e).__name__)
        _es_client = None

    if _mysql_pool is not None:
        try:
            _mysql_pool.close()
            await _mysql_pool.wait_closed()
            logger.info("mysql_pool_closed")
        except Exception as e:
            logger.error("mysql_close_error", error=str(e), error_type=type(e).__name__)
        _mysql_pool = None

    close_elapsed_ms = (time.perf_counter() - close_start) * 1000
    logger.info("all_db_clients_closed", elapsed_ms=round(close_elapsed_ms, 1))
