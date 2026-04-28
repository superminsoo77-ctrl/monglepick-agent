"""
DB 클라이언트 초기화 및 컬렉션/인덱스 설정.

§10-2-1 Qdrant 컬렉션, §10-8 Elasticsearch 인덱스 설정을 포함한다.
모든 클라이언트는 싱글턴 패턴으로 관리하며, FastAPI lifespan에서 초기화/종료한다.

[C-A1] asyncio 환경에서의 싱글턴 안전성:
asyncio에서는 여러 코루틴이 동시에 `if _xxx is None:` 체크와 할당 사이에
진입할 수 있어 경쟁 조건(Race Condition)이 발생한다.
`_get_init_lock()`을 사용한 Double-Checked Locking으로 이를 방지한다:
  1. Lock 진입 전 빠른 None 체크 (이미 초기화된 경우 Lock 진입 비용 없음)
  2. Lock 획득: 동시 초기화 직렬화
  3. 2차 체크: Lock 대기 중 다른 코루틴이 이미 초기화했을 경우 중복 생성 방지
  4. Lock 내부에서만 실제 초기화 수행

[C-A2] asyncio.Lock 지연 생성(Lazy Initialization) 이유:
Python에서 `asyncio.Lock()`을 모듈 임포트 시점(이벤트 루프 시작 전)에 생성하면
Python 3.10 미만에서는 Lock이 기본 이벤트 루프에 바인딩되어
`RuntimeError: Task got Future attached to a different loop` 가 발생한다.
Python 3.10+에서는 이 문제가 해결됐지만, 호환성과 안전성을 위해
`_get_init_lock()`을 통해 첫 호출 시점에 실행 중인 루프에서 Lock을 생성한다.

[C-A3] close_all_clients() 경쟁 조건:
종료 시에도 Lock을 획득한 후 None 재설정을 수행하여,
종료 중에 다른 코루틴이 get_*()를 호출해 재초기화하는 상황을 방지한다.
"""

from __future__ import annotations

import asyncio
import time
import traceback

import aiomysql
import structlog
from elasticsearch import AsyncElasticsearch
from neo4j import AsyncDriver, AsyncGraphDatabase
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

# ── [C-A1, C-A2] DB 클라이언트 초기화 전용 Lock (지연 생성) ──
# asyncio.Lock()을 모듈 임포트 시점에 생성하면 이벤트 루프 바인딩 문제가 발생한다.
# _get_init_lock()을 통해 첫 호출 시점(이벤트 루프 실행 중)에 Lock을 생성한다.
# asyncio.Lock은 단일 이벤트 루프 스레드 안에서만 안전하며,
# 스레드 간 공유가 필요한 경우 threading.Lock을 별도로 사용해야 한다.
_init_lock: asyncio.Lock | None = None


def _get_init_lock() -> asyncio.Lock:
    """
    [C-A2] 초기화 Lock을 지연 생성하여 반환한다.

    asyncio.Lock()은 이벤트 루프가 실행 중인 상태에서 생성해야 한다.
    모듈 임포트 시점이 아닌 첫 get_*() 호출 시점(이벤트 루프 실행 중)에 생성한다.
    이미 생성된 경우 재사용한다.

    주의: 이 함수 자체의 호출은 asyncio 환경에서 동일한 이벤트 루프 스레드 내에서
    이뤄지므로, Lock 인스턴스 생성(`asyncio.Lock()`) 단계는 GIL에 의해 원자적으로
    처리된다. 따라서 Lock 생성 자체에 대한 별도 동기화는 불필요하다.
    """
    global _init_lock
    if _init_lock is None:
        # asyncio.Lock() 생성은 CPython GIL 범위 내 단일 바이트코드 연산이므로
        # 이 시점에서의 경쟁 조건은 발생하지 않는다. 동일 이벤트 루프에서
        # 여러 코루틴이 동시에 여기 도달하더라도 한 번만 생성된다.
        _init_lock = asyncio.Lock()
    return _init_lock


# ============================================================
# Qdrant
# ============================================================

async def get_qdrant() -> AsyncQdrantClient:
    """
    Qdrant 비동기 클라이언트를 반환한다 (싱글턴, [C-A1][C-A2] asyncio-safe).

    Double-Checked Locking 패턴:
    - 1차 체크: Lock 진입 전 빠른 반환 (이미 초기화된 경우 비용 없음)
    - Lock 획득: _get_init_lock()을 통해 현재 이벤트 루프에 바인딩된 Lock 사용
    - 2차 체크: Lock 대기 중 다른 코루틴이 이미 초기화했을 경우 중복 생성 방지
    """
    global _qdrant_client

    # 1차 체크: 이미 초기화된 경우 Lock 없이 즉시 반환
    if _qdrant_client is not None:
        return _qdrant_client

    # [C-A2] _get_init_lock()으로 이벤트 루프 바인딩 문제 방지
    async with _get_init_lock():
        # 2차 체크: Lock 대기 중 다른 코루틴이 초기화했을 수 있음 (double-check)
        if _qdrant_client is not None:
            return _qdrant_client

        _qdrant_client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            check_compatibility=False,  # 클라이언트/서버 마이너 버전 차이 허용
            timeout=10,  # 요청 타임아웃 10초 (unhealthy 시 빠른 실패)
        )
        logger.info("qdrant_client_initialized", url=settings.QDRANT_URL)

    return _qdrant_client


async def ensure_qdrant_collection() -> None:
    """
    Qdrant 'movies' 컬렉션이 없으면 생성하고 payload 인덱스를 설정한다.

    §10-2-1 설정:
    - 벡터 크기: 4096 (Upstage Solar embedding-passage)
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
                size=settings.EMBEDDING_DIMENSION,  # 4096 (Upstage Solar)
                distance=Distance.COSINE,
                on_disk=True,  # 벡터를 디스크(mmap)에 저장 (117만건 × 4096차원 메모리 절감)
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
    """
    Neo4j 비동기 드라이버를 반환한다 (싱글턴, [C-A1][C-A2] asyncio-safe).

    Double-Checked Locking 패턴으로 경쟁 조건을 방지한다.
    _get_init_lock()으로 이벤트 루프 바인딩 문제를 방지한다.
    """
    global _neo4j_driver

    # 1차 체크: 이미 초기화된 경우 Lock 없이 즉시 반환
    if _neo4j_driver is not None:
        return _neo4j_driver

    # [C-A2] _get_init_lock()으로 이벤트 루프 바인딩 문제 방지
    async with _get_init_lock():
        # 2차 체크: Lock 대기 중 다른 코루틴이 초기화했을 수 있음 (double-check)
        if _neo4j_driver is not None:
            return _neo4j_driver

        _neo4j_driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            connection_timeout=5,               # 연결 수립 타임아웃 5초
            connection_acquisition_timeout=10,   # 풀에서 연결 획득 타임아웃 10초
            max_connection_lifetime=300,          # 연결 최대 수명 5분
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
    """
    Redis 비동기 클라이언트를 반환한다 (싱글턴, [C-A1][C-A2] asyncio-safe).

    Double-Checked Locking 패턴으로 경쟁 조건을 방지한다.
    _get_init_lock()으로 이벤트 루프 바인딩 문제를 방지한다.
    """
    global _redis_client

    # 1차 체크: 이미 초기화된 경우 Lock 없이 즉시 반환
    if _redis_client is not None:
        return _redis_client

    # [C-A2] _get_init_lock()으로 이벤트 루프 바인딩 문제 방지
    async with _get_init_lock():
        # 2차 체크: Lock 대기 중 다른 코루틴이 초기화했을 수 있음 (double-check)
        if _redis_client is not None:
            return _redis_client

        # ConnectionPool을 명시적으로 생성하여 최대 연결 수를 환경변수로 제어한다.
        # REDIS_MAX_CONNECTIONS 미만의 연결은 자동으로 관리되며,
        # 초과 시 연결 대기 후 타임아웃이 발생한다.
        from redis.asyncio import ConnectionPool
        pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
        )
        _redis_client = Redis(connection_pool=pool)
        logger.info(
            "redis_client_initialized",
            url=settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
        )

    return _redis_client


# ============================================================
# MySQL
# ============================================================

async def get_mysql() -> aiomysql.Pool:
    """
    MySQL 비동기 커넥션 풀을 반환한다 (싱글턴, [C-A1][C-A2] asyncio-safe).

    aiomysql.create_pool()로 커넥션 풀을 생성하며, 최소 1개 / 최대 10개 연결을 유지한다.
    charset은 utf8mb4로 설정하여 한국어/이모지를 지원한다.
    Double-Checked Locking 패턴으로 경쟁 조건을 방지한다.
    _get_init_lock()으로 이벤트 루프 바인딩 문제를 방지한다.
    """
    global _mysql_pool

    # 1차 체크: 이미 초기화된 경우 Lock 없이 즉시 반환
    if _mysql_pool is not None:
        return _mysql_pool

    # [C-A2] _get_init_lock()으로 이벤트 루프 바인딩 문제 방지
    async with _get_init_lock():
        # 2차 체크: Lock 대기 중 다른 코루틴이 초기화했을 수 있음 (double-check)
        if _mysql_pool is not None:
            return _mysql_pool

        # minsize/maxsize를 환경변수(MYSQL_POOL_MIN/MYSQL_POOL_MAX)로 제어하여
        # 운영 환경(maxsize 큼)과 개발 환경(maxsize 작음)을 분리한다.
        _mysql_pool = await aiomysql.create_pool(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            db=settings.MYSQL_DATABASE,
            charset="utf8mb4",
            minsize=settings.MYSQL_POOL_MIN,
            maxsize=settings.MYSQL_POOL_MAX,
            autocommit=True,
        )
        logger.info(
            "mysql_pool_initialized",
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            db=settings.MYSQL_DATABASE,
            pool_min=settings.MYSQL_POOL_MIN,
            pool_max=settings.MYSQL_POOL_MAX,
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
            # Phase ML (다국어 검색 개선): 대체 제목에 영문/일문 등 다국어 혼재
            # → standard analyzer 기본 + .korean 서브필드로 한국어 분석 병행
            "alternative_titles": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "korean": {"type": "text", "analyzer": "korean_analyzer"}
                }
            },
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
            # ── Phase D: 전체 수집 보강 필드 ──
            # 다국어 줄거리 (번역 텍스트 검색 지원)
            "overview_en": {"type": "text", "analyzer": "standard"},
            "overview_ja": {"type": "text", "analyzer": "standard"},
            # 소셜 미디어 ID (외부 연동용 필터링)
            "facebook_id": {"type": "keyword"},
            "instagram_id": {"type": "keyword"},
            "twitter_id": {"type": "keyword"},
            "wikidata_id": {"type": "keyword"},
            # TMDB 리스트 포함 수 (인기도 보조 지표)
            "tmdb_list_count": {"type": "integer"},
            # 로고 이미지 경로 (UI 표시용)
            "images_logos": {"type": "keyword"},
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
    """
    Elasticsearch 비동기 클라이언트를 반환한다 (싱글턴, [C-A1][C-A2] asyncio-safe).

    Double-Checked Locking 패턴으로 경쟁 조건을 방지한다.
    _get_init_lock()으로 이벤트 루프 바인딩 문제를 방지한다.
    """
    global _es_client

    # 1차 체크: 이미 초기화된 경우 Lock 없이 즉시 반환
    if _es_client is not None:
        return _es_client

    # [C-A2] _get_init_lock()으로 이벤트 루프 바인딩 문제 방지
    async with _get_init_lock():
        # 2차 체크: Lock 대기 중 다른 코루틴이 초기화했을 수 있음 (double-check)
        if _es_client is not None:
            return _es_client

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
# 고객센터 정책 RAG 컬렉션 (support_policy_v1)
# ============================================================

#: 고객센터 정책 RAG Qdrant 컬렉션명.
#: 'movies', 'admin_tool_registry' 와 동일 인스턴스 내에서 네임스페이스 분리.
#: 설계서: docs/고객센터_AI에이전트_v4_재설계.md §6.1
SUPPORT_POLICY_COLLECTION: str = "support_policy_v1"


async def ensure_support_policy_collection() -> None:
    """
    Qdrant `support_policy_v1` 컬렉션이 없으면 생성하고 payload 인덱스를 설정한다.

    고객센터 AI 봇(v4)의 정책 RAG 인프라 초기화 함수.
    Agent 시작 시 `init_all_clients()` 에서 자동 호출된다.

    설정 (docs/고객센터_AI에이전트_v4_재설계.md §6.1):
    - 벡터 크기: 4096 (settings.EMBEDDING_DIMENSION — Upstage Solar)
    - 거리 메트릭: Cosine
    - HNSW: M=16, ef_construct=100
    - on_disk=False — 정책 문서 청크는 수백~수천 개 수준으로 메모리 보관이 적절

    Payload 인덱스:
    - `policy_topic` (KEYWORD) — grade_benefit / ai_quota / subscription / refund / reward / payment / general
      필터 검색 가속: "BRONZE 등급 AI 한도" 질문 시 policy_topic="grade_benefit" 로 범위 한정
    - `doc_id` (KEYWORD) — 문서 단위 삭제·재인덱싱 시 must 필터로 기존 청크 일괄 삭제
    - `doc_path` (KEYWORD) — 출처 문서 경로 (디버깅/감사 추적용)

    이 함수는 멱등적이다 — 이미 존재하는 컬렉션이나 payload 인덱스에는 영향 없음.
    기존 'movies' / 'admin_tool_registry' 컬렉션은 절대 건드리지 않는다.

    Raises:
        Exception: Qdrant 연결 실패 또는 API 오류 (호출자인 init_all_clients 가 try/except 로 격리)
    """
    client = await get_qdrant()

    # ── 컬렉션 존재 여부 확인 ──
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]

    if SUPPORT_POLICY_COLLECTION not in existing_names:
        # 신규 생성 — 영화 컬렉션과 동일 벡터 설정, on_disk=False (소규모 컬렉션)
        await client.create_collection(
            collection_name=SUPPORT_POLICY_COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,  # 4096 (Upstage Solar)
                distance=Distance.COSINE,
                on_disk=False,  # 정책 청크는 수백~수천 개 — 메모리 보관이 빠름
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )
        logger.info(
            "support_policy_collection_created",
            name=SUPPORT_POLICY_COLLECTION,
            dim=settings.EMBEDDING_DIMENSION,
        )

    # ── Payload 인덱스 설정 (이미 있으면 idempotent) ──
    # Qdrant 는 인덱스가 이미 존재해도 에러를 던지는 경우가 있으므로 개별 try/except 처리.
    keyword_fields = [
        "policy_topic",  # grade_benefit / ai_quota / subscription / refund / reward / payment / general
        "doc_id",        # 문서 단위 필터링 (재인덱싱 시 기존 청크 삭제 용도)
        "doc_path",      # 출처 문서 경로 (감사 추적)
    ]
    for field in keyword_fields:
        try:
            await client.create_payload_index(
                collection_name=SUPPORT_POLICY_COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            # 이미 존재하거나 Qdrant 버전 차이로 에러가 날 수 있지만 운영상 무해.
            logger.debug(
                "support_policy_payload_index_skip",
                field=field,
                reason=str(e),
            )

    logger.info(
        "support_policy_collection_ready",
        collection=SUPPORT_POLICY_COLLECTION,
        existed=SUPPORT_POLICY_COLLECTION in existing_names,
    )


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

    # ── 고객센터 정책 RAG 컬렉션 부트스트랩 ──
    # Qdrant 클라이언트는 이미 위에서 초기화됐으므로 get_qdrant() 는 캐시 반환.
    # 'movies' / 'admin_tool_registry' 컬렉션과 독립적으로 격리 초기화한다.
    # 실패해도 채팅/추천 기능에 영향 없음 (고객센터 v4 기능만 degraded).
    try:
        db_start = time.perf_counter()
        await ensure_support_policy_collection()
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        logger.info("support_policy_collection_init_done", elapsed_ms=round(db_elapsed_ms, 1))
    except Exception as e:
        db_elapsed_ms = (time.perf_counter() - db_start) * 1000
        failed_dbs.append("support_policy_qdrant")
        logger.error(
            "support_policy_collection_init_error", error=str(e), error_type=type(e).__name__,
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

    [C-A3] Lock을 획득한 상태에서 클라이언트 종료 및 None 재설정을 수행한다.
    Lock 없이 None을 재설정하면, 종료 도중 다른 코루틴이 get_*()를 호출하여
    클라이언트를 재초기화한 뒤 곧바로 None으로 덮어쓰이는 경쟁 조건이 발생할 수 있다.

    개별 DB 종료 실패가 다른 DB 종료를 막지 않도록 각각 try/except로 격리한다.
    """
    global _qdrant_client, _neo4j_driver, _redis_client, _es_client, _mysql_pool
    close_start = time.perf_counter()

    # [C-A3] 종료 시에도 Lock을 획득하여 get_*()와 상호 배제 보장
    async with _get_init_lock():

        if _qdrant_client is not None:
            try:
                await _qdrant_client.close()
                logger.info("qdrant_client_closed")
            except Exception as e:
                logger.error("qdrant_close_error", error=str(e), error_type=type(e).__name__)
            finally:
                # 종료 성공/실패 여부와 관계없이 None으로 재설정하여 재사용 차단
                _qdrant_client = None

        if _neo4j_driver is not None:
            try:
                await _neo4j_driver.close()
                logger.info("neo4j_driver_closed")
            except Exception as e:
                logger.error("neo4j_close_error", error=str(e), error_type=type(e).__name__)
            finally:
                _neo4j_driver = None

        if _redis_client is not None:
            try:
                await _redis_client.close()
                logger.info("redis_client_closed")
            except Exception as e:
                logger.error("redis_close_error", error=str(e), error_type=type(e).__name__)
            finally:
                _redis_client = None

        if _es_client is not None:
            try:
                await _es_client.close()
                logger.info("elasticsearch_client_closed")
            except Exception as e:
                logger.error("elasticsearch_close_error", error=str(e), error_type=type(e).__name__)
            finally:
                _es_client = None

        if _mysql_pool is not None:
            try:
                _mysql_pool.close()
                await _mysql_pool.wait_closed()
                logger.info("mysql_pool_closed")
            except Exception as e:
                logger.error("mysql_close_error", error=str(e), error_type=type(e).__name__)
            finally:
                _mysql_pool = None

    close_elapsed_ms = (time.perf_counter() - close_start) * 1000
    logger.info("all_db_clients_closed", elapsed_ms=round(close_elapsed_ms, 1))
