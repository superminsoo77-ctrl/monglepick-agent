"""
관리자 전용 API 라우터.

시스템 탭:
- GET /api/v1/admin/system/db — 5개 DB(MySQL/Qdrant/Neo4j/ES/Redis) 상태 조회
- GET /api/v1/admin/system/ollama — Ollama 모델 로드 상태 조회
"""

import time
import traceback

import httpx
import structlog
from fastapi import APIRouter

from monglepick.config import settings
from monglepick.db.clients import (
    get_elasticsearch,
    get_mysql,
    get_neo4j,
    get_qdrant,
    get_redis,
)

logger = structlog.get_logger()

admin_router = APIRouter(prefix="/admin", tags=["admin"])


# ============================================================
# GET /admin/system/db — 5개 DB 상태 조회
# ============================================================

async def _check_qdrant() -> dict:
    """Qdrant 연결 상태 + 벡터/세그먼트 수 조회."""
    try:
        client = await get_qdrant()
        info = await client.get_collection(settings.QDRANT_COLLECTION)
        return {
            "connected": True,
            "vectorCount": info.points_count,
            "segmentCount": info.segments_count,
            "status": str(info.status),
        }
    except Exception as e:
        logger.warning("admin_db_check_qdrant_failed", error=str(e))
        return {"connected": False, "error": str(e)}


async def _check_neo4j() -> dict:
    """Neo4j 연결 상태 + 노드/관계 수 조회."""
    try:
        driver = await get_neo4j()
        async with driver.session() as session:
            # 노드 수
            result = await session.run("MATCH (n) RETURN count(n) AS cnt")
            record = await result.single()
            node_count = record["cnt"] if record else 0

            # 관계 수
            result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            record = await result.single()
            rel_count = record["cnt"] if record else 0

        return {
            "connected": True,
            "nodeCount": node_count,
            "relationshipCount": rel_count,
        }
    except Exception as e:
        logger.warning("admin_db_check_neo4j_failed", error=str(e))
        return {"connected": False, "error": str(e)}


async def _check_elasticsearch() -> dict:
    """Elasticsearch 클러스터 상태 + 문서 수 조회."""
    try:
        es = await get_elasticsearch()
        # 클러스터 헬스
        health = await es.cluster.health()
        cluster_status = health.get("status", "unknown")

        # movies 인덱스 문서 수
        stats = await es.indices.stats(index="movies_bm25")
        doc_count = stats["_all"]["primaries"]["docs"]["count"]
        index_size = stats["_all"]["primaries"]["store"]["size_in_bytes"]
        size_mb = f"{index_size / (1024 * 1024):.1f}MB"

        return {
            "connected": True,
            "clusterStatus": cluster_status,
            "documentCount": doc_count,
            "indexSize": size_mb,
        }
    except Exception as e:
        logger.warning("admin_db_check_es_failed", error=str(e))
        return {"connected": False, "error": str(e)}


async def _check_redis() -> dict:
    """Redis 연결 상태 + 키 수/메모리 조회."""
    try:
        redis = await get_redis()
        info = await redis.info("memory")
        keyspace = await redis.info("keyspace")
        stats = await redis.info("stats")

        # 키 수 합산
        key_count = 0
        for db_info in keyspace.values():
            if isinstance(db_info, dict):
                key_count += db_info.get("keys", 0)

        memory_used = info.get("used_memory_human", "?")
        memory_max = info.get("maxmemory_human", "?")

        # 히트율 계산
        hits = stats.get("keyspace_hits", 0)
        misses = stats.get("keyspace_misses", 0)
        hit_rate = round(hits / (hits + misses) * 100, 1) if (hits + misses) > 0 else 0

        return {
            "connected": True,
            "keyCount": key_count,
            "memoryUsed": memory_used,
            "memoryMax": memory_max if memory_max != "0B" else "제한 없음",
            "hitRate": hit_rate,
        }
    except Exception as e:
        logger.warning("admin_db_check_redis_failed", error=str(e))
        return {"connected": False, "error": str(e)}


async def _check_mysql() -> dict:
    """MySQL 연결 상태 + 활성 커넥션 조회."""
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # movies 테이블 행 수
                await cur.execute("SELECT COUNT(*) FROM movies")
                row = await cur.fetchone()
                movie_count = row[0] if row else 0

                # 활성 커넥션 수
                await cur.execute("SHOW STATUS LIKE 'Threads_connected'")
                row = await cur.fetchone()
                active_connections = int(row[1]) if row else 0

        return {
            "connected": True,
            "movieCount": movie_count,
            "activeConnections": active_connections,
        }
    except Exception as e:
        logger.warning("admin_db_check_mysql_failed", error=str(e))
        return {"connected": False, "error": str(e)}


@admin_router.get(
    "/system/db",
    summary="5개 DB 상태 조회",
    description="MySQL, Qdrant, Neo4j, Elasticsearch, Redis의 연결 상태 및 주요 지표를 반환한다.",
)
async def get_db_status():
    """5개 DB의 연결 상태와 주요 지표를 조회한다."""
    import asyncio

    # 5개 DB를 병렬로 체크
    results = await asyncio.gather(
        _check_mysql(),
        _check_qdrant(),
        _check_neo4j(),
        _check_elasticsearch(),
        _check_redis(),
        return_exceptions=True,
    )

    # 예외가 발생한 경우 처리
    def safe_result(r):
        if isinstance(r, Exception):
            return {"connected": False, "error": str(r)}
        return r

    return {
        "mysql": safe_result(results[0]),
        "qdrant": safe_result(results[1]),
        "neo4j": safe_result(results[2]),
        "elasticsearch": safe_result(results[3]),
        "redis": safe_result(results[4]),
    }


# ============================================================
# GET /admin/system/ollama — Ollama 모델 상태 조회
# ============================================================

@admin_router.get(
    "/system/ollama",
    summary="Ollama 모델 상태 조회",
    description="Ollama 서버 연결 상태, 로드된 모델, VRAM 사용량을 반환한다.",
)
async def get_ollama_status():
    """Ollama REST API를 호출하여 모델 로드 상태를 조회한다."""
    ollama_url = getattr(settings, "OLLAMA_URL", "http://localhost:11434")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # 서버 버전 확인
            version_resp = await client.get(f"{ollama_url}/api/version")
            version = version_resp.json().get("version", "unknown") if version_resp.status_code == 200 else "unknown"

            # 현재 로드된 모델 목록
            ps_resp = await client.get(f"{ollama_url}/api/ps")
            loaded_models = []
            if ps_resp.status_code == 200:
                ps_data = ps_resp.json()
                for model in ps_data.get("models", []):
                    loaded_models.append({
                        "name": model.get("name", ""),
                        "loaded": True,
                        "vram": _format_bytes(model.get("size_vram", 0)),
                        "lastUsed": model.get("expires_at", ""),
                    })

            # 사용 가능한 전체 모델 목록
            tags_resp = await client.get(f"{ollama_url}/api/tags")
            all_models = []
            if tags_resp.status_code == 200:
                for model in tags_resp.json().get("models", []):
                    all_models.append(model.get("name", ""))

        # 로드되지 않은 모델 추가
        loaded_names = {m["name"] for m in loaded_models}
        for name in all_models:
            if name not in loaded_names:
                loaded_models.append({"name": name, "loaded": False, "vram": None, "lastUsed": None})

        return {
            "connected": True,
            "version": version,
            "maxLoadedModels": getattr(settings, "OLLAMA_MAX_LOADED_MODELS", 2),
            "models": loaded_models,
            "queue": {"waiting": 0, "processing": len([m for m in loaded_models if m["loaded"]])},
        }

    except Exception as e:
        logger.warning("admin_ollama_check_failed", error=str(e))
        return {
            "connected": False,
            "version": None,
            "models": [],
            "error": str(e),
        }


def _format_bytes(n: int) -> str:
    """바이트를 사람 읽기 편한 형식으로 변환."""
    if n == 0:
        return "0B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"
