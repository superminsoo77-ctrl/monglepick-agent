"""
관리자 데이터 관리 API 라우터.

설계서: docs/관리자페이지_설계서.md §3.6 데이터 관리(15 API)
담당:   윤형주

15개 엔드포인트:
  데이터 현황 (3):
    GET  /admin/data/overview        — 5DB 건수 + 인덱스 정보 카드
    GET  /admin/data/distribution    — 소스별 데이터 분포 (TMDB/Kaggle/KOBIS/KMDb)
    GET  /admin/data/quality         — 중복률/NULL 률 등 품질 지표
  영화 CRUD (3):
    GET  /admin/movies               — 검색/필터/페이징 (5DB 통합)
    GET  /admin/movies/{id}          — 단건 상세 (모든 필드)
    PUT  /admin/movies/{id}          — 수정 (MySQL + Qdrant + Neo4j + ES 동기 반영)
    DELETE /admin/movies/{id}        — 삭제 (4DB 동기 삭제)
  파이프라인 (8):
    GET  /admin/pipeline             — 9개 작업 목록
    POST /admin/pipeline/run         — subprocess 실행 (백그라운드)
    POST /admin/pipeline/cancel      — 실행 중인 작업 취소 (체크포인트 저장)
    GET  /admin/pipeline/logs        — SSE 실시간 로그 스트리밍
    GET  /admin/pipeline/history     — 실행 이력 페이징
    GET  /admin/pipeline/stats       — 성공/실패 통계
    POST /admin/pipeline/retry-failed — 실패 건 재시도
    GET  /admin/collection/{name}/status — Qdrant 컬렉션 상태

설계 결정:
  - 인증: 최상위 main.py 의 ServiceKey 미들웨어 또는 Spring Boot 게이트웨이가 보호.
    Agent 자체에서는 별도 admin role 검증을 하지 않으나, 운영 환경에서는 Nginx
    또는 Spring Boot 프록시 단에서 ADMIN role 헤더를 검증한다.
  - 파이프라인 실행: subprocess.Popen 으로 비차단 실행 + 작업 상태를 메모리 dict 로 추적.
    Agent 재시작 시 상태가 사라지므로, 영구 이력은 logs/pipeline_history.jsonl 에 append.
  - SSE 로그: 작업의 stdout 을 실시간으로 클라이언트에 전송. EventSourceResponse 사용.
  - 5DB 동기 반영: MySQL 트랜잭션 + Qdrant/Neo4j/ES 베스트-에포트(실패 시 로그만).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from monglepick.config import settings
from monglepick.db.clients import (
    ES_INDEX_NAME,
    get_elasticsearch,
    get_mysql,
    get_neo4j,
    get_qdrant,
    get_redis,
)

logger = structlog.get_logger()

# 라우터: admin_router 와 동일 prefix /admin 사용
admin_data_router = APIRouter(prefix="/admin", tags=["admin-data"])


# ============================================================
# 모듈 레벨 상태 — 파이프라인 작업 추적
# ============================================================

# 사용 가능한 파이프라인 작업 정의 (설계서 §3.6 — 9개 작업)
# key: 작업 코드 (불변), value: 메타데이터 + 실행 명령
PIPELINE_TASKS: dict[str, dict[str, Any]] = {
    "tmdb_collect": {
        "name": "TMDB 영화 기본정보 수집",
        "description": "TMDB API 로 영화 메타데이터를 수집하여 JSONL 로 저장한다.",
        "script": "scripts/run_tmdb_full_collection.py",
        "category": "collection",
    },
    "kaggle_supplement": {
        "name": "Kaggle 데이터 보강",
        "description": "Kaggle 26M 시청 이력 데이터로 CF 학습 데이터를 보강한다.",
        "script": "scripts/run_kaggle_supplement.py",
        "category": "collection",
    },
    "kobis_load": {
        "name": "KOBIS 한국영화 적재",
        "description": "KOBIS API 로 한국영화 정보를 적재한다.",
        "script": "scripts/run_kobis_load.py",
        "category": "collection",
    },
    "kmdb_load": {
        "name": "KMDb 영화 메타데이터 적재",
        "description": "KMDb API 로 한국영화 상세 메타데이터를 적재한다.",
        "script": "scripts/run_kmdb_load.py",
        "category": "collection",
    },
    "mood_enrichment": {
        "name": "무드 태그 보강",
        "description": "Ollama 또는 Solar 로 영화 무드 태그를 생성/보강한다.",
        "script": "scripts/run_mood_enrichment.py",
        "category": "enrichment",
    },
    "full_reload": {
        "name": "전체 재적재 (TMDB + Kaggle + CF)",
        "description": "1.17M 건 TMDB JSONL + Kaggle 보강 + CF 재구축. ~6~10시간 소요.",
        "script": "scripts/run_full_reload.py",
        "category": "load",
    },
    "es_sync": {
        "name": "Elasticsearch 동기화",
        "description": "MySQL → Elasticsearch 인덱스 동기화 (Nori 한국어 분석기).",
        "script": "scripts/run_es_sync.py",
        "category": "load",
    },
    "mysql_sync": {
        "name": "MySQL 동기화",
        "description": "외부 소스 → MySQL movies 테이블 동기화.",
        "script": "scripts/run_mysql_sync.py",
        "category": "load",
    },
    "cf_only": {
        "name": "CF 매트릭스 재구축",
        "description": "Kaggle 시청 이력 기반 CF 매트릭스를 재구축하여 Redis 캐시.",
        "script": "scripts/run_cf_only.py",
        "category": "rebuild",
    },
}

# 실행 중/완료된 파이프라인 작업 상태 (메모리)
# key: job_id (UUID), value: {task_code, status, started_at, ended_at, exit_code, log_lines, process}
# status: PENDING / RUNNING / SUCCESS / FAILED / CANCELLED
PIPELINE_JOBS: dict[str, dict[str, Any]] = {}

# 작업별 출력 로그 큐 (SSE 스트리밍용)
# key: job_id, value: asyncio.Queue (stdout 라인 단위)
PIPELINE_LOG_QUEUES: dict[str, asyncio.Queue] = {}

# 영구 이력 파일 경로 (Agent 재시작 후에도 보존)
PIPELINE_HISTORY_FILE = Path("data/pipeline_history.jsonl")
PIPELINE_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Pydantic 모델
# ============================================================

class PipelineRunRequest(BaseModel):
    """파이프라인 실행 요청."""

    task_code: str = Field(..., description="실행할 작업 코드 (PIPELINE_TASKS 키)")
    args: list[str] = Field(
        default_factory=list,
        description="스크립트에 전달할 추가 인자 (예: ['--clear-db', '--chunk-size', '5000'])",
    )


class PipelineRunResponse(BaseModel):
    """파이프라인 실행 응답."""

    job_id: str
    task_code: str
    status: str
    started_at: str
    message: str


class PipelineCancelRequest(BaseModel):
    """파이프라인 취소 요청."""

    job_id: str = Field(..., description="취소할 작업 ID")


class MovieUpdateRequest(BaseModel):
    """영화 수정 요청 — 부분 업데이트 지원."""

    title: Optional[str] = None
    title_en: Optional[str] = None
    overview: Optional[str] = None
    genres: Optional[str] = Field(None, description="JSON 배열 문자열 (예: '[\"액션\", \"SF\"]')")
    release_year: Optional[int] = None
    rating: Optional[float] = None
    runtime: Optional[int] = None
    director: Optional[str] = None
    poster_url: Optional[str] = None


# ============================================================
# 1. 데이터 현황 (3 EP)
# ============================================================

@admin_data_router.get(
    "/data/overview",
    summary="5DB 데이터 건수 카드",
    description="MySQL/Qdrant/Neo4j/Elasticsearch/Redis 의 핵심 건수와 인덱스 정보를 반환한다.",
)
async def get_data_overview() -> dict:
    """5DB 의 영화/벡터/그래프/문서/캐시 건수를 병렬 조회한다."""

    async def _mysql_count() -> dict:
        try:
            pool = await get_mysql()
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT COUNT(*) FROM movies")
                    row = await cur.fetchone()
                    movie_count = row[0] if row else 0

                    # users 테이블도 확인 (운영 시 유저 수 확인용)
                    try:
                        await cur.execute("SELECT COUNT(*) FROM users")
                        row = await cur.fetchone()
                        user_count = row[0] if row else 0
                    except Exception:
                        user_count = None

            return {"name": "MySQL", "movieCount": movie_count, "userCount": user_count}
        except Exception as e:
            logger.warning("data_overview_mysql_failed", error=str(e))
            return {"name": "MySQL", "error": str(e)}

    async def _qdrant_count() -> dict:
        try:
            client = await get_qdrant()
            info = await client.get_collection(settings.QDRANT_COLLECTION)
            return {
                "name": "Qdrant",
                "vectorCount": info.points_count,
                "collection": settings.QDRANT_COLLECTION,
                "vectorSize": info.config.params.vectors.size if hasattr(info.config.params, "vectors") else None,
            }
        except Exception as e:
            logger.warning("data_overview_qdrant_failed", error=str(e))
            return {"name": "Qdrant", "error": str(e)}

    async def _neo4j_count() -> dict:
        try:
            driver = await get_neo4j()
            async with driver.session() as session:
                # 노드/관계 수 한 번에 조회
                result = await session.run(
                    "MATCH (n) "
                    "OPTIONAL MATCH ()-[r]->() "
                    "RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS rels"
                )
                record = await result.single()
                return {
                    "name": "Neo4j",
                    "nodeCount": record["nodes"] if record else 0,
                    "relationshipCount": record["rels"] if record else 0,
                }
        except Exception as e:
            logger.warning("data_overview_neo4j_failed", error=str(e))
            return {"name": "Neo4j", "error": str(e)}

    async def _es_count() -> dict:
        try:
            es = await get_elasticsearch()
            stats = await es.indices.stats(index=ES_INDEX_NAME)
            doc_count = stats["_all"]["primaries"]["docs"]["count"]
            size_bytes = stats["_all"]["primaries"]["store"]["size_in_bytes"]
            return {
                "name": "Elasticsearch",
                "documentCount": doc_count,
                "indexName": ES_INDEX_NAME,
                "indexSizeMB": round(size_bytes / (1024 * 1024), 2),
            }
        except Exception as e:
            logger.warning("data_overview_es_failed", error=str(e))
            return {"name": "Elasticsearch", "error": str(e)}

    async def _redis_count() -> dict:
        try:
            redis = await get_redis()
            keyspace = await redis.info("keyspace")
            key_count = 0
            for db_info in keyspace.values():
                if isinstance(db_info, dict):
                    key_count += db_info.get("keys", 0)
            return {"name": "Redis", "keyCount": key_count}
        except Exception as e:
            logger.warning("data_overview_redis_failed", error=str(e))
            return {"name": "Redis", "error": str(e)}

    results = await asyncio.gather(
        _mysql_count(), _qdrant_count(), _neo4j_count(), _es_count(), _redis_count(),
        return_exceptions=True,
    )

    def _safe(r) -> dict:
        if isinstance(r, Exception):
            return {"error": str(r)}
        return r

    return {
        "mysql": _safe(results[0]),
        "qdrant": _safe(results[1]),
        "neo4j": _safe(results[2]),
        "elasticsearch": _safe(results[3]),
        "redis": _safe(results[4]),
        "checkedAt": datetime.utcnow().isoformat() + "Z",
    }


@admin_data_router.get(
    "/data/distribution",
    summary="소스별 데이터 분포",
    description="movies 테이블의 source 컬럼을 GROUP BY 하여 소스별 영화 수를 반환한다.",
)
async def get_data_distribution() -> dict:
    """소스별 데이터 분포 (TMDB/Kaggle/KOBIS/KMDb 등) 를 집계한다."""
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # source 컬럼이 존재하지 않을 수 있으므로 try-except 로 방어
                try:
                    await cur.execute(
                        "SELECT COALESCE(source, 'unknown') AS src, COUNT(*) "
                        "FROM movies GROUP BY src ORDER BY COUNT(*) DESC"
                    )
                    rows = await cur.fetchall()
                    distribution = [
                        {"source": row[0], "count": row[1]} for row in rows
                    ]
                except Exception:
                    # source 컬럼 미존재 — country 또는 origin_country 로 대체
                    await cur.execute(
                        "SELECT COALESCE(country, 'unknown') AS c, COUNT(*) "
                        "FROM movies GROUP BY c ORDER BY COUNT(*) DESC LIMIT 20"
                    )
                    rows = await cur.fetchall()
                    distribution = [
                        {"source": row[0], "count": row[1]} for row in rows
                    ]

        total = sum(item["count"] for item in distribution)
        for item in distribution:
            item["percentage"] = round(item["count"] / total * 100.0, 1) if total > 0 else 0.0

        return {"distribution": distribution, "total": total}
    except Exception as e:
        logger.error("data_distribution_failed", error=str(e), traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"분포 조회 실패: {e}")


@admin_data_router.get(
    "/data/quality",
    summary="데이터 품질 지표",
    description="중복률, NULL 률, 평균 평점 등 movies 테이블의 품질 지표를 반환한다.",
)
async def get_data_quality() -> dict:
    """movies 테이블의 데이터 품질 지표를 계산한다."""
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM movies")
                total = (await cur.fetchone())[0]

                # NULL 비율
                async def count_null(col: str) -> int:
                    try:
                        await cur.execute(f"SELECT COUNT(*) FROM movies WHERE {col} IS NULL OR {col} = ''")
                        return (await cur.fetchone())[0]
                    except Exception:
                        return 0

                null_overview = await count_null("overview")
                null_genres = await count_null("genres")
                null_poster = await count_null("poster_url")
                null_director = await count_null("director")

                # 중복 제목 검출 (동명이인 영화)
                try:
                    await cur.execute(
                        "SELECT COUNT(*) FROM ("
                        "  SELECT title FROM movies GROUP BY title HAVING COUNT(*) > 1"
                        ") t"
                    )
                    dup_titles = (await cur.fetchone())[0]
                except Exception:
                    dup_titles = 0

                # 평균 rating
                try:
                    await cur.execute("SELECT AVG(rating) FROM movies WHERE rating IS NOT NULL")
                    avg_rating = (await cur.fetchone())[0] or 0.0
                except Exception:
                    avg_rating = 0.0

        def _ratio(n: int) -> float:
            return round(n / total * 100.0, 2) if total > 0 else 0.0

        return {
            "totalMovies": total,
            "nullRates": {
                "overview": {"count": null_overview, "ratio": _ratio(null_overview)},
                "genres": {"count": null_genres, "ratio": _ratio(null_genres)},
                "posterUrl": {"count": null_poster, "ratio": _ratio(null_poster)},
                "director": {"count": null_director, "ratio": _ratio(null_director)},
            },
            "duplicateTitles": dup_titles,
            "averageRating": round(float(avg_rating), 2),
            "checkedAt": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error("data_quality_failed", error=str(e), traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"품질 분석 실패: {e}")


# ============================================================
# 2. 영화 CRUD (4 EP) — 5DB 통합
# ============================================================

@admin_data_router.get(
    "/movies",
    summary="영화 검색/필터/페이징 (관리자)",
    description="MySQL movies 테이블에서 키워드 검색 + 페이징 결과를 반환한다.",
)
async def list_movies(
    keyword: Optional[str] = Query(None, description="제목 검색 키워드"),
    source: Optional[str] = Query(None, description="소스 필터"),
    page: int = Query(0, ge=0, description="페이지 번호 (0-base)"),
    size: int = Query(20, ge=1, le=100, description="페이지 크기 (1~100)"),
) -> dict:
    """관리자 영화 목록 조회 — keyword + source 동적 필터, 최신순 페이징."""
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # 동적 WHERE 절 + 파라미터 바인딩 (SQL 인젝션 방지)
                where_clauses = []
                params: list[Any] = []
                if keyword:
                    where_clauses.append("(title LIKE %s OR title_en LIKE %s)")
                    kw = f"%{keyword}%"
                    params.extend([kw, kw])
                if source:
                    where_clauses.append("source = %s")
                    params.append(source)

                where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

                # 총 건수
                count_sql = f"SELECT COUNT(*) FROM movies {where_sql}"
                await cur.execute(count_sql, params)
                total = (await cur.fetchone())[0]

                # 페이징 조회 (selected 컬럼만 — 응답 페이로드 최소화)
                offset = page * size
                list_sql = (
                    f"SELECT movie_id, title, title_en, release_year, rating, "
                    f"director, poster_url, runtime "
                    f"FROM movies {where_sql} "
                    f"ORDER BY release_year DESC, movie_id DESC "
                    f"LIMIT %s OFFSET %s"
                )
                await cur.execute(list_sql, params + [size, offset])
                rows = await cur.fetchall()

                items = [
                    {
                        "movieId": row[0],
                        "title": row[1],
                        "titleEn": row[2],
                        "releaseYear": row[3],
                        "rating": row[4],
                        "director": row[5],
                        "posterUrl": row[6],
                        "runtime": row[7],
                    }
                    for row in rows
                ]

        return {
            "items": items,
            "page": page,
            "size": size,
            "total": total,
            "totalPages": (total + size - 1) // size if size > 0 else 0,
        }
    except Exception as e:
        logger.error("list_movies_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"영화 목록 조회 실패: {e}")


@admin_data_router.get(
    "/movies/{movie_id}",
    summary="영화 단건 상세 (5DB 통합)",
    description="MySQL + Qdrant + Neo4j + ES 의 데이터를 통합하여 영화 상세 정보를 반환한다.",
)
async def get_movie_detail(movie_id: str) -> dict:
    """단일 영화의 5DB 통합 상세 정보."""
    response: dict[str, Any] = {"movieId": movie_id}

    # 1. MySQL — 핵심 메타데이터
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM movies WHERE movie_id = %s LIMIT 1",
                    (movie_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    raise HTTPException(status_code=404, detail=f"영화를 찾을 수 없습니다: {movie_id}")

                # 컬럼명 추출
                col_names = [desc[0] for desc in cur.description]
                response["mysql"] = dict(zip(col_names, row))
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("movie_detail_mysql_failed", error=str(e), movie_id=movie_id)
        response["mysql"] = {"error": str(e)}

    # 2. Qdrant — 벡터 존재 여부
    try:
        client = await get_qdrant()
        points = await client.retrieve(
            collection_name=settings.QDRANT_COLLECTION,
            ids=[movie_id],
            with_payload=True,
            with_vectors=False,
        )
        response["qdrant"] = {
            "exists": len(points) > 0,
            "payload": points[0].payload if points else None,
        }
    except Exception as e:
        response["qdrant"] = {"error": str(e)}

    # 3. Neo4j — 관계 그래프 (1-hop)
    try:
        driver = await get_neo4j()
        async with driver.session() as session:
            result = await session.run(
                "MATCH (m:Movie {movieId: $id}) "
                "OPTIONAL MATCH (m)-[r]-(other) "
                "RETURN type(r) AS rel, count(other) AS cnt",
                id=movie_id,
            )
            relations = []
            async for record in result:
                if record["rel"]:
                    relations.append({"type": record["rel"], "count": record["cnt"]})
            response["neo4j"] = {"relations": relations}
    except Exception as e:
        response["neo4j"] = {"error": str(e)}

    # 4. Elasticsearch — 인덱싱 여부
    try:
        es = await get_elasticsearch()
        es_doc = await es.get(index=ES_INDEX_NAME, id=movie_id, ignore=[404])
        response["elasticsearch"] = {
            "exists": es_doc.get("found", False),
            "source": es_doc.get("_source") if es_doc.get("found") else None,
        }
    except Exception as e:
        response["elasticsearch"] = {"error": str(e)}

    return response


@admin_data_router.put(
    "/movies/{movie_id}",
    summary="영화 수정 (4DB 동기 반영)",
    description="MySQL 업데이트 후 Qdrant/Neo4j/ES 에 베스트-에포트로 동기 반영한다.",
)
async def update_movie(movie_id: str, request: MovieUpdateRequest) -> dict:
    """영화 메타데이터 수정 — MySQL 우선, 나머지 4DB 베스트-에포트."""
    # 1. 동적 UPDATE SET 절 구성
    update_dict = request.dict(exclude_none=True)
    if not update_dict:
        raise HTTPException(status_code=400, detail="변경할 필드가 없습니다.")

    set_clauses = ", ".join(f"{k} = %s" for k in update_dict.keys())
    params = list(update_dict.values()) + [movie_id]

    sync_results: dict[str, Any] = {}

    # 2. MySQL UPDATE (필수)
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE movies SET {set_clauses} WHERE movie_id = %s",
                    params,
                )
                affected = cur.rowcount
                if affected == 0:
                    raise HTTPException(status_code=404, detail=f"영화를 찾을 수 없습니다: {movie_id}")
        sync_results["mysql"] = {"updated": True, "affectedRows": affected}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("movie_update_mysql_failed", error=str(e), movie_id=movie_id)
        raise HTTPException(status_code=500, detail=f"MySQL 수정 실패: {e}")

    # 3. Qdrant payload 업데이트 (베스트-에포트)
    if any(k in update_dict for k in ("title", "title_en", "overview", "genres")):
        try:
            client = await get_qdrant()
            await client.set_payload(
                collection_name=settings.QDRANT_COLLECTION,
                payload={k: v for k, v in update_dict.items() if k in ("title", "title_en", "overview", "genres")},
                points=[movie_id],
            )
            sync_results["qdrant"] = {"updated": True}
        except Exception as e:
            logger.warning("movie_update_qdrant_failed", error=str(e), movie_id=movie_id)
            sync_results["qdrant"] = {"updated": False, "error": str(e)}

    # 4. ES 부분 업데이트 (베스트-에포트)
    try:
        es = await get_elasticsearch()
        await es.update(
            index=ES_INDEX_NAME,
            id=movie_id,
            body={"doc": update_dict},
        )
        sync_results["elasticsearch"] = {"updated": True}
    except Exception as e:
        logger.warning("movie_update_es_failed", error=str(e), movie_id=movie_id)
        sync_results["elasticsearch"] = {"updated": False, "error": str(e)}

    # 5. Neo4j 노드 속성 업데이트 (베스트-에포트)
    try:
        driver = await get_neo4j()
        async with driver.session() as session:
            # 동적 SET 절 (Cypher)
            set_props = ", ".join(f"m.{k} = ${k}" for k in update_dict.keys())
            query = f"MATCH (m:Movie {{movieId: $id}}) SET {set_props} RETURN m"
            params_neo = {"id": movie_id, **update_dict}
            await session.run(query, **params_neo)
        sync_results["neo4j"] = {"updated": True}
    except Exception as e:
        logger.warning("movie_update_neo4j_failed", error=str(e), movie_id=movie_id)
        sync_results["neo4j"] = {"updated": False, "error": str(e)}

    return {
        "success": True,
        "movieId": movie_id,
        "updatedFields": list(update_dict.keys()),
        "syncResults": sync_results,
    }


@admin_data_router.delete(
    "/movies/{movie_id}",
    summary="영화 삭제 (4DB 동기)",
    description="MySQL 삭제 후 Qdrant/Neo4j/ES 에서 베스트-에포트로 동기 삭제한다.",
)
async def delete_movie(movie_id: str) -> dict:
    """영화 삭제 — MySQL 우선, 나머지 4DB 베스트-에포트."""
    sync_results: dict[str, Any] = {}

    # 1. MySQL DELETE
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM movies WHERE movie_id = %s", (movie_id,))
                affected = cur.rowcount
                if affected == 0:
                    raise HTTPException(status_code=404, detail=f"영화를 찾을 수 없습니다: {movie_id}")
        sync_results["mysql"] = {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("movie_delete_mysql_failed", error=str(e), movie_id=movie_id)
        raise HTTPException(status_code=500, detail=f"MySQL 삭제 실패: {e}")

    # 2. Qdrant 포인트 삭제
    try:
        client = await get_qdrant()
        await client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector={"points": [movie_id]},
        )
        sync_results["qdrant"] = {"deleted": True}
    except Exception as e:
        logger.warning("movie_delete_qdrant_failed", error=str(e), movie_id=movie_id)
        sync_results["qdrant"] = {"deleted": False, "error": str(e)}

    # 3. ES 문서 삭제
    try:
        es = await get_elasticsearch()
        await es.delete(index=ES_INDEX_NAME, id=movie_id, ignore=[404])
        sync_results["elasticsearch"] = {"deleted": True}
    except Exception as e:
        logger.warning("movie_delete_es_failed", error=str(e), movie_id=movie_id)
        sync_results["elasticsearch"] = {"deleted": False, "error": str(e)}

    # 4. Neo4j 노드 + 관계 삭제
    try:
        driver = await get_neo4j()
        async with driver.session() as session:
            await session.run(
                "MATCH (m:Movie {movieId: $id}) DETACH DELETE m",
                id=movie_id,
            )
        sync_results["neo4j"] = {"deleted": True}
    except Exception as e:
        logger.warning("movie_delete_neo4j_failed", error=str(e), movie_id=movie_id)
        sync_results["neo4j"] = {"deleted": False, "error": str(e)}

    return {"success": True, "movieId": movie_id, "syncResults": sync_results}


# ============================================================
# 3. 파이프라인 (8 EP)
# ============================================================

@admin_data_router.get(
    "/pipeline",
    summary="파이프라인 작업 목록",
    description="실행 가능한 9개 파이프라인 작업의 메타데이터를 반환한다.",
)
async def list_pipelines() -> dict:
    """PIPELINE_TASKS 정의를 응답으로 변환한다."""
    return {
        "tasks": [
            {
                "code": code,
                "name": meta["name"],
                "description": meta["description"],
                "category": meta["category"],
            }
            for code, meta in PIPELINE_TASKS.items()
        ]
    }


@admin_data_router.post(
    "/pipeline/run",
    summary="파이프라인 실행",
    description="task_code 에 해당하는 스크립트를 subprocess 로 비동기 실행하고 job_id 를 반환한다.",
)
async def run_pipeline(request: PipelineRunRequest) -> PipelineRunResponse:
    """subprocess.Popen 으로 파이프라인 실행 — 비차단."""
    task = PIPELINE_TASKS.get(request.task_code)
    if not task:
        raise HTTPException(status_code=400, detail=f"알 수 없는 작업 코드: {request.task_code}")

    # 동시 실행 방지: 동일 task_code 가 RUNNING 상태이면 거부
    for existing_id, job in PIPELINE_JOBS.items():
        if job["task_code"] == request.task_code and job["status"] == "RUNNING":
            raise HTTPException(
                status_code=409,
                detail=f"이미 실행 중인 작업입니다: {request.task_code} (job_id={existing_id})",
            )

    # 새 job_id 생성
    job_id = str(uuid.uuid4())
    started_at = datetime.utcnow().isoformat() + "Z"

    # subprocess 명령 구성 (PYTHONPATH=src 보장)
    script_path = task["script"]
    if not Path(script_path).exists():
        raise HTTPException(status_code=500, detail=f"스크립트 미존재: {script_path}")

    cmd = [sys.executable, script_path] + request.args
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    # 비차단 실행 (stdout/stderr 파이프)
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
    except Exception as e:
        logger.error("pipeline_subprocess_failed", error=str(e), task=request.task_code)
        raise HTTPException(status_code=500, detail=f"subprocess 실행 실패: {e}")

    # 작업 상태 등록
    PIPELINE_JOBS[job_id] = {
        "job_id": job_id,
        "task_code": request.task_code,
        "task_name": task["name"],
        "args": request.args,
        "status": "RUNNING",
        "started_at": started_at,
        "ended_at": None,
        "exit_code": None,
        "log_lines": [],
        "process": process,
    }

    # SSE 큐 초기화
    PIPELINE_LOG_QUEUES[job_id] = asyncio.Queue()

    # 백그라운드 태스크: stdout 라인 단위 읽기 + 큐에 push
    asyncio.create_task(_read_subprocess_output(job_id, process))

    logger.info("pipeline_started", job_id=job_id, task=request.task_code)
    return PipelineRunResponse(
        job_id=job_id,
        task_code=request.task_code,
        status="RUNNING",
        started_at=started_at,
        message=f"작업이 시작되었습니다. job_id={job_id}",
    )


async def _read_subprocess_output(job_id: str, process: asyncio.subprocess.Process) -> None:
    """subprocess 의 stdout 을 라인 단위로 읽어 PIPELINE_LOG_QUEUES 에 push 한다."""
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        return

    try:
        # stdout 이 None 이 아닌지 검증
        if process.stdout is None:
            return

        async for raw_line in process.stdout:
            try:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            except Exception:
                line = repr(raw_line)
            job["log_lines"].append(line)
            # SSE 큐에 push (큐 크기 제한 — 메모리 보호)
            queue = PIPELINE_LOG_QUEUES.get(job_id)
            if queue is not None and queue.qsize() < 5000:
                await queue.put(line)

        # 프로세스 종료 대기
        exit_code = await process.wait()
        job["exit_code"] = exit_code
        job["ended_at"] = datetime.utcnow().isoformat() + "Z"
        job["status"] = "SUCCESS" if exit_code == 0 else "FAILED"

        # 종료 신호를 큐에 push (None 센티넬)
        queue = PIPELINE_LOG_QUEUES.get(job_id)
        if queue is not None:
            await queue.put(None)

        # 영구 이력 append
        _append_pipeline_history(job)

        logger.info("pipeline_finished", job_id=job_id, status=job["status"], exit_code=exit_code)
    except Exception as e:
        logger.error("pipeline_read_output_failed", job_id=job_id, error=str(e))
        job["status"] = "FAILED"
        job["ended_at"] = datetime.utcnow().isoformat() + "Z"
        _append_pipeline_history(job)


def _append_pipeline_history(job: dict) -> None:
    """완료된 작업을 영구 이력 파일에 append 한다 (Agent 재시작 후에도 유지)."""
    try:
        # process 객체는 직렬화 불가 — 제외
        record = {k: v for k, v in job.items() if k not in ("process", "log_lines")}
        # 마지막 200줄만 저장 (이력 파일 비대화 방지)
        record["lastLogLines"] = job.get("log_lines", [])[-200:]
        with open(PIPELINE_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("pipeline_history_append_failed", error=str(e))


@admin_data_router.post(
    "/pipeline/cancel",
    summary="파이프라인 작업 취소",
    description="실행 중인 작업을 SIGTERM 으로 종료한다. 체크포인트가 있으면 보존된다.",
)
async def cancel_pipeline(request: PipelineCancelRequest) -> dict:
    """job_id 에 해당하는 작업을 취소한다."""
    job = PIPELINE_JOBS.get(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {request.job_id}")
    if job["status"] != "RUNNING":
        raise HTTPException(status_code=400, detail=f"실행 중이 아닙니다: status={job['status']}")

    process: asyncio.subprocess.Process = job["process"]
    try:
        process.terminate()  # SIGTERM
        # 5초 대기 후 강제 종료
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()  # SIGKILL
    except Exception as e:
        logger.warning("pipeline_cancel_failed", error=str(e), job_id=request.job_id)

    job["status"] = "CANCELLED"
    job["ended_at"] = datetime.utcnow().isoformat() + "Z"
    _append_pipeline_history(job)

    return {"success": True, "jobId": request.job_id, "status": "CANCELLED"}


@admin_data_router.get(
    "/pipeline/logs",
    summary="파이프라인 SSE 로그 스트리밍",
    description="job_id 에 해당하는 실행 중인 작업의 stdout 을 SSE 로 실시간 전송한다.",
)
async def stream_pipeline_logs(job_id: str = Query(..., description="작업 ID")) -> EventSourceResponse:
    """SSE 로 작업의 stdout 라인을 실시간 전송한다."""
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {job_id}")

    queue = PIPELINE_LOG_QUEUES.get(job_id)
    if queue is None:
        raise HTTPException(status_code=404, detail=f"로그 큐가 없습니다: {job_id}")

    async def event_generator():
        # 1) 이미 누적된 로그를 우선 전송 (재접속 시 전체 컨텍스트 복원)
        for past_line in job.get("log_lines", []):
            yield {"event": "log", "data": past_line}

        # 2) 실시간 라인 전송 — None 센티넬 도착 시 종료
        while True:
            try:
                line = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # keep-alive 핑 (30초마다)
                yield {"event": "ping", "data": "keep-alive"}
                continue
            if line is None:
                # 종료 신호
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "status": job["status"],
                        "exitCode": job["exit_code"],
                    }),
                }
                break
            yield {"event": "log", "data": line}

    return EventSourceResponse(event_generator())


@admin_data_router.get(
    "/pipeline/history",
    summary="파이프라인 실행 이력 조회",
    description="data/pipeline_history.jsonl 에서 최근 N건을 페이징 조회한다.",
)
async def get_pipeline_history(
    page: int = Query(0, ge=0),
    size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="필터: SUCCESS/FAILED/CANCELLED"),
) -> dict:
    """JSONL 이력 파일을 역순으로 페이징 읽기."""
    if not PIPELINE_HISTORY_FILE.exists():
        return {"items": [], "page": page, "size": size, "total": 0}

    try:
        with open(PIPELINE_HISTORY_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error("pipeline_history_read_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"이력 파일 읽기 실패: {e}")

    # 역순 (최신순)
    records = []
    for line in reversed(lines):
        try:
            rec = json.loads(line)
            if status and rec.get("status") != status:
                continue
            records.append(rec)
        except Exception:
            continue

    # 메모리 내 + RUNNING 작업도 포함
    in_memory_running = [
        {k: v for k, v in job.items() if k not in ("process", "log_lines")}
        for job in PIPELINE_JOBS.values()
        if job["status"] == "RUNNING"
    ]
    if not status or status == "RUNNING":
        records = in_memory_running + records

    total = len(records)
    start = page * size
    end = start + size
    return {
        "items": records[start:end],
        "page": page,
        "size": size,
        "total": total,
        "totalPages": (total + size - 1) // size if size > 0 else 0,
    }


@admin_data_router.get(
    "/pipeline/stats",
    summary="파이프라인 실행 통계",
    description="이력 파일 전체에 대한 성공/실패/취소 카운트와 평균 실행 시간을 반환한다.",
)
async def get_pipeline_stats() -> dict:
    """이력 통계 — 메모리 내 + 영구 이력 합산."""
    counts = {"SUCCESS": 0, "FAILED": 0, "CANCELLED": 0, "RUNNING": 0}
    total_duration = 0.0
    duration_count = 0

    # 1. 영구 이력
    if PIPELINE_HISTORY_FILE.exists():
        try:
            with open(PIPELINE_HISTORY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        st = rec.get("status", "FAILED")
                        counts[st] = counts.get(st, 0) + 1
                        # 시작/종료 시각 차이로 실행 시간 계산
                        if rec.get("started_at") and rec.get("ended_at"):
                            try:
                                start = datetime.fromisoformat(rec["started_at"].rstrip("Z"))
                                end = datetime.fromisoformat(rec["ended_at"].rstrip("Z"))
                                total_duration += (end - start).total_seconds()
                                duration_count += 1
                            except Exception:
                                pass
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("pipeline_stats_history_failed", error=str(e))

    # 2. 메모리 내 RUNNING 작업
    for job in PIPELINE_JOBS.values():
        if job["status"] == "RUNNING":
            counts["RUNNING"] = counts.get("RUNNING", 0) + 1

    avg_duration = round(total_duration / duration_count, 1) if duration_count > 0 else 0.0
    return {
        "counts": counts,
        "averageDurationSeconds": avg_duration,
        "totalRuns": sum(counts.values()),
    }


@admin_data_router.post(
    "/pipeline/retry-failed",
    summary="실패한 파이프라인 재시도",
    description="이력에서 가장 최근 실패한 작업을 동일 인자로 재실행한다.",
)
async def retry_failed_pipeline() -> dict:
    """가장 최근 FAILED 작업을 찾아 재실행한다."""
    if not PIPELINE_HISTORY_FILE.exists():
        raise HTTPException(status_code=404, detail="이력 파일이 없습니다.")

    last_failed: Optional[dict] = None
    try:
        with open(PIPELINE_HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "FAILED":
                        last_failed = rec  # 가장 마지막에 등장한 FAILED 가 최신
                except Exception:
                    continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이력 읽기 실패: {e}")

    if last_failed is None:
        raise HTTPException(status_code=404, detail="실패한 작업이 없습니다.")

    # 동일 task_code + 인자로 재실행
    request = PipelineRunRequest(
        task_code=last_failed["task_code"],
        args=last_failed.get("args", []),
    )
    response = await run_pipeline(request)
    return {
        "success": True,
        "originalJobId": last_failed.get("job_id"),
        "newJobId": response.job_id,
        "taskCode": response.task_code,
    }


@admin_data_router.get(
    "/collection/{name}/status",
    summary="Qdrant 컬렉션 상태",
    description="지정한 Qdrant 컬렉션의 벡터 수, 세그먼트 수, 메모리 사용량을 반환한다.",
)
async def get_collection_status(name: str) -> dict:
    """Qdrant 컬렉션 상세 상태 조회."""
    try:
        client = await get_qdrant()
        info = await client.get_collection(name)
        return {
            "name": name,
            "vectorCount": info.points_count,
            "segmentCount": info.segments_count,
            "indexedVectorsCount": info.indexed_vectors_count,
            "status": str(info.status),
            "vectorSize": (
                info.config.params.vectors.size
                if hasattr(info.config.params, "vectors") else None
            ),
        }
    except Exception as e:
        logger.warning("collection_status_failed", error=str(e), name=name)
        raise HTTPException(status_code=404, detail=f"컬렉션 조회 실패: {name} — {e}")


# ============================================================
# 4. AI 리뷰 생성 (Phase 7) — Backend AdminAiOpsService 와 연동
# ============================================================

class GenerateReviewRequest(BaseModel):
    """AI 리뷰 생성 요청 (Backend 의 AdminAiOpsDto.GenerateReviewRequest 와 1:1 매핑)."""

    movie_id: str = Field(..., description="대상 영화 ID")
    style: Optional[str] = Field(None, description="리뷰 스타일 (critical/enthusiastic/neutral)")
    length: Optional[str] = Field(None, description="리뷰 길이 (short/medium/long)")


class GenerateReviewResponse(BaseModel):
    """AI 리뷰 생성 응답."""

    success: bool
    movie_id: str
    review_text: str
    word_count: int
    style: str
    length: str
    message: str


@admin_data_router.post(
    "/ai/review/generate",
    summary="AI 영화 리뷰 생성",
    description=(
        "지정한 영화에 대해 Explanation LLM 으로 리뷰 초안을 생성한다. "
        "스타일/길이 옵션으로 톤을 조절할 수 있다. Backend 의 AdminAiOpsService.generateReview 가 "
        "이 엔드포인트를 호출한다."
    ),
)
async def admin_generate_review(request: GenerateReviewRequest) -> GenerateReviewResponse:
    """Explanation LLM 기반 영화 리뷰 생성.

    1. MySQL 에서 영화 메타데이터(제목/장르/감독/줄거리) 조회
    2. 스타일/길이 파라미터로 프롬프트 조정
    3. get_explanation_llm() 호출 → guarded_ainvoke 로 안전 실행
    4. 생성된 텍스트를 반환 (Backend 가 실제 reviews 테이블 INSERT 는 별도 처리)
    """
    style = (request.style or "neutral").lower()
    length = (request.length or "medium").lower()

    # 길이별 단어 수 가이드
    length_guide = {
        "short": "50~80 단어",
        "medium": "120~180 단어",
        "long": "200~300 단어",
    }.get(length, "120~180 단어")

    # 스타일별 톤 가이드
    style_guide = {
        "critical": "비평적이고 분석적인 톤. 단점도 함께 언급",
        "enthusiastic": "열정적이고 긍정적인 톤. 추천 포인트 강조",
        "neutral": "균형 잡힌 중립적 톤. 객관적 사실 기반",
    }.get(style, "균형 잡힌 중립적 톤")

    # 1. 영화 메타데이터 조회
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT title, title_en, overview, genres, director, release_year, rating "
                    "FROM movies WHERE movie_id = %s LIMIT 1",
                    (request.movie_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"영화를 찾을 수 없습니다: {request.movie_id}",
                    )
                title, title_en, overview, genres, director, release_year, rating = row
    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin_generate_review_db_failed", error=str(e), movie_id=request.movie_id)
        raise HTTPException(status_code=500, detail=f"영화 조회 실패: {e}")

    # 2. LLM 호출
    try:
        # 지연 임포트 — 모듈 로드 시 LLM 초기화 회피
        from langchain_core.messages import HumanMessage, SystemMessage

        from monglepick.llm.factory import get_explanation_llm, guarded_ainvoke

        system_prompt = (
            "당신은 영화 평론가입니다. 주어진 영화 메타데이터를 바탕으로 한국어 리뷰를 작성합니다. "
            f"톤: {style_guide}. "
            f"분량: {length_guide}. "
            "스포일러는 포함하지 마세요. 결말이나 반전은 언급 금지. "
            "리뷰만 출력하고 다른 설명은 하지 마세요."
        )
        user_prompt = (
            f"제목: {title}\n"
            f"원제: {title_en or '-'}\n"
            f"개봉: {release_year or '-'}\n"
            f"감독: {director or '-'}\n"
            f"평점: {rating or '-'}\n"
            f"장르: {genres or '-'}\n"
            f"줄거리: {overview or '(없음)'}\n\n"
            "위 정보를 바탕으로 영화 리뷰를 작성해주세요."
        )

        llm = get_explanation_llm()
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # guarded_ainvoke 시그니처가 다양할 수 있어 단순 ainvoke 우선 시도
        try:
            response = await llm.ainvoke(messages)
            review_text = response.content if hasattr(response, "content") else str(response)
        except Exception:
            # 폴백: guarded_ainvoke 사용
            review_text = await guarded_ainvoke(llm, messages, fallback="")  # type: ignore

        review_text = (review_text or "").strip()
        if not review_text:
            raise RuntimeError("LLM 응답이 비어 있습니다.")

        word_count = len(review_text.split())
        return GenerateReviewResponse(
            success=True,
            movie_id=request.movie_id,
            review_text=review_text,
            word_count=word_count,
            style=style,
            length=length,
            message=f"리뷰 생성 완료 ({word_count} 단어)",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("admin_generate_review_llm_failed", error=str(e), movie_id=request.movie_id)
        raise HTTPException(status_code=500, detail=f"LLM 호출 실패: {e}")
