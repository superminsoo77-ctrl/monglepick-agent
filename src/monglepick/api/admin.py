"""
관리자 전용 API 라우터.

시스템 탭:
- GET /api/v1/admin/system/db — 5개 DB(MySQL/Qdrant/Neo4j/ES/Redis) 상태 조회
- GET /api/v1/admin/system/ollama — Ollama 모델 로드 상태 조회
- GET /api/v1/admin/system/vllm   — vLLM 모델 상태 조회 (Chat/Vision 2개 엔드포인트)

AI 운영 탭:
- POST /api/v1/admin/ai/quiz/generate — LLM 기반 영화 퀴즈 자동 생성 + quizzes 테이블 PENDING INSERT
"""

import time
import traceback
from typing import Any, Optional

import httpx
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator


from monglepick.config import settings
from monglepick.db.clients import (
    get_elasticsearch,
    get_mysql,
    get_neo4j,
    get_qdrant,
    get_redis,
)

logger = structlog.get_logger()

# 라우터 레벨에는 가드를 걸지 않는다.
# Admin SPA 가 agentApi 로 /admin/system/**, /admin/ai/chat/** 등을 **직접** 호출하기 때문.
# ServiceKey 는 쓰기성/민감 엔드포인트(예: review-verification/verify) 에 개별 적용한다.
# admin_data_router 도 동일 정책 — 자세한 근거는 api/auth_deps.py 문서 참고.
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
    """
    Neo4j 연결 상태 + 노드/관계 수 조회.

    라벨/타입 없는 `MATCH (n)` 은 전체 노드를 메모리에 로드하므로
    트랜잭션 메모리 한도(256MB)를 초과할 수 있다.
    라벨별·타입별 count-store O(1) 쿼리를 UNION ALL 로 합산하여 메모리 사용을 최소화한다.
    """
    # 프로젝트에서 사용하는 노드 라벨 / 관계 타입
    _NODE_LABELS = [
        "Movie", "Person", "Genre", "Keyword", "MoodTag",
        "OTTPlatform", "Studio", "Collection", "Country",
    ]
    _REL_TYPES = [
        "DIRECTED", "ACTED_IN", "HAS_GENRE", "HAS_KEYWORD", "HAS_MOOD",
        "AVAILABLE_ON", "PRODUCED_BY", "PART_OF_COLLECTION", "PRODUCED_IN",
        "SHOT_BY", "COMPOSED_BY", "WRITTEN_BY", "PRODUCED", "EDITED_BY",
        "EXECUTIVE_PRODUCED", "DESIGNED", "COSTUMED", "BASED_ON",
        "RECOMMENDED", "SIMILAR_TO",
    ]
    try:
        driver = await get_neo4j()
        async with driver.session() as session:
            # 노드 수: 라벨별 count-store O(1) 쿼리
            node_cypher = " UNION ALL ".join(
                f"MATCH (n:{label}) RETURN count(n) AS cnt"
                for label in _NODE_LABELS
            )
            result = await session.run(node_cypher)
            records = await result.data()
            node_count = sum(r.get("cnt", 0) for r in records)

            # 관계 수: 타입별 count-store O(1) 쿼리
            rel_cypher = " UNION ALL ".join(
                f"MATCH ()-[r:{rtype}]->() RETURN count(r) AS cnt"
                for rtype in _REL_TYPES
            )
            result = await session.run(rel_cypher)
            records = await result.data()
            rel_count = sum(r.get("cnt", 0) for r in records)

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
    """
    MySQL 연결 상태 + 운영 지표 조회.

    관리자 대시보드 DbStatus 카드에서 기대하는 필드(totalRows / diskUsage /
    slowQueries / activeConnections)를 모두 채워 반환한다.

    - totalRows         : information_schema 기반 전체 테이블 행 수 합계 (근사치)
    - diskUsage         : MySQL 데이터 볼륨의 추정 디스크 사용량 (MB)
    - slowQueries       : `SHOW GLOBAL STATUS LIKE 'Slow_queries'` 누적 값
    - activeConnections : `Threads_connected` 현재 값
    """
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # 현재 사용 중인 스키마 전체 행 수 합계 (정확치는 아님 — InnoDB 통계)
                await cur.execute(
                    """
                    SELECT COALESCE(SUM(TABLE_ROWS), 0) AS total_rows,
                           COALESCE(SUM(DATA_LENGTH + INDEX_LENGTH), 0) AS total_bytes
                    FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = DATABASE()
                    """
                )
                row = await cur.fetchone()
                total_rows = int(row[0]) if row else 0
                total_bytes = int(row[1]) if row else 0

                # 누적 슬로우 쿼리 수
                await cur.execute("SHOW GLOBAL STATUS LIKE 'Slow_queries'")
                row = await cur.fetchone()
                slow_queries = int(row[1]) if row else 0

                # 현재 활성 커넥션 수
                await cur.execute("SHOW STATUS LIKE 'Threads_connected'")
                row = await cur.fetchone()
                active_connections = int(row[1]) if row else 0

        # 디스크 사용량을 MB 단위로 환산 (1 KB 미만은 "<1MB")
        disk_mb = total_bytes / (1024 * 1024)
        disk_usage = f"{disk_mb:.1f}MB" if disk_mb >= 1 else "<1MB"

        return {
            "connected": True,
            "totalRows": total_rows,
            "diskUsage": disk_usage,
            "slowQueries": slow_queries,
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


# ============================================================
# GET /admin/system/vllm — vLLM 모델 상태 조회 (Chat / Vision)
# ============================================================
#
# 운영서버(VM4) vLLM 상태 확인용. hybrid/api_only 모드에서 실제로 호출되는
# LLM 서버 2종을 개별 카드로 노출한다.
#
# - Chat   : EXAONE 4.0 1.2B  (VLLM_CHAT_BASE_URL   기본 :18000/v1)
# - Vision : Qwen2.5-VL-3B    (VLLM_VISION_BASE_URL 기본 :18001/v1)
#
# vLLM은 OpenAI 호환 API를 제공하므로 `/v1/models` 로 사용 가능 모델을,
# `/health` 로 서버 헬스를 확인할 수 있다. (vLLM `/health` 는 /v1 을 prefix로
# 가지지 않음 — base_url 의 `/v1` 을 제거하고 조회)
#
# Ollama 와 다른 점:
# - vLLM 은 프로세스당 하나의 모델만 로드한다(모델 리스트는 보통 1건).
# - VRAM/로드 상태를 API 로 노출하지 않으므로 "연결 여부 + 모델 ID" 만 보고한다.
# - VLLM_ENABLED=False 면 호출을 스킵하고 enabled=False 만 반환한다.
#
# 실패 시 /ollama 엔드포인트와 동일하게 500이 아닌 200 JSON에 connected=False 로
# 응답한다 (관리자 카드가 "장애" 뱃지를 띄울 수 있도록).


async def _probe_vllm(base_url: str, expected_model: str, timeout: float = 5.0) -> dict:
    """
    단일 vLLM 엔드포인트의 모델 리스트를 조회한다.

    base_url 예시: "http://10.20.0.10:18000/v1"
      - /models  : OpenAI 호환 모델 리스트 (200 = 서버 정상 + 서빙 중인 모델)

    반환 필드 (프론트 카드가 기대):
      - connected      : bool       (/models 200 응답 여부)
      - baseUrl        : 호출 대상 (디버깅용 노출)
      - expectedModel  : .env 에 지정된 모델 ID
      - loadedModels   : 서버가 실제로 서빙 중인 모델 ID 목록
      - healthStatus   : "ok" | "unreachable" | 에러 문자열
      - error          : 연결 실패 시 원인

    2026-04-15 성능 개선:
    - 기존에는 `/health` + `/models` 를 순차 호출하여 endpoint 당 최대 2*timeout
      (예: vision localhost 미연결 시 10초) 을 소모했다. `/models` 만으로도 연결
      상태와 서빙 모델을 모두 판단 가능하므로 `/health` 호출을 제거했다.
      이로써 관리자 대시보드 응답이 기존 ~10초 → ~5초로 단축.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # 사용 가능 모델 조회 (OpenAI 호환). 응답 200 이면 서버가 모델을
            # 서빙 중이며 네트워크 연결도 정상임을 의미한다.
            models_resp = await client.get(f"{base_url.rstrip('/')}/models")
            if models_resp.status_code != 200:
                return {
                    "connected": False,
                    "baseUrl": base_url,
                    "expectedModel": expected_model,
                    "loadedModels": [],
                    "healthStatus": f"http_{models_resp.status_code}",
                    "error": f"/models returned HTTP {models_resp.status_code}",
                }

            data = models_resp.json()
            loaded_models: list[str] = []
            for m in data.get("data", []) or []:
                model_id = m.get("id") or m.get("name")
                if model_id:
                    loaded_models.append(model_id)

        return {
            "connected": True,
            "baseUrl": base_url,
            "expectedModel": expected_model,
            "loadedModels": loaded_models,
            "healthStatus": "ok",
            "error": None,
        }

    except Exception as e:
        # 연결 실패 / 타임아웃 — 상세 에러를 그대로 관리자 UI 로 노출한다.
        return {
            "connected": False,
            "baseUrl": base_url,
            "expectedModel": expected_model,
            "loadedModels": [],
            "healthStatus": "unreachable",
            "error": str(e),
        }


@admin_router.get(
    "/system/vllm",
    summary="vLLM 모델 상태 조회",
    description=(
        "운영서버 vLLM(Chat + Vision)의 연결 상태와 서빙 중인 모델 ID를 반환한다. "
        "VLLM_ENABLED=False 인 환경에서는 enabled=False 만 반환한다."
    ),
)
async def get_vllm_status():
    """
    vLLM Chat / Vision 2개 엔드포인트를 병렬 조회한다.

    응답 구조:
        {
          "enabled": bool,                  # settings.VLLM_ENABLED
          "timeoutSeconds": int,            # settings.VLLM_TIMEOUT
          "chat":   { connected, baseUrl, expectedModel, loadedModels, healthStatus, error },
          "vision": { connected, baseUrl, expectedModel, loadedModels, healthStatus, error }
        }

    VLLM_ENABLED 가 False 여도 baseUrl/expectedModel 만 반환하여 관리자가
    설정값을 확인할 수 있게 한다(실제 네트워크 호출은 하지 않음).
    """
    import asyncio

    enabled = bool(getattr(settings, "VLLM_ENABLED", False))
    timeout = float(getattr(settings, "VLLM_TIMEOUT", 30))

    chat_base = getattr(settings, "VLLM_CHAT_BASE_URL", "http://localhost:18000/v1")
    chat_model = getattr(settings, "VLLM_CHAT_MODEL", "")
    vision_base = getattr(settings, "VLLM_VISION_BASE_URL", "http://localhost:18001/v1")
    vision_model = getattr(settings, "VLLM_VISION_MODEL", "")

    # 비활성화 상태여도 설정값은 노출한다 (카드에서 "비활성" 뱃지로 표시).
    if not enabled:
        stub = lambda base, model: {
            "connected": False,
            "baseUrl": base,
            "expectedModel": model,
            "loadedModels": [],
            "healthStatus": "disabled",
            "error": "VLLM_ENABLED=False",
        }
        return {
            "enabled": False,
            "timeoutSeconds": int(timeout),
            "chat": stub(chat_base, chat_model),
            "vision": stub(vision_base, vision_model),
        }

    # Chat/Vision 을 병렬로 프로브 (서로 다른 VM 포트라 한쪽이 느려도 합산 지연 최소화)
    probe_timeout = min(timeout, 5.0)  # 관리자 대시보드는 짧은 타임아웃이 낫다
    try:
        chat_result, vision_result = await asyncio.gather(
            _probe_vllm(chat_base, chat_model, timeout=probe_timeout),
            _probe_vllm(vision_base, vision_model, timeout=probe_timeout),
        )
    except Exception as e:
        # asyncio.gather 자체가 실패할 일은 거의 없지만 방어적으로 캐치.
        logger.warning("admin_vllm_check_failed", error=str(e))
        return {
            "enabled": True,
            "timeoutSeconds": int(timeout),
            "chat": {
                "connected": False, "baseUrl": chat_base, "expectedModel": chat_model,
                "loadedModels": [], "healthStatus": "error", "error": str(e),
            },
            "vision": {
                "connected": False, "baseUrl": vision_base, "expectedModel": vision_model,
                "loadedModels": [], "healthStatus": "error", "error": str(e),
            },
        }

    return {
        "enabled": True,
        "timeoutSeconds": int(timeout),
        "chat": chat_result,
        "vision": vision_result,
    }


# ============================================================
# POST /admin/ai/quiz/generate — LangGraph 기반 영화 퀴즈 자동 생성
# ============================================================
#
# 관리자 페이지 "AI 운영 → 퀴즈 생성" 카드가 호출하는 엔드포인트.
#
# 변경 이력:
# - 2026-04-08: Backend 의 dead-code 스텁을 Agent(FastAPI)로 이관하여 인라인 LLM 호출 구현.
# - 2026-04-28: 인라인 헬퍼들을 LangGraph 7노드 에이전트(`agents.quiz_generation`)로 승격.
#               영화 후보군 인기·다양성 가중, 메타데이터(시놉시스/감독/출연) 보강,
#               카테고리 라운드로빈, 스포일러 검증, 배치 내 중복 제거를 추가했다.
#               본 핸들러는 그래프 호출 + 응답 변환만 담당한다 (단일 진실 원본).
#
# 입력: {genre?, difficulty?, count?, excludeRecentDays?, rewardPoint?}
#  - genre: 장르 LIKE 필터 (None/빈 문자열 → 전체)
#  - difficulty: easy/medium/hard 난이도 힌트 (LLM 프롬프트 전달)
#  - count: 1~50 생성 목표 편수
#  - excludeRecentDays: 최근 N 일 quiz 가 있는 영화 제외 (기본 7)
#  - rewardPoint: 정답 시 지급 포인트 (기본 10)
#
# 응답: {success, count, message, quizzes[]} — 빈 DB / 매칭 실패 시 success=False (HTTP 500 금지).


class GenerateQuizRequest(BaseModel):
    """
    AI 퀴즈 자동 생성 요청 DTO.

    AiTriggerPanel(monglepick-admin) 폼과 호환되는 필드 + 운영 옵션.
    movieId 를 지정하면 movie_selector 샘플링을 건너뛰고 해당 영화만 사용한다.
    """

    genre: Optional[str] = Field(
        default=None,
        description="장르 필터 (movies.genres LIKE 검색). 빈 문자열/None 이면 전체.",
    )
    difficulty: str = Field(
        default="medium",
        description="난이도 힌트 (easy/medium/hard). LLM 프롬프트에 전달.",
    )
    count: int = Field(
        default=5,
        ge=1,
        le=50,
        description="생성할 퀴즈 개수 (1~50).",
    )
    excludeRecentDays: int = Field(
        default=7,
        ge=0,
        le=90,
        description="최근 N 일 동안 quiz 가 생성된 영화 제외 (0=비활성).",
    )
    rewardPoint: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="정답 시 지급할 보상 포인트 (기본 10).",
    )
    movieId: Optional[str] = Field(
        default=None,
        description="관리자 지정 영화 ID. 설정 시 movie_selector 샘플링을 건너뛰고 해당 영화만 사용.",
    )
    quizType: str = Field(
        default="auto",
        description="퀴즈 유형: 'auto'(자동/카테고리 라운드로빈) | 'plot'(줄거리) | 'cast'(출연진) | 'director'(감독) | 'genre'(장르)",
    )


class GeneratedQuizItem(BaseModel):
    """생성된 퀴즈 1건 응답 DTO."""

    quizId: int
    movieId: str
    movieTitle: str
    question: str
    correctAnswer: str
    options: list[str]
    explanation: Optional[str] = None
    rewardPoint: int
    status: str = "PENDING"


class GenerateQuizResponse(BaseModel):
    """AI 퀴즈 생성 응답 DTO."""

    success: bool
    count: int
    message: str
    quizzes: list[GeneratedQuizItem]


@admin_router.post(
    "/ai/quiz/generate",
    response_model=GenerateQuizResponse,
    summary="AI 퀴즈 자동 생성 (LangGraph)",
    description=(
        "LangGraph 7노드 에이전트(quiz_generation_graph)로 영화 퀴즈를 자동 생성하고 "
        "quizzes 테이블에 PENDING 으로 저장한다. 관리자 검수 후 APPROVED/PUBLISHED "
        "로 전환한다. 흐름: movie_selector → metadata_enricher → question_generator "
        "→ quality_validator → diversity_checker → fallback_filler → persistence."
    ),
)
async def generate_admin_quiz(request: GenerateQuizRequest) -> GenerateQuizResponse:
    """
    관리자 AI 퀴즈 자동 생성 핸들러.

    LangGraph quiz_generation 에이전트를 1회 ainvoke 호출하고,
    그래프 종단 상태(persisted, final_message, success) 를 응답 DTO 로 변환한다.

    그래프 자체가 빈 DB / LLM 실패 / 검증 실패에 대해 fallback 처리를 보장하므로
    여기서는 추가 분기 로직을 두지 않는다. 모든 처리/로깅은 노드 내부에서 수행된다.
    """
    # 그래프 모듈은 핸들러 호출 시점에 import — 순환 import 방지 + 모듈 로드 비용 분산.
    from monglepick.agents.quiz_generation.graph import quiz_generation_graph

    logger.info(
        "admin_quiz_generate_start",
        genre=request.genre,
        difficulty=request.difficulty,
        count=request.count,
        exclude_recent_days=request.excludeRecentDays,
        reward_point=request.rewardPoint,
        movie_id=request.movieId,
        quiz_type=request.quizType,
    )

    # ── LangGraph 초기 상태 구성 (camelCase → snake_case) ──
    initial_state: dict = {
        "genre": (request.genre or "").strip() or None,
        "difficulty": request.difficulty,
        "count": request.count,
        "exclude_recent_days": request.excludeRecentDays,
        "reward_point": request.rewardPoint,
        "forced_movie_id": (request.movieId or "").strip() or None,
        "quiz_type": request.quizType or "auto",
    }

    try:
        result_state = await quiz_generation_graph.ainvoke(initial_state)
    except Exception as e:
        # 그래프 자체가 예외를 throw 하는 일은 거의 없지만(노드별 try/except 가 흡수),
        # 안전 그물로 500 대신 500 메시지를 200 으로 감싸 UI 에 안내한다.
        logger.error(
            "admin_quiz_generate_graph_failed",
            error=str(e),
            trace=traceback.format_exc()[-500:],
        )
        return GenerateQuizResponse(
            success=False,
            count=0,
            message=f"퀴즈 생성 그래프 실행 중 오류: {e}",
            quizzes=[],
        )

    # ── 상태 → 응답 DTO 변환 ──
    persisted = result_state.get("persisted") or []
    items: list[GeneratedQuizItem] = [
        GeneratedQuizItem(
            quizId=p.quiz_id,
            movieId=p.movie_id,
            movieTitle=p.movie_title,
            question=p.question,
            correctAnswer=p.correct_answer,
            options=p.options,
            explanation=p.explanation,
            rewardPoint=p.reward_point,
            status=p.status,
        )
        for p in persisted
    ]

    count = len(items)
    logger.info(
        "admin_quiz_generate_complete",
        requested=request.count,
        generated=count,
    )

    return GenerateQuizResponse(
        success=bool(result_state.get("success", count > 0)),
        count=count,
        message=str(result_state.get("final_message") or ""),
        quizzes=items,
    )


# =========================================================================
# 도장깨기 리뷰 검증 에이전트 — 2026-04-14 계약만 정의(placeholder).
#
# 목적:
#     사용자가 도장깨기 코스에서 작성한 리뷰를 영화 줄거리와 비교하여 "실제 관람으로
#     볼 수 있는가"를 판정한다. 임베딩 유사도 + 키워드 매칭 + (선택) LLM 판단 하이브리드로
#     구현할 예정이며, 판정 결과는 Backend 의 CourseVerification 엔티티에
#     similarity_score / matched_keywords / ai_confidence / review_status 로 업서트된다.
#
# 현 상태:
#     계약(입력/출력 스키마)만 확정되었고 실제 알고리즘은 미구현. 본 엔드포인트는
#     HTTP 503 을 반환하여 Admin UI 가 "에이전트 준비 중" 배너를 띄우도록 한다.
#     자세한 설계는 docs/도장깨기_리뷰검증_에이전트_설계서.md 참조.
# =========================================================================


class ReviewVerificationRequest(BaseModel):
    """
    리뷰 인증 에이전트 호출 요청.

    Backend 가 "이 리뷰를 AI 로 검증해달라"고 에이전트에 전달하는 payload.
    Backend 측 CourseVerification row 에 필요한 컨텍스트(사용자/코스/영화)와
    판정에 쓰일 원본 텍스트 2개(리뷰 본문, 영화 줄거리)를 한 번에 담는다.
    """

    verification_id: int = Field(..., description="course_verification PK — 판정 결과 upsert 대상")
    user_id: str = Field(default="", description="리뷰 작성자 user_id")
    course_id: str = Field(default="", description="도장깨기 코스 ID")
    movie_id: str = Field(default="", description="영화 ID")
    review_id: Optional[int] = Field(default=None, description="course_review PK (로깅용)")
    review_text: str = Field(default="", description="사용자가 작성한 리뷰 본문 (원문)")
    movie_plot: str = Field(default="", description="비교 기준이 될 영화 줄거리/시놉시스")

    @model_validator(mode="before")
    @classmethod
    def _normalize_keys(cls, data: Any) -> Any:
        """camelCase → snake_case 변환 + null 문자열 필드를 빈 문자열로 정규화."""
        if not isinstance(data, dict):
            return data
        logger.info("review_verification_raw_body", keys=list(data.keys()), data=data)
        mapping = {
            "verificationId": "verification_id",
            "userId":         "user_id",
            "courseId":       "course_id",
            "movieId":        "movie_id",
            "reviewId":       "review_id",
            "reviewText":     "review_text",
            "moviePlot":      "movie_plot",
        }
        for camel, snake in mapping.items():
            if camel in data and snake not in data:
                data[snake] = data.pop(camel)
        # Java null String → 빈 문자열
        for str_field in ("user_id", "course_id", "movie_id", "review_text", "movie_plot"):
            if data.get(str_field) is None:
                data[str_field] = ""
        return data


class ReviewVerificationResponse(BaseModel):
    """
    리뷰 인증 에이전트 판정 결과.

    Backend 는 이 응답을 받아 CourseVerification.applyAiDecision() 을 호출하여
    similarity_score / matched_keywords / ai_confidence / review_status / decision_reason
    을 세팅한다. is_verified 전환은 review_status=AUTO_VERIFIED 일 때만 수행된다.
    """

    verification_id: int = Field(..., description="입력과 동일한 PK (멱등성 체크용)")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="영화 줄거리 ↔ 리뷰 본문 유사도 (임베딩 코사인/BM25/하이브리드)",
    )
    matched_keywords: list[str] = Field(
        default_factory=list,
        description="영화 줄거리와 리뷰에서 공통으로 추출된 핵심 키워드",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="종합 신뢰도 점수 (유사도 + 키워드 + 길이/어휘 다양성 가중합)",
    )
    review_status: str = Field(
        ...,
        description=(
            "판정 상태: AUTO_VERIFIED / NEEDS_REVIEW / AUTO_REJECTED. "
            "관리자 수동 오버라이드(ADMIN_APPROVED/ADMIN_REJECTED) 는 Backend 에서만 세팅된다."
        ),
    )
    rationale: str = Field(
        ...,
        description="판정 근거 한 줄 요약 (감사 로그 및 관리자 UI 에 노출)",
    )


@admin_router.post(
    "/ai/review-verification/verify",
    response_model=ReviewVerificationResponse,
    summary="도장깨기 리뷰 인증 AI 판정",
    description=(
        "영화 줄거리 ↔ 사용자 리뷰 유사도를 계산하여 시청 여부를 자동 인증한다. "
        "Solar 임베딩 유사도 + 키워드 매칭 + (선택) LLM 재검증 3단계 하이브리드 판정."
    ),
)
async def verify_course_review(request: ReviewVerificationRequest) -> ReviewVerificationResponse:
    """
    리뷰 인증 에이전트 엔드포인트.

    LangGraph ReviewVerificationGraph 를 실행하여 5단계로 판정한다:
        1) preprocessor         : 텍스트 정제, 20자 미만 조기 종료
        2) embedding_similarity : Solar 임베딩 코사인 유사도
        3) keyword_matcher      : 한국어 명사 교집합 키워드 추출
        4) llm_revalidator      : confidence_draft 0.5~0.8 구간에서만 LLM 재검증
        5) threshold_decider    : 임계값 기준 AUTO_VERIFIED/NEEDS_REVIEW/AUTO_REJECTED
    """
    from monglepick.agents.review_verification.graph import review_verification_graph

    logger.info(
        "review_verification_start",
        verification_id=request.verification_id,
        movie_id=request.movie_id,
        review_len=len(request.review_text),
    )

    try:
        initial_state = {
            "verification_id": request.verification_id,
            "user_id":         request.user_id,
            "course_id":       request.course_id,
            "movie_id":        request.movie_id,
            "review_id":       request.review_id,
            "review_text":     request.review_text,
            "movie_plot":      request.movie_plot,
        }

        result = await review_verification_graph.ainvoke(initial_state)

        logger.info(
            "review_verification_complete",
            verification_id=request.verification_id,
            review_status=result.get("review_status"),
            confidence=round(result.get("confidence", 0.0), 4),
        )

        return ReviewVerificationResponse(
            verification_id=result["verification_id"],
            similarity_score=result.get("similarity_score", 0.0),
            matched_keywords=result.get("matched_keywords", []),
            confidence=result.get("confidence", 0.0),
            review_status=result.get("review_status", "NEEDS_REVIEW"),
            rationale=result.get("rationale", ""),
        )

    except Exception as e:
        logger.error(
            "review_verification_failed",
            verification_id=request.verification_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "review_verification_agent_error",
                "message": f"AI 리뷰 검증 중 오류가 발생했습니다: {str(e)}",
                "verification_id": request.verification_id,
            },
        )
