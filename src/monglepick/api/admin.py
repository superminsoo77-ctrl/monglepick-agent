"""
관리자 전용 API 라우터.

시스템 탭:
- GET /api/v1/admin/system/db — 5개 DB(MySQL/Qdrant/Neo4j/ES/Redis) 상태 조회
- GET /api/v1/admin/system/ollama — Ollama 모델 로드 상태 조회
- GET /api/v1/admin/system/vllm   — vLLM 모델 상태 조회 (Chat/Vision 2개 엔드포인트)

AI 운영 탭:
- POST /api/v1/admin/ai/quiz/generate — LLM 기반 영화 퀴즈 자동 생성 + quizzes 테이블 PENDING INSERT
"""

import json
import re
import time
import traceback
from typing import Any, Optional

import httpx
import structlog
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from monglepick.config import settings
from monglepick.db.clients import (
    get_elasticsearch,
    get_mysql,
    get_neo4j,
    get_qdrant,
    get_redis,
)
from monglepick.llm import get_conversation_llm, guarded_ainvoke

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
# POST /admin/ai/quiz/generate — LLM 기반 영화 퀴즈 자동 생성
# ============================================================
#
# 관리자 페이지 "AI 운영 → 퀴즈 생성" 카드가 호출하는 엔드포인트.
# 기존에는 Backend(Spring Boot)가 관리자 입력값을 그대로 quizzes 테이블에 INSERT 하는
# 스텁 경로를 사용했으나, 2026-04-08부로 Agent(FastAPI)가 LLM을 이용해 실제 문항을
# 자동 생성하고 PENDING 상태로 저장하는 경로로 전환한다.
#
# 설계 결정:
# - 입력: {genre?, difficulty?, count?} (AiTriggerPanel 폼과 일치)
# - 1) MySQL에서 count 편 수의 후보 영화 샘플링 (genre LIKE 필터)
# - 2) 영화마다 LLM 호출하여 4지선다 퀴즈 1개 생성
# - 3) LLM 실패 시 장르/개봉연도 기반 fallback 템플릿 사용
# - 4) quizzes 테이블에 status=PENDING 으로 직접 INSERT (Agent도 MySQL read/write 권한 보유)
# - 5) 관리자는 이후 Backend의 quiz_history / approve 엔드포인트로 검수한다
# - 빈 DB 대응: 영화가 한 편도 없을 경우 success=False, count=0 로 반환 (500 금지)


class GenerateQuizRequest(BaseModel):
    """
    AI 퀴즈 자동 생성 요청 DTO.

    AiTriggerPanel(monglepick-admin) 폼과 일치하는 필드만 받는다.
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


# ──────────────────────────────────────────────────────────────
# 내부 유틸: JSON 파서, fallback 생성기, 영화 조회, LLM 체인
# ──────────────────────────────────────────────────────────────

# LLM에 전달할 퀴즈 생성 시스템 프롬프트 (단일 영화/단일 문항용).
# 로드맵 에이전트(agents/roadmap/nodes.py)의 _QUIZ_SYSTEM_PROMPT를 참고하되,
# "객관식 1문항" 스키마로 단순화하여 Backend 엔티티 컬럼과 1:1 매핑한다.
_ADMIN_QUIZ_SYSTEM_PROMPT = """당신은 영화 교육 퀴즈 전문가입니다.
주어진 영화 한 편에 대해 객관식 4지선다 퀴즈 1문항을 생성하세요.

반환 형식(JSON 객체만 출력, 설명/마크다운 금지):
{
  "question": "질문 텍스트",
  "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
  "correctAnswer": "정답 선택지(options 배열 중 하나와 정확히 일치)",
  "explanation": "정답 해설(1~2문장)"
}

규칙:
- 스포일러 절대 금지 (결말/반전/인물 사망 등 핵심 내용 언급 불가)
- 객관식 선택지는 반드시 4개
- 오답 선택지는 그럴듯하되 사실과 다르게 구성
- correctAnswer 는 options 배열에 존재하는 문자열과 100% 동일해야 함
- 난이도는 '{difficulty}' 수준으로 맞출 것
- 순수 JSON 객체만 출력 (마크다운 코드블록, 설명 문구 금지)"""


def _parse_quiz_json(text: str) -> dict:
    """
    LLM 응답에서 퀴즈 JSON 객체를 추출한다.

    마크다운 코드블록(```json ... ```)을 제거하고 파싱한다.
    실패 시 빈 dict 반환.
    """
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
        cleaned = cleaned.rstrip("`").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 중괄호 블록 추출 시도
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        logger.warning("admin_quiz_json_parse_failed", preview=text[:200])
        return {}


def _make_admin_fallback_quiz(movie: dict) -> dict:
    """
    LLM 생성 실패 시 fallback 퀴즈 1문항을 장르/연도 기반 템플릿으로 구성한다.

    Args:
        movie: {"id", "title", "genres": list, "release_year"} 형태의 dict

    Returns:
        {question, options, correctAnswer, explanation} dict
    """
    title = movie.get("title", "이 영화")
    genres: list[str] = movie.get("genres") or ["드라마"]
    main_genre = genres[0] if genres else "드라마"

    decoy_pool = [
        "액션", "드라마", "코미디", "공포", "SF",
        "스릴러", "로맨스", "판타지", "애니메이션",
    ]
    decoys = [g for g in decoy_pool if g != main_genre][:3]
    options = [main_genre] + decoys

    return {
        "question": f"'{title}' 영화의 주요 장르는 무엇인가요?",
        "options": options,
        "correctAnswer": main_genre,
        "explanation": (
            f"'{title}'은(는) '{main_genre}' 장르의 대표적인 작품 중 하나입니다. "
            f"포스터·로그라인·장르 태그를 통해 확인할 수 있습니다."
        ),
    }


async def _sample_movies_for_quiz(
    genre: Optional[str],
    count: int,
) -> list[dict]:
    """
    quizzes 생성용 후보 영화를 MySQL 에서 샘플링한다.

    - 장르 필터가 있으면 genres 컬럼 LIKE 매칭
    - popularity 상위 1000편 중 RAND() 로 count 편 랜덤 추출
    - 빈 DB 환경(행 0개)에서도 빈 리스트를 안전하게 반환

    Args:
        genre: 장르 필터 (None/빈 문자열 이면 전체)
        count: 추출할 편수

    Returns:
        [{"id","title","genres": list, "release_year"}] 형태의 dict 리스트
    """
    movies: list[dict] = []
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # 장르 LIKE 필터 파라미터 구성
                if genre:
                    sql = (
                        "SELECT movie_id, title, original_title, genres, release_date "
                        "FROM movies "
                        "WHERE genres LIKE %s "
                        "ORDER BY RAND() "
                        "LIMIT %s"
                    )
                    params: tuple = (f"%{genre}%", count)
                else:
                    # 장르 지정 없을 때는 전체 테이블에서 랜덤 추출.
                    # LIMIT 이 작으므로 full-scan RAND() 비용은 수용 가능.
                    sql = (
                        "SELECT movie_id, title, original_title, genres, release_date "
                        "FROM movies "
                        "ORDER BY RAND() "
                        "LIMIT %s"
                    )
                    params = (count,)

                await cur.execute(sql, params)
                rows = await cur.fetchall()

        for row in rows:
            movie_id, title, original_title, genres_raw, release_date = row

            # 장르 파싱: JSON 배열 문자열 또는 쉼표 구분
            parsed_genres: list[str] = []
            if genres_raw:
                try:
                    g = json.loads(genres_raw)
                    if isinstance(g, list):
                        parsed_genres = [str(x) for x in g]
                except (json.JSONDecodeError, TypeError):
                    parsed_genres = [
                        s.strip() for s in str(genres_raw).split(",") if s.strip()
                    ]

            # 연도 추출
            release_year = ""
            if release_date:
                try:
                    release_year = str(release_date)[:4]
                except Exception:
                    release_year = ""

            movies.append({
                "id": str(movie_id),
                "title": title or original_title or "(제목 없음)",
                "genres": parsed_genres,
                "release_year": release_year,
            })

    except Exception as e:
        logger.error("admin_quiz_sample_movies_failed", error=str(e))

    return movies


async def _generate_single_quiz_llm(movie: dict, difficulty: str) -> dict:
    """
    단일 영화에 대해 LLM 으로 4지선다 퀴즈 1문항을 생성한다.

    LLM 실패 또는 파싱 실패 시 fallback 템플릿으로 돌린다.

    Args:
        movie:      {"id","title","genres","release_year"} dict
        difficulty: 난이도 힌트 (easy/medium/hard)

    Returns:
        {question, options, correctAnswer, explanation} dict
    """
    title = movie.get("title", "")
    genres = ", ".join(movie.get("genres") or [])
    release_year = movie.get("release_year") or "미상"

    user_prompt = (
        f"다음 영화에 대해 객관식 4지선다 퀴즈 1문항을 만들어 주세요.\n\n"
        f"- 제목: {title}\n"
        f"- 장르: {genres or '정보 없음'}\n"
        f"- 개봉연도: {release_year}\n\n"
        f"위 규칙을 모두 지키고 JSON 객체로만 응답하세요."
    )

    try:
        llm = get_conversation_llm()
        messages = [
            SystemMessage(content=_ADMIN_QUIZ_SYSTEM_PROMPT.format(difficulty=difficulty)),
            HumanMessage(content=user_prompt),
        ]
        response = await guarded_ainvoke(llm, messages)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )
        parsed = _parse_quiz_json(response_text)

        # 스키마 검증: 필수 필드 존재 + options 4개 + correctAnswer 일치
        if (
            isinstance(parsed, dict)
            and parsed.get("question")
            and isinstance(parsed.get("options"), list)
            and len(parsed["options"]) == 4
            and parsed.get("correctAnswer") in parsed["options"]
        ):
            return {
                "question": str(parsed["question"]),
                "options": [str(o) for o in parsed["options"]],
                "correctAnswer": str(parsed["correctAnswer"]),
                "explanation": str(parsed.get("explanation") or ""),
            }

        logger.warning(
            "admin_quiz_llm_invalid_schema",
            movie_id=movie.get("id"),
            parsed_keys=list(parsed.keys()) if isinstance(parsed, dict) else None,
        )
    except Exception as e:
        logger.warning(
            "admin_quiz_llm_failed",
            movie_id=movie.get("id"),
            error=str(e),
        )

    # fallback
    return _make_admin_fallback_quiz(movie)


async def _insert_quiz_pending(
    movie_id: str,
    question: str,
    correct_answer: str,
    options: list[str],
    explanation: Optional[str],
    reward_point: int = 10,
) -> int:
    """
    생성된 퀴즈 1건을 quizzes 테이블에 PENDING 상태로 INSERT 하고 새 PK 를 반환한다.

    Backend 의 Quiz 엔티티와 동일한 컬럼 구성이며, created_at 은 NOW() 로 세팅한다.
    created_by 는 'ai-agent' 문자열을 박아 넣어 운영 로그에서 구분할 수 있게 한다.

    Args:
        movie_id:       movies.movie_id (VARCHAR(50))
        question:       퀴즈 문제
        correct_answer: 정답 문자열
        options:        선택지 리스트 (JSON 으로 직렬화하여 저장)
        explanation:    해설 (nullable)
        reward_point:   보상 포인트 (기본 10)

    Returns:
        신규 quiz_id (BIGINT)
    """
    options_json = json.dumps(options, ensure_ascii=False)
    pool = await get_mysql()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            # Backend(Quiz 엔티티)와 동일 컬럼. status 는 'PENDING' 고정.
            # created_by/updated_by 는 운영 식별용으로 'ai-agent' 저장.
            await cur.execute(
                """
                INSERT INTO quizzes (
                    movie_id, question, explanation, correct_answer,
                    options, reward_point, status, quiz_date,
                    created_at, updated_at, created_by, updated_by
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, 'PENDING', NULL,
                    NOW(), NOW(), 'ai-agent', 'ai-agent'
                )
                """,
                (
                    movie_id,
                    question,
                    explanation,
                    correct_answer,
                    options_json,
                    reward_point,
                ),
            )
            await conn.commit()
            # LAST_INSERT_ID() — aiomysql 드라이버는 cursor.lastrowid 로 직접 제공
            return int(cur.lastrowid or 0)


@admin_router.post(
    "/ai/quiz/generate",
    response_model=GenerateQuizResponse,
    summary="AI 퀴즈 자동 생성 (LLM)",
    description=(
        "LLM 을 사용해 영화 퀴즈를 자동 생성하고 quizzes 테이블에 "
        "PENDING 상태로 저장한다. 관리자 검수 후 APPROVED/PUBLISHED 로 전환한다."
    ),
)
async def generate_admin_quiz(request: GenerateQuizRequest) -> GenerateQuizResponse:
    """
    관리자 AI 퀴즈 자동 생성 핸들러.

    흐름:
        1) MySQL 에서 후보 영화 count 편 샘플링 (장르 필터 선택)
        2) 각 영화마다 LLM 호출 또는 fallback 으로 4지선다 퀴즈 1문항 생성
        3) quizzes 테이블에 PENDING 상태로 INSERT
        4) 생성 결과를 배열로 반환 (quizId 포함)

    빈 DB / 장르 매칭 실패 시 success=False, count=0 로 반환하여
    UI 가 에러가 아닌 안내 메시지를 표시할 수 있게 한다.
    """
    logger.info(
        "admin_quiz_generate_start",
        genre=request.genre,
        difficulty=request.difficulty,
        count=request.count,
    )

    # ── 1) 후보 영화 샘플링 ──
    movies = await _sample_movies_for_quiz(
        genre=(request.genre or "").strip() or None,
        count=request.count,
    )

    if not movies:
        logger.warning(
            "admin_quiz_generate_no_movies",
            genre=request.genre,
        )
        # 빈 DB 또는 장르 매칭 실패 — UI 에 안내 메시지로 표시
        return GenerateQuizResponse(
            success=False,
            count=0,
            message=(
                "후보 영화가 없습니다. 영화 데이터를 먼저 적재하거나 "
                "장르 필터를 해제해 주세요."
            ),
            quizzes=[],
        )

    # ── 2~3) 영화마다 LLM 호출 → DB INSERT ──
    generated: list[GeneratedQuizItem] = []
    for movie in movies:
        try:
            quiz_body = await _generate_single_quiz_llm(movie, request.difficulty)

            quiz_id = await _insert_quiz_pending(
                movie_id=movie["id"],
                question=quiz_body["question"],
                correct_answer=quiz_body["correctAnswer"],
                options=quiz_body["options"],
                explanation=quiz_body.get("explanation") or None,
                reward_point=10,
            )

            generated.append(GeneratedQuizItem(
                quizId=quiz_id,
                movieId=movie["id"],
                movieTitle=movie["title"],
                question=quiz_body["question"],
                correctAnswer=quiz_body["correctAnswer"],
                options=quiz_body["options"],
                explanation=quiz_body.get("explanation") or None,
                rewardPoint=10,
                status="PENDING",
            ))
        except Exception as e:
            # 한 편 실패는 전체 실패로 전파하지 않고 로그만 남기고 다음으로.
            logger.error(
                "admin_quiz_generate_per_movie_failed",
                movie_id=movie.get("id"),
                error=str(e),
                trace=traceback.format_exc()[-500:],
            )

    count = len(generated)
    logger.info(
        "admin_quiz_generate_complete",
        requested=request.count,
        generated=count,
    )

    return GenerateQuizResponse(
        success=count > 0,
        count=count,
        message=(
            f"AI 가 퀴즈 {count}개를 생성하여 PENDING 으로 등록했습니다."
            if count > 0
            else "퀴즈 생성에 실패했습니다. 로그를 확인해 주세요."
        ),
        quizzes=generated,
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
    user_id: str = Field(..., description="리뷰 작성자 user_id")
    course_id: str = Field(..., description="도장깨기 코스 ID")
    movie_id: str = Field(..., description="영화 ID")
    review_id: Optional[int] = Field(
        None,
        description="reviews 테이블의 review_id (있다면 감사/로깅용). course_review 만 있는 경우 생략 가능.",
    )
    review_text: str = Field(..., description="사용자가 작성한 리뷰 본문 (원문)")
    movie_plot: str = Field(..., description="비교 기준이 될 영화 줄거리/시놉시스")


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
    summary="도장깨기 리뷰 인증 AI 판정 (미구현)",
    description=(
        "영화 줄거리 ↔ 사용자 리뷰 유사도를 계산하여 시청 여부를 자동 인증한다. "
        "현재는 계약만 정의되어 있고 실제 구현은 대기 중이다 — HTTP 503 을 반환한다."
    ),
)
async def verify_course_review(request: ReviewVerificationRequest) -> ReviewVerificationResponse:
    """
    리뷰 인증 에이전트 엔드포인트 — 현 시점은 placeholder.

    추후 구현 시 다음 단계를 수행할 예정:
        1) 영화 줄거리와 리뷰 본문을 Solar 임베딩으로 벡터화 → 코사인 유사도
        2) 동시 발생 키워드 추출 (명사 추출 + stop-word 제거 + BM25 상위 토큰)
        3) (선택) LLM 으로 "리뷰가 영화 내용에 관한 것인가" yes/no 재검증
        4) 임계값 (application.yml: app.ai.review-verification.threshold) 기준으로
           AUTO_VERIFIED / NEEDS_REVIEW / AUTO_REJECTED 분기
        5) Backend 의 CourseVerification.applyAiDecision() 를 호출하도록 결과 반환

    이 함수가 구현되기 전까지 Admin UI 의 "AI 재검증" 버튼은 상태만 PENDING 으로
    되돌리고 실제로는 이 엔드포인트를 호출하지 않는다 (Backend 측에서 단락).
    """
    logger.warning(
        "review_verification_agent_not_implemented",
        verification_id=request.verification_id,
        user_id=request.user_id,
    )
    # 503 Service Unavailable — "의도적으로 현재 가용하지 않음" 의 표준 코드.
    # Admin UI 는 이 응답 대신 agentAvailable=false 플래그(Backend 에서 단락)로 안내한다.
    raise HTTPException(
        status_code=503,
        detail={
            "error": "review_verification_agent_not_implemented",
            "message": (
                "AI 리뷰 검증 에이전트가 아직 구현되지 않았습니다. "
                "설계: docs/도장깨기_리뷰검증_에이전트_설계서.md"
            ),
            "verification_id": request.verification_id,
        },
    )
