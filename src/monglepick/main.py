"""
몽글픽 AI Agent FastAPI 앱.

Phase 3: lifespan 추가 (5개 DB 클라이언트 초기화/종료), chat_router 등록.
LangSmith: LANGCHAIN_API_KEY 설정 시 LLM 호출/그래프 실행 자동 트레이싱.
Ollama Warmup: 앱 시작 시 두 모델(qwen3.5, exaone-32b)에 dummy 호출하여 첫 요청 cold start 제거.
"""

import asyncio
import os
import time
import traceback
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

# ── 로깅 초기화 (다른 모듈 import 이전에 반드시 실행) ──
# JSON 포맷/stdout 출력 → docker json-file → Filebeat → Logstash agent_json 분기
from monglepick.utils.logging_config import configure_logging

configure_logging()

from monglepick.api.admin import admin_router
from monglepick.api.admin_assistant import admin_assistant_router
from monglepick.api.admin_data import admin_data_router
from monglepick.api.chat import chat_router
from monglepick.api.match import match_router
from monglepick.api.middleware import RateLimitMiddleware, TimeoutMiddleware
from monglepick.api.router import api_router
from monglepick.config import settings
from monglepick.db.clients import close_all_clients, init_all_clients

logger = structlog.get_logger()

# ── LangSmith 트레이싱 자동 활성화 ──
# LANGCHAIN_API_KEY가 설정되어 있으면 LangChain/LangGraph의 모든 LLM 호출과
# 그래프 노드 실행을 LangSmith 대시보드에 자동 추적한다.
# 환경변수 방식이므로 코드 내 콜백 등록 없이 자동 동작한다.
if settings.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    logger.info(
        "langsmith_tracing_enabled",
        project=settings.LANGCHAIN_PROJECT,
        endpoint=settings.LANGCHAIN_ENDPOINT,
    )

# 앱 버전
APP_VERSION = "0.3.0"


async def _warmup_ollama_models() -> None:
    """
    Ollama 모델에 dummy 호출을 병렬 수행하여 첫 API 요청의 cold start를 제거한다.

    Ollama 서버가 모델을 처음 로드할 때 시간이 소요될 수 있다.
    앱 시작 시 짧은 dummy 호출을 수행하면 사용자 첫 요청 전에
    모델이 완전히 준비된다.

    [W-A3 개선] asyncio.gather()로 모든 모델을 병렬 warmup하여
    기존 순차 실행(최대 N×120초) 대비 시작 시간을 대폭 단축한다.
    - 기존: 2모델 × 120초 = 최대 240초 (순차)
    - 개선: max(120초, 120초) = 최대 120초 (병렬)

    대상 모델:
    1. qwen3.5:35b-a3b (의도+감정 분류, 이미지 분석)
    2. exaone-32b:latest (선호 추출, 대화, 추천 이유)

    warmup 실패 시에도 앱은 정상 기동한다 (에러 전파 금지).
    """
    from langchain_core.messages import HumanMessage

    from monglepick.llm.factory import get_llm

    # 사전 로드할 모델 목록 (중복 제거)
    # Ollama 단일 서버에 dummy 호출을 수행하여 모델을 로드한다.
    models_to_warmup: list[str] = []
    seen: set[str] = set()
    for model_name in [settings.INTENT_MODEL, settings.CONVERSATION_MODEL]:
        if model_name not in seen:
            models_to_warmup.append(model_name)
            seen.add(model_name)

    logger.info(
        "ollama_warmup_start",
        models=models_to_warmup,
        mode="parallel",
    )

    async def _warmup_single(model_name: str) -> None:
        """
        단일 모델의 warmup을 수행한다.

        temperature=0.1, num_predict=1로 최소 토큰만 생성하여 빠르게 완료.
        타임아웃(120초) 초과 또는 기타 예외 발생 시 경고 로그만 남기고 진행한다.

        Args:
            model_name: Ollama 모델명 (예: "qwen3.5:35b-a3b")
        """
        warmup_start = time.perf_counter()
        try:
            llm = get_llm(model=model_name, temperature=0.1, num_predict=1)
            await asyncio.wait_for(
                llm.ainvoke([HumanMessage(content="ping")]),
                timeout=120.0,  # 최대 2분 대기
            )
            elapsed_sec = time.perf_counter() - warmup_start
            logger.info(
                "ollama_model_warmed_up",
                model=model_name,
                elapsed_sec=round(elapsed_sec, 1),
            )
        except asyncio.TimeoutError:
            elapsed_sec = time.perf_counter() - warmup_start
            logger.warning(
                "ollama_warmup_timeout",
                model=model_name,
                elapsed_sec=round(elapsed_sec, 1),
                timeout_sec=120,
            )
        except Exception as e:
            elapsed_sec = time.perf_counter() - warmup_start
            # Ollama 서버 미실행 등 — warmup 실패해도 앱은 정상 기동
            logger.warning(
                "ollama_warmup_failed",
                model=model_name,
                error=str(e),
                error_type=type(e).__name__,
                elapsed_sec=round(elapsed_sec, 1),
            )

    # [W-A3] asyncio.gather()로 모든 모델을 병렬 warmup.
    # return_exceptions=True: 개별 모델 실패가 다른 모델 warmup에 영향을 주지 않는다.
    await asyncio.gather(
        *[_warmup_single(m) for m in models_to_warmup],
        return_exceptions=True,
    )

    logger.info("ollama_warmup_complete", models=models_to_warmup)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱 라이프사이클 관리.

    startup: 5개 DB 클라이언트 초기화 (Qdrant/Neo4j/Redis/ES/MySQL)
    shutdown: 모든 DB 연결 정리
    """
    # ── Startup ──
    startup_start = time.perf_counter()
    logger.info("app_startup", version=APP_VERSION)

    # ── [1] 5개 DB 클라이언트 초기화 ──
    try:
        await init_all_clients()
        db_elapsed_ms = (time.perf_counter() - startup_start) * 1000
        logger.info("db_clients_initialized", elapsed_ms=round(db_elapsed_ms, 1))
    except Exception as e:
        # DB 연결 실패 시에도 앱은 기동 (health 엔드포인트는 동작)
        db_elapsed_ms = (time.perf_counter() - startup_start) * 1000
        logger.error(
            "app_startup_db_error", error=str(e), error_type=type(e).__name__,
            stack_trace=traceback.format_exc(), elapsed_ms=round(db_elapsed_ms, 1),
        )

    # ── [1.5] 보안 설정 점검 ──
    # JWT_SECRET 미설정 시 경고: 프로덕션에서는 JWT 검증이 비활성화됨
    if not settings.JWT_SECRET:
        logger.warning(
            "jwt_secret_not_configured",
            message="JWT_SECRET이 설정되지 않았습니다. JWT 검증이 비활성화됩니다. (개발 환경 전용)",
        )

    # ── [2] Ollama 모델 warmup ──
    # local_only 모드에서만 실행 (hybrid/api_only에서는 Ollama 미사용)
    if settings.LLM_MODE == "local_only":
        await _warmup_ollama_models()
    else:
        logger.info("ollama_warmup_skipped", mode=settings.LLM_MODE, reason="Ollama 미사용")

    # ── [3] Admin Assistant Checkpointer setup ──
    # ADMIN_REDIS_CHECKPOINTER_ENABLED=true 면 Redis Search 인덱스 생성.
    # MemorySaver 면 no-op. 실패해도 앱 기동 차단 X (Agent 는 체크포인트 없이도 동작).
    try:
        from monglepick.agents.admin_assistant.graph import setup_admin_assistant_checkpointer
        await setup_admin_assistant_checkpointer()
    except Exception as e:
        logger.error(
            "admin_checkpointer_setup_unexpected_error",
            error=str(e), error_type=type(e).__name__,
        )

    # ── [3.5] Support Assistant Checkpointer setup ──
    # SUPPORT_REDIS_CHECKPOINTER_ENABLED=true 면 Redis Search 인덱스 생성.
    # MemorySaver(기본) 면 no-op. 실패해도 앱 기동 차단 X.
    try:
        from monglepick.agents.support_assistant.graph import setup_support_assistant_checkpointer
        await setup_support_assistant_checkpointer()
    except Exception as e:
        logger.error(
            "support_checkpointer_setup_unexpected_error",
            error=str(e), error_type=type(e).__name__,
        )

    startup_elapsed_ms = (time.perf_counter() - startup_start) * 1000
    logger.info("app_startup_complete", version=APP_VERSION, elapsed_ms=round(startup_elapsed_ms, 1))

    yield

    # ── Shutdown ──
    shutdown_start = time.perf_counter()
    logger.info("app_shutdown")
    # httpx 클라이언트 정리 (C-2: 리소스 누수 방지)
    from monglepick.api.admin_backend_client import close_admin_client
    from monglepick.api.point_client import close_client
    await close_client()
    await close_admin_client()
    await close_all_clients()
    shutdown_elapsed_ms = (time.perf_counter() - shutdown_start) * 1000
    logger.info("app_shutdown_complete", elapsed_ms=round(shutdown_elapsed_ms, 1))


# ── OpenAPI 태그 메타데이터 ──
# Swagger UI에서 엔드포인트를 태그별로 그룹화하여 표시한다.
openapi_tags = [
    {
        "name": "chat",
        "description": "영화 추천 채팅 API. SSE 스트리밍, 동기 JSON, 이미지 업로드 3가지 방식을 지원한다.",
    },
    {
        "name": "match",
        "description": "Movie Match API. 두 영화 교집합 기반 함께 볼 영화 추천. SSE 스트리밍 및 동기 JSON 지원.",
    },
    {
        "name": "system",
        "description": "서버 상태 확인용 시스템 엔드포인트.",
    },
    {
        "name": "admin_assistant",
        "description": (
            "관리자 AI 어시스턴트 — 자연어로 관리자 기능(조회/CRUD/통계/보고서)을 구동. "
            "Step 1(2026-04-23): Intent 분류 + smalltalk 응답만 구현. "
            "query/action/stats/report 는 후속 Step 에서 Tool RAG 로 확장."
        ),
    },
]

app = FastAPI(
    title="몽글픽 AI Agent",
    description=(
        "## 영화 추천 AI 에이전트 API\n\n"
        "LangGraph 기반 대화형 영화 추천 서비스.\n\n"
        "### 주요 기능\n"
        "- **대화형 추천**: 사용자의 감정·취향을 분석하여 맞춤 영화 추천\n"
        "- **이미지 분석**: 영화 포스터·분위기 사진 업로드 시 VLM 기반 추천\n"
        "- **하이브리드 검색**: Qdrant(벡터) + Elasticsearch(BM25) + Neo4j(그래프) RRF 합산\n"
        "- **추천 엔진**: CF + CBF 하이브리드, Cold Start 처리, MMR 다양성 재정렬\n\n"
        "### 데이터 규모\n"
        "- 영화 910,140편 (TMDB + Kaggle + KOBIS + KMDb)\n"
        "- 시청 이력 26M건, 유저 270K명\n\n"
        "### 기술 스택\n"
        "LangGraph · Ollama (EXAONE 4.0 / Qwen 3.5) · Upstage Solar 임베딩 · "
        "Qdrant · Neo4j · Elasticsearch · Redis · MySQL\n\n"
        "### ⚠️ Swagger UI 테스트 안내\n"
        "- **SSE 엔드포인트** (`/chat`, `/match`)는 Swagger UI에서 "
        "Try it out 시 응답이 끝나지 않습니다 (text/event-stream). "
        "대신 **`/chat/sync`**, **`/match/sync`** 동기 엔드포인트를 사용하세요.\n"
        "- 인증이 필요한 엔드포인트는 Authorization 헤더에 Bearer JWT 토큰을 입력하세요."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    openapi_tags=openapi_tags,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "몽글픽 팀",
    },
    license_info={
        "name": "MIT",
    },
)

# ── CORS 설정 ──
# settings.CORS_ALLOWED_ORIGINS가 "*"이면 전체 허용 (개발 환경),
# 그 외에는 쉼표 구분 오리진 목록만 허용 (프로덕션 환경).
_cors_origins_raw = settings.CORS_ALLOWED_ORIGINS.strip()
if _cors_origins_raw == "*":
    _cors_origins = ["*"]
else:
    _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── 전역 미들웨어 등록 ──
# Starlette 미들웨어는 등록 역순으로 실행된다.
# 실행 순서: TimeoutMiddleware → RateLimitMiddleware → CORS → 엔드포인트
#
# TimeoutMiddleware를 가장 안쪽(나중에 등록)에 두어
# Rate Limit 체크를 통과한 요청에만 타임아웃을 적용한다.
# RateLimitMiddleware는 바깥쪽(먼저 등록)에 두어
# 타임아웃 리소스 소모 전에 과도한 요청을 차단한다.
#
# 비인증: 분당 RATE_LIMIT_ANON_RPM 회 (기본 30) /
# 인증:   분당 RATE_LIMIT_AUTH_RPM 회 (기본 60) — config.py 참조
app.add_middleware(TimeoutMiddleware)
app.add_middleware(RateLimitMiddleware)

# API 라우터 등록
app.include_router(api_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(match_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
# Phase 6 (2026-04-08): 데이터 관리 admin 라우터 등록 — 영화 CRUD + 파이프라인 SSE + 데이터 현황
app.include_router(admin_data_router, prefix="/api/v1")
# 관리자 AI 어시스턴트 (2026-04-23, Step 1):
# 자연어로 관리자 기능(조회/CRUD/통계/보고서)을 구동하는 LangGraph 에이전트.
# 현재는 Intent 분류 + smalltalk 응답만 구현되어 있고, query/action/stats/report 는
# 후속 Step 에서 Tool RAG + tool-calling 으로 확장된다.
# 설계서: docs/관리자_AI에이전트_설계서.md (v2.0)
app.include_router(admin_assistant_router, prefix="/api/v1")

# ── Prometheus 메트릭 엔드포인트 ──
# prometheus-fastapi-instrumentator 가 /metrics 경로에 자동으로
# http_requests_total, http_request_duration_seconds, http_request_size_bytes 등
# 표준 메트릭을 노출한다. LangGraph 노드 커스텀 메트릭은 Phase 3에서 structlog → OpenTelemetry 로 확장 예정.
#
# 제외 경로:
#  - /health, /metrics 자체는 Prometheus 스크레이프 잡음이 되므로 제외
#  - /docs, /redoc, /openapi.json 은 Swagger UI 로드 시 노이즈
#
# should_respect_env_var: `ENABLE_METRICS` 환경변수가 "true" 가 아니면 비활성화되도록
# 설정할 수 있으나, 운영 VM3 에서 상시 스크레이프하므로 기본 활성화로 둔다.
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/health", "/metrics", "/docs", "/redoc", "/openapi.json"],
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True,
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False, tags=["system"])


@app.get(
    "/health",
    tags=["system"],
    summary="헬스 체크",
    response_description="서버 상태 및 연결된 DB 설정 정보",
    responses={
        200: {
            "description": "서버 정상 동작 중",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "version": "0.2.0",
                        "settings": {
                            "qdrant_url": "http://localhost:6333",
                            "redis_url": "redis://localhost:6379/0",
                            "elasticsearch_url": "http://localhost:9200",
                            "neo4j_uri": "bolt://localhost:7687",
                            "embedding_model": "solar-embedding-1-large-passage",
                        },
                    }
                }
            },
        }
    },
)
async def health_check():
    """헬스 체크 엔드포인트. 서버 상태와 연결된 DB 설정 정보를 반환한다."""
    return {
        "status": "ok",
        "version": APP_VERSION,
        "settings": {
            "qdrant_url": settings.QDRANT_URL,
            "redis_url": settings.REDIS_URL,
            "elasticsearch_url": settings.ELASTICSEARCH_URL,
            "neo4j_uri": settings.NEO4J_URI,
            "embedding_model": settings.EMBEDDING_MODEL,
        },
    }
