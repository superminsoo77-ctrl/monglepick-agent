"""
몽글픽 AI Agent FastAPI 앱.

Phase 3: lifespan 추가 (5개 DB 클라이언트 초기화/종료), chat_router 등록.
LangSmith: LANGCHAIN_API_KEY 설정 시 LLM 호출/그래프 실행 자동 트레이싱.
"""

import os
import time
import traceback
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from monglepick.api.chat import chat_router
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
APP_VERSION = "0.2.0"


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
    try:
        await init_all_clients()
        startup_elapsed_ms = (time.perf_counter() - startup_start) * 1000
        logger.info("app_startup_complete", version=APP_VERSION, elapsed_ms=round(startup_elapsed_ms, 1))
    except Exception as e:
        # DB 연결 실패 시에도 앱은 기동 (health 엔드포인트는 동작)
        startup_elapsed_ms = (time.perf_counter() - startup_start) * 1000
        logger.error(
            "app_startup_db_error", error=str(e), error_type=type(e).__name__,
            stack_trace=traceback.format_exc(), elapsed_ms=round(startup_elapsed_ms, 1),
        )

    yield

    # ── Shutdown ──
    shutdown_start = time.perf_counter()
    logger.info("app_shutdown")
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
        "name": "system",
        "description": "서버 상태 확인용 시스템 엔드포인트.",
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
        "- 영화 157,194편 (TMDB + Kaggle + KOBIS + KMDb)\n"
        "- 시청 이력 26M건, 유저 270K명\n\n"
        "### 기술 스택\n"
        "LangGraph · Ollama (EXAONE 4.0 / Qwen 3.5) · Upstage Solar 임베딩 · "
        "Qdrant · Neo4j · Elasticsearch · Redis · MySQL"
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(api_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


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
