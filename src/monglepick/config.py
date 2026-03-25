"""
프로젝트 설정 (pydantic-settings 기반).

§17 Phase 0: 27개 환경 변수 + Ollama 로컬 LLM 설정 추가.
.env 파일에서 환경 변수를 로드한다.
"""

import logging

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_config_logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── API Keys ──
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    UPSTAGE_API_KEY: str = ""
    TMDB_API_KEY: str = ""
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    KOBIS_API_KEY: str = ""
    KOBIS_BASE_URL: str = "http://www.kobis.or.kr/kobisopenapi/webservice/rest"
    KAKAO_API_KEY: str = ""

    # ── KMDb (한국영화 데이터베이스) ──
    KMDB_API_KEY: str = ""
    KMDB_BASE_URL: str = "http://api.koreafilm.or.kr/openapi-data2/wisenut/search_api/search_json2.jsp"

    # ── Ollama (로컬 LLM 서버) ──
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # 동시에 GPU에 로드할 수 있는 최대 모델 수.
    # Mac 64GB 통합 메모리에서 qwen3.5:35b-a3b + exaone-32b 2개 동시 로드 가능.
    # 1이면 모델 스왑이 발생하여 매 요청 30~90초 추가 지연.
    OLLAMA_MAX_LOADED_MODELS: int = 2

    # ── Qdrant ──
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "movies"

    # ── Redis ──
    REDIS_URL: str = "redis://localhost:6379"

    # ── Elasticsearch ──
    ELASTICSEARCH_URL: str = "http://localhost:9200"

    # ── Neo4j ──
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""  # .env 또는 환경변수로 설정 필수

    # ── MySQL ──
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DATABASE: str = "monglepick"
    MYSQL_USER: str = "monglepick"
    MYSQL_PASSWORD: str = ""  # .env 또는 환경변수로 설정 필수

    # ── Embedding (Upstage Solar) ──
    EMBEDDING_MODEL: str = "Upstage/solar-embedding-1-large"
    EMBEDDING_DIMENSION: int = 4096

    # ── LLM Models (Ollama 로컬 모델) ──
    # 구조화 출력 (JSON) + 비전: Qwen3.5 35B-A3B (텍스트 분류 + 이미지 분석 통합)
    INTENT_MODEL: str = "qwen3.5:35b-a3b"
    EMOTION_MODEL: str = "qwen3.5:35b-a3b"
    MOOD_MODEL: str = "qwen3.5:35b-a3b"
    # 한국어 자연어 생성: EXAONE 4.0 32B (비추론 모드: temperature < 0.6)
    PREFERENCE_MODEL: str = "exaone-32b:latest"
    CONVERSATION_MODEL: str = "exaone-32b:latest"
    # 경량 한국어 생성: EXAONE 4.0 32B (1.2B 미다운로드, 32B로 대체)
    QUESTION_MODEL: str = "exaone-32b:latest"
    # 추천 이유 생성: EXAONE 4.0 32B (자연어 설명)
    EXPLANATION_MODEL: str = "exaone-32b:latest"
    # Vision (이미지 분석): Qwen3.5 35B-A3B (비전 내장 멀티모달 모델)
    VISION_MODEL: str = "qwen3.5:35b-a3b"

    # ── Image Upload ──
    # 이미지 업로드 최대 크기 (MB)
    IMAGE_MAX_SIZE_MB: int = 10
    # 이미지 분석(VLM) 호출 타임아웃(초). 초과 시 분석 생략 후 진행
    VISION_TIMEOUT_SEC: int = 90
    # 이미지 리사이즈 최대 변 길이 (px). 긴 변이 이 값을 초과하면 비율 유지하여 축소
    IMAGE_MAX_DIMENSION: int = 1024

    # ── Session / Conversation ──
    SESSION_TTL_DAYS: int = 30
    MAX_CONVERSATION_TURNS: int = 20

    # ── LangSmith (LLM 관측성 + 트레이싱) ──
    # LangSmith 트레이싱 활성화 여부 (True이면 LLM 호출/그래프 실행 자동 추적)
    LANGCHAIN_TRACING_V2: bool = False
    # LangSmith API 키 (https://smith.langchain.com 에서 발급)
    LANGCHAIN_API_KEY: str = ""
    # LangSmith 프로젝트 이름 (대시보드에서 프로젝트별 분류)
    LANGCHAIN_PROJECT: str = "monglepick-agent"
    # LangSmith 엔드포인트 (기본값 사용, SaaS)
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # ── Point (포인트 시스템) ──
    # Spring Boot Backend 내부 URL (포인트 API 호출용)
    BACKEND_BASE_URL: str = "http://localhost:8080"
    # AI 추천 1회당 차감할 포인트 수 (init.sql point_items 기준 100P)
    POINT_COST_PER_RECOMMENDATION: int = 100
    # 포인트 체크 활성화 여부 (False이면 포인트 체크 생략 — 개발/테스트 환경)
    POINT_CHECK_ENABLED: bool = True
    # 회원가입 시 무료 포인트 지급 수
    FREE_POINTS_ON_SIGNUP: int = 5

    # ── Security ──
    SERVICE_API_KEY: str = ""
    # JWT 시크릿 키 (Backend application.yml의 app.jwt.secret과 동일해야 함)
    # Client → Agent 요청 시 Authorization: Bearer {JWT} 헤더의 user_id를 검증한다.
    # 미설정 시 JWT 검증을 건너뛰고 body의 user_id를 그대로 사용 (개발 환경용)
    JWT_SECRET: str = ""
    DAILY_TOKEN_LIMIT: int = 1_000_000
    # CORS 허용 오리진 (쉼표 구분). "*"이면 전체 허용 (개발 환경)
    CORS_ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8000"
    # 허용 이미지 MIME 타입 (쉼표 구분). 매직바이트 검증에도 사용
    ALLOWED_IMAGE_MIMES: str = "image/jpeg,image/png"
    # IP당 분당 최대 이미지 업로드 횟수. 초과 시 429 반환
    IMAGE_UPLOAD_RATE_LIMIT: int = 10
    # VLM 동시 처리 세마포어. GPU 메모리 보호용
    VLM_CONCURRENCY_LIMIT: int = 2
    # Pillow DecompressionBomb 방어: 최대 허용 픽셀 수 (25MP)
    IMAGE_MAX_PIXELS: int = 25_000_000

    # ── Concurrency Control (동시 요청 성능 최적화) ──
    # API 레벨: 동시 실행 가능한 최대 Chat Agent 그래프 수.
    # 초과 요청은 큐에 대기하며 SSE로 "대기 중" 알림을 전송한다.
    MAX_CONCURRENT_REQUESTS: int = 3
    # Ollama 모델별 동시 LLM 호출 제한.
    # Ollama는 GPU 추론을 모델당 직렬 처리하므로, 2 이상이면 큐 점유만 증가.
    # 1 = 활성 추론 1개 + 대기 1개 허용 (모델 스왑 방지)
    LLM_PER_MODEL_CONCURRENCY: int = 2
    # 추천 이유를 LLM으로 생성할 최대 영화 수.
    # 초과 영화는 메타데이터 기반 템플릿(_build_fallback_explanation)으로 대체하여
    # Ollama 큐 점유를 줄인다.
    MAX_EXPLANATION_MOVIES: int = 3


    @model_validator(mode='after')
    def _warn_empty_passwords(self):
        """비밀번호가 비어있으면 경고 로그를 출력한다. (W-1)"""
        if not self.NEO4J_PASSWORD:
            _config_logger.warning("NEO4J_PASSWORD가 설정되지 않았습니다. .env 파일을 확인하세요.")
        if not self.MYSQL_PASSWORD:
            _config_logger.warning("MYSQL_PASSWORD가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return self


settings = Settings()
