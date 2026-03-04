"""
프로젝트 설정 (pydantic-settings 기반).

§17 Phase 0: 27개 환경 변수 + Ollama 로컬 LLM 설정 추가.
.env 파일에서 환경 변수를 로드한다.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    NEO4J_PASSWORD: str = "monglepick_dev"

    # ── MySQL ──
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_DATABASE: str = "monglepick"
    MYSQL_USER: str = "monglepick"
    MYSQL_PASSWORD: str = "monglepick_dev"

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
    VISION_TIMEOUT_SEC: int = 180

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

    # ── Security ──
    SERVICE_API_KEY: str = ""
    DAILY_TOKEN_LIMIT: int = 1_000_000


settings = Settings()
