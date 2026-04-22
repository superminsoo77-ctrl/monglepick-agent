"""
프로젝트 설정 (pydantic-settings 기반).

§17 Phase 0: 27개 환경 변수 + Ollama(로컬 LLM) 설정.
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
    # 백업 키. 메인 키 (UPSTAGE_API_KEY) 가 401/quota_exceeded 로 실패할 때
    # 새 작업 (Phase 2~9 / scripts/*) 이 자동 fallback 으로 사용한다.
    # Task #5 (이미 메모리에 메인 키 보유) 에는 영향 없음 — swap 절차는 별도.
    UPSTAGE_API_KEY2: str = ""
    TMDB_API_KEY: str = ""
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    KOBIS_API_KEY: str = ""
    KOBIS_BASE_URL: str = "http://www.kobis.or.kr/kobisopenapi/webservice/rest"
    KAKAO_API_KEY: str = ""

    # ── KMDb (한국영화 데이터베이스) ──
    KMDB_API_KEY: str = ""
    KMDB_BASE_URL: str = "http://api.koreafilm.or.kr/openapi-data2/wisenut/search_api/search_json2.jsp"

    # ── OMDb (IMDb/Rotten Tomatoes/Metacritic 평점) ──
    # Phase 6: movie_external_ratings 테이블 적재용. 무료 1000/day.
    OMDB_API_KEY: str = ""
    OMDB_BASE_URL: str = "http://www.omdbapi.com/"

    # ── Ollama (로컬 LLM 서버) ──
    # Ollama는 단일 서버에서 OLLAMA_MAX_LOADED_MODELS 수만큼 모델을 동시 로드한다.
    # Mac 64GB 기준: qwen3.5:35b-a3b + exaone-32b 동시 로드 가능 (2모델)
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

    # ── LLM 라우팅 모드 ──
    # "hybrid": 몽글이(로컬, 빠른 응답) + Solar API(품질 중요) 혼합
    # "local_only": 모든 체인을 Ollama 로컬 모델로 처리 (기존 EXAONE/Qwen 또는 몽글이)
    # "api_only": 모든 체인을 Solar API로 처리 (Ollama 장애 시)
    LLM_MODE: str = "local_only"

    # ── Solar API (Upstage, 추론 품질이 중요한 체인) ──
    # OpenAI 호환 API — langchain-openai의 ChatOpenAI로 연동
    # hybrid/api_only 모드에서 의도분류, 감정분석, 선호추출, 추천이유, 이미지분석에 사용
    # UPSTAGE_API_KEY는 상단 API Keys 섹션에서 설정
    SOLAR_API_BASE_URL: str = "https://api.upstage.ai/v1"
    SOLAR_API_MODEL: str = "solar-pro"
    SOLAR_API_TIMEOUT: int = 30
    SOLAR_API_MAX_RETRIES: int = 2
    # Solar API 분당 최대 호출 수 (rate limit 보호용 세마포어)
    SOLAR_API_RATE_LIMIT: int = 60

    # ── 몽글이 (EXAONE 4.0 1.2B LoRA, Ollama 로컬, 텍스트 생성 전용) ──
    # hybrid 모드에서 일반대화/후속질문에 사용 (분석/추론 없이 "말하기"만 담당)
    # 파인튜닝 전에는 EXAONE 4.0 1.2B 베이스 모델 사용 가능
    MONGLE_MODEL: str = "mongle"
    MONGLE_TEMPERATURE: float = 0.5

    # ── vLLM (운영서버 GPU, OpenAI 호환 API) ──
    # hybrid 모드에서 Ollama 대신 운영서버 vLLM을 사용할 때 활성화
    # vLLM은 OpenAI 호환 API를 제공하므로 ChatOpenAI로 연동
    VLLM_ENABLED: bool = False
    # vLLM 채팅 모델 (EXAONE 4.0 1.2B, 일반대화/후속질문)
    VLLM_CHAT_BASE_URL: str = "http://localhost:18000/v1"
    VLLM_CHAT_MODEL: str = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    # vLLM 비전 모델 (Qwen2.5-VL-3B, 이미지 분석)
    VLLM_VISION_BASE_URL: str = "http://localhost:18001/v1"
    VLLM_VISION_MODEL: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    VLLM_TIMEOUT: int = 30  # 기존 60 → 30초로 단축 (연결 실패 시 빠른 폴백)
    VLLM_MAX_RETRIES: int = 2

    # ── Ollama 서빙 옵션 ──
    # 컨텍스트 윈도우 크기 (토큰 수). 모델 기본값(qwen3.5=262K)은 KV 캐시로 GPU 메모리를
    # 과도하게 소모하므로, 영화 추천 대화에 충분한 8192로 제한하여 메모리를 절약한다.
    # 64GB Mac 기준: num_ctx=8192이면 qwen3.5(~24GB) + exaone-32b(~20GB) = ~44GB (여유 20GB)
    OLLAMA_NUM_CTX: int = 8192
    # keep_alive: 모델 메모리 유지 시간. "10m"이면 마지막 요청 후 10분간 GPU에 상주.
    # 짧으면 메모리 절약, 길면 응답 빠름. "-1"이면 영구 유지, "0"이면 즉시 언로드.
    OLLAMA_KEEP_ALIVE: str = "10m"

    # ── LLM Models (Ollama 로컬 모델) ──
    # local_only 모드에서 사용되는 Ollama 모델명
    # hybrid 모드에서는 CONVERSATION_MODEL/QUESTION_MODEL만 로컬 사용 (나머지는 Solar API)
    # 모델명은 ollama pull/run 시 사용하는 이름과 일치해야 한다.
    #
    # 구조화 출력 (JSON) + 비전: Qwen3.5 35B-A3B (텍스트 분류 + 이미지 분석 통합)
    INTENT_MODEL: str = "qwen3.5:35b-a3b"
    EMOTION_MODEL: str = "qwen3.5:35b-a3b"
    MOOD_MODEL: str = "qwen3.5:35b-a3b"
    # 한국어 자연어 생성: EXAONE 4.0 32B (비추론 모드: temperature < 0.6)
    PREFERENCE_MODEL: str = "exaone-32b:latest"
    CONVERSATION_MODEL: str = "exaone-32b:latest"
    # 경량 한국어 생성: EXAONE 4.0 32B
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
    # ── Recommend FastAPI 내부 URL ──
    # Movie Match 의 Co-watched CF 후보 조회 (/api/v2/match/co-watched) 등에 사용.
    # Docker 네트워크에서는 http://monglepick-recommend:8001 로 오버라이드한다.
    RECOMMEND_BASE_URL: str = "http://localhost:8001"
    # Co-watched CF 조회 타임아웃 (초) — CF 실패 시 Qdrant/ES/Neo4j 결과만으로도
    # 매치가 가능하도록 짧게 설정해 전체 그래프 지연을 최소화한다.
    MATCH_COWATCH_TIMEOUT: float = 3.0
    # Co-watched CF 조회 top_k — RRF 에 투입할 후보 수
    MATCH_COWATCH_TOP_K: int = 20
    # AI 추천 1회당 차감할 포인트 수 (init.sql point_items 기준 100P)
    POINT_COST_PER_RECOMMENDATION: int = 100
    # 포인트 체크 활성화 여부 (False이면 포인트 체크 생략 — 개발/테스트 환경)
    POINT_CHECK_ENABLED: bool = True
    # 회원가입 시 무료 포인트 지급 수
    FREE_POINTS_ON_SIGNUP: int = 5

    # ── Security ──
    # Backend 측 application.yml 의 app.service.key 와 동일해야 한다.
    # 운영 환경(.env.prod)에서는 반드시 랜덤 문자열로 교체.
    SERVICE_API_KEY: str = "dev-service-key-change-me"
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
    # Ollama는 GPU 메모리 내에서 동시 요청을 처리하지만,
    # 과부하 방지를 위해 모델별 동시 호출 수를 제한한다.
    LLM_PER_MODEL_CONCURRENCY: int = 2
    # 추천 이유를 LLM으로 생성할 최대 영화 수.
    # RECOMMENDATION_TOP_K(5)와 동일하게 맞춰 모든 추천 영화에 LLM 설명을 생성한다.
    # 초과 영화는 메타데이터 기반 템플릿(_build_fallback_explanation)으로 대체.
    MAX_EXPLANATION_MOVIES: int = 5

    # ── Retrieval Quality Thresholds (RAG 검색 품질 판정) ──
    # 최소 후보 수: 이 값 미만이면 검색 품질 미달
    RETRIEVAL_MIN_CANDIDATES: int = 3
    # Top-1 RRF 점수 최소값
    # 2026-04-15 하향(0.015 → 0.010): "애매하면 재질문" 정책 반영.
    # 점수가 0.01~0.015 구간이면 품질 미달로 보고 soft-ambiguous 분기로 보낸다.
    RETRIEVAL_MIN_TOP_SCORE: float = 0.010
    # 상위 5개 평균 RRF 점수 최소값 (같은 취지로 하향)
    RETRIEVAL_QUALITY_MIN_AVG: float = 0.008
    # 선호 충분성 판정 임계값 (가중치 합산이 이 값 이상이면 추천 진행)
    # 2026-04-15 하향(2.5 → 2.0): 모호한 입력에서도 재질문을 더 자주 띄우도록 완화.
    SUFFICIENCY_THRESHOLD: float = 2.0
    # Soft-ambiguous 임계값 — route_after_retrieval 에서 "후보는 있지만 점수가 애매한"
    # 구간을 판별할 때 사용한다. top_score 가 이 값 미만이면 similar_fallback_search 대신
    # question_generator 로 보낸다. (turn_count < TURN_COUNT_OVERRIDE 인 경우에 한정)
    RETRIEVAL_SOFT_AMBIGUOUS_TOP_SCORE: float = 0.020
    # 턴 카운트 오버라이드 (이 턴 이상이면 선호 부족해도 추천 진행)
    # Phase ML-3: 2→3으로 상향하여 재질문 기회를 최대 2회 제공
    # 기존(2): 턴1 모호 → 재질문 → 턴2 모호 → 강제 추천 (재질문 1회)
    # 개선(3): 턴1 모호 → 재질문 → 턴2 모호 → 재질문 → 턴3 강제 추천 (재질문 2회)
    TURN_COUNT_OVERRIDE: int = 3

    # ── Recommendation Engine Thresholds (추천 엔진) ──
    # Cold Start 판정: 시청 이력이 이 값 미만이면 Cold Start
    COLD_START_THRESHOLD: int = 5
    # Warm Start 판정: 시청 이력이 이 값 미만이면 Warm Start (이상이면 정상)
    WARM_START_THRESHOLD: int = 30
    # MMR 다양성 파라미터 (λ=0.7: 관련성 70% + 다양성 30%)
    MMR_LAMBDA: float = 0.7
    # 최종 추천 영화 수
    RECOMMENDATION_TOP_K: int = 5
    # Movie Match v3 — 커플/개인 대상 핵심 추천 개수 (2026-04-14 유저 요청: 3→5 복귀)
    # 5편은 커플에게 선택 폭을 충분히 제공하면서 LLM 리랭커 품질도 담보 가능한 개수
    MATCH_TOP_K: int = 5

    # ── Hybrid Search (하이브리드 검색) ──
    # RRF Reciprocal Rank Fusion 상수
    RRF_K: int = 60

    # ── 전역 API Rate Limiting (RateLimitMiddleware) ──
    # 비인증 사용자(IP 기반): 분당 최대 요청 수
    RATE_LIMIT_ANON_RPM: int = 30
    # 인증 사용자(Bearer 토큰 기반): 분당 최대 요청 수
    RATE_LIMIT_AUTH_RPM: int = 60

    # ── 요청 타임아웃 (TimeoutMiddleware) ──
    # SSE 엔드포인트(/api/v1/chat, /api/v1/match): LangGraph 전체 실행 포함 (초)
    TIMEOUT_SSE_SEC: int = 300
    # 일반 엔드포인트 타임아웃 (초)
    TIMEOUT_DEFAULT_SEC: int = 30

    # ── MySQL 커넥션 풀 ──
    # 풀 최소 연결 수 (유휴 상태에서도 유지할 연결)
    MYSQL_POOL_MIN: int = 1
    # 풀 최대 연결 수 (동시 쿼리 허용 상한)
    MYSQL_POOL_MAX: int = 10

    # ── Redis 커넥션 풀 ──
    # Redis 클라이언트 최대 연결 수
    REDIS_MAX_CONNECTIONS: int = 50

    @model_validator(mode='after')
    def _warn_empty_passwords(self):
        """비밀번호 및 필수 API 키가 비어있으면 경고 로그를 출력한다. (W-1)"""
        if not self.NEO4J_PASSWORD:
            _config_logger.warning("NEO4J_PASSWORD가 설정되지 않았습니다. .env 파일을 확인하세요.")
        if not self.MYSQL_PASSWORD:
            _config_logger.warning("MYSQL_PASSWORD가 설정되지 않았습니다. .env 파일을 확인하세요.")
        # Upstage Solar API 키는 임베딩 + LLM API 모두에 필수
        # hybrid/api_only 모드에서 미설정 시 Solar API 호출 실패
        if not self.UPSTAGE_API_KEY:
            _config_logger.warning(
                "UPSTAGE_API_KEY가 설정되지 않았습니다. "
                "임베딩 검색 및 Solar API LLM 호출이 실패합니다. .env 파일을 확인하세요."
            )
        return self


settings = Settings()
