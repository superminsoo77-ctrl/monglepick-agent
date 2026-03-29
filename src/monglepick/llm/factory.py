"""
LLM 팩토리 — 하이브리드 라우팅 (Ollama 로컬 + Solar API).

LLM_MODE 설정에 따라 체인별로 최적의 LLM 백엔드를 선택한다:
  - "local_only" (기본): 모든 체인을 Ollama 로컬 모델로 처리 (기존 EXAONE/Qwen)
  - "hybrid": 빠른 응답 체인은 몽글이(Ollama), 품질 중요 체인은 Solar API
  - "api_only": 모든 체인을 Solar API로 처리 (Ollama 장애 시)

체인별 라우팅 (hybrid 모드):
  - 몽글이 (Ollama 로컬, 빠른 응답):
    - general_chat_chain (일반 대화)
    - question_chain (후속 질문 생성)
  - Solar API (Upstage, 추론 품질):
    - intent_emotion_chain, intent_chain, emotion_chain (의도+감정 분류)
    - preference_chain (선호 추출)
    - explanation_chain (추천 이유 생성)
    - image_analysis_chain (이미지 분석)

local_only 모드:
  - 기존과 동일하게 settings의 모델명(EXAONE/Qwen 등)을 그대로 사용
  - Solar API 호출 없음, Ollama만 사용

캐싱:
  - Ollama: (model, temperature, format) 튜플 → ChatOllama 인스턴스
  - Solar API: (model, temperature) 튜플 → ChatOpenAI 인스턴스
  - 구조화 출력: (backend, model, temperature, schema_name) → Runnable
"""

from __future__ import annotations

import threading
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from monglepick.config import settings
from monglepick.llm.concurrency import acquire_model_slot, release_model_slot

logger = structlog.get_logger()

# ============================================================
# 모듈 레벨 캐시
# ============================================================

# Ollama 캐시: (model, temperature, format) → ChatOllama
_ollama_cache: dict[tuple[str, float, str | None], ChatOllama] = {}

# Solar API 캐시: (model, temperature) → ChatOpenAI
_solar_cache: dict[tuple[str, float], ChatOpenAI] = {}

# vLLM 캐시: (base_url, model, temperature) → ChatOpenAI
_vllm_cache: dict[tuple[str, str, float], ChatOpenAI] = {}

# 구조화 출력 캐시: (backend, model, temperature, schema_name) → Runnable
_structured_cache: dict[tuple[str, str, float, str], Runnable] = {}

# 캐시 접근 보호용 락 — RLock: 구조화 출력이 내부에서 기본 LLM을 호출하므로 재진입 허용
_cache_lock = threading.RLock()


# ============================================================
# Ollama (로컬) LLM 생성
# ============================================================

def get_ollama_llm(
    model: str | None = None,
    temperature: float | None = None,
    format: str | None = None,
    num_predict: int | None = None,
) -> ChatOllama:
    """
    ChatOllama 인스턴스를 생성하거나 캐시에서 반환한다.

    Ollama 서버에 연결하여 LLM 모델을 사용한다.
    동일 (model, temperature, format) 조합은 싱글턴으로 재사용된다.

    Args:
        model: Ollama 모델명 (기본값: settings.CONVERSATION_MODEL)
        temperature: 생성 온도 (기본값: 0.5)
        format: 응답 형식 ("json" 또는 None)
        num_predict: 최대 생성 토큰 수

    Returns:
        ChatOllama 인스턴스
    """
    model = model or settings.CONVERSATION_MODEL
    temperature = temperature if temperature is not None else 0.5

    cache_key = (model, temperature, format)

    with _cache_lock:
        if cache_key not in _ollama_cache:
            kwargs: dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "base_url": settings.OLLAMA_BASE_URL,
                # GPU 메모리 보호: 모델 기본 context(qwen3.5=262K)가 KV 캐시로
                # 과도한 VRAM을 소모하므로, 설정값으로 제한한다.
                "num_ctx": settings.OLLAMA_NUM_CTX,
                # 모델 메모리 유지 시간: 마지막 요청 후 이 시간이 지나면 GPU에서 언로드.
                # 2개 모델 동시 서빙 시 메모리 관리에 필수.
                "keep_alive": settings.OLLAMA_KEEP_ALIVE,
            }
            if format == "json":
                kwargs["format"] = "json"
            if num_predict is not None:
                kwargs["num_predict"] = num_predict

            _ollama_cache[cache_key] = ChatOllama(**kwargs)
            logger.info(
                "ollama_llm_created",
                model=model,
                temperature=temperature,
                format=format,
                base_url=settings.OLLAMA_BASE_URL,
            )
        else:
            logger.debug("ollama_cache_hit", model=model, temperature=temperature)

        return _ollama_cache[cache_key]


# ============================================================
# Solar API (Upstage) LLM 생성
# ============================================================

def get_solar_api_llm(
    temperature: float = 0.3,
    model: str | None = None,
) -> ChatOpenAI:
    """
    Upstage Solar API LLM 인스턴스를 생성하거나 캐시에서 반환한다.

    Solar API는 OpenAI 호환 프로토콜이므로 ChatOpenAI 클래스를 사용한다.
    동일 (model, temperature) 조합은 싱글턴으로 재사용된다.

    Args:
        temperature: 생성 온도 (기본값: 0.3)
        model: Solar 모델명 (기본값: settings.SOLAR_API_MODEL)

    Returns:
        ChatOpenAI 인스턴스 (Solar API 백엔드)
    """
    model = model or settings.SOLAR_API_MODEL

    cache_key = (model, temperature)

    with _cache_lock:
        if cache_key not in _solar_cache:
            _solar_cache[cache_key] = ChatOpenAI(
                model=model,
                temperature=temperature,
                base_url=settings.SOLAR_API_BASE_URL,
                api_key=settings.UPSTAGE_API_KEY,
                max_tokens=2048,
                timeout=settings.SOLAR_API_TIMEOUT,
                max_retries=settings.SOLAR_API_MAX_RETRIES,
            )
            logger.info(
                "solar_api_llm_created",
                model=model,
                temperature=temperature,
                base_url=settings.SOLAR_API_BASE_URL,
            )
        else:
            logger.debug("solar_cache_hit", model=model, temperature=temperature)

        return _solar_cache[cache_key]


# ============================================================
# vLLM (운영서버 GPU, OpenAI 호환 API) LLM 생성
# ============================================================

def get_vllm_llm(
    temperature: float = 0.5,
    base_url: str | None = None,
    model: str | None = None,
) -> ChatOpenAI:
    """
    운영서버 vLLM LLM 인스턴스를 생성하거나 캐시에서 반환한다.

    vLLM은 OpenAI 호환 프로토콜이므로 ChatOpenAI 클래스를 사용한다.
    동일 (base_url, model, temperature) 조합은 싱글턴으로 재사용된다.

    Args:
        temperature: 생성 온도 (기본값: 0.5)
        base_url: vLLM 서버 URL (기본값: settings.VLLM_CHAT_BASE_URL)
        model: vLLM 모델명 (기본값: settings.VLLM_CHAT_MODEL)

    Returns:
        ChatOpenAI 인스턴스 (vLLM 백엔드)
    """
    base_url = base_url or settings.VLLM_CHAT_BASE_URL
    model = model or settings.VLLM_CHAT_MODEL

    cache_key = (base_url, model, temperature)

    with _cache_lock:
        if cache_key not in _vllm_cache:
            _vllm_cache[cache_key] = ChatOpenAI(
                model=model,
                temperature=temperature,
                base_url=base_url,
                # vLLM은 API 키 불필요 — 더미 값 전달 (ChatOpenAI 필수 인자)
                api_key="EMPTY",
                # 몽글이(1.2B)는 max_model_len=2048. 프롬프트(~500 tok) 여유를 두고 출력 제한
                max_tokens=512,
                timeout=settings.VLLM_TIMEOUT,
                max_retries=settings.VLLM_MAX_RETRIES,
            )
            logger.info(
                "vllm_llm_created",
                model=model,
                temperature=temperature,
                base_url=base_url,
            )
        else:
            logger.debug("vllm_cache_hit", model=model, temperature=temperature)

        return _vllm_cache[cache_key]


# ============================================================
# 하이브리드 LLM 선택 헬퍼
# ============================================================

def _use_solar_api() -> bool:
    """Solar API를 사용해야 하는지 판단 (hybrid 또는 api_only 모드)."""
    return settings.LLM_MODE in ("hybrid", "api_only")


def _use_local() -> bool:
    """로컬 Ollama를 사용해야 하는지 판단 (hybrid 또는 local_only 모드)."""
    return settings.LLM_MODE in ("hybrid", "local_only")


# ============================================================
# 구조화 출력 LLM
# ============================================================

def get_structured_llm(
    schema: type[BaseModel],
    model: str | None = None,
    temperature: float | None = None,
    use_api: bool = True,
) -> Runnable:
    """
    구조화 출력(Pydantic 모델 자동 검증) LLM을 반환한다.

    LLM_MODE에 따라 Solar API 또는 Ollama 백엔드를 선택한다.

    Args:
        schema: 출력 Pydantic 모델 클래스
        model: 모델명 (Local 시 Ollama 모델, API 시 무시됨)
        temperature: 생성 온도 (기본값: 0.1)
        use_api: True이면 hybrid/api_only에서 Solar API 사용, False이면 항상 로컬

    Returns:
        구조화 출력이 적용된 Runnable
    """
    temperature = temperature if temperature is not None else 0.1

    # LLM_MODE에 따른 백엔드 결정
    if use_api and _use_solar_api():
        backend = "solar_api"
        resolved_model = settings.SOLAR_API_MODEL
        cache_key = (backend, resolved_model, temperature, schema.__name__)
    else:
        backend = "ollama"
        resolved_model = model or settings.INTENT_MODEL
        cache_key = (backend, resolved_model, temperature, schema.__name__)

    with _cache_lock:
        if cache_key not in _structured_cache:
            if backend == "solar_api":
                # Solar API — ChatOpenAI + 구조화 출력
                llm = get_solar_api_llm(temperature=temperature)
                structured = llm.with_structured_output(schema, method="json_schema")
                logger.info(
                    "structured_solar_created",
                    model=resolved_model,
                    temperature=temperature,
                    schema=schema.__name__,
                )
            else:
                # Ollama 로컬 — ChatOllama(format="json") + 구조화 출력
                llm = get_ollama_llm(
                    model=resolved_model, temperature=temperature, format="json",
                )
                structured = llm.with_structured_output(schema, method="json_schema")
                logger.info(
                    "structured_ollama_created",
                    model=resolved_model,
                    temperature=temperature,
                    schema=schema.__name__,
                )

            _structured_cache[cache_key] = structured
        else:
            logger.debug(
                "structured_cache_hit",
                backend=backend,
                schema=schema.__name__,
            )

        return _structured_cache[cache_key]


# ============================================================
# 용도별 편의 함수 — 하이브리드 라우팅
# ============================================================

# ── Solar API 우선 체인 (hybrid 모드에서 품질 중요) ──

def get_intent_llm() -> Runnable:
    """
    의도 분류용 구조화 출력 LLM (temp=0.1).

    hybrid/api_only → Solar API (분류 정확도 중요)
    local_only → Ollama (settings.INTENT_MODEL, 기본 Qwen 35B)
    """
    from monglepick.agents.chat.models import IntentResult

    logger.debug("get_intent_llm_called", mode=settings.LLM_MODE)
    return get_structured_llm(
        schema=IntentResult,
        model=settings.INTENT_MODEL,
        temperature=0.1,
        use_api=True,
    )


def get_emotion_llm() -> Runnable:
    """
    감정 분석용 구조화 출력 LLM (temp=0.1).

    hybrid/api_only → Solar API
    local_only → Ollama (settings.EMOTION_MODEL, 기본 Qwen 35B)
    """
    from monglepick.agents.chat.models import EmotionResult

    logger.debug("get_emotion_llm_called", mode=settings.LLM_MODE)
    return get_structured_llm(
        schema=EmotionResult,
        model=settings.EMOTION_MODEL,
        temperature=0.1,
        use_api=True,
    )


def get_intent_emotion_llm() -> Runnable:
    """
    의도+감정 통합 분석용 구조화 출력 LLM (temp=0.1).

    hybrid/api_only → Solar API
    local_only → Ollama (settings.INTENT_MODEL, 기본 Qwen 35B)
    """
    from monglepick.agents.chat.models import IntentEmotionResult

    logger.debug("get_intent_emotion_llm_called", mode=settings.LLM_MODE)
    return get_structured_llm(
        schema=IntentEmotionResult,
        model=settings.INTENT_MODEL,
        temperature=0.1,
        use_api=True,
    )


def get_preference_llm() -> Runnable:
    """
    선호 추출용 구조화 출력 LLM (temp=0.3).

    hybrid/api_only → Solar API (복잡한 한국어 선호 파싱)
    local_only → Ollama (settings.PREFERENCE_MODEL, 기본 EXAONE 32B)
    """
    from monglepick.agents.chat.models import ExtractedPreferences

    logger.debug("get_preference_llm_called", mode=settings.LLM_MODE)
    return get_structured_llm(
        schema=ExtractedPreferences,
        model=settings.PREFERENCE_MODEL,
        temperature=0.3,
        use_api=True,
    )


def get_explanation_llm() -> BaseChatModel:
    """
    추천 이유 생성용 LLM (temp=0.5).

    hybrid/api_only → Solar API (설명 품질 중요)
    local_only → Ollama (settings.EXPLANATION_MODEL, 기본 EXAONE 32B)

    자유 텍스트 생성 — 구조화 출력 미적용.
    """
    logger.debug("get_explanation_llm_called", mode=settings.LLM_MODE)
    if _use_solar_api():
        return get_solar_api_llm(temperature=0.5)
    return get_ollama_llm(model=settings.EXPLANATION_MODEL, temperature=0.5)


def get_vision_llm() -> BaseChatModel:
    """
    비전(이미지 분석)용 LLM (temp=0.2).

    hybrid/api_only → Solar API (멀티모달 품질)
    local_only → Ollama (settings.VISION_MODEL, 기본 Qwen 35B)

    Note: format="json" 미사용. _parse_json_response()의 3단계 폴백으로 충분.
    """
    logger.debug("get_vision_llm_called", mode=settings.LLM_MODE)
    if _use_solar_api():
        return get_solar_api_llm(temperature=0.2)
    return get_ollama_llm(model=settings.VISION_MODEL, temperature=0.2)


# ── 몽글이(로컬) 우선 체인 (hybrid 모드에서 빠른 응답) ──

def get_conversation_llm() -> BaseChatModel:
    """
    일반 대화용 LLM (temp=0.5).

    hybrid + VLLM_ENABLED → vLLM EXAONE 1.2B (운영서버 GPU, 빠른 응답)
    hybrid + !VLLM_ENABLED → 몽글이 (settings.MONGLE_MODEL, Ollama 로컬)
    local_only → Ollama (settings.CONVERSATION_MODEL, 기본 EXAONE 32B)
    api_only → Solar API

    자유 텍스트 생성 — 구조화 출력 미적용.
    """
    logger.debug("get_conversation_llm_called", mode=settings.LLM_MODE, vllm=settings.VLLM_ENABLED)
    if settings.LLM_MODE == "api_only":
        return get_solar_api_llm(temperature=0.5)
    if settings.LLM_MODE == "hybrid":
        if settings.VLLM_ENABLED:
            # hybrid + vLLM: 운영서버 EXAONE 1.2B로 빠른 응답
            return get_vllm_llm(temperature=settings.MONGLE_TEMPERATURE)
        # hybrid + Ollama: 몽글이(파인튜닝 모델)로 빠른 응답
        return get_ollama_llm(
            model=settings.MONGLE_MODEL,
            temperature=settings.MONGLE_TEMPERATURE,
        )
    # local_only: 기존 설정 모델 사용 (EXAONE 등)
    return get_ollama_llm(model=settings.CONVERSATION_MODEL, temperature=0.5)


def get_question_llm() -> BaseChatModel:
    """
    후속 질문 생성용 LLM (temp=0.5).

    hybrid + VLLM_ENABLED → vLLM EXAONE 1.2B (운영서버 GPU, 빠른 응답)
    hybrid + !VLLM_ENABLED → 몽글이 (settings.MONGLE_MODEL, Ollama 로컬)
    local_only → Ollama (settings.QUESTION_MODEL, 기본 EXAONE 32B)
    api_only → Solar API

    자유 텍스트 생성 — 구조화 출력 미적용.
    """
    logger.debug("get_question_llm_called", mode=settings.LLM_MODE, vllm=settings.VLLM_ENABLED)
    if settings.LLM_MODE == "api_only":
        return get_solar_api_llm(temperature=0.5)
    if settings.LLM_MODE == "hybrid":
        if settings.VLLM_ENABLED:
            # hybrid + vLLM: 운영서버 EXAONE 1.2B로 빠른 응답
            return get_vllm_llm(temperature=settings.MONGLE_TEMPERATURE)
        # hybrid + Ollama: 몽글이(파인튜닝 모델)로 빠른 응답
        return get_ollama_llm(
            model=settings.MONGLE_MODEL,
            temperature=settings.MONGLE_TEMPERATURE,
        )
    # local_only: 기존 설정 모델 사용 (EXAONE 등)
    return get_ollama_llm(model=settings.QUESTION_MODEL, temperature=0.5)


# ============================================================
# 하위 호환 함수 — 기존 코드에서 get_llm() 직접 호출하는 경우 대응
# ============================================================

def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    format: str | None = None,
    num_predict: int | None = None,
) -> ChatOllama:
    """
    하위 호환용 — Ollama LLM 인스턴스 반환.

    get_ollama_llm()의 별칭. 기존 코드에서 get_llm()을 직접 호출하는 경우를 위해 유지.
    새 코드에서는 용도별 편의 함수(get_conversation_llm 등) 사용을 권장.
    """
    return get_ollama_llm(
        model=model, temperature=temperature,
        format=format, num_predict=num_predict,
    )


# ============================================================
# 동시성 제어 래퍼
# ============================================================

async def guarded_ainvoke(
    llm: Runnable | BaseChatModel,
    prompt_value: Any,
    model: str,
    request_id: str = "",
) -> Any:
    """
    모델별 세마포어로 감싼 LLM ainvoke.

    Ollama 로컬 호출 시: 모델별 세마포어로 동시 호출 수 제한.
    Solar API 호출 시: "solar_api" 키로 rate limit 세마포어 보호.

    Args:
        llm: LangChain Runnable 또는 BaseChatModel 인스턴스
        prompt_value: LLM에 전달할 프롬프트
        model: 모델명 (세마포어 키, Solar API는 "solar_api" 사용 권장)
        request_id: 요청 식별자 (로깅용)

    Returns:
        LLM 응답 (BaseMessage 또는 구조화 출력 Pydantic 모델)
    """
    await acquire_model_slot(model, request_id)
    try:
        return await llm.ainvoke(prompt_value)
    finally:
        release_model_slot(model)
