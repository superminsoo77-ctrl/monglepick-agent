"""
LLM 팩토리 — ChatOllama 인스턴스 생성 및 캐싱.

로컬 Ollama 서버의 LLM 모델을 용도별로 생성하는 팩토리 함수 모음.
동일 파라미터 조합에 대해 싱글턴 캐싱을 적용하여 중복 인스턴스 생성을 방지한다.

모델 라우팅:
- 구조화 출력 (intent, emotion): Qwen 14B (정확한 JSON 생성)
- 한국어 생성 (preference, conversation, question, explanation): EXAONE 32B
- 구조화 출력은 with_structured_output(schema)으로 Pydantic 모델 자동 검증

캐싱:
- (model, temperature, format) 튜플을 키로 모듈 레벨 dict 캐싱
- get_structured_llm은 schema별로 별도 캐싱 (.with_structured_output은 새 Runnable)
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from monglepick.config import settings
from monglepick.llm.concurrency import acquire_model_slot, release_model_slot

logger = structlog.get_logger()

# ============================================================
# 모듈 레벨 캐시: (model, temperature, format) → ChatOllama 인스턴스
# ============================================================

_llm_cache: dict[tuple[str, float, str | None], ChatOllama] = {}

# 구조화 출력 캐시: (model, temperature, schema_name) → Runnable
_structured_cache: dict[tuple[str, float, str], Runnable] = {}


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    format: str | None = None,
    num_predict: int | None = None,
) -> ChatOllama:
    """
    ChatOllama 인스턴스를 생성하거나 캐시에서 반환한다.

    동일 (model, temperature, format) 조합은 싱글턴으로 재사용된다.
    num_predict는 캐시 키에 포함되지 않으며, 호출 시마다 적용할 수 있다.

    Args:
        model: Ollama 모델명 (기본값: settings.CONVERSATION_MODEL)
        temperature: 생성 온도 (기본값: 0.5)
        format: 응답 형식 ("json" 또는 None)
        num_predict: 최대 생성 토큰 수 (None이면 모델 기본값)

    Returns:
        ChatOllama 인스턴스
    """
    # 기본값 설정
    model = model or settings.CONVERSATION_MODEL
    temperature = temperature if temperature is not None else 0.5

    # 캐시 키 생성
    cache_key = (model, temperature, format)

    if cache_key not in _llm_cache:
        # ChatOllama 인스턴스 생성
        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "base_url": settings.OLLAMA_BASE_URL,
        }
        if format is not None:
            kwargs["format"] = format
        if num_predict is not None:
            kwargs["num_predict"] = num_predict

        _llm_cache[cache_key] = ChatOllama(**kwargs)
        # 캐시 미스: 새 인스턴스 생성 로깅
        logger.info(
            "llm_instance_created",
            model=model,
            temperature=temperature,
            format=format,
            num_predict=num_predict,
            base_url=settings.OLLAMA_BASE_URL,
        )
    else:
        # 캐시 히트: 기존 인스턴스 재사용 로깅 (DEBUG 레벨, 과다 로깅 방지)
        logger.debug(
            "llm_cache_hit",
            model=model,
            temperature=temperature,
            format=format,
        )

    return _llm_cache[cache_key]


def get_structured_llm(
    schema: type[BaseModel],
    model: str | None = None,
    temperature: float | None = None,
) -> Runnable:
    """
    구조화 출력(Pydantic 모델 자동 검증) LLM을 반환한다.

    ChatOllama에 .with_structured_output(schema, method="json_schema")를 적용하여
    LLM 응답을 자동으로 Pydantic 모델로 파싱한다.

    Args:
        schema: 출력 Pydantic 모델 클래스
        model: Ollama 모델명 (기본값: settings.INTENT_MODEL)
        temperature: 생성 온도 (기본값: 0.1)

    Returns:
        구조화 출력이 적용된 Runnable
    """
    model = model or settings.INTENT_MODEL
    temperature = temperature if temperature is not None else 0.1

    # 구조화 출력 캐시 키
    cache_key = (model, temperature, schema.__name__)

    if cache_key not in _structured_cache:
        # 기본 ChatOllama 인스턴스 가져오기
        llm = get_llm(model=model, temperature=temperature, format="json")
        # 구조화 출력 적용
        _structured_cache[cache_key] = llm.with_structured_output(
            schema, method="json_schema",
        )
        # 캐시 미스: 새 구조화 출력 인스턴스 생성 로깅
        logger.info(
            "structured_llm_created",
            model=model,
            temperature=temperature,
            schema=schema.__name__,
        )
    else:
        # 캐시 히트 로깅 (DEBUG 레벨)
        logger.debug(
            "structured_llm_cache_hit",
            model=model,
            temperature=temperature,
            schema=schema.__name__,
        )

    return _structured_cache[cache_key]


# ============================================================
# 용도별 편의 함수
# ============================================================

def get_intent_llm() -> Runnable:
    """
    의도 분류용 구조화 출력 LLM (Qwen 14B, temp=0.1).

    IntentResult Pydantic 모델을 자동 파싱한다.
    """
    from monglepick.agents.chat.models import IntentResult

    logger.debug("get_intent_llm_called", model=settings.INTENT_MODEL)
    return get_structured_llm(
        schema=IntentResult,
        model=settings.INTENT_MODEL,
        temperature=0.1,
    )


def get_emotion_llm() -> Runnable:
    """
    감정 분석용 구조화 출력 LLM (Qwen 14B, temp=0.1).

    EmotionResult Pydantic 모델을 자동 파싱한다.
    """
    from monglepick.agents.chat.models import EmotionResult

    logger.debug("get_emotion_llm_called", model=settings.EMOTION_MODEL)
    return get_structured_llm(
        schema=EmotionResult,
        model=settings.EMOTION_MODEL,
        temperature=0.1,
    )


def get_preference_llm() -> Runnable:
    """
    선호 추출용 구조화 출력 LLM (EXAONE 32B, temp=0.3).

    ExtractedPreferences Pydantic 모델을 자동 파싱한다.
    """
    from monglepick.agents.chat.models import ExtractedPreferences

    logger.debug("get_preference_llm_called", model=settings.PREFERENCE_MODEL)
    return get_structured_llm(
        schema=ExtractedPreferences,
        model=settings.PREFERENCE_MODEL,
        temperature=0.3,
    )


def get_conversation_llm() -> ChatOllama:
    """
    일반 대화용 LLM (EXAONE 32B, temp=0.5).

    자유 텍스트 생성 — 구조화 출력 미적용.
    """
    logger.debug("get_conversation_llm_called", model=settings.CONVERSATION_MODEL)
    return get_llm(
        model=settings.CONVERSATION_MODEL,
        temperature=0.5,
    )


def get_question_llm() -> ChatOllama:
    """
    후속 질문 생성용 LLM (EXAONE 32B, temp=0.5).

    자유 텍스트 생성 — 구조화 출력 미적용.
    """
    logger.debug("get_question_llm_called", model=settings.QUESTION_MODEL)
    return get_llm(
        model=settings.QUESTION_MODEL,
        temperature=0.5,
    )


def get_explanation_llm() -> ChatOllama:
    """
    추천 이유 생성용 LLM (EXAONE 32B, temp=0.5).

    자유 텍스트 생성 — 구조화 출력 미적용.
    """
    logger.debug("get_explanation_llm_called", model=settings.EXPLANATION_MODEL)
    return get_llm(
        model=settings.EXPLANATION_MODEL,
        temperature=0.5,
    )


def get_intent_emotion_llm() -> Runnable:
    """
    의도+감정 통합 분석용 구조화 출력 LLM (qwen3.5:35b-a3b, temp=0.1).

    IntentEmotionResult Pydantic 모델을 자동 파싱한다.
    기존 get_intent_llm() + get_emotion_llm() 2회 호출을 1회로 통합.
    """
    from monglepick.agents.chat.models import IntentEmotionResult

    logger.debug("get_intent_emotion_llm_called", model=settings.INTENT_MODEL)
    return get_structured_llm(
        schema=IntentEmotionResult,
        model=settings.INTENT_MODEL,
        temperature=0.1,
    )


def get_vision_llm() -> ChatOllama:
    """
    비전(이미지 분석)용 LLM (Qwen3.5 35B-A3B, temp=0.2).

    멀티모달 모델로 이미지 + 텍스트 입력을 받아 JSON 구조화 출력을 생성한다.
    영화 포스터/분위기 사진에서 장르/무드/시각요소를 추출하는 데 사용된다.

    Note: format="json" 미사용. constrained decoding이 VLM 속도를 저하시키며,
    _parse_json_response()의 3단계 폴백(코드 블록, 직접 JSON, 중괄호 추출)으로 충분.
    """
    logger.debug("get_vision_llm_called", model=settings.VISION_MODEL)
    return get_llm(
        model=settings.VISION_MODEL,
        temperature=0.2,
    )


# ============================================================
# 동시성 제어 래퍼
# ============================================================

async def guarded_ainvoke(
    llm: Runnable | ChatOllama,
    prompt_value: Any,
    model: str,
    request_id: str = "",
) -> Any:
    """
    모델별 세마포어로 감싼 LLM ainvoke.

    Ollama는 GPU 추론을 모델당 직렬 처리하므로,
    동일 모델에 대한 동시 호출 수를 제한하여 Ollama 큐 점유를 방지한다.
    세마포어 획득 → ainvoke → 세마포어 반환의 안전한 흐름을 보장한다.

    Args:
        llm: LangChain Runnable 또는 ChatOllama 인스턴스
        prompt_value: LLM에 전달할 프롬프트 (포맷 완료된 PromptValue 또는 메시지 리스트)
        model: Ollama 모델명 (세마포어 키로 사용)
        request_id: 요청 식별자 (로깅용, 빈 문자열이면 생략)

    Returns:
        LLM 응답 (BaseMessage 또는 구조화 출력 Pydantic 모델)
    """
    await acquire_model_slot(model, request_id)
    try:
        return await llm.ainvoke(prompt_value)
    finally:
        release_model_slot(model)
