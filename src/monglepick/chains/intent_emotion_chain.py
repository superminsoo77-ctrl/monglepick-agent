"""
의도+감정 통합 체인.

기존 intent_chain.py + emotion_chain.py의 로직을 1회 LLM 호출로 통합한다.
동일 모델(qwen3.5:35b-a3b)로 동일 입력을 분석하므로
2번 순차 호출(~95초) → 1번 통합 호출(~50초)로 지연 시간을 절감한다.

처리 흐름:
1. ChatPromptTemplate으로 통합 프롬프트 구성
2. get_intent_emotion_llm() (qwen3.5, structured output) 호출
3. confidence < 0.6 → intent="general"로 보정 (intent_chain에서 가져옴)
4. emotion→무드 매핑 + MOOD_WHITELIST 필터링 (emotion_chain에서 가져옴)
5. 에러 시: 1회 재시도 → 실패 시 IntentEmotionResult(intent="general", confidence=0.0)
"""

from __future__ import annotations

import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.agents.chat.models import IntentEmotionResult
from monglepick.config import settings
from monglepick.data_pipeline.preprocessor import MOOD_WHITELIST
from monglepick.llm.factory import get_intent_emotion_llm
from monglepick.prompts.emotion import EMOTION_TO_MOOD_MAP
from monglepick.prompts.intent_emotion import (
    INTENT_EMOTION_HUMAN_PROMPT,
    INTENT_EMOTION_SYSTEM_PROMPT,
)

logger = structlog.get_logger()

# 신뢰도 임계값: 이 값 미만이면 general로 보정 (intent_chain과 동일)
CONFIDENCE_THRESHOLD = 0.6

# 최대 재시도 횟수
MAX_RETRIES = 1


def _validate_mood_tags(tags: list[str]) -> list[str]:
    """
    무드 태그를 MOOD_WHITELIST (25개)로 필터링한다.

    화이트리스트에 없는 태그는 제거하여 일관된 무드 태그 체계를 유지한다.

    Args:
        tags: 검증할 무드 태그 목록

    Returns:
        화이트리스트에 포함된 태그만 남긴 목록
    """
    valid_tags = [tag for tag in tags if tag in MOOD_WHITELIST]
    if len(valid_tags) < len(tags):
        removed = [tag for tag in tags if tag not in MOOD_WHITELIST]
        logger.debug(
            "intent_emotion_mood_tags_filtered",
            removed=removed,
            remaining=valid_tags,
        )
    return valid_tags


async def classify_intent_and_emotion(
    current_input: str,
    recent_messages: str = "",
) -> IntentEmotionResult:
    """
    사용자 메시지의 의도와 감정을 동시에 분석한다 (1회 LLM 호출).

    기존 classify_intent() + analyze_emotion() 2회 호출을 통합.
    - confidence < 0.6 → intent="general" 보정 (intent_chain에서 가져옴)
    - emotion→무드 매핑 + MOOD_WHITELIST 필터링 (emotion_chain에서 가져옴)
    - 에러 시 1회 재시도, 최종 실패 시 fallback 반환

    Args:
        current_input: 현재 사용자 입력 텍스트
        recent_messages: 최근 대화 이력 (포맷된 문자열)

    Returns:
        IntentEmotionResult(intent, confidence, emotion, mood_tags)
        - 에러 시: IntentEmotionResult(intent="general", confidence=0.0)
    """
    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_EMOTION_SYSTEM_PROMPT),
        ("human", INTENT_EMOTION_HUMAN_PROMPT),
    ])

    # 구조화 출력 LLM (qwen3.5:35b-a3b, IntentEmotionResult 자동 파싱)
    llm = get_intent_emotion_llm()

    # 입력 변수
    inputs = {
        "current_input": current_input,
        "recent_messages": recent_messages or "(대화 시작)",
    }

    logger.info(
        "intent_emotion_chain_start",
        input_preview=current_input[:100],
        recent_messages_preview=recent_messages[:100] if recent_messages else "(없음)",
    )

    # 최대 1회 재시도
    for attempt in range(MAX_RETRIES + 1):
        try:
            # LLM 파이프라인 타이밍 측정
            llm_start = time.perf_counter()

            # 프롬프트 포맷 → LLM 호출 (명시적 2단계)
            prompt_value = await prompt.ainvoke(inputs)
            logger.debug(
                "intent_emotion_chain_prompt_formatted",
                prompt_preview=str(prompt_value)[:300],
            )
            result: IntentEmotionResult = await llm.ainvoke(prompt_value)

            # LLM 응답 시간 (밀리초)
            elapsed_ms = (time.perf_counter() - llm_start) * 1000

            logger.info(
                "intent_emotion_chain_llm_response",
                raw_intent=result.intent,
                raw_confidence=result.confidence,
                raw_emotion=result.emotion,
                raw_mood_tags=result.mood_tags,
                elapsed_ms=round(elapsed_ms, 1),
            )

            # ── 의도 보정: 신뢰도 < 0.6 → general ──
            if result.confidence < CONFIDENCE_THRESHOLD:
                logger.info(
                    "intent_emotion_confidence_below_threshold",
                    original_intent=result.intent,
                    confidence=result.confidence,
                    threshold=CONFIDENCE_THRESHOLD,
                )
                result = IntentEmotionResult(
                    intent="general",
                    confidence=result.confidence,
                    emotion=result.emotion,
                    mood_tags=result.mood_tags,
                )

            # ── 감정→무드 매핑 보완 (합집합) ──
            if result.emotion and result.emotion in EMOTION_TO_MOOD_MAP:
                mapped_tags = EMOTION_TO_MOOD_MAP[result.emotion]
                # LLM 추출 태그 + 매핑 태그 합집합 (순서 유지, 중복 제거)
                combined = list(dict.fromkeys(result.mood_tags + mapped_tags))
                result = IntentEmotionResult(
                    intent=result.intent,
                    confidence=result.confidence,
                    emotion=result.emotion,
                    mood_tags=combined,
                )

            # ── MOOD_WHITELIST 필터링 ──
            result = IntentEmotionResult(
                intent=result.intent,
                confidence=result.confidence,
                emotion=result.emotion,
                mood_tags=_validate_mood_tags(result.mood_tags),
            )

            logger.info(
                "intent_emotion_classified",
                intent=result.intent,
                confidence=result.confidence,
                emotion=result.emotion,
                mood_tags=result.mood_tags,
                input_preview=current_input[:50],
                elapsed_ms=round(elapsed_ms, 1),
                model=settings.INTENT_MODEL,
            )
            return result

        except Exception as e:
            logger.warning(
                "intent_emotion_classification_error",
                attempt=attempt + 1,
                max_retries=MAX_RETRIES + 1,
                error=str(e),
                input_preview=current_input[:50],
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc(),
            )
            # 마지막 시도에서 실패 → fallback 반환
            if attempt >= MAX_RETRIES:
                logger.error(
                    "intent_emotion_classification_fallback",
                    error=str(e),
                    error_type=type(e).__name__,
                    stack_trace=traceback.format_exc(),
                )
                return IntentEmotionResult(
                    intent="general",
                    confidence=0.0,
                    emotion=None,
                    mood_tags=[],
                )

    # 도달 불가 — 안전망
    return IntentEmotionResult(
        intent="general",
        confidence=0.0,
        emotion=None,
        mood_tags=[],
    )
