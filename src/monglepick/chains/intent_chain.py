"""
의도 분류 체인 (§6-2 Node 2).

사용자 메시지를 6가지 의도 중 하나로 분류하는 체인.
Qwen 14B 구조화 출력으로 IntentResult를 반환한다.

의도 종류: recommend, search, info, theater, booking, general
신뢰도 < 0.6이면 general로 보정한다.

처리 흐름:
1. ChatPromptTemplate으로 프롬프트 구성
2. get_intent_llm() (Qwen 14B, structured output) 호출
3. confidence < 0.6 → intent="general"로 보정
4. 에러 시: 1회 재시도 → 실패 시 IntentResult(intent="general", confidence=0.0)
"""

from __future__ import annotations

import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.agents.chat.models import IntentResult
from monglepick.config import settings
from monglepick.llm.factory import get_intent_llm
from monglepick.prompts.intent import INTENT_HUMAN_PROMPT, INTENT_SYSTEM_PROMPT

logger = structlog.get_logger()

# 신뢰도 임계값: 이 값 미만이면 general로 보정
CONFIDENCE_THRESHOLD = 0.6

# 최대 재시도 횟수
MAX_RETRIES = 1


async def classify_intent(
    current_input: str,
    recent_messages: str = "",
) -> IntentResult:
    """
    사용자 메시지의 의도를 분류한다.

    Args:
        current_input: 현재 사용자 입력 텍스트
        recent_messages: 최근 대화 이력 (포맷된 문자열)

    Returns:
        IntentResult(intent, confidence)
        - 에러 시: IntentResult(intent="general", confidence=0.0)
    """
    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_SYSTEM_PROMPT),
        ("human", INTENT_HUMAN_PROMPT),
    ])

    # 구조화 출력 LLM (Qwen 14B, IntentResult 자동 파싱)
    llm = get_intent_llm()

    # 입력 변수
    inputs = {
        "current_input": current_input,
        "recent_messages": recent_messages or "(대화 시작)",
    }

    logger.info(
        "intent_chain_start",
        input_preview=current_input[:100],
        recent_messages_preview=recent_messages[:100] if recent_messages else "(없음)",
    )

    # 최대 1회 재시도
    for attempt in range(MAX_RETRIES + 1):
        try:
            # LLM 파이프라인 타이밍 측정 시작 (프롬프트 포맷 + LLM 호출)
            llm_start = time.perf_counter()

            # 프롬프트 포맷 → LLM 호출 (명시적 2단계)
            prompt_value = await prompt.ainvoke(inputs)
            logger.debug(
                "intent_chain_prompt_formatted",
                prompt_preview=str(prompt_value)[:300],
            )
            # 전체 프롬프트 텍스트 디버그 로그 (상세 디버깅용)
            logger.debug(
                "intent_chain_prompt_full",
                prompt_text=str(prompt_value),
                model=settings.INTENT_MODEL,
            )
            result: IntentResult = await llm.ainvoke(prompt_value)

            # LLM 응답 시간 계산 (밀리초 단위)
            elapsed_ms = (time.perf_counter() - llm_start) * 1000

            # LLM 원시 응답 디버그 로그 (파싱 전 전체 응답)
            logger.debug(
                "intent_chain_llm_raw_response",
                raw_response=str(result),
                model=settings.INTENT_MODEL,
            )

            # 신뢰도 보정: 임계값 미만이면 general로 변경
            if result.confidence < CONFIDENCE_THRESHOLD:
                logger.info(
                    "intent_confidence_below_threshold",
                    original_intent=result.intent,
                    confidence=result.confidence,
                    threshold=CONFIDENCE_THRESHOLD,
                )
                result = IntentResult(intent="general", confidence=result.confidence)

            logger.info(
                "intent_classified",
                intent=result.intent,
                confidence=result.confidence,
                input_preview=current_input[:50],
                elapsed_ms=round(elapsed_ms, 1),
                model=settings.INTENT_MODEL,
            )
            return result

        except Exception as e:
            logger.warning(
                "intent_classification_error",
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
                    "intent_classification_fallback",
                    error=str(e),
                    error_type=type(e).__name__,
                    stack_trace=traceback.format_exc(),
                )
                return IntentResult(intent="general", confidence=0.0)

    # 도달 불가 — 안전망
    return IntentResult(intent="general", confidence=0.0)
