"""
감정 분석 체인 (§6-2 Node 3).

사용자 메시지에서 감정을 분석하고 무드 태그를 추출하는 체인.
Qwen 14B 구조화 출력으로 EmotionResult를 반환한다.

처리 흐름:
1. get_emotion_llm() (Qwen 14B, structured output) 호출
2. emotion이 있으면 매핑 테이블에서 mood_tags 보완 (합집합)
3. mood_tags를 MOOD_WHITELIST (25개)로 필터링
4. 에러 시: EmotionResult(emotion=None, mood_tags=[])
"""

from __future__ import annotations

import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.agents.chat.models import EmotionResult
from monglepick.config import settings
from monglepick.data_pipeline.preprocessor import MOOD_WHITELIST
from monglepick.llm.factory import get_emotion_llm
from monglepick.prompts.emotion import (
    EMOTION_HUMAN_PROMPT,
    EMOTION_SYSTEM_PROMPT,
    EMOTION_TO_MOOD_MAP,
)

logger = structlog.get_logger()


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
            "mood_tags_filtered",
            removed=removed,
            remaining=valid_tags,
        )
    return valid_tags


async def analyze_emotion(
    current_input: str,
    recent_messages: str = "",
) -> EmotionResult:
    """
    사용자 메시지의 감정을 분석하고 무드 태그를 추출한다.

    LLM이 추출한 무드 태그에 감정→무드 매핑 테이블의 태그를 합집합으로 보완하고,
    전체를 MOOD_WHITELIST로 필터링한다.

    Args:
        current_input: 현재 사용자 입력 텍스트
        recent_messages: 최근 대화 이력 (포맷된 문자열)

    Returns:
        EmotionResult(emotion, mood_tags)
        - 에러 시: EmotionResult(emotion=None, mood_tags=[])
    """
    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", EMOTION_SYSTEM_PROMPT),
        ("human", EMOTION_HUMAN_PROMPT),
    ])

    # 구조화 출력 LLM (Qwen 14B, EmotionResult 자동 파싱)
    llm = get_emotion_llm()

    # 입력 변수
    inputs = {
        "current_input": current_input,
        "recent_messages": recent_messages or "(대화 시작)",
    }

    logger.info(
        "emotion_chain_start",
        input_preview=current_input[:100],
        recent_messages_preview=recent_messages[:100] if recent_messages else "(없음)",
    )

    try:
        # LLM 파이프라인 타이밍 측정 시작 (프롬프트 포맷 + LLM 호출)
        llm_start = time.perf_counter()

        # 프롬프트 포맷 → LLM 호출 (명시적 2단계)
        prompt_value = await prompt.ainvoke(inputs)
        logger.debug(
            "emotion_chain_prompt_formatted",
            prompt_preview=str(prompt_value)[:300],
        )
        # 전체 프롬프트 텍스트 디버그 로그 (상세 디버깅용)
        logger.debug(
            "emotion_chain_prompt_full",
            prompt_text=str(prompt_value),
            model=settings.EMOTION_MODEL,
        )
        result: EmotionResult = await llm.ainvoke(prompt_value)

        # LLM 응답 시간 계산 (밀리초 단위)
        elapsed_ms = (time.perf_counter() - llm_start) * 1000

        # LLM 원시 응답 디버그 로그 (파싱 전 전체 응답)
        logger.debug(
            "emotion_chain_llm_raw_response",
            raw_response=str(result),
            model=settings.EMOTION_MODEL,
        )

        logger.info(
            "emotion_chain_llm_response",
            raw_emotion=result.emotion,
            raw_mood_tags=result.mood_tags,
        )

        # 감정이 감지되면 매핑 테이블에서 무드 태그 보완 (합집합)
        if result.emotion and result.emotion in EMOTION_TO_MOOD_MAP:
            mapped_tags = EMOTION_TO_MOOD_MAP[result.emotion]
            # LLM 추출 태그 + 매핑 태그 합집합 (순서 유지, 중복 제거)
            combined = list(dict.fromkeys(result.mood_tags + mapped_tags))
            result = EmotionResult(
                emotion=result.emotion,
                mood_tags=combined,
            )

        # MOOD_WHITELIST 필터링
        result = EmotionResult(
            emotion=result.emotion,
            mood_tags=_validate_mood_tags(result.mood_tags),
        )

        logger.info(
            "emotion_analyzed",
            emotion=result.emotion,
            mood_tags=result.mood_tags,
            input_preview=current_input[:50],
            elapsed_ms=round(elapsed_ms, 1),
            model=settings.EMOTION_MODEL,
        )
        return result

    except Exception as e:
        logger.error(
            "emotion_analysis_error",
            error=str(e),
            input_preview=current_input[:50],
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
        )
        return EmotionResult(emotion=None, mood_tags=[])
