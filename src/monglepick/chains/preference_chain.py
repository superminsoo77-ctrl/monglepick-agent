"""
선호 추출 체인 (§6-2 Node 4, Intent-First + Dynamic Filter).

사용자 메시지에서 영화 추천 의도(user_intent), 동적 필터(dynamic_filters),
검색 키워드(search_keywords), 구조화된 선호 필드를 추출하고
이전 선호 조건과 병합하는 체인.
EXAONE 32B 구조화 출력으로 ExtractedPreferences를 반환한다.

처리 흐름:
1. 이전 선호 조건을 문자열로 포맷하여 프롬프트에 포함
2. get_preference_llm() (EXAONE 32B, structured output) 호출
3. merge_preferences(previous, current) 로 병합 (동적 필터 포함)
4. 에러 시: 이전 선호 조건 그대로 반환 (첫 턴이면 빈 ExtractedPreferences)
"""

from __future__ import annotations

import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.agents.chat.models import (
    ExtractedPreferences,
    merge_preferences,
)
from monglepick.config import settings
from monglepick.llm.factory import get_preference_llm, guarded_ainvoke
from monglepick.prompts.preference import (
    PREFERENCE_HUMAN_PROMPT,
    PREFERENCE_SYSTEM_PROMPT,
)

logger = structlog.get_logger()


def _format_existing_preferences(prefs: ExtractedPreferences | None) -> str:
    """
    이전 선호 조건을 프롬프트용 문자열로 포맷한다.

    Intent-First 필드(user_intent, dynamic_filters, search_keywords)도 포함하여
    LLM이 이전에 파악된 조건을 인지하고 중복 추출을 방지한다.

    Args:
        prefs: 이전 턴까지 누적된 선호 조건 (None이면 없음)

    Returns:
        프롬프트에 삽입할 선호 조건 문자열
    """
    if prefs is None:
        return "(아직 파악된 선호 조건 없음)"

    parts = []

    # ── Intent-First 필드 ──
    if prefs.user_intent:
        parts.append(f"- 추천 의도: {prefs.user_intent}")
    if prefs.dynamic_filters:
        filter_strs = [
            f"{f.field} {f.operator} {f.value}" for f in prefs.dynamic_filters
        ]
        parts.append(f"- 동적 필터: {', '.join(filter_strs)}")
    if prefs.search_keywords:
        parts.append(f"- 검색 키워드: {', '.join(prefs.search_keywords)}")

    # ── 기존 구조화 필드 ──
    if prefs.genre_preference:
        parts.append(f"- 장르: {prefs.genre_preference}")
    if prefs.mood:
        parts.append(f"- 분위기: {prefs.mood}")
    if prefs.viewing_context:
        parts.append(f"- 시청 상황: {prefs.viewing_context}")
    if prefs.platform:
        parts.append(f"- 플랫폼: {prefs.platform}")
    if prefs.reference_movies:
        parts.append(f"- 참조 영화: {', '.join(prefs.reference_movies)}")
    if prefs.era:
        parts.append(f"- 시대: {prefs.era}")
    if prefs.exclude:
        parts.append(f"- 제외: {prefs.exclude}")

    if not parts:
        return "(아직 파악된 선호 조건 없음)"

    return "\n".join(parts)


async def extract_preferences(
    current_input: str,
    previous_preferences: ExtractedPreferences | None = None,
) -> ExtractedPreferences:
    """
    사용자 메시지에서 선호 조건을 추출하고 이전 선호와 병합한다.

    Args:
        current_input: 현재 사용자 입력 텍스트
        previous_preferences: 이전 턴까지 누적된 선호 조건

    Returns:
        병합된 ExtractedPreferences
        - 에러 시: 이전 선호 조건 그대로 (첫 턴이면 빈 ExtractedPreferences)
    """
    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", PREFERENCE_SYSTEM_PROMPT),
        ("human", PREFERENCE_HUMAN_PROMPT),
    ])

    # 구조화 출력 LLM (EXAONE 32B, ExtractedPreferences 자동 파싱)
    llm = get_preference_llm()

    # 이전 선호 조건을 포맷
    existing_prefs_str = _format_existing_preferences(previous_preferences)

    # 입력 변수
    inputs = {
        "current_input": current_input,
        "existing_prefs": existing_prefs_str,
    }

    logger.info(
        "preference_chain_start",
        input_preview=current_input[:100],
        existing_prefs_preview=existing_prefs_str[:200],
    )

    try:
        # LLM 파이프라인 타이밍 측정 시작 (프롬프트 포맷 + LLM 호출)
        llm_start = time.perf_counter()

        # 프롬프트 포맷 → LLM 호출 (명시적 2단계)
        prompt_value = await prompt.ainvoke(inputs)
        logger.debug(
            "preference_chain_prompt_formatted",
            prompt_preview=str(prompt_value)[:300],
        )
        # 전체 프롬프트 텍스트 디버그 로그 (상세 디버깅용)
        logger.debug(
            "preference_chain_prompt_full",
            prompt_text=str(prompt_value),
            model=settings.PREFERENCE_MODEL,
        )
        # 모델별 세마포어로 동시 호출 제한 (Ollama 큐 점유 방지)
        extracted: ExtractedPreferences = await guarded_ainvoke(
            llm, prompt_value, model=settings.PREFERENCE_MODEL,
        )

        # LLM 응답 시간 계산 (밀리초 단위)
        elapsed_ms = (time.perf_counter() - llm_start) * 1000

        # LLM 원시 응답 디버그 로그 (파싱 전 전체 응답)
        logger.debug(
            "preference_chain_llm_raw_response",
            raw_response=str(extracted),
            model=settings.PREFERENCE_MODEL,
        )

        logger.info(
            "preference_chain_llm_response",
            raw_user_intent=extracted.user_intent[:100] if extracted.user_intent else "",
            raw_dynamic_filters=[
                {"field": f.field, "op": f.operator, "value": f.value}
                for f in extracted.dynamic_filters
            ] if extracted.dynamic_filters else [],
            raw_search_keywords=extracted.search_keywords,
            raw_genre=extracted.genre_preference,
            raw_mood=extracted.mood,
            raw_context=extracted.viewing_context,
            raw_reference=extracted.reference_movies,
            raw_era=extracted.era,
        )

        # 이전 선호와 병합
        merged = merge_preferences(previous_preferences, extracted)

        logger.info(
            "preferences_extracted",
            user_intent=merged.user_intent[:80] if merged.user_intent else "",
            dynamic_filter_count=len(merged.dynamic_filters),
            search_keywords=merged.search_keywords[:5],
            genre=merged.genre_preference,
            mood=merged.mood,
            reference_movies=merged.reference_movies,
            input_preview=current_input[:50],
            elapsed_ms=round(elapsed_ms, 1),
            model=settings.PREFERENCE_MODEL,
        )
        return merged

    except Exception as e:
        logger.error(
            "preference_extraction_error",
            error=str(e),
            input_preview=current_input[:50],
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
        )
        # 에러 시: 이전 선호 유지 (첫 턴이면 빈 선호)
        return previous_preferences or ExtractedPreferences()
