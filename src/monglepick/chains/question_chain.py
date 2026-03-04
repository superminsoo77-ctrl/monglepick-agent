"""
후속 질문 생성 체인 (§6-2 Node 5).

사용자 선호 조건이 부족할 때 자연스러운 후속 질문을 생성하는 체인.
EXAONE 32B (자유 텍스트)로 실행한다.

처리 흐름:
1. 부족 필드를 가중치 내림차순으로 정렬
2. 파악된 선호 + 부족 필드 + 감정을 프롬프트에 포함
3. get_question_llm() (EXAONE 32B, 자유 텍스트) 호출
4. 에러 시: DEFAULT_QUESTIONS[최고가중치_부족필드] 반환
"""

from __future__ import annotations

import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.agents.chat.models import (
    PREFERENCE_WEIGHTS,
    ExtractedPreferences,
)
from monglepick.config import settings
from monglepick.llm.factory import get_question_llm
from monglepick.prompts.question import QUESTION_HUMAN_PROMPT, QUESTION_SYSTEM_PROMPT

logger = structlog.get_logger()

# ============================================================
# 기본 질문 (LLM 에러 시 fallback)
# ============================================================

DEFAULT_QUESTIONS: dict[str, str] = {
    "genre_preference": "어떤 장르의 영화를 좋아하세요? 🎬",
    "mood": "오늘 어떤 분위기의 영화가 끌리세요?",
    "reference_movies": "최근에 재미있게 본 영화가 있으세요?",
    "viewing_context": "누구와 함께 볼 예정이에요?",
    "platform": "어디서 볼 계획이에요? (넷플릭스, 극장 등)",
    "era": "최신 영화가 좋으세요, 클래식도 괜찮으세요?",
    "exclude": "혹시 피하고 싶은 장르나 주제가 있으세요?",
}

# 모든 필드가 채워졌을 때의 기본 질문
DEFAULT_FALLBACK_QUESTION = "어떤 영화를 찾으시는지 좀 더 알려주세요! 🎬"


def _get_missing_fields(prefs: ExtractedPreferences) -> list[tuple[str, float]]:
    """
    채워지지 않은 선호 필드를 가중치 내림차순으로 반환한다.

    Args:
        prefs: 현재까지 파악된 사용자 선호 조건

    Returns:
        [(필드명, 가중치), ...] 리스트 (가중치 내림차순)
    """
    missing = []

    # 각 필드를 확인하여 None/빈 값이면 부족 필드로 추가
    if not prefs.genre_preference:
        missing.append(("genre_preference", PREFERENCE_WEIGHTS["genre_preference"]))
    if not prefs.mood:
        missing.append(("mood", PREFERENCE_WEIGHTS["mood"]))
    if not prefs.reference_movies:
        missing.append(("reference_movies", PREFERENCE_WEIGHTS["reference_movies"]))
    if not prefs.viewing_context:
        missing.append(("viewing_context", PREFERENCE_WEIGHTS["viewing_context"]))
    if not prefs.platform:
        missing.append(("platform", PREFERENCE_WEIGHTS["platform"]))
    if not prefs.era:
        missing.append(("era", PREFERENCE_WEIGHTS["era"]))
    if not prefs.exclude:
        missing.append(("exclude", PREFERENCE_WEIGHTS["exclude"]))

    # 가중치 내림차순 정렬
    missing.sort(key=lambda x: x[1], reverse=True)
    return missing


def _format_known_preferences(prefs: ExtractedPreferences) -> str:
    """
    파악된 선호 조건을 프롬프트용 문자열로 포맷한다.

    Args:
        prefs: 현재까지 파악된 사용자 선호 조건

    Returns:
        포맷된 문자열 (비어있으면 "(아직 없음)")
    """
    parts = []
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

    return "\n".join(parts) if parts else "(아직 없음)"


async def generate_question(
    extracted_preferences: ExtractedPreferences,
    emotion: str | None = None,
    turn_count: int = 0,
) -> str:
    """
    부족한 선호 정보를 파악하기 위한 자연스러운 후속 질문을 생성한다.

    Args:
        extracted_preferences: 현재까지 파악된 선호 조건
        emotion: 감지된 감정 (None이면 감지 안 됨)
        turn_count: 현재 대화 턴 수

    Returns:
        한국어 후속 질문 문자열
        - 에러 시: DEFAULT_QUESTIONS에서 최고 가중치 부족 필드의 기본 질문
    """
    # 부족 필드 파악
    missing = _get_missing_fields(extracted_preferences)

    # 모든 필드가 채워져 있으면 기본 질문
    if not missing:
        return DEFAULT_FALLBACK_QUESTION

    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUESTION_SYSTEM_PROMPT),
        ("human", QUESTION_HUMAN_PROMPT),
    ])

    # 자유 텍스트 LLM (EXAONE 32B, temp=0.5)
    llm = get_question_llm()

    # 부족 필드를 포맷
    missing_str = "\n".join(
        f"- {field} (가중치 {weight})" for field, weight in missing
    )

    # 입력 변수
    inputs = {
        "known_preferences": _format_known_preferences(extracted_preferences),
        "missing_fields": missing_str,
        "emotion": emotion or "감지 안 됨",
        "turn_count": str(turn_count),
    }

    logger.info(
        "question_chain_start",
        missing_fields=[f[0] for f in missing],
        turn_count=turn_count,
        emotion=emotion,
        known_prefs_preview=_format_known_preferences(extracted_preferences)[:200],
    )

    try:
        # LLM 파이프라인 타이밍 측정 시작 (프롬프트 포맷 + LLM 호출)
        llm_start = time.perf_counter()

        # 프롬프트 포맷 → LLM 호출 (명시적 2단계)
        prompt_value = await prompt.ainvoke(inputs)
        logger.debug(
            "question_chain_prompt_formatted",
            prompt_preview=str(prompt_value)[:300],
        )
        # 전체 프롬프트 텍스트 디버그 로그 (상세 디버깅용)
        logger.debug(
            "question_chain_prompt_full",
            prompt_text=str(prompt_value),
            model=settings.QUESTION_MODEL,
        )
        response = await llm.ainvoke(prompt_value)

        # LLM 응답 시간 계산 (밀리초 단위)
        elapsed_ms = (time.perf_counter() - llm_start) * 1000

        # LLM 원시 응답 디버그 로그 (파싱 전 전체 응답)
        logger.debug(
            "question_chain_llm_raw_response",
            raw_response=str(response),
            model=settings.QUESTION_MODEL,
        )

        # LangChain BaseMessage → 문자열 추출
        question = response.content if hasattr(response, "content") else str(response)
        question = question.strip() if isinstance(question, str) else str(question).strip()

        logger.info(
            "question_generated",
            question_preview=question[:50],
            missing_count=len(missing),
            turn_count=turn_count,
            elapsed_ms=round(elapsed_ms, 1),
            model=settings.QUESTION_MODEL,
        )
        return question

    except Exception as e:
        logger.error(
            "question_generation_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
        )
        # fallback: 최고 가중치 부족 필드의 기본 질문
        top_missing_field = missing[0][0]
        return DEFAULT_QUESTIONS.get(top_missing_field, DEFAULT_FALLBACK_QUESTION)
