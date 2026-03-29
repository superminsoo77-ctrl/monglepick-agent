"""
몽글이 최종 응답 생성 체인.

Solar API가 분석/처리한 데이터(추천 영화, 추천 이유, 감정, 선호 등)를
받아서 사용자에게 자연스럽고 예쁜 대화체로 최종 답변을 생성한다.

몽글이(EXAONE 1.2B LoRA / vLLM)가 서비스의 "목소리" 역할을 담당한다.

처리 흐름:
1. Solar가 생성한 영화 데이터 + 사용자 정보를 프롬프트에 포맷
2. get_conversation_llm() (몽글이, temp=0.5) 호출
3. 에러 시: Solar 데이터 기반 기계적 폴백 텍스트 반환
"""

from __future__ import annotations

import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.agents.chat.models import (
    ExtractedPreferences,
    RankedMovie,
)
from monglepick.config import settings
from monglepick.llm.factory import get_conversation_llm, guarded_ainvoke
from monglepick.prompts.response_generation import (
    QUESTION_RESPONSE_HUMAN_PROMPT,
    QUESTION_RESPONSE_SYSTEM_PROMPT,
    RESPONSE_GENERATION_HUMAN_PROMPT,
    RESPONSE_GENERATION_SYSTEM_PROMPT,
)

logger = structlog.get_logger()


def _format_movie_data(ranked_movies: list[RankedMovie]) -> str:
    """
    추천 영화 리스트를 몽글이에게 전달할 텍스트 형식으로 포맷한다.

    Solar가 생성한 추천 이유, 메타데이터를 모두 포함하여
    몽글이가 자연스러운 대화체로 변환할 수 있도록 구조화한다.

    Args:
        ranked_movies: Solar가 분석/랭킹한 추천 영화 목록

    Returns:
        영화 데이터 포맷 문자열
    """
    parts: list[str] = []
    for movie in ranked_movies:
        # 장르
        genres_str = ", ".join(movie.genres[:4]) if movie.genres else "-"
        # 출연진 (상위 3명)
        cast_str = ", ".join(movie.cast[:3]) if movie.cast else "-"
        # 무드태그
        mood_str = ", ".join(movie.mood_tags[:3]) if movie.mood_tags else "-"
        # OTT 플랫폼
        ott_str = ", ".join(movie.ott_platforms[:3]) if movie.ott_platforms else "-"
        # 개봉연도
        year_str = str(movie.release_year) if movie.release_year else "-"
        # 줄거리 (200자 제한)
        overview = (movie.overview or "")[:200]

        entry = (
            f"[영화 {movie.rank}]\n"
            f"  제목: {movie.title}\n"
            f"  장르: {genres_str}\n"
            f"  감독: {movie.director or '-'}\n"
            f"  출연: {cast_str}\n"
            f"  평점: {movie.rating:.1f}\n"
            f"  개봉: {year_str}\n"
            f"  분위기: {mood_str}\n"
            f"  시청 가능: {ott_str}\n"
            f"  줄거리: {overview}\n"
            f"  추천 이유: {movie.explanation or '-'}"
        )
        parts.append(entry)

    return "\n\n".join(parts)


def _format_preferences(preferences: ExtractedPreferences | None) -> str:
    """
    사용자 선호 조건을 텍스트로 포맷한다.

    Args:
        preferences: 사용자 선호 조건

    Returns:
        선호 조건 포맷 문자열
    """
    if not preferences:
        return "(파악된 선호 없음)"

    parts: list[str] = []
    if preferences.genre_preference:
        parts.append(f"선호 장르: {preferences.genre_preference}")
    if preferences.mood:
        parts.append(f"원하는 분위기: {preferences.mood}")
    if preferences.viewing_context:
        parts.append(f"시청 상황: {preferences.viewing_context}")
    if preferences.platform:
        parts.append(f"시청 플랫폼: {preferences.platform}")
    if preferences.reference_movies:
        parts.append(f"좋아하는 영화: {', '.join(preferences.reference_movies)}")
    if preferences.era:
        parts.append(f"선호 시대: {preferences.era}")
    if preferences.exclude:
        parts.append(f"제외 조건: {preferences.exclude}")

    return ", ".join(parts) if parts else "(파악된 선호 없음)"


def _build_fallback_response(
    ranked_movies: list[RankedMovie],
    emotion: str | None = None,
) -> str:
    """
    몽글이 LLM 호출 실패 시, Solar 데이터를 기계적으로 조합한 폴백 응답.

    LLM 없이도 최소한의 응답을 보장한다.

    Args:
        ranked_movies: 추천 영화 목록
        emotion: 사용자 감정

    Returns:
        폴백 응답 텍스트
    """
    parts = ["추천 영화를 찾았어요! 🎬\n"]
    for movie in ranked_movies:
        genres_str = ", ".join(movie.genres[:4]) if movie.genres else "-"
        cast_str = ", ".join(movie.cast[:3]) if movie.cast else ""
        mood_str = ", ".join(movie.mood_tags[:3]) if movie.mood_tags else ""
        ott_str = ", ".join(movie.ott_platforms[:3]) if movie.ott_platforms else ""
        year_str = f" ({movie.release_year})" if movie.release_year else ""

        # 줄거리 (150자, 문장 단위 절단)
        overview_text = ""
        if movie.overview:
            raw = movie.overview[:150]
            last_period = raw.rfind(".")
            if last_period > 30:
                overview_text = raw[:last_period + 1]
            else:
                overview_text = raw + "..."

        card = (
            f"{movie.rank}. **{movie.title}**{year_str}\n"
            f"   - 장르: {genres_str}\n"
            f"   - 감독: {movie.director or '-'}\n"
        )
        if cast_str:
            card += f"   - 출연: {cast_str}\n"
        card += f"   - 평점: {movie.rating:.1f}\n"
        if mood_str:
            card += f"   - 분위기: {mood_str}\n"
        if ott_str:
            card += f"   - 시청 가능: {ott_str}\n"
        if overview_text:
            card += f"   - 줄거리: {overview_text}\n"
        if movie.explanation:
            card += f"   > {movie.explanation}\n"
        parts.append(card)

    return "\n".join(parts)


# ============================================================
# 추천 응답 생성 (메인)
# ============================================================

async def generate_recommendation_response(
    ranked_movies: list[RankedMovie],
    emotion: str | None = None,
    preferences: ExtractedPreferences | None = None,
    user_message: str = "",
) -> str:
    """
    Solar가 처리한 추천 데이터를 몽글이가 자연스러운 대화체로 변환한다.

    Args:
        ranked_movies: Solar가 분석/랭킹한 추천 영화 목록 (explanation 포함)
        emotion: Solar가 분류한 사용자 감정
        preferences: Solar가 추출한 사용자 선호 조건
        user_message: 사용자 원래 입력 메시지

    Returns:
        몽글이가 작성한 자연스러운 추천 응답 텍스트
        - 에러 시: Solar 데이터 기반 기계적 폴백 텍스트
    """
    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_GENERATION_SYSTEM_PROMPT),
        ("human", RESPONSE_GENERATION_HUMAN_PROMPT),
    ])

    # 몽글이 LLM (hybrid: EXAONE 1.2B LoRA/vLLM, local: EXAONE 32B)
    llm = get_conversation_llm()

    # 입력 데이터 포맷
    movie_data = _format_movie_data(ranked_movies)
    pref_text = _format_preferences(preferences)

    inputs = {
        "emotion": emotion or "미감지",
        "preferences": pref_text,
        "user_message": user_message or "(메시지 없음)",
        "movie_data": movie_data,
    }

    logger.info(
        "response_generation_chain_start",
        movie_count=len(ranked_movies),
        emotion=emotion,
        preferences_preview=pref_text[:100],
        user_message_preview=user_message[:50],
    )

    try:
        llm_start = time.perf_counter()
        prompt_value = await prompt.ainvoke(inputs)
        logger.debug(
            "response_generation_prompt_formatted",
            prompt_preview=str(prompt_value)[:300],
        )

        # 몽글이 호출 (세마포어 적용)
        response = await guarded_ainvoke(
            llm, prompt_value, model=settings.CONVERSATION_MODEL,
        )
        elapsed_ms = (time.perf_counter() - llm_start) * 1000

        # LangChain BaseMessage → 문자열 추출
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip() if isinstance(text, str) else str(text).strip()

        logger.info(
            "response_generation_completed",
            response_preview=text[:80],
            elapsed_ms=round(elapsed_ms, 1),
            movie_count=len(ranked_movies),
            model=settings.CONVERSATION_MODEL,
        )
        return text

    except Exception as e:
        logger.error(
            "response_generation_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            movie_count=len(ranked_movies),
        )
        # 폴백: Solar 데이터를 기계적으로 조합
        return _build_fallback_response(ranked_movies, emotion)


# ============================================================
# 후속 질문 응답 생성
# ============================================================

async def generate_question_response(
    question: str,
    hints: list[str] | None = None,
    emotion: str | None = None,
    preferences: ExtractedPreferences | None = None,
    user_message: str = "",
) -> str:
    """
    Solar가 생성한 후속 질문을 몽글이가 자연스러운 대화체로 변환한다.

    Args:
        question: Solar가 생성한 후속 질문 텍스트
        hints: 질문에 대한 힌트 목록 (선택지)
        emotion: Solar가 분류한 사용자 감정
        preferences: Solar가 추출한 사용자 선호 조건
        user_message: 사용자 원래 입력 메시지

    Returns:
        몽글이가 작성한 자연스러운 질문 응답 텍스트
        - 에러 시: Solar 질문 그대로 반환
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUESTION_RESPONSE_SYSTEM_PROMPT),
        ("human", QUESTION_RESPONSE_HUMAN_PROMPT),
    ])

    llm = get_conversation_llm()

    pref_text = _format_preferences(preferences)
    hints_text = ", ".join(hints) if hints else "(없음)"

    inputs = {
        "emotion": emotion or "미감지",
        "preferences": pref_text,
        "user_message": user_message or "(메시지 없음)",
        "question": question,
        "hints": hints_text,
    }

    logger.info(
        "question_response_chain_start",
        question_preview=question[:50],
        emotion=emotion,
    )

    try:
        llm_start = time.perf_counter()
        prompt_value = await prompt.ainvoke(inputs)

        response = await guarded_ainvoke(
            llm, prompt_value, model=settings.CONVERSATION_MODEL,
        )
        elapsed_ms = (time.perf_counter() - llm_start) * 1000

        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip() if isinstance(text, str) else str(text).strip()

        logger.info(
            "question_response_completed",
            response_preview=text[:80],
            elapsed_ms=round(elapsed_ms, 1),
            model=settings.CONVERSATION_MODEL,
        )
        return text

    except Exception as e:
        logger.error(
            "question_response_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
        )
        # 폴백: Solar 질문 그대로 반환
        return question
