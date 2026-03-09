"""
추천 이유 생성 체인 (§6-2 Node 9).

추천된 영화에 대해 사용자 맞춤 추천 이유를 생성하는 체인.
EXAONE 32B (자유 텍스트)로 실행한다.

처리 흐름:
1. 영화 정보 + 사용자 상태를 프롬프트에 포함
2. get_explanation_llm() (EXAONE 32B, 자유 텍스트) 호출
3. 배치 버전: asyncio.gather로 3~5편 병렬 생성
4. 에러 시: _build_fallback_explanation(movie) 메타데이터 기반 템플릿
"""

from __future__ import annotations

import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.agents.chat.models import (
    CandidateMovie,
    ExtractedPreferences,
    RankedMovie,
    ScoreDetail,
)
from monglepick.config import settings
from monglepick.llm.factory import get_explanation_llm, guarded_ainvoke
from monglepick.prompts.explanation import (
    EXPLANATION_HUMAN_PROMPT,
    EXPLANATION_SYSTEM_PROMPT,
)

logger = structlog.get_logger()


def _build_fallback_explanation(movie: dict) -> str:
    """
    LLM 에러 시 메타데이터 기반으로 기본 추천 이유를 생성한다.

    Args:
        movie: 영화 정보 dict (title, genres, rating, director 등)

    Returns:
        기본 추천 이유 문자열
    """
    title = movie.get("title", "이 영화")
    genres = movie.get("genres", [])
    rating = movie.get("rating", 0.0)
    director = movie.get("director", "")

    # 장르 텍스트
    genre_text = ", ".join(genres[:3]) if genres else "다양한 장르"

    # 기본 템플릿
    parts = [f"{genre_text} 장르의 인기 영화예요."]

    if rating > 0:
        parts.append(f"평점 {rating:.1f}점으로 많은 분들이 좋아하는 작품이에요.")

    if director:
        parts.append(f"{director} 감독의 연출이 돋보여요.")

    return " ".join(parts)


def _movie_to_dict(movie: RankedMovie | CandidateMovie | dict) -> dict:
    """
    다양한 영화 타입을 dict로 변환한다.

    Args:
        movie: RankedMovie, CandidateMovie, 또는 dict

    Returns:
        영화 정보 dict
    """
    if isinstance(movie, dict):
        return movie
    return movie.model_dump()


async def generate_explanation(
    movie: RankedMovie | CandidateMovie | dict,
    emotion: str | None = None,
    preferences: ExtractedPreferences | None = None,
    watch_history_titles: list[str] | None = None,
    score_detail: ScoreDetail | None = None,
) -> str:
    """
    추천된 영화에 대해 사용자 맞춤 추천 이유를 생성한다.

    Args:
        movie: 추천 영화 (RankedMovie, CandidateMovie, 또는 dict)
        emotion: 사용자 감정 (None이면 미감지)
        preferences: 사용자 선호 조건 (None이면 없음)
        watch_history_titles: 시청 이력 영화 제목 목록 (None이면 없음)
        score_detail: 추천 점수 상세 (None이면 없음)

    Returns:
        2~3문장의 한국어 추천 이유 문자열
        - 에러 시: 메타데이터 기반 기본 설명
    """
    # 영화 정보 dict 변환
    movie_dict = _movie_to_dict(movie)

    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXPLANATION_SYSTEM_PROMPT),
        ("human", EXPLANATION_HUMAN_PROMPT),
    ])

    # 자유 텍스트 LLM (EXAONE 32B, temp=0.5)
    llm = get_explanation_llm()

    # 선호 조건 텍스트 포맷
    pref_text = "(없음)"
    if preferences:
        pref_parts = []
        if preferences.genre_preference:
            pref_parts.append(f"장르: {preferences.genre_preference}")
        if preferences.mood:
            pref_parts.append(f"분위기: {preferences.mood}")
        if preferences.reference_movies:
            pref_parts.append(f"참조 영화: {', '.join(preferences.reference_movies)}")
        pref_text = ", ".join(pref_parts) if pref_parts else "(없음)"

    # 시청 이력 텍스트 포맷
    history_text = "(없음)"
    if watch_history_titles:
        history_text = ", ".join(watch_history_titles[:5])

    # 점수 상세 텍스트 포맷
    score_text = "(없음)"
    if score_detail:
        score_text = (
            f"CF={score_detail.cf_score:.2f}, "
            f"CBF={score_detail.cbf_score:.2f}, "
            f"장르 일치={score_detail.genre_match:.0%}, "
            f"무드 일치={score_detail.mood_match:.0%}"
        )

    # 입력 변수
    inputs = {
        "title": movie_dict.get("title", ""),
        "genres": ", ".join(movie_dict.get("genres", [])),
        "director": movie_dict.get("director", ""),
        "rating": str(movie_dict.get("rating", 0.0)),
        "overview": (movie_dict.get("overview", "") or "")[:300],
        "emotion": emotion or "미감지",
        "preferences": pref_text,
        "watch_history": history_text,
        "score_detail": score_text,
    }

    logger.info(
        "explanation_chain_start",
        title=movie_dict.get("title", ""),
        emotion=emotion,
        preferences_preview=pref_text[:100],
        score_detail_preview=score_text[:100],
    )

    try:
        # 프롬프트 포맷 → LLM 호출 (명시적 2단계)
        llm_start = time.perf_counter()
        prompt_value = await prompt.ainvoke(inputs)
        logger.debug(
            "explanation_chain_prompt_formatted",
            title=movie_dict.get("title", ""),
            prompt_preview=str(prompt_value)[:300],
        )
        logger.debug(
            "explanation_chain_prompt_full",
            title=movie_dict.get("title", ""),
            prompt_text=str(prompt_value),
            model=settings.EXPLANATION_MODEL,
        )
        # 모델별 세마포어로 동시 호출 제한 (Ollama 큐 점유 방지)
        response = await guarded_ainvoke(
            llm, prompt_value, model=settings.EXPLANATION_MODEL,
        )
        elapsed_ms = (time.perf_counter() - llm_start) * 1000

        # LangChain BaseMessage → 문자열 추출
        explanation = response.content if hasattr(response, "content") else str(response)
        explanation = explanation.strip() if isinstance(explanation, str) else str(explanation).strip()

        logger.debug(
            "explanation_chain_llm_raw_response",
            title=movie_dict.get("title", ""),
            raw_response=str(response),
            model=settings.EXPLANATION_MODEL,
        )
        logger.info(
            "explanation_generated",
            title=movie_dict.get("title", ""),
            explanation_preview=explanation[:50],
            elapsed_ms=round(elapsed_ms, 1),
            model=settings.EXPLANATION_MODEL,
        )
        return explanation

    except Exception as e:
        logger.error(
            "explanation_generation_error",
            error=str(e),
            title=movie_dict.get("title", ""),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
        )
        return _build_fallback_explanation(movie_dict)


async def generate_explanations_batch(
    movies: list[RankedMovie | CandidateMovie | dict],
    emotion: str | None = None,
    preferences: ExtractedPreferences | None = None,
    watch_history_titles: list[str] | None = None,
) -> list[str]:
    """
    여러 영화에 대해 추천 이유를 순차 생성한다.

    Ollama는 GPU 추론을 모델당 직렬 처리하므로, asyncio.gather 병렬 호출은
    Ollama 큐만 점유하고 실질적 병렬성은 없다. 따라서 순차 실행으로 변경하여
    다른 요청이 Ollama에 접근할 수 있도록 공정하게 배분한다.

    MAX_EXPLANATION_MOVIES(기본 3)편까지만 LLM으로 생성하고,
    초과 영화는 _build_fallback_explanation() 템플릿을 사용한다.

    Args:
        movies: 추천 영화 목록 (3~5편)
        emotion: 사용자 감정
        preferences: 사용자 선호 조건
        watch_history_titles: 시청 이력 제목 목록

    Returns:
        영화 순서대로 추천 이유 문자열 목록
    """
    # 배치 전체 소요시간 측정 시작
    batch_start = time.perf_counter()
    max_llm = settings.MAX_EXPLANATION_MOVIES

    explanations: list[str] = []

    for i, movie in enumerate(movies):
        movie_dict = _movie_to_dict(movie)

        # MAX_EXPLANATION_MOVIES 초과 → 템플릿 fallback (LLM 호출 생략)
        if i >= max_llm:
            logger.info(
                "explanation_fallback_over_limit",
                title=movie_dict.get("title", ""),
                index=i,
                max_llm=max_llm,
            )
            explanations.append(_build_fallback_explanation(movie_dict))
            continue

        # score_detail 추출 (RankedMovie만 해당)
        score_detail = None
        if isinstance(movie, RankedMovie):
            score_detail = movie.score_detail
        elif isinstance(movie, dict) and "score_detail" in movie:
            score_detail = movie["score_detail"]

        # 순차 LLM 호출 (세마포어는 generate_explanation 내부에서 적용)
        try:
            explanation = await generate_explanation(
                movie=movie,
                emotion=emotion,
                preferences=preferences,
                watch_history_titles=watch_history_titles,
                score_detail=score_detail,
            )
            explanations.append(explanation)
        except Exception as e:
            logger.error(
                "batch_explanation_error",
                title=movie_dict.get("title", ""),
                error=str(e),
                error_type=type(e).__name__,
            )
            explanations.append(_build_fallback_explanation(movie_dict))

    batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000

    logger.info(
        "explanations_batch_completed",
        movie_count=len(movies),
        llm_count=min(len(movies), max_llm),
        fallback_count=max(0, len(movies) - max_llm),
        elapsed_ms=round(batch_elapsed_ms, 1),
        model=settings.EXPLANATION_MODEL,
    )

    return explanations
