"""
후속 질문 생성 체인 (§6-2 Node 5).

사용자 선호 조건이 부족하거나 검색 품질이 애매할 때 자연스러운 후속 질문 +
Claude Code 스타일 제안 카드(2~4개)를 한 번의 구조화 출력 LLM 호출로 생성한다.

2026-04-15 리팩터링:
- 기존: generate_question() 이 질문 텍스트(str) 만 반환 → question_generator 노드가
        FIELD_HINTS 상수로 옵션을 채웠음 (정적 옵션).
- 변경: generate_clarification() 이 ClarificationLLMOutput(question + suggestions) 을
        반환 → 노드는 그대로 UI 로 내려보냄 (AI 가 생성한 옵션).
- 하위 호환: generate_question() 은 유지하되 내부적으로 generate_clarification() 을
             호출해 question 문자열만 꺼낸다 (기존 호출자 영향 없음).

처리 흐름:
1. 부족 필드를 가중치 내림차순으로 정렬
2. 파악된 선호 + 부족 필드 + 감정 + 최근 후보(있으면) 를 프롬프트에 포함
3. Solar API (hybrid/api_only) 또는 Ollama(로컬) 로 구조화 출력 호출
4. 실패 시: DEFAULT_QUESTIONS[최고가중치_필드] + FIELD_HINTS 기반 fallback 제안
"""

from __future__ import annotations

import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from monglepick.agents.chat.models import (
    FIELD_HINTS,
    PREFERENCE_WEIGHTS,
    ExtractedPreferences,
    SuggestedOption,
)
from monglepick.config import settings
from monglepick.llm.factory import (
    get_question_llm,
    get_structured_llm,
    guarded_ainvoke,
)
from monglepick.prompts.question import QUESTION_HUMAN_PROMPT, QUESTION_SYSTEM_PROMPT

logger = structlog.get_logger()

# ============================================================
# 구조화 출력 스키마 (LLM → Pydantic 자동 검증)
# ============================================================


class ClarificationLLMOutput(BaseModel):
    """
    후속 질문 + 제안 카드 구조화 출력 스키마.

    Solar API `method="json_schema"` 로 강제 검증된다.
    노드(question_generator)에서 ClarificationResponse 로 변환되어
    SSE clarification 이벤트로 발행된다.
    """

    question: str = Field(
        ...,
        description=(
            "사용자 의도를 좁히기 위한 한 문장짜리 자연스러운 한국어 후속 질문. "
            "친근한 존댓말(~요/~에요), 이모지는 최대 1개."
        ),
    )
    suggestions: list[SuggestedOption] = Field(
        default_factory=list,
        description=(
            "사용자가 한 번의 클릭으로 답할 수 있는 자연어 제안 2~4개. "
            "각 옵션의 value 는 채팅 입력에 바로 삽입 가능한 한 문장이어야 한다. "
            "이미 파악된 선호와 겹치지 않는 구체적 선택지를 제시한다."
        ),
    )


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


# ============================================================
# 확장 프롬프트 (structured output 전용)
# ============================================================

# 기존 QUESTION_SYSTEM_PROMPT 에 "제안 카드 규칙"을 덧붙인다.
STRUCTURED_SYSTEM_SUFFIX = """

## 제안 카드(suggestions) 생성 규칙 (매우 중요)
- 사용자가 한 번의 클릭으로 답할 수 있는 짧고 구체적인 선택지 2~4개를 만들어요.
- 각 카드의 value 는 채팅창에 그대로 보낼 수 있는 자연어 한 문장으로 작성해요.
  예) "잔잔하고 따뜻한 힐링 영화가 좋아요", "오늘은 스릴 있는 범죄 스릴러 보고 싶어요"
- text 는 카드에 표시될 짧은 라벨(8~16자 내외). value 는 더 구체적이어야 해요.
- 이미 파악된 선호와 중복되지 않는, 서로 다른 방향의 옵션을 제시해요.
- reason 에는 "이 선택을 하면 어떤 영화가 추천될지" 한 줄로 힌트를 적어요 (생략 가능).
- tags 에는 이 제안이 주로 보완하는 필드명(genre_preference / mood / viewing_context 등)을 넣어요.
- 제안은 질문과 일관돼야 해요. 질문이 "어떤 분위기가 끌리세요?" 면 suggestions 도 분위기 옵션이어야 해요.
"""

STRUCTURED_HUMAN_SUFFIX = """

## 추가 컨텍스트
- 검색 피드백: {retrieval_feedback}
- 최근 후보 영화 요약: {candidate_hint}

위 정보를 바탕으로 question(자연스러운 후속 질문 1개) + suggestions(2~4개 제안 카드)
를 JSON 으로 한 번에 생성하세요.
"""


# ============================================================
# 유틸: 부족 필드 / 파악된 선호 포맷
# ============================================================


def _get_missing_fields(prefs: ExtractedPreferences) -> list[tuple[str, float]]:
    """
    채워지지 않은 선호 필드를 가중치 내림차순으로 반환한다.

    Args:
        prefs: 현재까지 파악된 사용자 선호 조건

    Returns:
        [(필드명, 가중치), ...] 리스트 (가중치 내림차순)
    """
    missing = []

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

    missing.sort(key=lambda x: x[1], reverse=True)
    return missing


def _format_known_preferences(prefs: ExtractedPreferences) -> str:
    """파악된 선호 조건을 프롬프트용 문자열로 포맷한다."""
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
    if prefs.user_intent:
        parts.append(f"- 사용자 의도: {prefs.user_intent}")
    if prefs.dynamic_filters:
        parts.append(f"- 동적 필터: {prefs.dynamic_filters}")

    return "\n".join(parts) if parts else "(아직 없음)"


def _build_fallback_suggestions(
    missing_fields: list[tuple[str, float]],
    max_count: int = 4,
) -> list[SuggestedOption]:
    """
    LLM 실패 시 FIELD_HINTS 정적 옵션으로 제안 카드를 구성한다.

    부족 필드 상위 2개에서 각 2개씩 옵션을 뽑아 value 를 문장으로 조합한다.
    (예: genre_preference + "액션" → value="액션 영화 추천해주세요")
    """
    suggestions: list[SuggestedOption] = []
    value_templates: dict[str, str] = {
        "genre_preference": "{opt} 장르 영화 추천해주세요",
        "mood": "{opt}한 분위기 영화 보고 싶어요",
        "viewing_context": "{opt} 볼 영화 추천해주세요",
        "platform": "{opt}에서 볼 수 있는 영화로 추천해주세요",
        "era": "{opt} 영화가 좋아요",
        "exclude": "{opt}",
    }

    for field_name, _weight in missing_fields[:2]:
        hint_info = FIELD_HINTS.get(field_name)
        if not hint_info or not hint_info["options"]:
            continue
        template = value_templates.get(field_name, "{opt}")
        for option_text in hint_info["options"][:2]:
            suggestions.append(
                SuggestedOption(
                    text=str(option_text),
                    value=template.format(opt=option_text),
                    reason=f"{hint_info['label']} 선택",
                    tags=[field_name],
                )
            )
            if len(suggestions) >= max_count:
                return suggestions
    return suggestions


# ============================================================
# 메인 체인: 구조화 출력 (question + suggestions 동시 생성)
# ============================================================


async def generate_clarification(
    extracted_preferences: ExtractedPreferences,
    emotion: str | None = None,
    turn_count: int = 0,
    retrieval_feedback: str = "",
    candidate_hint: str = "",
) -> ClarificationLLMOutput:
    """
    후속 질문 + 제안 카드(2~4개)를 한 번의 LLM 호출로 생성한다.

    Args:
        extracted_preferences: 현재까지 파악된 선호 조건
        emotion: 감지된 감정 (None 이면 감지 안 됨)
        turn_count: 현재 대화 턴 수
        retrieval_feedback: 검색 품질 미달 메시지 (있으면 질문에 반영)
        candidate_hint: 최근 검색 후보 요약 (선택)

    Returns:
        ClarificationLLMOutput (question + suggestions)
        - 에러 시: DEFAULT_QUESTIONS + FIELD_HINTS 기반 fallback 구조
    """
    missing = _get_missing_fields(extracted_preferences)

    # 모든 필드가 채워진 상태라면 기본 질문만 반환 (제안 없음)
    if not missing and not retrieval_feedback:
        return ClarificationLLMOutput(
            question=DEFAULT_FALLBACK_QUESTION,
            suggestions=[],
        )

    # 프롬프트 구성 — 기존 템플릿에 suffix 를 덧붙인다.
    system_prompt = QUESTION_SYSTEM_PROMPT + STRUCTURED_SYSTEM_SUFFIX
    human_prompt = QUESTION_HUMAN_PROMPT + STRUCTURED_HUMAN_SUFFIX
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    missing_str = (
        "\n".join(f"- {field} (가중치 {weight})" for field, weight in missing)
        or "(없음 — 이미 모든 선호가 파악됨)"
    )

    inputs = {
        "known_preferences": _format_known_preferences(extracted_preferences),
        "missing_fields": missing_str,
        "emotion": emotion or "감지 안 됨",
        "turn_count": str(turn_count),
        "retrieval_feedback": retrieval_feedback or "(없음)",
        "candidate_hint": candidate_hint or "(없음)",
    }

    logger.info(
        "clarification_chain_start",
        missing_fields=[f[0] for f in missing],
        turn_count=turn_count,
        emotion=emotion,
        has_retrieval_feedback=bool(retrieval_feedback),
    )

    try:
        # 구조화 출력 LLM — Solar API 우선 (hybrid/api_only), 로컬은 Ollama JSON
        llm = get_structured_llm(
            schema=ClarificationLLMOutput,
            temperature=0.5,
        )

        llm_start = time.perf_counter()
        prompt_value = await prompt.ainvoke(inputs)

        # with_structured_output 은 BaseChatModel.invoke/ainvoke 를 이미 감싸므로
        # ainvoke 로 직접 호출한다 (guarded_ainvoke 는 BaseChatModel 전용이라 미적용).
        result: ClarificationLLMOutput = await llm.ainvoke(prompt_value)
        elapsed_ms = (time.perf_counter() - llm_start) * 1000

        # 검증: suggestions 가 없거나 4개 초과면 후처리
        suggestions = list(result.suggestions or [])[:4]
        # LLM 결과가 str 이 아닌 경우(테스트 mock 등) 방어 — Pydantic validation 전에 걸러낸다.
        raw_question = result.question if isinstance(result.question, str) else ""
        question = raw_question.strip() or DEFAULT_FALLBACK_QUESTION

        logger.info(
            "clarification_generated",
            question_preview=question[:50],
            suggestion_count=len(suggestions),
            elapsed_ms=round(elapsed_ms, 1),
            model=settings.SOLAR_API_MODEL if settings.LLM_MODE != "local_only" else settings.INTENT_MODEL,
        )
        return ClarificationLLMOutput(question=question, suggestions=suggestions)

    except Exception as e:
        logger.error(
            "clarification_generation_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
        )
        # ── Fallback 1: 자유 텍스트 질문만 기존 EXAONE 32B 경로로 한 번 더 시도 ──
        fallback_question = DEFAULT_QUESTIONS.get(
            missing[0][0] if missing else "",
            DEFAULT_FALLBACK_QUESTION,
        )
        try:
            text_llm = get_question_llm()
            text_prompt = ChatPromptTemplate.from_messages([
                ("system", QUESTION_SYSTEM_PROMPT),
                ("human", QUESTION_HUMAN_PROMPT),
            ])
            text_inputs = {
                "known_preferences": inputs["known_preferences"],
                "missing_fields": missing_str,
                "emotion": inputs["emotion"],
                "turn_count": inputs["turn_count"],
            }
            text_prompt_value = await text_prompt.ainvoke(text_inputs)
            response = await guarded_ainvoke(
                text_llm, text_prompt_value, model=settings.QUESTION_MODEL,
            )
            raw = response.content if hasattr(response, "content") else str(response)
            if isinstance(raw, str) and raw.strip():
                fallback_question = raw.strip()
        except Exception as fallback_err:
            logger.warning(
                "clarification_text_fallback_failed",
                error=str(fallback_err),
                error_type=type(fallback_err).__name__,
            )

        # ── Fallback 2: FIELD_HINTS 정적 옵션으로 suggestions 구성 ──
        fallback_suggestions = _build_fallback_suggestions(missing)
        return ClarificationLLMOutput(
            question=fallback_question,
            suggestions=fallback_suggestions,
        )


# ============================================================
# 하위 호환: generate_question() 은 question 문자열만 반환
# ============================================================


async def generate_question(
    extracted_preferences: ExtractedPreferences,
    emotion: str | None = None,
    turn_count: int = 0,
) -> str:
    """
    (하위 호환) 후속 질문 텍스트만 반환한다.

    내부적으로 generate_clarification() 을 호출해 question 부분만 꺼낸다.
    기존 호출자(그래프 외부 유닛 테스트 등)는 영향 없이 동작한다.
    """
    result = await generate_clarification(
        extracted_preferences=extracted_preferences,
        emotion=emotion,
        turn_count=turn_count,
    )
    return result.question or DEFAULT_FALLBACK_QUESTION
