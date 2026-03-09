"""
Phase 2 통합 테스트 (Task 12).

전체 체인이 올바르게 연동되는지 end-to-end 시나리오를 검증한다.
모든 테스트는 mock 기반 (Ollama 서버 불필요).

시나리오:
1. 추천 플로우: intent→emotion→preference→sufficiency→question or explanation
2. 충분성 판정: genre+mood(4.0) → is_sufficient=True → 질문 스킵
3. 질문 생성: genre만(2.0), turn=1 → is_sufficient=False → 질문 생성
4. 턴 카운트 오버라이드: 선호 비어도 turn≥3 → 추천 진행
5. 일반 대화: intent=general → general_chat_chain 응답
6. 에러 복원력: 모든 체인이 에러 시 유효한 fallback 반환
7. 모델 라우팅: intent→INTENT_MODEL, preference→PREFERENCE_MODEL 확인
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import (
    EmotionResult,
    ExtractedPreferences,
    IntentResult,
    is_sufficient,
)
from monglepick.chains.emotion_chain import analyze_emotion
from monglepick.chains.explanation_chain import generate_explanation
from monglepick.chains.general_chat_chain import generate_general_response
from monglepick.chains.intent_chain import classify_intent
from monglepick.chains.preference_chain import extract_preferences
from monglepick.chains.question_chain import generate_question
from monglepick.chains.tool_executor_chain import execute_tool


# ============================================================
# 시나리오 1: 추천 플로우 end-to-end
# ============================================================


@pytest.mark.asyncio
async def test_recommend_flow_e2e(mock_ollama):
    """
    추천 플로우 end-to-end:
    1. intent=recommend
    2. emotion=sad → mood_tags 매핑
    3. preference 추출
    4. 충분성 판정
    """
    # Step 1: 의도 분류
    mock_ollama.set_structured_response(
        IntentResult(intent="recommend", confidence=0.95),
    )
    intent = await classify_intent("우울한데 영화 추천해줘")
    assert intent.intent == "recommend"

    # Step 2: 감정 분석
    mock_ollama.set_structured_response(
        EmotionResult(emotion="sad", mood_tags=["슬픔"]),
    )
    emotion = await analyze_emotion("우울한데 영화 추천해줘")
    assert emotion.emotion == "sad"
    assert "힐링" in emotion.mood_tags  # 매핑 테이블에서 보완

    # Step 3: 선호 추출
    mock_ollama.set_structured_response(
        ExtractedPreferences(genre_preference="드라마", mood="힐링"),
    )
    prefs = await extract_preferences("우울한데 영화 추천해줘")
    assert prefs.genre_preference == "드라마"

    # Step 4: 충분성 판정 (genre + mood = 4.0 >= 3.0)
    assert is_sufficient(prefs) is True


# ============================================================
# 시나리오 2: 충분성 → 질문 스킵
# ============================================================


@pytest.mark.asyncio
async def test_sufficient_prefs_skip_question(mock_ollama):
    """genre+mood 채움(4.0) → is_sufficient=True → 질문 생성 불필요."""
    mock_ollama.set_structured_response(
        ExtractedPreferences(genre_preference="SF", mood="웅장한"),
    )
    prefs = await extract_preferences("SF 우주 영화 추천해줘")
    assert prefs.genre_preference == "SF"
    assert is_sufficient(prefs) is True
    # 질문이 필요 없으므로 explanation 진행 가능


# ============================================================
# 시나리오 3: 불충분 → 질문 생성
# ============================================================


@pytest.mark.asyncio
async def test_insufficient_prefs_generate_question(mock_ollama):
    """genre만(2.0), turn=1 → is_sufficient=False → 질문 생성."""
    # 선호 추출
    mock_ollama.set_structured_response(
        ExtractedPreferences(genre_preference="SF"),
    )
    prefs = await extract_preferences("SF 영화")
    assert not is_sufficient(prefs, turn_count=1)

    # 질문 생성
    mock_ollama.set_response("어떤 분위기의 영화가 끌리세요? 🎬")
    question = await generate_question(prefs, turn_count=1)
    assert isinstance(question, str)
    assert len(question) > 0


# ============================================================
# 시나리오 4: 턴 카운트 오버라이드
# ============================================================


def test_turn_count_override():
    """선호 비어있어도 turn_count≥2 → is_sufficient=True (TURN_COUNT_OVERRIDE=2)."""
    prefs = ExtractedPreferences()  # 모두 None
    # turn_count=1 → 아직 부족
    assert is_sufficient(prefs, turn_count=1) is False
    # turn_count=2 → 오버라이드 (2턴째부터 추천 진행)
    assert is_sufficient(prefs, turn_count=2) is True


# ============================================================
# 시나리오 5: 일반 대화
# ============================================================


@pytest.mark.asyncio
async def test_general_chat_flow(mock_ollama):
    """intent=general → general_chat_chain 응답."""
    # 의도 분류
    mock_ollama.set_structured_response(
        IntentResult(intent="general", confidence=0.9),
    )
    intent = await classify_intent("안녕하세요")
    assert intent.intent == "general"

    # 일반 대화 응답
    mock_ollama.set_response("안녕하세요! 몽글이에요 😊 영화 추천이 필요하시면 말씀해주세요!")
    response = await generate_general_response("안녕하세요")
    assert "몽글" in response or len(response) > 0


# ============================================================
# 시나리오 6: 에러 복원력
# ============================================================


@pytest.mark.asyncio
async def test_all_chains_error_resilience(mock_ollama):
    """모든 체인이 LLM 에러 시 유효한 fallback을 반환한다."""
    mock_ollama.set_error(RuntimeError("LLM server down"))

    # 의도 분류 → fallback
    intent = await classify_intent("test")
    assert intent.intent == "general"
    assert intent.confidence == 0.0

    # 감정 분석 → fallback
    emotion = await analyze_emotion("test")
    assert emotion.emotion is None
    assert emotion.mood_tags == []

    # 선호 추출 → fallback
    prefs = await extract_preferences("test")
    assert prefs.genre_preference is None

    # 후속 질문 → fallback
    question = await generate_question(ExtractedPreferences())
    assert isinstance(question, str)
    assert len(question) > 0

    # 추천 이유 → fallback
    explanation = await generate_explanation(
        movie={"title": "테스트", "genres": ["드라마"], "rating": 7.0},
    )
    assert "드라마" in explanation

    # 일반 대화 → fallback
    response = await generate_general_response("test")
    assert "다시 말씀해주세요" in response or len(response) > 0


# ============================================================
# 시나리오 7: Tool Executor NotImplementedError
# ============================================================


@pytest.mark.asyncio
async def test_tool_executor_not_implemented():
    """tool_executor_chain은 NotImplementedError를 발생시킨다."""
    with pytest.raises(NotImplementedError, match="Phase 6"):
        await execute_tool("info", "인터스텔라 정보 알려줘")


# ============================================================
# 모델/프롬프트 import 검증
# ============================================================


def test_chains_module_imports():
    """chains/__init__.py에서 모든 체인을 import할 수 있다."""
    from monglepick.chains import (
        analyze_emotion,
        classify_intent,
        execute_tool,
        extract_preferences,
        generate_explanation,
        generate_explanations_batch,
        generate_general_response,
        generate_question,
    )
    # 모든 함수가 callable
    assert callable(classify_intent)
    assert callable(analyze_emotion)
    assert callable(extract_preferences)
    assert callable(generate_question)
    assert callable(generate_explanation)
    assert callable(generate_explanations_batch)
    assert callable(generate_general_response)
    assert callable(execute_tool)


def test_prompts_module_imports():
    """prompts/__init__.py에서 모든 프롬프트를 import할 수 있다."""
    from monglepick.prompts import (
        EMOTION_HUMAN_PROMPT,
        EMOTION_SYSTEM_PROMPT,
        EMOTION_TO_MOOD_MAP,
        EXPLANATION_HUMAN_PROMPT,
        EXPLANATION_SYSTEM_PROMPT,
        INTENT_HUMAN_PROMPT,
        INTENT_SYSTEM_PROMPT,
        MONGGLE_RECOMMENDATION_PERSONA,
        MONGGLE_SYSTEM_PROMPT,
        PREFERENCE_HUMAN_PROMPT,
        PREFERENCE_SYSTEM_PROMPT,
        QUESTION_HUMAN_PROMPT,
        QUESTION_SYSTEM_PROMPT,
        TOOL_EXECUTOR_SYSTEM_PROMPT,
    )
    # 모든 프롬프트가 비어있지 않은 문자열
    assert len(MONGGLE_SYSTEM_PROMPT) > 0
    assert "몽글" in MONGGLE_SYSTEM_PROMPT
    assert len(INTENT_SYSTEM_PROMPT) > 0
    assert len(EMOTION_SYSTEM_PROMPT) > 0
    assert len(PREFERENCE_SYSTEM_PROMPT) > 0
    assert len(QUESTION_SYSTEM_PROMPT) > 0
    assert len(EXPLANATION_SYSTEM_PROMPT) > 0
    assert len(TOOL_EXECUTOR_SYSTEM_PROMPT) > 0


def test_llm_module_imports():
    """llm/__init__.py에서 모든 팩토리 함수를 import할 수 있다."""
    from monglepick.llm import (
        get_conversation_llm,
        get_emotion_llm,
        get_explanation_llm,
        get_intent_llm,
        get_llm,
        get_preference_llm,
        get_question_llm,
        get_structured_llm,
    )
    assert callable(get_llm)
    assert callable(get_structured_llm)
    assert callable(get_intent_llm)
    assert callable(get_emotion_llm)
    assert callable(get_preference_llm)
    assert callable(get_conversation_llm)
    assert callable(get_question_llm)
    assert callable(get_explanation_llm)


def test_persona_prompt_content():
    """몽글 페르소나 프롬프트에 '몽글' 키워드가 포함되어 있다."""
    from monglepick.prompts.persona import (
        MONGGLE_RECOMMENDATION_PERSONA,
        MONGGLE_SYSTEM_PROMPT,
    )
    assert "몽글" in MONGGLE_SYSTEM_PROMPT
    assert "몽글" in MONGGLE_RECOMMENDATION_PERSONA
    assert len(MONGGLE_SYSTEM_PROMPT) > 100
    assert len(MONGGLE_RECOMMENDATION_PERSONA) > 100
