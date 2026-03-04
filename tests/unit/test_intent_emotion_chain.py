"""
의도+감정 통합 체인 단위 테스트.

classify_intent_and_emotion() 함수를 테스트한다.
기존 intent_chain + emotion_chain의 로직이 1회 LLM 호출로 통합되었는지 검증한다.

테스트 시나리오:
1. 추천 의도 + 감정 동시 분류
2. 일반 대화 의도 + 감정 없음
3. 신뢰도 < 0.6 → general 보정
4. 감정→무드 매핑 보완 (EMOTION_TO_MOOD_MAP 합집합)
5. MOOD_WHITELIST 필터링
6. LLM 에러 → fallback (1회 재시도)
7. 빈 입력 처리
8. 대화 이력 포함 분류
9. 모든 의도 타입 분류 (6가지)
10. 무드 태그 중복 제거
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import IntentEmotionResult


class TestClassifyIntentAndEmotion:
    """classify_intent_and_emotion 통합 체인 테스트."""

    @pytest.mark.asyncio
    async def test_recommend_with_sad_emotion(self, mock_ollama):
        """추천 의도 + 슬픈 감정 동시 분류."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="recommend",
                confidence=0.95,
                emotion="sad",
                mood_tags=["힐링", "감동"],
            )
        )
        from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion

        result = await classify_intent_and_emotion(
            current_input="우울한데 영화 추천해줘",
        )
        assert result.intent == "recommend"
        assert result.confidence == 0.95
        assert result.emotion == "sad"
        # EMOTION_TO_MOOD_MAP에서 sad→[힐링, 감동, 따뜻, 잔잔, 카타르시스] 합집합
        assert "힐링" in result.mood_tags
        assert "감동" in result.mood_tags

    @pytest.mark.asyncio
    async def test_general_intent_no_emotion(self, mock_ollama):
        """일반 대화: 의도=general, 감정=None."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="general",
                confidence=0.85,
                emotion=None,
                mood_tags=[],
            )
        )
        from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion

        result = await classify_intent_and_emotion(
            current_input="안녕하세요",
        )
        assert result.intent == "general"
        assert result.emotion is None
        assert result.mood_tags == []

    @pytest.mark.asyncio
    async def test_low_confidence_correction(self, mock_ollama):
        """신뢰도 < 0.6 → general로 보정."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="recommend",
                confidence=0.4,  # 임계값(0.6) 미만
                emotion="happy",
                mood_tags=["유쾌"],
            )
        )
        from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion

        result = await classify_intent_and_emotion(
            current_input="음...",
        )
        # 신뢰도 낮으면 general로 보정
        assert result.intent == "general"
        assert result.confidence == 0.4
        # 감정은 그대로 유지
        assert result.emotion == "happy"

    @pytest.mark.asyncio
    async def test_emotion_to_mood_mapping(self, mock_ollama):
        """감정→무드 매핑 보완 (EMOTION_TO_MOOD_MAP 합집합)."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="recommend",
                confidence=0.9,
                emotion="excited",  # excited→[스릴, 몰입, 웅장, 모험, 판타지]
                mood_tags=["스릴"],  # LLM이 1개만 추출
            )
        )
        from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion

        result = await classify_intent_and_emotion(
            current_input="신나는 영화 추천해줘",
        )
        assert result.emotion == "excited"
        # LLM 추출 태그 + 매핑 태그 합집합
        assert "스릴" in result.mood_tags
        assert "몰입" in result.mood_tags
        assert "웅장" in result.mood_tags

    @pytest.mark.asyncio
    async def test_mood_whitelist_filtering(self, mock_ollama):
        """MOOD_WHITELIST에 없는 태그 제거."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="recommend",
                confidence=0.9,
                emotion=None,
                mood_tags=["힐링", "없는무드태그", "감동", "가짜태그"],
            )
        )
        from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion

        result = await classify_intent_and_emotion(
            current_input="힐링 영화",
        )
        # 화이트리스트에 있는 태그만 남음
        assert "힐링" in result.mood_tags
        assert "감동" in result.mood_tags
        assert "없는무드태그" not in result.mood_tags
        assert "가짜태그" not in result.mood_tags

    @pytest.mark.asyncio
    async def test_error_fallback(self, mock_ollama):
        """LLM 에러 시 → general fallback (1회 재시도 후)."""
        mock_ollama.set_error(RuntimeError("LLM 연결 실패"))
        from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion

        result = await classify_intent_and_emotion(
            current_input="테스트",
        )
        assert result.intent == "general"
        assert result.confidence == 0.0
        assert result.emotion is None
        assert result.mood_tags == []

    @pytest.mark.asyncio
    async def test_with_recent_messages(self, mock_ollama):
        """대화 이력 포함 분류."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="recommend",
                confidence=0.9,
                emotion="calm",
                mood_tags=["잔잔"],
            )
        )
        from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion

        result = await classify_intent_and_emotion(
            current_input="잔잔한 거 없어?",
            recent_messages="user: 영화 추천해줘\nassistant: 어떤 장르를 좋아하세요?",
        )
        assert result.intent == "recommend"
        assert result.emotion == "calm"
        # calm→[잔잔, 힐링, 철학적, 레트로, 따뜻] 매핑
        assert "잔잔" in result.mood_tags
        assert "힐링" in result.mood_tags

    @pytest.mark.asyncio
    async def test_all_intent_types(self, mock_ollama):
        """6가지 의도 타입 모두 분류 가능."""
        from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion

        for intent_type in ["recommend", "search", "general", "info", "theater", "booking"]:
            mock_ollama.set_structured_response(
                IntentEmotionResult(
                    intent=intent_type,
                    confidence=0.9,
                    emotion=None,
                    mood_tags=[],
                )
            )
            result = await classify_intent_and_emotion(current_input="테스트")
            assert result.intent == intent_type

    @pytest.mark.asyncio
    async def test_mood_tags_dedup(self, mock_ollama):
        """무드 태그 중복 제거 (LLM 추출 + 매핑 합집합 시)."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="recommend",
                confidence=0.9,
                emotion="happy",  # happy→[유쾌, 모험, 따뜻, 로맨틱, 카타르시스]
                mood_tags=["유쾌", "따뜻", "모험"],  # LLM이 매핑 태그와 겹치는 것을 추출
            )
        )
        from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion

        result = await classify_intent_and_emotion(current_input="기분 좋은 영화")
        # 중복 제거: 유쾌, 따뜻, 모험이 1번씩만
        tag_counts = {tag: result.mood_tags.count(tag) for tag in result.mood_tags}
        for count in tag_counts.values():
            assert count == 1, f"무드 태그 중복: {tag_counts}"

    @pytest.mark.asyncio
    async def test_empty_input(self, mock_ollama):
        """빈 입력 → 정상 처리 (에러 없음)."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="general",
                confidence=0.3,
                emotion=None,
                mood_tags=[],
            )
        )
        from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion

        result = await classify_intent_and_emotion(current_input="")
        # 빈 입력이어도 에러 없이 처리 (신뢰도 낮으면 general 보정)
        assert result.intent == "general"
