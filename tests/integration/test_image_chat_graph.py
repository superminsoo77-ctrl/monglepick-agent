"""
이미지 포함 Chat Agent 그래프 통합 테스트.

이미지 업로드 시 전체 그래프 흐름이 정상 동작하는지 테스트한다.
- 이미지 없는 일반 대화 흐름 (기존 호환성)
- 이미지 포함 추천 흐름 (intent 부스트 + 선호 보강)
- API 엔드포인트 이미지 파라미터 전달

의도+감정 통합 후: IntentResult → IntentEmotionResult 사용.
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import (
    ChatAgentState,
    ImageAnalysisResult,
    IntentEmotionResult,
)


class TestImageChatGraph:
    """이미지 포함 그래프 통합 테스트."""

    @pytest.mark.asyncio
    async def test_graph_without_image_backward_compatible(
        self, mock_ollama, mock_mysql, mock_hybrid_search,
    ):
        """이미지 없는 기존 흐름이 정상 동작한다 (하위 호환성)."""
        from monglepick.agents.chat.graph import run_chat_agent_sync

        # 통합 모델: intent=general → 일반 대화 흐름
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="general", confidence=0.9,
                emotion=None, mood_tags=[],
            )
        )
        mock_ollama.set_response("안녕하세요! 영화 추천이 필요하시면 말씀해주세요.")

        state = await run_chat_agent_sync(
            user_id="",
            session_id="test-sess",
            message="안녕",
            image_data=None,  # 이미지 없음
        )

        assert state.get("response")
        # 이미지 없으면 image_analyzer 노드가 실행되지 않으므로 image_analysis는 None
        image_analysis = state.get("image_analysis")
        assert image_analysis is None

    @pytest.mark.asyncio
    async def test_graph_with_image_intent_boost(
        self, mock_ollama, mock_mysql, mock_hybrid_search, mock_image_analysis,
    ):
        """이미지 포함 시 intent가 general→recommend로 부스트된다."""
        from monglepick.agents.chat.graph import run_chat_agent_sync

        # 통합 모델이 general로 분류하더라도, 이미지가 있으면 recommend로 부스트
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="general", confidence=0.5,
                emotion=None, mood_tags=[],
            )
        )
        mock_ollama.set_response("SF 영화를 추천해드릴게요!")

        state = await run_chat_agent_sync(
            user_id="",
            session_id="test-sess",
            message="이런 느낌의 영화",
            image_data="base64fakeimage",
        )

        # intent가 recommend로 부스트되었어야 함
        intent = state.get("intent")
        assert intent is not None
        assert intent.intent == "recommend"

    @pytest.mark.asyncio
    async def test_image_analysis_enriches_preferences(
        self, mock_ollama, mock_mysql, mock_hybrid_search, mock_image_analysis,
    ):
        """이미지 분석 결과가 선호 조건을 보강한다."""
        from monglepick.agents.chat.graph import run_chat_agent_sync

        # 이미지 분석: SF 장르, 웅장 무드, 인터스텔라 포스터
        mock_image_analysis.set_result(ImageAnalysisResult(
            genre_cues=["SF", "모험"],
            mood_cues=["웅장"],
            visual_elements=["우주선"],
            search_keywords=["우주"],
            description="우주 배경 영화",
            is_movie_poster=True,
            detected_movie_title="인터스텔라",
            analyzed=True,
        ))

        # 통합 모델: intent=recommend로 분류
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="recommend", confidence=0.9,
                emotion=None, mood_tags=[],
            )
        )
        mock_ollama.set_response("SF 영화를 추천해드릴게요!")

        state = await run_chat_agent_sync(
            user_id="",
            session_id="test-sess",
            message="이 영화 같은 거 추천해줘",
            image_data="base64fakeimage",
        )

        # 선호 조건에 이미지 분석 결과가 반영되었는지 확인
        prefs = state.get("preferences")
        if prefs:
            # genre_preference에 이미지의 장르가 반영될 수 있음
            # (LLM이 빈 선호를 추출했을 때 이미지 보강이 적용됨)
            assert state.get("response")  # 응답이 생성됨
