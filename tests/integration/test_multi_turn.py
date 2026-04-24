"""
멀티턴 대화 통합 테스트.

세션 영속화를 통해 여러 턴에 걸친 대화 맥락이 유지되는지 검증한다.
mock_session_store fixture를 사용하여 Redis 없이 테스트한다.

테스트 시나리오:
1. 턴 1 → 후속 질문 → 세션 저장 확인
2. 턴 2 → 세션에서 이전 감정/선호 복원 확인
3. 턴 3 → turn_count ≥ 3 오버라이드로 추천 진행
4. 새 세션 → messages=[]로 시작 확인
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from monglepick.agents.chat.models import (
    EmotionResult,
    ExtractedPreferences,
    IntentEmotionResult,
)
from monglepick.rag.hybrid_search import SearchResult


def _make_multi_turn_mocks(
    intent_emotion: IntentEmotionResult | None = None,
    preferences: ExtractedPreferences | None = None,
    search_results: list[SearchResult] | None = None,
    general_response: str = "안녕하세요!",
    question_response: str = "어떤 장르를 좋아하세요?",
    explanation_response: str = "좋은 영화입니다.",
):
    """
    멀티턴 테스트용 mock 패치 목록.
    세션 저장소 mock은 포함하지 않는다 (conftest의 mock_session_store fixture 사용).
    """
    if intent_emotion is None:
        intent_emotion = IntentEmotionResult(
            intent="recommend", confidence=0.9,
            emotion="happy", mood_tags=["유쾌"],
        )
    if preferences is None:
        preferences = ExtractedPreferences(genre_preference="SF", mood="웅장한")
    if search_results is None:
        search_results = [
            SearchResult(
                movie_id="157336", title="인터스텔라", score=0.05,
                source="rrf", metadata={"genres": ["SF"], "director": "놀란", "rating": 8.7, "release_year": 2014, "overview": "우주"},
            ),
            SearchResult(
                movie_id="27205", title="인셉션", score=0.04,
                source="rrf", metadata={"genres": ["SF"], "director": "놀란", "rating": 8.4, "release_year": 2010, "overview": "꿈"},
            ),
            SearchResult(
                movie_id="49026", title="다크 나이트", score=0.03,
                source="rrf", metadata={"genres": ["액션"], "director": "놀란", "rating": 7.8, "release_year": 2012, "overview": "배트맨"},
            ),
        ]

    patches = [
        patch("monglepick.agents.chat.nodes.get_mysql", side_effect=Exception("mock")),
        patch("monglepick.agents.chat.nodes.classify_intent_and_emotion", new_callable=AsyncMock, return_value=intent_emotion),
        patch("monglepick.agents.chat.nodes.extract_preferences", new_callable=AsyncMock, return_value=preferences),
        patch("monglepick.agents.chat.nodes.generate_question", new_callable=AsyncMock, return_value=question_response),
        patch("monglepick.agents.chat.nodes.generate_explanations_batch", new_callable=AsyncMock, return_value=[explanation_response] * len(search_results)),
        patch("monglepick.agents.chat.nodes.generate_general_response", new_callable=AsyncMock, return_value=general_response),
        patch("monglepick.agents.chat.nodes.hybrid_search", new_callable=AsyncMock, return_value=search_results),
    ]
    return patches


class TestMultiTurn:
    """멀티턴 대화 통합 테스트."""

    @pytest.mark.asyncio
    async def test_turn1_saves_session(self, mock_session_store):
        """턴 1: 후속 질문 → 세션 저장 (messages, turn_count 기록).

        Note: SUFFICIENCY_THRESHOLD=2.0 이고 has_emotion 만으로 mood 가중치(2.0) 가
        부여되므로, 빈 선호 + 감정 있음 = 충분 판정(needs_clarification=False) 이 된다.
        본 테스트는 "선호 부족 → 질문" 흐름을 검증하므로 emotion 도 없앤다.
        """
        patches = _make_multi_turn_mocks(
            intent_emotion=IntentEmotionResult(
                intent="recommend", confidence=0.9,
                emotion=None, mood_tags=[],
            ),
            preferences=ExtractedPreferences(),  # 빈 선호 + 감정 없음 → 불충분 → 후속 질문
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="multi-turn-test",
                message="영화 추천해줘",
            )

        # 후속 질문이 생성됨
        assert state.get("needs_clarification") is True
        assert state.get("response")
        # 세션 저장 확인
        saved = mock_session_store.get_saved("multi-turn-test")
        assert saved is not None
        assert saved["turn_count"] == 1
        assert len(saved["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_turn2_restores_session(self, mock_session_store):
        """턴 2: 세션에서 이전 감정/선호 복원 + 현재 메시지 추가."""
        # 세션에 턴 1 데이터를 설정
        mock_session_store.set_session({
            "messages": [
                {"role": "user", "content": "우울한데 영화 추천해줘"},
                {"role": "assistant", "content": "어떤 장르를 좋아하세요?"},
            ],
            "preferences": ExtractedPreferences(mood="우울한"),
            "emotion": EmotionResult(emotion="sad", mood_tags=["힐링", "감동"]),
            "turn_count": 1,
            "user_profile": {},
            "watch_history": [],
        })

        patches = _make_multi_turn_mocks(
            intent_emotion=IntentEmotionResult(
                intent="recommend", confidence=0.9,
                emotion="sad", mood_tags=["힐링", "감동"],
            ),
            preferences=ExtractedPreferences(genre_preference="SF", mood="우울한"),
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="multi-turn-test",
                message="SF 영화로 추천해줘",
            )

        # 세션이 복원되어 turn_count=2
        assert state.get("turn_count") == 2
        # 세션에 이전 메시지 + 현재 메시지 포함
        messages = state.get("messages", [])
        assert len(messages) >= 3  # 이전 2개 + 현재 user 1개

    @pytest.mark.asyncio
    async def test_turn3_override_forces_recommendation(self, mock_session_store):
        """턴 3: turn_count ≥ 3 → 선호 부족해도 추천 진행."""
        # 세션에 턴 2 데이터를 설정 (turn_count=2)
        mock_session_store.set_session({
            "messages": [
                {"role": "user", "content": "질문 1"},
                {"role": "assistant", "content": "응답 1"},
                {"role": "user", "content": "질문 2"},
                {"role": "assistant", "content": "응답 2"},
            ],
            "preferences": ExtractedPreferences(),  # 빈 선호
            "emotion": None,
            "turn_count": 2,
            "user_profile": {},
            "watch_history": [],
        })

        patches = _make_multi_turn_mocks(
            intent_emotion=IntentEmotionResult(
                intent="recommend", confidence=0.9,
                emotion=None, mood_tags=[],
            ),
            preferences=ExtractedPreferences(),  # 여전히 빈 선호
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="multi-turn-test",
                message="아니요 없어요 그냥 추천해주세요",
            )

        # turn_count=3 → TURN_COUNT_OVERRIDE → 추천 진행
        assert state.get("turn_count") == 3
        # 추천이 진행되었거나 응답이 존재해야 함
        assert state.get("response")

    @pytest.mark.asyncio
    async def test_new_session_starts_fresh(self, mock_session_store):
        """세션 없이 요청 → 새 세션 자동 생성, messages=[]로 시작."""
        # mock_session_store 기본: load_session → None (신규 세션)
        patches = _make_multi_turn_mocks(
            intent_emotion=IntentEmotionResult(
                intent="general", confidence=0.8,
                emotion=None, mood_tags=[],
            ),
            general_response="안녕하세요! 몽글이에요!",
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="",  # 빈 세션 ID → 자동 생성
                message="안녕",
            )

        # 세션 ID가 자동 생성됨
        assert state.get("session_id")
        assert len(state["session_id"]) > 0
        # turn_count는 1 (첫 턴)
        assert state.get("turn_count") == 1
        # 응답 존재
        assert "몽글" in state.get("response", "")
