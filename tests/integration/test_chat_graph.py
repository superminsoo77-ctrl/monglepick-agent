"""
Chat Agent 그래프 통합 테스트 (Phase 3 + 의도+감정 통합 + 구조화 힌트 + RAG 품질).

전체 그래프 흐름을 테스트한다 (mock LLM + mock DB).
run_chat_agent_sync()로 동기 실행하여 최종 State를 검증한다.

테스트 시나리오:
1. 추천 흐름: recommend 의도+감정 → 선호 → 검색 → 순위 → 설명 → 응답
2. 일반 대화 흐름: general 의도 → general_responder → 응답
3. 후속 질문 흐름: 선호 부족 → question_generator → 질문+힌트 응답
4. 빈 검색 결과 → 빈 추천 + 기본 응답
5. tool_executor 흐름: info/theater/booking → 안내 메시지
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monglepick.agents.chat.models import (
    EmotionResult,
    ExtractedPreferences,
    IntentEmotionResult,
    IntentResult,
)
from monglepick.rag.hybrid_search import SearchResult


# ============================================================
# 통합 테스트용 mock 래퍼
# ============================================================

def _make_all_mocks(
    intent_emotion: IntentEmotionResult | None = None,
    preferences: ExtractedPreferences | None = None,
    search_results: list[SearchResult] | None = None,
    general_response: str = "안녕하세요!",
    question_response: str = "어떤 장르를 좋아하세요?",
    explanation_response: str = "좋은 영화입니다.",
):
    """
    모든 외부 의존성(LLM 체인, DB, 검색, 세션 저장소)을 mock하는 패치 목록을 반환한다.

    의도+감정 통합 후: classify_intent + analyze_emotion 2개 패치가
    classify_intent_and_emotion 1개 패치로 통합되었다.

    세션 저장소: load_session → None (신규 세션), save_session → no-op

    Returns:
        list of mock.patch context managers
    """
    if intent_emotion is None:
        intent_emotion = IntentEmotionResult(
            intent="recommend", confidence=0.9,
            emotion="happy", mood_tags=["유쾌"],
        )
    if preferences is None:
        preferences = ExtractedPreferences(genre_preference="SF", mood="웅장한")
    if search_results is None:
        # 최소 3개 이상 + Top-1 RRF ≥ 0.02 + 평균 ≥ 0.015 (검색 품질 통과 조건)
        search_results = [
            SearchResult(
                movie_id="157336",
                title="인터스텔라",
                score=0.05,
                source="rrf",
                metadata={
                    "genres": ["SF", "드라마"],
                    "director": "놀란",
                    "rating": 8.7,
                    "release_year": 2014,
                    "overview": "우주 탐사 영화",
                },
            ),
            SearchResult(
                movie_id="27205",
                title="인셉션",
                score=0.04,
                source="rrf",
                metadata={
                    "genres": ["SF", "액션"],
                    "director": "놀란",
                    "rating": 8.4,
                    "release_year": 2010,
                    "overview": "꿈 속의 꿈",
                },
            ),
            SearchResult(
                movie_id="49026",
                title="다크 나이트 라이즈",
                score=0.03,
                source="rrf",
                metadata={
                    "genres": ["액션", "드라마"],
                    "director": "놀란",
                    "rating": 7.8,
                    "release_year": 2012,
                    "overview": "배트맨 시리즈",
                },
            ),
        ]

    patches = [
        # MySQL mock (익명 사용자로 처리하기 위해 get_mysql을 에러로)
        patch(
            "monglepick.agents.chat.nodes.get_mysql",
            side_effect=Exception("mock: DB 미사용"),
        ),
        # 통합 의도+감정 체인 mock (기존 classify_intent + analyze_emotion 대체)
        patch(
            "monglepick.agents.chat.nodes.classify_intent_and_emotion",
            new_callable=AsyncMock,
            return_value=intent_emotion,
        ),
        patch(
            "monglepick.agents.chat.nodes.extract_preferences",
            new_callable=AsyncMock,
            return_value=preferences,
        ),
        patch(
            "monglepick.agents.chat.nodes.generate_question",
            new_callable=AsyncMock,
            return_value=question_response,
        ),
        patch(
            "monglepick.agents.chat.nodes.generate_explanations_batch",
            new_callable=AsyncMock,
            return_value=[explanation_response] * len(search_results),
        ),
        patch(
            "monglepick.agents.chat.nodes.generate_general_response",
            new_callable=AsyncMock,
            return_value=general_response,
        ),
        # 하이브리드 검색 mock
        patch(
            "monglepick.agents.chat.nodes.hybrid_search",
            new_callable=AsyncMock,
            return_value=search_results,
        ),
        # 세션 저장소 mock (load → None, save → no-op)
        patch(
            "monglepick.agents.chat.graph.load_session",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "monglepick.agents.chat.graph.save_session",
            new_callable=AsyncMock,
        ),
    ]
    return patches


# ============================================================
# 통합 테스트
# ============================================================

class TestChatGraphIntegration:
    """Chat Agent 그래프 통합 테스트."""

    @pytest.mark.asyncio
    async def test_recommend_flow(self):
        """추천 흐름: intent_emotion → preference → search → rank → explain → format."""
        patches = _make_all_mocks(
            intent_emotion=IntentEmotionResult(
                intent="recommend", confidence=0.9,
                emotion="sad", mood_tags=["힐링", "감동"],
            ),
            preferences=ExtractedPreferences(genre_preference="SF", mood="웅장한"),
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="test",
                message="우울한데 영화 추천해줘",
            )

        # 최종 State 검증
        assert state.get("intent") is not None
        assert state["intent"].intent == "recommend"
        assert state.get("response")
        assert "인터스텔라" in state["response"]
        assert len(state.get("ranked_movies", [])) >= 1

    @pytest.mark.asyncio
    async def test_general_flow(self):
        """일반 대화 흐름: general → general_responder → format."""
        patches = _make_all_mocks(
            intent_emotion=IntentEmotionResult(
                intent="general", confidence=0.8,
                emotion=None, mood_tags=[],
            ),
            general_response="안녕하세요! 영화 추천 도우미 몽글이에요!",
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="test",
                message="안녕",
            )

        assert state["intent"].intent == "general"
        assert "몽글" in state["response"]

    @pytest.mark.asyncio
    async def test_question_flow(self):
        """후속 질문 흐름: 선호 부족 → question_generator → 질문+힌트 응답."""
        patches = _make_all_mocks(
            intent_emotion=IntentEmotionResult(
                intent="recommend", confidence=0.9,
                emotion=None, mood_tags=[],
            ),
            preferences=ExtractedPreferences(),  # 빈 선호 → 불충분
            question_response="어떤 장르의 영화를 좋아하세요?",
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="test",
                message="영화 추천해줘",
            )

        assert state.get("needs_clarification") is True
        assert "장르" in state["response"]
        # 구조화된 힌트가 포함되어야 함
        clarification = state.get("clarification")
        assert clarification is not None
        assert len(clarification.hints) > 0

    @pytest.mark.asyncio
    async def test_empty_search_results(self):
        """빈 검색 결과 → 검색 품질 미달 → 추가 질문 또는 빈 추천."""
        patches = _make_all_mocks(
            intent_emotion=IntentEmotionResult(
                intent="recommend", confidence=0.9,
                emotion=None, mood_tags=[],
            ),
            preferences=ExtractedPreferences(genre_preference="SF", mood="웅장한"),
            search_results=[],
            explanation_response="",
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="test",
                message="엄청 특이한 영화 추천해줘",
            )

        # 빈 검색 결과 → route_after_retrieval에서 question_generator로 분기
        # 또는 turn_count ≥ 3이면 빈 ranked_movies로 진행
        assert state.get("response")  # 응답은 존재

    @pytest.mark.asyncio
    async def test_tool_executor_flow(self):
        """도구 실행 흐름: info → tool_executor_node → format."""
        patches = _make_all_mocks(
            intent_emotion=IntentEmotionResult(
                intent="info", confidence=0.9,
                emotion=None, mood_tags=[],
            ),
        )

        from monglepick.agents.chat.graph import run_chat_agent_sync

        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            state = await run_chat_agent_sync(
                user_id="",
                session_id="test",
                message="인터스텔라 상세 정보 알려줘",
            )

        assert state["intent"].intent == "info"
        assert "준비" in state["response"]
