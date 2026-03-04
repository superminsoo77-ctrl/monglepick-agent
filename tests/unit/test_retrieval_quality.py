"""
RAG 검색 품질 판정 로직 테스트.

route_after_retrieval 라우터와 _rag_retriever_with_quality_check 래퍼를 테스트한다.

테스트 시나리오:
1. 품질 OK (3개 이상 + Top-1 ≥ 0.015 + 평균 ≥ 0.01) → recommendation_ranker
2. 품질 미달 (후보 부족) → question_generator
3. 품질 미달 (Top-1 점수 낮음) → question_generator
4. 품질 미달 (평균 점수 낮음) → question_generator
5. 빈 결과 → question_generator
6. 품질 미달 + turn_count ≥ 3 → recommendation_ranker (무한 루프 방지)
7. 경계값 테스트 (정확히 임계값)
8. _rag_retriever_with_quality_check 래퍼 state 업데이트 검증
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from monglepick.agents.chat.models import (
    RETRIEVAL_MIN_CANDIDATES,
    RETRIEVAL_MIN_TOP_SCORE,
    RETRIEVAL_QUALITY_MIN_AVG,
    CandidateMovie,
    ChatAgentState,
)


def _make_candidates(scores: list[float]) -> list[CandidateMovie]:
    """RRF 점수 목록으로 CandidateMovie 리스트를 생성한다."""
    return [
        CandidateMovie(id=str(i), title=f"영화{i}", rrf_score=score)
        for i, score in enumerate(scores)
    ]


class TestRouteAfterRetrieval:
    """route_after_retrieval 라우터 테스트."""

    def test_quality_ok(self):
        """품질 OK → recommendation_ranker."""
        from monglepick.agents.chat.graph import route_after_retrieval

        state: ChatAgentState = {
            "candidate_movies": _make_candidates([0.05, 0.04, 0.03, 0.02, 0.015]),
            "turn_count": 1,
        }
        result = route_after_retrieval(state)
        assert result == "recommendation_ranker"

    def test_insufficient_candidates(self):
        """후보 2개 (최소 3개 미만) → question_generator."""
        from monglepick.agents.chat.graph import route_after_retrieval

        state: ChatAgentState = {
            "candidate_movies": _make_candidates([0.05, 0.04]),
            "turn_count": 1,
        }
        result = route_after_retrieval(state)
        assert result == "question_generator"

    def test_low_top_score(self):
        """Top-1 점수 낮음 (0.01 < 0.015) → question_generator."""
        from monglepick.agents.chat.graph import route_after_retrieval

        state: ChatAgentState = {
            "candidate_movies": _make_candidates([0.01, 0.008, 0.005]),
            "turn_count": 1,
        }
        result = route_after_retrieval(state)
        assert result == "question_generator"

    def test_low_average_score(self):
        """Top-1은 OK이나 평균 낮음 → question_generator."""
        from monglepick.agents.chat.graph import route_after_retrieval

        # Top-1=0.03, 나머지 매우 낮아 평균 < 0.01
        state: ChatAgentState = {
            "candidate_movies": _make_candidates([0.03, 0.002, 0.001, 0.001, 0.001]),
            "turn_count": 1,
        }
        result = route_after_retrieval(state)
        # avg = (0.03 + 0.002 + 0.001 + 0.001 + 0.001) / 5 = 0.007 < 0.01
        assert result == "question_generator"

    def test_empty_results(self):
        """빈 결과 → question_generator."""
        from monglepick.agents.chat.graph import route_after_retrieval

        state: ChatAgentState = {
            "candidate_movies": [],
            "turn_count": 1,
        }
        result = route_after_retrieval(state)
        assert result == "question_generator"

    def test_quality_fail_but_turn_count_override(self):
        """품질 미달 + turn_count ≥ 3 → recommendation_ranker (무한 루프 방지)."""
        from monglepick.agents.chat.graph import route_after_retrieval

        state: ChatAgentState = {
            "candidate_movies": _make_candidates([0.01]),  # 명백한 품질 미달
            "turn_count": 3,  # 오버라이드 임계값
        }
        result = route_after_retrieval(state)
        assert result == "recommendation_ranker"

    def test_boundary_exact_threshold(self):
        """정확히 임계값 → 품질 OK."""
        from monglepick.agents.chat.graph import route_after_retrieval

        state: ChatAgentState = {
            "candidate_movies": _make_candidates([
                RETRIEVAL_MIN_TOP_SCORE,
                RETRIEVAL_QUALITY_MIN_AVG,
                RETRIEVAL_QUALITY_MIN_AVG,
            ]),
            "turn_count": 1,
        }
        result = route_after_retrieval(state)
        # 정확히 임계값이면 >= 조건이므로 PASS
        assert result == "recommendation_ranker"

    def test_boundary_just_below_threshold(self):
        """임계값 바로 아래 → 품질 미달."""
        from monglepick.agents.chat.graph import route_after_retrieval

        state: ChatAgentState = {
            "candidate_movies": _make_candidates([
                RETRIEVAL_MIN_TOP_SCORE - 0.001,
                0.01,
                0.01,
            ]),
            "turn_count": 1,
        }
        result = route_after_retrieval(state)
        assert result == "question_generator"


class TestRetrievalConstants:
    """검색 품질 판정 상수 테스트."""

    def test_min_candidates_value(self):
        """최소 후보 수 = 3."""
        assert RETRIEVAL_MIN_CANDIDATES == 3

    def test_min_top_score_value(self):
        """Top-1 최소 점수 = 0.015 (RRF k=60 단일 엔진 최대 ~0.01639 고려)."""
        assert RETRIEVAL_MIN_TOP_SCORE == 0.015

    def test_min_avg_score_value(self):
        """평균 최소 점수 = 0.01."""
        assert RETRIEVAL_QUALITY_MIN_AVG == 0.01

    def test_top_score_greater_than_avg(self):
        """Top-1 임계값 > 평균 임계값 (논리적 일관성)."""
        assert RETRIEVAL_MIN_TOP_SCORE > RETRIEVAL_QUALITY_MIN_AVG


class TestRagRetrieverWithQualityCheck:
    """_rag_retriever_with_quality_check 래퍼 테스트."""

    @pytest.mark.asyncio
    async def test_quality_passed_state(self, mock_hybrid_search):
        """품질 OK → retrieval_quality_passed=True, retrieval_feedback=''."""
        from monglepick.agents.chat.graph import _rag_retriever_with_quality_check
        from monglepick.agents.chat.models import SearchQuery
        from monglepick.rag.hybrid_search import SearchResult

        mock_hybrid_search.set_results([
            SearchResult(
                movie_id=str(i), title=f"영화{i}",
                score=0.05 - i * 0.005, source="rrf",
                metadata={"genres": ["SF"], "rating": 8.0, "release_year": 2020},
            )
            for i in range(5)
        ])

        state: ChatAgentState = {
            "search_query": SearchQuery(semantic_query="SF 영화"),
            "emotion": None,
        }
        result = await _rag_retriever_with_quality_check(state)
        assert result["retrieval_quality_passed"] is True
        assert result["retrieval_feedback"] == ""

    @pytest.mark.asyncio
    async def test_quality_failed_state(self, mock_hybrid_search):
        """품질 미달 → retrieval_quality_passed=False, retrieval_feedback에 메시지."""
        from monglepick.agents.chat.graph import _rag_retriever_with_quality_check
        from monglepick.agents.chat.models import SearchQuery

        # 빈 검색 결과 → 품질 미달
        mock_hybrid_search.set_results([])

        state: ChatAgentState = {
            "search_query": SearchQuery(semantic_query="없는 영화"),
            "emotion": None,
        }
        result = await _rag_retriever_with_quality_check(state)
        assert result["retrieval_quality_passed"] is False
        assert result["retrieval_feedback"]  # 비어있지 않음
        assert "못했" in result["retrieval_feedback"] or "부족" in result["retrieval_feedback"]

    @pytest.mark.asyncio
    async def test_low_score_feedback_message(self, mock_hybrid_search):
        """Top-1 점수 낮음 → '조건과 딱 맞는 영화를 찾기 어려웠어요' 피드백."""
        from monglepick.agents.chat.graph import _rag_retriever_with_quality_check
        from monglepick.agents.chat.models import SearchQuery
        from monglepick.rag.hybrid_search import SearchResult

        mock_hybrid_search.set_results([
            SearchResult(
                movie_id=str(i), title=f"영화{i}",
                score=0.01 - i * 0.001, source="rrf",
                metadata={"genres": ["드라마"], "rating": 6.0, "release_year": 2015},
            )
            for i in range(5)
        ])

        state: ChatAgentState = {
            "search_query": SearchQuery(semantic_query="특이한 영화"),
            "emotion": None,
        }
        result = await _rag_retriever_with_quality_check(state)
        assert result["retrieval_quality_passed"] is False
        assert "어려웠어요" in result["retrieval_feedback"]
