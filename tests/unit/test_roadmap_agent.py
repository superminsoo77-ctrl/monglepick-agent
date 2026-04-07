"""
개인화 로드맵 에이전트 단위 테스트 — Phase 7.

테스트 항목:
1. user_segment_analyzer: 시청 이력 기반 수준 판정 (beginner/intermediate/expert)
2. State 모델 검증: RoadmapAgentState TypedDict + Pydantic 모델
3. 그래프 구조 검증: 4개 노드 등록 확인
"""

from __future__ import annotations

import pytest

from monglepick.agents.roadmap.state import (
    RoadmapAgentState,
    FormattedRoadmap,
    RoadmapMovie,
    RoadmapStage,
    QuizQuestion,
    Quiz,
)
from monglepick.agents.roadmap.nodes import user_segment_analyzer
from monglepick.agents.roadmap.graph import roadmap_graph


# ============================================================
# user_segment_analyzer 테스트
# ============================================================

class TestUserSegmentAnalyzer:
    """user_segment_analyzer 노드 단위 테스트."""

    @pytest.mark.asyncio
    async def test_empty_history_is_beginner(self):
        """시청 이력 없음 → beginner 판정."""
        state: RoadmapAgentState = {
            "user_id": "user_1",
            "user_profile": {},
            "watch_history": [],
            "theme": "SF 명작",
        }
        result = await user_segment_analyzer(state)

        assert result["user_level"] == "beginner"
        assert "level_detail" in result
        assert result["level_detail"]["total_watched"] == 0

    @pytest.mark.asyncio
    async def test_few_movies_is_beginner(self):
        """시청 10편 미만 → beginner."""
        history = [
            {"movie_id": str(i), "genres": ["드라마"], "rating": 4.0, "watched_at": "2026-01-01"}
            for i in range(5)
        ]
        state: RoadmapAgentState = {
            "user_id": "user_1",
            "user_profile": {},
            "watch_history": history,
            "theme": "드라마",
        }
        result = await user_segment_analyzer(state)
        assert result["user_level"] == "beginner"

    @pytest.mark.asyncio
    async def test_moderate_history_is_intermediate(self):
        """시청 30편 + 장르 5개 → intermediate."""
        history = []
        genres_pool = ["드라마", "액션", "SF", "공포", "코미디"]
        for i in range(30):
            history.append({
                "movie_id": str(i),
                "genres": [genres_pool[i % len(genres_pool)]],
                "rating": 3.5,
                "watched_at": "2026-01-01",
            })
        state: RoadmapAgentState = {
            "user_id": "user_1",
            "user_profile": {},
            "watch_history": history,
            "theme": "액션",
        }
        result = await user_segment_analyzer(state)
        assert result["user_level"] == "intermediate"

    @pytest.mark.asyncio
    async def test_heavy_watcher_is_expert(self):
        """시청 110편 → expert."""
        history = [
            {"movie_id": str(i), "genres": ["드라마"], "rating": 4.0, "watched_at": "2026-01-01"}
            for i in range(110)
        ]
        state: RoadmapAgentState = {
            "user_id": "user_1",
            "user_profile": {},
            "watch_history": history,
            "theme": "드라마",
        }
        result = await user_segment_analyzer(state)
        assert result["user_level"] == "expert"

    @pytest.mark.asyncio
    async def test_level_detail_fields(self):
        """level_detail에 필수 필드가 모두 포함된다."""
        state: RoadmapAgentState = {
            "user_id": "user_1",
            "user_profile": {},
            "watch_history": [
                {"movie_id": "1", "genres": ["SF"], "rating": 4.5, "watched_at": "2026-01-01"}
            ],
            "theme": "SF",
        }
        result = await user_segment_analyzer(state)
        detail = result["level_detail"]

        assert "total_watched" in detail
        assert "unique_genres" in detail
        assert "top_genre" in detail
        assert "top_genre_count" in detail
        assert "avg_rating" in detail
        assert "determination_reason" in detail

    @pytest.mark.asyncio
    async def test_no_error_on_missing_fields(self):
        """watch_history 항목에 genres/rating 누락 시에도 에러 없이 처리된다."""
        history = [
            {"movie_id": "1", "watched_at": "2026-01-01"},  # genres/rating 없음
            {"movie_id": "2", "watched_at": "2026-01-02", "genres": []},
        ]
        state: RoadmapAgentState = {
            "user_id": "user_1",
            "user_profile": {},
            "watch_history": history,
            "theme": "테스트",
        }
        result = await user_segment_analyzer(state)
        assert "user_level" in result
        assert result["user_level"] in ("beginner", "intermediate", "expert")


# ============================================================
# State 모델 테스트
# ============================================================

class TestRoadmapModels:
    """Pydantic State 모델 검증."""

    def test_quiz_question_multiple_choice(self):
        """객관식 QuizQuestion 생성."""
        q = QuizQuestion(
            type="multiple_choice",
            question="이 영화의 감독은?",
            options=["봉준호", "박찬욱", "홍상수", "김지운"],
            answer="봉준호",
            hint="기생충 감독이에요.",
        )
        assert q.type == "multiple_choice"
        assert len(q.options) == 4

    def test_quiz_question_short_answer_defaults(self):
        """주관식 QuizQuestion — options 기본값 빈 리스트."""
        q = QuizQuestion(
            type="short_answer",
            question="주인공의 이름은?",
            answer="기택",
        )
        assert q.options == []
        assert q.hint == ""

    def test_roadmap_movie_defaults(self):
        """RoadmapMovie 기본값 검증."""
        m = RoadmapMovie(id="tt123", title="인셉션")
        assert m.genres == []
        assert m.poster_url == ""
        assert m.rating == 0.0
        assert m.completed is False
        assert m.quiz is None

    def test_formatted_roadmap_structure(self):
        """FormattedRoadmap 3단계 구조 생성."""
        stages = [
            RoadmapStage(
                name="입문",
                description="입문 설명",
                movies=[
                    RoadmapMovie(id=str(i), title=f"영화{i}")
                    for i in range(5)
                ],
            )
            for _ in range(3)  # 3단계 동일 구조로 간단 생성
        ]
        roadmap = FormattedRoadmap(
            roadmap_id="test-uuid",
            theme="SF 명작",
            user_level="beginner",
            created_at="2026-04-07T12:00:00",
            stages=stages,
        )
        assert len(roadmap.stages) == 3
        assert roadmap.total_progress == 0


# ============================================================
# 그래프 구조 테스트
# ============================================================

class TestRoadmapGraph:
    """로드맵 에이전트 LangGraph 구조 검증."""

    def test_graph_is_compiled(self):
        """roadmap_graph가 컴파일된 CompiledGraph 인스턴스다."""
        assert roadmap_graph is not None

    def test_graph_has_required_nodes(self):
        """4개 필수 노드가 그래프에 등록되어 있다."""
        # CompiledGraph의 nodes 속성 또는 graph 속성으로 노드 목록 확인
        graph_nodes = set(roadmap_graph.get_graph().nodes.keys())
        required = {"user_segment_analyzer", "roadmap_generator", "quiz_generator", "roadmap_formatter"}
        assert required.issubset(graph_nodes), f"누락된 노드: {required - graph_nodes}"
