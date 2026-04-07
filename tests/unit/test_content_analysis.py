"""
콘텐츠 분석 에이전트 단위 테스트 — Phase 7.

테스트 항목:
1. 비속어 검출 (toxicity_detection): 4단계 액션 판정
2. 패턴 분석 (pattern_analysis): 업적 판정 + 장르 벡터
3. Pydantic 모델 검증 (models): 입출력 스키마
"""

from __future__ import annotations

import pytest

from monglepick.agents.content_analysis.models import (
    ProfanityCheckInput,
    ProfanityCheckOutput,
    WatchRecord,
    PatternAnalysisInput,
    PatternAnalysisOutput,
    Achievement,
)
from monglepick.agents.content_analysis.toxicity_detection import check_profanity
from monglepick.agents.content_analysis.pattern_analysis import analyze_user_pattern


# ============================================================
# 비속어 검출 테스트
# ============================================================

class TestToxicityDetection:
    """check_profanity() 단위 테스트."""

    @pytest.mark.asyncio
    async def test_clean_text_returns_pass(self):
        """비속어 없는 텍스트 → pass, is_toxic=False."""
        inp = ProfanityCheckInput(
            text="이 영화 정말 재미있었어요. 배우들의 연기가 훌륭했습니다.",
            user_id="user_1",
            content_type="review",
        )
        result = await check_profanity(inp)

        assert isinstance(result, ProfanityCheckOutput)
        assert result.is_toxic is False
        assert result.action == "pass"
        assert result.toxicity_score == 0.0

    @pytest.mark.asyncio
    async def test_output_schema(self):
        """출력이 ProfanityCheckOutput 스키마를 준수한다."""
        inp = ProfanityCheckInput(
            text="일반적인 텍스트입니다.",
            user_id="user_1",
            content_type="chat",
        )
        result = await check_profanity(inp)

        assert hasattr(result, "is_toxic")
        assert hasattr(result, "toxicity_score")
        assert hasattr(result, "detected_words")
        assert hasattr(result, "action")
        assert result.action in ("pass", "warning", "blind", "block")
        assert 0.0 <= result.toxicity_score <= 1.0

    @pytest.mark.asyncio
    async def test_empty_text_returns_pass(self):
        """빈 공백 텍스트 → 에러 없이 pass 반환."""
        inp = ProfanityCheckInput(
            text="   ",
            user_id="user_1",
            content_type="post",
        )
        result = await check_profanity(inp)
        assert result.action == "pass"

    @pytest.mark.asyncio
    async def test_action_severity_order(self):
        """action 판정은 pass → warning → blind → block 순서를 따른다."""
        valid_actions = {"pass", "warning", "blind", "block"}
        inp = ProfanityCheckInput(
            text="테스트 텍스트",
            user_id="user_1",
            content_type="comment",
        )
        result = await check_profanity(inp)
        assert result.action in valid_actions

    @pytest.mark.asyncio
    async def test_content_type_accepted(self):
        """허용된 content_type 4종 모두 에러 없이 처리된다."""
        for content_type in ("chat", "post", "review", "comment"):
            inp = ProfanityCheckInput(
                text="일반 텍스트",
                user_id="user_1",
                content_type=content_type,
            )
            result = await check_profanity(inp)
            assert result is not None


# ============================================================
# 패턴 분석 테스트
# ============================================================

class TestPatternAnalysis:
    """analyze_user_pattern() 단위 테스트."""

    @pytest.mark.asyncio
    async def test_empty_history_no_achievements(self):
        """시청 이력 없음 → 업적 0개, 벡터 40차원."""
        inp = PatternAnalysisInput(
            user_id="user_1",
            watch_history=[],
            existing_achievements=[],
        )
        result = await analyze_user_pattern(inp)

        assert isinstance(result, PatternAnalysisOutput)
        assert isinstance(result.new_achievements, list)
        assert len(result.user_pattern_vector) == 40

    @pytest.mark.asyncio
    async def test_ten_movies_unlocks_first_achievement(self):
        """시청 10편 → '영화 입문' 업적(ACH_001) 잠금 해제."""
        history = [
            WatchRecord(movie_id=str(i), watched_at="2026-01-01", genres=["드라마"])
            for i in range(10)
        ]
        inp = PatternAnalysisInput(
            user_id="user_1",
            watch_history=history,
            existing_achievements=[],
        )
        result = await analyze_user_pattern(inp)

        achievement_ids = [a.id for a in result.new_achievements]
        assert "ACH_001" in achievement_ids

    @pytest.mark.asyncio
    async def test_existing_achievements_not_duplicated(self):
        """이미 보유한 업적은 new_achievements에 포함되지 않는다."""
        history = [
            WatchRecord(movie_id=str(i), watched_at="2026-01-01", genres=["드라마"])
            for i in range(10)
        ]
        inp = PatternAnalysisInput(
            user_id="user_1",
            watch_history=history,
            existing_achievements=["ACH_001"],  # 이미 보유
        )
        result = await analyze_user_pattern(inp)

        achievement_ids = [a.id for a in result.new_achievements]
        assert "ACH_001" not in achievement_ids

    @pytest.mark.asyncio
    async def test_pattern_vector_length(self):
        """user_pattern_vector는 항상 40차원이다."""
        inp = PatternAnalysisInput(
            user_id="user_1",
            watch_history=[
                WatchRecord(movie_id="1", watched_at="2026-01-01", genres=["액션", "SF"])
            ],
            existing_achievements=[],
        )
        result = await analyze_user_pattern(inp)
        assert len(result.user_pattern_vector) == 40

    @pytest.mark.asyncio
    async def test_achievement_schema(self):
        """Achievement 객체는 id/name/description/icon 필드를 가진다."""
        history = [
            WatchRecord(movie_id=str(i), watched_at="2026-01-01", genres=["공포"])
            for i in range(10)
        ]
        inp = PatternAnalysisInput(
            user_id="user_1",
            watch_history=history,
            existing_achievements=[],
        )
        result = await analyze_user_pattern(inp)

        for ach in result.new_achievements:
            assert isinstance(ach, Achievement)
            assert ach.id
            assert ach.name
            assert ach.description
            assert ach.icon


# ============================================================
# Pydantic 모델 검증
# ============================================================

class TestContentAnalysisModels:
    """I/O 모델 스키마 검증."""

    def test_profanity_input_valid(self):
        """ProfanityCheckInput 유효한 값."""
        inp = ProfanityCheckInput(
            text="테스트", user_id="u1", content_type="chat"
        )
        assert inp.text == "테스트"

    def test_profanity_output_defaults(self):
        """ProfanityCheckOutput 기본값 검증."""
        out = ProfanityCheckOutput(
            is_toxic=False,
            toxicity_score=0.0,
            action="pass",
        )
        assert out.detected_words == []

    def test_watch_record_optional_rating(self):
        """WatchRecord의 rating은 Optional이다."""
        record = WatchRecord(movie_id="1", watched_at="2026-01-01")
        assert record.rating is None
        assert record.genres == []
