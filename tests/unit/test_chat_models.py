"""
Chat Agent Pydantic 모델 단위 테스트 (Task 2).

테스트 대상:
- IntentResult: 유효/무효 값 검증
- EmotionResult: emotion=None 허용
- ExtractedPreferences: 전체 None 허용
- merge_preferences: 병합 로직
- calculate_sufficiency: 가중치 합산
- is_sufficient: 충분성 판정
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from monglepick.agents.chat.models import (
    SUFFICIENCY_THRESHOLD,
    TURN_COUNT_OVERRIDE,
    CandidateMovie,
    EmotionResult,
    ExtractedPreferences,
    IntentResult,
    RankedMovie,
    ScoreDetail,
    SearchQuery,
    calculate_sufficiency,
    is_sufficient,
    merge_preferences,
)


# ============================================================
# IntentResult 테스트
# ============================================================


class TestIntentResult:
    """IntentResult 모델 검증 테스트."""

    def test_valid_intents(self):
        """6가지 유효한 intent 값이 모두 통과한다."""
        for intent in ["recommend", "search", "info", "theater", "booking", "general"]:
            result = IntentResult(intent=intent, confidence=0.9)
            assert result.intent == intent

    def test_confidence_range(self):
        """confidence가 0.0~1.0 범위 내에서 유효하다."""
        # 유효 범위
        assert IntentResult(intent="general", confidence=0.0).confidence == 0.0
        assert IntentResult(intent="general", confidence=1.0).confidence == 1.0
        assert IntentResult(intent="general", confidence=0.5).confidence == 0.5

    def test_confidence_out_of_range(self):
        """confidence가 0.0~1.0 범위를 벗어나면 ValidationError."""
        with pytest.raises(ValidationError):
            IntentResult(intent="general", confidence=-0.1)
        with pytest.raises(ValidationError):
            IntentResult(intent="general", confidence=1.1)

    def test_invalid_intent(self):
        """유효하지 않은 intent 값은 ValidationError."""
        with pytest.raises(ValidationError):
            IntentResult(intent="invalid_intent", confidence=0.5)

    def test_default_values(self):
        """기본값이 general/0.0이다."""
        result = IntentResult()
        assert result.intent == "general"
        assert result.confidence == 0.0


# ============================================================
# EmotionResult 테스트
# ============================================================


class TestEmotionResult:
    """EmotionResult 모델 검증 테스트."""

    def test_emotion_none_allowed(self):
        """emotion=None이 허용된다 (감정 미감지)."""
        result = EmotionResult(emotion=None, mood_tags=[])
        assert result.emotion is None

    def test_valid_emotion(self):
        """유효한 감정 값이 저장된다."""
        result = EmotionResult(emotion="happy", mood_tags=["유쾌"])
        assert result.emotion == "happy"
        assert result.mood_tags == ["유쾌"]

    def test_empty_mood_tags(self):
        """mood_tags가 빈 리스트일 수 있다."""
        result = EmotionResult(emotion="sad", mood_tags=[])
        assert result.mood_tags == []

    def test_default_values(self):
        """기본값이 None/빈 리스트이다."""
        result = EmotionResult()
        assert result.emotion is None
        assert result.mood_tags == []


# ============================================================
# ExtractedPreferences 테스트
# ============================================================


class TestExtractedPreferences:
    """ExtractedPreferences 모델 검증 테스트."""

    def test_all_none_allowed(self):
        """모든 필드가 None/빈 값이어도 유효하다."""
        prefs = ExtractedPreferences()
        assert prefs.genre_preference is None
        assert prefs.mood is None
        assert prefs.viewing_context is None
        assert prefs.platform is None
        assert prefs.reference_movies == []
        assert prefs.era is None
        assert prefs.exclude is None

    def test_partial_fill(self):
        """일부 필드만 채워도 유효하다."""
        prefs = ExtractedPreferences(
            genre_preference="SF",
            reference_movies=["인터스텔라"],
        )
        assert prefs.genre_preference == "SF"
        assert prefs.mood is None
        assert prefs.reference_movies == ["인터스텔라"]


# ============================================================
# merge_preferences 테스트
# ============================================================


class TestMergePreferences:
    """선호 조건 병합 로직 테스트."""

    def test_merge_new_overrides_prev(self):
        """새 값이 non-None이면 이전 값을 덮어쓴다."""
        prev = ExtractedPreferences(genre_preference="액션", mood="유쾌")
        curr = ExtractedPreferences(genre_preference="SF", mood=None)
        merged = merge_preferences(prev, curr)
        # SF가 액션을 덮어씀
        assert merged.genre_preference == "SF"
        # mood=None이므로 이전 값 유지
        assert merged.mood == "유쾌"

    def test_merge_none_preserves_prev(self):
        """새 값이 None이면 이전 값을 유지한다."""
        prev = ExtractedPreferences(platform="넷플릭스", era="2020년대")
        curr = ExtractedPreferences()  # 모두 None
        merged = merge_preferences(prev, curr)
        assert merged.platform == "넷플릭스"
        assert merged.era == "2020년대"

    def test_merge_reference_movies_union(self):
        """reference_movies는 합집합 (중복 제거)."""
        prev = ExtractedPreferences(reference_movies=["인셉션", "인터스텔라"])
        curr = ExtractedPreferences(reference_movies=["인터스텔라", "테넷"])
        merged = merge_preferences(prev, curr)
        assert merged.reference_movies == ["인셉션", "인터스텔라", "테넷"]

    def test_merge_prev_none(self):
        """prev=None이면 curr 그대로 반환."""
        curr = ExtractedPreferences(genre_preference="코미디")
        merged = merge_preferences(None, curr)
        assert merged.genre_preference == "코미디"

    def test_merge_both_empty(self):
        """둘 다 비어있으면 빈 결과."""
        merged = merge_preferences(ExtractedPreferences(), ExtractedPreferences())
        assert merged.genre_preference is None
        assert merged.reference_movies == []


# ============================================================
# calculate_sufficiency 테스트
# ============================================================


class TestCalculateSufficiency:
    """선호 조건 충분성 점수 계산 테스트."""

    def test_genre_plus_mood_equals_4(self):
        """genre + mood = 2.0 + 2.0 = 4.0."""
        prefs = ExtractedPreferences(genre_preference="SF", mood="웅장한")
        assert calculate_sufficiency(prefs) == 4.0

    def test_all_none_equals_zero(self):
        """모든 필드가 None이면 0.0."""
        prefs = ExtractedPreferences()
        assert calculate_sufficiency(prefs) == 0.0

    def test_has_emotion_adds_mood_weight(self):
        """has_emotion=True이면 mood 없어도 2.0 추가."""
        prefs = ExtractedPreferences(genre_preference="SF")
        # mood 없음, emotion 있음 → genre(2.0) + mood(2.0) = 4.0
        assert calculate_sufficiency(prefs, has_emotion=True) == 4.0

    def test_has_emotion_no_double_count(self):
        """mood가 이미 있으면 has_emotion으로 이중 계산하지 않는다."""
        prefs = ExtractedPreferences(genre_preference="SF", mood="웅장한")
        # mood(2.0)와 has_emotion이 동시 → mood 가중치 1번만
        assert calculate_sufficiency(prefs, has_emotion=True) == 4.0

    def test_all_filled(self):
        """모든 필드가 채워지면 8.5."""
        prefs = ExtractedPreferences(
            genre_preference="SF",
            mood="웅장한",
            viewing_context="혼자",
            platform="넷플릭스",
            reference_movies=["인터스텔라"],
            era="2020년대",
            exclude="공포 제외",
        )
        # 2.0 + 2.0 + 1.0 + 1.0 + 1.5 + 0.5 + 0.5 = 8.5
        assert calculate_sufficiency(prefs) == 8.5

    def test_reference_movies_weight(self):
        """reference_movies가 채워지면 1.5 추가."""
        prefs = ExtractedPreferences(reference_movies=["인셉션"])
        assert calculate_sufficiency(prefs) == 1.5


# ============================================================
# is_sufficient 테스트
# ============================================================


class TestIsSufficient:
    """추천 진행 가능 여부 판정 테스트."""

    def test_sufficient_by_score(self):
        """가중치 합산 >= 2.5이면 충분."""
        prefs = ExtractedPreferences(genre_preference="SF", mood="웅장한")
        # 4.0 >= 2.5 → True
        assert is_sufficient(prefs) is True

    def test_insufficient_by_score(self):
        """핵심 필드/의도/동적필터 모두 없으면 불충분 (Intent-First)."""
        # viewing_context만 있으면 불충분 (핵심 필드가 아님)
        prefs = ExtractedPreferences(viewing_context="혼자")
        assert is_sufficient(prefs) is False

        # 아무것도 없으면 불충분
        prefs_empty = ExtractedPreferences()
        assert is_sufficient(prefs_empty) is False

    def test_intent_first_sufficient(self):
        """Intent-First: user_intent가 있으면 즉시 충분."""
        prefs = ExtractedPreferences(user_intent="평점 높은 인기 영화")
        assert is_sufficient(prefs) is True

    def test_dynamic_filter_sufficient(self):
        """Intent-First: dynamic_filters가 있으면 즉시 충분."""
        from monglepick.agents.chat.models import FilterCondition
        prefs = ExtractedPreferences(
            dynamic_filters=[FilterCondition(field="rating", operator="gte", value=7.0)]
        )
        assert is_sufficient(prefs) is True

    def test_core_field_sufficient(self):
        """Intent-First: genre/mood/reference_movies 중 하나면 충분."""
        # genre만 있으면 충분
        assert is_sufficient(ExtractedPreferences(genre_preference="SF")) is True
        # mood만 있으면 충분
        assert is_sufficient(ExtractedPreferences(mood="웅장한")) is True
        # reference_movies만 있으면 충분
        assert is_sufficient(ExtractedPreferences(reference_movies=["인셉션"])) is True

    def test_turn_count_override(self):
        """turn_count >= 2이면 선호 부족해도 충분 (TURN_COUNT_OVERRIDE=2)."""
        prefs = ExtractedPreferences()  # 0.0
        assert is_sufficient(prefs, turn_count=2) is True
        assert is_sufficient(prefs, turn_count=5) is True

    def test_turn_count_below_threshold(self):
        """turn_count < 2이고 핵심 정보 없으면 불충분."""
        prefs = ExtractedPreferences()
        assert is_sufficient(prefs, turn_count=1) is False

    def test_emotion_contributes_to_sufficiency(self):
        """감정이 감지되면 가중치 합산에 기여한다 (calculate_sufficiency)."""
        from monglepick.agents.chat.models import calculate_sufficiency
        prefs = ExtractedPreferences(genre_preference="SF")
        # genre(2.0) = 2.0 (감정 없이)
        score_no_emotion = calculate_sufficiency(prefs)
        # genre(2.0) + mood(2.0, emotion→mood) = 4.0
        score_with_emotion = calculate_sufficiency(prefs, has_emotion=True)
        assert score_with_emotion > score_no_emotion
        # Intent-First에서 genre만 있으면 이미 충분
        assert is_sufficient(prefs) is True
        assert is_sufficient(prefs, has_emotion=True) is True


# ============================================================
# Phase 3/4 모델 기본 검증
# ============================================================


class TestPhase3Models:
    """Phase 3/4 준비 모델의 기본 생성 테스트."""

    def test_search_query_defaults(self):
        """SearchQuery 기본값이 올바르다."""
        sq = SearchQuery()
        assert sq.semantic_query == ""
        assert sq.limit == 15

    def test_candidate_movie_creation(self):
        """CandidateMovie가 정상 생성된다."""
        movie = CandidateMovie(id="123", title="테스트 영화")
        assert movie.id == "123"
        assert movie.rrf_score == 0.0

    def test_score_detail_defaults(self):
        """ScoreDetail 기본값이 0.0이다."""
        sd = ScoreDetail()
        assert sd.cf_score == 0.0
        assert sd.similar_to == []

    def test_ranked_movie_with_score(self):
        """RankedMovie에 ScoreDetail이 포함된다."""
        rm = RankedMovie(
            id="123",
            title="테스트",
            rank=1,
            score_detail=ScoreDetail(cf_score=0.3, cbf_score=0.8),
        )
        assert rm.score_detail.cf_score == 0.3
        assert rm.rank == 1
