"""
구조화된 후속 질문 힌트 모델/상수 검증 테스트.

ClarificationHint, ClarificationResponse 모델과 FIELD_HINTS 상수를 테스트한다.

테스트 시나리오:
1. ClarificationHint 모델 생성/직렬화
2. ClarificationResponse 모델 생성/직렬화
3. FIELD_HINTS 상수 무결성 (7개 필드, 각 필드에 label/options)
4. FIELD_HINTS 옵션 내용 검증 (장르 17개, 분위기 14개 등)
5. ClarificationResponse.model_dump() JSON 호환성
6. 빈 hints 허용
7. primary_field 검증
8. FIELD_HINTS와 ExtractedPreferences 필드 매핑
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import (
    FIELD_HINTS,
    ClarificationHint,
    ClarificationResponse,
    ExtractedPreferences,
)


class TestClarificationHint:
    """ClarificationHint 모델 테스트."""

    def test_basic_creation(self):
        """기본 힌트 생성."""
        hint = ClarificationHint(
            field="genre_preference",
            label="장르",
            options=["액션", "SF", "로맨스"],
        )
        assert hint.field == "genre_preference"
        assert hint.label == "장르"
        assert len(hint.options) == 3

    def test_empty_options(self):
        """빈 옵션 리스트 (reference_movies 용)."""
        hint = ClarificationHint(
            field="reference_movies",
            label="참조 영화",
            options=[],
        )
        assert hint.options == []

    def test_model_dump(self):
        """JSON 직렬화."""
        hint = ClarificationHint(
            field="mood",
            label="분위기",
            options=["힐링", "감동"],
        )
        data = hint.model_dump()
        assert data["field"] == "mood"
        assert data["label"] == "분위기"
        assert data["options"] == ["힐링", "감동"]


class TestClarificationResponse:
    """ClarificationResponse 모델 테스트."""

    def test_basic_creation(self):
        """기본 응답 생성."""
        resp = ClarificationResponse(
            question="어떤 장르를 좋아하세요?",
            hints=[
                ClarificationHint(field="genre_preference", label="장르", options=["액션", "SF"]),
            ],
            primary_field="genre_preference",
        )
        assert resp.question == "어떤 장르를 좋아하세요?"
        assert len(resp.hints) == 1
        assert resp.primary_field == "genre_preference"

    def test_empty_hints_allowed(self):
        """빈 힌트 리스트 허용 (모든 선호가 충분할 때)."""
        resp = ClarificationResponse(
            question="더 자세히 알려주세요.",
            hints=[],
            primary_field="",
        )
        assert resp.hints == []
        assert resp.primary_field == ""

    def test_max_three_hints(self):
        """힌트 최대 3개 구성."""
        hints = [
            ClarificationHint(field="genre_preference", label="장르", options=["액션"]),
            ClarificationHint(field="mood", label="분위기", options=["힐링"]),
            ClarificationHint(field="era", label="시대", options=["최신"]),
        ]
        resp = ClarificationResponse(
            question="테스트",
            hints=hints,
            primary_field="genre_preference",
        )
        assert len(resp.hints) == 3

    def test_model_dump_json_compatible(self):
        """model_dump()가 JSON 호환 dict를 반환한다 (SSE 이벤트용)."""
        import json

        resp = ClarificationResponse(
            question="어떤 장르?",
            hints=[
                ClarificationHint(field="genre_preference", label="장르", options=["액션"]),
            ],
            primary_field="genre_preference",
        )
        data = resp.model_dump()
        # JSON 직렬화 가능해야 함
        json_str = json.dumps(data, ensure_ascii=False)
        assert "genre_preference" in json_str
        assert "장르" in json_str


class TestFieldHints:
    """FIELD_HINTS 상수 검증 테스트."""

    def test_seven_fields_exist(self):
        """7개 필드가 모두 존재한다."""
        expected_fields = {
            "genre_preference", "mood", "viewing_context",
            "platform", "era", "exclude", "reference_movies",
        }
        assert set(FIELD_HINTS.keys()) == expected_fields

    def test_each_field_has_label_and_options(self):
        """각 필드에 label과 options 키가 있다."""
        for field_name, hint_info in FIELD_HINTS.items():
            assert "label" in hint_info, f"{field_name}에 label 없음"
            assert "options" in hint_info, f"{field_name}에 options 없음"
            assert isinstance(hint_info["label"], str)
            assert isinstance(hint_info["options"], list)

    def test_genre_options_count(self):
        """장르 옵션이 17개 존재한다."""
        assert len(FIELD_HINTS["genre_preference"]["options"]) == 17
        assert "액션" in FIELD_HINTS["genre_preference"]["options"]
        assert "SF" in FIELD_HINTS["genre_preference"]["options"]

    def test_mood_options_count(self):
        """분위기 옵션이 14개 존재한다."""
        assert len(FIELD_HINTS["mood"]["options"]) == 14
        assert "힐링" in FIELD_HINTS["mood"]["options"]
        assert "감동" in FIELD_HINTS["mood"]["options"]

    def test_viewing_context_options(self):
        """시청 상황 옵션 4개."""
        assert len(FIELD_HINTS["viewing_context"]["options"]) == 4
        assert "혼자" in FIELD_HINTS["viewing_context"]["options"]

    def test_platform_options(self):
        """플랫폼 옵션에 주요 OTT 포함."""
        platforms = FIELD_HINTS["platform"]["options"]
        assert "넷플릭스" in platforms
        assert "극장" in platforms

    def test_reference_movies_empty_options(self):
        """참조 영화 옵션은 빈 배열 (자유 입력)."""
        assert FIELD_HINTS["reference_movies"]["options"] == []

    def test_field_hints_maps_to_extracted_preferences(self):
        """FIELD_HINTS 키가 ExtractedPreferences 필드명과 매핑된다."""
        prefs = ExtractedPreferences()
        prefs_fields = set(prefs.model_fields.keys())
        # FIELD_HINTS의 모든 키가 ExtractedPreferences 필드에 존재해야 함
        for field_name in FIELD_HINTS:
            assert field_name in prefs_fields, (
                f"FIELD_HINTS['{field_name}']이 ExtractedPreferences에 없음"
            )
