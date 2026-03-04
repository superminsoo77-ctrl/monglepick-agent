"""
이미지 분석 체인 단위 테스트.

analyze_image() 체인의 정상 동작, JSON 파싱, 무드 필터링, 에러 복원력을 테스트한다.
Ollama 서버 없이 mock 기반으로 실행된다.
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import ImageAnalysisResult
from monglepick.chains.image_analysis_chain import (
    MOOD_WHITELIST,
    _parse_json_response,
    analyze_image,
)


# ============================================================
# 1. _parse_json_response 유닛 테스트
# ============================================================

class TestParseJsonResponse:
    """JSON 파싱 헬퍼 테스트."""

    def test_direct_json(self):
        """직접 JSON 문자열을 파싱한다."""
        raw = '{"genre_cues": ["SF"], "mood_cues": ["웅장"]}'
        result = _parse_json_response(raw)
        assert result is not None
        assert result["genre_cues"] == ["SF"]

    def test_code_block_json(self):
        """```json ... ``` 코드 블록 내 JSON을 추출한다."""
        raw = '```json\n{"genre_cues": ["액션"]}\n```'
        result = _parse_json_response(raw)
        assert result is not None
        assert result["genre_cues"] == ["액션"]

    def test_brace_extraction(self):
        """텍스트 중간의 { ... } JSON을 추출한다."""
        raw = '분석 결과: {"genre_cues": ["드라마"]} 입니다.'
        result = _parse_json_response(raw)
        assert result is not None
        assert result["genre_cues"] == ["드라마"]

    def test_empty_input(self):
        """빈 입력 시 None을 반환한다."""
        assert _parse_json_response("") is None
        assert _parse_json_response(None) is None

    def test_invalid_json(self):
        """잘못된 JSON 시 None을 반환한다."""
        assert _parse_json_response("not json at all") is None


# ============================================================
# 2. analyze_image 체인 테스트 (mock LLM)
# ============================================================

class TestAnalyzeImage:
    """이미지 분석 체인 테스트."""

    @pytest.mark.asyncio
    async def test_no_image_data_returns_not_analyzed(self):
        """이미지 데이터 없으면 analyzed=False를 반환한다."""
        result = await analyze_image(image_data="", current_input="추천해줘")
        assert isinstance(result, ImageAnalysisResult)
        assert result.analyzed is False

    @pytest.mark.asyncio
    async def test_analyze_image_success(self, mock_ollama):
        """정상 이미지 분석 — JSON 응답을 ImageAnalysisResult로 변환한다."""
        # VLM이 반환할 JSON 응답 설정
        mock_ollama.set_response(
            '{"genre_cues": ["SF", "모험"], "mood_cues": ["웅장", "몰입"], '
            '"visual_elements": ["우주선", "별"], "search_keywords": ["우주"], '
            '"description": "우주 배경 SF 이미지", "is_movie_poster": true, '
            '"detected_movie_title": "인터스텔라"}'
        )
        result = await analyze_image(
            image_data="base64encodedimage",
            current_input="이런 영화 추천해줘",
        )
        assert result.analyzed is True
        assert "SF" in result.genre_cues
        assert "웅장" in result.mood_cues
        assert result.is_movie_poster is True
        assert result.detected_movie_title == "인터스텔라"

    @pytest.mark.asyncio
    async def test_mood_whitelist_filtering(self, mock_ollama):
        """mood_cues가 MOOD_WHITELIST 범위 내로 필터링된다."""
        # 화이트리스트에 없는 무드 포함
        mock_ollama.set_response(
            '{"genre_cues": ["액션"], "mood_cues": ["웅장", "격렬한", "스릴", "잘못된무드"], '
            '"visual_elements": [], "search_keywords": [], '
            '"description": "", "is_movie_poster": false, "detected_movie_title": null}'
        )
        result = await analyze_image(image_data="base64data")
        assert result.analyzed is True
        # "격렬한"과 "잘못된무드"는 화이트리스트에 없으므로 필터링됨
        assert "격렬한" not in result.mood_cues
        assert "잘못된무드" not in result.mood_cues
        assert "웅장" in result.mood_cues
        assert "스릴" in result.mood_cues

    @pytest.mark.asyncio
    async def test_llm_error_returns_not_analyzed(self, mock_ollama):
        """LLM 에러 시 analyzed=False를 반환한다 (에러 전파 금지)."""
        mock_ollama.set_error(RuntimeError("VLM connection failed"))
        result = await analyze_image(image_data="base64data")
        assert isinstance(result, ImageAnalysisResult)
        assert result.analyzed is False

    @pytest.mark.asyncio
    async def test_invalid_json_response(self, mock_ollama):
        """LLM이 잘못된 JSON을 반환하면 analyzed=False."""
        mock_ollama.set_response("This is not valid JSON")
        result = await analyze_image(image_data="base64data")
        assert result.analyzed is False


# ============================================================
# 3. MOOD_WHITELIST 검증
# ============================================================

class TestMoodWhitelist:
    """무드 화이트리스트 검증."""

    def test_whitelist_has_25_items(self):
        """화이트리스트가 25개 항목을 포함한다."""
        assert len(MOOD_WHITELIST) == 25

    def test_common_moods_in_whitelist(self):
        """주요 무드 태그가 화이트리스트에 존재한다."""
        expected = {"유쾌", "따뜻", "스릴", "웅장", "힐링", "다크", "로맨틱", "판타지"}
        assert expected.issubset(MOOD_WHITELIST)
