"""
추천 이유 생성 체인 단위 테스트 (Task 9).

테스트 대상:
- Mock LLM → 4~6문장 한국어 설명 (유저 감정/상황/무드태그 반영)
- fallback 함수 → 장르+평점+배우+무드태그+감정 공감 포함된 기본 설명
- 배치: 3편 → 3개 설명 반환 (user_message, user_mood_tags 전달)
- 에러 → fallback 반환
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import (
    CandidateMovie,
    ExtractedPreferences,
    RankedMovie,
    ScoreDetail,
)
from monglepick.chains.explanation_chain import (
    _build_fallback_explanation,
    generate_explanation,
    generate_explanations_batch,
)


@pytest.mark.asyncio
async def test_generate_explanation_returns_text(mock_ollama):
    """Mock LLM → 한국어 설명 문자열을 반환한다."""
    mock_ollama.set_response(
        "따뜻한 가족 이야기를 찾으시는 분께 딱 맞는 영화예요. "
        "진솔한 감정이 돋보이는 작품이에요."
    )
    result = await generate_explanation(
        movie={"title": "리틀 미스 선샤인", "genres": ["코미디", "드라마"], "rating": 8.0},
    )
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_generate_explanation_with_preferences(mock_ollama, sample_preferences):
    """선호 조건이 프롬프트에 포함되어 LLM 응답이 반환된다."""
    mock_ollama.set_response("SF를 좋아하시는 분께 강력 추천해요.")
    result = await generate_explanation(
        movie={"title": "인터스텔라", "genres": ["SF"], "rating": 8.7},
        preferences=sample_preferences,
    )
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_generate_explanation_error_returns_fallback(mock_ollama):
    """LLM 에러 → fallback 설명 반환."""
    mock_ollama.set_error(RuntimeError("LLM timeout"))
    result = await generate_explanation(
        movie={"title": "인터스텔라", "genres": ["SF", "드라마"], "rating": 8.7},
    )
    assert "SF" in result
    assert "8.7" in result


@pytest.mark.asyncio
async def test_generate_explanation_with_ranked_movie(mock_ollama, sample_ranked_movie):
    """RankedMovie 타입이 정상 처리된다."""
    mock_ollama.set_response("놀란 감독의 걸작이에요.")
    result = await generate_explanation(movie=sample_ranked_movie)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_generate_explanation_with_candidate_movie(mock_ollama, sample_candidate_movie):
    """CandidateMovie 타입이 정상 처리된다."""
    mock_ollama.set_response("웅장한 스케일이 돋보여요.")
    result = await generate_explanation(movie=sample_candidate_movie)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_batch_explanations(mock_ollama):
    """배치: 3편 → 3개 설명 반환."""
    mock_ollama.set_response("좋은 영화예요.")
    movies = [
        {"title": "영화1", "genres": ["액션"], "rating": 7.0},
        {"title": "영화2", "genres": ["코미디"], "rating": 8.0},
        {"title": "영화3", "genres": ["드라마"], "rating": 9.0},
    ]
    results = await generate_explanations_batch(movies)
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)


class TestBuildFallbackExplanation:
    """_build_fallback_explanation 유틸 함수 테스트."""

    def test_with_genres_and_rating(self):
        """장르와 평점이 포함된 기본 설명."""
        result = _build_fallback_explanation({
            "title": "인터스텔라",
            "genres": ["SF", "드라마"],
            "rating": 8.7,
            "director": "크리스토퍼 놀란",
        })
        assert "SF" in result
        assert "8.7" in result
        assert "놀란" in result

    def test_with_emotion_adds_empathy(self):
        """감정이 주어지면 공감 문구가 포함된다."""
        result = _build_fallback_explanation(
            {"title": "리틀 미스 선샤인", "genres": ["코미디"], "rating": 8.0},
            emotion="sad",
        )
        assert "위로" in result

    def test_with_cast_and_mood_tags(self):
        """배우와 무드태그가 포함된 풍부한 설명."""
        result = _build_fallback_explanation({
            "title": "인터스텔라",
            "genres": ["SF"],
            "rating": 8.7,
            "director": "크리스토퍼 놀란",
            "cast": ["매튜 매커너히", "앤 해서웨이"],
            "mood_tags": ["몰입", "웅장", "감동"],
        })
        assert "매튜 매커너히" in result
        assert "몰입" in result

    def test_minimal_info(self):
        """최소 정보로도 유효한 설명 생성."""
        result = _build_fallback_explanation({"title": "영화"})
        assert len(result) > 0
        assert "다양한 장르" in result

    def test_empty_dict(self):
        """빈 dict → 기본 설명 생성."""
        result = _build_fallback_explanation({})
        assert len(result) > 0
