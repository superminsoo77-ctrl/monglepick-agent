"""
now_showing_cache 단위 테스트 (2026-04-27 신규).

대상: monglepick.agents.chat.now_showing_cache

핵심 계약:
 1) 정규화 — 공백/구두점/괄호 제거 + 소문자 변환
 2) 캐시 — 첫 호출 시 KOBIS fetch, 두 번째 호출은 캐시 hit (재호출 X)
 3) annotate_movies — RankedMovie.is_now_showing 을 매칭 결과로 in-place 갱신
 4) KOBIS 실패 (문자열 반환) → 빈 set, 모든 영화 False 유지 (graceful)
 5) title_en fallback — 한국어 매칭 실패 시 영문 제목으로 재시도
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from monglepick.agents.chat.models import RankedMovie, ScoreDetail
from monglepick.agents.chat.now_showing_cache import (
    _normalize_title,
    annotate_movies,
    get_now_showing_titles,
    is_now_showing,
    reset_cache_for_tests,
)


# ── 매 테스트마다 캐시 초기화 ────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_cache():
    reset_cache_for_tests()
    yield
    reset_cache_for_tests()


# ────────────────────────────────────────────────────────────────────
# 1) 정규화 단위 테스트
# ────────────────────────────────────────────────────────────────────

class TestNormalizeTitle:
    """제목 정규화는 매칭 정확도의 기반이므로 다양한 케이스 검증."""

    def test_strips_whitespace_and_punctuation(self):
        assert _normalize_title("아바타: 물의 길") == "아바타물의길"

    def test_lowercases_english(self):
        assert _normalize_title("Mission Impossible 7") == "missionimpossible7"

    def test_handles_full_width_space(self):
        # 전각 공백 (U+3000) 도 제거
        assert _normalize_title("기생충　") == "기생충"

    def test_handles_quotes_and_brackets(self):
        assert _normalize_title("(아노라)") == "아노라"
        assert _normalize_title('"위키드"') == "위키드"

    def test_handles_korean_punctuation(self):
        # 가운뎃점, 말줄임표 등 한국어 구두점도 제거
        assert _normalize_title("로미오·줄리엣") == "로미오줄리엣"

    def test_returns_empty_for_falsy_input(self):
        assert _normalize_title(None) == ""
        assert _normalize_title("") == ""


# ────────────────────────────────────────────────────────────────────
# 2) 캐시 동작
# ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cache_hits_on_second_call():
    """첫 호출에서만 KOBIS 가 호출되고, 두 번째는 캐시 hit."""
    sample_movies = [
        {"rank": 1, "movie_cd": "01", "movie_nm": "기생충", "audi_acc": 1, "open_dt": "20240101", "rank_inten": 0, "rank_old_and_new": "OLD"},
        {"rank": 2, "movie_cd": "02", "movie_nm": "아바타: 물의 길", "audi_acc": 1, "open_dt": "20240101", "rank_inten": 0, "rank_old_and_new": "OLD"},
    ]
    mock_invoke = AsyncMock(return_value=sample_movies)

    with patch(
        "monglepick.agents.chat.now_showing_cache._invoke_kobis_top10",
        mock_invoke,
    ):
        first = await get_now_showing_titles()
        second = await get_now_showing_titles()

    assert first == {"기생충", "아바타물의길"}
    assert second == first
    # 두 번째 호출은 캐시 hit 으로 KOBIS 재호출 없음
    assert mock_invoke.call_count == 1


@pytest.mark.asyncio
async def test_returns_empty_set_when_kobis_fails():
    """KOBIS 도구가 안내 문자열을 반환하면 빈 set 으로 graceful degrade."""
    mock_invoke = AsyncMock(return_value="현재 상영작 정보를 잠시 불러올 수 없어요")

    with patch(
        "monglepick.agents.chat.now_showing_cache._invoke_kobis_top10",
        mock_invoke,
    ):
        titles = await get_now_showing_titles()

    assert titles == set()


@pytest.mark.asyncio
async def test_returns_empty_set_when_kobis_raises():
    """KOBIS ainvoke 예외 발생도 빈 set 으로 graceful (안전망)."""
    mock_invoke = AsyncMock(side_effect=RuntimeError("network down"))

    with patch(
        "monglepick.agents.chat.now_showing_cache._invoke_kobis_top10",
        mock_invoke,
    ):
        titles = await get_now_showing_titles()

    assert titles == set()


# ────────────────────────────────────────────────────────────────────
# 3) is_now_showing — 단일 영화 매칭
# ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_is_now_showing_matches_korean_title():
    sample = [{"rank": 1, "movie_cd": "01", "movie_nm": "기생충", "audi_acc": 1,
               "open_dt": "20240101", "rank_inten": 0, "rank_old_and_new": "OLD"}]
    with patch(
        "monglepick.agents.chat.now_showing_cache._invoke_kobis_top10",
        AsyncMock(return_value=sample),
    ):
        assert await is_now_showing("기생충") is True
        # 공백/구두점 차이는 정규화로 흡수
        assert await is_now_showing("기 생 충") is True


@pytest.mark.asyncio
async def test_is_now_showing_falls_back_to_title_en():
    """한국어 매칭 실패 시 영문 제목으로 재시도."""
    # KOBIS 가 영문 제목 그대로 내려준 가상 케이스
    sample = [{"rank": 1, "movie_cd": "01", "movie_nm": "Wicked", "audi_acc": 1,
               "open_dt": "20240101", "rank_inten": 0, "rank_old_and_new": "OLD"}]
    with patch(
        "monglepick.agents.chat.now_showing_cache._invoke_kobis_top10",
        AsyncMock(return_value=sample),
    ):
        # 한국어 제목은 KOBIS 응답에 없음 → False 가 아니라 영문 fallback 으로 True
        assert await is_now_showing("위키드", title_en="Wicked") is True


@pytest.mark.asyncio
async def test_is_now_showing_returns_false_for_unmatched():
    sample = [{"rank": 1, "movie_cd": "01", "movie_nm": "기생충", "audi_acc": 1,
               "open_dt": "20240101", "rank_inten": 0, "rank_old_and_new": "OLD"}]
    with patch(
        "monglepick.agents.chat.now_showing_cache._invoke_kobis_top10",
        AsyncMock(return_value=sample),
    ):
        assert await is_now_showing("라스베가스에서만 생길 수 있는 일") is False


# ────────────────────────────────────────────────────────────────────
# 4) annotate_movies — RankedMovie 리스트 in-place 갱신
# ────────────────────────────────────────────────────────────────────

def _make_ranked(title: str, title_en: str = "") -> RankedMovie:
    """테스트용 RankedMovie 인스턴스 헬퍼."""
    return RankedMovie(
        id=f"test-{title}",
        title=title,
        title_en=title_en,
        rank=1,
        score_detail=ScoreDetail(),
    )


@pytest.mark.asyncio
async def test_annotate_movies_sets_flag_only_for_matched():
    """매칭된 영화만 is_now_showing=True 가 되고 나머지는 default(False) 유지."""
    sample = [
        {"rank": 1, "movie_cd": "01", "movie_nm": "기생충", "audi_acc": 1,
         "open_dt": "20240101", "rank_inten": 0, "rank_old_and_new": "OLD"},
        {"rank": 2, "movie_cd": "02", "movie_nm": "아바타: 물의 길", "audi_acc": 1,
         "open_dt": "20240101", "rank_inten": 0, "rank_old_and_new": "OLD"},
    ]
    movies = [
        _make_ranked("기생충"),
        _make_ranked("올드보이"),                # 매칭 X
        _make_ranked("아바타: 물의 길"),         # 정규화로 매칭 O
    ]

    with patch(
        "monglepick.agents.chat.now_showing_cache._invoke_kobis_top10",
        AsyncMock(return_value=sample),
    ):
        await annotate_movies(movies)

    assert movies[0].is_now_showing is True
    assert movies[1].is_now_showing is False
    assert movies[2].is_now_showing is True


@pytest.mark.asyncio
async def test_annotate_movies_leaves_all_false_when_kobis_unavailable():
    """KOBIS 미가용 시 모든 영화 False 유지 — 영화관 버튼 거짓 긍정 방지."""
    movies = [_make_ranked("기생충"), _make_ranked("올드보이")]

    with patch(
        "monglepick.agents.chat.now_showing_cache._invoke_kobis_top10",
        AsyncMock(return_value="현재 상영작 정보를 잠시 불러올 수 없어요"),
    ):
        await annotate_movies(movies)

    assert all(m.is_now_showing is False for m in movies)


@pytest.mark.asyncio
async def test_annotate_movies_handles_empty_list():
    """빈 리스트는 KOBIS 호출 없이 즉시 반환 (불필요한 네트워크 호출 방지)."""
    mock_invoke = AsyncMock(return_value=[])
    with patch(
        "monglepick.agents.chat.now_showing_cache._invoke_kobis_top10",
        mock_invoke,
    ):
        await annotate_movies([])
    mock_invoke.assert_not_called()
