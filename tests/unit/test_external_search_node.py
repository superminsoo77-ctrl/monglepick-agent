"""
external_search_node (v2) + search_external_movies_v2 + 장르매핑 단위 테스트.

2026-04-27: DuckDuckGo 단일 경로 → TMDB/KOBIS/KMDb fan-out 재설계에 따른 테스트 전면 갱신.

테스트 구조:
 1. TestHasRecencySignal          — _has_recency_signal 판정 (기존 유지)
 2. TestRouteAfterRetrievalExternal — route_after_retrieval 분기 (기존 유지)
 3. TestGenreMapping               — 한국어 장르 → TMDB genre_id 매핑 (신규)
 4. TestSearchExternalMoviesV2     — search_external_movies_v2 단위 (신규)
 5. TestExternalSearchNodeV2       — external_search_node 통합 (v2 스키마)

핵심 계약:
 1) "최신/올해/2026년" 키워드 OR dynamic_filters[release_year>=N] 이 current_year-1 이상
    → _has_recency_signal True
 2) 후보 0건 + recency_signal True → route_after_retrieval 이 "external_search_node" 반환
 3) SF 2026 글로벌 → TMDB Discover 가 genre_id=878 + year=2026 으로 호출됨
 4) "올해 한국 신작" → KOBIS 우선 호출
 5) TMDB 만 응답, KOBIS 실패 → 부분 실패 fallback (TMDB 결과 반환)
 6) 모든 소스 0건 → 빈 리스트
 7) external_search_node: genre_preference="SF", release_year_gte=2026 →
    ranked_movies[0].id 가 "external_tmdb_" 또는 "external_kobis_" 등 "external_" 로 시작
 8) 모든 소스 실패해도 ranked_movies=[] 로 graceful degrade
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monglepick.agents.chat.graph import _has_recency_signal, route_after_retrieval
from monglepick.agents.chat.models import ExtractedPreferences, FilterCondition
from monglepick.agents.chat.nodes import external_search_node
from monglepick.tools.external_movie_search import (
    _resolve_genre_ids,
    search_external_movies_v2,
)


# ============================================================
# 1. _has_recency_signal (기존 유지)
# ============================================================

class TestHasRecencySignal:
    """_has_recency_signal: 최신 영화 시그널 판정."""

    def test_recent_filter_returns_true(self):
        """dynamic_filters[release_year>=current_year] 이면 True."""
        current_year = datetime.now().year
        prefs = ExtractedPreferences(
            dynamic_filters=[
                FilterCondition(field="release_year", operator="gte", value=str(current_year)),
            ],
        )
        state = {"current_input": "", "preferences": prefs}
        assert _has_recency_signal(state) is True

    def test_old_filter_returns_false(self):
        """release_year>=2010 같은 오래된 필터는 최신 시그널이 아님."""
        prefs = ExtractedPreferences(
            dynamic_filters=[
                FilterCondition(field="release_year", operator="gte", value="2010"),
            ],
        )
        state = {"current_input": "추천해줘", "preferences": prefs}
        assert _has_recency_signal(state) is False

    @pytest.mark.parametrize("keyword", ["최신", "최근", "올해", "신작", "요즘"])
    def test_recency_keywords_in_input(self, keyword):
        """원문 입력에 최신 키워드가 있으면 True."""
        state = {"current_input": f"{keyword} 영화 추천"}
        assert _has_recency_signal(state) is True

    def test_future_year_in_input(self):
        """current_year 숫자가 직접 언급되면 True."""
        current_year = datetime.now().year
        state = {"current_input": f"{current_year}년 개봉 영화"}
        assert _has_recency_signal(state) is True

    def test_plain_request_returns_false(self):
        """아무 시그널도 없는 일반 요청은 False."""
        state = {"current_input": "재밌는 영화 추천해줘"}
        assert _has_recency_signal(state) is False

    def test_invalid_filter_value_does_not_crash(self):
        """release_year value 가 숫자가 아니어도 graceful — False 반환."""
        prefs = ExtractedPreferences(
            dynamic_filters=[
                FilterCondition(field="release_year", operator="gte", value="not-a-number"),
            ],
        )
        state = {"current_input": "", "preferences": prefs}
        assert _has_recency_signal(state) is False


# ============================================================
# 2. route_after_retrieval: 최신 시그널 분기 (기존 유지)
# ============================================================

class TestRouteAfterRetrievalExternal:
    """route_after_retrieval: 후보 0건 + 최신 시그널 → external_search_node."""

    def test_zero_candidates_with_recency_routes_external(self):
        """후보 0건 + 2026년 입력 → external_search_node."""
        state = {
            "candidate_movies": [],
            "current_input": "2026년 영화 추천",
            "turn_count": 1,
        }
        assert route_after_retrieval(state) == "external_search_node"

    def test_zero_candidates_without_recency_routes_question(self):
        """최신 시그널 없이 후보 0건이면 기존대로 question_generator."""
        state = {
            "candidate_movies": [],
            "current_input": "영화 추천",
            "turn_count": 1,
        }
        assert route_after_retrieval(state) == "question_generator"

    def test_has_candidates_ignores_external_branch(self):
        """후보가 있으면 최신 시그널과 무관하게 external 로 가지 않는다."""
        from monglepick.agents.chat.models import CandidateMovie
        candidates = [
            CandidateMovie(id=str(i), title=f"M{i}", rrf_score=0.1) for i in range(5)
        ]
        state = {
            "candidate_movies": candidates,
            "current_input": "최신 영화 추천",
            "turn_count": 1,
        }
        assert route_after_retrieval(state) != "external_search_node"


# ============================================================
# 3. 장르 매핑 테이블 단위 테스트 (신규)
# ============================================================

class TestGenreMapping:
    """_resolve_genre_ids: 한국어 장르 → TMDB genre_id 매핑."""

    def test_sf_maps_to_878(self):
        """'SF' → 878 (Science Fiction)."""
        assert _resolve_genre_ids(["SF"]) == [878]

    def test_sf_synonyms(self):
        """SF 동의어: '사이언스픽션', '공상과학' 모두 878 로 매핑."""
        assert _resolve_genre_ids(["사이언스픽션"]) == [878]
        assert _resolve_genre_ids(["공상과학"]) == [878]

    def test_horror_synonyms(self):
        """'공포' 와 '호러' 모두 27 로 매핑."""
        assert _resolve_genre_ids(["공포"]) == [27]
        assert _resolve_genre_ids(["호러"]) == [27]

    def test_thriller_maps_to_53(self):
        """'스릴러' → 53."""
        assert _resolve_genre_ids(["스릴러"]) == [53]

    def test_romance_synonyms(self):
        """'로맨스', '멜로', '로맨틱' → 10749."""
        for kw in ["로맨스", "멜로", "로맨틱"]:
            assert _resolve_genre_ids([kw]) == [10749], f"failed for: {kw}"

    def test_multi_genre_slash_notation(self):
        """'SF/스릴러' 슬래시 복합 장르 → [878, 53]."""
        result = _resolve_genre_ids(["SF/스릴러"])
        assert 878 in result
        assert 53 in result

    def test_multi_genre_list(self):
        """리스트로 여러 장르 전달 → 각각 매핑."""
        result = _resolve_genre_ids(["액션", "코미디"])
        assert 28 in result   # 액션
        assert 35 in result   # 코미디

    def test_unmapped_genre_returns_empty(self):
        """알 수 없는 장르 → 빈 리스트 (에러 없음)."""
        assert _resolve_genre_ids(["블록버스터어블록"]) == []

    def test_none_input_returns_empty(self):
        """None 입력 → 빈 리스트."""
        assert _resolve_genre_ids(None) == []

    def test_empty_list_returns_empty(self):
        """빈 리스트 입력 → 빈 리스트."""
        assert _resolve_genre_ids([]) == []

    def test_duplicate_genre_ids_deduplicated(self):
        """'SF' + '사이언스픽션' 동의어 중복 → genre_id 1개만."""
        result = _resolve_genre_ids(["SF", "사이언스픽션"])
        assert result.count(878) == 1

    def test_all_19_genres_have_mapping(self):
        """19개 장르 기본 키워드 모두 매핑 가능한지 확인."""
        representative_keywords = [
            "액션", "어드벤처", "애니메이션", "코미디", "범죄",
            "다큐멘터리", "드라마", "가족", "판타지", "역사",
            "공포", "음악", "미스터리", "로맨스", "SF",
            "tv영화", "스릴러", "전쟁", "서부",
        ]
        for kw in representative_keywords:
            result = _resolve_genre_ids([kw])
            assert len(result) >= 1, f"장르 '{kw}' 가 매핑되지 않음"


# ============================================================
# 4. search_external_movies_v2 단위 테스트 (신규)
# ============================================================

# TMDB Discover 가짜 응답 데이터
_FAKE_TMDB_RESULT = [
    {
        "source": "tmdb",
        "external_id": "external_tmdb_12345",
        "title": "듄: 파트 3",
        "original_title": "Dune: Part Three",
        "release_year": 2026,
        "overview": "아라키스의 서사시가 계속된다.",
        "poster_url": "https://image.tmdb.org/t/p/w500/abc.jpg",
        "extra": {"genres_kr": [], "popularity": 95.0, "vote_average": 8.5, "vote_count": 1000},
    }
]

# KOBIS 가짜 응답 데이터
_FAKE_KOBIS_RESULT = [
    {
        "source": "kobis",
        "external_id": "external_kobis_20260301",
        "title": "한국 신작 영화",
        "original_title": "한국 신작 영화",
        "release_year": 2026,
        "overview": "",
        "poster_url": None,
        "extra": {"genres_kr": [], "popularity": 500000.0, "audi_acc": 500000, "rank": 1},
    }
]


class TestSearchExternalMoviesV2:
    """search_external_movies_v2: 소스별 fan-out 동작 검증."""

    @pytest.mark.asyncio
    async def test_sf_2026_calls_tmdb_with_correct_genre(self):
        """SF 2026 글로벌 → TMDB Discover 가 genre_id=878 포함해 호출되는지 검증."""
        tmdb_call_args: dict = {}

        async def mock_tmdb_discover(year_gte, genre_ids, **kwargs):
            tmdb_call_args["year_gte"] = year_gte
            tmdb_call_args["genre_ids"] = genre_ids
            return _FAKE_TMDB_RESULT

        async def mock_kobis(**kwargs):
            return []

        with (
            patch(
                "monglepick.tools.external_movie_search._fetch_tmdb_discover",
                side_effect=mock_tmdb_discover,
            ),
            patch(
                "monglepick.tools.external_movie_search._fetch_kobis_recent_boxoffice",
                side_effect=mock_kobis,
            ),
        ):
            result = await search_external_movies_v2(
                user_intent="SF 신작",
                current_input="올해 SF 신작 뭐 봐야 해?",
                release_year_gte=2026,
                genres=["SF"],
                is_korean_focus=False,
                max_movies=5,
            )

        # TMDB 가 genre_id=878(SF) 로 호출됐는지 검증
        assert 878 in tmdb_call_args.get("genre_ids", [])
        assert tmdb_call_args.get("year_gte") == 2026
        # 결과에 TMDB 영화 포함
        assert len(result) >= 1
        assert result[0]["external_id"] == "external_tmdb_12345"

    @pytest.mark.asyncio
    async def test_korean_focus_calls_kobis_first(self):
        """'한국 신작' → is_korean_focus=True 시 KOBIS 가 호출되는지 검증."""
        kobis_called = False

        async def mock_kobis(days):
            nonlocal kobis_called
            kobis_called = True
            return _FAKE_KOBIS_RESULT

        async def mock_tmdb(**kwargs):
            return []

        async def mock_kmdb(**kwargs):
            return []

        with (
            patch(
                "monglepick.tools.external_movie_search._fetch_kobis_recent_boxoffice",
                side_effect=mock_kobis,
            ),
            patch(
                "monglepick.tools.external_movie_search._fetch_tmdb_discover",
                side_effect=mock_tmdb,
            ),
            patch(
                "monglepick.tools.external_movie_search._fetch_kmdb_search",
                side_effect=mock_kmdb,
            ),
        ):
            result = await search_external_movies_v2(
                user_intent="한국 최신 영화",
                current_input="올해 한국 신작 알려줘",
                release_year_gte=2026,
                genres=[],
                is_korean_focus=True,
                max_movies=5,
            )

        assert kobis_called is True
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_kobis_failure_falls_back_to_tmdb(self):
        """KOBIS 실패해도 TMDB 결과로 부분 fallback."""

        async def mock_kobis(days):
            raise RuntimeError("KOBIS API down")

        async def mock_tmdb(**kwargs):
            return _FAKE_TMDB_RESULT

        with (
            patch(
                "monglepick.tools.external_movie_search._fetch_kobis_recent_boxoffice",
                side_effect=mock_kobis,
            ),
            patch(
                "monglepick.tools.external_movie_search._fetch_tmdb_discover",
                side_effect=mock_tmdb,
            ),
        ):
            result = await search_external_movies_v2(
                user_intent="SF 영화",
                current_input="SF 영화 추천",
                release_year_gte=2026,
                genres=["SF"],
                is_korean_focus=False,
                max_movies=5,
            )

        # KOBIS 실패했지만 TMDB 결과는 살아있어야 함
        assert len(result) >= 1
        assert result[0]["source"] == "tmdb"

    @pytest.mark.asyncio
    async def test_all_sources_empty_returns_empty_list(self):
        """모든 소스가 0건이면 빈 리스트 반환."""

        async def mock_empty(**kwargs):
            return []

        with (
            patch(
                "monglepick.tools.external_movie_search._fetch_tmdb_discover",
                side_effect=mock_empty,
            ),
            patch(
                "monglepick.tools.external_movie_search._fetch_kobis_recent_boxoffice",
                side_effect=mock_empty,
            ),
            patch(
                "monglepick.tools.external_movie_search._fetch_kmdb_search",
                side_effect=mock_empty,
            ),
        ):
            result = await search_external_movies_v2(
                user_intent="",
                current_input="최신 영화",
                release_year_gte=None,
                genres=None,
                is_korean_focus=False,
                max_movies=5,
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_deduplication_removes_same_title_year(self):
        """TMDB + KOBIS 에 동일 제목+연도 영화가 있으면 1편으로 중복 제거."""
        duplicate_movie_tmdb = {
            "source": "tmdb",
            "external_id": "external_tmdb_999",
            "title": "중복영화",
            "original_title": "Duplicate",
            "release_year": 2026,
            "overview": "TMDB 줄거리",
            "poster_url": "https://image.tmdb.org/t/p/w500/dup.jpg",
            "extra": {"popularity": 80.0},
        }
        duplicate_movie_kobis = {
            "source": "kobis",
            "external_id": "external_kobis_DUP",
            "title": "중복영화",
            "original_title": "중복영화",
            "release_year": 2026,
            "overview": "",
            "poster_url": None,
            "extra": {"popularity": 100000.0},
        }

        async def mock_tmdb(**kwargs):
            return [duplicate_movie_tmdb]

        async def mock_kobis(days):
            return [duplicate_movie_kobis]

        # Wikipedia 보강: overview 있는 tmdb 항목은 그대로 반환하는 async mock
        async def mock_enrich(candidate):
            return candidate

        with (
            patch(
                "monglepick.tools.external_movie_search._fetch_tmdb_discover",
                side_effect=mock_tmdb,
            ),
            patch(
                "monglepick.tools.external_movie_search._fetch_kobis_recent_boxoffice",
                side_effect=mock_kobis,
            ),
            # Wikipedia 보강 스킵: 후보를 그대로 반환 (Python 3.12 호환 async mock)
            patch(
                "monglepick.tools.external_movie_search._enrich_with_wikipedia",
                side_effect=mock_enrich,
            ),
        ):
            result = await search_external_movies_v2(
                user_intent="",
                current_input="최신 영화",
                release_year_gte=2026,
                genres=None,
                is_korean_focus=False,
                max_movies=5,
            )

        # 중복이 1편으로 줄어야 함 (dedup 이 tmdb 우선 선택)
        titles = [r["title"] for r in result]
        assert titles.count("중복영화") == 1

    @pytest.mark.asyncio
    async def test_tmdb_api_key_missing_skips_tmdb(self):
        """TMDB_API_KEY 미설정 시 TMDB 소스를 건너뛰고 KOBIS 로만 동작."""

        async def mock_kobis(days):
            return _FAKE_KOBIS_RESULT

        # TMDB API 키를 빈 문자열로 패치
        import monglepick.tools.external_movie_search as esm
        original_key = esm.settings.TMDB_API_KEY

        try:
            esm.settings.TMDB_API_KEY = ""  # type: ignore[attr-defined]
            with patch(
                "monglepick.tools.external_movie_search._fetch_kobis_recent_boxoffice",
                side_effect=mock_kobis,
            ):
                result = await search_external_movies_v2(
                    user_intent="한국 영화",
                    current_input="한국 영화 추천",
                    release_year_gte=2026,
                    genres=None,
                    is_korean_focus=True,
                    max_movies=5,
                )
        finally:
            esm.settings.TMDB_API_KEY = original_key  # type: ignore[attr-defined]

        # KOBIS 결과는 있어야 함
        assert len(result) >= 1
        assert result[0]["source"] == "kobis"


# ============================================================
# 5. external_search_node 통합 테스트 (v2 스키마)
# ============================================================

class TestExternalSearchNodeV2:
    """external_search_node: v2 스키마 기반 RankedMovie 변환 통합 검증."""

    @pytest.mark.asyncio
    async def test_sf_2026_produces_external_tmdb_id(self):
        """genre_preference='SF', release_year_gte=2026 → ranked_movies[0].id 가 'external_tmdb_' 시작."""
        prefs = ExtractedPreferences(
            user_intent="SF 신작 추천",
            genre_preference="SF",
            dynamic_filters=[
                FilterCondition(field="release_year", operator="gte", value="2026"),
            ],
        )
        state = {
            "preferences": prefs,
            "current_input": "올해 SF 신작 뭐 봐야 해?",
            "session_id": "s1",
            "user_id": "u1",
        }

        async def mock_v2(**kwargs):
            return _FAKE_TMDB_RESULT

        with (
            patch(
                "monglepick.agents.chat.nodes.search_external_movies_v2",
                side_effect=mock_v2,
            ),
            # now_showing 어노테이션은 스킵
            patch(
                "monglepick.agents.chat.now_showing_cache.annotate_movies",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await external_search_node(state)

        ranked = result["ranked_movies"]
        assert len(ranked) == 1
        assert ranked[0].id.startswith("external_tmdb_")
        assert ranked[0].title == "듄: 파트 3"
        assert ranked[0].release_year == 2026
        assert ranked[0].rank == 1
        # TMDB 출처 explanation 문구
        assert "TMDB" in ranked[0].explanation

    @pytest.mark.asyncio
    async def test_kobis_source_produces_correct_explanation(self):
        """source='kobis' 영화 → explanation 에 '박스오피스' 포함."""
        prefs = ExtractedPreferences(user_intent="한국 영화")
        state = {
            "preferences": prefs,
            "current_input": "한국 최신 영화",
            "session_id": "s2",
            "user_id": "u2",
        }

        async def mock_v2(**kwargs):
            return _FAKE_KOBIS_RESULT

        with (
            patch(
                "monglepick.agents.chat.nodes.search_external_movies_v2",
                side_effect=mock_v2,
            ),
            patch(
                "monglepick.agents.chat.now_showing_cache.annotate_movies",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await external_search_node(state)

        ranked = result["ranked_movies"]
        assert len(ranked) == 1
        assert ranked[0].id.startswith("external_kobis_")
        assert "박스오피스" in ranked[0].explanation

    @pytest.mark.asyncio
    async def test_kmdb_source_produces_correct_explanation(self):
        """source='kmdb' 영화 → explanation 에 '한국영상자료원' 포함."""
        fake_kmdb = [
            {
                "source": "kmdb",
                "external_id": "external_kmdb_K12345",
                "title": "한국 독립영화",
                "original_title": "Korean Indie",
                "release_year": 2026,
                "overview": "독립 영화 줄거리",
                "poster_url": None,
                "extra": {"genres_kr": ["드라마"], "popularity": 2026.0, "director": "홍길동"},
            }
        ]

        prefs = ExtractedPreferences(user_intent="한국 독립영화")
        state = {
            "preferences": prefs,
            "current_input": "한국 독립영화 추천",
            "session_id": "s3",
            "user_id": "u3",
        }

        async def mock_v2(**kwargs):
            return fake_kmdb

        with (
            patch(
                "monglepick.agents.chat.nodes.search_external_movies_v2",
                side_effect=mock_v2,
            ),
            patch(
                "monglepick.agents.chat.now_showing_cache.annotate_movies",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await external_search_node(state)

        ranked = result["ranked_movies"]
        assert len(ranked) == 1
        assert ranked[0].id.startswith("external_kmdb_")
        assert "한국영상자료원" in ranked[0].explanation

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_ranked(self):
        """v2 가 빈 리스트 반환 → ranked_movies=[]."""
        state = {
            "preferences": ExtractedPreferences(user_intent=""),
            "current_input": "최신 영화",
            "session_id": "s4",
            "user_id": "u4",
        }

        async def mock_v2(**kwargs):
            return []

        with patch(
            "monglepick.agents.chat.nodes.search_external_movies_v2",
            side_effect=mock_v2,
        ):
            result = await external_search_node(state)

        assert result["ranked_movies"] == []

    @pytest.mark.asyncio
    async def test_v2_exception_returns_empty_gracefully(self):
        """search_external_movies_v2 가 예외를 던져도 ranked_movies=[] graceful."""
        state = {
            "preferences": ExtractedPreferences(),
            "current_input": "최신 영화",
            "session_id": "s5",
            "user_id": "u5",
        }

        async def raise_exc(**kwargs):
            raise RuntimeError("all sources down")

        with patch(
            "monglepick.agents.chat.nodes.search_external_movies_v2",
            side_effect=raise_exc,
        ):
            result = await external_search_node(state)

        assert result == {"ranked_movies": []}

    @pytest.mark.asyncio
    async def test_release_year_extracted_from_dynamic_filters(self):
        """dynamic_filters 의 release_year 하한이 v2 함수에 전달된다."""
        prefs = ExtractedPreferences(
            dynamic_filters=[
                FilterCondition(field="release_year", operator="gte", value="2026"),
            ],
        )
        state = {
            "preferences": prefs,
            "current_input": "추천",
            "session_id": "s6",
            "user_id": "u6",
        }

        captured: dict = {}

        async def capture_v2(**kwargs):
            captured.update(kwargs)
            return []

        with patch(
            "monglepick.agents.chat.nodes.search_external_movies_v2",
            side_effect=capture_v2,
        ):
            await external_search_node(state)

        assert captured.get("release_year_gte") == 2026

    @pytest.mark.asyncio
    async def test_genre_preference_passed_to_v2(self):
        """preferences.genre_preference 가 genres 리스트로 v2 에 전달된다."""
        prefs = ExtractedPreferences(
            genre_preference="SF",
            user_intent="SF 영화 추천",
        )
        state = {
            "preferences": prefs,
            "current_input": "SF 영화 추천",
            "session_id": "s7",
            "user_id": "u7",
        }

        captured: dict = {}

        async def capture_v2(**kwargs):
            captured.update(kwargs)
            return []

        with patch(
            "monglepick.agents.chat.nodes.search_external_movies_v2",
            side_effect=capture_v2,
        ):
            await external_search_node(state)

        # genres 파라미터에 "SF" 가 포함되어야 함
        assert "SF" in captured.get("genres", [])

    @pytest.mark.asyncio
    async def test_korean_focus_detected_from_input(self):
        """'한국' 키워드가 current_input 에 있으면 is_korean_focus=True 로 전달."""
        prefs = ExtractedPreferences(user_intent="한국 영화")
        state = {
            "preferences": prefs,
            "current_input": "한국 영화 추천해줘",
            "session_id": "s8",
            "user_id": "u8",
        }

        captured: dict = {}

        async def capture_v2(**kwargs):
            captured.update(kwargs)
            return []

        with patch(
            "monglepick.agents.chat.nodes.search_external_movies_v2",
            side_effect=capture_v2,
        ):
            await external_search_node(state)

        assert captured.get("is_korean_focus") is True

    @pytest.mark.asyncio
    async def test_poster_path_set_from_tmdb_poster_url(self):
        """TMDB 결과의 poster_url 이 RankedMovie.poster_path 에 올바르게 복사된다."""
        fake_with_poster = [
            {
                "source": "tmdb",
                "external_id": "external_tmdb_777",
                "title": "포스터 있는 영화",
                "original_title": "Movie With Poster",
                "release_year": 2026,
                "overview": "줄거리",
                "poster_url": "https://image.tmdb.org/t/p/w500/poster.jpg",
                "extra": {"popularity": 50.0},
            }
        ]

        state = {
            "preferences": ExtractedPreferences(user_intent="최신 영화"),
            "current_input": "최신 영화",
            "session_id": "s9",
            "user_id": "u9",
        }

        async def mock_v2(**kwargs):
            return fake_with_poster

        with (
            patch(
                "monglepick.agents.chat.nodes.search_external_movies_v2",
                side_effect=mock_v2,
            ),
            patch(
                "monglepick.agents.chat.now_showing_cache.annotate_movies",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await external_search_node(state)

        ranked = result["ranked_movies"]
        assert ranked[0].poster_path == "https://image.tmdb.org/t/p/w500/poster.jpg"

    @pytest.mark.asyncio
    async def test_recent_recommended_ids_updated(self):
        """외부 검색 결과 ID 가 recent_recommended_ids 롤링 윈도우에 추가된다."""
        prefs = ExtractedPreferences(user_intent="최신 영화")
        state = {
            "preferences": prefs,
            "current_input": "최신 영화",
            "session_id": "s10",
            "user_id": "u10",
            "recent_recommended_ids": [],
        }

        async def mock_v2(**kwargs):
            return _FAKE_TMDB_RESULT

        with (
            patch(
                "monglepick.agents.chat.nodes.search_external_movies_v2",
                side_effect=mock_v2,
            ),
            patch(
                "monglepick.agents.chat.now_showing_cache.annotate_movies",
                new=AsyncMock(return_value=None),
            ),
        ):
            result = await external_search_node(state)

        updated_ids = result.get("recent_recommended_ids", [])
        assert "external_tmdb_12345" in updated_ids
