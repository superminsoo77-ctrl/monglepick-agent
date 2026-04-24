"""
추천 엔진 서브그래프 노드 단위 테스트 (Phase 4).

7개 노드 함수를 개별적으로 테스트한다.
모든 테스트는 mock_redis_cf fixture를 사용하여 Redis 서버 없이 실행된다.
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import (
    CandidateMovie,
    EmotionResult,
    ExtractedPreferences,
    RankedMovie,
    ScoreDetail,
)
from monglepick.agents.recommendation.models import RecommendationEngineState
from monglepick.agents.recommendation.nodes import (
    _extract_crew_frequency,
    _extract_liked_genres,
    _find_similar_watched,
    _jaccard,
    _min_max_normalize,
    cold_start_checker,
    collaborative_filter,
    content_based_filter,
    diversity_reranker,
    hybrid_merger,
    popularity_fallback,
    score_finalizer,
)


# ============================================================
# 테스트 헬퍼: 후보 영화 생성
# ============================================================

def _make_candidate(
    id: str = "1",
    title: str = "테스트 영화",
    genres: list[str] | None = None,
    director: str = "",
    cast: list[str] | None = None,
    rating: float = 7.0,
    mood_tags: list[str] | None = None,
    rrf_score: float = 0.5,
    release_year: int = 2020,
) -> CandidateMovie:
    """테스트용 CandidateMovie를 생성한다."""
    return CandidateMovie(
        id=id,
        title=title,
        genres=genres or ["드라마"],
        director=director,
        cast=cast or [],
        rating=rating,
        mood_tags=mood_tags or [],
        rrf_score=rrf_score,
        release_year=release_year,
    )


def _make_candidates() -> list[CandidateMovie]:
    """다양한 장르의 테스트 후보 영화 5편을 생성한다."""
    return [
        _make_candidate(
            id="1", title="인터스텔라", genres=["SF", "드라마"],
            director="놀란", cast=["매튜"], rating=8.7,
            mood_tags=["웅장", "감동"], rrf_score=0.95,
        ),
        _make_candidate(
            id="2", title="기생충", genres=["드라마", "스릴러"],
            director="봉준호", cast=["송강호"], rating=8.5,
            mood_tags=["사회비판", "다크"], rrf_score=0.90,
        ),
        _make_candidate(
            id="3", title="어벤져스", genres=["액션", "SF"],
            director="루소", cast=["로버트"], rating=8.0,
            mood_tags=["모험", "웅장"], rrf_score=0.85,
        ),
        _make_candidate(
            id="4", title="라라랜드", genres=["로맨스", "뮤지컬"],
            director="차젤", cast=["라이언"], rating=7.9,
            mood_tags=["로맨틱", "감동"], rrf_score=0.80,
        ),
        _make_candidate(
            id="5", title="겟아웃", genres=["공포", "스릴러"],
            director="필", cast=["다니엘"], rating=7.5,
            mood_tags=["스릴", "반전"], rrf_score=0.75,
        ),
    ]


# ============================================================
# 유틸리티 함수 테스트
# ============================================================

class TestJaccard:
    """Jaccard 유사도 함수 테스트."""

    def test_identical_sets(self):
        """동일한 집합이면 1.0을 반환한다."""
        assert _jaccard({"SF", "드라마"}, {"SF", "드라마"}) == 1.0

    def test_disjoint_sets(self):
        """겹치지 않는 집합이면 0.0을 반환한다."""
        assert _jaccard({"SF"}, {"로맨스"}) == 0.0

    def test_partial_overlap(self):
        """부분적으로 겹치면 올바른 Jaccard를 반환한다."""
        # |{SF} ∩ {SF, 드라마}| / |{SF} ∪ {SF, 드라마}| = 1/2
        assert _jaccard({"SF"}, {"SF", "드라마"}) == 0.5

    def test_empty_sets(self):
        """빈 집합이면 0.0을 반환한다."""
        assert _jaccard(set(), set()) == 0.0

    def test_one_empty(self):
        """한쪽이 빈 집합이면 0.0을 반환한다."""
        assert _jaccard({"SF"}, set()) == 0.0


class TestMinMaxNormalize:
    """min-max 정규화 테스트."""

    def test_normal(self):
        """정상적인 정규화를 수행한다."""
        scores = {"a": 1.0, "b": 3.0, "c": 5.0}
        result = _min_max_normalize(scores)
        assert result["a"] == 0.0
        assert result["c"] == 1.0
        assert abs(result["b"] - 0.5) < 1e-6

    def test_all_same(self):
        """모든 값이 동일하면 0.5를 반환한다."""
        scores = {"a": 2.0, "b": 2.0}
        result = _min_max_normalize(scores)
        assert result["a"] == 0.5
        assert result["b"] == 0.5

    def test_empty(self):
        """빈 딕셔너리를 반환한다."""
        assert _min_max_normalize({}) == {}


class TestExtractLikedGenres:
    """선호 장르 추출 테스트."""

    def test_with_list_genres(self):
        """genres가 list인 시청 이력에서 장르를 추출한다."""
        history = [
            {"genres": ["SF", "드라마"]},
            {"genres": ["SF", "액션"]},
            {"genres": ["드라마"]},
        ]
        result = _extract_liked_genres(history, top_k=2)
        # SF: 2, 드라마: 2, 액션: 1 → top 2 = {SF, 드라마}
        assert "SF" in result
        assert "드라마" in result

    def test_empty_history(self):
        """빈 시청 이력이면 빈 set을 반환한다."""
        assert _extract_liked_genres([], top_k=5) == set()

    def test_no_genres_field(self):
        """genres 필드가 없으면 빈 set을 반환한다."""
        history = [{"title": "테스트"}]
        assert _extract_liked_genres(history) == set()


# ============================================================
# 1. cold_start_checker 테스트
# ============================================================

class TestColdStartChecker:
    """Cold Start 판정 노드 테스트."""

    @pytest.mark.asyncio
    async def test_no_history_is_cold_start(self):
        """시청 이력 0편이면 Cold Start이다."""
        state: RecommendationEngineState = {"watch_history": []}
        result = await cold_start_checker(state)
        assert result["is_cold_start"] is True

    @pytest.mark.asyncio
    async def test_few_history_is_cold_start(self):
        """시청 이력 4편이면 Cold Start이다 (< 5편)."""
        state: RecommendationEngineState = {
            "watch_history": [{"movie_id": str(i)} for i in range(4)]
        }
        result = await cold_start_checker(state)
        assert result["is_cold_start"] is True

    @pytest.mark.asyncio
    async def test_enough_history_not_cold_start(self):
        """시청 이력 5편이면 Cold Start가 아니다."""
        state: RecommendationEngineState = {
            "watch_history": [{"movie_id": str(i)} for i in range(5)]
        }
        result = await cold_start_checker(state)
        assert result["is_cold_start"] is False

    @pytest.mark.asyncio
    async def test_none_history_is_cold_start(self):
        """watch_history가 없으면 Cold Start로 간주한다."""
        state: RecommendationEngineState = {}
        result = await cold_start_checker(state)
        assert result["is_cold_start"] is True


# ============================================================
# 2. collaborative_filter 테스트
# ============================================================

class TestCollaborativeFilter:
    """협업 필터링(CF) 점수 계산 노드 테스트."""

    @pytest.mark.asyncio
    async def test_cf_score_calculation(self, mock_redis_cf):
        """유사 유저가 평가한 영화의 CF 점수를 계산한다."""
        # 유사 유저 설정
        mock_redis_cf.set_similar_users("user1", [("user2", 0.9), ("user3", 0.8)])
        # user2: movie1=4.0, user3: movie1=5.0
        mock_redis_cf.set_user_ratings("user2", {"1": "4.0"})
        mock_redis_cf.set_user_ratings("user3", {"1": "5.0"})

        state: RecommendationEngineState = {
            "user_id": "user1",
            "candidate_movies": [_make_candidate(id="1")],
        }
        result = await collaborative_filter(state)

        cf_scores = result["cf_scores"]
        assert "1" in cf_scores
        # CF = (0.9*4.0 + 0.8*5.0) / (0.9 + 0.8) = 7.6 / 1.7 ≈ 4.47
        # min-max 정규화: 단일 영화이면 0.5
        assert cf_scores["1"] == 0.5  # 단일 영화 → 동일 값 → 0.5

    @pytest.mark.asyncio
    async def test_cf_anonymous_user(self, mock_redis_cf):
        """익명 사용자(user_id 없음)면 기본값 0.5를 반환한다."""
        state: RecommendationEngineState = {
            "user_id": "",
            "candidate_movies": [_make_candidate(id="1")],
        }
        result = await collaborative_filter(state)
        assert result["cf_scores"]["1"] == 0.5

    @pytest.mark.asyncio
    async def test_cf_cache_miss(self, mock_redis_cf):
        """Redis에 유사 유저가 없으면 기본값 0.5를 반환한다."""
        # 유사 유저를 설정하지 않음 (캐시 미스)
        state: RecommendationEngineState = {
            "user_id": "unknown_user",
            "candidate_movies": [_make_candidate(id="1")],
        }
        result = await collaborative_filter(state)
        assert result["cf_scores"]["1"] == 0.5

    @pytest.mark.asyncio
    async def test_cf_no_rated_movies(self, mock_redis_cf):
        """유사 유저가 후보 영화를 평가하지 않았으면 CF 점수 0 → 기본값 0.5."""
        mock_redis_cf.set_similar_users("user1", [("user2", 0.9)])
        # user2는 movie99만 평가 (후보가 아님)
        mock_redis_cf.set_user_ratings("user2", {"99": "5.0"})

        state: RecommendationEngineState = {
            "user_id": "user1",
            "candidate_movies": [_make_candidate(id="1")],
        }
        result = await collaborative_filter(state)
        # 모든 CF 점수가 0 → all-zero fallback → 기본값 0.5
        assert result["cf_scores"]["1"] == 0.5

    @pytest.mark.asyncio
    async def test_cf_empty_candidates(self, mock_redis_cf):
        """후보 영화가 없으면 빈 딕셔너리를 반환한다."""
        state: RecommendationEngineState = {
            "user_id": "user1",
            "candidate_movies": [],
        }
        result = await collaborative_filter(state)
        assert result["cf_scores"] == {}

    @pytest.mark.asyncio
    async def test_cf_multiple_candidates(self, mock_redis_cf):
        """여러 후보 영화에 대해 서로 다른 CF 점수를 계산한다."""
        mock_redis_cf.set_similar_users("user1", [("user2", 0.9)])
        # user2: movie1=5.0, movie2=2.0
        mock_redis_cf.set_user_ratings("user2", {"1": "5.0", "2": "2.0"})

        candidates = [_make_candidate(id="1"), _make_candidate(id="2")]
        state: RecommendationEngineState = {
            "user_id": "user1",
            "candidate_movies": candidates,
        }
        result = await collaborative_filter(state)

        cf_scores = result["cf_scores"]
        # movie1: 5.0이 더 높으므로 정규화 후 1.0
        # movie2: 2.0이 더 낮으므로 정규화 후 0.0
        assert cf_scores["1"] > cf_scores["2"]


# ============================================================
# 3. content_based_filter 테스트
# ============================================================

class TestContentBasedFilter:
    """컨텐츠 기반 필터링(CBF) 점수 계산 노드 테스트."""

    @pytest.mark.asyncio
    async def test_cbf_with_emotion(self):
        """감정이 있으면 무드 매칭을 포함한 CBF 점수를 계산한다."""
        candidates = [
            _make_candidate(
                id="1", genres=["SF"], mood_tags=["웅장", "감동"], rrf_score=0.9,
            ),
        ]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "watch_history": [{"genres": ["SF"], "title": "테스트"}],
            "emotion": EmotionResult(emotion="excited", mood_tags=["웅장"]),
            "mood_tags": ["웅장"],
            "preferences": ExtractedPreferences(genre_preference="SF"),
        }
        result = await content_based_filter(state)
        assert "1" in result["cbf_scores"]
        # 장르 일치 + 무드 매칭 + 키워드 매칭 + RRF → 높은 점수
        assert result["cbf_scores"]["1"] == 0.5  # 단일 영화 → 정규화 0.5

    @pytest.mark.asyncio
    async def test_cbf_without_emotion(self):
        """감정이 없으면 무드 가중치 0으로 CBF 점수를 계산한다."""
        candidates = [
            _make_candidate(id="1", genres=["SF"], mood_tags=["웅장"]),
        ]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "watch_history": [],
            "emotion": None,
            "mood_tags": [],
            "preferences": None,
        }
        result = await content_based_filter(state)
        assert "1" in result["cbf_scores"]

    @pytest.mark.asyncio
    async def test_cbf_empty_candidates(self):
        """후보 영화가 없으면 빈 딕셔너리를 반환한다."""
        state: RecommendationEngineState = {
            "candidate_movies": [],
            "watch_history": [],
            "emotion": None,
            "mood_tags": [],
            "preferences": None,
        }
        result = await content_based_filter(state)
        assert result["cbf_scores"] == {}

    @pytest.mark.asyncio
    async def test_cbf_genre_match_higher(self):
        """장르가 일치하는 영화가 더 높은 CBF 점수를 받는다."""
        candidates = [
            _make_candidate(id="1", genres=["SF", "드라마"], rrf_score=0.5),
            _make_candidate(id="2", genres=["로맨스"], rrf_score=0.5),
        ]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "watch_history": [
                {"genres": ["SF"], "title": "영화1"},
                {"genres": ["SF"], "title": "영화2"},
            ],
            "emotion": None,
            "mood_tags": [],
            "preferences": ExtractedPreferences(genre_preference="SF"),
        }
        result = await content_based_filter(state)
        # SF 장르 일치하는 movie1이 더 높은 점수
        assert result["cbf_scores"]["1"] > result["cbf_scores"]["2"]


# ============================================================
# 4. hybrid_merger 테스트
# ============================================================

class TestHybridMerger:
    """CF+CBF 가중 합산 노드 테스트."""

    @pytest.mark.asyncio
    async def test_heavy_user_no_emotion(self):
        """시청 30편+ & 감정 없음: CF 0.60, CBF 0.40."""
        candidates = [_make_candidate(id="1")]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {"1": 0.8},
            "cbf_scores": {"1": 0.6},
            "watch_history": [{"movie_id": str(i)} for i in range(35)],
            "emotion": None,
        }
        result = await hybrid_merger(state)
        # 0.60 * 0.8 + 0.40 * 0.6 = 0.48 + 0.24 = 0.72
        assert abs(result["hybrid_scores"]["1"] - 0.72) < 1e-6

    @pytest.mark.asyncio
    async def test_heavy_user_with_emotion(self):
        """시청 30편+ & 감정 있음: CF 0.50, CBF 0.50."""
        candidates = [_make_candidate(id="1")]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {"1": 0.8},
            "cbf_scores": {"1": 0.6},
            "watch_history": [{"movie_id": str(i)} for i in range(35)],
            "emotion": EmotionResult(emotion="happy", mood_tags=["유쾌"]),
        }
        result = await hybrid_merger(state)
        # 0.50 * 0.8 + 0.50 * 0.6 = 0.40 + 0.30 = 0.70
        assert abs(result["hybrid_scores"]["1"] - 0.70) < 1e-6

    @pytest.mark.asyncio
    async def test_warm_user_with_emotion(self):
        """시청 5~29편 & 감정 있음: CF 0.30, CBF 0.70."""
        candidates = [_make_candidate(id="1")]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {"1": 0.8},
            "cbf_scores": {"1": 0.6},
            "watch_history": [{"movie_id": str(i)} for i in range(10)],
            "emotion": EmotionResult(emotion="sad", mood_tags=["힐링"]),
        }
        result = await hybrid_merger(state)
        # 0.30 * 0.8 + 0.70 * 0.6 = 0.24 + 0.42 = 0.66
        assert abs(result["hybrid_scores"]["1"] - 0.66) < 1e-6

    @pytest.mark.asyncio
    async def test_cf_cache_miss_cbf_only(self):
        """CF 캐시 미스(명시적 플래그)면 CBF에 전적 의존한다."""
        candidates = [_make_candidate(id="1")]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {"1": 0.5},
            "cf_cache_miss": True,  # 명시적 캐시 미스 플래그
            "cbf_scores": {"1": 0.9},
            "watch_history": [{"movie_id": str(i)} for i in range(10)],
            "emotion": None,
        }
        result = await hybrid_merger(state)
        # w_cf=0.0, w_cbf=1.0 → hybrid = 0.0 * 0.5 + 1.0 * 0.9 = 0.9
        assert abs(result["hybrid_scores"]["1"] - 0.9) < 1e-6

    @pytest.mark.asyncio
    async def test_both_zero_rrf_fallback(self):
        """CF 캐시 미스 & CBF 전부 0이면 RRF 점수를 사용한다."""
        candidates = [_make_candidate(id="1", rrf_score=0.8)]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {"1": 0.5},
            "cf_cache_miss": True,  # 명시적 캐시 미스 플래그
            "cbf_scores": {"1": 0.0},
            "watch_history": [],
            "emotion": None,
        }
        result = await hybrid_merger(state)
        # w_cf=0, w_cbf=0 → RRF 점수 / max_rrf = 0.8 / 0.8 = 1.0
        assert abs(result["hybrid_scores"]["1"] - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_empty_candidates(self):
        """후보 영화가 없으면 빈 딕셔너리를 반환한다."""
        state: RecommendationEngineState = {
            "candidate_movies": [],
            "cf_scores": {},
            "cbf_scores": {},
            "watch_history": [],
            "emotion": None,
        }
        result = await hybrid_merger(state)
        assert result["hybrid_scores"] == {}


# ============================================================
# 5. popularity_fallback 테스트
# ============================================================

class TestPopularityFallback:
    """Cold Start 인기도 기반 점수 계산 노드 테스트."""

    @pytest.mark.asyncio
    async def test_basic_popularity(self):
        """rating 기반 기본 인기도 점수를 계산한다."""
        candidates = [
            _make_candidate(id="1", rating=8.0, rrf_score=0.5),
            _make_candidate(id="2", rating=6.0, rrf_score=0.3),
        ]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "preferences": None,
            "mood_tags": [],
        }
        result = await popularity_fallback(state)
        # rating 높은 movie1이 더 높은 점수
        assert result["hybrid_scores"]["1"] > result["hybrid_scores"]["2"]
        # CF/CBF 점수는 0.0 (Cold Start)
        assert result["cf_scores"]["1"] == 0.0
        assert result["cbf_scores"]["1"] == 0.0

    @pytest.mark.asyncio
    async def test_genre_boost(self):
        """선호 장르가 일치하면 부스트를 받는다."""
        candidates = [
            _make_candidate(id="1", rating=7.0, genres=["SF"], rrf_score=0.5),
            _make_candidate(id="2", rating=7.0, genres=["로맨스"], rrf_score=0.5),
        ]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "preferences": ExtractedPreferences(genre_preference="SF"),
            "mood_tags": [],
        }
        result = await popularity_fallback(state)
        # SF 장르가 일치하는 movie1이 부스트를 받아 더 높음
        assert result["hybrid_scores"]["1"] > result["hybrid_scores"]["2"]

    @pytest.mark.asyncio
    async def test_mood_boost(self):
        """무드 태그가 일치하면 부스트를 받는다."""
        candidates = [
            _make_candidate(id="1", rating=7.0, mood_tags=["웅장"], rrf_score=0.5),
            _make_candidate(id="2", rating=7.0, mood_tags=["잔잔"], rrf_score=0.5),
        ]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "preferences": None,
            "mood_tags": ["웅장"],
        }
        result = await popularity_fallback(state)
        assert result["hybrid_scores"]["1"] > result["hybrid_scores"]["2"]

    @pytest.mark.asyncio
    async def test_empty_candidates(self):
        """후보 영화가 없으면 빈 결과를 반환한다."""
        state: RecommendationEngineState = {
            "candidate_movies": [],
            "preferences": None,
            "mood_tags": [],
        }
        result = await popularity_fallback(state)
        assert result["hybrid_scores"] == {}
        assert result["cf_scores"] == {}
        assert result["cbf_scores"] == {}


# ============================================================
# 6. diversity_reranker 테스트
# ============================================================

class TestDiversityReranker:
    """MMR 다양성 재정렬 노드 테스트."""

    @pytest.mark.asyncio
    async def test_mmr_diversifies_genres(self):
        """MMR이 같은 장르 영화를 뒤로 밀어 다양성을 높인다."""
        candidates = [
            _make_candidate(id="1", genres=["SF"], rrf_score=0.9),
            _make_candidate(id="2", genres=["SF"], rrf_score=0.85),
            _make_candidate(id="3", genres=["로맨스"], rrf_score=0.80),
        ]
        hybrid_scores = {"1": 0.9, "2": 0.85, "3": 0.80}

        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "hybrid_scores": hybrid_scores,
        }
        result = await diversity_reranker(state)

        reranked = result["candidate_movies"]
        assert len(reranked) == 3
        # 첫 번째: 최고 점수 movie1 (SF)
        assert reranked[0].id == "1"
        # 두 번째: movie3 (로맨스)가 movie2 (SF)보다 앞에 (다양성)
        # MMR: movie2 = 0.7*0.85 - 0.3*1.0 = 0.295
        # MMR: movie3 = 0.7*0.80 - 0.3*0.0 = 0.56
        assert reranked[1].id == "3"

    @pytest.mark.asyncio
    async def test_mmr_less_than_top_k(self):
        """후보가 TOP_K 미만이면 있는 만큼만 선택한다."""
        candidates = [_make_candidate(id="1"), _make_candidate(id="2")]
        hybrid_scores = {"1": 0.9, "2": 0.5}

        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "hybrid_scores": hybrid_scores,
        }
        result = await diversity_reranker(state)
        assert len(result["candidate_movies"]) == 2

    @pytest.mark.asyncio
    async def test_mmr_empty_candidates(self):
        """후보가 없으면 빈 리스트를 반환한다."""
        state: RecommendationEngineState = {
            "candidate_movies": [],
            "hybrid_scores": {},
        }
        result = await diversity_reranker(state)
        assert result["candidate_movies"] == []

    @pytest.mark.asyncio
    async def test_mmr_single_candidate(self):
        """후보가 1편이면 그대로 반환한다."""
        candidates = [_make_candidate(id="1")]
        hybrid_scores = {"1": 0.9}

        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "hybrid_scores": hybrid_scores,
        }
        result = await diversity_reranker(state)
        assert len(result["candidate_movies"]) == 1
        assert result["candidate_movies"][0].id == "1"

    # ── Popular / Hidden gem slot quota 동작 검증 (2026-04-15 신규) ──

    @pytest.mark.asyncio
    async def test_slot_quota_popular_wins_top_slot(self):
        """평점 0.0 무명작이 rrf_score 1.0 이어도 1순위를 차지하지 못한다.

        BM25 제목매칭으로 무명작이 hybrid_score=1.0 을 받은 과거 버그를 재현 차단.
        """
        candidates = [
            # hidden 후보 — rrf 최상위지만 평점 0.0, vote_count 부재
            _make_candidate(id="hid1", title="무명인디", rating=0.0, rrf_score=1.0),
            # popular 후보 — 평점 7.0 (>= POPULAR_MIN_RATING=5.0)
            _make_candidate(id="pop1", title="검증된영화A", rating=7.5, rrf_score=0.5),
            _make_candidate(id="pop2", title="검증된영화B", rating=6.5, rrf_score=0.4),
        ]
        hybrid_scores = {"hid1": 1.0, "pop1": 0.5, "pop2": 0.4}

        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "hybrid_scores": hybrid_scores,
        }
        result = await diversity_reranker(state)
        reranked = result["candidate_movies"]

        # 1순위는 반드시 popular 풀에서 나와야 함 (hidden 은 뒤로 밀림)
        assert reranked[0].id in {"pop1", "pop2"}
        # hidden 영화는 HIDDEN_SLOTS 범위 안에 들어가되 최상위는 아님
        ids = [m.id for m in reranked]
        assert "hid1" in ids
        assert ids.index("hid1") > 0

    @pytest.mark.asyncio
    async def test_slot_quota_hidden_gem_included(self):
        """popular 영화가 충분해도 hidden gem 풀에서 HIDDEN_SLOTS 개가 포함된다.

        사용자 요구: "마지막 1~2개 정도는 무명작도 추천받아도 OK". popular 가 5편 이상이어도
        hidden 이 있으면 최소 HIDDEN_SLOTS(=2) 편 확보되어야 한다.
        """
        candidates = [
            _make_candidate(id=f"pop{i}", rating=7.0 + i * 0.1, rrf_score=0.9 - i * 0.05)
            for i in range(5)
        ]
        candidates.extend([
            _make_candidate(id="hid1", title="숨은보석1", rating=0.0, rrf_score=0.3),
            _make_candidate(id="hid2", title="숨은보석2", rating=0.0, rrf_score=0.25),
        ])
        hybrid_scores = {c.id: c.rrf_score for c in candidates}

        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "hybrid_scores": hybrid_scores,
        }
        result = await diversity_reranker(state)
        selected_ids = {m.id for m in result["candidate_movies"]}

        # TOP_K=5 만큼 선택, hidden 2편 모두 포함
        assert len(selected_ids) == 5
        assert "hid1" in selected_ids
        assert "hid2" in selected_ids

    @pytest.mark.asyncio
    async def test_slot_quota_fallback_when_pool_empty(self):
        """popular 풀이 비면 hidden 만으로 TOP_K 까지 채운다 (fallback).

        반대 케이스(hidden 부재)도 popular 로 채워져야 한다. 이 테스트는 전자 검증.
        """
        candidates = [
            _make_candidate(id=f"hid{i}", rating=0.0, rrf_score=0.5 - i * 0.05)
            for i in range(6)
        ]
        hybrid_scores = {c.id: c.rrf_score for c in candidates}

        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "hybrid_scores": hybrid_scores,
        }
        result = await diversity_reranker(state)
        # popular 풀이 비었어도 hidden 에서 TOP_K(5)편 전부 채워짐
        assert len(result["candidate_movies"]) == 5


# ============================================================
# 7. score_finalizer 테스트
# ============================================================

class TestScoreFinalizer:
    """최종 점수 첨부 + RankedMovie 변환 노드 테스트."""

    @pytest.mark.asyncio
    async def test_ranked_movie_conversion(self):
        """CandidateMovie를 RankedMovie로 올바르게 변환한다."""
        candidates = [_make_candidate(id="1", title="인터스텔라")]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {"1": 0.7},
            "cbf_scores": {"1": 0.8},
            "hybrid_scores": {"1": 0.75},
            "watch_history": [],
            "mood_tags": [],
        }
        result = await score_finalizer(state)

        ranked = result["ranked_movies"]
        assert len(ranked) == 1
        assert isinstance(ranked[0], RankedMovie)
        assert ranked[0].title == "인터스텔라"
        assert ranked[0].rank == 1
        assert ranked[0].score_detail.cf_score == 0.7
        assert ranked[0].score_detail.cbf_score == 0.8
        assert ranked[0].score_detail.hybrid_score == 0.75

    @pytest.mark.asyncio
    async def test_genre_match_calculated(self):
        """장르 일치도가 올바르게 계산된다."""
        candidates = [_make_candidate(id="1", genres=["SF", "드라마"])]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {"1": 0.5},
            "cbf_scores": {"1": 0.5},
            "hybrid_scores": {"1": 0.5},
            "watch_history": [
                {"genres": ["SF"], "title": "영화1"},
                {"genres": ["SF"], "title": "영화2"},
            ],
            "mood_tags": [],
        }
        result = await score_finalizer(state)
        # liked_genres = {SF}, movie.genres = {SF, 드라마}
        # Jaccard = 1/2 = 0.5
        assert result["ranked_movies"][0].score_detail.genre_match == 0.5

    @pytest.mark.asyncio
    async def test_mood_match_calculated(self):
        """무드 일치도가 올바르게 계산된다."""
        candidates = [_make_candidate(id="1", mood_tags=["웅장", "감동"])]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {"1": 0.5},
            "cbf_scores": {"1": 0.5},
            "hybrid_scores": {"1": 0.5},
            "watch_history": [],
            "mood_tags": ["웅장"],
        }
        result = await score_finalizer(state)
        # movie_moods = {웅장, 감동}, user_moods = {웅장}
        # 수정된 공식: len(교집합) / len(user_moods) = |{웅장}| / max(1, 1) = 1/1 = 1.0
        # (기존: len(교집합) / len(movie_moods) = 1/2 = 0.5 — 다무드 영화 불이익 해소)
        assert result["ranked_movies"][0].score_detail.mood_match == 1.0

    @pytest.mark.asyncio
    async def test_similar_to_from_history(self):
        """시청 이력에서 같은 장르인 유사 영화를 찾는다."""
        candidates = [_make_candidate(id="1", genres=["SF"])]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {"1": 0.5},
            "cbf_scores": {"1": 0.5},
            "hybrid_scores": {"1": 0.5},
            "watch_history": [
                {"title": "인셉션", "genres": ["SF", "액션"]},
                {"title": "라라랜드", "genres": ["로맨스"]},
            ],
            "mood_tags": [],
        }
        result = await score_finalizer(state)
        # 인셉션(SF)만 매칭, 라라랜드(로맨스)는 미매칭
        assert "인셉션" in result["ranked_movies"][0].score_detail.similar_to
        assert "라라랜드" not in result["ranked_movies"][0].score_detail.similar_to

    @pytest.mark.asyncio
    async def test_rank_order(self):
        """rank가 1부터 순서대로 부여된다."""
        candidates = _make_candidates()[:3]
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {c.id: 0.5 for c in candidates},
            "cbf_scores": {c.id: 0.5 for c in candidates},
            "hybrid_scores": {c.id: 0.5 for c in candidates},
            "watch_history": [],
            "mood_tags": [],
        }
        result = await score_finalizer(state)
        ranks = [m.rank for m in result["ranked_movies"]]
        assert ranks == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_empty_candidates(self):
        """후보가 없으면 빈 리스트를 반환한다."""
        state: RecommendationEngineState = {
            "candidate_movies": [],
            "cf_scores": {},
            "cbf_scores": {},
            "hybrid_scores": {},
            "watch_history": [],
            "mood_tags": [],
        }
        result = await score_finalizer(state)
        assert result["ranked_movies"] == []

    @pytest.mark.asyncio
    async def test_requested_count_one_limits_output(self):
        """requested_count=1 이면 1편만 반환한다 ('인생영화 한 편만 추천해줘' 시나리오)."""
        candidates = _make_candidates()  # 5편
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {c.id: 0.5 for c in candidates},
            "cbf_scores": {c.id: 0.5 for c in candidates},
            "hybrid_scores": {c.id: 0.5 for c in candidates},
            "watch_history": [],
            "mood_tags": [],
            "preferences": ExtractedPreferences(requested_count=1),
        }
        result = await score_finalizer(state)
        assert len(result["ranked_movies"]) == 1
        assert result["ranked_movies"][0].rank == 1

    @pytest.mark.asyncio
    async def test_requested_count_three_limits_output(self):
        """requested_count=3 이면 3편만 반환한다."""
        candidates = _make_candidates()
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {c.id: 0.5 for c in candidates},
            "cbf_scores": {c.id: 0.5 for c in candidates},
            "hybrid_scores": {c.id: 0.5 for c in candidates},
            "watch_history": [],
            "mood_tags": [],
            "preferences": ExtractedPreferences(requested_count=3),
        }
        result = await score_finalizer(state)
        assert len(result["ranked_movies"]) == 3
        assert [m.rank for m in result["ranked_movies"]] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_requested_count_none_uses_default_top_k(self):
        """requested_count=None 이면 기본 TOP_K(5)편을 반환한다."""
        candidates = _make_candidates()  # 5편
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {c.id: 0.5 for c in candidates},
            "cbf_scores": {c.id: 0.5 for c in candidates},
            "hybrid_scores": {c.id: 0.5 for c in candidates},
            "watch_history": [],
            "mood_tags": [],
            "preferences": ExtractedPreferences(),  # requested_count 미지정
        }
        result = await score_finalizer(state)
        assert len(result["ranked_movies"]) == 5

    @pytest.mark.asyncio
    async def test_requested_count_without_preferences_uses_default(self):
        """preferences=None(과거 호환) 이면 기본 TOP_K(5)편을 반환한다."""
        candidates = _make_candidates()
        state: RecommendationEngineState = {
            "candidate_movies": candidates,
            "cf_scores": {c.id: 0.5 for c in candidates},
            "cbf_scores": {c.id: 0.5 for c in candidates},
            "hybrid_scores": {c.id: 0.5 for c in candidates},
            "watch_history": [],
            "mood_tags": [],
            # preferences 키 자체가 없음
        }
        result = await score_finalizer(state)
        assert len(result["ranked_movies"]) == 5
