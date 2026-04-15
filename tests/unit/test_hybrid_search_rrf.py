"""
Hybrid search RRF 합산 단위 테스트.

2026-04-15 추가 — popularity prior 가 RRF 점수에 제대로 주입되는지 검증한다.
배경: 평점 0.0 무명작이 세 엔진 모두에서 상위로 올라오면 RRF 합산만으로는
hybrid_score 최상위가 되는 구조적 문제를 막기 위한 popularity prior 가산 로직이
`reciprocal_rank_fusion()` 내부에 추가됐다.
"""

from __future__ import annotations

import math

import pytest

from monglepick.rag.hybrid_search import SearchResult, reciprocal_rank_fusion


def _sr(mid: str, title: str, metadata: dict | None = None) -> SearchResult:
    """간결한 SearchResult 팩토리."""
    return SearchResult(
        movie_id=mid,
        title=title,
        score=0.0,  # RRF 이전 원본 스코어는 합산에 쓰이지 않음
        source="test",
        metadata=metadata or {},
    )


class TestReciprocalRankFusion:
    """RRF 합산 로직 (popularity prior 포함) 검증."""

    def test_rrf_basic_ranking(self):
        """같은 rank 에 있는 두 후보는 popularity prior 만으로 순위가 갈린다.

        ES / Qdrant / Neo4j 모두에서 rank 1 로 동률인 두 영화 A(평점 8, 투표 500)와
        B(평점 0, 투표 0). RRF score 는 동일하지만 A 의 popularity_prior 가 더 커야 한다.
        """
        result_a = [_sr("a", "검증작", {"vote_count": 500, "vote_average": 8.0})]
        result_b = [_sr("b", "무명작", {"vote_count": 0, "vote_average": 0.0})]

        # 두 영화는 각각 다른 엔진 결과에만 등장해 RRF 기본 점수는 동일
        fused = reciprocal_rank_fusion([result_a, result_b])

        # A(검증작) 가 먼저 와야 함
        assert fused[0].movie_id == "a"
        assert fused[1].movie_id == "b"
        # A 의 최종 score 가 B 보다 확연히 높음 (popularity prior 효과)
        assert fused[0].score > fused[1].score + 0.01

    def test_rrf_missing_metadata_safe(self):
        """metadata 에 vote_count/vote_average 가 아예 없어도 예외 없이 처리된다."""
        results = [
            _sr("x", "메타없음", {}),  # vote_count/vote_average 키 없음
            _sr("y", "null값", {"vote_count": None, "vote_average": None}),
        ]
        # None 값/누락 필드 모두 popularity_prior=0 으로 안전 처리
        fused = reciprocal_rank_fusion([results])
        assert len(fused) == 2
        # 둘 다 prior=0 이므로 원본 rank 순서 유지
        assert fused[0].movie_id == "x"
        assert fused[1].movie_id == "y"

    def test_rrf_prior_scale_is_comparable_to_rrf(self):
        """popularity prior 의 스케일이 RRF 점수와 비슷해 BM25 지배를 뒤집을 수 있다.

        설계 의도: 평점 0.0 무명작이 세 엔진에서 rank 1 로 올라와도, 평점·투표수 강한
        검증작이 더 낮은 rank 에서 popularity prior 로 최종 상위가 되어야 한다.
        prior 가 너무 작으면(BM25 sum 모드의 과거 버그) 무명작이 계속 1 위.

        케이스: rank 1 무명작(평점 0) vs rank 5 검증작(평점 8.5, 투표 1000)
        - unknown: 1/(60+1) + 0 = 0.01639
        - known:   1/(60+5) + log1p(1000)*0.003 + 8.5*0.001 = 0.01538 + 0.0290 = 0.04438
        검증작이 최종 1 위가 되어야 한다.
        """
        engine = [
            _sr("unknown", "무명작", {"vote_count": 0, "vote_average": 0}),
            *[_sr(f"pad{i}", f"pad{i}") for i in range(3)],
            _sr("known", "검증작", {"vote_count": 1000, "vote_average": 8.5}),
        ]
        fused = reciprocal_rank_fusion([engine])

        # 검증작이 popularity prior 효과로 최종 1 위
        assert fused[0].movie_id == "known"

    def test_rrf_prior_flips_same_rank_tie(self):
        """rank 동률일 때는 popularity prior 가 순위를 결정한다."""
        # 두 영화가 각각 별도 엔진에서 rank 1
        engine_a = [_sr("popular", "인기작", {"vote_count": 1000, "vote_average": 8.5})]
        engine_b = [_sr("indie", "무명작", {"vote_count": 0, "vote_average": 0})]

        fused = reciprocal_rank_fusion([engine_a, engine_b])

        assert fused[0].movie_id == "popular"
        # 점수 차이는 대략 log1p(1000)*0.003 + 8.5*0.001 ≈ 0.029
        expected_gap = math.log1p(1000) * 0.003 + 8.5 * 0.001
        assert abs((fused[0].score - fused[1].score) - expected_gap) < 0.001

    def test_rrf_empty_input(self):
        """빈 입력은 빈 리스트 반환 (regression 방지)."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_rrf_accumulates_across_engines(self):
        """세 엔진 모두에서 상위에 오른 후보가 단일 엔진 후보보다 높은 점수를 받는다."""
        common = _sr("all3", "삼엔진", {"vote_count": 50, "vote_average": 6.0})
        single = _sr("single", "단일엔진", {"vote_count": 50, "vote_average": 6.0})

        # all3 는 세 엔진 rank 1, single 은 한 엔진 rank 1
        fused = reciprocal_rank_fusion([
            [common, _sr("x", "x")],
            [common, _sr("y", "y")],
            [common, single],
        ])

        assert fused[0].movie_id == "all3"
        # single 이 상위로 올라오더라도 all3 뒤에 있어야 함
        single_rank = next(i for i, r in enumerate(fused) if r.movie_id == "single")
        assert single_rank > 0
