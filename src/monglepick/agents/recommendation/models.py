"""
추천 엔진 서브그래프 State 모델 (§7-1).

Chat Agent의 recommendation_ranker 노드에서 호출하는 서브그래프의 State를 정의한다.
입력은 Chat Agent에서 전달받고, 출력(ranked_movies)은 Chat Agent로 반환된다.

RecommendationEngineState 필드:
- 입력: candidate_movies, user_id, user_profile, watch_history, emotion, mood_tags, preferences
- 내부: is_cold_start, cf_scores, cbf_scores, hybrid_scores
- 출력: ranked_movies
"""

from __future__ import annotations

from typing import Any, TypedDict

from monglepick.agents.chat.models import (
    CandidateMovie,
    EmotionResult,
    ExtractedPreferences,
    RankedMovie,
)


class RecommendationEngineState(TypedDict, total=False):
    """
    추천 엔진 서브그래프 State (§7-1).

    Chat Agent의 rag_retriever 출력(candidate_movies)을 입력으로 받아,
    CF+CBF 하이브리드 추천 → MMR 다양성 재정렬 → ScoreDetail 첨부를 거쳐
    최종 ranked_movies를 반환한다.

    total=False: 모든 키가 Optional (노드가 점진적으로 채워나감).
    """

    # ── 입력 (Chat Agent에서 전달) ──
    candidate_movies: list[CandidateMovie]    # rag_retriever 출력 후보 영화 목록
    user_id: str                              # 사용자 ID (빈 문자열이면 익명)
    user_profile: dict[str, Any]              # MySQL 유저 프로필
    watch_history: list[dict[str, Any]]       # MySQL 시청 이력 (최근 50건)
    emotion: EmotionResult | None             # 감정 분석 결과
    mood_tags: list[str]                      # emotion.mood_tags (감정→무드 매핑)
    preferences: ExtractedPreferences | None  # 사용자 선호 조건 (7개 필드)

    # ── 내부 상태 ──
    is_cold_start: bool                       # Cold Start 여부 (시청 < 5편)
    cf_cache_miss: bool                       # CF 캐시 미스 여부 (Redis에 유사 유저 없음)
    cf_scores: dict[str, float]               # {movie_id: cf_score} 협업 필터링 점수
    cbf_scores: dict[str, float]              # {movie_id: cbf_score} 컨텐츠 기반 점수
    hybrid_scores: dict[str, float]           # {movie_id: hybrid_score} 가중 합산 점수

    # ── 출력 ──
    ranked_movies: list[RankedMovie]          # 최종 순위 결과 (Top 5)
