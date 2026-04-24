"""
추천 엔진 서브그래프 노드 함수 (§7-2 Node 1~7).

LangGraph StateGraph의 각 노드로 등록되는 7개 async 함수.
시그니처: async def node_name(state: RecommendationEngineState) -> dict

모든 노드는 try/except로 감싸고, 에러 시 유효한 기본값을 반환한다 (에러 전파 금지).
LLM을 호출하지 않으며, 규칙 기반 + Redis CF 캐시로만 동작한다.

노드 목록:
1. cold_start_checker       — 시청 이력 기반 Cold Start 판정
2. collaborative_filter     — Redis CF 캐시에서 유사 유저 기반 점수 계산
3. content_based_filter     — 장르/감독/배우/무드 매칭 기반 CBF 점수 계산
4. hybrid_merger            — CF + CBF 가중 합산 (시청이력/감정에 따라 가중치 조절)
5. popularity_fallback      — Cold Start 유저용 인기도 기반 점수
6. diversity_reranker       — MMR(λ=0.7) 장르 기반 다양성 재정렬
7. score_finalizer          — Top 5편 선정 + RankedMovie 변환 + ScoreDetail 첨부
"""

from __future__ import annotations

import math
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import structlog
from langsmith import traceable

from monglepick.agents.chat.models import (
    CandidateMovie,
    RankedMovie,
    ScoreDetail,
)
from monglepick.agents.recommendation.models import RecommendationEngineState
from monglepick.config import settings as _settings
from monglepick.db.clients import get_redis

logger = structlog.get_logger()

# ── 상수 (config.py에서 환경변수로 설정 가능) ──

# Cold Start 임계값 (§7-2 Node 1)
COLD_START_THRESHOLD = _settings.COLD_START_THRESHOLD     # 시청 < 5편: Cold Start
WARM_START_THRESHOLD = _settings.WARM_START_THRESHOLD     # 시청 5~29편: Warm Start, 30편+: 정상

# CF 캐시 미스 시 기본값 (내부 구현 상수, config 미포함)
CF_DEFAULT_SCORE = 0.5

# MMR 파라미터 (§7-2 Node 6)
MMR_LAMBDA = _settings.MMR_LAMBDA  # 관련성 70% + 다양성 30%

# 최종 선택 영화 수
TOP_K = _settings.RECOMMENDATION_TOP_K

# Redis 키 접두사 (cf_builder.py와 동일)
KEY_SIMILAR_USERS = "cf:similar_users:{user_id}"
KEY_USER_RATINGS = "cf:user_ratings:{user_id}"


# ============================================================
# 1. cold_start_checker — Cold Start 판정
# ============================================================

@traceable(
    name="cold_start_checker",
    run_type="chain",
    metadata={"node": "rec_1/7", "subgraph": "recommendation_engine"},
)
async def cold_start_checker(state: RecommendationEngineState) -> dict:
    """
    사용자의 시청 이력 수를 기반으로 Cold Start 여부를 판정한다.

    판정 기준 (§7-2 Node 1):
    - 시청 < 5편: Cold Start (is_cold_start=True) → popularity_fallback 경로
    - 시청 5편+: 정상 (is_cold_start=False) → CF+CBF 경로

    Args:
        state: RecommendationEngineState (watch_history 필요)

    Returns:
        dict: is_cold_start(bool) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    try:
        watch_history = state.get("watch_history", [])
        history_count = len(watch_history)

        # 시청 이력이 COLD_START_THRESHOLD 미만이면 Cold Start
        is_cold_start = history_count < COLD_START_THRESHOLD

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "cold_start_checked",
            history_count=history_count,
            is_cold_start=is_cold_start,
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"is_cold_start": is_cold_start}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("cold_start_checker_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1))
        # 에러 시 Cold Start로 간주 (안전한 기본값)
        return {"is_cold_start": True}


# ============================================================
# 2. collaborative_filter — CF 점수 계산
# ============================================================

@traceable(
    name="collaborative_filter",
    run_type="chain",
    metadata={"node": "rec_2/7", "subgraph": "recommendation_engine"},
)
async def collaborative_filter(state: RecommendationEngineState) -> dict:
    """
    Redis CF 캐시에서 유사 유저를 조회하고, 후보 영화별 CF 점수를 계산한다.

    CF_score(u, m) = Σ sim(u,v) × r(v,m) / Σ |sim(u,v)|
    - v: Top-50 유사 유저 중 영화 m을 평가한 유저
    - sim(u,v): 유저 u와 v의 코사인 유사도
    - r(v,m): 유저 v가 영화 m에 부여한 평점

    Redis에 유저가 없으면 (익명/캐시 미스): 모든 후보에 cf_score=0.5
    CF 점수 전부 0이면: 모든 후보에 cf_score=0.5
    결과를 min-max 정규화하여 [0, 1] 범위로 변환

    Args:
        state: RecommendationEngineState (user_id, candidate_movies 필요)

    Returns:
        dict: cf_scores({movie_id: float}) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    try:
        user_id = state.get("user_id", "")
        candidates = state.get("candidate_movies", [])

        if not candidates:
            return {"cf_scores": {}}

        candidate_ids = {c.id for c in candidates}
        cf_scores: dict[str, float] = {}

        # 익명 사용자이거나 user_id가 없으면 기본값 반환 + 캐시 미스 명시
        if not user_id:
            cf_scores = {c.id: CF_DEFAULT_SCORE for c in candidates}
            logger.info("cf_anonymous_user", score=CF_DEFAULT_SCORE)
            return {"cf_scores": cf_scores, "cf_cache_miss": True}

        # similar_users_raw를 try 블록 밖에서 초기화하여 로깅 시 안전하게 접근 (dir() 안티패턴 제거)
        similar_users_raw: list = []

        try:
            redis = await get_redis()

            # 유사 유저 Top-50 조회 (Sorted Set, 높은 점수순)
            similar_users_key = KEY_SIMILAR_USERS.format(user_id=user_id)
            similar_users_raw = await redis.zrevrangebyscore(
                similar_users_key,
                max="+inf",
                min="-inf",
                withscores=True,
                start=0,
                num=50,
            )

            # Redis에 유사 유저가 없으면 캐시 미스
            # Phase 3: implicit_ratings가 있으면 fallback으로 활용, 없으면 기본값 0.5
            if not similar_users_raw:
                implicit_ratings = state.get("implicit_ratings", {})
                if implicit_ratings:
                    # 암시적 평점을 0~1 범위로 정규화하여 CF 점수로 사용
                    cf_scores = {}
                    for c in candidates:
                        raw = implicit_ratings.get(c.id)
                        if raw is not None:
                            cf_scores[c.id] = min(float(raw) / 5.0, 1.0)
                        else:
                            cf_scores[c.id] = CF_DEFAULT_SCORE
                    logger.info("cf_cache_miss_implicit_fallback",
                                user_id=user_id, implicit_count=len(implicit_ratings))
                else:
                    cf_scores = {c.id: CF_DEFAULT_SCORE for c in candidates}
                    logger.info("cf_cache_miss", user_id=user_id)
                return {"cf_scores": cf_scores, "cf_cache_miss": True}

            # 유사 유저별 평점 조회 (pipeline으로 배치 조회)
            sim_user_ids = [uid for uid, _ in similar_users_raw]
            sim_scores_map = {uid: score for uid, score in similar_users_raw}

            pipe = redis.pipeline()
            for sim_uid in sim_user_ids:
                ratings_key = KEY_USER_RATINGS.format(user_id=sim_uid)
                pipe.hgetall(ratings_key)
            all_ratings = await pipe.execute()

            # 후보 영화별 CF 점수 계산
            for c_id in candidate_ids:
                numerator = 0.0
                denominator = 0.0

                for idx, sim_uid in enumerate(sim_user_ids):
                    user_ratings = all_ratings[idx]
                    if not user_ratings:
                        continue

                    # 해당 유사 유저가 이 영화를 평가했는지 확인
                    if c_id in user_ratings:
                        sim_val = sim_scores_map[sim_uid]
                        try:
                            rating_val = float(user_ratings[c_id])
                        except (ValueError, TypeError):
                            # Redis 데이터 손상 시 해당 유저의 평점을 건너뜀
                            continue
                        numerator += sim_val * rating_val
                        denominator += abs(sim_val)

                if denominator > 0:
                    cf_scores[c_id] = numerator / denominator
                else:
                    cf_scores[c_id] = 0.0

            # CF 점수 전부 0이면 기본값으로 대체
            if all(v == 0.0 for v in cf_scores.values()):
                cf_scores = {c_id: CF_DEFAULT_SCORE for c_id in candidate_ids}
                logger.info("cf_all_zero_fallback")
            else:
                # min-max 정규화 [0, 1]
                cf_scores = _min_max_normalize(cf_scores)

        except Exception as redis_err:
            # Redis 연결 실패 시 기본값으로 graceful degradation
            logger.warning("cf_redis_error", error=str(redis_err))
            cf_scores = {c.id: CF_DEFAULT_SCORE for c in candidates}

        # CF 점수 상위 5개 영화 상세 로깅
        sorted_cf = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_title_map = {c.id: c.title for c in candidates}
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "cf_scores_calculated",
            user_id=user_id,
            candidate_count=len(cf_scores),
            avg_score=round(sum(cf_scores.values()) / max(len(cf_scores), 1), 4),
            similar_user_count=len(similar_users_raw),
            top_cf_scores=[
                {
                    "title": candidate_title_map.get(mid, mid),
                    "cf_score": round(score, 4),
                }
                for mid, score in sorted_cf[:5]
            ],
            elapsed_ms=round(elapsed_ms, 1),
        )
        # 정상 경로: CF 데이터가 실제로 존재 → 캐시 미스 아님
        return {"cf_scores": cf_scores, "cf_cache_miss": False}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("collaborative_filter_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1))
        candidates = state.get("candidate_movies", [])
        # Redis 에러 시에도 캐시 미스로 명시
        return {"cf_scores": {c.id: CF_DEFAULT_SCORE for c in candidates}, "cf_cache_miss": True}


# ============================================================
# 3. content_based_filter — CBF 점수 계산
# ============================================================

@traceable(
    name="content_based_filter",
    run_type="chain",
    metadata={"node": "rec_3/7", "subgraph": "recommendation_engine"},
)
async def content_based_filter(state: RecommendationEngineState) -> dict:
    """
    후보 영화별 컨텐츠 기반 필터링(CBF) 점수를 계산한다.

    비교 요소 및 가중치 (감정 있음 / 감정 없음):
    - 장르 일치도:     0.25 / 0.30  (Jaccard)
    - 감독/배우 빈도:  0.20 / 0.24  (시청 이력 빈도 × decay)
    - 무드 매칭:       0.15 / 0.00  (교집합 비율)
    - 키워드 매칭:     0.15 / 0.16  (선호 키워드 Jaccard)
    - RRF 점수 보존:   0.25 / 0.30  (정규화된 rrf_score)

    감정 없음일 때 무드 가중치를 0으로 설정하고 나머지를 재분배한다.

    Args:
        state: RecommendationEngineState (candidate_movies, watch_history,
               emotion, preferences, mood_tags 필요)

    Returns:
        dict: cbf_scores({movie_id: float}) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    try:
        candidates = state.get("candidate_movies", [])
        watch_history = state.get("watch_history", [])
        emotion = state.get("emotion")
        preferences = state.get("preferences")
        mood_tags = state.get("mood_tags", [])

        if not candidates:
            return {"cbf_scores": {}}

        # 감정 존재 여부 판정
        has_emotion = (
            emotion is not None
            and emotion.emotion is not None
            and len(mood_tags) > 0
        )

        # ── 사용자 프로필 구축 (시청 이력 기반) ──
        liked_genres = _extract_liked_genres(watch_history, top_k=5)
        director_freq, actor_freq = _extract_crew_frequency(watch_history)

        # 선호 키워드: preferences에서 추출
        pref_keywords: set[str] = set()
        if preferences:
            if preferences.genre_preference:
                pref_keywords.update(
                    g.strip() for g in preferences.genre_preference.split(",") if g.strip()
                )
            if preferences.mood:
                pref_keywords.add(preferences.mood.strip())

        # ── 가중치 설정 ──
        if has_emotion:
            w_genre = 0.25
            w_crew = 0.20
            w_mood = 0.15
            w_keyword = 0.15
            w_rrf = 0.25
        else:
            # Phase ML-3: 감정 미감지 시에도 무드 가중치를 최소 5% 유지
            # 기존: w_mood=0.00 → 무드태그가 완전히 무시되어 분위기 매칭 불가
            # 개선: w_mood=0.05 → 무드태그가 있는 영화에 약간의 가산점 부여
            # (사용자가 감정을 명시하지 않아도 영화 자체의 무드 다양성에 기여)
            w_genre = 0.30
            w_crew = 0.22
            w_mood = 0.05
            w_keyword = 0.15
            w_rrf = 0.28

        # RRF 점수 정규화용 max 값
        max_rrf = max((c.rrf_score for c in candidates), default=1.0) or 1.0

        # ── 후보 영화별 CBF 점수 계산 ──
        cbf_scores: dict[str, float] = {}

        for movie in candidates:
            # 1. 장르 일치도 (Jaccard)
            genre_score = _jaccard(set(movie.genres), liked_genres)

            # 2. 감독/배우 빈도 매칭
            crew_score = _crew_match_score(movie, director_freq, actor_freq)

            # 3. 무드 매칭 — 유저 무드 기준 비율 (유저가 원하는 무드가 영화에 몇 개 있는지)
            # 기존: len(교집합) / len(영화무드) → 다양한 무드태그를 가진 영화가 불이익
            # 수정: len(교집합) / len(유저무드) → 유저가 원하는 것 중 매칭 비율
            if has_emotion and mood_tags:
                movie_moods = set(movie.mood_tags) if movie.mood_tags else set()
                user_moods = set(mood_tags)
                mood_score = (
                    len(movie_moods & user_moods) / max(len(user_moods), 1)
                )
            else:
                mood_score = 0.0

            # 4. 키워드/장르 매칭 (선호 키워드 vs 영화 장르+무드)
            movie_keywords = set(movie.genres) | set(movie.mood_tags or [])
            keyword_score = _jaccard(movie_keywords, pref_keywords) if pref_keywords else 0.0

            # 5. RRF 점수 보존 (정규화)
            rrf_norm = movie.rrf_score / max_rrf

            # 가중 합산
            total = (
                w_genre * genre_score
                + w_crew * crew_score
                + w_mood * mood_score
                + w_keyword * keyword_score
                + w_rrf * rrf_norm
            )
            cbf_scores[movie.id] = total

        # min-max 정규화 [0, 1]
        cbf_scores = _min_max_normalize(cbf_scores)

        # CBF 점수 상위 5개 영화 상세 로깅
        sorted_cbf = sorted(cbf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_title_map = {c.id: c.title for c in candidates}
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "cbf_scores_calculated",
            candidate_count=len(cbf_scores),
            has_emotion=has_emotion,
            liked_genres=list(liked_genres)[:5],
            weights={"genre": w_genre, "crew": w_crew, "mood": w_mood, "keyword": w_keyword, "rrf": w_rrf},
            top_cbf_scores=[
                {
                    "title": candidate_title_map.get(mid, mid),
                    "cbf_score": round(score, 4),
                }
                for mid, score in sorted_cbf[:5]
            ],
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"cbf_scores": cbf_scores}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("content_based_filter_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1))
        candidates = state.get("candidate_movies", [])
        return {"cbf_scores": {c.id: 0.0 for c in candidates}}


# ============================================================
# 4. hybrid_merger — CF + CBF 가중 합산
# ============================================================

@traceable(
    name="hybrid_merger",
    run_type="chain",
    metadata={"node": "rec_4/7", "subgraph": "recommendation_engine"},
)
async def hybrid_merger(state: RecommendationEngineState) -> dict:
    """
    CF와 CBF 점수를 가중 합산하여 hybrid_score를 계산한다.

    가중치 조건표 (§7-2 Node 4):
    - 시청 30편+ AND 감정 없음: CF 0.60, CBF 0.40
    - 시청 30편+ AND 감정 있음: CF 0.50, CBF 0.50
    - 시청 5~29편 AND 감정 없음: CF 0.40, CBF 0.60
    - 시청 5~29편 AND 감정 있음: CF 0.30, CBF 0.70

    특수 케이스:
    - CF 전부 0.5 (캐시 미스): w_cf=0.0, w_cbf=1.0 (CBF에 전적 의존)
    - CBF 전부 0: w_cf=1.0, w_cbf=0.0 (CF에 전적 의존)
    - 양쪽 모두 0: RRF 점수를 hybrid_score로 사용

    Args:
        state: RecommendationEngineState (cf_scores, cbf_scores, watch_history,
               emotion, candidate_movies 필요)

    Returns:
        dict: hybrid_scores({movie_id: float}) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    try:
        cf_scores = state.get("cf_scores", {})
        cbf_scores = state.get("cbf_scores", {})
        watch_history = state.get("watch_history", [])
        emotion = state.get("emotion")
        candidates = state.get("candidate_movies", [])

        if not candidates:
            return {"hybrid_scores": {}}

        history_count = len(watch_history)
        has_emotion = emotion is not None and emotion.emotion is not None

        # ── CF 캐시 미스 감지: collaborative_filter 노드의 명시적 플래그 사용 ──
        # 기존 로직(점수 전부 0.5 비교)은 정규화 결과와 캐시 미스를 구분 못하는 오탐 존재.
        # 이제 collaborative_filter가 cf_cache_miss 플래그를 반환한다.
        cf_is_default = state.get("cf_cache_miss", False)

        # ── CBF 전부 0 감지 ──
        cbf_values = list(cbf_scores.values())
        cbf_all_zero = len(cbf_values) > 0 and all(v == 0.0 for v in cbf_values)

        # ── 가중치 결정 ──
        if cf_is_default and cbf_all_zero:
            # 양쪽 모두 무의미: RRF 점수 사용
            w_cf, w_cbf = 0.0, 0.0
        elif cf_is_default:
            # CF 캐시 미스: CBF에 전적 의존
            w_cf, w_cbf = 0.0, 1.0
        elif cbf_all_zero:
            # CBF 전부 0: CF에 전적 의존
            w_cf, w_cbf = 1.0, 0.0
        elif history_count >= WARM_START_THRESHOLD:
            # 시청 30편+
            if has_emotion:
                w_cf, w_cbf = 0.50, 0.50
            else:
                w_cf, w_cbf = 0.60, 0.40
        else:
            # 시청 5~29편 (Warm Start)
            if has_emotion:
                w_cf, w_cbf = 0.30, 0.70
            else:
                w_cf, w_cbf = 0.40, 0.60

        # ── Phase 4: taste_consistency 기반 동적 가중치 보정 ──
        # user_behavior_profile이 존재하면 취향 일관성에 따라 CF/CBF 비율 조정
        # - 취향 편향(consistency > 0.7): CBF 비중 +0.10 (자기 취향에 맞는 추천)
        # - 탐색형(consistency < 0.3): CF 비중 +0.10 (다양한 발견)
        profile = state.get("user_behavior_profile", {})
        taste_consistency = profile.get("taste_consistency")
        if taste_consistency is not None and w_cf + w_cbf > 0:
            if taste_consistency > 0.7:
                # 편향적 취향: CBF 의존도 높임
                boost = 0.10
                w_cbf = min(w_cbf + boost, 1.0)
                w_cf = max(w_cf - boost, 0.0)
            elif taste_consistency < 0.3:
                # 탐색형 유저: CF 의존도 높임 (다양한 유저 기반 추천)
                boost = 0.10
                w_cf = min(w_cf + boost, 1.0)
                w_cbf = max(w_cbf - boost, 0.0)
            # 합이 1.0 유지되도록 정규화
            total = w_cf + w_cbf
            if total > 0:
                w_cf /= total
                w_cbf /= total

        # ── hybrid_score 계산 ──
        hybrid_scores: dict[str, float] = {}

        # RRF 점수 맵 구성 (fallback용)
        rrf_map = {c.id: c.rrf_score for c in candidates}
        max_rrf = max(rrf_map.values(), default=1.0) or 1.0

        for movie in candidates:
            mid = movie.id
            cf_val = cf_scores.get(mid, 0.0)
            cbf_val = cbf_scores.get(mid, 0.0)

            if w_cf == 0.0 and w_cbf == 0.0:
                # 양쪽 모두 무의미: RRF 점수를 hybrid_score로 사용
                hybrid_scores[mid] = rrf_map.get(mid, 0.0) / max_rrf
            else:
                hybrid_scores[mid] = w_cf * cf_val + w_cbf * cbf_val

        # hybrid 점수 상위 5개 영화 상세 로깅
        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_title_map = {c.id: c.title for c in candidates}
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "hybrid_scores_merged",
            w_cf=w_cf,
            w_cbf=w_cbf,
            history_count=history_count,
            has_emotion=has_emotion,
            cf_is_default=cf_is_default,
            cbf_all_zero=cbf_all_zero,
            top_hybrid_scores=[
                {
                    "title": candidate_title_map.get(mid, mid),
                    "hybrid": round(score, 4),
                    "cf": round(cf_scores.get(mid, 0.0), 4),
                    "cbf": round(cbf_scores.get(mid, 0.0), 4),
                }
                for mid, score in sorted_hybrid[:5]
            ],
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"hybrid_scores": hybrid_scores}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("hybrid_merger_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1))
        # fallback: RRF 점수를 hybrid_score로 사용
        candidates = state.get("candidate_movies", [])
        max_rrf = max((c.rrf_score for c in candidates), default=1.0) or 1.0
        return {
            "hybrid_scores": {
                c.id: c.rrf_score / max_rrf for c in candidates
            }
        }


# ============================================================
# 5. popularity_fallback — Cold Start 인기도 기반 점수
# ============================================================

@traceable(
    name="popularity_fallback",
    run_type="chain",
    metadata={"node": "rec_5/7", "subgraph": "recommendation_engine"},
)
async def popularity_fallback(state: RecommendationEngineState) -> dict:
    """
    Cold Start 유저용 인기도 기반 점수를 계산한다.

    기본 점수 = rating / 10.0 (0~1 범위)
    + 장르 매칭 부스트: 선호 장르와 겹치면 +0.1
    + 무드 매칭 부스트: 무드 태그가 겹치면 +0.05

    결과를 hybrid_scores에 저장 (CF+CBF 대체).

    Args:
        state: RecommendationEngineState (candidate_movies, preferences,
               mood_tags, emotion 필요)

    Returns:
        dict: hybrid_scores({movie_id: float}), cf_scores, cbf_scores 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    try:
        candidates = state.get("candidate_movies", [])
        preferences = state.get("preferences")
        mood_tags = state.get("mood_tags", [])

        if not candidates:
            return {
                "hybrid_scores": {},
                "cf_scores": {},
                "cbf_scores": {},
            }

        # 선호 장르 추출 (preferences에서)
        pref_genres: set[str] = set()
        if preferences and preferences.genre_preference:
            pref_genres = {
                g.strip()
                for g in preferences.genre_preference.split(",")
                if g.strip()
            }

        user_moods = set(mood_tags) if mood_tags else set()

        # ── 인기도 기반 점수 계산 (평점 가중치 강화) ──
        # 기존 문제: RRF 점수가 평점을 압도하여 무명 영화가 상위에 노출됨
        # 수정: 평점 비중 40%, RRF 비중 15%로 조정하여 고평점 영화 우선
        hybrid_scores: dict[str, float] = {}

        for movie in candidates:
            # 평점 점수: rating / 10.0 (TMDB 평점 0~10 → 0~1), 가중치 40%
            # Phase Q-2: 평점 0 = 평가 데이터 없음 → 0.1로 강한 페널티 (기존 0.3)
            # 추가로 포스터/줄거리 없는 영화도 페널티를 적용하여 데이터 부족 영화 하위 배치
            rating_score = min(movie.rating / 10.0, 1.0) if movie.rating and movie.rating >= 1.0 else 0.1

            # 장르 매칭 부스트: 가중치 20%
            genre_score = 0.0
            if pref_genres and movie.genres:
                matched = set(movie.genres) & pref_genres
                genre_score = len(matched) / max(len(pref_genres), 1)

            # 무드 매칭 부스트: 가중치 10%
            mood_score = 0.0
            if user_moods and movie.mood_tags:
                matched = set(movie.mood_tags) & user_moods
                mood_score = len(matched) / max(len(user_moods), 1)

            # RRF 점수 (검색 관련성): 가중치 15%
            rrf_score = movie.rrf_score

            # 인기도 점수 (TMDB popularity 기반): 가중치 15%
            # Phase 0-4: CandidateMovie.popularity_score 실제 값 사용
            # TMDB popularity는 0~1000+ 범위이므로 log 정규화하여 0~1로 변환
            raw_pop = movie.popularity_score if movie.popularity_score else 0.0
            popularity_score = min(math.log1p(raw_pop) / math.log1p(1000), 1.0)

            # ── Phase Q-2: 데이터 품질 페널티 ──
            # 포스터/줄거리/평점이 부족한 영화는 최종 점수에 페널티를 적용한다.
            # 3개 중 충족 개수로 품질 계수 결정: 3개=1.0, 2개=0.8, 1개=0.5, 0개=0.2
            data_quality_fields = 0
            if movie.poster_path and movie.poster_path.strip():
                data_quality_fields += 1
            if movie.overview and len(movie.overview.strip()) >= 20:
                data_quality_fields += 1
            if movie.rating and movie.rating >= 1.0:
                data_quality_fields += 1
            data_quality_multiplier = {3: 1.0, 2: 0.8, 1: 0.5, 0: 0.2}.get(
                data_quality_fields, 0.2
            )

            # 가중 합산 (총 100%) × 데이터 품질 계수
            hybrid_scores[movie.id] = (
                0.40 * rating_score
                + 0.20 * genre_score
                + 0.10 * mood_score
                + 0.15 * rrf_score
                + 0.15 * popularity_score
            ) * data_quality_multiplier

        # min-max 정규화
        hybrid_scores = _min_max_normalize(hybrid_scores)

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "popularity_fallback_calculated",
            candidate_count=len(hybrid_scores),
            pref_genres=list(pref_genres),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {
            "hybrid_scores": hybrid_scores,
            # Cold Start: CF/CBF 점수는 0.0
            "cf_scores": {c.id: 0.0 for c in candidates},
            "cbf_scores": {c.id: 0.0 for c in candidates},
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("popularity_fallback_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1))
        candidates = state.get("candidate_movies", [])
        return {
            "hybrid_scores": {c.id: c.rrf_score for c in candidates},
            "cf_scores": {c.id: 0.0 for c in candidates},
            "cbf_scores": {c.id: 0.0 for c in candidates},
        }


# ── 데이터 품질 보너스 헬퍼 (Phase Q-2.1) ──
# 포스터/줄거리/평점 완비도에 따라 0.0~0.005 가산점을 반환한다.
# MMR 점수에 반영하여 메타데이터가 충실한 영화를 우선 추천한다.
# 가산점 범위(0.005)는 RRF 평균(~0.01)의 절반 수준으로, 관련성을 뒤집지 않되
# 비슷한 점수의 후보 간에 품질 좋은 쪽을 선호하도록 설계.
DATA_QUALITY_BONUS = 0.005


def _data_quality_bonus(movie) -> float:
    """포스터/줄거리/평점 완비도에 따른 MMR 가산점 (0.0 ~ DATA_QUALITY_BONUS)."""
    fields = 0
    total = 3
    if movie.poster_path and movie.poster_path.strip():
        fields += 1
    if movie.overview and len(movie.overview.strip()) >= 20:
        fields += 1
    if movie.rating and movie.rating >= 1.0:
        fields += 1
    return DATA_QUALITY_BONUS * (fields / total)


# ── Popular / Hidden Gem 슬롯 quota ──
# 최종 TOP_K(5) 중 몇 개를 "인기작(=검증된 품질)" 으로 채울지 결정하는 임계값.
# 설계 의도:
#   1) 평점 0.0/트레일러·포스터 부재인 무명작이 상위 1~3순위를 독점하는 것을 막는다.
#   2) 그렇다고 무명작을 완전히 배제하지는 않는다 — hidden gem 발굴도 서비스 매력의 일부.
#   3) 해결: 슬롯을 분리해 "인기작 풀" 에서 POPULAR_SLOTS 개, "그 외(hidden) 풀" 에서
#      HIDDEN_SLOTS 개를 각각 MMR 로 뽑아 최종 리스트를 구성한다.
# 각 풀이 부족하면 다른 풀로 채움 (총 TOP_K 유지).
POPULAR_SLOTS = 3
HIDDEN_SLOTS = 2  # TOP_K=5 기준 ( POPULAR_SLOTS + HIDDEN_SLOTS == TOP_K )

# 인기작 분류 기준 — rating 과 vote_count 둘 중 하나만 충족해도 popular 풀에 포함.
# 평점만 있고 투표수 없는 구작, 또는 투표수 많고 평점이 가지각색인 블록버스터 모두 커버.
POPULAR_MIN_RATING = 5.0
POPULAR_MIN_VOTE_COUNT = 50


def _is_popular(movie) -> bool:
    """인기작/검증작 여부 판정 (rating 또는 vote_count 임계값 충족)."""
    rating_ok = bool(movie.rating and movie.rating >= POPULAR_MIN_RATING)
    vote_ok = bool(
        getattr(movie, "vote_count", None) is not None
        and movie.vote_count >= POPULAR_MIN_VOTE_COUNT
    )
    return rating_ok or vote_ok


def _mmr_select(
    pool_ids: set[str],
    candidate_map: dict,
    hybrid_scores: dict[str, float],
    already_selected: list,
    k: int,
) -> list:
    """
    주어진 풀에서 MMR 알고리즘으로 최대 k 편을 greedy 선택한다.

    diversity_reranker 의 기존 루프를 함수로 분리해 popular / hidden 슬롯 양쪽에
    동일 로직을 재사용하기 위함. `already_selected` 가 비어 있으면 첫 선택은
    hybrid_score + quality_bonus 최고점, 이후는 MMR 점수 최고점을 선택한다.

    Args:
        pool_ids: 이 풀에서 선택 가능한 영화 id 집합 (호출 측이 소비할 때마다 줄어듦)
        candidate_map: {id: CandidateMovie} 조회용 dict
        hybrid_scores: {id: float} — 관련성 점수
        already_selected: 이미 선택된 영화 리스트 (다른 풀에서 선택된 것도 포함 가능)
        k: 이 풀에서 뽑을 최대 편수

    Returns:
        선택된 CandidateMovie 리스트 (최대 k 편, 풀이 부족하면 그만큼만)
    """
    selected: list = []
    remaining = set(pool_ids)  # 호출자 집합을 건드리지 않도록 복사

    for i in range(k):
        if not remaining:
            break

        # 첫 선택이고 already_selected 도 비어있을 때만 단순 hybrid+quality 최고점
        if i == 0 and not already_selected:
            best_id = max(
                remaining,
                key=lambda mid: hybrid_scores.get(mid, 0.0)
                + _data_quality_bonus(candidate_map[mid]),
            )
        else:
            # MMR: 이미 선택된 영화(already_selected + selected) 와의 최대 유사도 계산
            best_id = None
            best_mmr = float("-inf")
            reference = list(already_selected) + selected

            for mid in remaining:
                movie = candidate_map[mid]
                relevance = hybrid_scores.get(mid, 0.0)

                # 이미 선택된 영화와의 최대 유사도 (장르 60% + 감독 25% + 배우 15%)
                max_sim = 0.0
                for sel in reference:
                    genre_sim = _jaccard(set(movie.genres), set(sel.genres))
                    director_sim = (
                        1.0
                        if (movie.director and movie.director == sel.director)
                        else 0.0
                    )
                    cast_overlap = (
                        len(set(movie.cast[:3]) & set(sel.cast[:3])) / 3.0
                        if movie.cast and sel.cast
                        else 0.0
                    )
                    sim = 0.60 * genre_sim + 0.25 * director_sim + 0.15 * cast_overlap
                    if sim > max_sim:
                        max_sim = sim

                quality_bonus = _data_quality_bonus(movie)
                mmr_score = (
                    MMR_LAMBDA * relevance
                    - (1 - MMR_LAMBDA) * max_sim
                    + quality_bonus
                )

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_id = mid

        if best_id is None:
            break
        selected.append(candidate_map[best_id])
        remaining.discard(best_id)
        pool_ids.discard(best_id)  # 호출자 풀에서도 제거 → 중복 선택 방지

    return selected


# ============================================================
# 6. diversity_reranker — MMR 다양성 재정렬
# ============================================================

@traceable(
    name="diversity_reranker",
    run_type="chain",
    metadata={"node": "rec_6/7", "subgraph": "recommendation_engine"},
)
async def diversity_reranker(state: RecommendationEngineState) -> dict:
    """
    MMR(Maximum Marginal Relevance) 알고리즘으로 다양성을 고려한 재정렬을 수행한다.

    MMR_score(m) = λ × relevance(m) - (1 - λ) × max_genre_sim(m, S)

    - λ = 0.7 (관련성 70% + 다양성 30%)
    - relevance(m) = hybrid_scores[m.id]
    - max_genre_sim(m, S) = max(Jaccard(m.genres, s.genres) for s in S)
    - S: 이미 선택된 영화 집합

    Greedy 선택: 첫 영화는 최고 hybrid_score, 나머지는 MMR_score 최고점 순차 선택.
    TOP_K편 선택 (후보 < TOP_K편이면 있는 만큼).

    Args:
        state: RecommendationEngineState (hybrid_scores, candidate_movies 필요)

    Returns:
        dict: candidate_movies(재정렬된 list[CandidateMovie]) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    try:
        hybrid_scores = state.get("hybrid_scores", {})
        candidates = state.get("candidate_movies", [])

        if not candidates:
            return {"candidate_movies": []}

        # 후보를 dict로 변환 (빠른 조회용)
        candidate_map: dict[str, CandidateMovie] = {c.id: c for c in candidates}

        # ── 1단계: popular / hidden 풀 분리 ──
        # 평점 ≥5.0 또는 투표수 ≥50 을 충족하는 "검증된" 영화를 popular 풀로 분리.
        # 나머지(평점 0.0 무명작 포함)는 hidden gem 풀로 두어 최종 HIDDEN_SLOTS 개만 경쟁.
        # 이 분리가 필요한 이유: BM25 제목 매칭으로 올라온 평점 0.0 무명작이 hybrid_score=1.0
        # 으로 1순위 차지하는 구조를 원천 차단하기 위함. (사용자 피드백: "가끔 1~2개 정도
        # 무명작은 추천돼도 괜찮다")
        popular_ids: set[str] = set()
        hidden_ids: set[str] = set()
        for mid, movie in candidate_map.items():
            if _is_popular(movie):
                popular_ids.add(mid)
            else:
                hidden_ids.add(mid)

        # ── 2단계: 슬롯별 MMR 선택 ──
        selected: list[CandidateMovie] = []

        # 2-1) popular 풀에서 POPULAR_SLOTS 개 우선 선택
        popular_pick = _mmr_select(
            pool_ids=popular_ids,
            candidate_map=candidate_map,
            hybrid_scores=hybrid_scores,
            already_selected=selected,
            k=POPULAR_SLOTS,
        )
        selected.extend(popular_pick)

        # 2-2) hidden 풀에서 HIDDEN_SLOTS 개 선택 (이미 선택된 영화와의 다양성 고려)
        hidden_pick = _mmr_select(
            pool_ids=hidden_ids,
            candidate_map=candidate_map,
            hybrid_scores=hybrid_scores,
            already_selected=selected,
            k=HIDDEN_SLOTS,
        )
        selected.extend(hidden_pick)

        # 2-3) 총 TOP_K 미달 시 남은 풀에서 채움 (어느 풀이 모자랐든 대체).
        # 예) popular 1편 + hidden 2편만 있으면 → 3편만 반환되는 것을 방지.
        shortage = TOP_K - len(selected)
        if shortage > 0:
            leftover_ids = (popular_ids | hidden_ids)  # _mmr_select 가 이미 소비한 후 남은 것
            if leftover_ids:
                extra_pick = _mmr_select(
                    pool_ids=leftover_ids,
                    candidate_map=candidate_map,
                    hybrid_scores=hybrid_scores,
                    already_selected=selected,
                    k=shortage,
                )
                selected.extend(extra_pick)

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "diversity_reranked",
            original_count=len(candidates),
            popular_pool_size=len(popular_ids) + len(popular_pick),  # 소비 전 크기 복원
            hidden_pool_size=len(hidden_ids) + len(hidden_pick),
            popular_selected=[m.title for m in popular_pick],
            hidden_selected=[m.title for m in hidden_pick],
            selected_count=len(selected),
            selected_titles=[m.title for m in selected],
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"candidate_movies": selected}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("diversity_reranker_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1))
        # fallback: hybrid_score 내림차순으로 상위 TOP_K편 선택
        candidates = state.get("candidate_movies", [])
        hybrid_scores = state.get("hybrid_scores", {})
        sorted_candidates = sorted(
            candidates,
            key=lambda c: hybrid_scores.get(c.id, 0.0),
            reverse=True,
        )
        return {"candidate_movies": sorted_candidates[:TOP_K]}


# ============================================================
# 7. score_finalizer — 최종 점수 첨부 + RankedMovie 변환
# ============================================================

@traceable(
    name="score_finalizer",
    run_type="chain",
    metadata={"node": "rec_7/7", "subgraph": "recommendation_engine"},
)
async def score_finalizer(state: RecommendationEngineState) -> dict:
    """
    Top 5편을 선정하고, RankedMovie로 변환하며, ScoreDetail을 첨부한다.

    ScoreDetail 필드:
    - cf_score: cf_scores[movie_id] (Cold Start 시 0.0)
    - cbf_score: cbf_scores[movie_id] (Cold Start 시 0.0)
    - hybrid_score: hybrid_scores[movie_id]
    - genre_match: Jaccard(movie.genres, liked_genres)
    - mood_match: |movie.mood_tags ∩ user.mood_tags| / max(|movie.mood_tags|, 1)
    - similar_to: watch_history에서 같은 장르인 최근 시청 영화 제목 (최대 2개)

    Args:
        state: RecommendationEngineState (candidate_movies, cf_scores, cbf_scores,
               hybrid_scores, watch_history, mood_tags 필요)

    Returns:
        dict: ranked_movies(list[RankedMovie]) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    try:
        candidates = state.get("candidate_movies", [])
        cf_scores = state.get("cf_scores", {})
        cbf_scores = state.get("cbf_scores", {})
        hybrid_scores = state.get("hybrid_scores", {})
        watch_history = state.get("watch_history", [])
        mood_tags = state.get("mood_tags", [])
        preferences = state.get("preferences")

        if not candidates:
            return {"ranked_movies": []}

        # ── 사용자 명시 편수 우선 (2026-04-24) ──
        # "인생영화 한 편만 추천해줘" 같이 ExtractedPreferences.requested_count 가 채워지면
        # 해당 값으로 최종 편수를 제한한다. 미지정(None) 이면 기본 TOP_K(5).
        # 유효 범위 [1, TOP_K] 로 clamp — 프롬프트에서 1~5 만 허용하지만 방어적 처리.
        effective_top_k = TOP_K
        if preferences is not None and preferences.requested_count is not None:
            effective_top_k = max(1, min(TOP_K, preferences.requested_count))

        # diversity_reranker가 이미 TOP_K편을 선택했으므로 그 결과에서 effective_top_k 만 slice
        selected = candidates[:effective_top_k]

        # 사용자 선호 장르 추출
        liked_genres = _extract_liked_genres(watch_history, top_k=5)
        user_moods = set(mood_tags) if mood_tags else set()

        # ── RankedMovie 변환 ──
        ranked_movies: list[RankedMovie] = []

        for rank_idx, movie in enumerate(selected):
            mid = movie.id

            # ScoreDetail 구성
            cf_val = cf_scores.get(mid, 0.0)
            cbf_val = cbf_scores.get(mid, 0.0)
            hybrid_val = hybrid_scores.get(mid, 0.0)

            # 장르 일치도
            genre_match = _jaccard(set(movie.genres), liked_genres)

            # 무드 일치도 — 유저 무드 기준 비율 (CBF와 동일한 공식)
            movie_moods = set(movie.mood_tags) if movie.mood_tags else set()
            mood_match = (
                len(movie_moods & user_moods) / max(len(user_moods), 1)
                if user_moods
                else 0.0
            )

            # 유사 영화 (시청 이력에서 같은 장르인 최근 영화 제목, 최대 2개)
            similar_to = _find_similar_watched(movie, watch_history, max_count=2)

            score_detail = ScoreDetail(
                cf_score=round(cf_val, 4),
                cbf_score=round(cbf_val, 4),
                hybrid_score=round(hybrid_val, 4),
                genre_match=round(genre_match, 4),
                mood_match=round(mood_match, 4),
                similar_to=similar_to,
            )

            ranked_movie = RankedMovie(
                id=movie.id,
                title=movie.title,
                title_en=movie.title_en,
                genres=movie.genres,
                director=movie.director,
                cast=movie.cast,
                rating=movie.rating,
                release_year=movie.release_year,
                overview=movie.overview,
                mood_tags=movie.mood_tags,
                poster_path=movie.poster_path,
                ott_platforms=movie.ott_platforms,
                certification=movie.certification,
                trailer_url=movie.trailer_url,
                rank=rank_idx + 1,
                score_detail=score_detail,
                explanation="",  # explanation_generator가 채울 예정
                # ── 확장 메타데이터 필드 CandidateMovie → RankedMovie 복사 ──
                # SSE movie_card 이벤트를 통해 프론트엔드로 전달되므로 반드시 복사해야 한다.
                runtime=movie.runtime,
                popularity_score=movie.popularity_score,
                vote_count=movie.vote_count,
                backdrop_path=movie.backdrop_path,
            )
            ranked_movies.append(ranked_movie)

        # 최종 랭킹 전체 영화 상세 로깅
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "score_finalized",
            ranked_count=len(ranked_movies),
            ranked_details=[
                {
                    "rank": m.rank,
                    "title": m.title,
                    "genres": m.genres[:3],
                    "cf_score": round(m.score_detail.cf_score, 4),
                    "cbf_score": round(m.score_detail.cbf_score, 4),
                    "hybrid_score": round(m.score_detail.hybrid_score, 4),
                    "genre_match": round(m.score_detail.genre_match, 4),
                    "mood_match": round(m.score_detail.mood_match, 4),
                    "similar_to": m.score_detail.similar_to,
                }
                for m in ranked_movies
            ],
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"ranked_movies": ranked_movies}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("score_finalizer_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1))
        # fallback: CandidateMovie를 RankedMovie로 최소 변환 (requested_count 존중)
        candidates = state.get("candidate_movies", [])
        preferences = state.get("preferences")
        fallback_top_k = TOP_K
        if preferences is not None and preferences.requested_count is not None:
            fallback_top_k = max(1, min(TOP_K, preferences.requested_count))
        ranked = [
            RankedMovie(
                id=c.id,
                title=c.title,
                title_en=c.title_en,
                genres=c.genres,
                director=c.director,
                cast=c.cast,
                rating=c.rating,
                release_year=c.release_year,
                overview=c.overview,
                mood_tags=c.mood_tags,
                poster_path=c.poster_path,
                ott_platforms=c.ott_platforms,
                certification=c.certification,
                trailer_url=c.trailer_url,
                rank=i + 1,
                score_detail=ScoreDetail(hybrid_score=c.rrf_score),
                explanation="",
                # ── 확장 메타데이터 필드 CandidateMovie → RankedMovie 복사 (fallback 경로) ──
                runtime=c.runtime,
                popularity_score=c.popularity_score,
                vote_count=c.vote_count,
                backdrop_path=c.backdrop_path,
            )
            for i, c in enumerate(candidates[:fallback_top_k])
        ]
        return {"ranked_movies": ranked}


# ============================================================
# 유틸리티 함수
# ============================================================

def _jaccard(set_a: set, set_b: set) -> float:
    """
    두 집합의 Jaccard 유사도를 계산한다.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set_a: 집합 A
        set_b: 집합 B

    Returns:
        Jaccard 유사도 (0.0~1.0), 양쪽 모두 빈 집합이면 0.0
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """
    점수 딕셔너리를 min-max 정규화하여 [0, 1] 범위로 변환한다.

    모든 값이 동일하면 0.5를 반환한다.

    Args:
        scores: {id: score} 딕셔너리

    Returns:
        정규화된 {id: score} 딕셔너리
    """
    if not scores:
        return scores

    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)

    # 모든 값이 동일하면 0.5
    if max_val - min_val < 1e-9:
        return {k: 0.5 for k in scores}

    return {
        k: (v - min_val) / (max_val - min_val)
        for k, v in scores.items()
    }


def _temporal_weight(watched_at: Any) -> float:
    """
    시청 시점 기반 시간 감쇠 가중치 (Phase 0-3).

    최근 시청일수록 높은 가중치를 부여하여 현재 취향을 더 잘 반영한다.
    - 3개월 이내: 1.0 (감쇠 없음)
    - 3~6개월: 0.75
    - 6개월 이상: 0.5

    Args:
        watched_at: 시청 일시 (datetime 또는 ISO 문자열)

    Returns:
        시간 감쇠 가중치 (0.5 ~ 1.0)
    """
    if not watched_at:
        return 1.0
    try:
        if isinstance(watched_at, str):
            watched_at = datetime.fromisoformat(watched_at)
        age_days = (datetime.now() - watched_at).days
        if age_days > 180:  # 6개월 이상
            return 0.5
        elif age_days > 90:  # 3~6개월
            return 0.75
        return 1.0
    except (ValueError, TypeError):
        return 1.0


def _rating_weight(rating: Any) -> float:
    """
    사용자 평점 기반 가중치 (Phase 0-2).

    높은 평점의 영화에서 추출한 장르/감독/배우를 더 중요하게 반영한다.
    - 4.0+ (5점 만점): 1.5배 (강한 선호)
    - 3.0~4.0: 1.0배 (보통)
    - 3.0 미만: 0.5배 (비선호, 장르 오염 방지)
    - 평점 없음: 1.0배 (중립)

    Args:
        rating: 사용자 평점 (1.0~5.0, nullable)

    Returns:
        평점 기반 가중치 (0.5 ~ 1.5)
    """
    if rating is None:
        return 1.0
    try:
        rating = float(rating)
        if rating >= 4.0:
            return 1.5
        elif rating < 3.0:
            return 0.5
        return 1.0
    except (ValueError, TypeError):
        return 1.0


def _extract_liked_genres(
    watch_history: list[dict[str, Any]],
    top_k: int = 5,
) -> set[str]:
    """
    시청 이력에서 가중치 기반 상위 top_k개 선호 장르를 추출한다.

    Phase 0 개선: 평점(rating)과 시간 감쇠(temporal decay)를 반영하여
    최근에 높은 평점을 준 영화의 장르를 더 중요하게 취급한다.

    Args:
        watch_history: MySQL 시청 이력 리스트 (genres, rating, watched_at 포함)
        top_k: 상위 장르 수

    Returns:
        선호 장르 set
    """
    genre_counter: Counter[str] = Counter()

    for wh in watch_history:
        genres = wh.get("genres")
        if isinstance(genres, str):
            # JSON 문자열 파싱 (context_loader에서 이미 파싱하지만 안전 장치)
            try:
                import json
                parsed = json.loads(genres)
                genres = parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                genres = []

        if not isinstance(genres, list) or not genres:
            continue

        # Phase 0-2: 평점 가중치 + Phase 0-3: 시간 감쇠 가중치
        weight = _rating_weight(wh.get("rating")) * _temporal_weight(wh.get("watched_at"))

        for g in genres:
            genre_counter[g] += weight

    if not genre_counter:
        return set()

    return {genre for genre, _ in genre_counter.most_common(top_k)}


def _extract_crew_frequency(
    watch_history: list[dict[str, Any]],
) -> tuple[Counter[str], Counter[str]]:
    """
    시청 이력에서 감독/배우 출현 빈도를 가중치 기반으로 추출한다.

    Phase 0 개선: 평점(rating)과 시간 감쇠(temporal decay)를 반영하여
    높은 평점 + 최근 시청한 영화의 감독/배우에 더 높은 빈도를 부여한다.

    Args:
        watch_history: MySQL 시청 이력 리스트 (director, cast, rating, watched_at 포함)

    Returns:
        (director_frequency, actor_frequency) Counter 튜플
    """
    director_freq: Counter[str] = Counter()
    actor_freq: Counter[str] = Counter()

    for wh in watch_history:
        # Phase 0-2 + 0-3: 평점 × 시간 감쇠 가중치
        weight = _rating_weight(wh.get("rating")) * _temporal_weight(wh.get("watched_at"))

        director = wh.get("director")
        if director and isinstance(director, str):
            director_freq[director] += weight

        cast = wh.get("cast")
        if isinstance(cast, str):
            try:
                import json
                parsed = json.loads(cast)
                cast = parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                cast = []
        if isinstance(cast, list):
            for actor in cast:
                actor_freq[actor] += weight

    return director_freq, actor_freq


def _crew_match_score(
    movie: CandidateMovie,
    director_freq: Counter[str],
    actor_freq: Counter[str],
) -> float:
    """
    영화의 감독/배우가 시청 이력에서 얼마나 자주 등장하는지 점수를 계산한다.

    감독 점수: 빈도 / (전체 빈도 합 + 1) (정규화)
    배우 점수: 상위 3명의 빈도 합 / (전체 빈도 합 + 1) × decay(0.9^순서)

    Args:
        movie: 후보 영화
        director_freq: 감독 빈도 Counter
        actor_freq: 배우 빈도 Counter

    Returns:
        감독/배우 매칭 점수 (0.0~1.0)
    """
    total_dir = sum(director_freq.values()) + 1
    total_act = sum(actor_freq.values()) + 1

    # 감독 점수
    dir_score = 0.0
    if movie.director and movie.director in director_freq:
        dir_score = director_freq[movie.director] / total_dir

    # 배우 점수 (상위 3명, decay 0.9^순서)
    act_score = 0.0
    if movie.cast:
        for order, actor in enumerate(movie.cast[:3]):
            if actor in actor_freq:
                decay = 0.9 ** order
                act_score += (actor_freq[actor] / total_act) * decay

    # 감독 40%, 배우 60% 합산 (감독이 더 결정적이지만 배우가 더 다양)
    return min(dir_score * 0.4 + act_score * 0.6, 1.0)


def _find_similar_watched(
    movie: CandidateMovie,
    watch_history: list[dict[str, Any]],
    max_count: int = 2,
) -> list[str]:
    """
    시청 이력에서 추천 영화와 같은 장르인 최근 시청 영화 제목을 찾는다.

    Args:
        movie: 추천 영화
        watch_history: MySQL 시청 이력 리스트
        max_count: 최대 반환 수

    Returns:
        유사 시청 영화 제목 목록 (최대 max_count개)
    """
    if not movie.genres or not watch_history:
        return []

    movie_genres = set(movie.genres)
    similar: list[str] = []

    for wh in watch_history:
        title = wh.get("title", "")
        if not title:
            continue

        wh_genres = wh.get("genres")
        if isinstance(wh_genres, list):
            wh_genre_set = set(wh_genres)
        elif isinstance(wh_genres, str):
            try:
                import json
                parsed = json.loads(wh_genres)
                wh_genre_set = set(parsed) if isinstance(parsed, list) else set()
            except (json.JSONDecodeError, TypeError):
                wh_genre_set = set()
        else:
            wh_genre_set = set()

        # 장르 겹침이 있으면 유사 영화로 판정
        if movie_genres & wh_genre_set:
            similar.append(title)
            if len(similar) >= max_count:
                break

    return similar
