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

import time
import traceback
from collections import Counter
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

            # Redis에 유사 유저가 없으면 캐시 미스 → 기본값 + 명시적 플래그
            if not similar_users_raw:
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
            # 감정 없음: 무드 가중치 0, 나머지 재분배
            w_genre = 0.30
            w_crew = 0.24
            w_mood = 0.00
            w_keyword = 0.16
            w_rrf = 0.30

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
            rating_score = min(movie.rating / 10.0, 1.0) if movie.rating > 0 else 0.3

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

            # 인기도 점수 (vote_count 기반): 가중치 15%
            # popularity_score가 metadata에 있으면 사용, 없으면 rating 기반 추정
            popularity_score = 0.5  # 기본값

            # 가중 합산 (총 100%)
            hybrid_scores[movie.id] = (
                0.40 * rating_score
                + 0.20 * genre_score
                + 0.10 * mood_score
                + 0.15 * rrf_score
                + 0.15 * popularity_score
            )

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

        # 선택 가능한 후보 ID 집합
        remaining = set(candidate_map.keys())
        selected: list[CandidateMovie] = []

        # Greedy 선택 루프
        select_count = min(TOP_K, len(candidates))

        for i in range(select_count):
            if not remaining:
                break

            if i == 0:
                # 첫 영화: 최고 hybrid_score 선택
                best_id = max(remaining, key=lambda mid: hybrid_scores.get(mid, 0.0))
            else:
                # MMR_score 계산: λ × relevance - (1 - λ) × max_genre_sim
                best_id = None
                best_mmr = float("-inf")

                for mid in remaining:
                    movie = candidate_map[mid]
                    relevance = hybrid_scores.get(mid, 0.0)

                    # 이미 선택된 영화와의 최대 유사도 (장르 60% + 감독 25% + 배우 15%)
                    # 기존: 장르 Jaccard만 사용 → 같은 감독/배우 영화가 중복 추천
                    # 수정: 감독/배우까지 포함하여 다차원 다양성 확보
                    max_sim = 0.0
                    for sel in selected:
                        genre_sim = _jaccard(set(movie.genres), set(sel.genres))
                        director_sim = 1.0 if (movie.director and movie.director == sel.director) else 0.0
                        cast_overlap = (
                            len(set(movie.cast[:3]) & set(sel.cast[:3])) / 3.0
                            if movie.cast and sel.cast else 0.0
                        )
                        sim = 0.60 * genre_sim + 0.25 * director_sim + 0.15 * cast_overlap
                        if sim > max_sim:
                            max_sim = sim

                    mmr_score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * max_sim

                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_id = mid

            if best_id is not None:
                selected.append(candidate_map[best_id])
                remaining.discard(best_id)

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "diversity_reranked",
            original_count=len(candidates),
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

        if not candidates:
            return {"ranked_movies": []}

        # diversity_reranker가 이미 TOP_K편을 선택했으므로 그대로 사용
        selected = candidates[:TOP_K]

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
        # fallback: CandidateMovie를 RankedMovie로 최소 변환
        candidates = state.get("candidate_movies", [])
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
            )
            for i, c in enumerate(candidates[:TOP_K])
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


def _extract_liked_genres(
    watch_history: list[dict[str, Any]],
    top_k: int = 5,
) -> set[str]:
    """
    시청 이력에서 빈도 상위 top_k개 장르를 추출한다.

    watch_history 각 항목에 'genres' 키가 있을 수 있다 (JSON 파싱).
    없으면 빈 set 반환.

    Args:
        watch_history: MySQL 시청 이력 리스트
        top_k: 상위 장르 수

    Returns:
        선호 장르 set
    """
    genre_counter: Counter[str] = Counter()

    for wh in watch_history:
        genres = wh.get("genres")
        if isinstance(genres, list):
            genre_counter.update(genres)
        elif isinstance(genres, str):
            # JSON 문자열일 수 있음
            try:
                import json
                parsed = json.loads(genres)
                if isinstance(parsed, list):
                    genre_counter.update(parsed)
            except (json.JSONDecodeError, TypeError):
                pass

    if not genre_counter:
        return set()

    return {genre for genre, _ in genre_counter.most_common(top_k)}


def _extract_crew_frequency(
    watch_history: list[dict[str, Any]],
) -> tuple[Counter[str], Counter[str]]:
    """
    시청 이력에서 감독/배우 출현 빈도를 추출한다.

    Args:
        watch_history: MySQL 시청 이력 리스트

    Returns:
        (director_frequency, actor_frequency) Counter 튜플
    """
    director_freq: Counter[str] = Counter()
    actor_freq: Counter[str] = Counter()

    for wh in watch_history:
        director = wh.get("director")
        if director and isinstance(director, str):
            director_freq[director] += 1

        cast = wh.get("cast")
        if isinstance(cast, list):
            actor_freq.update(cast)
        elif isinstance(cast, str):
            try:
                import json
                parsed = json.loads(cast)
                if isinstance(parsed, list):
                    actor_freq.update(parsed)
            except (json.JSONDecodeError, TypeError):
                pass

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
