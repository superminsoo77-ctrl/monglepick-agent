"""
Movie Match Agent 노드 함수 (§21-3 노드 1~6).

LangGraph StateGraph의 각 노드로 등록되는 6개 async 함수.
시그니처: async def node_name(state: MovieMatchState) -> dict

모든 노드는 try/except로 감싸고, 에러 시 유효한 기본값을 반환한다 (에러 전파 금지).
반환값은 dict — LangGraph 컨벤션 (TypedDict State 일부 업데이트).

노드 목록:
1. movie_loader         — 두 영화 메타데이터+벡터 로드 (Qdrant → MySQL fallback)
2. feature_extractor    — 교집합 특성 추출 + LLM 유사성 요약 생성
3. query_builder        — 공통 특성 기반 RAG 검색 쿼리 구성 (규칙 기반, LLM 없음)
4. rag_retriever        — 하이브리드 검색 (Qdrant+ES+Neo4j → RRF k=60) + 임베딩 일괄 조회
5. match_scorer         — min(simA, simB) 스코어링 + MMR 다양성 리랭킹 → Top 5
6. explanation_generator — "두 사람 모두 좋아할 이유" 배치 생성
"""

from __future__ import annotations

import asyncio
import math
import time
import traceback
from typing import Any

import structlog
from langsmith import traceable

from monglepick.agents.match.models import (
    MatchScoreDetail,
    MatchedMovie,
    MovieMatchState,
    SharedFeatures,
    calculate_match_score,
    jaccard,
)
from monglepick.api.match_cowatch_client import fetch_cowatched_candidates
from monglepick.chains.match_explanation_chain import (
    generate_match_explanations_batch,
    generate_similarity_summary,
)
from monglepick.chains.match_llm_reranker_chain import rerank_match_candidates
from monglepick.config import settings
from monglepick.db.clients import get_mysql, get_qdrant
from monglepick.rag.hybrid_search import hybrid_search, reciprocal_rank_fusion
from monglepick.utils.qdrant_helpers import to_point_id

logger = structlog.get_logger()


def _compute_embedding_centroid(
    vec_a: list[float] | None,
    vec_b: list[float] | None,
) -> list[float] | None:
    """
    두 영화 임베딩 벡터의 centroid(평균 + L2 정규화)를 계산한다.

    Movie Match 의 "둘의 공통점 찾기" 검색에서 Qdrant 벡터 검색을 수행할 때
    텍스트를 재임베딩하는 대신, 두 영화 벡터의 중간 지점으로 직접 검색하기 위해 사용.
    Qdrant 컬렉션이 cosine distance 를 사용하므로 결과 벡터를 L2 정규화한다.

    두 벡터 중 하나라도 None/빈 리스트면 None 반환 → 호출자가 텍스트 재임베딩으로 fallback.

    Args:
        vec_a: 영화 A 임베딩 (4096차원 Upstage Solar). None/빈 리스트 가능.
        vec_b: 영화 B 임베딩 (동일 차원). None/빈 리스트 가능.

    Returns:
        정규화된 centroid 벡터 (list[float]) 또는 None.
    """
    if not vec_a or not vec_b:
        return None
    if len(vec_a) != len(vec_b):
        logger.warning(
            "centroid_dim_mismatch",
            dim_a=len(vec_a),
            dim_b=len(vec_b),
        )
        return None

    # 방어적 float 변환 + 평균 계산
    try:
        centroid = [(float(a) + float(b)) / 2.0 for a, b in zip(vec_a, vec_b)]
    except (TypeError, ValueError):
        return None

    # L2 정규화 (Qdrant cosine 검색과 정합)
    norm = math.sqrt(sum(x * x for x in centroid))
    if norm <= 1e-9:
        return None  # 영벡터 가드
    return [x / norm for x in centroid]


def _merge_unique_results(
    primary: list[Any],
    secondary: list[Any],
) -> list[Any]:
    """
    두 하이브리드 검색 결과 리스트를 movie_id 기준으로 중복 제거하여 병합한다.

    완화 재검색 단계에서 1차 결과를 유지하면서 새 결과를 덧붙인다.
    이미 포함된 영화는 score/rank 를 유지하고 재등장한 영화는 무시한다.
    단계별 완화가 누적되도록 설계되어 상위 단계의 고품질 결과가 보존된다.

    Args:
        primary: 1차(또는 기존) 결과 리스트. 우선순위 유지.
        secondary: 2차(또는 완화된) 결과 리스트. primary 에 없는 것만 추가.

    Returns:
        중복 제거된 병합 결과 (primary 순서 유지 + secondary 신규 append).
    """
    if not primary:
        return list(secondary)
    if not secondary:
        return list(primary)

    seen_ids = {r.movie_id for r in primary}
    merged = list(primary)
    for r in secondary:
        if r.movie_id not in seen_ids:
            merged.append(r)
            seen_ids.add(r.movie_id)
    return merged


def _payload_to_movie_dict(payload: dict[str, Any], point_id: Any) -> dict[str, Any]:
    """
    Qdrant 포인트 payload를 노드에서 사용하는 movie dict로 변환한다.

    Qdrant payload 키 이름이 표준 movie dict 키와 다를 수 있으므로
    안전하게 매핑한다. embedding 벡터는 별도 인자로 추가한다.

    Args:
        payload : Qdrant point payload dict
        point_id: Qdrant point ID (로그용)

    Returns:
        정규화된 movie dict (embedding 키는 별도로 추가 필요)
    """
    return {
        "id": payload.get("movie_id") or payload.get("id") or str(point_id),
        "title": payload.get("title", ""),
        "title_en": payload.get("title_en", ""),
        "genres": payload.get("genres", []),
        "mood_tags": payload.get("mood_tags", []),
        "keywords": payload.get("keywords", []),
        "director": payload.get("director", ""),
        "cast": payload.get("cast", []),
        "cast_members": payload.get("cast", []),  # MatchedMovie 호환 키 별칭
        "release_year": payload.get("release_year"),
        "rating": payload.get("rating"),
        "poster_path": payload.get("poster_path"),
        "overview": payload.get("overview", ""),
        "ott_platforms": payload.get("ott_platforms", []),
        "certification": payload.get("certification", ""),
        "trailer_url": payload.get("trailer_url", ""),
        "popularity_score": payload.get("popularity_score", 0.0),
    }


# ============================================================
# 노드 1: movie_loader — 두 영화 메타데이터+벡터 로드
# ============================================================

@traceable(name="match_movie_loader", run_type="chain", metadata={"node": "1/6", "db": "qdrant+mysql"})
async def movie_loader(state: MovieMatchState) -> dict:
    """
    두 영화의 전체 메타데이터와 임베딩 벡터를 Qdrant에서 로드한다.

    조회 전략:
    1. Qdrant client.retrieve(ids, with_vectors=True) 로 벡터+payload 일괄 조회
    2. Qdrant 미발견 시 MySQL movies 테이블 fallback (벡터 없음)
    3. 두 영화 모두 조회 실패 시 error 필드에 MOVIE_NOT_FOUND 메시지 반환

    Args:
        state: MovieMatchState (movie_id_1, movie_id_2 필수)

    Returns:
        {"movie_1": dict, "movie_2": dict}         — 정상 시
        {"error": "MOVIE_NOT_FOUND:...", ...}       — 에러 시 (그래프 종료)
    """
    node_start = time.perf_counter()
    movie_id_1 = state.get("movie_id_1", "")
    movie_id_2 = state.get("movie_id_2", "")

    logger.info(
        "match_movie_loader_start",
        movie_id_1=movie_id_1,
        movie_id_2=movie_id_2,
    )

    try:
        # ── [1] Qdrant에서 두 영화 일괄 조회 (with_vectors=True) ──
        client = await get_qdrant()
        point_id_1 = to_point_id(movie_id_1)
        point_id_2 = to_point_id(movie_id_2)

        # 두 포인트를 한 번의 API 호출로 조회 (네트워크 왕복 최소화)
        points = await client.retrieve(
            collection_name=settings.QDRANT_COLLECTION,
            ids=[point_id_1, point_id_2],
            with_vectors=True,    # 유사도 계산을 위해 임베딩 벡터 포함
            with_payload=True,    # 메타데이터 포함
        )

        # 조회된 포인트를 point_id → point 매핑으로 변환
        point_map: dict[Any, Any] = {p.id: p for p in points}

        def _load_from_qdrant(movie_id: str, point_id: Any) -> dict[str, Any] | None:
            """Qdrant 조회 결과에서 영화 dict를 구성한다."""
            point = point_map.get(point_id)
            if point is None:
                return None
            movie = _payload_to_movie_dict(point.payload or {}, point_id)
            # movie_id를 원본 문자열로 덮어쓰기 (Qdrant ID와 다를 수 있음)
            movie["id"] = movie_id
            # 임베딩 벡터 추가 (유사도 계산의 핵심)
            if point.vector is not None:
                # Qdrant는 벡터를 list[float] 또는 dict 형태로 반환
                vec = point.vector
                movie["embedding"] = vec if isinstance(vec, list) else list(vec.values())[0]
            else:
                movie["embedding"] = None
            return movie

        movie_1 = _load_from_qdrant(movie_id_1, point_id_1)
        movie_2 = _load_from_qdrant(movie_id_2, point_id_2)

        # ── [2] Qdrant 미발견 시 MySQL fallback ──
        # 재적재 전 데이터가 Qdrant에 없을 수 있으므로 MySQL을 백업으로 사용
        missing_ids: list[str] = []
        if movie_1 is None:
            missing_ids.append(movie_id_1)
        if movie_2 is None:
            missing_ids.append(movie_id_2)

        if missing_ids:
            logger.warning(
                "match_movie_loader_qdrant_miss",
                missing_ids=missing_ids,
                fallback="mysql",
            )
            try:
                pool = await get_mysql()
                async with pool.acquire() as conn:
                    import aiomysql
                    async with conn.cursor(aiomysql.DictCursor) as cursor:
                        placeholders = ", ".join(["%s"] * len(missing_ids))
                        await cursor.execute(
                            f"""
                            SELECT movie_id AS id, title, title_en, genres,
                                   director, rating, release_year, overview,
                                   poster_path, certification, trailer_url
                            FROM movies
                            WHERE movie_id IN ({placeholders})
                            """,
                            tuple(missing_ids),
                        )
                        rows = await cursor.fetchall()
                        mysql_map: dict[str, dict] = {}
                        for row in rows:
                            r = dict(row)
                            mid = r.get("id", "")
                            # MySQL genres 필드가 JSON 문자열일 수 있으므로 파싱
                            raw_genres = r.get("genres", "[]")
                            if isinstance(raw_genres, str):
                                import json
                                try:
                                    r["genres"] = json.loads(raw_genres)
                                except Exception:
                                    r["genres"] = []
                            r["embedding"] = None   # MySQL에는 벡터 없음
                            r["mood_tags"] = []
                            r["keywords"] = []
                            r["cast"] = []
                            r["cast_members"] = []
                            r["ott_platforms"] = []
                            mysql_map[mid] = r

                        # MySQL 결과로 None 필드 채우기
                        if movie_1 is None:
                            movie_1 = mysql_map.get(movie_id_1)
                        if movie_2 is None:
                            movie_2 = mysql_map.get(movie_id_2)

                        # MySQL fallback 영화는 embedding/mood/keyword 없음 → 스코어링 품질 저하 경고
                        mysql_loaded = [mid for mid in missing_ids if mid in mysql_map]
                        if mysql_loaded:
                            logger.warning(
                                "match_movie_loader_mysql_fallback_quality",
                                mysql_loaded_ids=mysql_loaded,
                                detail="MySQL fallback 영화는 embedding/mood_tags/keywords가 없어 "
                                       "유사도 계산 시 가중치가 재정규화됩니다 (장르 기반으로 축소).",
                            )
            except Exception as db_err:
                logger.warning(
                    "match_movie_loader_mysql_error",
                    error=str(db_err),
                    missing_ids=missing_ids,
                )

        # ── [3] 최종 조회 실패 시 에러 반환 ──
        not_found: list[str] = []
        if movie_1 is None:
            not_found.append(movie_id_1)
        if movie_2 is None:
            not_found.append(movie_id_2)

        if not_found:
            elapsed_ms = (time.perf_counter() - node_start) * 1000
            error_msg = f"MOVIE_NOT_FOUND:{', '.join(not_found)}"
            logger.error(
                "match_movie_loader_not_found",
                not_found=not_found,
                elapsed_ms=round(elapsed_ms, 1),
            )
            return {"error": error_msg}

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "match_movie_loader_complete",
            movie_1_title=movie_1.get("title", ""),
            movie_2_title=movie_2.get("title", ""),
            movie_1_has_vector=movie_1.get("embedding") is not None,
            movie_2_has_vector=movie_2.get("embedding") is not None,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return {"movie_1": movie_1, "movie_2": movie_2}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "match_movie_loader_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            elapsed_ms=round(elapsed_ms, 1),
        )
        # ── 인프라 장애 vs 비즈니스 에러 구분 ──
        # ConnectionError, TimeoutError 등 인프라 예외는 SERVICE_UNAVAILABLE로 분리하여
        # 모니터링/알림 시스템이 MOVIE_NOT_FOUND(정상)와 구분할 수 있도록 한다.
        # qdrant_client의 통신 에러(httpx 기반)도 인프라 장애로 분류한다.
        infra_errors = (ConnectionError, TimeoutError, OSError)
        error_type_name = type(e).__name__
        infra_keywords = ("connect", "timeout", "unreachable", "refused", "reset")
        is_infra = (
            isinstance(e, infra_errors)
            or any(kw in error_type_name.lower() for kw in infra_keywords)
            or any(kw in str(e).lower() for kw in infra_keywords)
        )

        if is_infra:
            return {"error": f"SERVICE_UNAVAILABLE:{error_type_name} - {str(e)[:100]}"}
        return {"error": f"MOVIE_NOT_FOUND:{movie_id_1},{movie_id_2}"}


# ============================================================
# 노드 2: feature_extractor — 교집합 특성 추출 + 유사성 요약
# ============================================================

@traceable(name="match_feature_extractor", run_type="chain", metadata={"node": "2/6"})
async def feature_extractor(state: MovieMatchState) -> dict:
    """
    두 영화의 공통 특성(교집합)을 추출하고 LLM으로 유사성 요약을 생성한다.

    set intersection으로 공통 장르/무드/키워드/감독/배우를 계산하고,
    EXAONE 32B로 1~2문장 유사성 요약을 생성하여 SharedFeatures를 구성한다.
    SSE shared_features 이벤트로 프론트엔드에 즉시 전달된다.

    Args:
        state: MovieMatchState (movie_1, movie_2 필수)

    Returns:
        {"shared_features": SharedFeatures}
    """
    node_start = time.perf_counter()
    movie_1: dict = state.get("movie_1", {})
    movie_2: dict = state.get("movie_2", {})

    logger.info(
        "match_feature_extractor_start",
        title_1=movie_1.get("title", ""),
        title_2=movie_2.get("title", ""),
    )

    try:
        # ── [1] set intersection으로 공통 특성 계산 ──

        # 공통 장르
        genres_1 = set(movie_1.get("genres", []))
        genres_2 = set(movie_2.get("genres", []))
        common_genres = sorted(genres_1 & genres_2)

        # 공통 무드태그
        moods_1 = set(movie_1.get("mood_tags", []))
        moods_2 = set(movie_2.get("mood_tags", []))
        common_moods = sorted(moods_1 & moods_2)

        # 공통 키워드
        kw_1 = set(movie_1.get("keywords", []))
        kw_2 = set(movie_2.get("keywords", []))
        common_keywords = sorted(kw_1 & kw_2)

        # 공통 감독 (동일 감독이 두 영화 모두 연출한 경우)
        dir_1 = movie_1.get("director", "")
        dir_2 = movie_2.get("director", "")
        common_directors: list[str] = []
        if dir_1 and dir_2 and dir_1 == dir_2:
            common_directors = [dir_1]

        # 공통 출연진 (두 영화 모두 출연한 배우)
        cast_1 = set(movie_1.get("cast", []) or movie_1.get("cast_members", []))
        cast_2 = set(movie_2.get("cast", []) or movie_2.get("cast_members", []))
        common_cast = sorted(cast_1 & cast_2)

        # ── [2] 개봉연도 범위 (±5년 확장) ──
        year_1 = movie_1.get("release_year") or 2000
        year_2 = movie_2.get("release_year") or 2000
        era_range = (min(year_1, year_2) - 5, max(year_1, year_2) + 5)

        # ── [3] 평균 평점 ──
        rating_1 = float(movie_1.get("rating") or 0.0)
        rating_2 = float(movie_2.get("rating") or 0.0)
        avg_rating = round((rating_1 + rating_2) / 2, 2)

        # ── [4] LLM으로 유사성 요약 생성 ──
        # 실패해도 SharedFeatures는 정상 생성 (summary만 규칙 기반 fallback)
        similarity_summary = await generate_similarity_summary(
            movie_1=movie_1,
            movie_2=movie_2,
            common_genres=common_genres,
            common_moods=common_moods,
        )

        # SharedFeatures 구성
        shared_features = SharedFeatures(
            common_genres=common_genres,
            common_moods=common_moods,
            common_keywords=common_keywords[:10],  # 최대 10개로 제한
            common_directors=common_directors,
            common_cast=common_cast[:5],            # 최대 5명으로 제한
            era_range=era_range,
            avg_rating=avg_rating,
            similarity_summary=similarity_summary,
        )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "match_feature_extractor_complete",
            common_genres=common_genres,
            common_moods=common_moods,
            common_keywords_count=len(common_keywords),
            common_cast_count=len(common_cast),
            avg_rating=avg_rating,
            summary_preview=similarity_summary[:60],
            elapsed_ms=round(elapsed_ms, 1),
        )

        return {"shared_features": shared_features}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "match_feature_extractor_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            elapsed_ms=round(elapsed_ms, 1),
        )
        # 에러 시 빈 SharedFeatures 반환 (그래프 계속 진행)
        return {"shared_features": SharedFeatures()}


# ============================================================
# 노드 3: query_builder — 공통 특성 기반 RAG 검색 쿼리 구성
# ============================================================

@traceable(name="match_query_builder", run_type="chain", metadata={"node": "3/6"})
async def query_builder(state: MovieMatchState) -> dict:
    """
    공통 특성을 기반으로 RAG 하이브리드 검색 쿼리를 규칙 기반으로 구성한다.

    LLM을 사용하지 않고 SharedFeatures에서 직접 쿼리를 생성하므로
    빠르게 실행된다. hybrid_search() 함수의 파라미터 형식에 맞게 구성한다.

    쿼리 구성 규칙:
    - semantic_query: 공통 장르 + 공통 무드 + similarity_summary 결합
    - genre_filter: 공통 장르 (비어있으면 두 영화 합집합으로 확장)
    - mood_filter: 공통 무드 (비어있으면 필터 없음)
    - year_range: era_range 적용
    - min_rating: avg_rating * 0.7 (최소 품질 보장)
    - exclude_ids: [movie_id_1, movie_id_2] (입력 영화 제외)
    - top_k: 20 (후보 풀 확보)

    Args:
        state: MovieMatchState (shared_features, movie_1, movie_2 필수)

    Returns:
        {"search_query": dict}
    """
    node_start = time.perf_counter()
    shared: SharedFeatures = state.get("shared_features") or SharedFeatures()
    movie_1: dict = state.get("movie_1", {})
    movie_2: dict = state.get("movie_2", {})

    try:
        # ── [1] semantic_query 구성 ──
        # 공통 장르 + 공통 무드 + similarity_summary를 자연어로 결합
        query_parts: list[str] = []

        if shared.common_genres:
            query_parts.append(" ".join(shared.common_genres[:3]))
        if shared.common_moods:
            query_parts.append(" ".join(shared.common_moods[:3]))
        if shared.similarity_summary:
            query_parts.append(shared.similarity_summary[:100])
        if shared.common_keywords:
            query_parts.append(" ".join(shared.common_keywords[:5]))

        # 쿼리가 비어있으면 두 영화 제목으로 기본 쿼리 구성
        semantic_query = " ".join(query_parts).strip()
        if not semantic_query:
            semantic_query = (
                f"{movie_1.get('title', '')} {movie_2.get('title', '')} 비슷한 영화"
            ).strip()

        # ── [2] 장르 필터 (Level 1-B 개선: 교집합 대신 합집합 기본) ──
        # 기존: 교집합 우선 → 매우 다른 두 영화에서 공약수 1개만 남아 검색이 허술해짐
        # 개선: 항상 합집합 사용 — centroid 벡터 검색이 의미적 유사성을 보장하므로
        #       필터는 "넓게 범위 확보" 역할만 담당. 최대 6개(교집합 * 2배)까지 확장.
        union_genres = list(
            set(movie_1.get("genres", [])) | set(movie_2.get("genres", []))
        )
        if shared.common_genres and len(shared.common_genres) >= 2:
            # 교집합이 2개 이상이면 (두 영화가 본질적으로 유사) 교집합 사용
            genre_filter = shared.common_genres[:5]
        else:
            # 교집합이 0~1개이면 합집합 사용 (다양성 확보)
            genre_filter = union_genres[:6] if union_genres else None

        # ── [3] 무드 필터 (Level 1-B 개선: 1단계 기본 비활성화) ──
        # 공통 무드는 교집합이 매우 sparse 하고, 과도하게 specific 한 태그가 많아
        # 1차 검색에서는 제약으로 사용하지 않는다. match_scorer 의 mood_overlap 스코어에
        # 반영되므로 "필터"보다 "스코어링"에서 가중치를 주는 것이 합리적.
        mood_filter = None

        # ── [4] 연도 범위 필터 (Level 1-B 개선: ±5 → ±10 확장) ──
        # era_range 는 이미 (min-5, max+5) 로 계산되어 있으므로 추가 패딩으로 ±5 더 확장
        era_min, era_max = shared.era_range
        year_range = (max(1900, era_min - 5), min(2030, era_max + 5))

        # ── [5] 최소 평점 필터 (Level 1-B 개선: 0.7 → 0.6 완화) ──
        # centroid 벡터 검색이 품질 필터 역할을 일부 담당하므로 평점 임계값을 낮춘다.
        min_rating = round(shared.avg_rating * 0.6, 1) if shared.avg_rating > 0 else None

        # ── [6] 입력 영화 제외 ID 목록 ──
        movie_id_1 = state.get("movie_id_1", "")
        movie_id_2 = state.get("movie_id_2", "")
        exclude_ids = [mid for mid in [movie_id_1, movie_id_2] if mid]

        # ── [7] 검색 쿼리 dict 구성 (hybrid_search 파라미터 형식) ──
        search_query: dict[str, Any] = {
            "semantic_query": semantic_query,
            "genre_filter": genre_filter or None,
            "mood_filter": mood_filter,
            "year_range": year_range,
            "min_rating": min_rating,
            "exclude_ids": exclude_ids,
            "top_k": 20,            # 후보 풀을 넉넉하게 확보
            "top_k_qdrant": 30,     # Qdrant 내부 검색 수
            "top_k_es": 20,         # ES 내부 검색 수
            "top_k_neo4j": 15,      # Neo4j 내부 검색 수
        }

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "match_query_builder_complete",
            semantic_query_preview=semantic_query[:80],
            genre_filter=genre_filter,
            mood_filter=mood_filter,
            year_range=year_range,
            min_rating=min_rating,
            exclude_count=len(exclude_ids),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return {"search_query": search_query}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "match_query_builder_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            elapsed_ms=round(elapsed_ms, 1),
        )
        # 에러 시 최소한의 기본 쿼리 반환
        return {
            "search_query": {
                "semantic_query": "인기 영화",
                "genre_filter": None,
                "mood_filter": None,
                "year_range": None,
                "min_rating": None,
                "exclude_ids": [],
                "top_k": 20,
            }
        }


# ============================================================
# 노드 4: rag_retriever — 하이브리드 검색 + 임베딩 일괄 조회
# ============================================================

@traceable(name="match_rag_retriever", run_type="retriever", metadata={"node": "4/6", "db": "qdrant+es+neo4j"})
async def rag_retriever(state: MovieMatchState) -> dict:
    """
    하이브리드 검색 파이프라인으로 후보 영화를 조회한다.

    hybrid_search() → RRF(k=60) → 후보 movie_id 목록 확보
    → Qdrant client.retrieve(with_vectors=True)로 임베딩 벡터 일괄 조회
    (match_scorer에서 cosine similarity 계산에 사용)

    ### Level 1-A 개선: Vector Centroid 검색
    두 영화 임베딩의 centroid((vec_A + vec_B)/2 정규화) 를 query_vector 로 전달하여
    Qdrant 벡터 검색이 텍스트를 재임베딩하지 않고 두 영화 "사이 지점"을 직접 탐색하게 한다.
    기존에는 공통 장르/무드 텍스트를 다시 임베딩해 semantic 정보가 손실되었다.

    ### Level 1-B 개선: 3단계 필터 완화 (기존 1회 → 3회)
    1단계: 교집합 장르 + 무드 + 연도 + 평점 + centroid (초기 설정)
    2단계: 무드 제거 + 평점 0.6배 (공통 무드가 sparse 한 케이스 대응)
    3단계: 장르 합집합 + 무드/연도/평점 전부 제거 (블록버스터 교차 조합 대응)
    4단계(absolute last): 장르만 합집합으로 2개까지 축약 + top_k=30 (최후 fallback)

    ### 치명 버그 수정 (2026-04-14)
    기존 코드는 hybrid_search 호출 시 `mood_filter=` 키워드 사용.
    그러나 hybrid_search 시그니처는 `mood_tags=` 이므로 TypeError 발생 → try/except 가 삼켜
    *모든* 매치 검색이 실제로는 실행되지 않고 빈 결과 반환. → `mood_tags=` 로 수정.

    Args:
        state: MovieMatchState (search_query 필수)

    Returns:
        {"candidate_movies": list[dict]}  — 각 dict에 embedding 포함
    """
    node_start = time.perf_counter()
    search_query: dict = state.get("search_query") or {}

    logger.info(
        "match_rag_retriever_start",
        semantic_query_preview=search_query.get("semantic_query", "")[:60],
        genre_filter=search_query.get("genre_filter"),
        mood_filter=search_query.get("mood_filter"),
    )

    try:
        # ── [0] Vector Centroid 계산 ──
        # 두 영화 임베딩이 모두 있으면 (Qdrant 로드 성공 케이스) centroid 를 계산.
        # L2 정규화하여 Qdrant cosine 검색과 정합시킨다.
        movie_1: dict = state.get("movie_1", {})
        movie_2: dict = state.get("movie_2", {})
        centroid_vec: list[float] | None = _compute_embedding_centroid(
            movie_1.get("embedding"),
            movie_2.get("embedding"),
        )
        if centroid_vec is not None:
            logger.info(
                "match_rag_retriever_centroid_ready",
                vector_dim=len(centroid_vec),
            )

        async def _do_hybrid_search(query: dict) -> list[Any]:
            """hybrid_search() 파라미터 변환 후 호출.

            mood_filter (내부 키) → mood_tags (hybrid_search 파라미터) 로 매핑한다.
            query_vector 는 centroid 가 준비된 경우에만 전달 (그 외에는 semantic_query 를 재임베딩).
            """
            return await hybrid_search(
                query=query.get("semantic_query", "인기 영화"),
                top_k=query.get("top_k", 20),
                genre_filter=query.get("genre_filter"),
                # ⚠️ 버그 수정: mood_filter= → mood_tags= (파라미터 이름 정합성)
                mood_tags=query.get("mood_filter"),
                ott_filter=query.get("ott_filter"),
                min_rating=query.get("min_rating"),
                year_range=query.get("year_range"),
                exclude_ids=query.get("exclude_ids", []),
                query_vector=centroid_vec,   # Level 1-A: centroid 우선 사용
            )

        # ── [1] 1차 검색 + CF 병렬 실행 ──
        # Level 2-B: Co-watched CF 후보를 Recommend FastAPI 에서 병렬 조회하고 RRF 병합.
        # 하이브리드 검색(Qdrant/ES/Neo4j)과 CF 를 동시에 수행하여 추가 지연을 최소화.
        movie_id_1 = state.get("movie_id_1", "")
        movie_id_2 = state.get("movie_id_2", "")

        hybrid_task = _do_hybrid_search(search_query)
        cf_task = fetch_cowatched_candidates(
            movie_id_1=movie_id_1,
            movie_id_2=movie_id_2,
            top_k=settings.MATCH_COWATCH_TOP_K,
        )
        hybrid_results, cf_results = await asyncio.gather(
            hybrid_task, cf_task, return_exceptions=False,
        )

        logger.info(
            "match_rag_retriever_sources",
            hybrid_count=len(hybrid_results),
            cf_count=len(cf_results),
        )

        # ── CF 와 하이브리드 결과 RRF 병합 (k=60, 2개 소스) ──
        # CF 는 단일 신호(공통 사용자 선호)이므로 하이브리드 결과 비중이 자연스럽게 우세하고,
        # CF 에서만 상위에 나오는 영화가 final 에 진입할 수 있도록 RRF 로 합친다.
        # CF 결과가 비면 hybrid_results 를 그대로 사용 (RRF 오버헤드 회피).
        if cf_results:
            results = reciprocal_rank_fusion([hybrid_results, cf_results], k=60)
            # CF-only 영화가 exclude_ids(입력 영화) 와 일치할 수 있으므로 재필터
            exclude_set = set(search_query.get("exclude_ids", []))
            if exclude_set:
                results = [r for r in results if r.movie_id not in exclude_set]
        else:
            results = hybrid_results

        stage_counts: list[tuple[str, int]] = [
            ("stage_1_hybrid+cf", len(results)),
        ]

        # ── [2] 2단계 완화: 무드 제거 + 평점 0.6배 ──
        # 공통 무드가 모호하게 들어간 케이스 (e.g. mood_tags 가 과도하게 specific) 대응
        if len(results) < 5:
            relaxed_2 = {
                **search_query,
                "mood_filter": None,
                "min_rating": (
                    round(search_query.get("min_rating", 0) * 0.6 / 0.7, 1)
                    if search_query.get("min_rating")
                    else None
                ),
                "top_k": 25,
            }
            results_2 = await _do_hybrid_search(relaxed_2)
            # stage_2 결과를 기존과 합치되 중복 제거 (id 기준)
            results = _merge_unique_results(results, results_2)
            stage_counts.append(("stage_2_no_mood", len(results)))

        # ── [3] 3단계 완화: 장르 합집합 + 무드/연도/평점 전부 제거 ──
        if len(results) < 5:
            union_genres = list(
                set(movie_1.get("genres", [])) | set(movie_2.get("genres", []))
            )
            relaxed_3 = {
                **search_query,
                "genre_filter": union_genres[:5] if union_genres else None,
                "mood_filter": None,
                "year_range": None,
                "min_rating": None,
                "top_k": 25,
            }
            results_3 = await _do_hybrid_search(relaxed_3)
            results = _merge_unique_results(results, results_3)
            stage_counts.append(("stage_3_union_all_relaxed", len(results)))

        # ── [4] 4단계 absolute last resort: 장르 합집합 축약 + top_k=30 ──
        # 여기까지 와도 비어있으면 센트로이드 벡터 근접성만으로 뽑는다.
        if len(results) < 3:
            union_genres = list(
                set(movie_1.get("genres", [])) | set(movie_2.get("genres", []))
            )
            relaxed_4 = {
                "semantic_query": (
                    f"{movie_1.get('title', '')} {movie_2.get('title', '')} 비슷한 영화"
                ).strip(),
                "genre_filter": union_genres[:2] if union_genres else None,
                "mood_filter": None,
                "year_range": None,
                "min_rating": None,
                "exclude_ids": search_query.get("exclude_ids", []),
                "top_k": 30,
            }
            results_4 = await _do_hybrid_search(relaxed_4)
            results = _merge_unique_results(results, results_4)
            stage_counts.append(("stage_4_last_resort", len(results)))

        logger.info(
            "match_rag_retriever_stage_counts",
            stages=stage_counts,
        )

        if not results:
            logger.warning("match_rag_retriever_no_results_after_all_stages")
            return {"candidate_movies": []}

        # ── [3] 후보 movie_id 목록 추출 ──
        candidate_ids: list[str] = [r.movie_id for r in results]

        # ── [4] Qdrant에서 임베딩 벡터 일괄 조회 ──
        # hybrid_search() 결과에는 벡터가 없으므로 별도 retrieve 호출 필요
        client = await get_qdrant()
        point_ids = [to_point_id(mid) for mid in candidate_ids]

        try:
            points = await client.retrieve(
                collection_name=settings.QDRANT_COLLECTION,
                ids=point_ids,
                with_vectors=True,
                with_payload=True,
            )
            # point_id → (payload, vector) 매핑
            vector_map: dict[Any, tuple[dict, list[float] | None]] = {}
            for p in points:
                vec = p.vector
                if isinstance(vec, dict):
                    # 네임드 벡터 케이스 — 첫 번째 값 사용
                    vec = list(vec.values())[0] if vec else None
                vector_map[p.id] = (p.payload or {}, vec)
        except Exception as vec_err:
            logger.warning(
                "match_rag_retriever_vector_fetch_error",
                error=str(vec_err),
                candidate_count=len(candidate_ids),
            )
            vector_map = {}

        # ── [5] 검색 결과 + 임베딩 벡터 결합하여 candidate_movies 구성 ──
        candidate_movies: list[dict[str, Any]] = []
        result_map = {r.movie_id: r for r in results}

        for mid in candidate_ids:
            result = result_map.get(mid)
            if result is None:
                continue

            # 기본 메타데이터는 hybrid_search 결과의 metadata에서 가져옴
            meta: dict = result.metadata or {}
            movie_dict: dict[str, Any] = {
                "id": mid,
                "title": result.title or meta.get("title", ""),
                "title_en": meta.get("title_en", ""),
                "genres": meta.get("genres", []),
                "mood_tags": meta.get("mood_tags", []),
                "keywords": meta.get("keywords", []),
                "director": meta.get("director", ""),
                "cast": meta.get("cast", []),
                "cast_members": meta.get("cast", []),
                "release_year": meta.get("release_year"),
                "rating": meta.get("rating"),
                "poster_path": meta.get("poster_path"),
                "overview": meta.get("overview", ""),
                "ott_platforms": meta.get("ott_platforms", []),
                "rrf_score": result.score,
                # Match v3: CF 소스 영화면 cf_score 를 top-level 로 승격 (match_scorer 조회 용이)
                # CF 가 아닌 영화는 None — calculate_match_score 에서 가중치 재정규화
                "cf_score": meta.get("cf_score"),
                "embedding": None,  # 기본값
            }

            # Qdrant 벡터 오버레이 (더 상세한 payload 포함)
            pid = to_point_id(mid)
            if pid in vector_map:
                payload, vec = vector_map[pid]
                # payload에서 더 풍부한 메타데이터 보강
                for key in ("genres", "mood_tags", "keywords", "director", "cast",
                            "release_year", "rating", "poster_path", "overview",
                            "ott_platforms"):
                    if payload.get(key):
                        movie_dict[key] = payload[key]
                movie_dict["cast_members"] = movie_dict["cast"]
                movie_dict["embedding"] = vec  # 임베딩 벡터 설정

            candidate_movies.append(movie_dict)

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        vector_count = sum(1 for c in candidate_movies if c.get("embedding") is not None)
        logger.info(
            "match_rag_retriever_complete",
            candidate_count=len(candidate_movies),
            vector_count=vector_count,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return {"candidate_movies": candidate_movies}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "match_rag_retriever_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"candidate_movies": []}


# ============================================================
# 노드 5: llm_reranker — Solar LLM 배치 점수화 (Match v3, 2026-04-14)
# ============================================================

@traceable(name="match_llm_reranker", run_type="chain", metadata={"node": "5/7", "llm": "solar"})
async def llm_reranker(state: MovieMatchState) -> dict:
    """
    rag_retriever 가 수집한 후보 영화들을 Solar LLM 에 배치 전달하여
    "두 영화 A/B 를 모두 좋아할 사용자 관점" 의 점수(0~1) 를 계산한다.

    ### 도입 배경 (2026-04-14)
    기존 Match 는 Jaccard + cosine + harmonic 의 결정론적 수학만으로 순위를 매기고,
    LLM 은 최종 설명 생성에만 쓰였다. Chat Agent 처럼 LLM 리랭커를 중간에 배치하면
    "두 영화 동시 선호" 라는 목표에 LLM 의 세계 지식을 직접 활용할 수 있다.

    ### 동작
    1. candidates 상위 10편을 Solar 에 배치 전달 (`rerank_match_candidates`)
    2. {movie_id: llm_score_0_to_1} 딕셔너리를 state["llm_scores"] 에 저장
    3. match_scorer 가 이를 읽어 calculate_match_score(llm_score=...) 에 주입

    ### 에러 처리 (채팅 영향 격리)
    LLM 호출 실패/타임아웃 시 체인 레벨에서 빈 dict 반환 → 노드도 {"llm_scores": {}}.
    match_scorer 는 빈 dict 를 보고 harmonic+cf 로 graceful fallback.
    → Solar API 장애가 Match 그래프 전체를 막지 않음.

    Args:
        state: MovieMatchState (candidate_movies, movie_1, movie_2 필수)

    Returns:
        {"llm_scores": dict[str, float]}
    """
    node_start = time.perf_counter()
    candidates: list[dict] = state.get("candidate_movies", []) or []
    movie_1: dict = state.get("movie_1", {}) or {}
    movie_2: dict = state.get("movie_2", {}) or {}
    shared = state.get("shared_features")
    shared_summary = getattr(shared, "similarity_summary", "") if shared else ""

    logger.info(
        "match_llm_reranker_node_start",
        candidate_count=len(candidates),
        movie_1_title=movie_1.get("title", ""),
        movie_2_title=movie_2.get("title", ""),
    )

    # 후보가 비면 LLM 호출 없이 빈 dict 반환
    if not candidates:
        return {"llm_scores": {}}

    try:
        scores = await rerank_match_candidates(
            candidates=candidates,
            movie_1=movie_1,
            movie_2=movie_2,
            shared_summary=shared_summary,
        )
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "match_llm_reranker_node_complete",
            scored_count=len(scores),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"llm_scores": scores}
    except Exception as e:
        # rerank_match_candidates 는 내부에서 예외를 삼켜 빈 dict 를 반환하지만,
        # 방어적으로 한 번 더 감싼다 (에이전트 graceful fallback 원칙).
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "match_llm_reranker_node_error",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=round(elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )
        return {"llm_scores": {}}


# ============================================================
# 노드 6: match_scorer — LLM + harmonic + CF 가중합 스코어링 + MMR (Match v3)
# ============================================================

@traceable(name="match_scorer", run_type="chain", metadata={"node": "6/7", "top_k": "MATCH_TOP_K"})
async def match_scorer(state: MovieMatchState) -> dict:
    """
    각 후보 영화와 두 입력 영화 간 유사도를 계산하고 MMR로 Top 5를 선별한다.

    스코어링 흐름:
    1. 모든 후보에 calculate_match_score() 적용 → MatchScoreDetail 생성
    2. match_score(= min(sim_1, sim_2)) 내림차순 정렬
    3. MMR 그리디 알고리즘으로 다양성 보장 Top 5 선별 (λ=0.7)
    4. MatchedMovie 리스트 구성

    MMR 공식:
    MMR(c) = 0.7 × match_score(c) − 0.3 × max(0.7×genre_jaccard + 0.3×mood_jaccard for s in selected)

    Args:
        state: MovieMatchState (candidate_movies, movie_1, movie_2, shared_features 필수)

    Returns:
        {"ranked_movies": list[MatchedMovie]}  — Top 5, rank 필드 1~5 부여
    """
    node_start = time.perf_counter()
    candidates: list[dict] = state.get("candidate_movies", [])
    movie_1: dict = state.get("movie_1", {})
    movie_2: dict = state.get("movie_2", {})

    # Match v3 (2026-04-14): llm_reranker 가 계산한 점수 + CF 후보의 cf_score 를 융합.
    # llm_scores 는 dict[movie_id, 0~1 점수]. 없는 후보는 None 으로 전달.
    llm_scores: dict[str, float] = state.get("llm_scores", {}) or {}
    top_k: int = getattr(settings, "MATCH_TOP_K", 3)

    logger.info(
        "match_scorer_start",
        candidate_count=len(candidates),
        llm_scored_count=len(llm_scores),
        top_k=top_k,
    )

    try:
        if not candidates:
            logger.warning("match_scorer_no_candidates")
            return {"ranked_movies": []}

        # ── [1] 모든 후보에 매치 스코어 계산 (LLM + harmonic + CF 가중합) ──
        scored: list[tuple[dict, MatchScoreDetail]] = []
        for candidate in candidates:
            movie_id = candidate.get("id", "")
            # LLM 리랭커 결과에 없는 영화는 None → calculate_match_score 에서 가중치 재정규화
            llm_score = llm_scores.get(movie_id) if movie_id else None
            # CF 후보는 rag_retriever 단계에서 metadata 에 cf_score 가 포함됨.
            # RRF 병합 중 덮어쓰일 수 있으므로 방어적으로 읽어온다.
            cf_score_raw = candidate.get("cf_score")
            if cf_score_raw is None:
                # 메타데이터 경로에서 직접 추출 (CF 소스 후보에만 존재)
                cf_score_raw = (candidate.get("metadata") or {}).get("cf_score")
            cf_score = (
                float(cf_score_raw) if cf_score_raw is not None else None
            )

            score_detail = calculate_match_score(
                candidate=candidate,
                movie_1=movie_1,
                movie_2=movie_2,
                llm_score=llm_score,
                cf_score=cf_score,
            )
            scored.append((candidate, score_detail))

        # ── [2] match_score 내림차순 정렬 ──
        scored.sort(key=lambda x: x[1].match_score, reverse=True)

        logger.debug(
            "match_scorer_top_scores",
            top_scores=[
                {
                    "title": c.get("title", ""),
                    "match_score": round(sd.match_score, 3),
                    "llm_score": round(sd.llm_score, 3),
                    "cf_score": round(sd.cf_score, 3),
                    "sim_1": round(sd.sim_to_movie_1, 3),
                    "sim_2": round(sd.sim_to_movie_2, 3),
                }
                for c, sd in scored[: max(top_k, 5)]
            ],
        )

        # ── [3] MMR 그리디 알고리즘으로 Top K 선별 (Match v3: 기본 3편) ──
        # λ=0.7: 점수 70%, 다양성 30%
        mmr_lambda = 0.7
        selected: list[tuple[dict, MatchScoreDetail]] = []

        # 1위 영화는 match_score가 가장 높은 영화를 바로 선택
        if scored:
            selected.append(scored[0])
            remaining = scored[1:]
        else:
            remaining = []

        # 2 ~ top_k 위: MMR 그리디 선택
        while len(selected) < top_k and remaining:
            best_mmr = -float("inf")
            best_idx = 0

            for i, (candidate, score_detail) in enumerate(remaining):
                # 현재까지 선택된 영화들과의 최대 유사도 계산 (장르 + 무드 가중 평균)
                # 장르만 사용하면 감독/무드가 같아도 장르만 다르면 "다양하다"고 오판할 수 있다.
                # 장르 70% + 무드 30% 가중 평균으로 다양성을 더 정교하게 판단한다.
                max_sim_to_selected = max(
                    0.7 * jaccard(
                        set(candidate.get("genres", [])),
                        set(s.get("genres", [])),
                    )
                    + 0.3 * jaccard(
                        set(candidate.get("mood_tags", [])),
                        set(s.get("mood_tags", [])),
                    )
                    for s, _ in selected
                ) if selected else 0.0

                # MMR 점수 계산
                mmr_score = (
                    mmr_lambda * score_detail.match_score
                    - (1 - mmr_lambda) * max_sim_to_selected
                )

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i

            # 최고 MMR 점수의 영화를 선택 목록에 추가
            selected.append(remaining[best_idx])
            remaining = remaining[:best_idx] + remaining[best_idx + 1:]

        # ── [4] MatchedMovie 리스트 구성 ──
        ranked_movies: list[MatchedMovie] = []
        for rank, (candidate, score_detail) in enumerate(selected, start=1):
            matched = MatchedMovie(
                movie_id=candidate.get("id", ""),
                title=candidate.get("title", ""),
                title_en=candidate.get("title_en") or None,
                genres=candidate.get("genres", []),
                mood_tags=candidate.get("mood_tags", []),
                release_year=candidate.get("release_year"),
                rating=candidate.get("rating"),
                poster_path=candidate.get("poster_path") or None,
                overview=candidate.get("overview") or None,
                ott_platforms=candidate.get("ott_platforms", []),
                director=candidate.get("director") or None,
                cast_members=candidate.get("cast_members") or candidate.get("cast", []),
                score_detail=score_detail,
                explanation="",     # explanation_generator에서 채워짐
                rank=rank,
            )
            ranked_movies.append(matched)

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "match_scorer_complete",
            ranked_count=len(ranked_movies),
            top_match_score=ranked_movies[0].score_detail.match_score if ranked_movies else 0.0,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return {"ranked_movies": ranked_movies}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "match_scorer_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"ranked_movies": []}


# ============================================================
# 노드 6: explanation_generator — "두 사람 모두 좋아할 이유" 배치 생성
# ============================================================

@traceable(name="match_explanation_generator", run_type="chain", metadata={"node": "6/6", "llm": "exaone-32b"})
async def explanation_generator(state: MovieMatchState) -> dict:
    """
    Top 5 추천 영화 각각에 대해 "두 사람 모두 좋아할 이유"를 생성한다.

    generate_match_explanations_batch()를 호출하여 EXAONE 32B로
    순차 설명을 생성하고, ranked_movies의 explanation 필드를 채운다.
    Ollama 직렬 처리 특성상 순차 실행이 병렬보다 효율적이다.

    Args:
        state: MovieMatchState (ranked_movies, movie_1, movie_2, shared_features 필수)

    Returns:
        {"ranked_movies": list[MatchedMovie]}  — explanation 필드가 채워진 목록
    """
    node_start = time.perf_counter()
    ranked_movies: list[MatchedMovie] = state.get("ranked_movies", [])
    movie_1: dict = state.get("movie_1", {})
    movie_2: dict = state.get("movie_2", {})
    shared: SharedFeatures = state.get("shared_features") or SharedFeatures()

    logger.info(
        "match_explanation_generator_start",
        movie_count=len(ranked_movies),
        movie_1_title=movie_1.get("title", ""),
        movie_2_title=movie_2.get("title", ""),
    )

    try:
        if not ranked_movies:
            logger.warning("match_explanation_generator_no_movies")
            return {"ranked_movies": []}

        # 각 영화의 score_detail을 dict로 변환 (체인에 전달)
        movie_dicts: list[dict] = []
        score_detail_dicts: list[dict] = []

        for m in ranked_movies:
            movie_dicts.append({
                "id": m.movie_id,
                "title": m.title,
                "genres": m.genres,
                "mood_tags": m.mood_tags,
                "overview": m.overview or "",
                "director": m.director or "",
                "rating": m.rating or 0.0,
            })
            score_detail_dicts.append(m.score_detail.model_dump())

        # 배치 설명 생성 (순차 실행)
        explanations = await generate_match_explanations_batch(
            movies=movie_dicts,
            movie_1=movie_1,
            movie_2=movie_2,
            shared_features_summary=shared.similarity_summary,
            score_details=score_detail_dicts,
        )

        # explanation 필드 업데이트 — Pydantic 모델 재생성 (불변성 보장)
        updated_movies: list[MatchedMovie] = []
        for movie, explanation in zip(ranked_movies, explanations):
            updated = movie.model_copy(update={"explanation": explanation})
            updated_movies.append(updated)

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "match_explanation_generator_complete",
            movie_count=len(updated_movies),
            explanations_preview=[e[:40] for e in explanations],
            elapsed_ms=round(elapsed_ms, 1),
        )

        return {"ranked_movies": updated_movies}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "match_explanation_generator_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            elapsed_ms=round(elapsed_ms, 1),
        )
        # 에러 시 explanation 없는 원본 반환 (설명만 없을 뿐 추천 결과는 유지)
        return {"ranked_movies": ranked_movies}
