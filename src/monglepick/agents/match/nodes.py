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
from monglepick.chains.match_explanation_chain import (
    generate_match_explanations_batch,
    generate_similarity_summary,
)
from monglepick.config import settings
from monglepick.db.clients import get_mysql, get_qdrant
from monglepick.rag.hybrid_search import hybrid_search
from monglepick.utils.qdrant_helpers import to_point_id

logger = structlog.get_logger()


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

        # ── [2] 장르 필터: 공통 장르 우선, 없으면 합집합 ──
        if shared.common_genres:
            genre_filter = shared.common_genres
        else:
            # 공통 장르가 없으면 두 영화 장르 합집합으로 넓게 검색
            all_genres = list(
                set(movie_1.get("genres", [])) | set(movie_2.get("genres", []))
            )
            genre_filter = all_genres[:5]   # 최대 5개로 제한

        # ── [3] 무드 필터: 공통 무드만 사용 (없으면 필터 없음) ──
        mood_filter = shared.common_moods if shared.common_moods else None

        # ── [4] 연도 범위 필터 ──
        era_min, era_max = shared.era_range
        year_range = (era_min, era_max)

        # ── [5] 최소 평점 필터: avg_rating * 0.7 ──
        min_rating = round(shared.avg_rating * 0.7, 1) if shared.avg_rating > 0 else None

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
    기존 하이브리드 검색 파이프라인으로 후보 영화를 조회한다.

    hybrid_search() → RRF(k=60) → 후보 movie_id 목록 확보
    → Qdrant client.retrieve(with_vectors=True)로 임베딩 벡터 일괄 조회
    (match_scorer에서 cosine similarity 계산에 사용)

    후보 3편 미만 시 필터 완화 재검색:
    1. 장르 교집합 → 합집합으로 확장
    2. year_range 제거
    3. min_rating 제거

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
        async def _do_hybrid_search(query: dict) -> list[Any]:
            """hybrid_search() 파라미터 변환 후 호출."""
            return await hybrid_search(
                query=query.get("semantic_query", "인기 영화"),
                top_k=query.get("top_k", 20),
                genre_filter=query.get("genre_filter"),
                mood_filter=query.get("mood_filter"),
                ott_filter=query.get("ott_filter"),
                min_rating=query.get("min_rating"),
                year_range=query.get("year_range"),
                exclude_ids=query.get("exclude_ids", []),
            )

        # ── [1] 1차 하이브리드 검색 ──
        results = await _do_hybrid_search(search_query)

        # ── [2] 후보 3편 미만 시 필터 완화 재검색 ──
        if len(results) < 3:
            logger.warning(
                "match_rag_retriever_too_few_results",
                count=len(results),
                action="relaxing_filters",
            )
            # 영화 1, 2의 장르 합집합으로 확장 (query_builder의 교집합 → 합집합)
            movie_1: dict = state.get("movie_1", {})
            movie_2: dict = state.get("movie_2", {})
            union_genres = list(
                set(movie_1.get("genres", [])) | set(movie_2.get("genres", []))
            )
            relaxed_query = {
                **search_query,
                "genre_filter": union_genres[:5] if union_genres else None,
                "mood_filter": None,    # 무드 필터 제거 — 특수 무드 태그로 인한 0건 방지
                "year_range": None,     # 연도 제한 제거
                "min_rating": None,     # 최소 평점 제거
                "top_k": 20,
            }
            results = await _do_hybrid_search(relaxed_query)

            logger.info(
                "match_rag_retriever_relaxed",
                count=len(results),
                union_genres=union_genres[:5],
            )

        if not results:
            logger.warning("match_rag_retriever_no_results")
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
# 노드 5: match_scorer — min(simA, simB) 스코어링 + MMR 리랭킹
# ============================================================

@traceable(name="match_scorer", run_type="chain", metadata={"node": "5/6"})
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

    logger.info(
        "match_scorer_start",
        candidate_count=len(candidates),
    )

    try:
        if not candidates:
            logger.warning("match_scorer_no_candidates")
            return {"ranked_movies": []}

        # ── [1] 모든 후보에 매치 스코어 계산 ──
        scored: list[tuple[dict, MatchScoreDetail]] = []
        for candidate in candidates:
            score_detail = calculate_match_score(candidate, movie_1, movie_2)
            scored.append((candidate, score_detail))

        # ── [2] match_score 내림차순 정렬 ──
        scored.sort(key=lambda x: x[1].match_score, reverse=True)

        logger.debug(
            "match_scorer_top_scores",
            top_5_scores=[
                {
                    "title": c.get("title", ""),
                    "match_score": round(sd.match_score, 3),
                    "sim_1": round(sd.sim_to_movie_1, 3),
                    "sim_2": round(sd.sim_to_movie_2, 3),
                }
                for c, sd in scored[:5]
            ],
        )

        # ── [3] MMR 그리디 알고리즘으로 Top 5 선별 ──
        # λ=0.7: 점수 70%, 다양성 30%
        mmr_lambda = 0.7
        selected: list[tuple[dict, MatchScoreDetail]] = []

        # 1위 영화는 match_score가 가장 높은 영화를 바로 선택
        if scored:
            selected.append(scored[0])
            remaining = scored[1:]
        else:
            remaining = []

        # 2~5위: MMR 그리디 선택
        while len(selected) < 5 and remaining:
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
