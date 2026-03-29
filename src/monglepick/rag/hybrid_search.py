"""
하이브리드 검색 (Qdrant 벡터 + Elasticsearch BM25 + Neo4j 그래프 + RRF 합산).

§11-1 하이브리드 검색 시 5개 저장소 협업 흐름:
1. Qdrant 벡터 검색: 의미적 유사도 Top-30
2. ES BM25 검색: 키워드 매칭 Top-20
3. Neo4j 그래프 검색: 무드+장르 관계 탐색 Top-15
4. RRF 합산: Reciprocal Rank Fusion (k=60) → 최종 후보 15편

§6-2-1 Chat Agent의 query_builder/rag_retriever 노드에서 호출된다.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog
from langsmith import traceable
from qdrant_client.models import FieldCondition, Filter, MatchAny

from monglepick.config import settings
from monglepick.data_pipeline.embedder import embed_query_async
from monglepick.db.clients import ES_INDEX_NAME, get_elasticsearch, get_neo4j, get_qdrant

logger = structlog.get_logger()

# RRF 상수 (§11-1: k=60) — config.py에서 환경변수로 설정 가능
RRF_K = settings.RRF_K


@dataclass
class SearchResult:
    """검색 결과 단일 항목."""
    movie_id: str
    title: str = ""
    score: float = 0.0
    source: str = ""  # "qdrant", "es", "neo4j"
    metadata: dict = field(default_factory=dict)


# ============================================================
# 1. Qdrant 벡터 검색
# ============================================================

@traceable(name="search_qdrant", run_type="retriever", metadata={"db": "qdrant", "type": "vector"})
async def search_qdrant(
    query: str,
    top_k: int = 30,
    genre_filter: list[str] | None = None,
    mood_filter: list[str] | None = None,
    ott_filter: list[str] | None = None,
    min_rating: float | None = None,
    year_range: tuple[int, int] | None = None,
    has_trailer: bool | None = None,
) -> list[SearchResult]:
    """
    Qdrant 벡터 검색: 쿼리의 의미적 유사도로 영화를 검색한다.

    §11-1 ①: 쿼리 벡터 생성 → 코사인 유사도 Top-30 + 메타데이터 필터
    동적 필터(min_rating, has_trailer) 지원 추가.
    """
    # Qdrant 검색 타이밍 측정 시작
    qdrant_start = time.perf_counter()
    client = await get_qdrant()

    # 쿼리 임베딩 — 비동기 래퍼로 event loop 블로킹 방지 (C-1)
    # Upstage API 장애 시 무한 대기 방지를 위해 30초 타임아웃 적용
    try:
        query_vector = (await asyncio.wait_for(
            embed_query_async(query),
            timeout=30.0,
        )).tolist()
    except asyncio.TimeoutError:
        logger.error("embedding_api_timeout", query_preview=query[:80], timeout_sec=30)
        return []  # 임베딩 실패 시 빈 결과 반환 (RRF에서 다른 엔진 결과만 사용)

    # 필터 조건 구성 (§10-2-1 payload 인덱스 활용)
    conditions = []

    if genre_filter:
        conditions.append(FieldCondition(key="genres", match=MatchAny(any=genre_filter)))
    if mood_filter:
        conditions.append(FieldCondition(key="mood_tags", match=MatchAny(any=mood_filter)))
    if ott_filter:
        conditions.append(FieldCondition(key="ott_platforms", match=MatchAny(any=ott_filter)))
    if min_rating is not None:
        conditions.append(FieldCondition(key="rating", range={"gte": min_rating}))
    if year_range is not None:
        conditions.append(FieldCondition(key="release_year", range={"gte": year_range[0], "lte": year_range[1]}))
    # 트레일러 존재 여부 필터: trailer_url 필드가 비어있지 않은 문서만 검색
    if has_trailer:
        # Qdrant에서 문자열 필드의 존재/비존재 필터링:
        # trailer_url이 빈 문자열("")이 아닌 경우만 포함하도록 range 조건 사용 불가하므로
        # MatchExcept로 빈 문자열을 제외하는 방식으로 구현
        from qdrant_client.models import MatchExcept
        conditions.append(FieldCondition(key="trailer_url", match=MatchExcept(except_=["", "null"])))

    query_filter = Filter(must=conditions) if conditions else None

    logger.info(
        "qdrant_search_start",
        query_preview=query[:80],
        top_k=top_k,
        filter_count=len(conditions),
        genre_filter=genre_filter,
        mood_filter=mood_filter,
        ott_filter=ott_filter,
        year_range=year_range,
    )

    # 벡터 검색 실행 (qdrant-client v1.17+: search → query_points)
    response = await client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    results = [
        SearchResult(
            movie_id=str(hit.id),
            title=hit.payload.get("title", "") if hit.payload else "",
            score=hit.score,
            source="qdrant",
            metadata=dict(hit.payload) if hit.payload else {},
        )
        for hit in response.points
    ]

    # Qdrant 검색 소요 시간 계산
    qdrant_elapsed_ms = (time.perf_counter() - qdrant_start) * 1000

    # 상세 검색 결과 로깅
    logger.info(
        "qdrant_search_results",
        result_count=len(results),
        elapsed_ms=round(qdrant_elapsed_ms, 1),
        top_results=[
            {"title": r.title, "score": round(r.score, 4), "id": r.movie_id}
            for r in results[:5]
        ],
    )

    return results


# ============================================================
# 2. Elasticsearch BM25 검색
# ============================================================

@traceable(name="search_elasticsearch", run_type="retriever", metadata={"db": "elasticsearch", "type": "bm25"})
async def search_elasticsearch(
    query: str,
    top_k: int = 20,
    genre_filter: list[str] | None = None,
    mood_filter: list[str] | None = None,
    min_rating: float | None = None,
    has_trailer: bool | None = None,
) -> list[SearchResult]:
    """
    Elasticsearch BM25 검색: Nori 한국어 형태소 분석 기반 키워드 매칭.

    §11-1 ②: multi_match + function_score (무드태그 부스트)
    동적 필터(min_rating, has_trailer) 지원 추가.
    """
    # ES 검색 타이밍 측정 시작
    es_start = time.perf_counter()
    client = await get_elasticsearch()

    # multi_match 쿼리 (title, director, overview, cast, keywords 대상)
    must_query: dict = {
        "multi_match": {
            "query": query,
            "fields": [
                "title^3.0",
                "director^2.5",
                "cast^2.0",
                "keywords^1.5",
                "overview^1.0",
            ],
            "type": "best_fields",
        }
    }

    # 필터 조건
    filter_clauses = []
    if genre_filter:
        filter_clauses.append({"terms": {"genres": genre_filter}})
    if mood_filter:
        filter_clauses.append({"terms": {"mood_tags": mood_filter}})
    # 동적 필터: 최소 평점
    if min_rating is not None:
        filter_clauses.append({"range": {"rating": {"gte": min_rating}}})
    # 동적 필터: 트레일러 존재 여부 (빈 문자열이 아닌 경우만)
    if has_trailer:
        filter_clauses.append({"exists": {"field": "trailer_url"}})
        # 빈 문자열 제외 (exists만으로는 ""도 포함되므로)
        filter_clauses.append({"bool": {"must_not": [{"term": {"trailer_url.keyword": ""}}]}})

    # function_score로 인기도 부스트
    body = {
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [must_query],
                        "filter": filter_clauses if filter_clauses else [],
                    }
                },
                "functions": [
                    {
                        "field_value_factor": {
                            "field": "popularity_score",
                            "modifier": "log1p",
                            "factor": 0.1,
                        }
                    }
                ],
                "boost_mode": "sum",
            }
        },
        "size": top_k,
    }

    logger.info(
        "es_search_start",
        query_preview=query[:80],
        top_k=top_k,
        genre_filter=genre_filter,
        mood_filter=mood_filter,
    )

    resp = await client.search(index=ES_INDEX_NAME, body=body)
    hits = resp["hits"]["hits"]

    results = [
        SearchResult(
            movie_id=hit["_id"],
            title=hit["_source"].get("title", ""),
            score=hit["_score"],
            source="es",
            metadata=hit["_source"],
        )
        for hit in hits
    ]

    # ES 검색 소요 시간 계산
    es_elapsed_ms = (time.perf_counter() - es_start) * 1000

    # 상세 검색 결과 로깅
    logger.info(
        "es_search_results",
        result_count=len(results),
        elapsed_ms=round(es_elapsed_ms, 1),
        top_results=[
            {"title": r.title, "score": round(r.score, 4), "id": r.movie_id}
            for r in results[:5]
        ],
    )

    return results


# ============================================================
# 3. Neo4j 그래프 검색
# ============================================================

@traceable(name="search_neo4j", run_type="retriever", metadata={"db": "neo4j", "type": "graph"})
async def search_neo4j(
    mood_tags: list[str] | None = None,
    genres: list[str] | None = None,
    director: str | None = None,
    similar_to_movie_id: str | None = None,
    top_k: int = 15,
) -> list[SearchResult]:
    """
    Neo4j 그래프 검색: 무드/장르/감독 관계를 기반으로 영화를 탐색한다.

    §11-1 ③:
    - MATCH (m)-[:HAS_MOOD]->(mt:MoodTag) 무드 기반
    - SIMILAR_TO 관계로 후보 확장
    - 감독/배우 관계 탐색
    """
    # Neo4j 검색 타이밍 측정 시작
    neo4j_start = time.perf_counter()
    driver = await get_neo4j()

    results: list[SearchResult] = []

    logger.info(
        "neo4j_search_start",
        mood_tags=mood_tags,
        genres=genres,
        director=director,
        similar_to_movie_id=similar_to_movie_id,
        top_k=top_k,
    )

    async with driver.session() as session:
        # 전략 1: 무드태그 + 장르 조합 검색
        if mood_tags or genres:
            conditions = []
            params: dict = {}

            if mood_tags:
                conditions.append("(m)-[:HAS_MOOD]->(:MoodTag {name: mood})")
                params["mood_tags"] = mood_tags

            if genres:
                conditions.append("(m)-[:HAS_GENRE]->(:Genre {name: genre})")
                params["genres"] = genres

            # 무드 + 장르 조합 쿼리
            cypher = """
            MATCH (m:Movie)
            WHERE
            """
            where_clauses = []
            if mood_tags:
                where_clauses.append(
                    "EXISTS { MATCH (m)-[:HAS_MOOD]->(mt:MoodTag) WHERE mt.name IN $mood_tags }"
                )
            if genres:
                where_clauses.append(
                    "EXISTS { MATCH (m)-[:HAS_GENRE]->(g:Genre) WHERE g.name IN $genres }"
                )

            cypher += " AND ".join(where_clauses)

            # mood_tags가 있을 때만 OPTIONAL MATCH로 무드 일치 수를 세고 정렬에 사용.
            # mood_tags가 없으면(감정 미감지) mood_match를 0으로 고정하고 인기도로만 정렬.
            # 기존 문제: mood_tags=[] 일 때 OPTIONAL MATCH ... WHERE mt.name IN [] → 항상 0 매칭
            if mood_tags:
                cypher += """
                OPTIONAL MATCH (m)-[:HAS_MOOD]->(mt:MoodTag) WHERE mt.name IN $mood_tags
                WITH m, count(mt) AS mood_match
                RETURN m.id AS movie_id, m.title AS title,
                       m.rating AS rating, m.popularity_score AS popularity,
                       mood_match
                ORDER BY mood_match DESC, m.popularity_score DESC
                LIMIT $top_k
                """
            else:
                # 무드태그 없음 (감정 미감지): 무드 매칭 없이 인기도/평점 기반 정렬
                cypher += """
                RETURN m.id AS movie_id, m.title AS title,
                       m.rating AS rating, m.popularity_score AS popularity,
                       0 AS mood_match
                ORDER BY m.popularity_score DESC, m.rating DESC
                LIMIT $top_k
                """

            params["top_k"] = top_k
            # mood_tags 파라미터가 Cypher 바인딩에 필요할 수 있으므로 안전하게 포함
            if "mood_tags" not in params:
                params["mood_tags"] = []

            result = await session.run(cypher, params)
            records = await result.data()

            for i, record in enumerate(records):
                results.append(SearchResult(
                    movie_id=str(record["movie_id"]),
                    title=record.get("title", ""),
                    score=float(top_k - i),  # 순위 기반 점수
                    source="neo4j",
                    metadata={"rating": record.get("rating"), "mood_match": record.get("mood_match")},
                ))

            logger.info(
                "neo4j_mood_genre_results",
                result_count=len(records),
                top_results=[
                    {"title": r.get("title", ""), "mood_match": r.get("mood_match", 0)}
                    for r in records[:5]
                ],
            )

        # 전략 2: SIMILAR_TO 관계 확장
        if similar_to_movie_id:
            cypher = """
            MATCH (source:Movie {id: $movie_id})-[r:SIMILAR_TO]->(m:Movie)
            RETURN m.id AS movie_id, m.title AS title,
                   r.score AS similarity, m.rating AS rating
            ORDER BY r.score DESC
            LIMIT $top_k
            """
            result = await session.run(cypher, {"movie_id": similar_to_movie_id, "top_k": top_k})
            records = await result.data()

            for record in records:
                results.append(SearchResult(
                    movie_id=str(record["movie_id"]),
                    title=record.get("title", ""),
                    score=float(record.get("similarity", 0)),
                    source="neo4j",
                    metadata={"rating": record.get("rating")},
                ))

            logger.info(
                "neo4j_similar_to_results",
                source_movie_id=similar_to_movie_id,
                result_count=len(records),
                top_results=[
                    {"title": r.get("title", ""), "similarity": r.get("similarity", 0)}
                    for r in records[:5]
                ],
            )

        # 전략 3: 감독 기반 탐색
        if director:
            cypher = """
            MATCH (p:Person {name: $director})-[:DIRECTED]->(m:Movie)
            RETURN m.id AS movie_id, m.title AS title, m.rating AS rating
            ORDER BY m.rating DESC
            LIMIT $top_k
            """
            result = await session.run(cypher, {"director": director, "top_k": top_k})
            records = await result.data()

            for i, record in enumerate(records):
                results.append(SearchResult(
                    movie_id=str(record["movie_id"]),
                    title=record.get("title", ""),
                    score=float(top_k - i),
                    source="neo4j",
                    metadata={"rating": record.get("rating")},
                ))

            logger.info(
                "neo4j_director_results",
                director=director,
                result_count=len(records),
                top_results=[
                    {"title": r.get("title", ""), "rating": r.get("rating", 0)}
                    for r in records[:5]
                ],
            )

    # Neo4j 검색 소요 시간 계산
    neo4j_elapsed_ms = (time.perf_counter() - neo4j_start) * 1000

    logger.info(
        "neo4j_search_completed",
        total_result_count=len(results),
        elapsed_ms=round(neo4j_elapsed_ms, 1),
        top_results=[
            {"title": r.title, "score": round(r.score, 4), "id": r.movie_id}
            for r in results[:5]
        ],
    )

    return results


# ============================================================
# 4. RRF (Reciprocal Rank Fusion) 합산
# ============================================================

def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = RRF_K,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion으로 여러 검색 결과를 합산한다.

    §6-2-1 rag_retriever 노드의 RRF 합산 로직:
    RRF_score(d) = Σ 1 / (k + rank_i(d))

    Args:
        result_lists: 각 검색 엔진의 결과 리스트 (순위 순서)
        k: RRF 상수 (기본 60)

    Returns:
        RRF 점수 내림차순으로 정렬된 합산 결과
    """
    # 영화별 RRF 점수 누적
    rrf_scores: dict[str, float] = {}
    metadata_cache: dict[str, dict] = {}
    title_cache: dict[str, str] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            mid = result.movie_id
            rrf_scores[mid] = rrf_scores.get(mid, 0.0) + 1.0 / (k + rank)

            # 메타데이터는 가장 상세한 것을 보존
            if mid not in metadata_cache or len(result.metadata) > len(metadata_cache[mid]):
                metadata_cache[mid] = result.metadata
                title_cache[mid] = result.title

    # RRF 점수 내림차순 정렬
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    return [
        SearchResult(
            movie_id=mid,
            title=title_cache.get(mid, ""),
            score=rrf_scores[mid],
            source="rrf",
            metadata=metadata_cache.get(mid, {}),
        )
        for mid in sorted_ids
    ]


# ============================================================
# 통합 하이브리드 검색 인터페이스
# ============================================================

@traceable(name="hybrid_search", run_type="chain", metadata={"fusion": "RRF", "k": 60})
async def hybrid_search(
    query: str,
    top_k: int = 15,
    genre_filter: list[str] | None = None,
    mood_tags: list[str] | None = None,
    ott_filter: list[str] | None = None,
    min_rating: float | None = None,
    year_range: tuple[int, int] | None = None,
    director: str | None = None,
    similar_to_movie_id: str | None = None,
    exclude_ids: list[str] | None = None,
    has_trailer: bool | None = None,
) -> list[SearchResult]:
    """
    3개 검색 엔진을 동시 실행하고 RRF로 합산하여 최종 후보를 반환한다.

    §11-1 하이브리드 검색 흐름:
    ① Qdrant 벡터 검색 (의미) — min_rating, has_trailer 필터 지원
    ② ES BM25 검색 (키워드) — min_rating, has_trailer 필터 지원
    ③ Neo4j 그래프 검색 (관계)
    ④ 시청 완료 영화 제외 (RRF 전)
    ⑤ RRF 합산 → 최종 후보

    Args:
        query: 사용자 검색 쿼리
        top_k: 최종 반환 결과 수 (기본 15)
        genre_filter: 장르 필터
        mood_tags: 무드태그 필터
        ott_filter: OTT 플랫폼 필터
        min_rating: 최소 평점 (동적 필터)
        year_range: (시작연도, 끝연도)
        director: 감독명 (Neo4j 검색용, 동적 필터)
        similar_to_movie_id: 유사 영화 기준 ID (Neo4j SIMILAR_TO)
        exclude_ids: 제외할 영화 ID 목록
        has_trailer: 트레일러 존재 여부 필터 (동적 필터)

    Returns:
        RRF 합산 점수 기준 상위 top_k 결과
    """
    import asyncio

    # 병렬 검색 전체 타이밍 측정 시작
    search_start = time.perf_counter()

    # 3개 검색 엔진 동시 실행 — 개별 장애 시 해당 소스만 건너뜀 (W-6)
    qdrant_results: list[SearchResult] = []
    es_results: list[SearchResult] = []
    neo4j_results: list[SearchResult] = []

    async def _safe_search_qdrant() -> list[SearchResult]:
        """Qdrant 검색 래퍼. 실패 시 빈 리스트 반환."""
        try:
            return await search_qdrant(
                query=query,
                top_k=30,
                genre_filter=genre_filter,
                mood_filter=mood_tags,
                ott_filter=ott_filter,
                min_rating=min_rating,
                year_range=year_range,
                has_trailer=has_trailer,
            )
        except Exception as e:
            logger.warning("qdrant_search_failed_skipping", error=str(e), error_type=type(e).__name__)
            return []

    async def _safe_search_es() -> list[SearchResult]:
        """ES 검색 래퍼. 실패 시 빈 리스트 반환."""
        try:
            return await search_elasticsearch(
                query=query,
                top_k=20,
                genre_filter=genre_filter,
                mood_filter=mood_tags,
                min_rating=min_rating,
                has_trailer=has_trailer,
            )
        except Exception as e:
            logger.warning("es_search_failed_skipping", error=str(e), error_type=type(e).__name__)
            return []

    async def _safe_search_neo4j() -> list[SearchResult]:
        """Neo4j 검색 래퍼. 실패 시 빈 리스트 반환."""
        try:
            return await search_neo4j(
                mood_tags=mood_tags,
                genres=genre_filter,
                director=director,
                similar_to_movie_id=similar_to_movie_id,
                top_k=15,
            )
        except Exception as e:
            logger.warning("neo4j_search_failed_skipping", error=str(e), error_type=type(e).__name__)
            return []

    qdrant_results, es_results, neo4j_results = await asyncio.gather(
        _safe_search_qdrant(), _safe_search_es(), _safe_search_neo4j(),
    )

    # 병렬 검색 소요 시간 계산
    search_elapsed_ms = (time.perf_counter() - search_start) * 1000

    logger.info(
        "hybrid_search_engine_results",
        query_preview=query[:80],
        qdrant_count=len(qdrant_results),
        es_count=len(es_results),
        neo4j_count=len(neo4j_results),
        parallel_search_elapsed_ms=round(search_elapsed_ms, 1),
        qdrant_top3=[r.title for r in qdrant_results[:3]],
        es_top3=[r.title for r in es_results[:3]],
        neo4j_top3=[r.title for r in neo4j_results[:3]],
    )

    # ── 시청 완료 영화 사전 제외 (RRF 전) ──
    # RRF 합산 후에 제거하면 이미 본 영화가 상위 슬롯을 차지하여
    # 최종 후보 수가 top_k 미만으로 줄어들 수 있다. 사전 제거로 방지.
    if exclude_ids:
        exclude_set = set(exclude_ids)
        qdrant_results = [r for r in qdrant_results if r.movie_id not in exclude_set]
        es_results = [r for r in es_results if r.movie_id not in exclude_set]
        neo4j_results = [r for r in neo4j_results if r.movie_id not in exclude_set]
        logger.info(
            "exclude_ids_applied_before_rrf",
            exclude_count=len(exclude_set),
            after_qdrant=len(qdrant_results),
            after_es=len(es_results),
            after_neo4j=len(neo4j_results),
        )

    # RRF 합산 타이밍 측정 시작
    rrf_start = time.perf_counter()

    # RRF 합산 (§11-1 ⑤)
    fused = reciprocal_rank_fusion(
        [qdrant_results, es_results, neo4j_results],
        k=RRF_K,
    )

    final = fused[:top_k]

    # RRF 합산 소요 시간 계산
    rrf_elapsed_ms = (time.perf_counter() - rrf_start) * 1000
    # 전체 하이브리드 검색 소요 시간
    total_elapsed_ms = (time.perf_counter() - search_start) * 1000

    # 최종 RRF 결과 상세 로깅
    logger.info(
        "hybrid_search_rrf_final",
        query_preview=query[:80],
        total_fused=len(fused),
        returned=len(final),
        rrf_elapsed_ms=round(rrf_elapsed_ms, 1),
        total_elapsed_ms=round(total_elapsed_ms, 1),
        final_results=[
            {"rank": i + 1, "title": r.title, "rrf_score": round(r.score, 6), "id": r.movie_id}
            for i, r in enumerate(final)
        ],
    )

    return final
