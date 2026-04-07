"""
GraphQueryPlan → (Cypher 쿼리, 파라미터) 변환기.

relation Intent에서 사용하는 Neo4j 멀티홉 Cypher 쿼리를 생성한다.

보안 원칙:
- 모든 사용자 입력값은 Cypher 파라미터($var)로 전달 — 인젝션 방지
- 관계 타입(DIRECTED, ACTED_IN 등)은 화이트리스트로 검증 후 쿼리에 인라인
  (Neo4j 드라이버가 관계 타입 파라미터화를 지원하지 않으므로 인라인 불가피,
   화이트리스트로 허용된 값만 통과시켜 인젝션을 차단한다)

지원 query_type:
- chain: A 감독/배우 → 영화(+장르 필터) → 배우/감독 → 결과 영화
- intersection: N명 모두 출연/감독한 영화 교집합
- person_filmography: 특정 인물의 필모그래피 전체
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()


# ============================================================
# 화이트리스트 — Cypher 관계 타입 인젝션 방지
# ============================================================

# Neo4j 드라이버는 관계 타입을 파라미터로 바인딩할 수 없으므로,
# 쿼리 문자열에 직접 포함한다.
# 허용된 관계 타입만 통과시켜 사용자 입력으로 인한 인젝션을 차단한다.
_ALLOWED_RELATIONS: frozenset[str] = frozenset({
    "DIRECTED",
    "ACTED_IN",
    "SIMILAR_TO",
    "HAS_GENRE",
    "HAS_MOOD",
})


def _sanitize_relation(rel: str | None, default: str = "ACTED_IN") -> str:
    """
    관계 타입 문자열을 화이트리스트로 검증한다.

    허용된 관계 타입이면 대문자로 변환하여 반환.
    허용되지 않거나 None이면 default 값을 반환한다.

    Args:
        rel: 검증할 관계 타입 문자열
        default: 검증 실패 시 대체값

    Returns:
        화이트리스트를 통과한 관계 타입 문자열
    """
    if rel and rel.upper() in _ALLOWED_RELATIONS:
        return rel.upper()
    if rel:
        logger.warning(
            "cypher_builder_invalid_relation",
            input_rel=rel,
            fallback=default,
        )
    return default


# ============================================================
# chain 유형 Cypher 생성
# ============================================================

def build_chain_cypher(plan: dict) -> tuple[str, dict]:
    """
    chain 유형: A 감독/배우 → 영화(+장르 필터) → 배우 → 배우의 다른 영화.

    예시:
    - "봉준호 감독이 찍은 스릴러에 나온 배우들이 찍은 영화"
      → (봉준호)-[DIRECTED]->(mid:Movie, genre=스릴러)<-[ACTED_IN]-(actor)-[ACTED_IN]->(result)
    - "박찬욱과 협업한 배우들의 다른 영화"
      → (박찬욱)-[DIRECTED]->(mid)<-[ACTED_IN]-(actor)-[ACTED_IN]->(result)

    결과 정렬:
    - 여러 배우가 겹칠수록(actor_count) 상위 → 해당 감독 세계관을 가장 잘 반영하는 영화
    - 동점이면 popularity_score 내림차순

    Args:
        plan: GraphQueryPlan dict

    Returns:
        (cypher_query, params) 튜플.
        start_entity가 없으면 폴백 인기작 쿼리를 반환한다.
    """
    start_entity: str = plan.get("start_entity") or ""
    start_rel: str = _sanitize_relation(plan.get("start_relation"), "DIRECTED")
    hop_genre: str | None = plan.get("hop_genre")
    target_rel: str = _sanitize_relation(plan.get("target_relation"), "ACTED_IN")

    # 시작 엔티티가 없으면 폴백
    if not start_entity:
        logger.warning("chain_cypher_no_start_entity", plan=plan)
        return _build_fallback_cypher(), {"top_k": 20}

    params: dict = {
        "start_entity": start_entity,
        "top_k": 20,
    }

    if hop_genre:
        # 장르 필터 있음: 중간 영화에 HAS_GENRE 조건 추가
        params["hop_genre"] = hop_genre
        cypher = f"""
        MATCH (p:Person)-[:{start_rel}]->(mid:Movie)-[:HAS_GENRE]->(g:Genre)
        WHERE (p.name = $start_entity OR p.name_en = $start_entity)
          AND g.name = $hop_genre
        MATCH (actor:Person)-[:{target_rel}]->(mid)
        MATCH (actor)-[:{target_rel}]->(result:Movie)
        WHERE result <> mid
        WITH DISTINCT result, count(DISTINCT actor) AS actor_count
        RETURN
            result.id            AS movie_id,
            result.title         AS title,
            result.rating        AS rating,
            result.popularity_score AS popularity,
            actor_count          AS relation_score
        ORDER BY actor_count DESC, result.popularity_score DESC
        LIMIT $top_k
        """
    else:
        # 장르 필터 없음: 감독/배우의 전체 영화 → 출연 배우의 다른 영화
        cypher = f"""
        MATCH (p:Person)-[:{start_rel}]->(mid:Movie)
        WHERE p.name = $start_entity OR p.name_en = $start_entity
        MATCH (actor:Person)-[:{target_rel}]->(mid)
        MATCH (actor)-[:{target_rel}]->(result:Movie)
        WHERE result <> mid
        WITH DISTINCT result, count(DISTINCT actor) AS actor_count
        RETURN
            result.id            AS movie_id,
            result.title         AS title,
            result.rating        AS rating,
            result.popularity_score AS popularity,
            actor_count          AS relation_score
        ORDER BY actor_count DESC, result.popularity_score DESC
        LIMIT $top_k
        """

    logger.debug(
        "chain_cypher_built",
        start_entity=start_entity,
        start_rel=start_rel,
        hop_genre=hop_genre,
        target_rel=target_rel,
    )
    return cypher, params


# ============================================================
# intersection 유형 Cypher 생성
# ============================================================

def build_intersection_cypher(plan: dict) -> tuple[str, dict]:
    """
    intersection 유형: N명 모두 출연/감독한 영화 교집합.

    예시:
    - "최민식과 송강호 둘 다 나온 영화"
      → 최민식이 ACTED_IN이고 동시에 송강호도 ACTED_IN인 영화

    구현 방식:
    - 각 인물에 대한 MATCH 절을 WITH m으로 연결하여 교집합을 구성한다.
    - N명 모두가 관계를 맺고 있는 영화(m)만 결과로 반환된다.

    Args:
        plan: GraphQueryPlan dict

    Returns:
        (cypher_query, params) 튜플.
        persons가 2명 미만이면 폴백 쿼리를 반환한다.
    """
    persons: list[str] = plan.get("persons") or []
    relation_type: str = _sanitize_relation(plan.get("relation_type"), "ACTED_IN")

    if len(persons) < 2:
        logger.warning("intersection_cypher_insufficient_persons", persons=persons)
        return _build_fallback_cypher(), {"top_k": 20}

    params: dict = {"top_k": 20}

    # 첫 번째 인물: MATCH + WITH
    # 이후 인물: MATCH (같은 m 변수 재사용) → 교집합 구현
    cypher_parts: list[str] = []

    for i, person in enumerate(persons):
        param_key = f"person_{i}"
        params[param_key] = person

        if i == 0:
            # 첫 번째 인물: MATCH로 영화 집합 초기화
            cypher_parts.append(
                f"MATCH (p{i}:Person)-[:{relation_type}]->(m:Movie)\n"
                f"WHERE p{i}.name = ${param_key} OR p{i}.name_en = ${param_key}"
            )
        else:
            # 이후 인물: 같은 m에 대해 추가 MATCH (AND 교집합)
            cypher_parts.append(
                f"MATCH (p{i}:Person)-[:{relation_type}]->(m)\n"
                f"WHERE p{i}.name = ${param_key} OR p{i}.name_en = ${param_key}"
            )

    cypher = "\n".join(cypher_parts)
    cypher += """
    WITH DISTINCT m
    RETURN
        m.id             AS movie_id,
        m.title          AS title,
        m.rating         AS rating,
        m.popularity_score AS popularity,
        1.0              AS relation_score
    ORDER BY m.popularity_score DESC, m.rating DESC
    LIMIT $top_k
    """

    logger.debug(
        "intersection_cypher_built",
        persons=persons,
        relation_type=relation_type,
        num_persons=len(persons),
    )
    return cypher, params


# ============================================================
# person_filmography 유형 Cypher 생성
# ============================================================

def _build_filmography_cypher(plan: dict) -> tuple[str, dict]:
    """
    person_filmography 유형: 특정 인물의 필모그래피 전체.

    예시:
    - "설경구가 나온 영화 모두" → ACTED_IN
    - "봉준호 감독 전작" → DIRECTED

    Args:
        plan: GraphQueryPlan dict

    Returns:
        (cypher_query, params) 튜플.
        start_entity가 없으면 폴백 쿼리를 반환한다.
    """
    entity: str = plan.get("start_entity") or ""
    rel: str = _sanitize_relation(plan.get("start_relation"), "DIRECTED")

    if not entity:
        logger.warning("filmography_cypher_no_entity", plan=plan)
        return _build_fallback_cypher(), {"top_k": 20}

    cypher = f"""
    MATCH (p:Person)-[:{rel}]->(m:Movie)
    WHERE p.name = $entity OR p.name_en = $entity
    RETURN
        m.id             AS movie_id,
        m.title          AS title,
        m.rating         AS rating,
        m.popularity_score AS popularity,
        1.0              AS relation_score
    ORDER BY m.popularity_score DESC
    LIMIT $top_k
    """

    logger.debug("filmography_cypher_built", entity=entity, rel=rel)
    return cypher, {"entity": entity, "top_k": 20}


# ============================================================
# 폴백 Cypher — 엔티티 추출 실패 시 인기작 반환
# ============================================================

def _build_fallback_cypher() -> str:
    """
    엔티티 추출 실패 또는 알 수 없는 query_type일 때 사용하는 폴백 쿼리.

    popularity_score 기준 상위 top_k편을 반환한다.
    relation_score=0.0으로 설정하여 search_neo4j_relation에서 낮은 가중치가 적용된다.

    Returns:
        폴백 Cypher 쿼리 문자열 (파라미터: $top_k)
    """
    return """
    MATCH (m:Movie)
    WHERE m.popularity_score IS NOT NULL
    RETURN
        m.id             AS movie_id,
        m.title          AS title,
        m.rating         AS rating,
        m.popularity_score AS popularity,
        0.0              AS relation_score
    ORDER BY m.popularity_score DESC
    LIMIT $top_k
    """


# ============================================================
# 메인 진입점 — query_type에 따른 Cypher 분기
# ============================================================

def build_cypher_from_plan(plan: dict) -> tuple[str, dict]:
    """
    GraphQueryPlan → (Cypher 쿼리, 파라미터) 메인 진입점.

    query_type에 따라 적합한 Cypher 빌더를 호출한다:
    - "chain"              → build_chain_cypher
    - "intersection"       → build_intersection_cypher
    - "person_filmography" → _build_filmography_cypher
    - 기타/알 수 없음      → _build_filmography_cypher (단순 폴백)

    모든 사용자 입력값은 params dict를 통해 파라미터로 전달된다.
    관계 타입(DIRECTED, ACTED_IN)은 화이트리스트 검증 후 쿼리 문자열에 인라인된다.

    Args:
        plan: extract_graph_query_plan()이 반환한 GraphQueryPlan dict

    Returns:
        (cypher_query, params) 튜플
    """
    query_type: str = plan.get("query_type", "chain")

    logger.info(
        "cypher_builder_dispatch",
        query_type=query_type,
        start_entity=plan.get("start_entity"),
        persons=plan.get("persons"),
    )

    if query_type == "chain":
        return build_chain_cypher(plan)
    elif query_type == "intersection":
        return build_intersection_cypher(plan)
    elif query_type == "person_filmography":
        return _build_filmography_cypher(plan)
    else:
        # 알 수 없는 유형: start_entity가 있으면 필모그래피, 없으면 폴백
        logger.warning("cypher_builder_unknown_query_type", query_type=query_type)
        return _build_filmography_cypher(plan)
