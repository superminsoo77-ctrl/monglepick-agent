"""
Neo4j 영화 관계 그래프 탐색 도구 (Phase 6 Tool 7).

Neo4j 그래프 DB에서 영화, 감독, 배우, 장르, 무드태그 노드와
그 사이의 관계(DIRECTED, ACTED_IN, HAS_GENRE, HAS_MOOD 등)를 탐색한다.

"봉준호 감독 영화 알려줘", "배우 송강호가 나온 영화" 등
관계 기반 탐색 질의에 응답하는 info/search 의도 보조 도구이다.

Neo4j 스키마 (§10-3-1):
    노드: Movie(id, title, rating, popularity_score, release_year, original_language, origin_country)
          Person(name, name_en)
          Genre(name)
          MoodTag(name)
          OTTPlatform(name)
    관계: (Person)-[:DIRECTED]->(Movie)
          (Person)-[:ACTED_IN]->(Movie)
          (Movie)-[:HAS_GENRE]->(Genre)
          (Movie)-[:HAS_MOOD]->(MoodTag)
          (Movie)-[:AVAILABLE_ON]->(OTTPlatform)
          (Movie)-[:SIMILAR_TO {score}]->(Movie)
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog
from langchain_core.tools import tool

from monglepick.db.clients import get_neo4j

logger = structlog.get_logger()

# Neo4j 쿼리 타임아웃 (초)
_NEO4J_TIMEOUT_SEC = 5.0

# 탐색 깊이 최대값 — 너무 깊으면 그래프가 폭발적으로 커짐
_MAX_DEPTH = 3

# 각 탐색 전략의 최대 결과 수
_MAX_RESULTS_PER_STRATEGY = 10


@tool
async def graph_explorer(
    query: str,
    depth: int = 2,
) -> dict[str, Any]:
    """
    Neo4j 영화 관계 그래프를 탐색하여 관련 영화·인물·관계를 반환한다.

    쿼리에서 감독/배우/장르/무드 키워드를 추출하여 적합한 Cypher 쿼리를 생성한다.
    "봉준호 감독", "송강호 배우", "SF 장르", "우울한 분위기" 등의 키워드를 지원한다.

    Args:
        query: 자연어 탐색 쿼리 (예: "봉준호 감독의 영화", "송강호가 나온 영화 목록")
        depth: 탐색 깊이 (기본 2, 최대 3). depth=1이면 직접 연결만, depth=2이면 2홉 탐색.

    Returns:
        그래프 탐색 결과 dict:
        {
            "nodes": [
                {
                    "id": str,          # 노드 고유 ID (Movie의 경우 movie_id)
                    "type": str,        # 노드 타입 (Movie / Person / Genre / MoodTag)
                    "label": str,       # 표시 이름
                    "properties": dict, # 추가 속성 (rating, release_year 등)
                }
            ],
            "edges": [
                {
                    "source": str,      # 출발 노드 ID
                    "target": str,      # 도착 노드 ID
                    "relation": str,    # 관계 유형 (DIRECTED, ACTED_IN, HAS_GENRE 등)
                }
            ],
            "summary": str,             # 탐색 결과 요약 (한국어)
        }
        에러 또는 결과 없음: 빈 dict {} 반환 (에러 전파 금지).
    """
    # depth 범위 보정
    safe_depth = max(1, min(int(depth), _MAX_DEPTH))

    try:
        # 쿼리에서 탐색 대상 키워드 추출 (규칙 기반)
        extracted = _extract_keywords(query)

        logger.info(
            "graph_explorer_tool_start",
            query_preview=query[:80],
            depth=safe_depth,
            extracted=extracted,
        )

        driver = await get_neo4j()

        # 탐색 전략 선택 및 Cypher 실행
        nodes: list[dict] = []
        edges: list[dict] = []

        async with driver.session() as session:
            # 전략 1: 감독 기반 탐색
            if extracted.get("directors"):
                n, e = await asyncio.wait_for(
                    _search_by_director(session, extracted["directors"], safe_depth),
                    timeout=_NEO4J_TIMEOUT_SEC,
                )
                nodes.extend(n)
                edges.extend(e)

            # 전략 2: 배우 기반 탐색
            if extracted.get("actors"):
                n, e = await asyncio.wait_for(
                    _search_by_actor(session, extracted["actors"], safe_depth),
                    timeout=_NEO4J_TIMEOUT_SEC,
                )
                nodes.extend(n)
                edges.extend(e)

            # 전략 3: 장르 기반 탐색
            if extracted.get("genres"):
                n, e = await asyncio.wait_for(
                    _search_by_genre(session, extracted["genres"]),
                    timeout=_NEO4J_TIMEOUT_SEC,
                )
                nodes.extend(n)
                edges.extend(e)

            # 전략 4: 무드 기반 탐색
            if extracted.get("moods"):
                n, e = await asyncio.wait_for(
                    _search_by_mood(session, extracted["moods"]),
                    timeout=_NEO4J_TIMEOUT_SEC,
                )
                nodes.extend(n)
                edges.extend(e)

            # 추출된 키워드가 없으면 제목 기반 탐색으로 fallback
            if not any(extracted.values()):
                n, e = await asyncio.wait_for(
                    _search_by_title_keyword(session, query),
                    timeout=_NEO4J_TIMEOUT_SEC,
                )
                nodes.extend(n)
                edges.extend(e)

        # 노드·엣지 중복 제거 (id 기준)
        nodes = _deduplicate(nodes, key="id")
        edges = _deduplicate(edges, key=("source", "target", "relation"))

        # 탐색 요약 생성
        summary = _build_summary(extracted, nodes, edges)

        result = {"nodes": nodes, "edges": edges, "summary": summary}

        logger.info(
            "graph_explorer_tool_done",
            query_preview=query[:80],
            node_count=len(nodes),
            edge_count=len(edges),
        )
        return result

    except asyncio.TimeoutError:
        logger.error(
            "graph_explorer_tool_timeout",
            query_preview=query[:80],
            timeout_sec=_NEO4J_TIMEOUT_SEC,
        )
        return {}

    except Exception as e:
        # Neo4j 연결 실패, Cypher 오류 등 모든 예외 처리 (에러 전파 금지)
        logger.error(
            "graph_explorer_tool_error",
            query_preview=query[:80],
            error=str(e),
            error_type=type(e).__name__,
        )
        return {}


# ============================================================
# 키워드 추출 (규칙 기반)
# ============================================================

# 감독/배우 구분 트리거 단어
_DIRECTOR_TRIGGERS = ["감독", "연출", "메가폰"]
_ACTOR_TRIGGERS = ["배우", "주연", "출연", "나온"]

# 장르 키워드 → Neo4j Genre.name 매핑
_GENRE_KEYWORDS: dict[str, str] = {
    "액션": "Action", "SF": "Science Fiction", "공상과학": "Science Fiction",
    "공포": "Horror", "호러": "Horror", "코미디": "Comedy", "로맨스": "Romance",
    "로맨틱": "Romance", "스릴러": "Thriller", "드라마": "Drama",
    "애니메이션": "Animation", "애니": "Animation", "범죄": "Crime",
    "다큐": "Documentary", "다큐멘터리": "Documentary", "판타지": "Fantasy",
    "어드벤처": "Adventure", "모험": "Adventure", "미스터리": "Mystery",
    "뮤지컬": "Music", "전쟁": "War", "서부": "Western", "가족": "Family",
    "역사": "History",
}

# 무드 키워드 → Neo4j MoodTag.name 매핑 (25개 화이트리스트 기준)
_MOOD_KEYWORDS: dict[str, str] = {
    "따뜻한": "따뜻한", "따뜻함": "따뜻한", "힐링": "힐링",
    "긴장감": "긴장감 있는", "긴장": "긴장감 있는", "스릴": "긴장감 있는",
    "유쾌한": "유쾌한", "재미있는": "유쾌한", "웃긴": "유쾌한",
    "슬픈": "슬픈", "감동적인": "감동적인", "감동": "감동적인",
    "어두운": "어두운", "우울한": "우울한",
    "신나는": "신나는", "활기찬": "신나는",
    "로맨틱한": "로맨틱한", "달달한": "로맨틱한",
    "몽환적인": "몽환적인", "신비로운": "몽환적인",
    "잔잔한": "잔잔한", "평온한": "잔잔한",
    "박진감": "박진감 넘치는", "박진감 넘치는": "박진감 넘치는",
    "공포스러운": "공포스러운", "무서운": "공포스러운",
    "철학적인": "철학적인", "심오한": "철학적인",
}


def _extract_keywords(query: str) -> dict[str, list[str]]:
    """
    자연어 쿼리에서 감독·배우·장르·무드 키워드를 규칙 기반으로 추출한다.

    패턴:
    - "감독/연출" 앞의 이름 → directors
    - "배우/주연/출연/나온" 앞의 이름 → actors
    - 장르 키워드 테이블 매칭 → genres (Neo4j 영문 장르명으로 변환)
    - 무드 키워드 테이블 매칭 → moods (무드태그 한국어명)

    Args:
        query: 사용자 자연어 쿼리

    Returns:
        {"directors": [...], "actors": [...], "genres": [...], "moods": [...]}
    """
    result: dict[str, list[str]] = {
        "directors": [],
        "actors": [],
        "genres": [],
        "moods": [],
    }

    # ── 감독 추출: "이름 감독" 패턴 ──
    # 예: "봉준호 감독", "박찬욱 연출"
    for trigger in _DIRECTOR_TRIGGERS:
        # 트리거 단어 앞에 2~6글자 한국어 이름 또는 영문 이름 추출
        pattern = rf"([가-힣A-Za-z\s]{{2,15}})\s*{trigger}"
        matches = re.findall(pattern, query)
        for m in matches:
            name = m.strip()
            if name and name not in result["directors"]:
                result["directors"].append(name)

    # ── 배우 추출: "이름 배우/나온/출연" 패턴 ──
    # 예: "송강호 배우", "최민식이 나온"
    for trigger in _ACTOR_TRIGGERS:
        pattern = rf"([가-힣A-Za-z\s]{{2,15}})\s*(?:이|가)?\s*{trigger}"
        matches = re.findall(pattern, query)
        for m in matches:
            name = m.strip()
            if name and name not in result["actors"]:
                result["actors"].append(name)

    # ── 장르 추출: 키워드 테이블 매칭 ──
    for kor_keyword, eng_genre in _GENRE_KEYWORDS.items():
        if kor_keyword in query:
            if eng_genre not in result["genres"]:
                result["genres"].append(eng_genre)

    # ── 무드 추출: 키워드 테이블 매칭 ──
    for kor_keyword, mood_tag in _MOOD_KEYWORDS.items():
        if kor_keyword in query:
            if mood_tag not in result["moods"]:
                result["moods"].append(mood_tag)

    return result


# ============================================================
# Cypher 탐색 전략별 헬퍼 함수
# ============================================================

async def _search_by_director(
    session: Any,
    directors: list[str],
    depth: int,
) -> tuple[list[dict], list[dict]]:
    """
    감독 이름으로 영화를 탐색한다 (DIRECTED 관계).

    depth=2이면 해당 영화와 SIMILAR_TO 관계의 유사 영화도 함께 반환한다.

    Args:
        session: Neo4j AsyncSession
        directors: 감독 이름 목록
        depth: 탐색 깊이 (1 또는 2)

    Returns:
        (nodes, edges) 튜플
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    for director_name in directors:
        # DIRECTED 관계: Person → Movie
        cypher = """
        MATCH (p:Person)-[:DIRECTED]->(m:Movie)
        WHERE p.name CONTAINS $name OR p.name_en CONTAINS $name
        RETURN p.name AS director_name, m.id AS movie_id, m.title AS title,
               m.rating AS rating, m.release_year AS release_year,
               m.popularity_score AS popularity
        ORDER BY m.popularity_score DESC
        LIMIT $limit
        """
        result = await session.run(
            cypher,
            {"name": director_name, "limit": _MAX_RESULTS_PER_STRATEGY},
        )
        records = await result.data()

        # Person 노드 추가 (중복 방지를 위해 이름을 ID로 사용)
        if records:
            nodes.append({
                "id": f"person_{director_name}",
                "type": "Person",
                "label": records[0].get("director_name", director_name),
                "properties": {"role": "감독"},
            })

        for record in records:
            movie_id = str(record.get("movie_id", ""))
            if not movie_id:
                continue

            # Movie 노드 추가
            nodes.append({
                "id": movie_id,
                "type": "Movie",
                "label": record.get("title", ""),
                "properties": {
                    "rating": record.get("rating"),
                    "release_year": record.get("release_year"),
                },
            })

            # DIRECTED 엣지 추가
            edges.append({
                "source": f"person_{director_name}",
                "target": movie_id,
                "relation": "DIRECTED",
            })

            # depth=2: SIMILAR_TO 관계 2홉 탐색
            if depth >= 2:
                sim_cypher = """
                MATCH (m:Movie {id: $movie_id})-[r:SIMILAR_TO]->(s:Movie)
                RETURN s.id AS sim_id, s.title AS sim_title, r.score AS sim_score
                ORDER BY r.score DESC
                LIMIT 3
                """
                sim_result = await session.run(sim_cypher, {"movie_id": movie_id})
                sim_records = await sim_result.data()
                for sim in sim_records:
                    sim_id = str(sim.get("sim_id", ""))
                    if sim_id:
                        nodes.append({
                            "id": sim_id,
                            "type": "Movie",
                            "label": sim.get("sim_title", ""),
                            "properties": {"similarity_to": record.get("title", "")},
                        })
                        edges.append({
                            "source": movie_id,
                            "target": sim_id,
                            "relation": "SIMILAR_TO",
                        })

    return nodes, edges


async def _search_by_actor(
    session: Any,
    actors: list[str],
    depth: int,
) -> tuple[list[dict], list[dict]]:
    """
    배우 이름으로 영화를 탐색한다 (ACTED_IN 관계).

    Args:
        session: Neo4j AsyncSession
        actors: 배우 이름 목록
        depth: 탐색 깊이 (1이면 직접 출연작만)

    Returns:
        (nodes, edges) 튜플
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    for actor_name in actors:
        cypher = """
        MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
        WHERE p.name CONTAINS $name OR p.name_en CONTAINS $name
        RETURN p.name AS actor_name, m.id AS movie_id, m.title AS title,
               m.rating AS rating, m.release_year AS release_year
        ORDER BY m.rating DESC
        LIMIT $limit
        """
        result = await session.run(
            cypher,
            {"name": actor_name, "limit": _MAX_RESULTS_PER_STRATEGY},
        )
        records = await result.data()

        # Person 노드 추가
        if records:
            nodes.append({
                "id": f"person_{actor_name}",
                "type": "Person",
                "label": records[0].get("actor_name", actor_name),
                "properties": {"role": "배우"},
            })

        for record in records:
            movie_id = str(record.get("movie_id", ""))
            if not movie_id:
                continue

            nodes.append({
                "id": movie_id,
                "type": "Movie",
                "label": record.get("title", ""),
                "properties": {
                    "rating": record.get("rating"),
                    "release_year": record.get("release_year"),
                },
            })
            edges.append({
                "source": f"person_{actor_name}",
                "target": movie_id,
                "relation": "ACTED_IN",
            })

    return nodes, edges


async def _search_by_genre(
    session: Any,
    genres: list[str],
) -> tuple[list[dict], list[dict]]:
    """
    장르 이름으로 영화를 탐색한다 (HAS_GENRE 관계).

    Args:
        session: Neo4j AsyncSession
        genres: 장르 이름 목록 (영문, 예: ["Action", "Science Fiction"])

    Returns:
        (nodes, edges) 튜플
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    for genre_name in genres:
        cypher = """
        MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre {name: $genre})
        RETURN g.name AS genre_name, m.id AS movie_id, m.title AS title,
               m.rating AS rating, m.release_year AS release_year,
               m.popularity_score AS popularity
        ORDER BY m.popularity_score DESC
        LIMIT $limit
        """
        result = await session.run(
            cypher,
            {"genre": genre_name, "limit": _MAX_RESULTS_PER_STRATEGY},
        )
        records = await result.data()

        # Genre 노드 추가
        if records:
            nodes.append({
                "id": f"genre_{genre_name}",
                "type": "Genre",
                "label": genre_name,
                "properties": {},
            })

        for record in records:
            movie_id = str(record.get("movie_id", ""))
            if not movie_id:
                continue

            nodes.append({
                "id": movie_id,
                "type": "Movie",
                "label": record.get("title", ""),
                "properties": {
                    "rating": record.get("rating"),
                    "release_year": record.get("release_year"),
                },
            })
            edges.append({
                "source": movie_id,
                "target": f"genre_{genre_name}",
                "relation": "HAS_GENRE",
            })

    return nodes, edges


async def _search_by_mood(
    session: Any,
    moods: list[str],
) -> tuple[list[dict], list[dict]]:
    """
    무드 태그로 영화를 탐색한다 (HAS_MOOD 관계).

    Args:
        session: Neo4j AsyncSession
        moods: 무드 태그 목록 (한국어, 예: ["따뜻한", "감동적인"])

    Returns:
        (nodes, edges) 튜플
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    for mood_name in moods:
        cypher = """
        MATCH (m:Movie)-[:HAS_MOOD]->(mt:MoodTag {name: $mood})
        RETURN mt.name AS mood_name, m.id AS movie_id, m.title AS title,
               m.rating AS rating, m.release_year AS release_year,
               m.popularity_score AS popularity
        ORDER BY m.popularity_score DESC
        LIMIT $limit
        """
        result = await session.run(
            cypher,
            {"mood": mood_name, "limit": _MAX_RESULTS_PER_STRATEGY},
        )
        records = await result.data()

        # MoodTag 노드 추가
        if records:
            nodes.append({
                "id": f"mood_{mood_name}",
                "type": "MoodTag",
                "label": mood_name,
                "properties": {},
            })

        for record in records:
            movie_id = str(record.get("movie_id", ""))
            if not movie_id:
                continue

            nodes.append({
                "id": movie_id,
                "type": "Movie",
                "label": record.get("title", ""),
                "properties": {
                    "rating": record.get("rating"),
                    "release_year": record.get("release_year"),
                },
            })
            edges.append({
                "source": movie_id,
                "target": f"mood_{mood_name}",
                "relation": "HAS_MOOD",
            })

    return nodes, edges


async def _search_by_title_keyword(
    session: Any,
    keyword: str,
) -> tuple[list[dict], list[dict]]:
    """
    영화 제목 키워드로 영화를 탐색한다 (키워드 추출 실패 시 fallback).

    Args:
        session: Neo4j AsyncSession
        keyword: 검색 키워드 (자연어 그대로)

    Returns:
        (nodes, edges) 튜플
    """
    nodes: list[dict] = []
    edges: list[dict] = []

    cypher = """
    MATCH (m:Movie)
    WHERE m.title CONTAINS $keyword
    RETURN m.id AS movie_id, m.title AS title,
           m.rating AS rating, m.release_year AS release_year
    ORDER BY m.rating DESC
    LIMIT $limit
    """
    result = await session.run(
        cypher,
        {"keyword": keyword[:20], "limit": _MAX_RESULTS_PER_STRATEGY},
    )
    records = await result.data()

    for record in records:
        movie_id = str(record.get("movie_id", ""))
        if not movie_id:
            continue
        nodes.append({
            "id": movie_id,
            "type": "Movie",
            "label": record.get("title", ""),
            "properties": {
                "rating": record.get("rating"),
                "release_year": record.get("release_year"),
            },
        })

    return nodes, edges


# ============================================================
# 중복 제거 유틸
# ============================================================

def _deduplicate(
    items: list[dict],
    key: str | tuple,
) -> list[dict]:
    """
    dict 목록에서 지정 key 기준으로 중복을 제거한다.

    Args:
        items: dict 목록
        key: 중복 판별 키 (단일 문자열 또는 복합 키 튜플)

    Returns:
        중복 제거된 dict 목록 (첫 번째 등장 순서 유지)
    """
    seen: set = set()
    result: list[dict] = []
    for item in items:
        if isinstance(key, tuple):
            # 복합 키: (source, target, relation) 등
            dedup_key = tuple(item.get(k, "") for k in key)
        else:
            dedup_key = item.get(key, "")

        if dedup_key not in seen:
            seen.add(dedup_key)
            result.append(item)
    return result


# ============================================================
# 탐색 결과 요약 생성
# ============================================================

def _build_summary(
    extracted: dict[str, list[str]],
    nodes: list[dict],
    edges: list[dict],
) -> str:
    """
    그래프 탐색 결과를 한국어 요약 문자열로 생성한다.

    Args:
        extracted: 추출된 키워드 dict
        nodes: 탐색된 노드 목록
        edges: 탐색된 엣지 목록

    Returns:
        한국어 요약 문자열
    """
    movie_nodes = [n for n in nodes if n.get("type") == "Movie"]
    movie_titles = [n.get("label", "") for n in movie_nodes if n.get("label")][:5]

    parts: list[str] = []

    if extracted.get("directors"):
        parts.append(f"감독 '{', '.join(extracted['directors'])}'")
    if extracted.get("actors"):
        parts.append(f"배우 '{', '.join(extracted['actors'])}'")
    if extracted.get("genres"):
        parts.append(f"장르 '{', '.join(extracted['genres'])}'")
    if extracted.get("moods"):
        parts.append(f"분위기 '{', '.join(extracted['moods'])}'")

    subject = " / ".join(parts) if parts else "관련"

    if movie_titles:
        title_str = ", ".join(f"'{t}'" for t in movie_titles)
        suffix = f" 외 {len(movie_nodes) - len(movie_titles)}편" if len(movie_nodes) > 5 else ""
        return f"{subject} 영화 {len(movie_nodes)}편 탐색 완료: {title_str}{suffix}"
    else:
        return f"'{', '.join(parts) if parts else '해당 조건'}' 관련 영화를 찾지 못했어요"
