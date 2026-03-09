"""
Neo4j 지식그래프 적재기.

§11-7 Neo4j 그래프 구축 명세:
- 노드: Movie, Person, Genre, Keyword, MoodTag, OTTPlatform
- 관계: DIRECTED, ACTED_IN, HAS_GENRE, HAS_KEYWORD, HAS_MOOD, AVAILABLE_ON, SIMILAR_TO

§11-7-2 배치 적재 패턴:
- UNWIND 배치로 노드/관계 생성 (500건/배치)
- 모든 노드/관계에 MERGE 사용 (중복 방지)
- 적재 순서: 노드 → 관계 → SIMILAR_TO → 인덱스 확인
"""

from __future__ import annotations

import structlog

from monglepick.data_pipeline.models import MovieDocument
from monglepick.db.clients import get_neo4j

logger = structlog.get_logger()

# 배치 크기 (§11-7-2)
NODE_BATCH_SIZE = 500
RELATION_BATCH_SIZE = 500


async def _run_batch(cypher: str, params: dict) -> None:
    """
    Neo4j Cypher 쿼리를 단일 실행한다.

    Args:
        cypher: 실행할 Cypher 쿼리 문자열 ($batch 파라미터 포함)
        params: 쿼리 파라미터 딕셔너리 (예: {"batch": [...]})
    """
    driver = await get_neo4j()
    async with driver.session() as session:
        await session.run(cypher, params)


async def _batch_execute(cypher: str, data: list[dict], batch_size: int, label: str) -> None:
    """
    데이터를 배치로 나누어 UNWIND Cypher를 실행한다.

    대량 데이터를 한 번에 전송하면 Neo4j 메모리 부족이 발생할 수 있으므로
    batch_size 단위로 분할하여 순차 실행한다.

    Args:
        cypher: UNWIND $batch 패턴을 포함하는 Cypher 쿼리
        data: UNWIND 대상 딕셔너리 리스트
        batch_size: 배치당 최대 건수 (기본 500)
        label: 로깅용 라벨 (예: "movie_nodes", "directed_relations")
    """
    if not data:
        logger.info(f"neo4j_{label}_skipped", count=0)
        return

    # batch_size가 0 이하이면 기본값 사용 (Genre/OTT 등 소량 데이터에서 len(data)=0일 때 방어)
    if batch_size <= 0:
        batch_size = NODE_BATCH_SIZE

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        await _run_batch(cypher, {"batch": batch})

    logger.info(f"neo4j_{label}_loaded", count=len(data))


# ============================================================
# Step 1: 노드 생성
# ============================================================

async def load_movie_nodes(documents: list[MovieDocument]) -> None:
    """
    Movie 노드를 배치 생성한다.

    §11-7-2: 500건/배치, MERGE 사용
    Phase B: budget/revenue/tagline/imdb_id/original_language/adult/status/backdrop_path 추가
    """
    data = [
        {
            "id": doc.id,
            "title": doc.title,
            "title_en": doc.title_en,
            "release_year": doc.release_year,
            "runtime": doc.runtime,
            "rating": doc.rating,
            "popularity_score": doc.popularity_score,
            "poster_path": doc.poster_path,
            # Phase A: 보강 속성
            "certification": doc.certification,
            "vote_count": doc.vote_count,
            "trailer_url": doc.trailer_url,
            # Phase B: 재무/텍스트 메타데이터
            "budget": doc.budget,
            "revenue": doc.revenue,
            "tagline": doc.tagline,
            "imdb_id": doc.imdb_id,
            "original_language": doc.original_language,
            "adult": doc.adult,
            "status": doc.status,
            "backdrop_path": doc.backdrop_path,
            "homepage": doc.homepage,
            "overview": doc.overview,
            "spoken_languages": doc.spoken_languages,
            # KMDb 보강 속성
            "kmdb_id": doc.kmdb_id,
            "awards": doc.awards,
            "audience_count": doc.audience_count,
            "filming_location": doc.filming_location,
            # Phase C: 완전 데이터 추출
            "origin_country": doc.origin_country,
            "director_id": doc.director_id,
            "kr_release_date": doc.kr_release_date,
            "video_flag": doc.video_flag,
            "collection_poster_path": doc.collection_poster_path,
            "collection_backdrop_path": doc.collection_backdrop_path,
            "production_country_names": doc.production_country_names,
            "spoken_language_names": doc.spoken_language_names,
            # KOBIS 보강 필드
            "kobis_movie_cd": doc.kobis_movie_cd,
            "sales_acc": doc.sales_acc,
            "screen_count": doc.screen_count,
            "kobis_nation": doc.kobis_nation,
            "kobis_watch_grade": doc.kobis_watch_grade,
            "kobis_open_dt": doc.kobis_open_dt,
            "kobis_type_nm": doc.kobis_type_nm,
            # Phase D: 전체 수집 보강 필드
            "overview_en": doc.overview_en,
            "overview_ja": doc.overview_ja,
            "facebook_id": doc.facebook_id,
            "instagram_id": doc.instagram_id,
            "twitter_id": doc.twitter_id,
            "wikidata_id": doc.wikidata_id,
            "tmdb_list_count": doc.tmdb_list_count,
            # 데이터 출처 추적
            "source": doc.source,
        }
        for doc in documents
    ]

    cypher = """
    UNWIND $batch AS m
    MERGE (movie:Movie {id: m.id})
    SET movie.title = m.title,
        movie.title_en = m.title_en,
        movie.release_year = m.release_year,
        movie.runtime = m.runtime,
        movie.rating = m.rating,
        movie.popularity_score = m.popularity_score,
        movie.poster_path = m.poster_path,
        movie.certification = m.certification,
        movie.vote_count = m.vote_count,
        movie.trailer_url = m.trailer_url,
        movie.budget = m.budget,
        movie.revenue = m.revenue,
        movie.tagline = m.tagline,
        movie.imdb_id = m.imdb_id,
        movie.original_language = m.original_language,
        movie.adult = m.adult,
        movie.status = m.status,
        movie.backdrop_path = m.backdrop_path,
        movie.homepage = m.homepage,
        movie.overview = m.overview,
        movie.spoken_languages = m.spoken_languages,
        movie.kmdb_id = m.kmdb_id,
        movie.awards = m.awards,
        movie.audience_count = m.audience_count,
        movie.filming_location = m.filming_location,
        movie.origin_country = m.origin_country,
        movie.director_id = m.director_id,
        movie.kr_release_date = m.kr_release_date,
        movie.video_flag = m.video_flag,
        movie.collection_poster_path = m.collection_poster_path,
        movie.collection_backdrop_path = m.collection_backdrop_path,
        movie.production_country_names = m.production_country_names,
        movie.spoken_language_names = m.spoken_language_names,
        movie.kobis_movie_cd = m.kobis_movie_cd,
        movie.sales_acc = m.sales_acc,
        movie.screen_count = m.screen_count,
        movie.kobis_nation = m.kobis_nation,
        movie.kobis_watch_grade = m.kobis_watch_grade,
        movie.kobis_open_dt = m.kobis_open_dt,
        movie.kobis_type_nm = m.kobis_type_nm,
        movie.overview_en = m.overview_en,
        movie.overview_ja = m.overview_ja,
        movie.facebook_id = m.facebook_id,
        movie.instagram_id = m.instagram_id,
        movie.twitter_id = m.twitter_id,
        movie.wikidata_id = m.wikidata_id,
        movie.tmdb_list_count = m.tmdb_list_count,
        movie.source = m.source
    """
    await _batch_execute(cypher, data, NODE_BATCH_SIZE, "movie_nodes")


async def load_genre_nodes(documents: list[MovieDocument]) -> None:
    """Genre 노드를 생성한다. (~20개, 배치 불필요)"""
    genres = list({genre for doc in documents for genre in doc.genres})
    data = [{"name": g} for g in genres]

    cypher = """
    UNWIND $batch AS g
    MERGE (:Genre {name: g.name})
    """
    await _batch_execute(cypher, data, len(data), "genre_nodes")


async def load_mood_tag_nodes(documents: list[MovieDocument]) -> None:
    """MoodTag 노드를 생성한다. (~25개, 배치 불필요)"""
    tags = list({tag for doc in documents for tag in doc.mood_tags})
    data = [{"name": t} for t in tags]

    cypher = """
    UNWIND $batch AS t
    MERGE (:MoodTag {name: t.name})
    """
    await _batch_execute(cypher, data, len(data), "mood_tag_nodes")


async def load_ott_nodes(documents: list[MovieDocument]) -> None:
    """OTTPlatform 노드를 생성한다. (~10개, 배치 불필요)"""
    platforms = list({p for doc in documents for p in doc.ott_platforms})
    data = [{"name": p} for p in platforms]

    cypher = """
    UNWIND $batch AS p
    MERGE (:OTTPlatform {name: p.name})
    """
    await _batch_execute(cypher, data, len(data), "ott_nodes")


async def load_person_nodes(documents: list[MovieDocument]) -> None:
    """
    Person 노드를 생성한다.

    Phase C 확장: 모든 크루 직군 포함 + TMDB person ID/프로필 사진 속성 추가.
    감독 + 배우 + 촬영감독 + 작곡가 + 각본가 + 프로듀서 + 편집자
    + 총괄 프로듀서 + 프로덕션 디자이너 + 의상 디자이너 + 원작 작가

    Args:
        documents: Person 노드를 추출할 MovieDocument 리스트
    """
    # 이름을 키로 사용하여 중복 제거하면서 ID/프로필 정보는 덮어쓴다
    persons_by_name: dict[str, dict] = {}
    for doc in documents:
        # 감독은 ID/프로필 정보가 있으므로 우선 등록
        if doc.director:
            persons_by_name[doc.director] = {
                "name": doc.director,
                "person_id": doc.director_id,
                "profile_path": doc.director_profile_path,
            }
        # 배우 (cast_characters에 ID/프로필 있음)
        for actor_info in doc.cast_characters:
            name = actor_info.get("name", "")
            if name:
                persons_by_name[name] = {
                    "name": name,
                    "person_id": actor_info.get("id", 0),
                    "profile_path": actor_info.get("profile_path", ""),
                }
        # cast 배열에만 있고 cast_characters에 없는 배우 (fallback)
        for name in doc.cast:
            if name and name not in persons_by_name:
                persons_by_name[name] = {"name": name, "person_id": 0, "profile_path": ""}
        # Phase B: 확장 크루
        for name in [doc.cinematographer, doc.composer, doc.editor,
                      doc.production_designer, doc.costume_designer, doc.source_author]:
            if name and name not in persons_by_name:
                persons_by_name[name] = {"name": name, "person_id": 0, "profile_path": ""}
        for name_list in [doc.screenwriters, doc.producers, doc.executive_producers]:
            for name in name_list:
                if name and name not in persons_by_name:
                    persons_by_name[name] = {"name": name, "person_id": 0, "profile_path": ""}

    data = list(persons_by_name.values())

    # MERGE로 중복 방지, CASE WHEN으로 기존 값이 있으면 유지 (더 상세한 정보 우선)
    cypher = """
    UNWIND $batch AS p
    MERGE (person:Person {name: p.name})
    SET person.person_id = CASE WHEN p.person_id > 0 THEN p.person_id ELSE person.person_id END,
        person.profile_path = CASE WHEN p.profile_path <> '' THEN p.profile_path ELSE person.profile_path END
    """
    await _batch_execute(cypher, data, NODE_BATCH_SIZE, "person_nodes")


async def load_keyword_nodes(documents: list[MovieDocument]) -> None:
    """Keyword 노드를 배치 생성한다. 영화 수가 많으면 수만 개의 키워드가 생성될 수 있어 배치 처리한다."""
    keywords = list({kw for doc in documents for kw in doc.keywords})
    data = [{"name": k} for k in keywords if k]

    cypher = """
    UNWIND $batch AS k
    MERGE (:Keyword {name: k.name})
    """
    await _batch_execute(cypher, data, NODE_BATCH_SIZE, "keyword_nodes")


# ============================================================
# Step 2: 관계 생성
# ============================================================

async def load_directed_relations(documents: list[MovieDocument]) -> None:
    """DIRECTED 관계를 배치 생성한다. §11-7: director 필드가 존재하고 비어있지 않을 때."""
    data = [
        {"movie_id": doc.id, "name": doc.director}
        for doc in documents
        if doc.director
    ]

    cypher = """
    UNWIND $batch AS d
    MATCH (p:Person {name: d.name})
    MATCH (m:Movie {id: d.movie_id})
    MERGE (p)-[:DIRECTED]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "directed_relations")


async def load_acted_in_relations(documents: list[MovieDocument]) -> None:
    """ACTED_IN 관계를 배치 생성한다."""
    data = [
        {"movie_id": doc.id, "name": actor}
        for doc in documents
        for actor in doc.cast
        if actor
    ]

    cypher = """
    UNWIND $batch AS a
    MATCH (p:Person {name: a.name})
    MATCH (m:Movie {id: a.movie_id})
    MERGE (p)-[:ACTED_IN]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "acted_in_relations")


async def load_has_genre_relations(documents: list[MovieDocument]) -> None:
    """HAS_GENRE 관계를 배치 생성한다."""
    data = [
        {"movie_id": doc.id, "name": genre}
        for doc in documents
        for genre in doc.genres
    ]

    cypher = """
    UNWIND $batch AS g
    MATCH (m:Movie {id: g.movie_id})
    MATCH (genre:Genre {name: g.name})
    MERGE (m)-[:HAS_GENRE]->(genre)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "has_genre_relations")


async def load_has_keyword_relations(documents: list[MovieDocument]) -> None:
    """HAS_KEYWORD 관계를 배치 생성한다."""
    data = [
        {"movie_id": doc.id, "name": kw}
        for doc in documents
        for kw in doc.keywords
        if kw
    ]

    cypher = """
    UNWIND $batch AS k
    MATCH (m:Movie {id: k.movie_id})
    MATCH (kw:Keyword {name: k.name})
    MERGE (m)-[:HAS_KEYWORD]->(kw)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "has_keyword_relations")


async def load_has_mood_relations(documents: list[MovieDocument]) -> None:
    """HAS_MOOD 관계를 배치 생성한다."""
    data = [
        {"movie_id": doc.id, "name": tag}
        for doc in documents
        for tag in doc.mood_tags
    ]

    cypher = """
    UNWIND $batch AS t
    MATCH (m:Movie {id: t.movie_id})
    MATCH (mt:MoodTag {name: t.name})
    MERGE (m)-[:HAS_MOOD]->(mt)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "has_mood_relations")


async def load_available_on_relations(documents: list[MovieDocument]) -> None:
    """AVAILABLE_ON 관계를 배치 생성한다."""
    data = [
        {"movie_id": doc.id, "name": platform}
        for doc in documents
        for platform in doc.ott_platforms
    ]

    cypher = """
    UNWIND $batch AS o
    MATCH (m:Movie {id: o.movie_id})
    MATCH (ott:OTTPlatform {name: o.name})
    MERGE (m)-[:AVAILABLE_ON]->(ott)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "available_on_relations")


# ============================================================
# Phase B: 새 노드 타입 (Studio, Collection, Country)
# ============================================================


async def load_studio_nodes(documents: list[MovieDocument]) -> None:
    """Studio 노드를 생성한다 (제작사). Phase B."""
    studios: dict[int, str] = {}
    for doc in documents:
        for company in doc.production_companies:
            cid = company.get("id", 0)
            cname = company.get("name", "")
            if cname and cid:
                studios[cid] = cname
    data = [{"id": sid, "name": sname} for sid, sname in studios.items()]

    cypher = """
    UNWIND $batch AS s
    MERGE (studio:Studio {id: s.id})
    SET studio.name = s.name
    """
    await _batch_execute(cypher, data, NODE_BATCH_SIZE, "studio_nodes")


async def load_collection_nodes(documents: list[MovieDocument]) -> None:
    """Collection 노드를 생성한다 (프랜차이즈/시리즈). Phase B."""
    collections: dict[int, str] = {}
    for doc in documents:
        if doc.collection_id and doc.collection_name:
            collections[doc.collection_id] = doc.collection_name
    data = [{"id": cid, "name": cname} for cid, cname in collections.items()]

    cypher = """
    UNWIND $batch AS c
    MERGE (col:Collection {id: c.id})
    SET col.name = c.name
    """
    await _batch_execute(cypher, data, max(len(data), 1), "collection_nodes")


async def load_country_nodes(documents: list[MovieDocument]) -> None:
    """Country 노드를 생성한다 (제작 국가). Phase B."""
    countries = list({c for doc in documents for c in doc.production_countries if c})
    data = [{"iso_code": c} for c in countries]

    cypher = """
    UNWIND $batch AS c
    MERGE (:Country {iso_code: c.iso_code})
    """
    await _batch_execute(cypher, data, max(len(data), 1), "country_nodes")


# ============================================================
# Phase B: 새 관계 타입
# ============================================================


async def load_produced_by_relations(documents: list[MovieDocument]) -> None:
    """PRODUCED_BY 관계를 배치 생성한다 (Movie → Studio). Phase B."""
    data = [
        {"movie_id": doc.id, "studio_id": c["id"]}
        for doc in documents
        for c in doc.production_companies
        if c.get("id")
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (m:Movie {id: r.movie_id})
    MATCH (s:Studio {id: r.studio_id})
    MERGE (m)-[:PRODUCED_BY]->(s)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "produced_by_relations")


async def load_part_of_collection_relations(documents: list[MovieDocument]) -> None:
    """PART_OF_COLLECTION 관계를 배치 생성한다 (Movie → Collection). Phase B."""
    data = [
        {"movie_id": doc.id, "collection_id": doc.collection_id}
        for doc in documents
        if doc.collection_id
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (m:Movie {id: r.movie_id})
    MATCH (c:Collection {id: r.collection_id})
    MERGE (m)-[:PART_OF_COLLECTION]->(c)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "part_of_collection_relations")


async def load_produced_in_relations(documents: list[MovieDocument]) -> None:
    """PRODUCED_IN 관계를 배치 생성한다 (Movie → Country). Phase B."""
    data = [
        {"movie_id": doc.id, "iso_code": c}
        for doc in documents
        for c in doc.production_countries
        if c
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (m:Movie {id: r.movie_id})
    MATCH (c:Country {iso_code: r.iso_code})
    MERGE (m)-[:PRODUCED_IN]->(c)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "produced_in_relations")


async def load_shot_by_relations(documents: list[MovieDocument]) -> None:
    """SHOT_BY 관계를 배치 생성한다 (Person → Movie, 촬영감독). Phase B."""
    data = [
        {"movie_id": doc.id, "name": doc.cinematographer}
        for doc in documents
        if doc.cinematographer
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (p:Person {name: r.name})
    MATCH (m:Movie {id: r.movie_id})
    MERGE (p)-[:SHOT_BY]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "shot_by_relations")


async def load_composed_by_relations(documents: list[MovieDocument]) -> None:
    """COMPOSED_BY 관계를 배치 생성한다 (Person → Movie, 작곡가). Phase B."""
    data = [
        {"movie_id": doc.id, "name": doc.composer}
        for doc in documents
        if doc.composer
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (p:Person {name: r.name})
    MATCH (m:Movie {id: r.movie_id})
    MERGE (p)-[:COMPOSED_BY]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "composed_by_relations")


async def load_written_by_relations(documents: list[MovieDocument]) -> None:
    """WRITTEN_BY 관계를 배치 생성한다 (Person → Movie, 각본가). Phase B."""
    data = [
        {"movie_id": doc.id, "name": writer}
        for doc in documents
        for writer in doc.screenwriters
        if writer
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (p:Person {name: r.name})
    MATCH (m:Movie {id: r.movie_id})
    MERGE (p)-[:WRITTEN_BY]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "written_by_relations")


async def load_produced_relations(documents: list[MovieDocument]) -> None:
    """PRODUCED 관계를 배치 생성한다 (Person → Movie, 프로듀서). Phase B."""
    data = [
        {"movie_id": doc.id, "name": producer}
        for doc in documents
        for producer in doc.producers
        if producer
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (p:Person {name: r.name})
    MATCH (m:Movie {id: r.movie_id})
    MERGE (p)-[:PRODUCED]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "produced_relations")


async def load_edited_by_relations(documents: list[MovieDocument]) -> None:
    """EDITED_BY 관계를 배치 생성한다 (Person → Movie, 편집자). Phase B."""
    data = [
        {"movie_id": doc.id, "name": doc.editor}
        for doc in documents
        if doc.editor
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (p:Person {name: r.name})
    MATCH (m:Movie {id: r.movie_id})
    MERGE (p)-[:EDITED_BY]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "edited_by_relations")


# ============================================================
# Phase C: 추가 크루 관계
# ============================================================


async def load_executive_produced_relations(documents: list[MovieDocument]) -> None:
    """EXECUTIVE_PRODUCED 관계를 배치 생성한다 (Person → Movie, 총괄 프로듀서). Phase C."""
    data = [
        {"movie_id": doc.id, "name": name}
        for doc in documents
        for name in doc.executive_producers
        if name
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (p:Person {name: r.name})
    MATCH (m:Movie {id: r.movie_id})
    MERGE (p)-[:EXECUTIVE_PRODUCED]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "executive_produced_relations")


async def load_designed_relations(documents: list[MovieDocument]) -> None:
    """DESIGNED 관계를 배치 생성한다 (Person → Movie, 프로덕션 디자이너). Phase C."""
    data = [
        {"movie_id": doc.id, "name": doc.production_designer}
        for doc in documents
        if doc.production_designer
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (p:Person {name: r.name})
    MATCH (m:Movie {id: r.movie_id})
    MERGE (p)-[:DESIGNED]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "designed_relations")


async def load_costumed_relations(documents: list[MovieDocument]) -> None:
    """COSTUMED 관계를 배치 생성한다 (Person → Movie, 의상 디자이너). Phase C."""
    data = [
        {"movie_id": doc.id, "name": doc.costume_designer}
        for doc in documents
        if doc.costume_designer
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (p:Person {name: r.name})
    MATCH (m:Movie {id: r.movie_id})
    MERGE (p)-[:COSTUMED]->(m)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "costumed_relations")


async def load_based_on_relations(documents: list[MovieDocument]) -> None:
    """BASED_ON 관계를 배치 생성한다 (Movie → Person, 원작 작가). Phase C."""
    data = [
        {"movie_id": doc.id, "name": doc.source_author}
        for doc in documents
        if doc.source_author
    ]
    cypher = """
    UNWIND $batch AS r
    MATCH (m:Movie {id: r.movie_id})
    MATCH (p:Person {name: r.name})
    MERGE (m)-[:BASED_ON]->(p)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "based_on_relations")


async def load_recommended_relations(documents: list[MovieDocument]) -> None:
    """RECOMMENDED 관계를 TMDB recommendations 데이터로 생성한다 (similar_to와 다른 알고리즘). Phase C."""
    data = [
        {"movie_id": doc.id, "rec_id": rid}
        for doc in documents
        for rid in doc.recommendation_ids
        if rid
    ]
    if not data:
        logger.info("neo4j_recommended_skipped", count=0)
        return

    cypher = """
    UNWIND $batch AS r
    MATCH (m1:Movie {id: r.movie_id})
    MATCH (m2:Movie {id: r.rec_id})
    MERGE (m1)-[:RECOMMENDED]->(m2)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "recommended_relations")


# ============================================================
# Phase A: SIMILAR_TO 관계
# ============================================================


async def load_similar_to_relations(documents: list[MovieDocument]) -> None:
    """
    SIMILAR_TO 관계를 TMDB similar 데이터로 생성한다.

    Phase A: 단방향 관계 (TMDB 기준). 양쪽 Movie 노드가 모두 존재할 때만 관계를 생성한다.
    MATCH를 사용하여 존재하는 노드 간에만 MERGE 수행.
    """
    # movie_id → similar_movie_id 쌍 생성
    data = [
        {"movie_id": doc.id, "similar_id": sid}
        for doc in documents
        for sid in doc.similar_movie_ids
        if sid  # 빈 문자열 방어
    ]

    if not data:
        logger.info("neo4j_similar_to_skipped", count=0)
        return

    cypher = """
    UNWIND $batch AS s
    MATCH (m1:Movie {id: s.movie_id})
    MATCH (m2:Movie {id: s.similar_id})
    MERGE (m1)-[:SIMILAR_TO]->(m2)
    """
    await _batch_execute(cypher, data, RELATION_BATCH_SIZE, "similar_to_relations")


# ============================================================
# 전체 적재 오케스트레이션
# ============================================================

async def load_to_neo4j(documents: list[MovieDocument]) -> None:
    """
    MovieDocument 리스트를 Neo4j 그래프에 적재한다.

    적재 순서가 중요하다: 반드시 노드를 먼저 생성한 뒤 관계를 생성해야 한다.
    MATCH 기반 관계 생성은 양쪽 노드가 존재하지 않으면 무시된다.

    §11-7-2 적재 순서:
      Step 1: 9종 노드 생성 (Movie, Person, Genre, Keyword, MoodTag, OTTPlatform, Studio, Collection, Country)
      Step 2: 6종 기본 관계 (DIRECTED, ACTED_IN, HAS_GENRE, HAS_KEYWORD, HAS_MOOD, AVAILABLE_ON)
      Step 3: SIMILAR_TO 관계 (Phase A)
      Step 4: 8종 확장 관계 (Phase B — 제작사/컬렉션/국가/확장크루)
      Step 5: 5종 추가 관계 (Phase C — 총괄프로듀서/디자이너/원작/RECOMMENDED)

    Args:
        documents: 적재할 MovieDocument 리스트
    """
    logger.info("neo4j_load_started", count=len(documents))

    # ── Step 1: 노드 생성 (9종) ──
    # 노드가 먼저 존재해야 관계 MERGE가 동작한다
    await load_movie_nodes(documents)
    await load_person_nodes(documents)
    await load_genre_nodes(documents)
    await load_keyword_nodes(documents)
    await load_mood_tag_nodes(documents)
    await load_ott_nodes(documents)
    await load_studio_nodes(documents)
    await load_collection_nodes(documents)
    await load_country_nodes(documents)

    # ── Step 2: 기본 관계 생성 (6종) ──
    await load_directed_relations(documents)
    await load_acted_in_relations(documents)
    await load_has_genre_relations(documents)
    await load_has_keyword_relations(documents)
    await load_has_mood_relations(documents)
    await load_available_on_relations(documents)

    # ── Step 3: Phase A — SIMILAR_TO 관계 ──
    await load_similar_to_relations(documents)

    # ── Step 4: Phase B — 확장 관계 (8종) ──
    await load_produced_by_relations(documents)
    await load_part_of_collection_relations(documents)
    await load_produced_in_relations(documents)
    await load_shot_by_relations(documents)
    await load_composed_by_relations(documents)
    await load_written_by_relations(documents)
    await load_produced_relations(documents)
    await load_edited_by_relations(documents)

    # ── Step 5: Phase C — 추가 크루 관계 + RECOMMENDED (5종) ──
    await load_executive_produced_relations(documents)
    await load_designed_relations(documents)
    await load_costumed_relations(documents)
    await load_based_on_relations(documents)
    await load_recommended_relations(documents)

    logger.info("neo4j_load_complete", count=len(documents))
