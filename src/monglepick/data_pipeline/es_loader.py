"""
Elasticsearch BM25 인덱스 적재기.

§10-8 Elasticsearch 인덱스 설정:
- 인덱스명: movies_bm25
- Nori 한국어 분석기 (형태소 분석)
- 12개 필드 매핑 (text + keyword + numeric)
- 배치 적재: bulk API 사용
"""

from __future__ import annotations

import structlog
from elasticsearch.helpers import async_bulk

from monglepick.data_pipeline.models import MovieDocument
from monglepick.db.clients import ES_INDEX_NAME, get_elasticsearch

logger = structlog.get_logger()


def _movie_to_es_doc(doc: MovieDocument) -> dict:
    """
    MovieDocument를 Elasticsearch bulk API 형식의 dict로 변환한다.

    반환되는 dict는 elasticsearch-py의 async_bulk()이 요구하는 형태로,
    _index, _id, _source 키를 포함한다.
    list[dict] 필드(cast, keywords, production_companies 등)는 공백 구분 텍스트로
    변환하여 Nori 형태소 분석기가 토큰화할 수 있게 한다.

    Args:
        doc: 변환할 MovieDocument 인스턴스

    Returns:
        dict: ES bulk API 형식의 문서 (_index, _id, _source 포함)
    """
    return {
        "_index": ES_INDEX_NAME,
        "_id": doc.id,
        "_source": {
            # ── 기본 메타데이터 ──
            "id": doc.id,
            "title": doc.title,
            "title_en": doc.title_en,
            "director": doc.director,
            "overview": doc.overview,
            # 배열 필드는 공백 구분 텍스트로 변환 (Nori 분석기 토큰화 대상)
            "cast": " ".join(doc.cast),
            "keywords": " ".join(doc.keywords),
            "genres": doc.genres,
            "mood_tags": doc.mood_tags,
            "ott_platforms": doc.ott_platforms,
            "release_year": doc.release_year,
            "rating": doc.rating,
            "popularity_score": doc.popularity_score,
            "runtime": doc.runtime,
            "poster_path": doc.poster_path,
            # ── Phase A: 리뷰/관람등급/트레일러 ──
            "reviews": " ".join(doc.reviews[:3]),
            "certification": doc.certification,
            "trailer_url": doc.trailer_url,
            "vote_count": doc.vote_count,
            "behind_the_scenes": doc.behind_the_scenes,
            "similar_movie_ids": doc.similar_movie_ids,
            # ── KMDb 보강 필드 ──
            "awards": doc.awards,
            "filming_location": doc.filming_location,
            "audience_count": doc.audience_count,
            # ── Phase B: 텍스트 검색 필드 ──
            "tagline": doc.tagline,
            "collection_name": doc.collection_name,
            "production_companies": " ".join(c.get("name", "") for c in doc.production_companies),
            "screenwriters": " ".join(doc.screenwriters),
            "cinematographer": doc.cinematographer,
            "composer": doc.composer,
            "producers": " ".join(doc.producers),
            "editor": doc.editor,
            # ── Phase B: 필터링 필드 ──
            "original_language": doc.original_language,
            "production_countries": doc.production_countries,
            "budget": doc.budget,
            "revenue": doc.revenue,
            "adult": doc.adult,
            "status": doc.status,
            "collection_id": doc.collection_id,
            "imdb_id": doc.imdb_id,
            "spoken_languages": doc.spoken_languages,
            "backdrop_path": doc.backdrop_path,
            "homepage": doc.homepage,
            # ── Phase C: 완전 데이터 추출 ──
            "origin_country": doc.origin_country,
            "director_id": doc.director_id,
            "alternative_titles": [t.get("title", "") for t in doc.alternative_titles],
            "recommendation_ids": doc.recommendation_ids,
            "kr_release_date": doc.kr_release_date,
            "video_flag": doc.video_flag,
            "executive_producers": " ".join(doc.executive_producers),
            "production_designer": doc.production_designer,
            "costume_designer": doc.costume_designer,
            "source_author": doc.source_author,
            "production_country_names": doc.production_country_names,
            "spoken_language_names": doc.spoken_language_names,
            # 캐스트 캐릭터: "배우명(역할)" 형식으로 결합하여 전문 검색 가능하게 함
            "cast_characters": " ".join(
                f"{c.get('name', '')}({c.get('character', '')})"
                for c in doc.cast_characters
            ),
            # 임베딩 텍스트 (디버깅 및 감사 추적용, 검색 대상 아님)
            "embedding_text": doc.embedding_text,
            # ── KOBIS 보강 필드 ──
            "kobis_movie_cd": doc.kobis_movie_cd,
            "sales_acc": doc.sales_acc,
            "screen_count": doc.screen_count,
            "kobis_genres": doc.kobis_genres,
            "kobis_nation": doc.kobis_nation,
            "kobis_watch_grade": doc.kobis_watch_grade,
            "kobis_open_dt": doc.kobis_open_dt,
            "kobis_type_nm": doc.kobis_type_nm,
            # KOBIS list[dict] 필드는 ES에 직접 저장 불가 → 텍스트로 변환
            # (Neo4j도 동일 이유로 dict 필드를 제외함)
            "kobis_directors": " ".join(
                d.get("peopleNm", "") for d in doc.kobis_directors
            ),
            "kobis_actors": " ".join(
                f"{a.get('peopleNm', '')}({a.get('cast', '')})"
                for a in doc.kobis_actors
            ),
            "kobis_companies": " ".join(
                f"{c.get('companyNm', '')}({c.get('companyPartNm', '')})"
                for c in doc.kobis_companies
            ),
            "kobis_staffs": " ".join(
                f"{s.get('peopleNm', '')}({s.get('staffRoleNm', '')})"
                for s in doc.kobis_staffs
            ),
            # ── Phase D: 전체 수집 보강 필드 ──
            "overview_en": doc.overview_en,
            "overview_ja": doc.overview_ja,
            "facebook_id": doc.facebook_id,
            "instagram_id": doc.instagram_id,
            "twitter_id": doc.twitter_id,
            "wikidata_id": doc.wikidata_id,
            "tmdb_list_count": doc.tmdb_list_count,
            "images_logos": doc.images_logos,
            # ── 데이터 출처 ──
            "source": doc.source,
        },
    }


async def load_to_elasticsearch(
    documents: list[MovieDocument],
    chunk_size: int = 500,
) -> int:
    """
    MovieDocument 리스트를 Elasticsearch에 bulk 적재한다.

    §10-8: movies_bm25 인덱스에 Nori 분석기로 인덱싱.
    배치 적재 시 refresh_interval을 -1로 비활성화 후 완료 시 복원한다.

    Args:
        documents: 적재할 MovieDocument 리스트
        chunk_size: bulk API 배치 크기 (기본 500)

    Returns:
        int: 성공적으로 적재된 문서 수
    """
    client = await get_elasticsearch()

    # 대량 적재 중 세그먼트 병합(refresh)을 비활성화하여 I/O 부하를 줄인다.
    # 적재 완료 후 refresh_interval을 복원하고 강제 refresh를 수행한다.
    await client.indices.put_settings(
        index=ES_INDEX_NAME,
        body={"refresh_interval": "-1"},
    )

    # bulk 적재
    actions = [_movie_to_es_doc(doc) for doc in documents]

    success_count, errors = await async_bulk(
        client,
        actions,
        chunk_size=chunk_size,
        raise_on_error=False,
    )

    if errors:
        logger.warning("es_bulk_errors", error_count=len(errors))

    # refresh 복원 + 강제 refresh
    await client.indices.put_settings(
        index=ES_INDEX_NAME,
        body={"refresh_interval": "30s"},
    )
    await client.indices.refresh(index=ES_INDEX_NAME)

    # 적재 확인
    count_resp = await client.count(index=ES_INDEX_NAME)
    total = count_resp["count"]

    logger.info(
        "es_load_complete",
        loaded=success_count,
        total_in_index=total,
    )

    return success_count


# ============================================================
# Partial update 헬퍼 (Phase 2 enrichment 3DB 동기화용)
# ============================================================


def _serialize_enrichment_for_es(payload: dict) -> dict:
    """
    KOBIS/KMDb enrichment payload 를 ES 매핑에 맞게 직렬화한다.

    ES 는 list[dict] 를 직접 저장하지 못하므로 (nested/object 타입 지정 안 됨)
    KOBIS list[dict] 필드들을 BM25 토큰화 가능한 텍스트로 변환한다.
    기존 _movie_to_es_doc() 의 변환 로직과 동일 규칙.

    Args:
        payload: KOBIS/KMDb enricher 가 생성한 부분 업데이트 dict.
                 값은 str/int/float/list[str]/list[dict] 모두 가능.

    Returns:
        ES update body 의 `doc` 에 바로 들어갈 수 있는 dict.
    """
    out: dict = {}
    for key, value in payload.items():
        if value is None or value == "" or value == 0:
            continue

        # KOBIS list[dict] 필드 → 공백 구분 텍스트
        if key == "kobis_directors" and isinstance(value, list):
            out[key] = " ".join(
                d.get("peopleNm", "") for d in value if isinstance(d, dict)
            )
            continue
        if key == "kobis_actors" and isinstance(value, list):
            out[key] = " ".join(
                f"{a.get('peopleNm', '')}({a.get('cast', '')})"
                for a in value if isinstance(a, dict)
            )
            continue
        if key == "kobis_companies" and isinstance(value, list):
            out[key] = " ".join(
                f"{c.get('companyNm', '')}({c.get('companyPartNm', '')})"
                for c in value if isinstance(c, dict)
            )
            continue
        if key == "kobis_staffs" and isinstance(value, list):
            out[key] = " ".join(
                f"{s.get('peopleNm', '')}({s.get('staffRoleNm', '')})"
                for s in value if isinstance(s, dict)
            )
            continue
        if key == "production_companies" and isinstance(value, list):
            out[key] = " ".join(
                c.get("name", "") for c in value if isinstance(c, dict)
            )
            continue
        if key == "alternative_titles" and isinstance(value, list):
            out[key] = [
                t.get("title", "") for t in value
                if isinstance(t, dict) and t.get("title")
            ]
            continue
        if key == "cast_characters" and isinstance(value, list):
            out[key] = " ".join(
                f"{c.get('name', '')}({c.get('character', '')})"
                for c in value if isinstance(c, dict)
            )
            continue
        # cast / cast_original_names / keywords: list[str] → 공백 구분
        if key in ("cast", "cast_members", "cast_original_names", "keywords") and isinstance(value, list):
            out[key] = " ".join(str(x) for x in value if x)
            continue

        # 나머지 (기본 타입 + list[str] + list[numeric]) 그대로
        out[key] = value

    return out


async def update_movie_partial(movie_id: str, enrichment: dict) -> bool:
    """
    단일 영화의 특정 필드만 부분 업데이트 (ES Update API).

    KOBIS/KMDb enrichment 를 기존 ES 문서에 병합할 때 사용한다.
    문서가 없으면 404 → False 반환 (new movie 는 load_to_elasticsearch 경유).

    Args:
        movie_id: ES `_id` (보통 TMDB ID 문자열)
        enrichment: 업데이트할 필드 dict. KOBIS list[dict] 등은 자동 직렬화.

    Returns:
        성공 True, 문서 없음/실패 False
    """
    if not enrichment:
        return False

    client = await get_elasticsearch()
    es_doc = _serialize_enrichment_for_es(enrichment)
    if not es_doc:
        return False

    try:
        await client.update(
            index=ES_INDEX_NAME,
            id=movie_id,
            body={"doc": es_doc},
        )
        return True
    except Exception as e:
        msg = str(e)
        if "not_found" in msg or "404" in msg:
            logger.debug("es_update_not_found", movie_id=movie_id)
            return False
        logger.warning("es_update_failed", movie_id=movie_id, error=msg[:200])
        return False


async def update_movies_partial_bulk(updates: list[tuple[str, dict]]) -> int:
    """
    여러 영화의 부분 업데이트를 bulk API 로 일괄 처리.

    Args:
        updates: [(movie_id, enrichment_dict), ...]

    Returns:
        성공 건수
    """
    if not updates:
        return 0

    client = await get_elasticsearch()

    actions = []
    for movie_id, enrichment in updates:
        es_doc = _serialize_enrichment_for_es(enrichment)
        if not es_doc:
            continue
        actions.append({
            "_op_type": "update",
            "_index": ES_INDEX_NAME,
            "_id": movie_id,
            "doc": es_doc,
            "doc_as_upsert": False,  # 존재하지 않는 doc 은 무시
        })

    if not actions:
        return 0

    success, errors = await async_bulk(
        client,
        actions,
        chunk_size=500,
        raise_on_error=False,
        raise_on_exception=False,
    )

    if errors:
        # not_found 는 정상 (enrichment 대상이 아직 적재 안 됨)
        real_errors = [
            e for e in errors
            if "not_found" not in str(e) and "404" not in str(e)
        ]
        if real_errors:
            logger.warning("es_bulk_update_errors", count=len(real_errors))

    await client.indices.refresh(index=ES_INDEX_NAME)
    logger.info("es_partial_bulk_done", success=success, total=len(actions))
    return success
