"""
Person Qdrant 적재기 (Phase ML §9.5 Phase 1 — C-3).

TMDB Person 수집 + LLM 보강이 끝난 dict 리스트를 받아서:
    1. embedding_text 구성 (이름/국적/직업/대표작/페르소나/전기)
    2. Solar embedding (4096d)
    3. Qdrant `persons` 컬렉션 신규 생성 + payload upsert

기존 `qdrant_loader.py` 는 movies 컬렉션 전용. Person 은 별도 컬렉션이라
이 모듈을 신규 작성하여 도메인 분리 + Task #5 영향 방지.

설계 진실 원본:
    docs/데이터_적재_프로세스_전체분석_및_개선계획.md §9.2 D1, §9.5 Phase 1 (C-3)

Qdrant `persons` 컬렉션 스키마:
    - 차원: 4096 (Solar embedding, movies 와 동일 공간)
    - 거리: Cosine
    - on_disk: True (572K+ 메모리 절감)
    - point_id: TMDB person_id (정수)
    - payload:
        tmdb_id, name, original_name, profile_path, popularity,
        known_for_department, place_of_birth, birthday, deathday,
        gender, imdb_id,
        style_tags, persona, top_movies, biography_ko,
        embedding_text, source ('tmdb_person')

사용처:
    `scripts/run_persons_qdrant_load.py` 에서 사용 (별도 작성 예정)
    또는 한 번에 처리하는 통합 스크립트
"""

from __future__ import annotations

import asyncio

import numpy as np
import structlog
from qdrant_client.models import Distance, HnswConfigDiff, PayloadSchemaType, PointStruct, VectorParams
from tenacity import retry, stop_after_attempt, wait_exponential

from monglepick.config import settings
from monglepick.data_pipeline.embedder import embed_texts
from monglepick.db.clients import get_qdrant

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 컬렉션 상수
# ══════════════════════════════════════════════════════════════

PERSONS_COLLECTION = "persons"
EMBED_BATCH_SIZE = 50  # Solar embedding 배치
QDRANT_UPSERT_BATCH_SIZE = 100
_upsert_semaphore = asyncio.Semaphore(4)


# ══════════════════════════════════════════════════════════════
# Qdrant `persons` 컬렉션 생성 / payload 인덱스
# ══════════════════════════════════════════════════════════════


async def ensure_persons_collection() -> None:
    """
    Qdrant `persons` 컬렉션이 없으면 생성한다 + payload 인덱스 설정.

    movies 컬렉션과 동일한 4096d Cosine + on_disk + HNSW 설정.
    """
    client = await get_qdrant()
    collections = await client.get_collections()
    existing = [c.name for c in collections.collections]

    if PERSONS_COLLECTION not in existing:
        await client.create_collection(
            collection_name=PERSONS_COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,  # 4096
                distance=Distance.COSINE,
                on_disk=True,  # 572K+ 메모리 절감
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )
        logger.info("qdrant_persons_collection_created", name=PERSONS_COLLECTION)
    else:
        logger.info("qdrant_persons_collection_exists", name=PERSONS_COLLECTION)

    # Payload 인덱스 (필터 가능 필드)
    keyword_fields = [
        "name",
        "original_name",
        "known_for_department",
        "imdb_id",
        "source",
    ]
    for field in keyword_fields:
        try:
            await client.create_payload_index(
                collection_name=PERSONS_COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # 이미 존재

    integer_fields = ["tmdb_id", "gender"]
    for field in integer_fields:
        try:
            await client.create_payload_index(
                collection_name=PERSONS_COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.INTEGER,
            )
        except Exception:
            pass

    float_fields = ["popularity"]
    for field in float_fields:
        try:
            await client.create_payload_index(
                collection_name=PERSONS_COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.FLOAT,
            )
        except Exception:
            pass

    logger.info("qdrant_persons_payload_indexes_configured")


# ══════════════════════════════════════════════════════════════
# Person → embedding_text + payload
# ══════════════════════════════════════════════════════════════


def build_person_embedding_text(person: dict) -> str:
    """
    LLM 보강된 Person dict 로부터 임베딩 입력 텍스트를 생성한다.

    형식:
        [이름] {name} ({original_name})
        [국적] {place_of_birth}
        [직업] {known_for_department}
        [활동시기] {birthday[:4]} ~ {deathday[:4] or '현재'}
        [페르소나] {llm_persona}
        [스타일] {llm_style_tags}
        [대표작] {llm_top_movies}
        [전기] {llm_biography_ko[:500]}
    """
    parts = []
    name = person.get("name", "")
    parts.append(f"[이름] {name}")

    # 원어 이름 (also_known_as 첫 번째)
    aka = person.get("also_known_as") or []
    if aka and aka[0] and aka[0] != name:
        parts.append(f"[원어이름] {aka[0]}")

    if person.get("place_of_birth"):
        parts.append(f"[국적/출생지] {person['place_of_birth']}")

    if person.get("known_for_department"):
        parts.append(f"[직업] {person['known_for_department']}")

    # 활동 시기
    birthday = (person.get("birthday") or "")[:4]
    deathday = (person.get("deathday") or "")[:4]
    if birthday:
        period = f"{birthday} ~ {deathday}" if deathday else f"{birthday} ~ 현재"
        parts.append(f"[활동시기] {period}")

    # LLM 보강 필드
    persona = person.get("llm_persona") or ""
    if persona:
        parts.append(f"[페르소나] {persona}")

    style_tags = person.get("llm_style_tags") or []
    if style_tags:
        parts.append(f"[스타일] {', '.join(style_tags[:10])}")

    top_movies = person.get("llm_top_movies") or []
    if top_movies:
        parts.append(f"[대표작] {', '.join(top_movies[:5])}")

    bio = person.get("llm_biography_ko") or ""
    if bio:
        parts.append(f"[전기] {bio[:500]}")

    return " ".join(parts)


def _person_to_payload(person: dict, embedding_text: str) -> dict:
    """
    TMDB Person 응답 + LLM 보강을 Qdrant payload 로 정리한다.

    embedding_text 도 payload 에 포함하여 디버깅/감사 추적 가능.
    """
    return {
        "tmdb_id": int(person.get("id") or 0),
        "name": person.get("name", "") or "",
        "original_name": (person.get("also_known_as") or [""])[0] if person.get("also_known_as") else "",
        "profile_path": person.get("profile_path") or "",
        "popularity": float(person.get("popularity") or 0.0),
        "known_for_department": person.get("known_for_department", "") or "",
        "place_of_birth": person.get("place_of_birth", "") or "",
        "birthday": person.get("birthday", "") or "",
        "deathday": person.get("deathday", "") or "",
        "gender": int(person.get("gender") or 0),
        "imdb_id": person.get("imdb_id", "") or "",
        # 외부 ID
        "external_ids": person.get("external_ids", {}) or {},
        # LLM 보강
        "style_tags": person.get("llm_style_tags") or [],
        "persona": person.get("llm_persona") or "",
        "top_movies": person.get("llm_top_movies") or [],
        "biography_ko": person.get("llm_biography_ko") or "",
        # 임베딩 텍스트 (감사 추적용)
        "embedding_text": embedding_text,
        "source": "tmdb_person",
    }


# ══════════════════════════════════════════════════════════════
# Qdrant 배치 upsert
# ══════════════════════════════════════════════════════════════


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=4))
async def _upsert_batch(points: list[PointStruct]) -> None:
    """단일 배치 upsert (최대 100건). Semaphore + 지수 백오프."""
    async with _upsert_semaphore:
        client = await get_qdrant()
        await client.upsert(
            collection_name=PERSONS_COLLECTION,
            points=points,
            wait=True,
        )


# ══════════════════════════════════════════════════════════════
# 메인 진입점
# ══════════════════════════════════════════════════════════════


async def load_persons_to_qdrant(
    persons: list[dict],
    embed_batch_size: int = EMBED_BATCH_SIZE,
    upsert_batch_size: int = QDRANT_UPSERT_BATCH_SIZE,
) -> int:
    """
    LLM 보강된 Person dict 리스트를 Qdrant `persons` 컬렉션에 적재한다.

    파이프라인:
        1. ensure_persons_collection (없으면 생성)
        2. 각 person → embedding_text 생성
        3. Solar embedding 배치 호출 (4096d)
        4. PointStruct 변환 + 100건씩 upsert

    Args:
        persons: TMDB Person + LLM 보강 dict 리스트
            (각 person 에 llm_biography_ko / llm_style_tags / llm_persona /
             llm_top_movies 키가 있어야 임베딩 품질이 좋음)
        embed_batch_size: Solar embedding 배치 (기본 50)
        upsert_batch_size: Qdrant upsert 배치 (기본 100)

    Returns:
        적재 완료된 person 수
    """
    if not persons:
        logger.info("persons_load_empty")
        return 0

    # 1) 컬렉션 보장
    await ensure_persons_collection()

    # 2) embedding_text 생성
    valid_persons = []
    embedding_texts = []
    for p in persons:
        if not p.get("id") or not p.get("name"):
            continue
        text = build_person_embedding_text(p)
        if not text or len(text) < 10:
            # embedding_text 가 너무 짧으면 skip
            continue
        valid_persons.append(p)
        embedding_texts.append(text)

    if not valid_persons:
        logger.warning("persons_all_invalid_skipped", input=len(persons))
        return 0

    logger.info(
        "persons_embedding_start",
        count=len(valid_persons),
        embed_batch_size=embed_batch_size,
    )

    # 3) Solar embedding (executor 로 이벤트 루프 블로킹 방지)
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None, embed_texts, embedding_texts, embed_batch_size
    )

    # 4) PointStruct 변환
    points: list[PointStruct] = []
    for person, text, vector in zip(valid_persons, embedding_texts, embeddings):
        try:
            payload = _person_to_payload(person, text)
            points.append(
                PointStruct(
                    id=int(person["id"]),  # TMDB person_id 정수
                    vector=vector.tolist() if isinstance(vector, np.ndarray) else list(vector),
                    payload=payload,
                )
            )
        except Exception as e:
            logger.warning(
                "person_pointstruct_failed",
                person_id=person.get("id"),
                error=str(e)[:200],
            )

    if not points:
        logger.warning("persons_no_points_to_upsert")
        return 0

    # 5) 배치 upsert
    logger.info("persons_qdrant_upsert_start", count=len(points), batch_size=upsert_batch_size)
    tasks = []
    for i in range(0, len(points), upsert_batch_size):
        batch = points[i:i + upsert_batch_size]
        tasks.append(_upsert_batch(batch))

    await asyncio.gather(*tasks)

    # 적재 검증
    client = await get_qdrant()
    info = await client.get_collection(PERSONS_COLLECTION)
    logger.info(
        "persons_load_complete",
        loaded=len(points),
        total_in_collection=info.points_count,
    )

    return len(points)
