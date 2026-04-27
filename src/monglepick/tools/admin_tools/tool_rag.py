"""
관리자 AI 에이전트 Tool RAG (Step 7a, 2026-04-27 착수).

설계 배경:
- v3 재설계로 ADMIN_TOOL_REGISTRY 가 76개로 증가 (Read 54 + Draft 10 + Navigate 12).
- 매 턴 76개를 모두 `bind_tools` 에 싣는 건 Solar-pro 프롬프트 토큰 + tool 선택 정확도 저하.
- 기존 `tool_filter.py` 는 카테고리·키워드 매칭(Qdrant 0 의존)으로 충분한 1차 후보 추출이 가능하나,
  의미 유사도 매칭(예: "환불" → `goto_order_refund`, "사용자 정지" → `goto_user_suspend`) 가 약함.
- 본 모듈은 76 tool 의 description + example_questions 를 Qdrant `admin_tool_registry`
  컬렉션에 임베딩하여 의미 기반 후보 추출을 추가한다.

본 모듈은 **`tool_filter.py` 와 공존**한다 (대체 아님):
- `tool_filter.shortlist_tools_by_category()` — 빠르고 의존성 0, intent_kind 정확도 높을 때
- `tool_rag.search_similar_tools()` — 의미 유사도, Qdrant 의존, intent_kind 모호할 때

`tool_selector` 그래프 노드는 환경변수 `ADMIN_TOOL_RAG_ENABLED` 로 둘 중 선택하거나
**하이브리드** (RAG top-K + filter 우선순위 머지) 로 결합 가능.

공개 API:
- `ADMIN_TOOL_COLLECTION` (str) — Qdrant 컬렉션명
- `ensure_admin_tool_collection()` — 컬렉션 미존재 시 생성
- `upsert_admin_tool_embeddings()` — 76 tool 임베딩 + upsert (1회 인덱싱 스크립트에서 호출)
- `search_similar_tools(query, ...)` — 의미 유사도 top-K tool 이름

Fallback:
- Qdrant 장애 → 빈 리스트 반환 (호출측이 `tool_filter` 로 폴백)
- Upstage 장애 → 빈 리스트
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import structlog
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from monglepick.config import settings
from monglepick.data_pipeline.embedder import embed_query_async, embed_texts
from monglepick.db.clients import get_qdrant
from monglepick.tools.admin_tools import ADMIN_TOOL_REGISTRY, ToolSpec

logger = structlog.get_logger(__name__)


# ============================================================
# 상수
# ============================================================

#: Qdrant 컬렉션명. 영화 컬렉션(`movies`)과 분리 — 동일 인스턴스 안에서 격리.
ADMIN_TOOL_COLLECTION: str = "admin_tool_registry"

#: 임베딩 텍스트 ID 네임스페이스 — 같은 tool_name 은 항상 같은 UUID 로 매핑.
#: 재인덱싱 시 중복 INSERT 가 아닌 UPSERT 가 되어 운영이 멱등적.
_TOOL_UUID_NAMESPACE: uuid.UUID = uuid.UUID("6f8a9c10-1234-5678-9abc-def012345678")


# ============================================================
# 분류 헬퍼 — tool_filter 와 동일 규칙
# ============================================================

def _classify_tool_kind(tool_name: str) -> str:
    """
    tool 이름만으로 kind 를 판정한다 (`tool_filter.classify_tool_kind` 와 동일 규칙).

    의존성 사이클 회피를 위해 `tool_filter` 를 import 하지 않고 자체 구현.
    우선순위: draft > navigate > stats > read(기본).

    Args:
        tool_name: 등록된 tool 의 name 필드 (snake_case 영문).

    Returns:
        "draft" | "navigate" | "stats" | "read"
    """
    if tool_name.endswith("_draft"):
        return "draft"
    if tool_name.startswith("goto_"):
        return "navigate"
    if tool_name.startswith("stats_") or tool_name.startswith("dashboard_"):
        return "stats"
    return "read"


def _tool_name_to_uuid(tool_name: str) -> str:
    """
    tool_name → deterministic UUID5 문자열.

    같은 입력은 항상 같은 출력 — 재인덱싱 시 같은 ID 로 UPSERT 되어 중복 방지.
    Qdrant point ID 로 사용된다.
    """
    return str(uuid.uuid5(_TOOL_UUID_NAMESPACE, tool_name))


def _build_embedding_text(spec: ToolSpec) -> str:
    """
    Tool 임베딩 텍스트 합성.

    Solar embedding-passage 모델은 한국어 + 영문을 모두 잘 처리한다.
    description(한글) + example_questions(한글) + name(영문) 조합으로 의미 + 컨벤션을 모두 흡수.

    형식 예시:
        ```
        notice_draft
        공지사항 등록 폼 사전 작성
        예: 새 공지 등록할게요 / 점검 공지 만들어줘 / 이벤트 공지 초안 만들어줘
        ```

    Args:
        spec: ToolSpec 인스턴스

    Returns:
        임베딩 입력 텍스트 (개행으로 구분된 단일 문자열)
    """
    parts = [spec.name, spec.description.strip()]
    if spec.example_questions:
        examples_joined = " / ".join(q.strip() for q in spec.example_questions if q.strip())
        if examples_joined:
            parts.append(f"예: {examples_joined}")
    return "\n".join(parts)


# ============================================================
# 컬렉션 보장
# ============================================================

async def ensure_admin_tool_collection() -> None:
    """
    Qdrant `admin_tool_registry` 컬렉션이 없으면 생성한다.

    설정:
    - 벡터 크기: `settings.EMBEDDING_DIMENSION` (4096, Upstage Solar)
    - 거리 메트릭: Cosine
    - HNSW: M=16, ef_construct=100 (영화 컬렉션과 동일)
    - on_disk=False — 76개 작은 컬렉션이라 메모리 보관이 빠름

    Payload 인덱스:
    - `name` (KEYWORD) — 정확 매칭 디버깅 용
    - `kind` (KEYWORD) — read/draft/navigate/stats 카테고리 필터
    - `tier` (INTEGER) — 0~4 위험도 필터
    - `required_roles` (KEYWORD, 배열) — admin_role 필터

    이 함수는 멱등적이다 — 이미 존재하는 컬렉션에는 영향 없음.
    """
    client = await get_qdrant()

    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]

    if ADMIN_TOOL_COLLECTION not in existing_names:
        await client.create_collection(
            collection_name=ADMIN_TOOL_COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
                on_disk=False,
            ),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
        logger.info("admin_tool_collection_created", name=ADMIN_TOOL_COLLECTION)

    # Payload 인덱스 (이미 있으면 idempotent)
    keyword_fields = ["name", "kind", "required_roles"]
    for field in keyword_fields:
        try:
            await client.create_payload_index(
                collection_name=ADMIN_TOOL_COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            # 이미 존재하면 Qdrant 가 에러를 던지지만 운영상 무해.
            logger.debug("admin_tool_payload_index_skip", field=field, reason=str(e))

    try:
        await client.create_payload_index(
            collection_name=ADMIN_TOOL_COLLECTION,
            field_name="tier",
            field_schema=PayloadSchemaType.INTEGER,
        )
    except Exception as e:
        logger.debug("admin_tool_payload_index_skip", field="tier", reason=str(e))

    logger.info(
        "admin_tool_collection_ready",
        collection=ADMIN_TOOL_COLLECTION,
        existed=ADMIN_TOOL_COLLECTION in existing_names,
    )


# ============================================================
# 인덱싱
# ============================================================

async def upsert_admin_tool_embeddings(
    tool_specs: list[ToolSpec] | None = None,
    *,
    batch_size: int = 50,
) -> int:
    """
    `tool_specs`(기본 ADMIN_TOOL_REGISTRY 전체) 의 description + example_questions 를
    임베딩하고 Qdrant 에 UPSERT 한다.

    `_tool_name_to_uuid` 가 deterministic 이므로 재실행 시 같은 ID 로 덮어쓰기 — 멱등적.
    Tool 추가/수정/제거 후 1회 실행하면 운영 인덱스가 정합 상태로 복귀한다.

    Args:
        tool_specs: 인덱싱할 tool 목록. None 이면 ADMIN_TOOL_REGISTRY 전체.
        batch_size: 임베딩 배치 크기 (기본 50, Upstage TPM 고려).

    Returns:
        upsert 한 point 개수 (= 입력 tool 수).

    Raises:
        Qdrant/Upstage 호출 실패는 그대로 전파 (1회 인덱싱 스크립트에서 수동 재시도).
    """
    specs = tool_specs if tool_specs is not None else list(ADMIN_TOOL_REGISTRY.values())
    if not specs:
        logger.warning("admin_tool_upsert_empty_registry")
        return 0

    # 1) 임베딩 텍스트 합성
    texts = [_build_embedding_text(s) for s in specs]

    # 2) 임베딩 (동기 함수를 to_thread 로 비동기화 — embed_texts 가 batch 처리 + 딜레이 포함)
    vectors = await asyncio.to_thread(embed_texts, texts, batch_size)
    if vectors.shape[0] != len(specs):
        raise RuntimeError(
            f"embedding count mismatch: expected {len(specs)} got {vectors.shape[0]}"
        )

    # 3) Qdrant point 구성
    points = [
        PointStruct(
            id=_tool_name_to_uuid(spec.name),
            vector=vec.tolist(),
            payload={
                "name": spec.name,
                "kind": _classify_tool_kind(spec.name),
                "tier": spec.tier,
                "required_roles": sorted(spec.required_roles),
                "description": spec.description,
            },
        )
        for spec, vec in zip(specs, vectors, strict=True)
    ]

    # 4) UPSERT
    client = await get_qdrant()
    await client.upsert(
        collection_name=ADMIN_TOOL_COLLECTION,
        points=points,
        wait=True,
    )

    logger.info(
        "admin_tool_upsert_done",
        collection=ADMIN_TOOL_COLLECTION,
        count=len(points),
    )
    return len(points)


# ============================================================
# 검색
# ============================================================

async def search_similar_tools(
    query: str,
    *,
    allowed_tool_names: set[str] | None = None,
    top_k: int = 20,
    score_threshold: float = 0.25,
) -> list[str]:
    """
    `query` 와 의미적으로 유사한 tool 이름을 top-K 로 반환한다.

    알고리즘:
    1. query 임베딩 (Upstage embedding-query)
    2. Qdrant `admin_tool_registry` 에서 cosine top-K 검색
    3. (allowed_tool_names 가 있으면) 교집합 — role 매트릭스 필터
    4. score_threshold 미만은 제외 (의미 유사도가 너무 낮으면 노이즈)

    Args:
        query: 자연어 질의 (현재 턴 관리자 발화).
        allowed_tool_names: role 매트릭스로 필터된 후보 집합. None 이면 필터 없음.
            **호출측은 항상 `list_tools_for_role(admin_role)` 결과를 넘기는 게 안전.**
        top_k: 후보 상한 (기본 20).
        score_threshold: cosine 유사도 하한 (기본 0.25, 2026-04-27 튜닝).
            짧은 발화("FAQ 추가"/"도움말 추가"/"이용권 등록") 가 0.29 근처에서 모이는 분포가
            관찰되어 0.30 → 0.25 로 완화. 0.20~0.25 구간은 노이즈 비율이 높아 제외.

    Returns:
        score 내림차순 tool 이름 리스트. 권한 없거나 결과가 비면 빈 리스트.

    실패 모드:
        - Qdrant/Upstage 장애 → 빈 리스트 반환 (호출측이 `tool_filter` 로 fallback).
        - 빈 query → 빈 리스트.
    """
    if not query or not query.strip():
        logger.info("admin_tool_search_empty_query")
        return []

    # 1) 쿼리 임베딩
    try:
        query_vec = await embed_query_async(query)
    except Exception as e:
        logger.warning("admin_tool_search_embedding_failed", error=str(e))
        return []

    # 2) Qdrant 검색 (qdrant-client v1.17+: search → query_points)
    try:
        client = await get_qdrant()
        # role 필터를 Qdrant 측에 위임할 수도 있으나, allowed_tool_names 가 작아서
        # 후처리(파이썬 set 교집합) 가 더 단순하고 빠름.
        response = await client.query_points(
            collection_name=ADMIN_TOOL_COLLECTION,
            query=query_vec.tolist(),
            limit=top_k,
            with_payload=True,
            score_threshold=score_threshold,
        )
    except Exception as e:
        logger.warning("admin_tool_search_qdrant_failed", error=str(e))
        return []

    hits = response.points

    # 3) 결과 정렬 + 필터
    ranked: list[tuple[str, float]] = []
    for hit in hits:
        payload: dict[str, Any] = hit.payload or {}
        name = payload.get("name")
        if not name:
            continue
        if allowed_tool_names is not None and name not in allowed_tool_names:
            continue
        ranked.append((name, hit.score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    names = [n for n, _ in ranked]

    logger.info(
        "admin_tool_search_done",
        query_preview=query[:60],
        hits=len(hits),
        returned=len(names),
        top_score=hits[0].score if hits else None,
    )
    return names


# ============================================================
# Step 7b — 하이브리드 후보 빌더 (filter + RAG 머지)
# ============================================================

def _is_rag_enabled() -> bool:
    """ADMIN_TOOL_RAG_ENABLED 환경변수 — true/1/yes 외에는 비활성."""
    return os.getenv("ADMIN_TOOL_RAG_ENABLED", "false").lower() in ("true", "1", "yes")


async def build_admin_tool_candidates_async(
    *,
    user_message: str,
    admin_role: str,
    intent_kind: str,
    max_tools: int = 30,
) -> list[str]:
    """
    `tool_selector` 노드용 하이브리드 후보 빌더 (Step 7b).

    1) 항상 `tool_filter.shortlist_tools_by_category()` 로 카테고리 + 키워드 후보 산출 (Qdrant 0 의존)
    2) `ADMIN_TOOL_RAG_ENABLED=true` 면 `search_similar_tools()` 추가 호출 (의미 유사도)
    3) filter 결과 우선 + RAG 결과 보강 — 중복 제거 + 상한 절단

    이렇게 하면:
    - **filter 단독**: 항상 빠르고 안정적, intent_kind 가 정확하면 충분.
    - **RAG 추가**: intent_kind 가 모호하거나 키워드 매칭이 안 될 때(예: \"환불 처리해줘\"
      → goto_order_refund) 의미 유사도로 보강.
    - **장애 시 자동 폴백**: RAG 가 빈 리스트 반환해도 filter 결과는 유지된다.

    `nodes.py:tool_selector` 가 이 함수를 호출하도록 교체. 직접 `shortlist_tools_by_category`
    를 호출하던 코드는 본 함수에 위임.

    Args:
        user_message: 현재 턴 관리자 발화 원문.
        admin_role: 정규화된 AdminRoleEnum 값.
        intent_kind: intent_classifier 가 내려준 kind 문자열 (query/action/stats/...).
        max_tools: bind_tools 에 실을 최대 tool 이름 수.

    Returns:
        tool 이름 list (중복 없음, filter 우선 정렬).
    """
    # 1) 카테고리 필터 — 의존성 0
    from monglepick.tools.admin_tools.tool_filter import shortlist_tools_by_category

    filter_names = shortlist_tools_by_category(
        user_message=user_message,
        admin_role=admin_role,
        intent_kind=intent_kind,
        max_tools=max_tools,
    )

    # 2) RAG 비활성이면 즉시 반환
    if not _is_rag_enabled():
        return filter_names

    # 3) RAG 활성 — role 매트릭스 후보를 allowed_tool_names 로 제한
    from monglepick.tools.admin_tools import list_tools_for_role

    role_allowed: set[str] = {s.name for s in list_tools_for_role(admin_role)}
    if not role_allowed:
        # admin_role 이 비었거나 매트릭스 결과 0 — RAG 도 무의미
        return filter_names

    rag_names = await search_similar_tools(
        user_message,
        allowed_tool_names=role_allowed,
        top_k=max_tools,
    )

    # 4) 머지 — filter 우선, RAG 로 보강
    seen: set[str] = set()
    merged: list[str] = []
    for name in filter_names + rag_names:
        if name not in seen:
            seen.add(name)
            merged.append(name)

    truncated = merged[:max_tools]
    logger.info(
        "admin_tool_candidates_built",
        intent_kind=intent_kind,
        role=admin_role,
        filter_total=len(filter_names),
        rag_total=len(rag_names),
        merged=len(truncated),
        rag_only_added=len([n for n in rag_names if n not in set(filter_names)]),
    )
    return truncated
