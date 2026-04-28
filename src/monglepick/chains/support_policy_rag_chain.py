"""
고객센터 정책 RAG 검색 체인.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §6 (정책 RAG 스펙)

Qdrant `support_policy_v1` 컬렉션에서 정책 청크를 벡터 유사도 검색한다.
하이브리드 검색(RRF / ES / Neo4j) 없이 단순 vector top-K 만 수행한다.

사용처:
- `tools/support_tools/policy.py` 의 `lookup_policy` 핸들러가 호출
- Phase 1.8 에서 support_assistant v4 그래프의 narrator 노드가 결과를 인용

에러 정책:
- 모든 예외는 try/except 로 포착하여 빈 리스트 반환 (에러 전파 금지).
  호출 측 fallback 이 가능하도록 설계.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog

from monglepick.data_pipeline.embedder import embed_query_async
from monglepick.db.clients import SUPPORT_POLICY_COLLECTION, get_qdrant

logger = structlog.get_logger()

# ── 지원 policy_topic 값 목록 (문서화 목적) ──
# Qdrant payload 인덱스 `policy_topic` 에 저장된 값과 일치해야 한다.
# (scripts/index_support_policy.py 의 _infer_policy_topic 참조)
VALID_TOPICS: frozenset[str] = frozenset({
    "grade_benefit",   # 등급별 혜택 (6등급 팝콘 테마)
    "ai_quota",        # AI 쿼터 정책 (3-소스 순서, daily/monthly/purchased)
    "subscription",    # 구독 플랜 (4 플랜, 자동 갱신, 해지)
    "refund",          # 환불 정책 (결제 취소, 부분 환불)
    "reward",          # 리워드 적립 (출석, 활동, 이벤트)
    "payment",         # 결제 수단·정책 (포인트 1P=10원)
    "general",         # 일반 서비스 이용 안내
})


@dataclass
class PolicyChunk:
    """
    정책 RAG 검색 결과 단일 청크.

    Qdrant `support_policy_v1` 컬렉션의 payload 필드와 1:1 매핑된다.
    설계서: docs/고객센터_AI에이전트_v4_재설계.md §6.2 (메타데이터 스키마)
    """

    doc_id: str          # 문서 식별자 (파일 stem, 예: "리워드_결제_설계서")
    doc_path: str        # 원본 문서 경로 (감사 추적용)
    section: str         # 섹션 식별자 (예: "§4.5 AI 쿼터 정책")
    headings: list[str]  # 청크가 속한 헤딩 경로 (예: ["##등급 혜택", "###BRONZE"])
    policy_topic: str    # 정책 토픽 (grade_benefit / ai_quota / ... / general)
    text: str            # 청크 본문 텍스트
    score: float         # Qdrant 코사인 유사도 점수 (0.0 ~ 1.0)


async def search_policy(
    query: str,
    top_k: int = 5,
    topic_filter: str | None = None,
    request_id: str = "",
) -> list[PolicyChunk]:
    """
    정책 RAG 검색 — Qdrant `support_policy_v1` 컬렉션 vector top-K 검색.

    사용자 발화를 Upstage Solar embedding-query 모델로 임베딩한 뒤,
    코사인 유사도로 가장 관련 높은 정책 청크를 반환한다.

    topic_filter 가 주어지면 해당 policy_topic 에 속한 청크만 검색한다.
    예: topic_filter="grade_benefit" → 등급 혜택 관련 청크만 검색.

    에러 시 (임베딩 실패, Qdrant 연결 오류 등) 빈 리스트를 반환한다 (에러 전파 금지).
    호출 측(lookup_policy handler, narrator 노드)이 fallback 처리할 수 있도록 설계.

    Args:
        query: 검색 쿼리 (사용자 발화에서 핵심 키워드 추출 권장)
        top_k: 반환할 최대 청크 수 (기본 5)
        topic_filter: 정책 토픽 필터 — None 이면 전체 컬렉션 검색.
                      유효값: grade_benefit / ai_quota / subscription /
                              refund / reward / payment / general
        request_id: 요청 추적 ID (로그 상관관계 추적용)

    Returns:
        PolicyChunk 리스트 (유사도 내림차순). 에러 시 빈 리스트.
    """
    start_time = time.perf_counter()

    try:
        # ── 1. 쿼리 임베딩 ──
        # embed_query_async 는 asyncio.to_thread 래퍼 — event loop 블로킹 없음.
        # Upstage API 장애 대비 30초 타임아웃 적용 (hybrid_search.py 동일 패턴).
        try:
            query_vec = await asyncio.wait_for(
                embed_query_async(query),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.error(
                "policy_rag_embed_timeout",
                query_preview=query[:80],
                timeout_sec=30,
                request_id=request_id,
            )
            return []

        # numpy array → list[float] 변환 (qdrant-client API 요구사항)
        query_vector: list[float] = query_vec.tolist()

        # ── 2. topic_filter 가 있으면 Qdrant Filter 구성 ──
        # policy_topic 은 KEYWORD 인덱스 필드이므로 MatchValue 로 정확 매칭한다.
        query_filter = None
        if topic_filter is not None:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="policy_topic",
                        match=MatchValue(value=topic_filter),
                    )
                ]
            )

        # ── 3. Qdrant vector 검색 실행 ──
        client = await get_qdrant()
        response = await client.query_points(
            collection_name=SUPPORT_POLICY_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        # ── 4. 결과를 PolicyChunk 로 변환 ──
        chunks: list[PolicyChunk] = []
        for hit in response.points:
            payload = hit.payload or {}
            chunks.append(
                PolicyChunk(
                    doc_id=payload.get("doc_id", ""),
                    doc_path=payload.get("doc_path", ""),
                    section=payload.get("section", ""),
                    headings=payload.get("headings", []),
                    policy_topic=payload.get("policy_topic", "general"),
                    text=payload.get("text", ""),
                    score=float(hit.score),
                )
            )

        # ── 5. 구조화 로그 ──
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "policy_rag_search_done",
            query_preview=query[:80],
            top_k=top_k,
            topic=topic_filter,
            hit_count=len(chunks),
            latency_ms=round(elapsed_ms, 1),
            request_id=request_id,
            top_hits=[
                {
                    "doc_id": c.doc_id,
                    "section": c.section,
                    "score": round(c.score, 4),
                }
                for c in chunks[:3]
            ],
        )

        return chunks

    except Exception as exc:
        # 예상치 못한 에러 — 빈 리스트 반환, 에러 전파 금지
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "policy_rag_search_error",
            error=str(exc),
            error_type=type(exc).__name__,
            query_preview=query[:80],
            topic=topic_filter,
            latency_ms=round(elapsed_ms, 1),
            request_id=request_id,
        )
        return []
