"""
고객센터 FAQ ES 검색 모듈 (support_assistant v3.3).

### 역할
Elasticsearch `support_faq` 인덱스에서 사용자 발화와 가장 관련 있는 FAQ 후보를
Nori BM25 검색으로 빠르게 가져온다.

### 인덱스 스펙 (Backend 팀과 합의 — 절대 변경 금지)
- 인덱스명: `support_faq`
- Nori analyzer: `nori_analyzer`
- 주요 필드:
  - `faq_id`      (long)     — 문서 _id 와 동일
  - `category`    (keyword)  — GENERAL/ACCOUNT/CHAT/RECOMMENDATION/COMMUNITY/PAYMENT
  - `question`    (text, nori) — 부스트 ×3
  - `keywords`    (text, nori) — 동의어 태그 (쉼표 구분). 예: "환불,반환,취소". 부스트 ×2
  - `answer`      (text, nori) — 부스트 ×1
  - `is_published`(boolean)  — 검색 시 반드시 true 필터
  - `helpful_count`(integer)
  - `sort_order`  (integer)
  - `updated_at`  (date)

### 검색 전략
multi_match best_fields 로 question^3 + keywords^2 + answer 를 동시에 검색한다.
`is_published=true` 필터를 항상 적용해 미공개 FAQ 는 절대 노출되지 않는다.
`minimum_should_match="50%"` 로 짧은 쿼리에서 오매칭을 줄인다.

### 실패 처리
- ES 연결 실패 / 타임아웃: 빈 리스트 반환 + warn 로그
  → 상위 support_reply_chain 이 Backend HTTP fallback 경로로 전환한다.
- 인덱스 미존재: 같은 처리 (es.search 는 404 예외를 던짐)
"""

from __future__ import annotations

import time

import structlog
from pydantic import BaseModel

from monglepick.config import settings
from monglepick.db.clients import get_elasticsearch

logger = structlog.get_logger(__name__)

# ES 인덱스명 상수 — config 를 단일 진실 원본으로 유지하되 모듈 내 참조용 별칭도 선언
SUPPORT_FAQ_INDEX = settings.SUPPORT_FAQ_INDEX_NAME


# =============================================================================
# Pydantic DTO — ES 검색 결과 단건
# =============================================================================


class FaqCandidate(BaseModel):
    """
    ES 검색 결과 한 건을 담는 DTO.

    faq_id  : ES 문서 _id (= MySQL support_faq.faq_id)
    category: FAQ 카테고리 (GENERAL/ACCOUNT/CHAT/RECOMMENDATION/COMMUNITY/PAYMENT)
    question: 질문 원문
    answer  : 답변 원문 (답변 생성 단계에서 근거로 사용)
    keywords: 동의어 태그 쉼표 구분 문자열. 인덱스에 없으면 None.
    score   : BM25 점수. 임계값 분기(HIGH/MID) 판정에 사용.
    """

    faq_id: int
    category: str
    question: str
    answer: str
    keywords: str | None = None
    score: float


# =============================================================================
# 검색 함수
# =============================================================================


async def search_faq_candidates(
    user_message: str,
    top_k: int = 5,
) -> list[FaqCandidate]:
    """
    ES Nori BM25 으로 사용자 발화와 가장 관련 있는 FAQ 후보를 검색한다.

    쿼리 구조:
    - bool.must: multi_match (question^3, keywords^2, answer, type=best_fields)
    - bool.filter: is_published=true
    - minimum_should_match: "50%" (짧은 쿼리 오매칭 방지)

    Args:
        user_message: 사용자 발화 텍스트
        top_k:        반환할 최대 후보 수 (기본 5)

    Returns:
        FaqCandidate 리스트 (점수 내림차순). ES 실패 시 빈 리스트.
    """
    if not user_message or not user_message.strip():
        # 빈 발화 — ES 호출 불필요
        logger.debug("support_es_search_empty_query")
        return []

    started = time.perf_counter()

    # ES 비동기 클라이언트 가져오기 (싱글턴, lifespan 초기화)
    try:
        es = await get_elasticsearch()
    except Exception as exc:  # noqa: BLE001 — 에러 전파 금지
        logger.warning(
            "support_es_client_unavailable",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return []

    # 검색 쿼리 구성
    # question 필드에 가장 높은 부스트(^3)를 주어 정확한 질문 매칭을 우선한다.
    # keywords 필드(^2)는 동의어(환불/반환/취소 등)로 커버리지를 넓힌다.
    # answer 필드(^1)는 부스트 없이 풀텍스트 보조 검색.
    query_body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": user_message,
                            "fields": ["question^3", "keywords^2", "answer"],
                            "type": "best_fields",
                            "operator": "or",
                            # 쿼리 형태소의 50% 이상이 매칭돼야 결과에 포함됨.
                            # 1~2어절 짧은 쿼리에서 오매칭을 줄이는 효과.
                            "minimum_should_match": "50%",
                        }
                    }
                ],
                # is_published=true 인 FAQ 만 검색 (미공개 FAQ 절대 노출 불가)
                "filter": [{"term": {"is_published": True}}],
            }
        },
    }

    try:
        resp = await es.search(
            index=SUPPORT_FAQ_INDEX,
            body=query_body,
            # 타임아웃: config 에서 조정 가능 (기본 2.0s)
            request_timeout=settings.SUPPORT_ES_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 — 인덱스 없음/네트워크 오류 등 모두 처리
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning(
            "support_es_search_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            elapsed_ms=round(elapsed_ms, 1),
        )
        return []

    hits = resp.get("hits", {}).get("hits", [])
    elapsed_ms = (time.perf_counter() - started) * 1000

    # 결과 변환 — ES hit → FaqCandidate
    candidates: list[FaqCandidate] = []
    for hit in hits:
        src = hit.get("_source", {})
        score = float(hit.get("_score") or 0.0)
        try:
            candidates.append(
                FaqCandidate(
                    faq_id=int(src.get("faq_id") or hit.get("_id") or 0),
                    category=str(src.get("category") or "GENERAL"),
                    question=str(src.get("question") or ""),
                    answer=str(src.get("answer") or ""),
                    keywords=src.get("keywords") or None,
                    score=score,
                )
            )
        except Exception as parse_exc:  # noqa: BLE001 — 한 건 실패가 전체를 막지 않도록
            logger.debug(
                "support_es_hit_parse_skip",
                error=str(parse_exc),
                hit_id=hit.get("_id"),
            )

    top_score = candidates[0].score if candidates else 0.0

    # 임계값 밴드 결정 (로그 가독성용)
    if top_score >= settings.SUPPORT_ES_SCORE_HIGH:
        threshold_band = "HIGH"
    elif top_score >= settings.SUPPORT_ES_SCORE_MID:
        threshold_band = "MID"
    else:
        threshold_band = "LOW"

    logger.info(
        "support_es_search",
        query_preview=user_message[:80],
        hits=len(candidates),
        score_top=round(top_score, 2),
        threshold_band=threshold_band,
        elapsed_ms=round(elapsed_ms, 1),
    )

    return candidates
