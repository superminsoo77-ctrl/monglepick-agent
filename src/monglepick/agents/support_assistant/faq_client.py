"""
고객센터 AI 챗봇 — Backend FAQ 조회 클라이언트 (v3).

v2 의 `faq_cache.py` (Solar 임베딩 + 키워드 스코어링) 를 폐기하고
**매 요청마다 Backend GET /api/v1/support/faq 를 직접 호출**하는 단순 fetcher
로 대체한다.

### 왜 캐시 없이 매번 조회하는가
- FAQ 는 RDB 한 곳에 수십 건 규모 — 조회 비용이 미미 (< 10ms)
- Agent ↔ Backend 는 같은 VM 내부 네트워크 호출 (< 50ms)
- 관리자가 FAQ 를 추가/수정/삭제하면 **다음 챗봇 요청부터 즉시** 반영됨 (동기화 지연 0)
- 캐시 무효화 로직, TTL 튜닝, stale 서빙 등 운영 복잡도가 사라짐
- 챗봇 트래픽이 높아져 RDB 부담이 문제가 된다면 그때 가서 짧은 TTL 캐시를 다시 도입

### 실패 대응
- Backend 장애/타임아웃 시 빈 리스트 반환 → support_agent 가 FAQ 없이
  `kind="complaint"` fallback 을 내 1:1 문의 유도.
"""

from __future__ import annotations

import httpx
import structlog

from monglepick.agents.support_assistant.models import FaqDoc
from monglepick.config import settings

logger = structlog.get_logger(__name__)


# Backend 호출 타임아웃 — 챗봇 응답이 통째로 지연되지 않도록 짧게.
_BACKEND_TIMEOUT_SECONDS = 3.0

# 호출 싱글턴 httpx 클라이언트 (FastAPI worker 수명 동안 연결 풀 재사용).
_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    """FAQ 조회 전용 httpx.AsyncClient 싱글턴."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.BACKEND_BASE_URL,
            timeout=_BACKEND_TIMEOUT_SECONDS,
        )
    return _client


async def fetch_faqs() -> list[FaqDoc]:
    """
    Backend 에서 공개 FAQ 전체를 가져와 FaqDoc 리스트로 변환해 반환한다.

    실패 시 빈 리스트 반환 (에러 전파 금지). support_agent 는 빈 목록이어도
    `kind="complaint"` 폴백으로 1:1 문의 안내를 할 수 있다.
    """
    client = await _get_client()
    try:
        response = await client.get("/api/v1/support/faq")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning(
            "support_faq_fetch_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return []

    raw = response.json()
    if not isinstance(raw, list):
        logger.warning("support_faq_unexpected_shape", type=type(raw).__name__)
        return []

    faqs: list[FaqDoc] = []
    for row in raw:
        try:
            faqs.append(
                FaqDoc(
                    faq_id=int(row.get("faqId", 0)),
                    category=str(row.get("category", "GENERAL")),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    sort_order=row.get("sortOrder"),
                )
            )
        except Exception as exc:  # noqa: BLE001 — 한 건 실패가 전체를 막지 않도록
            logger.debug("support_faq_row_skip", error=str(exc), row=row)
    return faqs
