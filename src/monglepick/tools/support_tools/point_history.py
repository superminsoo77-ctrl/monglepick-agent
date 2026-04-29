"""
고객센터 AI 에이전트 v4 — `lookup_my_point_history` tool.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.1

Backend EP: GET /api/v1/point/history?page=0&size=N
응답 (Spring Page<HistoryResponse>):
  {
    "content": [
      {
        "id": <Long>,
        "pointChange": <int>,    # 양수=적립 / 음수=차감
        "pointAfter":  <int>,    # 변동 후 잔액
        "pointType":   <str>,    # earn / spend / expire / bonus 등
        "description": <str>,    # 변동 사유 (예: "리뷰 작성 보상")
        "createdAt":   "<ISO-8601>"
      }, ...
    ],
    "pageable": {...}, "totalElements": ..., "totalPages": ..., ...
  }

Backend 는 `?days=N` 쿼리를 지원하지 않으므로 본 tool 이 클라이언트 측에서 최근 N일을
필터링해 narrator 친화 평면 배열로 정규화한다 (`{amount, type, description, createdAt}`).

용도: "포인트 안 들어왔어요", "포인트 언제 빠져나갔어요" 같은 포인트 이력 문의.
requires_login=True — 로그인한 사용자의 본인 데이터 조회이므로 게스트 차단.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from pydantic import BaseModel, Field

from . import ToolContext, SupportToolSpec, register_support_tool
from . import _base


logger = structlog.get_logger(__name__)


# ============================================================
# 입력 스키마
# ============================================================

class LookupPointHistoryArgs(BaseModel):
    """
    `lookup_my_point_history` tool 입력 스키마.

    LLM 은 사용자 발화에서 기간(일수)을 추출해 days 에 담는다.
    "최근 일주일" → 7, "이번 달" → 30, 기간 언급 없으면 기본 7일.
    """

    days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="조회 기간 (일). 1~365 사이. 기본 7일.",
    )


# ============================================================
# Backend 응답 정규화 헬퍼
# ============================================================

# Backend 한 번의 페이지 호출로 가져올 최대 건수.
# 사용자가 7~30일 이력을 묻는 일반 케이스에서 한 번으로 충분하도록 100 으로 설정.
# (Backend MAX_PAGE_SIZE 와 동일 — 더 큰 값을 보내도 limitPageSize 가 100 으로 깎음)
_PAGE_SIZE = 100


def _parse_iso8601(value: Any) -> datetime | None:
    """
    ISO-8601 문자열을 timezone-aware datetime 으로 파싱한다.

    Backend Jackson 직렬화는 LocalDateTime 을 "2026-04-29T10:00:00" 형태로 출력하므로
    timezone 정보가 없을 수 있다. 그 경우 운영 정책상 KST(UTC+9) 로 가정한다.

    파싱 실패 시 None 반환 — 호출부에서 필터에 포함시키지 않고 무시한다.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        # "Z" 접미사를 fromisoformat 가 처리하도록 보정.
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        # naive → KST(UTC+9) 로 가정
        dt = dt.replace(tzinfo=timezone(timedelta(hours=9)))
    return dt


def _normalize_history_items(
    raw_content: list[dict[str, Any]] | None,
    days: int,
) -> list[dict[str, Any]]:
    """
    Backend Page.content 를 narrator 가 해석하기 쉬운 평면 배열로 변환한다.

    - `pointChange` → `amount` (alias 유지: 기존 narrator 가이드 호환)
    - `pointType`   → `type`
    - `description` 그대로
    - `createdAt`   그대로 (ISO-8601 문자열)
    - `id` 는 LLM 컨텍스트에 불필요하므로 제거.

    days 필터: createdAt 이 (now - days) 이전이면 제외.
    파싱 실패 항목은 보수적으로 포함 (Backend 가 정렬해 상위 페이지에 최신 순으로 줬으므로).
    """
    if not isinstance(raw_content, list):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    normalized: list[dict[str, Any]] = []
    for item in raw_content:
        if not isinstance(item, dict):
            continue
        created_at_raw = item.get("createdAt")
        created_at_dt = _parse_iso8601(created_at_raw)
        # 파싱 성공한 경우에만 cutoff 비교. 파싱 실패는 일단 살린다.
        if created_at_dt is not None and created_at_dt < cutoff:
            continue
        normalized.append(
            {
                "amount": item.get("pointChange"),
                "type": item.get("pointType"),
                "description": item.get("description"),
                "createdAt": created_at_raw,
            }
        )
    return normalized


# ============================================================
# Handler
# ============================================================

async def _handle(ctx: ToolContext, days: int = 7) -> dict:
    """
    lookup_my_point_history tool 실행 핸들러.

    게스트 차단 후 Backend GET /api/v1/point/history 를 호출한다.
    ctx.user_id 는 _base.call_backend_get 내부에서 X-User-Id 헤더로 강제 주입되므로
    params 에 user_id 를 별도로 담지 않는다.

    Backend 는 days 파라미터를 지원하지 않으므로 page=0 / size=100 으로 가져와
    클라이언트 측에서 최근 N일 필터링 + 필드 평면화를 수행한다.

    반환 스키마:
        ok=True  → {"ok": True, "data": [{amount, type, description, createdAt}, ...]}
        ok=False → {"ok": False, "error": "<사유>"}
    """
    # 게스트 차단 — requires_login=True 이지만 handler 에서도 이중 방어
    if ctx.is_guest:
        return {
            "ok": False,
            "error": "login_required",
            "reason": "포인트 이력은 로그인 후 확인할 수 있어요.",
        }

    # Backend Pageable 시그니처에 맞춰 page / size 전달.
    raw = await _base.call_backend_get(
        ctx,
        "/api/v1/point/history",
        params={"page": 0, "size": _PAGE_SIZE},
    )
    if not raw.get("ok"):
        return raw

    backend_payload = raw.get("data") or {}
    # 일반 Page 응답은 dict + "content" 키. 안전하게 fallback 처리.
    if isinstance(backend_payload, dict):
        content = backend_payload.get("content")
    elif isinstance(backend_payload, list):
        # 향후 Backend 가 평면 배열로 바뀔 가능성 대비 호환 유지.
        content = backend_payload
    else:
        content = None

    items = _normalize_history_items(content, days)
    logger.info(
        "lookup_my_point_history_normalized",
        days=days,
        raw_count=len(content) if isinstance(content, list) else 0,
        filtered_count=len(items),
        request_id=ctx.request_id,
    )
    return {"ok": True, "data": items}


# ============================================================
# 레지스트리 등록 (import side-effect)
# ============================================================

register_support_tool(
    SupportToolSpec(
        name="lookup_my_point_history",
        description=(
            "본인의 포인트 적립/차감 이력을 조회합니다. "
            "'포인트 안 들어왔어요', '포인트 왜 빠졌어요', '최근 포인트 내역 보여줘' "
            "같은 포인트 이력 문의에 사용하세요. 로그인 필수."
        ),
        args_schema=LookupPointHistoryArgs,
        handler=_handle,
        requires_login=True,
    )
)
