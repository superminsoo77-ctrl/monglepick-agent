"""
관리자 AI 에이전트 — Navigate Tool 12개.

설계서: docs/관리자_AI에이전트_v3_재설계.md §1.3 Navigate Tool, §4.3, §5 Role 매트릭스

핵심 원칙:
- Backend **쓰기(POST/PUT/DELETE) 절대 호출 금지**. 대상 식별을 위한 GET 읽기만 허용.
- 환불·계정 제재·포인트 조정 등 위험한 작업은 "찾아주고 링크 거는" 역할까지.
- 실제 처리는 관리자가 해당 관리 화면에서 직접 수행한다.

handler 결과 3가지 케이스:
1. 후보 1건  → {target_path, label, context_summary, tool_name}
2. 후보 여러 건 → {target_path: None, label: "~를 선택하세요", context_summary,
                   candidates: [{target_path, label}, ...], tool_name}
3. 후보 0건  → AdminApiResult(ok=False, error="해당 조건의 대상이 없어요...")

등록 tool 목록 (총 12개):
- goto_user_detail       : user 검색 후 상세 화면
- goto_user_suspend      : user 검색 후 정지 폼
- goto_user_activate     : user 검색 후 복구 폼
- goto_user_role_change  : user 검색 후 역할 변경 폼
- goto_points_adjust     : user 검색 후 포인트 조정 폼
- goto_token_grant       : user 검색 후 이용권 발급 폼
- goto_order_detail      : order 검색 후 상세 화면
- goto_order_refund      : order 검색 후 환불 폼
- goto_subscription_manage : subscription 검색 후 관리 화면
- goto_report_detail     : report 검색 후 처리 화면
- goto_ticket_detail     : ticket 검색 후 상세 화면
- goto_audit_log         : 조건부 감사 로그 검색 화면 (검색 없이 링크만 생성)

Role 매트릭스 (설계서 §5):
- user/* navigate     → SUPER_ADMIN, ADMIN, MODERATOR, SUPPORT_ADMIN
- points/token nav    → SUPER_ADMIN, ADMIN, FINANCE_ADMIN
- order/subscription  → SUPER_ADMIN, ADMIN, FINANCE_ADMIN
- report/ticket       → SUPER_ADMIN, ADMIN, MODERATOR, SUPPORT_ADMIN
- audit               → SUPER_ADMIN, ADMIN
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlencode

import structlog
from pydantic import BaseModel, Field

from monglepick.api.admin_backend_client import (
    AdminApiResult,
    get_admin_json,
    unwrap_api_response,
)
from monglepick.tools.admin_tools import (
    ToolContext,
    ToolSpec,
    register_tool,
)

logger = structlog.get_logger(__name__)


# ============================================================
# Role 매트릭스 — §5
# ============================================================

# user 검색 계열 (상세/정지/복구/역할변경) — MODERATOR, SUPPORT_ADMIN 포함
_USER_NAV_ROLES: set[str] = {
    "SUPER_ADMIN", "ADMIN", "MODERATOR", "SUPPORT_ADMIN",
}

# 포인트/이용권 조정 — 재무 영역
_FINANCE_NAV_ROLES: set[str] = {
    "SUPER_ADMIN", "ADMIN", "FINANCE_ADMIN",
}

# 결제/구독 — 재무 영역
_PAYMENT_NAV_ROLES: set[str] = {
    "SUPER_ADMIN", "ADMIN", "FINANCE_ADMIN",
}

# 신고/티켓 — MODERATOR, SUPPORT_ADMIN 포함
_CONTENT_SUPPORT_NAV_ROLES: set[str] = {
    "SUPER_ADMIN", "ADMIN", "MODERATOR", "SUPPORT_ADMIN",
}

# 감사 로그 — SUPER_ADMIN, ADMIN 만
_AUDIT_NAV_ROLES: set[str] = {
    "SUPER_ADMIN", "ADMIN",
}


# ============================================================
# Args Schemas
# ============================================================

class _UserKeywordArgs(BaseModel):
    """
    user 검색 계열 6개 공통 — 이메일/닉네임/user_id 키워드로 검색.

    2026-04-28 보강 (길 A v3): suspend / role_change / points_adjust / token_grant 같이
    부가 정보(사유·기간·금액·역할 등)가 있는 발화를 위해 prefill 필드를 옵션으로 허용한다.
    handler 가 query string 으로 실어 보내면 Client 가 폼에 prefill 한다.
    """

    keyword: str = Field(
        ...,
        min_length=1,
        description=(
            "이동할 사용자를 찾기 위한 검색어. 이메일·닉네임·user_id 중 하나 또는 일부."
            " 예: 'chulsoo', 'chulsoo@test.com', 'usr_abc123'."
        ),
    )

    # 2026-04-28 prefill 필드 (모든 user_navigate handler 가 공통으로 받지만,
    # action 의미가 맞는 handler 만 query string 에 실어 보낸다)
    reason: Optional[str] = Field(
        default=None,
        description=(
            "정지/역할 변경/포인트 조정 등의 사유. 사용자 발화에 '~사유는 X', '~때문에' "
            "같은 어휘가 있으면 그 부분을 그대로 채운다. handler 가 폼 prefill 로 전달."
        ),
    )
    suspendUntil: Optional[str] = Field(
        default=None,
        description=(
            "정지 해제 일시 (ISO 8601). '7일 정지', '한 달 정지' 같이 기간이 있으면 LLM 이 "
            "오늘 기준으로 미래 시각으로 환산해 채운다. goto_user_suspend 만 사용."
        ),
    )
    targetRole: Optional[str] = Field(
        default=None,
        description=(
            "변경할 역할 (예: ADMIN, MODERATOR, USER). goto_user_role_change 만 사용."
        ),
    )
    pointAmount: Optional[int] = Field(
        default=None,
        description=(
            "조정할 포인트 양 (양수=지급, 음수=차감). goto_points_adjust 만 사용."
        ),
    )
    tokenAmount: Optional[int] = Field(
        default=None,
        description=(
            "발급할 AI 이용권 수량. goto_token_grant 만 사용."
        ),
    )


class _OrderNavArgs(BaseModel):
    """order 검색용 — orderId 우선, 없으면 userId 로 fallback."""

    orderId: Optional[str] = Field(
        default=None,
        description=(
            "조회할 주문 ID (UUID 형태). 알고 있으면 이 값을 우선 사용한다."
        ),
    )
    userId: Optional[str] = Field(
        default=None,
        description=(
            "특정 사용자의 주문을 검색할 때 사용하는 사용자 ID. "
            "orderId 가 없는 경우에만 사용된다."
        ),
    )


class _SubscriptionNavArgs(BaseModel):
    """subscription 검색용 — userId / subscriptionId / status 조합."""

    userId: Optional[str] = Field(
        default=None,
        description="특정 사용자의 구독을 검색할 때 사용하는 사용자 ID.",
    )
    subscriptionId: Optional[str] = Field(
        default=None,
        description="구독 ID. 알고 있으면 이 값을 우선 사용한다.",
    )
    status: Optional[str] = Field(
        default=None,
        description="구독 상태 필터 (ACTIVE / CANCELLED / EXPIRED). 선택 사항.",
    )


class _ReportNavArgs(BaseModel):
    """report 화면 이동 — reportId 단건 또는 최신 목록 선택."""

    reportId: Optional[int] = Field(
        default=None,
        description="이동할 신고 ID. 알고 있으면 즉시 단건 링크를 생성한다.",
    )
    page: Optional[int] = Field(
        default=None,
        ge=0,
        description="reportId 가 없을 때 가져올 최신 신고 목록의 페이지 번호 (0부터).",
    )


class _TicketNavArgs(BaseModel):
    """ticket 화면 이동 — ticketId 우선, 없으면 userId 로 검색."""

    ticketId: Optional[int] = Field(
        default=None,
        description="이동할 고객 문의 티켓 ID. 알고 있으면 즉시 링크를 생성한다.",
    )
    userId: Optional[str] = Field(
        default=None,
        description="특정 사용자의 티켓을 찾을 때 사용하는 사용자 ID.",
    )


class _AuditLogArgs(BaseModel):
    """감사 로그 화면 — 검색 생략, 조건만 URL 에 실어 링크 생성."""

    q: str = Field(
        ...,
        min_length=1,
        description="감사 로그에서 조회할 검색어 (관리자 닉네임, 작업 유형, 대상 리소스 등).",
    )
    actor: Optional[str] = Field(
        default=None,
        description="감사 로그 작성자(관리자) ID 또는 닉네임 필터. 선택 사항.",
    )
    action: Optional[str] = Field(
        default=None,
        description="감사 로그 액션 유형 필터 (예: USER_SUSPEND, POINTS_ADJUST). 선택 사항.",
    )


# ============================================================
# 공통 내부 헬퍼
# ============================================================

def _extract_items(data: object) -> list[dict]:
    """
    Backend 응답 data 에서 리스트를 추출한다.

    Spring Data Page 래퍼 ({content: [...], ...}) 와 plain list 양쪽을 처리한다.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Spring Page 응답
        for key in ("content", "items", "data", "orders", "subscriptions", "tickets"):
            val = data.get(key)
            if isinstance(val, list):
                return val
    return []


async def _search_users(
    ctx: ToolContext,
    keyword: str,
) -> AdminApiResult:
    """
    GET /api/v1/admin/users?keyword={keyword}&size=10 — user 검색 공통 헬퍼.

    6개 user 검색 navigate tool 이 동일한 검색 로직을 재사용한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/users",
        admin_jwt=ctx.admin_jwt,
        params={"keyword": keyword, "size": 10},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


def _build_user_navigation_result(
    items: list[dict],
    keyword: str,
    action_suffix: str,
    tool_name: str,
    action_label_template: str,
    prefill_qs: str = "",
) -> dict:
    """
    후보 user 목록을 받아 navigation payload dict 를 구성한다.

    Args:
        items:                검색 결과 user 리스트
        keyword:              사용자가 입력한 검색어
        action_suffix:        target_path 에 붙을 action 쿼리 (예: "&action=suspend")
        tool_name:            이 tool 의 이름 (SSE navigation 이벤트 페이로드)
        action_label_template: label 템플릿. "{display}" 를 실제 이름으로 치환한다.
        prefill_qs:           target_path 에 추가로 붙을 prefill 쿼리 (예: "&reason=...&suspendUntil=...").
                              빈 문자열이면 미적용. 2026-04-28 길 A 보강.

    Returns:
        navigation payload dict (AdminApiResult.data 에 담겨 반환됨)
    """
    def _uid(u: dict) -> str:
        return str(u.get("userId") or u.get("user_id") or u.get("id") or "")

    def _display(u: dict) -> str:
        email = u.get("email") or ""
        nick = u.get("nickname") or ""
        if email and nick:
            return f"{nick} ({email})"
        return email or nick or _uid(u) or "unknown"

    def _joined(u: dict) -> str:
        created = u.get("createdAt") or u.get("created_at") or ""
        return str(created)[:10]  # YYYY-MM-DD 앞 10자

    if len(items) == 1:
        u = items[0]
        uid = _uid(u)
        display = _display(u)
        return {
            "target_path": f"/admin/users?userId={uid}{action_suffix}{prefill_qs}",
            "label": action_label_template.replace("{display}", display),
            "context_summary": f"'{keyword}' 로 1명을 찾았어요. 해당 화면으로 이동하실 수 있어요.",
            "tool_name": tool_name,
        }

    # 여러 명 — candidates 목록 제공
    candidates = []
    for u in items[:10]:
        uid = _uid(u)
        display = _display(u)
        joined = _joined(u)
        label = display + (f" ({joined} 가입)" if joined else "")
        candidates.append({
            "target_path": f"/admin/users?userId={uid}{action_suffix}{prefill_qs}",
            "label": label,
        })

    return {
        "target_path": None,
        "label": "사용자를 선택하세요",
        "context_summary": (
            f"'{keyword}' 로 {len(items)}명이 검색됐어요. "
            "이동할 계정을 골라주세요."
        ),
        "candidates": candidates,
        "tool_name": tool_name,
    }


def _build_prefill_qs(prefill: dict[str, Any]) -> str:
    """
    suspend/role_change/points_adjust/token_grant 등에서 받은 prefill dict 를
    target_path 끝에 붙일 query string 으로 변환한다.

    None / 빈 문자열은 제외. urlencode 로 한국어·특수문자 안전 처리.
    반환 형식: "&key1=val1&key2=val2" (앞에 & 가 붙어 기존 path 끝에 그대로 concat).
    """
    cleaned = {k: v for k, v in prefill.items() if v not in (None, "", 0)}
    if not cleaned:
        return ""
    return "&" + urlencode({k: str(v) for k, v in cleaned.items()})


async def _handle_user_navigate(
    ctx: ToolContext,
    keyword: str,
    action_suffix: str,
    tool_name: str,
    action_label_template: str,
    prefill: dict[str, Any] | None = None,
) -> AdminApiResult:
    """
    user 검색 계열 6개 handler 의 공통 구현.

    1. GET /api/v1/admin/users?keyword=...&size=10 으로 후보 검색
    2. 후보 0건 → ok=False 에러
    3. 후보 1건 → 단건 navigation payload
    4. 후보 여러 건 → candidates 배열 포함 payload

    prefill: 정지 사유·기간·역할·금액 등 폼 prefill 용 dict. None 또는 빈 dict 면 미적용.
    """
    result = await _search_users(ctx, keyword)
    if not result.ok:
        # 검색 자체 실패 — 그대로 전파
        return result

    items = _extract_items(result.data)

    if not items:
        return AdminApiResult(
            ok=False,
            status_code=200,
            data=None,
            error=f"'{keyword}' 로 찾은 사용자가 없어요. 다른 조건으로 시도해 보세요.",
            latency_ms=result.latency_ms,
            row_count=0,
        )

    prefill_qs = _build_prefill_qs(prefill or {})

    payload = _build_user_navigation_result(
        items=items,
        keyword=keyword,
        action_suffix=action_suffix,
        tool_name=tool_name,
        action_label_template=action_label_template,
        prefill_qs=prefill_qs,
    )

    return AdminApiResult(
        ok=True,
        status_code=200,
        data=payload,
        latency_ms=result.latency_ms,
        row_count=len(items),
    )


# ============================================================
# Handlers — user 검색 계열 (6개)
# ============================================================

async def _handle_goto_user_detail(
    ctx: ToolContext,
    keyword: str,
    reason: str | None = None,
    suspendUntil: str | None = None,
    targetRole: str | None = None,
    pointAmount: int | None = None,
    tokenAmount: int | None = None,
) -> AdminApiResult:
    """
    user 검색 → 상세 화면 이동. (상세 보기는 prefill 필드 미사용)
    target_path: /admin/users?userId={userId}
    """
    return await _handle_user_navigate(
        ctx=ctx,
        keyword=keyword,
        action_suffix="",
        tool_name="goto_user_detail",
        action_label_template="{display} 상세 화면 열기",
        prefill=None,
    )


async def _handle_goto_user_suspend(
    ctx: ToolContext,
    keyword: str,
    reason: str | None = None,
    suspendUntil: str | None = None,
    targetRole: str | None = None,
    pointAmount: int | None = None,
    tokenAmount: int | None = None,
) -> AdminApiResult:
    """
    user 검색 → 정지 처리 폼 화면 이동.
    target_path: /admin/users?userId={userId}&action=suspend[&reason=...&suspendUntil=...]
    """
    return await _handle_user_navigate(
        ctx=ctx,
        keyword=keyword,
        action_suffix="&action=suspend",
        tool_name="goto_user_suspend",
        action_label_template="{display} 계정 정지 화면으로 이동",
        prefill={"reason": reason, "suspendUntil": suspendUntil},
    )


async def _handle_goto_user_activate(
    ctx: ToolContext,
    keyword: str,
    reason: str | None = None,
    suspendUntil: str | None = None,
    targetRole: str | None = None,
    pointAmount: int | None = None,
    tokenAmount: int | None = None,
) -> AdminApiResult:
    """
    user 검색 → 계정 복구 폼 화면 이동.
    target_path: /admin/users?userId={userId}&action=activate[&reason=...]
    """
    return await _handle_user_navigate(
        ctx=ctx,
        keyword=keyword,
        action_suffix="&action=activate",
        tool_name="goto_user_activate",
        action_label_template="{display} 계정 복구 화면으로 이동",
        prefill={"reason": reason},
    )


async def _handle_goto_user_role_change(
    ctx: ToolContext,
    keyword: str,
    reason: str | None = None,
    suspendUntil: str | None = None,
    targetRole: str | None = None,
    pointAmount: int | None = None,
    tokenAmount: int | None = None,
) -> AdminApiResult:
    """
    user 검색 → 역할 변경 폼 화면 이동.
    target_path: /admin/users?userId={userId}&action=role[&targetRole=...&reason=...]
    """
    return await _handle_user_navigate(
        ctx=ctx,
        keyword=keyword,
        action_suffix="&action=role",
        tool_name="goto_user_role_change",
        action_label_template="{display} 역할 변경 화면으로 이동",
        prefill={"targetRole": targetRole, "reason": reason},
    )


async def _handle_goto_points_adjust(
    ctx: ToolContext,
    keyword: str,
    reason: str | None = None,
    suspendUntil: str | None = None,
    targetRole: str | None = None,
    pointAmount: int | None = None,
    tokenAmount: int | None = None,
) -> AdminApiResult:
    """
    user 검색 → 포인트 조정 폼 화면 이동.
    target_path: /admin/users?userId={userId}&action=points-adjust[&pointAmount=...&reason=...]
    """
    return await _handle_user_navigate(
        ctx=ctx,
        keyword=keyword,
        action_suffix="&action=points-adjust",
        tool_name="goto_points_adjust",
        action_label_template="{display} 포인트 조정 화면으로 이동",
        prefill={"pointAmount": pointAmount, "reason": reason},
    )


async def _handle_goto_token_grant(
    ctx: ToolContext,
    keyword: str,
    reason: str | None = None,
    suspendUntil: str | None = None,
    targetRole: str | None = None,
    pointAmount: int | None = None,
    tokenAmount: int | None = None,
) -> AdminApiResult:
    """
    user 검색 → 이용권 발급 폼 화면 이동.
    target_path: /admin/users?userId={userId}&action=tokens-grant[&tokenAmount=...&reason=...]
    """
    return await _handle_user_navigate(
        ctx=ctx,
        keyword=keyword,
        action_suffix="&action=tokens-grant",
        tool_name="goto_token_grant",
        action_label_template="{display} 이용권 발급 화면으로 이동",
        prefill={"tokenAmount": tokenAmount, "reason": reason},
    )


# ============================================================
# Handler — goto_order_detail
# ============================================================

async def _handle_goto_order_detail(
    ctx: ToolContext,
    orderId: Optional[str] = None,
    userId: Optional[str] = None,
) -> AdminApiResult:
    """
    주문 검색 후 상세 화면 이동.

    orderId 가 있으면 GET /api/v1/admin/payment/orders/{orderId} 단건 조회.
    없으면 GET /api/v1/admin/payment/orders?userId={userId}&size=10 으로 fallback.
    target_path: /admin/payment?tab=orders&orderId={orderId}
    """
    if not orderId and not userId:
        return AdminApiResult(
            ok=False,
            status_code=400,
            data=None,
            error="orderId 또는 userId 중 하나는 반드시 입력해야 해요.",
            latency_ms=0,
            row_count=0,
        )

    tool_name = "goto_order_detail"

    if orderId:
        # 단건 직접 조회
        raw = await get_admin_json(
            f"/api/v1/admin/payment/orders/{orderId}",
            admin_jwt=ctx.admin_jwt,
            invocation_id=ctx.invocation_id,
        )
        result = unwrap_api_response(raw)
        if not result.ok:
            return result

        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": f"/admin/payment?tab=orders&orderId={orderId}",
                "label": f"주문 {orderId} 상세 화면 열기",
                "context_summary": f"주문 {orderId} 를 찾았어요. 상세 화면으로 이동하실 수 있어요.",
                "tool_name": tool_name,
            },
            latency_ms=result.latency_ms,
            row_count=1,
        )

    # userId fallback — 목록 검색
    raw = await get_admin_json(
        "/api/v1/admin/payment/orders",
        admin_jwt=ctx.admin_jwt,
        params={"userId": userId, "size": 10},
        invocation_id=ctx.invocation_id,
    )
    result = unwrap_api_response(raw)
    if not result.ok:
        return result

    items = _extract_items(result.data)
    if not items:
        return AdminApiResult(
            ok=False,
            status_code=200,
            data=None,
            error=f"userId={userId} 의 주문이 없어요. 다른 조건으로 시도해 보세요.",
            latency_ms=result.latency_ms,
            row_count=0,
        )

    def _oid(o: dict) -> str:
        return str(o.get("orderId") or o.get("order_id") or o.get("id") or "")

    def _olabel(o: dict) -> str:
        oid = _oid(o)
        created = str(o.get("createdAt") or o.get("created_at") or "")[:10]
        amount = o.get("totalAmount") or o.get("amount") or ""
        parts = [oid]
        if amount:
            parts.append(f"{amount}원")
        if created:
            parts.append(created)
        return " / ".join(parts)

    if len(items) == 1:
        o = items[0]
        oid = _oid(o)
        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": f"/admin/payment?tab=orders&orderId={oid}",
                "label": f"주문 {oid} 상세 화면 열기",
                "context_summary": f"userId={userId} 의 주문 1건을 찾았어요.",
                "tool_name": tool_name,
            },
            latency_ms=result.latency_ms,
            row_count=1,
        )

    candidates = [
        {
            "target_path": f"/admin/payment?tab=orders&orderId={_oid(o)}",
            "label": _olabel(o),
        }
        for o in items[:10]
    ]
    return AdminApiResult(
        ok=True,
        status_code=200,
        data={
            "target_path": None,
            "label": "주문을 선택하세요",
            "context_summary": (
                f"userId={userId} 의 주문 {len(items)}건이 조회됐어요. "
                "이동할 주문을 골라주세요."
            ),
            "candidates": candidates,
            "tool_name": tool_name,
        },
        latency_ms=result.latency_ms,
        row_count=len(items),
    )


# ============================================================
# Handler — goto_order_refund
# ============================================================

async def _handle_goto_order_refund(
    ctx: ToolContext,
    orderId: Optional[str] = None,
    userId: Optional[str] = None,
) -> AdminApiResult:
    """
    주문 검색 후 환불 처리 폼 화면 이동.

    orderId 탐색 로직은 goto_order_detail 과 동일. target_path 만 action=refund 추가.
    실제 환불 처리는 관리자가 해당 폼에서 직접 실행한다.
    target_path: /admin/payment?tab=orders&orderId={orderId}&action=refund
    """
    if not orderId and not userId:
        return AdminApiResult(
            ok=False,
            status_code=400,
            data=None,
            error="orderId 또는 userId 중 하나는 반드시 입력해야 해요.",
            latency_ms=0,
            row_count=0,
        )

    tool_name = "goto_order_refund"

    if orderId:
        raw = await get_admin_json(
            f"/api/v1/admin/payment/orders/{orderId}",
            admin_jwt=ctx.admin_jwt,
            invocation_id=ctx.invocation_id,
        )
        result = unwrap_api_response(raw)
        if not result.ok:
            return result

        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": f"/admin/payment?tab=orders&orderId={orderId}&action=refund",
                "label": f"주문 {orderId} 환불 화면으로 이동",
                "context_summary": f"주문 {orderId} 를 찾았어요. 환불 처리는 해당 화면에서 직접 해주세요.",
                "tool_name": tool_name,
            },
            latency_ms=result.latency_ms,
            row_count=1,
        )

    # userId fallback
    raw = await get_admin_json(
        "/api/v1/admin/payment/orders",
        admin_jwt=ctx.admin_jwt,
        params={"userId": userId, "size": 10},
        invocation_id=ctx.invocation_id,
    )
    result = unwrap_api_response(raw)
    if not result.ok:
        return result

    items = _extract_items(result.data)
    if not items:
        return AdminApiResult(
            ok=False,
            status_code=200,
            data=None,
            error=f"userId={userId} 의 주문이 없어요. 다른 조건으로 시도해 보세요.",
            latency_ms=result.latency_ms,
            row_count=0,
        )

    def _oid(o: dict) -> str:
        return str(o.get("orderId") or o.get("order_id") or o.get("id") or "")

    def _olabel(o: dict) -> str:
        oid = _oid(o)
        created = str(o.get("createdAt") or o.get("created_at") or "")[:10]
        amount = o.get("totalAmount") or o.get("amount") or ""
        parts = [oid]
        if amount:
            parts.append(f"{amount}원")
        if created:
            parts.append(created)
        return " / ".join(parts)

    if len(items) == 1:
        o = items[0]
        oid = _oid(o)
        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": f"/admin/payment?tab=orders&orderId={oid}&action=refund",
                "label": f"주문 {oid} 환불 화면으로 이동",
                "context_summary": f"userId={userId} 의 주문 1건을 찾았어요. 환불 처리는 해당 화면에서 직접 해주세요.",
                "tool_name": tool_name,
            },
            latency_ms=result.latency_ms,
            row_count=1,
        )

    candidates = [
        {
            "target_path": f"/admin/payment?tab=orders&orderId={_oid(o)}&action=refund",
            "label": _olabel(o),
        }
        for o in items[:10]
    ]
    return AdminApiResult(
        ok=True,
        status_code=200,
        data={
            "target_path": None,
            "label": "환불할 주문을 선택하세요",
            "context_summary": (
                f"userId={userId} 의 주문 {len(items)}건이 조회됐어요. "
                "환불할 주문을 골라주세요."
            ),
            "candidates": candidates,
            "tool_name": tool_name,
        },
        latency_ms=result.latency_ms,
        row_count=len(items),
    )


# ============================================================
# Handler — goto_subscription_manage
# ============================================================

async def _handle_goto_subscription_manage(
    ctx: ToolContext,
    userId: Optional[str] = None,
    subscriptionId: Optional[str] = None,
    status: Optional[str] = None,
) -> AdminApiResult:
    """
    구독 검색 후 관리 화면 이동.

    subscriptionId 가 있으면 바로 단건 링크 생성.
    없으면 GET /api/v1/admin/subscription?userId=...&status=...&size=10 으로 검색.
    target_path:
      - 단건: /admin/payment?tab=subscriptions&subscriptionId={id}
      - userId 만 있으면: /admin/payment?tab=subscriptions&userId={userId}
      - candidates 배열: target_path=None
    """
    tool_name = "goto_subscription_manage"

    if subscriptionId:
        # subscriptionId 가 명시된 경우 — 검색 없이 즉시 링크 생성
        target = f"/admin/payment?tab=subscriptions&subscriptionId={subscriptionId}"
        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": target,
                "label": f"구독 {subscriptionId} 관리 화면으로 이동",
                "context_summary": f"구독 {subscriptionId} 화면으로 바로 이동할 수 있어요.",
                "tool_name": tool_name,
            },
            latency_ms=0,
            row_count=1,
        )

    if not userId and not status:
        return AdminApiResult(
            ok=False,
            status_code=400,
            data=None,
            error="subscriptionId, userId, status 중 하나는 입력해야 해요.",
            latency_ms=0,
            row_count=0,
        )

    # userId 또는 status 로 목록 검색
    params: dict = {"size": 10}
    if userId:
        params["userId"] = userId
    if status:
        params["status"] = status

    raw = await get_admin_json(
        "/api/v1/admin/subscription",
        admin_jwt=ctx.admin_jwt,
        params=params,
        invocation_id=ctx.invocation_id,
    )
    result = unwrap_api_response(raw)
    if not result.ok:
        # userId 단독으로 링크를 그냥 생성해 fallback 제공
        if userId:
            target = f"/admin/payment?tab=subscriptions&userId={userId}"
            return AdminApiResult(
                ok=True,
                status_code=200,
                data={
                    "target_path": target,
                    "label": f"userId={userId} 구독 관리 화면으로 이동",
                    "context_summary": (
                        "구독 조회 API 가 실패했지만 userId 기반 화면 링크를 제공해요. "
                        "직접 확인해 주세요."
                    ),
                    "tool_name": tool_name,
                },
                latency_ms=result.latency_ms,
                row_count=0,
            )
        return result

    items = _extract_items(result.data)

    if not items:
        # 조회 결과가 없어도 userId 가 있으면 기본 링크 제공
        if userId:
            target = f"/admin/payment?tab=subscriptions&userId={userId}"
            return AdminApiResult(
                ok=True,
                status_code=200,
                data={
                    "target_path": target,
                    "label": f"userId={userId} 구독 관리 화면으로 이동",
                    "context_summary": "해당 조건의 구독이 없지만 사용자 구독 화면으로 이동할 수 있어요.",
                    "tool_name": tool_name,
                },
                latency_ms=result.latency_ms,
                row_count=0,
            )
        return AdminApiResult(
            ok=False,
            status_code=200,
            data=None,
            error="해당 조건의 구독이 없어요. 다른 조건으로 시도해 보세요.",
            latency_ms=result.latency_ms,
            row_count=0,
        )

    def _sid(s: dict) -> str:
        return str(s.get("subscriptionId") or s.get("subscription_id") or s.get("id") or "")

    def _slabel(s: dict) -> str:
        sid = _sid(s)
        plan = s.get("planCode") or s.get("plan_code") or ""
        sub_status = s.get("status") or ""
        parts = [f"구독 {sid}"]
        if plan:
            parts.append(plan)
        if sub_status:
            parts.append(sub_status)
        return " / ".join(parts)

    if len(items) == 1:
        s = items[0]
        sid = _sid(s)
        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": f"/admin/payment?tab=subscriptions&subscriptionId={sid}",
                "label": f"구독 {sid} 관리 화면으로 이동",
                "context_summary": "조건에 맞는 구독 1건을 찾았어요. 해당 화면으로 이동하실 수 있어요.",
                "tool_name": tool_name,
            },
            latency_ms=result.latency_ms,
            row_count=1,
        )

    candidates = [
        {
            "target_path": f"/admin/payment?tab=subscriptions&subscriptionId={_sid(s)}",
            "label": _slabel(s),
        }
        for s in items[:10]
    ]
    return AdminApiResult(
        ok=True,
        status_code=200,
        data={
            "target_path": None,
            "label": "구독을 선택하세요",
            "context_summary": (
                f"조건에 맞는 구독 {len(items)}건이 조회됐어요. "
                "이동할 구독을 골라주세요."
            ),
            "candidates": candidates,
            "tool_name": tool_name,
        },
        latency_ms=result.latency_ms,
        row_count=len(items),
    )


# ============================================================
# Handler — goto_report_detail
# ============================================================

async def _handle_goto_report_detail(
    ctx: ToolContext,
    reportId: Optional[int] = None,
    page: Optional[int] = None,
) -> AdminApiResult:
    """
    신고 처리 화면 이동.

    reportId 가 있으면 즉시 단건 링크 생성 (Backend 조회 없이 ID 그대로 사용).
    없으면 GET /api/v1/admin/reports?page=0&size=10 최신 목록을 가져와 candidates 제공.
    target_path: /admin/board?tab=reports&reportId={id}
    """
    tool_name = "goto_report_detail"

    if reportId is not None:
        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": f"/admin/board?tab=reports&reportId={reportId}",
                "label": f"신고 {reportId}번 처리 화면으로 이동",
                "context_summary": f"신고 {reportId}번 화면으로 바로 이동할 수 있어요.",
                "tool_name": tool_name,
            },
            latency_ms=0,
            row_count=1,
        )

    # 최신 신고 목록 조회
    fetch_page = page if page is not None else 0
    raw = await get_admin_json(
        "/api/v1/admin/reports",
        admin_jwt=ctx.admin_jwt,
        params={"page": fetch_page, "size": 10},
        invocation_id=ctx.invocation_id,
    )
    result = unwrap_api_response(raw)
    if not result.ok:
        return result

    items = _extract_items(result.data)
    if not items:
        return AdminApiResult(
            ok=False,
            status_code=200,
            data=None,
            error="해당 조건의 신고가 없어요. 다른 조건으로 시도해 보세요.",
            latency_ms=result.latency_ms,
            row_count=0,
        )

    def _rid(r: dict) -> int | str:
        return r.get("reportId") or r.get("id") or ""

    def _rlabel(r: dict) -> str:
        rid = _rid(r)
        reason = r.get("reason") or r.get("reportReason") or ""
        status = r.get("status") or ""
        created = str(r.get("createdAt") or "")[:10]
        parts = [f"신고 {rid}번"]
        if reason:
            parts.append(str(reason)[:20])
        if status:
            parts.append(status)
        if created:
            parts.append(created)
        return " / ".join(parts)

    if len(items) == 1:
        r = items[0]
        rid = _rid(r)
        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": f"/admin/board?tab=reports&reportId={rid}",
                "label": f"신고 {rid}번 처리 화면으로 이동",
                "context_summary": "신고 1건을 찾았어요. 해당 화면으로 이동하실 수 있어요.",
                "tool_name": tool_name,
            },
            latency_ms=result.latency_ms,
            row_count=1,
        )

    candidates = [
        {
            "target_path": f"/admin/board?tab=reports&reportId={_rid(r)}",
            "label": _rlabel(r),
        }
        for r in items[:10]
    ]
    return AdminApiResult(
        ok=True,
        status_code=200,
        data={
            "target_path": None,
            "label": "신고를 선택하세요",
            "context_summary": (
                f"최근 신고 {len(items)}건을 가져왔어요. "
                "이동할 신고를 골라주세요."
            ),
            "candidates": candidates,
            "tool_name": tool_name,
        },
        latency_ms=result.latency_ms,
        row_count=len(items),
    )


# ============================================================
# Handler — goto_ticket_detail
# ============================================================

async def _handle_goto_ticket_detail(
    ctx: ToolContext,
    ticketId: Optional[int] = None,
    userId: Optional[str] = None,
) -> AdminApiResult:
    """
    고객 문의 티켓 화면 이동.

    ticketId 가 있으면 GET /api/v1/admin/tickets/{id} 단건 조회 후 링크 생성.
    없으면 GET /api/v1/admin/tickets?userId=... 로 목록 검색.
    target_path: /admin/support?tab=tickets&ticketId={id}
    """
    tool_name = "goto_ticket_detail"

    if ticketId is not None:
        raw = await get_admin_json(
            f"/api/v1/admin/tickets/{ticketId}",
            admin_jwt=ctx.admin_jwt,
            invocation_id=ctx.invocation_id,
        )
        result = unwrap_api_response(raw)
        if not result.ok:
            return result

        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": f"/admin/support?tab=tickets&ticketId={ticketId}",
                "label": f"티켓 {ticketId}번 상세 화면으로 이동",
                "context_summary": f"티켓 {ticketId}번 화면으로 바로 이동할 수 있어요.",
                "tool_name": tool_name,
            },
            latency_ms=result.latency_ms,
            row_count=1,
        )

    if not userId:
        return AdminApiResult(
            ok=False,
            status_code=400,
            data=None,
            error="ticketId 또는 userId 중 하나는 반드시 입력해야 해요.",
            latency_ms=0,
            row_count=0,
        )

    # userId 로 목록 검색
    raw = await get_admin_json(
        "/api/v1/admin/tickets",
        admin_jwt=ctx.admin_jwt,
        params={"userId": userId, "size": 10},
        invocation_id=ctx.invocation_id,
    )
    result = unwrap_api_response(raw)
    if not result.ok:
        return result

    items = _extract_items(result.data)
    if not items:
        return AdminApiResult(
            ok=False,
            status_code=200,
            data=None,
            error=f"userId={userId} 의 티켓이 없어요. 다른 조건으로 시도해 보세요.",
            latency_ms=result.latency_ms,
            row_count=0,
        )

    def _tid(t: dict) -> int | str:
        return t.get("ticketId") or t.get("id") or ""

    def _tlabel(t: dict) -> str:
        tid = _tid(t)
        title = t.get("title") or t.get("subject") or ""
        status = t.get("status") or ""
        created = str(t.get("createdAt") or "")[:10]
        parts = [f"티켓 {tid}번"]
        if title:
            parts.append(str(title)[:30])
        if status:
            parts.append(status)
        if created:
            parts.append(created)
        return " / ".join(parts)

    if len(items) == 1:
        t = items[0]
        tid = _tid(t)
        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": f"/admin/support?tab=tickets&ticketId={tid}",
                "label": f"티켓 {tid}번 상세 화면으로 이동",
                "context_summary": f"userId={userId} 의 티켓 1건을 찾았어요.",
                "tool_name": tool_name,
            },
            latency_ms=result.latency_ms,
            row_count=1,
        )

    candidates = [
        {
            "target_path": f"/admin/support?tab=tickets&ticketId={_tid(t)}",
            "label": _tlabel(t),
        }
        for t in items[:10]
    ]
    return AdminApiResult(
        ok=True,
        status_code=200,
        data={
            "target_path": None,
            "label": "티켓을 선택하세요",
            "context_summary": (
                f"userId={userId} 의 티켓 {len(items)}건이 조회됐어요. "
                "이동할 티켓을 골라주세요."
            ),
            "candidates": candidates,
            "tool_name": tool_name,
        },
        latency_ms=result.latency_ms,
        row_count=len(items),
    )


# ============================================================
# 2026-04-28 신설 — Notice navigate (공지사항 상세 / 목록)
# ============================================================
# 기존에는 공지사항 navigate tool 자체가 없어 "공지 옛날거 삭제해줘" 발화 시 LLM 이
# notice_draft 로 잘못 매칭 → modal=create 폼이 떴다. 길 A 보강으로 별도 navigate tool 신설.

class _NoticeNavArgs(BaseModel):
    """공지 navigate 공통 — id 우선, 없으면 keyword 로 목록 화면."""

    noticeId: Optional[int] = Field(
        default=None,
        description="이동할 공지 ID. 알고 있으면 즉시 단건 링크를 생성한다.",
    )
    keyword: Optional[str] = Field(
        default=None,
        description="공지 제목/본문 검색어. id 가 없을 때 목록 화면으로 보낼 검색어로 사용.",
    )


async def _handle_goto_notice_detail(
    ctx: ToolContext,
    noticeId: Optional[int] = None,
    keyword: Optional[str] = None,
) -> AdminApiResult:
    """
    공지사항 상세 화면 이동 (수정·삭제 모두 이 화면에서 수행).

    target_path: /admin/support?tab=notice&noticeId={id}
    Backend 조회 없이 ID 그대로 링크. ID 모르면 목록 화면(keyword 검색)으로 보냄.
    실제 수정·삭제는 관리자가 화면에서 직접 수행한다.
    """
    tool_name = "goto_notice_detail"

    if noticeId is not None:
        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": f"/admin/support?tab=notice&noticeId={noticeId}",
                "label": f"공지 #{noticeId} 상세 화면으로 이동",
                "context_summary": (
                    f"공지 #{noticeId} 화면으로 바로 이동할 수 있어요. "
                    "삭제·수정은 화면 우측 상단 메뉴에서 직접 수행해 주세요."
                ),
                "tool_name": tool_name,
            },
            latency_ms=0,
            row_count=1,
        )

    if not keyword:
        return AdminApiResult(
            ok=False,
            status_code=400,
            data=None,
            error="noticeId 또는 keyword 중 하나는 입력해야 해요.",
            latency_ms=0,
            row_count=0,
        )

    # keyword 만 있으면 목록 화면 + 검색어로 fallback
    qs = urlencode({"q": keyword})
    return AdminApiResult(
        ok=True,
        status_code=200,
        data={
            "target_path": f"/admin/support?tab=notice&{qs}",
            "label": f"공지 '{keyword}' 검색 화면으로 이동",
            "context_summary": (
                f"공지 ID 를 모르므로 '{keyword}' 검색 결과 화면으로 이동해요. "
                "목록에서 대상 공지를 클릭한 뒤 삭제·수정해 주세요."
            ),
            "tool_name": tool_name,
        },
        latency_ms=0,
        row_count=None,
    )


async def _handle_goto_notice_list(
    ctx: ToolContext,
    noticeId: Optional[int] = None,
    keyword: Optional[str] = None,
) -> AdminApiResult:
    """
    공지사항 목록 화면 이동 (전체 목록 + 선택적 검색어).

    target_path: /admin/support?tab=notice[&q=keyword]
    keyword 미입력이면 전체 목록.
    """
    tool_name = "goto_notice_list"

    if keyword:
        qs = urlencode({"q": keyword})
        return AdminApiResult(
            ok=True,
            status_code=200,
            data={
                "target_path": f"/admin/support?tab=notice&{qs}",
                "label": f"공지 '{keyword}' 검색 화면으로 이동",
                "context_summary": f"공지 '{keyword}' 검색 결과 화면으로 이동해요.",
                "tool_name": tool_name,
            },
            latency_ms=0,
            row_count=None,
        )

    return AdminApiResult(
        ok=True,
        status_code=200,
        data={
            "target_path": "/admin/support?tab=notice",
            "label": "공지사항 목록 화면으로 이동",
            "context_summary": "전체 공지사항 목록 화면으로 이동해요.",
            "tool_name": tool_name,
        },
        latency_ms=0,
        row_count=None,
    )


# ============================================================
# Handler — goto_audit_log
# ============================================================

async def _handle_goto_audit_log(
    ctx: ToolContext,
    q: str,
    actor: Optional[str] = None,
    action: Optional[str] = None,
) -> AdminApiResult:
    """
    감사 로그 검색 화면 이동.

    Backend 조회 없이 입력 파라미터를 URL 쿼리스트링에 그대로 실어 링크만 생성한다.
    target_path: /admin/system?q={q}[&actor={actor}][&action={action}]

    2026-04-23 수정: AuditLogTab 은 `/admin/settings` 가 아니라 `/admin/system` 페이지에
    상시 렌더된다(탭 구분 없음). Settings 경로에서는 해당 탭이 없어 이동해도 빈 화면.
    SystemPage 가 useQueryParams 로 q/actor/action 을 읽어 검색 필터 초기값으로 주입.
    """
    tool_name = "goto_audit_log"

    # 쿼리스트링 파라미터 조립 — SystemPage 의 AuditLogTab 초기 필터값으로 매핑
    qs_parts: dict[str, str] = {"q": q}
    if actor:
        qs_parts["actor"] = actor
    if action:
        qs_parts["action"] = action

    # URL 인코딩으로 특수문자 안전 처리
    qs = urlencode(qs_parts)
    target = f"/admin/system?{qs}"

    # context_summary 구성
    summary_parts = [f"검색어 '{q}'"]
    if actor:
        summary_parts.append(f"관리자 '{actor}'")
    if action:
        summary_parts.append(f"액션 '{action}'")
    summary = " / ".join(summary_parts) + " 조건으로 감사 로그 화면으로 이동할 수 있어요."

    return AdminApiResult(
        ok=True,
        status_code=200,
        data={
            "target_path": target,
            "label": f"감사 로그 '{q}' 검색 화면으로 이동",
            "context_summary": summary,
            "tool_name": tool_name,
        },
        latency_ms=0,
        row_count=None,
    )


# ============================================================
# Registration — 12개 tool 등록
# ============================================================

register_tool(ToolSpec(
    name="goto_user_detail",
    tier=0,
    required_roles=_USER_NAV_ROLES,
    description=(
        "이메일·닉네임·user_id 키워드로 사용자를 검색해 상세 관리 화면으로 이동할 링크를 "
        "제공합니다. 실제 처리는 관리자가 화면에서 직접 수행합니다. "
        "후보가 여러 명이면 선택지 목록을 제공합니다."
    ),
    example_questions=[
        "chulsoo 유저 상세 보고 싶어",
        "dhgapdl@test.com 계정 화면으로 데려가줘",
        "닉네임에 '몽글' 들어간 유저 찾아줘",
    ],
    args_schema=_UserKeywordArgs,
    handler=_handle_goto_user_detail,
))


register_tool(ToolSpec(
    name="goto_user_suspend",
    tier=0,
    required_roles=_USER_NAV_ROLES,
    description=(
        "이메일·닉네임·user_id 키워드로 사용자를 검색해 계정 정지 처리 폼 화면으로 이동할 "
        "링크를 제공합니다. **사용자 발화에 정지 사유나 기간이 있으면 reason / suspendUntil "
        "필드를 함께 채워야 합니다.** 예: '~사유는 X' → reason='X', '7일 정지' → suspendUntil 을 "
        "오늘+7일 ISO 8601 로 환산. Client 폼이 prefill 받아 자동 채움. "
        "실제 처리는 관리자가 화면에서 직접 수행합니다."
    ),
    example_questions=[
        "chulsoo 계정 정지시켜줘",
        "spammer@test.com 유저 정지 처리 화면으로 이동",
        "abuser 닉네임 계정 정지 폼 열어줘",
        "이민수 계정 정지해줘 사유는 욕설 반복이야",   # reason 채우기
        "chulsoo 7일 정지 처리해줘",                   # suspendUntil 채우기
    ],
    args_schema=_UserKeywordArgs,
    handler=_handle_goto_user_suspend,
))


register_tool(ToolSpec(
    name="goto_user_activate",
    tier=0,
    required_roles=_USER_NAV_ROLES,
    description=(
        "이메일·닉네임·user_id 키워드로 사용자를 검색해 계정 복구(정지 해제) 폼 화면으로 "
        "이동할 링크를 제공합니다. 실제 처리는 관리자가 화면에서 직접 수행합니다. "
        "검색으로 대상을 찾아 해당 관리 화면으로 이동할 링크를 제공합니다."
    ),
    example_questions=[
        "suspended_user 정지 풀어줘",
        "chulsoo@test.com 계정 복구 화면으로 이동",
        "정지된 몽글팝콘 유저 활성화 폼 열어줘",
    ],
    args_schema=_UserKeywordArgs,
    handler=_handle_goto_user_activate,
))


register_tool(ToolSpec(
    name="goto_user_role_change",
    tier=0,
    required_roles=_USER_NAV_ROLES,
    description=(
        "이메일·닉네임·user_id 키워드로 사용자를 검색해 역할 변경 폼 화면으로 이동할 링크를 "
        "제공합니다. 실제 처리는 관리자가 화면에서 직접 수행합니다. "
        "검색으로 대상을 찾아 해당 관리 화면으로 이동할 링크를 제공합니다."
    ),
    example_questions=[
        "chulsoo 역할을 ADMIN 으로 바꿔줘",
        "moderator@test.com 권한 변경 화면으로 이동",
        "testuser 역할 수정 폼 열어줘",
    ],
    args_schema=_UserKeywordArgs,
    handler=_handle_goto_user_role_change,
))


register_tool(ToolSpec(
    name="goto_points_adjust",
    tier=0,
    required_roles=_FINANCE_NAV_ROLES,
    description=(
        "이메일·닉네임·user_id 키워드로 사용자를 검색해 포인트 수동 조정 폼 화면으로 이동할 "
        "링크를 제공합니다. 실제 처리는 관리자가 화면에서 직접 수행합니다. "
        "검색으로 대상을 찾아 해당 관리 화면으로 이동할 링크를 제공합니다."
    ),
    example_questions=[
        "chulsoo 포인트 조정해줘",
        "testuser@test.com 포인트 수동 추가 화면으로 이동",
        "mongle 유저 포인트 조정 폼 열어줘",
    ],
    args_schema=_UserKeywordArgs,
    handler=_handle_goto_points_adjust,
))


register_tool(ToolSpec(
    name="goto_token_grant",
    tier=0,
    required_roles=_FINANCE_NAV_ROLES,
    description=(
        "이메일·닉네임·user_id 키워드로 사용자를 검색해 AI 이용권 발급 폼 화면으로 이동할 "
        "링크를 제공합니다. 실제 처리는 관리자가 화면에서 직접 수행합니다. "
        "검색으로 대상을 찾아 해당 관리 화면으로 이동할 링크를 제공합니다."
    ),
    example_questions=[
        "chulsoo 이용권 발급해줘",
        "vip@test.com AI 쿠폰 지급 화면으로 이동",
        "이용권 부족한 mongle 유저 발급 폼 열어줘",
    ],
    args_schema=_UserKeywordArgs,
    handler=_handle_goto_token_grant,
))


register_tool(ToolSpec(
    name="goto_order_detail",
    tier=0,
    required_roles=_PAYMENT_NAV_ROLES,
    description=(
        "주문 ID 또는 사용자 ID 로 결제 주문을 조회해 상세 화면으로 이동할 링크를 "
        "제공합니다. orderId 를 알면 단건, 모르면 userId 로 목록을 가져와 선택지를 제공합니다. "
        "실제 처리는 관리자가 화면에서 직접 수행합니다."
    ),
    example_questions=[
        "주문 ord_abc123 상세 화면으로 이동",
        "chulsoo 최근 결제 주문 보여줘",
        "ord_xxx 주문 상태 확인하고 싶어",
    ],
    args_schema=_OrderNavArgs,
    handler=_handle_goto_order_detail,
))


register_tool(ToolSpec(
    name="goto_order_refund",
    tier=0,
    required_roles=_PAYMENT_NAV_ROLES,
    description=(
        "주문 ID 또는 사용자 ID 로 결제 주문을 조회해 환불 처리 폼 화면으로 이동할 링크를 "
        "제공합니다. 실제 환불 처리는 관리자가 해당 화면에서 직접 수행합니다. "
        "검색으로 대상을 찾아 해당 관리 화면으로 이동할 링크를 제공합니다."
    ),
    example_questions=[
        "chulsoo 환불해줘",
        "ord_abc123 환불 처리 화면으로 이동",
        "userId=usr_xxx 최근 주문 환불 폼 열어줘",
    ],
    args_schema=_OrderNavArgs,
    handler=_handle_goto_order_refund,
))


register_tool(ToolSpec(
    name="goto_subscription_manage",
    tier=0,
    required_roles=_PAYMENT_NAV_ROLES,
    description=(
        "사용자 ID·구독 ID·상태 필터로 구독을 조회해 구독 관리 화면으로 이동할 링크를 "
        "제공합니다. 구독 취소·플랜 변경 등 실제 처리는 관리자가 화면에서 직접 수행합니다. "
        "검색으로 대상을 찾아 해당 관리 화면으로 이동할 링크를 제공합니다."
    ),
    example_questions=[
        "sub_abc 구독 관리 화면으로 이동",
        "chulsoo 구독 취소 처리 화면 보여줘",
        "ACTIVE 구독 목록 화면으로 이동",
    ],
    args_schema=_SubscriptionNavArgs,
    handler=_handle_goto_subscription_manage,
))


register_tool(ToolSpec(
    name="goto_report_detail",
    tier=0,
    required_roles=_CONTENT_SUPPORT_NAV_ROLES,
    description=(
        "신고 ID 로 바로 이동하거나, ID 를 모르면 최신 신고 목록을 가져와 처리 화면 링크를 "
        "제공합니다. 실제 처리(승인/기각)는 관리자가 화면에서 직접 수행합니다. "
        "검색으로 대상을 찾아 해당 관리 화면으로 이동할 링크를 제공합니다."
    ),
    example_questions=[
        "신고 12번 처리 화면으로 이동",
        "reports 최신 신고 보여줘",
        "신고 처리해야 하는 거 링크 줘",
    ],
    args_schema=_ReportNavArgs,
    handler=_handle_goto_report_detail,
))


register_tool(ToolSpec(
    name="goto_ticket_detail",
    tier=0,
    required_roles=_CONTENT_SUPPORT_NAV_ROLES,
    description=(
        "고객 문의 티켓 ID 또는 사용자 ID 로 티켓을 조회해 상세 화면으로 이동할 링크를 "
        "제공합니다. 실제 처리(답변/종결)는 관리자가 화면에서 직접 수행합니다. "
        "검색으로 대상을 찾아 해당 관리 화면으로 이동할 링크를 제공합니다."
    ),
    example_questions=[
        "티켓 5번 상세 화면으로 이동",
        "chulsoo 고객 문의 티켓 보여줘",
        "ticket 99 처리 화면 링크 줘",
    ],
    args_schema=_TicketNavArgs,
    handler=_handle_goto_ticket_detail,
))


# 2026-04-28 신설 — 공지사항 navigate 2개
register_tool(ToolSpec(
    name="goto_notice_detail",
    tier=0,
    required_roles=_CONTENT_SUPPORT_NAV_ROLES,
    description=(
        "공지 ID 또는 제목 키워드로 공지사항을 찾아 상세 관리 화면으로 이동할 링크를 제공합니다. "
        "수정·삭제는 모두 이 화면 우측 상단 메뉴에서 직접 수행합니다. "
        "공지 삭제/수정/조회 의도 발화에 모두 이 도구를 사용하세요. "
        "ID 가 없으면 목록 검색 화면으로 fallback 합니다."
    ),
    example_questions=[
        "공지 1번 보여줘",
        "최신 공지 화면으로 이동",
        "공지사항 옛날거 삭제하러 가자",
        "서비스 점검 공지 화면으로 이동",
    ],
    args_schema=_NoticeNavArgs,
    handler=_handle_goto_notice_detail,
))

register_tool(ToolSpec(
    name="goto_notice_list",
    tier=0,
    required_roles=_CONTENT_SUPPORT_NAV_ROLES,
    description=(
        "공지사항 전체 목록 화면으로 이동할 링크를 제공합니다. keyword 가 있으면 검색 결과 화면. "
        "어떤 공지를 다룰지 특정되지 않을 때(여러 공지 정리·검토) 사용하세요."
    ),
    example_questions=[
        "공지 목록 보여줘",
        "공지사항 화면으로 이동",
        "이번 달 공지 검색해줘",
    ],
    args_schema=_NoticeNavArgs,
    handler=_handle_goto_notice_list,
))


register_tool(ToolSpec(
    name="goto_audit_log",
    tier=0,
    required_roles=_AUDIT_NAV_ROLES,
    description=(
        "검색어·관리자·액션 유형 조건을 URL 에 실어 감사 로그 검색 화면으로 이동할 링크를 "
        "제공합니다. Backend 조회 없이 조건만 링크로 구성합니다. "
        "검색으로 대상을 찾아 해당 관리 화면으로 이동할 링크를 제공합니다."
    ),
    example_questions=[
        "USER_SUSPEND 감사 로그 화면으로 이동",
        "admin01 이 한 작업 감사 로그 보고 싶어",
        "포인트 조정 감사 로그 검색 화면 링크 줘",
    ],
    args_schema=_AuditLogArgs,
    handler=_handle_goto_audit_log,
))
