"""
고객센터 AI 에이전트 v4 — `lookup_my_orders` tool.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.1

Backend EP: GET /api/v1/payment/orders?days=N
응답: [{orderId, amount, status, paidAt}]  (camelCase, PII 마스킹됨)

용도: "결제 내역 확인해줘", "환불됐는지 확인해줘", "최근 결제가 뭐가 있어요"
requires_login=True — 로그인한 사용자의 본인 결제 내역 조회이므로 게스트 차단.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import ToolContext, SupportToolSpec, register_support_tool
from . import _base


# ============================================================
# 입력 스키마
# ============================================================

class LookupOrdersArgs(BaseModel):
    """
    `lookup_my_orders` tool 입력 스키마.

    LLM 은 사용자 발화에서 기간(일수)을 추출해 days 에 담는다.
    "이번 달 결제 내역" → 30, 기간 언급 없으면 기본 30일.
    """

    days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="조회 기간 (일). 1~365 사이. 기본 30일.",
    )


# ============================================================
# Handler
# ============================================================

async def _handle(ctx: ToolContext, days: int = 30) -> dict:
    """
    lookup_my_orders tool 실행 핸들러.

    게스트 차단 후 Backend GET /api/v1/payment/orders 를 호출한다.
    Backend 응답은 PII 마스킹이 적용된 상태로 반환된다.

    반환 스키마:
        ok=True  → {"ok": True, "data": [{orderId, amount, status, paidAt}, ...]}
        ok=False → {"ok": False, "error": "<사유>"}

    Narrator 해석 참고:
      - status: PAID / CANCELLED / REFUNDED / PENDING
      - amount: 원(KRW) 단위 정수
    """
    if ctx.is_guest:
        return {
            "ok": False,
            "error": "login_required",
            "reason": "결제 내역은 로그인 후 확인할 수 있어요.",
        }
    return await _base.call_backend_get(ctx, "/api/v1/payment/orders", params={"days": days})


# ============================================================
# 레지스트리 등록 (import side-effect)
# ============================================================

register_support_tool(
    SupportToolSpec(
        name="lookup_my_orders",
        description=(
            "본인의 결제 주문 내역(주문 ID, 금액, 상태, 결제일)을 조회합니다. "
            "'결제 내역 확인해줘', '환불됐는지 확인해줘', '최근 결제가 뭐가 있어요', "
            "'구독 결제 됐나요' 같은 결제 관련 문의에 사용하세요. 로그인 필수."
        ),
        args_schema=LookupOrdersArgs,
        handler=_handle,
        requires_login=True,
    )
)
