"""
고객센터 AI 에이전트 v4 — `lookup_my_subscription` tool.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.1

Backend EP: GET /api/v1/subscription/status
응답: {isActive, plan, expiryDate, nextBillingDate}  (camelCase)

용도: "구독 중인지 확인해줘", "구독 언제 끝나요", "다음 결제일이 언제예요"
requires_login=True — 로그인한 사용자의 본인 구독 정보 조회이므로 게스트 차단.
"""

from __future__ import annotations

from pydantic import BaseModel

from . import ToolContext, SupportToolSpec, register_support_tool
from . import _base


# ============================================================
# 입력 스키마 (인자 없음 — 항상 현재 사용자 기준)
# ============================================================

class LookupSubscriptionArgs(BaseModel):
    """
    `lookup_my_subscription` tool 입력 스키마.

    구독 상태는 "현재 로그인한 사용자"의 상태를 조회하므로 추가 인자가 없다.
    """


# ============================================================
# Handler
# ============================================================

async def _handle(ctx: ToolContext) -> dict:
    """
    lookup_my_subscription tool 실행 핸들러.

    게스트 차단 후 Backend GET /api/v1/subscription/status 를 호출한다.

    반환 스키마:
        ok=True  → {"ok": True, "data": {isActive, plan, expiryDate, nextBillingDate}}
        ok=False → {"ok": False, "error": "<사유>"}

    Narrator 해석 참고:
      - isActive=False 이면 구독 중이 아님 → 구독 플랜 안내로 연결 가능.
      - plan: monthly_basic / monthly_premium / yearly_basic / yearly_premium / null
    """
    if ctx.is_guest:
        return {
            "ok": False,
            "error": "login_required",
            "reason": "구독 상태는 로그인 후 확인할 수 있어요.",
        }
    return await _base.call_backend_get(ctx, "/api/v1/subscription/status")


# ============================================================
# 레지스트리 등록 (import side-effect)
# ============================================================

register_support_tool(
    SupportToolSpec(
        name="lookup_my_subscription",
        description=(
            "본인의 구독 상태(활성 여부, 플랜 종류, 만료일, 다음 결제일)를 조회합니다. "
            "'구독 중인지 확인해줘', '구독 언제 끝나요', '다음 결제일이 언제예요', "
            "'구독 해지했는데 왜 아직 구독이에요' 같은 구독 관련 문의에 사용하세요. "
            "로그인 필수."
        ),
        args_schema=LookupSubscriptionArgs,
        handler=_handle,
        requires_login=True,
    )
)
