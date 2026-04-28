"""
고객센터 AI 에이전트 v4 — `lookup_my_ai_quota` tool.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.1

Backend EP: GET /api/v1/point/ai-quota  (Phase 1.2 신규)
응답: {dailyAiUsed, dailyAiLimit, remainingAiBonus, purchasedAiTokens,
       monthlyCouponUsed, monthlyCouponLimit, resetAt}  (camelCase)

Sentinel 값 주의:
  - dailyAiLimit == -1   → DIAMOND 등급 무제한
  - remainingAiBonus == -1 → 활성 구독 없음

용도: "AI 추천 횟수 왜 안 돼요", "오늘 AI 몇 번 남았어요", "구독 보너스 얼마나 남았어요"
requires_login=True — 로그인한 사용자의 본인 데이터 조회이므로 게스트 차단.
"""

from __future__ import annotations

from pydantic import BaseModel

from . import ToolContext, SupportToolSpec, register_support_tool
from . import _base


# ============================================================
# 입력 스키마 (인자 없음 — 항상 현재 사용자 기준)
# ============================================================

class LookupAiQuotaArgs(BaseModel):
    """
    `lookup_my_ai_quota` tool 입력 스키마.

    AI 쿼터는 "현재 로그인한 사용자"의 상태를 조회하므로 추가 인자가 없다.
    """


# ============================================================
# Handler
# ============================================================

async def _handle(ctx: ToolContext) -> dict:
    """
    lookup_my_ai_quota tool 실행 핸들러.

    게스트 차단 후 Backend GET /api/v1/point/ai-quota 를 호출한다.

    Narrator 가 해석해야 할 sentinel 값:
      - data.dailyAiLimit == -1   : DIAMOND 등급 — "무제한"으로 표시
      - data.remainingAiBonus == -1: 구독 없음 — "구독 미가입"으로 표시

    반환 스키마:
        ok=True  → {"ok": True, "data": {dailyAiUsed, dailyAiLimit,
                                         remainingAiBonus, purchasedAiTokens,
                                         monthlyCouponUsed, monthlyCouponLimit, resetAt}}
        ok=False → {"ok": False, "error": "<사유>"}
    """
    if ctx.is_guest:
        return {
            "ok": False,
            "error": "login_required",
            "reason": "AI 쿼터 정보는 로그인 후 확인할 수 있어요.",
        }
    return await _base.call_backend_get(ctx, "/api/v1/point/ai-quota")


# ============================================================
# 레지스트리 등록 (import side-effect)
# ============================================================

register_support_tool(
    SupportToolSpec(
        name="lookup_my_ai_quota",
        description=(
            "본인의 AI 추천 이용 쿼터(일일 사용량·잔여량, 구독 보너스, 구매 이용권)를 조회합니다. "
            "'AI 추천이 왜 안 돼요', '오늘 AI 몇 번 남았어요', "
            "'구독 AI 보너스 얼마나 남았어요', 'AI 한도 초과됐대요' 같은 문의에 사용하세요. "
            "로그인 필수."
        ),
        args_schema=LookupAiQuotaArgs,
        handler=_handle,
        requires_login=True,
    )
)
