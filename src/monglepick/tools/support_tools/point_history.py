"""
고객센터 AI 에이전트 v4 — `lookup_my_point_history` tool.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.1

Backend EP: GET /api/v1/point/history?days=N
응답: [{amount, type, source, createdAt}]  (camelCase)

용도: "포인트 안 들어왔어요", "포인트 언제 빠져나갔어요" 같은 포인트 이력 문의.
requires_login=True — 로그인한 사용자의 본인 데이터 조회이므로 게스트 차단.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import ToolContext, SupportToolSpec, register_support_tool
from . import _base


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
# Handler
# ============================================================

async def _handle(ctx: ToolContext, days: int = 7) -> dict:
    """
    lookup_my_point_history tool 실행 핸들러.

    게스트 차단 후 Backend GET /api/v1/point/history 를 호출한다.
    ctx.user_id 는 _base.call_backend_get 내부에서 X-User-Id 헤더로 강제 주입되므로
    params 에 user_id 를 별도로 담지 않는다.

    반환 스키마:
        ok=True  → {"ok": True, "data": [{amount, type, source, createdAt}, ...]}
        ok=False → {"ok": False, "error": "<사유>"}
    """
    # 게스트 차단 — requires_login=True 이지만 handler 에서도 이중 방어
    if ctx.is_guest:
        return {
            "ok": False,
            "error": "login_required",
            "reason": "포인트 이력은 로그인 후 확인할 수 있어요.",
        }
    return await _base.call_backend_get(ctx, "/api/v1/point/history", params={"days": days})


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
