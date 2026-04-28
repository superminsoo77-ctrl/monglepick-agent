"""
고객센터 AI 에이전트 v4 — `lookup_my_tickets` tool.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.1

Backend EP: GET /api/v1/support/tickets
응답: [{id, status, category, title, createdAt}]  (camelCase)

용도: "문의 접수됐는지 확인해줘", "내 문의 처리됐나요", "이전에 문의한 것 알려줘"
requires_login=True — 로그인한 사용자의 본인 문의 내역 조회이므로 게스트 차단.
"""

from __future__ import annotations

from pydantic import BaseModel

from . import ToolContext, SupportToolSpec, register_support_tool
from . import _base


# ============================================================
# 입력 스키마 (인자 없음 — 항상 현재 사용자 기준)
# ============================================================

class LookupTicketsArgs(BaseModel):
    """
    `lookup_my_tickets` tool 입력 스키마.

    문의 내역은 "현재 로그인한 사용자"의 전체 이력을 조회하므로 추가 인자가 없다.
    """


# ============================================================
# Handler
# ============================================================

async def _handle(ctx: ToolContext) -> dict:
    """
    lookup_my_tickets tool 실행 핸들러.

    게스트 차단 후 Backend GET /api/v1/support/tickets 를 호출한다.

    반환 스키마:
        ok=True  → {"ok": True, "data": [{id, status, category, title, createdAt}, ...]}
        ok=False → {"ok": False, "error": "<사유>"}

    Narrator 해석 참고:
      - status: OPEN / IN_PROGRESS / RESOLVED / CLOSED
      - category: 문의 카테고리 (결제, 계정, AI 추천, 기타 등)
    """
    if ctx.is_guest:
        return {
            "ok": False,
            "error": "login_required",
            "reason": "문의 내역은 로그인 후 확인할 수 있어요.",
        }
    return await _base.call_backend_get(ctx, "/api/v1/support/tickets")


# ============================================================
# 레지스트리 등록 (import side-effect)
# ============================================================

register_support_tool(
    SupportToolSpec(
        name="lookup_my_tickets",
        description=(
            "본인이 접수한 문의 내역(문의 ID, 상태, 카테고리, 제목, 접수일)을 조회합니다. "
            "'문의 접수됐는지 확인해줘', '내 문의 처리됐나요', "
            "'이전에 문의한 것 알려줘', '1:1 문의 현황' 같은 문의 내역 확인에 사용하세요. "
            "로그인 필수."
        ),
        args_schema=LookupTicketsArgs,
        handler=_handle,
        requires_login=True,
    )
)
