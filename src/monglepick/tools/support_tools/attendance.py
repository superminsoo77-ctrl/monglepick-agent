"""
고객센터 AI 에이전트 v4 — `lookup_my_attendance` tool.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.1

Backend EP: GET /api/v1/point/attendance/status
응답: {streak, totalDays, monthlyDates[], todayChecked}  (camelCase)

용도: "출석 체크 안 됐어요", "연속 출석이 몇 일이에요", "오늘 출석 체크 했나요" 같은
      출석/리워드 관련 문의.
requires_login=True — 로그인한 사용자의 본인 데이터 조회이므로 게스트 차단.
"""

from __future__ import annotations

from pydantic import BaseModel

from . import ToolContext, SupportToolSpec, register_support_tool
from . import _base


# ============================================================
# 입력 스키마 (인자 없음 — 항상 현재 사용자 기준)
# ============================================================

class LookupAttendanceArgs(BaseModel):
    """
    `lookup_my_attendance` tool 입력 스키마.

    출석 현황은 "현재 로그인한 사용자"의 상태를 조회하므로 추가 인자가 없다.
    """


# ============================================================
# Handler
# ============================================================

async def _handle(ctx: ToolContext) -> dict:
    """
    lookup_my_attendance tool 실행 핸들러.

    게스트 차단 후 Backend GET /api/v1/point/attendance/status 를 호출한다.

    반환 스키마:
        ok=True  → {"ok": True, "data": {streak, totalDays, monthlyDates, todayChecked}}
        ok=False → {"ok": False, "error": "<사유>"}
    """
    if ctx.is_guest:
        return {
            "ok": False,
            "error": "login_required",
            "reason": "출석 현황은 로그인 후 확인할 수 있어요.",
        }
    return await _base.call_backend_get(ctx, "/api/v1/point/attendance/status")


# ============================================================
# 레지스트리 등록 (import side-effect)
# ============================================================

register_support_tool(
    SupportToolSpec(
        name="lookup_my_attendance",
        description=(
            "본인의 출석 체크 현황을 조회합니다. "
            "'출석 체크 안 됐어요', '연속 출석 며칠이에요', '오늘 출석했나요', "
            "'이번 달 출석 일수' 같은 출석/리워드 문의에 사용하세요. 로그인 필수."
        ),
        args_schema=LookupAttendanceArgs,
        handler=_handle,
        requires_login=True,
    )
)
