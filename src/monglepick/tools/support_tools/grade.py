"""
고객센터 AI 에이전트 v4 — `lookup_my_grade_progress` tool.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.1

Backend EP: GET /api/v1/point/balance
응답: {grade, balance, progressPercent, nextGrade, nextGradeRequirements}  (camelCase)

등급 6단계: NORMAL(알갱이) → BRONZE(강냉이) → SILVER(팝콘) →
            GOLD(카라멜팝콘) → PLATINUM(몽글팝콘) → DIAMOND(몽아일체)

용도: "내 등급이 뭐예요", "다음 등급 되려면 얼마나 남았어요", "포인트 잔액 얼마예요"
requires_login=True — 로그인한 사용자의 본인 등급·잔액 조회이므로 게스트 차단.
"""

from __future__ import annotations

from pydantic import BaseModel

from . import ToolContext, SupportToolSpec, register_support_tool
from . import _base


# ============================================================
# 입력 스키마 (인자 없음 — 항상 현재 사용자 기준)
# ============================================================

class LookupGradeProgressArgs(BaseModel):
    """
    `lookup_my_grade_progress` tool 입력 스키마.

    등급/잔액 정보는 "현재 로그인한 사용자" 기준이므로 추가 인자가 없다.
    """


# ============================================================
# Handler
# ============================================================

async def _handle(ctx: ToolContext) -> dict:
    """
    lookup_my_grade_progress tool 실행 핸들러.

    게스트 차단 후 Backend GET /api/v1/point/balance 를 호출한다.

    반환 스키마:
        ok=True  → {"ok": True, "data": {grade, balance, progressPercent,
                                         nextGrade, nextGradeRequirements}}
        ok=False → {"ok": False, "error": "<사유>"}

    Narrator 해석 참고:
      - grade: NORMAL / BRONZE / SILVER / GOLD / PLATINUM / DIAMOND
      - progressPercent: 0~100, 현재 등급 내 진행률
      - nextGrade: null 이면 최고 등급(DIAMOND)
      - nextGradeRequirements: 다음 등급까지 필요한 조건 설명 문자열
    """
    if ctx.is_guest:
        return {
            "ok": False,
            "error": "login_required",
            "reason": "등급 및 포인트 정보는 로그인 후 확인할 수 있어요.",
        }
    return await _base.call_backend_get(ctx, "/api/v1/point/balance")


# ============================================================
# 레지스트리 등록 (import side-effect)
# ============================================================

register_support_tool(
    SupportToolSpec(
        name="lookup_my_grade_progress",
        description=(
            "본인의 현재 등급, 포인트 잔액, 다음 등급까지의 진행률을 조회합니다. "
            "'내 등급이 뭐예요', '브론즈 되려면 얼마나 남았어요', "
            "'포인트 잔액 얼마예요', '등급 혜택 확인하고 싶어요' 같은 "
            "등급/포인트 잔액 문의에 사용하세요. 로그인 필수."
        ),
        args_schema=LookupGradeProgressArgs,
        handler=_handle,
        requires_login=True,
    )
)
