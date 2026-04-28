"""
고객센터 AI 에이전트 v4 — `lookup_my_recent_activity` tool.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.1

Backend EP (2개 병렬 호출):
  - GET /api/v1/users/me/reviews?days=N&page=0&size=20  (Phase 1.3 신규)
    응답: Page<MyReviewSummary> {
            content: [{reviewId, movieId, movieTitle, rating, contentPreview,
                       createdAt, pointAwarded, pointAwardedAt}],
            totalElements, ...
          }
    pointAwardedAt == null → 포인트 미지급 상태
  - GET /api/v1/watch-history
    응답: [{movieId, watchedAt, rating}]

두 EP 를 asyncio.gather 로 병렬 호출한 뒤 {reviews, watchHistory} 로 합쳐 반환.
한쪽이 실패해도 다른 쪽 결과는 그대로 포함 (partial result 허용).

용도: "최근에 본 영화 뭐예요", "리뷰 썼는데 포인트 안 들어왔어요",
      "최근 활동 내역 보여줘"
requires_login=True — 로그인한 사용자의 본인 데이터 조회이므로 게스트 차단.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from . import ToolContext, SupportToolSpec, register_support_tool
from . import _base


# ============================================================
# 입력 스키마
# ============================================================

class LookupRecentActivityArgs(BaseModel):
    """
    `lookup_my_recent_activity` tool 입력 스키마.

    LLM 은 사용자 발화에서 기간(일수)을 추출해 days 에 담는다.
    "이번 달 활동" → 30, 기간 언급 없으면 기본 30일.
    리뷰 조회는 page=0, size=20 고정 (최신 20건).
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
    lookup_my_recent_activity tool 실행 핸들러.

    게스트 차단 후 리뷰 EP + 시청 이력 EP 를 asyncio.gather 로 병렬 호출한다.
    한쪽 실패 시 해당 키에 {"ok": False, "error": ...} 를 담고 다른 쪽은 정상 반환.

    반환 스키마:
        ok=True  → {
            "ok": True,
            "data": {
                "reviews":      <리뷰 Page 응답 또는 {"ok": False, ...}>,
                "watchHistory": <시청 이력 리스트 또는 {"ok": False, ...}>
            }
        }
        ok=False (게스트) → {"ok": False, "error": "login_required", "reason": "..."}

    Narrator 해석 참고:
      - reviews.content[].pointAwardedAt == null → 리뷰 포인트 미지급 상태
      - reviews.content[].pointAwarded: 지급된 포인트 (null 이면 미지급)
      - watchHistory 는 rating 이 null 일 수 있음 (평점 없이 시청만 한 경우)
    """
    if ctx.is_guest:
        return {
            "ok": False,
            "error": "login_required",
            "reason": "최근 활동 내역은 로그인 후 확인할 수 있어요.",
        }

    # 두 EP 를 병렬 호출 — 한쪽 실패가 다른 쪽을 막지 않음
    reviews_result, watch_result = await asyncio.gather(
        _base.call_backend_get(
            ctx,
            "/api/v1/users/me/reviews",
            params={"days": days, "page": 0, "size": 20},
        ),
        _base.call_backend_get(ctx, "/api/v1/watch-history"),
        return_exceptions=False,  # 예외는 call_backend_get 내부에서 ok=False 로 흡수됨
    )

    return {
        "ok": True,
        "data": {
            "reviews": reviews_result,
            "watchHistory": watch_result,
        },
    }


# ============================================================
# 레지스트리 등록 (import side-effect)
# ============================================================

register_support_tool(
    SupportToolSpec(
        name="lookup_my_recent_activity",
        description=(
            "본인의 최근 리뷰 작성 이력과 영화 시청 이력을 함께 조회합니다. "
            "'최근에 본 영화 뭐예요', '리뷰 썼는데 포인트 안 들어왔어요', "
            "'리뷰 포인트 지급됐는지 확인해줘', '최근 활동 내역 보여줘' "
            "같은 활동 이력 관련 문의에 사용하세요. 로그인 필수."
        ),
        args_schema=LookupRecentActivityArgs,
        handler=_handle,
        requires_login=True,
    )
)
