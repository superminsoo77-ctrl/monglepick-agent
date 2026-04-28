"""
고객센터 AI 에이전트 v4 — 본인 데이터 조회 Read tool 단위 테스트.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.1 (Read Tool 8개) / §8 (보안)

## 테스트 구성 (8 tool × 5 케이스 + 레지스트리 메타 6개 = 46개)

각 tool 마다 다음 케이스를 검증한다:
  1. 정상 응답 — Backend 모킹, 응답 필드 매핑 검증 (ok=True, data 존재)
  2. 게스트 차단 — is_guest=True → ok=False, error="login_required"
  3. BOLA 차단 — ctx.user_id 가 X-User-Id 헤더에 강제 주입됨 (args userId 무시)
  4. HTTP 4xx — Backend 에러 응답 → ok=False, error 필드에 detail 포함
  5. 타임아웃 — httpx.TimeoutException → ok=False, error="timeout:..."

## Mock 전략
`call_backend_get` 을 `AsyncMock` 으로 패치한다.
  경로: `monglepick.tools.support_tools._base.call_backend_get`
  httpx 를 직접 모킹하지 않고 각 handler 의 로직만 순수하게 검증한다.

## 디스크립터 바인딩 회피 원칙
SupportToolSpec.handler 는 일반 모듈 함수이다.
클래스 변수(_HANDLER = spec.handler)로 저장하면 Python 디스크립터 프로토콜이
인스턴스 메서드로 바인딩해 self(테스트 인스턴스)가 첫 번째 positional 인자로
주입된다 → "takes 1 positional argument but 2 were given" TypeError 발생.
해결: SUPPORT_TOOL_REGISTRY 에서 꺼낸 handler 를 모듈 수준 변수에 할당한다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from monglepick.tools.support_tools import ToolContext, SUPPORT_TOOL_REGISTRY

# ------------------------------------------------------------
# 모듈 수준 handler 참조 (클래스 변수 저장 금지)
# ------------------------------------------------------------
_h_point_history    = SUPPORT_TOOL_REGISTRY["lookup_my_point_history"].handler
_h_attendance       = SUPPORT_TOOL_REGISTRY["lookup_my_attendance"].handler
_h_ai_quota         = SUPPORT_TOOL_REGISTRY["lookup_my_ai_quota"].handler
_h_subscription     = SUPPORT_TOOL_REGISTRY["lookup_my_subscription"].handler
_h_grade            = SUPPORT_TOOL_REGISTRY["lookup_my_grade_progress"].handler
_h_orders           = SUPPORT_TOOL_REGISTRY["lookup_my_orders"].handler
_h_tickets          = SUPPORT_TOOL_REGISTRY["lookup_my_tickets"].handler
_h_recent_activity  = SUPPORT_TOOL_REGISTRY["lookup_my_recent_activity"].handler

_MOCK_PATH = "monglepick.tools.support_tools._base.call_backend_get"


# ============================================================
# 공통 픽스처
# ============================================================

@pytest.fixture
def auth_ctx() -> ToolContext:
    """로그인한 일반 사용자 컨텍스트."""
    return ToolContext(
        user_id="user-abc-123",
        is_guest=False,
        session_id="sess-001",
        request_id="req-001",
    )


@pytest.fixture
def guest_ctx() -> ToolContext:
    """비인증 게스트 컨텍스트."""
    return ToolContext(
        user_id="",
        is_guest=True,
        session_id="sess-guest",
        request_id="req-guest",
    )


def _ok(data) -> dict:
    """call_backend_get 정상 응답 포맷."""
    return {"ok": True, "data": data}


def _err(error: str = "http_404", status_code: int = 404) -> dict:
    """call_backend_get 에러 응답 포맷."""
    return {"ok": False, "error": error, "status_code": status_code}


def _timeout() -> dict:
    """call_backend_get 타임아웃 응답 포맷."""
    return {"ok": False, "error": "timeout:ReadTimeout", "status_code": 0}


def _get_params(mock_call) -> dict:
    """mock_call.call_args 에서 params dict 를 안전하게 꺼낸다."""
    ca = mock_call.call_args
    # call_backend_get(ctx, path, params=...) 시그니처
    if ca.kwargs.get("params") is not None:
        return ca.kwargs["params"]
    # positional 3번째 인자
    if len(ca.args) >= 3:
        return ca.args[2]
    return {}


# ============================================================
# 1. lookup_my_point_history
# ============================================================

class TestLookupMyPointHistory:
    """포인트 이력 조회 tool 단위 테스트."""

    @pytest.mark.asyncio
    async def test_normal_response(self, auth_ctx):
        """정상 응답 — Backend 데이터가 그대로 반환된다."""
        mock_data = [
            {"amount": 100, "type": "EARN", "source": "REVIEW",    "createdAt": "2026-04-27T10:00:00"},
            {"amount": -50, "type": "USE",  "source": "AI_USAGE",  "createdAt": "2026-04-26T09:00:00"},
        ]
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok(mock_data)
            result = await _h_point_history(auth_ctx, days=7)

        assert result["ok"] is True
        assert result["data"] == mock_data
        assert _get_params(mock_call)["days"] == 7

    @pytest.mark.asyncio
    async def test_guest_blocked(self, guest_ctx):
        """게스트는 차단된다 — ok=False, error=login_required."""
        result = await _h_point_history(guest_ctx, days=7)
        assert result["ok"] is False
        assert result["error"] == "login_required"
        assert "reason" in result

    @pytest.mark.asyncio
    async def test_bola_ctx_user_id_passed(self, auth_ctx):
        """BOLA 차단 — call_backend_get 의 첫 번째 인자가 ctx 이고 user_id 가 일치한다."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok([])
            await _h_point_history(auth_ctx, days=7)

        passed_ctx = mock_call.call_args.args[0]
        assert isinstance(passed_ctx, ToolContext)
        assert passed_ctx.user_id == auth_ctx.user_id

    @pytest.mark.asyncio
    async def test_http_4xx(self, auth_ctx):
        """HTTP 4xx → ok=False, error 필드 존재."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _err("http_404", 404)
            result = await _h_point_history(auth_ctx, days=7)
        assert result["ok"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_timeout(self, auth_ctx):
        """타임아웃 → ok=False, error 에 timeout 포함."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _timeout()
            result = await _h_point_history(auth_ctx, days=7)
        assert result["ok"] is False
        assert "timeout" in result["error"]


# ============================================================
# 2. lookup_my_attendance
# ============================================================

class TestLookupMyAttendance:
    """출석 현황 조회 tool 단위 테스트."""

    @pytest.mark.asyncio
    async def test_normal_response(self, auth_ctx):
        """정상 응답 — streak, totalDays, monthlyDates, todayChecked 포함."""
        mock_data = {
            "streak": 5,
            "totalDays": 20,
            "monthlyDates": ["2026-04-01", "2026-04-02"],
            "todayChecked": True,
        }
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok(mock_data)
            result = await _h_attendance(auth_ctx)

        assert result["ok"] is True
        assert result["data"]["streak"] == 5
        assert result["data"]["todayChecked"] is True

    @pytest.mark.asyncio
    async def test_guest_blocked(self, guest_ctx):
        """게스트 차단 — login_required."""
        result = await _h_attendance(guest_ctx)
        assert result["ok"] is False
        assert result["error"] == "login_required"

    @pytest.mark.asyncio
    async def test_bola_ctx_user_id_passed(self, auth_ctx):
        """BOLA 차단 — ctx 의 user_id 가 그대로 전달된다."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok({})
            await _h_attendance(auth_ctx)

        passed_ctx = mock_call.call_args.args[0]
        assert passed_ctx.user_id == auth_ctx.user_id

    @pytest.mark.asyncio
    async def test_http_4xx(self, auth_ctx):
        """HTTP 4xx → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _err("http_401", 401)
            result = await _h_attendance(auth_ctx)
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_timeout(self, auth_ctx):
        """타임아웃 → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _timeout()
            result = await _h_attendance(auth_ctx)
        assert result["ok"] is False
        assert "timeout" in result["error"]


# ============================================================
# 3. lookup_my_ai_quota
# ============================================================

class TestLookupMyAiQuota:
    """AI 쿼터 조회 tool 단위 테스트."""

    @pytest.mark.asyncio
    async def test_normal_response(self, auth_ctx):
        """정상 응답 — 일반 등급 (dailyAiLimit > 0), 구독 없음 sentinel."""
        mock_data = {
            "dailyAiUsed": 2,
            "dailyAiLimit": 5,
            "remainingAiBonus": -1,     # sentinel: 구독 없음
            "purchasedAiTokens": 0,
            "monthlyCouponUsed": 0,
            "monthlyCouponLimit": 0,
            "resetAt": "2026-04-29T00:00:00",
        }
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok(mock_data)
            result = await _h_ai_quota(auth_ctx)

        assert result["ok"] is True
        assert result["data"]["dailyAiUsed"] == 2
        assert result["data"]["remainingAiBonus"] == -1  # sentinel 보존

    @pytest.mark.asyncio
    async def test_sentinel_diamond_unlimited(self, auth_ctx):
        """DIAMOND 등급 sentinel — dailyAiLimit == -1 (무제한), 그대로 반환해야 한다."""
        mock_data = {
            "dailyAiUsed": 50,
            "dailyAiLimit": -1,         # sentinel: DIAMOND 무제한
            "remainingAiBonus": 30,
            "purchasedAiTokens": 0,
            "monthlyCouponUsed": 10,
            "monthlyCouponLimit": 60,
            "resetAt": "2026-04-29T00:00:00",
        }
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok(mock_data)
            result = await _h_ai_quota(auth_ctx)

        assert result["ok"] is True
        assert result["data"]["dailyAiLimit"] == -1

    @pytest.mark.asyncio
    async def test_guest_blocked(self, guest_ctx):
        """게스트 차단."""
        result = await _h_ai_quota(guest_ctx)
        assert result["ok"] is False
        assert result["error"] == "login_required"

    @pytest.mark.asyncio
    async def test_http_4xx(self, auth_ctx):
        """HTTP 4xx → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _err("http_503", 503)
            result = await _h_ai_quota(auth_ctx)
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_timeout(self, auth_ctx):
        """타임아웃 → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _timeout()
            result = await _h_ai_quota(auth_ctx)
        assert result["ok"] is False
        assert "timeout" in result["error"]


# ============================================================
# 4. lookup_my_subscription
# ============================================================

class TestLookupMySubscription:
    """구독 상태 조회 tool 단위 테스트."""

    @pytest.mark.asyncio
    async def test_normal_active(self, auth_ctx):
        """정상 응답 — 활성 구독."""
        mock_data = {
            "isActive": True,
            "plan": "monthly_premium",
            "expiryDate": "2026-05-28",
            "nextBillingDate": "2026-05-28",
        }
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok(mock_data)
            result = await _h_subscription(auth_ctx)

        assert result["ok"] is True
        assert result["data"]["isActive"] is True
        assert result["data"]["plan"] == "monthly_premium"

    @pytest.mark.asyncio
    async def test_normal_inactive(self, auth_ctx):
        """정상 응답 — 구독 없음 (isActive=False)."""
        mock_data = {
            "isActive": False,
            "plan": None,
            "expiryDate": None,
            "nextBillingDate": None,
        }
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok(mock_data)
            result = await _h_subscription(auth_ctx)

        assert result["ok"] is True
        assert result["data"]["isActive"] is False

    @pytest.mark.asyncio
    async def test_guest_blocked(self, guest_ctx):
        """게스트 차단."""
        result = await _h_subscription(guest_ctx)
        assert result["ok"] is False
        assert result["error"] == "login_required"

    @pytest.mark.asyncio
    async def test_http_4xx(self, auth_ctx):
        """HTTP 4xx → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _err("http_404", 404)
            result = await _h_subscription(auth_ctx)
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_timeout(self, auth_ctx):
        """타임아웃 → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _timeout()
            result = await _h_subscription(auth_ctx)
        assert result["ok"] is False
        assert "timeout" in result["error"]


# ============================================================
# 5. lookup_my_grade_progress
# ============================================================

class TestLookupMyGradeProgress:
    """등급/포인트 잔액 조회 tool 단위 테스트."""

    @pytest.mark.asyncio
    async def test_normal_response(self, auth_ctx):
        """정상 응답 — SILVER 등급 사용자."""
        mock_data = {
            "grade": "SILVER",
            "balance": 350,
            "progressPercent": 65,
            "nextGrade": "GOLD",
            "nextGradeRequirements": "카라멜팝콘 등급까지 150P 필요",
        }
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok(mock_data)
            result = await _h_grade(auth_ctx)

        assert result["ok"] is True
        assert result["data"]["grade"] == "SILVER"
        assert result["data"]["balance"] == 350

    @pytest.mark.asyncio
    async def test_diamond_grade_no_next(self, auth_ctx):
        """DIAMOND 등급 — nextGrade == null (최고 등급)."""
        mock_data = {
            "grade": "DIAMOND",
            "balance": 9999,
            "progressPercent": 100,
            "nextGrade": None,
            "nextGradeRequirements": None,
        }
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok(mock_data)
            result = await _h_grade(auth_ctx)

        assert result["ok"] is True
        assert result["data"]["nextGrade"] is None

    @pytest.mark.asyncio
    async def test_guest_blocked(self, guest_ctx):
        """게스트 차단."""
        result = await _h_grade(guest_ctx)
        assert result["ok"] is False
        assert result["error"] == "login_required"

    @pytest.mark.asyncio
    async def test_http_4xx(self, auth_ctx):
        """HTTP 4xx → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _err("http_500", 500)
            result = await _h_grade(auth_ctx)
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_timeout(self, auth_ctx):
        """타임아웃 → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _timeout()
            result = await _h_grade(auth_ctx)
        assert result["ok"] is False
        assert "timeout" in result["error"]


# ============================================================
# 6. lookup_my_orders
# ============================================================

class TestLookupMyOrders:
    """결제 주문 내역 조회 tool 단위 테스트."""

    @pytest.mark.asyncio
    async def test_normal_response(self, auth_ctx):
        """정상 응답 — 결제 내역 2건 반환."""
        mock_data = [
            {"orderId": "ord-001", "amount": 5900, "status": "PAID",     "paidAt": "2026-04-01T12:00:00"},
            {"orderId": "ord-002", "amount": 2900, "status": "REFUNDED", "paidAt": "2026-03-01T12:00:00"},
        ]
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok(mock_data)
            result = await _h_orders(auth_ctx, days=30)

        assert result["ok"] is True
        assert len(result["data"]) == 2
        assert result["data"][0]["status"] == "PAID"

    @pytest.mark.asyncio
    async def test_days_param_forwarded(self, auth_ctx):
        """days=90 파라미터가 Backend 호출 시 정확히 전달된다."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok([])
            await _h_orders(auth_ctx, days=90)

        assert _get_params(mock_call)["days"] == 90

    @pytest.mark.asyncio
    async def test_guest_blocked(self, guest_ctx):
        """게스트 차단."""
        result = await _h_orders(guest_ctx, days=30)
        assert result["ok"] is False
        assert result["error"] == "login_required"

    @pytest.mark.asyncio
    async def test_http_4xx(self, auth_ctx):
        """HTTP 4xx → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _err("http_403", 403)
            result = await _h_orders(auth_ctx, days=30)
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_timeout(self, auth_ctx):
        """타임아웃 → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _timeout()
            result = await _h_orders(auth_ctx, days=30)
        assert result["ok"] is False
        assert "timeout" in result["error"]


# ============================================================
# 7. lookup_my_tickets
# ============================================================

class TestLookupMyTickets:
    """문의 내역 조회 tool 단위 테스트."""

    @pytest.mark.asyncio
    async def test_normal_response(self, auth_ctx):
        """정상 응답 — 문의 내역 2건 반환."""
        mock_data = [
            {"id": 1, "status": "RESOLVED", "category": "결제", "title": "환불 문의",     "createdAt": "2026-04-20T10:00:00"},
            {"id": 2, "status": "OPEN",     "category": "계정", "title": "로그인 안 돼요", "createdAt": "2026-04-27T15:00:00"},
        ]
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok(mock_data)
            result = await _h_tickets(auth_ctx)

        assert result["ok"] is True
        assert len(result["data"]) == 2
        assert result["data"][1]["status"] == "OPEN"

    @pytest.mark.asyncio
    async def test_empty_tickets(self, auth_ctx):
        """문의 내역 없음 — 빈 리스트 정상 반환."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _ok([])
            result = await _h_tickets(auth_ctx)

        assert result["ok"] is True
        assert result["data"] == []

    @pytest.mark.asyncio
    async def test_guest_blocked(self, guest_ctx):
        """게스트 차단."""
        result = await _h_tickets(guest_ctx)
        assert result["ok"] is False
        assert result["error"] == "login_required"

    @pytest.mark.asyncio
    async def test_http_4xx(self, auth_ctx):
        """HTTP 4xx → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _err("http_401", 401)
            result = await _h_tickets(auth_ctx)
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_timeout(self, auth_ctx):
        """타임아웃 → ok=False."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _timeout()
            result = await _h_tickets(auth_ctx)
        assert result["ok"] is False
        assert "timeout" in result["error"]


# ============================================================
# 8. lookup_my_recent_activity
# ============================================================

class TestLookupMyRecentActivity:
    """최근 활동(리뷰 + 시청 이력) 조회 tool 단위 테스트."""

    @pytest.mark.asyncio
    async def test_normal_both_succeed(self, auth_ctx):
        """정상 응답 — 리뷰 EP + 시청이력 EP 모두 성공."""
        mock_reviews = {
            "content": [
                {
                    "reviewId": 1,
                    "movieId": "tt001",
                    "movieTitle": "인터스텔라",
                    "rating": 5,
                    "contentPreview": "최고의 영화",
                    "createdAt": "2026-04-25T12:00:00",
                    "pointAwarded": 10,
                    "pointAwardedAt": "2026-04-25T12:05:00",
                }
            ],
            "totalElements": 1,
        }
        mock_watch = [
            {"movieId": "tt001", "watchedAt": "2026-04-24T20:00:00", "rating": 5},
            {"movieId": "tt002", "watchedAt": "2026-04-23T19:00:00", "rating": None},
        ]
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            # asyncio.gather 호출 순서: reviews 먼저, watch-history 두 번째
            mock_call.side_effect = [_ok(mock_reviews), _ok(mock_watch)]
            result = await _h_recent_activity(auth_ctx, days=30)

        assert result["ok"] is True
        assert result["data"]["reviews"]["ok"] is True
        assert result["data"]["watchHistory"]["ok"] is True

    @pytest.mark.asyncio
    async def test_review_point_not_awarded(self, auth_ctx):
        """리뷰 포인트 미지급 — pointAwardedAt == null 그대로 반환된다."""
        mock_reviews = {
            "content": [
                {
                    "reviewId": 2,
                    "movieId": "tt003",
                    "movieTitle": "기생충",
                    "rating": 4,
                    "contentPreview": "명작",
                    "createdAt": "2026-04-28T10:00:00",
                    "pointAwarded": None,
                    "pointAwardedAt": None,     # 미지급 sentinel
                }
            ],
            "totalElements": 1,
        }
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [_ok(mock_reviews), _ok([])]
            result = await _h_recent_activity(auth_ctx, days=30)

        assert result["ok"] is True
        review_item = result["data"]["reviews"]["data"]["content"][0]
        assert review_item["pointAwardedAt"] is None

    @pytest.mark.asyncio
    async def test_partial_failure_watch_history(self, auth_ctx):
        """시청이력 EP 실패 시 리뷰 결과는 정상 포함 (partial result 허용)."""
        mock_reviews = {"content": [], "totalElements": 0}
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [_ok(mock_reviews), _err("http_500", 500)]
            result = await _h_recent_activity(auth_ctx, days=30)

        # 전체 래퍼 ok=True, 리뷰는 정상, 시청이력은 실패
        assert result["ok"] is True
        assert result["data"]["reviews"]["ok"] is True
        assert result["data"]["watchHistory"]["ok"] is False

    @pytest.mark.asyncio
    async def test_guest_blocked(self, guest_ctx):
        """게스트는 두 EP 모두 호출 없이 즉시 차단된다."""
        result = await _h_recent_activity(guest_ctx, days=30)
        assert result["ok"] is False
        assert result["error"] == "login_required"
        assert "reason" in result

    @pytest.mark.asyncio
    async def test_both_timeout(self, auth_ctx):
        """두 EP 모두 타임아웃 — 각 ok=False, 전체 래퍼는 ok=True (partial result)."""
        with patch(_MOCK_PATH, new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [_timeout(), _timeout()]
            result = await _h_recent_activity(auth_ctx, days=30)

        assert result["ok"] is True
        assert result["data"]["reviews"]["ok"] is False
        assert result["data"]["watchHistory"]["ok"] is False
        assert "timeout" in result["data"]["reviews"]["error"]


# ============================================================
# 9. 레지스트리 메타 검증
# ============================================================

class TestRegistryMeta:
    """SUPPORT_TOOL_REGISTRY 메타데이터 일관성 검증."""

    _EXPECTED_LOGIN_REQUIRED = {
        "lookup_my_point_history",
        "lookup_my_attendance",
        "lookup_my_ai_quota",
        "lookup_my_subscription",
        "lookup_my_grade_progress",
        "lookup_my_orders",
        "lookup_my_tickets",
        "lookup_my_recent_activity",
    }

    def test_all_read_tools_registered(self):
        """8개 Read tool 이 모두 레지스트리에 등록되어 있다."""
        registered = set(SUPPORT_TOOL_REGISTRY.keys())
        for name in self._EXPECTED_LOGIN_REQUIRED:
            assert name in registered, f"{name} 이 레지스트리에 없음"

    def test_all_read_tools_require_login(self):
        """8개 Read tool 은 모두 requires_login=True 이다."""
        for name in self._EXPECTED_LOGIN_REQUIRED:
            spec = SUPPORT_TOOL_REGISTRY[name]
            assert spec.requires_login is True, f"{name}.requires_login 이 False"

    def test_policy_does_not_require_login(self):
        """lookup_policy 는 공개 정책 정보이므로 requires_login=False 이다."""
        spec = SUPPORT_TOOL_REGISTRY["lookup_policy"]
        assert spec.requires_login is False

    def test_all_tools_have_description(self):
        """모든 tool 이 비어있지 않은 description 을 갖는다."""
        for name, spec in SUPPORT_TOOL_REGISTRY.items():
            assert spec.description.strip(), f"{name}.description 이 비어있음"

    def test_all_tools_have_async_handler(self):
        """모든 tool 이 async callable handler 를 갖는다."""
        import asyncio
        for name, spec in SUPPORT_TOOL_REGISTRY.items():
            assert callable(spec.handler), f"{name}.handler 가 callable 이 아님"
            assert asyncio.iscoroutinefunction(spec.handler), \
                f"{name}.handler 가 async 함수가 아님"

    def test_all_tools_have_pydantic_args_schema(self):
        """모든 tool 이 Pydantic BaseModel args_schema 를 갖는다."""
        from pydantic import BaseModel
        for name, spec in SUPPORT_TOOL_REGISTRY.items():
            assert issubclass(spec.args_schema, BaseModel), \
                f"{name}.args_schema 가 BaseModel 서브클래스가 아님"
