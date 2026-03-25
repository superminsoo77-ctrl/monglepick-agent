"""
Backend 포인트 API 비동기 클라이언트.

Agent가 추천 완료(movie_card 발행) 시점에 포인트 차감을 요청하고,
그래프 실행 전 포인트 사전 체크를 수행한다.

과금 단위: "추천 완료" (movie_card 발행 시점에만 1회 차감)
- AI의 후속 질문(clarification) 턴은 과금하지 않음
- recommendation_ranker 완료 → movie_card 발행 직전에 deduct 호출

Backend 통신: httpx.AsyncClient (내부 HTTP, 같은 VM/네트워크)
인증: X-Service-Key 헤더 (settings.SERVICE_API_KEY)
"""

from __future__ import annotations

import asyncio

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

# ── 응답 모델 ──


class PointCheckResult(BaseModel):
    """
    포인트 사전 체크 결과. Agent가 그래프 실행 여부를 결정할 때 사용한다.

    Phase R-3에서 Backend에 추가된 등급별 쿼터 필드를 포함한다.
    - max_input_length: 등급별 사용자 입력 글자수 제한 (BRONZE=200, SILVER=300, ...)
    - daily_used / daily_limit: 일일 AI 추천 사용 횟수 / 한도 (-1이면 무제한)
    - monthly_used / monthly_limit: 월간 AI 추천 사용 횟수 / 한도
    - free_remaining: 오늘 무료 잔여 횟수 (등급별 무료 횟수 - daily_used)
    - effective_cost: 실제 차감 포인트 (무료 잔여가 있으면 0, 없으면 cost)
    """

    allowed: bool       # 진행 가능 여부 (잔액 >= cost)
    balance: int        # 현재 잔액
    cost: int           # 필요 포인트
    message: str = ""   # 부족 시 사용자 안내 메시지
    # ── 쿼터 필드 (Phase R-3 Backend에서 추가됨) ──
    max_input_length: int = 200       # 등급별 사용자 입력 글자수 제한
    daily_used: int = 0               # 오늘 AI 추천 사용 횟수
    daily_limit: int = 3              # 일일 한도 (-1이면 무제한)
    monthly_used: int = 0             # 이번 달 사용 횟수
    monthly_limit: int = 30           # 월간 한도
    free_remaining: int = 0           # 오늘 무료 잔여 횟수
    effective_cost: int = 0           # 실제 차감 포인트 (무료면 0)


class PointDeductResult(BaseModel):
    """포인트 차감 결과. 차감 성공 여부와 변동 후 잔액을 포함한다."""

    success: bool           # 차감 성공 여부
    balance_after: int = 0  # 변동 후 잔액
    transaction_id: int | None = None  # 거래 이력 ID (추적용)


# ── 싱글턴 클라이언트 ──

# httpx.AsyncClient는 앱 수명 동안 한 번만 생성하여 커넥션 풀을 재사용한다.
_client = None
# 싱글턴 동시 생성 방지용 Lock (W-5)
_client_lock = asyncio.Lock()


async def _get_http_client():
    """httpx.AsyncClient 싱글턴을 반환한다. Lock으로 동시 생성을 방지한다."""
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        # double-check: Lock 대기 중 다른 코루틴이 이미 생성했을 수 있음
        if _client is None:
            import httpx
            from monglepick.config import settings

            _client = httpx.AsyncClient(
                base_url=settings.BACKEND_BASE_URL,
                timeout=httpx.Timeout(5.0),  # 내부 통신 5초 타임아웃
                headers={"X-Service-Key": settings.SERVICE_API_KEY},
            )
    return _client


async def close_client():
    """앱 종료 시 httpx 클라이언트를 정리한다. (C-2)"""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def check_point(user_id: str, cost: int = 1) -> PointCheckResult:
    """
    포인트 잔액이 충분한지 사전 확인한다 (실제 차감 없음).

    Agent가 그래프 실행 전 호출하여 잔액 부족 시 그래프를 실행하지 않고
    SSE error 이벤트(INSUFFICIENT_POINT)를 즉시 반환한다.

    Args:
        user_id: 사용자 ID
        cost: 필요 포인트 (기본: 1)

    Returns:
        PointCheckResult: 진행 가능 여부 + 잔액 정보

    Note:
        Backend 응답 실패 시 graceful degradation:
        allowed=True를 반환하여 추천은 정상 진행하되 로그를 기록한다.
    """
    try:
        client = await _get_http_client()
        resp = await client.post(
            "/api/v1/point/check",
            json={"userId": user_id, "cost": cost},
        )

        if resp.status_code == 200:
            data = resp.json()
            # Backend JSON 응답은 camelCase → Python snake_case로 매핑.
            # data.get()으로 안전하게 읽어 Backend가 아직 쿼터 필드를
            # 반환하지 않는 경우에도 기본값으로 정상 동작한다.
            return PointCheckResult(
                allowed=data.get("allowed", False),
                balance=data.get("balance", 0),
                cost=data.get("cost", cost),
                message=data.get("message", ""),
                max_input_length=data.get("maxInputLength", 200),
                daily_used=data.get("dailyUsed", 0),
                daily_limit=data.get("dailyLimit", 3),
                monthly_used=data.get("monthlyUsed", 0),
                monthly_limit=data.get("monthlyLimit", 30),
                free_remaining=data.get("freeRemaining", 0),
                effective_cost=data.get("effectiveCost", 0),
            )

        # 4xx/5xx 응답: 로그 후 graceful degradation
        logger.warning(
            "point_check_failed",
            user_id=user_id,
            status_code=resp.status_code,
            body=resp.text[:200],
        )
        return PointCheckResult(allowed=True, balance=-1, cost=cost, message="")

    except Exception as e:
        # 네트워크 오류: 로그 후 graceful degradation (추천은 진행)
        logger.error(
            "point_check_error",
            user_id=user_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return PointCheckResult(allowed=True, balance=-1, cost=cost, message="")


async def deduct_point(
    user_id: str,
    session_id: str,
    amount: int = 1,
    description: str = "AI 추천 사용",
) -> PointDeductResult:
    """
    포인트를 차감한다.

    호출 시점: recommendation_ranker 완료 → movie_card 발행 직전.
    이 시점에서만 차감되므로 AI의 후속 질문 턴은 과금되지 않는다.

    Args:
        user_id: 사용자 ID
        session_id: 채팅 세션 ID (거래 이력 추적용)
        amount: 차감할 포인트 (기본: 1)
        description: 차감 사유 설명

    Returns:
        PointDeductResult: 차감 성공 여부 + 변동 후 잔액

    Note:
        Backend 응답 실패 시 graceful degradation:
        success=False를 반환하지만 추천은 정상 전달한다.
        미차감 이벤트는 로그에 기록되어 사후 정산 가능하다.
    """
    try:
        client = await _get_http_client()
        resp = await client.post(
            "/api/v1/point/deduct",
            json={
                "userId": user_id,
                "amount": amount,
                "sessionId": session_id,
                "description": description,
            },
        )

        if resp.status_code == 200:
            data = resp.json()
            return PointDeductResult(
                success=data.get("success", False),
                balance_after=data.get("balanceAfter", 0),
                transaction_id=data.get("transactionId"),
            )

        # 402 Payment Required: 잔액 부족
        if resp.status_code == 402:
            data = resp.json()
            logger.info(
                "point_deduct_insufficient",
                user_id=user_id,
                balance=data.get("balance", 0),
                required=data.get("required", amount),
            )
            return PointDeductResult(success=False, balance_after=data.get("balance", 0))

        # 기타 에러
        logger.warning(
            "point_deduct_failed",
            user_id=user_id,
            session_id=session_id,
            status_code=resp.status_code,
            body=resp.text[:200],
        )
        return PointDeductResult(success=False)

    except Exception as e:
        # 네트워크 오류: 로그 기록 (사후 정산용)
        logger.error(
            "point_deduct_error",
            user_id=user_id,
            session_id=session_id,
            amount=amount,
            error=str(e),
            error_type=type(e).__name__,
        )
        return PointDeductResult(success=False)


async def get_balance(user_id: str) -> int:
    """
    사용자의 포인트 잔액을 조회한다.

    Args:
        user_id: 사용자 ID

    Returns:
        포인트 잔액 (오류 시 -1)
    """
    try:
        client = await _get_http_client()
        resp = await client.get(
            "/api/v1/point/balance",
            params={"userId": user_id},
        )

        if resp.status_code == 200:
            return resp.json().get("balance", 0)

        return -1

    except Exception as e:
        logger.error("point_balance_error", user_id=user_id, error=str(e))
        return -1
