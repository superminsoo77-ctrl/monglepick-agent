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


class PointConsumeResult(BaseModel):
    """
    AI 쿼터 소비 결과 (2026-04-15 신규).

    Backend `POST /api/v1/point/consume` 응답. `movie_card` 발행 시점에 실제 1회 차감된 후
    각 소스별 잔여 카운트와 안내 메시지를 Client 까지 전파한다.

    - allowed: 차감 성공 여부. False 면 check/consume 사이 레이스로 한도 소진된 경우 —
      movie_card 는 이미 생성됐으므로 graceful(SSE error 만 내려보냄)로 처리.
    - source: GRADE_FREE / SUB_BONUS / PURCHASED / BLOCKED
    - daily_used / daily_limit: UI "오늘 N/M" 배너 표시용
    - sub_bonus_remaining / purchased_remaining: 각 소스별 잔여 횟수 (-1 은 미보유)
    """

    allowed: bool
    balance: int = 0
    source: str = "GRADE_FREE"
    message: str = ""
    max_input_length: int = 200
    daily_used: int = 0
    daily_limit: int = 3
    sub_bonus_remaining: int = -1
    purchased_remaining: int = 0


class PointDeductResult(BaseModel):
    """
    포인트 차감 결과. 차감 성공 여부와 변동 후 잔액을 포함한다.

    error_code: Backend의 402 응답 body에서 파싱한 세분화된 에러 코드.
      - INSUFFICIENT_POINT: 포인트 잔액 부족
      - DAILY_LIMIT_EXCEEDED: 일일 AI 추천 사용 한도 초과
      - MONTHLY_LIMIT_EXCEEDED: 월간 AI 추천 사용 한도 초과
      - None: 에러 없음 (차감 성공) 또는 미분류 에러
    error_message: 사용자에게 표시할 안내 메시지 (Backend 응답 message 필드)
    """

    success: bool                       # 차감 성공 여부
    balance_after: int = 0              # 변동 후 잔액
    transaction_id: int | None = None   # 거래 이력 ID (추적용)
    # ── 에러 세분화 필드 (이슈 2: Backend 402 응답 파싱) ──
    error_code: str | None = None       # 세분화된 에러 코드 (SSE error 이벤트에 포함)
    error_message: str = ""             # 사용자 안내 메시지


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
                # [FIX] 타임아웃 확장: 5초 → connect 5초 / read 15초.
                # Backend 콜드 스타트(JPA 스키마 검증, DB 커넥션 풀 워밍업) 시
                # 첫 요청이 5초를 초과할 수 있어 세션 저장이 실패하던 문제 해결.
                timeout=httpx.Timeout(15.0, connect=5.0),
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

        # 4xx/5xx 응답: 쿼터 미확인 상태에서 허용하면 일일 한도가 우회되므로 차단
        logger.warning(
            "point_check_failed",
            user_id=user_id,
            status_code=resp.status_code,
            body=resp.text[:200],
        )
        return PointCheckResult(
            allowed=False, balance=-1, cost=cost,
            message="포인트 시스템 연결에 실패했습니다. 잠시 후 다시 시도해주세요.",
        )

    except Exception as e:
        # 네트워크 오류: 쿼터 미확인 상태에서 허용하면 일일 한도가 우회되므로 차단
        logger.error(
            "point_check_error",
            user_id=user_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return PointCheckResult(
            allowed=False, balance=-1, cost=cost,
            message="포인트 시스템 연결에 실패했습니다. 잠시 후 다시 시도해주세요.",
        )


async def consume_point(user_id: str) -> PointConsumeResult:
    """
    AI 쿼터를 실제로 1회 차감한다 (2026-04-15 신규).

    호출 시점: {@code recommendation_ranker} 완료 → {@code movie_card} 발행 직전.
    이전까지는 {@link check_point} 가 체크 + 차감을 동시에 수행했으나, "추천 완료
    전에 쿼터가 깎여버리는" 정책 버그를 해소하기 위해 분리되었다. 본 함수가 유일한
    쓰기 경로이며, Backend `POST /api/v1/point/consume` 를 호출한다.

    Args:
        user_id: 사용자 ID

    Returns:
        PointConsumeResult: 차감 결과 (allowed, source, daily_used/limit, 잔여 카운트 등)

    Note:
        Backend 응답 실패 시 graceful degradation:
        movie_card 는 이미 만들어진 상태이므로 allowed=True 반환 (추천 노출 유지).
        다음 턴 check 에서 쿼터 상태는 다시 올바르게 조회되므로 일시적 불일치만 발생.
    """
    try:
        client = await _get_http_client()
        resp = await client.post(
            "/api/v1/point/consume",
            json={"userId": user_id, "cost": 0},  # v3.0 AI 무과금: cost 무시 (DTO 호환용)
        )

        if resp.status_code == 200:
            data = resp.json()
            return PointConsumeResult(
                allowed=data.get("allowed", True),
                balance=data.get("balance", 0),
                source=data.get("source", "GRADE_FREE"),
                message=data.get("message", ""),
                max_input_length=data.get("maxInputLength", 200),
                daily_used=data.get("dailyUsed", 0),
                daily_limit=data.get("dailyLimit", 3),
                sub_bonus_remaining=data.get("subBonusRemaining", -1),
                purchased_remaining=data.get("purchasedRemaining", 0),
            )

        logger.error(
            "point_consume_failed",
            user_id=user_id,
            status_code=resp.status_code,
            body=resp.text[:200],
        )
        # movie_card 는 이미 yield 준비 중 — graceful: 노출 유지, 다음 턴에서 정정
        return PointConsumeResult(
            allowed=True, source="GRADE_FREE",
            message="쿼터 차감 일시 오류 (추천은 정상 진행됩니다).",
        )

    except Exception as e:
        logger.error(
            "point_consume_error",
            user_id=user_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return PointConsumeResult(
            allowed=True, source="GRADE_FREE",
            message="쿼터 차감 일시 오류 (추천은 정상 진행됩니다).",
        )


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

        # 402 Payment Required: 잔액 부족 또는 쿼터 초과
        # Backend는 402 응답 body에 error_code 필드로 에러 유형을 세분화한다.
        # 가능한 error_code 값:
        #   INSUFFICIENT_POINT      — 포인트 잔액 부족 (포인트 구매 유도)
        #   DAILY_LIMIT_EXCEEDED    — 일일 AI 추천 사용 한도 초과
        #   MONTHLY_LIMIT_EXCEEDED  — 월간 AI 추천 사용 한도 초과
        # error_code를 PointDeductResult에 담아 chat.py의 SSE error 이벤트로 전달한다.
        if resp.status_code == 402:
            data = resp.json()
            # Backend error_code 파싱: 없으면 잔액 부족으로 간주
            error_code = data.get("errorCode") or data.get("error_code") or "INSUFFICIENT_POINT"
            error_message = data.get("message", "")
            logger.info(
                "point_deduct_insufficient",
                user_id=user_id,
                balance=data.get("balance", 0),
                required=data.get("required", amount),
                error_code=error_code,
                error_message=error_message,
            )
            return PointDeductResult(
                success=False,
                balance_after=data.get("balance", 0),
                error_code=error_code,
                error_message=error_message,
            )

        # 기타 에러 (4xx/5xx)
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
