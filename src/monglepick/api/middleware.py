"""
FastAPI 전역 미들웨어 모음.

Phase 5: FastAPI 최적화 — 전역 Rate Limiting + 요청 타임아웃

1. RateLimitMiddleware
   - Redis Sorted Set 슬라이딩 윈도우 방식 (기존 chat.py 이미지 rate limit과 동일 패턴)
   - IP 기반(비인증) 또는 user_id 기반(인증 사용자) 키 분리
   - 비인증: 분당 RATE_LIMIT_ANON_RPM 회 (기본 30)
   - 인증:   분당 RATE_LIMIT_AUTH_RPM 회 (기본 60)
   - Redis 연결 실패 시 rate limit 무시 (서비스 가용성 우선)
   - /health, /docs, /redoc, /openapi.json, /api/v1/ping 헬스체크 경로 제외

2. TimeoutMiddleware
   - asyncio.wait_for()로 요청 처리에 타임아웃 적용
   - SSE 엔드포인트(/api/v1/chat, /api/v1/match): TIMEOUT_SSE_SEC 초 (기본 300)
   - 일반 엔드포인트: TIMEOUT_DEFAULT_SEC 초 (기본 30)
   - 타임아웃 초과 시 504 Gateway Timeout 반환
   - StreamingResponse(SSE)는 첫 청크 전달 이후 타임아웃 우회됨 — 연결 자체는 유지

주의:
- TimeoutMiddleware는 Starlette의 일반 요청에 적용된다.
  SSE(StreamingResponse)는 응답 스트리밍 자체에는 타임아웃이 적용되지 않으므로
  그래프 노드별 개별 타임아웃(asyncio.wait_for)은 별도 유지해야 한다.
- RateLimitMiddleware는 SSE/sync/upload 모두에 적용된다.
  이미지 업로드(chat.py의 _check_upload_rate_limit)는 별도 per-IP 제한이므로
  중복 적용이지만 레이어 분리를 위해 양쪽 모두 유지한다.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from monglepick.config import settings

logger = structlog.get_logger()


# ============================================================
# 상수 정의
# ============================================================

# Rate Limit Redis 키 접두사
# "rate_limit:api:ip:{ip}" 또는 "rate_limit:api:user:{user_id}" 형태
_RATE_LIMIT_KEY_PREFIX_IP: str = "rate_limit:api:ip:"
_RATE_LIMIT_KEY_PREFIX_USER: str = "rate_limit:api:user:"

# 슬라이딩 윈도우 크기 (초)
_RATE_LIMIT_WINDOW_SEC: int = 60

# Authorization 헤더에서 user_id를 추출하기 위한 JWT 파싱 없이
# 헤더 존재 여부만 확인하여 인증/비인증을 구분한다.
# (정확한 JWT 파싱은 chat.py의 _extract_user_id_from_jwt에서 수행)
_BEARER_PREFIX: str = "Bearer "

# Rate Limit 제외 경로 집합 — 헬스체크, 문서, 핑은 rate limit 대상에서 제외
_EXEMPT_PATHS: frozenset[str] = frozenset(
    [
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/ping",
    ]
)

# SSE 엔드포인트 접두사 — 긴 타임아웃 적용 대상
_SSE_PATH_PREFIXES: tuple[str, ...] = (
    "/api/v1/chat",
    "/api/v1/match",
)


# ============================================================
# 헬퍼: 클라이언트 IP 추출
# ============================================================

def _get_client_ip(request: Request) -> str:
    """
    요청 객체에서 클라이언트 IP를 추출한다.

    Nginx 리버스 프록시 환경에서는 X-Forwarded-For 헤더에 실제 IP가 담긴다.
    X-Forwarded-For가 없으면 TCP 연결 주소(request.client.host)를 사용한다.

    여러 프록시를 거친 경우 X-Forwarded-For 값이 쉼표로 구분된 IP 목록이므로
    가장 첫 번째(클라이언트에 가장 가까운) IP를 사용한다.

    Args:
        request: Starlette Request 객체

    Returns:
        클라이언트 IP 문자열. 알 수 없으면 "unknown" 반환.
    """
    # X-Forwarded-For: client, proxy1, proxy2, ...
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        # 첫 번째 IP = 실제 클라이언트 IP
        return forwarded_for.split(",")[0].strip()

    # X-Real-IP: Nginx에서 설정하는 경우
    real_ip = request.headers.get("X-Real-IP", "")
    if real_ip:
        return real_ip.strip()

    # TCP 연결 직접 주소 (로컬 개발 환경)
    if request.client:
        return request.client.host

    return "unknown"


def _get_rate_limit_key(request: Request) -> tuple[str, int]:
    """
    Rate Limit Redis 키와 적용할 한도(RPM)를 결정한다.

    Authorization: Bearer {token} 헤더가 있으면 인증 사용자로 판단한다.
    인증 사용자는 헤더 값을 해시하여 user별 키를 생성한다.
    (JWT 파싱 없이 헤더 존재 + 값 기반 식별 — 정확한 user_id 추출은 엔드포인트에서)

    Args:
        request: Starlette Request 객체

    Returns:
        (redis_key, rpm_limit) 튜플
    """
    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith(_BEARER_PREFIX):
        # 인증 사용자: 토큰 값을 그대로 키로 사용 (SHA-256 해시는 비용이 있으므로
        # Redis 키 길이 제한을 고려해 앞 32자만 사용)
        token_prefix = auth_header[len(_BEARER_PREFIX):len(_BEARER_PREFIX) + 32]
        key = f"{_RATE_LIMIT_KEY_PREFIX_USER}{token_prefix}"
        rpm = settings.RATE_LIMIT_AUTH_RPM
    else:
        # 비인증 사용자: IP 기반 키
        client_ip = _get_client_ip(request)
        key = f"{_RATE_LIMIT_KEY_PREFIX_IP}{client_ip}"
        rpm = settings.RATE_LIMIT_ANON_RPM

    return key, rpm


# ============================================================
# 1. RateLimitMiddleware — 전역 API Rate Limiting
# ============================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    전역 API Rate Limiting 미들웨어.

    Redis Sorted Set 슬라이딩 윈도우 방식으로 IP(비인증) 또는
    Bearer 토큰 접두사(인증 사용자) 기반으로 분당 요청 수를 제한한다.

    동작 방식:
    1. 헬스체크/문서 경로는 제외
    2. Authorization 헤더로 인증/비인증 판별 → Redis 키 + RPM 결정
    3. Redis 파이프라인으로 원자적 슬라이딩 윈도우 체크
       a. zremrangebyscore: 60초 이전 타임스탬프 제거
       b. zcard: 현재 윈도우 내 요청 수 조회
       c. 한도 초과 시 → 429 반환 (zadd 전에 거부)
       d. 한도 미만 시 → zadd + expire 후 요청 통과
    4. Redis 연결 실패 시 → rate limit 무시, 요청 통과 (graceful degradation)

    기존 chat.py의 _check_upload_rate_limit는 이미지 업로드 전용 추가 제한으로
    이 미들웨어와 독립적으로 유지된다.
    """

    def __init__(self, app: ASGIApp) -> None:
        """
        미들웨어 초기화.

        Redis 클라이언트는 요청 처리 시점에 get_redis()로 지연 획득한다.
        (미들웨어 초기화 시점에는 이벤트 루프가 아직 준비되지 않을 수 있음)

        Args:
            app: ASGI 앱 인스턴스 (FastAPI)
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        요청을 rate limit 체크 후 처리한다.

        Args:
            request: Starlette Request 객체
            call_next: 다음 미들웨어 또는 엔드포인트 핸들러

        Returns:
            정상 처리된 응답 또는 429 Too Many Requests
        """
        path = request.url.path

        # 헬스체크/문서 경로는 rate limit 제외
        if path in _EXEMPT_PATHS:
            return await call_next(request)

        # Redis 키 + RPM 결정
        rate_key, rpm_limit = _get_rate_limit_key(request)
        now = time.time()

        try:
            # Redis 클라이언트 획득 (싱글턴, DB 클라이언트 초기화 이후 항상 존재)
            from monglepick.db.clients import get_redis  # 순환 임포트 방지를 위해 지연 임포트
            redis = await get_redis()

            # 파이프라인으로 원자적 슬라이딩 윈도우 체크
            pipe = redis.pipeline()

            # [1] 윈도우 밖(60초 이전) 오래된 타임스탬프 제거
            pipe.zremrangebyscore(rate_key, 0, now - _RATE_LIMIT_WINDOW_SEC)

            # [2] 현재 윈도우 내 요청 수 조회 (추가 전 카운트)
            pipe.zcard(rate_key)

            results = await pipe.execute()

            # results[1] = zcard 결과 (현재 윈도우 내 요청 수)
            current_count: int = results[1]

            if current_count >= rpm_limit:
                # 한도 초과 → 429 반환 (타임스탬프 추가하지 않음)
                logger.warning(
                    "rate_limit_exceeded",
                    path=path,
                    rate_key=rate_key,
                    current_count=current_count,
                    limit=rpm_limit,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "요청이 너무 많습니다. 잠시 후 다시 시도해주세요.",
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "retry_after": _RATE_LIMIT_WINDOW_SEC,
                    },
                    headers={
                        # 클라이언트가 retry_after 이후 재시도할 수 있도록 안내
                        "Retry-After": str(_RATE_LIMIT_WINDOW_SEC),
                        "X-RateLimit-Limit": str(rpm_limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(now + _RATE_LIMIT_WINDOW_SEC)),
                    },
                )

            # 한도 미만 → 현재 타임스탬프 추가 + TTL 설정
            pipe2 = redis.pipeline()
            # 타임스탬프를 스코어로 사용하여 각 요청을 고유하게 식별
            # 같은 초에 여러 요청이 올 수 있으므로 마이크로초 단위로 키 생성
            pipe2.zadd(rate_key, {f"{now:.6f}": now})
            # 키 TTL: 윈도우(60초) + 여유(10초) — 미요청 시 자동 정리
            pipe2.expire(rate_key, _RATE_LIMIT_WINDOW_SEC + 10)
            await pipe2.execute()

        except Exception as e:
            # Redis 연결 실패 또는 기타 예외 → rate limit 무시 (서비스 가용성 우선)
            logger.warning(
                "rate_limit_middleware_redis_error",
                path=path,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Redis 실패 시에도 요청은 통과시킴

        # Rate limit 통과 → 다음 핸들러로 위임
        return await call_next(request)


# ============================================================
# 2. TimeoutMiddleware — 요청 타임아웃
# ============================================================

class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    요청 처리 타임아웃 미들웨어.

    asyncio.wait_for()를 사용하여 요청 처리에 타임아웃을 적용한다.
    Ollama LLM 호출이나 DB 쿼리가 hang하는 경우 리소스 점유를 방지한다.

    타임아웃 정책:
    - SSE 엔드포인트(/api/v1/chat, /api/v1/match): TIMEOUT_SSE_SEC 초 (기본 300초)
      → LangGraph 전체 그래프 실행 시간을 포함하므로 여유있게 설정
      → 단, SSE 스트리밍 응답은 첫 청크 반환 후 타임아웃에서 벗어남
    - 일반 엔드포인트: TIMEOUT_DEFAULT_SEC 초 (기본 30초)

    주의:
    - Starlette BaseHTTPMiddleware는 StreamingResponse를 버퍼링하지 않는다.
      SSE 응답의 경우 call_next()가 첫 청크를 반환하면 타임아웃이 해제되므로
      노드별 개별 타임아웃(asyncio.wait_for)을 그래프 내부에서 유지해야 한다.
    - 타임아웃 발생 시 504 Gateway Timeout을 반환한다.
    """

    def __init__(self, app: ASGIApp) -> None:
        """
        미들웨어 초기화.

        Args:
            app: ASGI 앱 인스턴스 (FastAPI)
        """
        super().__init__(app)

    def _get_timeout(self, path: str) -> float:
        """
        경로에 따라 적용할 타임아웃(초)을 결정한다.

        SSE 엔드포인트(/api/v1/chat, /api/v1/match)는 긴 타임아웃 적용.
        그 외 엔드포인트는 짧은 타임아웃 적용.

        Args:
            path: 요청 경로 (예: "/api/v1/chat", "/health")

        Returns:
            타임아웃 초(float)
        """
        # SSE 엔드포인트 여부 확인 (접두사 매칭)
        for prefix in _SSE_PATH_PREFIXES:
            if path.startswith(prefix):
                return float(settings.TIMEOUT_SSE_SEC)

        # 일반 엔드포인트
        return float(settings.TIMEOUT_DEFAULT_SEC)

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        요청에 타임아웃을 적용하여 처리한다.

        Args:
            request: Starlette Request 객체
            call_next: 다음 미들웨어 또는 엔드포인트 핸들러

        Returns:
            정상 처리된 응답 또는 504 Gateway Timeout
        """
        path = request.url.path
        timeout_sec = self._get_timeout(path)

        try:
            # asyncio.wait_for로 타임아웃 적용
            # call_next(request)는 코루틴이므로 wait_for와 호환된다.
            response = await asyncio.wait_for(
                call_next(request),
                timeout=timeout_sec,
            )
            return response

        except asyncio.TimeoutError:
            # 타임아웃 초과 → 504 반환
            logger.error(
                "request_timeout",
                path=path,
                method=request.method,
                timeout_sec=timeout_sec,
            )
            return JSONResponse(
                status_code=504,
                content={
                    "detail": f"요청 처리 시간이 초과되었습니다. ({timeout_sec:.0f}초)",
                    "error_code": "GATEWAY_TIMEOUT",
                },
            )
        except Exception as e:
            # 예상치 못한 예외 → 500 반환 (미들웨어는 에러를 전파하지 않음)
            logger.error(
                "timeout_middleware_unexpected_error",
                path=path,
                error=str(e),
                error_type=type(e).__name__,
            )
            # 예외를 직접 처리하지 않고 상위로 전파하여 FastAPI 예외 핸들러에게 위임
            raise
