"""
고객센터 AI 에이전트 v4 — Backend GET 공통 HTTP 헬퍼.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.1 (Read Tool) / §8 (보안)

핵심 원칙:
- X-Service-Key: settings.SERVICE_API_KEY 헤더 강제 (point_client.py 와 동일한 패턴).
- ctx.user_id 는 항상 쿼리 파라미터로 강제 주입한다. 사용자 발화 내 userId 는 무시.
  → BOLA(Broken Object Level Authorization) 차단.
- ok=False 응답은 에러 전파 없이 {"ok": False, "error": ...} dict 로 반환.
- 싱글턴 httpx.AsyncClient 를 분리 유지:
    (1) admin_backend_client.py 는 관리자 JWT forwarding 전용 (헤더 비움).
    (2) 본 모듈은 X-Service-Key + X-User-Id 고정 헤더가 있는 고객센터 전용 클라이언트.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog

from monglepick.config import settings
from . import ToolContext

logger = structlog.get_logger(__name__)


# ============================================================
# httpx 싱글턴 (admin_backend_client 와 분리)
# ============================================================

_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_support_http_client() -> httpx.AsyncClient:
    """
    고객센터 Backend 호출용 httpx.AsyncClient 싱글턴을 반환한다.

    기본 헤더에 X-Service-Key 를 포함한다. user_id 는 호출 시점에 X-User-Id 헤더로
    per-request 주입되므로 기본 헤더에 넣지 않는다.

    Timeout: connect 3s / read 5s. 고객센터 조회 EP 는 단순 조회라 충분.
    """
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        # double-check: Lock 대기 중 다른 코루틴이 이미 생성했을 수 있음
        if _client is None:
            _client = httpx.AsyncClient(
                base_url=settings.BACKEND_BASE_URL,
                timeout=httpx.Timeout(5.0, connect=3.0),
                headers={"X-Service-Key": settings.SERVICE_API_KEY},
            )
    return _client


async def close_support_client() -> None:
    """앱 종료 시 httpx 클라이언트 정리 (main.py lifespan shutdown 에서 호출 가능)."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ============================================================
# 핵심 호출 함수
# ============================================================

async def call_backend_get(
    ctx: ToolContext,
    path: str,
    params: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """
    Backend GET EP 를 ServiceKey + X-User-Id 패턴으로 호출한다.

    보안 원칙:
    - X-Service-Key: settings.SERVICE_API_KEY (싱글턴 클라이언트 기본 헤더에 포함됨).
    - X-User-Id: ctx.user_id 강제 주입. LLM 이 args 에 담아준 userId 는 절대 사용하지 않음.
      Backend 가 X-User-Id 헤더로 JWT 없이 사용자 식별하는 서비스 내부 패턴.
    - 게스트(is_guest=True) 체크는 각 tool handler 에서 먼저 수행한다. 이 함수는
      is_guest 체크를 하지 않으며, tool handler 가 이미 거른 뒤에만 호출된다.

    반환:
        ok=True  → {"ok": True, "data": <응답 JSON>}
        ok=False → {"ok": False, "error": "<사유>", "status_code": <int>}
                   (HTTP 4xx/5xx, 타임아웃, JSON 디코딩 실패 모두 포함)

    Args:
        ctx:       런타임 컨텍스트. user_id / session_id / request_id 사용.
        path:      /api/v1/... 형태의 절대 경로 (BACKEND_BASE_URL 기준 상대).
        params:    쿼리스트링 dict. user_id 는 여기에 넣지 말 것 — 헤더로 강제 주입됨.
        timeout_s: 타임아웃(초). 기본 5s.
    """
    started = time.perf_counter()

    # ctx.user_id 를 X-User-Id 헤더로 강제 주입 — BOLA 방어.
    # 사용자 발화에서 파싱된 userId 가 params 에 섞여 들어올 수 있으므로 별도 헤더로 분리.
    per_req_headers: dict[str, str] = {}
    if ctx.user_id:
        per_req_headers["X-User-Id"] = ctx.user_id
    if ctx.request_id:
        per_req_headers["X-Request-Id"] = ctx.request_id

    try:
        client = await _get_support_http_client()
        resp = await client.get(
            path,
            params=params,
            headers=per_req_headers,
            timeout=timeout_s,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if 200 <= resp.status_code < 300:
            try:
                data = resp.json()
            except Exception as je:
                logger.warning(
                    "support_backend_json_decode_failed",
                    path=path,
                    status=resp.status_code,
                    error=str(je),
                    request_id=ctx.request_id,
                )
                return {
                    "ok": False,
                    "error": f"json_decode_error:{type(je).__name__}",
                    "status_code": resp.status_code,
                }
            logger.info(
                "support_backend_ok",
                path=path,
                status=resp.status_code,
                latency_ms=elapsed_ms,
                user_id=ctx.user_id,
                request_id=ctx.request_id,
            )
            return {"ok": True, "data": data}

        # 2xx 외 → Backend 에러. body 의 message/detail 필드 추출 시도.
        detail = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = body.get("message") or body.get("detail") or ""
        except Exception:
            pass

        logger.warning(
            "support_backend_non_2xx",
            path=path,
            status=resp.status_code,
            detail=detail[:200],
            latency_ms=elapsed_ms,
            user_id=ctx.user_id,
            request_id=ctx.request_id,
        )
        return {
            "ok": False,
            "error": detail or f"http_{resp.status_code}",
            "status_code": resp.status_code,
        }

    except httpx.TimeoutException as te:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "support_backend_timeout",
            path=path,
            elapsed_ms=elapsed_ms,
            user_id=ctx.user_id,
            request_id=ctx.request_id,
        )
        return {
            "ok": False,
            "error": f"timeout:{type(te).__name__}",
            "status_code": 0,
        }
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "support_backend_unexpected_error",
            path=path,
            error=str(e),
            error_type=type(e).__name__,
            elapsed_ms=elapsed_ms,
            user_id=ctx.user_id,
            request_id=ctx.request_id,
        )
        return {
            "ok": False,
            "error": f"unexpected:{type(e).__name__}",
            "status_code": 0,
        }
