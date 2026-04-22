"""
Agent API 인증 의존성 (FastAPI Depends).

현재 정의된 가드:
  - verify_service_key: X-Service-Key 헤더 검증 (내부 서비스 호출 전용)

사용 예:
    from fastapi import APIRouter, Depends
    from monglepick.api.auth_deps import verify_service_key

    admin_router = APIRouter(
        prefix="/admin",
        tags=["admin"],
        dependencies=[Depends(verify_service_key)],
    )

Backend ↔ Agent 내부 호출은 전부 X-Service-Key 헤더를 첨부한다.
Admin SPA / 브라우저 직접 호출은 Backend 를 경유하므로 Agent 에서 직접 받을 일이 없다.
따라서 Agent 의 /admin/** 전체를 이 가드로 보호하는 것이 안전.
"""

from __future__ import annotations

import hmac

import structlog
from fastapi import Header, HTTPException, status

from monglepick.config import settings

logger = structlog.get_logger()

# 공개 헤더 이름 — Backend AppConstants.HEADER_SERVICE_KEY 와 동일해야 한다.
SERVICE_KEY_HEADER = "X-Service-Key"

# 운영 환경에서 반드시 교체해야 할 개발 기본 키.
# Backend application.yml 의 app.service.key 기본값과 동일.
_DEV_DEFAULT_KEY = "dev-service-key-change-me"


async def verify_service_key(
    x_service_key: str | None = Header(default=None, alias=SERVICE_KEY_HEADER),
) -> None:
    """
    X-Service-Key 헤더 값이 settings.SERVICE_API_KEY 와 일치하는지 검증한다.

    - 헤더 누락: 401 Unauthorized
    - 헤더 값 불일치: 401 Unauthorized
    - 상수 시간 비교(hmac.compare_digest) 사용 — 타이밍 공격 방지

    운영 환경에서는 .env.prod 의 SERVICE_API_KEY 를 랜덤 문자열로 설정해야 한다.
    dev-service-key-change-me 기본값이 그대로 노출되면 WARN 로그를 남긴다.
    """
    expected = settings.SERVICE_API_KEY or _DEV_DEFAULT_KEY

    if expected == _DEV_DEFAULT_KEY:
        # 개발 환경 경고 — 프로덕션에서는 반드시 교체 필요
        logger.warning(
            "service_api_key_using_default_dev_key",
            header=SERVICE_KEY_HEADER,
            hint="운영 배포 시 SERVICE_API_KEY 환경변수를 랜덤 문자열로 설정하세요.",
        )

    if not x_service_key:
        logger.warning(
            "service_key_missing",
            header=SERVICE_KEY_HEADER,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "missing_service_key",
                "message": f"{SERVICE_KEY_HEADER} 헤더가 필요합니다.",
            },
        )

    # 상수 시간 비교 — 타이밍 공격 방지
    if not hmac.compare_digest(x_service_key, expected):
        logger.warning(
            "service_key_invalid",
            header=SERVICE_KEY_HEADER,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_service_key",
                "message": "유효하지 않은 서비스 키입니다.",
            },
        )
