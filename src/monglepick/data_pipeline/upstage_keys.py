"""
Upstage API 키 회전(rotation) 헬퍼.

설계 배경:
    - Upstage 무료 크레딧이 한정적이므로 메인 키 (UPSTAGE_API_KEY) 가 만료될 수 있음.
    - 사용자가 .env 에 백업 키 (UPSTAGE_API_KEY2) 를 추가해둔 경우, 메인 키가
      401 Unauthorized / quota_exceeded 로 실패하면 자동으로 백업 키로 전환한다.

사용 정책:
    - 본 모듈은 **Task #5 (run_full_reload.py PID 16164) 에 영향이 없는** Phase 2~9
      신규 스크립트만 사용한다.
    - Task #5 는 이미 메모리에 메인 키를 import 한 상태이므로 swap 불가능.
      Task #5 swap 절차는 docs/Phase_ML4_후속_실행_체크리스트.md §10 폴백 매트릭스 참조.

사용 예:
    from monglepick.data_pipeline.upstage_keys import UpstageKeyRotator

    async def main():
        rotator = UpstageKeyRotator()  # .env 에서 자동 로드
        async with rotator.openai_client() as client:
            response = await client.chat.completions.create(...)
            # 401 발생 시 rotator 가 백업 키로 전환 후 재시도

    # 또는 직접 키 가져오기
    api_key = rotator.current_key()  # 현재 활성 키
    rotator.mark_failed()              # 실패 시 다음 키로 전환
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from openai import AsyncOpenAI

from monglepick.config import settings

logger = logging.getLogger(__name__)

UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"


class UpstageKeyExhausted(Exception):
    """모든 등록된 Upstage 키가 만료/실패했을 때 발생."""

    pass


class UpstageKeyRotator:
    """
    Upstage 키 라운드 로빈 / fallback 매니저.

    초기 구성:
        - 키 1: settings.UPSTAGE_API_KEY (메인)
        - 키 2: settings.UPSTAGE_API_KEY2 (백업, .env 에 있을 때만)

    `mark_failed()` 호출 시 다음 키로 전환. 모든 키 실패 시 UpstageKeyExhausted.

    스레드 안전: 단일 이벤트 루프 내 단일 인스턴스 사용 가정.
    """

    def __init__(self, keys: list[str] | None = None):
        if keys is None:
            keys = []
            if settings.UPSTAGE_API_KEY:
                keys.append(settings.UPSTAGE_API_KEY)
            if settings.UPSTAGE_API_KEY2:
                keys.append(settings.UPSTAGE_API_KEY2)

        # 빈 문자열 제거
        keys = [k for k in keys if k and k.strip()]

        if not keys:
            raise ValueError(
                "Upstage API 키가 .env 에 하나도 설정되지 않았습니다 "
                "(UPSTAGE_API_KEY / UPSTAGE_API_KEY2)"
            )

        self._keys = keys
        self._idx = 0
        self._failed_idxs: set[int] = set()

    def current_key(self) -> str:
        """현재 활성 키 반환."""
        if self._idx >= len(self._keys):
            raise UpstageKeyExhausted(
                f"모든 Upstage 키 ({len(self._keys)} 개) 가 실패했습니다."
            )
        return self._keys[self._idx]

    def current_index(self) -> int:
        """현재 키 인덱스 (디버깅용)."""
        return self._idx

    def total_keys(self) -> int:
        return len(self._keys)

    def remaining_keys(self) -> int:
        return len(self._keys) - len(self._failed_idxs)

    def mark_failed(self, reason: str = "") -> bool:
        """
        현재 키를 실패로 마킹하고 다음 키로 전환.

        Returns:
            bool: True 이면 다음 키 사용 가능, False 이면 모든 키 소진
        """
        self._failed_idxs.add(self._idx)
        logger.warning(
            "upstage_key_failed",
            extra={
                "failed_idx": self._idx,
                "failed_count": len(self._failed_idxs),
                "total": len(self._keys),
                "reason": reason,
            },
        )

        # 다음 사용 가능 키 찾기
        for next_idx in range(self._idx + 1, len(self._keys)):
            if next_idx not in self._failed_idxs:
                self._idx = next_idx
                logger.info(
                    "upstage_key_rotated",
                    extra={"new_idx": next_idx, "total": len(self._keys)},
                )
                return True

        # 더 이상 사용 가능 키 없음
        return False

    @asynccontextmanager
    async def openai_client(self) -> AsyncIterator[AsyncOpenAI]:
        """
        OpenAI 호환 클라이언트 컨텍스트 매니저.

        주의: 이 클라이언트는 단일 키만 들고 있다. 401 발생 시 호출 측이
        명시적으로 mark_failed() 후 클라이언트를 재생성해야 한다.

        사용 예:
            rotator = UpstageKeyRotator()
            while rotator.remaining_keys() > 0:
                try:
                    async with rotator.openai_client() as client:
                        response = await client.chat.completions.create(...)
                        break
                except UnauthorizedError:
                    if not rotator.mark_failed("401 Unauthorized"):
                        raise UpstageKeyExhausted()
        """
        client = AsyncOpenAI(
            api_key=self.current_key(),
            base_url=UPSTAGE_BASE_URL,
        )
        try:
            yield client
        finally:
            try:
                await client.close()
            except Exception:
                pass


def is_upstage_auth_error(error: Exception) -> bool:
    """
    예외가 Upstage 키 만료/인증 실패인지 판별.

    인식 패턴:
        - 401 Unauthorized
        - quota_exceeded
        - insufficient_quota
        - invalid_api_key
        - credit_exhausted
    """
    msg = str(error).lower()
    patterns = [
        "401",
        "unauthorized",
        "quota_exceeded",
        "insufficient_quota",
        "invalid_api_key",
        "credit_exhausted",
        "incorrect api key",
    ]
    return any(p in msg for p in patterns)
