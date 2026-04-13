"""
Backend 채팅 세션 API 비동기 클라이언트.

Agent가 매 턴마다 Backend를 통해 MySQL에 세션 상태를 저장/로드한다.
Redis를 대체하여 MySQL이 세션 데이터의 단일 진실 소스(Source of Truth)가 된다.

Backend 통신: httpx.AsyncClient (내부 HTTP, 같은 VM/네트워크)
인증: X-Service-Key 헤더 (settings.SERVICE_API_KEY)

엔드포인트:
  - POST /api/v1/chat/internal/session/save  — 세션 upsert (매 턴)
  - POST /api/v1/chat/internal/session/load  — 세션 로드 (이어하기)
"""

from __future__ import annotations

from typing import Any

import structlog

# point_client.py의 httpx 싱글턴 클라이언트를 공유한다 (같은 BACKEND_BASE_URL + X-Service-Key)
from monglepick.api.point_client import _get_http_client

logger = structlog.get_logger(__name__)


async def save_session_to_backend(
    user_id: str,
    session_id: str,
    messages: str,
    turn_count: int,
    title: str | None = None,
    session_state: str | None = None,
    intent_summary: str | None = None,
) -> dict[str, Any] | None:
    """
    Backend에 세션 데이터를 저장한다 (upsert).

    세션이 없으면 신규 생성, 있으면 messages/sessionState 업데이트.
    Agent가 매 턴마다 그래프 실행 완료 후 호출한다.

    Args:
        user_id: 사용자 ID
        session_id: 세션 UUID
        messages: 전체 대화 내역 JSON 문자열
        turn_count: 현재 턴 수
        title: 세션 제목 (nullable — 첫 턴에서 Backend가 자동 생성)
        session_state: Agent 세션 상태 JSON (preferences, emotion 등)
        intent_summary: 의도 요약 JSON

    Returns:
        저장 응답 dict {chatSessionArchiveId, sessionId, created} 또는 None (실패 시)
    """
    try:
        client = await _get_http_client()
        payload = {
            "userId": user_id,
            "sessionId": session_id,
            "messages": messages,
            "turnCount": turn_count,
            "title": title,
            "sessionState": session_state,
            "intentSummary": intent_summary,
        }
        resp = await client.post("/api/v1/chat/internal/session/save", json=payload)

        if resp.status_code == 200:
            data = resp.json()
            logger.info(
                "chat_session_saved_to_backend",
                session_id=session_id,
                user_id=user_id,
                created=data.get("created"),
            )
            return data

        # [FIX] 저장 실패 시 로그 레벨을 error로 상향 + 상세 정보 추가.
        # 기존 warning 레벨은 로그가 묻혀 문제 파악이 불가능했음.
        logger.error(
            "chat_session_save_failed",
            session_id=session_id,
            user_id=user_id,
            status=resp.status_code,
            body=resp.text[:500],
        )
        return None

    except Exception as e:
        # [FIX] 로그 레벨 error로 상향 + 스택트레이스 포함.
        # httpx 타임아웃, 네트워크 에러 등이 warning으로 묻혀 진단이 불가능했음.
        logger.error(
            "chat_session_save_error",
            session_id=session_id,
            user_id=user_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        return None


async def load_session_from_backend(
    user_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    """
    Backend에서 세션 데이터를 로드한다 (이어하기).

    Args:
        user_id: 사용자 ID
        session_id: 세션 UUID

    Returns:
        세션 데이터 dict {sessionId, messages, turnCount, sessionState, intentSummary}
        또는 None (세션 없음/실패)
    """
    try:
        client = await _get_http_client()
        payload = {
            "userId": user_id,
            "sessionId": session_id,
        }
        resp = await client.post("/api/v1/chat/internal/session/load", json=payload)

        if resp.status_code == 200:
            data = resp.json()
            # Backend가 세션을 찾지 못하면 body가 null일 수 있음
            if data is None or data.get("sessionId") is None:
                logger.debug("chat_session_load_miss", session_id=session_id)
                return None
            logger.debug(
                "chat_session_loaded_from_backend",
                session_id=session_id,
                turn_count=data.get("turnCount", 0),
            )
            return data

        logger.warning(
            "chat_session_load_failed",
            session_id=session_id,
            status=resp.status_code,
            body=resp.text[:200],
        )
        return None

    except Exception as e:
        # 로드 실패 시 None 반환 (신규 세션으로 진행)
        logger.warning(
            "chat_session_load_error",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None
