"""
하이브리드 세션 저장소 — Redis 핫 캐시 + MySQL(Backend) 아카이브.

매 턴마다 초기화되던 Chat Agent State를 Redis(우선)·MySQL(아카이브)에 저장/복원하여
대화 맥락(messages, preferences, emotion, turn_count 등)을 유지한다.

아키텍처 (Option B: Redis cache + MySQL write-behind):
  - Redis: per-turn hot path. TTL = `SESSION_TTL_DAYS` 일. 매 턴 R/W.
  - MySQL (via Backend `/api/v1/chat/internal/session/*`): 영구 아카이브.
    save 시 fire-and-forget 백그라운드 task 로 async flush → SSE 응답을 블로킹하지 않는다.
    이력 목록(`/chat/history`) 및 이어하기(재진입) 은 MySQL 이 단일 진실 원본.
  - Redis miss 시 Backend load 로 폴백하여 Redis 에 재적재한다 (cache-aside 패턴).
  - Redis 연결 실패/에러 시 Backend 직접 R/W 로 graceful degradation.

과거에는 MySQL 단일 저장소(2024-04 전환) 였으나, 매 턴 HTTP 라운드트립이 SSE 레이턴시를
늘리고 JWT/userId 검증 실패 시 400 으로 세션 복원이 통째로 실패하는 문제가 있었다.
Option B 로 복귀하면서 Redis 를 1차 저장소로, MySQL 을 영구 아카이브로 역할 분리한다.

저장 대상 필드 (영속 필요):
- messages: 전체 대화 이력 → messages JSON
- preferences: 누적된 사용자 선호 (ExtractedPreferences) → session_state JSON 내부
- emotion: 마지막 감정 분석 결과 (EmotionResult) → session_state JSON 내부
- turn_count: 현재 턴 수
- user_profile: MySQL 유저 프로필 (세션 내 캐싱) → session_state JSON 내부
- watch_history: MySQL 시청 이력 (세션 내 캐싱) → session_state JSON 내부

저장하지 않는 필드 (매 턴 새로 도출):
- intent, search_query, candidate_movies, ranked_movies
- response, clarification, retrieval_quality_passed, retrieval_feedback
- image_data, image_analysis, error

Redis 키 규격:
  chat:session:{user_id}:{session_id}  → JSON {messages, turn_count, session_state}
  TTL: settings.SESSION_TTL_DAYS * 86400 초 (기본 30일). 쓰기마다 TTL 갱신(rolling).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from monglepick.agents.chat.models import EmotionResult, ExtractedPreferences
from monglepick.api.chat_client import (
    load_session_from_backend,
    save_session_to_backend,
)
from monglepick.config import settings
from monglepick.db.clients import get_redis

logger = structlog.get_logger()

# ── 세션 설정 상수 ──

# 최대 대화 턴 수: 이 값을 초과하면 messages 앞부분을 자동 truncation
MAX_CONVERSATION_TURNS: int = 20

# Redis 키 prefix 및 TTL(초). TTL 은 config.SESSION_TTL_DAYS(기본 30일) 기반.
_REDIS_KEY_PREFIX = "chat:session"
_REDIS_TTL_SECONDS: int = int(settings.SESSION_TTL_DAYS) * 86400

# fire-and-forget Backend flush 를 추적하기 위한 Task 집합.
# 참조를 유지하지 않으면 가비지 컬렉터가 Task 를 중도 수거할 수 있다(Python 3.11+ 문서 권고).
# 테스트에서는 `_wait_for_pending_flushes()` 로 완료를 대기한다.
_pending_flushes: set[asyncio.Task[Any]] = set()


async def _wait_for_pending_flushes() -> None:
    """
    미완료된 Backend flush 백그라운드 task 를 모두 대기한다.

    프로덕션 코드에서는 호출할 필요 없으며, 단위 테스트가 `save_session` 직후
    mock assertion 을 수행하기 전에 이 함수를 호출해 race condition 을 없애기 위한 용도.
    """
    if not _pending_flushes:
        return
    await asyncio.gather(*list(_pending_flushes), return_exceptions=True)


def _session_key(user_id: str, session_id: str) -> str:
    """
    Redis 세션 키를 생성한다.

    user_id 를 포함하여 다른 사용자의 세션을 조회 불가하도록 네임스페이스를 분리한다.
    빈 user_id 는 사전에 가드하므로 여기서는 검증하지 않는다.
    """
    return f"{_REDIS_KEY_PREFIX}:{user_id}:{session_id}"


async def _redis_get_session(user_id: str, session_id: str) -> dict[str, Any] | None:
    """
    Redis 에서 세션을 로드한다. 연결 실패/파싱 실패 시 None 반환(예외 전파 금지).

    Returns:
        Backend `/session/load` 응답과 호환되는 dict
        ({sessionId, messages, turnCount, sessionState, intentSummary}).
        Redis miss / 에러 시 None.
    """
    try:
        redis = await get_redis()
        raw = await redis.get(_session_key(user_id, session_id))
        if raw is None:
            return None

        # Redis 에는 Backend 응답 포맷 그대로 저장하여 일관성 유지.
        payload = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        if not isinstance(payload, dict):
            logger.warning("redis_session_invalid_payload", session_id=session_id)
            return None
        return payload
    except Exception as e:
        # Redis 장애 시 상위 호출자가 Backend 폴백을 시도하도록 None 반환.
        logger.warning(
            "redis_session_get_error",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


async def _redis_set_session(
    user_id: str,
    session_id: str,
    payload: dict[str, Any],
) -> bool:
    """
    Redis 에 세션을 저장한다(TTL 갱신). 실패해도 호출자에게 예외 전파 금지.

    Returns:
        성공 여부 (True 면 Redis 쓰기 성공).
    """
    try:
        redis = await get_redis()
        await redis.set(
            _session_key(user_id, session_id),
            json.dumps(payload, ensure_ascii=False, default=str),
            ex=_REDIS_TTL_SECONDS,
        )
        return True
    except Exception as e:
        logger.warning(
            "redis_session_set_error",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return False


def _flush_to_backend_in_background(
    user_id: str,
    session_id: str,
    messages_json: str,
    turn_count: int,
    session_state_json: str,
) -> None:
    """
    Backend MySQL 에 세션을 async flush 한다 (fire-and-forget).

    SSE 응답 완료 후 Agent 턴 루프를 블로킹하지 않기 위해 asyncio.create_task 로 분리.
    실패 시 로그만 남기고 다음 턴에서 동일한 전체 상태가 다시 저장되므로 유실은 제한적.
    """
    async def _run() -> None:
        try:
            result = await save_session_to_backend(
                user_id=user_id,
                session_id=session_id,
                messages=messages_json,
                turn_count=turn_count,
                title=None,  # Backend 가 첫 턴에서 자동 생성
                session_state=session_state_json,
                intent_summary=None,
            )
            if result is None:
                logger.error(
                    "session_backend_flush_returned_none",
                    session_id=session_id,
                    user_id=user_id,
                    turn_count=turn_count,
                )
        except Exception as e:
            logger.error(
                "session_backend_flush_error",
                session_id=session_id,
                user_id=user_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

    # 백그라운드 실행. Task 참조를 _pending_flushes 에 보관해야 GC 로 중도 수거되지 않는다.
    task = asyncio.create_task(_run())
    _pending_flushes.add(task)
    task.add_done_callback(_pending_flushes.discard)


async def load_session(user_id: str, session_id: str) -> dict[str, Any] | None:
    """
    세션을 로드한다 — Redis 우선, Backend MySQL 폴백(cache-aside).

    load 순서:
      1) user_id / session_id 빈 값 가드 (Backend @NotBlank 검증 실패 방지)
      2) Redis 조회 → 히트 시 그대로 반환
      3) Backend `/session/load` 조회 → 히트 시 Redis 에 재적재 후 반환
      4) 둘 다 miss → None (신규 세션)

    Args:
        user_id: 사용자 ID
        session_id: 세션 ID

    Returns:
        dict (messages, preferences, emotion, turn_count, user_profile, watch_history)
        또는 None (세션 없음/익명/에러)
    """
    # [가드] 빈 session_id → 신규 세션이므로 저장소 조회 불필요
    if not session_id:
        return None
    # [가드] 빈 user_id → Backend @NotBlank 검증이 400 을 반환하므로 조회하지 않는다.
    # 익명 사용자는 세션 복원 대상에서 제외(매 턴 새 세션으로 동작).
    if not user_id:
        logger.debug("session_load_skipped_no_user_id", session_id=session_id)
        return None

    try:
        # 1) Redis 핫 캐시 조회
        raw = await _redis_get_session(user_id, session_id)
        source = "redis" if raw is not None else None

        # 2) Redis miss → Backend 폴백 + Redis 재적재
        if raw is None:
            raw = await load_session_from_backend(user_id, session_id)
            if raw is None:
                logger.debug("session_load_miss", session_id=session_id)
                return None
            source = "backend"
            # Backend hit → Redis 에 재적재 (다음 턴부터는 Redis 에서 직접 읽음)
            await _redis_set_session(user_id, session_id, raw)

        # Backend 응답 포맷 → Agent 내부 포맷 변환
        messages_str = raw.get("messages", "[]")
        messages = json.loads(messages_str) if isinstance(messages_str, str) else messages_str

        # sessionState JSON 파싱 → 개별 필드로 풀기
        session_state_str = raw.get("sessionState")
        session_state: dict[str, Any] = {}
        if session_state_str:
            session_state = (
                json.loads(session_state_str)
                if isinstance(session_state_str, str)
                else session_state_str
            )

        # Pydantic 모델 복원: dict → ExtractedPreferences / EmotionResult
        # 스키마 변경 시에도 세션 자체는 유지되도록 개별 예외 처리
        preferences = session_state.get("preferences")
        if preferences is not None:
            try:
                preferences = ExtractedPreferences(**preferences)
            except Exception:
                logger.warning("preferences_restore_failed", session_id=session_id)
                preferences = None

        emotion = session_state.get("emotion")
        if emotion is not None:
            try:
                emotion = EmotionResult(**emotion)
            except Exception:
                logger.warning("emotion_restore_failed", session_id=session_id)
                emotion = None

        # 세션 내 최근 추천 ID 롤링 윈도우 — 스키마 이전 세션은 비어 있는 기본값
        recent_recommended_ids_raw = session_state.get("recent_recommended_ids") or []
        recent_recommended_ids: list[str] = [
            str(rid) for rid in recent_recommended_ids_raw if rid
        ]

        data: dict[str, Any] = {
            "messages": messages,
            "preferences": preferences,
            "emotion": emotion,
            "turn_count": raw.get("turnCount", 0),
            "user_profile": session_state.get("user_profile", {}),
            "watch_history": session_state.get("watch_history", []),
            "recent_recommended_ids": recent_recommended_ids,
        }

        logger.info(
            "session_loaded",
            session_id=session_id,
            source=source,
            turn_count=data["turn_count"],
            message_count=len(data["messages"]),
            has_preferences=data["preferences"] is not None,
            has_emotion=data["emotion"] is not None,
        )
        return data

    except Exception as e:
        # 세션 로드 실패 시 None 반환 (신규 세션으로 진행)
        logger.warning(
            "session_load_error",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


async def save_session(user_id: str, session_id: str, state: dict[str, Any]) -> None:
    """
    그래프 실행 완료 후 세션을 저장한다 — Redis 즉시 반영 + Backend async flush(write-behind).

    저장 필드 (6개):
    - messages: list[dict]        — 전체 대화 이력 → messages JSON
    - preferences: dict | None    — ExtractedPreferences → session_state 내부
    - emotion: dict | None        — EmotionResult → session_state 내부
    - turn_count: int             — 현재 턴 수
    - user_profile: dict          — MySQL 유저 프로필 (캐싱) → session_state 내부
    - watch_history: list[dict]   — MySQL 시청 이력 (캐싱) → session_state 내부

    MAX_CONVERSATION_TURNS(20) 초과 시 messages 앞부분 자동 truncation.

    쓰기 순서:
      1) Redis set (TTL 갱신) — 다음 턴이 즉시 새 상태를 읽도록 동기 await
      2) asyncio.create_task 로 Backend `/session/save` flush — fire-and-forget

    Args:
        user_id: 사용자 ID
        session_id: 세션 ID
        state: 그래프 실행 완료 후의 전체 State dict
    """
    # [가드] 빈 session_id / user_id 는 저장 대상 아님 (익명·에러 케이스)
    # Phase 1 진단 로그 강화 (2026-04-15):
    # "로그인 사용자인데 JWT 검증 실패 → user_id 빈 문자열 → 저장 스킵" 케이스가
    # 실제 운영에서 채팅 이력 미표시의 근본원인이므로 WARNING → ERROR 로 상향하여
    # 알림/대시보드에서 즉시 포착되도록 한다. turn_count 를 함께 기록하여
    # "첫 턴 저장 실패" vs "N턴 진행 후 실패" 구분 가능.
    if not session_id:
        logger.error(
            "session_save_skipped_no_session_id",
            user_id=user_id,
            turn_count=state.get("turn_count", 0),
        )
        return
    if not user_id:
        logger.error(
            "session_save_skipped_no_user_id",
            session_id=session_id,
            turn_count=state.get("turn_count", 0),
            message_count=len(state.get("messages", [])),
        )
        return

    try:
        # messages: 대화 이력 (truncation 적용)
        messages = list(state.get("messages", []))
        if len(messages) > MAX_CONVERSATION_TURNS * 2:
            # user+assistant 쌍 기준으로 최근 MAX_CONVERSATION_TURNS 턴만 유지
            messages = messages[-(MAX_CONVERSATION_TURNS * 2):]

        # [가드] 빈 messages → 저장 스킵 (Backend @NotBlank 검증 실패 방지)
        if not messages:
            logger.warning("session_save_skipped_no_messages", session_id=session_id)
            return

        # session_state: preferences, emotion, user_profile, watch_history 를 묶어 저장
        session_state: dict[str, Any] = {}

        # preferences: Pydantic → dict 직렬화
        prefs = state.get("preferences")
        if prefs is not None and hasattr(prefs, "model_dump"):
            session_state["preferences"] = prefs.model_dump()
        elif isinstance(prefs, dict):
            session_state["preferences"] = prefs
        else:
            session_state["preferences"] = None

        # emotion: Pydantic → dict 직렬화
        emotion = state.get("emotion")
        if emotion is not None and hasattr(emotion, "model_dump"):
            session_state["emotion"] = emotion.model_dump()
        elif isinstance(emotion, dict):
            session_state["emotion"] = emotion
        else:
            session_state["emotion"] = None

        # user_profile: dict (MySQL 에서 로드된 프로필 캐싱)
        session_state["user_profile"] = state.get("user_profile", {})

        # watch_history: list[dict] (datetime 객체 직렬화 대응)
        watch_history = state.get("watch_history", [])
        serializable_history = []
        for wh in watch_history:
            item = dict(wh)
            if "watched_at" in item and hasattr(item["watched_at"], "isoformat"):
                item["watched_at"] = item["watched_at"].isoformat()
            serializable_history.append(item)
        session_state["watch_history"] = serializable_history

        # recent_recommended_ids: 세션 내 최근 추천 영화 ID 롤링 윈도우.
        # query_builder 가 다음 턴 exclude_ids 에 병합해 같은 영화 반복 추천을 방지한다
        # (2026-04-24 버그: "한 편 더 추천" 시 동일 포스터가 반복 노출).
        recent_recommended_ids = state.get("recent_recommended_ids") or []
        session_state["recent_recommended_ids"] = [
            str(rid) for rid in recent_recommended_ids if rid
        ]

        # JSON 직렬화
        messages_json = json.dumps(messages, ensure_ascii=False, default=str)
        session_state_json = json.dumps(session_state, ensure_ascii=False, default=str)

        turn_count = state.get("turn_count", 0)

        # 1) Redis 즉시 반영 — 다음 턴에서 바로 읽도록 동기 await.
        #    Backend 응답 포맷과 동일한 필드명으로 저장해서 load 경로에서 분기 없이 재사용.
        redis_payload = {
            "sessionId": session_id,
            "messages": messages_json,
            "turnCount": turn_count,
            "sessionState": session_state_json,
            "intentSummary": None,
        }
        redis_ok = await _redis_set_session(user_id, session_id, redis_payload)

        # 2) Backend MySQL 에 async flush — fire-and-forget (SSE 루프 블로킹 방지)
        _flush_to_backend_in_background(
            user_id=user_id,
            session_id=session_id,
            messages_json=messages_json,
            turn_count=turn_count,
            session_state_json=session_state_json,
        )

        logger.info(
            "session_saved_hybrid",
            session_id=session_id,
            user_id=user_id,
            turn_count=turn_count,
            message_count=len(messages),
            redis_ok=redis_ok,
        )

    except Exception as e:
        # 예외 전파 금지 — 저장 실패가 응답 자체를 중단시키지 않게 한다.
        logger.error(
            "session_save_error",
            session_id=session_id,
            user_id=user_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
