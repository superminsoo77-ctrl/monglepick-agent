"""
MySQL 기반 세션 저장소 — 멀티턴 대화 상태 영속화.

매 턴마다 초기화되던 Chat Agent State를 Backend API를 통해 MySQL에 저장/복원하여
대화 맥락(messages, preferences, emotion, turn_count 등)을 유지한다.

Redis 대체: MySQL이 세션 데이터의 단일 진실 소스(Source of Truth)가 된다.
이전 채팅 목록 조회 및 이어하기 기능을 지원한다.

저장 대상 필드 (영속 필요):
- messages: 전체 대화 이력 → chat_session_archive.messages (JSON)
- preferences: 누적된 사용자 선호 (ExtractedPreferences) → session_state JSON 내부
- emotion: 마지막 감정 분석 결과 (EmotionResult) → session_state JSON 내부
- turn_count: 현재 턴 수 → chat_session_archive.turn_count
- user_profile: MySQL 유저 프로필 (세션 내 캐싱) → session_state JSON 내부
- watch_history: MySQL 시청 이력 (세션 내 캐싱) → session_state JSON 내부

저장하지 않는 필드 (매 턴 새로 도출):
- intent, search_query, candidate_movies, ranked_movies
- response, clarification, retrieval_quality_passed, retrieval_feedback
- image_data, image_analysis, error

Backend API:
  - POST /api/v1/chat/internal/session/save  — 세션 upsert
  - POST /api/v1/chat/internal/session/load  — 세션 로드
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from monglepick.agents.chat.models import EmotionResult, ExtractedPreferences
from monglepick.api.chat_client import (
    load_session_from_backend,
    save_session_to_backend,
)

logger = structlog.get_logger()

# ── 세션 설정 상수 ──

# 최대 대화 턴 수: 이 값을 초과하면 messages 앞부분을 자동 truncation
MAX_CONVERSATION_TURNS: int = 20


async def load_session(user_id: str, session_id: str) -> dict[str, Any] | None:
    """
    Backend API를 통해 MySQL에서 세션 데이터를 로드한다.

    Args:
        user_id: 사용자 ID
        session_id: 세션 ID

    Returns:
        dict (messages, preferences, emotion, turn_count, user_profile, watch_history)
        또는 None (세션 없음/에러)
    """
    if not session_id:
        return None

    try:
        # Backend API로 세션 로드
        raw = await load_session_from_backend(user_id, session_id)
        if raw is None:
            logger.debug("session_load_miss", session_id=session_id)
            return None

        # Backend 응답에서 messages JSON 파싱
        messages_str = raw.get("messages", "[]")
        messages = json.loads(messages_str) if isinstance(messages_str, str) else messages_str

        # sessionState JSON 파싱 → 개별 필드로 풀기
        session_state_str = raw.get("sessionState")
        session_state: dict[str, Any] = {}
        if session_state_str:
            session_state = json.loads(session_state_str) if isinstance(session_state_str, str) else session_state_str

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

        data: dict[str, Any] = {
            "messages": messages,
            "preferences": preferences,
            "emotion": emotion,
            "turn_count": raw.get("turnCount", 0),
            "user_profile": session_state.get("user_profile", {}),
            "watch_history": session_state.get("watch_history", []),
        }

        logger.info(
            "session_loaded",
            session_id=session_id,
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
    그래프 실행 완료 후 Backend API를 통해 MySQL에 세션 데이터를 저장한다.

    저장 필드 (6개):
    - messages: list[dict]        — 전체 대화 이력 → messages JSON
    - preferences: dict | None    — ExtractedPreferences → session_state 내부
    - emotion: dict | None        — EmotionResult → session_state 내부
    - turn_count: int             — 현재 턴 수
    - user_profile: dict          — MySQL 유저 프로필 (캐싱) → session_state 내부
    - watch_history: list[dict]   — MySQL 시청 이력 (캐싱) → session_state 내부

    MAX_CONVERSATION_TURNS(20) 초과 시 messages 앞부분 자동 truncation.

    Args:
        user_id: 사용자 ID
        session_id: 세션 ID
        state: 그래프 실행 완료 후의 전체 State dict
    """
    # [FIX] user_id도 빈 문자열이면 저장 불가 (Backend @NotBlank 검증 실패)
    if not session_id:
        logger.warning("session_save_skipped_no_session_id")
        return
    if not user_id:
        logger.warning("session_save_skipped_no_user_id", session_id=session_id)
        return

    try:
        # messages: 대화 이력 (truncation 적용)
        messages = list(state.get("messages", []))
        if len(messages) > MAX_CONVERSATION_TURNS * 2:
            # user+assistant 쌍 기준으로 최근 MAX_CONVERSATION_TURNS 턴만 유지
            messages = messages[-(MAX_CONVERSATION_TURNS * 2):]

        # [FIX] messages가 비어있으면 저장하지 않음 (Backend @NotBlank 검증 실패 방지)
        if not messages:
            logger.warning("session_save_skipped_no_messages", session_id=session_id)
            return

        # session_state: preferences, emotion, user_profile, watch_history를 묶어 저장
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

        # user_profile: dict (MySQL에서 로드된 프로필 캐싱)
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

        # JSON 직렬화
        messages_json = json.dumps(messages, ensure_ascii=False, default=str)
        session_state_json = json.dumps(session_state, ensure_ascii=False, default=str)

        turn_count = state.get("turn_count", 0)

        # Backend API로 세션 저장
        result = await save_session_to_backend(
            user_id=user_id,
            session_id=session_id,
            messages=messages_json,
            turn_count=turn_count,
            title=None,  # Backend가 첫 턴에서 자동 생성
            session_state=session_state_json,
            intent_summary=None,
        )

        # [FIX] 저장 성공/실패를 명확히 구분하여 로깅.
        # 기존에는 save_session_to_backend가 None을 반환해도 "session_saved"로 로깅되어
        # 실제로 저장이 실패했는지 성공했는지 로그만으로는 구분이 불가능했음.
        if result is not None:
            logger.info(
                "session_saved_ok",
                session_id=session_id,
                user_id=user_id,
                turn_count=turn_count,
                message_count=len(messages),
                created=result.get("created", False),
            )
        else:
            logger.error(
                "session_save_returned_none",
                session_id=session_id,
                user_id=user_id,
                turn_count=turn_count,
                message_count=len(messages),
            )

    except Exception as e:
        # [FIX] 로그 레벨 error + 스택트레이스
        logger.error(
            "session_save_error",
            session_id=session_id,
            user_id=user_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
