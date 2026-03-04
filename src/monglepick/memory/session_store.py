"""
Redis 기반 세션 저장소 — 멀티턴 대화 상태 영속화.

매 턴마다 초기화되던 Chat Agent State를 Redis에 저장/복원하여
대화 맥락(messages, preferences, emotion, turn_count 등)을 유지한다.

저장 대상 필드 (영속 필요):
- messages: 전체 대화 이력
- preferences: 누적된 사용자 선호 (ExtractedPreferences)
- emotion: 마지막 감정 분석 결과 (EmotionResult)
- turn_count: 현재 턴 수
- user_profile: MySQL 유저 프로필 (세션 내 캐싱)
- watch_history: MySQL 시청 이력 (세션 내 캐싱)

저장하지 않는 필드 (매 턴 새로 도출):
- intent, search_query, candidate_movies, ranked_movies
- response, clarification, retrieval_quality_passed, retrieval_feedback
- image_data, image_analysis, error

Redis 키 형식: "session:{session_id}" (JSON 직렬화)
TTL: SESSION_TTL_SECONDS (기본 30일)
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from monglepick.agents.chat.models import EmotionResult, ExtractedPreferences
from monglepick.db.clients import get_redis

logger = structlog.get_logger()

# ── 세션 설정 상수 ──

# 세션 TTL: 30일 (초 단위)
SESSION_TTL_SECONDS: int = 30 * 24 * 60 * 60  # 2,592,000초

# 최대 대화 턴 수: 이 값을 초과하면 messages 앞부분을 자동 truncation
MAX_CONVERSATION_TURNS: int = 20

# Redis 키 접두사
SESSION_KEY_PREFIX: str = "session:"

# 세션에 저장할 필드 목록 (이 필드들만 Redis에 직렬화)
_PERSIST_FIELDS: set[str] = {
    "messages",
    "preferences",
    "emotion",
    "turn_count",
    "user_profile",
    "watch_history",
}


def _session_key(session_id: str) -> str:
    """세션 ID로 Redis 키를 생성한다."""
    return f"{SESSION_KEY_PREFIX}{session_id}"


async def load_session(session_id: str) -> dict[str, Any] | None:
    """
    Redis에서 세션 데이터를 로드한다.

    키: "session:{session_id}" (JSON 문자열)
    TTL 자동 갱신: 접근할 때마다 SESSION_TTL_SECONDS로 TTL 리셋

    Args:
        session_id: 세션 ID

    Returns:
        dict (messages, preferences, emotion, turn_count, user_profile, watch_history)
        또는 None (세션 없음/만료/에러)
    """
    if not session_id:
        return None

    try:
        redis = await get_redis()
        key = _session_key(session_id)

        # Redis에서 JSON 문자열 조회
        raw = await redis.get(key)
        if raw is None:
            logger.debug("session_load_miss", session_id=session_id)
            return None

        # JSON 파싱
        data: dict[str, Any] = json.loads(raw)

        # Pydantic 모델 복원: dict → ExtractedPreferences / EmotionResult
        if data.get("preferences") is not None:
            data["preferences"] = ExtractedPreferences(**data["preferences"])
        if data.get("emotion") is not None:
            data["emotion"] = EmotionResult(**data["emotion"])

        # TTL 갱신: 접근할 때마다 만료 시간을 리셋
        await redis.expire(key, SESSION_TTL_SECONDS)

        logger.info(
            "session_loaded",
            session_id=session_id,
            turn_count=data.get("turn_count", 0),
            message_count=len(data.get("messages", [])),
            has_preferences=data.get("preferences") is not None,
            has_emotion=data.get("emotion") is not None,
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


async def save_session(session_id: str, state: dict[str, Any]) -> None:
    """
    그래프 실행 완료 후 세션 데이터를 Redis에 저장한다.

    저장 필드 (6개):
    - messages: list[dict]        — 전체 대화 이력
    - preferences: dict | None    — ExtractedPreferences.model_dump()
    - emotion: dict | None        — EmotionResult.model_dump()
    - turn_count: int             — 현재 턴 수
    - user_profile: dict          — MySQL 유저 프로필 (캐싱)
    - watch_history: list[dict]   — MySQL 시청 이력 (캐싱)

    MAX_CONVERSATION_TURNS(20) 초과 시 messages 앞부분 자동 truncation.
    TTL: SESSION_TTL_SECONDS (기본 30일)

    Args:
        session_id: 세션 ID
        state: 그래프 실행 완료 후의 전체 State dict
    """
    if not session_id:
        return

    try:
        # 영속 필드만 추출
        session_data: dict[str, Any] = {}

        # messages: 대화 이력 (truncation 적용)
        messages = list(state.get("messages", []))
        if len(messages) > MAX_CONVERSATION_TURNS * 2:
            # user+assistant 쌍 기준으로 최근 MAX_CONVERSATION_TURNS 턴만 유지
            # 첫 번째 메시지(시스템/초기)는 보존하고 중간을 잘라냄
            messages = messages[-(MAX_CONVERSATION_TURNS * 2):]
        session_data["messages"] = messages

        # preferences: Pydantic → dict 직렬화
        prefs = state.get("preferences")
        if prefs is not None and hasattr(prefs, "model_dump"):
            session_data["preferences"] = prefs.model_dump()
        elif isinstance(prefs, dict):
            session_data["preferences"] = prefs
        else:
            session_data["preferences"] = None

        # emotion: Pydantic → dict 직렬화
        emotion = state.get("emotion")
        if emotion is not None and hasattr(emotion, "model_dump"):
            session_data["emotion"] = emotion.model_dump()
        elif isinstance(emotion, dict):
            session_data["emotion"] = emotion
        else:
            session_data["emotion"] = None

        # turn_count: 정수
        session_data["turn_count"] = state.get("turn_count", 0)

        # user_profile: dict (MySQL에서 로드된 프로필 캐싱)
        session_data["user_profile"] = state.get("user_profile", {})

        # watch_history: list[dict] (MySQL에서 로드된 시청 이력 캐싱)
        watch_history = state.get("watch_history", [])
        # datetime 객체 직렬화 대응: watched_at이 datetime이면 isoformat으로 변환
        serializable_history = []
        for wh in watch_history:
            item = dict(wh)
            if "watched_at" in item and hasattr(item["watched_at"], "isoformat"):
                item["watched_at"] = item["watched_at"].isoformat()
            serializable_history.append(item)
        session_data["watch_history"] = serializable_history

        # Redis에 JSON 직렬화하여 저장 (TTL 설정)
        redis = await get_redis()
        key = _session_key(session_id)
        await redis.set(
            key,
            json.dumps(session_data, ensure_ascii=False, default=str),
            ex=SESSION_TTL_SECONDS,
        )

        logger.info(
            "session_saved",
            session_id=session_id,
            turn_count=session_data["turn_count"],
            message_count=len(session_data["messages"]),
            has_preferences=session_data["preferences"] is not None,
            has_emotion=session_data["emotion"] is not None,
        )

    except Exception as e:
        # 세션 저장 실패는 대화 흐름에 영향을 주지 않는다 (로그만 남김)
        logger.warning(
            "session_save_error",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
        )
