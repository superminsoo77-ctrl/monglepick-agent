"""
Redis 세션 저장소 단위 테스트.

load_session(), save_session() 함수의 동작을 검증한다.
Redis 서버 없이 mock 기반으로 실행한다.

테스트 항목:
1. load_session: 빈 세션 ID → None
2. load_session: 세션 없음 → None
3. load_session: 정상 로드 + Pydantic 복원
4. load_session: Redis 에러 → None (graceful)
5. save_session: 빈 세션 ID → no-op
6. save_session: 정상 저장 (Pydantic 직렬화)
7. save_session: messages truncation (MAX_CONVERSATION_TURNS 초과)
8. save_session: datetime 직렬화
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from monglepick.agents.chat.models import EmotionResult, ExtractedPreferences


# ============================================================
# load_session 테스트
# ============================================================

class TestLoadSession:
    """load_session() 단위 테스트."""

    @pytest.mark.asyncio
    async def test_empty_session_id_returns_none(self):
        """빈 세션 ID → None 반환 (Redis 호출 없음)."""
        from monglepick.memory.session_store import load_session

        result = await load_session("")
        assert result is None

    @pytest.mark.asyncio
    async def test_session_not_found_returns_none(self):
        """Redis에 세션 없음 → None 반환."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        with patch("monglepick.memory.session_store.get_redis", return_value=mock_redis):
            from monglepick.memory.session_store import load_session
            result = await load_session("nonexistent-session")

        assert result is None
        mock_redis.get.assert_called_once_with("session:nonexistent-session")

    @pytest.mark.asyncio
    async def test_load_restores_pydantic_models(self):
        """정상 로드 시 preferences/emotion이 Pydantic 모델로 복원된다."""
        session_data = {
            "messages": [{"role": "user", "content": "영화 추천해줘"}],
            "preferences": {"genre_preference": "SF", "mood": "웅장한"},
            "emotion": {"emotion": "happy", "mood_tags": ["유쾌"]},
            "turn_count": 1,
            "user_profile": {"user_id": "test"},
            "watch_history": [],
        }

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(session_data))
        mock_redis.expire = AsyncMock()

        with patch("monglepick.memory.session_store.get_redis", return_value=mock_redis):
            from monglepick.memory.session_store import load_session
            result = await load_session("test-session")

        assert result is not None
        # Pydantic 모델로 복원 확인
        assert isinstance(result["preferences"], ExtractedPreferences)
        assert result["preferences"].genre_preference == "SF"
        assert isinstance(result["emotion"], EmotionResult)
        assert result["emotion"].emotion == "happy"
        # 기타 필드 확인
        assert result["turn_count"] == 1
        assert len(result["messages"]) == 1
        # TTL 갱신 확인
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_with_null_preferences_emotion(self):
        """preferences/emotion이 None인 세션 로드."""
        session_data = {
            "messages": [],
            "preferences": None,
            "emotion": None,
            "turn_count": 0,
            "user_profile": {},
            "watch_history": [],
        }

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps(session_data))
        mock_redis.expire = AsyncMock()

        with patch("monglepick.memory.session_store.get_redis", return_value=mock_redis):
            from monglepick.memory.session_store import load_session
            result = await load_session("test-session")

        assert result is not None
        assert result["preferences"] is None
        assert result["emotion"] is None

    @pytest.mark.asyncio
    async def test_redis_error_returns_none(self):
        """Redis 에러 시 None 반환 (graceful degradation)."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=ConnectionError("Redis 연결 실패"))

        with patch("monglepick.memory.session_store.get_redis", return_value=mock_redis):
            from monglepick.memory.session_store import load_session
            result = await load_session("error-session")

        assert result is None


# ============================================================
# save_session 테스트
# ============================================================

class TestSaveSession:
    """save_session() 단위 테스트."""

    @pytest.mark.asyncio
    async def test_empty_session_id_noop(self):
        """빈 세션 ID → 저장하지 않음 (Redis 호출 없음)."""
        from monglepick.memory.session_store import save_session

        # Redis mock 없이 호출 — 에러 없이 반환되어야 함
        await save_session("", {"messages": []})

    @pytest.mark.asyncio
    async def test_save_serializes_pydantic_models(self):
        """Pydantic 모델(ExtractedPreferences, EmotionResult)이 dict로 직렬화된다."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()

        state = {
            "messages": [{"role": "user", "content": "테스트"}],
            "preferences": ExtractedPreferences(genre_preference="SF", mood="웅장한"),
            "emotion": EmotionResult(emotion="happy", mood_tags=["유쾌"]),
            "turn_count": 1,
            "user_profile": {},
            "watch_history": [],
        }

        with patch("monglepick.memory.session_store.get_redis", return_value=mock_redis):
            from monglepick.memory.session_store import save_session
            await save_session("test-session", state)

        # Redis.set이 호출되었는지 확인
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args

        # 저장된 JSON 파싱
        saved_key = call_args[0][0]
        saved_json = call_args[0][1]
        saved_data = json.loads(saved_json)

        assert saved_key == "session:test-session"
        # preferences가 dict로 직렬화됨
        assert isinstance(saved_data["preferences"], dict)
        assert saved_data["preferences"]["genre_preference"] == "SF"
        # emotion이 dict로 직렬화됨
        assert isinstance(saved_data["emotion"], dict)
        assert saved_data["emotion"]["emotion"] == "happy"

    @pytest.mark.asyncio
    async def test_save_truncates_long_messages(self):
        """MAX_CONVERSATION_TURNS(20) 초과 시 messages 앞부분이 잘린다."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()

        # 25턴 분량의 메시지 (50개: user+assistant 쌍)
        messages = []
        for i in range(25):
            messages.append({"role": "user", "content": f"질문 {i}"})
            messages.append({"role": "assistant", "content": f"응답 {i}"})

        state = {
            "messages": messages,
            "preferences": None,
            "emotion": None,
            "turn_count": 25,
            "user_profile": {},
            "watch_history": [],
        }

        with patch("monglepick.memory.session_store.get_redis", return_value=mock_redis):
            from monglepick.memory.session_store import save_session
            await save_session("test-session", state)

        # 저장된 JSON 파싱
        saved_json = mock_redis.set.call_args[0][1]
        saved_data = json.loads(saved_json)

        # MAX_CONVERSATION_TURNS * 2 = 40개로 잘림
        assert len(saved_data["messages"]) == 40

    @pytest.mark.asyncio
    async def test_save_handles_datetime(self):
        """watch_history의 datetime 객체가 isoformat 문자열로 변환된다."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()

        now = datetime(2026, 3, 4, 12, 0, 0)
        state = {
            "messages": [],
            "preferences": None,
            "emotion": None,
            "turn_count": 0,
            "user_profile": {},
            "watch_history": [
                {"movie_id": "1", "title": "인셉션", "watched_at": now},
            ],
        }

        with patch("monglepick.memory.session_store.get_redis", return_value=mock_redis):
            from monglepick.memory.session_store import save_session
            await save_session("test-session", state)

        saved_json = mock_redis.set.call_args[0][1]
        saved_data = json.loads(saved_json)

        # datetime이 isoformat 문자열로 변환됨
        assert saved_data["watch_history"][0]["watched_at"] == "2026-03-04T12:00:00"
