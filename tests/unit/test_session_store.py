"""
세션 저장소 단위 테스트.

load_session(), save_session() 함수의 동작을 검증한다.
Backend API(load_session_from_backend / save_session_to_backend)를
mock으로 대체하여 외부 의존성 없이 실행한다.

테스트 항목:
1. load_session: 빈 세션 ID → None
2. load_session: Backend가 None 반환 → None
3. load_session: 정상 로드 + Pydantic 복원
4. load_session: preferences/emotion이 None인 세션 로드
5. load_session: Backend 에러 → None (graceful)
6. save_session: 빈 세션 ID → no-op
7. save_session: 정상 저장 (Pydantic 직렬화)
8. save_session: messages truncation (MAX_CONVERSATION_TURNS 초과)
9. save_session: datetime 직렬화
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
        """빈 세션 ID → None 반환 (Backend 호출 없음)."""
        from monglepick.memory.session_store import load_session

        result = await load_session("user-1", "")
        assert result is None

    @pytest.mark.asyncio
    async def test_session_not_found_returns_none(self):
        """Backend가 None 반환 → None 반환."""
        with patch(
            "monglepick.memory.session_store.load_session_from_backend",
            new=AsyncMock(return_value=None),
        ) as mock_load:
            from monglepick.memory.session_store import load_session
            result = await load_session("user-1", "nonexistent-session")

        assert result is None
        mock_load.assert_called_once_with("user-1", "nonexistent-session")

    @pytest.mark.asyncio
    async def test_load_restores_pydantic_models(self):
        """정상 로드 시 preferences/emotion이 Pydantic 모델로 복원된다."""
        # Backend 응답 형식: messages/sessionState는 JSON 문자열
        session_state = {
            "preferences": {"genre_preference": "SF", "mood": "웅장한"},
            "emotion": {"emotion": "happy", "mood_tags": ["유쾌"]},
            "user_profile": {"user_id": "test"},
            "watch_history": [],
        }
        backend_response = {
            "sessionId": "test-session",
            "messages": json.dumps([{"role": "user", "content": "영화 추천해줘"}]),
            "sessionState": json.dumps(session_state),
            "turnCount": 1,
        }

        with patch(
            "monglepick.memory.session_store.load_session_from_backend",
            new=AsyncMock(return_value=backend_response),
        ):
            from monglepick.memory.session_store import load_session
            result = await load_session("user-1", "test-session")

        assert result is not None
        # Pydantic 모델로 복원 확인
        assert isinstance(result["preferences"], ExtractedPreferences)
        assert result["preferences"].genre_preference == "SF"
        assert isinstance(result["emotion"], EmotionResult)
        assert result["emotion"].emotion == "happy"
        # 기타 필드 확인
        assert result["turn_count"] == 1
        assert len(result["messages"]) == 1
        assert result["user_profile"] == {"user_id": "test"}
        assert result["watch_history"] == []

    @pytest.mark.asyncio
    async def test_load_with_null_preferences_emotion(self):
        """preferences/emotion이 None인 세션 로드."""
        session_state = {
            "preferences": None,
            "emotion": None,
            "user_profile": {},
            "watch_history": [],
        }
        backend_response = {
            "sessionId": "test-session",
            "messages": json.dumps([]),
            "sessionState": json.dumps(session_state),
            "turnCount": 0,
        }

        with patch(
            "monglepick.memory.session_store.load_session_from_backend",
            new=AsyncMock(return_value=backend_response),
        ):
            from monglepick.memory.session_store import load_session
            result = await load_session("user-1", "test-session")

        assert result is not None
        assert result["preferences"] is None
        assert result["emotion"] is None

    @pytest.mark.asyncio
    async def test_backend_error_returns_none(self):
        """Backend 에러 시 None 반환 (graceful degradation)."""
        with patch(
            "monglepick.memory.session_store.load_session_from_backend",
            new=AsyncMock(side_effect=ConnectionError("Backend 연결 실패")),
        ):
            from monglepick.memory.session_store import load_session
            result = await load_session("user-1", "error-session")

        assert result is None


# ============================================================
# save_session 테스트
# ============================================================

class TestSaveSession:
    """save_session() 단위 테스트."""

    @pytest.mark.asyncio
    async def test_empty_session_id_noop(self):
        """빈 세션 ID → 저장하지 않음 (Backend 호출 없음)."""
        with patch(
            "monglepick.memory.session_store.save_session_to_backend",
            new=AsyncMock(),
        ) as mock_save:
            from monglepick.memory.session_store import save_session
            await save_session("user-1", "", {"messages": []})

        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_serializes_pydantic_models(self):
        """Pydantic 모델(ExtractedPreferences, EmotionResult)이 dict로 직렬화된다."""
        state = {
            "messages": [{"role": "user", "content": "테스트"}],
            "preferences": ExtractedPreferences(genre_preference="SF", mood="웅장한"),
            "emotion": EmotionResult(emotion="happy", mood_tags=["유쾌"]),
            "turn_count": 1,
            "user_profile": {},
            "watch_history": [],
        }

        mock_save = AsyncMock(return_value={"created": True})
        with patch(
            "monglepick.memory.session_store.save_session_to_backend",
            new=mock_save,
        ):
            from monglepick.memory.session_store import save_session
            await save_session("user-1", "test-session", state)

        mock_save.assert_called_once()
        kwargs = mock_save.call_args.kwargs

        assert kwargs["user_id"] == "user-1"
        assert kwargs["session_id"] == "test-session"
        assert kwargs["turn_count"] == 1

        # session_state JSON 파싱 후 Pydantic이 dict로 직렬화되었는지 확인
        session_state = json.loads(kwargs["session_state"])
        assert isinstance(session_state["preferences"], dict)
        assert session_state["preferences"]["genre_preference"] == "SF"
        assert isinstance(session_state["emotion"], dict)
        assert session_state["emotion"]["emotion"] == "happy"

        # messages JSON도 정상 직렬화
        messages = json.loads(kwargs["messages"])
        assert len(messages) == 1
        assert messages[0]["content"] == "테스트"

    @pytest.mark.asyncio
    async def test_save_truncates_long_messages(self):
        """MAX_CONVERSATION_TURNS(20) 초과 시 messages 앞부분이 잘린다."""
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

        mock_save = AsyncMock(return_value={"created": False})
        with patch(
            "monglepick.memory.session_store.save_session_to_backend",
            new=mock_save,
        ):
            from monglepick.memory.session_store import save_session
            await save_session("user-1", "test-session", state)

        kwargs = mock_save.call_args.kwargs
        saved_messages = json.loads(kwargs["messages"])
        # MAX_CONVERSATION_TURNS * 2 = 40개로 잘림
        assert len(saved_messages) == 40
        # 가장 마지막 메시지는 보존되어야 함
        assert saved_messages[-1]["content"] == "응답 24"

    @pytest.mark.asyncio
    async def test_save_handles_datetime(self):
        """watch_history의 datetime 객체가 isoformat 문자열로 변환된다."""
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

        mock_save = AsyncMock(return_value={"created": True})
        with patch(
            "monglepick.memory.session_store.save_session_to_backend",
            new=mock_save,
        ):
            from monglepick.memory.session_store import save_session
            await save_session("user-1", "test-session", state)

        kwargs = mock_save.call_args.kwargs
        session_state = json.loads(kwargs["session_state"])
        # datetime이 isoformat 문자열로 변환됨
        assert session_state["watch_history"][0]["watched_at"] == "2026-03-04T12:00:00"
