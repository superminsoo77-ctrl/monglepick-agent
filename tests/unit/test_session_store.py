"""
세션 저장소 단위 테스트.

load_session(), save_session() 함수의 동작을 검증한다.
Backend API(load_session_from_backend / save_session_to_backend) 와
Redis 클라이언트(get_redis) 를 mock 으로 대체하여 외부 의존성 없이 실행한다.

아키텍처 변경 (2026-04-14, Option B: Redis 캐시 + MySQL 아카이브 write-behind):
  - load_session: Redis 우선 → miss 시 Backend 폴백 + Redis 재적재
  - save_session: Redis 동기 쓰기 + Backend 는 fire-and-forget 백그라운드 task
    → 테스트에서는 `_wait_for_pending_flushes()` 로 flush 완료 후 assertion.

테스트 항목:
1. load_session: 빈 세션 ID → None
2. load_session: 빈 user_id → None (Backend @NotBlank 가드)
3. load_session: Redis miss + Backend 도 None → None
4. load_session: Redis miss + Backend 히트 → Pydantic 복원 + Redis 재적재
5. load_session: Redis 히트 → Backend 호출 없음
6. load_session: preferences/emotion 이 None 인 세션 로드
7. load_session: Backend 에러 → None (graceful)
8. save_session: 빈 세션 ID → no-op
9. save_session: 빈 user_id → no-op
10. save_session: Redis 쓰기 + Backend flush (백그라운드 task)
11. save_session: messages truncation (MAX_CONVERSATION_TURNS 초과)
12. save_session: datetime 직렬화
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monglepick.agents.chat.models import EmotionResult, ExtractedPreferences


# ============================================================
# 공통 fixture — Redis 를 인메모리 dict 로 mock
# ============================================================

@pytest.fixture
def mock_redis():
    """
    `monglepick.memory.session_store.get_redis` 를 패치하여
    인메모리 dict 기반 mock Redis 를 반환한다.

    get/set/(ex 옵션 포함) 만 구현 — 세션 저장소가 사용하는 연산만 커버.
    """
    store: dict[str, str] = {}

    async def _get(key: str):
        return store.get(key)

    async def _set(key: str, value: str, ex: int | None = None, **_):
        store[key] = value
        return True

    client = MagicMock()
    client.get = AsyncMock(side_effect=_get)
    client.set = AsyncMock(side_effect=_set)

    async def _get_redis():
        return client

    with patch(
        "monglepick.memory.session_store.get_redis",
        new=AsyncMock(side_effect=_get_redis),
    ):
        # store 도 외부에서 접근 가능하도록 fixture 로 노출
        yield {"client": client, "store": store}


# ============================================================
# load_session 테스트
# ============================================================

class TestLoadSession:
    """load_session() 단위 테스트."""

    @pytest.mark.asyncio
    async def test_empty_session_id_returns_none(self, mock_redis):
        """빈 세션 ID → None 반환 (Redis/Backend 호출 없음)."""
        from monglepick.memory.session_store import load_session

        result = await load_session("user-1", "")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_user_id_returns_none(self, mock_redis):
        """빈 user_id → None 반환 (Backend @NotBlank 400 방지 가드)."""
        with patch(
            "monglepick.memory.session_store.load_session_from_backend",
            new=AsyncMock(),
        ) as mock_load:
            from monglepick.memory.session_store import load_session
            result = await load_session("", "some-session")

        assert result is None
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_not_found_returns_none(self, mock_redis):
        """Redis miss + Backend도 None → None 반환."""
        with patch(
            "monglepick.memory.session_store.load_session_from_backend",
            new=AsyncMock(return_value=None),
        ) as mock_load:
            from monglepick.memory.session_store import load_session
            result = await load_session("user-1", "nonexistent-session")

        assert result is None
        mock_load.assert_called_once_with("user-1", "nonexistent-session")

    @pytest.mark.asyncio
    async def test_load_restores_pydantic_models_from_backend(self, mock_redis):
        """Redis miss → Backend 정상 로드 시 preferences/emotion이 Pydantic 모델로 복원된다."""
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

        # Backend 응답이 Redis 에 재적재되었는지 확인 (cache-aside 패턴)
        assert "chat:session:user-1:test-session" in mock_redis["store"]

    @pytest.mark.asyncio
    async def test_load_redis_hit_skips_backend(self, mock_redis):
        """Redis 히트 시 Backend 호출하지 않는다 (핫 캐시 정상 동작)."""
        # Redis 에 미리 세션 데이터를 적재
        session_state = {
            "preferences": None,
            "emotion": None,
            "user_profile": {},
            "watch_history": [],
        }
        cached = {
            "sessionId": "test-session",
            "messages": json.dumps([{"role": "user", "content": "캐시된 대화"}]),
            "sessionState": json.dumps(session_state),
            "turnCount": 3,
        }
        mock_redis["store"]["chat:session:user-1:test-session"] = json.dumps(cached)

        mock_backend_load = AsyncMock(return_value=None)
        with patch(
            "monglepick.memory.session_store.load_session_from_backend",
            new=mock_backend_load,
        ):
            from monglepick.memory.session_store import load_session
            result = await load_session("user-1", "test-session")

        assert result is not None
        assert result["turn_count"] == 3
        assert result["messages"][0]["content"] == "캐시된 대화"
        # Backend 호출되지 않음
        mock_backend_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_with_null_preferences_emotion(self, mock_redis):
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
    async def test_backend_error_returns_none(self, mock_redis):
        """Redis miss + Backend 에러 시 None 반환 (graceful degradation)."""
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
    async def test_empty_session_id_noop(self, mock_redis):
        """빈 세션 ID → 저장하지 않음 (Redis/Backend 호출 없음)."""
        with patch(
            "monglepick.memory.session_store.save_session_to_backend",
            new=AsyncMock(),
        ) as mock_save:
            from monglepick.memory.session_store import (
                save_session,
                _wait_for_pending_flushes,
            )
            await save_session("user-1", "", {"messages": []})
            await _wait_for_pending_flushes()

        mock_save.assert_not_called()
        assert len(mock_redis["store"]) == 0

    @pytest.mark.asyncio
    async def test_empty_user_id_noop(self, mock_redis):
        """빈 user_id → 저장하지 않음."""
        with patch(
            "monglepick.memory.session_store.save_session_to_backend",
            new=AsyncMock(),
        ) as mock_save:
            from monglepick.memory.session_store import (
                save_session,
                _wait_for_pending_flushes,
            )
            await save_session("", "test-session", {"messages": [{"role": "user", "content": "x"}]})
            await _wait_for_pending_flushes()

        mock_save.assert_not_called()
        assert len(mock_redis["store"]) == 0

    @pytest.mark.asyncio
    async def test_save_writes_redis_and_schedules_backend_flush(self, mock_redis):
        """Pydantic 직렬화 + Redis 즉시 쓰기 + Backend fire-and-forget flush."""
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
            from monglepick.memory.session_store import (
                save_session,
                _wait_for_pending_flushes,
            )
            await save_session("user-1", "test-session", state)
            # fire-and-forget 백그라운드 task 완료 대기
            await _wait_for_pending_flushes()

        # Redis 에 즉시 반영되었는지 확인
        assert "chat:session:user-1:test-session" in mock_redis["store"]
        redis_payload = json.loads(mock_redis["store"]["chat:session:user-1:test-session"])
        assert redis_payload["sessionId"] == "test-session"
        assert redis_payload["turnCount"] == 1

        # Backend flush 가 호출되었는지 확인 (fire-and-forget)
        mock_save.assert_called_once()
        kwargs = mock_save.call_args.kwargs

        assert kwargs["user_id"] == "user-1"
        assert kwargs["session_id"] == "test-session"
        assert kwargs["turn_count"] == 1

        # session_state JSON 파싱 후 Pydantic 이 dict 로 직렬화되었는지 확인
        session_state = json.loads(kwargs["session_state"])
        assert isinstance(session_state["preferences"], dict)
        assert session_state["preferences"]["genre_preference"] == "SF"
        assert isinstance(session_state["emotion"], dict)
        assert session_state["emotion"]["emotion"] == "happy"

        # messages JSON 도 정상 직렬화
        messages = json.loads(kwargs["messages"])
        assert len(messages) == 1
        assert messages[0]["content"] == "테스트"

    @pytest.mark.asyncio
    async def test_save_truncates_long_messages(self, mock_redis):
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
            from monglepick.memory.session_store import (
                save_session,
                _wait_for_pending_flushes,
            )
            await save_session("user-1", "test-session", state)
            await _wait_for_pending_flushes()

        kwargs = mock_save.call_args.kwargs
        saved_messages = json.loads(kwargs["messages"])
        # MAX_CONVERSATION_TURNS * 2 = 40개로 잘림
        assert len(saved_messages) == 40
        # 가장 마지막 메시지는 보존되어야 함
        assert saved_messages[-1]["content"] == "응답 24"

    @pytest.mark.asyncio
    async def test_save_handles_datetime(self, mock_redis):
        """watch_history의 datetime 객체가 isoformat 문자열로 변환된다."""
        now = datetime(2026, 3, 4, 12, 0, 0)
        state = {
            "messages": [{"role": "user", "content": "x"}],
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
            from monglepick.memory.session_store import (
                save_session,
                _wait_for_pending_flushes,
            )
            await save_session("user-1", "test-session", state)
            await _wait_for_pending_flushes()

        kwargs = mock_save.call_args.kwargs
        session_state = json.loads(kwargs["session_state"])
        # datetime이 isoformat 문자열로 변환됨
        assert session_state["watch_history"][0]["watched_at"] == "2026-03-04T12:00:00"
