"""
Chat API 엔드포인트 통합 테스트 (Phase 3 + 의도+감정 통합).

httpx + ASGITransport로 FastAPI 앱을 직접 테스트한다.
SSE/sync 엔드포인트 동작과 유효성 검사를 검증한다.

테스트 시나리오:
1. SSE 엔드포인트 → text/event-stream 응답
2. 동기 엔드포인트 → JSON 응답 (clarification 필드 포함)
3. 빈 메시지 → 422 Validation Error
4. 긴 메시지(2000자 초과) → 422 Validation Error
5. health 엔드포인트 → 200 OK
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from monglepick.agents.chat.models import (
    ExtractedPreferences,
    IntentEmotionResult,
)


@pytest.fixture
def mock_all_deps():
    """
    Chat Agent의 모든 외부 의존성을 mock하는 fixture.

    lifespan의 init_all_clients/close_all_clients도 패치하여 DB 연결을 방지한다.
    의도+감정 통합 후: classify_intent + analyze_emotion → classify_intent_and_emotion.
    """
    patches = [
        # lifespan DB 초기화/종료 패치
        patch("monglepick.main.init_all_clients", new_callable=AsyncMock),
        patch("monglepick.main.close_all_clients", new_callable=AsyncMock),
        # 노드 의존성 패치
        patch(
            "monglepick.agents.chat.nodes.get_mysql",
            side_effect=Exception("mock"),
        ),
        # 통합 의도+감정 체인 mock (기존 classify_intent + analyze_emotion 대체)
        patch(
            "monglepick.agents.chat.nodes.classify_intent_and_emotion",
            new_callable=AsyncMock,
            return_value=IntentEmotionResult(
                intent="general", confidence=0.8,
                emotion=None, mood_tags=[],
            ),
        ),
        patch(
            "monglepick.agents.chat.nodes.extract_preferences",
            new_callable=AsyncMock,
            return_value=ExtractedPreferences(),
        ),
        patch(
            "monglepick.agents.chat.nodes.generate_question",
            new_callable=AsyncMock,
            return_value="테스트 질문",
        ),
        patch(
            "monglepick.agents.chat.nodes.generate_explanations_batch",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "monglepick.agents.chat.nodes.generate_general_response",
            new_callable=AsyncMock,
            return_value="안녕하세요! 몽글이에요!",
        ),
        patch(
            "monglepick.agents.chat.nodes.hybrid_search",
            new_callable=AsyncMock,
            return_value=[],
        ),
        # 세션 저장소 mock (load → None, save → no-op)
        patch(
            "monglepick.agents.chat.graph.load_session",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "monglepick.agents.chat.graph.save_session",
            new_callable=AsyncMock,
        ),
    ]

    # 모든 패치를 동시에 적용
    started = [p.start() for p in patches]
    yield
    for p in patches:
        p.stop()


# ============================================================
# API 테스트
# ============================================================

class TestChatAPI:
    """Chat API 엔드포인트 테스트."""

    @pytest.mark.asyncio
    async def test_sync_endpoint(self, mock_all_deps):
        """동기 엔드포인트: POST /api/v1/chat/sync → JSON 응답 (clarification 필드 포함)."""
        from monglepick.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/sync",
                json={"message": "안녕"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["intent"] == "general"
        # session_id 자동 생성 확인 (빈 문자열 요청 → UUID 반환)
        assert "session_id" in data
        assert len(data["session_id"]) > 0
        # clarification 필드 존재 확인 (일반 대화 시 None)
        assert "clarification" in data
        assert data["clarification"] is None

    @pytest.mark.asyncio
    async def test_sse_endpoint(self, mock_all_deps):
        """SSE 엔드포인트: POST /api/v1/chat → text/event-stream."""
        from monglepick.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat",
                json={"message": "안녕"},
            )
        assert response.status_code == 200
        # SSE 응답의 content-type 확인
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type

    @pytest.mark.asyncio
    async def test_empty_message_validation(self, mock_all_deps):
        """빈 메시지 → 422 Validation Error."""
        from monglepick.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/sync",
                json={"message": ""},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_long_message_validation(self, mock_all_deps):
        """긴 메시지(2001자) → 422 Validation Error."""
        from monglepick.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/chat/sync",
                json={"message": "a" * 2001},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_health_endpoint(self, mock_all_deps):
        """헬스 체크: GET /health → 200 OK."""
        from monglepick.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.2.0"
