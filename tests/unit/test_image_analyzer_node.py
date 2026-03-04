"""
image_analyzer 노드 단위 테스트.

image_analyzer 노드의 패스스루, 정상 분석, 에러 복원력을 테스트한다.
"""

from __future__ import annotations

import pytest

from monglepick.agents.chat.models import ChatAgentState, ImageAnalysisResult
from monglepick.agents.chat.nodes import image_analyzer


class TestImageAnalyzerNode:
    """image_analyzer 노드 테스트."""

    @pytest.mark.asyncio
    async def test_passthrough_when_no_image(self, mock_ollama, mock_mysql):
        """이미지 데이터 없으면 패스스루 (analyzed=False)."""
        state: ChatAgentState = {
            "user_id": "test",
            "session_id": "sess1",
            "current_input": "영화 추천해줘",
            "image_data": None,
            "messages": [],
        }
        result = await image_analyzer(state)
        assert "image_analysis" in result
        assert result["image_analysis"].analyzed is False

    @pytest.mark.asyncio
    async def test_passthrough_when_empty_image(self, mock_ollama, mock_mysql):
        """빈 문자열 이미지 데이터도 패스스루."""
        state: ChatAgentState = {
            "user_id": "test",
            "session_id": "sess1",
            "current_input": "영화 추천해줘",
            "image_data": "",
            "messages": [],
        }
        result = await image_analyzer(state)
        assert result["image_analysis"].analyzed is False

    @pytest.mark.asyncio
    async def test_analyze_with_image(self, mock_image_analysis, mock_ollama, mock_mysql):
        """이미지 데이터가 있으면 분석을 수행한다."""
        state: ChatAgentState = {
            "user_id": "test",
            "session_id": "sess1",
            "current_input": "이런 분위기 영화 추천해줘",
            "image_data": "base64encodedimage",
            "messages": [],
        }
        result = await image_analyzer(state)
        analysis = result["image_analysis"]
        assert analysis.analyzed is True
        assert len(analysis.genre_cues) > 0

    @pytest.mark.asyncio
    async def test_error_returns_not_analyzed(self, mock_ollama, mock_mysql):
        """analyze_image 에러 시 analyzed=False 반환 (에러 전파 금지)."""
        from unittest.mock import AsyncMock, patch

        # analyze_image가 예외를 발생시키도록 mock
        with patch(
            "monglepick.agents.chat.nodes.analyze_image",
            side_effect=RuntimeError("VLM error"),
        ):
            state: ChatAgentState = {
                "user_id": "test",
                "session_id": "sess1",
                "current_input": "추천해줘",
                "image_data": "base64data",
                "messages": [],
            }
            result = await image_analyzer(state)
            assert result["image_analysis"].analyzed is False
