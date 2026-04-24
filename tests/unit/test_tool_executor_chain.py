"""
tool_executor_chain 단위 테스트 (Phase 6 외부 지도 연동).

대상: monglepick.chains.tool_executor_chain.execute_tool

핵심 계약:
 1) 알 수 없는 intent → 빈 dict
 2) theater 의도 + location → theater_search + kobis_now_showing 모두 병렬 실행
 3) theater 의도 + location 누락 → theater_search 호출 스킵 (kobis_now_showing 만 실행)
 4) info 의도 + movie_id 없음 → movie_detail/ott/similar 모두 스킵, 키도 미존재
 5) 도구가 timeout → 결과는 빈 list 로 fallback (전체 실패 X)
 6) 도구가 예외 → 결과는 빈 list 로 fallback
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monglepick.chains.tool_executor_chain import execute_tool


def _make_async_tool(return_value):
    """ainvoke(...) 가 return_value 를 반환하는 가짜 LangChain Tool."""
    tool = MagicMock()
    tool.ainvoke = AsyncMock(return_value=return_value)
    return tool


def _make_failing_tool(exc: Exception):
    tool = MagicMock()
    tool.ainvoke = AsyncMock(side_effect=exc)
    return tool


class TestExecuteToolDispatch:
    @pytest.mark.asyncio
    async def test_unknown_intent_returns_empty(self):
        result = await execute_tool(intent="nonexistent")
        assert result == {}

    @pytest.mark.asyncio
    async def test_theater_intent_runs_both_tools(self):
        """theater 의도 + location 있음 → theater_search + kobis_now_showing 모두 실행."""
        fake_theaters = [{"name": "CGV 강남"}]
        fake_now_showing = [{"rank": 1, "movie_nm": "테스트"}]
        fake_registry = {
            "theater_search": _make_async_tool(fake_theaters),
            "kobis_now_showing": _make_async_tool(fake_now_showing),
        }
        with patch("monglepick.chains.tool_executor_chain.TOOL_REGISTRY", fake_registry):
            result = await execute_tool(
                intent="theater",
                location={"latitude": 37.5, "longitude": 127.0},
            )
        assert result["theater_search"] == fake_theaters
        assert result["kobis_now_showing"] == fake_now_showing
        # 도구별 호출 인자 검증
        fake_registry["theater_search"].ainvoke.assert_awaited_once()
        fake_registry["kobis_now_showing"].ainvoke.assert_awaited_once()
        ts_args = fake_registry["theater_search"].ainvoke.await_args.args[0]
        assert ts_args["latitude"] == 37.5
        assert ts_args["longitude"] == 127.0
        assert ts_args["radius"] == 5000  # 기본값

    @pytest.mark.asyncio
    async def test_theater_intent_no_location_skips_theater(self):
        """location 누락 → theater_search 호출 스킵, kobis_now_showing 만 실행."""
        fake_now_showing = [{"rank": 1, "movie_nm": "X"}]
        fake_registry = {
            "theater_search": _make_async_tool([{"name": "should not be called"}]),
            "kobis_now_showing": _make_async_tool(fake_now_showing),
        }
        with patch("monglepick.chains.tool_executor_chain.TOOL_REGISTRY", fake_registry):
            result = await execute_tool(intent="theater", location=None)
        assert "theater_search" not in result  # 호출 스킵 = 키 미존재
        assert result["kobis_now_showing"] == fake_now_showing
        fake_registry["theater_search"].ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_info_intent_no_movie_id_skips_detail_tools(self):
        """movie_id 없음 → movie_detail/ott/similar 모두 호출 스킵."""
        fake_registry = {
            "movie_detail": _make_async_tool({}),
            "ott_availability": _make_async_tool([]),
            "similar_movies": _make_async_tool([]),
            "web_search_movie": _make_async_tool([{"title": "X"}]),
        }
        with patch("monglepick.chains.tool_executor_chain.TOOL_REGISTRY", fake_registry):
            result = await execute_tool(
                intent="info",
                movie_id=None,
                movie_title="기생충",
            )
        assert "movie_detail" not in result
        assert "ott_availability" not in result
        assert "similar_movies" not in result
        # web_search_movie 는 movie_title 만으로도 호출 가능하므로 살아있다
        assert result["web_search_movie"] == [{"title": "X"}]

    @pytest.mark.asyncio
    async def test_tool_timeout_falls_back_to_empty(self):
        """asyncio.TimeoutError → 결과 빈 list, 다른 도구는 영향 없음."""
        fake_registry = {
            "theater_search": _make_failing_tool(asyncio.TimeoutError()),
            "kobis_now_showing": _make_async_tool([{"rank": 1}]),
        }
        with patch("monglepick.chains.tool_executor_chain.TOOL_REGISTRY", fake_registry):
            result = await execute_tool(
                intent="theater",
                location={"latitude": 37.5, "longitude": 127.0},
            )
        assert result["theater_search"] == []
        assert result["kobis_now_showing"] == [{"rank": 1}]

    @pytest.mark.asyncio
    async def test_tool_exception_falls_back_to_empty(self):
        """일반 Exception → 결과 빈 list, 다른 도구는 영향 없음."""
        fake_registry = {
            "theater_search": _make_failing_tool(RuntimeError("boom")),
            "kobis_now_showing": _make_async_tool([{"rank": 1}]),
        }
        with patch("monglepick.chains.tool_executor_chain.TOOL_REGISTRY", fake_registry):
            result = await execute_tool(
                intent="theater",
                location={"latitude": 37.5, "longitude": 127.0},
            )
        assert result["theater_search"] == []
        assert result["kobis_now_showing"] == [{"rank": 1}]

    # ──────────────────────────────────────────────────────────────
    # gating: "근처 상영중 영화가 있을 때만 영화관 카드를 노출" 요구사항
    # ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_theater_gated_skip_when_kobis_empty(self):
        """kobis 결과가 빈 list → theater_search 호출 자체를 스킵."""
        fake_registry = {
            "theater_search": _make_async_tool([{"name": "should not be called"}]),
            "kobis_now_showing": _make_async_tool([]),  # 빈 박스오피스
        }
        with patch("monglepick.chains.tool_executor_chain.TOOL_REGISTRY", fake_registry):
            result = await execute_tool(
                intent="theater",
                location={"latitude": 37.5, "longitude": 127.0},
            )
        # kobis 결과는 포함되지만 값은 빈 list, theater_search 는 키조차 없어야 한다
        assert result["kobis_now_showing"] == []
        assert "theater_search" not in result
        fake_registry["theater_search"].ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_theater_gated_skip_when_kobis_timeout(self):
        """kobis 가 타임아웃 → 상영중 확인 불가 → theater_search 스킵 (보수적)."""
        fake_registry = {
            "theater_search": _make_async_tool([{"name": "nope"}]),
            "kobis_now_showing": _make_failing_tool(asyncio.TimeoutError()),
        }
        with patch("monglepick.chains.tool_executor_chain.TOOL_REGISTRY", fake_registry):
            result = await execute_tool(
                intent="theater",
                location={"latitude": 37.5, "longitude": 127.0},
            )
        assert result["kobis_now_showing"] == []
        assert "theater_search" not in result
        fake_registry["theater_search"].ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_booking_gated_skip_also_blocks_search_movies(self):
        """booking 도 gating 대상 — kobis 비었을 때 theater_search 와 search_movies 모두 스킵."""
        fake_registry = {
            "theater_search": _make_async_tool([{"name": "nope"}]),
            "kobis_now_showing": _make_async_tool([]),
            "search_movies": _make_async_tool([{"title": "nope"}]),
        }
        with patch("monglepick.chains.tool_executor_chain.TOOL_REGISTRY", fake_registry):
            result = await execute_tool(
                intent="booking",
                location={"latitude": 37.5, "longitude": 127.0},
                movie_title="아무 영화",
            )
        assert result["kobis_now_showing"] == []
        assert "theater_search" not in result
        assert "search_movies" not in result
        fake_registry["theater_search"].ainvoke.assert_not_awaited()
        fake_registry["search_movies"].ainvoke.assert_not_awaited()
