"""
Tool Executor 체인 (Phase 6 — 외부 지도 연동 라운드에서 stub 제거 + 실구현).

§6-2 Node 13 (`tool_executor_node`) 의 도구 실행 오케스트레이션 체인.

설계:
- LLM 기반 ReAct 루프 대신 **규칙 기반 디스패처** 채택.
  · intent → INTENT_TOOL_MAP → 해당 도구들을 asyncio.gather 로 병렬 호출.
  · LLM 호출 0회 → 평균 200~500ms 단축 + Solar API 토큰 비용 0.
- 도구별 입력 인자는 본 체인이 state 기반으로 직접 조립한다.
  · theater/booking 의 `theater_search` → location 필요 (없으면 노드 측에서 사전 분기)
  · info 의 `movie_detail` / `ott_availability` / `similar_movies` → movie_id 필요
- 도구 단위 timeout (기본 10초) 으로 한 도구 지연이 전체를 깨뜨리지 않도록 격리.
- 모든 예외는 도구 단위에서 흡수, 호출자에게는 빈 결과(`[]`) 로만 전파.

Phase 6 이후 ReAct/Function-Calling 으로 확장이 필요해지면 이 체인 내부만 교체하면 된다.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from monglepick.tools import INTENT_TOOL_MAP, TOOL_REGISTRY

logger = structlog.get_logger()

# 도구 단위 타임아웃 (초). 한 도구가 멈춰도 다른 도구 결과는 살리기 위함.
_TOOL_TIMEOUT_SEC: float = 10.0


async def execute_tool(
    intent: str,
    *,
    location: dict | None = None,
    movie_id: str | None = None,
    movie_title: str | None = None,
    user_id: str = "",
    radius_m: int = 5000,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    의도(intent) 에 맞는 도구를 INTENT_TOOL_MAP 으로 디스패치하고 결과를 묶어 반환한다.

    각 도구는 비동기 + 독립 실행이므로 asyncio.gather 로 병렬 호출하여
    네트워크 왕복(TMDB / 카카오 / KOBIS / Qdrant 등) 을 시간 측면에서 평탄화한다.

    Args:
        intent: 사용자 의도 ("info" | "theater" | "booking" | "search")
        location: 사용자 위치 dict
                  {"latitude": float, "longitude": float, "address": str?}
                  theater/booking 의도에서 theater_search 호출에 사용. 없으면 해당 도구 skip.
        movie_id: TMDB 영화 ID (info 의도에서 movie_detail / ott_availability / similar_movies 입력)
        movie_title: 영화 제목 (info / booking 의도에서 search_movies 의 query 로 사용)
        user_id: 사용자 ID (현재는 도구에 직접 전달하지 않음 — 추후 user_history 연동용)
        radius_m: theater_search 검색 반경(미터)
        top_n: kobis_now_showing 의 박스오피스 Top-N

    Returns:
        도구 이름 → 결과 dict.
        예) theater 의도:
        {
            "theater_search": [<theater_dict>, ...] | "안내 문자열",
            "kobis_now_showing": [<movie_dict>, ...] | "안내 문자열",
        }
        실행되지 않은 도구는 결과에 포함되지 않는다 (키 누락 = 미실행).
    """
    tool_names = INTENT_TOOL_MAP.get(intent, [])
    if not tool_names:
        logger.info("tool_executor_chain_unknown_intent", intent=intent)
        return {}

    # ── gating 대상 인텐트 (theater/booking) ──
    # "근처 상영중 영화가 실제로 있을 때만 영화관 카드를 노출한다" 요구사항을 만족시키기 위해
    # kobis_now_showing 을 선행 실행하고, 결과가 비면 theater_search / search_movies 를 전부 스킵한다.
    # info 는 kobis 를 쓰지 않으므로 기존 병렬 경로를 유지한다.
    _GATED_INTENTS = ("theater", "booking")

    # 도구 호출 인자 조립 헬퍼 — 도구가 요구하는 입력만 골라서 전달한다.
    # 인자 누락 시 (예: theater_search 인데 location 없음) None 반환 → 호출 스킵.
    def _build_args(name: str) -> dict[str, Any] | None:
        # ── 위치 기반 영화관 검색 ──
        if name == "theater_search":
            if not location:
                return None
            return {
                "latitude": location.get("latitude") or location.get("lat"),
                "longitude": location.get("longitude") or location.get("lng"),
                "radius": radius_m,
            }
        # ── KOBIS 박스오피스 (입력 없음, top_n 만 조정 가능) ──
        if name == "kobis_now_showing":
            return {"top_n": top_n}
        # ── TMDB 영화 상세 ──
        if name == "movie_detail":
            if not movie_id:
                return None
            return {"movie_id": str(movie_id)}
        # ── TMDB OTT 가용성 ──
        if name == "ott_availability":
            if not movie_id:
                return None
            return {"movie_id": str(movie_id)}
        # ── Qdrant 유사 영화 ──
        if name == "similar_movies":
            if not movie_id:
                return None
            return {"movie_id": str(movie_id)}
        # ── 내부 RAG 영화 검색 (제목 기반) ──
        if name == "search_movies":
            if not movie_title:
                return None
            return {"query": movie_title}
        # ── 외부 웹 검색 ──
        if name == "web_search_movie":
            if not movie_title:
                return None
            return {"query": movie_title}
        # ── Neo4j 관계 탐색 ──
        if name == "graph_explorer":
            # graph_explorer 는 별도 LLM 추출 결과 필요 → 본 디스패처 범위 밖.
            # 호출 스킵 (필요 시 graph_traversal_node 에서 직접 호출).
            return None
        return None

    # 도구별 호출 코루틴 빌드 — 인자 누락(None)은 미실행으로 분류
    pending: dict[str, Any] = {}        # name -> 결과 (이미 fail-fast 분류된 항목)
    coros: list[tuple[str, Any]] = []   # (name, coroutine) — 실제로 실행할 것만

    # ── theater/booking 인텐트 gating ──
    # kobis_now_showing 을 먼저 호출하여 "현재 상영중인 영화가 있는가" 를 확인한다.
    # 비어 있으면 (API 장애 / 박스오피스 공백 등) theater_search / search_movies 를 건너뛰어
    # "근처에 영화관만 덩그러니 떠 있고 볼 영화는 없음" 상태를 방지한다.
    # 정상(>0) 이면 kobis 결과를 pending 에 확정해두고 나머지 도구만 병렬로 실행한다.
    if intent in _GATED_INTENTS and "kobis_now_showing" in tool_names:
        kobis_tool = TOOL_REGISTRY.get("kobis_now_showing")
        if kobis_tool is not None:
            kobis_args = _build_args("kobis_now_showing") or {}
            try:
                kobis_result = await asyncio.wait_for(
                    kobis_tool.ainvoke(kobis_args),
                    timeout=_TOOL_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "tool_executor_chain_tool_timeout",
                    tool="kobis_now_showing",
                    intent=intent,
                    timeout_sec=_TOOL_TIMEOUT_SEC,
                )
                kobis_result = []
            except Exception as e:  # noqa: BLE001 — 도구 경계에서 모든 예외 흡수
                logger.error(
                    "tool_executor_chain_tool_error",
                    tool="kobis_now_showing",
                    intent=intent,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                kobis_result = []

            # kobis 는 결과 유형이 list | str 둘 다 허용 → list 이고 원소가 있을 때만 "상영중" 으로 판정.
            has_now_showing = isinstance(kobis_result, list) and len(kobis_result) > 0
            pending["kobis_now_showing"] = kobis_result

            if not has_now_showing:
                # 현재 상영중 영화가 없음 → 영화관/예매 관련 도구를 일괄 스킵.
                logger.info(
                    "tool_executor_chain_gated_skip",
                    intent=intent,
                    reason="kobis_now_showing_empty",
                    skipped_tools=[n for n in tool_names if n != "kobis_now_showing"],
                )
                return pending

            # 정상: 나머지 도구만 이후 루프에서 처리하도록 kobis 제외
            tool_names = [n for n in tool_names if n != "kobis_now_showing"]

    for name in tool_names:
        tool_obj = TOOL_REGISTRY.get(name)
        if tool_obj is None:
            logger.warning("tool_executor_chain_unknown_tool", tool=name, intent=intent)
            continue
        args = _build_args(name)
        if args is None:
            # 입력 인자 부족 — 호출조차 하지 않고 결과에 미포함 (호출자가 키 부재로 인지)
            logger.info(
                "tool_executor_chain_skip_no_args",
                tool=name,
                intent=intent,
                reason="missing_required_args",
            )
            continue
        # asyncio.wait_for 로 도구 단위 타임아웃 부여
        coros.append((name, asyncio.wait_for(tool_obj.ainvoke(args), timeout=_TOOL_TIMEOUT_SEC)))

    # 모든 도구 병렬 실행 — return_exceptions=True 로 한 도구 실패가 다른 결과를 깨뜨리지 않게.
    if coros:
        names = [name for name, _ in coros]
        results = await asyncio.gather(
            *(coro for _, coro in coros),
            return_exceptions=True,
        )
        for name, result in zip(names, results, strict=False):
            if isinstance(result, asyncio.TimeoutError):
                logger.warning(
                    "tool_executor_chain_tool_timeout",
                    tool=name,
                    intent=intent,
                    timeout_sec=_TOOL_TIMEOUT_SEC,
                )
                # 타임아웃은 빈 결과로 — 호출자가 자연어 응답에서 해당 블록을 생략하도록.
                pending[name] = []
            elif isinstance(result, Exception):
                logger.error(
                    "tool_executor_chain_tool_error",
                    tool=name,
                    intent=intent,
                    error=str(result),
                    error_type=type(result).__name__,
                )
                pending[name] = []
            else:
                pending[name] = result

    logger.info(
        "tool_executor_chain_done",
        intent=intent,
        executed_tools=list(pending.keys()),
        skipped_tools=[n for n in tool_names if n not in pending],
    )
    return pending
