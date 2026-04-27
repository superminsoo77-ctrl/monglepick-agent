"""
KOBIS 일별 박스오피스 기반 "현재 상영중" 매칭 헬퍼 (2026-04-27).

목적
----
- 추천 영화 카드(`movie_card` SSE)에 `is_now_showing` 플래그를 채워주기 위한 경량 매칭 모듈.
- Client `MovieCard` 의 "🏢 영화관" 버튼은 이 플래그가 True 인 영화에만 노출된다.
- KOBIS 가 영화관별 시간표 API 를 공개하지 않으므로, 일별 박스오피스 Top-10 을
  "전국 어딘가에서 상영 중" 의 약한 신호로 사용한다 (CLAUDE.md "현실적 절충안" 참고).

설계
----
- 모듈 단위 in-memory TTL 캐시 (기본 30분). 추천 1턴마다 KOBIS 5초 호출이 들어가지 않도록.
- 동시 호출 시 중복 fetch 방지용 단일 asyncio.Lock.
- 매칭 키는 정규화된 한국어 영화 제목 (공백/구두점/괄호 제거 + 소문자). 영문 제목은 보조.
- KOBIS Top-10 자체가 작아 set 매칭 O(1). 후보 영화 5편 × 매칭 1회 = 무시할 수 있는 비용.
- KOBIS API 실패 시 set() 반환 → 모든 영화의 is_now_showing=False 가 되어 영화관 버튼이 숨는
  graceful degradation. 사용자에게 거짓 긍정(미상영작에 영화관 버튼)이 나가는 것보다는
  거짓 부정(상영작인데 버튼 미노출) 이 안전하다는 판단.
"""

from __future__ import annotations

import asyncio
import re
import time

import structlog

from monglepick.tools.kobis_now_showing import kobis_now_showing

logger = structlog.get_logger()

# ── 캐시 설정 ────────────────────────────────────────────────────────────
# TTL: KOBIS 박스오피스가 D-1 단위로 갱신되므로 30분이면 충분히 신선하다.
# 운영 중 강제 무효화는 프로세스 재시작으로 처리.
_CACHE_TTL_SEC = 30 * 60

# ── 캐시 상태 (모듈 전역) ────────────────────────────────────────────────
# (정규화 제목 set, 만료 epoch 초). 미초기화 상태는 (None, 0).
_cache: tuple[set[str] | None, float] = (None, 0.0)
_lock = asyncio.Lock()


# ── 영화명 정규화 ─────────────────────────────────────────────────────────
# 매칭 정확도를 위해 양쪽 모두 동일 규칙으로 정규화한다.
# 제거 대상:
#   - 모든 공백 (전각 포함)
#   - ASCII/한글 구두점 + 괄호류
#   - 시리즈 표기 차이("아바타: 물의 길" vs "아바타 물의길") 흡수
# 유지: 한글, 영문, 숫자
_NORM_STRIP_RE = re.compile(r"[\s　.,!?:;'\"`~\-_/\\|@#$%^&*+=\[\]\{\}()<>·•‥…“”‘’、。]")


def _normalize_title(title: str | None) -> str:
    """제목을 매칭용으로 정규화한다 — 공백/구두점/괄호 제거 + 소문자.

    ex) "아바타: 물의 길" → "아바타물의길"
        "Mission Impossible 7" → "missionimpossible7"
    """
    if not title:
        return ""
    return _NORM_STRIP_RE.sub("", title).lower()


async def _invoke_kobis_top10():
    """KOBIS Top-10 호출을 한 겹 감싸 테스트에서 patch 가능하게 한다.

    langchain `@tool` 데코레이터가 만드는 `StructuredTool` 의 `ainvoke` 는
    pydantic frozen 속성이라 `unittest.mock.patch` 로 직접 monkeypatch 가 불가하다.
    이 모듈 함수를 patch 대상으로 노출해 단위 테스트가 KOBIS 결과를 stub 한다.
    """
    return await kobis_now_showing.ainvoke({"top_n": 10})


async def _fetch_now_showing_titles() -> set[str]:
    """KOBIS Top-10 을 호출해 정규화된 제목 set 을 반환한다 (실패 시 빈 set)."""
    try:
        result = await _invoke_kobis_top10()
    except Exception as e:
        # 도구 자체에서 모든 예외를 흡수하고 문자열을 반환하지만, 안전망으로 한 겹 더.
        logger.warning("now_showing_cache_kobis_invoke_error", error=str(e))
        return set()

    # KOBIS 도구는 실패 시 한국어 안내 문자열을 반환한다 (list 가 아님).
    if not isinstance(result, list):
        logger.info("now_showing_cache_kobis_unavailable", result_type=type(result).__name__)
        return set()

    titles: set[str] = set()
    for item in result:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_title(item.get("movie_nm"))
        if normalized:
            titles.add(normalized)
    return titles


async def get_now_showing_titles() -> set[str]:
    """캐시된 정규화 제목 set 을 반환한다. TTL 만료 시 1회 KOBIS 재조회.

    동시 다발적인 호출 시 중복 fetch 가 발생하지 않도록 asyncio.Lock 보호.
    """
    global _cache
    titles, expires_at = _cache
    now = time.monotonic()

    # 캐시 유효 — 락 없이 빠른 경로
    if titles is not None and now < expires_at:
        return titles

    async with _lock:
        # 락을 잡는 동안 다른 코루틴이 갱신했을 수 있으므로 재확인 (double-checked locking)
        titles, expires_at = _cache
        now = time.monotonic()
        if titles is not None and now < expires_at:
            return titles

        # 실제 fetch
        fetched = await _fetch_now_showing_titles()
        _cache = (fetched, now + _CACHE_TTL_SEC)
        logger.info(
            "now_showing_cache_refreshed",
            count=len(fetched),
            ttl_sec=_CACHE_TTL_SEC,
        )
        return fetched


async def is_now_showing(title: str | None, title_en: str | None = None) -> bool:
    """주어진 제목이 KOBIS Top-10 안에 있는지 매칭한다.

    한국어 제목 우선 매칭, 실패 시 영문 제목 fallback.
    KOBIS 가 한국 개봉 영화를 한국어로만 내려보내므로 영문 제목 매칭은
    "현지화되지 않은 외화" 의 매우 드문 케이스에만 동작한다.
    """
    titles = await get_now_showing_titles()
    if not titles:
        return False

    primary = _normalize_title(title)
    if primary and primary in titles:
        return True

    secondary = _normalize_title(title_en)
    if secondary and secondary in titles:
        return True

    return False


async def annotate_movies(movies: list) -> None:
    """RankedMovie 리스트의 각 항목에 `is_now_showing` 플래그를 in-place 로 채운다.

    KOBIS Top-10 set 을 1회 조회하고, 영화별 정규화 매칭으로 bool 을 채운다.
    Pydantic v2 BaseModel 은 기본적으로 mutable 이므로 세터 직접 할당.
    캐시 미스 / API 실패 시 모든 영화의 플래그가 False 로 유지 — 영화관 버튼이 숨는 안전한 폴백.
    """
    if not movies:
        return

    titles = await get_now_showing_titles()
    if not titles:
        # KOBIS 미가용 — 모든 영화 False (이미 default). 굳이 순회하지 않는다.
        return

    for m in movies:
        primary = _normalize_title(getattr(m, "title", None))
        if primary and primary in titles:
            m.is_now_showing = True
            continue
        secondary = _normalize_title(getattr(m, "title_en", None))
        if secondary and secondary in titles:
            m.is_now_showing = True


def reset_cache_for_tests() -> None:
    """테스트 전용 — 캐시 무효화 (운영 코드에서는 호출하지 않는다)."""
    global _cache
    _cache = (None, 0.0)
