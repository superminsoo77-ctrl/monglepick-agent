"""
LangChain Tools 패키지 — Phase 6 도구 모음.

tool_executor_node에서 info/theater/booking 의도에 대해 호출되는 7개 도구.

도구 목록:
1. search_movies      — 내부 RAG(Qdrant+ES+Neo4j) 기반 영화 검색
2. movie_detail       — TMDB API 영화 상세 정보 조회
3. theater_search     — 카카오맵 API 기반 근처 영화관 검색
4. ott_availability   — TMDB Watch Providers API OTT 시청 가능 여부
5. similar_movies     — Qdrant 코사인 유사도 기반 유사 영화 검색
6. user_history       — MySQL watch_history 사용자 시청 이력 조회
7. graph_explorer     — Neo4j 영화 관계 그래프 탐색

각 도구는 @tool 데코레이터가 적용된 async 함수이며,
에러 발생 시 빈 값([], {}, 안내 문자열)을 반환하고 절대 에러를 전파하지 않는다.

사용 예시 (tool_executor_node에서):
    from monglepick.tools import TOOL_REGISTRY
    func = TOOL_REGISTRY.get("movie_detail")
    result = await func.ainvoke({"movie_id": "157336"})
"""

from __future__ import annotations

from monglepick.tools.graph_explorer import graph_explorer
from monglepick.tools.movie_detail import movie_detail
from monglepick.tools.ott_availability import ott_availability
from monglepick.tools.search_movies import search_movies
from monglepick.tools.similar_movies import similar_movies
from monglepick.tools.theater_search import theater_search
from monglepick.tools.user_history import user_history

# 도구 이름 → LangChain Tool 인스턴스 매핑
# tool_executor_node에서 의도별 도구를 선택할 때 사용한다.
TOOL_REGISTRY: dict[str, object] = {
    "search_movies": search_movies,
    "movie_detail": movie_detail,
    "theater_search": theater_search,
    "ott_availability": ott_availability,
    "similar_movies": similar_movies,
    "user_history": user_history,
    "graph_explorer": graph_explorer,
}

# 의도별 기본 도구 매핑 (tool_executor_node의 intent → tool name)
# info 의도: 영화 상세 정보 + OTT 가용성 + 유사 영화
# theater 의도: 영화관 검색
# booking 의도: 영화 검색 후 상세 정보 (예매 링크 미구현, 영화 정보 제공으로 대체)
INTENT_TOOL_MAP: dict[str, list[str]] = {
    "info": ["movie_detail", "ott_availability", "similar_movies"],
    "theater": ["theater_search"],
    "booking": ["movie_detail", "search_movies"],
    "search": ["search_movies", "graph_explorer"],
}

__all__ = [
    "search_movies",
    "movie_detail",
    "theater_search",
    "ott_availability",
    "similar_movies",
    "user_history",
    "graph_explorer",
    "TOOL_REGISTRY",
    "INTENT_TOOL_MAP",
]
