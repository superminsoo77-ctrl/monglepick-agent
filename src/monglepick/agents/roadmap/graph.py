"""
로드맵 에이전트 LangGraph 그래프 정의 (§9-3, Phase 7).

4노드 순차 실행 그래프:
START → user_segment_analyzer → roadmap_generator → quiz_generator → roadmap_formatter → END

조건 분기 없음 — 모든 노드가 에러 시 fallback을 반환하므로 항상 END까지 진행한다.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from monglepick.agents.roadmap.nodes import (
    quiz_generator,
    roadmap_formatter,
    roadmap_generator,
    user_segment_analyzer,
)
from monglepick.agents.roadmap.state import RoadmapAgentState


def build_roadmap_graph() -> StateGraph:
    """
    개인화 로드맵 에이전트 StateGraph를 빌드하고 컴파일한다.

    그래프 흐름 (순차):
    START
      → user_segment_analyzer  : 시청 이력 기반 사용자 레벨 판정 (beginner/intermediate/expert)
      → roadmap_generator      : MySQL 테마 검색 → 3단계별 5편 선정
      → quiz_generator         : 15편 영화 퀴즈 생성 (LLM + fallback 템플릿)
      → roadmap_formatter      : 단계별 소개글 생성 + UUID + 최종 구조 조립
      → END

    Returns:
        컴파일된 CompiledStateGraph 인스턴스
    """
    builder = StateGraph(RoadmapAgentState)

    # ── 노드 등록 ──
    builder.add_node("user_segment_analyzer", user_segment_analyzer)
    builder.add_node("roadmap_generator",     roadmap_generator)
    builder.add_node("quiz_generator",        quiz_generator)
    builder.add_node("roadmap_formatter",     roadmap_formatter)

    # ── 엣지 정의 (순차 실행) ──
    builder.add_edge(START,                    "user_segment_analyzer")
    builder.add_edge("user_segment_analyzer",  "roadmap_generator")
    builder.add_edge("roadmap_generator",      "quiz_generator")
    builder.add_edge("quiz_generator",         "roadmap_formatter")
    builder.add_edge("roadmap_formatter",      END)

    return builder.compile()


# ── 모듈 로드 시 그래프 인스턴스 생성 (싱글턴) ──
# API 레이어에서 `from monglepick.agents.roadmap import roadmap_graph` 로 임포트하여 사용한다.
roadmap_graph = build_roadmap_graph()
