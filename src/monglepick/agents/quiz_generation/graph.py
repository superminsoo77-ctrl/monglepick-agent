"""
영화 퀴즈 생성 에이전트 LangGraph 그래프 정의.

7노드 순차 실행:
    START → movie_selector → metadata_enricher → question_generator
          → quality_validator → diversity_checker → fallback_filler
          → persistence → END

조건 분기 없음 — 각 노드는 입력이 비어있으면 빈 결과로 pass-through 한다.
빈 DB / 영화 매칭 실패 시에도 persistence 노드까지 흘러가며 final_message 로
사용자에게 안내된다 (HTTP 500 금지).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from monglepick.agents.quiz_generation.models import QuizGenerationState
from monglepick.agents.quiz_generation.nodes import (
    diversity_checker,
    fallback_filler,
    metadata_enricher,
    movie_selector,
    persistence,
    quality_validator,
    question_generator,
)


def build_quiz_generation_graph() -> StateGraph:
    """
    퀴즈 생성 그래프 빌더.

    노드 등록 → 엣지 연결 → compile() 순으로 그래프를 구성한다.
    review_verification 그래프와 동일한 단일 경로 패턴이다.
    """
    builder = StateGraph(QuizGenerationState)

    builder.add_node("movie_selector",      movie_selector)
    builder.add_node("metadata_enricher",   metadata_enricher)
    builder.add_node("question_generator",  question_generator)
    builder.add_node("quality_validator",   quality_validator)
    builder.add_node("diversity_checker",   diversity_checker)
    builder.add_node("fallback_filler",     fallback_filler)
    builder.add_node("persistence",         persistence)

    builder.add_edge(START,                 "movie_selector")
    builder.add_edge("movie_selector",      "metadata_enricher")
    builder.add_edge("metadata_enricher",   "question_generator")
    builder.add_edge("question_generator",  "quality_validator")
    builder.add_edge("quality_validator",   "diversity_checker")
    builder.add_edge("diversity_checker",   "fallback_filler")
    builder.add_edge("fallback_filler",     "persistence")
    builder.add_edge("persistence",         END)

    return builder.compile()


# 모듈 import 시점에 1회 컴파일 — admin.py 핸들러가 ainvoke 만 호출.
quiz_generation_graph = build_quiz_generation_graph()
