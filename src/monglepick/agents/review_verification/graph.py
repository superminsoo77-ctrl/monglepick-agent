"""
리뷰 검증 에이전트 LangGraph 그래프 정의.

5노드 순차 실행:
START → preprocessor → embedding_similarity → keyword_matcher
      → llm_revalidator → threshold_decider → END

조건 분기 없음 — early_exit 플래그를 각 노드가 체크하여 pass-through한다.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from monglepick.agents.review_verification.models import ReviewVerificationState
from monglepick.agents.review_verification.nodes import (
    embedding_similarity,
    keyword_matcher,
    llm_revalidator,
    preprocessor,
    threshold_decider,
)


def build_review_verification_graph() -> StateGraph:
    builder = StateGraph(ReviewVerificationState)

    builder.add_node("preprocessor",         preprocessor)
    builder.add_node("embedding_similarity", embedding_similarity)
    builder.add_node("keyword_matcher",      keyword_matcher)
    builder.add_node("llm_revalidator",      llm_revalidator)
    builder.add_node("threshold_decider",    threshold_decider)

    builder.add_edge(START,                  "preprocessor")
    builder.add_edge("preprocessor",         "embedding_similarity")
    builder.add_edge("embedding_similarity", "keyword_matcher")
    builder.add_edge("keyword_matcher",      "llm_revalidator")
    builder.add_edge("llm_revalidator",      "threshold_decider")
    builder.add_edge("threshold_decider",    END)

    return builder.compile()


review_verification_graph = build_review_verification_graph()
