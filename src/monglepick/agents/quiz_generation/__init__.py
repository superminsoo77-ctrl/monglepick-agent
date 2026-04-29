"""
영화 퀴즈 생성 에이전트 (Quiz Generation Agent).

관리자 페이지의 "AI 운영 → 퀴즈 생성" 트리거가 호출하는 LangGraph 기반 에이전트.

주요 흐름 (7노드):
    movie_selector       → 후보 영화 풀 샘플링 (장르 / 인기도 / 최근 7일 quiz 제외)
    metadata_enricher    → 시놉시스·감독·출연·키워드·연도 메타데이터 보강
    question_generator   → 영화별 LLM 호출하여 4지선다 1문항 생성
    quality_validator    → 스키마 / 정답-options 일치 / 옵션 중복 / 스포일러 키워드 필터
    diversity_checker    → 동일 패턴 중복 질문 제거
    fallback_filler      → 검증 실패 영화에 대해 장르 기반 템플릿 fallback 생성
    persistence          → quizzes 테이블 PENDING INSERT (생성된 quiz_id 수집)

설계 원칙:
    - State: TypedDict (LangGraph 호환)
    - 모든 노드: async def + try/except + 안전한 fallback (에러 전파 금지)
    - 단일 진실 원본: 기존 admin.py 인라인 헬퍼들은 본 모듈로 이관 후 삭제
"""

from monglepick.agents.quiz_generation.graph import (
    build_quiz_generation_graph,
    quiz_generation_graph,
)
from monglepick.agents.quiz_generation.models import (
    QuizGenerationState,
    QuizDraft,
    GeneratedQuizRecord,
)

__all__ = [
    "build_quiz_generation_graph",
    "quiz_generation_graph",
    "QuizGenerationState",
    "QuizDraft",
    "GeneratedQuizRecord",
]
