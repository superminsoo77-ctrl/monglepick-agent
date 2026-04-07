"""
로드맵 에이전트 LangGraph State 및 Pydantic 모델 정의 (§9-3, Phase 7).

모델 구조:
- QuizQuestion   : 퀴즈 문항 단건 (객관식/주관식)
- Quiz           : 영화 1편에 대한 퀴즈 묶음
- RoadmapMovie   : 로드맵에 포함된 영화 단건 (퀴즈 포함)
- RoadmapStage   : 학습 단계 (입문/심화/매니아) + 소속 영화 목록
- FormattedRoadmap : 최종 반환 로드맵 전체 구조

LangGraph State:
- RoadmapAgentState : TypedDict (total=False) — 4개 노드가 공유하는 그래프 상태
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


# ============================================================
# 퀴즈 모델
# ============================================================

class QuizQuestion(BaseModel):
    """퀴즈 문항 단건."""

    type: str = Field(
        default="multiple_choice",
        description="문항 유형 (multiple_choice | short_answer)",
    )
    question: str = Field(description="질문 텍스트")
    options: list[str] = Field(
        default_factory=list,
        description="선택지 목록 (객관식일 때만 사용, 주관식은 빈 리스트)",
    )
    answer: str = Field(description="정답 문자열")
    hint: str = Field(default="", description="힌트 텍스트 (선택)")


class Quiz(BaseModel):
    """영화 1편에 대한 퀴즈 묶음."""

    movie_id: str = Field(description="영화 ID")
    questions: list[QuizQuestion] = Field(
        default_factory=list,
        description="해당 영화의 퀴즈 문항 목록",
    )


# ============================================================
# 로드맵 구성 모델
# ============================================================

class RoadmapMovie(BaseModel):
    """로드맵에 포함된 영화 단건."""

    id: str = Field(description="영화 ID (movie_id)")
    title: str = Field(description="영화 제목")
    genres: list[str] = Field(default_factory=list, description="장르 목록")
    poster_url: str = Field(default="", description="포스터 이미지 URL")
    rating: float = Field(default=0.0, description="TMDB 평점 (0~10)")
    hybrid_score: float = Field(
        default=0.0,
        description="하이브리드 추천 점수 (CF+CBF 결합, 0~1)",
    )
    popularity_score: float = Field(
        default=0.0,
        description="TMDB popularity 정규화 점수 (0~1)",
    )
    quiz: Optional[Quiz] = Field(default=None, description="해당 영화 퀴즈 (생성 후 채워짐)")
    completed: bool = Field(default=False, description="사용자 학습 완료 여부")


class RoadmapStage(BaseModel):
    """학습 단계 묶음 (입문/심화/매니아)."""

    name: str = Field(description="단계 이름 (입문 | 심화 | 매니아)")
    description: str = Field(default="", description="단계 소개글 (LLM 생성, 1~2문장)")
    movies: list[RoadmapMovie] = Field(
        default_factory=list,
        description="해당 단계의 영화 목록 (5편)",
    )


class FormattedRoadmap(BaseModel):
    """최종 반환 로드맵 전체 구조."""

    roadmap_id: str = Field(description="UUID 기반 로드맵 고유 ID")
    theme: str = Field(description="로드맵 테마 키워드 (예: 봉준호, 느와르, 1990년대)")
    user_level: str = Field(
        description="사용자 레벨 판정 결과 (beginner | intermediate | expert)",
    )
    created_at: str = Field(description="생성 시각 (ISO 8601 UTC 문자열)")
    stages: list[RoadmapStage] = Field(
        default_factory=list,
        description="학습 단계 목록 (입문 → 심화 → 매니아 순서)",
    )
    total_progress: int = Field(
        default=0,
        description="전체 영화 중 완료한 편 수 (0~15)",
    )


# ============================================================
# LangGraph State
# ============================================================

class RoadmapAgentState(TypedDict, total=False):
    """
    로드맵 에이전트 그래프 전체 상태.

    total=False: 모든 키가 선택적 — 노드는 자신이 담당하는 키만 업데이트한다.

    키 목록:
    - user_id       : 사용자 ID (str)
    - user_profile  : 사용자 프로필 dict (취향, 등급 등)
    - watch_history : 시청 이력 list[dict] (movie_id, genres, rating 포함)
    - theme         : 로드맵 테마 키워드 (str)
    - user_level    : 레벨 판정 결과 (beginner | intermediate | expert)
    - level_detail  : 레벨 판정 상세 dict (total_watched, unique_genres, ...)
    - course_movies : 단계별 영화 dict {"beginner": [...], "intermediate": [...], "expert": [...]}
    - quizzes       : 퀴즈 목록 list[dict] (movie_id별)
    - formatted_roadmap : 최종 로드맵 dict (FormattedRoadmap.model_dump())
    - error         : 에러 메시지 (있으면 조기 종료 신호)
    """

    user_id: str
    user_profile: dict
    watch_history: list[dict]
    theme: str
    user_level: str
    level_detail: dict
    course_movies: dict
    quizzes: list[dict]
    formatted_roadmap: dict
    error: str
