"""
퀴즈 생성 에이전트 상태 정의.

LangGraph 호환을 위해 그래프 상태는 TypedDict 로 선언한다.
하위 데이터(영화 후보 / 퀴즈 초안 / 저장 결과)는 Pydantic BaseModel 로
정의하여 검증·직렬화에 활용한다.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ============================================================
# 하위 데이터 모델 (Pydantic)
# ============================================================


class CandidateMovie(BaseModel):
    """
    퀴즈 생성 후보 영화 메타.

    movie_selector → metadata_enricher 단계까지 전달되며,
    question_generator 가 LLM 프롬프트의 컨텍스트로 사용한다.
    """

    movie_id: str = Field(..., description="movies.movie_id (VARCHAR(50))")
    title: str = Field(..., description="한국어 제목 (없으면 원제 fallback)")
    genres: list[str] = Field(default_factory=list, description="장르 목록")
    release_year: str = Field(default="", description="개봉 연도(YYYY) 문자열")
    overview: str = Field(default="", description="영화 줄거리/시놉시스 (TEXT)")
    director: str = Field(default="", description="감독명 (영화당 1명 가정)")
    cast_members: list[str] = Field(
        default_factory=list,
        description="주요 출연진 (상위 5인)",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="영화 키워드 태그 (최대 10개)",
    )
    tagline: str = Field(default="", description="태그라인(짧은 홍보 문구)")


class QuizDraft(BaseModel):
    """
    LLM 또는 fallback 으로 생성된 4지선다 퀴즈 1문항 초안.

    quality_validator → diversity_checker → persistence 단계로 흘러가며,
    검증을 통과한 초안만 quizzes 테이블에 INSERT 된다.
    """

    movie_id: str = Field(..., description="대상 영화 ID")
    movie_title: str = Field(..., description="응답용 영화 제목 캐시")
    question: str = Field(..., description="퀴즈 문제 본문")
    options: list[str] = Field(..., description="객관식 4지선다 (정확히 4개)")
    correct_answer: str = Field(..., description="정답 (options 중 하나와 정확히 일치)")
    explanation: str = Field(default="", description="해설 (1~2문장)")
    category: str = Field(
        default="general",
        description="질문 카테고리 (genre/director/year/cast/plot/general)",
    )
    is_fallback: bool = Field(
        default=False,
        description="True 면 LLM 생성 실패 → 템플릿 fallback 사용",
    )
    valid: bool = Field(
        default=True,
        description="quality_validator 통과 여부. False 면 persistence 가 스킵.",
    )
    reject_reason: str = Field(
        default="",
        description="valid=False 일 때 사유 (감사 로그용)",
    )


class GeneratedQuizRecord(BaseModel):
    """
    persistence 노드가 INSERT 후 생성한 응답용 레코드.

    api/admin.py 의 GeneratedQuizItem 으로 변환되어 관리자 UI 에 노출된다.
    """

    quiz_id: int
    movie_id: str
    movie_title: str
    question: str
    correct_answer: str
    options: list[str]
    explanation: Optional[str] = None
    reward_point: int = 10
    status: str = "PENDING"


# ============================================================
# 그래프 상태 (TypedDict — LangGraph 호환)
# ============================================================


class QuizGenerationState(TypedDict, total=False):
    """
    퀴즈 생성 에이전트 그래프 전체 상태.

    total=False: 모든 키가 선택적 — 각 노드는 자신이 담당하는 키만 업데이트한다.
    입력 필드는 GenerateQuizRequest 와 1:1 대응한다.
    """

    # ── 입력 (API 에서 초기화) ──────────────────────────────────
    genre: Optional[str]            # 장르 필터 (None 이면 전체)
    difficulty: str                 # easy / medium / hard
    count: int                      # 생성 목표 편수 (1~50)
    exclude_recent_days: int        # 최근 N 일 quiz 가 있는 영화 제외 (기본 7)
    reward_point: int               # 정답 시 포인트 (기본 10)

    # ── movie_selector 출력 ─────────────────────────────────────
    candidates: list[CandidateMovie]      # 샘플링된 후보 영화 풀
    selector_message: str                  # 빈 결과 등 안내 메시지

    # ── metadata_enricher 출력 ──────────────────────────────────
    enriched_candidates: list[CandidateMovie]   # 메타 보강된 후보

    # ── question_generator 출력 ─────────────────────────────────
    drafts: list[QuizDraft]                # LLM 또는 fallback 초안 리스트

    # ── quality_validator 출력 ──────────────────────────────────
    validated_drafts: list[QuizDraft]      # valid=True/False 마킹된 초안

    # ── diversity_checker 출력 ──────────────────────────────────
    diversified_drafts: list[QuizDraft]    # 동일 패턴 중복 제거 후 초안

    # ── fallback_filler 출력 ────────────────────────────────────
    final_drafts: list[QuizDraft]          # persistence 입력으로 사용

    # ── persistence 출력 (최종) ─────────────────────────────────
    persisted: list[GeneratedQuizRecord]   # INSERT 성공한 레코드들
    final_message: str                      # 최종 안내 메시지
    success: bool                           # persisted 가 1건 이상이면 True
