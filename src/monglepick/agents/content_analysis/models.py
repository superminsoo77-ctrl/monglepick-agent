"""
콘텐츠 분석 에이전트 Pydantic 모델 정의 (§8, Phase 7).

기능별 입출력 모델:
- 기능1: 포스터 분석 — PosterAnalysisInput / PosterAnalysisOutput
- 기능2: 커뮤니티 언급 분석 — CommunityAnalysisInput / CommunityAnalysisOutput
- 기능3: 사용자 패턴 분석 + 업적 판정 — PatternAnalysisInput / PatternAnalysisOutput
- 기능4: 비속어/혐오 표현 검출 — ProfanityCheckInput / ProfanityCheckOutput
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# 기능1: 포스터 분석
# ============================================================

class PosterAnalysis(BaseModel):
    """VLM이 분석한 영화 포스터 시각적 정보."""

    mood_tags: list[str] = Field(
        default_factory=list,
        description="포스터에서 추출한 분위기 태그 (3~5개). 예: ['긴장감', '어두운', '서늘한']",
    )
    color_palette: list[str] = Field(
        default_factory=list,
        description="포스터 주요 색감 (2~4개). 예: ['딥 블루', '그레이', '레드']",
    )
    visual_impression: str = Field(
        default="",
        description="포스터 첫인상 한 줄 요약 (50자 이내)",
    )
    atmosphere: str = Field(
        default="",
        description="포스터가 전달하는 전체적인 분위기 (1~2문장)",
    )


class PosterAnalysisInput(BaseModel):
    """포스터 분석 요청 입력 모델."""

    movie_id: str = Field(description="영화 고유 ID (VARCHAR50)")
    poster_url: str = Field(description="포스터 이미지 URL")
    movie_metadata: dict = Field(
        default_factory=dict,
        description="영화 메타데이터 (title, genres, overview, director, release_year 등)",
    )
    user_rating: Optional[float] = Field(
        default=None,
        description="사용자가 매긴 평점 (0~5). 리뷰 초안 어조 조정에 사용",
    )


class PosterAnalysisOutput(BaseModel):
    """포스터 분석 결과 출력 모델."""

    poster_analysis: PosterAnalysis = Field(
        default_factory=PosterAnalysis,
        description="VLM 포스터 시각 분석 결과",
    )
    review_draft: str = Field(
        default="",
        description="포스터+메타데이터 기반 리뷰 초안 (3~5문장, 스포일러 금지)",
    )
    review_keywords: list[str] = Field(
        default_factory=list,
        description="리뷰 초안에서 추출한 키워드 (5~7개)",
    )


# ============================================================
# 기능2: 커뮤니티 언급 분석
# ============================================================

class PostData(BaseModel):
    """분석 대상 커뮤니티 게시글 단건."""

    post_id: str = Field(description="게시글 ID")
    content: str = Field(description="게시글 본문 텍스트")
    created_at: str = Field(description="작성 시각 (ISO 8601 문자열)")


class MovieMention(BaseModel):
    """특정 영화에 대한 커뮤니티 언급 집계."""

    movie_id: str = Field(description="언급된 영화 ID")
    title: str = Field(description="영화 제목")
    count: int = Field(default=0, description="언급 횟수")
    posts: list[str] = Field(
        default_factory=list,
        description="언급된 게시글 ID 목록",
    )


class TrendingMovie(BaseModel):
    """트렌딩 영화 집계 결과."""

    movie_id: str = Field(description="영화 ID")
    title: str = Field(description="영화 제목")
    mention_count: int = Field(default=0, description="현재 기간 언급 횟수")
    growth_rate: float = Field(
        default=0.0,
        description="이전 기간 대비 성장률 (0.5 = +50%). 이전 기간 없으면 0.0",
    )
    trending_score: float = Field(
        default=0.0,
        description="트렌딩 점수 = mention_count × (1 + growth_rate)",
    )


class CommunityAnalysisInput(BaseModel):
    """커뮤니티 언급 분석 요청 입력 모델."""

    posts: list[PostData] = Field(
        default_factory=list,
        description="분석 대상 게시글 목록",
    )
    period: str = Field(
        default="weekly",
        description="집계 기간 (daily | weekly | monthly)",
    )


class CommunityAnalysisOutput(BaseModel):
    """커뮤니티 언급 분석 결과 출력 모델."""

    mention_counts: list[MovieMention] = Field(
        default_factory=list,
        description="영화별 언급 횟수 목록 (언급 있는 영화만 포함)",
    )
    trending_movies: list[TrendingMovie] = Field(
        default_factory=list,
        description="트렌딩 점수 상위 10개 영화",
    )


# ============================================================
# 기능3: 사용자 시청 패턴 분석 + 업적 판정
# ============================================================

class WatchRecord(BaseModel):
    """시청 이력 단건."""

    movie_id: str = Field(description="영화 ID")
    watched_at: str = Field(description="시청 시각 (ISO 8601 문자열)")
    rating: Optional[float] = Field(
        default=None,
        description="사용자 평점 (0~5). 없으면 None",
    )
    genres: list[str] = Field(
        default_factory=list,
        description="해당 영화의 장르 목록",
    )


class Achievement(BaseModel):
    """사용자가 획득한 업적 정보."""

    id: str = Field(description="업적 고유 코드 (예: ACH_001)")
    name: str = Field(description="업적 이름 (예: 영화 입문)")
    description: str = Field(description="업적 달성 조건 설명")
    icon: str = Field(description="업적 아이콘 이모지")


class PatternAnalysisInput(BaseModel):
    """사용자 패턴 분석 요청 입력 모델."""

    user_id: str = Field(description="분석 대상 사용자 ID")
    watch_history: list[WatchRecord] = Field(
        default_factory=list,
        description="사용자 전체 시청 이력",
    )
    existing_achievements: list[str] = Field(
        default_factory=list,
        description="이미 보유한 업적 ID 목록 (중복 지급 방지)",
    )


class PatternAnalysisOutput(BaseModel):
    """사용자 패턴 분석 결과 출력 모델."""

    new_achievements: list[Achievement] = Field(
        default_factory=list,
        description="이번 분석에서 새로 획득한 업적 목록",
    )
    user_pattern_vector: list[float] = Field(
        default_factory=list,
        description="장르 분포 기반 40차원 사용자 패턴 벡터 (미구현 차원은 0.0)",
    )


# ============================================================
# 기능4: 비속어/혐오 표현 검출
# ============================================================

class ProfanityCheckInput(BaseModel):
    """비속어 검출 요청 입력 모델."""

    text: str = Field(description="검사할 텍스트")
    user_id: str = Field(description="작성자 사용자 ID (로그용)")
    content_type: str = Field(
        default="post",
        description="콘텐츠 유형 (chat | post | review | comment). 민감도 가중치 결정에 사용",
    )


class ProfanityCheckOutput(BaseModel):
    """비속어 검출 결과 출력 모델."""

    is_toxic: bool = Field(default=False, description="비속어/혐오표현 포함 여부")
    toxicity_score: float = Field(
        default=0.0,
        description="독성 점수 (0~1). 검출 단어 수 / 총 단어 수 × 가중치",
    )
    detected_words: list[str] = Field(
        default_factory=list,
        description="검출된 비속어/혐오표현 목록",
    )
    action: str = Field(
        default="pass",
        description="권장 처리 방식 (pass | warning | blind | block)",
    )
