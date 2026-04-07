"""
콘텐츠 분석 에이전트 패키지 (§8, Phase 7).

제공 기능:
- analyze_poster          : 영화 포스터 시각 분석 + 리뷰 초안 생성 (VLM)
- analyze_community_mentions : 커뮤니티 게시글에서 영화 언급 집계 + 트렌딩 산출
- analyze_user_pattern    : 시청 이력 패턴 분석 + 업적 판정 (규칙 기반)
- check_profanity         : 한국어 비속어/혐오표현 검출 + 처리 액션 결정

모델:
- PosterAnalysis, PosterAnalysisInput, PosterAnalysisOutput
- PostData, MovieMention, TrendingMovie, CommunityAnalysisInput, CommunityAnalysisOutput
- WatchRecord, Achievement, PatternAnalysisInput, PatternAnalysisOutput
- ProfanityCheckInput, ProfanityCheckOutput
"""

from monglepick.agents.content_analysis.models import (
    Achievement,
    CommunityAnalysisInput,
    CommunityAnalysisOutput,
    MovieMention,
    PatternAnalysisInput,
    PatternAnalysisOutput,
    PostData,
    PosterAnalysis,
    PosterAnalysisInput,
    PosterAnalysisOutput,
    ProfanityCheckInput,
    ProfanityCheckOutput,
    TrendingMovie,
    WatchRecord,
)
from monglepick.agents.content_analysis.poster_analysis import analyze_poster
from monglepick.agents.content_analysis.community_analysis import (
    analyze_community_mentions,
)
from monglepick.agents.content_analysis.pattern_analysis import analyze_user_pattern
from monglepick.agents.content_analysis.toxicity_detection import check_profanity

__all__ = [
    # 분석 함수
    "analyze_poster",
    "analyze_community_mentions",
    "analyze_user_pattern",
    "check_profanity",
    # 모델 — 포스터 분석
    "PosterAnalysis",
    "PosterAnalysisInput",
    "PosterAnalysisOutput",
    # 모델 — 커뮤니티 분석
    "PostData",
    "MovieMention",
    "TrendingMovie",
    "CommunityAnalysisInput",
    "CommunityAnalysisOutput",
    # 모델 — 패턴 분석
    "WatchRecord",
    "Achievement",
    "PatternAnalysisInput",
    "PatternAnalysisOutput",
    # 모델 — 비속어 검출
    "ProfanityCheckInput",
    "ProfanityCheckOutput",
]
