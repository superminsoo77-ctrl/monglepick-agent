"""
콘텐츠 분석 API 라우터 (§8, Phase 7).

엔드포인트:
- POST /api/v1/content/poster-analysis  — 영화 포스터 시각 분석 + 리뷰 초안 생성
- POST /api/v1/content/toxicity-check   — 한국어 비속어/혐오표현 검출

인증: 없음 (내부 서비스 호출 전용, X-Service-Key 헤더는 선택)
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter

from monglepick.agents.content_analysis import (
    PosterAnalysisInput,
    PosterAnalysisOutput,
    ProfanityCheckInput,
    ProfanityCheckOutput,
    analyze_poster,
    check_profanity,
)

logger = structlog.get_logger()

content_router = APIRouter(prefix="/content", tags=["content"])


@content_router.post(
    "/poster-analysis",
    response_model=PosterAnalysisOutput,
    summary="포스터 분석 + 리뷰 초안",
    responses={
        200: {
            "description": "포스터 시각 분석 결과와 한국어 리뷰 초안 반환",
            "content": {
                "application/json": {
                    "example": {
                        "poster_analysis": {
                            "mood_tags": ["어두운", "긴장감", "서늘한"],
                            "color_palette": ["딥 블루", "차가운 회색"],
                            "visual_impression": "냉혹한 도시의 긴장감",
                            "atmosphere": "밤의 도시를 배경으로 긴박한 분위기가 감돈다.",
                        },
                        "review_draft": "이 영화는 긴장감 넘치는 장면들로 가득합니다.",
                        "review_keywords": ["긴장감", "스릴러", "도시"],
                    }
                }
            },
        }
    },
)
async def poster_analysis(req: PosterAnalysisInput) -> PosterAnalysisOutput:
    """
    영화 포스터 URL을 받아 시각 분석 결과와 리뷰 초안을 반환한다.

    처리 과정:
    1. poster_url에서 이미지 다운로드 (5MB 이하, 실패 시 메타데이터 기반)
    2. VLM으로 포스터 분위기/색감/인상 분석
    3. Explanation LLM으로 3~5문장 한국어 리뷰 초안 생성
    4. 리뷰 키워드 5~7개 추출

    에러 발생 시에도 fallback 결과를 반환한다 (500 에러 없음).
    """
    logger.info(
        "content_poster_analysis_request",
        movie_id=req.movie_id,
        poster_url=req.poster_url[:80] if req.poster_url else "",
    )
    return await analyze_poster(req)


@content_router.post(
    "/toxicity-check",
    response_model=ProfanityCheckOutput,
    summary="비속어/혐오표현 검출",
    responses={
        200: {
            "description": "비속어 검출 결과 및 처리 액션 반환",
            "content": {
                "application/json": {
                    "example": {
                        "is_toxic": True,
                        "toxicity_score": 0.12,
                        "detected_words": ["비속어예시"],
                        "action": "blind",
                    }
                }
            },
        }
    },
)
async def toxicity_check(req: ProfanityCheckInput) -> ProfanityCheckOutput:
    """
    텍스트에서 한국어 비속어/혐오표현을 검출하고 처리 액션을 반환한다.

    content_type별 민감도:
    - chat    : 느슨 (가중치 0.7)
    - post    : 기본 (가중치 1.0)
    - comment : 약간 엄격 (가중치 1.1)
    - review  : 엄격 (가중치 1.3)

    action 기준:
    - pass    : 정상
    - warning : 경미한 비속어 감지
    - blind   : 콘텐츠 블라인드 권장
    - block   : 작성 차단 권장

    LLM 없이 정규식 기반으로 처리하여 응답이 빠르다.
    에러 시 안전 기본값("pass") 반환.
    """
    logger.info(
        "content_toxicity_check_request",
        user_id=req.user_id,
        content_type=req.content_type,
        text_length=len(req.text),
    )
    return await check_profanity(req)
