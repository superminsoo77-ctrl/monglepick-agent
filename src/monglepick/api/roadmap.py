"""
로드맵 에이전트 API 라우터 (§9, Phase 7).

엔드포인트:
- POST /api/v1/roadmap/generate     — 개인화 로드맵 생성 (LangGraph 4노드 순차 실행)
- POST /api/v1/roadmap/verify-quiz  — 퀴즈 정답 검증 (객관식: 정확 일치, 주관식: 포함 검사)

인증: 없음 (내부 서비스 호출 전용)
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

from monglepick.agents.roadmap import FormattedRoadmap, roadmap_graph

logger = structlog.get_logger()

roadmap_router = APIRouter(prefix="/roadmap", tags=["roadmap"])


# ============================================================
# 요청/응답 모델
# ============================================================

class RoadmapRequest(BaseModel):
    """로드맵 생성 요청 모델."""

    user_id: str = Field(description="사용자 ID")
    theme: str = Field(
        description="로드맵 테마 키워드 (예: '봉준호', '느와르', '1990년대 홍콩')",
    )
    user_profile: dict = Field(
        default_factory=dict,
        description="사용자 프로필 dict (취향, 등급, 구독 정보 등). 선택적.",
    )
    watch_history: list[dict] = Field(
        default_factory=list,
        description=(
            "사용자 시청 이력 목록. 각 항목은 "
            "{movie_id, genres, rating, watched_at} 포함 권장."
        ),
    )


class QuizVerifyRequest(BaseModel):
    """퀴즈 정답 검증 요청 모델."""

    movie_id: str = Field(description="퀴즈 대상 영화 ID")
    question: str = Field(description="퀴즈 질문 텍스트")
    answer: str = Field(description="정답 문자열")
    user_answer: str = Field(description="사용자가 입력한 답변")
    question_type: str = Field(
        default="multiple_choice",
        description="문항 유형 (multiple_choice | short_answer)",
    )


class QuizVerifyResponse(BaseModel):
    """퀴즈 정답 검증 결과 모델."""

    is_correct: bool = Field(description="정답 여부")
    correct_answer: str = Field(description="정답 문자열")
    explanation: str = Field(
        default="",
        description="정오답 설명 (현재는 빈 문자열, 추후 LLM 확장 가능)",
    )


# ============================================================
# 엔드포인트
# ============================================================

@roadmap_router.post(
    "/generate",
    response_model=dict,
    summary="개인화 로드맵 생성",
    responses={
        200: {
            "description": "3단계(입문/심화/매니아) 각 5편 + 퀴즈 포함 로드맵 반환",
            "content": {
                "application/json": {
                    "example": {
                        "roadmap_id": "550e8400-e29b-41d4-a716-446655440000",
                        "theme": "봉준호",
                        "user_level": "intermediate",
                        "created_at": "2026-04-07T00:00:00+00:00",
                        "stages": [
                            {
                                "name": "입문",
                                "description": "봉준호 감독의 대중적인 작품으로 시작하세요.",
                                "movies": [],
                            }
                        ],
                        "total_progress": 0,
                    }
                }
            },
        }
    },
)
async def generate_roadmap(req: RoadmapRequest) -> dict:
    """
    사용자 시청 이력과 테마 키워드를 기반으로 개인화 로드맵을 생성한다.

    LangGraph 4노드 순차 실행:
    1. user_segment_analyzer : 시청 이력 → 레벨 판정 (beginner/intermediate/expert)
    2. roadmap_generator     : MySQL 테마 검색 → 단계별 5편 선정
    3. quiz_generator        : 15편 영화 퀴즈 생성 (LLM + fallback 템플릿)
    4. roadmap_formatter     : 단계 소개글 + UUID + 최종 조립

    에러 발생 시에도 빈 stages를 포함한 유효한 dict를 반환한다 (500 에러 없음).

    Args:
        req: RoadmapRequest (user_id, theme, user_profile, watch_history)

    Returns:
        FormattedRoadmap.model_dump() 형태의 dict
    """
    logger.info(
        "roadmap_generate_request",
        user_id=req.user_id,
        theme=req.theme,
        watch_history_count=len(req.watch_history),
    )

    try:
        result = await roadmap_graph.ainvoke({
            "user_id":       req.user_id,
            "theme":         req.theme,
            "user_profile":  req.user_profile,
            "watch_history": req.watch_history,
        })
        formatted = result.get("formatted_roadmap", {})

        logger.info(
            "roadmap_generate_complete",
            user_id=req.user_id,
            roadmap_id=formatted.get("roadmap_id", ""),
            stage_count=len(formatted.get("stages", [])),
        )
        return formatted

    except Exception as e:
        logger.error(
            "roadmap_generate_error",
            user_id=req.user_id,
            theme=req.theme,
            error=str(e),
        )
        # 에러 시 최소한의 유효한 응답 반환 (500 에러 방지)
        return {
            "roadmap_id": "",
            "theme": req.theme,
            "user_level": "beginner",
            "created_at": "",
            "stages": [],
            "total_progress": 0,
            "error": "로드맵 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        }


@roadmap_router.post(
    "/verify-quiz",
    response_model=QuizVerifyResponse,
    summary="퀴즈 정답 검증",
    responses={
        200: {
            "description": "정오답 여부와 정답 반환",
            "content": {
                "application/json": {
                    "example": {
                        "is_correct": True,
                        "correct_answer": "드라마",
                        "explanation": "",
                    }
                }
            },
        }
    },
)
async def verify_quiz(req: QuizVerifyRequest) -> QuizVerifyResponse:
    """
    퀴즈 문항의 정답 여부를 검증한다.

    검증 방식:
    - multiple_choice: 사용자 답변과 정답의 정확 일치 (strip 후 비교)
    - short_answer   : 상호 포함 검사 (user_answer in answer 또는 answer in user_answer,
                       대소문자 무시). 짧은 연도/감독명 등의 주관식에 적합.

    LLM 없이 규칙 기반으로 처리하여 응답이 즉각적이다.

    Args:
        req: QuizVerifyRequest (movie_id, question, answer, user_answer, question_type)

    Returns:
        QuizVerifyResponse (is_correct, correct_answer, explanation)
    """
    logger.info(
        "quiz_verify_request",
        movie_id=req.movie_id,
        question_type=req.question_type,
    )

    try:
        if req.question_type == "multiple_choice":
            # 객관식: 정확 일치 (앞뒤 공백 제거 후 비교)
            is_correct = req.user_answer.strip() == req.answer.strip()
        else:
            # 주관식: 상호 포함 검사 (대소문자 무시)
            user = req.user_answer.strip().lower()
            correct = req.answer.strip().lower()
            is_correct = bool(user and correct and (user in correct or correct in user))

        logger.info(
            "quiz_verify_result",
            movie_id=req.movie_id,
            is_correct=is_correct,
        )

        return QuizVerifyResponse(
            is_correct=is_correct,
            correct_answer=req.answer,
            explanation="",
        )

    except Exception as e:
        logger.error("quiz_verify_error", movie_id=req.movie_id, error=str(e))
        return QuizVerifyResponse(
            is_correct=False,
            correct_answer=req.answer,
            explanation="검증 중 오류가 발생했습니다.",
        )
