"""
Chat Agent SSE/동기 엔드포인트 (§6 Phase 3).

3개 엔드포인트:
- POST /api/v1/chat       — SSE 스트리밍 (EventSourceResponse)
- POST /api/v1/chat/sync  — 동기 JSON (디버그/테스트용)
- POST /api/v1/chat/upload — 멀티파트 이미지 업로드 (파일+메시지)

요청 모델: ChatRequest (user_id, session_id, message, image)
응답 모델: ChatSyncResponse (동기 전용)
"""

from __future__ import annotations

import base64
import time

import structlog
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from monglepick.agents.chat.graph import run_chat_agent, run_chat_agent_sync
from monglepick.config import settings

logger = structlog.get_logger()

# APIRouter 생성 (prefix는 main.py에서 설정)
chat_router = APIRouter(tags=["chat"])


# ============================================================
# 요청/응답 모델
# ============================================================

class ChatRequest(BaseModel):
    """
    채팅 요청 모델.

    user_id: 사용자 ID (빈 문자열이면 익명)
    session_id: 세션 ID (빈 문자열이면 신규 세션)
    message: 사용자 입력 메시지 (1~2000자, 필수)
    image: base64 인코딩된 이미지 데이터 (None이면 이미지 없음)
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_123",
                    "session_id": "sess_abc",
                    "message": "우울한데 영화 추천해줘",
                    "image": None,
                },
                {
                    "user_id": "",
                    "session_id": "",
                    "message": "봉준호 감독 영화 중에 볼만한 거 추천해줘",
                    "image": None,
                },
            ]
        }
    }

    user_id: str = Field(
        default="",
        description="사용자 ID (빈 문자열이면 익명)",
    )
    session_id: str = Field(
        default="",
        description="세션 ID (빈 문자열이면 신규 세션)",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="사용자 입력 메시지 (1~2000자)",
    )
    image: str | None = Field(
        default=None,
        description="base64 인코딩된 이미지 데이터 (영화 포스터/분위기 사진 등)",
    )


class ChatSyncResponse(BaseModel):
    """
    동기 채팅 응답 모델 (디버그/테스트용).

    session_id: 세션 ID (다음 요청에 전달하면 대화 맥락 유지)
    response: 최종 응답 텍스트
    intent: 분류된 의도
    emotion: 감지된 감정 (None이면 미감지)
    movie_count: 추천된 영화 수
    image_analyzed: 이미지 분석 수행 여부
    clarification: 후속 질문 힌트 (None이면 힌트 없음)
    """

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "abc123-...",
                    "response": "우울할 때 보면 좋은 영화를 추천해드릴게요! 🎬\n\n1. **인사이드 아웃** ...",
                    "intent": "recommend",
                    "emotion": "sad",
                    "movie_count": 5,
                    "image_analyzed": False,
                    "clarification": None,
                }
            ]
        }
    }

    session_id: str = Field(default="", description="세션 ID (다음 요청에 전달하면 대화 맥락 유지)")
    response: str = Field(default="", description="최종 응답 텍스트")
    intent: str = Field(default="", description="분류된 의도 (recommend/search/general/info/theater/booking)")
    emotion: str | None = Field(default=None, description="감지된 감정 (happy/sad/excited/angry/calm 또는 None)")
    movie_count: int = Field(default=0, description="추천된 영화 수 (최대 5)")
    image_analyzed: bool = Field(default=False, description="이미지 분석 수행 여부")
    clarification: dict | None = Field(default=None, description="후속 질문 힌트 (ClarificationResponse JSON)")


# ============================================================
# SSE 스트리밍 엔드포인트
# ============================================================

@chat_router.post(
    "/chat",
    summary="SSE 스트리밍 채팅",
    response_description="SSE 이벤트 스트림 (text/event-stream)",
    responses={
        200: {
            "description": "SSE 스트리밍 응답. 이벤트 타입: status, movie_card, token, done, error",
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: status\ndata: {"phase": "intent", "message": "의도 분석 중..."}\n\n'
                        'event: token\ndata: {"delta": "우울할 때 보면 좋은 영화를 추천해드릴게요!"}\n\n'
                        'event: movie_card\ndata: {"title": "인사이드 아웃", "genres": ["애니메이션", "가족"]}\n\n'
                        "event: done\ndata: {}\n\n"
                    )
                }
            },
        },
    },
)
async def chat_sse(request: ChatRequest):
    """
    SSE 스트리밍 채팅 엔드포인트.

    Chat Agent 그래프를 실행하며, 각 노드 완료 시 SSE 이벤트를 발행한다.
    Content-Type: text/event-stream

    SSE 이벤트:
    - status: 현재 처리 단계 (phase, message)
    - movie_card: 추천 영화 데이터 (RankedMovie JSON)
    - token: 응답 텍스트 (delta)
    - done: 완료 신호
    - error: 에러 메시지

    Args:
        request: ChatRequest (user_id, session_id, message)

    Returns:
        EventSourceResponse (SSE 스트리밍)
    """
    # 요청 수신 타이밍 측정 시작
    request_start = time.perf_counter()

    logger.info(
        "chat_sse_request",
        user_id=request.user_id or "(anonymous)",
        session_id=request.session_id,
        message_preview=request.message[:50],
        has_image=request.image is not None,
    )

    async def event_generator():
        """SSE 이벤트 생성기 — run_chat_agent()의 이벤트를 relay한다."""
        async for sse_event in run_chat_agent(
            user_id=request.user_id,
            session_id=request.session_id,
            message=request.message,
            image_data=request.image,
        ):
            yield sse_event
        # SSE 스트리밍 완료 시 타이밍 로깅
        elapsed_ms = (time.perf_counter() - request_start) * 1000
        logger.info(
            "chat_sse_completed",
            user_id=request.user_id or "(anonymous)",
            session_id=request.session_id,
            elapsed_ms=round(elapsed_ms, 1),
        )

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ============================================================
# 동기 엔드포인트 (디버그/테스트용)
# ============================================================

@chat_router.post(
    "/chat/sync",
    response_model=ChatSyncResponse,
    summary="동기 JSON 채팅 (디버그용)",
    responses={
        200: {
            "description": "채팅 응답 JSON. 추천 영화 수, 감정/의도 분류 결과 포함.",
        },
    },
)
async def chat_sync(request: ChatRequest):
    """
    동기 JSON 채팅 엔드포인트 (디버그/테스트용).

    Chat Agent 그래프를 동기 실행하고, 최종 State에서 주요 정보를 추출하여 JSON으로 반환한다.

    Args:
        request: ChatRequest (user_id, session_id, message)

    Returns:
        ChatSyncResponse (response, intent, emotion, movie_count)
    """
    # 요청 수신 타이밍 측정 시작
    request_start = time.perf_counter()

    logger.info(
        "chat_sync_request",
        user_id=request.user_id or "(anonymous)",
        session_id=request.session_id,
        message_preview=request.message[:50],
        has_image=request.image is not None,
    )

    state = await run_chat_agent_sync(
        user_id=request.user_id,
        session_id=request.session_id,
        message=request.message,
        image_data=request.image,
    )

    # State에서 응답 정보 추출
    # session_id는 graph.py에서 자동 생성되어 state에 포함됨
    result_session_id = state.get("session_id", request.session_id)
    intent = state.get("intent")
    emotion = state.get("emotion")
    ranked = state.get("ranked_movies", [])
    image_analysis = state.get("image_analysis")

    # 동기 요청 완료 타이밍 로깅
    elapsed_ms = (time.perf_counter() - request_start) * 1000
    logger.info(
        "chat_sync_completed",
        user_id=request.user_id or "(anonymous)",
        session_id=result_session_id,
        elapsed_ms=round(elapsed_ms, 1),
        intent=intent.intent if intent else "",
        movie_count=len(ranked),
    )

    # 후속 질문 힌트 추출
    clarification = state.get("clarification")
    clarification_dict = (
        clarification.model_dump()
        if clarification and hasattr(clarification, "model_dump")
        else None
    )

    return ChatSyncResponse(
        session_id=result_session_id,
        response=state.get("response", ""),
        intent=intent.intent if intent else "",
        emotion=emotion.emotion if emotion else None,
        movie_count=len(ranked),
        image_analyzed=bool(image_analysis and image_analysis.analyzed),
        clarification=clarification_dict,
    )


# ============================================================
# 멀티파트 이미지 업로드 엔드포인트
# ============================================================

@chat_router.post(
    "/chat/upload",
    summary="이미지 업로드 채팅 (멀티파트)",
    response_description="SSE 이벤트 스트림 (text/event-stream)",
    responses={
        200: {
            "description": "이미지 분석 후 SSE 스트리밍 응답. 이미지를 VLM으로 분석하여 영화 추천.",
            "content": {
                "text/event-stream": {
                    "example": (
                        'event: status\ndata: {"phase": "image_analysis", "message": "이미지 분석 중..."}\n\n'
                        'event: token\ndata: {"delta": "업로드하신 이미지를 분석해봤어요!"}\n\n'
                        "event: done\ndata: {}\n\n"
                    )
                }
            },
        },
        413: {
            "description": "이미지 크기 초과 (최대 10MB)",
            "content": {
                "application/json": {
                    "example": {"detail": "이미지 크기가 10MB를 초과합니다."}
                }
            },
        },
    },
)
async def chat_upload(
    message: str = Form(..., min_length=1, max_length=2000, description="사용자 입력 메시지"),
    user_id: str = Form(default="", description="사용자 ID"),
    session_id: str = Form(default="", description="세션 ID"),
    image: UploadFile | None = File(default=None, description="이미지 파일 (JPEG/PNG, 최대 10MB)"),
):
    """
    멀티파트 이미지 업로드 채팅 엔드포인트 (SSE 스트리밍).

    이미지 파일을 직접 업로드할 수 있다 (base64 변환 불필요).
    Content-Type: multipart/form-data

    Args:
        message: 사용자 입력 메시지 (필수)
        user_id: 사용자 ID (빈 문자열이면 익명)
        session_id: 세션 ID (빈 문자열이면 신규 세션)
        image: 이미지 파일 (JPEG/PNG, 최대 10MB)

    Returns:
        EventSourceResponse (SSE 스트리밍)
    """
    # 이미지 파일 → base64 변환
    image_data: str | None = None
    if image is not None:
        # 파일 크기 검증
        contents = await image.read()
        max_bytes = settings.IMAGE_MAX_SIZE_MB * 1024 * 1024
        if len(contents) > max_bytes:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=413,
                content={"detail": f"이미지 크기가 {settings.IMAGE_MAX_SIZE_MB}MB를 초과합니다."},
            )
        image_data = base64.b64encode(contents).decode("utf-8")

    # 요청 수신 타이밍 측정 시작
    upload_start = time.perf_counter()

    logger.info(
        "chat_upload_request",
        user_id=user_id or "(anonymous)",
        session_id=session_id,
        message_preview=message[:50],
        has_image=image_data is not None,
        image_size_kb=len(image_data) // 1024 if image_data else 0,
    )

    async def event_generator():
        """SSE 이벤트 생성기 — run_chat_agent()의 이벤트를 relay한다."""
        async for sse_event in run_chat_agent(
            user_id=user_id,
            session_id=session_id,
            message=message,
            image_data=image_data,
        ):
            yield sse_event
        # 업로드 요청 완료 타이밍 로깅
        elapsed_ms = (time.perf_counter() - upload_start) * 1000
        logger.info(
            "chat_upload_completed",
            user_id=user_id or "(anonymous)",
            session_id=session_id,
            elapsed_ms=round(elapsed_ms, 1),
        )

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
    )
