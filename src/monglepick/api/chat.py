"""
Chat Agent SSE/동기 엔드포인트 (§6 Phase 3).

3개 엔드포인트:
- POST /api/v1/chat       — SSE 스트리밍 (EventSourceResponse)
- POST /api/v1/chat/sync  — 동기 JSON (디버그/테스트용)
- POST /api/v1/chat/upload — 멀티파트 이미지 업로드 (파일+메시지)

보안 강화:
- Data URL 접두사 제거 + base64 패딩 보정 (_strip_base64_prefix)
- 매직바이트 기반 이미지 검증 (_validate_image_bytes)
- IP당 분당 업로드 횟수 제한 (_check_upload_rate_limit)
- VLM 동시 처리 Semaphore (_vlm_semaphore)
- Pillow DecompressionBomb 방어 (IMAGE_MAX_PIXELS)

요청 모델: ChatRequest (user_id, session_id, message, image)
응답 모델: ChatSyncResponse (동기 전용)
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import re
import time

import structlog
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import jwt

from monglepick.agents.chat.graph import run_chat_agent, run_chat_agent_sync
from monglepick.config import settings
from monglepick.db.clients import get_redis

logger = structlog.get_logger()


# ============================================================
# 보안 상수 + 모듈 레벨 상태
# ============================================================

# 허용 이미지 MIME 타입 → 매직바이트 매핑
_IMAGE_MAGIC_BYTES: dict[str, bytes] = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png": b"\x89PNG",
}

# settings에서 허용 MIME 타입 세트 구성
_ALLOWED_MIMES: set[str] = set(settings.ALLOWED_IMAGE_MIMES.split(","))

# VLM 동시 처리 세마포어 — GPU 메모리 보호
_vlm_semaphore = asyncio.Semaphore(settings.VLM_CONCURRENCY_LIMIT)

# Chat Agent 그래프 동시 실행 세마포어 — Ollama 과부하 방지.
# MAX_CONCURRENT_REQUESTS(기본 3)개를 초과하는 요청은 큐에 대기하며
# SSE로 "대기 중" 알림을 전송한다.
_graph_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

# Rate Limiting: Redis 기반 슬라이딩 윈도우 (서버 재시작/멀티 인스턴스에서도 유지)
_RATE_LIMIT_KEY_PREFIX: str = "rate_limit:upload:"
_RATE_LIMIT_WINDOW_SEC: int = 60

# Data URL 접두사 패턴: "data:image/jpeg;base64," 또는 "data:image/png;base64," 등
_DATA_URL_RE = re.compile(r"^data:[^;]+;base64,", re.IGNORECASE)


# ============================================================
# JWT 검증 (Client → Agent 요청의 user_id 위조 방지)
# ============================================================

def _extract_user_id_from_jwt(raw_request: Request) -> str | None:
    """
    Authorization 헤더에서 JWT를 추출하고 user_id를 반환한다.

    JWT_SECRET이 미설정이면 검증을 건너뛴다 (개발 환경 호환).
    JWT가 유효하면 subject(user_id)를 반환하고, 유효하지 않으면 None을 반환한다.

    Args:
        raw_request: FastAPI Request 객체

    Returns:
        JWT에서 추출한 user_id 또는 None (JWT 없음/무효/미설정)
    """
    # JWT_SECRET 미설정 → 검증 건너뜀 (개발 환경)
    if not settings.JWT_SECRET:
        return None

    # Authorization 헤더 추출
    auth_header = raw_request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:]  # "Bearer " 이후

    try:
        # Backend와 동일한 HS256 알고리즘으로 검증
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256"],
        )
        # Refresh Token은 거부 (access token만 허용)
        if payload.get("type") == "refresh":
            logger.warning("jwt_refresh_token_rejected")
            return None
        # subject = user_id
        user_id = payload.get("sub", "")
        if user_id:
            return user_id
        return None
    except jwt.ExpiredSignatureError:
        logger.debug("jwt_expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug("jwt_invalid", error=str(e))
        return None


def _resolve_user_id(request_user_id: str, raw_request: Request) -> str:
    """
    JWT와 요청 body의 user_id를 비교하여 최종 user_id를 결정한다.

    우선순위:
    1. JWT가 유효하면 JWT의 user_id를 사용 (body의 user_id 무시)
    2. JWT가 없거나 무효하고 JWT_SECRET이 설정되어 있으면 body의 user_id도 거부 → 익명
    3. JWT_SECRET이 미설정이면 body의 user_id를 그대로 사용 (개발 환경 호환)

    Args:
        request_user_id: 요청 body의 user_id
        raw_request: FastAPI Request 객체

    Returns:
        검증된 user_id (빈 문자열이면 익명)
    """
    jwt_user_id = _extract_user_id_from_jwt(raw_request)

    if jwt_user_id:
        # JWT 유효 → JWT의 user_id 사용
        if request_user_id and request_user_id != jwt_user_id:
            logger.warning(
                "user_id_mismatch_jwt_overrides",
                body_user_id=request_user_id,
                jwt_user_id=jwt_user_id,
            )
        return jwt_user_id

    if settings.JWT_SECRET:
        # JWT_SECRET 설정됨 + JWT 없음/무효 → body의 user_id 무시 (스푸핑 방지)
        if request_user_id:
            logger.warning(
                "no_valid_jwt_body_user_id_ignored",
                body_user_id=request_user_id,
            )
        return ""  # 익명 처리

    # JWT_SECRET 미설정 → 개발 환경, body의 user_id 그대로 사용
    return request_user_id


# ============================================================
# 보안 헬퍼 함수
# ============================================================

def _strip_base64_prefix(data: str) -> str:
    """
    Data URL 접두사를 제거하고 base64 패딩을 보정한다.

    프론트엔드에서 `data:image/png;base64,...` 형태의 Data URL을 보내면
    base64.b64decode()가 접두사 때문에 `binascii.Error: Incorrect padding`을
    발생시킨다. 이 함수는:
    1. Data URL 접두사(`data:image/...;base64,`) 제거
    2. base64 패딩 보정 (4의 배수가 되도록 `=` 추가)

    Args:
        data: base64 인코딩된 문자열 (Data URL 접두사 포함 가능)

    Returns:
        접두사가 제거되고 패딩이 보정된 순수 base64 문자열
    """
    # 1. Data URL 접두사 제거 ("data:image/jpeg;base64," → "")
    stripped = _DATA_URL_RE.sub("", data)

    # 2. base64 패딩 보정 — 길이가 4의 배수가 아니면 `=`로 채움
    remainder = len(stripped) % 4
    if remainder:
        stripped += "=" * (4 - remainder)

    return stripped


def _validate_image_bytes(image_bytes: bytes) -> None:
    """
    이미지 바이트의 매직바이트를 검증하여 허용된 이미지 형식인지 확인한다.

    JPEG(FF D8 FF)와 PNG(89 50 4E 47)만 허용한다.
    GIF, SVG, 실행 파일 등 다른 형식은 거부한다.

    Args:
        image_bytes: 디코딩된 원본 이미지 바이트

    Raises:
        ValueError: 빈 데이터일 때 (status_code=400)
        ValueError: 매직바이트가 허용 목록에 없을 때 (status_code=415)
    """
    if not image_bytes:
        raise ValueError("400:이미지 데이터가 비어있습니다.")

    # 매직바이트로 실제 파일 형식 확인
    for mime, magic in _IMAGE_MAGIC_BYTES.items():
        if mime in _ALLOWED_MIMES and image_bytes[:len(magic)] == magic:
            return  # 허용된 형식

    raise ValueError("415:허용되지 않는 이미지 형식입니다. JPEG 또는 PNG만 지원합니다.")


async def _check_upload_rate_limit(client_ip: str) -> None:
    """
    IP당 분당 이미지 업로드 횟수를 Redis 기반 슬라이딩 윈도우로 제한한다.

    Redis Sorted Set을 사용하여 서버 재시작 및 멀티 인스턴스 환경에서도
    Rate Limiting이 유지된다. 60초 이전의 타임스탬프는 자동 만료된다.

    Args:
        client_ip: 클라이언트 IP 주소

    Raises:
        ValueError: 분당 업로드 한도 초과 시 (status_code=429)
    """
    now = time.time()
    key = f"{_RATE_LIMIT_KEY_PREFIX}{client_ip}"

    try:
        redis = await get_redis()

        # 파이프라인으로 원자적 실행: 만료 제거 → 카운트 조회 → 추가 → TTL 설정
        pipe = redis.pipeline()
        # 1. 윈도우 밖의 오래된 타임스탬프 제거
        pipe.zremrangebyscore(key, 0, now - _RATE_LIMIT_WINDOW_SEC)
        # 2. 현재 윈도우 내 요청 수 조회
        pipe.zcard(key)
        # 3. 현재 요청 타임스탬프 추가
        pipe.zadd(key, {str(now): now})
        # 4. 키 TTL 설정 (윈도우 + 여유 10초, 자동 정리)
        pipe.expire(key, _RATE_LIMIT_WINDOW_SEC + 10)
        results = await pipe.execute()

        # results[1] = zcard 결과 (추가 전 카운트)
        current_count = results[1]

        if current_count >= settings.IMAGE_UPLOAD_RATE_LIMIT:
            # 한도 초과: 방금 추가한 타임스탬프 제거 (롤백)
            await redis.zrem(key, str(now))
            raise ValueError("429:이미지 업로드 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")

    except ValueError:
        # ValueError는 Rate Limit 초과 — 그대로 re-raise
        raise
    except Exception as e:
        # Redis 연결 실패 시 요청을 차단하지 않고 경고만 남긴다 (graceful degradation)
        logger.warning(
            "rate_limit_redis_error",
            client_ip=client_ip,
            error=str(e),
            error_type=type(e).__name__,
        )


def _handle_security_error(error: ValueError) -> JSONResponse:
    """
    보안 헬퍼의 ValueError를 HTTP 응답으로 변환한다.

    에러 메시지 형식: "status_code:detail_message"

    Args:
        error: 보안 헬퍼에서 발생한 ValueError

    Returns:
        적절한 HTTP 상태 코드의 JSONResponse
    """
    msg = str(error)
    if ":" in msg:
        code_str, detail = msg.split(":", 1)
        try:
            status_code = int(code_str)
        except ValueError:
            status_code = 400
            detail = msg
    else:
        status_code = 400
        detail = msg

    return JSONResponse(status_code=status_code, content={"detail": detail})


# ============================================================
# 이미지 리사이즈 헬퍼
# ============================================================

def _resize_image_bytes(
    image_bytes: bytes,
    max_dim: int | None = None,
) -> bytes:
    """
    이미지 바이트를 max_dim 이하로 리사이즈한다.

    긴 변이 max_dim을 초과하면 비율을 유지하여 축소하고,
    EXIF 회전 보정 후 JPEG quality=85로 압축한다.
    이미 작은 이미지는 원본 바이트를 그대로 반환한다.

    DecompressionBomb 방어: Image.open() 전에 MAX_IMAGE_PIXELS를 설정하여
    과도하게 큰 이미지(예: 100,000×100,000px)에 의한 메모리 폭발을 방지한다.

    Args:
        image_bytes: 원본 이미지 바이트 (JPEG/PNG 등)
        max_dim: 최대 변 길이 (px). None이면 settings.IMAGE_MAX_DIMENSION 사용

    Returns:
        리사이즈된 JPEG 이미지 바이트
    """
    max_dim = max_dim or settings.IMAGE_MAX_DIMENSION
    original_size = len(image_bytes)

    try:
        # DecompressionBomb 방어: 허용 최대 픽셀 수 설정
        Image.MAX_IMAGE_PIXELS = settings.IMAGE_MAX_PIXELS

        img = Image.open(io.BytesIO(image_bytes))

        # EXIF 회전 보정 (사진이 90/180/270도 회전된 경우 정상 방향으로 복원)
        img = ImageOps.exif_transpose(img)

        # 긴 변이 max_dim 이하이면 리사이즈 불필요
        if max(img.size) <= max_dim:
            logger.debug(
                "image_resize_skipped",
                width=img.size[0],
                height=img.size[1],
                size_kb=original_size // 1024,
                reason="already_small_enough",
            )
            return image_bytes

        # 비율 유지하여 축소 (LANCZOS: 고품질 다운샘플링)
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        # RGB 변환 (PNG 투명도 등 처리) + JPEG 압축
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        resized_bytes = buf.getvalue()

        logger.info(
            "image_resized",
            original_size_kb=original_size // 1024,
            resized_size_kb=len(resized_bytes) // 1024,
            width=img.size[0],
            height=img.size[1],
            max_dim=max_dim,
        )
        return resized_bytes

    except Exception as e:
        # 리사이즈 실패 시 원본 반환 (에러 전파 금지)
        logger.warning(
            "image_resize_failed",
            error=str(e),
            original_size_kb=original_size // 1024,
        )
        return image_bytes

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
                    "response": "우울할 때 보면 좋은 영화를 추천해드릴게요!\n\n1. **인사이드 아웃** ...",
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
        400: {"description": "base64 디코드 실패 또는 빈 이미지 데이터"},
        415: {"description": "허용되지 않는 이미지 형식 (JPEG/PNG만 지원)"},
        429: {"description": "이미지 업로드 Rate Limit 초과"},
    },
)
async def chat_sse(request: ChatRequest, raw_request: Request):
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
        request: ChatRequest (user_id, session_id, message, image)
        raw_request: FastAPI Request (클라이언트 IP 추출용)

    Returns:
        EventSourceResponse (SSE 스트리밍)
    """
    # 요청 수신 타이밍 측정 시작
    request_start = time.perf_counter()

    # JWT 검증: Authorization 헤더에서 user_id 추출 (body의 user_id보다 우선)
    verified_user_id = _resolve_user_id(request.user_id, raw_request)
    request.user_id = verified_user_id

    # base64 이미지가 있으면 보안 검증 → 디코드 → 리사이즈 → 재인코딩
    image_data = request.image
    if image_data:
        # Rate Limiting — IP당 분당 업로드 횟수 제한
        client_ip = raw_request.client.host if raw_request.client else "unknown"
        try:
            await _check_upload_rate_limit(client_ip)
        except ValueError as e:
            return _handle_security_error(e)

        # Data URL 접두사 제거 + base64 패딩 보정
        stripped = _strip_base64_prefix(image_data)

        # base64 디코드
        try:
            raw_bytes = base64.b64decode(stripped)
        except (binascii.Error, ValueError) as e:
            logger.warning("chat_sse_base64_error", error=str(e))
            return JSONResponse(
                status_code=400,
                content={"detail": "base64 이미지 디코딩에 실패했습니다."},
            )

        # 매직바이트 검증 — JPEG/PNG만 허용
        try:
            _validate_image_bytes(raw_bytes)
        except ValueError as e:
            return _handle_security_error(e)

        # 리사이즈 + 재인코딩
        resized_bytes = _resize_image_bytes(raw_bytes)
        image_data = base64.b64encode(resized_bytes).decode("utf-8")

    logger.info(
        "chat_sse_request",
        user_id=request.user_id or "(anonymous)",
        session_id=request.session_id,
        message_preview=request.message[:50],
        has_image=image_data is not None,
    )

    async def event_generator():
        """SSE 이벤트 생성기 — 포인트 체크 + 글로벌/VLM 세마포어로 동시 처리 제한 후 relay."""
        import json as _json

        # ── 포인트 사전 체크 + 쿼터 검증 (익명 사용자는 생략) ──
        # 과금 단위: "추천 완료" (movie_card 발행 시점)
        # 사전 체크에서는 차감하지 않고 잔액/쿼터만 확인하여 조기 차단한다.
        # AI 후속 질문만 하는 턴에서는 포인트가 차감되지 않는다.
        # effective_cost: 실제 차감 포인트 (무료 잔여가 있으면 0). graph.py deduct에 전달.
        from monglepick.config import settings as _settings
        _effective_cost: int = _settings.POINT_COST_PER_RECOMMENDATION  # 기본값 (체크 실패 시 사용)
        if _settings.POINT_CHECK_ENABLED and request.user_id:
            from monglepick.api.point_client import check_point
            point_check = await check_point(
                user_id=request.user_id,
                cost=_settings.POINT_COST_PER_RECOMMENDATION,
            )
            # effective_cost를 로컬 변수에 저장 (graph.py deduct에 전달)
            _effective_cost = point_check.effective_cost

            # 1) 사용자 입력 글자수 제한 검증 (등급별 max_input_length)
            message_text = request.message or ""
            if len(message_text) > point_check.max_input_length:
                logger.info(
                    "chat_sse_input_too_long",
                    user_id=request.user_id,
                    max_input_length=point_check.max_input_length,
                    current_length=len(message_text),
                )
                yield {
                    "event": "error",
                    "data": _json.dumps({
                        "message": f"입력 글자수가 {point_check.max_input_length}자를 초과했습니다. (현재: {len(message_text)}자)",
                        "error_code": "INPUT_TOO_LONG",
                        "max_input_length": point_check.max_input_length,
                        "current_length": len(message_text),
                    }, ensure_ascii=False),
                }
                yield {"event": "done", "data": "{}"}
                return

            # 2) 포인트/쿼터 부족 → 그래프 실행 없이 에러 반환
            if not point_check.allowed:
                logger.info(
                    "chat_sse_point_insufficient",
                    user_id=request.user_id,
                    balance=point_check.balance,
                    cost=point_check.effective_cost,
                    daily_used=point_check.daily_used,
                    daily_limit=point_check.daily_limit,
                )
                yield {
                    "event": "error",
                    "data": _json.dumps({
                        "message": point_check.message,
                        "error_code": "INSUFFICIENT_POINT",
                        "balance": point_check.balance,
                        "cost": point_check.effective_cost,
                        "needs_purchase": True,
                        "daily_used": point_check.daily_used,
                        "daily_limit": point_check.daily_limit,
                        "monthly_used": point_check.monthly_used,
                        "monthly_limit": point_check.monthly_limit,
                    }, ensure_ascii=False),
                }
                yield {"event": "done", "data": "{}"}
                return

        # 글로벌 그래프 세마포어 — 동시 실행 요청 수 제한
        # 슬롯이 없으면 대기 중 SSE 알림을 먼저 전송
        if _graph_semaphore.locked():
            # 대기 중임을 사용자에게 SSE로 알림
            yield {
                "event": "status",
                "data": _json.dumps(
                    {"phase": "queued", "message": "요청이 많아 잠시 대기 중이에요..."},
                    ensure_ascii=False,
                ),
            }
            logger.info(
                "chat_sse_queued",
                user_id=request.user_id or "(anonymous)",
                session_id=request.session_id,
            )

        async with _graph_semaphore:
            # VLM 세마포어 — 이미지가 있으면 동시 처리 수를 제한
            if image_data:
                async with _vlm_semaphore:
                    async for sse_event in run_chat_agent(
                        user_id=request.user_id,
                        session_id=request.session_id,
                        message=request.message,
                        image_data=image_data,
                        effective_cost=_effective_cost,
                    ):
                        yield sse_event
            else:
                async for sse_event in run_chat_agent(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    message=request.message,
                    image_data=image_data,
                    effective_cost=_effective_cost,
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
        400: {"description": "base64 디코드 실패 또는 빈 이미지 데이터"},
        415: {"description": "허용되지 않는 이미지 형식 (JPEG/PNG만 지원)"},
    },
)
async def chat_sync(request: ChatRequest, raw_request: Request):
    """
    동기 JSON 채팅 엔드포인트 (디버그/테스트용).

    Chat Agent 그래프를 동기 실행하고, 최종 State에서 주요 정보를 추출하여 JSON으로 반환한다.

    Args:
        request: ChatRequest (user_id, session_id, message)
        raw_request: FastAPI Request (JWT 추출용)

    Returns:
        ChatSyncResponse (response, intent, emotion, movie_count)
    """
    # 요청 수신 타이밍 측정 시작
    request_start = time.perf_counter()

    # JWT 검증: Authorization 헤더에서 user_id 추출
    verified_user_id = _resolve_user_id(request.user_id, raw_request)
    request.user_id = verified_user_id

    # base64 이미지가 있으면 보안 검증
    image_for_agent = request.image
    if image_for_agent:
        # Data URL 접두사 제거 + base64 패딩 보정
        stripped = _strip_base64_prefix(image_for_agent)

        # base64 디코드 검증
        try:
            raw_bytes = base64.b64decode(stripped)
        except (binascii.Error, ValueError) as e:
            logger.warning("chat_sync_base64_error", error=str(e))
            return JSONResponse(
                status_code=400,
                content={"detail": "base64 이미지 디코딩에 실패했습니다."},
            )

        # 매직바이트 검증 — JPEG/PNG만 허용
        try:
            _validate_image_bytes(raw_bytes)
        except ValueError as e:
            return _handle_security_error(e)

        # 검증 통과 시 정제된 base64 사용
        image_for_agent = stripped

    logger.info(
        "chat_sync_request",
        user_id=request.user_id or "(anonymous)",
        session_id=request.session_id,
        message_preview=request.message[:50],
        has_image=image_for_agent is not None,
    )

    # 글로벌 그래프 세마포어 + VLM 세마포어로 동시 처리 제한
    # 동기 엔드포인트는 디버그/테스트 전용 → 포인트 체크 없이 기본값(0) 전달
    async with _graph_semaphore:
        if image_for_agent:
            async with _vlm_semaphore:
                state = await run_chat_agent_sync(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    message=request.message,
                    image_data=image_for_agent,
                    effective_cost=0,
                )
        else:
            state = await run_chat_agent_sync(
                user_id=request.user_id,
                session_id=request.session_id,
                message=request.message,
                image_data=image_for_agent,
                effective_cost=0,
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
        415: {"description": "허용되지 않는 이미지 형식 (JPEG/PNG만 지원)"},
        429: {"description": "이미지 업로드 Rate Limit 초과"},
    },
)
async def chat_upload(
    raw_request: Request,
    message: str = Form(..., min_length=1, max_length=2000, description="사용자 입력 메시지"),
    user_id: str = Form(default="", description="사용자 ID"),
    session_id: str = Form(default="", description="세션 ID"),
    image: UploadFile | None = File(default=None, description="이미지 파일 (JPEG/PNG, 최대 10MB)"),
):
    """
    멀티파트 이미지 업로드 채팅 엔드포인트 (SSE 스트리밍).

    이미지 파일을 직접 업로드할 수 있다 (base64 변환 불필요).
    Content-Type: multipart/form-data

    보안 검증:
    - MIME 타입 확인 (Content-Type 헤더)
    - 매직바이트 검증 (파일 내용 실제 검사)
    - 파일 크기 제한 (IMAGE_MAX_SIZE_MB)
    - IP당 분당 업로드 횟수 제한

    Args:
        raw_request: FastAPI Request (클라이언트 IP 추출용)
        message: 사용자 입력 메시지 (필수)
        user_id: 사용자 ID (빈 문자열이면 익명)
        session_id: 세션 ID (빈 문자열이면 신규 세션)
        image: 이미지 파일 (JPEG/PNG, 최대 10MB)

    Returns:
        EventSourceResponse (SSE 스트리밍)
    """
    # JWT 검증: Authorization 헤더에서 user_id 추출
    verified_user_id = _resolve_user_id(user_id, raw_request)
    user_id = verified_user_id

    # 이미지 파일 → 보안 검증 → 리사이즈 → base64 변환
    image_data: str | None = None
    if image is not None:
        # Rate Limiting — IP당 분당 업로드 횟수 제한
        client_ip = raw_request.client.host if raw_request.client else "unknown"
        try:
            await _check_upload_rate_limit(client_ip)
        except ValueError as e:
            return _handle_security_error(e)

        # MIME 타입 검증 (Content-Type 헤더 기반, 1차 방어)
        if image.content_type and image.content_type not in _ALLOWED_MIMES:
            logger.warning(
                "chat_upload_mime_rejected",
                content_type=image.content_type,
                allowed=list(_ALLOWED_MIMES),
            )
            return JSONResponse(
                status_code=415,
                content={"detail": f"허용되지 않는 파일 형식입니다: {image.content_type}. JPEG 또는 PNG만 지원합니다."},
            )

        # 파일 크기 검증
        contents = await image.read()
        max_bytes = settings.IMAGE_MAX_SIZE_MB * 1024 * 1024
        if len(contents) > max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"이미지 크기가 {settings.IMAGE_MAX_SIZE_MB}MB를 초과합니다."},
            )

        # 매직바이트 검증 (파일 내용 실제 검사, 2차 방어 — MIME 스푸핑 방지)
        try:
            _validate_image_bytes(contents)
        except ValueError as e:
            return _handle_security_error(e)

        # 리사이즈 후 base64 인코딩 (원본 10MB → ~200KB로 축소)
        resized = _resize_image_bytes(contents)
        image_data = base64.b64encode(resized).decode("utf-8")

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
        """SSE 이벤트 생성기 — 포인트 체크 + 글로벌/VLM 세마포어로 동시 처리 제한 후 relay."""
        import json as _json

        # ── 포인트 사전 체크 + 쿼터 검증 (업로드 엔드포인트) ──
        from monglepick.config import settings as _settings
        _effective_cost_upload: int = _settings.POINT_COST_PER_RECOMMENDATION  # 기본값
        if _settings.POINT_CHECK_ENABLED and user_id:
            from monglepick.api.point_client import check_point
            point_check = await check_point(
                user_id=user_id,
                cost=_settings.POINT_COST_PER_RECOMMENDATION,
            )
            # effective_cost를 로컬 변수에 저장 (graph.py deduct에 전달)
            _effective_cost_upload = point_check.effective_cost

            # 1) 사용자 입력 글자수 제한 검증 (등급별 max_input_length)
            if len(message) > point_check.max_input_length:
                logger.info(
                    "chat_upload_input_too_long",
                    user_id=user_id,
                    max_input_length=point_check.max_input_length,
                    current_length=len(message),
                )
                yield {
                    "event": "error",
                    "data": _json.dumps({
                        "message": f"입력 글자수가 {point_check.max_input_length}자를 초과했습니다. (현재: {len(message)}자)",
                        "error_code": "INPUT_TOO_LONG",
                        "max_input_length": point_check.max_input_length,
                        "current_length": len(message),
                    }, ensure_ascii=False),
                }
                yield {"event": "done", "data": "{}"}
                return

            # 2) 포인트/쿼터 부족 → 그래프 실행 없이 에러 반환
            if not point_check.allowed:
                logger.info(
                    "chat_upload_point_insufficient",
                    user_id=user_id,
                    balance=point_check.balance,
                    cost=point_check.effective_cost,
                    daily_used=point_check.daily_used,
                    daily_limit=point_check.daily_limit,
                )
                yield {
                    "event": "error",
                    "data": _json.dumps({
                        "message": point_check.message,
                        "error_code": "INSUFFICIENT_POINT",
                        "balance": point_check.balance,
                        "cost": point_check.effective_cost,
                        "needs_purchase": True,
                        "daily_used": point_check.daily_used,
                        "daily_limit": point_check.daily_limit,
                        "monthly_used": point_check.monthly_used,
                        "monthly_limit": point_check.monthly_limit,
                    }, ensure_ascii=False),
                }
                yield {"event": "done", "data": "{}"}
                return

        # 글로벌 그래프 세마포어 — 동시 실행 요청 수 제한
        if _graph_semaphore.locked():
            yield {
                "event": "status",
                "data": _json.dumps(
                    {"phase": "queued", "message": "요청이 많아 잠시 대기 중이에요..."},
                    ensure_ascii=False,
                ),
            }
            logger.info(
                "chat_upload_queued",
                user_id=user_id or "(anonymous)",
                session_id=session_id,
            )

        async with _graph_semaphore:
            # VLM 세마포어 — 이미지가 있으면 동시 처리 수를 제한
            if image_data:
                async with _vlm_semaphore:
                    async for sse_event in run_chat_agent(
                        user_id=user_id,
                        session_id=session_id,
                        message=message,
                        image_data=image_data,
                        effective_cost=_effective_cost_upload,
                    ):
                        yield sse_event
            else:
                async for sse_event in run_chat_agent(
                    user_id=user_id,
                    session_id=session_id,
                    message=message,
                    image_data=image_data,
                    effective_cost=_effective_cost_upload,
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
