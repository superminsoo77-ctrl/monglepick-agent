"""
관리자 AI 에이전트 — Tier 0 AI 운영 Read-only Tool (7개).

설계서: docs/관리자_AI에이전트_v3_재설계.md §4.1

Backend `AdminAiController` (/api/v1/admin/ai prefix):
- quizzes_list                  — GET /api/v1/admin/ai/quiz/history?page&size
- chatbot_sessions_list         — GET /api/v1/admin/ai/chat/sessions?page&size[&userId]
- chatbot_session_messages      — GET /api/v1/admin/ai/chat/sessions/{sessionId}/messages
- chatbot_stats                 — GET /api/v1/admin/ai/chat/stats
- review_verifications_list     — GET /api/v1/admin/ai/review-verification/queue?status&page&size
- review_verification_detail    — GET /api/v1/admin/ai/review-verification/{id}
- review_verifications_overview — GET /api/v1/admin/ai/review-verification/overview

Role matrix (§4.2):
- 퀴즈/챗봇 계열 → SUPER_ADMIN, ADMIN, AI_OPS_ADMIN
- review_verification 계열 → SUPER_ADMIN, ADMIN, AI_OPS_ADMIN, MODERATOR

설계 결정 — 퀴즈 운영은 자연어 흐름 밖이다:
    퀴즈 _생성_ 자체는 AiTriggerPanel 폼+버튼만으로 트리거되므로 자연어 ReAct
    루프에 들어오지 않는다. 통계 가시성은 GenerationHistory 상단의
    QuizStatsCard 화면 컴포넌트가 GET /admin/ai/quiz/stats 를 직접 호출하여
    제공한다 (2026-04-28). 따라서 자연어 quiz_stats Read tool 은 도입하지 않는다.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from monglepick.api.admin_backend_client import (
    AdminApiResult,
    get_admin_json,
    unwrap_api_response,
)
from monglepick.tools.admin_tools import (
    ToolContext,
    ToolSpec,
    register_tool,
)


# ============================================================
# Role matrix
# ============================================================

# 퀴즈·챗봇 계열 — AI_OPS_ADMIN 포함
_AI_OPS_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN", "AI_OPS_ADMIN"}

# 리뷰 검증 계열 — 콘텐츠 모더레이션 담당 MODERATOR 추가
_REVIEW_VERIFICATION_ROLES: set[str] = {"SUPER_ADMIN", "ADMIN", "AI_OPS_ADMIN", "MODERATOR"}


# ============================================================
# Args Schemas
# ============================================================

class _QuizzesListArgs(BaseModel):
    """퀴즈 생성 이력 조회 args."""

    page: int = Field(default=0, ge=0, description="페이지 번호 (0-indexed).")
    size: int = Field(default=20, ge=1, le=100, description="페이지 크기 (최대 100).")


class _ChatbotSessionsListArgs(BaseModel):
    """챗봇 세션 목록 조회 args."""

    page: int = Field(default=0, ge=0, description="페이지 번호 (0-indexed).")
    size: int = Field(default=20, ge=1, le=100, description="페이지 크기 (최대 100).")
    userId: Optional[str] = Field(
        default=None,
        description="특정 사용자 ID로 필터링. 생략하면 전체 세션 조회.",
    )


class _ChatbotSessionMessagesArgs(BaseModel):
    """챗봇 세션 상세 메시지 조회 args."""

    sessionId: str = Field(description="조회할 챗봇 세션 ID (필수 path variable).")


class _NoArgs(BaseModel):
    """파라미터 없는 EP 용 빈 스키마."""

    pass


class _ReviewVerificationsListArgs(BaseModel):
    """리뷰 검증 큐 목록 조회 args."""

    status: Literal["", "PENDING", "APPROVED", "REJECTED", "FLAGGED"] = Field(
        default="",
        description=(
            "검증 상태 필터. "
            "'PENDING'=대기, 'APPROVED'=승인, 'REJECTED'=거절, 'FLAGGED'=플래그. "
            "빈 문자열이면 전체."
        ),
    )
    page: int = Field(default=0, ge=0, description="페이지 번호 (0-indexed).")
    size: int = Field(default=20, ge=1, le=100, description="페이지 크기 (최대 100).")


class _ReviewVerificationDetailArgs(BaseModel):
    """리뷰 검증 단건 상세 조회 args."""

    id: int = Field(description="조회할 리뷰 검증 레코드 ID (Long, 필수).")


# ============================================================
# Handlers
# ============================================================

async def _handle_quizzes_list(
    ctx: ToolContext,
    page: int = 0,
    size: int = 20,
) -> AdminApiResult:
    """`GET /api/v1/admin/ai/quiz/history?page=...&size=...` 호출."""
    raw = await get_admin_json(
        "/api/v1/admin/ai/quiz/history",
        admin_jwt=ctx.admin_jwt,
        params={"page": page, "size": size},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_chatbot_sessions_list(
    ctx: ToolContext,
    page: int = 0,
    size: int = 20,
    userId: Optional[str] = None,
) -> AdminApiResult:
    """`GET /api/v1/admin/ai/chat/sessions?page=...&size=...[&userId=...]` 호출."""
    params: dict = {"page": page, "size": size}
    if userId:
        params["userId"] = userId
    raw = await get_admin_json(
        "/api/v1/admin/ai/chat/sessions",
        admin_jwt=ctx.admin_jwt,
        params=params,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_chatbot_session_messages(
    ctx: ToolContext,
    sessionId: str,
) -> AdminApiResult:
    """`GET /api/v1/admin/ai/chat/sessions/{sessionId}/messages` 호출.

    path variable(sessionId) 을 f-string 으로 경로에 조립한 뒤 get_admin_json 을 호출한다.
    """
    raw = await get_admin_json(
        f"/api/v1/admin/ai/chat/sessions/{sessionId}/messages",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_chatbot_stats(
    ctx: ToolContext,
) -> AdminApiResult:
    """`GET /api/v1/admin/ai/chat/stats` 호출 (파라미터 없음)."""
    raw = await get_admin_json(
        "/api/v1/admin/ai/chat/stats",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_review_verifications_list(
    ctx: ToolContext,
    status: str = "",
    page: int = 0,
    size: int = 20,
) -> AdminApiResult:
    """`GET /api/v1/admin/ai/review-verification/queue?status=...&page=...&size=...` 호출."""
    params: dict = {"page": page, "size": size}
    if status:
        params["status"] = status
    raw = await get_admin_json(
        "/api/v1/admin/ai/review-verification/queue",
        admin_jwt=ctx.admin_jwt,
        params=params,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_review_verification_detail(
    ctx: ToolContext,
    id: int,
) -> AdminApiResult:
    """`GET /api/v1/admin/ai/review-verification/{id}` 호출.

    path variable(id) 을 f-string 으로 조립.
    """
    raw = await get_admin_json(
        f"/api/v1/admin/ai/review-verification/{id}",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_review_verifications_overview(
    ctx: ToolContext,
) -> AdminApiResult:
    """`GET /api/v1/admin/ai/review-verification/overview` 호출 (파라미터 없음)."""
    raw = await get_admin_json(
        "/api/v1/admin/ai/review-verification/overview",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration
# ============================================================

register_tool(ToolSpec(
    name="quizzes_list",
    tier=0,
    required_roles=_AI_OPS_ROLES,
    description=(
        "AI 퀴즈 생성 이력 페이징 조회. 퀴즈 생성 일시·대상 영화·생성 상태를 확인한다. "
        "'최근에 생성된 퀴즈 있어?', '퀴즈 이력 보여줘' 질문에 사용한다."
    ),
    example_questions=[
        "최근 생성된 퀴즈 목록 보여줘",
        "퀴즈 생성 이력 확인",
        "AI 퀴즈 얼마나 만들어졌어?",
    ],
    args_schema=_QuizzesListArgs,
    handler=_handle_quizzes_list,
))


register_tool(ToolSpec(
    name="chatbot_sessions_list",
    tier=0,
    required_roles=_AI_OPS_ROLES,
    description=(
        "AI 챗봇 세션 목록 페이징 조회. 전체 세션 또는 특정 userId 기준으로 필터링 가능. "
        "'챗봇 세션 몇 개야?', '특정 유저 챗봇 대화 확인' 질문에 사용한다."
    ),
    example_questions=[
        "전체 챗봇 세션 목록",
        "유저 ID 123 챗봇 세션 목록 보여줘",
        "최근 챗봇 대화 세션 얼마나 있어?",
    ],
    args_schema=_ChatbotSessionsListArgs,
    handler=_handle_chatbot_sessions_list,
))


register_tool(ToolSpec(
    name="chatbot_session_messages",
    tier=0,
    required_roles=_AI_OPS_ROLES,
    description=(
        "특정 챗봇 세션의 대화 메시지 상세 조회. sessionId 로 세션을 지정한다. "
        "'세션 abc 대화 내용 보여줘', '챗봇 세션 메시지 확인' 질문에 사용한다."
    ),
    example_questions=[
        "세션 ID abc123 대화 내용",
        "챗봇 세션 메시지 상세 확인",
        "특정 세션 대화 기록 보고 싶어",
    ],
    args_schema=_ChatbotSessionMessagesArgs,
    handler=_handle_chatbot_session_messages,
))


register_tool(ToolSpec(
    name="chatbot_stats",
    tier=0,
    required_roles=_AI_OPS_ROLES,
    description=(
        "AI 챗봇 전체 통계 조회. 총 세션 수·평균 턴 수·일별 활성 세션 등 운영 지표를 반환. "
        "'챗봇 이용 통계', '챗봇 총 대화 수' 질문에 사용한다. 파라미터 없음."
    ),
    example_questions=[
        "챗봇 이용 현황 통계",
        "AI 챗봇 총 세션 수 알려줘",
        "챗봇 평균 대화 턴 얼마나 돼?",
    ],
    args_schema=_NoArgs,
    handler=_handle_chatbot_stats,
))


register_tool(ToolSpec(
    name="review_verifications_list",
    tier=0,
    required_roles=_REVIEW_VERIFICATION_ROLES,
    description=(
        "AI 리뷰 검증 큐 목록 페이징 조회. status(PENDING/APPROVED/REJECTED/FLAGGED) 필터 가능. "
        "'검증 대기 중인 리뷰 몇 개야?', '플래그된 리뷰 확인' 질문에 사용한다."
    ),
    example_questions=[
        "검증 대기(PENDING) 리뷰 목록",
        "플래그된 리뷰 확인해줘",
        "리뷰 검증 큐 현황",
    ],
    args_schema=_ReviewVerificationsListArgs,
    handler=_handle_review_verifications_list,
))


register_tool(ToolSpec(
    name="review_verification_detail",
    tier=0,
    required_roles=_REVIEW_VERIFICATION_ROLES,
    description=(
        "AI 리뷰 검증 레코드 단건 상세 조회. id(Long) 로 특정 검증 항목을 조회한다. "
        "'검증 ID 5 상세 보여줘', '리뷰 검증 결과 확인' 질문에 사용한다."
    ),
    example_questions=[
        "리뷰 검증 ID 5 상세",
        "검증 항목 7번 결과 보여줘",
        "특정 검증 레코드 내용 확인",
    ],
    args_schema=_ReviewVerificationDetailArgs,
    handler=_handle_review_verification_detail,
))


register_tool(ToolSpec(
    name="review_verifications_overview",
    tier=0,
    required_roles=_REVIEW_VERIFICATION_ROLES,
    description=(
        "AI 리뷰 검증 전체 현황 개요 조회. 상태별 건수·처리율 등 요약 지표를 반환. "
        "'리뷰 검증 현황', '자동 검증 처리율' 질문에 사용한다. 파라미터 없음."
    ),
    example_questions=[
        "리뷰 검증 전체 현황",
        "자동 검증 처리 비율 알려줘",
        "검증 대기/완료 건수 요약",
    ],
    args_schema=_NoArgs,
    handler=_handle_review_verifications_overview,
))
