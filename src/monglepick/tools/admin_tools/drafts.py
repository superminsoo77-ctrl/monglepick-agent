"""
관리자 AI 에이전트 — Draft Tool 10개 (단일 파일).

설계서: docs/관리자_AI에이전트_v3_재설계.md §1.2 Draft Tool / §4.2 Draft Tool 표 전체

핵심 원칙:
- **Backend 를 절대 호출하지 않는다.**
  handler 는 Pydantic 검증된 인자 + ctx 만으로 payload dict 를 조립해
  AdminApiResult(ok=True, ...) 를 반환한다.
- LLM 이 tool 호출 시 사용자 발화·이전 read 결과를 바탕으로 폼 필드를 직접 채운
  인자를 넣어준다. handler 는 그 인자를 받아 {target_path, draft_fields, action_label,
  summary, tool_name} 형태로 감싸는 역할만 한다.
- 실제 저장·수정·삭제는 관리자가 해당 관리 페이지의 저장 버튼을 직접 눌러야만 반영된다.

SSE: response_formatter 가 AdminApiResult.data 를 'form_prefill' 이벤트로 발행한다.
      Client 는 draft_fields 를 location.state.draft 로 해당 페이지에 전달한다.

Role 매트릭스 (§5):
  notice/faq/help    — SUPER_ADMIN / ADMIN / SUPPORT_ADMIN / MODERATOR
  banner             — SUPER_ADMIN / ADMIN
  quiz               — SUPER_ADMIN / ADMIN / AI_OPS_ADMIN
  chat_suggestion    — SUPER_ADMIN / ADMIN / AI_OPS_ADMIN
  term               — SUPER_ADMIN / ADMIN
  worldcup_candidate — SUPER_ADMIN / ADMIN
  reward_policy      — SUPER_ADMIN / ADMIN / FINANCE_ADMIN
  point_pack         — SUPER_ADMIN / ADMIN / FINANCE_ADMIN
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from monglepick.api.admin_backend_client import AdminApiResult
from monglepick.tools.admin_tools import ToolContext, ToolSpec, register_tool


# ============================================================
# Role 집합 상수 (§5 매트릭스 기반)
# ============================================================

# 공지/FAQ/도움말: 고객 대면 콘텐츠 — 운영/고객지원/모더레이터까지 허용
_SUPPORT_CONTENT_ROLES: set[str] = {
    "SUPER_ADMIN",
    "ADMIN",
    "SUPPORT_ADMIN",
    "MODERATOR",
}

# 배너/이벤트/약관: 서비스 전면에 노출되는 콘텐츠 — 최고 권한만 허용
_SITE_CONTENT_ROLES: set[str] = {
    "SUPER_ADMIN",
    "ADMIN",
}

# 퀴즈/채팅 추천 칩: AI 운영 도메인 — AI 운영 관리자까지 허용
_AI_OPS_ROLES: set[str] = {
    "SUPER_ADMIN",
    "ADMIN",
    "AI_OPS_ADMIN",
}

# 리워드 정책/포인트 팩: 재무 도메인 — 재무 관리자까지 허용
_FINANCE_ROLES: set[str] = {
    "SUPER_ADMIN",
    "ADMIN",
    "FINANCE_ADMIN",
}


# ============================================================
# Args Schemas (LLM bind 용 Pydantic 모델)
# ============================================================

class NoticeDraftArgs(BaseModel):
    """공지사항 초안 생성 인자."""

    title: str = Field(
        description="공지 제목 (예: '서비스 점검 안내', '신기능 업데이트').",
    )
    type: str = Field(
        default="NOTICE",
        description="공지 유형. 'NOTICE'(일반 공지) | 'NEWS'(뉴스) | 'EVENT'(이벤트).",
    )
    pinned: bool = Field(
        default=False,
        description="상단 고정 여부. 중요 공지는 True 로 설정.",
    )
    content: str = Field(
        default="",
        description="공지 본문 내용.",
    )
    startAt: Optional[str] = Field(
        default=None,
        description="게시 시작 일시 (ISO 8601, 예: '2026-05-01T00:00:00'). 없으면 즉시 게시.",
    )
    endAt: Optional[str] = Field(
        default=None,
        description="게시 종료 일시 (ISO 8601). 없으면 무기한.",
    )


class FaqDraftArgs(BaseModel):
    """FAQ 초안 생성 인자."""

    category: str = Field(
        description="FAQ 카테고리 (예: '결제', '회원', 'AI 추천', '서비스 이용').",
    )
    question: str = Field(
        description="FAQ 질문 문구 (예: '포인트 환불이 가능한가요?').",
    )
    answer: str = Field(
        description="FAQ 답변 내용.",
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="검색·분류용 태그 목록 (예: ['포인트', '환불']). 없으면 빈 리스트.",
    )


class HelpArticleDraftArgs(BaseModel):
    """도움말 아티클 초안 생성 인자."""

    title: str = Field(
        description="도움말 제목 (예: 'AI 추천 서비스 이용 방법').",
    )
    category: str = Field(
        description="도움말 카테고리 (예: 'AI 서비스', '결제', '계정').",
    )
    content: str = Field(
        description="도움말 본문. 마크다운 허용.",
    )


class BannerDraftArgs(BaseModel):
    """배너 초안 생성 인자."""

    title: str = Field(
        description="배너 제목 또는 문구.",
    )
    imageUrl: str = Field(
        default="",
        description="배너 이미지 URL. 확정 전이면 빈 문자열로 두어도 됨.",
    )
    link: str = Field(
        default="",
        description="배너 클릭 시 이동할 URL 또는 내부 경로.",
    )
    position: str = Field(
        default="HOME",
        description="배너 노출 위치. 'HOME'(홈 메인) | 'EVENT'(이벤트 페이지) | 기타 관리자 정의 슬롯.",
    )
    priority: int = Field(
        default=0,
        description="노출 우선순위. 숫자가 클수록 상위 노출 (0이 기본값).",
    )


class QuizDraftArgs(BaseModel):
    """영화 퀴즈 초안 생성 인자."""

    movieId: str = Field(
        description="퀴즈 대상 영화 ID (movie_id 문자열).",
    )
    question: str = Field(
        description="퀴즈 질문 문구 (예: '이 영화의 감독은?').",
    )
    choices: list[str] = Field(
        description="선택지 목록 (2개 이상). 예: ['봉준호', '박찬욱', '홍상수', '이창동'].",
        min_length=2,
    )
    answerIndex: int = Field(
        description="정답 선택지의 0-based 인덱스. choices[answerIndex] 가 정답.",
    )
    explanation: Optional[str] = Field(
        default=None,
        description="정답 해설 (선택). 없으면 미노출.",
    )


class ChatSuggestionDraftArgs(BaseModel):
    """채팅 추천 칩(빠른 질문) 초안 생성 인자."""

    surface: str = Field(
        description=(
            "노출 채널. 'user_chat'(사용자 채팅) | 'admin_assistant'(관리자 AI) | "
            "'faq_chatbot'(고객센터 챗봇)."
        ),
    )
    text: str = Field(
        description="추천 칩에 표시될 질문 또는 단문 텍스트.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="이 칩을 추천하는 이유 또는 의도 메모 (관리 참고용).",
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="분류·필터링용 태그 목록.",
    )


class TermDraftArgs(BaseModel):
    """약관 초안 생성 인자."""

    type: str = Field(
        description=(
            "약관 유형. 'SERVICE'(서비스 이용약관) | 'PRIVACY'(개인정보처리방침) | "
            "기타 관리자 정의 타입."
        ),
    )
    version: str = Field(
        description="약관 버전 (예: '2026-05-01', 'v2.3').",
    )
    content: str = Field(
        description="약관 전문 내용. 마크다운 허용.",
    )


class WorldcupCandidateDraftArgs(BaseModel):
    """이상형 월드컵 후보 초안 생성 인자."""

    movieId: str = Field(
        description="월드컵 후보로 추가할 영화 ID (movie_id 문자열).",
    )
    tier: Optional[str] = Field(
        default=None,
        description="후보 티어 분류 (예: 'S', 'A', 'B'). 운영 정책에 따라 선택 입력.",
    )


class RewardPolicyDraftArgs(BaseModel):
    """리워드 정책 초안 생성 인자."""

    code: str = Field(
        description=(
            "정책 코드 (예: 'REVIEW_WRITE', 'DAILY_LOGIN', 'FIRST_AI_USE'). "
            "영문 대문자 + 언더스코어."
        ),
    )
    pointAmount: int = Field(
        description="지급할 포인트 양 (양의 정수).",
        ge=1,
    )
    condition: str = Field(
        description=(
            "지급 조건 설명 (예: '리뷰 작성 시 1회', '매일 첫 로그인 시'). "
            "관리자 참고용 자유 텍스트."
        ),
    )


class PointPackDraftArgs(BaseModel):
    """포인트 팩 상품 초안 생성 인자."""

    packCode: str = Field(
        description="팩 고유 코드 (예: 'PACK_10P', 'PACK_50P'). 영문 대문자 + 언더스코어.",
    )
    points: int = Field(
        description="팩에 포함된 포인트 수 (양의 정수).",
        ge=1,
    )
    priceKrw: int = Field(
        description="판매 가격 (원화, 양의 정수). 예: 990, 4900.",
        ge=1,
    )


# ============================================================
# Handlers (공통 패턴: Pydantic 검증 인자 → payload dict 조립 → AdminApiResult 반환)
# Backend 호출 없음. latency_ms=0.
# ============================================================

async def _handle_notice_draft(
    ctx: ToolContext,
    title: str,
    type: str = "NOTICE",
    pinned: bool = False,
    content: str = "",
    startAt: str | None = None,
    endAt: str | None = None,
) -> AdminApiResult:
    """
    공지사항 초안 payload 를 조립한다.

    Backend 호출 없이 draft_fields dict 를 구성해 AdminApiResult 로 래핑 반환.
    None 값 필드는 payload 에 그대로 포함 — Client 가 undefined 로 처리한다.
    """
    draft_fields: dict = {
        "title": title,
        "type": type,
        "pinned": pinned,
        "content": content,
        "startAt": startAt,
        "endAt": endAt,
    }
    data = {
        "target_path": "/admin/support?tab=notice&modal=create",
        "draft_fields": draft_fields,
        "action_label": "공지사항 작성 화면 열기",
        "summary": f"공지 초안 '{title}'을 준비했어요. 검토 후 저장해주세요.",
        "tool_name": "notice_draft",
    }
    return AdminApiResult(
        ok=True,
        status_code=200,
        data=data,
        error="",
        latency_ms=0,
        row_count=None,
    )


async def _handle_faq_draft(
    ctx: ToolContext,
    category: str,
    question: str,
    answer: str,
    tags: list[str] | None = None,
) -> AdminApiResult:
    """
    FAQ 초안 payload 를 조립한다.

    category/question/answer 는 필수. tags 는 선택 (미입력 시 빈 리스트로 정규화).
    """
    draft_fields: dict = {
        "category": category,
        "question": question,
        "answer": answer,
        "tags": tags or [],
    }
    data = {
        "target_path": "/admin/support?tab=faq&modal=create",
        "draft_fields": draft_fields,
        "action_label": "FAQ 작성 화면 열기",
        "summary": f"FAQ 초안 '{question}'을 준비했어요. 검토 후 저장해주세요.",
        "tool_name": "faq_draft",
    }
    return AdminApiResult(
        ok=True,
        status_code=200,
        data=data,
        error="",
        latency_ms=0,
        row_count=None,
    )


async def _handle_help_article_draft(
    ctx: ToolContext,
    title: str,
    category: str,
    content: str,
) -> AdminApiResult:
    """
    도움말 아티클 초안 payload 를 조립한다.

    title/category/content 모두 필수 — 도움말은 구조화된 내용이 핵심.
    """
    draft_fields: dict = {
        "title": title,
        "category": category,
        "content": content,
    }
    data = {
        "target_path": "/admin/support?tab=help&modal=create",
        "draft_fields": draft_fields,
        "action_label": "도움말 작성 화면 열기",
        "summary": f"도움말 초안 '{title}'을 준비했어요. 검토 후 저장해주세요.",
        "tool_name": "help_article_draft",
    }
    return AdminApiResult(
        ok=True,
        status_code=200,
        data=data,
        error="",
        latency_ms=0,
        row_count=None,
    )


async def _handle_banner_draft(
    ctx: ToolContext,
    title: str,
    imageUrl: str = "",
    link: str = "",
    position: str = "HOME",
    priority: int = 0,
) -> AdminApiResult:
    """
    배너 초안 payload 를 조립한다.

    imageUrl/link 는 기획 확정 전 빈 문자열 허용. Client 가 배너 편집 폼에 세팅 후 직접 업로드.
    """
    draft_fields: dict = {
        "title": title,
        "imageUrl": imageUrl,
        "link": link,
        "position": position,
        "priority": priority,
    }
    data = {
        "target_path": "/admin/settings?tab=banners&modal=create",
        "draft_fields": draft_fields,
        "action_label": "배너 작성 화면 열기",
        "summary": f"배너 초안 '{title}'을 준비했어요. 이미지 업로드 후 저장해주세요.",
        "tool_name": "banner_draft",
    }
    return AdminApiResult(
        ok=True,
        status_code=200,
        data=data,
        error="",
        latency_ms=0,
        row_count=None,
    )


async def _handle_quiz_draft(
    ctx: ToolContext,
    movieId: str,
    question: str,
    choices: list[str],
    answerIndex: int,
    explanation: str | None = None,
) -> AdminApiResult:
    """
    영화 퀴즈 초안 payload 를 조립한다.

    choices 는 2개 이상 필수. answerIndex 는 0-based — choices[answerIndex] 가 정답.
    """
    draft_fields: dict = {
        "movieId": movieId,
        "question": question,
        "choices": choices,
        "answerIndex": answerIndex,
        "explanation": explanation,
    }
    data = {
        # 2026-04-27 정정: quiz CRUD 는 ContentEventsPage 의 quiz 탭이 담당
        # (`features/contentEvents/components/QuizManagementTab.jsx`).
        # AiOpsPage 에는 quiz 탭이 없어 기존 `/admin/ai?tab=quiz` 로 이동 시
        # 첫 탭(trigger) 으로 폴백되어 모달이 열리지 않던 결함 수정.
        "target_path": "/admin/content-events?tab=quiz&modal=create",
        "draft_fields": draft_fields,
        "action_label": "퀴즈 작성 화면 열기",
        "summary": f"퀴즈 초안 '{question}'을 준비했어요. 정답 확인 후 저장해주세요.",
        "tool_name": "quiz_draft",
    }
    return AdminApiResult(
        ok=True,
        status_code=200,
        data=data,
        error="",
        latency_ms=0,
        row_count=None,
    )


async def _handle_chat_suggestion_draft(
    ctx: ToolContext,
    surface: str,
    text: str,
    reason: str | None = None,
    tags: list[str] | None = None,
) -> AdminApiResult:
    """
    채팅 추천 칩 초안 payload 를 조립한다.

    surface 3채널: user_chat / admin_assistant / faq_chatbot.
    tags 는 선택 (미입력 시 빈 리스트로 정규화).
    """
    draft_fields: dict = {
        "surface": surface,
        "text": text,
        "reason": reason,
        "tags": tags or [],
    }
    data = {
        "target_path": "/admin/ai?tab=chat-suggestions&modal=create",
        "draft_fields": draft_fields,
        "action_label": "채팅 추천 칩 작성 화면 열기",
        "summary": f"채팅 추천 칩 초안 '{text}'을 준비했어요. 검토 후 저장해주세요.",
        "tool_name": "chat_suggestion_draft",
    }
    return AdminApiResult(
        ok=True,
        status_code=200,
        data=data,
        error="",
        latency_ms=0,
        row_count=None,
    )


async def _handle_term_draft(
    ctx: ToolContext,
    type: str,
    version: str,
    content: str,
) -> AdminApiResult:
    """
    약관 초안 payload 를 조립한다.

    type/version/content 모두 필수. 약관 전문은 마크다운 형식 권장.
    """
    draft_fields: dict = {
        "type": type,
        "version": version,
        "content": content,
    }
    data = {
        "target_path": "/admin/settings?tab=terms&modal=create",
        "draft_fields": draft_fields,
        "action_label": "약관 작성 화면 열기",
        "summary": f"약관({type}) 버전 '{version}' 초안을 준비했어요. 법적 검토 후 저장해주세요.",
        "tool_name": "term_draft",
    }
    return AdminApiResult(
        ok=True,
        status_code=200,
        data=data,
        error="",
        latency_ms=0,
        row_count=None,
    )


async def _handle_worldcup_candidate_draft(
    ctx: ToolContext,
    movieId: str,
    tier: str | None = None,
) -> AdminApiResult:
    """
    이상형 월드컵 후보 초안 payload 를 조립한다.

    movieId 필수. tier 는 운영 정책에 따라 선택 입력.
    """
    draft_fields: dict = {
        "movieId": movieId,
        "tier": tier,
    }
    data = {
        # 2026-04-27 정정: ContentEventsPage SUB_TABS 의 실제 key 가
        # `worldcup_candidate` (snake_case 풀네임). 기존 `tab=worldcup` 으로 이동 시
        # 첫 탭(roadmap_course) 으로 폴백되던 결함 수정.
        "target_path": "/admin/content-events?tab=worldcup_candidate&modal=create",
        "draft_fields": draft_fields,
        "action_label": "월드컵 후보 추가 화면 열기",
        "summary": f"이상형 월드컵 후보(영화 ID: {movieId}) 초안을 준비했어요. 검토 후 저장해주세요.",
        "tool_name": "worldcup_candidate_draft",
    }
    return AdminApiResult(
        ok=True,
        status_code=200,
        data=data,
        error="",
        latency_ms=0,
        row_count=None,
    )


async def _handle_reward_policy_draft(
    ctx: ToolContext,
    code: str,
    pointAmount: int,
    condition: str,
) -> AdminApiResult:
    """
    리워드 정책 초안 payload 를 조립한다.

    code 는 영문 대문자 + 언더스코어 관례. pointAmount 는 1 이상 정수.
    """
    draft_fields: dict = {
        "code": code,
        "pointAmount": pointAmount,
        "condition": condition,
    }
    data = {
        "target_path": "/admin/payment?tab=reward_policy&modal=create",
        "draft_fields": draft_fields,
        "action_label": "리워드 정책 작성 화면 열기",
        "summary": (
            f"리워드 정책 초안 '{code}' ({pointAmount}P)을 준비했어요. "
            "조건 검토 후 저장해주세요."
        ),
        "tool_name": "reward_policy_draft",
    }
    return AdminApiResult(
        ok=True,
        status_code=200,
        data=data,
        error="",
        latency_ms=0,
        row_count=None,
    )


async def _handle_point_pack_draft(
    ctx: ToolContext,
    packCode: str,
    points: int,
    priceKrw: int,
) -> AdminApiResult:
    """
    포인트 팩 상품 초안 payload 를 조립한다.

    packCode 는 영문 대문자 + 언더스코어 관례. 1P=10원 정책(v3.2) 기준으로 가격 확인 권장.
    """
    draft_fields: dict = {
        "packCode": packCode,
        "points": points,
        "priceKrw": priceKrw,
    }
    data = {
        "target_path": "/admin/payment?tab=point_pack&modal=create",
        "draft_fields": draft_fields,
        "action_label": "포인트 팩 작성 화면 열기",
        "summary": (
            f"포인트 팩 초안 '{packCode}' ({points}P / {priceKrw:,}원)을 준비했어요. "
            "가격 정책 확인 후 저장해주세요."
        ),
        "tool_name": "point_pack_draft",
    }
    return AdminApiResult(
        ok=True,
        status_code=200,
        data=data,
        error="",
        latency_ms=0,
        row_count=None,
    )


# ============================================================
# 공통 description 후미 문구 (설계서 §4.2 Draft Tool 기본 원칙)
# ============================================================
_DRAFT_SUFFIX = (
    " 폼을 채워 해당 관리 화면을 열 수 있도록 초안을 생성합니다. "
    "저장은 관리자가 직접 화면에서 실행합니다."
)


# ============================================================
# Registration (정확히 10회)
# ============================================================

register_tool(ToolSpec(
    name="notice_draft",
    tier=0,
    required_roles=_SUPPORT_CONTENT_ROLES,
    description=(
        "공지사항 초안을 생성합니다. 제목·유형(공지/뉴스/이벤트)·본문·상단 고정 여부·게시 기간을 "
        "사용자 발화에서 추출해 공지사항 작성 폼에 미리 채워줍니다." + _DRAFT_SUFFIX
    ),
    example_questions=[
        "공지 초안 써줘",
        "서비스 점검 공지 만들어줘",
        "이번 주 이벤트 공지 초안 잡아줘",
    ],
    args_schema=NoticeDraftArgs,
    handler=_handle_notice_draft,
))

register_tool(ToolSpec(
    name="faq_draft",
    tier=0,
    required_roles=_SUPPORT_CONTENT_ROLES,
    description=(
        "FAQ 항목 초안을 생성합니다. 카테고리·질문·답변·태그를 사용자 발화에서 추출해 "
        "FAQ 작성 폼에 미리 채워줍니다." + _DRAFT_SUFFIX
    ),
    example_questions=[
        "FAQ 하나 초안으로 만들어줘",
        "포인트 환불 FAQ 초안 써줘",
        "AI 추천 이용 방법 FAQ 만들어줘",
    ],
    args_schema=FaqDraftArgs,
    handler=_handle_faq_draft,
))

register_tool(ToolSpec(
    name="help_article_draft",
    tier=0,
    required_roles=_SUPPORT_CONTENT_ROLES,
    description=(
        "도움말 아티클 초안을 생성합니다. 제목·카테고리·본문을 사용자 발화에서 추출해 "
        "도움말 작성 폼에 미리 채워줍니다." + _DRAFT_SUFFIX
    ),
    example_questions=[
        "도움말 아티클 초안 만들어줘",
        "AI 추천 서비스 이용 방법 도움말 초안 써줘",
        "회원 탈퇴 절차 도움말 초안 잡아줘",
    ],
    args_schema=HelpArticleDraftArgs,
    handler=_handle_help_article_draft,
))

register_tool(ToolSpec(
    name="banner_draft",
    tier=0,
    required_roles=_SITE_CONTENT_ROLES,
    description=(
        "배너 초안을 생성합니다. 제목·이미지 URL·링크·노출 위치·우선순위를 사용자 발화에서 "
        "추출해 배너 작성 폼에 미리 채워줍니다." + _DRAFT_SUFFIX
    ),
    example_questions=[
        "배너 초안 만들어줘",
        "홈 메인 배너 초안 잡아줘",
        "이벤트 배너 내용 초안 써줘",
    ],
    args_schema=BannerDraftArgs,
    handler=_handle_banner_draft,
))

register_tool(ToolSpec(
    name="quiz_draft",
    tier=0,
    required_roles=_AI_OPS_ROLES,
    description=(
        "영화 퀴즈 초안을 생성합니다. 대상 영화 ID·질문·선택지·정답 인덱스·해설을 사용자 "
        "발화에서 추출해 퀴즈 작성 폼에 미리 채워줍니다." + _DRAFT_SUFFIX
    ),
    example_questions=[
        "퀴즈 초안 만들어줘",
        "기생충 감독 퀴즈 초안 써줘",
        "영화 퀴즈 하나 만들어줘",
    ],
    args_schema=QuizDraftArgs,
    handler=_handle_quiz_draft,
))

register_tool(ToolSpec(
    name="chat_suggestion_draft",
    tier=0,
    required_roles=_AI_OPS_ROLES,
    description=(
        "채팅 추천 칩(빠른 질문) 초안을 생성합니다. 노출 채널(사용자 채팅/관리자 AI/고객센터 챗봇)·"
        "텍스트·이유·태그를 사용자 발화에서 추출해 추천 칩 작성 폼에 미리 채워줍니다." + _DRAFT_SUFFIX
    ),
    example_questions=[
        "채팅 추천 칩 초안 만들어줘",
        "사용자 채팅용 빠른 질문 초안 써줘",
        "관리자 AI 어시스턴트 추천 질문 추가해줘",
    ],
    args_schema=ChatSuggestionDraftArgs,
    handler=_handle_chat_suggestion_draft,
))

register_tool(ToolSpec(
    name="term_draft",
    tier=0,
    required_roles=_SITE_CONTENT_ROLES,
    description=(
        "약관 초안을 생성합니다. 약관 유형(서비스/개인정보처리방침 등)·버전·전문을 사용자 "
        "발화에서 추출해 약관 작성 폼에 미리 채워줍니다." + _DRAFT_SUFFIX
    ),
    example_questions=[
        "약관 초안 만들어줘",
        "개인정보처리방침 개정 초안 써줘",
        "서비스 이용약관 v2 초안 잡아줘",
    ],
    args_schema=TermDraftArgs,
    handler=_handle_term_draft,
))

register_tool(ToolSpec(
    name="worldcup_candidate_draft",
    tier=0,
    required_roles=_SITE_CONTENT_ROLES,
    description=(
        "이상형 월드컵 후보 추가 초안을 생성합니다. 대상 영화 ID·티어를 사용자 발화에서 "
        "추출해 월드컵 후보 추가 폼에 미리 채워줍니다." + _DRAFT_SUFFIX
    ),
    example_questions=[
        "월드컵 후보 추가 초안 만들어줘",
        "이상형 월드컵에 영화 추가해줘",
        "월드컵 후보 초안 잡아줘",
    ],
    args_schema=WorldcupCandidateDraftArgs,
    handler=_handle_worldcup_candidate_draft,
))

register_tool(ToolSpec(
    name="reward_policy_draft",
    tier=0,
    required_roles=_FINANCE_ROLES,
    description=(
        "리워드 정책 초안을 생성합니다. 정책 코드·지급 포인트·지급 조건을 사용자 발화에서 "
        "추출해 리워드 정책 작성 폼에 미리 채워줍니다." + _DRAFT_SUFFIX
    ),
    example_questions=[
        "리워드 정책 초안 만들어줘",
        "리뷰 작성 리워드 정책 초안 써줘",
        "출석 포인트 정책 초안 잡아줘",
    ],
    args_schema=RewardPolicyDraftArgs,
    handler=_handle_reward_policy_draft,
))

register_tool(ToolSpec(
    name="point_pack_draft",
    tier=0,
    required_roles=_FINANCE_ROLES,
    description=(
        "포인트 팩 상품 초안을 생성합니다. 팩 코드·포인트 수량·가격(원화)을 사용자 발화에서 "
        "추출해 포인트 팩 작성 폼에 미리 채워줍니다." + _DRAFT_SUFFIX
    ),
    example_questions=[
        "포인트 팩 초안 만들어줘",
        "10포인트 팩 상품 초안 써줘",
        "신규 포인트 상품 초안 잡아줘",
    ],
    args_schema=PointPackDraftArgs,
    handler=_handle_point_pack_draft,
))
