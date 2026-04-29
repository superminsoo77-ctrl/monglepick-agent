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

from typing import Literal, Optional

from pydantic import BaseModel, Field

from monglepick.api.admin_backend_client import AdminApiResult
from monglepick.tools.admin_tools import ToolContext, ToolSpec, register_tool


# ============================================================
# 2026-04-28 (길 A v3 보강) — Draft Tool 공용 상수
# ============================================================
# 모든 Draft Args 가 공유하는 두 개의 운영 필드:
#   - mode: "create" | "update" — 생성 vs 수정 분기. LLM 이 사용자 발화에서 ID·번호·
#     "수정/보강/다시 쓰" 어휘를 보고 update 로 판정해 target_id 와 함께 채워야 한다.
#   - target_id: 수정 대상 PK (정수 또는 문자열). update 일 때 필수, create 일 때 None.
# 핸들러는 이 두 값을 보고 target_path 의 modal 을 create/edit 으로 분기한다.

# FAQ 카테고리는 Backend SupportCategory enum (6종) 과 1:1 동기화.
# 자유 텍스트 허용 시 "환불" 같은 사용자 발화가 그대로 들어가 Backend 400 → "잘못된 입력입니다".
FaqCategoryLiteral = Literal[
    "GENERAL",         # 일반 문의
    "ACCOUNT",         # 계정 / 회원
    "CHAT",            # AI 채팅
    "RECOMMENDATION",  # 영화 추천
    "COMMUNITY",       # 커뮤니티 게시판
    "PAYMENT",         # 결제 / 구독 / 포인트
]

# 공지사항 노출 방식 — Backend NoticeCreateRequest.displayType (Size 20) 와 매칭.
NoticeDisplayTypeLiteral = Literal["LIST_ONLY", "BANNER", "POPUP", "MODAL"]

# Draft 공통 mode 타입 — 모든 Args 가 이 값을 default="create" 로 받는다.
DraftModeLiteral = Literal["create", "update"]


def _resolve_target_path(
    base_path: str,
    mode: str,
    target_id: Optional[int | str],
    create_modal: str = "create",
    edit_modal: str = "edit",
) -> str:
    """
    mode + target_id 조합으로 Client 가 열 모달 경로를 만든다.

    - mode=="update" + target_id 가 truthy → `{base_path}&modal={edit_modal}&id={target_id}`
    - 그 외 → `{base_path}&modal={create_modal}`

    Client(SupportPage, ContentEventsPage 등) 는 modal=edit&id= 를 보고 기존 데이터를
    Backend 에서 조회해 폼에 prefill 한 뒤 draft_fields 로 덮어쓴다 (보강 워크플로우).
    """
    base = base_path.rstrip("&")
    if mode == "update" and target_id not in (None, "", 0):
        return f"{base}&modal={edit_modal}&id={target_id}"
    return f"{base}&modal={create_modal}"


def _build_draft_summary(
    mode: str,
    target_id: Optional[int | str],
    create_phrase: str,
    update_phrase: str,
) -> str:
    """
    mode 에 따라 자연스러운 한국어 요약 문장을 만든다.

    create_phrase: "공지 초안 'X' 을(를) 준비했어요. 검토 후 저장해주세요."
    update_phrase: "공지 #ID 'X' 을(를) 보강했어요. 검토 후 저장해주세요."
    """
    if mode == "update" and target_id not in (None, "", 0):
        return update_phrase
    return create_phrase


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
    """
    공지사항 초안 생성/수정 인자.

    Backend DTO: AdminSupportDto.NoticeCreateRequest / NoticeUpdateRequest 와 1:1 매칭.
    필수 필드: title, content (NotBlank). 그 외는 선택 — Client 가 기본값으로 채운다.

    수정(update) 흐름:
    - 사용자 발화에 공지 ID/번호 또는 "수정·보강·다시 써" 어휘가 있으면 mode="update" + target_id.
    - LLM 은 먼저 read_notices/get_notice_by_id 로 기존 본문을 읽고, 그 본문을 토대로
      content 필드에 **보강된 전체 본문** 을 채워야 한다 (절대 "내용을 보강합니다" 같은
      메타 문구만 넣지 말 것).
    """

    mode: DraftModeLiteral = Field(
        default="create",
        description=(
            "동작 모드. 'create'=새 공지 생성, 'update'=기존 공지 수정. "
            "사용자 발화에 공지 ID·번호·'수정/보강' 어휘가 있으면 'update' 로 채우고 "
            "target_id 도 함께 지정한다."
        ),
    )
    target_id: Optional[int] = Field(
        default=None,
        description=(
            "mode='update' 일 때 수정 대상 공지의 PK (notice_id, BIGINT). "
            "발화에서 'N번 공지', '공지 #N', '첫번째 공지(=목록 첫 항목)' 같은 어휘가 있으면 "
            "먼저 read tool 로 ID 를 확인한 뒤 채워야 한다. mode='create' 일 때는 None."
        ),
    )
    title: str = Field(
        description=(
            "공지 제목 (예: '서비스 점검 안내', '신기능 업데이트'). "
            "수정 모드에서도 반드시 채워야 함 (NotBlank). 기존 제목 유지가 필요하면 "
            "read 로 가져온 원본 제목을 그대로 넣는다."
        ),
    )
    content: str = Field(
        description=(
            "공지 본문 — **즉시 게시 가능한 수준의 구체적이고 풍부한 내용**. "
            "수정 모드에서 '내용 보강' 요청이면 read 로 가져온 원본을 토대로 단락을 추가하고 "
            "구체 일정·범위·문의처 등을 보충해 완성된 본문을 채운다. "
            "절대 '내용을 보강합니다' 같은 placeholder 문구만 넣지 말 것."
        ),
    )
    noticeType: str = Field(
        default="NOTICE",
        description="콘텐츠 카테고리. 'NOTICE'(일반) | 'UPDATE'(업데이트) | 'MAINTENANCE'(점검) | 'EVENT'(이벤트).",
    )
    displayType: NoticeDisplayTypeLiteral = Field(
        default="LIST_ONLY",
        description=(
            "노출 방식. 'LIST_ONLY'(목록만) | 'BANNER'(배너) | 'POPUP'(팝업) | 'MODAL'(모달). "
            "Backend 기본값은 LIST_ONLY."
        ),
    )
    isPinned: bool = Field(
        default=False,
        description="상단 고정 여부. 중요 공지는 True.",
    )
    sortOrder: Optional[int] = Field(
        default=None,
        description="정렬 순서 (선택). 미입력 시 Backend 가 부여.",
    )
    publishedAt: Optional[str] = Field(
        default=None,
        description="공개 시각 (ISO 8601, 예: '2026-05-01T00:00:00'). 없으면 즉시 공개.",
    )
    linkUrl: Optional[str] = Field(
        default=None,
        description="배너/팝업 클릭 시 이동 URL (선택, 최대 500자).",
    )
    imageUrl: Optional[str] = Field(
        default=None,
        description="배너/팝업 이미지 URL (선택, 최대 500자).",
    )
    startAt: Optional[str] = Field(
        default=None,
        description="앱 메인 노출 시작 일시 (ISO 8601). 없으면 즉시.",
    )
    endAt: Optional[str] = Field(
        default=None,
        description="앱 메인 노출 종료 일시 (ISO 8601). 없으면 무기한.",
    )
    priority: Optional[int] = Field(
        default=None,
        description="앱 메인 노출 우선순위 (숫자가 클수록 상위). 미입력 시 0.",
    )
    isActive: Optional[bool] = Field(
        default=None,
        description="앱 메인 노출 활성 토글. 미입력 시 Client 기본값 사용.",
    )


class FaqDraftArgs(BaseModel):
    """
    FAQ 초안 생성/수정 인자.

    Backend DTO: AdminSupportDto.FaqCreateRequest / FaqUpdateRequest 와 1:1 매칭.
    필수: category, question(<=500자), answer (모두 NotBlank).

    category 강제 enum:
        GENERAL · ACCOUNT · CHAT · RECOMMENDATION · COMMUNITY · PAYMENT
    이 외의 값을 LLM 이 채우면 Backend 400 → "잘못된 입력입니다" 에러가 난다.
    사용자 발화 예시 → 매핑:
      "환불·결제·포인트 관련 FAQ" → PAYMENT
      "로그인·비밀번호·탈퇴 FAQ"   → ACCOUNT
      "AI 채팅 사용법 FAQ"        → CHAT
      "추천 알고리즘 FAQ"         → RECOMMENDATION
      "게시판/댓글/신고 FAQ"      → COMMUNITY
      "그 외 일반 FAQ"            → GENERAL
    """

    mode: DraftModeLiteral = Field(
        default="create",
        description="'create'=신규, 'update'=기존 FAQ 수정. 발화에 FAQ ID·번호가 있으면 'update'.",
    )
    target_id: Optional[int] = Field(
        default=None,
        description="mode='update' 일 때 수정 대상 FAQ 의 PK (faq_id, BIGINT).",
    )
    category: FaqCategoryLiteral = Field(
        description=(
            "FAQ 카테고리 — 반드시 6종 enum 중 하나. "
            "GENERAL/ACCOUNT/CHAT/RECOMMENDATION/COMMUNITY/PAYMENT. "
            "한국어 발화는 위 매핑표에 따라 영문 enum 으로 변환해서 채울 것."
        ),
    )
    question: str = Field(
        max_length=500,
        description=(
            "FAQ 질문 문구 (최대 500자, NotBlank). "
            "예: '포인트 환불이 가능한가요?', '비밀번호 재설정은 어떻게 하나요?'."
        ),
    )
    answer: str = Field(
        description=(
            "FAQ 답변 — **사용자가 바로 따라할 수 있는 구체적이고 친절한 안내**. "
            "절차가 있으면 단계별로, 정책이 있으면 조건과 제한을 명시. "
            "수정 모드에서 '내용 보강' 요청이면 기존 답변을 그대로 두지 말고 보강된 전체 본문을 채운다. "
            "절대 '내용을 보강합니다' 같은 placeholder 금지."
        ),
    )
    keywords: Optional[str] = Field(
        default=None,
        description=(
            "ES 검색 키워드 힌트 (쉼표 구분 동의어 문자열). "
            "예: '환불,반환,취소,돈' / '비밀번호,패스워드,암호,재설정'. "
            "Backend 는 list 가 아니라 문자열을 받으므로 반드시 쉼표 구분 1줄 문자열로 채울 것."
        ),
    )
    sortOrder: Optional[int] = Field(
        default=None,
        description="표시 순서 (선택). 미입력 시 Backend 가 부여.",
    )


class HelpArticleDraftArgs(BaseModel):
    """
    도움말 아티클 초안 생성/수정 인자.

    Backend DTO: AdminSupportDto.HelpArticleCreateRequest / HelpArticleUpdateRequest.
    필수: category, title(<=200자), content (모두 NotBlank). category 는 FAQ 와 동일 enum.
    """

    mode: DraftModeLiteral = Field(
        default="create",
        description="'create'=신규, 'update'=기존 도움말 수정. 발화에 도움말 ID/번호가 있으면 'update'.",
    )
    target_id: Optional[int] = Field(
        default=None,
        description="mode='update' 일 때 수정 대상 도움말 PK (article_id, BIGINT).",
    )
    title: str = Field(
        max_length=200,
        description=(
            "도움말 제목 (최대 200자, NotBlank). 예: 'AI 추천 서비스 이용 방법'."
        ),
    )
    category: FaqCategoryLiteral = Field(
        description=(
            "도움말 카테고리 — FAQ 와 동일 enum 6종. "
            "GENERAL/ACCOUNT/CHAT/RECOMMENDATION/COMMUNITY/PAYMENT."
        ),
    )
    content: str = Field(
        description=(
            "도움말 본문 (마크다운 허용). **즉시 게시 가능한 수준의 풍부한 내용** 으로 채움. "
            "단계별 설명, 스크린샷 자리표시, 자주 묻는 질문 섹션 등을 마크다운으로 구조화. "
            "수정 모드의 보강 요청 시 기존 본문을 토대로 단락을 추가하고 누락된 절차/예외/문의처를 보충."
        ),
    )


class TicketReplyDraftArgs(BaseModel):
    """
    고객센터 티켓 답변 초안 생성 인자 — 길 A v3 보강 (2026-04-28 신설).

    Backend DTO: AdminSupportDto.TicketReplyRequest (필수: content NotBlank).
    티켓별 단건 답변. 일괄("전부 답변") 발화는 LLM 이 첫 건에 대해서만 draft 를 만들고
    "1건 prefill 했어요. 저장 후 다음 티켓을 처리해 주세요" 라고 안내하도록 프롬프트가 강제.

    target_path: /admin/support?tab=tickets&ticketId={ticket_id}&modal=reply&prefill=1
    Client TicketTab 이 modal=reply 를 보고 답변 모달을 띄운 뒤 draft.content 를 textarea 에 주입.
    """

    ticket_id: int = Field(
        description=(
            "답변할 티켓의 PK (ticket_id, BIGINT). 사용자 발화에서 'N번 티켓'·'#N' 으로 "
            "특정되거나, 직전 read_tickets/get_ticket 결과에서 식별된 ID 를 그대로 채운다."
        ),
    )
    content: str = Field(
        description=(
            "티켓 답변 본문 — **사용자가 곧바로 받아볼 수 있는 친절한 한국어 응답**. "
            "사용자 질문을 한 줄로 요약 → 원인/안내 → 추가 도움 안내 (가능 시 도움말/FAQ 링크) → "
            "마무리 인사 순서로 구체적이고 풍부하게 작성한다. 절대 '내용을 보강합니다' 같은 "
            "placeholder 금지. 정책 모르면 LLM 이 추측하지 말고 짧게 '확인 후 답변드릴게요' 톤."
        ),
    )


class BannerDraftArgs(BaseModel):
    """배너 초안 생성/수정 인자."""

    mode: DraftModeLiteral = Field(
        default="create",
        description="'create'=신규, 'update'=기존 배너 수정.",
    )
    target_id: Optional[int] = Field(
        default=None,
        description="mode='update' 일 때 수정 대상 배너 PK.",
    )
    title: str = Field(
        description="배너 제목 또는 문구. 클릭 유도가 분명한 짧은 카피로 채운다.",
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
    """영화 퀴즈 초안 생성/수정 인자."""

    mode: DraftModeLiteral = Field(
        default="create",
        description="'create'=신규, 'update'=기존 퀴즈 수정.",
    )
    target_id: Optional[int] = Field(
        default=None,
        description="mode='update' 일 때 수정 대상 퀴즈 PK.",
    )
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
    """채팅 추천 칩(빠른 질문) 초안 생성/수정 인자."""

    mode: DraftModeLiteral = Field(
        default="create",
        description="'create'=신규, 'update'=기존 추천 칩 수정.",
    )
    target_id: Optional[int] = Field(
        default=None,
        description="mode='update' 일 때 수정 대상 추천 칩 PK.",
    )
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
    """약관 초안 생성/수정 인자."""

    mode: DraftModeLiteral = Field(
        default="create",
        description="'create'=신규 버전 등록, 'update'=기존 약관 버전 수정.",
    )
    target_id: Optional[int] = Field(
        default=None,
        description="mode='update' 일 때 수정 대상 약관 PK.",
    )
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
    """이상형 월드컵 후보 초안 생성/수정 인자."""

    mode: DraftModeLiteral = Field(
        default="create",
        description="'create'=후보 추가, 'update'=기존 후보 수정.",
    )
    target_id: Optional[int] = Field(
        default=None,
        description="mode='update' 일 때 수정 대상 후보 PK.",
    )
    movieId: str = Field(
        description="월드컵 후보로 추가할 영화 ID (movie_id 문자열).",
    )
    tier: Optional[str] = Field(
        default=None,
        description="후보 티어 분류 (예: 'S', 'A', 'B'). 운영 정책에 따라 선택 입력.",
    )


class RewardPolicyDraftArgs(BaseModel):
    """리워드 정책 초안 생성/수정 인자."""

    mode: DraftModeLiteral = Field(
        default="create",
        description="'create'=신규 정책 등록, 'update'=기존 정책 수정.",
    )
    target_id: Optional[int] = Field(
        default=None,
        description="mode='update' 일 때 수정 대상 리워드 정책 PK.",
    )
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
    """포인트 팩 상품 초안 생성/수정 인자."""

    mode: DraftModeLiteral = Field(
        default="create",
        description="'create'=신규 팩 등록, 'update'=기존 팩 수정.",
    )
    target_id: Optional[int] = Field(
        default=None,
        description="mode='update' 일 때 수정 대상 포인트 팩 PK.",
    )
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
    content: str,
    mode: str = "create",
    target_id: int | None = None,
    noticeType: str = "NOTICE",
    displayType: str = "LIST_ONLY",
    isPinned: bool = False,
    sortOrder: int | None = None,
    publishedAt: str | None = None,
    linkUrl: str | None = None,
    imageUrl: str | None = None,
    startAt: str | None = None,
    endAt: str | None = None,
    priority: int | None = None,
    isActive: bool | None = None,
) -> AdminApiResult:
    """
    공지사항 초안 payload 조립 — create/update 분기.

    update 시 target_path 에 modal=edit&id={target_id} 가 붙어 Client SupportPage 가
    edit 모달을 띄우고 Backend GET /api/v1/admin/notices/{id} 로 기존 데이터를 prefetch 한 뒤
    draft_fields 로 덮어쓴다.
    """
    # Backend NoticeCreateRequest / NoticeUpdateRequest 모든 필드를 draft_fields 에 담는다.
    # None 값은 Client 가 기본값으로 채우므로 그대로 전달.
    draft_fields: dict = {
        "title": title,
        "content": content,
        "noticeType": noticeType,
        "displayType": displayType,
        "isPinned": isPinned,
        "sortOrder": sortOrder,
        "publishedAt": publishedAt,
        "linkUrl": linkUrl,
        "imageUrl": imageUrl,
        "startAt": startAt,
        "endAt": endAt,
        "priority": priority,
        "isActive": isActive,
    }
    target_path = _resolve_target_path(
        base_path="/admin/support?tab=notice",
        mode=mode,
        target_id=target_id,
    )
    summary = _build_draft_summary(
        mode=mode,
        target_id=target_id,
        create_phrase=f"공지 초안 '{title}'을(를) 준비했어요. 검토 후 저장해주세요.",
        update_phrase=(
            f"공지 #{target_id} '{title}' 보강안을 준비했어요. "
            "수정 모달에서 검토 후 저장해주세요."
        ),
    )
    action_label = (
        "공지사항 수정 화면 열기" if mode == "update" and target_id else "공지사항 작성 화면 열기"
    )
    data = {
        "target_path": target_path,
        "draft_fields": draft_fields,
        "action_label": action_label,
        "summary": summary,
        "tool_name": "notice_draft",
        "mode": mode,
        "target_id": target_id,
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
    mode: str = "create",
    target_id: int | None = None,
    keywords: str | None = None,
    sortOrder: int | None = None,
) -> AdminApiResult:
    """
    FAQ 초안 payload 조립 — create/update 분기.

    Backend FaqCreateRequest/FaqUpdateRequest 와 1:1. category 는 SupportCategory enum 6종.
    keywords 는 list 가 아니라 쉼표 구분 문자열 (Backend 가 그대로 받음).
    """
    draft_fields: dict = {
        "category": category,
        "question": question,
        "answer": answer,
        "keywords": keywords,
        "sortOrder": sortOrder,
    }
    target_path = _resolve_target_path(
        base_path="/admin/support?tab=faq",
        mode=mode,
        target_id=target_id,
    )
    summary = _build_draft_summary(
        mode=mode,
        target_id=target_id,
        create_phrase=f"FAQ 초안 '{question}'을(를) 준비했어요. 검토 후 저장해주세요.",
        update_phrase=(
            f"FAQ #{target_id} '{question}' 보강안을 준비했어요. "
            "수정 모달에서 검토 후 저장해주세요."
        ),
    )
    action_label = (
        "FAQ 수정 화면 열기" if mode == "update" and target_id else "FAQ 작성 화면 열기"
    )
    data = {
        "target_path": target_path,
        "draft_fields": draft_fields,
        "action_label": action_label,
        "summary": summary,
        "tool_name": "faq_draft",
        "mode": mode,
        "target_id": target_id,
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
    mode: str = "create",
    target_id: int | None = None,
) -> AdminApiResult:
    """도움말 아티클 초안 payload — create/update 분기."""
    draft_fields: dict = {
        "title": title,
        "category": category,
        "content": content,
    }
    target_path = _resolve_target_path(
        base_path="/admin/support?tab=help",
        mode=mode,
        target_id=target_id,
    )
    summary = _build_draft_summary(
        mode=mode,
        target_id=target_id,
        create_phrase=f"도움말 초안 '{title}'을(를) 준비했어요. 검토 후 저장해주세요.",
        update_phrase=(
            f"도움말 #{target_id} '{title}' 보강안을 준비했어요. "
            "수정 모달에서 검토 후 저장해주세요."
        ),
    )
    action_label = (
        "도움말 수정 화면 열기" if mode == "update" and target_id else "도움말 작성 화면 열기"
    )
    data = {
        "target_path": target_path,
        "draft_fields": draft_fields,
        "action_label": action_label,
        "summary": summary,
        "tool_name": "help_article_draft",
        "mode": mode,
        "target_id": target_id,
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
# 2026-04-28 신설 — Ticket Reply Draft handler (11번째 Draft tool)
# ============================================================

async def _handle_ticket_reply_draft(
    ctx: ToolContext,
    ticket_id: int,
    content: str,
) -> AdminApiResult:
    """
    티켓 답변 초안 payload — Backend TicketReplyRequest(content) 와 매칭.

    target_path: /admin/support?tab=tickets&ticketId={ticket_id}&modal=reply&prefill=1
    Client TicketTab 이 modal=reply 를 보고 답변 모달을 띄운 뒤 draft.content 를 textarea
    에 주입한다. 관리자가 [전송] 버튼을 누르면 Backend POST /api/v1/admin/tickets/{id}/reply.
    """
    draft_fields: dict = {
        "content": content,
    }
    target_path = (
        f"/admin/support?tab=tickets&ticketId={ticket_id}&modal=reply&prefill=1"
    )
    data = {
        "target_path": target_path,
        "draft_fields": draft_fields,
        "action_label": f"티켓 #{ticket_id} 답변 모달 열기",
        "summary": (
            f"티켓 #{ticket_id} 답변 초안을 준비했어요. 모달에서 내용 확인 후 [전송] 을 눌러주세요. "
            "여러 티켓을 답변해야 하면 한 건씩 차례대로 처리해 주세요."
        ),
        "tool_name": "ticket_reply_draft",
        "mode": "create",
        "target_id": ticket_id,
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
    mode: str = "create",
    target_id: int | None = None,
) -> AdminApiResult:
    """배너 초안 payload — create/update 분기."""
    draft_fields: dict = {
        "title": title,
        "imageUrl": imageUrl,
        "link": link,
        "position": position,
        "priority": priority,
    }
    target_path = _resolve_target_path(
        base_path="/admin/settings?tab=banners",
        mode=mode,
        target_id=target_id,
    )
    summary = _build_draft_summary(
        mode=mode,
        target_id=target_id,
        create_phrase=f"배너 초안 '{title}'을(를) 준비했어요. 이미지 업로드 후 저장해주세요.",
        update_phrase=(
            f"배너 #{target_id} '{title}' 보강안을 준비했어요. "
            "수정 모달에서 검토 후 저장해주세요."
        ),
    )
    action_label = (
        "배너 수정 화면 열기" if mode == "update" and target_id else "배너 작성 화면 열기"
    )
    data = {
        "target_path": target_path,
        "draft_fields": draft_fields,
        "action_label": action_label,
        "summary": summary,
        "tool_name": "banner_draft",
        "mode": mode,
        "target_id": target_id,
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
    mode: str = "create",
    target_id: int | None = None,
) -> AdminApiResult:
    """영화 퀴즈 초안 payload — create/update 분기."""
    draft_fields: dict = {
        "movieId": movieId,
        "question": question,
        "choices": choices,
        "answerIndex": answerIndex,
        "explanation": explanation,
    }
    # 2026-04-27 정정: quiz CRUD 는 ContentEventsPage 의 quiz 탭이 담당.
    target_path = _resolve_target_path(
        base_path="/admin/content-events?tab=quiz",
        mode=mode,
        target_id=target_id,
    )
    summary = _build_draft_summary(
        mode=mode,
        target_id=target_id,
        create_phrase=f"퀴즈 초안 '{question}'을(를) 준비했어요. 정답 확인 후 저장해주세요.",
        update_phrase=f"퀴즈 #{target_id} 보강안을 준비했어요. 수정 모달에서 검토 후 저장해주세요.",
    )
    action_label = (
        "퀴즈 수정 화면 열기" if mode == "update" and target_id else "퀴즈 작성 화면 열기"
    )
    data = {
        "target_path": target_path,
        "draft_fields": draft_fields,
        "action_label": action_label,
        "summary": summary,
        "tool_name": "quiz_draft",
        "mode": mode,
        "target_id": target_id,
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
    mode: str = "create",
    target_id: int | None = None,
) -> AdminApiResult:
    """채팅 추천 칩 초안 payload — create/update 분기."""
    draft_fields: dict = {
        "surface": surface,
        "text": text,
        "reason": reason,
        "tags": tags or [],
    }
    target_path = _resolve_target_path(
        base_path="/admin/ai?tab=chat-suggestions",
        mode=mode,
        target_id=target_id,
    )
    summary = _build_draft_summary(
        mode=mode,
        target_id=target_id,
        create_phrase=f"채팅 추천 칩 초안 '{text}'을(를) 준비했어요. 검토 후 저장해주세요.",
        update_phrase=f"채팅 추천 칩 #{target_id} 보강안을 준비했어요. 수정 모달에서 검토 후 저장해주세요.",
    )
    action_label = (
        "채팅 추천 칩 수정 화면 열기" if mode == "update" and target_id
        else "채팅 추천 칩 작성 화면 열기"
    )
    data = {
        "target_path": target_path,
        "draft_fields": draft_fields,
        "action_label": action_label,
        "summary": summary,
        "tool_name": "chat_suggestion_draft",
        "mode": mode,
        "target_id": target_id,
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
    mode: str = "create",
    target_id: int | None = None,
) -> AdminApiResult:
    """약관 초안 payload — create/update 분기."""
    draft_fields: dict = {
        "type": type,
        "version": version,
        "content": content,
    }
    target_path = _resolve_target_path(
        base_path="/admin/settings?tab=terms",
        mode=mode,
        target_id=target_id,
    )
    summary = _build_draft_summary(
        mode=mode,
        target_id=target_id,
        create_phrase=f"약관({type}) 버전 '{version}' 초안을 준비했어요. 법적 검토 후 저장해주세요.",
        update_phrase=(
            f"약관 #{target_id} ({type}) 버전 '{version}' 보강안을 준비했어요. "
            "수정 모달에서 검토 후 저장해주세요."
        ),
    )
    action_label = (
        "약관 수정 화면 열기" if mode == "update" and target_id else "약관 작성 화면 열기"
    )
    data = {
        "target_path": target_path,
        "draft_fields": draft_fields,
        "action_label": action_label,
        "summary": summary,
        "tool_name": "term_draft",
        "mode": mode,
        "target_id": target_id,
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
    mode: str = "create",
    target_id: int | None = None,
) -> AdminApiResult:
    """이상형 월드컵 후보 초안 payload — create/update 분기."""
    draft_fields: dict = {
        "movieId": movieId,
        "tier": tier,
    }
    # 2026-04-27 정정: ContentEventsPage SUB_TABS 의 실제 key=worldcup_candidate.
    target_path = _resolve_target_path(
        base_path="/admin/content-events?tab=worldcup_candidate",
        mode=mode,
        target_id=target_id,
    )
    summary = _build_draft_summary(
        mode=mode,
        target_id=target_id,
        create_phrase=(
            f"이상형 월드컵 후보(영화 ID: {movieId}) 초안을 준비했어요. 검토 후 저장해주세요."
        ),
        update_phrase=f"월드컵 후보 #{target_id} 보강안을 준비했어요. 수정 모달에서 검토 후 저장해주세요.",
    )
    action_label = (
        "월드컵 후보 수정 화면 열기" if mode == "update" and target_id
        else "월드컵 후보 추가 화면 열기"
    )
    data = {
        "target_path": target_path,
        "draft_fields": draft_fields,
        "action_label": action_label,
        "summary": summary,
        "tool_name": "worldcup_candidate_draft",
        "mode": mode,
        "target_id": target_id,
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
    mode: str = "create",
    target_id: int | None = None,
) -> AdminApiResult:
    """리워드 정책 초안 payload — create/update 분기."""
    draft_fields: dict = {
        "code": code,
        "pointAmount": pointAmount,
        "condition": condition,
    }
    target_path = _resolve_target_path(
        base_path="/admin/payment?tab=reward_policy",
        mode=mode,
        target_id=target_id,
    )
    summary = _build_draft_summary(
        mode=mode,
        target_id=target_id,
        create_phrase=(
            f"리워드 정책 초안 '{code}' ({pointAmount}P)을(를) 준비했어요. "
            "조건 검토 후 저장해주세요."
        ),
        update_phrase=(
            f"리워드 정책 #{target_id} '{code}' 보강안을 준비했어요. "
            "수정 모달에서 검토 후 저장해주세요."
        ),
    )
    action_label = (
        "리워드 정책 수정 화면 열기" if mode == "update" and target_id
        else "리워드 정책 작성 화면 열기"
    )
    data = {
        "target_path": target_path,
        "draft_fields": draft_fields,
        "action_label": action_label,
        "summary": summary,
        "tool_name": "reward_policy_draft",
        "mode": mode,
        "target_id": target_id,
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
    mode: str = "create",
    target_id: int | None = None,
) -> AdminApiResult:
    """포인트 팩 상품 초안 payload — create/update 분기."""
    draft_fields: dict = {
        "packCode": packCode,
        "points": points,
        "priceKrw": priceKrw,
    }
    target_path = _resolve_target_path(
        base_path="/admin/payment?tab=point_pack",
        mode=mode,
        target_id=target_id,
    )
    summary = _build_draft_summary(
        mode=mode,
        target_id=target_id,
        create_phrase=(
            f"포인트 팩 초안 '{packCode}' ({points}P / {priceKrw:,}원)을(를) 준비했어요. "
            "가격 정책 확인 후 저장해주세요."
        ),
        update_phrase=(
            f"포인트 팩 #{target_id} '{packCode}' 보강안을 준비했어요. "
            "수정 모달에서 검토 후 저장해주세요."
        ),
    )
    action_label = (
        "포인트 팩 수정 화면 열기" if mode == "update" and target_id
        else "포인트 팩 작성 화면 열기"
    )
    data = {
        "target_path": target_path,
        "draft_fields": draft_fields,
        "action_label": action_label,
        "summary": summary,
        "tool_name": "point_pack_draft",
        "mode": mode,
        "target_id": target_id,
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
    "저장은 관리자가 직접 화면에서 실행합니다. "
    "**중요**: 본문/답변 같은 내용 필드는 placeholder 가 아니라 즉시 게시 가능한 수준으로 "
    "구체적이고 풍부하게 채워야 합니다. "
    "기존 항목 수정(보강) 요청이면 mode='update' + target_id 를 함께 채우세요. "
    "필요하면 먼저 read 도구로 원본을 가져온 뒤 그 내용을 토대로 보강안을 작성합니다."
)


# ============================================================
# Registration (정확히 10회)
# ============================================================

register_tool(ToolSpec(
    name="notice_draft",
    tier=0,
    required_roles=_SUPPORT_CONTENT_ROLES,
    description=(
        "공지사항 초안 생성 또는 수정용 폼 prefill. 제목·노출 방식·본문·상단 고정·게시 기간 등을 "
        "사용자 발화에서 추출해 공지 작성/수정 폼에 미리 채워줍니다. "
        "발화에 'N번 공지', '첫번째 공지 수정', '내용 보강' 같은 어휘가 있으면 mode='update' + "
        "target_id 를 채워 수정 모달로 보냅니다 (read 로 원본 본문을 먼저 가져온 뒤 보강)."
        + _DRAFT_SUFFIX
    ),
    example_questions=[
        "공지 초안 써줘",
        "서비스 점검 공지 만들어줘",
        "이번 주 이벤트 공지 초안 잡아줘",
        "1번 공지 내용 보강해서 수정해줘",   # update 모드 예시
        "최근 공지 내용 다시 써줘",          # update 모드 예시
    ],
    args_schema=NoticeDraftArgs,
    handler=_handle_notice_draft,
))

register_tool(ToolSpec(
    name="faq_draft",
    tier=0,
    required_roles=_SUPPORT_CONTENT_ROLES,
    description=(
        "FAQ 항목 초안 생성 또는 수정용 폼 prefill. category 는 반드시 "
        "GENERAL/ACCOUNT/CHAT/RECOMMENDATION/COMMUNITY/PAYMENT 6종 enum 중 하나로 채워야 합니다. "
        "한국어 발화는 의미상 가장 가까운 enum 으로 변환 (환불/결제 → PAYMENT, 비밀번호/탈퇴 → ACCOUNT). "
        "수정 의도면 mode='update' + target_id."
        + _DRAFT_SUFFIX
    ),
    example_questions=[
        "FAQ 하나 초안으로 만들어줘",
        "포인트 환불 FAQ 초안 써줘",
        "AI 추천 이용 방법 FAQ 만들어줘",
        "FAQ 3번 답변 다시 써줘",            # update 모드 예시
        "환불 FAQ 새로 작성해줘",            # category=PAYMENT 예시
    ],
    args_schema=FaqDraftArgs,
    handler=_handle_faq_draft,
))

register_tool(ToolSpec(
    name="help_article_draft",
    tier=0,
    required_roles=_SUPPORT_CONTENT_ROLES,
    description=(
        "도움말 아티클 초안 생성 또는 수정용 폼 prefill. category 는 FAQ 와 동일한 6종 enum. "
        "본문은 마크다운으로 단계별·구체적으로 작성. 수정 의도면 mode='update' + target_id."
        + _DRAFT_SUFFIX
    ),
    example_questions=[
        "도움말 아티클 초안 만들어줘",
        "AI 추천 서비스 이용 방법 도움말 초안 써줘",
        "회원 탈퇴 절차 도움말 초안 잡아줘",
        "5번 도움말 내용 보강해서 수정해줘",  # update 모드 예시
    ],
    args_schema=HelpArticleDraftArgs,
    handler=_handle_help_article_draft,
))

# 2026-04-28 신설 — 티켓 답변 Draft (11번째)
register_tool(ToolSpec(
    name="ticket_reply_draft",
    tier=0,
    required_roles=_SUPPORT_CONTENT_ROLES,
    description=(
        "고객 문의 티켓에 대한 관리자 답변 초안을 생성해 답변 모달을 엽니다. "
        "ticket_id 는 사용자 발화에서 'N번 티켓' 으로 특정되거나 직전 read 결과에서 식별된 ID. "
        "content 는 사용자가 곧바로 받아볼 수 있는 친절하고 구체적인 한국어 응답으로 작성합니다. "
        "여러 티켓을 답변해야 한다면 한 번에 한 건씩 차례대로 prefill 하고, 사용자에게 "
        "'1건 prefill 했어요. 저장 후 다음 티켓을 처리해 주세요' 라고 안내합니다."
        + _DRAFT_SUFFIX
    ),
    example_questions=[
        "5번 티켓에 답변 초안 써줘",
        "최근 OPEN 티켓 답변 초안 만들어줘",
        "환불 문의 티켓 답변 작성해줘",
        "이 티켓 답변 다시 써줘",
    ],
    args_schema=TicketReplyDraftArgs,
    handler=_handle_ticket_reply_draft,
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
