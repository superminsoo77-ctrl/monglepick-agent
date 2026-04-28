"""
고객센터 AI 챗봇 (support_assistant) Pydantic/TypedDict 모델.

### v3 설계 (2026-04-23)
임베딩 기반 의미 검색(v2) 은 과잉 설계였다. RDB 하나에만 들어있는 FAQ 수십 건을
굳이 Qdrant/임베딩 API 로 돌릴 이유가 없어, v3 는 매 요청마다 Backend RDB 에서
FAQ 전체를 바로 조회하고 Solar Pro LLM **1회 호출** 로 다음을 한 번에 해결한다.
    - 사용자 발화의 응답 모드 판단 (faq/partial/complaint/out_of_scope/smalltalk)
    - 참조한 FAQ ID 목록 수집
    - 몽글이 톤 답변 본문 생성
    - needs_human 플래그 결정

구조화 출력은 `SupportReply`.
State 는 TypedDict 로 LangGraph 규약 유지.

### v4 확장 (2026-04-28)
v4 9노드 골격에 맞춰 SupportAssistantState 에 필드 추가.
기존 v3 필드는 절대 변경 없이 v4 신규 필드만 append.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §3 (그래프) + §10 (SSE)
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

# =============================================================================
# 응답 모드 — 사용자 발화를 LLM 이 이 5종 중 하나로 분류한다
# =============================================================================
# v3 원본 (5종):
# - faq          : FAQ 하나(또는 여러 건)에 명확한 답이 있음. 친근한 톤으로 FAQ 인용.
# - partial      : 관련 주제의 FAQ 는 있지만 완전한 정답은 아님. 참고 안내 + 1:1 유도.
#   [DEPRECATED v4] partial 은 v4 에서 faq 로 통합. 하위 호환성 유지를 위해 Literal 에 잔류.
# - complaint    : 불만/버그 신고. 공감 + 1:1 유도(구체적 증상 확인 필요).
# - out_of_scope : 고객센터 범위 밖 (영화 취향 상담/잡지식 등). 적절한 창구로 안내.
#   [DEPRECATED v4] out_of_scope 는 v4 에서 redirect 로 통합. 하위 호환성 유지를 위해 잔류.
# - smalltalk    : 인사/감사/간단한 안부. 1~2문장 짧게 응대.
#
# v4 추가 (2종):
# - personal_data: 본인 데이터 조회 필요. Phase 1 은 정책 RAG 폴백. Phase 2 에서 개인화 tool.
# - policy       : 운영 정책 질문. Qdrant 정책 RAG 검색으로 답변.
# - redirect     : 고객센터 밖 채널로 연결 (메인 AI 채팅 등). out_of_scope 대체.
SupportReplyKind = Literal[
    "faq",
    "partial",           # DEPRECATED v4 — faq 로 통합. 하위 호환 유지.
    "complaint",
    "out_of_scope",      # DEPRECATED v4 — redirect 로 통합. 하위 호환 유지.
    "smalltalk",
    "personal_data",     # v4 신규
    "policy",            # v4 신규
    "redirect",          # v4 신규 (out_of_scope 대체)
]


class SupportPlan(BaseModel):
    """
    v3.2 Step 1 출력 — vLLM EXAONE 1.2B 가 FAQ question 목록만 보고 선정하는 계획.

    답변 본문 생성은 Step 2 에서 분리된다. 1.2B 의 max_model_len=2048 제약 때문에
    FAQ 전체 answer 를 맥락으로 실을 수 없어, 먼저 kind 분류와 FAQ id 선정만 수행한다.
    """

    kind: SupportReplyKind = Field(
        default="smalltalk",
        description=(
            "사용자 발화 분류. faq/partial/complaint/out_of_scope/smalltalk 중 하나."
        ),
    )
    matched_faq_ids: list[int] = Field(
        default_factory=list,
        description=(
            "답변 근거로 참조할 FAQ id 목록 (0~3개). kind=faq/partial 에서만 채워지며 "
            "그 외에는 빈 배열. Step 2 에서 이 id 들의 answer 본문만 로드해 프롬프트에 실음."
        ),
    )


class SupportReply(BaseModel):
    """
    Step 2 통합 결과 — 최종 사용자 노출 응답.

    - `kind` 가 partial/complaint/out_of_scope 면 `needs_human=True` 가 자연스럽다.
      단, 환각 방지를 위해 kind 와 `matched_faq_ids` 정합성은 프롬프트에서 강제
      하고(예: smalltalk 에서는 matched_faq_ids=[]) 코드에서도 방어적으로 체크한다.
    """

    kind: SupportReplyKind = Field(
        default="smalltalk",
        description=(
            "사용자 발화에 대한 응답 모드. "
            "faq=FAQ 직답, partial=부분 매칭, complaint=불만/버그, "
            "out_of_scope=고객센터 범위 밖, smalltalk=인사."
        ),
    )
    matched_faq_ids: list[int] = Field(
        default_factory=list,
        description=(
            "답변 근거로 참고한 FAQ ID 목록. kind=faq/partial 에서만 채워지며, "
            "complaint/out_of_scope/smalltalk 에서는 빈 배열을 유지한다."
        ),
    )
    answer: str = Field(
        default="",
        description="사용자에게 보일 최종 답변 텍스트 (몽글이 톤, 존댓말, 3~4문장).",
    )
    needs_human: bool = Field(
        default=False,
        description=(
            "상담원 연결 배너 노출 여부. partial/complaint 는 기본 True, "
            "out_of_scope 는 '다른 탭 안내' 이므로 False, smalltalk/faq 는 False."
        ),
    )


# =============================================================================
# FAQ DTO — Backend GET /api/v1/support/faqs 응답 한 건
# =============================================================================


class FaqDoc(BaseModel):
    """Backend 에서 가져온 단일 FAQ. 매 요청마다 조회되므로 메모리 캐시 아님."""

    faq_id: int
    category: str
    question: str
    answer: str
    sort_order: int | None = None


# =============================================================================
# SSE matched_faq 이벤트 페이로드용 축약 DTO
# =============================================================================


class MatchedFaq(BaseModel):
    """
    SSE matched_faq 이벤트에 실려 나가는 근거 FAQ 요약.

    LLM 이 돌려준 matched_faq_ids 를 Backend FAQ 목록에서 찾아 이 형태로 직렬화한다.
    Client(SupportChatbotWidget/ChatbotTab) 는 이 payload 를 받아 '관련 FAQ' 카드로 노출.
    """

    faq_id: int
    category: str
    question: str


# =============================================================================
# State (LangGraph TypedDict)
# =============================================================================


class MessageTurn(TypedDict, total=False):
    """세션 히스토리 한 턴 (후속 확장 예약 — MVP 미사용)."""

    role: Literal["user", "assistant"]
    content: str


def ensure_reply(value: Any) -> "SupportReply | None":
    """
    LangGraph checkpointer 가 Pydantic 을 dict 로 바꿔 복원하는 케이스 대비 방어.

    v2 의 admin_assistant.ensure_intent 와 동일한 패턴.
    """
    if value is None:
        return None
    if isinstance(value, SupportReply):
        return value
    if isinstance(value, dict):
        try:
            return SupportReply.model_validate(value)
        except Exception:  # noqa: BLE001
            return None
    return None


class SupportAssistantState(TypedDict, total=False):
    """
    support_assistant LangGraph State.

    ### v3 필드 (변경 금지)
    - user_id       : JWT 에서 추출한 사용자 ID (비로그인이면 "")
    - session_id    : 세션 ID (자동 생성)
    - user_message  : 현재 턴 사용자 발화
    - history       : 이전 대화 이력 (멀티턴 — MVP 미사용)
    - faqs          : context_loader 가 Backend 에서 가져온 FAQ 전체 목록
    - reply         : Solar Pro 가 돌려준 SupportReply 구조화 응답 (v3 chain 결과)
    - matched_faqs  : reply.matched_faq_ids → FAQ 메타 매핑 축약 리스트 (SSE 발행용)
    - response_text : 최종 본문 (response_formatter 가 한 번 더 검증)
    - needs_human_agent: 최종 상담원 배너 노출 여부
    - error         : 노드 에러 메시지 (에러 전파 금지 원칙상 fallback 후 기록)

    ### v4 신규 필드 (2026-04-28)
    - is_guest        : user_id 가 빈 문자열이면 True. context_loader 에서 결정.
    - intent          : support_intent_chain 의 SupportIntent 분류 결과.
                        (kind ∈ faq/personal_data/policy/redirect/smalltalk/complaint)
    - pending_tool_call: tool_selector 가 선택한 실행 예정 tool.
                        {tool_name: str, args: dict}
    - tool_call_history: 실행 완료된 tool 호출 이력 (observation 노드가 append).
                         Phase 2 ReAct 루프에서 multi-hop 시 사용.
    - tool_results_cache: ref_id(tool_name+call_index) → 실행 결과 dict.
                          {"ok": bool, "data": {...} | "error": str}
    - hop_count       : 현재 ReAct hop 카운터. Phase 1 = 항상 1.
                        Phase 2 에서 MAX_HOPS 초과 방어에 사용.
    - rag_chunks      : search_policy 결과 직렬화 리스트 (narrator 가 인용).
                        [{"doc_id", "section", "headings", "policy_topic", "text", "score"}, ...]
    - navigation      : redirect 의도 시 Client 에 전달할 내비게이션 페이로드.
                        {target_path: str, label: str, candidates: list[str]}
    """

    # ── v3 필드 (절대 변경 금지) ──────────────────────────────────────────────
    user_id: str
    session_id: str
    user_message: str
    history: list[MessageTurn]

    # context_loader 단계
    faqs: list[FaqDoc]

    # support_agent 단계 (v3 chain 결과)
    reply: SupportReply | None
    matched_faqs: list[MatchedFaq]

    # 출력
    response_text: str
    needs_human_agent: bool

    error: str | None

    # ── v4 신규 필드 ─────────────────────────────────────────────────────────
    # context_loader 가 결정 — not user_id → is_guest = True
    is_guest: bool

    # intent_classifier 가 채움 — SupportIntent(kind/confidence/reason)
    # 순환 import 방지: 런타임 시 chains.support_intent_chain 에서 가져온 SupportIntent.
    # TypedDict 는 런타임 타입 검사를 하지 않으므로 Any 로 선언하고 노드에서 캐스팅.
    intent: Any  # SupportIntent | None

    # tool_selector 가 채움 — {"tool_name": str, "args": dict}
    pending_tool_call: dict | None

    # observation 노드가 append — Phase 1 은 단일 hop 이므로 최대 1건
    tool_call_history: list[dict]

    # tool_executor 결과 누적 — ref_id → {"ok": bool, "data": {...}} or {"ok": False, "error": str}
    tool_results_cache: dict  # dict[str, dict]

    # Phase 1 = 항상 1 (tool_executor 직후 observation 이 1로 설정)
    # Phase 2 = MAX_HOPS=3 초과 시 narrator 로 강제 분기
    hop_count: int

    # lookup_policy 결과 직렬화 — narrator 가 컨텍스트로 인용
    rag_chunks: list[dict]

    # redirect 의도 시 navigation SSE 이벤트 페이로드
    # {"target_path": "/chat", "label": "AI 채팅으로 이동", "candidates": [...]}
    navigation: dict | None
