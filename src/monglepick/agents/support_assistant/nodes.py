"""
support_assistant LangGraph 노드 (v3.3 — ES-first + Solar 경계분류 + vLLM 답변).

그래프 (3노드 유지):
    START → context_loader → support_agent → response_formatter → END

### v3.3 변경점
context_loader 가 ES-first 전략을 채택한다.
- ES 검색은 support_reply_chain 내부에서 수행되므로 노드 자체는 단순하게 유지.
- context_loader 는 Backend HTTP FAQ 조회를 **지연(lazy)** 방식으로 처리한다:
    - ES 주요 경로: Backend FAQ 호출 생략 가능 (ES 가 검색을 담당)
    - 그러나 Solar/vLLM fallback 경로에서 faqs 인수가 필요하므로 항상 조회 유지.
    - 단, Backend 장애 시 faqs=[] 로 계속 진행 — ES 가 있으면 정상 응답 가능.

### 노드 역할
- context_loader     : Backend FAQ 조회 + state 초기화
- support_agent      : generate_support_reply() 위임 (ES+Solar+vLLM 3단계)
- response_formatter : 최종 검증 + 빈 응답 방어

모든 노드:
- async def
- 반환값은 state 업데이트용 dict (LangGraph 규약)
- 에러 전파 금지 — 실패 시 graceful fallback 응답으로 떨어진다
"""

from __future__ import annotations

import structlog

from monglepick.agents.support_assistant.faq_client import fetch_faqs
from monglepick.agents.support_assistant.models import (
    FaqDoc,
    MatchedFaq,
    SupportAssistantState,
    SupportReply,
    ensure_reply,
)
from monglepick.chains.support_reply_chain import generate_support_reply

logger = structlog.get_logger(__name__)


# =============================================================================
# 1) context_loader — Backend FAQ 조회 (ES fallback 안전망용)
# =============================================================================


async def context_loader(state: SupportAssistantState) -> dict:
    """
    진입 노드. Backend HTTP 에서 FAQ 전체를 조회해 state.faqs 에 싣고 기본 필드를 초기화한다.

    ### ES-first 전략에서 faqs 의 역할 (v3.3)
    v3.3 에서는 ES Nori BM25 가 FAQ 검색의 주요 경로이므로 이론상 Backend FAQ 전체
    조회가 불필요하다. 그러나 다음 이유로 유지한다:

    1. **Solar/vLLM fallback**: ES 실패 시 _classify_and_match 가 faqs 인수로
       question 목록을 필요로 한다 (최후 안전망).
    2. **답변 생성 보조**: _generate_answer_from_faq 가 ES 후보에 없는 id 를
       Backend FAQ 에서 찾는 로직을 포함한다 (교차 참조).
    3. **matched_faq_ids 환각 방어**: valid_ids = candidate_ids | faq_ids
       — Backend FAQ 가 있으면 Solar 가 돌려준 id 를 더 넓게 허용 가능.

    Backend 장애 시 faqs=[] 로 계속 진행한다. ES 가 정상이면 faqs=[] 여도
    HIGH/MID 경로에서 정상 응답이 가능하다.
    """
    user_message = (state.get("user_message") or "").strip()
    logger.info(
        "support_context_loader_start",
        session_id=state.get("session_id", ""),
        user_id=state.get("user_id", "") or "(guest)",
        message_preview=user_message[:120],
    )

    try:
        faqs = await fetch_faqs()
    except Exception as exc:  # noqa: BLE001 — 에러 전파 금지
        logger.warning(
            "support_context_loader_fetch_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        faqs = []

    return {
        "faqs": faqs,
        # 이전 턴 잔재가 혼입되지 않도록 초기화
        "reply": None,
        "matched_faqs": [],
        "response_text": "",
        "needs_human_agent": False,
        "error": None,
    }


# =============================================================================
# 2) support_agent — Solar Pro 1회 호출로 통합 응답 생성
# =============================================================================


def _select_matched_faqs(
    faqs: list[FaqDoc], matched_ids: list[int]
) -> list[MatchedFaq]:
    """
    LLM/ES 가 돌려준 matched_faq_ids 를 실제 FAQ 메타와 매핑해 SSE/UI 용 축약 리스트로 변환.

    - ID 순서를 제시된 대로 유지 (중요 FAQ 가 먼저 노출되도록)
    - v3.3: state.faqs(Backend fetch 결과)에서 못 찾은 ID 도 **드롭하지 않고 id-only
      MatchedFaq 로 유지**. chain 내부에서 ES candidate 로 이미 검증된 ID 이므로
      Backend fetch 가 일시 실패해도 근거 FAQ 표시를 포기하지 않기 위함.
      (v3.2 까지는 못 찾은 ID 를 조용히 스킵 → kind 강등 → complaint 로 수렴하는 버그)
    """
    by_id: dict[int, FaqDoc] = {f.faq_id: f for f in faqs}
    out: list[MatchedFaq] = []
    for fid in matched_ids:
        try:
            fid_int = int(fid)
        except (TypeError, ValueError):
            continue
        faq = by_id.get(fid_int)
        if faq is None:
            # Backend fetch 결과엔 없지만 chain(ES) 이 유효하다고 전한 ID —
            # id 만 담은 축약 레코드로 유지해 강등 트리거를 피한다.
            out.append(
                MatchedFaq(
                    faq_id=fid_int,
                    category="",
                    question="",
                )
            )
            continue
        out.append(
            MatchedFaq(
                faq_id=faq.faq_id,
                category=faq.category,
                question=faq.question,
            )
        )
    return out


async def support_agent(state: SupportAssistantState) -> dict:
    """
    Solar Pro structured output 으로 SupportReply 를 받아 state 에 반영한다.

    실패/범위 밖/환각은 `generate_support_reply` 내부에서 graceful 처리되어
    항상 SupportReply 인스턴스가 돌아온다. 여기서는 matched_faq_ids 를 실제
    FAQ 와 교차 매핑해 허위 ID 를 걸러낸다.
    """
    user_message = (state.get("user_message") or "").strip()
    faqs: list[FaqDoc] = state.get("faqs") or []

    reply: SupportReply = await generate_support_reply(
        user_message=user_message,
        faqs=faqs,
    )

    matched = _select_matched_faqs(faqs, reply.matched_faq_ids)

    # 환각/진짜 매칭 실패 방어.
    # v3.3: chain 이 ES candidate 로 이미 ID 를 검증하므로 reply.matched_faq_ids 가
    # 비어있을 때만 강등한다 (v3.2 처럼 matched 축약 리스트 기준으로 강등하면
    # Backend fetch 실패 시에도 잘못 강등되는 버그 — 2026-04-24 hotfix).
    if reply.kind in ("faq", "partial") and not reply.matched_faq_ids:
        logger.info(
            "support_agent_empty_matches_for_faq_kind",
            original_kind=reply.kind,
            original_ids=reply.matched_faq_ids,
        )
        reply = reply.model_copy(
            update={
                "kind": "complaint",
                "matched_faq_ids": [],
                "needs_human": True,
            }
        )

    logger.info(
        "support_agent_done",
        kind=reply.kind,
        matched_count=len(matched),
        needs_human=reply.needs_human,
    )

    return {
        "reply": reply,
        "matched_faqs": matched,
        "response_text": reply.answer,
        "needs_human_agent": bool(reply.needs_human),
    }


# =============================================================================
# 3) response_formatter — 최종 검증 + 빈 응답 방어
# =============================================================================


async def response_formatter(state: SupportAssistantState) -> dict:
    """
    최종 본문/배너 플래그를 한 번 더 가드한다.

    support_agent 가 이미 기본값을 채우지만, 극단적인 케이스(state 직렬화
    중 reply 손실 등) 에 대비해 방어적으로 보정한다.
    """
    text = (state.get("response_text") or "").strip()
    needs_human = bool(state.get("needs_human_agent", False))

    if not text:
        text = (
            "지금은 답변을 드리기가 어려워요. '문의하기' 탭에서 1:1 티켓으로 "
            "남겨주시면 담당자가 확인해 드릴게요."
        )
        needs_human = True

    # 체크포인트 복원 방어 — reply 가 dict 로 보존된 경우도 정상 복원 가능.
    reply = ensure_reply(state.get("reply"))
    if reply is not None:
        kind = reply.kind
    else:
        kind = "unknown"

    logger.info(
        "support_response_formatter_done",
        kind=kind,
        needs_human=needs_human,
        text_length=len(text),
    )

    return {"response_text": text, "needs_human_agent": needs_human}
