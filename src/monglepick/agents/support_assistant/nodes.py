"""
support_assistant LangGraph 노드 (v4 Phase 1 — 9노드 골격).

### v4 변경점 (2026-04-28)
기존 v3 3노드(context_loader / support_agent / response_formatter) 를 유지하면서
6개 신규 노드를 추가해 9노드 구조로 확장한다.

    v3 유지 노드 (v3 코드/동작 완전 보존):
        - support_agent       : v3 generate_support_reply 체인 그대로 사용.
                                v4 그래프에서 intent.kind 가 v3 호환 kind 일 때만 진입.

    v4 갱신 노드:
        - context_loader      : is_guest 결정 + v4 State 필드 초기화 추가
        - response_formatter  : complaint 고정 템플릿, redirect navigation 페이로드 처리 추가

    v4 신규 노드 (7개):
        - intent_classifier   : classify_support_intent 호출 → state.intent 채움
        - smalltalk_responder : vLLM 1.2B / Solar fallback 으로 소소한 인사 응대
        - tool_selector       : intent.kind 매핑 → pending_tool_call 채움
        - tool_executor       : SUPPORT_TOOL_REGISTRY / lookup_faq(ES) 실행
        - observation         : hop_count++, tool_call_history append (Phase 1 단일 hop stub)
        - narrator            : Solar Pro 로 tool_results_cache 를 인용해 최종 답변 생성
        - smart_fallback      : tool_selector 실패 시 안내 메시지

설계서: docs/고객센터_AI에이전트_v4_재설계.md §3 (그래프) §5 (tool) §6 (RAG)

### 에러 처리 원칙
모든 노드는 async def, try/except, 실패 시 graceful fallback 반환.
에러 전파 절대 금지.
"""

from __future__ import annotations

import structlog

from monglepick.agents.support_assistant.faq_client import fetch_faqs
from monglepick.agents.support_assistant.faq_search import search_faq_candidates
from monglepick.agents.support_assistant.models import (
    FaqDoc,
    MatchedFaq,
    SupportAssistantState,
    SupportReply,
    ensure_reply,
)
from monglepick.chains.support_reply_chain import generate_support_reply
from monglepick.prompts.support_assistant import (
    SUPPORT_COMPLAINT_TEMPLATE,
    SUPPORT_SMALLTALK_HUMAN_PROMPT,
    SUPPORT_SMALLTALK_SYSTEM_PROMPT,
)
from monglepick.tools.support_tools import SUPPORT_TOOL_REGISTRY, ToolContext

logger = structlog.get_logger(__name__)


# =============================================================================
# 1) context_loader — v4 갱신 (is_guest 결정 + v4 State 초기화 추가)
# =============================================================================


async def context_loader(state: SupportAssistantState) -> dict:
    """
    진입 노드. Backend HTTP 에서 FAQ 전체를 조회해 state.faqs 에 싣고 기본 필드를 초기화한다.

    ### v4 갱신 (2026-04-28)
    - is_guest = not bool(user_id) 결정. 빈 문자열 / None 이면 게스트.
    - v4 신규 State 필드 초기화:
        intent / pending_tool_call / tool_call_history / tool_results_cache /
        hop_count / rag_chunks / navigation

    ### v3 동작 보존
    - Backend FAQ 전체 조회 (ES-first fallback 안전망 + matched_faq 교차 참조용)
    - Backend 장애 시 faqs=[] 로 계속 진행 (에러 전파 금지)
    - reply/matched_faqs/response_text/needs_human_agent/error 초기화
    """
    user_message = (state.get("user_message") or "").strip()
    user_id = state.get("user_id") or ""
    is_guest = not bool(user_id)

    logger.info(
        "support_context_loader_start",
        session_id=state.get("session_id", ""),
        user_id=user_id or "(guest)",
        is_guest=is_guest,
        message_preview=user_message[:120],
    )

    # Backend FAQ 전체 조회 (ES fallback 안전망, kind 강등 방어 교차 참조용)
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
        "is_guest": is_guest,
        # ── v3 기존 필드 초기화 ──
        "reply": None,
        "matched_faqs": [],
        "response_text": "",
        "needs_human_agent": False,
        "error": None,
        # ── v4 신규 필드 초기화 ──
        "intent": None,
        "pending_tool_call": None,
        "tool_call_history": [],
        "tool_results_cache": {},
        "hop_count": 0,
        "rag_chunks": [],
        "navigation": None,
    }


# =============================================================================
# 2) support_agent — v3 원본 보존 (v4 라우터가 v3 호환 경로에서만 진입)
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
    [v3 원본 보존] Solar Pro structured output 으로 SupportReply 를 받아 state 에 반영한다.

    v4 그래프에서 이 노드는 진입하지 않는다. v3 그래프 (build_support_assistant_graph_v3)
    또는 직접 import 하는 기존 테스트 경로에서만 호출된다.

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
# 3) response_formatter — v4 갱신 (complaint 고정 템플릿 + redirect 처리 추가)
# =============================================================================


async def response_formatter(state: SupportAssistantState) -> dict:
    """
    최종 본문/배너 플래그를 한 번 더 가드한다.

    ### v4 갱신 (2026-04-28)
    - complaint 의도: SUPPORT_COMPLAINT_TEMPLATE 고정 문자열 + needs_human=True
      (v4 에서 complaint 는 narrator 를 거치지 않고 이 노드가 직접 처리)
    - redirect 의도: 고정 안내 메시지 + needs_human=False
      (navigation 페이로드는 narrator 또는 이 노드에서 이미 state 에 채워짐)
    - v3 호환 경로 (reply 존재): 기존 동작 완전 보존

    ### 에러 처리
    빈 본문이면 최후 fallback 메시지 + needs_human=True (v3 동작 유지).
    """
    intent = state.get("intent")
    intent_kind = getattr(intent, "kind", None) if intent is not None else None

    # v4 경로: complaint → 고정 템플릿 직접 발행
    if intent_kind == "complaint":
        logger.info(
            "support_response_formatter_complaint",
            intent_kind=intent_kind,
        )
        return {
            "response_text": SUPPORT_COMPLAINT_TEMPLATE,
            "needs_human_agent": True,
        }

    # v4 경로: redirect → 고정 안내 + navigation 페이로드 보존
    if intent_kind == "redirect":
        text = (state.get("response_text") or "").strip()
        if not text:
            # narrator 가 response_text 를 채워야 하지만 만일의 경우 방어
            text = (
                "영화 추천은 메인 AI 채팅 탭이 더 잘 도와드려요. "
                "상단 'AI 채팅' 메뉴에서 편하게 물어봐 주세요!"
            )
        logger.info(
            "support_response_formatter_redirect",
            intent_kind=intent_kind,
            has_navigation=bool(state.get("navigation")),
        )
        return {
            "response_text": text,
            "needs_human_agent": False,
        }

    # v3 호환 경로 + v4 faq/policy/personal_data/smalltalk 경로
    # support_agent 또는 narrator 가 채운 response_text 를 한 번 더 검증한다.
    text = (state.get("response_text") or "").strip()
    needs_human = bool(state.get("needs_human_agent", False))

    if not text:
        # 극단적 케이스(state 직렬화 중 reply 손실 등) 대비 최후 fallback
        text = (
            "지금은 답변을 드리기가 어려워요. '문의하기' 탭에서 1:1 티켓으로 "
            "남겨주시면 담당자가 확인해 드릴게요."
        )
        needs_human = True

    # 체크포인트 복원 방어 — reply 가 dict 로 보존된 경우도 정상 복원 가능.
    reply = ensure_reply(state.get("reply"))
    if reply is not None:
        kind = reply.kind
    elif intent_kind:
        kind = intent_kind
    else:
        kind = "unknown"

    logger.info(
        "support_response_formatter_done",
        kind=kind,
        needs_human=needs_human,
        text_length=len(text),
    )

    return {"response_text": text, "needs_human_agent": needs_human}


# =============================================================================
# 4) intent_classifier — v4 신규
# =============================================================================


async def intent_classifier(state: SupportAssistantState) -> dict:
    """
    [v4 신규] classify_support_intent() 로 SupportIntent 를 분류해 state 에 채운다.

    순환 import 방지를 위해 함수 내부에서 lazily import 한다.
    (chains.support_intent_chain → llm.factory → ... 의존성 체인이
    모듈 레벨 import 에서 순환 가능성 있음)

    실패 시 faq intent 로 폴백 (에러 전파 금지).
    """
    user_message = (state.get("user_message") or "").strip()
    is_guest = bool(state.get("is_guest", False))
    session_id = state.get("session_id", "")

    try:
        # lazy import — 순환 의존성 방지
        from monglepick.chains.support_intent_chain import classify_support_intent

        intent = await classify_support_intent(
            user_message=user_message,
            is_guest=is_guest,
            request_id=session_id,
        )
        logger.info(
            "support_intent_classifier_done",
            kind=intent.kind,
            confidence=round(intent.confidence, 2),
            is_guest=is_guest,
        )
        return {"intent": intent}
    except Exception as exc:  # noqa: BLE001 — 에러 전파 금지
        logger.warning(
            "support_intent_classifier_failed_faq_fallback",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        # 분류 실패 시 faq 폴백 — ES 검색이라도 시도하도록
        from monglepick.chains.support_intent_chain import SupportIntent
        return {
            "intent": SupportIntent(
                kind="faq",
                confidence=0.0,
                reason=f"intent_classifier_error:{type(exc).__name__}",
            )
        }


# =============================================================================
# 5) smalltalk_responder — v4 신규
# =============================================================================


async def smalltalk_responder(state: SupportAssistantState) -> dict:
    """
    [v4 신규] 인사/안부 발화에 짧은 몽글이 톤 응대를 생성한다.

    우선순위:
    1. vLLM EXAONE 1.2B (hybrid 모드 + VLLM_ENABLED 시)
    2. Solar API (hybrid/api_only 모드)
    3. Ollama (local_only 모드)

    실패 시 고정 인사 문자열 fallback (에러 전파 금지).
    """
    user_message = (state.get("user_message") or "").strip()

    _FALLBACK_SMALLTALK = (
        "안녕하세요! 몽글 고객센터 챗봇이에요. 궁금한 점 있으시면 편하게 말씀해 주세요."
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from monglepick.llm.factory import get_conversation_llm

        llm = get_conversation_llm()
        messages = [
            SystemMessage(content=SUPPORT_SMALLTALK_SYSTEM_PROMPT),
            HumanMessage(
                content=SUPPORT_SMALLTALK_HUMAN_PROMPT.format(
                    user_message=user_message
                )
            ),
        ]
        response = await llm.ainvoke(messages)
        text = (response.content or "").strip()

        if not text:
            text = _FALLBACK_SMALLTALK

        logger.info(
            "support_smalltalk_responder_done",
            text_length=len(text),
        )
        return {
            "response_text": text,
            "needs_human_agent": False,
        }

    except Exception as exc:  # noqa: BLE001 — 에러 전파 금지
        logger.warning(
            "support_smalltalk_responder_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return {
            "response_text": _FALLBACK_SMALLTALK,
            "needs_human_agent": False,
        }


# =============================================================================
# 6) tool_selector — v4 신규 (Phase 1 단순 매핑)
# =============================================================================

# intent.kind → tool_name 매핑 테이블.
# Phase 1: personal_data 는 lookup_policy 로 폴백 (설계서 §3 Phase 1 임시 처리)
# Phase 2: personal_data 전용 tool (lookup_my_*) 추가 예정.
_INTENT_TO_TOOL: dict[str, str] = {
    "faq": "lookup_faq",            # ES Nori BM25 (faq_search.search_faq_candidates)
    "policy": "lookup_policy",       # Qdrant 정책 RAG (support_policy_rag_chain)
    "personal_data": "lookup_policy",  # Phase 1 임시 폴백 — Phase 2 에서 personal_data tool 교체
}


async def tool_selector(state: SupportAssistantState) -> dict:
    """
    [v4 신규] intent.kind 에 따라 실행할 tool 을 선택하고 pending_tool_call 에 채운다.

    Phase 1 단순 매핑:
    - faq           → lookup_faq (ES BM25, faq_search 재사용)
    - policy        → lookup_policy (Qdrant RAG)
    - personal_data → lookup_policy (Phase 1 임시 폴백, Phase 2 에서 교체)

    실패 시 smart_fallback 경로로 전환 (pending_tool_call = None, error 기록).
    """
    intent = state.get("intent")
    intent_kind = getattr(intent, "kind", "faq") if intent is not None else "faq"
    user_message = (state.get("user_message") or "").strip()

    tool_name = _INTENT_TO_TOOL.get(intent_kind)
    if tool_name is None:
        # 매핑 없는 kind (redirect/smalltalk/complaint 는 tool_selector 에 도달하지 않음)
        logger.warning(
            "tool_selector_no_mapping",
            intent_kind=intent_kind,
        )
        return {"pending_tool_call": None, "error": f"no_tool_for:{intent_kind}"}

    # tool args 구성
    if tool_name == "lookup_faq":
        # ES 검색은 사용자 발화를 그대로 쿼리로 사용
        args: dict = {"query": user_message}
    elif tool_name == "lookup_policy":
        # RAG 검색은 발화 + personal_data 임시 폴백 힌트
        args = {"query": user_message, "topic": None}
        if intent_kind == "personal_data":
            # personal_data Phase 1 폴백: 사용자 문맥 힌트 추가
            args["_phase1_fallback_note"] = "personal_data_intent_phase1_policy_fallback"
    else:
        args = {"query": user_message}

    pending = {"tool_name": tool_name, "args": args}
    logger.info(
        "tool_selector_done",
        intent_kind=intent_kind,
        tool_name=tool_name,
    )
    return {"pending_tool_call": pending}


# =============================================================================
# 7) tool_executor — v4 신규
# =============================================================================


async def tool_executor(state: SupportAssistantState) -> dict:
    """
    [v4 신규] pending_tool_call 에 지정된 tool 을 실행하고 결과를 캐시에 저장한다.

    tool 분기:
    - "lookup_faq"    : faq_search.search_faq_candidates (ES Nori BM25 재사용)
                       결과를 matched_faqs + tool_results_cache 모두에 저장.
    - "lookup_policy" : SUPPORT_TOOL_REGISTRY["lookup_policy"].handler 호출
                       결과를 rag_chunks + tool_results_cache 모두에 저장.

    게스트 + requires_login=True tool 호출 시:
    - tool_results_cache 에 {"ok": False, "error": "login_required"} 저장
    - narrator 가 "로그인이 필요한 기능이에요" 안내 + 정책 RAG 보조 처리

    실패 시 tool_results_cache 에 {"ok": False, "error": ...} 저장.
    에러 전파 금지.
    """
    pending: dict | None = state.get("pending_tool_call")
    if not pending:
        logger.warning("tool_executor_no_pending_call")
        return {"tool_results_cache": {}}

    tool_name: str = pending.get("tool_name", "")
    args: dict = {
        k: v
        for k, v in pending.get("args", {}).items()
        if not k.startswith("_")  # _phase1_fallback_note 같은 내부 메타 필드 제거
    }

    is_guest = bool(state.get("is_guest", False))
    session_id = state.get("session_id", "")
    user_id = state.get("user_id") or ""

    # 현재 캐시 가져오기 (기존 결과 유지)
    existing_cache: dict = dict(state.get("tool_results_cache") or {})
    ref_id = f"{tool_name}_0"  # Phase 1 단일 hop — 인덱스 항상 0

    # ── lookup_faq 경로 (ES BM25 — faq_search 직접 호출) ──
    if tool_name == "lookup_faq":
        try:
            candidates = await search_faq_candidates(
                user_message=args.get("query", ""),
                top_k=5,
            )
            # FaqCandidate 리스트 → dict 직렬화 (tool_results_cache 저장용)
            items = [
                {
                    "faq_id": c.faq_id,
                    "category": c.category,
                    "question": c.question,
                    "answer": c.answer,
                    "keywords": c.keywords,
                    "score": c.score,
                }
                for c in candidates
            ]
            existing_cache[ref_id] = {"ok": True, "data": {"faqs": items}}

            # matched_faqs 도 동시 갱신 — SSE matched_faq 이벤트를 위해 필요
            faqs_in_state: list[FaqDoc] = state.get("faqs") or []
            by_id = {f.faq_id: f for f in faqs_in_state}
            matched: list[MatchedFaq] = []
            for item in items:
                fid = item["faq_id"]
                faq = by_id.get(fid)
                if faq:
                    matched.append(
                        MatchedFaq(
                            faq_id=faq.faq_id,
                            category=faq.category,
                            question=faq.question,
                        )
                    )
                else:
                    # Backend fetch 미스 — id-only 보존 (v3.3 방어 패턴 유지)
                    matched.append(
                        MatchedFaq(faq_id=fid, category="", question="")
                    )

            logger.info(
                "tool_executor_lookup_faq_done",
                hit_count=len(items),
                top_score=round(items[0]["score"], 2) if items else 0.0,
            )
            return {
                "tool_results_cache": existing_cache,
                "matched_faqs": matched,
            }

        except Exception as exc:  # noqa: BLE001 — 에러 전파 금지
            logger.warning(
                "tool_executor_lookup_faq_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            existing_cache[ref_id] = {"ok": False, "error": str(exc)}
            return {"tool_results_cache": existing_cache}

    # ── SUPPORT_TOOL_REGISTRY 경로 (lookup_policy 등) ──
    spec = SUPPORT_TOOL_REGISTRY.get(tool_name)
    if spec is None:
        logger.error("tool_executor_unknown_tool", tool_name=tool_name)
        existing_cache[ref_id] = {"ok": False, "error": f"unknown_tool:{tool_name}"}
        return {"tool_results_cache": existing_cache}

    # 게스트 로그인 필요 tool 거부
    if spec.requires_login and is_guest:
        logger.info(
            "tool_executor_login_required",
            tool_name=tool_name,
            is_guest=is_guest,
        )
        existing_cache[ref_id] = {"ok": False, "error": "login_required"}
        return {"tool_results_cache": existing_cache}

    try:
        ctx = ToolContext(
            user_id=user_id,
            is_guest=is_guest,
            session_id=session_id,
            request_id=session_id,
        )
        result = await spec.handler(ctx, **args)
        existing_cache[ref_id] = result

        # lookup_policy 결과 → rag_chunks 로 노출 (SSE policy_chunk 이벤트 소재)
        rag_chunks: list[dict] = []
        if tool_name == "lookup_policy" and result.get("ok"):
            chunks_raw = result.get("data", {}).get("chunks", [])
            rag_chunks = chunks_raw if isinstance(chunks_raw, list) else []

        logger.info(
            "tool_executor_registry_done",
            tool_name=tool_name,
            ok=result.get("ok"),
            rag_count=len(rag_chunks),
        )
        update = {"tool_results_cache": existing_cache}
        if rag_chunks:
            update["rag_chunks"] = rag_chunks
        return update

    except Exception as exc:  # noqa: BLE001 — 에러 전파 금지
        logger.error(
            "tool_executor_registry_failed",
            tool_name=tool_name,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        existing_cache[ref_id] = {"ok": False, "error": str(exc)}
        return {"tool_results_cache": existing_cache}


# =============================================================================
# 8) observation — v4 신규 (Phase 1 단일 hop stub)
# =============================================================================


async def observation(state: SupportAssistantState) -> dict:
    """
    [v4 신규] tool 실행 결과를 관찰하고 hop 카운터를 증가시킨다.

    ### Phase 1 동작 (단일 hop stub)
    hop_count 를 1 로 설정하고 tool_call_history 에 실행 기록을 추가한다.
    Phase 1 에서는 항상 즉시 narrator 로 진행 (루프 진입 X).

    ### Phase 2 예정
    hop_count >= MAX_HOPS (3) 이면 narrator 로 강제 분기.
    tool_results_cache 품질이 낮으면 다른 tool 선택 → 재실행.
    """
    pending: dict | None = state.get("pending_tool_call")
    tool_name = pending.get("tool_name", "") if pending else ""
    cache: dict = state.get("tool_results_cache") or {}
    ref_id = f"{tool_name}_0"
    result_summary = cache.get(ref_id, {})

    history: list[dict] = list(state.get("tool_call_history") or [])
    history.append(
        {
            "hop": 1,
            "tool_name": tool_name,
            "ok": result_summary.get("ok", False),
            "error": result_summary.get("error"),
        }
    )

    logger.info(
        "support_observation_done",
        tool_name=tool_name,
        hop_count=1,
        ok=result_summary.get("ok", False),
    )

    return {
        "hop_count": 1,
        "tool_call_history": history,
    }


# =============================================================================
# 9) narrator — v4 신규
# =============================================================================

# redirect 의도 고정 메시지 (LLM 생성 없이 Python 템플릿)
_REDIRECT_MESSAGE = (
    "영화 추천은 메인 AI 채팅 탭이 더 잘 도와드려요. "
    "상단 'AI 채팅' 메뉴에서 편하게 물어봐 주세요!"
)

# personal_data Phase 1 임시 안내 접두 문구
_PERSONAL_DATA_PHASE1_PREFIX = (
    "회원 정보 직접 조회는 Phase 2 에서 지원될 예정이에요. "
    "우선 관련 정책 내용으로 도와드릴게요.\n\n"
)

# login_required 안내 접미 문구
_LOGIN_REQUIRED_SUFFIX = (
    "\n\n로그인하신 후에 더 정확한 개인화 정보를 안내드릴 수 있어요."
)

# 검색 결과 없음 fallback
_NO_RESULT_FALLBACK = (
    "죄송해요, 지금 당장 해당 내용을 찾지 못했어요. "
    "'문의하기' 탭에서 1:1 티켓으로 남겨주시면 담당자가 확인해 드릴게요."
)

# Solar Pro narrator 시스템 프롬프트
_NARRATOR_SYSTEM_PROMPT = """\
당신은 영화 서비스 '몽글' 고객센터 AI 상담원 '몽글이'예요.
아래 [검색 결과] 를 근거로 2~4문장의 자연스러운 대화체로 답변하세요.

## 절대 규칙
- [검색 결과] 에 없는 수치·연락처·정책은 절대 지어내지 마세요
- ~요/~에요 존댓말, 이모지·마크다운 금지
- "아마", "~인 것 같아요" 같은 추측 표현 금지
- 총 2~4문장 이내
- 검색 결과가 없거나 관련 내용이 없으면 솔직하게 "해당 내용을 찾지 못했어요"라고 안내

## FAQ 근거 사용 시
- question/answer 를 자연스럽게 요약해 친근하게 전달
- FAQ 번호나 "FAQ 에 따르면" 같은 표현 없이 자연스럽게

## 정책 근거 사용 시
- 정책 청크의 핵심 내용만 발췌해 쉬운 말로 풀어서 설명
- 설계서 섹션 번호·기술 용어 그대로 인용 금지
"""


async def narrator(state: SupportAssistantState) -> dict:
    """
    [v4 신규] tool_results_cache 의 검색 결과를 컨텍스트로 삼아 Solar Pro 로 최종 답변을 생성한다.

    처리 분기:
    1. redirect 의도: LLM 생성 없이 고정 메시지 + navigation 페이로드 설정
    2. personal_data + login_required 에러: 정책 RAG 결과 + 로그인 권유 결합
    3. personal_data (Phase 1 폴백): RAG 결과 + Phase 1 안내 접두어 결합
    4. faq 의도: ES FAQ 결과 → Solar Pro 자연어 답변
    5. policy 의도: Qdrant RAG 결과 → Solar Pro 자연어 답변
    6. 결과 없음: _NO_RESULT_FALLBACK 반환

    실패 시 _NO_RESULT_FALLBACK 반환 (에러 전파 금지).
    """
    intent = state.get("intent")
    intent_kind = getattr(intent, "kind", "faq") if intent is not None else "faq"
    user_message = (state.get("user_message") or "").strip()
    cache: dict = state.get("tool_results_cache") or {}
    is_guest = bool(state.get("is_guest", False))

    # ── redirect: LLM 없이 고정 메시지 + navigation 페이로드 ──
    if intent_kind == "redirect":
        navigation = {
            "target_path": "/chat",
            "label": "AI 채팅으로 이동",
            "candidates": [],  # Phase 2 에서 추천 목록 추가 예정
        }
        logger.info("narrator_redirect_fixed_message")
        return {
            "response_text": _REDIRECT_MESSAGE,
            "needs_human_agent": False,
            "navigation": navigation,
        }

    # ── login_required 에러: 정책 RAG + 로그인 권유 ──
    tool_name = (state.get("pending_tool_call") or {}).get("tool_name", "")
    ref_id = f"{tool_name}_0"
    tool_result = cache.get(ref_id, {})
    is_login_required = tool_result.get("error") == "login_required"

    if is_login_required:
        # 정책 RAG 결과가 있으면 보조 안내로 활용
        rag_chunks: list[dict] = state.get("rag_chunks") or []
        rag_context = _build_rag_context(rag_chunks)
        if rag_context:
            context_block = f"[정책 참고]\n{rag_context}"
        else:
            context_block = ""
        base_text = await _generate_with_solar(
            user_message=user_message,
            context_block=context_block,
            fallback=_NO_RESULT_FALLBACK,
        )
        text = base_text + _LOGIN_REQUIRED_SUFFIX
        logger.info("narrator_login_required_policy_fallback")
        return {
            "response_text": text,
            "needs_human_agent": False,
        }

    # ── faq 의도: ES 결과 → Solar Pro ──
    if intent_kind in ("faq",):
        faq_items = tool_result.get("data", {}).get("faqs", []) if tool_result.get("ok") else []
        context_block = _build_faq_context(faq_items)
        if not context_block:
            logger.info("narrator_faq_no_results_fallback")
            return {
                "response_text": _NO_RESULT_FALLBACK,
                "needs_human_agent": True,
            }
        text = await _generate_with_solar(
            user_message=user_message,
            context_block=context_block,
            fallback=_NO_RESULT_FALLBACK,
        )
        logger.info("narrator_faq_done", text_length=len(text))
        return {
            "response_text": text,
            "needs_human_agent": False,
        }

    # ── policy 의도: RAG 결과 → Solar Pro ──
    if intent_kind in ("policy",):
        rag_chunks = state.get("rag_chunks") or []
        rag_context = _build_rag_context(rag_chunks)
        if not rag_context:
            # Qdrant 검색 실패 또는 결과 없음
            logger.info("narrator_policy_no_results_fallback")
            return {
                "response_text": _NO_RESULT_FALLBACK,
                "needs_human_agent": True,
            }
        context_block = f"[정책 근거]\n{rag_context}"
        text = await _generate_with_solar(
            user_message=user_message,
            context_block=context_block,
            fallback=_NO_RESULT_FALLBACK,
        )
        logger.info("narrator_policy_done", rag_count=len(rag_chunks), text_length=len(text))
        return {
            "response_text": text,
            "needs_human_agent": False,
        }

    # ── personal_data Phase 1 폴백: RAG 결과 + 안내 접두 ──
    if intent_kind == "personal_data":
        rag_chunks = state.get("rag_chunks") or []
        rag_context = _build_rag_context(rag_chunks)
        if rag_context:
            context_block = f"[정책 참고]\n{rag_context}"
            base_text = await _generate_with_solar(
                user_message=user_message,
                context_block=context_block,
                fallback=_NO_RESULT_FALLBACK,
            )
            text = _PERSONAL_DATA_PHASE1_PREFIX + base_text
        else:
            text = (
                _PERSONAL_DATA_PHASE1_PREFIX
                + "관련 정책 내용을 찾지 못했어요. "
                "'문의하기' 탭에서 1:1 티켓으로 남겨주시면 담당자가 확인해 드릴게요."
            )
        suffix = _LOGIN_REQUIRED_SUFFIX if is_guest else ""
        logger.info("narrator_personal_data_phase1", is_guest=is_guest)
        return {
            "response_text": text + suffix,
            "needs_human_agent": False,
        }

    # ── 기타 예상치 못한 kind ──
    logger.warning("narrator_unexpected_kind", intent_kind=intent_kind)
    return {
        "response_text": _NO_RESULT_FALLBACK,
        "needs_human_agent": True,
    }


def _build_faq_context(faq_items: list[dict]) -> str:
    """
    ES FAQ 검색 결과 dict 리스트를 narrator 프롬프트용 텍스트 블록으로 변환한다.

    상위 3건만 사용 (Solar Pro max_tokens=2048 여유 확보).
    """
    lines: list[str] = []
    for i, item in enumerate(faq_items[:3], start=1):
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            lines.append(f"[FAQ {i}] Q: {q}\nA: {a}")
    return "\n\n".join(lines)


def _build_rag_context(rag_chunks: list[dict]) -> str:
    """
    Qdrant 정책 RAG 청크 dict 리스트를 narrator 프롬프트용 텍스트 블록으로 변환한다.

    상위 3건 청크 본문만 사용.
    """
    lines: list[str] = []
    for i, chunk in enumerate(rag_chunks[:3], start=1):
        text = (chunk.get("text") or "").strip()
        section = (chunk.get("section") or "").strip()
        if text:
            header = f"[정책 {i}]" + (f" ({section})" if section else "")
            lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


async def _generate_with_solar(
    user_message: str,
    context_block: str,
    fallback: str,
) -> str:
    """
    Solar Pro API 로 narrator 최종 답변을 생성한다.

    실패 시 fallback 반환 (에러 전파 금지).
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from monglepick.llm.factory import get_solar_api_llm

        # Solar Pro API 직접 사용 (narrator 는 품질이 중요한 체인)
        llm = get_solar_api_llm(temperature=0.3)

        human_content = (
            f"[사용자 질문]\n{user_message}\n\n"
            f"{context_block}\n\n"
            "위 내용을 바탕으로 몽글이 톤으로 2~4문장으로 답변해 주세요. 본문만 출력하세요."
        )
        messages = [
            SystemMessage(content=_NARRATOR_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]
        response = await llm.ainvoke(messages)
        text = (response.content or "").strip()
        return text if text else fallback

    except Exception as exc:  # noqa: BLE001 — 에러 전파 금지
        logger.warning(
            "narrator_solar_generate_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return fallback


# =============================================================================
# 10) smart_fallback — v4 신규
# =============================================================================


async def smart_fallback(state: SupportAssistantState) -> dict:
    """
    [v4 신규] tool_selector 실패 또는 매핑 없는 intent 에서 진입하는 안전망 노드.

    "이런 분야 도와드릴 수 있어요" 형식의 메뉴 안내를 반환한다.
    에러 전파 금지.
    """
    intent = state.get("intent")
    intent_kind = getattr(intent, "kind", "unknown") if intent is not None else "unknown"
    error = state.get("error") or ""

    fallback_text = (
        "현재 요청을 처리하지 못했어요. 다음 분야에서 도와드릴 수 있어요:\n"
        "• 서비스 사용법 안내 (FAQ)\n"
        "• 등급·AI 쿼터·구독·결제 정책 안내\n"
        "• 기타 불편 사항 → '문의하기' 탭에서 1:1 티켓"
    )

    logger.info(
        "support_smart_fallback",
        intent_kind=intent_kind,
        error=error,
    )

    return {
        "response_text": fallback_text,
        "needs_human_agent": False,
    }
