"""
support_assistant LangGraph 노드 (v4 Phase 2 — 다중 hop ReAct).

### v4 Phase 2 변경점 (2026-04-28)
Phase 1 의 단일 hop 골격을 **다중 hop ReAct 루프** 로 확장한다.

    v4 Phase 1 유지 노드 (동작 완전 보존):
        - context_loader      : is_guest 결정 + v4 State 필드 초기화
        - support_agent       : v3 원본 보존 (v3 그래프 전용)
        - response_formatter  : complaint 고정 템플릿, redirect navigation 처리
        - intent_classifier   : classify_support_intent → state.intent
        - smalltalk_responder : vLLM 1.2B / Solar fallback 인사 응대
        - tool_executor       : SUPPORT_TOOL_REGISTRY / lookup_faq(ES) 실행
        - smart_fallback      : tool_selector 실패 안내

    v4 Phase 2 확장 노드 (3개 교체):
        - tool_selector   : Phase 1 단순 매핑 → Solar bind_tools (ReAct 다중 hop)
                           의도+게스트 여부에 따라 tool 목록을 동적으로 바인딩.
                           finish_task 가상 tool 종결 시그널 지원.
        - observation     : hop_count 누적 + route_after_observation 이 MAX_HOPS 분기.
                           Phase 1 에서는 항상 narrator 로 직행했으나,
                           Phase 2 에서 finish_task / MAX_HOPS 초과 시 narrator 강제 분기,
                           그 외 read tool 완료 시 tool_selector 재진입.
        - narrator        : 단일 hop 결과 단순 인용 → 다중 hop 누적 결과 종합 진단 답변.
                           support_narrator_chain.generate_narrator_response() 호출.

    v3 유지 노드 (v3 코드/동작 완전 보존):
        - support_agent : v3 generate_support_reply 체인. v3 그래프 전용.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §3 (그래프) §5 (tool) §6 (RAG) §7 (시나리오)

### 에러 처리 원칙
모든 노드는 async def, try/except, 실패 시 graceful fallback 반환.
에러 전파 절대 금지.

### MAX_HOPS
환경변수 SUPPORT_MAX_HOPS (기본 3). observation 노드와 graph.py route_after_observation 에서
참조한다. 3을 초과하면 narrator 로 강제 분기해 수집된 부분 결과로 답변한다.
"""

from __future__ import annotations

import os

import structlog

from monglepick.agents.support_assistant.faq_client import fetch_faqs
from monglepick.agents.support_assistant.faq_search import search_faq_candidates
from monglepick.agents.support_assistant.models import (
    FaqDoc,
    MatchedFaq,
    SupportAssistantState,
    ensure_reply,
)
from monglepick.prompts.support_assistant import (
    SUPPORT_COMPLAINT_TEMPLATE,
    SUPPORT_SMALLTALK_HUMAN_PROMPT,
    SUPPORT_SMALLTALK_SYSTEM_PROMPT,
)
from monglepick.tools.support_tools import SUPPORT_TOOL_REGISTRY, ToolContext

logger = structlog.get_logger(__name__)

# =============================================================================
# ReAct 루프 상한 (Phase 2)
# =============================================================================
# 환경변수 SUPPORT_MAX_HOPS 로 override 가능. 기본 3.
# observation 노드와 graph.py route_after_observation 이 이 값을 참조한다.
MAX_HOPS: int = int(os.getenv("SUPPORT_MAX_HOPS", "3"))


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

    ### 멀티턴 컨텍스트 보존 (2026-04-28)
    - history 는 LangGraph checkpointer 가 thread_id=session_id 단위로 보존한다.
    - 첫 턴(checkpointer 미존재)이면 state.get("history") 가 None → [] 로 초기화.
    - 이후 턴은 이전 history 를 그대로 유지(반환 dict 에 history 미포함).
    - response_formatter 가 매 턴 끝에 user/bot 두 턴을 append.

    ### v3 동작 보존
    - Backend FAQ 전체 조회 (ES-first fallback 안전망 + matched_faq 교차 참조용)
    - Backend 장애 시 faqs=[] 로 계속 진행 (에러 전파 금지)
    - reply/matched_faqs/response_text/needs_human_agent/error 초기화
    """
    user_message = (state.get("user_message") or "").strip()
    user_id = state.get("user_id") or ""
    is_guest = not bool(user_id)

    # 첫 턴이면 None → [], 이후 턴이면 보존된 history 가 들어 있다.
    existing_history = state.get("history")
    if existing_history is None:
        history_init: list = []
    else:
        history_init = list(existing_history)

    logger.info(
        "support_context_loader_start",
        session_id=state.get("session_id", ""),
        user_id=user_id or "(guest)",
        is_guest=is_guest,
        history_length=len(history_init),
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
        # 멀티턴: checkpointer 보존된 history 를 명시적으로 다시 set 해 None → [] 로 정규화.
        # 이후 노드는 이 값을 read 만 하고 response_formatter 가 새 턴을 append 한다.
        "history": history_init,
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


# Phase 2.5 cleanup: v3 `support_agent` 노드 제거. `_select_matched_faqs` 헬퍼는
# v4 `tool_executor` 의 lookup_faq 결과 매핑에서 그대로 재사용한다.


# =============================================================================
# 3) response_formatter — v4 갱신 (complaint 고정 템플릿 + redirect 처리 추가)
# =============================================================================


def _append_history_turn(
    history: list | None,
    user_message: str,
    bot_text: str,
    intent_kind: str,
    needs_human: bool,
    max_turns: int = 10,
) -> list:
    """
    멀티턴 history 에 이번 턴의 user/bot 메시지를 append.

    Args:
        history: 이전까지의 history 리스트 (None 이면 [] 로 초기화)
        user_message: 현재 턴 사용자 발화
        bot_text: 봇 최종 응답 본문
        intent_kind: 분류된 의도 (logging/디버그용 메타)
        needs_human: 1:1 유도 여부 (메타)
        max_turns: 보존할 최근 턴 수 (user+bot 쌍 기준 max_turns 쌍)

    Returns:
        새로 append 된 history 리스트. 길이는 최대 `max_turns * 2` (user+bot 쌍).
        오래된 항목은 버린다 (LangGraph state 무한 누적 방지).
    """
    new_history: list = list(history) if history else []
    if user_message:
        new_history.append({"role": "user", "content": user_message})
    if bot_text:
        new_history.append(
            {
                "role": "assistant",
                "content": bot_text,
                # 메타: 디버그/통계용. intent_classifier 는 read 하지 않음.
                "intent": intent_kind or "",
                "needs_human": bool(needs_human),
            }
        )
    # 최근 max_turns 쌍 (= 2*max_turns 항목) 만 유지
    cap = max_turns * 2
    if len(new_history) > cap:
        new_history = new_history[-cap:]
    return new_history


async def _log_chat_event_async(state: SupportAssistantState, response_text: str) -> None:
    """
    이번 턴을 fire-and-forget 으로 통계 로깅한다 (DB INSERT).

    실패해도 사용자 응답에 영향 주지 않도록 try/except 로 완전 무력화.
    상세 구현은 monglepick.agents.support_assistant.chat_log 모듈 참조.
    """
    try:
        # lazy import — chat_log 모듈 미존재(개발 환경) / DB 미설정 시 응답 차단 방지
        from monglepick.agents.support_assistant.chat_log import insert_support_chat_log

        intent = state.get("intent")
        await insert_support_chat_log(
            session_id=state.get("session_id", ""),
            user_id=state.get("user_id") or None,
            is_guest=bool(state.get("is_guest", False)),
            user_message=state.get("user_message") or "",
            response_text=response_text,
            intent_kind=getattr(intent, "kind", "unknown") if intent else "unknown",
            intent_confidence=float(getattr(intent, "confidence", 0.0)) if intent else 0.0,
            intent_reason=getattr(intent, "reason", "") if intent else "",
            needs_human=bool(state.get("needs_human_agent", False)),
            hop_count=int(state.get("hop_count") or 0),
            tool_call_history=list(state.get("tool_call_history") or []),
        )
    except Exception as exc:  # noqa: BLE001 — 로깅 실패 절대 응답 차단 금지
        logger.warning(
            "support_chat_log_insert_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )


async def response_formatter(state: SupportAssistantState) -> dict:
    """
    최종 본문/배너 플래그를 한 번 더 가드하고 history 에 이번 턴을 누적한다.

    ### v4 갱신 (2026-04-28)
    - complaint 의도: SUPPORT_COMPLAINT_TEMPLATE 고정 문자열 + needs_human=True
      (v4 에서 complaint 는 narrator 를 거치지 않고 이 노드가 직접 처리)
    - redirect 의도: 고정 안내 메시지 + needs_human=False
      (navigation 페이로드는 narrator 또는 이 노드에서 이미 state 에 채워짐)
    - v3 호환 경로 (reply 존재): 기존 동작 완전 보존

    ### 멀티턴 컨텍스트 (2026-04-28)
    - 모든 분기에서 _append_history_turn() 으로 user/bot 두 턴을 history 에 append.
    - 최근 10턴 쌍 (= 20 항목) 까지만 보존해 state 무한 누적 방지.
    - LangGraph checkpointer(MemorySaver/AsyncRedisSaver) 가 thread_id=session_id
      단위로 다음 턴까지 보존 → 이전 채팅과 컨텍스트 연계 가능.

    ### 통계·감사 (2026-04-28)
    - _log_chat_event_async() 로 매 턴 fire-and-forget DB INSERT.
    - 실패는 무시 (사용자 응답 차단 금지).

    ### 에러 처리
    빈 본문이면 최후 fallback 메시지 + needs_human=True (v3 동작 유지).
    """
    import asyncio  # 함수 단독 사용 — 모듈 상단 import 미오염

    intent = state.get("intent")
    intent_kind = getattr(intent, "kind", None) if intent is not None else None
    user_message = (state.get("user_message") or "").strip()
    prev_history = state.get("history") or []

    # v4 경로: complaint → 고정 템플릿 직접 발행
    if intent_kind == "complaint":
        logger.info(
            "support_response_formatter_complaint",
            intent_kind=intent_kind,
        )
        new_history = _append_history_turn(
            prev_history,
            user_message,
            SUPPORT_COMPLAINT_TEMPLATE,
            intent_kind=intent_kind,
            needs_human=True,
        )
        # fire-and-forget 통계 로깅 — await 하지 않아 응답 지연 0
        # 단, asyncio.create_task 는 현재 이벤트 루프 내에서만 안전
        asyncio.create_task(
            _log_chat_event_async(
                {**state, "needs_human_agent": True},
                SUPPORT_COMPLAINT_TEMPLATE,
            )
        )
        return {
            "response_text": SUPPORT_COMPLAINT_TEMPLATE,
            "needs_human_agent": True,
            "history": new_history,
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
        new_history = _append_history_turn(
            prev_history,
            user_message,
            text,
            intent_kind=intent_kind,
            needs_human=False,
        )
        asyncio.create_task(_log_chat_event_async(state, text))
        return {
            "response_text": text,
            "needs_human_agent": False,
            "history": new_history,
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

    new_history = _append_history_turn(
        prev_history,
        user_message,
        text,
        intent_kind=kind,
        needs_human=needs_human,
    )
    asyncio.create_task(
        _log_chat_event_async({**state, "needs_human_agent": needs_human}, text)
    )

    logger.info(
        "support_response_formatter_done",
        kind=kind,
        needs_human=needs_human,
        text_length=len(text),
        history_length=len(new_history),
    )

    return {
        "response_text": text,
        "needs_human_agent": needs_human,
        "history": new_history,
    }


# =============================================================================
# 4) intent_classifier — v4 신규
# =============================================================================


def _format_history_context(history: list, max_turns: int = 3) -> str:
    """
    멀티턴 history 의 최근 N턴(user+bot 쌍)을 LLM 프롬프트용 텍스트 블록으로 변환.

    Args:
        history: response_formatter 가 누적한 history. [{role, content, ...}, ...]
        max_turns: 최근 몇 쌍을 포함할지 (기본 3쌍 = 6 메시지)

    Returns:
        "사용자: ...\n몽글이: ...\n사용자: ...\n몽글이: ..." 형식의 멀티라인.
        history 가 비어 있으면 빈 문자열.
    """
    if not history:
        return ""
    # 최근 max_turns 쌍 = 2*max_turns 항목
    recent = history[-(max_turns * 2):]
    lines: list[str] = []
    for entry in recent:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role", "")
        content = (entry.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"사용자: {content}")
        elif role == "assistant":
            lines.append(f"몽글이: {content}")
    return "\n".join(lines)


async def intent_classifier(state: SupportAssistantState) -> dict:
    """
    [v4 신규] classify_support_intent() 로 SupportIntent 를 분류해 state 에 채운다.

    순환 import 방지를 위해 함수 내부에서 lazily import 한다.
    (chains.support_intent_chain → llm.factory → ... 의존성 체인이
    모듈 레벨 import 에서 순환 가능성 있음)

    ### 멀티턴 컨텍스트 (2026-04-28)
    - history 최근 3턴 쌍을 추가 컨텍스트로 분류기에 전달.
    - 짧은 후속 질문("그럼 환불은?", "그건 뭐예요?") 의 의도를 정확히 파악하기 위함.
    - history 가 빈 첫 턴은 기존 동작과 동일.

    실패 시 faq intent 로 폴백 (에러 전파 금지).
    """
    user_message = (state.get("user_message") or "").strip()
    is_guest = bool(state.get("is_guest", False))
    session_id = state.get("session_id", "")
    history = state.get("history") or []
    history_context = _format_history_context(history, max_turns=3)

    try:
        # lazy import — 순환 의존성 방지
        from monglepick.chains.support_intent_chain import classify_support_intent

        intent = await classify_support_intent(
            user_message=user_message,
            is_guest=is_guest,
            request_id=session_id,
            history_context=history_context,
        )
        logger.info(
            "support_intent_classifier_done",
            kind=intent.kind,
            confidence=round(intent.confidence, 2),
            is_guest=is_guest,
            history_turns=len(history) // 2,
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


# 사용자가 "뭘 할 수 있어 / 무엇을 도와줄 수 있어 / 어떤 기능 / 어떤 도움" 처럼
# 챗봇 capability 를 묻는 발화 패턴. 작은 LLM(vLLM 1.2B) 이 환각하기 쉬운 카테고리라
# Python 키워드 매칭으로 가로채 고정 문자열을 반환한다.
# (스마트한 LLM 분류 대신 짧은 휴리스틱으로 안정성 확보.)
_CAPABILITY_QUESTION_KEYWORDS: tuple[str, ...] = (
    "뭘 할 수 있",
    "뭘할 수 있",
    "뭘할수있",
    "무엇을 할 수 있",
    "무엇을 도와",
    "뭐 도와",
    "뭐 할 수 있",
    "뭐할 수 있",
    "뭐할수있",
    "어떤 기능",
    "어떤 도움",
    "어떻게 사용",
    "어떻게 써",
    "어떤 일을 해",
    "할 수 있는 일",
    "할 수 있는게",
    "할수있는게",
    "what can you",
    "capabilities",
)

# 서비스 이름 환각 자동 교정 — vLLM EXAONE 1.2B 가 "몽블랑" / "몽글" 등으로
# 잘못 출력해도 본문에서 후처리한다. (프롬프트 보강 + 정규화 이중 방어.)
_SERVICE_NAME_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("몽블랑", "몽글픽"),
    ("몽블랭", "몽글픽"),
    # 단어 경계 처리: 사용자가 정말 '몽글' 만 적은 경우는 적지만,
    # LLM 출력 텍스트에서 '몽글픽' 으로 보정해도 의미 손상은 없다.
    ("몽글 ", "몽글픽 "),
    ("몽글의", "몽글픽의"),
    ("몽글에서", "몽글픽에서"),
    ("몽글이라는", "몽글픽이라는"),
    ("몽글 서비스", "몽글픽 서비스"),
    ("몽글 고객센터", "몽글픽 고객센터"),
    ("몽글 챗봇", "몽글픽 챗봇"),
    # 몽글이(상담원 페르소나 이름) 는 보존해야 하므로 위치별로만 치환
)


def _is_capability_question(text: str) -> bool:
    """사용자 발화가 봇 capability 질의 인지 키워드 매칭으로 판정."""
    lowered = text.lower().replace(" ", "")
    for kw in _CAPABILITY_QUESTION_KEYWORDS:
        normalized_kw = kw.lower().replace(" ", "")
        if normalized_kw and normalized_kw in lowered:
            return True
    return False


def _normalize_service_name(text: str) -> str:
    """LLM 출력 본문에서 잘못된 서비스 이름을 '몽글픽' 으로 교정."""
    for src, dst in _SERVICE_NAME_REPLACEMENTS:
        text = text.replace(src, dst)
    return text


# 봇 capability 안내 — LLM 환각을 우회하기 위한 고정 문자열.
# 1:1 문의 채널까지 한 문장에 명시해 needs_human=True 와 함께 상담원 연결 배너 노출.
_CAPABILITY_FIXED_REPLY = (
    "안녕하세요, 몽글픽 고객센터 챗봇 몽글이예요. 다음 분야에서 도와드릴 수 있어요:\n"
    "• 서비스 사용법 안내 (FAQ): 리뷰 작성, 비밀번호 변경, 탈퇴 등\n"
    "• 등급·AI 쿼터·구독·결제 정책 안내\n"
    "• 본인 데이터 진단: 포인트 적립 이력, 출석 체크, AI 쿼터, 구독 상태 등\n"
    "환불·계정 제재·기타 직접 처리가 필요한 요청은 '문의하기' 탭에서 1:1 티켓으로 "
    "남겨주시면 담당자가 확인해 드릴게요."
)


async def smalltalk_responder(state: SupportAssistantState) -> dict:
    """
    [v4 신규] 인사/안부 발화에 짧은 몽글이 톤 응대를 생성한다.

    우선순위:
    1. capability 질의 ("뭘 할 수 있어?" 등) → 고정 문자열 + needs_human=True
       - vLLM EXAONE 1.2B 가 서비스 이름을 환각(몽블랑 등)하는 빈도가 높아
         LLM 호출을 우회하고 안정적인 fixed-string 으로 응답한다.
       - needs_human=True 로 Client 의 '상담원 연결' 배너도 함께 노출 → 1:1 유도 회복.
    2. 그 외 일반 smalltalk → LLM 생성 + 서비스 이름 후처리 정규화
       - hybrid + VLLM_ENABLED → vLLM EXAONE 1.2B
       - hybrid + Ollama → 몽글이 모델
       - api_only → Solar API
    3. LLM 실패 시 '몽글픽' 표기 fallback 문자열

    실패 시 고정 인사 문자열 fallback (에러 전파 금지).
    """
    user_message = (state.get("user_message") or "").strip()
    history = state.get("history") or []
    history_context = _format_history_context(history, max_turns=2)

    # capability 질의 가로채기 — LLM 환각 회피 + 1:1 유도 강제 노출
    if _is_capability_question(user_message):
        logger.info(
            "support_smalltalk_capability_intercept",
            user_message_preview=user_message[:60],
        )
        return {
            "response_text": _CAPABILITY_FIXED_REPLY,
            # 1:1 티켓 안내까지 본문에 포함했으므로 상담원 연결 배너 강제 노출
            "needs_human_agent": True,
        }

    _FALLBACK_SMALLTALK = (
        "안녕하세요! 몽글픽 고객센터 챗봇이에요. 궁금한 점 있으시면 편하게 말씀해 주세요."
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from monglepick.llm.factory import get_conversation_llm

        # 멀티턴 컨텍스트 — Human prompt 본문 앞에 [이전 대화] 블록을 prefix.
        # SUPPORT_SMALLTALK_HUMAN_PROMPT 의 placeholder 와 호환되도록 메시지 합성.
        if history_context:
            human_text = (
                f"[이전 대화]\n{history_context}\n\n"
                + SUPPORT_SMALLTALK_HUMAN_PROMPT.format(user_message=user_message)
            )
        else:
            human_text = SUPPORT_SMALLTALK_HUMAN_PROMPT.format(
                user_message=user_message
            )

        llm = get_conversation_llm()
        messages = [
            SystemMessage(content=SUPPORT_SMALLTALK_SYSTEM_PROMPT),
            HumanMessage(content=human_text),
        ]
        response = await llm.ainvoke(messages)
        text = (response.content or "").strip()

        if not text:
            text = _FALLBACK_SMALLTALK

        # 서비스 이름 환각 후처리 — 작은 LLM 출력을 '몽글픽' 으로 강제 교정
        text = _normalize_service_name(text)

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
# 6) tool_selector — v4 Phase 2 교체 (Solar bind_tools ReAct)
# =============================================================================

# faq 의도는 ES BM25 직접 경로 — Solar bind_tools 를 거치지 않고
# 항상 lookup_faq(ES) 로 고정 매핑한다. ES 검색은 SUPPORT_TOOL_REGISTRY 외부이므로
# bind_tools 목록에 포함되지 않는다.
_FAQ_DIRECT_TOOL = "lookup_faq"


async def tool_selector(state: SupportAssistantState) -> dict:
    """
    [v4 Phase 2] Solar bind_tools 로 최적 tool 하나를 선택하고 pending_tool_call 에 채운다.

    ### Phase 2 변경점 (단순 매핑 → Solar bind_tools)
    - faq 의도: ES 직접 경로 유지 (lookup_faq 고정 매핑). Solar LLM 호출 없음.
    - policy / personal_data 의도: Solar bind_tools 로 최적 tool 선택.
      * personal_data + 로그인: Read tool 8개 + lookup_policy 중 최적 선택
      * personal_data + 게스트: lookup_policy 만 바인딩 (본인 데이터 tool 차단)
      * policy: lookup_policy 만 바인딩
    - 이전 hop 이력(tool_call_history + tool_results_cache) 을 압축해 LLM 컨텍스트에 주입.
    - finish_task 가상 tool: Solar 가 "더 이상 조회 불필요" 판단 시 선택.
      → route_after_tool_select 가 이를 감지해 tool_executor 를 건너뛰고 narrator 로 직행.
    - Solar LLM 실패 / tool 없음 / 허용 목록 위반 → pending_tool_call=None → smart_fallback.

    ### ReAct 루프에서의 역할
    observation 노드가 hop_count < MAX_HOPS 이고 finish_task 가 아니면 이 노드로 재진입한다.
    매 진입마다 이전 hop 결과를 확인해 다음에 호출할 tool 을 다시 결정한다.

    실패 시 smart_fallback 경로로 전환 (pending_tool_call = None, error 기록).
    에러 전파 금지.
    """
    intent = state.get("intent")
    intent_kind = getattr(intent, "kind", "faq") if intent is not None else "faq"
    intent_confidence = float(getattr(intent, "confidence", 0.0)) if intent is not None else 0.0
    user_message = (state.get("user_message") or "").strip()
    is_guest = bool(state.get("is_guest", False))
    hop_count: int = state.get("hop_count") or 0
    tool_call_history: list[dict] = list(state.get("tool_call_history") or [])
    tool_results_cache: dict = dict(state.get("tool_results_cache") or {})
    session_id = state.get("session_id", "")

    # ── faq 의도: ES 직접 경로 (Solar bind_tools 미사용) ──
    if intent_kind == "faq":
        pending = {"tool_name": _FAQ_DIRECT_TOOL, "args": {"query": user_message}}
        logger.info(
            "tool_selector_faq_direct",
            tool_name=_FAQ_DIRECT_TOOL,
            hop_count=hop_count,
        )
        return {"pending_tool_call": pending}

    # ── policy / personal_data 의도: Solar bind_tools ──
    if intent_kind not in ("policy", "personal_data"):
        # redirect / smalltalk / complaint 는 tool_selector 에 도달하지 않아야 함
        logger.warning(
            "tool_selector_unexpected_kind",
            intent_kind=intent_kind,
        )
        return {"pending_tool_call": None, "error": f"no_tool_for:{intent_kind}"}

    # lazy import — 순환 의존성 방지 (체인이 llm.factory → ... 를 타는 긴 의존성 체인)
    try:
        from monglepick.chains.support_tool_selector_chain import select_support_tool

        selected = await select_support_tool(
            user_message=user_message,
            intent_kind=intent_kind,
            intent_confidence=intent_confidence,
            is_guest=is_guest,
            tool_call_history=tool_call_history,
            tool_results_cache=tool_results_cache,
            hop_count=hop_count,
            max_hops=MAX_HOPS,
            request_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001 — 에러 전파 금지
        logger.warning(
            "tool_selector_chain_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            intent_kind=intent_kind,
            hop_count=hop_count,
        )
        return {"pending_tool_call": None, "error": f"selector_chain_error:{type(exc).__name__}"}

    if selected is None:
        # Solar LLM 이 tool 을 선택하지 못한 경우 → smart_fallback 경로
        logger.info(
            "tool_selector_no_selection",
            intent_kind=intent_kind,
            hop_count=hop_count,
        )
        return {"pending_tool_call": None, "error": f"no_tool_selected:{intent_kind}"}

    # finish_task 가상 tool: tool_executor 를 건너뛰고 narrator 로 직행하는 시그널
    # route_after_tool_select 에서 이 이름을 감지해 narrator 로 라우팅한다.
    pending = {
        "tool_name": selected.name,
        "args": selected.arguments,
        "_rationale": selected.rationale,  # 내부 메타 — tool_executor 가 _ 접두 필드 제거
    }
    logger.info(
        "tool_selector_done",
        intent_kind=intent_kind,
        tool_name=selected.name,
        hop_count=hop_count,
        rationale=selected.rationale,
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
    # Phase 2: ref_id 는 현재 hop_count(0-indexed) 기반으로 생성한다.
    # observation 이 hop_count 를 +1 하기 전이므로 현재 값이 이번 실행 인덱스.
    current_hop: int = state.get("hop_count") or 0
    ref_id = f"{tool_name}_{current_hop}"

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
# 8) observation — v4 Phase 2 교체 (다중 hop ReAct 누적)
# =============================================================================


async def observation(state: SupportAssistantState) -> dict:
    """
    [v4 Phase 2] tool 실행 결과를 tool_call_history 에 누적하고 hop 카운터를 증가시킨다.

    ### Phase 2 변경점 (Phase 1 단일 hop stub → 다중 hop 누적)
    - hop_count 를 현재 값에서 +1 증가 (Phase 1 에서는 항상 1로 고정했으나 이제 누적).
    - pending_tool_call.tool_name 과 tool_results_cache 의 ref_id 를 hop 번호로 매핑.
      ref_id = "{tool_name}_{현재 hop 번호}" (0-indexed).
    - tool_call_history 에 {"hop", "tool_name", "ok", "error"} dict 를 append.
      이 이력이 다음 hop 에서 tool_selector 의 Solar bind_tools 컨텍스트로 주입된다.

    ### route_after_observation 분기 (graph.py 에서 처리)
    graph.py 의 route_after_observation 이 다음 목적지를 결정한다:
      - hop_count >= MAX_HOPS → narrator (상한 도달, 부분 결과로 답변)
      - last_call.tool_name == "finish_task" → narrator (Solar 가 종결 판단)
      - 그 외 read tool 완료 → tool_selector (다음 hop 재진입)

    observation 노드 자체는 항상 state 누적만 하고 라우팅 결정은 graph.py 에 위임한다.
    에러 전파 금지.
    """
    pending: dict | None = state.get("pending_tool_call")
    tool_name = pending.get("tool_name", "") if pending else ""

    # 현재 hop_count (이 노드 실행 전 값) — 이번 hop 의 ref_id 인덱스로 사용
    current_hop_count: int = state.get("hop_count") or 0
    # ref_id: tool_executor 가 "{tool_name}_{current_hop_count}" 로 저장
    # (tool_executor 의 ref_id = f"{tool_name}_{current_hop_count}")
    ref_id = f"{tool_name}_{current_hop_count}"
    cache: dict = state.get("tool_results_cache") or {}
    result_summary = cache.get(ref_id, {})

    # tool_call_history 에 이번 hop 기록 append
    history: list[dict] = list(state.get("tool_call_history") or [])
    new_hop_number = current_hop_count + 1
    history.append(
        {
            "hop": new_hop_number,
            "tool_name": tool_name,
            "ok": result_summary.get("ok", False),
            "error": result_summary.get("error"),
        }
    )

    logger.info(
        "support_observation_done",
        tool_name=tool_name,
        hop_count=new_hop_number,
        ok=result_summary.get("ok", False),
        max_hops=MAX_HOPS,
        will_continue=(
            new_hop_number < MAX_HOPS
            and tool_name != "finish_task"
        ),
    )

    return {
        "hop_count": new_hop_number,
        "tool_call_history": history,
    }


# =============================================================================
# 9) narrator — v4 Phase 2 교체 (다중 hop 진단 답변)
# =============================================================================

# redirect 의도 고정 메시지 (LLM 생성 없이 Python 템플릿)
_REDIRECT_MESSAGE = (
    "영화 추천은 메인 AI 채팅 탭이 더 잘 도와드려요. "
    "상단 'AI 채팅' 메뉴에서 편하게 물어봐 주세요!"
)

# 검색/조회 결과 없음 fallback
_NO_RESULT_FALLBACK = (
    "죄송해요, 지금 당장 해당 내용을 찾지 못했어요. "
    "'문의하기' 탭에서 1:1 티켓으로 남겨주시면 담당자가 확인해 드릴게요."
)


async def narrator(state: SupportAssistantState) -> dict:
    """
    [v4 Phase 2] 다중 hop 누적 결과를 종합해 Solar Pro 로 진단 답변을 생성한다.

    ### Phase 2 변경점 (단순 결과 인용 → 진단 답변)
    - support_narrator_chain.generate_narrator_response() 호출.
    - tool_call_history 전체 + tool_results_cache 전체를 컨텍스트로 주입.
    - 본인 데이터(ai_quota, point_history 등)와 정책 RAG 청크를 함께 참고해
      원인 진단 + 행동 안내를 포함한 4~7문장 답변 생성.
    - faq 의도: Phase 1 과 동일하게 ES FAQ 결과 기반 답변 (2~4문장 단순 인용 유지).

    ### 처리 분기
    1. redirect 의도: LLM 없이 고정 메시지 + navigation 페이로드 설정
    2. faq 의도: ES FAQ 결과 → 단순 인용 답변 (Phase 1 로직 유지)
    3. policy / personal_data 의도:
       - generate_narrator_response() 로 다중 hop 진단 답변 생성
       - 게스트 + personal_data: 로그인 권유 suffix 자동 추가 (체인 내부 처리)
    4. 결과 없음: _NO_RESULT_FALLBACK 반환

    실패 시 _NO_RESULT_FALLBACK 반환 (에러 전파 금지).
    """
    intent = state.get("intent")
    intent_kind = getattr(intent, "kind", "faq") if intent is not None else "faq"
    intent_confidence = float(getattr(intent, "confidence", 0.0)) if intent is not None else 0.0
    user_message = (state.get("user_message") or "").strip()
    cache: dict = state.get("tool_results_cache") or {}
    is_guest = bool(state.get("is_guest", False))
    hop_count: int = state.get("hop_count") or 0
    tool_call_history: list[dict] = list(state.get("tool_call_history") or [])
    rag_chunks: list[dict] = state.get("rag_chunks") or []
    # 멀티턴 컨텍스트: 최근 3턴 쌍을 narrator chain 에 함께 전달.
    # 사용자가 "그럼 환불은요?" 처럼 이전 답변에 의존하는 후속 질문을 할 때
    # 진단 답변 품질을 떨어뜨리지 않기 위함.
    history = state.get("history") or []
    history_context = _format_history_context(history, max_turns=3)

    # ── redirect: LLM 없이 고정 메시지 + navigation 페이로드 ──
    if intent_kind == "redirect":
        navigation = {
            "target_path": "/chat",
            "label": "AI 채팅으로 이동",
            "candidates": [],
        }
        logger.info("narrator_redirect_fixed_message")
        return {
            "response_text": _REDIRECT_MESSAGE,
            "needs_human_agent": False,
            "navigation": navigation,
        }

    # ── faq 의도: ES FAQ 결과 기반 단순 인용 (Phase 1 로직 유지) ──
    # faq 는 단일 hop 으로 충분하므로 다중 hop 진단 체인을 사용하지 않는다.
    if intent_kind == "faq":
        # hop_count=0 이면 ref_id는 "lookup_faq_0", hop_count 이후면 마지막 hop 인덱스 사용
        # 단, faq 는 항상 첫 번째 hop(index=0)에서 실행됨
        ref_id = "lookup_faq_0"
        tool_result = cache.get(ref_id, {})
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
            history_context=history_context,
        )
        logger.info("narrator_faq_done", text_length=len(text))
        return {
            "response_text": text,
            "needs_human_agent": False,
        }

    # ── policy / personal_data: 다중 hop 진단 답변 ──
    if intent_kind in ("policy", "personal_data"):
        try:
            from monglepick.chains.support_narrator_chain import generate_narrator_response

            text = await generate_narrator_response(
                user_message=user_message,
                intent_kind=intent_kind,
                intent_confidence=intent_confidence,
                tool_call_history=tool_call_history,
                tool_results_cache=cache,
                rag_chunks=rag_chunks,
                is_guest=is_guest,
                hop_count=hop_count,
                fallback=_NO_RESULT_FALLBACK,
                history_context=history_context,
            )
        except Exception as exc:  # noqa: BLE001 — 에러 전파 금지
            logger.warning(
                "narrator_chain_import_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                intent_kind=intent_kind,
            )
            text = _NO_RESULT_FALLBACK

        if not text:
            text = _NO_RESULT_FALLBACK

        # needs_human_agent: personal_data + 모든 tool 실패(login_required 포함) 시 True
        all_failed = all(
            not entry.get("ok", False) for entry in tool_call_history
        ) if tool_call_history else False
        needs_human = bool(all_failed and intent_kind == "personal_data" and is_guest)

        logger.info(
            "narrator_diagnostic_done",
            intent_kind=intent_kind,
            hop_count=hop_count,
            rag_count=len(rag_chunks),
            text_length=len(text),
            needs_human=needs_human,
        )
        return {
            "response_text": text,
            "needs_human_agent": needs_human,
        }

    # ── 기타 예상치 못한 kind ──
    logger.warning("narrator_unexpected_kind", intent_kind=intent_kind)
    return {
        "response_text": _NO_RESULT_FALLBACK,
        "needs_human_agent": True,
    }


def _build_faq_context(faq_items: list[dict]) -> str:
    """
    ES FAQ 검색 결과 dict 리스트를 단순 인용 답변용 텍스트 블록으로 변환한다.

    상위 3건만 사용 (Solar Pro max_tokens=2048 여유 확보).
    faq 의도의 Phase 1 로직을 그대로 유지한다.
    """
    lines: list[str] = []
    for i, item in enumerate(faq_items[:3], start=1):
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            lines.append(f"[FAQ {i}] Q: {q}\nA: {a}")
    return "\n\n".join(lines)


# Phase 1 에서 사용하던 단순 Solar 답변 생성 함수 — faq 경로에서 계속 사용.
# narrator 시스템 프롬프트는 Phase 1 용 경량 버전 유지 (faq 단순 인용 충분).
_NARRATOR_SIMPLE_SYSTEM_PROMPT = """\
당신은 영화 서비스 '몽글픽' 고객센터 AI 상담원 '몽글이'예요.
서비스 이름은 반드시 '몽글픽' 으로만 표기 (몽글/몽블랑 등 다른 이름 사용 금지).
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
"""


async def _generate_with_solar(
    user_message: str,
    context_block: str,
    fallback: str,
    history_context: str = "",
) -> str:
    """
    Solar Pro API 로 단순 FAQ 인용 답변을 생성한다 (faq 경로 전용).

    Phase 1 로직을 그대로 유지. 다중 hop 진단 답변은
    support_narrator_chain.generate_narrator_response() 에서 처리한다.

    Args:
        user_message: 현재 턴 사용자 발화
        context_block: ES FAQ 검색 결과 텍스트 블록
        fallback: LLM 실패 시 반환할 기본 텍스트
        history_context: 최근 멀티턴 대화 텍스트 블록 (2026-04-28 추가).
            비어 있으면 단일턴 응답.

    실패 시 fallback 반환 (에러 전파 금지).
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from monglepick.llm.factory import get_solar_api_llm

        llm = get_solar_api_llm(temperature=0.3)
        # 멀티턴 컨텍스트 prefix
        history_prefix = (
            f"[이전 대화]\n{history_context}\n\n" if history_context else ""
        )
        human_content = (
            f"{history_prefix}"
            f"[사용자 질문]\n{user_message}\n\n"
            f"{context_block}\n\n"
            "위 내용을 바탕으로 몽글이 톤으로 2~4문장으로 답변해 주세요. 본문만 출력하세요."
        )
        messages = [
            SystemMessage(content=_NARRATOR_SIMPLE_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]
        response = await llm.ainvoke(messages)
        text = (response.content or "").strip()
        # 서비스 이름 환각 후처리 — Solar 도 가끔 잘못된 표기 출력
        text = _normalize_service_name(text)
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
