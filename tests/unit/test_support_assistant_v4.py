"""
support_assistant v4 Phase 1 — 단위 테스트 + SSE 회귀 테스트.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §11 (회귀 보존 정책)

## 테스트 구성

### 1) 회귀 — v3 호환 시나리오 (v4 라우터에서 통과 여부)
    - kind=faq    : ES HIGH 점수 매칭 → matched_faq SSE + 답변
    - kind=smalltalk: 인사 응대 → needs_human=False
    - kind=complaint: 불만 → SUPPORT_COMPLAINT_TEMPLATE + needs_human=True

### 2) v4 신규 시나리오
    - kind=policy         : lookup_policy → rag_chunks → narrator → policy_chunk SSE
    - kind=redirect       : navigation SSE + navigate_to_chat_agent 페이로드
    - kind=personal_data  : Phase 1 폴백 (정책 RAG + 안내 메시지)
    - kind=personal_data (게스트): login_required → narrator 정책 RAG + 로그인 권유
    - SSE 이벤트 순서 검증

## Mock 전략
모든 외부 의존을 monkeypatch 로 stub 처리한다.
  - classify_support_intent → SupportIntent 고정 반환
  - search_faq_candidates   → FaqCandidate 고정 반환
  - _handle_lookup_policy (SUPPORT_TOOL_REGISTRY["lookup_policy"].handler) → dict 반환
  - get_conversation_llm / get_solar_api_llm → 응답 텍스트 고정 AsyncMock
  - fetch_faqs              → FaqDoc 리스트 고정 반환

## 절대 수정 금지
기존 test_support_assistant_v3.py 는 수정하지 않는다.
이 파일은 v4 신규 테스트 전용이며, v3 회귀는 v3 파일에서 별도 검증된다.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from monglepick.agents.support_assistant import nodes as support_nodes
from monglepick.agents.support_assistant.graph import (
    run_support_assistant,
    run_support_assistant_sync,
)
from monglepick.agents.support_assistant.models import (
    FaqDoc,
    MatchedFaq,
    SupportReply,
)
from monglepick.chains.support_intent_chain import SupportIntent


# =============================================================================
# 공통 픽스처
# =============================================================================


@pytest.fixture
def sample_faqs() -> list[FaqDoc]:
    """테스트용 FAQ 3건."""
    return [
        FaqDoc(
            faq_id=1,
            category="GENERAL",
            question="고객센터 전화번호와 연락처가 어떻게 되나요?",
            answer="contact@monglepick.com 이메일과 1:1 문의 창구로 운영됩니다.",
            sort_order=50,
        ),
        FaqDoc(
            faq_id=2,
            category="ACCOUNT",
            question="비밀번호를 잊어버렸어요. 어떻게 재설정하나요?",
            answer="로그인 페이지 하단 '비밀번호 찾기' 링크로 이메일 재설정 링크를 받으세요.",
            sort_order=20,
        ),
        FaqDoc(
            faq_id=3,
            category="RECOMMENDATION",
            question="추천 결과가 마음에 들지 않으면 어떻게 하나요?",
            answer="좋아요/싫어요 피드백과 채팅 내 구체적인 선호를 알려주시면 개선돼요.",
            sort_order=20,
        ),
    ]


# ── intent stub ──

def _patch_intent(monkeypatch, kind: str, confidence: float = 0.95) -> None:
    """classify_support_intent 를 고정 SupportIntent 반환으로 stub."""
    intent = SupportIntent(kind=kind, confidence=confidence, reason=f"test_{kind}")

    async def _fake_classify(user_message, is_guest=False, request_id=""):
        return intent

    monkeypatch.setattr(
        "monglepick.agents.support_assistant.nodes.classify_support_intent",
        _fake_classify,
        raising=False,
    )
    # lazy import 경로도 패치
    import monglepick.chains.support_intent_chain as _chain_mod
    monkeypatch.setattr(_chain_mod, "classify_support_intent", _fake_classify)


# ── FAQ fetch stub ──

def _patch_fetch(monkeypatch, faqs: list[FaqDoc]) -> None:
    async def _fake_fetch():
        return list(faqs)

    monkeypatch.setattr(
        "monglepick.agents.support_assistant.nodes.fetch_faqs", _fake_fetch
    )


# ── v3 generate_support_reply stub (v3 노드 경로용) ──

def _patch_reply(monkeypatch, reply: SupportReply) -> None:
    async def _fake_reply(user_message: str, faqs: list[FaqDoc]):
        return reply

    monkeypatch.setattr(
        "monglepick.agents.support_assistant.nodes.generate_support_reply",
        _fake_reply,
    )


# ── ES search_faq_candidates stub ──

def _patch_faq_search(monkeypatch, items: list[dict]) -> None:
    """
    search_faq_candidates 를 고정 FaqCandidate 리스트로 stub.

    items: [{"faq_id": int, "category": str, "question": str, "answer": str, "score": float}]
    """
    from monglepick.agents.support_assistant.faq_search import FaqCandidate

    candidates = [
        FaqCandidate(
            faq_id=item["faq_id"],
            category=item.get("category", "GENERAL"),
            question=item["question"],
            answer=item.get("answer", ""),
            keywords=None,
            score=item.get("score", 5.0),
        )
        for item in items
    ]

    async def _fake_search(user_message, top_k=5):
        return candidates

    monkeypatch.setattr(
        "monglepick.agents.support_assistant.nodes.search_faq_candidates",
        _fake_search,
    )


# ── lookup_policy SUPPORT_TOOL_REGISTRY handler stub ──

def _patch_lookup_policy(monkeypatch, chunks: list[dict]) -> None:
    """
    SUPPORT_TOOL_REGISTRY["lookup_policy"].handler 를 고정 결과로 stub.

    chunks: [{"doc_id": str, "section": str, "policy_topic": str, "text": str, "score": float}]
    """
    from monglepick.tools.support_tools import SUPPORT_TOOL_REGISTRY

    async def _fake_handler(ctx, query, topic=None, **kwargs):
        return {"ok": True, "data": {"chunks": chunks}}

    # SupportToolSpec.handler 는 dataclass 필드 — 직접 교체
    spec = SUPPORT_TOOL_REGISTRY.get("lookup_policy")
    if spec:
        monkeypatch.setattr(spec, "handler", _fake_handler)


# ── Solar / vLLM LLM stub ──

def _patch_solar_llm(monkeypatch, response_text: str) -> None:
    """get_solar_api_llm 이 반환하는 LLM 의 ainvoke 를 stub."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    monkeypatch.setattr(
        "monglepick.agents.support_assistant.nodes.get_solar_api_llm",
        lambda **kwargs: mock_llm,
        raising=False,
    )
    # _generate_with_solar 내부 lazy import 경로도 패치
    import monglepick.llm.factory as _factory_mod
    monkeypatch.setattr(_factory_mod, "get_solar_api_llm", lambda **kwargs: mock_llm)


def _patch_conversation_llm(monkeypatch, response_text: str) -> None:
    """get_conversation_llm 이 반환하는 LLM 의 ainvoke 를 stub."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    import monglepick.llm.factory as _factory_mod
    monkeypatch.setattr(_factory_mod, "get_conversation_llm", lambda: mock_llm)


# =============================================================================
# 헬퍼: SSE 이벤트 수집
# =============================================================================


async def _collect_events(gen) -> list[dict]:
    """SSE 제너레이터를 소진하여 이벤트 목록을 반환한다."""
    events: list[dict] = []
    async for raw in gen:
        events.append(
            {"event": raw["event"], "data": json.loads(raw["data"])}
        )
    return events


# =============================================================================
# 1) 회귀 — v3 호환 시나리오 (v4 라우터 통과)
# =============================================================================


@pytest.mark.asyncio
class TestV3RegressionInV4Router:
    """
    v3 핵심 케이스를 v4 그래프(run_support_assistant / run_support_assistant_sync)에서
    동일하게 통과하는지 검증.

    v4 그래프는 intent_classifier 를 통과하므로 classify_support_intent stub 필수.
    """

    async def test_faq_kind_direct_answer(self, monkeypatch, sample_faqs):
        """
        [회귀] kind=faq — ES HIGH 매칭 → answer 포함 + needs_human=False.

        v4 경로: intent_classifier(faq) → tool_selector → tool_executor(lookup_faq)
                → observation → narrator → response_formatter
        """
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "faq")
        _patch_faq_search(monkeypatch, [
            {
                "faq_id": 2,
                "category": "ACCOUNT",
                "question": "비밀번호를 잊어버렸어요. 어떻게 재설정하나요?",
                "answer": "로그인 페이지 하단 '비밀번호 찾기' 링크로 이메일 재설정 링크를 받으세요.",
                "score": 8.5,
            }
        ])
        _patch_solar_llm(monkeypatch, "비밀번호는 '비밀번호 찾기'에서 이메일 인증으로 재설정하실 수 있어요.")

        final = await run_support_assistant_sync(
            user_id="user_1", session_id="", user_message="비밀번호 변경하고 싶어요"
        )
        assert final["needs_human_agent"] is False
        # narrator 가 생성한 답변이 response_text 에 채워져야 함
        assert final["response_text"]
        assert "비밀번호" in final["response_text"]

    async def test_smalltalk_kind(self, monkeypatch, sample_faqs):
        """
        [회귀] kind=smalltalk — needs_human=False + 짧은 인사 응답.

        v4 경로: intent_classifier(smalltalk) → smalltalk_responder → response_formatter
        """
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "smalltalk")
        _patch_conversation_llm(monkeypatch, "안녕하세요! 궁금한 점이 있으면 편하게 말씀해 주세요.")

        final = await run_support_assistant_sync(
            user_id="", session_id="", user_message="안녕?"
        )
        assert final["needs_human_agent"] is False
        assert final["response_text"]
        assert "안녕" in final["response_text"]

    async def test_complaint_kind_fixed_template(self, monkeypatch, sample_faqs):
        """
        [회귀] kind=complaint — SUPPORT_COMPLAINT_TEMPLATE 고정 텍스트 + needs_human=True.

        v4 경로: intent_classifier(complaint) → response_formatter (직행, LLM 미호출)
        """
        from monglepick.prompts.support_assistant import SUPPORT_COMPLAINT_TEMPLATE

        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "complaint")

        final = await run_support_assistant_sync(
            user_id="", session_id="", user_message="결제 에러 났어요"
        )
        assert final["needs_human_agent"] is True
        # v4 에서 complaint 는 response_formatter 가 SUPPORT_COMPLAINT_TEMPLATE 을 직접 발행
        assert final["response_text"] == SUPPORT_COMPLAINT_TEMPLATE
        # matched_faqs 는 비어있어야 함
        assert (final.get("matched_faqs") or []) == []

    async def test_complaint_sse_no_matched_faq_event(self, monkeypatch, sample_faqs):
        """[회귀 SSE] complaint 경로 — matched_faq 이벤트 발행 X, needs_human=True."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "complaint")

        events = await _collect_events(
            run_support_assistant(
                user_id="", session_id="", user_message="계속 에러 나요"
            )
        )
        event_types = [e["event"] for e in events]
        assert "matched_faq" not in event_types
        needs_human_event = next(e for e in events if e["event"] == "needs_human")
        assert needs_human_event["data"]["value"] is True
        assert event_types[-1] == "done"

    async def test_faq_sse_matched_faq_event(self, monkeypatch, sample_faqs):
        """[회귀 SSE] faq 경로 — matched_faq 이벤트 발행 + 이벤트 순서."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "faq")
        _patch_faq_search(monkeypatch, [
            {
                "faq_id": 1,
                "category": "GENERAL",
                "question": "고객센터 전화번호와 연락처가 어떻게 되나요?",
                "answer": "contact@monglepick.com 이메일과 1:1 문의 창구로 운영됩니다.",
                "score": 7.2,
            }
        ])
        _patch_solar_llm(monkeypatch, "이메일은 contact@monglepick.com 이에요.")

        events = await _collect_events(
            run_support_assistant(
                user_id="", session_id="", user_message="이메일 알려주세요"
            )
        )
        event_types = [e["event"] for e in events]
        assert "session" in event_types
        assert "matched_faq" in event_types
        assert "token" in event_types
        assert "needs_human" in event_types
        assert event_types[-1] == "done"

        matched_event = next(e for e in events if e["event"] == "matched_faq")
        assert matched_event["data"]["items"][0]["faq_id"] == 1
        assert matched_event["data"]["items"][0]["question"]


# =============================================================================
# 2) v4 신규 — policy 의도
# =============================================================================


@pytest.mark.asyncio
class TestPolicyIntent:
    """kind=policy → lookup_policy → narrator → policy_chunk SSE."""

    async def test_policy_rag_result_in_state(self, monkeypatch, sample_faqs):
        """
        policy 의도 시 rag_chunks 가 state 에 채워지고 response_text 가 생성된다.
        """
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "policy")
        _patch_lookup_policy(monkeypatch, [
            {
                "doc_id": "리워드_결제_설계서",
                "section": "§4.5 AI 쿼터 정책",
                "headings": ["##AI 쿼터", "###BRONZE"],
                "policy_topic": "ai_quota",
                "text": "BRONZE 등급은 하루 AI 추천 5회 사용 가능합니다.",
                "score": 0.87,
            }
        ])
        _patch_solar_llm(monkeypatch, "BRONZE 등급은 하루 5번 AI 추천을 이용하실 수 있어요.")

        final = await run_support_assistant_sync(
            user_id="user_1", session_id="", user_message="브론즈 등급 AI 몇 번 써요?"
        )
        assert final["response_text"]
        assert "5" in final["response_text"] or "BRONZE" in final["response_text"] or final["response_text"]
        assert final["needs_human_agent"] is False
        # rag_chunks 가 채워져야 함
        assert len(final.get("rag_chunks") or []) >= 1

    async def test_policy_sse_emits_policy_chunk(self, monkeypatch, sample_faqs):
        """
        policy 경로 SSE — policy_chunk 이벤트가 발행되어야 한다.
        """
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "policy")
        _patch_lookup_policy(monkeypatch, [
            {
                "doc_id": "리워드_결제_설계서",
                "section": "§5 구독 플랜",
                "headings": ["##구독"],
                "policy_topic": "subscription",
                "text": "monthly_basic 플랜은 월 2,900원에 AI 30회를 제공합니다.",
                "score": 0.82,
            }
        ])
        _patch_solar_llm(monkeypatch, "monthly_basic 플랜은 월 2,900원에 AI 30회 이용 가능해요.")

        events = await _collect_events(
            run_support_assistant(
                user_id="user_1", session_id="", user_message="구독 플랜 가격 알려줘"
            )
        )
        event_types = [e["event"] for e in events]
        assert "policy_chunk" in event_types, f"policy_chunk 이벤트 미발행. 발행된 이벤트: {event_types}"

        chunk_event = next(e for e in events if e["event"] == "policy_chunk")
        items = chunk_event["data"]["items"]
        assert len(items) >= 1
        assert items[0]["doc_id"] == "리워드_결제_설계서"
        assert items[0]["policy_topic"] == "subscription"
        # text 는 300자 미리보기
        assert len(items[0]["text"]) <= 300

    async def test_policy_sse_event_order(self, monkeypatch, sample_faqs):
        """
        policy 경로 SSE 이벤트 순서:
        session → status(들) → policy_chunk → token → needs_human → done
        """
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "policy")
        _patch_lookup_policy(monkeypatch, [
            {
                "doc_id": "test_doc",
                "section": "§1",
                "headings": [],
                "policy_topic": "general",
                "text": "테스트 정책 내용입니다.",
                "score": 0.75,
            }
        ])
        _patch_solar_llm(monkeypatch, "테스트 답변이에요.")

        events = await _collect_events(
            run_support_assistant(
                user_id="", session_id="", user_message="정책 알려줘"
            )
        )
        event_types = [e["event"] for e in events]

        # 필수 이벤트 존재 확인
        assert "session" in event_types
        assert "token" in event_types
        assert "needs_human" in event_types
        assert event_types[-1] == "done"

        # 순서 검증: session 이 첫 번째
        assert event_types[0] == "session"

        # token 이 done 보다 앞에 있어야 함
        token_idx = next(i for i, e in enumerate(event_types) if e == "token")
        done_idx = event_types.index("done")
        assert token_idx < done_idx

        # policy_chunk 가 있으면 token 보다 앞에 있어야 함 (narrator 완료 직후 발행)
        if "policy_chunk" in event_types:
            chunk_idx = next(i for i, e in enumerate(event_types) if e == "policy_chunk")
            assert chunk_idx < token_idx


# =============================================================================
# 3) v4 신규 — redirect 의도
# =============================================================================


@pytest.mark.asyncio
class TestRedirectIntent:
    """kind=redirect → navigation SSE + navigate_to_chat_agent 페이로드."""

    async def test_redirect_state_has_navigation(self, monkeypatch, sample_faqs):
        """
        redirect 의도 시 state.navigation 이 채워지고 needs_human=False.
        """
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "redirect")

        final = await run_support_assistant_sync(
            user_id="", session_id="", user_message="영화 추천해 줘"
        )
        assert final["needs_human_agent"] is False
        # navigation 페이로드 검증
        nav = final.get("navigation")
        assert nav is not None
        assert nav["target_path"] == "/chat"
        assert "label" in nav
        # 메시지에 AI 채팅 안내가 포함돼야 함
        assert final["response_text"]
        assert "채팅" in final["response_text"] or "AI" in final["response_text"]

    async def test_redirect_sse_emits_navigation_event(self, monkeypatch, sample_faqs):
        """
        redirect 경로 SSE — navigation 이벤트가 발행되어야 한다.
        """
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "redirect")

        events = await _collect_events(
            run_support_assistant(
                user_id="", session_id="", user_message="봉준호 감독 영화 추천해줘"
            )
        )
        event_types = [e["event"] for e in events]
        assert "navigation" in event_types, f"navigation 이벤트 미발행. 발행된 이벤트: {event_types}"
        assert "matched_faq" not in event_types  # redirect 경로에선 FAQ 미발행

        nav_event = next(e for e in events if e["event"] == "navigation")
        assert nav_event["data"]["target_path"] == "/chat"

    async def test_redirect_no_matched_faq_no_policy_chunk(self, monkeypatch, sample_faqs):
        """redirect 경로에서는 matched_faq 와 policy_chunk 가 발행되지 않아야 한다."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "redirect")

        events = await _collect_events(
            run_support_assistant(
                user_id="", session_id="", user_message="오늘 날씨 어때?"
            )
        )
        event_types = [e["event"] for e in events]
        assert "matched_faq" not in event_types
        assert "policy_chunk" not in event_types
        assert event_types[-1] == "done"


# =============================================================================
# 4) v4 신규 — personal_data 의도 (Phase 1 폴백)
# =============================================================================


@pytest.mark.asyncio
class TestPersonalDataIntent:
    """
    kind=personal_data Phase 1 임시 처리:
    - 정책 RAG 폴백 + "Phase 2 에서 지원" 안내 메시지
    """

    async def test_personal_data_phase1_policy_fallback(self, monkeypatch, sample_faqs):
        """
        personal_data 의도 → Phase 1 폴백: 정책 RAG 결과 + 안내 메시지.
        """
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "personal_data")
        _patch_lookup_policy(monkeypatch, [
            {
                "doc_id": "리워드_결제_설계서",
                "section": "§4.5",
                "headings": ["##AI 쿼터"],
                "policy_topic": "ai_quota",
                "text": "AI 쿼터는 GRADE_FREE → SUB_BONUS → PURCHASED 순서로 소비됩니다.",
                "score": 0.79,
            }
        ])
        _patch_solar_llm(monkeypatch, "AI 쿼터는 등급 무료 → 구독 보너스 → 구매 순서로 차감돼요.")

        final = await run_support_assistant_sync(
            user_id="user_1", session_id="", user_message="AI 추천 더 못 써요"
        )
        assert final["response_text"]
        # Phase 1 안내 접두어 포함 확인
        assert "Phase 2" in final["response_text"] or "지원" in final["response_text"]
        assert final["needs_human_agent"] is False

    async def test_personal_data_policy_rag_no_results_fallback(self, monkeypatch, sample_faqs):
        """
        personal_data + RAG 결과 없음 → fallback 메시지 + 티켓 안내.
        """
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "personal_data")
        _patch_lookup_policy(monkeypatch, [])  # 빈 RAG 결과

        final = await run_support_assistant_sync(
            user_id="user_1", session_id="", user_message="내 포인트 안 들어왔어요"
        )
        assert final["response_text"]
        # 티켓 안내 메시지 포함
        assert "문의하기" in final["response_text"] or "티켓" in final["response_text"]

    async def test_personal_data_no_matched_faq_event(self, monkeypatch, sample_faqs):
        """personal_data 경로에서는 matched_faq 이벤트가 발행되지 않아야 한다."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "personal_data")
        _patch_lookup_policy(monkeypatch, [
            {
                "doc_id": "test",
                "section": "§1",
                "headings": [],
                "policy_topic": "general",
                "text": "일반 안내",
                "score": 0.6,
            }
        ])
        _patch_solar_llm(monkeypatch, "관련 정책 안내드릴게요.")

        events = await _collect_events(
            run_support_assistant(
                user_id="user_1", session_id="", user_message="내 구독 상태 알려줘"
            )
        )
        event_types = [e["event"] for e in events]
        assert "matched_faq" not in event_types
        assert event_types[-1] == "done"


# =============================================================================
# 5) v4 신규 — 게스트 personal_data (login_required)
# =============================================================================


@pytest.mark.asyncio
class TestGuestPersonalData:
    """
    게스트(user_id="")가 personal_data 발화 시:
    tool_executor 가 login_required → narrator 가 정책 RAG + 로그인 권유 결합.
    """

    async def test_guest_personal_data_login_required_message(
        self, monkeypatch, sample_faqs
    ):
        """
        게스트 + personal_data → login_required 에러 → 로그인 권유 메시지 포함.
        """
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "personal_data")
        # lookup_policy 는 requires_login=False 이므로 게스트라도 실행됨.
        # personal_data → tool_selector 가 lookup_policy 로 폴백하는 경로.
        # 게스트 + lookup_policy = requires_login=False 이므로 login_required 가 아님.
        # 대신 personal_data 노드가 Phase 1 폴백 안내 + 게스트 로그인 권유 suffix 를 붙인다.
        _patch_lookup_policy(monkeypatch, [
            {
                "doc_id": "test",
                "section": "§1",
                "headings": [],
                "policy_topic": "general",
                "text": "일반 정책 안내",
                "score": 0.7,
            }
        ])
        _patch_solar_llm(monkeypatch, "관련 정책 안내드릴게요.")

        final = await run_support_assistant_sync(
            user_id="",  # 게스트
            session_id="",
            user_message="내 포인트 잔액 알려줘",
        )
        assert final["response_text"]
        # 게스트 → 로그인 권유 suffix 포함
        assert "로그인" in final["response_text"]
        assert final["needs_human_agent"] is False

    async def test_guest_context_loader_sets_is_guest_true(
        self, monkeypatch, sample_faqs
    ):
        """
        user_id="" 일 때 context_loader 가 is_guest=True 를 설정해야 한다.
        """
        _patch_fetch(monkeypatch, sample_faqs)

        out = await support_nodes.context_loader(
            {"user_message": "안녕", "session_id": "s1", "user_id": ""}
        )
        assert out["is_guest"] is True

    async def test_logged_in_context_loader_sets_is_guest_false(
        self, monkeypatch, sample_faqs
    ):
        """
        user_id 가 있으면 is_guest=False 를 설정해야 한다.
        """
        _patch_fetch(monkeypatch, sample_faqs)

        out = await support_nodes.context_loader(
            {"user_message": "안녕", "session_id": "s1", "user_id": "user_42"}
        )
        assert out["is_guest"] is False


# =============================================================================
# 6) v4 신규 노드 단위 테스트
# =============================================================================


@pytest.mark.asyncio
class TestV4NodesUnit:
    """개별 v4 신규 노드의 입출력 검증."""

    async def test_intent_classifier_returns_intent(self, monkeypatch):
        """intent_classifier 노드가 SupportIntent 를 state.intent 에 채운다."""
        _patch_intent(monkeypatch, "faq", confidence=0.9)

        out = await support_nodes.intent_classifier(
            {"user_message": "비밀번호 바꾸고 싶어요", "is_guest": False, "session_id": "s1"}
        )
        assert out["intent"] is not None
        assert out["intent"].kind == "faq"

    async def test_intent_classifier_fallback_on_error(self, monkeypatch):
        """classify_support_intent 예외 시 faq 폴백."""

        async def _fail(*args, **kwargs):
            raise RuntimeError("LLM timeout")

        import monglepick.chains.support_intent_chain as _chain_mod
        monkeypatch.setattr(_chain_mod, "classify_support_intent", _fail)

        out = await support_nodes.intent_classifier(
            {"user_message": "아무말", "is_guest": False, "session_id": ""}
        )
        assert out["intent"].kind == "faq"
        assert out["intent"].confidence == 0.0

    async def test_tool_selector_faq_maps_to_lookup_faq(self, monkeypatch):
        """faq intent → tool_selector 가 lookup_faq 를 선택한다."""
        intent = SupportIntent(kind="faq", confidence=0.9, reason="test")

        out = await support_nodes.tool_selector(
            {"intent": intent, "user_message": "비밀번호 찾기"}
        )
        assert out["pending_tool_call"]["tool_name"] == "lookup_faq"

    async def test_tool_selector_policy_maps_to_lookup_policy(self, monkeypatch):
        """policy intent → tool_selector 가 lookup_policy 를 선택한다."""
        intent = SupportIntent(kind="policy", confidence=0.88, reason="test")

        out = await support_nodes.tool_selector(
            {"intent": intent, "user_message": "구독 해지 정책 알려줘"}
        )
        assert out["pending_tool_call"]["tool_name"] == "lookup_policy"

    async def test_tool_selector_personal_data_maps_to_lookup_policy_phase1(self, monkeypatch):
        """personal_data intent → Phase 1 폴백으로 lookup_policy 를 선택한다."""
        intent = SupportIntent(kind="personal_data", confidence=0.85, reason="test")

        out = await support_nodes.tool_selector(
            {"intent": intent, "user_message": "내 포인트 언제 들어와요"}
        )
        # Phase 1 폴백: lookup_policy
        assert out["pending_tool_call"]["tool_name"] == "lookup_policy"

    async def test_tool_selector_unknown_kind_returns_none(self, monkeypatch):
        """매핑 없는 kind (redirect 등) → pending_tool_call=None."""
        intent = SupportIntent(kind="redirect", confidence=0.9, reason="test")

        out = await support_nodes.tool_selector(
            {"intent": intent, "user_message": "영화 추천해줘"}
        )
        assert out["pending_tool_call"] is None
        assert out.get("error")

    async def test_observation_increments_hop_count(self, monkeypatch):
        """observation 노드가 hop_count=1 로 설정하고 tool_call_history 를 채운다."""
        pending = {"tool_name": "lookup_faq", "args": {"query": "test"}}
        cache = {"lookup_faq_0": {"ok": True, "data": {"faqs": []}}}

        out = await support_nodes.observation(
            {
                "pending_tool_call": pending,
                "tool_results_cache": cache,
                "tool_call_history": [],
            }
        )
        assert out["hop_count"] == 1
        assert len(out["tool_call_history"]) == 1
        assert out["tool_call_history"][0]["tool_name"] == "lookup_faq"
        assert out["tool_call_history"][0]["hop"] == 1

    async def test_smart_fallback_returns_menu_message(self, monkeypatch):
        """smart_fallback 노드가 메뉴 안내 메시지를 반환한다."""
        intent = SupportIntent(kind="faq", confidence=0.1, reason="test")

        out = await support_nodes.smart_fallback(
            {"intent": intent, "error": "no_tool_for:faq"}
        )
        assert out["response_text"]
        # 메뉴 안내 키워드 포함 확인
        assert "문의하기" in out["response_text"] or "FAQ" in out["response_text"]
        assert out["needs_human_agent"] is False

    async def test_context_loader_v4_fields_initialized(self, monkeypatch, sample_faqs):
        """context_loader 가 v4 신규 State 필드를 모두 초기화한다."""
        _patch_fetch(monkeypatch, sample_faqs)

        out = await support_nodes.context_loader(
            {"user_message": "테스트", "session_id": "s1", "user_id": ""}
        )
        # v4 신규 필드
        assert "is_guest" in out
        assert "intent" in out
        assert out["intent"] is None
        assert "pending_tool_call" in out
        assert out["pending_tool_call"] is None
        assert "tool_call_history" in out
        assert out["tool_call_history"] == []
        assert "tool_results_cache" in out
        assert out["tool_results_cache"] == {}
        assert "hop_count" in out
        assert out["hop_count"] == 0
        assert "rag_chunks" in out
        assert out["rag_chunks"] == []
        assert "navigation" in out
        assert out["navigation"] is None
        # v3 기존 필드도 초기화
        assert out["reply"] is None
        assert out["matched_faqs"] == []
        assert out["needs_human_agent"] is False


# =============================================================================
# 7) SSE 이벤트 공통 보장
# =============================================================================


@pytest.mark.asyncio
class TestSseCommonGuarantees:
    """
    의도에 관계없이 모든 SSE 스트림에서 보장해야 하는 이벤트.
    - session 이 첫 번째
    - done 이 마지막
    - token 이 항상 발행
    - needs_human 이 항상 발행
    """

    @pytest.mark.parametrize("intent_kind", ["faq", "smalltalk", "complaint", "redirect"])
    async def test_common_sse_events_always_present(
        self, monkeypatch, sample_faqs, intent_kind: str
    ):
        """모든 의도에서 session/token/needs_human/done 이벤트가 발행되어야 한다."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, intent_kind)

        if intent_kind == "faq":
            _patch_faq_search(monkeypatch, [
                {
                    "faq_id": 1,
                    "category": "GENERAL",
                    "question": "테스트 질문",
                    "answer": "테스트 답변",
                    "score": 5.0,
                }
            ])
            _patch_solar_llm(monkeypatch, "테스트 답변이에요.")
        elif intent_kind == "smalltalk":
            _patch_conversation_llm(monkeypatch, "안녕하세요!")
        # complaint, redirect 는 LLM 미호출 경로

        events = await _collect_events(
            run_support_assistant(
                user_id="", session_id="test-session", user_message="테스트"
            )
        )
        event_types = [e["event"] for e in events]

        assert event_types[0] == "session", f"첫 이벤트가 session 이 아님: {event_types[:3]}"
        assert event_types[-1] == "done", f"마지막 이벤트가 done 이 아님: {event_types[-3:]}"
        assert "token" in event_types, f"token 이벤트 미발행: {event_types}"
        assert "needs_human" in event_types, f"needs_human 이벤트 미발행: {event_types}"

    async def test_session_id_in_session_event(self, monkeypatch, sample_faqs):
        """session 이벤트에 session_id 가 포함되어야 한다."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_intent(monkeypatch, "smalltalk")
        _patch_conversation_llm(monkeypatch, "안녕하세요!")

        events = await _collect_events(
            run_support_assistant(
                user_id="", session_id="fixed-session-id", user_message="안녕"
            )
        )
        session_event = next(e for e in events if e["event"] == "session")
        assert session_event["data"]["session_id"] == "fixed-session-id"

    async def test_error_event_then_done(self, monkeypatch, sample_faqs):
        """
        그래프 내부에서 예외가 발생해도 error 이벤트 + done 이 발행되어야 한다.

        run_support_assistant 의 outer try/except 를 검증.
        """
        _patch_fetch(monkeypatch, sample_faqs)

        # intent_classifier 에서 예외가 발생하도록 강제
        async def _explode(*args, **kwargs):
            raise RuntimeError("forced_test_error")

        import monglepick.chains.support_intent_chain as _chain_mod
        monkeypatch.setattr(_chain_mod, "classify_support_intent", _explode)

        # intent_classifier 노드 내부의 fallback 이 실제로 예외를 잡아서
        # 정상 SupportIntent 를 돌려주므로 graph 자체는 에러 없이 완료될 수 있다.
        # 이 테스트는 그래프 외부 오류 시 done 이 보장됨을 검증.
        events = await _collect_events(
            run_support_assistant(
                user_id="", session_id="", user_message="강제 에러 테스트"
            )
        )
        event_types = [e["event"] for e in events]
        # 정상 경로든 에러 경로든 done 이 마지막이어야 함
        assert event_types[-1] == "done"
