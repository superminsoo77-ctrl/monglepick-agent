"""
support_assistant v3 (RDB 직접 조회 + Solar Pro 1회 호출) 단위 테스트.

모든 외부 의존 stub:
- `fetch_faqs` (Backend HTTP) → 고정 FaqDoc 리스트 반환
- `generate_support_reply` (Solar Pro structured output) → 시나리오별 SupportReply 반환

검증 범위:
- context_loader / support_agent / response_formatter 각 노드
- 그래프 E2E (5가지 kind 시나리오 + LLM 환각 방어)
- SSE matched_faq 이벤트가 kind ∈ {faq, partial} 에서만 발행되는지
"""

from __future__ import annotations

import json

import pytest

from monglepick.agents.support_assistant import nodes as support_nodes
from monglepick.agents.support_assistant.graph import (
    run_support_assistant,
    run_support_assistant_sync,
)
from monglepick.agents.support_assistant.models import (
    FaqDoc,
    SupportReply,
)


# =============================================================================
# 공통 픽스처 — FAQ 목록 / 기본 reply stub
# =============================================================================


@pytest.fixture
def sample_faqs() -> list[FaqDoc]:
    """테스트용 FAQ 3건 — 전화번호 / 비밀번호 / AI 추천."""
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


def _patch_fetch(monkeypatch, faqs: list[FaqDoc]) -> None:
    async def _fake_fetch():
        return list(faqs)

    monkeypatch.setattr(
        "monglepick.agents.support_assistant.nodes.fetch_faqs", _fake_fetch
    )


def _patch_reply(monkeypatch, reply: SupportReply) -> None:
    async def _fake_reply(user_message: str, faqs: list[FaqDoc]):
        return reply

    monkeypatch.setattr(
        "monglepick.agents.support_assistant.nodes.generate_support_reply",
        _fake_reply,
    )


# =============================================================================
# 1) 노드 단위 — context_loader / support_agent / response_formatter
# =============================================================================


@pytest.mark.asyncio
class TestNodes:
    async def test_context_loader_returns_faq_list(self, monkeypatch, sample_faqs):
        _patch_fetch(monkeypatch, sample_faqs)
        out = await support_nodes.context_loader(
            {"user_message": "안녕", "session_id": "s1"}
        )
        assert out["faqs"] == sample_faqs
        # 이전 턴 잔재 플래그가 초기화되어 있어야 한다
        assert out["matched_faqs"] == []
        assert out["needs_human_agent"] is False
        assert out["response_text"] == ""

    async def test_support_agent_maps_matched_ids_to_faqs(
        self, monkeypatch, sample_faqs
    ):
        """LLM 이 돌려준 matched_faq_ids 를 실제 FaqDoc 메타와 매핑."""
        reply = SupportReply(
            kind="faq",
            matched_faq_ids=[2],
            answer="비밀번호는 '비밀번호 찾기' 에서 이메일 인증으로 재설정하실 수 있어요.",
            needs_human=False,
        )
        _patch_reply(monkeypatch, reply)

        out = await support_nodes.support_agent(
            {"user_message": "비밀번호 바꿀래요", "faqs": sample_faqs}
        )
        assert out["reply"] == reply
        assert len(out["matched_faqs"]) == 1
        assert out["matched_faqs"][0].faq_id == 2
        assert out["matched_faqs"][0].question.startswith("비밀번호")
        assert out["needs_human_agent"] is False
        assert "비밀번호" in out["response_text"]

    async def test_support_agent_keeps_id_only_when_backend_fetch_misses(
        self, monkeypatch, sample_faqs
    ):
        """v3.3: state.faqs(Backend fetch) 에서 못 찾은 ID 여도 chain(ES)이 유효하다고 전한
        matched_faq_ids 가 있으면 **강등하지 않고 id-only MatchedFaq 로 유지** 한다.

        - v3.2: Backend fetch 결과에서 못 찾으면 complaint 로 강등(버그 — 환불/비밀번호
          스모크에서 '1:1 문의'로 수렴하는 원인)
        - v3.3: 강등은 reply.matched_faq_ids 가 비어있을 때만 (chain 책임)
        """
        reply = SupportReply(
            kind="faq",
            matched_faq_ids=[999],  # Backend fetch 결과에는 없음
            answer="자세한 내용은 FAQ 를 확인해 주세요.",
            needs_human=False,
        )
        _patch_reply(monkeypatch, reply)

        out = await support_nodes.support_agent(
            {"user_message": "가짜 질문", "faqs": sample_faqs}
        )
        # kind 유지 — chain 이 ES 로 이미 검증한 ID 이므로 신뢰
        assert out["reply"].kind == "faq"
        # id 는 살아있되 Backend fetch 미스로 category/question 은 빈 문자열
        assert len(out["matched_faqs"]) == 1
        assert out["matched_faqs"][0].faq_id == 999
        assert out["matched_faqs"][0].category == ""
        assert out["matched_faqs"][0].question == ""
        assert out["needs_human_agent"] is False

    async def test_support_agent_demotes_only_when_matched_ids_empty(
        self, monkeypatch, sample_faqs
    ):
        """faq/partial 인데 matched_faq_ids 자체가 비어있으면 complaint 로 강등."""
        reply = SupportReply(
            kind="faq",
            matched_faq_ids=[],
            answer="",
            needs_human=False,
        )
        _patch_reply(monkeypatch, reply)

        out = await support_nodes.support_agent(
            {"user_message": "애매한 질문", "faqs": sample_faqs}
        )
        assert out["reply"].kind == "complaint"
        assert out["matched_faqs"] == []
        assert out["needs_human_agent"] is True

    async def test_response_formatter_guards_empty_text(self, monkeypatch):
        """빈 본문이면 최후 fallback 메시지 + 상담원 배너 강제."""
        out = await support_nodes.response_formatter(
            {"response_text": "", "needs_human_agent": False, "reply": None}
        )
        assert out["response_text"]  # 빈 문자열이 아니어야 함
        assert out["needs_human_agent"] is True

    async def test_response_formatter_passthrough(self, monkeypatch):
        out = await support_nodes.response_formatter(
            {
                "response_text": "비밀번호는 이메일 인증으로 재설정해요.",
                "needs_human_agent": False,
                "reply": SupportReply(kind="faq", matched_faq_ids=[2], answer="x"),
            }
        )
        assert out["response_text"].startswith("비밀번호는")
        assert out["needs_human_agent"] is False


# =============================================================================
# 2) 그래프 E2E — 5가지 kind 시나리오
# =============================================================================


@pytest.mark.asyncio
class TestGraphEndToEnd:
    async def test_kind_faq_direct_answer(self, monkeypatch, sample_faqs):
        """FAQ 에 명확한 답이 있는 경우 → needs_human=False, 근거 FAQ 노출."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_reply(
            monkeypatch,
            SupportReply(
                kind="faq",
                matched_faq_ids=[2],
                answer="비밀번호는 '비밀번호 찾기' 에서 이메일 인증 후 재설정하실 수 있어요.",
                needs_human=False,
            ),
        )

        final = await run_support_assistant_sync(
            user_id="", session_id="", user_message="비밀번호 변경하고 싶어요"
        )
        assert final["needs_human_agent"] is False
        assert len(final["matched_faqs"]) == 1
        assert final["matched_faqs"][0].faq_id == 2
        assert "비밀번호" in final["response_text"]
        assert final["reply"].kind == "faq"

    async def test_kind_partial_shows_faq_but_human_too(
        self, monkeypatch, sample_faqs
    ):
        """관련 FAQ 가 있지만 정확한 답은 아님 → needs_human=True + 근거 노출."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_reply(
            monkeypatch,
            SupportReply(
                kind="partial",
                matched_faq_ids=[3],
                answer=(
                    "완전히 일치하는 안내는 아니지만 비슷한 내용이 있어요. "
                    "좋아요/싫어요 피드백을 주시면 개선돼요. "
                    "계속 같은 문제라면 '문의하기' 탭에 남겨주세요."
                ),
                needs_human=True,
            ),
        )

        final = await run_support_assistant_sync(
            user_id="", session_id="", user_message="AI 추천이 너무 비슷한 영화만 줘요"
        )
        assert final["needs_human_agent"] is True
        assert len(final["matched_faqs"]) == 1
        assert final["matched_faqs"][0].faq_id == 3
        assert "문의하기" in final["response_text"]

    async def test_kind_complaint_no_faq_reference(self, monkeypatch, sample_faqs):
        """불만/버그 신고 → matched_faqs 비어있음 + 상담원 유도."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_reply(
            monkeypatch,
            SupportReply(
                kind="complaint",
                matched_faq_ids=[],
                answer=(
                    "불편을 드려 죄송해요. 발생 시각·화면을 '문의하기' 탭에 남겨주시면 "
                    "담당자가 확인해 드릴게요."
                ),
                needs_human=True,
            ),
        )

        final = await run_support_assistant_sync(
            user_id="", session_id="", user_message="결제하다가 에러 떴어요 긴급"
        )
        assert final["needs_human_agent"] is True
        assert final["matched_faqs"] == []
        assert "죄송" in final["response_text"]

    async def test_kind_out_of_scope_redirects_to_chat_tab(
        self, monkeypatch, sample_faqs
    ):
        """영화 추천 요청 등 고객센터 범위 밖 → needs_human=False (AI 채팅 탭 안내)."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_reply(
            monkeypatch,
            SupportReply(
                kind="out_of_scope",
                matched_faq_ids=[],
                answer=(
                    "이 부분은 고객센터에서 도와드리기 어려운 내용이에요. "
                    "영화 추천은 'AI 채팅' 탭에서 도와드리고 있어요."
                ),
                needs_human=False,
            ),
        )

        final = await run_support_assistant_sync(
            user_id="", session_id="", user_message="봉준호 감독 영화 추천해줘"
        )
        assert final["needs_human_agent"] is False
        assert final["matched_faqs"] == []
        assert "AI 채팅" in final["response_text"]

    async def test_kind_smalltalk_keeps_short_answer(self, monkeypatch, sample_faqs):
        """인사에는 짧은 응대 + needs_human=False."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_reply(
            monkeypatch,
            SupportReply(
                kind="smalltalk",
                matched_faq_ids=[],
                answer="안녕하세요! 궁금한 점 있으시면 편하게 물어봐 주세요.",
                needs_human=False,
            ),
        )

        final = await run_support_assistant_sync(
            user_id="", session_id="", user_message="안녕?"
        )
        assert final["needs_human_agent"] is False
        assert final["matched_faqs"] == []
        assert "안녕하세요" in final["response_text"]


# =============================================================================
# 3) SSE 스트림 — matched_faq 이벤트 발행 조건
# =============================================================================


@pytest.mark.asyncio
class TestSseStream:
    async def _collect_events(self, gen) -> list[dict]:
        events: list[dict] = []
        async for raw in gen:
            # sse_starlette 호환 dict 포맷: {"event": "...", "data": "json_str"}
            events.append(
                {"event": raw["event"], "data": json.loads(raw["data"])}
            )
        return events

    async def test_faq_kind_emits_matched_faq_event(
        self, monkeypatch, sample_faqs
    ):
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_reply(
            monkeypatch,
            SupportReply(
                kind="faq",
                matched_faq_ids=[1],
                answer="이메일은 contact@monglepick.com 이에요.",
                needs_human=False,
            ),
        )

        events = await self._collect_events(
            run_support_assistant(
                user_id="", session_id="", user_message="이메일 알려주세요"
            )
        )
        event_types = [e["event"] for e in events]
        assert "session" in event_types
        assert "matched_faq" in event_types  # faq kind 에서만 발행
        assert "token" in event_types
        assert "needs_human" in event_types
        assert event_types[-1] == "done"

        # matched_faq 페이로드 검증
        matched_event = next(e for e in events if e["event"] == "matched_faq")
        assert matched_event["data"]["items"][0]["faq_id"] == 1
        # v3: score 필드는 이제 전송되지 않음
        assert "score" not in matched_event["data"]["items"][0]

    async def test_complaint_kind_skips_matched_faq_event(
        self, monkeypatch, sample_faqs
    ):
        """불만 케이스: matched_faqs 비어있어 matched_faq 이벤트는 발행하지 않는다."""
        _patch_fetch(monkeypatch, sample_faqs)
        _patch_reply(
            monkeypatch,
            SupportReply(
                kind="complaint",
                matched_faq_ids=[],
                answer="불편을 드려 죄송해요. '문의하기' 탭에 남겨주세요.",
                needs_human=True,
            ),
        )

        events = await self._collect_events(
            run_support_assistant(
                user_id="", session_id="", user_message="계속 에러 나요"
            )
        )
        event_types = [e["event"] for e in events]
        assert "matched_faq" not in event_types
        # needs_human=true 가 실려 나가는지 확인
        needs_human = next(e for e in events if e["event"] == "needs_human")
        assert needs_human["data"]["value"] is True
