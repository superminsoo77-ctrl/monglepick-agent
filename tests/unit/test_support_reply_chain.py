"""
support_reply_chain v3.3 단위 테스트.

### v3.3 아키텍처 (3단계 파이프라인)
[1단계] ES Nori BM25 검색 (search_faq_candidates)
[2단계] 점수 임계값 분기 (_determine_plan_from_candidates):
    - HIGH (>=12.0) → 즉시 faq 확정 (LLM 0회)
    - MID (>=4.0, <12.0) → Solar 재랭킹 (_solar_rerank)
    - LOW (<4.0) → Solar 무매칭 분류 (_solar_classify_no_match)
    Solar 실패 시 → vLLM 1.2B fallback (_classify_and_match)
[3단계] 답변 생성: vLLM 1.2B (faq/partial) 또는 고정 템플릿 (complaint/out_of_scope)

### 테스트 구성
Part 1: _parse_plan_json — JSON 파서 (v3.2 이월, vLLM fallback 경로에서 재사용)
Part 2: HIGH score 경로 — LLM 0회, kind=faq 즉시 확정
Part 3: MID score 경로 — Solar 재랭킹 mock → kind 결정 + Step 3 vLLM
Part 4: LOW score 경로 — Solar 무매칭 분류 mock
Part 5: ES 완전 실패 → Backend fallback vLLM 1.2B + _parse_plan_json 경로
Part 6: 기존 v3.2 호환 — 5가지 kind 분기 (graceful degrade 포함)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from monglepick.agents.support_assistant.faq_search import FaqCandidate
from monglepick.agents.support_assistant.models import FaqDoc, SupportPlan
from monglepick.chains import support_reply_chain as chain_mod
from monglepick.chains.support_reply_chain import (
    _parse_plan_json,
    generate_support_reply,
)
from monglepick.prompts.support_assistant import (
    SUPPORT_COMPLAINT_TEMPLATE,
    SUPPORT_OUT_OF_SCOPE_TEMPLATE,
)


# =============================================================================
# 공통 픽스처
# =============================================================================


@pytest.fixture
def sample_faqs() -> list[FaqDoc]:
    """테스트용 FAQ 2건 — Backend HTTP fallback 경로 검증용."""
    return [
        FaqDoc(
            faq_id=1,
            category="GENERAL",
            question="고객센터 전화번호와 연락처가 어떻게 되나요?",
            answer="이메일 contact@monglepick.com 과 1:1 문의 창구로 운영됩니다.",
            sort_order=50,
        ),
        FaqDoc(
            faq_id=2,
            category="ACCOUNT",
            question="비밀번호를 잊어버렸어요. 어떻게 재설정하나요?",
            answer="로그인 페이지 하단 '비밀번호 찾기' 링크로 이메일 재설정 링크를 받으세요.",
            sort_order=20,
        ),
    ]


def _make_candidate(faq_id: int, score: float, category: str = "GENERAL") -> FaqCandidate:
    """테스트용 FaqCandidate 생성 헬퍼."""
    return FaqCandidate(
        faq_id=faq_id,
        category=category,
        question=f"질문_{faq_id}",
        answer=f"답변_{faq_id}",
        score=score,
    )


# =============================================================================
# vLLM/Solar stub 설치 헬퍼
# =============================================================================


class _FakeInvokeRecorder:
    """
    guarded_ainvoke 를 대체하는 recorder.

    responses 큐에서 순서대로 응답을 꺼낸다. 호출된 request_id 를 calls 에 기록해
    "LLM 이 몇 번 어떤 순서로 호출됐는가" 를 검증할 수 있다.
    """

    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.calls: list[str] = []

    async def __call__(self, llm, prompt_value, model, request_id=""):
        self.calls.append(request_id)
        if not self._responses:
            raise RuntimeError(f"stub 응답 소진: request_id={request_id}")
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        # str 이면 SimpleNamespace(content=...) 로 감싸 반환 (vLLM 응답 형식)
        if isinstance(resp, str):
            return SimpleNamespace(content=resp)
        # Pydantic 모델 등 직접 반환 (Solar structured_output 형식)
        return resp


def _install_vllm_stub(monkeypatch, responses: list[Any]) -> _FakeInvokeRecorder:
    """
    guarded_ainvoke 와 get_vllm_llm 을 stub 으로 교체한다.
    이 stub 은 vLLM (문자열 응답) 경로에서만 사용.
    """
    rec = _FakeInvokeRecorder(responses)
    monkeypatch.setattr(chain_mod, "guarded_ainvoke", rec)
    monkeypatch.setattr(chain_mod, "get_vllm_llm", lambda temperature=0.0: object())
    return rec


def _install_solar_stub(monkeypatch, plan_response: Any) -> None:
    """
    Solar structured_output (_solar_rerank / _solar_classify_no_match) 을 mock.

    plan_response 가 Exception 이면 Solar 실패 시뮬레이션 (vLLM fallback 유발).
    그 외에는 _SolarSupportPlan 인스턴스처럼 .to_support_plan() 을 가진 객체.
    """
    from monglepick.chains.support_reply_chain import _SolarSupportPlan

    if isinstance(plan_response, Exception):
        # Solar 호출 자체를 실패시키는 mock
        async def _fail_rerank(user_message, candidates):
            raise plan_response

        async def _fail_no_match(user_message):
            raise plan_response

        monkeypatch.setattr(chain_mod, "_solar_rerank", _fail_rerank)
        monkeypatch.setattr(chain_mod, "_solar_classify_no_match", _fail_no_match)
    else:
        # 정상 반환 mock
        async def _ok_rerank(user_message, candidates):
            return plan_response

        async def _ok_no_match(user_message):
            return plan_response

        monkeypatch.setattr(chain_mod, "_solar_rerank", _ok_rerank)
        monkeypatch.setattr(chain_mod, "_solar_classify_no_match", _ok_no_match)


def _install_es_stub(monkeypatch, candidates: list[FaqCandidate]) -> None:
    """search_faq_candidates 를 고정 후보 리스트를 반환하는 stub 으로 교체."""

    async def _fake_search(user_message: str, top_k: int = 5) -> list[FaqCandidate]:
        return candidates

    monkeypatch.setattr(chain_mod, "search_faq_candidates", _fake_search)


# =============================================================================
# Part 1: _parse_plan_json — JSON 파서 (vLLM fallback 경로 재사용)
# =============================================================================


class TestParsePlanJson:
    """
    _parse_plan_json 은 vLLM 1.2B 응답을 파싱하는 유틸리티.
    Solar 실패 → vLLM fallback 경로에서 여전히 사용된다.
    """

    def test_plain_json(self):
        p = _parse_plan_json('{"kind": "faq", "matched_faq_ids": [1, 2]}')
        assert p is not None
        assert p.kind == "faq"
        assert p.matched_faq_ids == [1, 2]

    def test_code_fence_wrapped(self):
        p = _parse_plan_json(
            '```json\n{"kind": "smalltalk", "matched_faq_ids": []}\n```'
        )
        assert p is not None
        assert p.kind == "smalltalk"

    def test_with_surrounding_text(self):
        p = _parse_plan_json(
            '여기 결과입니다: {"kind": "complaint", "matched_faq_ids": []} 참고하세요.'
        )
        assert p is not None
        assert p.kind == "complaint"

    def test_non_json_returns_none(self):
        assert _parse_plan_json("그냥 텍스트") is None
        assert _parse_plan_json("") is None


# =============================================================================
# Part 2: HIGH score (>=12.0) — LLM 0회, kind=faq 즉시 확정
# =============================================================================


@pytest.mark.asyncio
class TestHighScorePath:
    """
    ES top 점수가 HIGH(12.0) 이상이면:
    - Solar / vLLM 분류 LLM 호출이 없어야 한다
    - 즉시 kind=faq, matched_faq_ids=[top_candidate.faq_id] 로 확정
    - Step 3 (답변 생성) vLLM 1회만 호출
    """

    async def test_high_score_skips_classification_llm(self, monkeypatch, sample_faqs):
        """
        top_score=18.5 (HIGH) → 분류 LLM 0회, 답변 vLLM 1회 = 총 1회.
        kind=faq, matched_faq_ids=[10] 이어야 한다.
        """
        # ES 후보: top=18.5 → HIGH 분기
        candidates = [
            _make_candidate(faq_id=10, score=18.5, category="PAYMENT"),
            _make_candidate(faq_id=7, score=9.0),
        ]
        _install_es_stub(monkeypatch, candidates)

        # Solar mock 은 불필요 (HIGH 에서 호출 안 됨) 하지만 혹시라도 호출되면
        # 테스트가 실패하도록 예외를 걸어 둔다.
        _install_solar_stub(
            monkeypatch,
            RuntimeError("HIGH score 경로에서 Solar 가 호출되었습니다 — 버그"),
        )

        # vLLM: Step 3 답변 생성 1회
        rec = _install_vllm_stub(
            monkeypatch,
            ["환불은 마이페이지 결제내역에서 신청하실 수 있어요."],
        )

        reply = await generate_support_reply(
            user_message="환불하고 싶어요",
            faqs=sample_faqs,
        )

        assert reply.kind == "faq"
        assert reply.matched_faq_ids == [10]
        assert "환불" in reply.answer
        assert reply.needs_human is False
        # 분류 LLM 호출 없음 → vLLM 은 답변 생성 1회만 호출
        assert len(rec.calls) == 1
        assert rec.calls[0] == "support_answer_faq"

    async def test_high_score_exact_boundary(self, monkeypatch, sample_faqs):
        """
        top_score=12.0 (경계값, HIGH 이상) → 즉시 faq 확정.
        """
        candidates = [_make_candidate(faq_id=5, score=12.0)]
        _install_es_stub(monkeypatch, candidates)
        _install_solar_stub(
            monkeypatch,
            RuntimeError("경계값 HIGH 에서 Solar 호출 금지"),
        )
        rec = _install_vllm_stub(monkeypatch, ["고객센터는 이메일로 운영합니다."])

        reply = await generate_support_reply(
            user_message="연락처 알려주세요",
            faqs=sample_faqs,
        )

        assert reply.kind == "faq"
        assert reply.matched_faq_ids == [5]
        assert len(rec.calls) == 1


# =============================================================================
# Part 3: MID score (4.0 <= score < 12.0) — Solar 재랭킹 → kind 결정 + vLLM
# =============================================================================


@pytest.mark.asyncio
class TestMidScorePath:
    """
    ES top 점수가 MID(4.0) 이상 HIGH(12.0) 미만이면:
    - Solar 재랭킹(_solar_rerank) 을 호출해 kind/faq_id 를 결정한다
    - Solar 결과 kind=faq → Step 3 vLLM 답변 생성 1회 호출
    - Solar 결과 kind=out_of_scope → 고정 템플릿 (vLLM 0회)
    """

    async def test_mid_score_solar_rerank_faq(self, monkeypatch, sample_faqs):
        """
        top_score=7.0 (MID) → Solar 재랭킹 호출, kind=faq, faq_id=2 선정.
        vLLM 답변 생성 1회 호출.
        """
        candidates = [
            _make_candidate(faq_id=2, score=7.0, category="ACCOUNT"),
            _make_candidate(faq_id=1, score=5.5),
        ]
        _install_es_stub(monkeypatch, candidates)

        # Solar 재랭킹 결과: kind=faq, faq_id=2
        solar_plan = SupportPlan(kind="faq", matched_faq_ids=[2])
        _install_solar_stub(monkeypatch, solar_plan)

        # vLLM: Step 3 답변 생성 1회
        rec = _install_vllm_stub(
            monkeypatch,
            ["비밀번호는 '비밀번호 찾기' 에서 재설정하실 수 있어요."],
        )

        reply = await generate_support_reply(
            user_message="비밀번호 잊어버렸어요",
            faqs=sample_faqs,
        )

        assert reply.kind == "faq"
        assert reply.matched_faq_ids == [2]
        assert "비밀번호" in reply.answer
        assert reply.needs_human is False
        # 답변 생성 vLLM 1회만 호출 (분류는 Solar mock 이 처리)
        assert len(rec.calls) == 1
        assert rec.calls[0] == "support_answer_faq"

    async def test_mid_score_solar_rerank_out_of_scope(self, monkeypatch, sample_faqs):
        """
        Solar 재랭킹 결과 kind=out_of_scope → 고정 템플릿, vLLM 0회.
        """
        candidates = [_make_candidate(faq_id=1, score=6.0)]
        _install_es_stub(monkeypatch, candidates)

        solar_plan = SupportPlan(kind="out_of_scope", matched_faq_ids=[])
        _install_solar_stub(monkeypatch, solar_plan)

        rec = _install_vllm_stub(monkeypatch, [])  # 호출 없어야 함

        reply = await generate_support_reply(
            user_message="봉준호 감독 영화 알려줘",
            faqs=sample_faqs,
        )

        assert reply.kind == "out_of_scope"
        assert reply.answer == SUPPORT_OUT_OF_SCOPE_TEMPLATE
        assert reply.needs_human is False
        # vLLM 호출 없어야 한다
        assert rec.calls == []

    async def test_mid_score_boundary_exact_4(self, monkeypatch, sample_faqs):
        """
        top_score=4.0 (MID 경계값) → Solar 재랭킹 경로 진입.
        """
        candidates = [_make_candidate(faq_id=2, score=4.0)]
        _install_es_stub(monkeypatch, candidates)

        solar_plan = SupportPlan(kind="faq", matched_faq_ids=[2])
        _install_solar_stub(monkeypatch, solar_plan)

        rec = _install_vllm_stub(monkeypatch, ["답변 텍스트"])

        reply = await generate_support_reply(
            user_message="질문",
            faqs=sample_faqs,
        )

        assert reply.kind == "faq"
        assert reply.matched_faq_ids == [2]


# =============================================================================
# Part 4: LOW score (<4.0) — Solar 무매칭 분류
# =============================================================================


@pytest.mark.asyncio
class TestLowScorePath:
    """
    ES top 점수가 LOW(<4.0) 이면:
    - Solar 무매칭 분류(_solar_classify_no_match) 를 호출한다
    - kind=smalltalk → vLLM 짧은 응대
    - kind=complaint → 고정 템플릿
    - kind=out_of_scope → 고정 템플릿
    """

    async def test_low_score_solar_no_match_smalltalk(self, monkeypatch, sample_faqs):
        """
        top_score=1.0 (LOW) → Solar 무매칭 분류 → kind=smalltalk → vLLM 1회.
        """
        candidates = [_make_candidate(faq_id=1, score=1.0)]
        _install_es_stub(monkeypatch, candidates)

        solar_plan = SupportPlan(kind="smalltalk", matched_faq_ids=[])
        _install_solar_stub(monkeypatch, solar_plan)

        rec = _install_vllm_stub(
            monkeypatch,
            ["안녕하세요! 궁금한 점 있으면 편하게 말씀해 주세요."],
        )

        reply = await generate_support_reply(
            user_message="안녕?",
            faqs=sample_faqs,
        )

        assert reply.kind == "smalltalk"
        assert reply.matched_faq_ids == []
        assert "안녕하세요" in reply.answer
        assert reply.needs_human is False
        assert rec.calls == ["support_smalltalk"]

    async def test_low_score_solar_no_match_complaint(self, monkeypatch, sample_faqs):
        """
        LOW score → Solar 무매칭 → kind=complaint → 고정 템플릿, vLLM 0회.
        """
        candidates = [_make_candidate(faq_id=1, score=0.5)]
        _install_es_stub(monkeypatch, candidates)

        solar_plan = SupportPlan(kind="complaint", matched_faq_ids=[])
        _install_solar_stub(monkeypatch, solar_plan)

        rec = _install_vllm_stub(monkeypatch, [])  # 호출 없어야 함

        reply = await generate_support_reply(
            user_message="앱이 계속 튕겨요",
            faqs=sample_faqs,
        )

        assert reply.kind == "complaint"
        assert reply.answer == SUPPORT_COMPLAINT_TEMPLATE
        assert reply.needs_human is True
        assert rec.calls == []

    async def test_empty_candidates_goes_low_path(self, monkeypatch, sample_faqs):
        """
        ES 후보가 0건이면 top_score=0.0 → LOW 경로 → Solar 무매칭 분류.
        """
        _install_es_stub(monkeypatch, [])  # 후보 없음

        solar_plan = SupportPlan(kind="out_of_scope", matched_faq_ids=[])
        _install_solar_stub(monkeypatch, solar_plan)

        rec = _install_vllm_stub(monkeypatch, [])

        reply = await generate_support_reply(
            user_message="완전히 관련 없는 질문",
            faqs=sample_faqs,
        )

        assert reply.kind == "out_of_scope"
        assert reply.answer == SUPPORT_OUT_OF_SCOPE_TEMPLATE
        assert rec.calls == []


# =============================================================================
# Part 5: ES 완전 실패 → Backend fallback vLLM 1.2B + _parse_plan_json 경로
# =============================================================================


@pytest.mark.asyncio
class TestEsFailureFallback:
    """
    ES 검색이 실패(빈 리스트 반환)하고 Solar 도 실패할 때:
    _classify_and_match (vLLM 1.2B + _parse_plan_json) 경로로 fallback.
    """

    async def test_es_fail_solar_fail_falls_back_to_vllm(self, monkeypatch, sample_faqs):
        """
        ES 실패(빈 리스트) + Solar 실패 → vLLM 1.2B Step 1 분류 + Step 3 답변 생성.
        """
        # ES 실패 → 빈 리스트 (LOW 경로로 진입)
        _install_es_stub(monkeypatch, [])

        # Solar 도 실패 → vLLM fallback
        _install_solar_stub(monkeypatch, ConnectionError("Solar API down"))

        # vLLM fallback: Step 1 분류 응답 + Step 3 답변 응답
        rec = _install_vllm_stub(
            monkeypatch,
            [
                # Step 1: _classify_and_match 에서 호출 (support_plan)
                '{"kind": "faq", "matched_faq_ids": [2]}',
                # Step 3: _generate_answer_from_faq 에서 호출 (support_answer_faq)
                "비밀번호는 '비밀번호 찾기' 에서 이메일 인증으로 재설정하실 수 있어요.",
            ],
        )

        reply = await generate_support_reply(
            user_message="비밀번호 재설정 방법",
            faqs=sample_faqs,
        )

        assert reply.kind == "faq"
        assert reply.matched_faq_ids == [2]
        assert "비밀번호" in reply.answer
        assert reply.needs_human is False
        # vLLM 이 분류(support_plan) + 답변 생성(support_answer_faq) 두 번 호출됨
        assert rec.calls == ["support_plan", "support_answer_faq"]

    async def test_es_fail_solar_fail_vllm_unparsable_complaint_fallback(
        self, monkeypatch, sample_faqs
    ):
        """
        ES 실패 + Solar 실패 + vLLM 도 JSON 파싱 불가 → complaint 템플릿.
        에러 전파 없이 graceful degrade.
        """
        _install_es_stub(monkeypatch, [])
        _install_solar_stub(monkeypatch, ConnectionError("Solar down"))

        rec = _install_vllm_stub(
            monkeypatch,
            ["이건 JSON 이 아닌 헛소리입니다"],  # _parse_plan_json 실패 유발
        )

        reply = await generate_support_reply(
            user_message="테스트",
            faqs=sample_faqs,
        )

        assert reply.kind == "complaint"
        assert reply.answer == SUPPORT_COMPLAINT_TEMPLATE
        assert reply.needs_human is True
        # vLLM 분류 1회만 호출 (파싱 실패 후 complaint 확정, Step 3 미호출)
        assert rec.calls == ["support_plan"]

    async def test_es_fail_no_faqs_complaint_template(self, monkeypatch):
        """
        ES 실패 + faqs=[] (Backend 도 장애) + Solar 실패 → complaint 템플릿.
        전체 외부 의존성 장애 상황의 최하단 안전망 검증.
        """
        _install_es_stub(monkeypatch, [])
        _install_solar_stub(monkeypatch, RuntimeError("all systems down"))

        # vLLM: faq_titles 가 빈 목록이므로 분류 실패 → complaint fallback
        rec = _install_vllm_stub(
            monkeypatch,
            ["알 수 없는 텍스트"],
        )

        reply = await generate_support_reply(
            user_message="도움이 필요해요",
            faqs=[],  # Backend FAQ 없음
        )

        assert reply.kind == "complaint"
        assert reply.needs_human is True


# =============================================================================
# Part 6: 기존 v3.2 호환 — 5가지 kind 분기 + graceful degrade
# =============================================================================
# v3.3 에서도 vLLM fallback 경로(_classify_and_match) 는 그대로 남아있다.
# vLLM 분류 응답이 주어졌을 때 kind 별 분기가 올바르게 동작하는지
# ES stub(빈 리스트) + Solar stub(실패) 로 vLLM fallback 경로를 강제하며 검증.


@pytest.mark.asyncio
class TestVllmFallbackKindBranching:
    """
    ES 실패 + Solar 실패 → vLLM fallback (_classify_and_match) 경로 강제.
    5가지 kind 분기가 올바르게 동작하는지 검증.
    """

    def _setup_fallback(self, monkeypatch):
        """ES 빈 리스트 + Solar 실패로 vLLM fallback 경로 강제."""
        _install_es_stub(monkeypatch, [])
        _install_solar_stub(monkeypatch, RuntimeError("force vllm fallback"))

    async def test_kind_faq_calls_answer_generation(self, monkeypatch, sample_faqs):
        """vLLM fallback: faq 분류 → Step 3 답변 생성 호출, needs_human=False."""
        self._setup_fallback(monkeypatch)
        rec = _install_vllm_stub(
            monkeypatch,
            [
                '{"kind": "faq", "matched_faq_ids": [2]}',
                "비밀번호는 '비밀번호 찾기' 에서 이메일 인증으로 재설정하실 수 있어요.",
            ],
        )

        reply = await generate_support_reply(
            user_message="비밀번호 변경하고 싶어요",
            faqs=sample_faqs,
        )

        assert reply.kind == "faq"
        assert reply.matched_faq_ids == [2]
        assert "비밀번호" in reply.answer
        assert reply.needs_human is False
        assert rec.calls == ["support_plan", "support_answer_faq"]

    async def test_kind_partial_sets_needs_human_true(self, monkeypatch, sample_faqs):
        self._setup_fallback(monkeypatch)
        rec = _install_vllm_stub(
            monkeypatch,
            [
                '{"kind": "partial", "matched_faq_ids": [2]}',
                "완전히 일치하는 안내는 아니지만 비슷한 내용이 있어요... '문의하기' 탭",
            ],
        )

        reply = await generate_support_reply(
            user_message="비밀번호 관련 문의",
            faqs=sample_faqs,
        )

        assert reply.kind == "partial"
        assert reply.matched_faq_ids == [2]
        assert reply.needs_human is True
        assert rec.calls == ["support_plan", "support_answer_partial"]

    async def test_kind_smalltalk_calls_smalltalk(self, monkeypatch, sample_faqs):
        self._setup_fallback(monkeypatch)
        rec = _install_vllm_stub(
            monkeypatch,
            [
                '{"kind": "smalltalk", "matched_faq_ids": []}',
                "안녕하세요! 도와드릴 일이 있으면 편하게 말씀해 주세요.",
            ],
        )

        reply = await generate_support_reply(
            user_message="안녕?",
            faqs=sample_faqs,
        )

        assert reply.kind == "smalltalk"
        assert reply.matched_faq_ids == []
        assert reply.needs_human is False
        assert reply.answer.startswith("안녕하세요")
        assert rec.calls == ["support_plan", "support_smalltalk"]

    async def test_kind_complaint_uses_template_no_step3(self, monkeypatch, sample_faqs):
        """complaint 는 Step 3 LLM 호출 없이 고정 템플릿."""
        self._setup_fallback(monkeypatch)
        rec = _install_vllm_stub(
            monkeypatch,
            ['{"kind": "complaint", "matched_faq_ids": []}'],
        )

        reply = await generate_support_reply(
            user_message="결제 에러 계속 나요 긴급",
            faqs=sample_faqs,
        )

        assert reply.kind == "complaint"
        assert reply.matched_faq_ids == []
        assert reply.needs_human is True
        assert reply.answer == SUPPORT_COMPLAINT_TEMPLATE
        assert rec.calls == ["support_plan"]

    async def test_kind_out_of_scope_uses_template_no_step3(self, monkeypatch, sample_faqs):
        self._setup_fallback(monkeypatch)
        rec = _install_vllm_stub(
            monkeypatch,
            ['{"kind": "out_of_scope", "matched_faq_ids": []}'],
        )

        reply = await generate_support_reply(
            user_message="봉준호 감독 영화 추천해줘",
            faqs=sample_faqs,
        )

        assert reply.kind == "out_of_scope"
        assert reply.answer == SUPPORT_OUT_OF_SCOPE_TEMPLATE
        assert reply.needs_human is False
        assert rec.calls == ["support_plan"]

    async def test_hallucinated_faq_ids_are_dropped_and_demoted(
        self, monkeypatch, sample_faqs
    ):
        """
        vLLM 이 faq 라고 하면서 실제 존재하지 않는 id(999) 를 돌려주면
        valid_ids 검증에서 제거 → cleaned_ids=[] → complaint 로 강등.
        """
        self._setup_fallback(monkeypatch)
        rec = _install_vllm_stub(
            monkeypatch,
            ['{"kind": "faq", "matched_faq_ids": [999]}'],
        )

        reply = await generate_support_reply(
            user_message="뭔가 찾는 거",
            faqs=sample_faqs,
        )

        assert reply.kind == "complaint"
        assert reply.matched_faq_ids == []
        assert rec.calls == ["support_plan"]

    async def test_step3_empty_answer_falls_back_to_faq_raw(
        self, monkeypatch, sample_faqs
    ):
        """vLLM 이 빈 답변을 돌려주면 FAQ 원문을 그대로 노출."""
        self._setup_fallback(monkeypatch)
        rec = _install_vllm_stub(
            monkeypatch,
            [
                '{"kind": "faq", "matched_faq_ids": [2]}',
                "",  # 빈 답변
            ],
        )

        reply = await generate_support_reply(
            user_message="비밀번호 잊어버렸어요",
            faqs=sample_faqs,
        )

        assert reply.kind == "faq"
        assert reply.matched_faq_ids == [2]
        # FAQ 원문 answer 가 노출되어야 함
        assert "비밀번호 찾기" in reply.answer
