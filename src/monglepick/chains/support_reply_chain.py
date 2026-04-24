"""
고객센터 AI 챗봇 — ES Nori 검색 + Solar 경계분류 + vLLM 1.2B 답변 체인 (v3.3).

### 근본 원인과 해결책
v3.2 는 vLLM EXAONE 1.2B 에게 FAQ 분류와 matched_faq_ids 선정을 동시에 맡겼다.
1.2B 모델이 JSON 파싱 실패 또는 환각 ID 를 반환하면 complaint 하드폴백으로 떨어져
사실상 모든 질문이 "1:1 문의" 로 수렴되는 문제가 있었다.

v3.3 은 역할을 3단계로 분리한다:

**[1단계] ES Nori BM25 검색 (Python, LLM 0회)**
    `faq_search.search_faq_candidates` → top-5 FaqCandidate (점수 포함)

**[2단계] 점수 임계값 분기 (Python 규칙 기반)**
    - top_score >= HIGH(12.0) → 즉시 faq 확정, LLM 호출 0회
    - MID(4.0) <= top_score < HIGH → Solar 재랭킹: top-3 에서 정답 선택
    - top_score < MID → Solar 무매칭 분류: smalltalk/complaint/out_of_scope 판정

**[3단계] 답변 생성 (vLLM 1.2B)**
    - kind=faq/partial → FAQ answer 근거 대화체 답변
    - kind=smalltalk   → 짧은 몽글이 응대
    - kind=complaint/out_of_scope → 고정 템플릿 (LLM 0회)

### fallback 계층
Solar 호출 실패
    → vLLM 1.2B fallback (_classify_and_match, 기존 v3.2 로직 재사용)
vLLM 1.2B 실패
    → complaint 템플릿
ES 실패
    → Backend HTTP FAQ 전체 조회 + vLLM 1.2B 분류 (context_loader 에서 faqs 인수)

### LLM 호출 수 (정상 경로)
- HIGH 히트: 0회 (분류) + 1회 (답변) = 1회
- MID 히트: 1회 (Solar 재랭킹) + 1회 (답변) = 2회
- LOW: 1회 (Solar 무매칭) + 0회 또는 1회 (smalltalk) = 최대 1회
"""

from __future__ import annotations

import json
import re
import time
import traceback
from typing import Optional

import structlog
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from monglepick.agents.support_assistant.faq_search import (
    FaqCandidate,
    search_faq_candidates,
)
from monglepick.agents.support_assistant.models import (
    FaqDoc,
    SupportPlan,
    SupportReply,
)
from monglepick.config import settings
from monglepick.llm.factory import (
    _use_solar_api,
    get_structured_llm,
    get_vllm_llm,
    guarded_ainvoke,
)
from monglepick.prompts.support_assistant import (
    SUPPORT_ANSWER_FROM_FAQ_HUMAN_PROMPT,
    SUPPORT_ANSWER_FROM_FAQ_SYSTEM_PROMPT,
    SUPPORT_COMPLAINT_TEMPLATE,
    SUPPORT_OUT_OF_SCOPE_TEMPLATE,
    SUPPORT_PLAN_HUMAN_PROMPT,
    SUPPORT_PLAN_SYSTEM_PROMPT,
    SUPPORT_SMALLTALK_HUMAN_PROMPT,
    SUPPORT_SMALLTALK_SYSTEM_PROMPT,
    SUPPORT_SOLAR_NOMATCH_HUMAN_PROMPT,
    SUPPORT_SOLAR_NOMATCH_SYSTEM_PROMPT,
    SUPPORT_SOLAR_RERANK_HUMAN_PROMPT,
    SUPPORT_SOLAR_RERANK_SYSTEM_PROMPT,
)

logger = structlog.get_logger(__name__)

# =============================================================================
# Solar 구조화 출력 스키마
# =============================================================================


class _SolarSupportPlan(BaseModel):
    """
    Solar(solar-pro) with_structured_output 출력 스키마.

    Solar 는 JSON Schema 를 엄격하게 준수하므로 vLLM 1.2B 처럼
    정규식 복원 없이 JSON 100% 보장된다.

    matched_faq_id : Solar 재랭킹 시 선정된 단일 FAQ id (kind=faq 는 1개, partial 은 1~2개).
                     out_of_scope / smalltalk / complaint 는 None.
    matched_faq_ids: 편의상 list 로 정규화. 비어있으면 빈 리스트.
    """

    kind: str = Field(
        description=(
            "응답 종류. faq / partial / out_of_scope / smalltalk / complaint 중 하나."
        )
    )
    matched_faq_id: Optional[int] = Field(
        default=None,
        description=(
            "선정된 FAQ id (단일). kind=faq/partial 에서만 채워지며 그 외는 null."
        ),
    )
    # partial 에서 2건까지 허용하기 위한 추가 필드 (선택)
    matched_faq_id_2: Optional[int] = Field(
        default=None,
        description="kind=partial 시 두 번째 참고 FAQ id (선택). 단 1개로 충분하면 null.",
    )

    def to_support_plan(self) -> SupportPlan:
        """SupportPlan 으로 정규화 — 기존 파이프라인과 인터페이스를 맞춘다."""
        ids: list[int] = []
        if self.matched_faq_id is not None:
            ids.append(self.matched_faq_id)
        if self.matched_faq_id_2 is not None:
            ids.append(self.matched_faq_id_2)
        # kind 값 안전화 — 예외적으로 모르는 값이 오면 complaint 로 강등
        valid_kinds = {"faq", "partial", "complaint", "out_of_scope", "smalltalk"}
        kind = self.kind if self.kind in valid_kinds else "complaint"
        return SupportPlan(kind=kind, matched_faq_ids=ids)


# =============================================================================
# 유틸리티 — 후보 목록 직렬화 (Solar 프롬프트용)
# =============================================================================


def _build_candidates_block(candidates: list[FaqCandidate]) -> str:
    """
    Solar 재랭킹 프롬프트에 실을 FAQ 후보 블록을 생성한다.

    형식:
        [faq_id=12] 카테고리: PAYMENT
        질문: 환불은 어떻게 하나요?
        답변: 마이페이지 > 결제내역에서 환불 신청을 하시면...
        ---

    answer 는 200자로 자른다 (Solar 컨텍스트 절약).
    """
    if not candidates:
        return "(후보 없음)"
    blocks: list[str] = []
    for c in candidates:
        answer_snippet = (c.answer or "").strip()
        if len(answer_snippet) > 200:
            answer_snippet = answer_snippet[:200] + "..."
        blocks.append(
            f"[faq_id={c.faq_id}] 카테고리: {c.category}\n"
            f"질문: {c.question}\n"
            f"답변: {answer_snippet}"
        )
    return "\n---\n".join(blocks)


# =============================================================================
# [2단계-B] Solar 재랭킹 — MID~HIGH 구간 (후보 있음)
# =============================================================================


async def _solar_rerank(
    user_message: str,
    candidates: list[FaqCandidate],
) -> SupportPlan:
    """
    ES top-3 후보를 Solar 에게 보여주고 가장 적합한 FAQ 와 kind 를 선택하게 한다.

    Solar with_structured_output(_SolarSupportPlan) 을 사용하므로 JSON 100% 보장.
    Solar 호출 실패 시 exception 을 raise 해 상위에서 vLLM fallback 으로 전환한다.

    Args:
        user_message: 사용자 발화
        candidates  : ES 검색 상위 후보 (최대 3건 권장)

    Returns:
        SupportPlan(kind, matched_faq_ids)

    Raises:
        Exception: Solar 호출 실패 시 (상위에서 fallback 처리)
    """
    started = time.perf_counter()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPPORT_SOLAR_RERANK_SYSTEM_PROMPT),
            ("human", SUPPORT_SOLAR_RERANK_HUMAN_PROMPT),
        ]
    )
    # get_structured_llm: LLM_MODE=hybrid/api_only → Solar, local_only → Ollama
    structured_llm = get_structured_llm(
        schema=_SolarSupportPlan,
        temperature=0.1,
        use_api=True,
    )

    inputs = {
        "user_message": user_message,
        "candidates_block": _build_candidates_block(candidates),
    }

    prompt_value = await prompt.ainvoke(inputs)
    result: _SolarSupportPlan = await guarded_ainvoke(
        structured_llm,
        prompt_value,
        model="solar_api",
        request_id="support_solar_rerank",
    )

    elapsed_ms = (time.perf_counter() - started) * 1000
    plan = result.to_support_plan()

    logger.info(
        "support_solar_rerank",
        kind=plan.kind,
        faq_id=plan.matched_faq_ids[0] if plan.matched_faq_ids else None,
        latency_ms=round(elapsed_ms, 1),
    )
    return plan


# =============================================================================
# [2단계-C] Solar 무매칭 분류 — LOW 구간 (후보 없거나 점수 낮음)
# =============================================================================


async def _solar_classify_no_match(user_message: str) -> SupportPlan:
    """
    ES 후보가 없거나 점수가 LOW 일 때 Solar 에게 발화 종류를 분류하게 한다.

    이 경로에서 kind 는 smalltalk / complaint / out_of_scope 중 하나.
    FAQ 매칭 없으므로 matched_faq_ids 는 항상 [].

    Solar 호출 실패 시 exception 을 raise 해 상위에서 vLLM fallback 으로 전환한다.

    Args:
        user_message: 사용자 발화

    Returns:
        SupportPlan(kind, matched_faq_ids=[])

    Raises:
        Exception: Solar 호출 실패 시
    """
    started = time.perf_counter()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPPORT_SOLAR_NOMATCH_SYSTEM_PROMPT),
            ("human", SUPPORT_SOLAR_NOMATCH_HUMAN_PROMPT),
        ]
    )
    structured_llm = get_structured_llm(
        schema=_SolarSupportPlan,
        temperature=0.1,
        use_api=True,
    )

    prompt_value = await prompt.ainvoke({"user_message": user_message})
    result: _SolarSupportPlan = await guarded_ainvoke(
        structured_llm,
        prompt_value,
        model="solar_api",
        request_id="support_solar_no_match",
    )

    elapsed_ms = (time.perf_counter() - started) * 1000

    # 무매칭 경로 — matched_faq_ids 는 강제 비움 (Solar 가 id 를 지어낼 수 없도록)
    plan = SupportPlan(kind=result.kind or "complaint", matched_faq_ids=[])

    logger.info(
        "support_solar_no_match",
        kind=plan.kind,
        latency_ms=round(elapsed_ms, 1),
    )
    return plan


# =============================================================================
# [2단계 통합] 점수 임계값 분기 → Solar 또는 즉시 확정
# =============================================================================


async def _determine_plan_from_candidates(
    user_message: str,
    candidates: list[FaqCandidate],
    faqs: list[FaqDoc],
) -> SupportPlan:
    """
    ES 후보 점수에 따라 분기해 SupportPlan 을 결정한다.

    HIGH 구간: LLM 호출 없이 즉시 확정 (가장 빠른 경로)
    MID~HIGH:  Solar 재랭킹 호출 (실패 시 vLLM fallback)
    LOW:       Solar 무매칭 분류 호출 (실패 시 vLLM fallback)

    Solar 호출이 실패하면 vLLM 1.2B 기존 분류 로직(_classify_and_match) 으로 내려간다.
    vLLM 마저 실패하면 complaint 템플릿.

    Args:
        user_message: 사용자 발화
        candidates  : ES 검색 결과 (빈 리스트 가능)
        faqs        : Backend 에서 가져온 전체 FAQ 목록 (vLLM fallback 용)

    Returns:
        SupportPlan
    """
    top_score = candidates[0].score if candidates else 0.0
    high = settings.SUPPORT_ES_SCORE_HIGH
    mid = settings.SUPPORT_ES_SCORE_MID

    # ── 경로 A: HIGH 이상 → 즉시 faq 확정 (LLM 0회) ──────────────────────────
    if top_score >= high:
        top = candidates[0]
        logger.info(
            "support_plan_high_score_shortcut",
            faq_id=top.faq_id,
            score=round(top_score, 2),
        )
        return SupportPlan(kind="faq", matched_faq_ids=[top.faq_id])

    # ── 경로 B: MID~HIGH → Solar 재랭킹 ─────────────────────────────────────
    if top_score >= mid:
        # top-3 만 전달해 Solar 컨텍스트 절약
        top3 = candidates[:3]
        try:
            return await _solar_rerank(user_message, top3)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "support_solar_fallback_to_vllm",
                reason=f"solar_rerank_failed: {exc!s}",
                error_type=type(exc).__name__,
            )
            # Solar 실패 → vLLM 1.2B 분류 (기존 v3.2 로직)
            return await _classify_and_match(user_message, faqs)

    # ── 경로 C: LOW → Solar 무매칭 분류 ─────────────────────────────────────
    # ES 후보가 아예 없거나 점수가 낮으면 FAQ 무관 발화로 간주
    try:
        return await _solar_classify_no_match(user_message)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "support_solar_fallback_to_vllm",
            reason=f"solar_no_match_failed: {exc!s}",
            error_type=type(exc).__name__,
        )
        # Solar 실패 → vLLM 1.2B 분류 (기존 v3.2 로직)
        return await _classify_and_match(user_message, faqs)


# =============================================================================
# [기존 v3.2] vLLM 1.2B 분류 — Solar 실패 시 fallback 전용
# =============================================================================

# Step 1 프롬프트는 question 만 포함. 한 건당 최대 120자 (극단적으로 긴 질문 자름).
_FAQ_QUESTION_BLOCK_LIMIT = 120

# LLM 이 JSON 을 코드블록으로 감쌌을 때 본문만 추출하기 위한 정규식
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _build_faq_titles(faqs: list[FaqDoc]) -> str:
    """
    vLLM fallback 용 — FAQ question 만 포함한 짧은 목록 직렬화.

    Solar 실패 시 1.2B 에게 넘기는 컨텍스트이므로 최대한 짧게 유지.
    """
    if not faqs:
        return "(등록된 FAQ 가 없습니다)"
    lines = []
    for faq in faqs:
        q = (faq.question or "").strip()
        if len(q) > _FAQ_QUESTION_BLOCK_LIMIT:
            q = q[:_FAQ_QUESTION_BLOCK_LIMIT] + "..."
        lines.append(f"[id={faq.faq_id}] {q}")
    return "\n".join(lines)


def _parse_plan_json(raw_text: str) -> SupportPlan | None:
    """
    vLLM fallback 경로용 — LLM 응답에서 SupportPlan JSON 을 파싱한다.

    1.2B 모델이 앞뒤에 코드블록/설명을 섞을 수 있으므로 중괄호 블록을 추출해 파싱.
    실패하면 None 반환 — 호출 측이 complaint 폴백으로 분기한다.

    Solar 재랭킹/무매칭 분류는 with_structured_output 으로 이 함수가 필요 없다.
    1.2B fallback 경로에서만 사용.
    """
    if not raw_text:
        return None
    text = raw_text.strip()
    # 코드펜스 제거
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE)
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    try:
        return SupportPlan.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.debug("support_plan_validate_failed", error=str(exc), data=data)
        return None


async def _classify_and_match(
    user_message: str, faqs: list[FaqDoc]
) -> SupportPlan:
    """
    vLLM EXAONE 1.2B fallback — FAQ question 목록만 주고 SupportPlan 을 받아온다.

    Solar 가 실패한 경우에만 호출된다. 분류 정확도는 Solar 보다 낮지만
    완전한 장애보다는 낫다.

    실패 시 SupportPlan(kind="complaint", []) 로 graceful degrade.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SUPPORT_PLAN_SYSTEM_PROMPT), ("human", SUPPORT_PLAN_HUMAN_PROMPT)]
    )
    # 분류는 결정적이어야 정확도 ↑ — temperature 0.0
    llm = get_vllm_llm(temperature=0.0)

    inputs = {
        "user_message": user_message,
        "faq_titles": _build_faq_titles(faqs),
    }

    try:
        prompt_value = await prompt.ainvoke(inputs)
        response = await guarded_ainvoke(
            llm,
            prompt_value,
            model="vllm_exaone_1_2b",
            request_id="support_plan",
        )
        raw = (getattr(response, "content", "") or "").strip()
        plan = _parse_plan_json(raw)
        if plan is None:
            logger.warning(
                "support_plan_unparsable",
                raw_preview=raw[:200],
            )
            return SupportPlan(kind="complaint", matched_faq_ids=[])
        return plan
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "support_plan_error",
            error=str(exc),
            error_type=type(exc).__name__,
            stack_trace=traceback.format_exc(),
        )
        return SupportPlan(kind="complaint", matched_faq_ids=[])


# =============================================================================
# Step 2-A — FAQ 기반 답변 생성 (kind ∈ {faq, partial}) — vLLM 1.2B
# =============================================================================

# Step 2 에서 선정된 FAQ answer 를 싣는 최대 문자 수.
# 한 건 400자 × 3건 = 1200자 ≈ 600 tok — 1.2B 의 2048 tok 컨텍스트에 여유.
_FAQ_ANSWER_TRUNCATE = 400


def _build_faq_answer_context(
    faqs: list[FaqDoc],
    candidates: list[FaqCandidate],
    matched_ids: list[int],
) -> str:
    """
    Step 2-A 프롬프트용 — 선정된 FAQ 의 full answer 를 추려 직렬화한다.

    우선순위:
    1. ES FaqCandidate (검색 경로, answer 포함)
    2. Backend FaqDoc (Backend HTTP fallback 경로)

    matched_ids 순서를 유지해 중요 FAQ 가 먼저 노출되도록 한다.
    """
    # ES 후보에서 먼저 찾기 (faq_id 기준 인덱스)
    by_id_candidate: dict[int, FaqCandidate] = {c.faq_id: c for c in candidates}
    # Backend FAQ 에서도 찾기 (fallback 경로 대비)
    by_id_faq: dict[int, FaqDoc] = {f.faq_id: f for f in faqs}

    blocks: list[str] = []
    for fid in matched_ids:
        # ES 후보 우선, 없으면 Backend FAQ 사용
        cand = by_id_candidate.get(int(fid))
        if cand is not None:
            answer = (cand.answer or "").strip()
            question = cand.question
        else:
            faq = by_id_faq.get(int(fid))
            if faq is None:
                continue
            answer = (faq.answer or "").strip()
            question = faq.question

        if len(answer) > _FAQ_ANSWER_TRUNCATE:
            answer = answer[:_FAQ_ANSWER_TRUNCATE] + "..."
        blocks.append(f"[id={fid}] 질문: {question}\n답변: {answer}")

    if not blocks:
        return "(근거 FAQ 없음)"
    return "\n\n".join(blocks)


async def _generate_answer_from_faq(
    user_message: str,
    faqs: list[FaqDoc],
    candidates: list[FaqCandidate],
    matched_ids: list[int],
    match_mode: str,  # "faq" | "partial"
) -> str:
    """
    Step 2-A: 선정된 FAQ 근거로 몽글이 톤 답변 자유 텍스트 생성 (vLLM 1.2B).

    ES FaqCandidate 와 Backend FaqDoc 양쪽을 참조해 answer 본문을 구성한다.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPPORT_ANSWER_FROM_FAQ_SYSTEM_PROMPT),
            ("human", SUPPORT_ANSWER_FROM_FAQ_HUMAN_PROMPT),
        ]
    )
    llm = get_vllm_llm(temperature=0.3)
    inputs = {
        "user_message": user_message,
        "match_mode": match_mode,
        "faq_context": _build_faq_answer_context(faqs, candidates, matched_ids),
    }
    try:
        prompt_value = await prompt.ainvoke(inputs)
        response = await guarded_ainvoke(
            llm,
            prompt_value,
            model="vllm_exaone_1_2b",
            request_id=f"support_answer_{match_mode}",
        )
        text = (getattr(response, "content", "") or "").strip()
        return text
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "support_answer_error",
            error=str(exc),
            error_type=type(exc).__name__,
            match_mode=match_mode,
        )
        # LLM 실패 — ES 후보 또는 Backend FAQ 원문을 그대로 노출 (정확한 정보 우선)
        by_id_c: dict[int, FaqCandidate] = {c.faq_id: c for c in candidates}
        by_id_f: dict[int, FaqDoc] = {f.faq_id: f for f in faqs}
        for fid in matched_ids:
            cand = by_id_c.get(int(fid))
            if cand is not None:
                return cand.answer
            faq = by_id_f.get(int(fid))
            if faq is not None:
                return faq.answer
        return ""


# =============================================================================
# Step 2-B — 스몰토크 응답 (kind == smalltalk) — vLLM 1.2B
# =============================================================================


async def _generate_smalltalk(user_message: str) -> str:
    """짧은 몽글이 응대 — FAQ 없이 1~2문장 (vLLM 1.2B)."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPPORT_SMALLTALK_SYSTEM_PROMPT),
            ("human", SUPPORT_SMALLTALK_HUMAN_PROMPT),
        ]
    )
    llm = get_vllm_llm(temperature=0.3)
    try:
        prompt_value = await prompt.ainvoke({"user_message": user_message})
        response = await guarded_ainvoke(
            llm,
            prompt_value,
            model="vllm_exaone_1_2b",
            request_id="support_smalltalk",
        )
        return (getattr(response, "content", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "support_smalltalk_error",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return "안녕하세요! 궁금한 점이 있으면 편하게 말씀해 주세요."


# =============================================================================
# 퍼블릭 — ES + Solar + vLLM 3단계 통합 dispatch
# =============================================================================


async def generate_support_reply(
    user_message: str,
    faqs: list[FaqDoc],
) -> SupportReply:
    """
    ES Nori 검색 → 점수 임계값 분기 → Solar/vLLM 분류 → vLLM 답변 생성.

    시그니처는 v3.2 와 동일 (`faqs` 인수 유지) — nodes.py 호출부 변경 없음.
    `faqs` 는 Solar/vLLM fallback 경로와 답변 생성 시 근거 FAQ 로 활용된다.
    ES 가 주요 경로이므로 `faqs` 가 빈 리스트여도 ES 가 있으면 정상 동작한다.

    ### 처리 흐름
    1. ES Nori 검색 → candidates (top-5, 점수 포함)
    2. 점수 임계값 분기 → SupportPlan (kind, matched_faq_ids)
       - HIGH: 즉시 확정
       - MID~HIGH: Solar 재랭킹
       - LOW: Solar 무매칭 분류
       (Solar 실패 시 vLLM fallback, vLLM 실패 시 complaint 템플릿)
    3. matched_faq_ids 환각 방어
       - ES 경로: candidates 에 존재하는 ID 인지 검증
       - Backend fallback 경로: faqs 에 존재하는 ID 인지 검증
    4. kind 별 답변 생성

    Args:
        user_message: 사용자 발화
        faqs        : context_loader 가 Backend 에서 가져온 FAQ 목록
                      (ES 실패 시 최후 안전망. ES 성공 시에도 답변 생성 근거 보조용)

    Returns:
        SupportReply (항상 유효한 응답 — 에러 전파 금지)
    """
    started = time.perf_counter()
    logger.info(
        "support_reply_start",
        input_preview=user_message[:100],
        faq_count=len(faqs),
    )

    # ─── [1단계] ES Nori BM25 검색 ───────────────────────────────────────────
    candidates = await search_faq_candidates(user_message, top_k=5)

    # ─── [2단계] 점수 임계값 분기 → SupportPlan ──────────────────────────────
    plan = await _determine_plan_from_candidates(user_message, candidates, faqs)

    # ─── [3단계] matched_faq_ids 환각 방어 ───────────────────────────────────
    # ES 경로: candidates 에 있는 ID 인지 검증 (Solar 가 지어낸 id 방지)
    candidate_ids = {c.faq_id for c in candidates}
    # Backend fallback 경로: faqs 에 있는 ID 도 허용 (vLLM fallback 대비)
    faq_ids = {f.faq_id for f in faqs}
    valid_ids = candidate_ids | faq_ids

    cleaned_ids = [fid for fid in plan.matched_faq_ids if int(fid) in valid_ids]

    # kind=faq/partial 인데 유효 id 가 0건이면 complaint 로 강등
    if plan.kind in ("faq", "partial") and not cleaned_ids:
        logger.info(
            "support_reply_demote_to_complaint_no_matches",
            original_kind=plan.kind,
            raw_ids=plan.matched_faq_ids,
        )
        plan = SupportPlan(kind="complaint", matched_faq_ids=[])
        cleaned_ids = []

    # ─── [4단계] kind 별 답변 생성 ────────────────────────────────────────────
    if plan.kind in ("faq", "partial"):
        answer = await _generate_answer_from_faq(
            user_message=user_message,
            faqs=faqs,
            candidates=candidates,
            matched_ids=cleaned_ids,
            match_mode=plan.kind,
        )
        if not answer.strip():
            # LLM 이 빈 답변을 주면 ES 후보 또는 FAQ 원문을 그대로 노출
            by_id_c = {c.faq_id: c for c in candidates}
            by_id_f = {f.faq_id: f for f in faqs}
            first_answer = ""
            for fid in cleaned_ids:
                cand = by_id_c.get(fid)
                if cand:
                    first_answer = cand.answer
                    break
                faq = by_id_f.get(fid)
                if faq:
                    first_answer = faq.answer
                    break
            answer = first_answer or SUPPORT_COMPLAINT_TEMPLATE

        needs_human = plan.kind == "partial"
        reply = SupportReply(
            kind=plan.kind,
            matched_faq_ids=cleaned_ids,
            answer=answer,
            needs_human=needs_human,
        )

    elif plan.kind == "smalltalk":
        answer = await _generate_smalltalk(user_message)
        if not answer.strip():
            answer = "안녕하세요! 궁금한 점이 있으면 편하게 말씀해 주세요."
        reply = SupportReply(
            kind="smalltalk",
            matched_faq_ids=[],
            answer=answer,
            needs_human=False,
        )

    elif plan.kind == "out_of_scope":
        reply = SupportReply(
            kind="out_of_scope",
            matched_faq_ids=[],
            answer=SUPPORT_OUT_OF_SCOPE_TEMPLATE,
            needs_human=False,
        )

    else:
        # complaint (또는 분류 실패 폴백)
        reply = SupportReply(
            kind="complaint",
            matched_faq_ids=[],
            answer=SUPPORT_COMPLAINT_TEMPLATE,
            needs_human=True,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "support_reply_done",
        kind=reply.kind,
        matched_count=len(reply.matched_faq_ids),
        needs_human=reply.needs_human,
        elapsed_ms=round(elapsed_ms, 1),
    )
    return reply
