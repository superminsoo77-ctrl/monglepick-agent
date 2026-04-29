"""
고객센터 AI 에이전트 v4 의도 분류 체인.

Solar Pro API 의 structured output(`method="json_schema"`) 로 사용자 발화를
6종 SupportIntentKind(faq/personal_data/policy/redirect/smalltalk/complaint) 중
하나로 분류한다.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §4

구조:
    발화 → ChatPromptTemplate → structured Solar LLM → SupportIntent

Fallback 정책:
- LLM 호출 실패(타임아웃/API 오류) 시 faq intent 로 폴백.
  "모르면 faq 로 강등 — ES 검색 한 번이라도 시도해야 봇 책임 다함" (설계서 §4.1).
- confidence < 0.5 는 faq 로 보정 — admin_assistant 의 smalltalk 강등과 다름.
  고객센터는 모든 발화에 답변을 시도하므로 가장 범용적인 faq 를 안전 기본값으로.
- 게스트 여부는 의도 분류에 영향을 주지 않음 — 게스트라도 personal_data 의도로
  분류하고 tool handler 에서 login_required 반환. narrator 가 정책 안내 + 로그인 권유.
"""

from __future__ import annotations

import time
import traceback
from typing import Literal

import structlog
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from monglepick.llm.factory import get_structured_llm, guarded_ainvoke
from monglepick.prompts.support_assistant import (
    SUPPORT_INTENT_HUMAN_PROMPT,
    SUPPORT_INTENT_SYSTEM_PROMPT,
)

logger = structlog.get_logger()


# ============================================================
# 출력 스키마 — SupportIntent
# ============================================================


class SupportIntent(BaseModel):
    """
    고객센터 봇 의도 분류 결과.

    kind: 6종 의도 (설계서 §4 표)
    confidence: 분류 신뢰도 (0.0~1.0). 0.5 미만이면 호출 측에서 faq 로 강등.
    reason: 분류 근거 한 줄 한국어 설명 (LangSmith 트레이스·로그용).
    """

    kind: Literal[
        "faq",
        "personal_data",
        "policy",
        "redirect",
        "smalltalk",
        "complaint",
    ]
    confidence: float = Field(ge=0.0, le=1.0, description="분류 신뢰도 0.0~1.0")
    reason: str = Field(description="분류 근거 한 줄 한국어 설명")


# ============================================================
# 프롬프트 템플릿 (모듈 레벨 싱글턴)
# ============================================================

_support_intent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SUPPORT_INTENT_SYSTEM_PROMPT),
        ("human", SUPPORT_INTENT_HUMAN_PROMPT),
    ]
)


# ============================================================
# 신뢰도 하한 — 아래면 faq 로 강제 보정
# ============================================================

_MIN_CONFIDENCE: float = 0.5

# confidence 미달 시 폴백 kind — admin_assistant(smalltalk) 와 의도적으로 다름.
# 고객센터 봇은 "모르면 일단 FAQ 검색 시도" 정책(설계서 §4.1).
_FALLBACK_KIND = "faq"


# ============================================================
# 체인 호출 — 공개 인터페이스
# ============================================================


async def classify_support_intent(
    user_message: str,
    is_guest: bool = False,
    request_id: str = "",
    history_context: str = "",
) -> SupportIntent:
    """
    사용자 발화를 SupportIntent(6종 kind) 로 분류한다.

    Solar Pro API 의 `with_structured_output(method="json_schema")` 로
    SupportIntent Pydantic 모델을 직접 받는다.

    게스트 여부(`is_guest`)는 프롬프트 컨텍스트로 전달되지만
    분류기 자체는 게스트라고 의도를 강등하지 않는다.
    game — 의도 강등은 tool handler 에서 처리(설계서 §4.1, §8.4).

    Args:
        user_message: 사용자가 입력한 발화 (1줄 이상)
        is_guest: True 면 게스트 사용자. 프롬프트에 힌트로 포함 (분류 결과는 무영향).
        request_id: 세마포어/로그용 요청 식별자 (선택)
        history_context: 최근 멀티턴 대화 텍스트 블록.
            "사용자: ... \n몽글이: ... " 형식. 비어 있으면 단일턴으로 분류.
            (2026-04-28 추가) 짧은 후속 질문 의도 정확도 향상을 위함.

    Returns:
        SupportIntent — kind/confidence/reason 3필드.
        실패 또는 confidence < 0.5 시 faq 로 폴백 (에러 전파 금지).
    """
    start = time.perf_counter()
    try:
        llm = get_structured_llm(schema=SupportIntent, temperature=0.1, use_api=True)

        # 멀티턴 컨텍스트 — 프롬프트 내 별도 [이전 대화] 섹션으로 주입.
        # SUPPORT_INTENT_HUMAN_PROMPT 의 placeholder 와 호환되도록 한 줄 prefix 만 추가.
        # ChatPromptTemplate 에 새 변수를 추가하면 기존 호출/테스트가 깨지므로
        # human prompt 본문에 합성하는 방식으로 호환성 유지.
        if history_context:
            user_message_with_context = (
                f"[이전 대화]\n{history_context}\n\n[현재 발화]\n{user_message.strip()}"
            )
        else:
            user_message_with_context = user_message.strip()

        # `_support_intent_prompt | llm` chain 패턴을 쓰지 않는 이유:
        # 단위 테스트에서 `with_structured_output()` 이 반환하는 MagicMock 에 `|` 연산자가
        # 없어 RunnableSequence 구성이 깨지기 때문. llm 직접 호출은 mock 과 실 모듈 모두 호환.
        # (admin_intent_chain.py 동일 패턴 참조)
        prompt_value = _support_intent_prompt.format_prompt(
            user_message=user_message_with_context,
            is_guest="게스트(비로그인)" if is_guest else "로그인 사용자",
        )
        result = await guarded_ainvoke(
            llm,
            prompt_value,
            model="solar_api",
            request_id=request_id or "support_intent",
        )

        # LLM 이 BaseModel 이 아닌 dict 를 돌려주는 경로 대비 graceful 파싱
        if not isinstance(result, SupportIntent):
            result = SupportIntent.model_validate(result)

        # confidence 하한 보정 — admin 과 달리 faq 로 강등
        if result.confidence < _MIN_CONFIDENCE and result.kind != _FALLBACK_KIND:
            logger.info(
                "support_intent_low_confidence_fallback_to_faq",
                kind=result.kind,
                confidence=result.confidence,
                reason=result.reason,
                is_guest=is_guest,
            )
            result = SupportIntent(
                kind=_FALLBACK_KIND,
                confidence=result.confidence,
                reason=f"low_confidence_fallback (원래: {result.kind}, {result.reason})",
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "support_intent_classified",
            kind=result.kind,
            confidence=round(result.confidence, 2),
            reason=result.reason[:80],
            is_guest=is_guest,
            elapsed_ms=round(elapsed_ms, 1),
        )
        return result

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "support_intent_classify_failed_fallback_faq",
            error=str(e),
            error_type=type(e).__name__,
            is_guest=is_guest,
            elapsed_ms=round(elapsed_ms, 1),
            stack_trace=traceback.format_exc(),
        )
        # 에러 전파 금지 (설계서 §3 "에러: 모든 노드/체인 try/except, 실패 시 fallback 반환")
        return SupportIntent(
            kind=_FALLBACK_KIND,
            confidence=0.0,
            reason=f"classify_error:{type(e).__name__}",
        )
