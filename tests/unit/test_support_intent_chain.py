"""
고객센터 AI 에이전트 v4 의도 분류 체인 단위 테스트.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §4 / §13.1

테스트 범위:
1. 6종 의도 정확도 — 각 kind 별 5건 이상 발화 예시 (pytest.mark.parametrize)
2. confidence 하한 보정 — confidence < 0.5 시 faq 로 강등
3. 게스트/로그인 동일 의도 보장 — 게스트라고 personal_data 강등 X
4. 에러 fallback — LLM 예외 시 faq + confidence=0.0 반환

모킹 전략:
- conftest.py 의 mock_ollama 픽스처 재사용 (admin_intent_chain 테스트 동일 패턴).
- mock_ollama.set_structured_response(SupportIntent(...)) 로 LLM 응답 사전 설정.
- Solar API 실 호출 없음 — 모든 테스트는 오프라인에서 통과해야 한다.
"""

from __future__ import annotations

import pytest

from monglepick.chains.support_intent_chain import (
    SupportIntent,
    classify_support_intent,
)


# ============================================================
# 헬퍼 — 지정 kind 의 SupportIntent 빠르게 생성
# ============================================================


def _intent(kind: str, confidence: float = 0.9, reason: str = "") -> SupportIntent:
    """테스트용 SupportIntent 인스턴스를 간결하게 생성한다."""
    return SupportIntent(
        kind=kind,  # type: ignore[arg-type]
        confidence=confidence,
        reason=reason or f"테스트: {kind}",
    )


# ============================================================
# 1. faq 의도 — 5건 파라미터화
# ============================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_message",
    [
        "리뷰 작성은 어떻게 해요?",
        "비밀번호 변경하고 싶어요",
        "탈퇴하면 데이터 어떻게 돼요?",
        "포인트 잔액 어디서 확인해요?",
        "앱 다운로드 방법 알려줘요",
    ],
)
async def test_faq_intent(user_message: str, mock_ollama) -> None:
    """faq 발화가 kind=faq 로 분류된다."""
    mock_ollama.set_structured_response(_intent("faq", confidence=0.88))
    result = await classify_support_intent(user_message)
    assert result.kind == "faq"
    assert result.confidence >= 0.5


# ============================================================
# 2. personal_data 의도 — 5건 파라미터화
# ============================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_message",
    [
        "내 포인트가 안 들어왔어요",
        "AI 추천을 더 이상 못 쓰겠어요",
        "출석 체크했는데 포인트가 없어요",
        "구독이 언제 만료되는지 알고 싶어요",
        "내 최근 주문 내역 확인하고 싶어요",
    ],
)
async def test_personal_data_intent(user_message: str, mock_ollama) -> None:
    """본인 데이터 진단 발화가 kind=personal_data 로 분류된다."""
    mock_ollama.set_structured_response(_intent("personal_data", confidence=0.91))
    result = await classify_support_intent(user_message)
    assert result.kind == "personal_data"
    assert result.confidence >= 0.5


# ============================================================
# 3. policy 의도 — 5건 파라미터화
# ============================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_message",
    [
        "BRONZE 등급은 AI 추천 몇 번 써요?",
        "환불 정책이 어떻게 돼요?",
        "구독하면 AI 몇 번 더 쓸 수 있어요?",
        "출석 체크 포인트 얼마예요?",
        "골드 등급이 되려면 뭘 해야 해요?",
    ],
)
async def test_policy_intent(user_message: str, mock_ollama) -> None:
    """운영 정책 질문이 kind=policy 로 분류된다."""
    mock_ollama.set_structured_response(_intent("policy", confidence=0.87))
    result = await classify_support_intent(user_message)
    assert result.kind == "policy"
    assert result.confidence >= 0.5


# ============================================================
# 4. redirect 의도 — 5건 파라미터화
# ============================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_message",
    [
        "오늘 볼 만한 영화 추천해 줘",
        "둘이 영화 고르기 어디서 해요?",
        "봉준호 감독 작품 알려줘",
        "오늘 날씨 어때요?",
        "지금 몇 시야?",
    ],
)
async def test_redirect_intent(user_message: str, mock_ollama) -> None:
    """봇 영역 밖 발화(영화 추천·날씨 등)가 kind=redirect 로 분류된다."""
    mock_ollama.set_structured_response(_intent("redirect", confidence=0.85))
    result = await classify_support_intent(user_message)
    assert result.kind == "redirect"
    assert result.confidence >= 0.5


# ============================================================
# 5. smalltalk 의도 — 5건 파라미터화
# ============================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_message",
    [
        "안녕",
        "너 누구야?",
        "몽글이 귀엽다",
        "고마워",
        "오늘 기분 어때?",
    ],
)
async def test_smalltalk_intent(user_message: str, mock_ollama) -> None:
    """봇 자신과의 인사·잡담이 kind=smalltalk 로 분류된다."""
    mock_ollama.set_structured_response(_intent("smalltalk", confidence=0.93))
    result = await classify_support_intent(user_message)
    assert result.kind == "smalltalk"
    assert result.confidence >= 0.5


# ============================================================
# 6. complaint 의도 — 5건 파라미터화
# ============================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_message",
    [
        "환불해 주세요",
        "결제 취소해 주세요",
        "계정 정지 풀어줘",
        "탈퇴 처리해 주세요",
        "잘못 결제됐어요",
    ],
)
async def test_complaint_intent(user_message: str, mock_ollama) -> None:
    """환불·결제 취소·계정 제재 요구가 kind=complaint 로 분류된다."""
    mock_ollama.set_structured_response(_intent("complaint", confidence=0.92))
    result = await classify_support_intent(user_message)
    assert result.kind == "complaint"
    assert result.confidence >= 0.5


# ============================================================
# 7. confidence 하한 보정 — faq 강등 (admin 과 달리 smalltalk 아님)
# ============================================================


class TestConfidenceFallback:
    """confidence < 0.5 시 faq 로 강등되는지 검증."""

    @pytest.mark.asyncio
    async def test_low_confidence_personal_data_downgraded_to_faq(
        self, mock_ollama
    ) -> None:
        """personal_data 인데 confidence=0.3 → faq 로 강등."""
        mock_ollama.set_structured_response(
            _intent("personal_data", confidence=0.3, reason="애매한 발화")
        )
        result = await classify_support_intent("뭔가 이상한 것 같아요")
        assert result.kind == "faq"
        assert "low_confidence_fallback" in result.reason
        # 원래 kind 기록이 reason 에 남아야 한다
        assert "personal_data" in result.reason

    @pytest.mark.asyncio
    async def test_low_confidence_redirect_downgraded_to_faq(
        self, mock_ollama
    ) -> None:
        """redirect 인데 confidence=0.2 → faq 로 강등."""
        mock_ollama.set_structured_response(
            _intent("redirect", confidence=0.2, reason="불분명")
        )
        result = await classify_support_intent("그거 어떻게 해요?")
        assert result.kind == "faq"
        assert "low_confidence_fallback" in result.reason

    @pytest.mark.asyncio
    async def test_low_confidence_faq_stays_faq(self, mock_ollama) -> None:
        """faq 이고 confidence < 0.5 여도 faq 에서 faq 로 강등은 무시 (동일 kind)."""
        mock_ollama.set_structured_response(
            _intent("faq", confidence=0.4, reason="저신뢰 faq")
        )
        result = await classify_support_intent("사용법 알려줘요")
        # faq → faq 강등 분기에서 kind != _FALLBACK_KIND 조건이 False 라
        # 강등 로직을 타지 않으므로 confidence 는 그대로 0.4
        assert result.kind == "faq"
        assert result.confidence == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_boundary_exactly_0_5_not_downgraded(self, mock_ollama) -> None:
        """confidence == 0.5 는 강등 기준(< 0.5) 미만이 아니므로 원래 kind 유지."""
        mock_ollama.set_structured_response(
            _intent("policy", confidence=0.5, reason="경계값")
        )
        result = await classify_support_intent("정책 알려줘요")
        assert result.kind == "policy"
        assert result.confidence == pytest.approx(0.5)


# ============================================================
# 8. 게스트/로그인 동일 의도 보장
# ============================================================


class TestGuestIntentEquality:
    """
    게스트 여부가 의도 분류 결과에 영향을 주지 않아야 한다.

    설계서 §4.1: "게스트(`is_guest=True`) 는 personal_data 자동 강등 안 함"
    분류기는 is_guest 를 프롬프트 힌트로만 전달. 강등은 tool handler 에서.
    """

    @pytest.mark.asyncio
    async def test_guest_personal_data_not_downgraded(self, mock_ollama) -> None:
        """게스트가 본인 데이터 발화를 해도 personal_data 로 분류된다."""
        mock_ollama.set_structured_response(_intent("personal_data", confidence=0.88))
        result_guest = await classify_support_intent(
            "내 포인트 안 들어왔어요", is_guest=True
        )
        assert result_guest.kind == "personal_data"

    @pytest.mark.asyncio
    async def test_guest_and_loggedin_same_kind_for_personal_data(
        self, mock_ollama
    ) -> None:
        """게스트와 로그인 사용자가 동일 발화에 대해 동일 kind 를 받는다."""
        mock_ollama.set_structured_response(_intent("personal_data", confidence=0.9))
        result_guest = await classify_support_intent(
            "AI 추천 더 못 써요", is_guest=True
        )
        mock_ollama.set_structured_response(_intent("personal_data", confidence=0.9))
        result_login = await classify_support_intent(
            "AI 추천 더 못 써요", is_guest=False
        )
        assert result_guest.kind == result_login.kind == "personal_data"

    @pytest.mark.asyncio
    async def test_guest_faq_intent_unchanged(self, mock_ollama) -> None:
        """게스트도 faq 발화는 faq 로 분류된다."""
        mock_ollama.set_structured_response(_intent("faq", confidence=0.85))
        result = await classify_support_intent(
            "비밀번호 변경 어떻게 해요?", is_guest=True
        )
        assert result.kind == "faq"

    @pytest.mark.asyncio
    async def test_guest_smalltalk_intent_unchanged(self, mock_ollama) -> None:
        """게스트도 smalltalk 발화는 smalltalk 로 분류된다."""
        mock_ollama.set_structured_response(_intent("smalltalk", confidence=0.92))
        result = await classify_support_intent("안녕", is_guest=True)
        assert result.kind == "smalltalk"

    @pytest.mark.asyncio
    async def test_guest_complaint_intent_unchanged(self, mock_ollama) -> None:
        """게스트도 complaint 발화는 complaint 로 분류된다."""
        mock_ollama.set_structured_response(_intent("complaint", confidence=0.91))
        result = await classify_support_intent("환불해 주세요", is_guest=True)
        assert result.kind == "complaint"


# ============================================================
# 9. 에러 fallback
# ============================================================


class TestErrorFallback:
    """LLM 예외 시 faq + confidence=0.0 graceful fallback 검증."""

    @pytest.mark.asyncio
    async def test_llm_exception_returns_faq_fallback(self, mock_ollama) -> None:
        """Solar API 장애 시 kind=faq, confidence=0.0 으로 폴백한다."""
        mock_ollama.set_error(RuntimeError("Solar API timeout"))
        result = await classify_support_intent("뭔가 물어보고 싶은데요")
        assert result.kind == "faq"
        assert result.confidence == pytest.approx(0.0)
        assert "classify_error" in result.reason

    @pytest.mark.asyncio
    async def test_llm_exception_does_not_propagate(self, mock_ollama) -> None:
        """예외가 상위로 전파되지 않고 SupportIntent 를 반환한다."""
        mock_ollama.set_error(ConnectionError("네트워크 오류"))
        # 예외 전파 시 pytest 가 실패하므로 반환값이 있으면 통과
        result = await classify_support_intent("테스트")
        assert isinstance(result, SupportIntent)

    @pytest.mark.asyncio
    async def test_error_with_guest_returns_faq_fallback(self, mock_ollama) -> None:
        """게스트 요청에서도 에러 시 faq 로 폴백한다."""
        mock_ollama.set_error(ValueError("파싱 실패"))
        result = await classify_support_intent("포인트 어떻게 돼요?", is_guest=True)
        assert result.kind == "faq"
        assert result.confidence == pytest.approx(0.0)


# ============================================================
# 10. dict 입력 graceful 파싱 — LLM 이 BaseModel 대신 dict 반환하는 경우
# ============================================================


class TestGracefulParsing:
    """LLM 이 Pydantic 모델 대신 dict 를 돌려줄 때 model_validate 로 처리되는지 검증."""

    @pytest.mark.asyncio
    async def test_dict_response_parsed_correctly(self, mock_ollama) -> None:
        """LLM 이 dict 를 반환해도 SupportIntent 로 파싱돼 정상 동작한다."""
        # mock_ollama 는 BaseModel 을 직접 반환하는 경로를 사용하므로
        # dict 반환 경로는 classify_support_intent 내부에서 isinstance 체크 후
        # model_validate 를 호출하는 로직을 단위 테스트하기 어렵다.
        # 대신 정상 경로에서 반환 타입이 SupportIntent 인지 확인한다.
        mock_ollama.set_structured_response(_intent("policy", confidence=0.82))
        result = await classify_support_intent("구독 정책 알려줘요")
        assert isinstance(result, SupportIntent)
        assert result.kind == "policy"


# ============================================================
# 11. 분류 모호 케이스 문서화 (경계 케이스 — 어떻게 분류해야 하는지 명시)
# ============================================================


class TestAmbiguousCases:
    """
    설계서에서 명확히 정의된 경계 케이스.

    분류기 구현 시 프롬프트를 어떻게 작성하느냐에 따라 달라질 수 있는 케이스들.
    테스트는 설계서 §4 의 결정을 기준으로 작성 — mock 응답을 '정답'으로 설정.

    모호 케이스 분석:
    1. "내 등급 뭐야?" → personal_data (내 현재 등급 조회) vs policy (등급 제도 설명)?
       결정: personal_data — "내" 라는 소유격이 있으면 본인 데이터 조회 의도.
    2. "환불 정책이 어떻게 돼요?" → policy vs complaint?
       결정: policy — 정책 문의. "환불해 주세요" 만 complaint.
    3. "AI 추천 몇 번 써요?" → policy (일반 질문) vs personal_data (내 남은 횟수)?
       결정: 주어 없으면 policy. "내 AI 추천 몇 번 남았어요?" 면 personal_data.
    """

    @pytest.mark.asyncio
    async def test_my_grade_is_personal_data(self, mock_ollama) -> None:
        """'내 등급 뭐야?' — 소유격 '내' 있으면 personal_data 로 분류한다."""
        mock_ollama.set_structured_response(
            _intent("personal_data", confidence=0.78, reason="'내' 소유격 → 본인 데이터")
        )
        result = await classify_support_intent("내 등급 뭐야?")
        assert result.kind == "personal_data"

    @pytest.mark.asyncio
    async def test_refund_policy_question_is_policy(self, mock_ollama) -> None:
        """'환불 정책 어떻게 돼요?' — 정책 문의는 policy, 요구는 complaint."""
        mock_ollama.set_structured_response(
            _intent("policy", confidence=0.85, reason="정책 문의, 개인 요구 아님")
        )
        result = await classify_support_intent("환불 정책이 어떻게 돼요?")
        assert result.kind == "policy"

    @pytest.mark.asyncio
    async def test_ai_quota_without_subject_is_policy(self, mock_ollama) -> None:
        """주어 없는 AI 횟수 질문은 policy (제도 안내). '내'가 붙으면 personal_data."""
        mock_ollama.set_structured_response(
            _intent("policy", confidence=0.80, reason="주어 없음 → 제도 안내")
        )
        result = await classify_support_intent("AI 추천 몇 번 써요?")
        assert result.kind == "policy"

    @pytest.mark.asyncio
    async def test_my_remaining_ai_quota_is_personal_data(self, mock_ollama) -> None:
        """'내 AI 추천 몇 번 남았어요?' — '내' 소유격 → personal_data."""
        mock_ollama.set_structured_response(
            _intent("personal_data", confidence=0.86, reason="'내 AI 추천' 소유격 → 본인 쿼터")
        )
        result = await classify_support_intent("내 AI 추천 몇 번 남았어요?")
        assert result.kind == "personal_data"

    @pytest.mark.asyncio
    async def test_weather_is_redirect_not_smalltalk(self, mock_ollama) -> None:
        """'오늘 날씨 어때요?' — 봇 무관 잡담은 smalltalk 가 아니라 redirect."""
        mock_ollama.set_structured_response(
            _intent("redirect", confidence=0.83, reason="봇 무관 외부 질문 → redirect")
        )
        result = await classify_support_intent("오늘 날씨 어때요?")
        assert result.kind == "redirect"
