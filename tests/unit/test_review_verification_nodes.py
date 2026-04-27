"""
도장깨기 리뷰 검증 에이전트 단위 테스트 (Step D, 2026-04-27).

대상 모듈: `monglepick.agents.review_verification.nodes`

테스트 영역 (5 노드 + 헬퍼 + 상수 외부화):
1. _clean_text       — HTML/마크다운 제거, 1500자 truncate
2. _extract_words    — 한글 2글자 이상 + 영문 대문자 시작 추출
3. preprocessor      — 정상 정제 / 20자 미만 early_exit / 예외 안전 fallback
4. embedding_similarity — 정상 코사인 유사도 / early_exit pass / 예외 시 0.0
5. keyword_matcher   — 교집합 / 스탑워드 제거 / early_exit pass / 예외 swallow
6. llm_revalidator   — 구간 안 LLM 호출 / 구간 밖 미호출 / YES/NO/모호 / 예외
7. threshold_decider — ≥HIGH AUTO_VERIFIED / 중간 NEEDS_REVIEW / <LOW AUTO_REJECTED / early_exit / 예외
8. 상수 외부화       — settings 로부터 HIGH/LOW/LLM_CALL 임계값이 올바르게 주입됨

Solar 임베딩과 LLM 은 모두 mock — CI 환경에서 인프라 의존성 0.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from monglepick.agents.review_verification import nodes as rv_nodes
from monglepick.agents.review_verification.nodes import (
    _LLM_CALL_HIGH,
    _LLM_CALL_LOW,
    _THRESHOLD_HIGH,
    _THRESHOLD_LOW,
    _clean_text,
    _extract_words,
    embedding_similarity,
    keyword_matcher,
    llm_revalidator,
    preprocessor,
    threshold_decider,
)


# ============================================================
# 0) 상수 외부화 — settings 와 동일해야 함
# ============================================================

class TestThresholdsExternalized:
    """nodes 의 상수가 settings 에서 오는지 검증 — 운영 튜닝 시 env 만 바꾸면 됨."""

    def test_high_low_match_settings(self):
        from monglepick.config import settings

        assert _THRESHOLD_HIGH == settings.REVIEW_VERIFICATION_THRESHOLD_HIGH
        assert _THRESHOLD_LOW == settings.REVIEW_VERIFICATION_THRESHOLD_LOW
        assert _LLM_CALL_LOW == settings.REVIEW_VERIFICATION_LLM_CALL_LOW
        assert _LLM_CALL_HIGH == settings.REVIEW_VERIFICATION_LLM_CALL_HIGH

    def test_default_values(self):
        """기본값 — 설계서 §4 와 동일."""
        assert _THRESHOLD_HIGH == 0.7
        assert _THRESHOLD_LOW == 0.3
        assert _LLM_CALL_LOW == 0.5
        assert _LLM_CALL_HIGH == 0.8

    def test_threshold_ordering(self):
        """LOW < LLM_CALL_LOW <= LLM_CALL_HIGH < HIGH 순서가 깨지면 로직 결함."""
        assert _THRESHOLD_LOW < _LLM_CALL_LOW
        assert _LLM_CALL_LOW <= _LLM_CALL_HIGH
        assert _LLM_CALL_HIGH < 1.0
        assert _THRESHOLD_HIGH > _LLM_CALL_LOW


# ============================================================
# 1) _clean_text — HTML/마크다운 정제
# ============================================================

class TestCleanText:
    def test_html_tags_removed(self):
        assert "script" not in _clean_text("<script>alert(1)</script>안녕")
        assert _clean_text("<p>본문</p>") == "본문"

    def test_markdown_special_chars_removed(self):
        cleaned = _clean_text("**굵게** _기울임_ `코드` ## 헤더")
        assert "**" not in cleaned
        assert "_" not in cleaned
        assert "`" not in cleaned
        assert "굵게" in cleaned

    def test_whitespace_collapsed(self):
        assert _clean_text("a  b   c\n\nd") == "a b c d"

    def test_truncate_long_text(self):
        long = "가" * 2000
        assert len(_clean_text(long)) == 1500

    def test_empty_input(self):
        assert _clean_text("") == ""


# ============================================================
# 2) _extract_words — 한글 + 영문 고유명사 추출
# ============================================================

class TestExtractWords:
    def test_korean_2chars_plus(self):
        words = _extract_words("기생충은 가족 이야기")
        assert "기생충은" in words or "기생충" in words
        assert "가족" in words
        assert "이야기" in words

    def test_filter_single_korean_char(self):
        """1글자 한글은 제외."""
        words = _extract_words("이 그 저")
        assert words == []

    def test_english_capitalized_only(self):
        """소문자 시작 영문은 제외, 대문자 시작 고유명사만."""
        words = _extract_words("Apple와 google 비교")
        assert "Apple" in words
        assert "google" not in words

    def test_mixed_korean_english(self):
        words = _extract_words("기생충 Parasite 봉준호 Bong")
        assert "기생충" in words
        assert "Parasite" in words
        assert "Bong" in words


# ============================================================
# 3) preprocessor — 정제 + early_exit
# ============================================================

@pytest.mark.asyncio
class TestPreprocessor:
    async def test_normal_review_proceeds(self):
        state = {
            "verification_id": 1,
            "review_text": "이 영화는 정말 흥미로웠다. 봉준호 감독의 연출력이 돋보였고 송강호의 연기가 압권이었다.",
            "movie_plot": "기택 가족이 박 사장 집에 차례로 침투한다",
        }
        result = await preprocessor(state)
        assert result["early_exit"] is False
        assert "clean_review" in result
        assert "clean_plot" in result

    async def test_short_review_triggers_early_exit(self):
        """20자 미만 리뷰 → early_exit=True + NEEDS_REVIEW 강등."""
        state = {
            "verification_id": 1,
            "review_text": "재밌어요",  # 4자
            "movie_plot": "줄거리",
        }
        result = await preprocessor(state)
        assert result["early_exit"] is True
        assert result["review_status"] == "NEEDS_REVIEW"
        assert result["confidence"] == 0.0
        assert "너무 짧" in result["rationale"]

    async def test_html_stripped_in_preprocessor(self):
        state = {
            "verification_id": 1,
            "review_text": "<p>긴 리뷰" + "내용" * 20 + "</p>",
            "movie_plot": "줄거리",
        }
        result = await preprocessor(state)
        assert "<p>" not in result.get("clean_review", "")
        assert result["early_exit"] is False

    async def test_exception_returns_safe_fallback(self):
        """내부 예외 시 안전 fallback (early_exit=False, 본문 그대로)."""
        with patch.object(rv_nodes, "_clean_text", side_effect=Exception("boom")):
            result = await preprocessor({
                "verification_id": 1,
                "review_text": "리뷰" * 30,
                "movie_plot": "줄거리",
            })
        assert result["early_exit"] is False
        assert "clean_review" in result


# ============================================================
# 4) embedding_similarity — Solar 코사인 유사도
# ============================================================

@pytest.mark.asyncio
class TestEmbeddingSimilarity:
    async def test_identical_vectors_similarity_1(self):
        """같은 벡터 → 코사인 유사도 ≈ 1."""
        v = np.ones(4096) / np.linalg.norm(np.ones(4096))

        with (
            patch.object(rv_nodes, "embed_query_async", new=AsyncMock(return_value=v)),
            patch.object(rv_nodes, "_embed_passage_async", new=AsyncMock(return_value=v)),
        ):
            result = await embedding_similarity({
                "early_exit": False,
                "clean_review": "리뷰",
                "clean_plot": "줄거리",
                "verification_id": 1,
            })

        assert result["similarity_score"] == pytest.approx(1.0, abs=1e-3)

    async def test_orthogonal_vectors_similarity_0(self):
        """직교 벡터 → 코사인 유사도 ≈ 0."""
        v1 = np.zeros(4096)
        v1[0] = 1.0
        v2 = np.zeros(4096)
        v2[1] = 1.0

        with (
            patch.object(rv_nodes, "embed_query_async", new=AsyncMock(return_value=v1)),
            patch.object(rv_nodes, "_embed_passage_async", new=AsyncMock(return_value=v2)),
        ):
            result = await embedding_similarity({
                "early_exit": False,
                "clean_review": "a",
                "clean_plot": "b",
                "verification_id": 1,
            })

        assert result["similarity_score"] == pytest.approx(0.0, abs=1e-6)

    async def test_clipped_to_zero_one(self):
        """음수 / 1 초과 → [0, 1] 클램프."""
        v_neg = np.zeros(4096)
        v_neg[0] = -1.0
        v_pos = np.zeros(4096)
        v_pos[0] = 1.0

        with (
            patch.object(rv_nodes, "embed_query_async", new=AsyncMock(return_value=v_neg)),
            patch.object(rv_nodes, "_embed_passage_async", new=AsyncMock(return_value=v_pos)),
        ):
            result = await embedding_similarity({
                "early_exit": False,
                "clean_review": "x", "clean_plot": "y", "verification_id": 1,
            })

        assert 0.0 <= result["similarity_score"] <= 1.0

    async def test_early_exit_returns_empty(self):
        result = await embedding_similarity({"early_exit": True, "verification_id": 1})
        assert result == {}

    async def test_embedding_failure_returns_zero(self):
        """Solar API 장애 시 similarity_score=0.0 안전 폴백."""
        with patch.object(rv_nodes, "embed_query_async", new=AsyncMock(side_effect=Exception("API down"))):
            result = await embedding_similarity({
                "early_exit": False,
                "clean_review": "x", "clean_plot": "y", "verification_id": 1,
            })

        assert result["similarity_score"] == 0.0


# ============================================================
# 5) keyword_matcher — 교집합 + 스탑워드
# ============================================================

@pytest.mark.asyncio
class TestKeywordMatcher:
    async def test_intersection_calculated(self):
        """겹치는 단어가 keyword_score 에 반영."""
        state = {
            "early_exit": False,
            "verification_id": 1,
            "clean_review": "기생충 봉준호 송강호 가족 침투",
            "clean_plot": "기생충 봉준호 송강호 가족 박사장",
        }
        result = await keyword_matcher(state)
        assert "기생충" in result["matched_keywords"]
        assert "봉준호" in result["matched_keywords"]
        assert result["keyword_score"] > 0.0

    async def test_stopwords_filtered(self):
        """'영화', '재미' 등은 교집합에서 제외."""
        state = {
            "early_exit": False,
            "verification_id": 1,
            "clean_review": "이 영화 정말 재미있다 너무 좋았다",
            "clean_plot": "이 영화 진짜 재미 너무 좋다",
        }
        result = await keyword_matcher(state)
        # 모두 스탑워드라 매칭 0
        assert result["matched_keywords"] == []
        assert result["keyword_score"] == 0.0

    async def test_score_capped_at_one(self):
        """교집합 5개 이상 → keyword_score = 1.0."""
        # 한글 2글자 이상 + 스탑워드 아닌 단어 10개 (정규식 [가-힣]{2,} 매칭)
        words = "기생충 봉준호 송강호 박사장 침투 비밀 지하 비극 인물 변두리"
        state = {
            "early_exit": False,
            "verification_id": 1,
            "clean_review": words,
            "clean_plot": words,
        }
        result = await keyword_matcher(state)
        assert result["keyword_score"] == 1.0
        assert len(result["matched_keywords"]) >= 5

    async def test_no_intersection_zero_score(self):
        state = {
            "early_exit": False,
            "verification_id": 1,
            "clean_review": "사과 바나나 포도",
            "clean_plot": "주연 배우 신인",
        }
        result = await keyword_matcher(state)
        assert result["matched_keywords"] == []
        assert result["keyword_score"] == 0.0

    async def test_early_exit_returns_empty(self):
        result = await keyword_matcher({"early_exit": True, "verification_id": 1})
        assert result == {}


# ============================================================
# 6) llm_revalidator — 구간별 호출/미호출 + YES/NO 조정
# ============================================================

@pytest.mark.asyncio
class TestLlmRevalidator:
    async def test_below_lower_band_skips_llm(self):
        """confidence_draft < LOW → LLM 미호출."""
        state = {
            "early_exit": False, "verification_id": 1,
            "clean_review": "x", "clean_plot": "y",
            "similarity_score": 0.1, "keyword_score": 0.0,
        }
        # confidence_draft = 0.7 * 0.1 + 0.3 * 0.0 = 0.07 (< 0.5)
        with patch.object(rv_nodes, "get_conversation_llm") as mock_llm:
            result = await llm_revalidator(state)
            mock_llm.assert_not_called()

        assert result["llm_adjustment"] == 0.0

    async def test_above_upper_band_skips_llm(self):
        """confidence_draft > HIGH → LLM 미호출 (이미 명확)."""
        state = {
            "early_exit": False, "verification_id": 1,
            "clean_review": "x", "clean_plot": "y",
            "similarity_score": 1.0, "keyword_score": 1.0,
        }
        # confidence_draft = 0.7 * 1.0 + 0.3 * 1.0 = 1.0 (> 0.8)
        with patch.object(rv_nodes, "get_conversation_llm") as mock_llm:
            result = await llm_revalidator(state)
            mock_llm.assert_not_called()

        assert result["llm_adjustment"] == 0.0

    async def test_in_band_calls_llm_and_yes_adjusts_plus(self):
        """구간 안 + YES → adjustment +0.1."""
        state = {
            "early_exit": False, "verification_id": 1,
            "clean_review": "x", "clean_plot": "y",
            "similarity_score": 0.7, "keyword_score": 0.5,
        }
        # confidence_draft = 0.7 * 0.7 + 0.3 * 0.5 = 0.64 (구간 안)
        fake_response = MagicMock()
        fake_response.content = "YES"

        with (
            patch.object(rv_nodes, "get_conversation_llm", return_value=MagicMock()),
            patch.object(rv_nodes, "guarded_ainvoke", new=AsyncMock(return_value=fake_response)),
        ):
            result = await llm_revalidator(state)

        assert result["llm_adjustment"] == 0.1

    async def test_in_band_no_adjusts_minus(self):
        state = {
            "early_exit": False, "verification_id": 1,
            "clean_review": "x", "clean_plot": "y",
            "similarity_score": 0.7, "keyword_score": 0.5,
        }
        fake_response = MagicMock()
        fake_response.content = "NO"

        with (
            patch.object(rv_nodes, "get_conversation_llm", return_value=MagicMock()),
            patch.object(rv_nodes, "guarded_ainvoke", new=AsyncMock(return_value=fake_response)),
        ):
            result = await llm_revalidator(state)

        assert result["llm_adjustment"] == -0.2

    async def test_in_band_ambiguous_response_zero_adjust(self):
        """LLM 응답에 YES/NO 둘 다 없으면 중립(0)."""
        state = {
            "early_exit": False, "verification_id": 1,
            "clean_review": "x", "clean_plot": "y",
            "similarity_score": 0.7, "keyword_score": 0.5,
        }
        fake_response = MagicMock()
        fake_response.content = "음... 잘 모르겠음"

        with (
            patch.object(rv_nodes, "get_conversation_llm", return_value=MagicMock()),
            patch.object(rv_nodes, "guarded_ainvoke", new=AsyncMock(return_value=fake_response)),
        ):
            result = await llm_revalidator(state)

        assert result["llm_adjustment"] == 0.0

    async def test_llm_exception_neutral(self):
        """LLM 호출 실패 → adjustment=0.0 으로 중립 처리, 에러 전파 X."""
        state = {
            "early_exit": False, "verification_id": 1,
            "clean_review": "x", "clean_plot": "y",
            "similarity_score": 0.7, "keyword_score": 0.5,
        }
        with (
            patch.object(rv_nodes, "get_conversation_llm", return_value=MagicMock()),
            patch.object(rv_nodes, "guarded_ainvoke", new=AsyncMock(side_effect=Exception("timeout"))),
        ):
            result = await llm_revalidator(state)

        assert result["llm_adjustment"] == 0.0

    async def test_early_exit_returns_empty(self):
        result = await llm_revalidator({"early_exit": True, "verification_id": 1})
        assert result == {}


# ============================================================
# 7) threshold_decider — 최종 분기
# ============================================================

@pytest.mark.asyncio
class TestThresholdDecider:
    async def test_high_confidence_auto_verified(self):
        state = {
            "early_exit": False, "verification_id": 1,
            "confidence_draft": 0.8, "llm_adjustment": 0.1,
            "similarity_score": 0.85, "matched_keywords": ["기생충", "봉준호"],
        }
        # final = clip(0.8 + 0.1, 0, 1) = 0.9 ≥ 0.7
        result = await threshold_decider(state)
        assert result["review_status"] == "AUTO_VERIFIED"
        assert result["confidence"] == pytest.approx(0.9)
        assert "자동 승인" in result["rationale"]

    async def test_mid_confidence_needs_review(self):
        state = {
            "early_exit": False, "verification_id": 1,
            "confidence_draft": 0.5, "llm_adjustment": 0.0,
            "similarity_score": 0.5, "matched_keywords": ["기생충"],
        }
        result = await threshold_decider(state)
        assert result["review_status"] == "NEEDS_REVIEW"
        assert "검수" in result["rationale"]

    async def test_low_confidence_auto_rejected(self):
        state = {
            "early_exit": False, "verification_id": 1,
            "confidence_draft": 0.2, "llm_adjustment": -0.1,
            "similarity_score": 0.15, "matched_keywords": [],
        }
        # final = clip(0.2 + (-0.1), 0, 1) = 0.1 < 0.3
        result = await threshold_decider(state)
        assert result["review_status"] == "AUTO_REJECTED"
        assert "자동 반려" in result["rationale"]

    async def test_clamps_negative_to_zero(self):
        state = {
            "early_exit": False, "verification_id": 1,
            "confidence_draft": 0.1, "llm_adjustment": -0.5,
            "similarity_score": 0.0, "matched_keywords": [],
        }
        result = await threshold_decider(state)
        assert result["confidence"] == 0.0

    async def test_clamps_overflow_to_one(self):
        state = {
            "early_exit": False, "verification_id": 1,
            "confidence_draft": 0.95, "llm_adjustment": 0.5,
            "similarity_score": 1.0, "matched_keywords": ["기생충"],
        }
        result = await threshold_decider(state)
        assert result["confidence"] == 1.0
        assert result["review_status"] == "AUTO_VERIFIED"

    async def test_early_exit_returns_empty(self):
        result = await threshold_decider({"early_exit": True, "verification_id": 1})
        assert result == {}

    async def test_exception_falls_back_to_needs_review(self):
        """내부 예외 시 안전한 fallback — NEEDS_REVIEW 로 강등."""
        # state.get 이 예외를 던지도록 mocked dict
        class BrokenDict(dict):
            def get(self, key, default=None):
                if key == "confidence_draft":
                    raise RuntimeError("boom")
                return super().get(key, default)

        state = BrokenDict({"early_exit": False, "verification_id": 1})
        result = await threshold_decider(state)
        assert result["review_status"] == "NEEDS_REVIEW"
        assert result["confidence"] == 0.0
