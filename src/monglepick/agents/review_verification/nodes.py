"""
리뷰 검증 에이전트 노드 함수.

5개 노드 순차 실행:
  preprocessor         → HTML/마크다운 제거, 1500자 truncate, 20자 미만 early_exit
  embedding_similarity → Solar 임베딩 코사인 유사도
  keyword_matcher      → 한국어 명사 추출 + 스탑워드 제거 + TOP-20 교집합
  llm_revalidator      → confidence_draft 0.5~0.8 구간에서만 Solar LLM yes/no 판정
  threshold_decider    → 최종 confidence + AUTO_VERIFIED/NEEDS_REVIEW/AUTO_REJECTED

모든 노드는 try/except로 감싸고, 실패 시 안전한 기본값을 반환한다 (에러 전파 금지).
early_exit=True인 노드는 preprocessor에서 이미 결과가 세팅되므로 pass-through한다.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import Any

import numpy as np
import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from monglepick.agents.review_verification.models import ReviewVerificationState
from monglepick.data_pipeline.embedder import embed_query_async
from monglepick.llm import get_solar_api_llm, guarded_ainvoke

logger = structlog.get_logger()

# ============================================================
# 상수
# ============================================================

_MAX_TEXT_LEN = 1500
_MIN_REVIEW_LEN = 20

_THRESHOLD_HIGH = 0.7   # AUTO_VERIFIED
_THRESHOLD_LOW = 0.3    # AUTO_REJECTED (미만이면)

_LLM_CALL_LOW = 0.5     # LLM 호출 하한
_LLM_CALL_HIGH = 0.8    # LLM 호출 상한

# 영화 리뷰에서 흔히 등장하지만 내용과 무관한 일반 어휘
_STOPWORDS: set[str] = {
    # 영화 일반 어휘
    "영화", "작품", "장면", "감독", "스토리", "내용", "연출", "배우",
    "연기", "촬영", "음악", "음향", "효과", "포스터", "예고편", "관람",
    # 감상 표현
    "재미", "재밌", "좋았", "싫었", "최고", "최악", "강추", "비추",
    "추천", "시청", "봤다", "봤어", "봤는데", "봤습니다", "느낌", "생각",
    # 일반 부사/형용사
    "정말", "너무", "매우", "굉장히", "상당히", "완전", "진짜", "되게",
    "처음", "마지막", "다시", "계속", "항상", "이번", "오늘", "정도",
    # 접속어
    "그리고", "하지만", "그런데", "그래서", "또한", "때문", "위해", "모습",
}

# ============================================================
# 내부 유틸
# ============================================================

def _clean_text(text: str) -> str:
    """HTML 태그, 마크다운 특수문자 제거 후 공백 정리."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[*_`#>~\[\]()\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_TEXT_LEN]


def _extract_words(text: str) -> list[str]:
    """한글 2글자 이상 단어 + 영문 대문자 시작 고유명사 추출."""
    korean = re.findall(r"[가-힣]{2,}", text)
    english = re.findall(r"[A-Z][a-zA-Z]{1,}", text)
    return korean + english


async def _embed_passage_async(text: str) -> np.ndarray:
    """
    embedding-passage 모델로 텍스트를 비동기 임베딩한다.

    줄거리처럼 긴 텍스트는 passage 모델이 더 적합하다.
    embedder.py의 _get_client()를 재사용하여 asyncio.to_thread()로 블로킹 방지.
    """
    from monglepick.data_pipeline.embedder import _get_client

    def _sync_embed(t: str) -> np.ndarray:
        client = _get_client()
        response = client.embeddings.create(model="embedding-passage", input=[t])
        return np.array(response.data[0].embedding)

    return await asyncio.to_thread(_sync_embed, text)


# ============================================================
# 노드 1: preprocessor
# ============================================================

@traceable(name="review_verification.preprocessor")
async def preprocessor(state: ReviewVerificationState) -> dict:
    """
    텍스트 정제 + 조기 종료 판단.

    - HTML/마크다운 제거, 1500자 truncate
    - 정제 후 리뷰 길이 < 20자이면 early_exit=True로 NEEDS_REVIEW 강등
    """
    try:
        clean_review = _clean_text(state.get("review_text", ""))
        clean_plot = _clean_text(state.get("movie_plot", ""))

        if len(clean_review) < _MIN_REVIEW_LEN:
            logger.info(
                "review_verification_early_exit",
                verification_id=state.get("verification_id"),
                review_len=len(clean_review),
            )
            return {
                "clean_review": clean_review,
                "clean_plot": clean_plot,
                "early_exit": True,
                "similarity_score": 0.0,
                "matched_keywords": [],
                "keyword_score": 0.0,
                "confidence_draft": 0.0,
                "llm_adjustment": 0.0,
                "confidence": 0.0,
                "review_status": "NEEDS_REVIEW",
                "rationale": f"리뷰 내용이 너무 짧습니다 ({len(clean_review)}자). 관리자 검수가 필요합니다.",
            }

        return {
            "clean_review": clean_review,
            "clean_plot": clean_plot,
            "early_exit": False,
        }

    except Exception as e:
        logger.warning(
            "review_verification_preprocessor_error",
            verification_id=state.get("verification_id"),
            error=str(e),
        )
        return {
            "clean_review": state.get("review_text", "")[:_MAX_TEXT_LEN],
            "clean_plot": state.get("movie_plot", "")[:_MAX_TEXT_LEN],
            "early_exit": False,
        }


# ============================================================
# 노드 2: embedding_similarity
# ============================================================

@traceable(name="review_verification.embedding_similarity")
async def embedding_similarity(state: ReviewVerificationState) -> dict:
    """
    Solar 임베딩으로 리뷰 ↔ 줄거리 코사인 유사도를 계산한다.

    - 리뷰: embedding-query 모델 (짧은 텍스트 최적화)
    - 줄거리: embedding-passage 모델 (긴 텍스트 최적화)
    - 두 벡터는 L2-normalized → 내적 = 코사인 유사도
    """
    if state.get("early_exit"):
        return {}

    try:
        review_vec, plot_vec = await asyncio.gather(
            embed_query_async(state["clean_review"]),
            _embed_passage_async(state["clean_plot"]),
        )

        similarity = float(np.dot(review_vec, plot_vec))
        similarity = max(0.0, min(1.0, similarity))

        logger.info(
            "review_verification_embedding_done",
            verification_id=state.get("verification_id"),
            similarity_score=round(similarity, 4),
        )
        return {"similarity_score": similarity}

    except Exception as e:
        logger.warning(
            "review_verification_embedding_error",
            verification_id=state.get("verification_id"),
            error=str(e),
        )
        return {"similarity_score": 0.0}


# ============================================================
# 노드 3: keyword_matcher
# ============================================================

@traceable(name="review_verification.keyword_matcher")
async def keyword_matcher(state: ReviewVerificationState) -> dict:
    """
    한국어 명사 추출 → 스탑워드 제거 → TOP-20 교집합으로 keyword_score 계산.

    konlpy/soynlp 없이 정규식 기반으로 처리하여 운영 환경 의존성을 최소화한다.
    keyword_score = min(교집합 크기 / 5, 1.0)
    """
    if state.get("early_exit"):
        return {}

    try:
        review_words = _extract_words(state.get("clean_review", ""))
        plot_words = _extract_words(state.get("clean_plot", ""))

        review_filtered = [w for w in review_words if w not in _STOPWORDS]
        plot_filtered = [w for w in plot_words if w not in _STOPWORDS]

        review_top20 = {w for w, _ in Counter(review_filtered).most_common(20)}
        plot_top20 = {w for w, _ in Counter(plot_filtered).most_common(20)}

        intersection = list(review_top20 & plot_top20)
        keyword_score = min(len(intersection) / 5.0, 1.0)

        logger.info(
            "review_verification_keyword_done",
            verification_id=state.get("verification_id"),
            matched_count=len(intersection),
            keyword_score=round(keyword_score, 4),
        )
        return {
            "matched_keywords": intersection,
            "keyword_score": keyword_score,
        }

    except Exception as e:
        logger.warning(
            "review_verification_keyword_error",
            verification_id=state.get("verification_id"),
            error=str(e),
        )
        return {"matched_keywords": [], "keyword_score": 0.0}


# ============================================================
# 노드 4: llm_revalidator
# ============================================================

_REVALIDATION_SYSTEM = """당신은 영화 리뷰 검증 전문가입니다.
아래 영화 줄거리와 유저 리뷰를 읽고, 이 리뷰가 해당 영화를 실제로 관람한 사람이 쓴 것인지 판단하세요.

판단 기준:
- 영화의 구체적인 인물, 사건, 장면이 언급되면 YES
- 줄거리와 전혀 무관한 일반적 감상만 있으면 NO
- 리뷰가 줄거리를 그대로 복사한 것처럼 보이면 NO

반드시 YES 또는 NO 중 하나만 출력하세요."""

_REVALIDATION_HUMAN = """[영화 줄거리]
{plot}

[유저 리뷰]
{review}

이 리뷰가 이 영화를 실제로 관람한 사람이 쓴 것입니까? (YES / NO)"""


@traceable(name="review_verification.llm_revalidator")
async def llm_revalidator(state: ReviewVerificationState) -> dict:
    """
    confidence_draft 0.5~0.8 구간일 때만 Solar LLM으로 yes/no 재검증한다.

    구간 밖(< 0.5 또는 > 0.8)은 LLM 호출 없이 adjustment=0.0으로 처리한다.
    LLM 호출 실패 시에도 adjustment=0.0으로 중립 처리하여 에러 전파를 막는다.
    """
    if state.get("early_exit"):
        return {}

    sim = state.get("similarity_score", 0.0)
    kw = state.get("keyword_score", 0.0)
    confidence_draft = 0.7 * sim + 0.3 * kw

    if not (_LLM_CALL_LOW <= confidence_draft <= _LLM_CALL_HIGH):
        logger.info(
            "review_verification_llm_skipped",
            verification_id=state.get("verification_id"),
            confidence_draft=round(confidence_draft, 4),
            reason="구간 밖" if confidence_draft < _LLM_CALL_LOW else "충분히 높음",
        )
        return {"confidence_draft": confidence_draft, "llm_adjustment": 0.0}

    try:
        llm = get_solar_api_llm(temperature=0.1)
        messages = [
            SystemMessage(content=_REVALIDATION_SYSTEM),
            HumanMessage(content=_REVALIDATION_HUMAN.format(
                plot=state.get("clean_plot", "")[:800],
                review=state.get("clean_review", "")[:500],
            )),
        ]
        response = await guarded_ainvoke(llm, messages, model="solar_api")
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        ).strip().upper()

        if "YES" in response_text:
            adjustment = 0.1
        elif "NO" in response_text:
            adjustment = -0.2
        else:
            adjustment = 0.0

        logger.info(
            "review_verification_llm_done",
            verification_id=state.get("verification_id"),
            llm_response=response_text[:10],
            adjustment=adjustment,
        )
        return {"confidence_draft": confidence_draft, "llm_adjustment": adjustment}

    except Exception as e:
        logger.warning(
            "review_verification_llm_error",
            verification_id=state.get("verification_id"),
            error=str(e),
        )
        return {"confidence_draft": confidence_draft, "llm_adjustment": 0.0}


# ============================================================
# 노드 5: threshold_decider
# ============================================================

@traceable(name="review_verification.threshold_decider")
async def threshold_decider(state: ReviewVerificationState) -> dict:
    """
    최종 confidence 계산 후 임계값 기준으로 review_status를 결정한다.

    confidence = clip(confidence_draft + llm_adjustment, 0.0, 1.0)
    >= 0.7 → AUTO_VERIFIED
    0.3 ~ 0.7 → NEEDS_REVIEW
    < 0.3 → AUTO_REJECTED
    """
    if state.get("early_exit"):
        return {}

    try:
        draft = state.get("confidence_draft", 0.0)
        adjustment = state.get("llm_adjustment", 0.0)
        confidence = max(0.0, min(1.0, draft + adjustment))

        sim = state.get("similarity_score", 0.0)
        kw_count = len(state.get("matched_keywords", []))
        kw_preview = ", ".join(state.get("matched_keywords", [])[:5]) or "없음"

        if confidence >= _THRESHOLD_HIGH:
            review_status = "AUTO_VERIFIED"
            rationale = (
                f"임베딩 유사도 {sim:.2f} + 키워드 {kw_count}개 매칭({kw_preview}) "
                f"→ confidence {confidence:.2f} 자동 승인"
            )
        elif confidence >= _THRESHOLD_LOW:
            review_status = "NEEDS_REVIEW"
            rationale = (
                f"임베딩 유사도 {sim:.2f}, 키워드 매칭 {kw_count}개 "
                f"→ confidence {confidence:.2f} 관리자 검수 필요"
            )
        else:
            review_status = "AUTO_REJECTED"
            rationale = (
                f"임베딩 유사도 {sim:.2f}, 키워드 매칭 {kw_count}개 "
                f"→ confidence {confidence:.2f} 영화와 무관한 리뷰로 판단하여 자동 반려"
            )

        logger.info(
            "review_verification_decision",
            verification_id=state.get("verification_id"),
            confidence=round(confidence, 4),
            review_status=review_status,
        )
        return {
            "confidence": confidence,
            "review_status": review_status,
            "rationale": rationale,
        }

    except Exception as e:
        logger.warning(
            "review_verification_threshold_error",
            verification_id=state.get("verification_id"),
            error=str(e),
        )
        return {
            "confidence": 0.0,
            "review_status": "NEEDS_REVIEW",
            "rationale": "판정 중 오류 발생 — 관리자 검수가 필요합니다.",
        }
