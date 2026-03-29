"""
LLM 체인 모듈.

각 노드별 LLM 체인 함수를 제공한다.
모든 체인은 async def이며, LLM 에러 시 유효한 fallback을 반환한다.
"""

from monglepick.chains.emotion_chain import analyze_emotion
from monglepick.chains.explanation_chain import (
    generate_explanation,
    generate_explanations_batch,
)
from monglepick.chains.general_chat_chain import generate_general_response
from monglepick.chains.image_analysis_chain import analyze_image
from monglepick.chains.intent_chain import classify_intent
from monglepick.chains.intent_emotion_chain import classify_intent_and_emotion
from monglepick.chains.preference_chain import extract_preferences
from monglepick.chains.question_chain import generate_question
from monglepick.chains.rerank_chain import rerank_candidates
from monglepick.chains.tool_executor_chain import execute_tool

__all__ = [
    # 의도 분류 — 6가지 intent (recommend/search/info/theater/booking/general)
    "classify_intent",
    # 의도+감정 통합 — 1회 LLM 호출로 의도 분류 + 감정 분석 동시 수행
    "classify_intent_and_emotion",
    # 감정 분석 — 5가지 감정 + 25개 무드태그 매핑
    "analyze_emotion",
    # 선호 추출 — 7가지 선호 필드 + 충분성 판정
    "extract_preferences",
    # 후속 질문 — 부족 정보 파악을 위한 자연스러운 질문 생성
    "generate_question",
    # 추천 이유 — 영화별 맞춤 설명 (단건/배치)
    "generate_explanation",
    "generate_explanations_batch",
    # 일반 대화 — 몽글 페르소나 기반 자유 대화
    "generate_general_response",
    # 이미지 분석 — VLM 멀티모달 포스터/분위기 사진 분석
    "analyze_image",
    # LLM 재랭킹 — 사용자 의도 기반 후보 재평가 (Phase Q)
    "rerank_candidates",
    # 도구 실행 — Phase 6 예정 (영화 상세/영화관/예매 등)
    "execute_tool",
]
