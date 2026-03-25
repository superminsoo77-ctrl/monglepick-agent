"""
프롬프트 템플릿 모듈.

각 LLM 체인에서 사용하는 시스템/휴먼 프롬프트를 제공한다.
"""

from monglepick.prompts.emotion import (
    EMOTION_HUMAN_PROMPT,
    EMOTION_SYSTEM_PROMPT,
    EMOTION_TO_MOOD_MAP,
)
from monglepick.prompts.explanation import (
    EXPLANATION_HUMAN_PROMPT,
    EXPLANATION_SYSTEM_PROMPT,
)
from monglepick.prompts.image_analysis import (
    IMAGE_ANALYSIS_HUMAN_PROMPT,
    IMAGE_ANALYSIS_SYSTEM_PROMPT,
)
from monglepick.prompts.intent import (
    INTENT_HUMAN_PROMPT,
    INTENT_SYSTEM_PROMPT,
)
from monglepick.prompts.intent_emotion import (
    INTENT_EMOTION_HUMAN_PROMPT,
    INTENT_EMOTION_SYSTEM_PROMPT,
)
from monglepick.prompts.persona import (
    MONGGLE_RECOMMENDATION_PERSONA,
    MONGGLE_SYSTEM_PROMPT,
)
from monglepick.prompts.preference import (
    PREFERENCE_HUMAN_PROMPT,
    PREFERENCE_SYSTEM_PROMPT,
)
from monglepick.prompts.question import (
    QUESTION_HUMAN_PROMPT,
    QUESTION_SYSTEM_PROMPT,
)
from monglepick.prompts.tool_executor import TOOL_EXECUTOR_SYSTEM_PROMPT

__all__ = [
    # 페르소나
    "MONGGLE_SYSTEM_PROMPT",
    "MONGGLE_RECOMMENDATION_PERSONA",
    # 의도 분류
    "INTENT_SYSTEM_PROMPT",
    "INTENT_HUMAN_PROMPT",
    # 감정 분석
    "EMOTION_SYSTEM_PROMPT",
    "EMOTION_HUMAN_PROMPT",
    "EMOTION_TO_MOOD_MAP",
    # 선호 추출
    "PREFERENCE_SYSTEM_PROMPT",
    "PREFERENCE_HUMAN_PROMPT",
    # 후속 질문
    "QUESTION_SYSTEM_PROMPT",
    "QUESTION_HUMAN_PROMPT",
    # 추천 이유
    "EXPLANATION_SYSTEM_PROMPT",
    "EXPLANATION_HUMAN_PROMPT",
    # 이미지 분석
    "IMAGE_ANALYSIS_SYSTEM_PROMPT",
    "IMAGE_ANALYSIS_HUMAN_PROMPT",
    # 의도+감정 통합
    "INTENT_EMOTION_SYSTEM_PROMPT",
    "INTENT_EMOTION_HUMAN_PROMPT",
    # 도구 실행
    "TOOL_EXECUTOR_SYSTEM_PROMPT",
]
