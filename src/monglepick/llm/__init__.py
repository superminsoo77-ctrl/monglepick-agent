"""
LLM 팩토리 + 동시성 제어 모듈.

용도별 ChatOllama 인스턴스를 생성하는 팩토리 함수를 제공한다.
동일 파라미터에 대해 싱글턴 캐싱을 적용한다.
guarded_ainvoke로 모델별 동시 호출 수를 제한한다.
"""

from monglepick.llm.concurrency import (
    acquire_model_slot,
    release_model_slot,
    reset_semaphores,
)
from monglepick.llm.factory import (
    get_conversation_llm,
    get_emotion_llm,
    get_explanation_llm,
    get_intent_emotion_llm,
    get_intent_llm,
    get_llm,
    get_preference_llm,
    get_question_llm,
    get_structured_llm,
    get_vision_llm,
    guarded_ainvoke,
)

__all__ = [
    "get_llm",
    "get_structured_llm",
    "get_intent_llm",
    "get_emotion_llm",
    "get_intent_emotion_llm",
    "get_preference_llm",
    "get_conversation_llm",
    "get_question_llm",
    "get_explanation_llm",
    "get_vision_llm",
    "guarded_ainvoke",
    "acquire_model_slot",
    "release_model_slot",
    "reset_semaphores",
]
