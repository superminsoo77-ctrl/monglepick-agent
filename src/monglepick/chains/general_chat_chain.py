"""
일반 대화 체인 (intent="general"일 때 호출).

영화와 무관한 일반 대화에 대해 몽글 페르소나로 응답하는 체인.
EXAONE 32B (자유 텍스트, temp=0.5)로 실행한다.

처리 흐름:
1. MONGGLE_SYSTEM_PROMPT 시스템 프롬프트 사용
2. get_conversation_llm() (EXAONE 32B, temp=0.5) 호출
3. 에러 시: "잠시 문제가 있었어요. 다시 말씀해주세요! 😊"
"""

from __future__ import annotations

import time
import traceback

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.config import settings
from monglepick.llm.factory import get_conversation_llm, guarded_ainvoke
from monglepick.prompts.persona import MONGGLE_SYSTEM_PROMPT

logger = structlog.get_logger()

# 에러 시 반환할 기본 메시지
DEFAULT_ERROR_MESSAGE = "잠시 문제가 있었어요. 다시 말씀해주세요! 😊"


async def generate_general_response(
    current_input: str,
    recent_messages: str = "",
) -> str:
    """
    일반 대화에 대해 몽글 페르소나로 응답한다.

    intent="general"일 때 호출되며, 인사/잡담/영화 무관 질문에 응답한다.

    Args:
        current_input: 현재 사용자 입력 텍스트
        recent_messages: 최근 대화 이력 (포맷된 문자열)

    Returns:
        한국어 응답 문자열
        - 에러 시: DEFAULT_ERROR_MESSAGE
    """
    # 프롬프트 구성 (몽글 페르소나 시스템 프롬프트)
    prompt = ChatPromptTemplate.from_messages([
        ("system", MONGGLE_SYSTEM_PROMPT),
        ("human", "{recent_messages}\n\n사용자: {current_input}"),
    ])

    # 자유 텍스트 LLM (EXAONE 32B, temp=0.5)
    llm = get_conversation_llm()

    # 입력 변수
    inputs = {
        "current_input": current_input,
        "recent_messages": recent_messages or "(대화 시작)",
    }

    logger.info(
        "general_chat_chain_start",
        input_preview=current_input[:100],
        recent_messages_preview=recent_messages[:100] if recent_messages else "(없음)",
    )

    try:
        # 프롬프트 포맷 → LLM 호출 (명시적 2단계)
        llm_start = time.perf_counter()
        prompt_value = await prompt.ainvoke(inputs)
        logger.debug(
            "general_chat_chain_prompt_formatted",
            prompt_preview=str(prompt_value)[:300],
        )
        logger.debug(
            "general_chat_chain_prompt_full",
            prompt_text=str(prompt_value),
            model=settings.CONVERSATION_MODEL,
        )
        # 모델별 세마포어로 동시 호출 제한 (Ollama 큐 점유 방지)
        response = await guarded_ainvoke(
            llm, prompt_value, model=settings.CONVERSATION_MODEL,
        )
        elapsed_ms = (time.perf_counter() - llm_start) * 1000

        # LangChain BaseMessage → 문자열 추출
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip() if isinstance(text, str) else str(text).strip()

        logger.debug(
            "general_chat_chain_llm_raw_response",
            raw_response=str(response),
            model=settings.CONVERSATION_MODEL,
        )
        logger.info(
            "general_response_generated",
            response_preview=text[:50],
            input_preview=current_input[:50],
            elapsed_ms=round(elapsed_ms, 1),
            model=settings.CONVERSATION_MODEL,
        )
        return text

    except Exception as e:
        logger.error(
            "general_response_error",
            error=str(e),
            input_preview=current_input[:50],
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
        )
        return DEFAULT_ERROR_MESSAGE
