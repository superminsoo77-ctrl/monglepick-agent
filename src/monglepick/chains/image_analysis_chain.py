"""
이미지 분석 체인 (VLM 멀티모달).

사용자가 업로드한 이미지(영화 포스터, 분위기 사진 등)를 VLM으로 분석하여
ImageAnalysisResult를 반환하는 체인.

처리 흐름:
1. SystemMessage + HumanMessage(content=[image_url(base64), text]) 멀티모달 메시지 구성
2. get_vision_llm() (Qwen3.5 35B-A3B, format=json) 호출
3. 응답 JSON → ImageAnalysisResult 파싱
4. mood_cues를 MOOD_WHITELIST로 필터링
5. 에러 시 ImageAnalysisResult(analyzed=False) 반환
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from monglepick.agents.chat.models import ImageAnalysisResult
from monglepick.config import settings
from monglepick.llm.factory import get_vision_llm, guarded_ainvoke
from monglepick.prompts.image_analysis import (
    IMAGE_ANALYSIS_HUMAN_PROMPT,
    IMAGE_ANALYSIS_SYSTEM_PROMPT,
)

logger = structlog.get_logger()

# 25개 무드 태그 화이트리스트 (data_pipeline/preprocessor.py 기준)
MOOD_WHITELIST: set[str] = {
    "유쾌", "따뜻", "감동", "스릴", "몰입", "웅장", "힐링", "잔잔",
    "다크", "반전", "로맨틱", "카타르시스", "철학적", "모험", "사회비판",
    "레트로", "판타지", "공포", "미스터리", "코미디", "비주얼", "실험적",
    "성장", "우정", "가족",
}


async def analyze_image(
    image_data: str,
    current_input: str = "",
) -> ImageAnalysisResult:
    """
    이미지를 VLM으로 분석하여 영화 추천에 활용할 정보를 추출한다.

    멀티모달 메시지를 구성하여 Qwen3.5 비전 모델에 전달하고,
    JSON 응답을 ImageAnalysisResult로 파싱한다.

    Args:
        image_data: base64 인코딩된 이미지 데이터
        current_input: 사용자 입력 텍스트 (이미지와 함께 전달된 메시지)

    Returns:
        ImageAnalysisResult — 분석 성공 시 analyzed=True, 실패 시 analyzed=False
    """
    if not image_data:
        logger.debug("image_analysis_skipped", reason="no_image_data")
        return ImageAnalysisResult(analyzed=False)

    try:
        # VLM LLM 인스턴스 가져오기 (Qwen3.5 35B-A3B, format=json)
        llm = get_vision_llm()

        # 사용자 메시지 포맷 (이미지와 함께 전달된 텍스트)
        user_message = current_input if current_input else "이 이미지를 분석해주세요."
        human_text = IMAGE_ANALYSIS_HUMAN_PROMPT.format(user_message=user_message)

        # 멀티모달 메시지 구성: SystemMessage + HumanMessage(이미지 + 텍스트)
        messages = [
            SystemMessage(content=IMAGE_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    # base64 이미지를 image_url 형식으로 전달 (Ollama 멀티모달 호환)
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                        },
                    },
                    # 텍스트 프롬프트
                    {
                        "type": "text",
                        "text": human_text,
                    },
                ],
            ),
        ]

        logger.info(
            "image_analysis_started",
            image_data_length=len(image_data),
            has_user_message=bool(current_input),
        )

        # VLM 호출 전 프롬프트 전문 디버그 로깅 + 타이밍 측정 시작
        logger.debug(
            "image_analysis_prompt_full",
            prompt_text=str(messages),
            model=settings.VISION_MODEL,
            image_data_length=len(image_data),
        )
        llm_start = time.perf_counter()
        timeout_sec = getattr(settings, "VISION_TIMEOUT_SEC", 180)
        try:
            # 모델별 세마포어로 동시 호출 제한 (Ollama 큐 점유 방지)
            response = await asyncio.wait_for(
                guarded_ainvoke(llm, messages, model=settings.VISION_MODEL),
                timeout=float(timeout_sec),
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - llm_start) * 1000
            logger.warning(
                "image_analysis_timeout",
                timeout_sec=timeout_sec,
                elapsed_ms=round(elapsed_ms, 1),
                model=settings.VISION_MODEL,
            )
            return ImageAnalysisResult(analyzed=False)
        elapsed_ms = (time.perf_counter() - llm_start) * 1000
        raw_content = response.content if hasattr(response, "content") else str(response)

        logger.debug(
            "image_analysis_llm_raw_response",
            raw_response=raw_content,
            model=settings.VISION_MODEL,
        )
        logger.info(
            "image_analysis_llm_response",
            response_length=len(raw_content),
            response_preview=raw_content[:200],
            elapsed_ms=round(elapsed_ms, 1),
            model=settings.VISION_MODEL,
        )

        # JSON 파싱
        parsed = _parse_json_response(raw_content)
        if parsed is None:
            logger.warning("image_analysis_json_parse_failed", raw=raw_content[:300])
            return ImageAnalysisResult(analyzed=False)

        # ImageAnalysisResult 구성 + mood_cues 화이트리스트 필터링
        result = ImageAnalysisResult(
            genre_cues=parsed.get("genre_cues", [])[:5],
            mood_cues=[m for m in parsed.get("mood_cues", []) if m in MOOD_WHITELIST][:5],
            visual_elements=parsed.get("visual_elements", [])[:8],
            search_keywords=parsed.get("search_keywords", [])[:5],
            description=parsed.get("description", "")[:500],
            is_movie_poster=bool(parsed.get("is_movie_poster", False)),
            detected_movie_title=parsed.get("detected_movie_title"),
            analyzed=True,
        )

        logger.info(
            "image_analysis_completed",
            genre_cues=result.genre_cues,
            mood_cues=result.mood_cues,
            visual_elements_count=len(result.visual_elements),
            is_poster=result.is_movie_poster,
            detected_title=result.detected_movie_title,
            elapsed_ms=round(elapsed_ms, 1),
        )
        return result

    except Exception as e:
        logger.error(
            "image_analysis_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
        )
        return ImageAnalysisResult(analyzed=False)


def _parse_json_response(raw: str) -> dict | None:
    """
    LLM 응답에서 JSON을 추출하여 파싱한다.

    JSON이 ```json ... ``` 코드 블록 안에 있거나 직접 JSON인 경우 모두 처리한다.

    Args:
        raw: LLM 응답 원문 텍스트

    Returns:
        파싱된 dict, 실패 시 None
    """
    if not raw:
        return None

    # 코드 블록 내 JSON 추출 시도
    import re
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 직접 JSON 파싱 시도
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # { ... } 패턴 추출 시도
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
