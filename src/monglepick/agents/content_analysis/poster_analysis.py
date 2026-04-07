"""
포스터 분석 모듈 (§8-2 기능1).

처리 흐름:
1. httpx로 poster_url에서 이미지 다운로드 (5MB 이하, 실패 시 skip)
2. get_vision_llm()으로 포스터 시각 분석 → JSON 파싱
3. get_explanation_llm()으로 리뷰 초안 생성 (3~5문장, 스포일러 금지)
4. 리뷰 초안에서 키워드 5~7개 추출
5. 에러 시 메타데이터만으로 fallback 처리

이미지가 없거나 다운로드 실패 시에도 메타데이터 기반으로 결과를 반환한다 (에러 전파 금지).
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

import httpx
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from monglepick.agents.content_analysis.models import (
    PosterAnalysis,
    PosterAnalysisInput,
    PosterAnalysisOutput,
)
from monglepick.llm import guarded_ainvoke, get_vision_llm, get_explanation_llm

logger = structlog.get_logger()

# ── 이미지 다운로드 제한 ──
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB
HTTP_TIMEOUT_SEC = 10.0

# ── 포스터 분석 시스템 프롬프트 ──
POSTER_ANALYSIS_SYSTEM = """당신은 영화 포스터 분석 전문가입니다.
포스터의 시각적 요소를 분석하여 아래 JSON 형식으로 반환하세요.

반환 형식 (JSON만 출력, 설명 없음):
{
  "mood_tags": ["분위기태그1", "분위기태그2", "분위기태그3"],
  "color_palette": ["색감1", "색감2"],
  "visual_impression": "첫인상 한 줄 요약 (50자 이내)",
  "atmosphere": "전체적인 분위기 1~2문장"
}

규칙:
- mood_tags: 3~5개의 한국어 분위기 키워드
- color_palette: 2~4개의 색감 표현 (예: "딥 블루", "차가운 회색")
- visual_impression: 50자 이내
- atmosphere: 1~2문장의 자연어 묘사
- 순수 JSON만 출력 (마크다운 코드블록 없음)"""

# ── 리뷰 초안 생성 시스템 프롬프트 ──
REVIEW_DRAFT_SYSTEM = """당신은 영화 리뷰 작가입니다.
포스터 분석 결과와 영화 정보를 바탕으로 한국어 리뷰 초안과 키워드를 작성하세요.

반환 형식 (JSON만 출력, 설명 없음):
{
  "review_draft": "3~5문장의 한국어 리뷰 초안",
  "review_keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"]
}

규칙:
- review_draft: 스포일러 절대 금지, 감정과 분위기 중심으로 3~5문장
- review_keywords: 리뷰에서 핵심 표현 5~7개 추출
- 사용자 평점이 있으면 어조에 반영 (높은 평점 → 긍정적, 낮은 평점 → 비판적)
- 순수 JSON만 출력 (마크다운 코드블록 없음)"""


async def _download_image(url: str) -> bytes | None:
    """
    poster_url에서 이미지를 다운로드한다.

    5MB 초과 또는 HTTP 오류 시 None 반환 (에러 전파 금지).
    timeout은 10초로 고정한다.

    Args:
        url: 이미지 URL

    Returns:
        이미지 바이트열, 실패 시 None
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SEC) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()

            # Content-Length 헤더로 사전 크기 체크
            content_length = int(resp.headers.get("content-length", 0))
            if content_length > MAX_IMAGE_BYTES:
                logger.warning(
                    "poster_image_too_large",
                    url=url,
                    content_length=content_length,
                    max_bytes=MAX_IMAGE_BYTES,
                )
                return None

            image_bytes = resp.content
            # 실제 수신 바이트 크기 재검증 (Content-Length 헤더 신뢰 불가 경우 대비)
            if len(image_bytes) > MAX_IMAGE_BYTES:
                logger.warning(
                    "poster_image_too_large_actual",
                    url=url,
                    actual_bytes=len(image_bytes),
                )
                return None

            return image_bytes

    except httpx.HTTPError as e:
        logger.warning("poster_image_download_http_error", url=url, error=str(e))
        return None
    except Exception as e:
        logger.warning("poster_image_download_error", url=url, error=str(e))
        return None


def _build_vision_message(
    image_bytes: bytes | None,
    movie_metadata: dict,
) -> list[SystemMessage | HumanMessage]:
    """
    VLM 분석을 위한 메시지 목록을 구성한다.

    이미지가 있으면 base64 인코딩 후 multimodal 메시지로,
    없으면 텍스트만으로 메타데이터 기반 분석을 요청한다.

    Args:
        image_bytes  : 이미지 바이트열 (없으면 None)
        movie_metadata: 영화 메타데이터 dict

    Returns:
        LangChain 메시지 목록
    """
    title = movie_metadata.get("title", "제목 미상")
    genres = movie_metadata.get("genres", [])
    overview = movie_metadata.get("overview", "")

    system_msg = SystemMessage(content=POSTER_ANALYSIS_SYSTEM)

    if image_bytes:
        # base64 인코딩 → multimodal content
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        human_msg = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        f"영화: {title}\n"
                        f"장르: {', '.join(genres) if genres else '미상'}\n"
                        f"줄거리 요약: {overview[:200] if overview else '없음'}\n\n"
                        "위 포스터를 분석하여 JSON으로 반환하세요."
                    ),
                },
            ]
        )
    else:
        # 이미지 없음 → 메타데이터만으로 텍스트 분석 요청
        human_msg = HumanMessage(
            content=(
                f"포스터 이미지를 불러올 수 없습니다. 아래 영화 정보만으로 분석하세요.\n\n"
                f"영화: {title}\n"
                f"장르: {', '.join(genres) if genres else '미상'}\n"
                f"줄거리 요약: {overview[:200] if overview else '없음'}\n\n"
                "영화 정보를 바탕으로 예상 분위기를 JSON으로 반환하세요."
            )
        )

    return [system_msg, human_msg]


def _parse_json_response(text: str, context: str = "") -> dict:
    """
    LLM 응답 텍스트에서 JSON을 파싱한다.

    마크다운 코드블록(```json ... ```) 을 제거하고 순수 JSON을 추출한다.

    Args:
        text   : LLM 응답 텍스트
        context: 로그용 컨텍스트 설명

    Returns:
        파싱된 dict, 실패 시 빈 dict
    """
    try:
        # 마크다운 코드블록 제거
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # JSON 블록 추출 시도 (중괄호 기준)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning("json_parse_failed", context=context, text_preview=text[:200])
        return {}


def _make_fallback_poster_analysis(movie_metadata: dict) -> PosterAnalysis:
    """
    포스터 분석 실패 시 메타데이터 기반 fallback PosterAnalysis를 생성한다.

    Args:
        movie_metadata: 영화 메타데이터 dict

    Returns:
        최소 정보로 채워진 PosterAnalysis
    """
    genres = movie_metadata.get("genres", [])
    title = movie_metadata.get("title", "")

    # 장르 기반 기본 분위기 태그 매핑
    genre_mood_map: dict[str, list[str]] = {
        "액션": ["역동적인", "긴장감"],
        "공포": ["어두운", "서늘한", "긴장감"],
        "로맨스": ["따뜻한", "감성적인"],
        "코미디": ["유쾌한", "밝은"],
        "드라마": ["감성적인", "진지한"],
        "SF": ["미래적인", "신비로운"],
        "애니메이션": ["밝은", "화사한"],
        "스릴러": ["긴박한", "어두운"],
        "범죄": ["어두운", "긴박한"],
        "다큐멘터리": ["진지한", "사실적인"],
    }
    mood_tags: list[str] = []
    for genre in genres[:2]:
        mood_tags.extend(genre_mood_map.get(genre, []))
    if not mood_tags:
        mood_tags = ["미상"]

    return PosterAnalysis(
        mood_tags=list(dict.fromkeys(mood_tags))[:5],  # 중복 제거 후 최대 5개
        color_palette=["미상"],
        visual_impression=f"{title} 포스터" if title else "포스터 분석 불가",
        atmosphere=f"{'·'.join(genres[:2])} 영화의 분위기" if genres else "분위기 분석 불가",
    )


async def analyze_poster(inp: PosterAnalysisInput) -> PosterAnalysisOutput:
    """
    영화 포스터를 분석하고 리뷰 초안을 생성한다.

    처리 순서:
    1. poster_url에서 이미지 다운로드 (실패해도 메타데이터 기반으로 계속 진행)
    2. VLM(get_vision_llm)으로 포스터 시각 분석
    3. Explanation LLM(get_explanation_llm)으로 리뷰 초안 생성
    4. 에러 시 fallback 반환 (에러 전파 금지)

    Args:
        inp: PosterAnalysisInput (movie_id, poster_url, movie_metadata, user_rating)

    Returns:
        PosterAnalysisOutput (poster_analysis, review_draft, review_keywords)
    """
    try:
        movie_id = inp.movie_id
        movie_metadata = inp.movie_metadata or {}
        title = movie_metadata.get("title", "제목 미상")

        logger.info("poster_analysis_start", movie_id=movie_id, title=title)

        # ── Step 1: 이미지 다운로드 ──
        image_bytes: bytes | None = None
        if inp.poster_url:
            image_bytes = await _download_image(inp.poster_url)
            if image_bytes is None:
                logger.info(
                    "poster_image_skipped_using_metadata",
                    movie_id=movie_id,
                    poster_url=inp.poster_url,
                )

        # ── Step 2: VLM 포스터 시각 분석 ──
        poster_analysis = _make_fallback_poster_analysis(movie_metadata)
        try:
            vision_llm = get_vision_llm()
            vision_messages = _build_vision_message(image_bytes, movie_metadata)
            vision_response = await guarded_ainvoke(vision_llm, vision_messages)
            vision_text = (
                vision_response.content
                if hasattr(vision_response, "content")
                else str(vision_response)
            )
            parsed = _parse_json_response(vision_text, context="poster_vision_analysis")
            if parsed:
                poster_analysis = PosterAnalysis(
                    mood_tags=parsed.get("mood_tags", poster_analysis.mood_tags),
                    color_palette=parsed.get("color_palette", poster_analysis.color_palette),
                    visual_impression=parsed.get(
                        "visual_impression", poster_analysis.visual_impression
                    ),
                    atmosphere=parsed.get("atmosphere", poster_analysis.atmosphere),
                )
        except Exception as e:
            # VLM 분석 실패 → fallback 유지, 계속 진행
            logger.warning(
                "poster_vision_analysis_failed",
                movie_id=movie_id,
                error=str(e),
            )

        # ── Step 3: 리뷰 초안 생성 ──
        review_draft = ""
        review_keywords: list[str] = []
        try:
            explanation_llm = get_explanation_llm()

            # 평점 어조 힌트 구성
            rating_hint = ""
            if inp.user_rating is not None:
                if inp.user_rating >= 4.0:
                    rating_hint = f"\n사용자 평점: {inp.user_rating}/5 (매우 긍정적 어조로 작성)"
                elif inp.user_rating <= 2.0:
                    rating_hint = f"\n사용자 평점: {inp.user_rating}/5 (비판적 어조 허용)"
                else:
                    rating_hint = f"\n사용자 평점: {inp.user_rating}/5"

            genres = movie_metadata.get("genres", [])
            review_prompt_content = (
                f"영화: {title}\n"
                f"장르: {', '.join(genres) if genres else '미상'}\n"
                f"포스터 분위기 태그: {', '.join(poster_analysis.mood_tags)}\n"
                f"색감: {', '.join(poster_analysis.color_palette)}\n"
                f"첫인상: {poster_analysis.visual_impression}\n"
                f"분위기: {poster_analysis.atmosphere}"
                f"{rating_hint}\n\n"
                "위 정보를 바탕으로 리뷰 초안(review_draft)과 키워드(review_keywords)를 JSON으로 반환하세요."
            )

            review_messages = [
                SystemMessage(content=REVIEW_DRAFT_SYSTEM),
                HumanMessage(content=review_prompt_content),
            ]
            review_response = await guarded_ainvoke(explanation_llm, review_messages)
            review_text = (
                review_response.content
                if hasattr(review_response, "content")
                else str(review_response)
            )
            review_parsed = _parse_json_response(review_text, context="review_draft")
            if review_parsed:
                review_draft = review_parsed.get("review_draft", "")
                review_keywords = review_parsed.get("review_keywords", [])

        except Exception as e:
            logger.warning(
                "review_draft_generation_failed",
                movie_id=movie_id,
                error=str(e),
            )
            # fallback 리뷰 초안 — 분위기 태그 기반 단순 문장
            review_draft = (
                f"'{title}'은(는) {', '.join(poster_analysis.mood_tags[:2])} 분위기의 영화입니다. "
                "포스터에서 느껴지는 시각적 인상이 인상적입니다."
            )
            review_keywords = poster_analysis.mood_tags[:5]

        logger.info(
            "poster_analysis_complete",
            movie_id=movie_id,
            mood_tags=poster_analysis.mood_tags,
            keyword_count=len(review_keywords),
        )

        return PosterAnalysisOutput(
            poster_analysis=poster_analysis,
            review_draft=review_draft,
            review_keywords=review_keywords,
        )

    except Exception as e:
        # 최외곽 예외 — 절대 에러 전파 금지
        logger.error(
            "analyze_poster_fatal_error",
            movie_id=getattr(inp, "movie_id", "unknown"),
            error=str(e),
        )
        return PosterAnalysisOutput(
            poster_analysis=PosterAnalysis(
                mood_tags=["분석 실패"],
                color_palette=["미상"],
                visual_impression="포스터 분석에 실패했습니다.",
                atmosphere="",
            ),
            review_draft="포스터 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            review_keywords=[],
        )
