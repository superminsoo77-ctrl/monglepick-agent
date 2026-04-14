"""
Movie Match v3 LLM 리랭커 체인 (Solar API, 배치 점수화).

### 배경
Level 1·2 재설계에서 Movie Match 는 여전히 결정론적 유사도(Jaccard + cosine + CF) 로만
후보 순위를 매기고 있었다. 이는 "두 영화를 모두 좋아할 사용자" 라는 목표를 달성하는데
한계가 있어 (유저 피드백 2026-04-14), Chat Agent 의 `llm_reranker` 패턴을 Match 에도 도입한다.

### 처리 흐름
1. 후보 영화 N편(<=10) + 두 선택 영화(A, B) 를 프롬프트에 포맷
2. Solar API 로 배치 점수화 (0~10점 + 한줄 이유)
3. JSON 파싱 → {movie_id: llm_score_normalized} 딕셔너리 반환 (0~1 범위)
4. 실패 시 빈 dict 반환 → match_scorer 가 harmonic+cf 로 graceful fallback

### 설계 원칙 (채팅 영향 격리 가드레일)
- Solar 세마포어는 `guarded_ainvoke` 로 채팅과 공유되나, **타임아웃 + 에러 시 빈 dict** 로
  즉시 반환하여 Match 실패가 채팅 응답을 지연시키지 않도록 한다.
- Solar API 장애 시에도 Match 그래프가 완주하도록 에러 전파 절대 금지.
- 채팅의 `rerank_chain.py` 와 독립된 모듈 — 공유 상태 없음.

### 참고
- Chat `chains/rerank_chain.py` 의 JSON 파싱 3단계 전략 재사용
- `prompts/match_reranker.py` 에 정의된 Match 전용 프롬프트 사용
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.config import settings
from monglepick.llm.factory import get_explanation_llm, guarded_ainvoke
from monglepick.metrics import (
    match_llm_reranker_duration_seconds,
    match_llm_reranker_total,
)
from monglepick.prompts.match_reranker import (
    MATCH_RERANK_HUMAN_PROMPT,
    MATCH_RERANK_SYSTEM_PROMPT,
)

logger = structlog.get_logger()

# LLM 에 전달할 최대 후보 수 — 토큰 절약 + 응답 지연 방지
# rag_retriever 가 보통 15~20 편 반환하므로, 상위 10편으로 샘플링 후 리랭킹한다.
MAX_RERANK_CANDIDATES: int = 10

# Match v3 LLM 리랭커 타임아웃 (초). 채팅 응답을 지연시키지 않도록 여유있게 설정.
# Solar API 의 p95 는 보통 3~5초이므로 10초는 충분.
MATCH_LLM_RERANK_TIMEOUT: float = 10.0


# ============================================================
# 내부 유틸 — 후보 영화 포맷 + 응답 파싱
# ============================================================

def _format_candidate_line(movie: dict, idx: int) -> str:
    """
    후보 영화 dict 를 리랭커 프롬프트 한 줄로 포맷.

    토큰 절약을 위해 핵심 필드만 추림. id 를 명시하여 LLM 이 응답에 정확히 참조하도록.

    Args:
        movie: 후보 영화 dict (rag_retriever 출력 형식)
        idx  : 1-based 번호 (사용자/LLM 가독성)

    Returns:
        "1. **인셉션** (2010) | 장르: SF, 스릴러 | 감독: 크리스토퍼 놀란 | 평점: 8.8 | ... [ID: tmdb_27205]"
    """
    parts: list[str] = []
    title = movie.get("title") or "제목 없음"
    year = movie.get("release_year")
    parts.append(f"{idx}. **{title}**" + (f" ({year})" if year else ""))

    if movie.get("genres"):
        parts.append(f"장르: {', '.join(movie['genres'][:5])}")
    if movie.get("mood_tags"):
        parts.append(f"무드: {', '.join(movie['mood_tags'][:4])}")
    if movie.get("director"):
        parts.append(f"감독: {movie['director']}")
    if movie.get("rating"):
        parts.append(f"평점: {movie['rating']}")

    overview = (movie.get("overview") or "").strip()
    if overview:
        parts.append(f"줄거리: {overview[:150]}" + ("..." if len(overview) > 150 else ""))
    else:
        parts.append("줄거리: 없음")

    # ID 를 마지막에 — LLM 이 응답 movie_id 에 동일 값을 써야 매칭됨
    parts.append(f"[ID: {movie.get('id', '')}]")
    return " | ".join(parts)


def _parse_json_array(response_text: str) -> list[dict[str, Any]] | None:
    """
    LLM 응답 텍스트에서 JSON 배열을 3단계 전략으로 파싱한다.

    채팅 `rerank_chain._parse_rerank_response` 의 전략을 Match 용으로 단순화.
    파싱 실패 시 None 반환 → 호출자가 fallback 동작.

    Args:
        response_text: LLM 원본 응답

    Returns:
        [{movie_id, score, reason}, ...] 또는 None (파싱 실패)
    """
    # 1단계: 전체 응답이 JSON 배열
    try:
        data = json.loads(response_text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 2단계: 마크다운 코드블록 안의 JSON
    blocks = re.findall(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", response_text)
    for block in blocks:
        try:
            data = json.loads(block)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue

    # 3단계: 첫 번째 [...] 패턴 직접 추출
    match = re.search(r"\[[\s\S]*\]", response_text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return None


def _validate_and_normalize(
    items: list[dict],
    valid_ids: set[str],
) -> dict[str, float]:
    """
    파싱된 JSON 배열을 검증하고 {movie_id: normalized_score} 로 변환한다.

    - movie_id 가 후보 목록에 없으면 제외 (부분 문자열 매칭 fallback 1회 허용)
    - score 는 0~10 → 0~1 범위로 정규화 (match_scorer 의 다른 점수와 동일 스케일)

    Args:
        items     : LLM 응답에서 파싱된 [{movie_id, score, reason}, ...]
        valid_ids : 유효한 후보 영화 ID 집합 (검증용)

    Returns:
        {movie_id: llm_score_0_to_1} 딕셔너리
    """
    result: dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        raw_id = str(item.get("movie_id", "")).strip()
        if not raw_id:
            continue

        # ID 매칭: 정확 일치 우선, 실패 시 부분 일치 fallback
        if raw_id in valid_ids:
            movie_id = raw_id
        else:
            # 부분 매칭 — LLM 이 prefix/suffix 차이로 다르게 적는 경우 구제
            matched = [vid for vid in valid_ids if vid in raw_id or raw_id in vid]
            if not matched:
                continue
            movie_id = matched[0]

        # score 파싱 + 0~10 → 0~1 정규화
        try:
            raw_score = float(item.get("score", 5.0))
        except (TypeError, ValueError):
            raw_score = 5.0
        # 클램핑 후 0~1 변환
        normalized = max(0.0, min(10.0, raw_score)) / 10.0
        result[movie_id] = round(normalized, 4)

    return result


# ============================================================
# 공개 함수 — rag_retriever/match_scorer 중간 노드에서 호출
# ============================================================

async def rerank_match_candidates(
    candidates: list[dict],
    movie_1: dict,
    movie_2: dict,
    shared_summary: str = "",
    max_candidates: int = MAX_RERANK_CANDIDATES,
) -> dict[str, float]:
    """
    두 영화 A/B 와 후보 리스트를 Solar LLM 에 배치 전달하여 0~1 정규화 점수를 반환한다.

    ### 동작
    1. candidates 상위 max_candidates 편을 프롬프트에 포맷
    2. Solar API 호출 (hybrid 모드 기준 `get_explanation_llm()`)
    3. JSON 응답 파싱 → {movie_id: score} 반환
    4. 실패/타임아웃 시 빈 dict 반환 (에러 전파 금지, 호출자가 fallback)

    ### 반환값의 매치 스코어 융합 방식
    match_scorer 가 이 dict 를 읽어 `llm_score = result.get(movie_id, None)` 로 조회.
    dict 에 없는 영화는 LLM 이 평가하지 않은 것 → 기존 harmonic+cf 만으로 점수 계산.

    Args:
        candidates     : rag_retriever 출력 후보 영화 dict 리스트 (상위 15~25 편 추천)
        movie_1        : 첫 번째 선택 영화 dict (title, genres, mood_tags, overview, director)
        movie_2        : 두 번째 선택 영화 dict (동일 스키마)
        shared_summary : SharedFeatures.similarity_summary (feature_extractor 생성) 또는 빈 문자열
        max_candidates : LLM 에 투입할 최대 후보 수 (기본 10)

    Returns:
        {movie_id: llm_score_0_to_1} — 실패 시 빈 dict
    """
    node_start = time.perf_counter()

    # 방어적 입력 검증 — 빈 리스트면 LLM 호출 없이 즉시 반환
    if not candidates:
        logger.info("match_llm_reranker_empty_candidates")
        return {}

    # ── [1] 상위 N편만 선택 (토큰 + 지연 절감) ──
    # rag_retriever 가 이미 RRF 로 정렬했으므로 상위부터 자름
    top_candidates = candidates[:max_candidates]
    candidate_ids = {str(c.get("id", "")) for c in top_candidates if c.get("id")}

    # ── [2] 프롬프트 입력 구성 ──
    candidate_lines = [
        _format_candidate_line(c, i + 1) for i, c in enumerate(top_candidates)
    ]

    inputs = {
        "movie_1_title": movie_1.get("title", "영화 A"),
        "movie_1_genres": ", ".join(movie_1.get("genres", []) or []) or "미분류",
        "movie_1_moods": ", ".join(movie_1.get("mood_tags", []) or []) or "정보 없음",
        "movie_1_director": movie_1.get("director", "") or "정보 없음",
        "movie_1_overview": (movie_1.get("overview", "") or "")[:200] or "정보 없음",
        "movie_2_title": movie_2.get("title", "영화 B"),
        "movie_2_genres": ", ".join(movie_2.get("genres", []) or []) or "미분류",
        "movie_2_moods": ", ".join(movie_2.get("mood_tags", []) or []) or "정보 없음",
        "movie_2_director": movie_2.get("director", "") or "정보 없음",
        "movie_2_overview": (movie_2.get("overview", "") or "")[:200] or "정보 없음",
        "shared_summary": shared_summary or "(분석 정보 없음)",
        "candidate_count": len(top_candidates),
        "candidate_list": "\n".join(candidate_lines),
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", MATCH_RERANK_SYSTEM_PROMPT),
        ("human", MATCH_RERANK_HUMAN_PROMPT),
    ])

    # hybrid 모드 기준 Solar API 로 라우팅, local_only 면 EXAONE 32B
    # (feedback_llm_mode 메모리 기준 hybrid 가 기본이므로 Solar 가 정상)
    llm = get_explanation_llm()

    logger.info(
        "match_llm_reranker_start",
        movie_1_title=inputs["movie_1_title"],
        movie_2_title=inputs["movie_2_title"],
        candidate_count=len(top_candidates),
    )

    # ── [3] Solar API 호출 (타임아웃 보호) ──
    try:
        prompt_value = await prompt.ainvoke(inputs)
        response = await asyncio.wait_for(
            guarded_ainvoke(llm, prompt_value, model=settings.EXPLANATION_MODEL),
            timeout=MATCH_LLM_RERANK_TIMEOUT,
        )
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - node_start
        logger.warning(
            "match_llm_reranker_timeout",
            timeout_sec=MATCH_LLM_RERANK_TIMEOUT,
            elapsed_sec=round(elapsed, 2),
        )
        match_llm_reranker_total.labels(outcome="timeout").inc()
        match_llm_reranker_duration_seconds.observe(elapsed)
        return {}
    except Exception as e:
        elapsed = time.perf_counter() - node_start
        logger.error(
            "match_llm_reranker_llm_error",
            error=str(e),
            error_type=type(e).__name__,
            elapsed_sec=round(elapsed, 2),
            stack_trace=traceback.format_exc(),
        )
        match_llm_reranker_total.labels(outcome="exception").inc()
        match_llm_reranker_duration_seconds.observe(elapsed)
        return {}

    # ── [4] 응답 파싱 + 검증 ──
    response_text = (
        response.content if hasattr(response, "content") else str(response)
    ).strip()

    if not response_text:
        elapsed = time.perf_counter() - node_start
        logger.warning("match_llm_reranker_empty_response")
        match_llm_reranker_total.labels(outcome="empty_response").inc()
        match_llm_reranker_duration_seconds.observe(elapsed)
        return {}

    parsed = _parse_json_array(response_text)
    if parsed is None:
        elapsed = time.perf_counter() - node_start
        logger.warning(
            "match_llm_reranker_parse_error",
            response_preview=response_text[:200],
            elapsed_sec=round(elapsed, 2),
        )
        match_llm_reranker_total.labels(outcome="parse_error").inc()
        match_llm_reranker_duration_seconds.observe(elapsed)
        return {}

    scores = _validate_and_normalize(parsed, candidate_ids)

    elapsed = time.perf_counter() - node_start
    logger.info(
        "match_llm_reranker_complete",
        scored_count=len(scores),
        top_preview=sorted(
            scores.items(), key=lambda kv: kv[1], reverse=True,
        )[:5],
        elapsed_sec=round(elapsed, 2),
    )
    match_llm_reranker_total.labels(outcome="ok").inc()
    match_llm_reranker_duration_seconds.observe(elapsed)
    return scores
