"""
LLM 재랭킹 체인 (Phase Q: 추천 품질 개선).

RAG 검색 결과를 사용자 의도 기반으로 의미론적 재평가한다.
DB 필터나 벡터 검색으로 잡을 수 없는 조건들을 LLM의 세계 지식으로 판단한다.

처리 흐름:
1. 후�� 영화 메타데이터를 간결한 텍스��로 포맷
2. 사용자 요청 + 감정/선호 컨텍스트 구성
3. Solar API LLM으로 적합��� 점수(0~10) 평가
4. 점수 기반으로 후보 재정렬 + 부적합 영화 제거
5. 에러 시: 원본 순서 유지 (graceful degradation)

사용 LLM: Solar API (hybrid 모드, temp=0.3)
"""

from __future__ import annotations

import json
import time
import traceback
from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate

from monglepick.agents.chat.models import (
    CandidateMovie,
    EmotionResult,
    ExtractedPreferences,
)
from monglepick.llm.factory import get_solar_api_llm, guarded_ainvoke
from monglepick.prompts.rerank import (
    RERANK_HUMAN_PROMPT,
    RERANK_SYSTEM_PROMPT,
)

logger = structlog.get_logger()

# 재랭킹 대상 최대 후보 수 (토큰 절약)
MAX_RERANK_CANDIDATES = 10

# 재랭킹 절대 하한 점수 — Solar LLM 이 0.0 을 거의 매기지 않는 경향을 고려한 최소 컷
MIN_RERANK_SCORE = 3.0

# 상대 임계값 배율 — 평균 점수가 높을수록 컷오프도 비례해 올라간다.
# 예) 평균 7.0 → 컷오프 = max(3.0, 7.0*0.7) = 4.9
#     평균 4.0 → 컷오프 = max(3.0, 4.0*0.7) = 3.0 (절대 하한)
# 이 방식이 필요한 이유: Solar 가 "애매해도 4~5점" 을 자비롭게 부여해 절대 하한만으로는
# 거의 필터링이 되지 않는다. 평균 대비 현저히 낮은 후보만 제외하는 형태로 동작.
RELATIVE_CUTOFF_RATIO = 0.7

# 재랭킹 후 최소 유지 후보 수 — TOP_K(5)와 동일하게 설정하여 항상 5편 추천 보장
MIN_KEEP_CANDIDATES = 5


def _format_candidate_for_rerank(movie: CandidateMovie, idx: int) -> str:
    """
    후보 영화를 재랭킹 프롬프트에 넣을 간결한 텍스트로 포맷한다.

    토큰 절약을 위해 핵심 메타데이터만 포함:
    - 제목, 장르, 감독, 평점, 개봉연도, 무드태그, 줄거리(150자)

    Args:
        movie: 후보 ���화
        idx: 1-based 인덱스

    Returns:
        "1. 인셉션 (2010) | 장르: SF, 스릴러 | 감독: 크리스토퍼 놀란 | 평점: 8.8 | ..."
    """
    parts = [f"{idx}. **{movie.title}**"]

    if movie.release_year:
        parts[0] += f" ({movie.release_year})"

    if movie.genres:
        parts.append(f"장르: {', '.join(movie.genres[:5])}")
    if movie.director:
        parts.append(f"감독: {movie.director}")
    if movie.rating > 0:
        parts.append(f"평점: {movie.rating}")
    if movie.mood_tags:
        parts.append(f"무드: {', '.join(movie.mood_tags[:4])}")
    if movie.trailer_url:
        parts.append("트레일러: 있음")
    else:
        parts.append("트레일러: 없음")

    # ── 국가/언어 정보 (한국영화 등 국가 기반 추천 시 LLM이 판단할 수 있도록 포함) ──
    if movie.original_language:
        parts.append(f"언어: {movie.original_language}")
    if movie.origin_country:
        parts.append(f"국가: {', '.join(movie.origin_country)}")

    # 줄거리는 150자로 제한 (토큰 절약)
    if movie.overview and len(movie.overview.strip()) >= 20:
        overview = movie.overview[:150]
        if len(movie.overview) > 150:
            overview += "..."
        parts.append(f"줄거리: {overview}")
    else:
        parts.append("줄거리: 없음")

    # ── Phase Q-2: 데이터 품질 신호를 LLM에 명시적으로 전달 ──
    # 포스터 유무 (없으면 LLM이 "데이터 부족" 감점 판단에 활용)
    if not movie.poster_path or not movie.poster_path.strip():
        parts.append("포스터: 없음")

    # movie_id를 마지막에 추가 (LLM이 응답에 포함하도록)
    parts.append(f"[ID: {movie.id}]")

    return " | ".join(parts)


def _format_user_context(
    emotion: EmotionResult | None,
    preferences: ExtractedPreferences | None,
) -> str:
    """
    사용자 감정/선호 컨텍스트를 프롬프트용 문자열로 포맷한다.

    Args:
        emotion: 감정 분석 결과
        preferences: 선호 조건

    Returns:
        컨텍스트 문자열 (없으면 "(없음)")
    """
    parts = []

    if emotion and emotion.emotion:
        parts.append(f"감정: {emotion.emotion}")
    if emotion and emotion.mood_tags:
        parts.append(f"무드: {', '.join(emotion.mood_tags[:5])}")
    if preferences:
        if preferences.user_intent:
            parts.append(f"추천 의도: {preferences.user_intent}")
        if preferences.genre_preference:
            parts.append(f"선호 장르: {preferences.genre_preference}")
        if preferences.mood:
            parts.append(f"선호 분위기: {preferences.mood}")
        if preferences.dynamic_filters:
            filter_strs = [
                f"{f.field} {f.operator} {f.value}" for f in preferences.dynamic_filters
            ]
            parts.append(f"필터 조건: {', '.join(filter_strs)}")

    return "\n".join(parts) if parts else "(없음)"


def _parse_rerank_response(response_text: str, candidate_ids: set[str]) -> list[dict[str, Any]]:
    """
    LLM 재랭킹 응답을 파싱하여 {movie_id, score, reason} ��스트를 반환한다.

    3단계 파싱 전략:
    1. 전체 JSON 배열 파싱
    2. JSON 코드블록 추출 후 파싱
    3. 개별 라인에서 JSON 객체 추출

    파싱 실패 시 빈 리스트 반환 (원본 순서 유지 fallback).

    Args:
        response_text: LLM 응답 텍스트
        candidate_ids: ���효한 후보 영화 ID 집합

    Returns:
        [{movie_id, score, reason}, ...] — 파싱 실패 시 빈 리스트
    """
    # 1단계: 전체 JSON 배열 파싱 시도
    try:
        data = json.loads(response_text)
        if isinstance(data, list):
            return _validate_rerank_items(data, candidate_ids)
    except json.JSONDecodeError:
        pass

    # 2단계: JSON 코드블��� 추출
    import re
    json_blocks = re.findall(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", response_text)
    for block in json_blocks:
        try:
            data = json.loads(block)
            if isinstance(data, list):
                return _validate_rerank_items(data, candidate_ids)
        except json.JSONDecodeError:
            continue

    # 3단계: 배열 패턴 추출 (코드블록 없이 직접 출력된 경우)
    array_match = re.search(r"\[[\s\S]*\]", response_text)
    if array_match:
        try:
            data = json.loads(array_match.group())
            if isinstance(data, list):
                return _validate_rerank_items(data, candidate_ids)
        except json.JSONDecodeError:
            pass

    logger.warning(
        "rerank_response_parse_failed",
        response_preview=response_text[:200],
    )
    return []


def _validate_rerank_items(
    items: list[dict],
    candidate_ids: set[str],
) -> list[dict[str, Any]]:
    """
    파싱된 재랭킹 ���이템을 검증하고 정규화한다.

    - movie_id가 후보 목록에 없으면 제외
    - score가 0~10 범위 밖이면 클램핑
    - reason이 없으면 빈 문자열

    Args:
        items: 파싱�� 아이템 리스트
        candidate_ids: 유효한 후보 영화 ID 집합

    Returns:
        검증된 [{movie_id, score, reason}, ...]
    """
    validated = []
    for item in items:
        if not isinstance(item, dict):
            continue

        movie_id = str(item.get("movie_id", ""))
        score = item.get("score", 5)
        reason = str(item.get("reason", ""))

        # movie_id 유효성 검증
        if movie_id not in candidate_ids:
            # ID가 정확히 매칭되지 않으면 부분 매칭 시도
            matched = [cid for cid in candidate_ids if cid in movie_id or movie_id in cid]
            if matched:
                movie_id = matched[0]
            else:
                continue

        # score 클램핑
        try:
            score = max(0.0, min(10.0, float(score)))
        except (ValueError, TypeError):
            score = 5.0

        validated.append({
            "movie_id": movie_id,
            "score": score,
            "reason": reason,
        })

    return validated


async def rerank_candidates(
    candidates: list[CandidateMovie],
    user_request: str,
    emotion: EmotionResult | None = None,
    preferences: ExtractedPreferences | None = None,
) -> list[CandidateMovie]:
    """
    LLM 기반으로 후보 영화를 사용자 의도에 맞게 재랭킹한다.

    RAG 검색 결과를 LLM의 세계 지식으로 재평가하여:
    - 사용자 ���청에 부적합한 후보를 제거/감점
    - 적합한 후보를 상위로 이동
    - DB에 없는 정보(수상 이력, 실화 여부 등)도 LLM 지식으로 판단

    에러 시 원본 순서를 그대로 반환한다 (graceful degradation).

    Args:
        candidates: RAG 검색 결과 후보 영화 ���스트 (RRF 순서)
        user_request: 사��자 원본 요청 텍스트
        emotion: 감정 분석 결과 (선택)
        preferences: 선호 조건 (선택)

    Returns:
        재랭킹된 CandidateMovie 리스트 (부적합 영화 제거됨)
    """
    # 후보가 없거나 1편이면 재랭킹 불필요
    if len(candidates) <= 1:
        return candidates

    rerank_start = time.perf_counter()

    try:
        # 재랭킹 대상을 MAX_RERANK_CANDIDATES로 제한 (토큰 절약)
        rerank_targets = candidates[:MAX_RERANK_CANDIDATES]
        remaining = candidates[MAX_RERANK_CANDIDATES:]

        # 프롬프트 ���성
        candidate_list = "\n".join(
            _format_candidate_for_rerank(m, i + 1)
            for i, m in enumerate(rerank_targets)
        )
        user_context = _format_user_context(emotion, preferences)

        prompt = ChatPromptTemplate.from_messages([
            ("system", RERANK_SYSTEM_PROMPT),
            ("human", RERANK_HUMAN_PROMPT),
        ])

        # Solar API LLM (temp=0.3, 안정적 판단)
        llm = get_solar_api_llm(temperature=0.3)

        inputs = {
            "user_request": user_request,
            "user_context": user_context,
            "candidate_count": len(rerank_targets),
            "candidate_list": candidate_list,
        }

        logger.info(
            "rerank_chain_start",
            candidate_count=len(rerank_targets),
            user_request_preview=user_request[:80],
        )

        # LLM 호출 (세마포어로 동시 호출 제한)
        prompt_value = await prompt.ainvoke(inputs)
        response = await guarded_ainvoke(
            llm, prompt_value, model="solar-pro",
        )

        # 응답 텍스트 추출
        response_text = response.content if hasattr(response, "content") else str(response)

        elapsed_ms = (time.perf_counter() - rerank_start) * 1000

        logger.debug(
            "rerank_chain_llm_response",
            response_preview=response_text[:300],
            elapsed_ms=round(elapsed_ms, 1),
        )

        # 응답 파싱
        candidate_ids = {m.id for m in rerank_targets}
        rerank_results = _parse_rerank_response(response_text, candidate_ids)

        if not rerank_results:
            # 파싱 실패: 원본 순서 유지
            logger.warning("rerank_parse_failed_keeping_original", elapsed_ms=round(elapsed_ms, 1))
            return candidates

        # 점수 맵 구성
        score_map: dict[str, float] = {}
        reason_map: dict[str, str] = {}
        for item in rerank_results:
            score_map[item["movie_id"]] = item["score"]
            reason_map[item["movie_id"]] = item["reason"]

        # LLM이 평가하지 않은 후보는 중간 점수(5.0) 부여
        for m in rerank_targets:
            if m.id not in score_map:
                score_map[m.id] = 5.0

        # 점수 기반 재정렬
        reranked = sorted(
            rerank_targets,
            key=lambda m: score_map.get(m.id, 5.0),
            reverse=True,
        )

        # 부적합 영화 제거 — 절대 하한(MIN_RERANK_SCORE)과 상대 임계값(평균의 70%) 중 큰 값.
        # 상대 임계값이 필요한 이유: Solar LLM 은 애매한 후보에도 4~5점을 자비롭게 부여해
        # 절대 하한만으로는 "애매한 후보 무더기" 를 걸러내지 못한다. 평균 대비 현저히 낮은
        # 후보만 쳐냄으로써 후보군이 전반적으로 강할 때는 컷오프가 자동으로 올라가도록 한다.
        evaluated_scores = [score_map.get(m.id, 5.0) for m in reranked]
        avg_score = (
            sum(evaluated_scores) / len(evaluated_scores) if evaluated_scores else 0.0
        )
        effective_cutoff = max(MIN_RERANK_SCORE, avg_score * RELATIVE_CUTOFF_RATIO)
        filtered = [m for m in reranked if score_map.get(m.id, 5.0) >= effective_cutoff]

        # 최소 후보 수 보장 (너무 많이 제거되면 상위 N개 유지)
        if len(filtered) < MIN_KEEP_CANDIDATES:
            filtered = reranked[:MIN_KEEP_CANDIDATES]

        # 나머지 후보(MAX_RERANK_CANDIDATES 초과분)를 뒤에 추가
        result = filtered + remaining

        elapsed_ms = (time.perf_counter() - rerank_start) * 1000
        logger.info(
            "rerank_completed",
            original_count=len(rerank_targets),
            reranked_count=len(filtered),
            removed_count=len(rerank_targets) - len(filtered),
            avg_llm_score=round(avg_score, 2),
            effective_cutoff=round(effective_cutoff, 2),
            top_results=[
                {
                    "title": m.title,
                    "llm_score": score_map.get(m.id, 5.0),
                    "reason": reason_map.get(m.id, "")[:50],
                    "rrf_score": round(m.rrf_score, 4),
                }
                for m in filtered[:5]
            ],
            elapsed_ms=round(elapsed_ms, 1),
        )

        return result

    except Exception as e:
        elapsed_ms = (time.perf_counter() - rerank_start) * 1000
        logger.error(
            "rerank_chain_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            elapsed_ms=round(elapsed_ms, 1),
        )
        # 에러 시: 원본 순서 유지 (graceful degradation)
        return candidates
