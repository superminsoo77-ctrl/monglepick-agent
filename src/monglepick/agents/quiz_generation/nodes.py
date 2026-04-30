"""
영화 퀴즈 생성 에이전트 노드 함수.

7개 노드 순차 실행:
    movie_selector       → 후보 영화 풀 샘플링 (장르 / 인기도 / 최근 N일 quiz 제외)
    metadata_enricher    → overview / director / cast_members / keywords 보강
    question_generator   → 영화별 LLM 호출하여 4지선다 1문항 생성 (라운드로빈 카테고리)
    quality_validator    → 스키마 / 정답-options 일치 / 옵션 중복 / 스포일러 검증
    diversity_checker    → 동일 영화·동일 카테고리 중복 제거
    fallback_filler      → 검증 실패 영화에 대해 장르 기반 fallback 1문항 생성
    persistence          → quizzes 테이블 PENDING INSERT (실패는 스킵)

모든 노드는 try/except 로 감싸고, 실패 시 안전한 기본값을 반환한다 (에러 전파 금지).
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from monglepick.agents.quiz_generation.models import (
    CandidateMovie,
    GeneratedQuizRecord,
    QuizDraft,
    QuizGenerationState,
)
from monglepick.agents.quiz_generation.prompts import (
    CATEGORY_GUIDES,
    CATEGORY_ROTATION,
    QUIZ_SYSTEM_PROMPT,
    SPOILER_BLACKLIST,
    build_user_prompt,
)
from monglepick.db.clients import get_mysql
from monglepick.llm import get_conversation_llm, guarded_ainvoke

logger = structlog.get_logger()


# ============================================================
# 내부 유틸
# ============================================================


# 영어 장르 코드 → DB 한국어 라벨 매핑 (2026-04-29).
# Admin 프론트가 영어 코드(`action`, `drama`)를 보내면 `WHERE genres LIKE '%action%'` 매칭에 실패한다.
# DB(`movies.genres`) 는 한국어 배열(`["SF","드라마"]`)로 적재되기 때문 (data_pipeline/models.py).
# Admin UI 는 이미 한국어 value 로 통일했지만, 외부 호출/하위 호환을 위한 안전망으로 둔다.
# 매핑되지 않는 값은 그대로 LIKE 파라미터로 사용한다.
_GENRE_EN_TO_KO: dict[str, str] = {
    "action": "액션",
    "drama": "드라마",
    "comedy": "코미디",
    "horror": "공포",
    "romance": "로맨스",
    "sci-fi": "SF",
    "scifi": "SF",
    "sf": "SF",
    "thriller": "스릴러",
    "animation": "애니메이션",
    "fantasy": "판타지",
    "crime": "범죄",
    "mystery": "미스터리",
    "documentary": "다큐멘터리",
    "war": "전쟁",
    "western": "서부",
    "musical": "뮤지컬",
    "family": "가족",
    "history": "역사",
    "adventure": "모험",
}


def _normalize_genre_filter(raw: Optional[str]) -> Optional[str]:
    """
    퀴즈 생성 요청의 genre 파라미터를 DB LIKE 매칭 가능한 형태로 정규화한다.

    - None/빈 문자열 → None (필터 비활성)
    - 한국어가 그대로 들어오면 그대로 사용
    - 영어 코드(소문자)면 한국어로 변환
    - 매핑이 없으면 원본 그대로 (커스텀 장르 케이스 보존)
    """
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    # ASCII 만 들어왔다면 영어 코드로 간주하여 매핑 시도
    if cleaned.isascii():
        return _GENRE_EN_TO_KO.get(cleaned.lower(), cleaned)
    return cleaned


def _parse_json_array(raw: Any) -> list[str]:
    """
    JSON 배열 컬럼(genres/cast_members/keywords) 을 list[str] 로 파싱한다.

    저장 형태가 깨져 있어도(쉼표 구분 문자열 등) 안전하게 fallback 한다.
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    # 콤마 구분 문자열 fallback
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return []


def _parse_quiz_json(text: str) -> dict:
    """
    LLM 응답에서 퀴즈 JSON 객체를 추출한다.

    마크다운 코드블록(```json ... ```) 제거 + 중괄호 블록 추출 fallback.
    실패 시 빈 dict 반환.
    """
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
        cleaned = cleaned.rstrip("`").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        logger.warning("quiz_json_parse_failed", preview=text[:200])
        return {}


def _is_valid_options(options: Any) -> bool:
    """객관식 4지선다 스키마 검증 — 정확히 4개 + 모두 서로 다른 비어있지 않은 문자열."""
    if not isinstance(options, list) or len(options) != 4:
        return False
    str_options = [str(o).strip() for o in options]
    if any(not o for o in str_options):
        return False
    return len(set(str_options)) == 4


def _contains_spoiler(text: str) -> bool:
    """question 또는 explanation 에 스포일러 블랙리스트 단어가 포함되어 있는지 검사."""
    if not text:
        return False
    return any(word in text for word in SPOILER_BLACKLIST)


def _build_fallback_draft(movie: CandidateMovie, quiz_type: str = "auto") -> QuizDraft:
    """
    LLM 생성 실패 또는 quality_validator 거부 시 사용할 장르 기반 템플릿 fallback.

    가장 안전한 카테고리("이 영화의 주요 장르는?")로 폴백하여,
    빈 메타에서도 정답을 보장할 수 있다.
    """
    genres = movie.genres or ["드라마"]
    main_genre = genres[0]

    # 오답 풀 — 메인 장르와 다른 일반 장르 9종 중 3개
    decoy_pool = [
        "액션", "드라마", "코미디", "공포", "SF",
        "스릴러", "로맨스", "판타지", "애니메이션",
    ]
    decoys = [g for g in decoy_pool if g != main_genre][:3]
    options = [main_genre] + decoys

    return QuizDraft(
        movie_id=movie.movie_id,
        movie_title=movie.title,
        question=f"'{movie.title}' 영화의 주요 장르는 무엇인가요?",
        options=options,
        correct_answer=main_genre,
        explanation=(
            f"'{movie.title}' 은(는) '{main_genre}' 장르의 작품입니다. "
            f"포스터·로그라인·장르 태그를 통해 확인할 수 있습니다."
        ),
        category="genre",
        quiz_type=quiz_type,
        is_fallback=True,
        valid=True,
    )


# ============================================================
# 노드 1: movie_selector
# ============================================================


@traceable(name="quiz_generation.movie_selector")
async def movie_selector(state: QuizGenerationState) -> dict:
    """
    퀴즈 생성 후보 영화를 MySQL 에서 샘플링한다.

    forced_movie_id 가 설정된 경우 해당 영화만 단건 조회하고 샘플링을 건너뛴다.
    그 외:
        - 장르 필터: genres LIKE '%<genre>%' (필요 시)
        - 최근 N 일 동안 quiz 가 이미 생성된 영화는 제외 (중복 출제 방지)
        - popularity_score DESC ORDER + RAND() 가벼운 셔플로 인기도 가중 + 다양성 확보
        - 빈 DB / 매칭 실패 시 빈 리스트 + selector_message 로 안내
    """
    forced_movie_id = (state.get("forced_movie_id") or "").strip() or None

    # ── 관리자 지정 영화 모드: 단건 조회 ──
    if forced_movie_id:
        try:
            pool = await get_mysql()
            async with pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT movie_id, title, title_en, genres, release_year, release_date "
                        "FROM movies WHERE movie_id = %s LIMIT 1",
                        (forced_movie_id,),
                    )
                    row = await cur.fetchone()

            if not row:
                logger.warning("quiz_generation_forced_movie_not_found", movie_id=forced_movie_id)
                return {
                    "candidates": [],
                    "selector_message": f"영화 ID '{forced_movie_id}'를 찾을 수 없습니다.",
                }

            movie_id, title, title_en, genres_raw, release_year, release_date = row
            year_str = str(release_year) if release_year else (str(release_date)[:4] if release_date else "")
            candidate = CandidateMovie(
                movie_id=str(movie_id),
                title=title or title_en or "(제목 없음)",
                genres=_parse_json_array(genres_raw),
                release_year=year_str,
            )
            logger.info("quiz_generation_forced_movie_selected", movie_id=forced_movie_id, title=candidate.title)
            return {"candidates": [candidate], "selector_message": ""}

        except Exception as e:
            logger.error("quiz_generation_forced_movie_failed", error=str(e))
            return {
                "candidates": [],
                "selector_message": f"지정 영화 조회 중 오류가 발생했습니다: {e}",
            }

    # ── 자동 샘플링 모드 ──
    # 영어 코드(action) → 한국어(액션) 정규화 — DB(movies.genres) 가 한국어 배열로 저장되므로 필수.
    # 한국어 입력은 그대로 통과한다.
    genre = _normalize_genre_filter(state.get("genre"))
    count = max(1, int(state.get("count") or 5))
    exclude_recent_days = max(0, int(state.get("exclude_recent_days") or 7))

    candidates: list[CandidateMovie] = []
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # WHERE 절 동적 조립
                where_clauses: list[str] = []
                params: list[Any] = []

                if genre:
                    where_clauses.append("genres LIKE %s")
                    params.append(f"%{genre}%")

                # 최근 N 일 quiz 가 있는 영화 제외 (서브쿼리)
                if exclude_recent_days > 0:
                    where_clauses.append(
                        "movie_id NOT IN ("
                        "  SELECT DISTINCT movie_id FROM quizzes "
                        "  WHERE movie_id IS NOT NULL "
                        "    AND created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)"
                        ")"
                    )
                    params.append(exclude_recent_days)

                where_sql = (
                    "WHERE " + " AND ".join(where_clauses)
                    if where_clauses
                    else ""
                )

                # popularity 상위 1000편 풀에서 RAND() 추출 → 인기·다양성 균형
                sql = (
                    "SELECT movie_id, title, title_en, genres, release_year, release_date "
                    "FROM movies "
                    f"{where_sql} "
                    "ORDER BY popularity_score DESC, RAND() "
                    "LIMIT %s"
                )
                params.append(count * 3)  # 후속 검증 실패 대비 3배 풀링

                await cur.execute(sql, tuple(params))
                rows = await cur.fetchall()

        # 상위 count 편만 선택 (3배 풀에서 LLM 단계가 일부 실패해도 count 충당 가능)
        for row in rows[:count]:
            movie_id, title, title_en, genres_raw, release_year, release_date = row

            year_str = ""
            if release_year:
                year_str = str(release_year)
            elif release_date:
                year_str = str(release_date)[:4]

            candidates.append(CandidateMovie(
                movie_id=str(movie_id),
                title=title or title_en or "(제목 없음)",
                genres=_parse_json_array(genres_raw),
                release_year=year_str,
            ))

        logger.info(
            "quiz_generation_selector_done",
            requested=count,
            sampled=len(candidates),
            genre=genre,
            exclude_days=exclude_recent_days,
        )

        if not candidates:
            return {
                "candidates": [],
                "selector_message": (
                    "후보 영화가 없습니다. 영화 데이터를 먼저 적재하거나 "
                    "장르 필터를 해제해 주세요."
                ),
            }
        return {"candidates": candidates, "selector_message": ""}

    except Exception as e:
        logger.error("quiz_generation_selector_failed", error=str(e))
        return {
            "candidates": [],
            "selector_message": f"후보 영화 샘플링 중 오류가 발생했습니다: {e}",
        }


# ============================================================
# 노드 2: metadata_enricher
# ============================================================


@traceable(name="quiz_generation.metadata_enricher")
async def metadata_enricher(state: QuizGenerationState) -> dict:
    """
    선별된 영화의 풍부한 메타데이터를 한 번의 IN-쿼리로 보강한다.

    movie_selector 가 가져오지 않은 컬럼:
        overview, director, cast_members, keywords, tagline
    이들을 추가로 SELECT 하여 question_generator 가 카테고리별로 활용한다.
    """
    candidates: list[CandidateMovie] = list(state.get("candidates") or [])
    if not candidates:
        return {"enriched_candidates": []}

    movie_ids = [c.movie_id for c in candidates]
    enriched: dict[str, dict] = {}

    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                # IN 절 placeholder 구성
                placeholders = ", ".join(["%s"] * len(movie_ids))
                sql = (
                    "SELECT movie_id, overview, director, cast_members, keywords, tagline "
                    f"FROM movies WHERE movie_id IN ({placeholders})"
                )
                await cur.execute(sql, tuple(movie_ids))
                rows = await cur.fetchall()

        for row in rows:
            mid, overview, director, cast_raw, kw_raw, tagline = row
            enriched[str(mid)] = {
                "overview": overview or "",
                "director": (director or "").strip(),
                "cast_members": _parse_json_array(cast_raw),
                "keywords": _parse_json_array(kw_raw),
                "tagline": (tagline or "").strip(),
            }

        # 후보 모델 갱신
        out: list[CandidateMovie] = []
        for c in candidates:
            extra = enriched.get(c.movie_id, {})
            out.append(c.model_copy(update={
                "overview": extra.get("overview", ""),
                "director": extra.get("director", ""),
                "cast_members": extra.get("cast_members", []),
                "keywords": extra.get("keywords", []),
                "tagline": extra.get("tagline", ""),
            }))

        logger.info(
            "quiz_generation_enricher_done",
            count=len(out),
            with_overview=sum(1 for c in out if c.overview),
            with_director=sum(1 for c in out if c.director),
        )
        return {"enriched_candidates": out}

    except Exception as e:
        logger.warning("quiz_generation_enricher_failed", error=str(e))
        # 메타 보강 실패해도 기본 필드만으로 진행 (에러 전파 금지)
        return {"enriched_candidates": candidates}


# ============================================================
# 노드 3: question_generator
# ============================================================


def _select_category(movie: CandidateMovie, index: int) -> str:
    """
    영화 메타에 맞춰 카테고리를 라운드로빈으로 선택한다.

    필수 메타가 비어 있는 카테고리는 자동으로 'general' 로 다운그레이드한다.
    예: director 카테고리인데 movie.director 가 빈 문자열 → general
    """
    rotated = CATEGORY_ROTATION[index % len(CATEGORY_ROTATION)]

    # 메타 가용성 체크 — 비면 general 로 강등하여 LLM 환각 방지
    if rotated == "director" and not movie.director:
        return "general"
    if rotated == "year" and not movie.release_year:
        return "general"
    if rotated == "cast" and not movie.cast_members:
        return "general"
    if rotated == "plot" and not movie.overview:
        return "general"
    return rotated


def _select_forced_category(movie: CandidateMovie, quiz_type: str) -> str:
    """
    관리자 지정 quiz_type 을 카테고리로 변환한다.

    메타가 비어있으면 'general' 로 강등하여 LLM 환각을 방지한다.
    """
    from monglepick.agents.quiz_generation.prompts import CATEGORY_GUIDES
    if quiz_type not in CATEGORY_GUIDES:
        return "general"
    if quiz_type == "plot" and not movie.overview:
        return "general"
    if quiz_type == "cast" and not movie.cast_members:
        return "general"
    if quiz_type == "director" and not movie.director:
        return "general"
    if quiz_type == "year" and not movie.release_year:
        return "general"
    return quiz_type


async def _generate_one_quiz_llm(
    movie: CandidateMovie,
    category: str,
    difficulty: str,
    quiz_type: str = "auto",
) -> QuizDraft:
    """
    단일 영화에 대해 LLM 호출 1회로 4지선다 1문항을 생성한다.

    LLM 응답 파싱 실패 / 스키마 미달 / 예외 발생 시 fallback 초안을 반환한다.
    """
    try:
        llm = get_conversation_llm()
        system_text = QUIZ_SYSTEM_PROMPT.format(
            category=category,
            difficulty=difficulty,
        )
        user_text = build_user_prompt(
            title=movie.title,
            genres=movie.genres,
            release_year=movie.release_year,
            director=movie.director,
            cast_members=movie.cast_members,
            overview=movie.overview,
            tagline=movie.tagline,
            keywords=movie.keywords,
            category=category,
        )

        response = await guarded_ainvoke(
            llm,
            [SystemMessage(content=system_text), HumanMessage(content=user_text)],
        )
        text = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_quiz_json(text)

        # 스키마 1차 검증 (질문 + options 4개 + 정답 일치)
        if (
            isinstance(parsed, dict)
            and parsed.get("question")
            and _is_valid_options(parsed.get("options"))
            and parsed.get("correctAnswer") in parsed["options"]
        ):
            return QuizDraft(
                movie_id=movie.movie_id,
                movie_title=movie.title,
                question=str(parsed["question"]).strip(),
                options=[str(o).strip() for o in parsed["options"]],
                correct_answer=str(parsed["correctAnswer"]).strip(),
                explanation=str(parsed.get("explanation") or "").strip(),
                category=str(parsed.get("category") or category),
                quiz_type=quiz_type,
                is_fallback=False,
                valid=True,
            )

        logger.warning(
            "quiz_generation_llm_invalid_schema",
            movie_id=movie.movie_id,
            parsed_keys=list(parsed.keys()) if isinstance(parsed, dict) else None,
        )
    except Exception as e:
        logger.warning(
            "quiz_generation_llm_failed",
            movie_id=movie.movie_id,
            error=str(e),
        )

    return _build_fallback_draft(movie, quiz_type=quiz_type)


@traceable(name="quiz_generation.question_generator")
async def question_generator(state: QuizGenerationState) -> dict:
    """
    enriched_candidates 각각에 대해 LLM 으로 4지선다 1문항을 생성한다.

    quiz_type 이 'auto' 가 아니면 해당 카테고리를 강제 사용한다.
    영화별 호출은 asyncio.gather 로 병렬화하여 전체 지연시간을 줄인다.
    개별 영화 실패는 _generate_one_quiz_llm 내부에서 fallback 초안으로 흡수된다.
    """
    movies: list[CandidateMovie] = list(state.get("enriched_candidates") or state.get("candidates") or [])
    if not movies:
        return {"drafts": []}

    difficulty = (state.get("difficulty") or "medium").lower()
    quiz_type = (state.get("quiz_type") or "auto").lower()

    tasks = []
    for i, m in enumerate(movies):
        if quiz_type == "auto":
            category = _select_category(m, i)
        else:
            category = _select_forced_category(m, quiz_type)
        tasks.append(_generate_one_quiz_llm(m, category, difficulty, quiz_type=quiz_type))

    drafts = await asyncio.gather(*tasks, return_exceptions=False)
    drafts_list: list[QuizDraft] = list(drafts)

    logger.info(
        "quiz_generation_question_done",
        total=len(drafts_list),
        quiz_type=quiz_type,
        fallback_count=sum(1 for d in drafts_list if d.is_fallback),
    )
    return {"drafts": drafts_list}


# ============================================================
# 노드 4: quality_validator
# ============================================================


@traceable(name="quiz_generation.quality_validator")
async def quality_validator(state: QuizGenerationState) -> dict:
    """
    각 초안에 대해 품질 검증을 수행하고 valid 플래그를 세팅한다.

    검증 항목:
        - question / correct_answer 가 비어 있지 않을 것
        - options 가 정확히 4개 + 모두 서로 다른 문자열
        - correct_answer 가 options 중 하나와 일치
        - question / explanation 에 스포일러 블랙리스트 단어 포함 금지
        - question 길이 10자 이상 / 200자 이하 (너무 짧거나 너무 긴 질문 차단)
    """
    drafts: list[QuizDraft] = list(state.get("drafts") or [])
    validated: list[QuizDraft] = []

    for draft in drafts:
        try:
            reasons: list[str] = []

            if not draft.question or not draft.question.strip():
                reasons.append("EMPTY_QUESTION")
            elif len(draft.question) < 10:
                reasons.append("QUESTION_TOO_SHORT")
            elif len(draft.question) > 200:
                reasons.append("QUESTION_TOO_LONG")

            if not _is_valid_options(draft.options):
                reasons.append("INVALID_OPTIONS")

            if draft.correct_answer not in draft.options:
                reasons.append("ANSWER_NOT_IN_OPTIONS")

            if _contains_spoiler(draft.question) or _contains_spoiler(draft.explanation):
                reasons.append("SPOILER_DETECTED")

            if reasons:
                # fallback 은 무조건 통과 (장르 템플릿은 스키마 안전)
                if draft.is_fallback:
                    validated.append(draft)
                    continue
                validated.append(draft.model_copy(update={
                    "valid": False,
                    "reject_reason": ",".join(reasons),
                }))
                logger.warning(
                    "quiz_generation_validator_rejected",
                    movie_id=draft.movie_id,
                    reasons=reasons,
                )
            else:
                validated.append(draft)

        except Exception as e:
            logger.warning(
                "quiz_generation_validator_error",
                movie_id=draft.movie_id,
                error=str(e),
            )
            validated.append(draft.model_copy(update={
                "valid": False,
                "reject_reason": "VALIDATOR_EXCEPTION",
            }))

    logger.info(
        "quiz_generation_validator_done",
        total=len(validated),
        valid=sum(1 for d in validated if d.valid),
    )
    return {"validated_drafts": validated}


# ============================================================
# 노드 5: diversity_checker
# ============================================================


@traceable(name="quiz_generation.diversity_checker")
async def diversity_checker(state: QuizGenerationState) -> dict:
    """
    동일 (movie_id, question 본문) 또는 동일 (movie_id, category) 중복을 제거한다.

    LLM 이 동일 영화에 대해 같은 질문을 반복 생성하는 케이스를 차단한다.
    배치 내 중복만 검사 — DB 레벨 중복 검사는 movie_selector 의
    exclude_recent_days 가 담당한다.
    """
    drafts: list[QuizDraft] = list(state.get("validated_drafts") or [])
    if not drafts:
        return {"diversified_drafts": []}

    seen_questions: set[tuple[str, str]] = set()
    seen_movie_category: set[tuple[str, str]] = set()
    out: list[QuizDraft] = []

    for d in drafts:
        if not d.valid:
            out.append(d)
            continue

        q_key = (d.movie_id, d.question.strip())
        c_key = (d.movie_id, d.category)

        if q_key in seen_questions or c_key in seen_movie_category:
            out.append(d.model_copy(update={
                "valid": False,
                "reject_reason": "DUPLICATE_IN_BATCH",
            }))
            logger.info(
                "quiz_generation_diversity_dedup",
                movie_id=d.movie_id,
                category=d.category,
            )
            continue

        seen_questions.add(q_key)
        seen_movie_category.add(c_key)
        out.append(d)

    logger.info(
        "quiz_generation_diversity_done",
        total=len(out),
        kept=sum(1 for d in out if d.valid),
    )
    return {"diversified_drafts": out}


# ============================================================
# 노드 6: fallback_filler
# ============================================================


@traceable(name="quiz_generation.fallback_filler")
async def fallback_filler(state: QuizGenerationState) -> dict:
    """
    valid=False 처리된 초안을 fallback 으로 1회 보충한다.

    관리자가 특정 영화를 직접 지정한 모드(forced_movie_id + quiz_type != 'auto')에서는
    장르 fallback 을 생성하지 않는다 — 퀄리티 보장 목적.
    그 외 자동 모드에서는 같은 영화의 valid 초안이 없을 때만 장르 fallback 1건을 추가한다.
    """
    drafts: list[QuizDraft] = list(state.get("diversified_drafts") or [])
    forced_movie_id = (state.get("forced_movie_id") or "").strip() or None
    quiz_type = (state.get("quiz_type") or "auto").lower()

    # 영화 선택 모드: 검증 통과한 초안만 반환 (장르 fallback 금지)
    if forced_movie_id and quiz_type != "auto":
        final = [d for d in drafts if d.valid]
        logger.info(
            "quiz_generation_fallback_skipped_forced_mode",
            kept=len(final),
            rejected=len(drafts) - len(final),
        )
        return {"final_drafts": final}

    candidates: list[CandidateMovie] = list(
        state.get("enriched_candidates") or state.get("candidates") or []
    )
    cand_index: dict[str, CandidateMovie] = {c.movie_id: c for c in candidates}

    valid_movie_ids = {d.movie_id for d in drafts if d.valid}
    final: list[QuizDraft] = []

    for d in drafts:
        if d.valid:
            final.append(d)
            continue

        # 이 영화가 이미 valid 초안을 가지고 있으면 fallback 추가 X
        if d.movie_id in valid_movie_ids:
            continue

        movie = cand_index.get(d.movie_id)
        if not movie:
            continue

        fb = _build_fallback_draft(movie, quiz_type=quiz_type)
        valid_movie_ids.add(d.movie_id)
        final.append(fb)

    logger.info(
        "quiz_generation_fallback_done",
        final=len(final),
        fallback_added=sum(1 for d in final if d.is_fallback),
    )
    return {"final_drafts": final}


# ============================================================
# 노드 7: persistence
# ============================================================


async def _insert_quiz_pending(
    movie_id: str,
    question: str,
    correct_answer: str,
    options: list[str],
    explanation: Optional[str],
    reward_point: int,
    quiz_type: str = "auto",
) -> int:
    """
    검증 통과한 퀴즈 초안 1건을 quizzes 테이블에 PENDING 으로 INSERT 한다.

    Backend(Quiz 엔티티) 컬럼 1:1 매핑. created_by/updated_by 는 'ai-agent'.
    quiz_type: 'auto' | 'plot' | 'cast' | 'director' | 'genre'
    """
    options_json = json.dumps(options, ensure_ascii=False)
    pool = await get_mysql()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO quizzes (
                    movie_id, question, explanation, correct_answer,
                    options, reward_point, status, quiz_date,
                    quiz_type, created_at, updated_at, created_by, updated_by
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, 'PENDING', NULL,
                    %s, NOW(), NOW(), 'ai-agent', 'ai-agent'
                )
                """,
                (
                    movie_id,
                    question,
                    explanation,
                    correct_answer,
                    options_json,
                    reward_point,
                    quiz_type,
                ),
            )
            await conn.commit()
            return int(cur.lastrowid or 0)


@traceable(name="quiz_generation.persistence")
async def persistence(state: QuizGenerationState) -> dict:
    """
    final_drafts 중 valid=True 만 quizzes 테이블에 PENDING INSERT 한다.

    개별 INSERT 실패는 로그만 남기고 다음 항목으로 넘어간다 (배치 단위 best-effort).
    """
    drafts: list[QuizDraft] = list(state.get("final_drafts") or [])
    reward_point = max(1, int(state.get("reward_point") or 10))
    persisted: list[GeneratedQuizRecord] = []

    for d in drafts:
        if not d.valid:
            continue
        try:
            quiz_id = await _insert_quiz_pending(
                movie_id=d.movie_id,
                question=d.question,
                correct_answer=d.correct_answer,
                options=d.options,
                explanation=d.explanation or None,
                reward_point=reward_point,
                quiz_type=d.quiz_type,
            )
            persisted.append(GeneratedQuizRecord(
                quiz_id=quiz_id,
                movie_id=d.movie_id,
                movie_title=d.movie_title,
                question=d.question,
                correct_answer=d.correct_answer,
                options=d.options,
                explanation=d.explanation or None,
                reward_point=reward_point,
                status="PENDING",
            ))
        except Exception as e:
            logger.error(
                "quiz_generation_persistence_failed",
                movie_id=d.movie_id,
                error=str(e),
            )

    count = len(persisted)
    selector_msg = state.get("selector_message") or ""

    if count > 0:
        message = f"AI 가 퀴즈 {count}개를 생성하여 PENDING 으로 등록했습니다."
    elif selector_msg:
        message = selector_msg
    else:
        message = "퀴즈 생성에 실패했습니다. 로그를 확인해 주세요."

    logger.info(
        "quiz_generation_persistence_done",
        persisted=count,
        success=count > 0,
    )
    return {
        "persisted": persisted,
        "final_message": message,
        "success": count > 0,
    }
