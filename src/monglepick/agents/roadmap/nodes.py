"""
로드맵 에이전트 노드 함수 (§9-3, §9-5, Phase 7).

LangGraph StateGraph의 각 노드로 등록되는 4개 async 함수.
시그니처: async def node_name(state: RoadmapAgentState) -> dict

노드 목록:
1. user_segment_analyzer — 시청 이력 기반 사용자 레벨 판정 (규칙 기반)
2. roadmap_generator     — MySQL 검색 → 단계별 5편 선정 (이미 본 영화 제외)
3. quiz_generator        — 15편 영화 퀴즈 생성 (LLM + fallback 템플릿)
4. roadmap_formatter     — 단계별 소개글 + UUID + 최종 구조 조립

모든 노드는 try/except로 감싸고, 에러 시 유효한 기본값을 반환한다 (에러 전파 금지).
반환값은 dict — LangGraph 컨벤션 (TypedDict State 일부 업데이트).
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from monglepick.agents.roadmap.state import (
    FormattedRoadmap,
    Quiz,
    QuizQuestion,
    RoadmapAgentState,
    RoadmapMovie,
    RoadmapStage,
)
from monglepick.db.clients import get_mysql
from monglepick.llm import get_conversation_llm, guarded_ainvoke

logger = structlog.get_logger()

# ── 단계별 선정 영화 수 ──
_MOVIES_PER_STAGE = 5

# ── popularity 백분위 경계 ──
# beginner  : popularity 상위 70% (대중적, 쉽게 접근 가능)
# intermediate : 30~70%
# expert    : 하위 30% (마니아 대상, 덜 알려진 작품)
_BEGINNER_PERCENTILE    = 0.70
_INTERMEDIATE_LOW       = 0.30

# ============================================================
# 내부 유틸
# ============================================================

def _parse_json_safe(text: str, context: str = "") -> dict | list:
    """
    LLM 응답 텍스트에서 JSON을 안전하게 파싱한다.

    마크다운 코드블록(```json ... ```)을 제거하고 순수 JSON을 추출한다.
    파싱 실패 시 빈 dict를 반환하고 로그를 남긴다.

    Args:
        text   : LLM 응답 텍스트
        context: 로그용 컨텍스트 설명

    Returns:
        파싱된 dict 또는 list, 실패 시 빈 dict
    """
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 중괄호 또는 대괄호 블록 추출 시도
        for pattern in (r"\{.*\}", r"\[.*\]"):
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        logger.warning("json_parse_failed", context=context, preview=text[:200])
        return {}


def _make_fallback_quiz(movie: dict) -> dict:
    """
    LLM 퀴즈 생성 실패 시 영화 메타데이터 기반 기본 템플릿 퀴즈를 생성한다.

    장르 객관식 + 개봉연도 주관식 2문항으로 구성한다.

    Args:
        movie: 영화 dict (id, title, genres, release_year 포함)

    Returns:
        Quiz 호환 dict
    """
    title = movie.get("title", "이 영화")
    genres = movie.get("genres", ["드라마"])
    main_genre = genres[0] if genres else "드라마"
    release_year = str(movie.get("release_year", ""))

    # 객관식 오답 보기 풀 (주요 장르 중 main_genre 제외)
    decoy_genres = [g for g in ["액션", "드라마", "코미디", "공포", "SF", "스릴러", "로맨스"] if g != main_genre]
    options = [main_genre] + decoy_genres[:3]

    return {
        "movie_id": movie.get("id", ""),
        "questions": [
            {
                "type": "multiple_choice",
                "question": f"'{title}' 영화의 주요 장르는 무엇인가요?",
                "options": options,
                "answer": main_genre,
                "hint": "영화 소개나 포스터에서 확인할 수 있어요.",
            },
            {
                "type": "short_answer",
                "question": f"'{title}' 영화가 개봉한 연도는?",
                "options": [],
                "answer": release_year,
                "hint": "영화 정보 페이지에서 개봉연도를 확인할 수 있어요.",
            },
        ],
    }


# ============================================================
# Node 1: user_segment_analyzer
# ============================================================

@traceable(name="user_segment_analyzer", run_type="chain", metadata={"node": "1/4"})
async def user_segment_analyzer(state: RoadmapAgentState) -> dict:
    """
    사용자 시청 이력을 분석하여 레벨을 판정한다 (규칙 기반, LLM 없음).

    판정 기준:
    - beginner     : 총 시청 < 20편 또는 고유 장르 <= 3
    - expert       : 총 시청 > 100편 또는 단일 장르 시청 >= 50편
    - intermediate : 그 외 모든 경우

    반환 state 키:
    - user_level  : "beginner" | "intermediate" | "expert"
    - level_detail: 판정 상세 dict

    Args:
        state: RoadmapAgentState

    Returns:
        {"user_level": str, "level_detail": dict}
    """
    try:
        watch_history: list[dict] = state.get("watch_history", [])
        user_id = state.get("user_id", "unknown")

        logger.info(
            "user_segment_analyzer_start",
            user_id=user_id,
            watch_count=len(watch_history),
        )

        # ── 통계 계산 ──
        total_watched = len(watch_history)

        genre_counter: Counter[str] = Counter()
        ratings: list[float] = []
        for rec in watch_history:
            for genre in rec.get("genres", []):
                genre_counter[genre.strip()] += 1
            r = rec.get("rating")
            if r is not None:
                ratings.append(float(r))

        unique_genres = len(genre_counter)
        top_genre, top_genre_count = ("", 0)
        if genre_counter:
            top_genre, top_genre_count = genre_counter.most_common(1)[0]
        avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

        # ── 레벨 판정 (expert 우선 검사 — 다작 시청자가 단일 장르여도 expert로 분류)
        # 설계서 §9-5: expert(100편+) > beginner(20편 미만, 장르 3 이하) > intermediate
        if total_watched > 100 or top_genre_count >= 50:
            user_level = "expert"
            determination_reason = (
                f"총 시청 {total_watched}편, 최다 장르 '{top_genre}' {top_genre_count}편 "
                f"(기준: 100편 초과 또는 단일 장르 50편+ → 매니아)"
            )
        elif total_watched < 20 or unique_genres <= 3:
            user_level = "beginner"
            determination_reason = (
                f"총 시청 {total_watched}편, 고유 장르 {unique_genres}개 "
                f"(기준: 20편 미만 또는 장르 3개 이하 → 입문)"
            )
        else:
            user_level = "intermediate"
            determination_reason = (
                f"총 시청 {total_watched}편, 고유 장르 {unique_genres}개 "
                f"(기준: 20~100편, 장르 4개+ → 중급)"
            )

        level_detail = {
            "total_watched": total_watched,
            "unique_genres": unique_genres,
            "top_genre": top_genre,
            "top_genre_count": top_genre_count,
            "avg_rating": avg_rating,
            "determination_reason": determination_reason,
        }

        logger.info(
            "user_segment_analyzer_complete",
            user_id=user_id,
            user_level=user_level,
            total_watched=total_watched,
        )

        return {"user_level": user_level, "level_detail": level_detail}

    except Exception as e:
        logger.error("user_segment_analyzer_error", error=str(e))
        # fallback: 안전하게 beginner로 처리
        return {
            "user_level": "beginner",
            "level_detail": {
                "total_watched": 0,
                "unique_genres": 0,
                "top_genre": "",
                "top_genre_count": 0,
                "avg_rating": 0.0,
                "determination_reason": f"분석 오류로 기본값 적용: {e}",
            },
        }


# ============================================================
# Node 2: roadmap_generator
# ============================================================

@traceable(name="roadmap_generator", run_type="chain", metadata={"node": "2/4"})
async def roadmap_generator(state: RoadmapAgentState) -> dict:
    """
    MySQL에서 테마 키워드로 영화를 검색하고 3단계별로 5편씩 선정한다.

    처리 순서:
    1. 시청 이력에서 이미 본 영화 ID 집합 구성
    2. MySQL에서 theme 키워드로 title/overview/genres/keywords LIKE 검색
    3. TMDB popularity 기준으로 3단계 분류
       - beginner    : 상위 70% (대중적)
       - intermediate: 30~70%
       - expert      : 하위 30% (마니아)
    4. 각 단계에서 5편 선정 (부족 시 인접 단계에서 보충)
    5. 결과 없으면 인기 영화로 fallback

    반환 state 키:
    - course_movies: {"beginner": [...], "intermediate": [...], "expert": [...]}

    Args:
        state: RoadmapAgentState

    Returns:
        {"course_movies": dict}
    """
    try:
        theme = state.get("theme", "")
        watch_history: list[dict] = state.get("watch_history", [])
        user_id = state.get("user_id", "unknown")

        logger.info(
            "roadmap_generator_start",
            user_id=user_id,
            theme=theme,
        )

        # ── Step 1: 이미 본 영화 ID 집합 ──
        watched_ids: set[str] = {
            str(rec.get("movie_id", "")) for rec in watch_history if rec.get("movie_id")
        }

        # ── Step 2: MySQL 테마 검색 ──
        candidate_movies: list[dict] = []
        try:
            mysql = await get_mysql()
            async with mysql.acquire() as conn:
                async with conn.cursor() as cur:
                    # title, overview, genres, keywords LIKE 검색
                    # popularity 내림차순으로 충분히 가져옴
                    like_pattern = f"%{theme}%"
                    # 2026-04-15 컬럼명 수정:
                    # MySQL `movies` 테이블의 실제 컬럼은 `rating`/`popularity_score`.
                    # 이전까지 `vote_average`/`popularity` 를 SELECT 해 Unknown column
                    # 에러로 try/except 에 빠지면서 로드맵 에이전트가 사실상 동작 불능이었다.
                    await cur.execute(
                        """
                        SELECT
                            movie_id,
                            title,
                            original_title,
                            genres,
                            poster_path,
                            rating,
                            popularity_score,
                            release_date,
                            overview,
                            keywords
                        FROM movies
                        WHERE (
                            title LIKE %s
                            OR original_title LIKE %s
                            OR overview LIKE %s
                            OR genres LIKE %s
                            OR keywords LIKE %s
                        )
                        ORDER BY popularity_score DESC
                        LIMIT 200
                        """,
                        (
                            like_pattern,
                            like_pattern,
                            like_pattern,
                            like_pattern,
                            like_pattern,
                        ),
                    )
                    rows = await cur.fetchall()

            for row in rows:
                (
                    movie_id, title, original_title, genres_str,
                    poster_path, rating_val, popularity_val,
                    release_date, overview, keywords_str,
                ) = row

                movie_id_str = str(movie_id)
                # 이미 시청한 영화 제외
                if movie_id_str in watched_ids:
                    continue

                # 장르 파싱 (JSON 배열 문자열 또는 쉼표 구분)
                genres: list[str] = []
                if genres_str:
                    try:
                        genres = json.loads(genres_str)
                    except (json.JSONDecodeError, TypeError):
                        genres = [g.strip() for g in str(genres_str).split(",") if g.strip()]

                # poster_path → URL 변환
                poster_url = ""
                if poster_path:
                    poster_url = (
                        f"https://image.tmdb.org/t/p/w500{poster_path}"
                        if str(poster_path).startswith("/")
                        else str(poster_path)
                    )

                # 개봉연도 추출
                release_year = ""
                if release_date:
                    try:
                        release_year = str(release_date)[:4]
                    except Exception:
                        release_year = ""

                candidate_movies.append({
                    "id": movie_id_str,
                    "title": title or original_title or "",
                    "genres": genres,
                    "poster_url": poster_url,
                    "rating": float(rating_val or 0.0),
                    "popularity": float(popularity_val or 0.0),
                    "release_year": release_year,
                })

        except Exception as db_err:
            logger.error("roadmap_generator_db_error", error=str(db_err))

        # ── Step 2-fallback: 검색 결과 없으면 인기 영화 ──
        if not candidate_movies:
            logger.warning(
                "roadmap_theme_no_results_fallback_to_popular",
                theme=theme,
            )
            try:
                mysql = await get_mysql()
                async with mysql.acquire() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            """
                            SELECT
                                movie_id, title, original_title, genres,
                                poster_path, vote_average, popularity, release_date
                            FROM movies
                            ORDER BY popularity DESC
                            LIMIT 100
                            """,
                        )
                        rows = await cur.fetchall()

                for row in rows:
                    (
                        movie_id, title, original_title, genres_str,
                        poster_path, vote_average, popularity, release_date,
                    ) = row
                    movie_id_str = str(movie_id)
                    if movie_id_str in watched_ids:
                        continue

                    genres = []
                    if genres_str:
                        try:
                            genres = json.loads(genres_str)
                        except (json.JSONDecodeError, TypeError):
                            genres = [g.strip() for g in str(genres_str).split(",") if g.strip()]

                    poster_url = ""
                    if poster_path:
                        poster_url = (
                            f"https://image.tmdb.org/t/p/w500{poster_path}"
                            if str(poster_path).startswith("/")
                            else str(poster_path)
                        )

                    release_year = str(release_date)[:4] if release_date else ""
                    candidate_movies.append({
                        "id": movie_id_str,
                        "title": title or original_title or "",
                        "genres": genres,
                        "poster_url": poster_url,
                        "rating": float(vote_average or 0.0),
                        "popularity": float(popularity or 0.0),
                        "release_year": release_year,
                    })
            except Exception as fallback_err:
                logger.error("roadmap_generator_fallback_error", error=str(fallback_err))

        # ── Step 3: popularity 기준 3단계 분류 ──
        # popularity 내림차순 정렬 (이미 DB ORDER BY로 정렬됨, 안전을 위해 재정렬)
        candidate_movies.sort(key=lambda m: m["popularity"], reverse=True)
        total = len(candidate_movies)

        beginner_cutoff     = int(total * _BEGINNER_PERCENTILE)
        intermediate_cutoff = int(total * _INTERMEDIATE_LOW)

        beginner_pool     = candidate_movies[:beginner_cutoff]
        intermediate_pool = candidate_movies[intermediate_cutoff:beginner_cutoff]
        expert_pool       = candidate_movies[beginner_cutoff:]

        def _pick(pool: list[dict], n: int, supplement: list[dict]) -> list[dict]:
            """
            pool에서 n편을 선정하고, 부족하면 supplement에서 보충한다.

            Args:
                pool      : 1차 선정 풀
                n         : 선정 편수
                supplement: 부족 시 보충 풀

            Returns:
                n편 이하의 영화 dict 목록
            """
            picked = pool[:n]
            if len(picked) < n:
                already = {m["id"] for m in picked}
                for m in supplement:
                    if m["id"] not in already:
                        picked.append(m)
                    if len(picked) >= n:
                        break
            return picked

        beginner_movies     = _pick(beginner_pool,     _MOVIES_PER_STAGE, intermediate_pool)
        intermediate_movies = _pick(intermediate_pool, _MOVIES_PER_STAGE, beginner_pool + expert_pool)
        expert_movies       = _pick(expert_pool,       _MOVIES_PER_STAGE, intermediate_pool)

        course_movies = {
            "beginner":     beginner_movies,
            "intermediate": intermediate_movies,
            "expert":       expert_movies,
        }

        total_selected = sum(len(v) for v in course_movies.values())
        logger.info(
            "roadmap_generator_complete",
            user_id=user_id,
            theme=theme,
            candidate_count=total,
            selected_count=total_selected,
        )

        return {"course_movies": course_movies}

    except Exception as e:
        logger.error("roadmap_generator_fatal_error", error=str(e))
        return {"course_movies": {"beginner": [], "intermediate": [], "expert": []}}


# ============================================================
# Node 3: quiz_generator
# ============================================================

# 퀴즈 생성 시스템 프롬프트 (§9-5 Node3)
_QUIZ_SYSTEM_PROMPT = """당신은 영화 교육 퀴즈 전문가입니다.
아래 영화 목록에 대해 각 영화별로 퀴즈 2문항을 생성하세요.

반환 형식 (JSON 배열만 출력, 설명 없음):
[
  {
    "movie_id": "영화ID",
    "questions": [
      {
        "type": "multiple_choice",
        "question": "질문 텍스트",
        "options": ["선택지1", "선택지2", "선택지3", "선택지4"],
        "answer": "정답 선택지",
        "hint": "힌트 텍스트"
      },
      {
        "type": "short_answer",
        "question": "질문 텍스트",
        "options": [],
        "answer": "정답",
        "hint": "힌트 텍스트"
      }
    ]
  }
]

규칙:
- 스포일러 절대 금지 (결말, 반전, 죽음 등 핵심 내용 언급 불가)
- 각 영화마다 객관식 1문항 + 주관식 1문항 (총 2문항)
- 객관식 선택지는 4개, 오답은 그럴듯하게 구성
- 주관식은 짧게 답할 수 있는 사실형 질문 (감독, 장르, 개봉연도 등)
- 순수 JSON 배열만 출력 (마크다운 코드블록 없음)"""


@traceable(name="quiz_generator", run_type="chain", metadata={"node": "3/4"})
async def quiz_generator(state: RoadmapAgentState) -> dict:
    """
    로드맵의 15편 영화에 대해 퀴즈를 생성한다.

    처리 순서:
    1. course_movies에서 모든 영화 수집 (beginner 5 + intermediate 5 + expert 5 = 최대 15편)
    2. LLM(get_conversation_llm)에 영화 목록 전달 → JSON 퀴즈 배열 수신
    3. JSON 파싱 실패 또는 특정 영화 누락 시 fallback 템플릿 퀴즈 사용
    4. quizzes 리스트 반환

    Args:
        state: RoadmapAgentState

    Returns:
        {"quizzes": list[dict]}
    """
    try:
        course_movies: dict = state.get("course_movies", {})
        user_id = state.get("user_id", "unknown")

        # 전체 영화 목록 수집 (단계별 순서 유지)
        all_movies: list[dict] = []
        for stage_key in ("beginner", "intermediate", "expert"):
            all_movies.extend(course_movies.get(stage_key, []))

        if not all_movies:
            logger.warning("quiz_generator_no_movies", user_id=user_id)
            return {"quizzes": []}

        logger.info(
            "quiz_generator_start",
            user_id=user_id,
            movie_count=len(all_movies),
        )

        # ── LLM 퀴즈 생성 ──
        # 영화 목록을 간결하게 요약하여 LLM에 전달
        movies_summary = "\n".join(
            f"- movie_id: {m['id']}, 제목: {m['title']}, "
            f"장르: {', '.join(m.get('genres', []))}, "
            f"개봉연도: {m.get('release_year', '미상')}"
            for m in all_movies
        )

        llm_quizzes: list[dict] = []
        try:
            llm = get_conversation_llm()
            messages = [
                SystemMessage(content=_QUIZ_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"아래 {len(all_movies)}편의 영화에 대해 퀴즈를 생성하세요.\n\n"
                        f"{movies_summary}"
                    )
                ),
            ]
            response = await guarded_ainvoke(llm, messages)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            parsed = _parse_json_safe(response_text, context="quiz_generator")
            if isinstance(parsed, list):
                llm_quizzes = parsed
            elif isinstance(parsed, dict) and "quizzes" in parsed:
                llm_quizzes = parsed["quizzes"]

        except Exception as llm_err:
            logger.warning("quiz_generator_llm_failed", error=str(llm_err))

        # ── fallback: LLM 결과에 없는 영화 → 템플릿 퀴즈 ──
        # LLM이 생성한 quiz의 movie_id 집합
        llm_covered_ids: set[str] = {
            q.get("movie_id", "") for q in llm_quizzes if isinstance(q, dict)
        }

        final_quizzes: list[dict] = list(llm_quizzes)  # LLM 결과 유지
        for movie in all_movies:
            if movie["id"] not in llm_covered_ids:
                # 해당 영화 퀴즈 누락 → fallback 템플릿
                final_quizzes.append(_make_fallback_quiz(movie))
                logger.debug(
                    "quiz_fallback_used",
                    movie_id=movie["id"],
                    title=movie.get("title"),
                )

        logger.info(
            "quiz_generator_complete",
            user_id=user_id,
            total_quizzes=len(final_quizzes),
            llm_generated=len(llm_quizzes),
            fallback_used=len(final_quizzes) - len(llm_quizzes),
        )

        return {"quizzes": final_quizzes}

    except Exception as e:
        logger.error("quiz_generator_fatal_error", error=str(e))
        return {"quizzes": []}


# ============================================================
# Node 4: roadmap_formatter
# ============================================================

# 단계 소개글 생성 시스템 프롬프트 (§9-5 Node4)
_STAGE_DESC_SYSTEM = """당신은 영화 교육 로드맵 설계 전문가입니다.
아래 로드맵 단계 정보를 바탕으로 해당 단계의 짧은 소개글을 작성하세요.

반환 형식 (JSON만 출력, 설명 없음):
{
  "beginner_desc": "입문 단계 소개글 1~2문장",
  "intermediate_desc": "심화 단계 소개글 1~2문장",
  "expert_desc": "매니아 단계 소개글 1~2문장"
}

규칙:
- 각 단계 소개글은 1~2문장 (50자 이내 권장)
- 단계별 영화 분위기/특성을 자연스럽게 소개
- 스포일러 없이 기대감을 높이는 어조
- 순수 JSON만 출력"""

# 단계별 기본 소개글 fallback
_STAGE_DEFAULT_DESCS: dict[str, str] = {
    "beginner":     "영화 입문자를 위한 대중적이고 접근하기 쉬운 작품들로 구성했습니다.",
    "intermediate": "기본기를 갖춘 분들을 위한 깊이 있는 작품들을 선별했습니다.",
    "expert":       "진정한 영화 마니아를 위한 숨겨진 걸작들을 소개합니다.",
}

# 단계 한국어 이름 매핑
_STAGE_KR_NAMES: dict[str, str] = {
    "beginner":     "입문",
    "intermediate": "심화",
    "expert":       "매니아",
}


@traceable(name="roadmap_formatter", run_type="chain", metadata={"node": "4/4"})
async def roadmap_formatter(state: RoadmapAgentState) -> dict:
    """
    course_movies + quizzes를 결합하여 최종 FormattedRoadmap을 조립한다.

    처리 순서:
    1. LLM으로 단계별 소개글 생성 (실패 시 기본 텍스트 사용)
    2. course_movies의 각 영화에 해당 quiz를 연결
    3. UUID + 현재 타임스탬프로 roadmap_id, created_at 생성
    4. FormattedRoadmap 구조로 직렬화하여 formatted_roadmap dict 반환

    Args:
        state: RoadmapAgentState

    Returns:
        {"formatted_roadmap": dict}
    """
    try:
        course_movies: dict = state.get("course_movies", {})
        quizzes: list[dict] = state.get("quizzes", [])
        theme = state.get("theme", "")
        user_level = state.get("user_level", "beginner")
        user_id = state.get("user_id", "unknown")

        logger.info(
            "roadmap_formatter_start",
            user_id=user_id,
            theme=theme,
            user_level=user_level,
        )

        # ── Step 1: quiz를 movie_id로 빠르게 조회할 수 있도록 dict 변환 ──
        quiz_by_movie_id: dict[str, dict] = {
            q.get("movie_id", ""): q for q in quizzes if isinstance(q, dict)
        }

        # ── Step 2: LLM 단계별 소개글 생성 ──
        stage_descs: dict[str, str] = dict(_STAGE_DEFAULT_DESCS)  # 기본값으로 초기화
        try:
            llm = get_conversation_llm()

            # 단계별 대표 영화 제목 목록
            def _titles(key: str) -> str:
                movies = course_movies.get(key, [])
                return ", ".join(m.get("title", "") for m in movies[:3])

            desc_prompt = (
                f"테마: {theme}\n"
                f"입문 대표 영화: {_titles('beginner')}\n"
                f"심화 대표 영화: {_titles('intermediate')}\n"
                f"매니아 대표 영화: {_titles('expert')}\n\n"
                "각 단계의 소개글을 JSON으로 반환하세요."
            )
            messages = [
                SystemMessage(content=_STAGE_DESC_SYSTEM),
                HumanMessage(content=desc_prompt),
            ]
            response = await guarded_ainvoke(llm, messages)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            parsed = _parse_json_safe(response_text, context="stage_desc")
            if isinstance(parsed, dict):
                # LLM 생성 결과로 교체 (없는 키는 기본값 유지)
                for key_suffix, stage_key in [
                    ("beginner_desc",     "beginner"),
                    ("intermediate_desc", "intermediate"),
                    ("expert_desc",       "expert"),
                ]:
                    if parsed.get(key_suffix):
                        stage_descs[stage_key] = parsed[key_suffix]

        except Exception as desc_err:
            logger.warning("roadmap_formatter_desc_llm_failed", error=str(desc_err))

        # ── Step 3: 단계별 RoadmapStage 조립 ──
        stages: list[RoadmapStage] = []
        for stage_key in ("beginner", "intermediate", "expert"):
            movies_raw = course_movies.get(stage_key, [])

            roadmap_movies: list[RoadmapMovie] = []
            for m in movies_raw:
                movie_id = m.get("id", "")

                # 해당 영화의 퀴즈 연결
                quiz_dict = quiz_by_movie_id.get(movie_id)
                quiz_obj: Quiz | None = None
                if quiz_dict:
                    try:
                        quiz_obj = Quiz(
                            movie_id=movie_id,
                            questions=[
                                QuizQuestion(**q)
                                for q in quiz_dict.get("questions", [])
                            ],
                        )
                    except Exception as q_err:
                        logger.warning(
                            "quiz_model_parse_error",
                            movie_id=movie_id,
                            error=str(q_err),
                        )

                roadmap_movies.append(
                    RoadmapMovie(
                        id=movie_id,
                        title=m.get("title", ""),
                        genres=m.get("genres", []),
                        poster_url=m.get("poster_url", ""),
                        rating=float(m.get("rating", 0.0)),
                        hybrid_score=float(m.get("hybrid_score", 0.0)),
                        popularity_score=float(m.get("popularity", 0.0)),
                        quiz=quiz_obj,
                        completed=False,
                    )
                )

            stages.append(
                RoadmapStage(
                    name=_STAGE_KR_NAMES.get(stage_key, stage_key),
                    description=stage_descs.get(stage_key, ""),
                    movies=roadmap_movies,
                )
            )

        # ── Step 4: FormattedRoadmap 생성 ──
        roadmap = FormattedRoadmap(
            roadmap_id=str(uuid.uuid4()),
            theme=theme,
            user_level=user_level,
            created_at=datetime.now(timezone.utc).isoformat(),
            stages=stages,
            total_progress=0,
        )

        formatted_roadmap = roadmap.model_dump()

        logger.info(
            "roadmap_formatter_complete",
            user_id=user_id,
            roadmap_id=roadmap.roadmap_id,
            stage_count=len(stages),
            total_movies=sum(len(s.movies) for s in stages),
        )

        return {"formatted_roadmap": formatted_roadmap}

    except Exception as e:
        logger.error("roadmap_formatter_fatal_error", error=str(e))
        # 최소한의 유효한 fallback 반환
        return {
            "formatted_roadmap": {
                "roadmap_id": str(uuid.uuid4()),
                "theme": state.get("theme", ""),
                "user_level": state.get("user_level", "beginner"),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "stages": [],
                "total_progress": 0,
                "error": str(e),
            }
        }
