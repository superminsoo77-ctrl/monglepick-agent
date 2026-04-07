"""
커뮤니티 언급 분석 모듈 (§8-2 기능2).

처리 흐름:
1. MySQL에서 전체 영화 제목 목록 로드 → in-memory dict (title → movie_id) 캐싱
2. 게시글 본문에서 영화 제목 문자열 매칭 (str.find 기반, ahocorasick 불필요)
3. 매칭이 없는 게시글에 한해 LLM 보조 추출 (get_conversation_llm)
4. 언급 횟수 집계 → 이전 기간 대비 growth_rate 계산 (이전 기간 없으면 0.0)
5. trending_score = mention_count × (1 + growth_rate) 기준 상위 10편 선정

에러 시 빈 결과를 반환하고 에러를 전파하지 않는다.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from monglepick.agents.content_analysis.models import (
    CommunityAnalysisInput,
    CommunityAnalysisOutput,
    MovieMention,
    TrendingMovie,
)
from monglepick.db.clients import get_mysql
from monglepick.llm import get_conversation_llm, guarded_ainvoke

logger = structlog.get_logger()

# ── 영화 제목 캐시 (메모리 내, 프로세스 수명 동안 유지) ──
# {title: movie_id} 형태로 저장. 재시작 시 자동 갱신.
_title_cache: dict[str, str] = {}
_title_cache_loaded: bool = False

# LLM 보조 추출 시 한 번에 처리할 게시글 최대 수
_LLM_BATCH_SIZE = 20

# 트렌딩 상위 반환 수
_TOP_TRENDING = 10

# LLM 보조 추출 시스템 프롬프트
_LLM_EXTRACT_SYSTEM = """당신은 게시글에서 언급된 영화 제목을 추출하는 전문가입니다.
아래 게시글들에서 언급된 한국어 또는 외래어 영화 제목을 찾아 JSON으로 반환하세요.

반환 형식:
{
  "mentions": [
    {"post_id": "게시글ID", "titles": ["영화제목1", "영화제목2"]}
  ]
}

규칙:
- 명확한 영화 제목만 포함 (일반 명사 제외)
- 언급이 없으면 titles를 빈 배열로 반환
- 순수 JSON만 출력 (마크다운 코드블록 없음)"""


async def _load_title_cache() -> dict[str, str]:
    """
    MySQL movies 테이블에서 전체 영화 제목 목록을 로드하여 캐싱한다.

    {제목: movie_id} 형태로 저장한다.
    이미 로드된 경우 캐시를 그대로 반환한다 (프로세스 수명 동안 1회 로드).

    Returns:
        {title: movie_id} dict
    """
    global _title_cache, _title_cache_loaded

    if _title_cache_loaded:
        return _title_cache

    try:
        mysql = await get_mysql()
        async with mysql.acquire() as conn:
            async with conn.cursor() as cur:
                # 제목(한국어) + 원제(영문) 모두 수집
                await cur.execute(
                    """
                    SELECT movie_id, title, original_title
                    FROM movies
                    WHERE title IS NOT NULL
                    LIMIT 50000
                    """
                )
                rows = await cur.fetchall()

        new_cache: dict[str, str] = {}
        for row in rows:
            movie_id, title, original_title = row[0], row[1], row[2]
            if title:
                new_cache[title.strip()] = str(movie_id)
            if original_title and original_title != title:
                new_cache[original_title.strip()] = str(movie_id)

        _title_cache = new_cache
        _title_cache_loaded = True
        logger.info("title_cache_loaded", count=len(_title_cache))

    except Exception as e:
        logger.error("title_cache_load_failed", error=str(e))
        # 캐시 로드 실패 시 빈 dict 반환, 다음 호출에서 재시도
        _title_cache_loaded = False

    return _title_cache


def _match_titles_in_text(text: str, title_dict: dict[str, str]) -> list[str]:
    """
    게시글 본문에서 영화 제목을 문자열 포함 검사(str in)로 매칭한다.

    ahocorasick 없이 단순 in 연산자를 사용한다.
    짧은 제목(2자 이하)은 오탐을 줄이기 위해 건너뛴다.

    Args:
        text       : 게시글 본문
        title_dict : {title: movie_id} 캐시 dict

    Returns:
        매칭된 movie_id 목록 (중복 제거)
    """
    matched_ids: list[str] = []
    seen: set[str] = set()

    for title, movie_id in title_dict.items():
        # 2자 이하 제목은 오탐 위험 → 건너뜀
        if len(title) <= 2:
            continue
        if title in text and movie_id not in seen:
            matched_ids.append(movie_id)
            seen.add(movie_id)

    return matched_ids


async def _llm_extract_mentions(
    posts: list[tuple[str, str]],  # [(post_id, content), ...]
    title_dict: dict[str, str],
) -> dict[str, list[str]]:
    """
    LLM으로 게시글에서 영화 제목을 추출하고, 제목을 movie_id로 변환한다.

    문자열 매칭으로 찾지 못한 게시글에만 사용한다.
    LLM 실패 시 빈 dict 반환 (에러 전파 금지).

    Args:
        posts      : [(post_id, content)] 목록
        title_dict : {title: movie_id} 캐시 dict

    Returns:
        {post_id: [movie_id, ...]} dict
    """
    if not posts:
        return {}

    result: dict[str, list[str]] = {}

    try:
        llm = get_conversation_llm()

        # 배치 단위로 처리 (LLM 부하 제한)
        for i in range(0, len(posts), _LLM_BATCH_SIZE):
            batch = posts[i : i + _LLM_BATCH_SIZE]

            # 프롬프트 구성
            posts_text = "\n\n".join(
                f"[post_id: {pid}]\n{content[:500]}"  # 본문 500자까지만
                for pid, content in batch
            )
            human_msg = HumanMessage(
                content=f"아래 게시글들에서 언급된 영화 제목을 추출하세요.\n\n{posts_text}"
            )
            messages = [SystemMessage(content=_LLM_EXTRACT_SYSTEM), human_msg]

            response = await guarded_ainvoke(llm, messages)
            text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # JSON 파싱
            cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("```").strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                parsed = json.loads(match.group()) if match else {}

            # 제목 → movie_id 변환
            for item in parsed.get("mentions", []):
                post_id = item.get("post_id", "")
                titles = item.get("titles", [])
                movie_ids: list[str] = []
                for title in titles:
                    # 정확 매칭 우선, 없으면 부분 매칭
                    if title in title_dict:
                        movie_ids.append(title_dict[title])
                    else:
                        for cached_title, mid in title_dict.items():
                            if title in cached_title or cached_title in title:
                                movie_ids.append(mid)
                                break
                if movie_ids:
                    result[post_id] = list(dict.fromkeys(movie_ids))  # 중복 제거

    except Exception as e:
        logger.warning("llm_mention_extraction_failed", error=str(e))

    return result


async def analyze_community_mentions(
    inp: CommunityAnalysisInput,
) -> CommunityAnalysisOutput:
    """
    커뮤니티 게시글에서 영화 언급을 분석하고 트렌딩 영화를 산출한다.

    처리 순서:
    1. 영화 제목 캐시 로드 (MySQL, 최초 1회)
    2. 문자열 매칭으로 게시글별 언급 영화 추출
    3. 미매칭 게시글에 LLM 보조 추출 적용
    4. movie_id별 언급 횟수 집계
    5. trending_score = mention_count × (1 + growth_rate) 기준 상위 10편 반환
       (이전 기간 데이터 없으므로 growth_rate = 0.0 고정)

    Args:
        inp: CommunityAnalysisInput (posts, period)

    Returns:
        CommunityAnalysisOutput (mention_counts, trending_movies)
    """
    try:
        logger.info(
            "community_analysis_start",
            post_count=len(inp.posts),
            period=inp.period,
        )

        # ── Step 1: 영화 제목 캐시 로드 ──
        title_dict = await _load_title_cache()

        # movie_id → (title, post_id 목록) 집계 구조
        # {movie_id: {"title": str, "posts": [post_id, ...]}}
        mention_map: dict[str, dict] = defaultdict(
            lambda: {"title": "", "posts": []}
        )

        # 제목 역방향 조회: movie_id → 첫 번째 제목
        id_to_title: dict[str, str] = {}
        for title, mid in title_dict.items():
            if mid not in id_to_title:
                id_to_title[mid] = title

        # ── Step 2: 문자열 매칭 ──
        unmatched_posts: list[tuple[str, str]] = []  # LLM 보조가 필요한 게시글
        for post in inp.posts:
            matched_ids = _match_titles_in_text(post.content, title_dict)
            if matched_ids:
                for mid in matched_ids:
                    mention_map[mid]["title"] = id_to_title.get(mid, "")
                    if post.post_id not in mention_map[mid]["posts"]:
                        mention_map[mid]["posts"].append(post.post_id)
            else:
                unmatched_posts.append((post.post_id, post.content))

        logger.info(
            "community_string_match_done",
            matched_posts=len(inp.posts) - len(unmatched_posts),
            unmatched_posts=len(unmatched_posts),
            unique_movies=len(mention_map),
        )

        # ── Step 3: 미매칭 게시글 LLM 보조 추출 ──
        if unmatched_posts and title_dict:
            llm_results = await _llm_extract_mentions(unmatched_posts, title_dict)
            for post_id, movie_ids in llm_results.items():
                for mid in movie_ids:
                    mention_map[mid]["title"] = id_to_title.get(mid, "")
                    if post_id not in mention_map[mid]["posts"]:
                        mention_map[mid]["posts"].append(post_id)

        # ── Step 4: MovieMention 목록 생성 ──
        mention_counts: list[MovieMention] = [
            MovieMention(
                movie_id=mid,
                title=data["title"],
                count=len(data["posts"]),
                posts=data["posts"],
            )
            for mid, data in mention_map.items()
            if data["posts"]  # 언급 있는 영화만
        ]

        # ── Step 5: 트렌딩 점수 계산 + 상위 10편 선정 ──
        # 이전 기간 데이터가 없으므로 growth_rate = 0.0 고정
        trending_movies: list[TrendingMovie] = []
        for m in mention_counts:
            growth_rate = 0.0  # 이전 기간 비교 데이터 없음
            trending_score = m.count * (1.0 + growth_rate)
            trending_movies.append(
                TrendingMovie(
                    movie_id=m.movie_id,
                    title=m.title,
                    mention_count=m.count,
                    growth_rate=growth_rate,
                    trending_score=trending_score,
                )
            )

        # trending_score 내림차순 정렬 후 상위 10편
        trending_movies.sort(key=lambda t: t.trending_score, reverse=True)
        trending_movies = trending_movies[:_TOP_TRENDING]

        logger.info(
            "community_analysis_complete",
            total_mentions=len(mention_counts),
            trending_count=len(trending_movies),
        )

        return CommunityAnalysisOutput(
            mention_counts=mention_counts,
            trending_movies=trending_movies,
        )

    except Exception as e:
        logger.error("analyze_community_mentions_fatal_error", error=str(e))
        # 에러 전파 금지 → 빈 결과 반환
        return CommunityAnalysisOutput(mention_counts=[], trending_movies=[])
