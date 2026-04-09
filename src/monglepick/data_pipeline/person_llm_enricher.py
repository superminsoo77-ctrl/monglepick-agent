"""
Person LLM 보강 모듈 (Phase ML §9.5 Phase 1 — C-2).

TMDB Person 데이터 (raw JSON) 를 Solar Pro 3 배치 호출로 보강하여
임베딩 텍스트 생성에 필요한 다음 항목을 추가한다:

    1. biography_ko:    영문 biography → 한국어 번역
    2. style_tags:      필모그래피 5편 → 공통 스타일 태그 5~10개
    3. persona:         페르소나 한 줄 요약 ("한국 작가주의 스릴러 거장")
    4. top_movies:      대표작 자동 선정 (popularity + vote_count)

mood_batch.py 와 동일한 RPMLimiter / generate_xxx_batch / enrich_xxx 패턴.

설계 진실 원본:
    docs/데이터_적재_프로세스_전체분석_및_개선계획.md §9.5 Phase 1 (C-2)

Person 임베딩 텍스트 (보강 후):
    [이름] {name} ({original_name})
    [국적] {place_of_birth}
    [직업] {known_for_department}
    [활동시기] {birthday[:4]} ~ {deathday[:4] or '현재'}
    [장르] {style_tags 중 장르 부분}
    [대표작] {top_movies 5편 한국어 제목}
    [페르소나] {persona}
    [전기] {biography_ko[:500]}

사용처:
    `scripts/run_tmdb_persons_enrich.py` 또는
    `scripts/run_persons_qdrant_load.py` 의 임베딩 직전 단계
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import structlog
from openai import AsyncOpenAI

from monglepick.data_pipeline.mood_batch import RPMLimiter

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 시스템 프롬프트
# ══════════════════════════════════════════════════════════════

#: Solar Pro 3 시스템 프롬프트 — 한 명씩 처리하는 단건 호출 (배치 X)
PERSON_SYSTEM_PROMPT = """당신은 영화 인물 분석 전문가입니다.
주어진 인물 정보(이름/필모그래피/영문 약력)를 한국어로 분석하여
JSON 객체로만 응답하세요.

[응답 JSON 형식 - 키 4개 필수]
{
  "biography_ko": "<한국어 약력 200~500자, 영문 biography 가 있으면 번역 + 요약, 없으면 필모그래피 기반 한 단락 작성>",
  "style_tags": ["<스타일 태그 5~10개>", ...],
  "persona": "<페르소나 한 줄 (15~40자) 예: '한국 작가주의 스릴러 거장'>",
  "top_movies": ["<대표작 한국어 제목 최대 5편>", ...]
}

[규칙]
1. JSON 객체로만 응답 (다른 텍스트 금지)
2. biography_ko: 한국어로 작성. 가족/사생활은 제외하고 작품 활동 중심
3. style_tags: 장르/연출 스타일/주제/시각적 특징 등 (예: ["사회비판", "블랙코미디", "느와르", "롱테이크"])
4. persona: 한 줄 요약 (15~40자, "~ 거장" / "~ 전문" / "~ 아이콘" 형태)
5. top_movies: 사용자가 들어본 대표작 위주로 5편. 한국 인물이면 한국 제목, 해외 인물이면 한국 개봉명 우선
"""


# ══════════════════════════════════════════════════════════════
# Person → LLM 입력 텍스트 변환
# ══════════════════════════════════════════════════════════════


def _build_person_user_prompt(person: dict) -> str:
    """
    TMDB Person 응답 dict → LLM 사용자 프롬프트.

    핵심 정보만 추려서 토큰을 절약한다.
    필모그래피는 movie_credits.cast + crew 에서 popularity 상위 10편만 사용.
    """
    name = person.get("name", "")
    original_name = person.get("also_known_as", [None])[0] if person.get("also_known_as") else ""
    biography = (person.get("biography") or "")[:1500]  # 1500자 제한
    birthday = person.get("birthday") or ""
    deathday = person.get("deathday") or ""
    place_of_birth = person.get("place_of_birth") or ""
    known_for = person.get("known_for_department", "")

    # 필모그래피 추출 (popularity 상위 10편)
    credits = person.get("movie_credits", {}) or {}
    cast = credits.get("cast", []) or []
    crew = credits.get("crew", []) or []
    all_credits = cast + crew

    # popularity 상위 10편 (제목 + 연도 + 직책)
    sorted_credits = sorted(
        all_credits,
        key=lambda c: c.get("popularity", 0) or 0,
        reverse=True,
    )[:10]

    filmography_lines = []
    for c in sorted_credits:
        title = c.get("title") or c.get("original_title") or ""
        date = c.get("release_date") or ""
        year = date[:4] if date else "?"
        # 직책: cast 면 character, crew 면 job
        role = c.get("character") or c.get("job") or ""
        filmography_lines.append(f"  - {title} ({year}) - {role}")

    filmography_block = "\n".join(filmography_lines) if filmography_lines else "  (정보 없음)"

    return f"""[이름] {name}
[원어이름] {original_name or name}
[출생] {birthday or '?'}{(' / 사망 ' + deathday) if deathday else ''}
[출생지] {place_of_birth or '?'}
[알려진 분야] {known_for or '?'}

[영문 약력]
{biography or '(약력 없음)'}

[필모그래피 (popularity 상위 10편)]
{filmography_block}

위 정보를 분석하여 명세된 JSON 객체로 응답하세요."""


# ══════════════════════════════════════════════════════════════
# 단건 LLM 호출 (Person 1명 → 보강 dict)
# ══════════════════════════════════════════════════════════════


async def generate_person_enrichment(
    client: AsyncOpenAI,
    person: dict,
    model: str,
    rpm_limiter: RPMLimiter,
    max_retries: int = 5,
) -> dict | None:
    """
    Person 1명에 대해 LLM 보강 결과를 생성한다.

    Returns:
        {"biography_ko": str, "style_tags": [...], "persona": str, "top_movies": [...]}
        또는 None (모든 시도 실패)
    """
    user_prompt = _build_person_user_prompt(person)

    for attempt in range(max_retries):
        try:
            await rpm_limiter.acquire()

            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": PERSON_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                    max_tokens=900,
                    response_format={"type": "json_object"},
                )
            except Exception as inner_e:
                msg = str(inner_e).lower()
                if "response_format" in msg or "json" in msg:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": PERSON_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.4,
                        max_tokens=900,
                    )
                else:
                    raise

            content = response.choices[0].message.content or "{}"

            # JSON 블록 추출
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
            if json_match:
                content = json_match.group()
            parsed = json.loads(content)

            # 필드 검증
            return {
                "biography_ko": str(parsed.get("biography_ko") or "")[:1500],
                "style_tags": [
                    str(t) for t in (parsed.get("style_tags") or [])[:10] if t
                ],
                "persona": str(parsed.get("persona") or "")[:80],
                "top_movies": [
                    str(m) for m in (parsed.get("top_movies") or [])[:5] if m
                ],
            }

        except json.JSONDecodeError as e:
            logger.warning(
                "person_llm_json_parse_error",
                attempt=attempt + 1,
                error=str(e),
                person_id=person.get("id"),
            )
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "too_many_requests" in error_str.lower()
            if is_rate_limit:
                wait = 30 * (attempt + 1)
                logger.warning(
                    "person_llm_rate_limit_backoff",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.warning(
                    "person_llm_api_error",
                    attempt=attempt + 1,
                    error=error_str[:200],
                    person_id=person.get("id"),
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))

    logger.warning(
        "person_llm_all_attempts_failed",
        person_id=person.get("id"),
    )
    return None


# ══════════════════════════════════════════════════════════════
# 배치 보강 진입점
# ══════════════════════════════════════════════════════════════


async def enrich_persons_with_solar_llm(
    persons: list[dict],
    api_key: str,
    model: str = "solar-pro3",
    rpm: int = 100,
    concurrency: int = 20,
) -> dict[str, int]:
    """
    Person 리스트의 enrichment 필드를 in-place 추가한다.

    각 person dict 에 다음 키를 추가:
        - llm_biography_ko
        - llm_style_tags
        - llm_persona
        - llm_top_movies

    Person 1명당 API 호출 1회 (필모그래피가 길어 배치 어려움).
    동시성은 Semaphore 로 제어, RPM 은 RPMLimiter 로 제어.

    Args:
        persons: TMDB Person 응답 dict 리스트
        api_key: Upstage API 키
        model: 모델명 (기본 solar-pro3)
        rpm: 분당 호출 한도 (기본 100)
        concurrency: 동시 호출 수 (기본 20)

    Returns:
        {"total": N, "enriched": N, "failed": N, "elapsed_s": float}
    """
    if not persons:
        return {"total": 0, "enriched": 0, "failed": 0, "elapsed_s": 0}

    start = time.time()

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1",
    )
    rpm_limiter = RPMLimiter(rpm)
    sem = asyncio.Semaphore(concurrency)

    enriched_count = 0
    failed_count = 0
    enriched_lock = asyncio.Lock()

    async def _process_person(person: dict) -> None:
        nonlocal enriched_count, failed_count
        async with sem:
            try:
                result = await generate_person_enrichment(
                    client=client,
                    person=person,
                    model=model,
                    rpm_limiter=rpm_limiter,
                )
            except Exception as e:
                logger.error(
                    "person_enrichment_unexpected_error",
                    person_id=person.get("id"),
                    error=str(e)[:200],
                )
                result = None

            if result:
                # in-place 업데이트 (LLM 결과를 별도 키로)
                person["llm_biography_ko"] = result["biography_ko"]
                person["llm_style_tags"] = result["style_tags"]
                person["llm_persona"] = result["persona"]
                person["llm_top_movies"] = result["top_movies"]
                async with enriched_lock:
                    enriched_count += 1
            else:
                async with enriched_lock:
                    failed_count += 1

    await asyncio.gather(*(_process_person(p) for p in persons))

    try:
        await client.close()
    except Exception:
        pass

    elapsed = time.time() - start
    logger.info(
        "person_llm_enriched",
        total=len(persons),
        enriched=enriched_count,
        failed=failed_count,
        elapsed_s=round(elapsed, 1),
        rpm=rpm,
        concurrency=concurrency,
    )

    return {
        "total": len(persons),
        "enriched": enriched_count,
        "failed": failed_count,
        "elapsed_s": round(elapsed, 1),
    }
