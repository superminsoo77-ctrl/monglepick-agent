"""
Upstage Solar Pro 3 기반 배치 무드태그 생성 모듈.

`scripts/run_mood_enrichment.py` 에서 공용 로직을 추출하여
`run_full_reload.py` 파이프라인에서도 재사용할 수 있게 한다.

사용 목적:
    Phase ML-4 재적재 시 `process_raw_movie()` 가 생성한 fallback 무드태그를
    Solar Pro 3 배치 호출로 덮어쓰고, 그 결과로 `build_embedding_text()` 를
    재실행하여 임베딩 벡터에 정밀 무드가 반영되도록 한다.

핵심 함수:
    - `enrich_documents_with_solar_mood()`: MovieDocument 리스트를 in-place 업데이트
    - `generate_mood_batch()`: 10건 배치 API 호출 (429 대응 지수백오프 포함)
    - `RPMLimiter`: 슬라이딩 윈도우 RPM 제한기

성능 (참고):
    - 배치 10건/호출, RPM 100, concurrency 20 기준
    - 청크 2,000건 → 200 API calls → 약 2분
    - 전체 1.18M건 → 약 20시간 (임베딩 단계와 병행 불가 기준)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Iterable

import structlog
from openai import AsyncOpenAI

from monglepick.data_pipeline.models import MovieDocument
from monglepick.data_pipeline.preprocessor import build_embedding_text

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수 (run_mood_enrichment.py 와 일치해야 함)
# ══════════════════════════════════════════════════════════════

#: 유효 무드태그 25개 화이트리스트 (§11-6 정의)
MOOD_WHITELIST: set[str] = {
    "몰입", "감동", "웅장", "긴장감", "힐링", "유쾌", "따뜻", "슬픔",
    "공포", "잔잔", "스릴", "카타르시스", "청춘", "우정", "가족애",
    "로맨틱", "미스터리", "반전", "철학적", "사회비판", "모험", "판타지",
    "레트로", "다크", "유머",
}

#: 장르 → 기본 무드 매핑 (LLM 실패 시 fallback)
GENRE_TO_MOOD: dict[str, list[str]] = {
    "액션": ["몰입", "스릴"], "모험": ["모험", "몰입"], "애니메이션": ["따뜻", "판타지"],
    "코미디": ["유쾌", "유머"], "범죄": ["긴장감", "다크"], "다큐멘터리": ["철학적", "사회비판"],
    "드라마": ["감동", "잔잔"], "가족": ["가족애", "따뜻"], "판타지": ["판타지", "모험"],
    "역사": ["웅장", "감동"], "공포": ["공포", "다크"], "음악": ["감동", "힐링"],
    "미스터리": ["미스터리", "긴장감"], "로맨스": ["로맨틱", "따뜻"], "SF": ["몰입", "웅장"],
    "TV 영화": ["잔잔"], "스릴러": ["스릴", "긴장감"], "전쟁": ["웅장", "카타르시스"],
    "서부": ["모험", "레트로"],
}

#: 시스템 프롬프트 — 무드 분석 전문가 역할 + 25개 태그 목록 + 규칙
SYSTEM_PROMPT = """당신은 영화 분위기 분석 전문가입니다.
주어진 영화들의 분위기를 분석하여 각각 무드 태그를 3~5개 선택해주세요.

[사용 가능한 무드 태그 - 반드시 이 목록에서만 선택]
몰입, 감동, 웅장, 긴장감, 힐링, 유쾌, 따뜻, 슬픔, 공포, 잔잔,
스릴, 카타르시스, 청춘, 우정, 가족애, 로맨틱, 미스터리, 반전,
철학적, 사회비판, 모험, 판타지, 레트로, 다크, 유머

[규칙]
1. 반드시 위 25개 태그에서만 선택하세요
2. 각 영화당 3~5개를 선택하세요
3. 장르, 키워드, 줄거리를 종합적으로 분석하세요
4. JSON 객체로만 응답하세요 (다른 텍스트 금지)"""


# ══════════════════════════════════════════════════════════════
# RPM 제한기 (슬라이딩 윈도우)
# ══════════════════════════════════════════════════════════════


class RPMLimiter:
    """
    분당 요청 수(RPM)를 제한하는 슬라이딩 윈도우 제한기.

    ``asyncio.Semaphore`` 와 함께 사용하여 동시 요청 수(concurrency)와 RPM 을
    동시에 제어한다. 60초 윈도우 내의 요청 타임스탬프를 추적하고, 한도 초과 시
    가장 오래된 요청이 만료될 때까지 대기한다.
    """

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.timestamps: list[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """요청 슬롯을 확보한다. RPM 초과 시 대기."""
        async with self.lock:
            now = time.time()
            # 60초 이전 타임스탬프 제거 (슬라이딩 윈도우)
            self.timestamps = [t for t in self.timestamps if now - t < 60]

            if len(self.timestamps) >= self.rpm:
                # 가장 오래된 요청이 60초 경과할 때까지 대기
                wait_until = self.timestamps[0] + 60
                wait_time = wait_until - now
                if wait_time > 0:
                    logger.debug("rpm_limit_wait", wait_seconds=round(wait_time, 1))
                    await asyncio.sleep(wait_time)
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < 60]

            self.timestamps.append(time.time())


# ══════════════════════════════════════════════════════════════
# 유틸: 프롬프트 구성 / fallback
# ══════════════════════════════════════════════════════════════


def build_user_prompt(movies: list[dict]) -> str:
    """
    배치 영화 정보를 사용자 프롬프트로 구성한다.

    각 영화를 1-indexed 번호와 함께 한 줄로 요약하여 LLM 입력 토큰을 절약한다.
    """
    lines = []
    for i, m in enumerate(movies, 1):
        title = m.get("title", "제목 없음")
        genres = ", ".join(m.get("genres", []))
        keywords = ", ".join(m.get("keywords", [])[:10])
        # 줄거리는 200자로 제한 (토큰 절약)
        overview = (m.get("overview") or "")[:200]
        lines.append(
            f"영화 {i}: {title} | 장르: {genres} | 키워드: {keywords} | 줄거리: {overview}"
        )

    movies_block = "\n".join(lines)
    return (
        f"{movies_block}\n\n"
        f"위 {len(movies)}개 영화의 무드 태그를 JSON 객체로 응답하세요.\n"
        f'키는 영화 번호(문자열), 값은 무드 태그 배열.\n'
        f'예: {{"1": ["몰입", "감동"], "2": ["유쾌", "따뜻"]}}'
    )


def _genre_fallback(genres: list[str]) -> list[str]:
    """장르 기반 기본 무드태그 (API 실패 시 fallback)."""
    moods: set[str] = set()
    for g in genres:
        moods.update(GENRE_TO_MOOD.get(g, []))
    return list(moods)[:5] if moods else ["잔잔"]


# ══════════════════════════════════════════════════════════════
# LLM 배치 호출 (1회 API → N건 무드 생성)
# ══════════════════════════════════════════════════════════════


async def generate_mood_batch(
    client: AsyncOpenAI,
    movies: list[dict],
    model: str,
    rpm_limiter: RPMLimiter,
    max_retries: int = 5,
) -> dict[str, list[str]]:
    """
    LLM 으로 배치 영화의 무드태그를 생성한다.

    N개 영화를 1회 API 호출로 처리하여 비용을 최소화한다.
    ``response_format=json_object`` 를 시도하고, 미지원 모델이면 텍스트에서
    정규식으로 JSON 블록을 추출한다. 429/네트워크 에러 발생 시 지수 백오프로
    재시도하며, 모든 시도 실패 시 장르 기반 fallback 을 반환한다.

    Args:
        client: AsyncOpenAI 클라이언트 (Upstage/OpenAI 호환)
        movies: [{"movie_id": str, "title": "...", "genres": [...], ...}, ...]
        model: 사용할 모델명 (예: "solar-pro3")
        rpm_limiter: 슬라이딩 윈도우 RPM 제한기
        max_retries: 최대 재시도 횟수 (기본 5)

    Returns:
        {movie_id: ["몰입", "감동", ...], ...}
    """
    user_prompt = build_user_prompt(movies)

    for attempt in range(max_retries):
        try:
            # RPM 제한 적용 (공유 인스턴스가 슬라이딩 윈도우로 관리)
            await rpm_limiter.acquire()

            # API 호출 — response_format=json_object 우선 시도
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
            except Exception as inner_e:
                # response_format 미지원만 fallback (429 등은 외부 except로 전파)
                msg = str(inner_e).lower()
                if "response_format" in msg or "json" in msg:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.3,
                        max_tokens=500,
                    )
                else:
                    raise

            content = response.choices[0].message.content or "{}"

            # JSON 추출: 텍스트에 JSON 이 포함된 경우 { } 블록만 추출
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
            if json_match:
                content = json_match.group()
            parsed = json.loads(content)

            # 결과 매핑: 영화 번호(1-indexed) → movie_id (문자열)
            result: dict[str, list[str]] = {}
            for i, m in enumerate(movies, 1):
                tags = parsed.get(str(i), [])
                # 화이트리스트 필터링 (유효하지 않은 태그 제거)
                valid_tags = [t for t in tags if t in MOOD_WHITELIST]
                if valid_tags:
                    result[m["movie_id"]] = valid_tags[:5]
                else:
                    # LLM 이 유효 태그를 반환하지 못한 경우 → fallback
                    result[m["movie_id"]] = _genre_fallback(m.get("genres", []))

            return result

        except json.JSONDecodeError as e:
            logger.warning("mood_json_parse_error", attempt=attempt + 1, error=str(e))
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "too_many_requests" in error_str.lower()
            if is_rate_limit:
                # 429 Rate Limit → 긴 대기 (30s, 60s, 90s, 120s, 150s)
                wait = 30 * (attempt + 1)
                logger.warning("mood_rate_limit_backoff", attempt=attempt + 1, wait_seconds=wait)
                await asyncio.sleep(wait)
            else:
                logger.warning("mood_api_error", attempt=attempt + 1, error=error_str)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))

    # 모든 시도 실패 → 장르 기반 fallback 으로 채움
    logger.warning(
        "mood_batch_all_attempts_failed",
        movie_ids=[m["movie_id"] for m in movies[:5]],
        batch_size=len(movies),
    )
    return {
        m["movie_id"]: _genre_fallback(m.get("genres", []))
        for m in movies
    }


# ══════════════════════════════════════════════════════════════
# MovieDocument in-place 무드 보강 (공용 진입점)
# ══════════════════════════════════════════════════════════════


async def enrich_documents_with_solar_mood(
    documents: list[MovieDocument],
    api_key: str,
    model: str = "solar-pro3",
    rpm: int = 100,
    concurrency: int = 20,
    batch_size: int = 10,
    rebuild_embedding_text: bool = True,
) -> dict[str, int]:
    """
    MovieDocument 리스트의 ``mood_tags`` 를 Solar Pro 3 배치 호출로 in-place 갱신한다.

    파이프라인:
        1. documents 를 ``batch_size`` 단위로 분할
        2. 각 배치를 ``concurrency`` 개 동시 실행 (Semaphore 로 제한)
        3. 공유 ``RPMLimiter`` 가 분당 호출 수를 제한
        4. 배치 결과의 유효 태그로 ``doc.mood_tags`` 덮어쓰기
        5. ``rebuild_embedding_text=True`` 이면 ``doc.embedding_text`` 재생성
           (Phase ML 의 다국어 + 정밀 무드가 벡터 단계에 반영되도록)

    네트워크 에러 / 429 / JSON 파싱 실패는 ``generate_mood_batch`` 내부에서
    재시도 + fallback 처리된다. 본 함수는 예외를 던지지 않고, 실패 배치는
    문서의 기존 mood_tags (전처리 단계의 fallback) 를 유지한다.

    Args:
        documents: in-place 로 업데이트될 MovieDocument 리스트
        api_key: Upstage API 키 (UPSTAGE_API_KEY)
        model: 사용할 모델명 (기본 ``solar-pro3``)
        rpm: 분당 최대 API 호출 수 (기본 100)
        concurrency: 동시 API 호출 수 (기본 20)
        batch_size: 1회 API 호출당 영화 수 (기본 10)
        rebuild_embedding_text: True 이면 무드 갱신 후 embedding_text 재계산

    Returns:
        {"total": 전체 문서 수, "enriched": 실제 업데이트된 문서 수,
         "batches": 처리 배치 수, "elapsed_s": 소요 초}
    """
    if not documents:
        return {"total": 0, "enriched": 0, "batches": 0, "elapsed_s": 0}

    start = time.time()

    # Upstage OpenAI 호환 클라이언트 (base_url 만 변경하면 동일 인터페이스)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1",
    )

    rpm_limiter = RPMLimiter(rpm)
    sem = asyncio.Semaphore(concurrency)

    # 배치로 분할 — 각 배치는 (시작 인덱스, MovieDocument 리스트)
    batches: list[tuple[int, list[MovieDocument]]] = []
    for start_idx in range(0, len(documents), batch_size):
        batches.append((start_idx, documents[start_idx:start_idx + batch_size]))

    enriched_count = 0
    enriched_count_lock = asyncio.Lock()

    async def _process_batch(start_idx: int, batch_docs: list[MovieDocument]) -> None:
        """단일 배치를 처리하고 mood_tags를 in-place 업데이트."""
        nonlocal enriched_count

        async with sem:
            # LLM 입력용 경량 dict 로 변환 (embedding_text 등 큰 필드 제외)
            movies_data = [
                {
                    "movie_id": doc.id,
                    "title": doc.title,
                    "genres": doc.genres,
                    "keywords": (doc.keywords or [])[:10],
                    "overview": (doc.overview or "")[:200],
                }
                for doc in batch_docs
            ]

            try:
                mood_map = await generate_mood_batch(
                    client=client,
                    movies=movies_data,
                    model=model,
                    rpm_limiter=rpm_limiter,
                )
            except Exception as e:
                # generate_mood_batch 내부에서 대부분 fallback 처리되지만
                # 예상치 못한 예외는 여기서 로깅만 하고 해당 배치 스킵
                logger.error(
                    "mood_batch_unexpected_error",
                    start_idx=start_idx,
                    batch_size=len(batch_docs),
                    error=str(e),
                )
                return

            # in-place 업데이트: movie_id 매칭으로 mood_tags 덮어쓰기
            batch_updated = 0
            for doc in batch_docs:
                new_moods = mood_map.get(doc.id)
                if new_moods:
                    doc.mood_tags = new_moods
                    batch_updated += 1

            async with enriched_count_lock:
                enriched_count += batch_updated

    # 모든 배치를 동시 실행 (Semaphore 가 concurrency 제한)
    await asyncio.gather(*(
        _process_batch(idx, docs) for idx, docs in batches
    ))

    # client close (AsyncOpenAI 는 내부 httpx 세션을 보유하므로 명시 종료)
    try:
        await client.close()
    except Exception:
        pass

    # embedding_text 재생성 (정밀 무드를 벡터에 반영하려면 필수)
    if rebuild_embedding_text:
        for doc in documents:
            doc.embedding_text = build_embedding_text(doc)

    elapsed = time.time() - start

    logger.info(
        "chunk_mood_enriched",
        total=len(documents),
        enriched=enriched_count,
        batches=len(batches),
        elapsed_s=round(elapsed, 1),
        rpm=rpm,
        concurrency=concurrency,
    )

    return {
        "total": len(documents),
        "enriched": enriched_count,
        "batches": len(batches),
        "elapsed_s": round(elapsed, 1),
    }
