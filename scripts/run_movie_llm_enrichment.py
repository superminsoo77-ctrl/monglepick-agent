"""
Movie LLM 보강 스크립트 (Phase ML §9.5 Phase 2 — D).

Task #5 (run_full_reload.py) 완료 후 실행. Qdrant `movies` 컬렉션의 영화에 대해
Solar Pro 3 로 다음 보강 필드를 생성하여 Qdrant payload 에 추가한다:

    - tagline_ko        영문 태그라인 → 한국어 번역
    - overview_ko       한글 줄거리가 비어있거나 짧은 영화에 대해 영문→한국어 번역
    - category_tags     입문작/마니아용/가족용/데이트용/스트레스해소 분류
    - one_line_summary  20자 한 줄 요약 (UI 카드용)
    - llm_keywords      줄거리 기반 핵심 키워드 5~10개

설계 진실 원본:
    docs/데이터_적재_프로세스_전체분석_및_개선계획.md §9.4 (LLM 보강 카테고리)
    §9.5 Phase 2

선결 조건:
    - Task #5 완료 (Qdrant movies 컬렉션에 정밀 mood 적용된 데이터 적재 완료)
    - .env UPSTAGE_API_KEY

처리 흐름:
    1. Qdrant `movies` 컬렉션 스트리밍 scroll (1,000건씩)
    2. 각 영화에 대해 Solar Pro 3 호출 (배치 5건/호출 — 영화는 mood보다 입력 큼)
    3. 응답 dict → Qdrant payload 부분 update (`set_payload`)
    4. 청크마다 체크포인트 저장 (재개 지원)

성능 추정 (1.18M 영화 기준):
    - 배치 5 / 호출 / 100 RPM → 1.18M / 5 / 100 = 2,360 분 ≈ 39시간
    - 단일 항목(tagline_ko)만 처리 시 응답 짧아 더 빠를 수 있음
    - 권장 분할: tagline_ko 먼저 → overview_ko → category_tags 순

사용법:
    # 전체 항목 한 번에 (39h~)
    PYTHONPATH=src uv run python scripts/run_movie_llm_enrichment.py --target all

    # tagline_ko 만 (가치 高, 시간 短)
    PYTHONPATH=src uv run python scripts/run_movie_llm_enrichment.py --target tagline

    # 재개
    PYTHONPATH=src uv run python scripts/run_movie_llm_enrichment.py --target tagline --resume

    # 처음 100건만 (테스트)
    PYTHONPATH=src uv run python scripts/run_movie_llm_enrichment.py --target all --limit 100

    # 상태 확인
    PYTHONPATH=src uv run python scripts/run_movie_llm_enrichment.py --status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# .env 로드
_env_file = _project_root / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import structlog  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from monglepick.config import settings  # noqa: E402
from monglepick.data_pipeline.mood_batch import RPMLimiter  # noqa: E402
from monglepick.db.clients import (  # noqa: E402
    init_all_clients,
    close_all_clients,
    get_qdrant,
)

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

CHECKPOINT_FILE = Path("data/movie_llm_enrich_checkpoint.json")
DEFAULT_CHUNK_SIZE = 1_000   # Qdrant scroll batch
DEFAULT_BATCH_SIZE = 5       # Solar API 1회 호출당 영화 수
DEFAULT_RPM = 100
DEFAULT_CONCURRENCY = 20

#: 보강 가능 항목
TARGETS = ("tagline", "overview", "category", "summary", "keyword", "all")

#: 카테고리 화이트리스트 (LLM 응답 검증)
CATEGORY_WHITELIST = {"입문작", "마니아용", "대중적", "예술적", "가족용", "데이트용", "스트레스해소"}


# ══════════════════════════════════════════════════════════════
# Solar Pro 3 시스템 프롬프트
# ══════════════════════════════════════════════════════════════

#: 보강 항목 5가지 모두 한 번에 요청 (배치 5건/호출)
ENRICHMENT_SYSTEM_PROMPT = """당신은 영화 메타데이터 분석 및 한국어 콘텐츠 작성 전문가입니다.
주어진 영화 N편에 대해 각각 다음 5가지 항목을 한국어로 생성하여 JSON 객체로 응답하세요.

[응답 JSON 형식 - 키는 영화 번호 문자열, 값은 항목 dict]
{
  "1": {
    "tagline_ko": "<영문 tagline 한국어 번역. tagline 없으면 빈 문자열>",
    "overview_ko": "<영문 overview 가 있고 한글 overview 가 없거나 짧으면 한국어 번역 (200~400자). 한글 충분하면 빈 문자열>",
    "category_tags": ["<입문작/마니아용/대중적/예술적/가족용/데이트용/스트레스해소 중 1~3개>"],
    "one_line_summary": "<20자 이내 한 줄 요약 (UI 카드용, 핵심 매력 한 문장)>",
    "llm_keywords": ["<줄거리 기반 핵심 한국어 키워드 5~10개>"]
  },
  "2": { ... }
}

[규칙]
1. JSON 객체로만 응답 (다른 텍스트 금지)
2. category_tags 는 화이트리스트만 사용 (입문작/마니아용/대중적/예술적/가족용/데이트용/스트레스해소)
3. one_line_summary 는 20자 이내 (영화 핵심 매력 한 문장)
4. tagline_ko / overview_ko 는 자연스러운 한국어로
5. llm_keywords 는 영화 줄거리/장르/주제 관련 한국어 단어/구"""


# ══════════════════════════════════════════════════════════════
# 체크포인트
# ══════════════════════════════════════════════════════════════


def _new_checkpoint(target: str) -> dict:
    return {
        "target": target,
        "phase": "",
        "total_processed": 0,
        "total_enriched": 0,
        "total_failed": 0,
        "last_qdrant_offset": None,    # Qdrant scroll offset (재개용)
        "processed_ids": [],            # 이미 처리한 movie_id (중복 방지)
        "start_time": datetime.now().isoformat(),
        "last_updated": "",
    }


def _load_checkpoint(target: str) -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
            if data.get("target") == target:
                data.setdefault("processed_ids", [])
                return data
        except Exception as e:
            logger.warning("checkpoint_load_failed", error=str(e))
    return _new_checkpoint(target)


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_FILE.write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════
# Qdrant 스트리밍 scroll
# ══════════════════════════════════════════════════════════════


async def _scroll_movies(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    skip_ids: set[str] | None = None,
    limit: int | None = None,
):
    """
    Qdrant `movies` 컬렉션을 청크 단위로 scroll 한다.

    Yields:
        list[dict]: [{point_id, payload}, ...]
    """
    skip_ids = skip_ids or set()
    client = await get_qdrant()
    offset = None
    yielded = 0

    while True:
        result = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=chunk_size,
            offset=offset,
            with_vectors=False,
            with_payload=True,
        )
        points = result[0]
        next_offset = result[1]

        if not points:
            break

        chunk = []
        for p in points:
            pid = str(p.id)
            payload = p.payload or {}
            mid = payload.get("id") or pid
            if str(mid) in skip_ids:
                continue
            chunk.append({"point_id": pid, "movie_id": str(mid), "payload": payload})

        if chunk:
            yielded += len(chunk)
            yield chunk

            if limit and yielded >= limit:
                return

        if next_offset is None:
            return
        offset = next_offset


# ══════════════════════════════════════════════════════════════
# Solar Pro 3 배치 호출
# ══════════════════════════════════════════════════════════════


def _build_user_prompt(movies: list[dict]) -> str:
    """배치 영화 정보 → 사용자 프롬프트 (각 영화당 입력 3~5줄)."""
    lines = []
    for i, m in enumerate(movies, 1):
        title = m.get("title", "제목 없음")
        title_en = m.get("title_en", "")
        genres = ", ".join(m.get("genres", []))
        tagline = m.get("tagline", "") or ""
        overview = (m.get("overview", "") or "")[:600]
        overview_en = (m.get("overview_en", "") or "")[:600]
        keywords = ", ".join((m.get("keywords") or [])[:8])

        block = f"""[영화 {i}]
제목: {title}
영문제목: {title_en}
장르: {genres}
태그라인: {tagline}
줄거리(한): {overview}
줄거리(영): {overview_en}
키워드: {keywords}"""
        lines.append(block)

    movies_block = "\n\n".join(lines)
    return f"""{movies_block}

위 {len(movies)}개 영화에 대해 명세된 JSON 객체로 응답하세요. 키는 영화 번호 문자열."""


async def _generate_enrichment_batch(
    client: AsyncOpenAI,
    movies: list[dict],
    model: str,
    rpm_limiter: RPMLimiter,
    target: str,
    max_retries: int = 5,
) -> dict[str, dict]:
    """
    배치 영화에 대해 LLM 보강 결과 생성.

    Returns:
        {movie_id: {tagline_ko, overview_ko, category_tags, one_line_summary, llm_keywords}}
    """
    user_prompt = _build_user_prompt(movies)

    for attempt in range(max_retries):
        try:
            await rpm_limiter.acquire()

            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                    max_tokens=1500,
                    response_format={"type": "json_object"},
                )
            except Exception as inner_e:
                msg = str(inner_e).lower()
                if "response_format" in msg or "json" in msg:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.4,
                        max_tokens=1500,
                    )
                else:
                    raise

            content = response.choices[0].message.content or "{}"

            # JSON 추출
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
            if json_match:
                content = json_match.group()
            parsed = json.loads(content)

            # 결과 매핑: 영화 번호 → movie_id
            result: dict[str, dict] = {}
            for i, m in enumerate(movies, 1):
                item = parsed.get(str(i), {})
                if not isinstance(item, dict):
                    continue

                # 항목 검증 + target 별 필터링
                cleaned = {}
                if target in ("tagline", "all"):
                    cleaned["tagline_ko"] = str(item.get("tagline_ko", "") or "")[:300]
                if target in ("overview", "all"):
                    cleaned["overview_ko"] = str(item.get("overview_ko", "") or "")[:1500]
                if target in ("category", "all"):
                    cats = item.get("category_tags") or []
                    valid_cats = [c for c in cats if c in CATEGORY_WHITELIST][:3]
                    cleaned["category_tags"] = valid_cats
                if target in ("summary", "all"):
                    cleaned["one_line_summary"] = str(item.get("one_line_summary", "") or "")[:50]
                if target in ("keyword", "all"):
                    kws = item.get("llm_keywords") or []
                    cleaned["llm_keywords"] = [str(k) for k in kws[:10] if k]

                if cleaned:
                    result[m["movie_id"]] = cleaned

            return result

        except json.JSONDecodeError as e:
            logger.warning("enrichment_json_parse_error", attempt=attempt + 1, error=str(e))
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "too_many_requests" in error_str.lower()
            if is_rate_limit:
                wait = 30 * (attempt + 1)
                logger.warning("enrichment_rate_limit_backoff", attempt=attempt + 1, wait=wait)
                await asyncio.sleep(wait)
            else:
                logger.warning(
                    "enrichment_api_error",
                    attempt=attempt + 1,
                    error=error_str[:200],
                    movie_count=len(movies),
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))

    logger.warning("enrichment_all_attempts_failed", movie_count=len(movies))
    return {}


# ══════════════════════════════════════════════════════════════
# Qdrant payload partial update
# ══════════════════════════════════════════════════════════════


async def _update_qdrant_payload(point_id: str, payload_delta: dict) -> bool:
    """
    Qdrant 포인트의 payload 일부 필드만 update (set_payload).

    Returns: 성공 여부
    """
    try:
        client = await get_qdrant()
        # point_id 가 string("123") 또는 uuid 일 수 있음
        try:
            pid: int | str = int(point_id)
        except ValueError:
            pid = point_id

        await client.set_payload(
            collection_name=settings.QDRANT_COLLECTION,
            payload=payload_delta,
            points=[pid],
        )
        return True
    except Exception as e:
        logger.warning("qdrant_set_payload_failed", point_id=point_id, error=str(e)[:200])
        return False


# ══════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════


async def run_movie_llm_enrichment(
    target: str = "tagline",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rpm: int = DEFAULT_RPM,
    concurrency: int = DEFAULT_CONCURRENCY,
    model: str = "solar-pro3",
    limit: int | None = None,
    resume: bool = False,
) -> None:
    """
    Qdrant `movies` 전체를 스트리밍하여 LLM 보강 적용.

    Args:
        target: 보강 항목 ("tagline", "overview", "category", "summary", "keyword", "all")
        chunk_size: Qdrant scroll 청크 (기본 1,000)
        batch_size: 1회 LLM 호출 영화 수 (기본 5)
        rpm: Solar API RPM (기본 100)
        concurrency: 동시 LLM 호출 (기본 20)
        model: Upstage 모델명
        limit: 처리 최대 건수 (테스트용)
        resume: 체크포인트 재개
    """
    pipeline_start = time.time()

    if target not in TARGETS:
        print(f"[ERROR] target 은 {TARGETS} 중 하나여야 합니다.")
        return

    if not settings.UPSTAGE_API_KEY:
        print("[ERROR] UPSTAGE_API_KEY 가 .env 에 설정되지 않았습니다.")
        return

    # 체크포인트
    checkpoint = _load_checkpoint(target) if resume else _new_checkpoint(target)
    skip_ids = set(checkpoint.get("processed_ids", [])) if resume else set()

    print(f"[Step 0] DB 클라이언트 초기화 (target={target}, resume={resume})")
    await init_all_clients()

    try:
        # 컬렉션 건수 확인
        client = await get_qdrant()
        info = await client.get_collection(settings.QDRANT_COLLECTION)
        total_points = info.points_count or 0
        print(f"  Qdrant {settings.QDRANT_COLLECTION}: {total_points:,} 포인트")
        if resume:
            print(f"  Skip ID: {len(skip_ids):,}")

        # LLM 클라이언트 + RPM
        llm_client = AsyncOpenAI(
            api_key=settings.UPSTAGE_API_KEY,
            base_url="https://api.upstage.ai/v1",
        )
        rpm_limiter = RPMLimiter(rpm)
        sem = asyncio.Semaphore(concurrency)

        print(f"\n[Step 1] LLM 보강 스트리밍 ({target}, chunk={chunk_size}, batch={batch_size}, rpm={rpm}, conc={concurrency})")
        print()

        checkpoint["phase"] = "enriching"
        chunk_idx = 0

        async for chunk in _scroll_movies(
            chunk_size=chunk_size,
            skip_ids=skip_ids,
            limit=limit,
        ):
            chunk_idx += 1
            chunk_start = time.time()

            # 청크 → 배치 분할
            batches: list[list[dict]] = []
            for i in range(0, len(chunk), batch_size):
                batches.append(chunk[i:i + batch_size])

            # 배치 동시 LLM 호출
            chunk_enriched = 0
            chunk_failed = 0

            async def _process_batch(batch: list[dict]) -> None:
                nonlocal chunk_enriched, chunk_failed
                async with sem:
                    try:
                        # batch 의 각 movie 를 dict (movie_id, title, ...) 로 LLM 입력 변환
                        movies_for_llm = [
                            {
                                "movie_id": m["movie_id"],
                                "title": m["payload"].get("title", ""),
                                "title_en": m["payload"].get("title_en", ""),
                                "genres": m["payload"].get("genres", []),
                                "tagline": m["payload"].get("tagline", ""),
                                "overview": m["payload"].get("overview", ""),
                                "overview_en": m["payload"].get("overview_en", ""),
                                "keywords": m["payload"].get("keywords", []),
                            }
                            for m in batch
                        ]

                        result_map = await _generate_enrichment_batch(
                            client=llm_client,
                            movies=movies_for_llm,
                            model=model,
                            rpm_limiter=rpm_limiter,
                            target=target,
                        )

                        # Qdrant payload update
                        for m in batch:
                            mid = m["movie_id"]
                            delta = result_map.get(mid)
                            if not delta:
                                chunk_failed += 1
                                continue
                            ok = await _update_qdrant_payload(m["point_id"], delta)
                            if ok:
                                chunk_enriched += 1
                                checkpoint["processed_ids"].append(mid)
                            else:
                                chunk_failed += 1
                    except Exception as e:
                        logger.error(
                            "batch_enrichment_unexpected_error",
                            error=str(e)[:200],
                            batch_size=len(batch),
                        )
                        chunk_failed += len(batch)

            await asyncio.gather(*(_process_batch(b) for b in batches))

            checkpoint["total_processed"] += len(chunk)
            checkpoint["total_enriched"] += chunk_enriched
            checkpoint["total_failed"] += chunk_failed
            _save_checkpoint(checkpoint)

            chunk_elapsed = time.time() - chunk_start
            total_elapsed = time.time() - pipeline_start
            rate = checkpoint["total_enriched"] / total_elapsed if total_elapsed > 0 else 0

            print(
                f"  [Chunk {chunk_idx:>5}] "
                f"+{chunk_enriched:>4} (fail {chunk_failed:>3}) | "
                f"누적 enriched {checkpoint['total_enriched']:>10,} | "
                f"속도 {rate:>5.1f}/s | "
                f"청크 {chunk_elapsed:>5.1f}s"
            )

        # 완료
        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        try:
            await llm_client.close()
        except Exception:
            pass

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[Movie LLM 보강 완료] target={target}")
        print(f"  처리:    {checkpoint['total_processed']:>10,}")
        print(f"  enriched: {checkpoint['total_enriched']:>10,}")
        print(f"  실패:    {checkpoint['total_failed']:>10,}")
        print(f"  소요:    {total_elapsed / 60:>10.1f} 분")
        print(f"{'=' * 60}")

    finally:
        await close_all_clients()


# ══════════════════════════════════════════════════════════════
# 상태 조회
# ══════════════════════════════════════════════════════════════


async def show_status() -> None:
    if not CHECKPOINT_FILE.exists():
        print("체크포인트 파일이 없습니다.")
        return
    cp = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    print("=" * 60)
    print(f"  Movie LLM Enrichment 체크포인트")
    print("=" * 60)
    print(f"  target:         {cp.get('target')}")
    print(f"  phase:          {cp.get('phase')}")
    print(f"  처리:           {cp.get('total_processed', 0):>10,}")
    print(f"  enriched:       {cp.get('total_enriched', 0):>10,}")
    print(f"  failed:         {cp.get('total_failed', 0):>10,}")
    print(f"  processed_ids:  {len(cp.get('processed_ids', [])):>10,}")
    print(f"  마지막 갱신:    {cp.get('last_updated', '-')}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Movie LLM 보강 (Solar Pro 3 → Qdrant payload update)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # tagline_ko 만 (가치 高, 시간 短)
  PYTHONPATH=src uv run python scripts/run_movie_llm_enrichment.py --target tagline

  # 전체 5항목
  PYTHONPATH=src uv run python scripts/run_movie_llm_enrichment.py --target all

  # 재개
  PYTHONPATH=src uv run python scripts/run_movie_llm_enrichment.py --target tagline --resume

  # 처음 100건만 (테스트)
  PYTHONPATH=src uv run python scripts/run_movie_llm_enrichment.py --target all --limit 100

  # 상태
  PYTHONPATH=src uv run python scripts/run_movie_llm_enrichment.py --status
        """,
    )
    parser.add_argument(
        "--target", type=str, default="tagline", choices=TARGETS,
        help=f"보강 항목 ({', '.join(TARGETS)})",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Qdrant scroll 청크 (기본 {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"1회 LLM 호출 영화 수 (기본 {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--rpm", type=int, default=DEFAULT_RPM,
        help=f"Solar API RPM (기본 {DEFAULT_RPM})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"동시 호출 (기본 {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--model", type=str, default="solar-pro3",
        help="Upstage 모델명 (기본 solar-pro3)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="처리 최대 건수 (테스트용)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="체크포인트 재개",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="현재 체크포인트 상태만 출력",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.status:
        asyncio.run(show_status())
    else:
        asyncio.run(
            run_movie_llm_enrichment(
                target=args.target,
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
                rpm=args.rpm,
                concurrency=args.concurrency,
                model=args.model,
                limit=args.limit,
                resume=args.resume,
            )
        )
