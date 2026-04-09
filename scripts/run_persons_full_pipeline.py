"""
Person 통합 실행 파이프라인 (Phase ML §9.5 Phase 1 — C-5).

C-1 (수집) → C-2 (LLM 보강) → C-3 (Qdrant 적재) 를 한 번에 실행한다.

설계 진실 원본:
    docs/데이터_적재_프로세스_전체분석_및_개선계획.md §9.5 Phase 1

처리 흐름:
    1. data/tmdb_persons/tmdb_persons.jsonl 스트리밍 청크 로드 (1,000명/청크)
    2. 각 청크에 대해:
        a) 이미 Qdrant `persons` 컬렉션에 있는 person_id skip (중복 방지)
        b) Solar Pro 3 LLM 보강 (biography_ko/style_tags/persona/top_movies)
        c) Qdrant `persons` 컬렉션에 적재 (4096d Solar embedding)
    3. 청크마다 체크포인트 저장 (재개 지원)

선결 조건:
    - run_tmdb_persons_collect.py 완료 (data/tmdb_persons/tmdb_persons.jsonl 존재)
    - .env UPSTAGE_API_KEY
    - Qdrant 가동 (`persons` 컬렉션 자동 생성)

중복 제거 정책 (사용자 요구사항):
    - person_id 기준 1차 필터 (Qdrant scroll 로 기존 ID 셋 로드)
    - 체크포인트의 processed_ids 2차 필터 (재개 시 중복 방지)
    - Qdrant upsert 자체가 멱등 (PointStruct.id 충돌 시 덮어쓰기)
    → 3중 안전장치

사용법:
    # 전체 실행 (신규)
    PYTHONPATH=src uv run python scripts/run_persons_full_pipeline.py

    # 재개 (체크포인트 + 기존 Qdrant ID skip)
    PYTHONPATH=src uv run python scripts/run_persons_full_pipeline.py --resume

    # 처음 N명만 (테스트)
    PYTHONPATH=src uv run python scripts/run_persons_full_pipeline.py --limit 100

    # mood/embedding 옵션 조정
    PYTHONPATH=src uv run python scripts/run_persons_full_pipeline.py \\
        --chunk-size 500 --llm-rpm 100 --llm-concurrency 20

    # 상태 확인
    PYTHONPATH=src uv run python scripts/run_persons_full_pipeline.py --status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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

from monglepick.config import settings  # noqa: E402
from monglepick.data_pipeline.person_llm_enricher import enrich_persons_with_solar_llm  # noqa: E402
from monglepick.data_pipeline.person_qdrant_loader import (  # noqa: E402
    PERSONS_COLLECTION,
    ensure_persons_collection,
    load_persons_to_qdrant,
)
from monglepick.db.clients import init_all_clients, close_all_clients, get_qdrant  # noqa: E402

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

INPUT_JSONL = Path("data/tmdb_persons/tmdb_persons.jsonl")
CHECKPOINT_FILE = Path("data/tmdb_persons/full_pipeline_checkpoint.json")
DEFAULT_CHUNK_SIZE = 1_000     # JSONL 청크 (LLM 보강 단위)
DEFAULT_LLM_RPM = 100
DEFAULT_LLM_CONCURRENCY = 20
DEFAULT_EMBED_BATCH = 50
DEFAULT_QDRANT_BATCH = 100


# ══════════════════════════════════════════════════════════════
# 체크포인트
# ══════════════════════════════════════════════════════════════


def _new_checkpoint() -> dict:
    return {
        "phase": "",
        "total_jsonl_lines": 0,
        "total_skipped_existing": 0,   # Qdrant 에 이미 있어서 skip
        "total_llm_enriched": 0,        # LLM 보강 성공
        "total_llm_failed": 0,
        "total_qdrant_loaded": 0,       # Qdrant 적재 성공
        "last_jsonl_line": 0,
        "start_time": datetime.now().isoformat(),
        "last_updated": "",
    }


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("checkpoint_load_failed", error=str(e))
    return _new_checkpoint()


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════
# 기존 Qdrant `persons` ID 셋 로드 (1차 중복 제거)
# ══════════════════════════════════════════════════════════════


async def _load_existing_persons_ids() -> set[int]:
    """
    Qdrant `persons` 컬렉션의 모든 person_id (point_id) 셋을 반환한다.

    재개 시 또는 부분 적재 후에 이미 있는 person 을 다시 처리하지 않도록.
    """
    client = await get_qdrant()
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]
    if PERSONS_COLLECTION not in existing_names:
        return set()

    print(f"  기존 {PERSONS_COLLECTION} 컬렉션 ID scroll 시작...")
    ids: set[int] = set()
    offset = None

    while True:
        result = await client.scroll(
            collection_name=PERSONS_COLLECTION,
            limit=5000,
            offset=offset,
            with_vectors=False,
            with_payload=False,
        )
        points = result[0]
        next_offset = result[1]

        if not points:
            break

        for p in points:
            try:
                ids.add(int(p.id))
            except (ValueError, TypeError):
                pass

        if next_offset is None:
            break
        offset = next_offset

    return ids


# ══════════════════════════════════════════════════════════════
# JSONL 스트리밍
# ══════════════════════════════════════════════════════════════


def _jsonl_chunks(path: Path, chunk_size: int, skip_lines: int = 0):
    """
    JSONL 파일을 청크 단위로 yield. dict 객체 리스트.
    skip_lines 이후 라인부터 시작.
    """
    chunk: list[dict] = []
    line_no = 0

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line_no += 1
            if line_no <= skip_lines:
                continue

            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("id"):
                    chunk.append(obj)
            except json.JSONDecodeError:
                continue

            if len(chunk) >= chunk_size:
                yield chunk, line_no
                chunk = []

    if chunk:
        yield chunk, line_no


# ══════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════


async def run_persons_full_pipeline(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    llm_rpm: int = DEFAULT_LLM_RPM,
    llm_concurrency: int = DEFAULT_LLM_CONCURRENCY,
    embed_batch_size: int = DEFAULT_EMBED_BATCH,
    qdrant_batch_size: int = DEFAULT_QDRANT_BATCH,
    llm_model: str = "solar-pro3",
    limit: int | None = None,
    resume: bool = False,
) -> None:
    """
    JSONL → LLM 보강 → Qdrant `persons` 적재 통합.

    Args:
        chunk_size: JSONL 청크 크기 (= LLM 보강 단위)
        llm_rpm: Solar API RPM 한도
        llm_concurrency: Solar API 동시 호출
        embed_batch_size: Solar embedding 배치 (Qdrant loader 내부)
        qdrant_batch_size: Qdrant upsert 배치
        llm_model: Upstage 모델명
        limit: 처리 최대 person 수 (테스트용)
        resume: 체크포인트 + Qdrant 기존 ID skip
    """
    pipeline_start = time.time()

    if not INPUT_JSONL.exists():
        print(f"[ERROR] JSONL 파일이 없습니다: {INPUT_JSONL}")
        print(f"        먼저 run_tmdb_persons_collect.py 를 실행하세요.")
        return

    if not settings.UPSTAGE_API_KEY:
        print("[ERROR] UPSTAGE_API_KEY 가 .env 에 설정되지 않았습니다.")
        return

    # ── 체크포인트 로드 ──
    checkpoint = _load_checkpoint() if resume else _new_checkpoint()
    skip_lines = checkpoint.get("last_jsonl_line", 0) if resume else 0

    # ── DB 클라이언트 ──
    print(f"[Step 0] DB 클라이언트 초기화")
    await init_all_clients()

    try:
        # ── Qdrant 컬렉션 보장 ──
        print(f"\n[Step 1] Qdrant {PERSONS_COLLECTION} 컬렉션 보장")
        await ensure_persons_collection()

        # ── 기존 Qdrant ID 로드 (1차 중복 제거) ──
        print(f"\n[Step 2] 기존 Qdrant {PERSONS_COLLECTION} ID 로드")
        existing_ids = await _load_existing_persons_ids()
        print(f"  기존 ID: {len(existing_ids):,}")

        # ── 청크 단위 처리 ──
        print(f"\n[Step 3] JSONL → LLM → Qdrant 통합 처리")
        print(f"  chunk_size={chunk_size:,}, llm_rpm={llm_rpm}, concurrency={llm_concurrency}")
        if resume and skip_lines > 0:
            print(f"  resume: 라인 {skip_lines:,} 부터 시작")
        if limit:
            print(f"  limit: 최대 {limit:,} 명만 처리")
        print()

        checkpoint["phase"] = "processing"
        chunk_idx = 0
        processed_total = 0

        for chunk, current_line in _jsonl_chunks(INPUT_JSONL, chunk_size, skip_lines):
            chunk_idx += 1
            chunk_start = time.time()

            # ── 1차 중복 제거: 기존 Qdrant ID skip ──
            before_filter = len(chunk)
            filtered_chunk = [p for p in chunk if int(p.get("id", 0)) not in existing_ids]
            skipped_existing = before_filter - len(filtered_chunk)
            checkpoint["total_skipped_existing"] += skipped_existing

            if not filtered_chunk:
                checkpoint["last_jsonl_line"] = current_line
                _save_checkpoint(checkpoint)
                continue

            # ── LLM 보강 ──
            try:
                llm_stats = await enrich_persons_with_solar_llm(
                    persons=filtered_chunk,
                    api_key=settings.UPSTAGE_API_KEY,
                    model=llm_model,
                    rpm=llm_rpm,
                    concurrency=llm_concurrency,
                )
                checkpoint["total_llm_enriched"] += llm_stats["enriched"]
                checkpoint["total_llm_failed"] += llm_stats["failed"]
            except Exception as e:
                logger.error("chunk_llm_failed", chunk_idx=chunk_idx, error=str(e)[:200])
                checkpoint["total_llm_failed"] += len(filtered_chunk)
                checkpoint["last_jsonl_line"] = current_line
                _save_checkpoint(checkpoint)
                continue

            # ── Qdrant 적재 ──
            try:
                loaded = await load_persons_to_qdrant(
                    persons=filtered_chunk,
                    embed_batch_size=embed_batch_size,
                    upsert_batch_size=qdrant_batch_size,
                )
                checkpoint["total_qdrant_loaded"] += loaded

                # 적재 성공한 ID 를 existing_ids 에 추가 (다음 청크에서 중복 방지)
                for p in filtered_chunk:
                    pid = int(p.get("id", 0))
                    if pid:
                        existing_ids.add(pid)
            except Exception as e:
                logger.error("chunk_qdrant_failed", chunk_idx=chunk_idx, error=str(e)[:200])

            # 체크포인트 갱신
            checkpoint["last_jsonl_line"] = current_line
            checkpoint["total_jsonl_lines"] = current_line
            _save_checkpoint(checkpoint)

            chunk_elapsed = time.time() - chunk_start
            total_elapsed = time.time() - pipeline_start
            rate = checkpoint["total_qdrant_loaded"] / total_elapsed if total_elapsed > 0 else 0

            print(
                f"  [Chunk {chunk_idx:>4}] "
                f"input {before_filter:>5} | skip_exist {skipped_existing:>4} | "
                f"llm {llm_stats['enriched']:>4} (fail {llm_stats['failed']:>2}) | "
                f"qdrant +{loaded:>4} | "
                f"누적 loaded {checkpoint['total_qdrant_loaded']:>8,} | "
                f"속도 {rate:>5.1f}/s | "
                f"청크 {chunk_elapsed:>5.1f}s"
            )

            processed_total += len(filtered_chunk)
            if limit and processed_total >= limit:
                print(f"\n  --limit {limit} 도달 → 중단")
                break

        # ── 완료 ──
        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[Person 통합 파이프라인 완료]")
        print(f"  마지막 라인:        {checkpoint['last_jsonl_line']:>10,}")
        print(f"  기존 ID skip:       {checkpoint['total_skipped_existing']:>10,}")
        print(f"  LLM 보강 성공:      {checkpoint['total_llm_enriched']:>10,}")
        print(f"  LLM 실패:           {checkpoint['total_llm_failed']:>10,}")
        print(f"  Qdrant 적재 성공:   {checkpoint['total_qdrant_loaded']:>10,}")
        print(f"  소요:               {total_elapsed / 60:>10.1f} 분")
        print(f"{'=' * 60}")

    finally:
        await close_all_clients()


# ══════════════════════════════════════════════════════════════
# 상태 조회
# ══════════════════════════════════════════════════════════════


async def show_status() -> None:
    cp = _load_checkpoint()
    print("=" * 60)
    print(f"  Person 통합 파이프라인 체크포인트")
    print("=" * 60)
    print(f"  단계:           {cp.get('phase', '미시작')}")
    print(f"  마지막 라인:    {cp.get('last_jsonl_line', 0):>10,}")
    print(f"  기존 ID skip:   {cp.get('total_skipped_existing', 0):>10,}")
    print(f"  LLM 보강 성공:  {cp.get('total_llm_enriched', 0):>10,}")
    print(f"  LLM 실패:       {cp.get('total_llm_failed', 0):>10,}")
    print(f"  Qdrant 적재:    {cp.get('total_qdrant_loaded', 0):>10,}")
    print(f"  마지막 갱신:    {cp.get('last_updated', '-')}")
    print()

    # Qdrant 라이브 카운트
    try:
        await init_all_clients()
        client = await get_qdrant()
        collections = await client.get_collections()
        existing = [c.name for c in collections.collections]
        if PERSONS_COLLECTION in existing:
            info = await client.get_collection(PERSONS_COLLECTION)
            print(f"  Qdrant {PERSONS_COLLECTION}: {info.points_count:>10,} 포인트")
        else:
            print(f"  Qdrant {PERSONS_COLLECTION}: 컬렉션 없음")
    except Exception as e:
        print(f"  Qdrant 조회 실패: {e}")
    finally:
        try:
            await close_all_clients()
        except Exception:
            pass

    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Person 통합 실행 파이프라인 (JSONL → LLM → Qdrant persons)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 실행 (신규)
  PYTHONPATH=src uv run python scripts/run_persons_full_pipeline.py

  # 재개
  PYTHONPATH=src uv run python scripts/run_persons_full_pipeline.py --resume

  # 처음 100명만 (테스트)
  PYTHONPATH=src uv run python scripts/run_persons_full_pipeline.py --limit 100

  # 상태 확인
  PYTHONPATH=src uv run python scripts/run_persons_full_pipeline.py --status
        """,
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"JSONL 청크 크기 (기본 {DEFAULT_CHUNK_SIZE:,})",
    )
    parser.add_argument(
        "--llm-rpm", type=int, default=DEFAULT_LLM_RPM,
        help=f"Solar API RPM (기본 {DEFAULT_LLM_RPM})",
    )
    parser.add_argument(
        "--llm-concurrency", type=int, default=DEFAULT_LLM_CONCURRENCY,
        help=f"Solar API 동시 호출 (기본 {DEFAULT_LLM_CONCURRENCY})",
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH,
        help=f"Solar embedding 배치 (기본 {DEFAULT_EMBED_BATCH})",
    )
    parser.add_argument(
        "--qdrant-batch-size", type=int, default=DEFAULT_QDRANT_BATCH,
        help=f"Qdrant upsert 배치 (기본 {DEFAULT_QDRANT_BATCH})",
    )
    parser.add_argument(
        "--llm-model", type=str, default="solar-pro3",
        help="Upstage 모델명",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="처리 최대 person 수 (테스트)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="체크포인트 + 기존 Qdrant ID skip",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="현재 상태 출력",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.status:
        asyncio.run(show_status())
    else:
        asyncio.run(
            run_persons_full_pipeline(
                chunk_size=args.chunk_size,
                llm_rpm=args.llm_rpm,
                llm_concurrency=args.llm_concurrency,
                embed_batch_size=args.embed_batch_size,
                qdrant_batch_size=args.qdrant_batch_size,
                llm_model=args.llm_model,
                limit=args.limit,
                resume=args.resume,
            )
        )
