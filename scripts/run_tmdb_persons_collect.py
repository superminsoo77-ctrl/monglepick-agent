"""
TMDB Person 수집 스크립트 (Phase ML §9.5 Phase 1).

Neo4j Person 노드(572K+)에서 person_id 를 추출하여 TMDB Person API 를 호출하고,
biography / 필모그래피 / 외부 ID / 이미지 / 다국어 번역을 JSONL 로 저장한다.

설계 진실 원본:
    docs/데이터_적재_프로세스_전체분석_및_개선계획.md §9.5 Phase 1
    Phase C-1 ~ C-4 (Person 임베딩 파이프라인)

선결 조건:
    - Task #5 (run_full_reload.py) 완료 후 실행 (TMDB API 쿼터 경합 회피)
    - Neo4j 기동 + Person 노드 적재 완료
    - .env TMDB_API_KEY

처리 단계:
    1. Neo4j 에서 Person 노드의 person_id 전체 추출 (이미 적재된 ID)
    2. (체크포인트 존재 시) 이미 처리한 ID skip
    3. 청크 단위 (5,000 ID) 로 TMDBPersonCollector 비동기 호출
    4. 결과를 data/tmdb_persons.jsonl 에 append
    5. 청크마다 체크포인트 저장 (Ctrl+C 안전)

성능 추정 (572K Person 기준):
    - 35 req/sec × 86,400 = 3M req/day
    - 572K / 35 = 16,343 sec ≈ 4.5 시간
    - max_workers 10 + 5,000 청크 × ~115 청크

사용법:
    # 기본 실행 (Neo4j 전체 → JSONL append)
    PYTHONPATH=src uv run python scripts/run_tmdb_persons_collect.py

    # 중단점부터 재개 (체크포인트 + 기존 JSONL 의 ID skip)
    PYTHONPATH=src uv run python scripts/run_tmdb_persons_collect.py --resume

    # 청크/워커 크기 조정
    PYTHONPATH=src uv run python scripts/run_tmdb_persons_collect.py \\
        --chunk-size 10000 --max-workers 15 --rps 40

    # 처음 N 명만 (테스트)
    PYTHONPATH=src uv run python scripts/run_tmdb_persons_collect.py --limit 100

    # 상태 확인
    PYTHONPATH=src uv run python scripts/run_tmdb_persons_collect.py --status
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

# .env 로드 (config.settings 가 자동 로드하지만 명시 보장)
_env_file = _project_root / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import structlog  # noqa: E402

from monglepick.config import settings  # noqa: E402
from monglepick.data_pipeline.tmdb_person_collector import (  # noqa: E402
    TMDBPersonCollector,
    DEFAULT_RATE_LIMIT_RPS,
    DEFAULT_MAX_WORKERS,
)
from monglepick.db.clients import get_neo4j, init_all_clients, close_all_clients  # noqa: E402

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

OUTPUT_JSONL = Path("data/tmdb_persons/tmdb_persons.jsonl")
CHECKPOINT_FILE = Path("data/tmdb_persons/checkpoint.json")
DEFAULT_CHUNK_SIZE = 5_000


# ══════════════════════════════════════════════════════════════
# 체크포인트
# ══════════════════════════════════════════════════════════════


def _new_checkpoint() -> dict:
    return {
        "phase": "",                  # init / collecting / done
        "total_persons_in_neo4j": 0,
        "total_collected": 0,         # JSONL 에 저장된 총 person 수
        "total_failed": 0,
        "last_chunk_idx": -1,
        "collected_ids": [],          # 메모리 효율 위해 set 으로 유지하다 저장 시 list
        "start_time": datetime.now().isoformat(),
        "last_updated": "",
    }


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
            data.setdefault("collected_ids", [])
            return data
        except Exception as e:
            logger.warning("checkpoint_load_failed", error=str(e))
    return _new_checkpoint()


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    # collected_ids 가 set 이면 list 로 변환
    if isinstance(state.get("collected_ids"), set):
        state["collected_ids"] = sorted(state["collected_ids"])
    CHECKPOINT_FILE.write_text(
        json.dumps(state, ensure_ascii=False),  # indent 없이 (수십만 ID 저장)
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════
# Neo4j 에서 Person ID 추출
# ══════════════════════════════════════════════════════════════


async def _fetch_neo4j_person_ids() -> list[int]:
    """
    Neo4j (:Person) 노드에서 person_id 전체 추출.

    Returns:
        list[int]: TMDB person ID 리스트 (오름차순)
    """
    driver = await get_neo4j()
    person_ids: list[int] = []

    async with driver.session() as session:
        result = await session.run(
            "MATCH (p:Person) WHERE p.person_id IS NOT NULL "
            "RETURN p.person_id AS pid ORDER BY pid"
        )
        async for record in result:
            pid = record["pid"]
            if isinstance(pid, int) and pid > 0:
                person_ids.append(pid)

    logger.info("neo4j_person_ids_loaded", count=len(person_ids))
    return person_ids


# ══════════════════════════════════════════════════════════════
# JSONL append (재개 안전)
# ══════════════════════════════════════════════════════════════


def _append_jsonl(persons: list[dict]) -> None:
    """수집한 person 리스트를 JSONL 에 append (한 줄 = 한 person)."""
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("a", encoding="utf-8") as f:
        for p in persons:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def _read_existing_jsonl_ids() -> set[int]:
    """기존 JSONL 에서 이미 수집된 person_id 추출 (재개용)."""
    if not OUTPUT_JSONL.exists():
        return set()
    ids: set[int] = set()
    try:
        with OUTPUT_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get("id")
                    if isinstance(pid, int):
                        ids.add(pid)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning("jsonl_read_failed", error=str(e))
    return ids


# ══════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════


async def run_tmdb_persons_collect(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    rps: int = DEFAULT_RATE_LIMIT_RPS,
    limit: int | None = None,
    resume: bool = False,
) -> None:
    """
    Neo4j Person 노드 → TMDB Person API → JSONL 저장.

    Args:
        chunk_size: 한 청크당 person 수 (기본 5,000)
        max_workers: 동시 API 호출 수 (기본 10)
        rps: 초당 요청 한도 (기본 35)
        limit: 처음 N 명만 처리 (테스트용)
        resume: True 이면 기존 JSONL + 체크포인트의 collected_ids skip
    """
    pipeline_start = time.time()

    if not settings.TMDB_API_KEY:
        print("[ERROR] TMDB_API_KEY 가 .env 에 설정되지 않았습니다.")
        return

    # ── Step 0: DB 클라이언트 초기화 ──
    await init_all_clients()

    try:
        # ── Step 1: Neo4j Person ID 추출 ──
        print("[Step 1] Neo4j Person ID 추출")
        all_person_ids = await _fetch_neo4j_person_ids()
        print(f"  Neo4j Person 노드: {len(all_person_ids):,}건")

        if not all_person_ids:
            print("  Neo4j 에 Person 노드가 없습니다. Task #5 (run_full_reload) 완료 확인 필요.")
            return

        # ── Step 2: 체크포인트 로드 + skip ID ──
        checkpoint = _load_checkpoint() if resume else _new_checkpoint()
        skip_ids: set[int] = set()

        if resume:
            # JSONL 에서 이미 수집된 ID 로드
            jsonl_ids = _read_existing_jsonl_ids()
            checkpoint_ids = set(checkpoint.get("collected_ids", []))
            skip_ids = jsonl_ids | checkpoint_ids
            print(f"\n[Step 2] 재개 모드 — 이미 처리된 ID skip")
            print(f"  JSONL 기존 ID:        {len(jsonl_ids):,}")
            print(f"  체크포인트 ID:        {len(checkpoint_ids):,}")
            print(f"  skip 합계 (중복 제거): {len(skip_ids):,}")
        else:
            print(f"\n[Step 2] 신규 실행 — 기존 JSONL 무시 (덮어쓰기)")
            # 신규 실행: 기존 JSONL 백업 후 빈 파일로 시작
            if OUTPUT_JSONL.exists():
                backup = OUTPUT_JSONL.with_suffix(
                    f".jsonl.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                OUTPUT_JSONL.rename(backup)
                print(f"  기존 JSONL 백업: {backup.name}")

        # 처리할 ID 리스트 (skip 제외)
        target_ids = [pid for pid in all_person_ids if pid not in skip_ids]
        if limit:
            target_ids = target_ids[:limit]
            print(f"  --limit {limit} 적용")

        print(f"\n  처리 대상: {len(target_ids):,} / {len(all_person_ids):,}")
        if not target_ids:
            print("  처리할 ID 가 없습니다.")
            return

        checkpoint["total_persons_in_neo4j"] = len(all_person_ids)
        checkpoint["phase"] = "collecting"
        _save_checkpoint(checkpoint)

        # ── Step 3: 청크 단위 비동기 수집 ──
        print(f"\n[Step 3] TMDB Person API 수집")
        print(f"  chunk_size={chunk_size:,}, max_workers={max_workers}, rps={rps}")
        print(f"  예상 소요: ~{len(target_ids) / rps / 60:.1f} 분 (rps 기준)")
        print()

        total_collected = checkpoint.get("total_collected", 0)
        total_failed = 0
        collected_set: set[int] = set(checkpoint.get("collected_ids", []))

        async with TMDBPersonCollector(rps=rps) as collector:
            chunk_idx = checkpoint.get("last_chunk_idx", -1) + 1

            for batch_start in range(0, len(target_ids), chunk_size):
                batch_end = min(batch_start + chunk_size, len(target_ids))
                batch_ids = target_ids[batch_start:batch_end]
                chunk_start_time = time.time()

                # 청크 비동기 수집
                results = await collector.collect_persons_batch(
                    batch_ids,
                    max_workers=max_workers,
                )

                # JSONL append
                _append_jsonl(results)

                # 카운트 + 체크포인트
                chunk_elapsed = time.time() - chunk_start_time
                chunk_collected = len(results)
                chunk_failed = len(batch_ids) - chunk_collected
                total_collected += chunk_collected
                total_failed += chunk_failed

                collected_set.update(p["id"] for p in results if "id" in p)
                checkpoint["total_collected"] = total_collected
                checkpoint["total_failed"] += chunk_failed
                checkpoint["last_chunk_idx"] = chunk_idx
                checkpoint["collected_ids"] = sorted(collected_set)
                _save_checkpoint(checkpoint)

                total_elapsed = time.time() - pipeline_start
                rate = total_collected / total_elapsed if total_elapsed > 0 else 0
                remaining = len(target_ids) - batch_end
                eta_min = remaining / rate / 60 if rate > 0 else 0

                print(
                    f"  [Chunk {chunk_idx:>4}] "
                    f"+{chunk_collected:>5} (fail {chunk_failed:>3}) | "
                    f"누적 {total_collected:>7,} / {len(target_ids):,} | "
                    f"속도 {rate:>5.1f}/s | "
                    f"청크 {chunk_elapsed:>5.1f}s | "
                    f"ETA {eta_min:>5.0f}m"
                )

                chunk_idx += 1

        # ── 완료 ──
        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[TMDB Person 수집 완료]")
        print(f"  Neo4j 전체:    {len(all_person_ids):>10,}")
        print(f"  처리 대상:     {len(target_ids):>10,}")
        print(f"  수집 성공:     {total_collected:>10,}")
        print(f"  실패/누락:     {total_failed:>10,}")
        print(f"  API 호출 수:   {collector.call_count:>10,}")
        print(f"  소요:          {total_elapsed / 60:>10.1f} 분")
        print(f"  출력:          {OUTPUT_JSONL}")
        print(f"{'=' * 60}")

    finally:
        await close_all_clients()


# ══════════════════════════════════════════════════════════════
# 상태 조회
# ══════════════════════════════════════════════════════════════


async def show_status() -> None:
    checkpoint = _load_checkpoint()

    print("=" * 60)
    print(f"  TMDB Person 수집 체크포인트")
    print("=" * 60)
    print(f"  단계:           {checkpoint.get('phase', '미시작')}")
    print(f"  Neo4j 전체:     {checkpoint.get('total_persons_in_neo4j', 0):>10,}")
    print(f"  수집 완료:      {checkpoint.get('total_collected', 0):>10,}")
    print(f"  실패:           {checkpoint.get('total_failed', 0):>10,}")
    print(f"  마지막 청크:    {checkpoint.get('last_chunk_idx', -1):>10}")
    print(f"  마지막 갱신:    {checkpoint.get('last_updated', '-')}")
    print()

    if OUTPUT_JSONL.exists():
        size_mb = OUTPUT_JSONL.stat().st_size / (1024 * 1024)
        # 라인 수 카운트 (대용량이면 건너뜀)
        if size_mb < 500:
            with OUTPUT_JSONL.open("r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
            print(f"  JSONL 라인:     {line_count:>10,}")
        print(f"  JSONL 크기:     {size_mb:>10,.1f} MB")
        print(f"  경로:           {OUTPUT_JSONL}")
    else:
        print(f"  JSONL:          없음 ({OUTPUT_JSONL})")

    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TMDB Person API 수집 (Neo4j → JSONL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 실행 (신규)
  PYTHONPATH=src uv run python scripts/run_tmdb_persons_collect.py

  # 중단점부터 재개
  PYTHONPATH=src uv run python scripts/run_tmdb_persons_collect.py --resume

  # 처음 100명만 (테스트)
  PYTHONPATH=src uv run python scripts/run_tmdb_persons_collect.py --limit 100

  # 상태 확인
  PYTHONPATH=src uv run python scripts/run_tmdb_persons_collect.py --status
        """,
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"청크 크기 (기본: {DEFAULT_CHUNK_SIZE:,})",
    )
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"동시 API 호출 수 (기본: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--rps", type=int, default=DEFAULT_RATE_LIMIT_RPS,
        help=f"초당 요청 한도 (기본: {DEFAULT_RATE_LIMIT_RPS})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="처리할 최대 ID 수 (테스트용)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="기존 JSONL + 체크포인트의 ID 를 skip 하고 재개",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="현재 체크포인트 + JSONL 상태만 확인",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.status:
        asyncio.run(show_status())
    else:
        asyncio.run(
            run_tmdb_persons_collect(
                chunk_size=args.chunk_size,
                max_workers=args.max_workers,
                rps=args.rps,
                limit=args.limit,
                resume=args.resume,
            )
        )
