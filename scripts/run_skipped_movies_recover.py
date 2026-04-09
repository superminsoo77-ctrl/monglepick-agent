"""
Task #5 에서 스킵된 영화 복구 스크립트 (Phase E-1).

run_full_reload.py 가 처리하지 못하고 스킵한 영화 (~9.8%, 약 11만 건) 를
완화된 검증 규칙으로 재처리하여 추가 적재한다.

배경:
    Task #5 에서 17.5% 진행 시점 기준 스킵률 약 9.8% (24,144건).
    주요 스킵 사유: validate_movie() 의 release_year=0 검증 실패.
    KOBIS / KMDb / Kaggle / 오래된 TMDB 영화 중 prod_year/release_date
    가 비어있는 영화가 전부 스킵됨.

복구 전략:
    1. JSONL 전체 재스캔 → process_raw_movie 로 변환
    2. 변환 성공 doc 중에서 **이미 Qdrant 에 있는 ID** 제외 (중복 방지)
    3. 완화된 검증 (validate_movie_relaxed) 적용:
       - release_year=0 허용
       - genres 빈 리스트 허용 (장르 없는 영화도 임베딩만 되면 OK)
    4. release_year=0 인 경우 1900 으로 fallback (validation 통과용 보정)
    5. mood_batch 적용 → embedding → Qdrant/Neo4j/ES 적재

주의:
    - Task #5 완료 후 실행
    - --clear-db 사용 안 함 (기존 데이터 보존)
    - 중복 적재 방지: Qdrant scroll 로 기존 ID 셋 로드
    - mood_batch 는 Solar Pro 3 정밀 mood 적용 (run_full_reload 와 동일 정책)

사용법:
    # 기본 실행 (전체 JSONL 재스캔, 스킵된 영화만 적재)
    PYTHONPATH=src uv run python scripts/run_skipped_movies_recover.py

    # 처음 N 건만 (테스트)
    PYTHONPATH=src uv run python scripts/run_skipped_movies_recover.py --limit 100

    # 체크포인트 재개
    PYTHONPATH=src uv run python scripts/run_skipped_movies_recover.py --resume

    # 상태 확인
    PYTHONPATH=src uv run python scripts/run_skipped_movies_recover.py --status

선결 조건:
    - Task #5 (run_full_reload.py) 완료
    - data/tmdb_full/tmdb_full_movies.jsonl 존재 (재수집 결과)
    - .env UPSTAGE_API_KEY (mood_batch + embedding)
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
from monglepick.data_pipeline.embedder import embed_texts  # noqa: E402
from monglepick.data_pipeline.es_loader import load_to_elasticsearch  # noqa: E402
from monglepick.data_pipeline.models import MovieDocument, TMDBRawMovie  # noqa: E402
from monglepick.data_pipeline.mood_batch import enrich_documents_with_solar_mood  # noqa: E402
from monglepick.data_pipeline.neo4j_loader import load_to_neo4j  # noqa: E402
from monglepick.data_pipeline.preprocessor import (  # noqa: E402
    build_embedding_text,
    process_raw_movie,
)
from monglepick.data_pipeline.qdrant_loader import load_to_qdrant  # noqa: E402
from monglepick.db.clients import (  # noqa: E402
    init_all_clients,
    close_all_clients,
    get_qdrant,
)

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

DEFAULT_JSONL_PATH = Path("data/tmdb_full/tmdb_full_movies.jsonl")
CHECKPOINT_FILE = Path("data/skipped_recover_checkpoint.json")
DEFAULT_CHUNK_SIZE = 2_000      # JSONL 청크
DEFAULT_BATCH_SIZE = 50         # Solar embedding 배치
DEFAULT_MOOD_RPM = 100
DEFAULT_MOOD_CONCURRENCY = 20
DEFAULT_MOOD_BATCH = 10


# ══════════════════════════════════════════════════════════════
# 완화된 검증
# ══════════════════════════════════════════════════════════════


def validate_movie_relaxed(doc: MovieDocument) -> bool:
    """
    완화된 영화 유효성 검증 (Phase E-1).

    원본 validate_movie() 대비 완화점:
        - release_year=0 허용 (강제 fallback 1900 적용)
        - genres 빈 리스트 허용 (장르 없는 영화도 임베딩 가능)
        - title 만 필수 (id 는 process_raw_movie 가 항상 채움)
    """
    if not doc.id or not doc.title:
        return False
    if doc.rating < 0 or doc.rating > 10:
        return False
    return True


def _normalize_release_year(doc: MovieDocument) -> None:
    """
    release_year 가 0 이거나 비정상일 때 1900 으로 fallback.

    validate_movie 의 release_year >= 1900 검증을 통과시키기 위함이지만,
    여기서는 validate_movie_relaxed 를 쓰므로 단순히 데이터 보정 목적.
    """
    if not doc.release_year or doc.release_year < 1900:
        doc.release_year = 1900
    # build_embedding_text 가 release_year 를 직접 사용하지는 않으므로
    # 주로 다운스트림 필터링용 보정.


# ══════════════════════════════════════════════════════════════
# 체크포인트
# ══════════════════════════════════════════════════════════════


def _new_checkpoint() -> dict:
    return {
        "phase": "",
        "last_jsonl_line": 0,
        "total_scanned": 0,
        "total_existing_skipped": 0,   # 이미 Qdrant 에 있어서 skip
        "total_validation_failed": 0,  # 완화 검증도 실패
        "total_recovered": 0,          # 추가 적재 성공
        "total_failed_loaders": 0,     # loader 단계 실패
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
# 기존 Qdrant ID 셋 로드 (중복 방지)
# ══════════════════════════════════════════════════════════════


async def _load_existing_qdrant_ids() -> set[str]:
    """Qdrant `movies` 컬렉션의 모든 movie_id 셋 로드 (payload.id 기준)."""
    print("[Step 1] 기존 Qdrant 영화 ID 로드 중...")
    client = await get_qdrant()
    ids: set[str] = set()
    offset = None

    while True:
        result = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=5000,
            offset=offset,
            with_vectors=False,
            with_payload=["id"],
        )
        points = result[0]
        next_offset = result[1]

        if not points:
            break

        for p in points:
            mid = (p.payload or {}).get("id") or str(p.id)
            ids.add(str(mid))

        if next_offset is None:
            break
        offset = next_offset

    print(f"  Qdrant 기존 ID: {len(ids):,}")
    return ids


# ══════════════════════════════════════════════════════════════
# JSONL 스트리밍
# ══════════════════════════════════════════════════════════════


def _jsonl_lines(path: Path, skip_lines: int = 0):
    """JSONL 파일을 라인 단위로 yield. skip_lines 만큼 건너뛴다."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            if line_no <= skip_lines:
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


# ══════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════


async def run_skipped_recover(
    jsonl_path: Path = DEFAULT_JSONL_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    embed_batch_size: int = DEFAULT_BATCH_SIZE,
    mood_provider: str = "upstage",
    mood_rpm: int = DEFAULT_MOOD_RPM,
    mood_concurrency: int = DEFAULT_MOOD_CONCURRENCY,
    mood_batch_size: int = DEFAULT_MOOD_BATCH,
    limit: int | None = None,
    resume: bool = False,
) -> None:
    """
    완화 검증으로 스킵된 영화를 재처리하여 추가 적재.

    Args:
        jsonl_path: TMDB JSONL 경로
        chunk_size: JSONL 청크 (기본 2,000)
        embed_batch_size: Solar embedding 배치
        mood_provider: 'upstage' | 'fallback'
        mood_rpm/concurrency/batch_size: Solar Pro 3 mood 옵션
        limit: 최대 스킵 영화 처리 수 (테스트용)
        resume: 체크포인트 재개
    """
    pipeline_start = time.time()

    if not jsonl_path.exists():
        print(f"[ERROR] JSONL 파일이 없습니다: {jsonl_path}")
        return

    checkpoint = _load_checkpoint() if resume else _new_checkpoint()
    skip_lines = checkpoint.get("last_jsonl_line", 0) if resume else 0

    print(f"[Step 0] DB 클라이언트 초기화")
    await init_all_clients()

    try:
        # 기존 ID 셋 로드 (중복 방지)
        existing_ids = await _load_existing_qdrant_ids()

        # mood_provider 검증
        upstage_api_key: str | None = None
        if mood_provider == "upstage":
            upstage_api_key = settings.UPSTAGE_API_KEY if hasattr(settings, "UPSTAGE_API_KEY") else None
            if not upstage_api_key:
                print("  [경고] UPSTAGE_API_KEY 없음 → fallback mood 사용")
                mood_provider = "fallback"
            else:
                print(f"  무드 모드: upstage solar-pro3 (rpm={mood_rpm})")

        print(f"\n[Step 2] JSONL 스캔 + 완화 검증 + 적재")
        print(f"  JSONL: {jsonl_path}")
        print(f"  chunk_size: {chunk_size:,}, skip_lines: {skip_lines:,}")
        print()

        checkpoint["phase"] = "recovering"

        # 청크 단위 처리
        chunk_buffer: list[MovieDocument] = []
        recovered_in_run = 0

        for line_no, raw_dict in _jsonl_lines(jsonl_path, skip_lines=skip_lines):
            checkpoint["total_scanned"] += 1
            checkpoint["last_jsonl_line"] = line_no

            try:
                # 정제 후 TMDBRawMovie 생성
                recs = raw_dict.get("recommendations", [])
                if recs and isinstance(recs[0], int):
                    raw_dict["recommendations"] = [{"id": r} for r in recs]
                raw_dict.pop("recommendation_ids_raw", None)
                raw_dict.pop("similar_movies_full", None)
                raw_dict.pop("changes", None)

                raw = TMDBRawMovie(**raw_dict)
                doc = await process_raw_movie(raw, generate_mood=False)
                if doc is None:
                    continue

                # 이미 Qdrant 에 있는 영화 스킵 (중복 방지)
                if doc.id in existing_ids:
                    checkpoint["total_existing_skipped"] += 1
                    continue

                # 완화된 검증 + release_year 보정
                if not validate_movie_relaxed(doc):
                    checkpoint["total_validation_failed"] += 1
                    continue
                _normalize_release_year(doc)

                # build_embedding_text 재실행 (필수)
                doc.embedding_text = build_embedding_text(doc)
                if not doc.embedding_text or len(doc.embedding_text) < 10:
                    checkpoint["total_validation_failed"] += 1
                    continue

                chunk_buffer.append(doc)

            except Exception as e:
                logger.debug("recover_preprocess_failed", line_no=line_no, error=str(e)[:200])
                continue

            # 청크 가득 차면 처리
            if len(chunk_buffer) >= chunk_size:
                await _process_chunk(
                    chunk_buffer,
                    embed_batch_size,
                    mood_provider,
                    upstage_api_key,
                    mood_rpm,
                    mood_concurrency,
                    mood_batch_size,
                    checkpoint,
                )
                recovered_in_run += len(chunk_buffer)
                chunk_buffer = []
                _save_checkpoint(checkpoint)

                # 진행률 출력
                total_elapsed = time.time() - pipeline_start
                rate = checkpoint["total_recovered"] / total_elapsed if total_elapsed > 0 else 0
                print(
                    f"  scanned {checkpoint['total_scanned']:>9,} | "
                    f"existing_skip {checkpoint['total_existing_skipped']:>7,} | "
                    f"valid_fail {checkpoint['total_validation_failed']:>5,} | "
                    f"recovered {checkpoint['total_recovered']:>7,} | "
                    f"rate {rate:>4.1f}/s"
                )

                # limit 검증
                if limit and checkpoint["total_recovered"] >= limit:
                    print(f"  --limit {limit} 도달 → 중단")
                    break

        # 마지막 청크 처리
        if chunk_buffer:
            await _process_chunk(
                chunk_buffer,
                embed_batch_size,
                mood_provider,
                upstage_api_key,
                mood_rpm,
                mood_concurrency,
                mood_batch_size,
                checkpoint,
            )
            _save_checkpoint(checkpoint)

        # 완료
        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[스킵 복구 완료]")
        print(f"  스캔:           {checkpoint['total_scanned']:>10,}")
        print(f"  기존 ID 스킵:   {checkpoint['total_existing_skipped']:>10,}")
        print(f"  완화 검증 실패: {checkpoint['total_validation_failed']:>10,}")
        print(f"  추가 적재 성공: {checkpoint['total_recovered']:>10,}")
        print(f"  loader 실패:    {checkpoint['total_failed_loaders']:>10,}")
        print(f"  소요:           {total_elapsed / 60:>10.1f} 분")
        print(f"{'=' * 60}")

    finally:
        await close_all_clients()


async def _process_chunk(
    chunk: list[MovieDocument],
    embed_batch_size: int,
    mood_provider: str,
    upstage_api_key: str | None,
    mood_rpm: int,
    mood_concurrency: int,
    mood_batch_size: int,
    checkpoint: dict,
) -> None:
    """단일 청크: mood 보강 → 임베딩 → 3DB 적재."""
    if not chunk:
        return

    # 1) Solar Pro 3 mood 보강 (선택)
    if mood_provider == "upstage" and upstage_api_key:
        try:
            await enrich_documents_with_solar_mood(
                documents=chunk,
                api_key=upstage_api_key,
                model="solar-pro3",
                rpm=mood_rpm,
                concurrency=mood_concurrency,
                batch_size=mood_batch_size,
                rebuild_embedding_text=True,
            )
        except Exception as e:
            logger.error("recover_mood_enrich_failed", error=str(e)[:200])

    # 2) 임베딩
    try:
        texts = [doc.embedding_text for doc in chunk]
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, embed_texts, texts, embed_batch_size)
    except Exception as e:
        logger.error("recover_embed_failed", error=str(e)[:200], count=len(chunk))
        checkpoint["total_failed_loaders"] += len(chunk)
        return

    # 3) 3DB 동시 적재
    try:
        results = await asyncio.gather(
            load_to_qdrant(chunk, embeddings),
            load_to_neo4j(chunk),
            load_to_elasticsearch(chunk),
            return_exceptions=True,
        )
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(
                    "recover_loader_failed",
                    db=["qdrant", "neo4j", "es"][i],
                    error=str(r)[:200],
                )
        checkpoint["total_recovered"] += len(chunk)
    except Exception as e:
        logger.error("recover_load_failed", error=str(e)[:200], count=len(chunk))
        checkpoint["total_failed_loaders"] += len(chunk)


# ══════════════════════════════════════════════════════════════
# 상태 조회
# ══════════════════════════════════════════════════════════════


def show_status() -> None:
    if not CHECKPOINT_FILE.exists():
        print("체크포인트 파일이 없습니다.")
        return
    cp = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    print("=" * 60)
    print(f"  Skipped Movies Recover 체크포인트")
    print("=" * 60)
    print(f"  phase:              {cp.get('phase', '미시작')}")
    print(f"  last_jsonl_line:    {cp.get('last_jsonl_line', 0):>10,}")
    print(f"  total_scanned:      {cp.get('total_scanned', 0):>10,}")
    print(f"  existing_skipped:   {cp.get('total_existing_skipped', 0):>10,}")
    print(f"  validation_failed:  {cp.get('total_validation_failed', 0):>10,}")
    print(f"  recovered:          {cp.get('total_recovered', 0):>10,}")
    print(f"  loader_failed:      {cp.get('total_failed_loaders', 0):>10,}")
    print(f"  마지막 갱신:        {cp.get('last_updated', '-')}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task #5 에서 스킵된 영화 복구 (완화 검증 + 추가 적재)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (전체 JSONL 재스캔)
  PYTHONPATH=src uv run python scripts/run_skipped_movies_recover.py

  # 처음 100건만 (테스트)
  PYTHONPATH=src uv run python scripts/run_skipped_movies_recover.py --limit 100

  # 재개
  PYTHONPATH=src uv run python scripts/run_skipped_movies_recover.py --resume

  # 상태 확인
  PYTHONPATH=src uv run python scripts/run_skipped_movies_recover.py --status
        """,
    )
    parser.add_argument(
        "--jsonl-path", type=str, default=str(DEFAULT_JSONL_PATH),
        help=f"TMDB JSONL 경로 (기본: {DEFAULT_JSONL_PATH})",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"JSONL 청크 크기 (기본: {DEFAULT_CHUNK_SIZE:,})",
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Solar embedding 배치 (기본: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--mood-provider", choices=["upstage", "fallback"], default="upstage",
        help="mood 생성 방식 (기본: upstage)",
    )
    parser.add_argument("--mood-rpm", type=int, default=DEFAULT_MOOD_RPM)
    parser.add_argument("--mood-concurrency", type=int, default=DEFAULT_MOOD_CONCURRENCY)
    parser.add_argument("--mood-batch-size", type=int, default=DEFAULT_MOOD_BATCH)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="복구 최대 건수 (테스트용)",
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
        show_status()
    else:
        asyncio.run(
            run_skipped_recover(
                jsonl_path=Path(args.jsonl_path),
                chunk_size=args.chunk_size,
                embed_batch_size=args.embed_batch_size,
                mood_provider=args.mood_provider,
                mood_rpm=args.mood_rpm,
                mood_concurrency=args.mood_concurrency,
                mood_batch_size=args.mood_batch_size,
                limit=args.limit,
                resume=args.resume,
            )
        )
