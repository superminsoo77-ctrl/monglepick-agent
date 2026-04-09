"""
Kaggle MovieLens ratings.csv → MySQL kaggle_watch_history 적재 스크립트.

Phase 3 재적재 (2026-04-08) 신규 스크립트:
    기존 `init.sql`이 만든 `kaggle_watch_history` 테이블에 Kaggle MovieLens
    ratings.csv (26M+ 행) 시드 데이터를 배치 INSERT 한다.

설계 진실 원본:
    `docs/데이터_적재_프로세스_전체분석_및_개선계획.md` §5

주요 정책:
    - **테이블명**: `kaggle_watch_history` (Phase 3에서 watch_history 에서 리네임)
    - **user_id 접두**: `kaggle_{userId}` — 실서비스 users.user_id 와 충돌 방지
    - **movie_id 매핑**: links.csv 의 (Kaggle movieId → TMDB id) 매핑 후 저장
    - **timestamp 변환**: Unix epoch → MySQL TIMESTAMP (1995~2020 범위)
    - **멱등성**: 기본 모드는 TRUNCATE → INSERT (재실행 안전)
    - **체크포인트**: data/kaggle_ratings_checkpoint.json (재개 지원)
    - **배치**: aiomysql executemany (5,000 건/배치)
    - **스트리밍**: pandas chunksize=50,000 으로 메모리 효율

데이터 흐름:
    1. links.csv 전체 로드 → dict[kaggle_movieId, tmdb_id]
    2. (옵션) 기존 kaggle_watch_history 건수 확인
    3. (옵션) TRUNCATE
    4. ratings.csv 청크 스트리밍 (50,000 행씩)
       - 매핑 없는 movieId 는 skip + skip_count++
       - 변환: user_id, movie_id, rating, watched_at
       - 5,000 행씩 executemany INSERT
       - 청크마다 체크포인트 저장
    5. 최종 검증: SELECT COUNT(*)

성능 추정:
    - 26M / 5,000 = 5,200 배치
    - 배치당 ~1초 (executemany + commit)
    - 총 ~90분 (디스크 IO 의존)

사용법:
    # 전체 적재 (TRUNCATE → INSERT)
    PYTHONPATH=src uv run python scripts/run_kaggle_ratings_load.py

    # 중단점부터 재개 (TRUNCATE 안 함, 이미 처리한 행 skip)
    PYTHONPATH=src uv run python scripts/run_kaggle_ratings_load.py --resume

    # 건수 확인만
    PYTHONPATH=src uv run python scripts/run_kaggle_ratings_load.py --status

    # TRUNCATE 없이 INSERT만 (기존 데이터 유지)
    PYTHONPATH=src uv run python scripts/run_kaggle_ratings_load.py --no-truncate

    # 배치/청크 크기 조정
    PYTHONPATH=src uv run python scripts/run_kaggle_ratings_load.py \\
        --chunk-size 100000 --batch-size 10000

선결 조건:
    1. MySQL `kaggle_watch_history` 테이블 존재 (init.sql 수정 후 수동 실행)
    2. data/kaggle_movies/ratings.csv (심볼릭 링크 또는 실파일)
    3. data/kaggle_movies/links.csv
    4. .env 의 MYSQL_HOST/PORT/USER/PASSWORD/DATABASE 설정
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

# 프로젝트 루트를 sys.path 에 추가
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# .env 로드 (config.settings 가 자동 로드하지만 명시적으로 보장)
_env_file = _project_root / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import aiomysql  # noqa: E402
import pandas as pd  # noqa: E402
import structlog  # noqa: E402

from monglepick.config import settings  # noqa: E402

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

#: Kaggle 데이터 디렉토리 (심볼릭 링크 가능)
DEFAULT_KAGGLE_DIR = Path("data/kaggle_movies")
RATINGS_CSV = "ratings.csv"
LINKS_CSV = "links.csv"

#: 체크포인트 파일
CHECKPOINT_FILE = Path("data/kaggle_ratings_checkpoint.json")

#: 기본 청크/배치 크기
DEFAULT_CHUNK_SIZE = 50_000   # pandas read_csv chunksize
DEFAULT_BATCH_SIZE = 5_000    # MySQL executemany 배치
DEFAULT_POOL_SIZE = 5         # aiomysql pool

#: MySQL 테이블 + INSERT 문
TABLE_NAME = "kaggle_watch_history"

INSERT_SQL = f"""
INSERT INTO {TABLE_NAME} (user_id, movie_id, rating, watched_at)
VALUES (%s, %s, %s, %s)
"""

TRUNCATE_SQL = f"TRUNCATE TABLE {TABLE_NAME}"
COUNT_SQL = f"SELECT COUNT(*) FROM {TABLE_NAME}"


# ══════════════════════════════════════════════════════════════
# 체크포인트 (재개 지원)
# ══════════════════════════════════════════════════════════════


def _new_checkpoint() -> dict:
    """새 체크포인트 구조."""
    return {
        "phase": "",                  # init / truncate / loading / done
        "rows_processed": 0,          # ratings.csv 에서 읽은 총 행 수 (skip 포함)
        "rows_inserted": 0,           # MySQL 에 INSERT 완료된 행 수
        "rows_skipped": 0,            # links.csv 매핑 없어서 skip 한 행 수
        "rows_failed": 0,             # 변환/INSERT 실패 행 수
        "last_chunk_idx": -1,         # 마지막 처리 완료 청크 인덱스 (재개용)
        "links_count": 0,
        "start_time": datetime.now().isoformat(),
        "last_updated": "",
    }


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("checkpoint_load_failed_using_new", error=str(e))
    return _new_checkpoint()


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════
# links.csv 로드 (Kaggle movieId → TMDB id 매핑)
# ══════════════════════════════════════════════════════════════


def load_links_mapping(kaggle_dir: Path) -> dict[int, int]:
    """
    links.csv 를 로드하여 Kaggle movieId → TMDB id 매핑 dict 를 반환한다.

    links.csv 컬럼: movieId, imdbId, tmdbId
    tmdbId 가 NaN 인 행은 매핑에서 제외한다.

    Args:
        kaggle_dir: Kaggle CSV 디렉토리

    Returns:
        dict[int, int]: {kaggle_movieId: tmdb_id}
    """
    links_path = kaggle_dir / LINKS_CSV
    if not links_path.exists():
        raise FileNotFoundError(f"links.csv 를 찾을 수 없습니다: {links_path}")

    logger.info("links_loading", path=str(links_path))
    df = pd.read_csv(links_path, dtype={"movieId": "Int64", "tmdbId": "Int64"})

    # tmdbId 가 NaN 인 행 제외
    before = len(df)
    df = df.dropna(subset=["tmdbId"])
    after = len(df)

    mapping: dict[int, int] = {
        int(row["movieId"]): int(row["tmdbId"])
        for _, row in df.iterrows()
    }

    logger.info(
        "links_loaded",
        total_rows=before,
        valid_mappings=after,
        unique_mappings=len(mapping),
        dropped_no_tmdb=before - after,
    )
    return mapping


# ══════════════════════════════════════════════════════════════
# ratings.csv 청크 변환 → MySQL 행 튜플
# ══════════════════════════════════════════════════════════════


def _chunk_to_rows(
    chunk_df: pd.DataFrame,
    id_map: dict[int, int],
) -> tuple[list[tuple], int]:
    """
    pandas chunk DataFrame 을 MySQL INSERT 행 튜플 리스트로 변환한다.

    Args:
        chunk_df: ratings.csv 청크 (userId, movieId, rating, timestamp)
        id_map: Kaggle movieId → TMDB id 매핑

    Returns:
        (rows: list[tuple], skipped: int)
            - rows: [(user_id, movie_id, rating, watched_at), ...]
            - skipped: 매핑 없어서 제외된 행 수
    """
    rows: list[tuple] = []
    skipped = 0

    # NumPy ndarray 로 변환하여 iterrows 보다 빠르게 처리
    user_ids = chunk_df["userId"].astype("int64").to_numpy()
    movie_ids = chunk_df["movieId"].astype("int64").to_numpy()
    ratings = chunk_df["rating"].astype("float64").to_numpy()
    timestamps = chunk_df["timestamp"].astype("int64").to_numpy()

    for u, m, r, ts in zip(user_ids, movie_ids, ratings, timestamps):
        # links.csv 에 매핑 없으면 skip
        tmdb_id = id_map.get(int(m))
        if tmdb_id is None:
            skipped += 1
            continue

        # user_id: kaggle_{userId} 접두 (실서비스 users 와 충돌 방지)
        user_id = f"kaggle_{int(u)}"
        # movie_id: TMDB id 문자열 (movies.movie_id 와 일관)
        movie_id = str(tmdb_id)
        # watched_at: Unix epoch → datetime
        watched_at = datetime.fromtimestamp(int(ts))
        # rating: float (0.5~5.0)
        rating_val = float(r)

        rows.append((user_id, movie_id, rating_val, watched_at))

    return rows, skipped


# ══════════════════════════════════════════════════════════════
# MySQL 헬퍼
# ══════════════════════════════════════════════════════════════


async def _create_pool() -> aiomysql.Pool:
    """aiomysql 커넥션 풀 생성."""
    pool = await aiomysql.create_pool(
        host=settings.MYSQL_HOST,
        port=int(getattr(settings, "MYSQL_PORT", 3306)),
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        db=settings.MYSQL_DATABASE,
        minsize=1,
        maxsize=DEFAULT_POOL_SIZE,
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=10,
    )
    logger.info(
        "mysql_pool_created",
        host=settings.MYSQL_HOST,
        db=settings.MYSQL_DATABASE,
        pool_max=DEFAULT_POOL_SIZE,
    )
    return pool


async def _verify_table_exists(pool: aiomysql.Pool) -> bool:
    """kaggle_watch_history 테이블 존재 확인."""
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema=%s AND table_name=%s",
                (settings.MYSQL_DATABASE, TABLE_NAME),
            )
            (cnt,) = await cur.fetchone()
            return cnt > 0


async def _truncate_table(pool: aiomysql.Pool) -> None:
    """kaggle_watch_history 테이블 비우기."""
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(TRUNCATE_SQL)
        await conn.commit()
    logger.info("table_truncated", table=TABLE_NAME)


async def _count_table(pool: aiomysql.Pool) -> int:
    """현재 kaggle_watch_history 건수 조회."""
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(COUNT_SQL)
            (cnt,) = await cur.fetchone()
            return cnt


async def _insert_batch(pool: aiomysql.Pool, rows: list[tuple]) -> int:
    """배치 INSERT (executemany + commit)."""
    if not rows:
        return 0
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            try:
                await cur.executemany(INSERT_SQL, rows)
                await conn.commit()
                return len(rows)
            except Exception:
                await conn.rollback()
                raise


# ══════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════


async def run_kaggle_ratings_load(
    kaggle_dir: Path = DEFAULT_KAGGLE_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    truncate: bool = True,
    resume: bool = False,
) -> None:
    """
    Kaggle ratings.csv → MySQL kaggle_watch_history 적재.

    Args:
        kaggle_dir: Kaggle CSV 디렉토리 (기본: data/kaggle_movies)
        chunk_size: pandas read_csv chunksize (기본: 50,000)
        batch_size: aiomysql executemany 배치 (기본: 5,000)
        truncate: True 이면 적재 전 TRUNCATE (기본 True)
        resume: True 이면 체크포인트의 last_chunk_idx 부터 재개 (truncate 자동 False)
    """
    pipeline_start = time.time()

    # ── 사전 검증 ──
    ratings_path = kaggle_dir / RATINGS_CSV
    if not ratings_path.exists():
        print(f"[ERROR] ratings.csv 를 찾을 수 없습니다: {ratings_path}")
        return

    # ── 체크포인트 로드 ──
    checkpoint = _load_checkpoint() if resume else _new_checkpoint()
    skip_chunks = checkpoint.get("last_chunk_idx", -1) + 1 if resume else 0
    if resume and skip_chunks > 0:
        print(f"[RESUME] 체크포인트에서 재개: 청크 {skip_chunks}부터 시작")
        truncate = False  # 재개 시에는 절대 TRUNCATE 금지
    else:
        checkpoint = _new_checkpoint()

    # ── links.csv 매핑 로드 ──
    print(f"[Step 1] links.csv 매핑 로드: {kaggle_dir / LINKS_CSV}")
    id_map = load_links_mapping(kaggle_dir)
    checkpoint["links_count"] = len(id_map)
    print(f"  매핑 수: {len(id_map):,} (Kaggle movieId → TMDB id)")

    # ── MySQL 풀 + 테이블 검증 ──
    print(f"\n[Step 2] MySQL 연결 + 테이블 검증")
    pool = await _create_pool()

    try:
        if not await _verify_table_exists(pool):
            print(
                f"[ERROR] {TABLE_NAME} 테이블이 존재하지 않습니다.\n"
                f"        먼저 init.sql 을 실행하여 테이블을 생성하세요:\n"
                f"        docker exec -i monglepick-mysql mysql -u {settings.MYSQL_USER} -p<pw> "
                f"{settings.MYSQL_DATABASE} < db_dumps/prod_old_backup/init.sql"
            )
            return

        before_count = await _count_table(pool)
        print(f"  현재 {TABLE_NAME} 건수: {before_count:,}")

        # ── TRUNCATE (기본) ──
        if truncate:
            print(f"\n[Step 3] {TABLE_NAME} TRUNCATE")
            await _truncate_table(pool)
            checkpoint["phase"] = "truncate"
            _save_checkpoint(checkpoint)
        else:
            print(f"\n[Step 3] TRUNCATE 스킵 (--no-truncate / --resume)")

        # ── ratings.csv 청크 스트리밍 + 배치 INSERT ──
        print(f"\n[Step 4] ratings.csv 청크 스트리밍 적재")
        print(f"  chunk_size={chunk_size:,}, batch_size={batch_size:,}")

        checkpoint["phase"] = "loading"
        chunk_iter = pd.read_csv(
            ratings_path,
            chunksize=chunk_size,
            dtype={
                "userId": "int64",
                "movieId": "int64",
                "rating": "float64",
                "timestamp": "int64",
            },
        )

        chunk_idx = -1
        for chunk_df in chunk_iter:
            chunk_idx += 1

            # 재개 모드: 이미 처리한 청크 skip
            if chunk_idx < skip_chunks:
                continue

            chunk_start = time.time()

            # 1) 청크 변환
            rows, skipped = _chunk_to_rows(chunk_df, id_map)
            checkpoint["rows_processed"] += len(chunk_df)
            checkpoint["rows_skipped"] += skipped

            # 2) 배치 분할 INSERT
            chunk_inserted = 0
            chunk_failed = 0
            for i in range(0, len(rows), batch_size):
                sub_batch = rows[i:i + batch_size]
                try:
                    inserted = await _insert_batch(pool, sub_batch)
                    chunk_inserted += inserted
                except Exception as e:
                    chunk_failed += len(sub_batch)
                    logger.error(
                        "batch_insert_failed",
                        chunk_idx=chunk_idx,
                        batch_offset=i,
                        batch_size=len(sub_batch),
                        error=str(e)[:200],
                    )

            checkpoint["rows_inserted"] += chunk_inserted
            checkpoint["rows_failed"] += chunk_failed
            checkpoint["last_chunk_idx"] = chunk_idx
            _save_checkpoint(checkpoint)

            chunk_elapsed = time.time() - chunk_start
            total_elapsed = time.time() - pipeline_start
            rate = checkpoint["rows_inserted"] / total_elapsed if total_elapsed > 0 else 0

            print(
                f"  [Chunk {chunk_idx:>5}] "
                f"+{chunk_inserted:>6,} (skip {skipped:>4}, fail {chunk_failed:>3}) | "
                f"누적 inserted {checkpoint['rows_inserted']:>10,} | "
                f"속도 {rate:>7,.0f}건/s | "
                f"청크 {chunk_elapsed:>5.1f}s"
            )

        # ── 완료 ──
        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        after_count = await _count_table(pool)
        total_elapsed = time.time() - pipeline_start

        print(f"\n{'=' * 60}")
        print(f"[Kaggle ratings 적재 완료]")
        print(f"  처리 행:    {checkpoint['rows_processed']:>12,}")
        print(f"  INSERT 성공: {checkpoint['rows_inserted']:>12,}")
        print(f"  매핑 없음 skip: {checkpoint['rows_skipped']:>9,}")
        print(f"  실패:       {checkpoint['rows_failed']:>12,}")
        print(f"  최종 건수:  {after_count:>12,}")
        print(f"  소요:       {total_elapsed / 60:>12.1f} 분")
        print(f"{'=' * 60}")

    finally:
        pool.close()
        await pool.wait_closed()


# ══════════════════════════════════════════════════════════════
# 상태 조회
# ══════════════════════════════════════════════════════════════


async def show_status() -> None:
    """현재 체크포인트 + MySQL 건수 출력."""
    checkpoint = _load_checkpoint()

    print("=" * 60)
    print(f"  Kaggle ratings 적재 체크포인트")
    print("=" * 60)
    print(f"  단계:           {checkpoint.get('phase', '미시작')}")
    print(f"  처리 행:        {checkpoint.get('rows_processed', 0):>12,}")
    print(f"  INSERT 성공:    {checkpoint.get('rows_inserted', 0):>12,}")
    print(f"  매핑 없음 skip: {checkpoint.get('rows_skipped', 0):>12,}")
    print(f"  실패:           {checkpoint.get('rows_failed', 0):>12,}")
    print(f"  마지막 청크:    {checkpoint.get('last_chunk_idx', -1):>12}")
    print(f"  links 매핑 수:  {checkpoint.get('links_count', 0):>12,}")
    print(f"  마지막 갱신:    {checkpoint.get('last_updated', '-')}")
    print()

    # MySQL 라이브 카운트
    try:
        pool = await _create_pool()
        try:
            cnt = await _count_table(pool)
            print(f"  MySQL 라이브 건수: {cnt:>12,}")
        finally:
            pool.close()
            await pool.wait_closed()
    except Exception as e:
        print(f"  MySQL 연결 실패: {e}")

    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kaggle ratings.csv → MySQL kaggle_watch_history 적재",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 적재 (TRUNCATE → INSERT)
  PYTHONPATH=src uv run python scripts/run_kaggle_ratings_load.py

  # 중단점부터 재개
  PYTHONPATH=src uv run python scripts/run_kaggle_ratings_load.py --resume

  # 건수 확인만
  PYTHONPATH=src uv run python scripts/run_kaggle_ratings_load.py --status

  # TRUNCATE 없이 INSERT만
  PYTHONPATH=src uv run python scripts/run_kaggle_ratings_load.py --no-truncate

  # 배치/청크 크기 조정
  PYTHONPATH=src uv run python scripts/run_kaggle_ratings_load.py \\
      --chunk-size 100000 --batch-size 10000
        """,
    )
    parser.add_argument(
        "--kaggle-dir",
        type=str,
        default=str(DEFAULT_KAGGLE_DIR),
        help=f"Kaggle CSV 디렉토리 (기본: {DEFAULT_KAGGLE_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"pandas read_csv chunksize (기본: {DEFAULT_CHUNK_SIZE:,})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"MySQL executemany 배치 크기 (기본: {DEFAULT_BATCH_SIZE:,})",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="TRUNCATE 없이 INSERT 만 (기존 데이터 유지)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="체크포인트의 last_chunk_idx + 1 부터 재개 (TRUNCATE 자동 비활성)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="현재 체크포인트 + MySQL 건수만 확인 (적재 안 함)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.status:
        asyncio.run(show_status())
    else:
        asyncio.run(
            run_kaggle_ratings_load(
                kaggle_dir=Path(args.kaggle_dir),
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
                truncate=not args.no_truncate,
                resume=args.resume,
            )
        )
