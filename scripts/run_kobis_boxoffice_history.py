"""
KOBIS 일별 박스오피스 시계열 적재 (Phase ML §10 G-1).

KOBIS searchDailyBoxOfficeList API 를 사용하여 N일치 일별 박스오피스 Top-10 을
시계열 형태로 MySQL `box_office_daily` 테이블에 적재한다.

기존 `run_kobis_load.py` 의 `collect_boxoffice_history()` 는 영화별 최대 누적치
1건만 보존했음 → 시계열 손실. 본 스크립트는 (movie_cd, target_dt) 단위로 모두 보존.

설계 진실 원본:
    docs/데이터_적재_프로세스_전체분석_및_개선계획.md §10.3 (2. box_office_daily)
    db_dumps/prod_old_backup/init.sql §1C box_office_daily CREATE TABLE

핵심 정책:
    - **무료 API**: KOBIS searchDailyBoxOfficeList (회원가입만, 일일 한도 거의 없음)
    - **시계열 보존**: (movie_cd, target_dt) UNIQUE KEY 로 일별 추세 그대로 저장
    - **중복 제거**: ON DUPLICATE KEY UPDATE 로 재실행 멱등성 보장
    - **rate limit 보수적**: KOBISCollector 의 기본 RPS (10 req/sec) 사용
    - **체크포인트**: 마지막 처리 일자 저장 → 재개 시 그 이후만 수집
    - **movie_id 매핑**: kobis_movie_cd → movies.movie_id 자동 조인 (사후 UPDATE)

선결 조건:
    - .env KOBIS_API_KEY
    - MySQL `box_office_daily` 테이블 존재 (init.sql 수정 후 수동 실행)

성능 추정:
    - 1년 (365일) × 1초/호출 ≈ 6분
    - 5년 (1825일) × 1초/호출 ≈ 30분
    - 일별 Top-10 만 수집되므로 총 행 수: 365 × 10 = 3,650 (1년)

사용법:
    # 최근 365일 (어제 기준)
    PYTHONPATH=src uv run python scripts/run_kobis_boxoffice_history.py

    # 최근 5년 (1825일)
    PYTHONPATH=src uv run python scripts/run_kobis_boxoffice_history.py --days 1825

    # 특정 종료일 기준 1년
    PYTHONPATH=src uv run python scripts/run_kobis_boxoffice_history.py --end-date 20241231

    # 재개 (체크포인트 last_processed_date 다음날부터)
    PYTHONPATH=src uv run python scripts/run_kobis_boxoffice_history.py --resume

    # 일일 cron (어제만)
    PYTHONPATH=src uv run python scripts/run_kobis_boxoffice_history.py --days 1

    # 상태 확인
    PYTHONPATH=src uv run python scripts/run_kobis_boxoffice_history.py --status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
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

import aiomysql  # noqa: E402
import structlog  # noqa: E402

from monglepick.config import settings  # noqa: E402
from monglepick.data_pipeline.kobis_collector import KOBISCollector  # noqa: E402
from monglepick.data_pipeline.models import KOBISBoxOffice  # noqa: E402

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

CHECKPOINT_FILE = Path("data/kobis_boxoffice_history_checkpoint.json")
DEFAULT_DAYS = 365
DEFAULT_BATCH_SIZE = 500       # MySQL executemany 배치
DEFAULT_POOL_SIZE = 5
TABLE_NAME = "box_office_daily"


INSERT_SQL = f"""
INSERT INTO {TABLE_NAME} (
    movie_cd, movie_id, movie_nm, target_dt,
    rank_no, rank_inten, rank_old_and_new,
    audi_cnt, audi_acc, sales_amt, sales_acc,
    scrn_cnt, show_cnt, open_dt
) VALUES (
    %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s
) ON DUPLICATE KEY UPDATE
    movie_id        = COALESCE(VALUES(movie_id), movie_id),
    movie_nm        = VALUES(movie_nm),
    rank_no         = VALUES(rank_no),
    rank_inten      = VALUES(rank_inten),
    rank_old_and_new = VALUES(rank_old_and_new),
    audi_cnt        = VALUES(audi_cnt),
    audi_acc        = VALUES(audi_acc),
    sales_amt       = VALUES(sales_amt),
    sales_acc       = VALUES(sales_acc),
    scrn_cnt        = VALUES(scrn_cnt),
    show_cnt        = VALUES(show_cnt),
    open_dt         = VALUES(open_dt)
"""

COUNT_SQL = f"SELECT COUNT(*) FROM {TABLE_NAME}"
TABLE_EXISTS_SQL = (
    "SELECT COUNT(*) FROM information_schema.tables "
    "WHERE table_schema=%s AND table_name=%s"
)

# kobis_movie_cd → movies.movie_id 매핑 캐시 SQL
LOAD_MOVIE_ID_MAP_SQL = (
    "SELECT kobis_movie_cd, movie_id FROM movies "
    "WHERE kobis_movie_cd IS NOT NULL AND kobis_movie_cd <> ''"
)


# ══════════════════════════════════════════════════════════════
# 체크포인트
# ══════════════════════════════════════════════════════════════


def _new_checkpoint() -> dict:
    return {
        "phase": "",
        "last_processed_date": None,    # YYYYMMDD (수집 완료된 마지막 날짜)
        "total_api_calls": 0,
        "total_rows_collected": 0,      # KOBIS 응답 raw 건수
        "total_rows_inserted": 0,        # MySQL upsert 성공 건수
        "total_failed_dates": 0,
        "failed_dates": [],
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
# MySQL 헬퍼
# ══════════════════════════════════════════════════════════════


async def _create_pool() -> aiomysql.Pool:
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
    logger.info("mysql_pool_created")
    return pool


async def _verify_table_exists(pool: aiomysql.Pool) -> bool:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                TABLE_EXISTS_SQL,
                (settings.MYSQL_DATABASE, TABLE_NAME),
            )
            (cnt,) = await cur.fetchone()
            return cnt > 0


async def _count_table(pool: aiomysql.Pool) -> int:
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(COUNT_SQL)
            (cnt,) = await cur.fetchone()
            return cnt


async def _load_movie_id_map(pool: aiomysql.Pool) -> dict[str, str]:
    """movies 테이블에서 (kobis_movie_cd → movie_id) 매핑 로드."""
    mapping: dict[str, str] = {}
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(LOAD_MOVIE_ID_MAP_SQL)
            rows = await cur.fetchall()
            for movie_cd, movie_id in rows:
                if movie_cd and movie_id:
                    mapping[str(movie_cd)] = str(movie_id)
    return mapping


async def _insert_batch(pool: aiomysql.Pool, rows: list[tuple]) -> int:
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
# KOBIS 응답 → MySQL 행 튜플
# ══════════════════════════════════════════════════════════════


def _box_to_row(
    box: KOBISBoxOffice,
    target_date_str: str,    # YYYYMMDD
    movie_id_map: dict[str, str],
) -> tuple:
    """KOBISBoxOffice → INSERT 행 튜플. movie_cd 매핑으로 movie_id 채움."""
    # YYYYMMDD → YYYY-MM-DD
    target_dt_iso = f"{target_date_str[:4]}-{target_date_str[4:6]}-{target_date_str[6:8]}"
    movie_id = movie_id_map.get(box.movie_cd)

    return (
        box.movie_cd,
        movie_id,
        (box.movie_nm or "")[:300],
        target_dt_iso,
        box.rank,
        box.rank_inten,
        box.rank_old_and_new or "OLD",
        box.audi_cnt,
        box.audi_acc,
        box.sales_amt,
        box.sales_acc,
        box.scrn_cnt,
        box.show_cnt,
        box.open_dt or None,
    )


# ══════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════


async def run_kobis_boxoffice_history(
    days: int = DEFAULT_DAYS,
    end_date: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = False,
) -> None:
    """
    KOBIS 일별 박스오피스 시계열 → MySQL box_office_daily 적재.

    Args:
        days: 수집할 일수 (기본 365)
        end_date: 종료 날짜 YYYYMMDD (기본 어제)
        batch_size: MySQL executemany 배치
        resume: 체크포인트의 last_processed_date 다음날부터 재개
    """
    pipeline_start = time.time()

    if not settings.KOBIS_API_KEY:
        print("[ERROR] KOBIS_API_KEY 가 .env 에 설정되지 않았습니다.")
        return

    # 종료일 결정 (기본: 어제)
    if not end_date:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=days - 1)

    # 체크포인트 로드 + 재개 시작일 보정
    checkpoint = _load_checkpoint() if resume else _new_checkpoint()
    if resume and checkpoint.get("last_processed_date"):
        # last_processed_date 다음날부터 시작
        last_dt = datetime.strptime(checkpoint["last_processed_date"], "%Y%m%d")
        resume_start_dt = last_dt + timedelta(days=1)
        if resume_start_dt > start_dt:
            start_dt = resume_start_dt
        print(f"[RESUME] 체크포인트 ({checkpoint['last_processed_date']}) → "
              f"재개 시작일: {start_dt.strftime('%Y%m%d')}")

    print(f"[Step 1] MySQL 풀 + box_office_daily 테이블 검증")
    pool = await _create_pool()

    try:
        if not await _verify_table_exists(pool):
            print(
                f"[ERROR] {TABLE_NAME} 테이블이 존재하지 않습니다.\n"
                f"        먼저 init.sql 을 실행하여 테이블을 생성하세요."
            )
            return

        before_count = await _count_table(pool)
        print(f"  현재 {TABLE_NAME} 건수: {before_count:,}")

        # movie_id 매핑 로드
        print(f"\n[Step 2] movies 테이블에서 (kobis_movie_cd → movie_id) 매핑 로드")
        movie_id_map = await _load_movie_id_map(pool)
        print(f"  매핑 수: {len(movie_id_map):,}")

        # ── KOBIS 일별 박스오피스 수집 ──
        total_days = (end_dt - start_dt).days + 1
        print(f"\n[Step 3] KOBIS 박스오피스 시계열 수집 ({total_days} 일)")
        print(f"  범위: {start_dt.strftime('%Y%m%d')} ~ {end_dt.strftime('%Y%m%d')}")
        print()

        checkpoint["phase"] = "collecting"

        batch_buffer: list[tuple] = []

        async with KOBISCollector() as collector:
            for day_offset in range(total_days):
                target_dt = start_dt + timedelta(days=day_offset)
                target_date_str = target_dt.strftime("%Y%m%d")

                try:
                    daily = await collector.collect_daily_boxoffice(target_date_str)
                    checkpoint["total_api_calls"] += 1
                    checkpoint["total_rows_collected"] += len(daily)
                except Exception as e:
                    logger.error(
                        "kobis_daily_collect_failed",
                        date=target_date_str,
                        error=str(e)[:200],
                    )
                    checkpoint["total_failed_dates"] += 1
                    checkpoint["failed_dates"].append(target_date_str)
                    continue

                # 응답 → MySQL 행 변환
                for box in daily:
                    if not box.movie_cd:
                        continue
                    row = _box_to_row(box, target_date_str, movie_id_map)
                    batch_buffer.append(row)

                # 배치가 가득 차면 INSERT
                if len(batch_buffer) >= batch_size:
                    try:
                        inserted = await _insert_batch(pool, batch_buffer)
                        checkpoint["total_rows_inserted"] += inserted
                    except Exception as e:
                        logger.error(
                            "kobis_box_batch_insert_failed",
                            batch_size=len(batch_buffer),
                            error=str(e)[:200],
                        )
                    batch_buffer = []

                # 마지막 처리 일자 갱신
                checkpoint["last_processed_date"] = target_date_str

                # 30일마다 체크포인트 + 진행률 출력
                if (day_offset + 1) % 30 == 0 or (day_offset + 1) == total_days:
                    _save_checkpoint(checkpoint)
                    elapsed = time.time() - pipeline_start
                    print(
                        f"  [Day {day_offset + 1:>4}/{total_days}] "
                        f"date {target_date_str} | "
                        f"calls {checkpoint['total_api_calls']:>5,} | "
                        f"collected {checkpoint['total_rows_collected']:>7,} | "
                        f"inserted {checkpoint['total_rows_inserted']:>7,} | "
                        f"failed_dates {checkpoint['total_failed_dates']:>3,} | "
                        f"elapsed {elapsed:>5.0f}s"
                    )

        # 마지막 남은 배치
        if batch_buffer:
            try:
                inserted = await _insert_batch(pool, batch_buffer)
                checkpoint["total_rows_inserted"] += inserted
            except Exception as e:
                logger.error(
                    "kobis_box_final_batch_failed",
                    batch_size=len(batch_buffer),
                    error=str(e)[:200],
                )

        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        after_count = await _count_table(pool)
        total_elapsed = time.time() - pipeline_start

        print(f"\n{'=' * 60}")
        print(f"[KOBIS 박스오피스 시계열 적재 완료]")
        print(f"  수집 일수:       {total_days:>10,}")
        print(f"  API 호출:        {checkpoint['total_api_calls']:>10,}")
        print(f"  raw 행:          {checkpoint['total_rows_collected']:>10,}")
        print(f"  inserted:        {checkpoint['total_rows_inserted']:>10,}")
        print(f"  failed_dates:    {checkpoint['total_failed_dates']:>10,}")
        print(f"  before:          {before_count:>10,}")
        print(f"  after:           {after_count:>10,}")
        print(f"  diff:            {after_count - before_count:>10,}")
        print(f"  소요:            {total_elapsed / 60:>10.1f} 분")
        print(f"{'=' * 60}")

    finally:
        pool.close()
        await pool.wait_closed()


# ══════════════════════════════════════════════════════════════
# 상태 조회
# ══════════════════════════════════════════════════════════════


async def show_status() -> None:
    cp = _load_checkpoint()
    print("=" * 60)
    print(f"  KOBIS Box Office History 체크포인트")
    print("=" * 60)
    print(f"  단계:                {cp.get('phase', '미시작')}")
    print(f"  마지막 처리 일자:    {cp.get('last_processed_date', '-')}")
    print(f"  API 호출:            {cp.get('total_api_calls', 0):>10,}")
    print(f"  raw 행:              {cp.get('total_rows_collected', 0):>10,}")
    print(f"  inserted:            {cp.get('total_rows_inserted', 0):>10,}")
    print(f"  failed_dates:        {cp.get('total_failed_dates', 0):>10,}")
    print(f"  마지막 갱신:         {cp.get('last_updated', '-')}")
    print()

    try:
        pool = await _create_pool()
        try:
            cnt = await _count_table(pool)
            print(f"  MySQL {TABLE_NAME} 라이브 건수: {cnt:>10,}")
        finally:
            pool.close()
            await pool.wait_closed()
    except Exception as e:
        print(f"  MySQL 조회 실패: {e}")

    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KOBIS 일별 박스오피스 시계열 → MySQL box_office_daily 적재",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 최근 365일
  PYTHONPATH=src uv run python scripts/run_kobis_boxoffice_history.py

  # 최근 5년
  PYTHONPATH=src uv run python scripts/run_kobis_boxoffice_history.py --days 1825

  # 일일 cron (어제만)
  PYTHONPATH=src uv run python scripts/run_kobis_boxoffice_history.py --days 1

  # 재개
  PYTHONPATH=src uv run python scripts/run_kobis_boxoffice_history.py --resume

  # 상태
  PYTHONPATH=src uv run python scripts/run_kobis_boxoffice_history.py --status
        """,
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"수집할 일수 (기본 {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="종료 날짜 YYYYMMDD (기본 어제)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"executemany 배치 크기 (기본 {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="체크포인트 last_processed_date 다음날부터 재개",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="현재 체크포인트 + MySQL 건수만 출력",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.status:
        asyncio.run(show_status())
    else:
        asyncio.run(
            run_kobis_boxoffice_history(
                days=args.days,
                end_date=args.end_date,
                batch_size=args.batch_size,
                resume=args.resume,
            )
        )
