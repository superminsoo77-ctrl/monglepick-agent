"""
Person → MySQL persons 적재 스크립트 (Phase ML §9.5 Phase 1 — C-6).

data/tmdb_persons/tmdb_persons.jsonl (TMDB Person + LLM 보강 결과) 를
MySQL `persons` 테이블에 ON DUPLICATE KEY UPDATE 로 적재한다.

설계 진실 원본:
    docs/데이터_적재_프로세스_전체분석_및_개선계획.md §10.3 (1. persons 테이블)
    db_dumps/prod_old_backup/init.sql §1B persons CREATE TABLE

선결 조건:
    - run_tmdb_persons_collect.py 완료 (data/tmdb_persons/tmdb_persons.jsonl 존재)
    - (선택) run_persons_full_pipeline.py 완료 — JSONL 에 LLM 보강 필드 (llm_*) 포함
      LLM 보강 안 된 JSONL 도 처리 가능 (해당 컬럼 NULL)
    - MySQL `persons` 테이블 존재 (init.sql 수정 후 수동 실행)
    - .env MYSQL_*

중복 제거 정책 (사용자 요구사항):
    - PK person_id 기준 ON DUPLICATE KEY UPDATE → 같은 person_id 는 1건만 존재
    - 같은 JSONL 라인이 여러 번 있어도 마지막 값으로 덮어쓰기
    - 청크 INSERT 실패 시 청크 단위 롤백 + 로그

성능 추정:
    - 572K Person × executemany 5,000 건 배치 → 약 115 배치
    - 배치당 ~1초 (LONGTEXT biography 포함 시 2~3초)
    - 총 5~10분

사용법:
    # 전체 적재
    PYTHONPATH=src uv run python scripts/run_persons_mysql_sync.py

    # 처음 N명만 (테스트)
    PYTHONPATH=src uv run python scripts/run_persons_mysql_sync.py --limit 100

    # 재개 (체크포인트 last_jsonl_line 부터)
    PYTHONPATH=src uv run python scripts/run_persons_mysql_sync.py --resume

    # 상태 확인
    PYTHONPATH=src uv run python scripts/run_persons_mysql_sync.py --status
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

import aiomysql  # noqa: E402
import structlog  # noqa: E402

from monglepick.config import settings  # noqa: E402

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

INPUT_JSONL = Path("data/tmdb_persons/tmdb_persons.jsonl")
CHECKPOINT_FILE = Path("data/tmdb_persons/mysql_sync_checkpoint.json")
DEFAULT_BATCH_SIZE = 5_000
DEFAULT_POOL_SIZE = 5
TABLE_NAME = "persons"


#: persons 테이블 ON DUPLICATE KEY UPDATE INSERT (init.sql 정의와 일치)
INSERT_SQL = f"""
INSERT INTO {TABLE_NAME} (
    person_id, name, original_name, profile_path,
    biography_ko, biography_en,
    place_of_birth, birthday, deathday,
    gender, known_for_department, popularity,
    imdb_id, homepage,
    style_tags, persona, top_movies, external_ids,
    source
) VALUES (
    %s, %s, %s, %s,
    %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s, %s,
    %s, %s, %s, %s,
    %s
) ON DUPLICATE KEY UPDATE
    name                = VALUES(name),
    original_name       = VALUES(original_name),
    profile_path        = VALUES(profile_path),
    biography_ko        = VALUES(biography_ko),
    biography_en        = VALUES(biography_en),
    place_of_birth      = VALUES(place_of_birth),
    birthday            = VALUES(birthday),
    deathday            = VALUES(deathday),
    gender              = VALUES(gender),
    known_for_department = VALUES(known_for_department),
    popularity          = VALUES(popularity),
    imdb_id             = VALUES(imdb_id),
    homepage            = VALUES(homepage),
    style_tags          = VALUES(style_tags),
    persona             = VALUES(persona),
    top_movies          = VALUES(top_movies),
    external_ids        = VALUES(external_ids),
    source              = VALUES(source)
"""

COUNT_SQL = f"SELECT COUNT(*) FROM {TABLE_NAME}"
TABLE_EXISTS_SQL = (
    "SELECT COUNT(*) FROM information_schema.tables "
    "WHERE table_schema=%s AND table_name=%s"
)


# ══════════════════════════════════════════════════════════════
# 체크포인트
# ══════════════════════════════════════════════════════════════


def _new_checkpoint() -> dict:
    return {
        "phase": "",
        "last_jsonl_line": 0,
        "total_processed": 0,
        "total_inserted": 0,    # ON DUPLICATE 로 갱신된 것 포함
        "total_failed": 0,
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
# Person dict → MySQL 행 튜플
# ══════════════════════════════════════════════════════════════


def _parse_date(s: str | None) -> str | None:
    """YYYY-MM-DD 형식 문자열만 통과시킨다 (MySQL DATE 호환)."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if len(s) < 10:
        return None
    # 간단 형식 검증 (YYYY-MM-DD)
    try:
        datetime.strptime(s[:10], "%Y-%m-%d")
        return s[:10]
    except ValueError:
        return None


def _person_to_row(person: dict) -> tuple | None:
    """
    TMDB Person dict (LLM 보강 가능) → INSERT 행 튜플.

    JSON 컬럼은 json.dumps(ensure_ascii=False) 로 직렬화.
    LLM 보강 필드 (llm_*) 가 없으면 NULL 로 채움.

    Returns:
        tuple | None — id 가 없거나 잘못되면 None
    """
    pid = person.get("id")
    if not pid:
        return None

    try:
        person_id = int(pid)
    except (ValueError, TypeError):
        return None

    name = (person.get("name") or "").strip()
    if not name:
        return None

    # also_known_as 첫 번째를 original_name 으로
    aka = person.get("also_known_as") or []
    original_name = (aka[0] or "").strip()[:200] if aka else None

    profile_path = (person.get("profile_path") or "")[:500] or None
    biography_en = person.get("biography") or None
    place_of_birth = (person.get("place_of_birth") or "")[:200] or None
    birthday = _parse_date(person.get("birthday"))
    deathday = _parse_date(person.get("deathday"))
    gender = int(person.get("gender") or 0)
    known_for = (person.get("known_for_department") or "")[:50] or None
    popularity = float(person.get("popularity") or 0.0)
    imdb_id = (person.get("imdb_id") or "")[:20] or None
    homepage = (person.get("homepage") or "")[:500] or None

    # external_ids: 다양한 형태 가능
    ext_ids = person.get("external_ids") or {}
    external_ids_json = (
        json.dumps(ext_ids, ensure_ascii=False)
        if isinstance(ext_ids, dict) and ext_ids
        else None
    )

    # LLM 보강 필드 (없으면 NULL)
    biography_ko = (person.get("llm_biography_ko") or "").strip() or None

    style_tags = person.get("llm_style_tags") or []
    style_tags_json = (
        json.dumps(style_tags, ensure_ascii=False)
        if isinstance(style_tags, list) and style_tags
        else None
    )

    persona = (person.get("llm_persona") or "").strip()[:150] or None

    top_movies = person.get("llm_top_movies") or []
    top_movies_json = (
        json.dumps(top_movies, ensure_ascii=False)
        if isinstance(top_movies, list) and top_movies
        else None
    )

    source = (person.get("source") or "tmdb")[:20]

    return (
        person_id,
        name[:200],
        original_name,
        profile_path,
        biography_ko,
        biography_en,
        place_of_birth,
        birthday,
        deathday,
        gender,
        known_for,
        popularity,
        imdb_id,
        homepage,
        style_tags_json,
        persona,
        top_movies_json,
        external_ids_json,
        source,
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
    logger.info("mysql_pool_created", db=settings.MYSQL_DATABASE)
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
# JSONL 스트리밍
# ══════════════════════════════════════════════════════════════


def _jsonl_chunks(path: Path, batch_size: int, skip_lines: int = 0):
    """JSONL → batch_size 단위로 (rows, last_line_no) yield."""
    rows: list[tuple] = []
    line_no = 0
    skipped_invalid = 0

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
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue

            row = _person_to_row(obj)
            if row is None:
                skipped_invalid += 1
                continue

            rows.append(row)

            if len(rows) >= batch_size:
                yield rows, line_no, skipped_invalid
                rows = []
                skipped_invalid = 0

    if rows:
        yield rows, line_no, skipped_invalid


# ══════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════


async def run_persons_mysql_sync(
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    resume: bool = False,
) -> None:
    pipeline_start = time.time()

    if not INPUT_JSONL.exists():
        print(f"[ERROR] JSONL 파일이 없습니다: {INPUT_JSONL}")
        print(f"        먼저 run_tmdb_persons_collect.py 를 실행하세요.")
        return

    checkpoint = _load_checkpoint() if resume else _new_checkpoint()
    skip_lines = checkpoint.get("last_jsonl_line", 0) if resume else 0

    print(f"[Step 1] MySQL 풀 + persons 테이블 확인")
    pool = await _create_pool()

    try:
        if not await _verify_table_exists(pool):
            print(
                f"[ERROR] {TABLE_NAME} 테이블이 존재하지 않습니다.\n"
                f"        먼저 init.sql 을 실행하여 persons 테이블을 생성하세요:\n"
                f"        docker exec -i monglepick-mysql mysql -u {settings.MYSQL_USER} -p<pw> "
                f"{settings.MYSQL_DATABASE} < db_dumps/prod_old_backup/init.sql"
            )
            return

        before_count = await _count_table(pool)
        print(f"  현재 persons 건수: {before_count:,}")

        print(f"\n[Step 2] JSONL → persons 적재 (ON DUPLICATE KEY UPDATE)")
        print(f"  JSONL: {INPUT_JSONL}")
        print(f"  batch_size: {batch_size:,}, skip_lines: {skip_lines:,}")
        if limit:
            print(f"  limit: {limit:,}")
        print()

        checkpoint["phase"] = "loading"

        for batch_rows, last_line, invalid_count in _jsonl_chunks(
            INPUT_JSONL, batch_size, skip_lines=skip_lines
        ):
            chunk_start = time.time()

            try:
                inserted = await _insert_batch(pool, batch_rows)
                checkpoint["total_inserted"] += inserted
            except Exception as e:
                logger.error(
                    "persons_batch_insert_failed",
                    last_line=last_line,
                    batch_size=len(batch_rows),
                    error=str(e)[:200],
                )
                checkpoint["total_failed"] += len(batch_rows)

            checkpoint["total_processed"] += len(batch_rows) + invalid_count
            checkpoint["last_jsonl_line"] = last_line
            _save_checkpoint(checkpoint)

            chunk_elapsed = time.time() - chunk_start
            total_elapsed = time.time() - pipeline_start
            rate = checkpoint["total_inserted"] / total_elapsed if total_elapsed > 0 else 0

            print(
                f"  [Batch] +{len(batch_rows):>5} (invalid {invalid_count:>3}) | "
                f"누적 inserted {checkpoint['total_inserted']:>8,} | "
                f"속도 {rate:>6,.0f}/s | "
                f"청크 {chunk_elapsed:>5.1f}s | "
                f"line {last_line:>9,}"
            )

            if limit and checkpoint["total_inserted"] >= limit:
                print(f"  --limit {limit} 도달 → 중단")
                break

        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        after_count = await _count_table(pool)
        total_elapsed = time.time() - pipeline_start

        print(f"\n{'=' * 60}")
        print(f"[persons MySQL sync 완료]")
        print(f"  처리:        {checkpoint['total_processed']:>10,}")
        print(f"  inserted:    {checkpoint['total_inserted']:>10,}")
        print(f"  failed:      {checkpoint['total_failed']:>10,}")
        print(f"  before:      {before_count:>10,}")
        print(f"  after:       {after_count:>10,}")
        print(f"  diff:        {after_count - before_count:>10,}")
        print(f"  소요:        {total_elapsed / 60:>10.1f} 분")
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
    print(f"  Person MySQL Sync 체크포인트")
    print("=" * 60)
    print(f"  단계:           {cp.get('phase', '미시작')}")
    print(f"  마지막 라인:    {cp.get('last_jsonl_line', 0):>10,}")
    print(f"  처리:           {cp.get('total_processed', 0):>10,}")
    print(f"  inserted:       {cp.get('total_inserted', 0):>10,}")
    print(f"  failed:         {cp.get('total_failed', 0):>10,}")
    print(f"  마지막 갱신:    {cp.get('last_updated', '-')}")
    print()

    try:
        pool = await _create_pool()
        try:
            cnt = await _count_table(pool)
            print(f"  MySQL persons 라이브 건수: {cnt:>10,}")
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
        description="Person JSONL → MySQL persons 적재",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"executemany 배치 크기 (기본 {DEFAULT_BATCH_SIZE:,})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="최대 적재 건수 (테스트)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="체크포인트 last_jsonl_line 부터 재개",
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
            run_persons_mysql_sync(
                batch_size=args.batch_size,
                limit=args.limit,
                resume=args.resume,
            )
        )
