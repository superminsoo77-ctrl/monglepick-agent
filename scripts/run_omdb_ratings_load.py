"""
OMDb 외부 평점 → MySQL movie_external_ratings 적재 (Phase ML §10 G-2).

OMDb API (http://www.omdbapi.com/) 로 IMDb / Rotten Tomatoes / Metacritic 점수를
보강하여 MySQL `movie_external_ratings` 테이블에 적재한다.

설계 진실 원본:
    docs/데이터_적재_프로세스_전체분석_및_개선계획.md §10.3 (3. movie_external_ratings)
    db_dumps/prod_old_backup/init.sql §1D movie_external_ratings CREATE TABLE

핵심 정책 — **무료 한도 1,000/day 우회 전략**:
    1. **인기 영화 우선**: Qdrant payload 의 popularity_score DESC 정렬
    2. **이미 적재된 영화 skip**: movie_external_ratings.movie_id UNIQUE 활용
    3. **하루 한도 자동 추적**: --max-calls 옵션 (기본 950, 안전 마진)
    4. **일일 cron 친화**: 매일 950건씩 → 1년에 약 35만 건 적재 가능
    5. **재실행 멱등**: ON DUPLICATE KEY UPDATE
    6. **체크포인트**: 마지막 처리 popularity 임계값 + 처리한 movie_id 셋

선결 조건:
    - .env OMDB_API_KEY (https://www.omdbapi.com/apikey.aspx 무료 가입)
    - MySQL `movie_external_ratings` 테이블 존재 (init.sql 실행 후)
    - movies 테이블에 imdb_id 가 있는 영화만 처리 (OMDb 는 imdb_id 로 조회)

사용법:
    # 일일 cron (950건/일, 인기순)
    PYTHONPATH=src uv run python scripts/run_omdb_ratings_load.py

    # 처음 100건만 (테스트)
    PYTHONPATH=src uv run python scripts/run_omdb_ratings_load.py --max-calls 100

    # 재개 (이미 적재된 movie_id skip)
    PYTHONPATH=src uv run python scripts/run_omdb_ratings_load.py --resume

    # 상태
    PYTHONPATH=src uv run python scripts/run_omdb_ratings_load.py --status

운영 권장:
    - macOS launchd / Linux cron 으로 매일 새벽 1회 실행
    - 35만 건 도달까지 약 12개월
    - 도달 후 인기 변동에 따라 갱신 위주
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
import httpx  # noqa: E402
import structlog  # noqa: E402

from monglepick.config import settings  # noqa: E402
from monglepick.db.clients import init_all_clients, close_all_clients, get_qdrant  # noqa: E402

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

OMDB_BASE_URL = "http://www.omdbapi.com/"
CHECKPOINT_FILE = Path("data/omdb_ratings_checkpoint.json")
DEFAULT_MAX_CALLS = 950           # 무료 1,000/day 안전 마진
DEFAULT_BATCH_INSERT = 50          # MySQL executemany 배치
DEFAULT_RPS = 5                    # OMDb 무료 RPS (서버 친화적)
DEFAULT_TIMEOUT = 10.0
TABLE_NAME = "movie_external_ratings"


INSERT_SQL = f"""
INSERT INTO {TABLE_NAME} (
    movie_id, imdb_id, imdb_rating, imdb_votes,
    rotten_tomatoes_score, metacritic_score,
    awards, box_office, rated, runtime_omdb, fetched_at
) VALUES (
    %s, %s, %s, %s,
    %s, %s,
    %s, %s, %s, %s, %s
) ON DUPLICATE KEY UPDATE
    imdb_id              = VALUES(imdb_id),
    imdb_rating          = VALUES(imdb_rating),
    imdb_votes           = VALUES(imdb_votes),
    rotten_tomatoes_score = VALUES(rotten_tomatoes_score),
    metacritic_score     = VALUES(metacritic_score),
    awards               = VALUES(awards),
    box_office           = VALUES(box_office),
    rated                = VALUES(rated),
    runtime_omdb         = VALUES(runtime_omdb),
    fetched_at           = VALUES(fetched_at)
"""

COUNT_SQL = f"SELECT COUNT(*) FROM {TABLE_NAME}"
EXISTING_IDS_SQL = f"SELECT movie_id FROM {TABLE_NAME}"
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
        "total_api_calls": 0,
        "total_inserted": 0,
        "total_skipped_no_imdb": 0,    # imdb_id 없어서 skip
        "total_skipped_existing": 0,    # 이미 DB 에 있어서 skip
        "total_failed": 0,              # 호출 실패 / 응답 에러
        "last_processed_movie_id": None,
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
# Rate Limiter
# ══════════════════════════════════════════════════════════════


class _RateLimiter:
    """초당 N건 제한 (간단 토큰 버킷)."""

    def __init__(self, rps: int):
        self.rps = rps
        self.last_call = 0.0
        self.lock = asyncio.Lock()
        self.min_interval = 1.0 / rps

    async def acquire(self) -> None:
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_call = time.monotonic()


# ══════════════════════════════════════════════════════════════
# OMDb API 호출
# ══════════════════════════════════════════════════════════════


async def _call_omdb(
    client: httpx.AsyncClient,
    api_key: str,
    imdb_id: str,
    rate_limiter: _RateLimiter,
    max_retries: int = 3,
) -> dict | None:
    """
    OMDb API 단건 호출 (imdb_id 기준).

    Returns:
        dict | None — 응답 비어있거나 Response='False' 면 None
    """
    for attempt in range(max_retries):
        await rate_limiter.acquire()
        try:
            response = await client.get(
                OMDB_BASE_URL,
                params={"apikey": api_key, "i": imdb_id},
            )
        except httpx.HTTPError as e:
            logger.warning(
                "omdb_network_error",
                imdb_id=imdb_id,
                attempt=attempt + 1,
                error=str(e)[:200],
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            try:
                data = response.json()
            except Exception as e:
                logger.warning("omdb_json_decode_error", imdb_id=imdb_id, error=str(e))
                return None
            # OMDb 는 에러도 200 + Response: 'False'
            if data.get("Response") == "True":
                return data
            else:
                # "Movie not found!" 등 — 정상적인 빈 응답
                return None

        if response.status_code == 401:
            # API 키 오류 — 재시도 불가
            logger.error("omdb_unauthorized", status=response.status_code)
            return None

        if response.status_code == 429:
            # rate limit hit
            logger.warning("omdb_rate_limit", attempt=attempt + 1)
            await asyncio.sleep(60)
            continue

        # 5xx
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)

    return None


def _parse_omdb_response(data: dict, movie_id: str) -> tuple:
    """
    OMDb 응답 dict → INSERT 행 튜플.

    OMDb 응답 예시:
        {
          "Title": "Inception", "imdbID": "tt1375666",
          "imdbRating": "8.8", "imdbVotes": "2,500,000",
          "Ratings": [{"Source":"Internet Movie Database","Value":"8.8/10"},
                      {"Source":"Rotten Tomatoes","Value":"87%"},
                      {"Source":"Metacritic","Value":"74/100"}],
          "Awards": "Won 4 Oscars. ...", "BoxOffice": "$292,587,330",
          "Rated": "PG-13", "Runtime": "148 min"
        }
    """
    imdb_id = data.get("imdbID") or None

    # IMDb 평점
    imdb_rating = None
    raw_imdb = data.get("imdbRating")
    if raw_imdb and raw_imdb != "N/A":
        try:
            imdb_rating = float(raw_imdb)
        except (ValueError, TypeError):
            pass

    # IMDb votes
    imdb_votes = None
    raw_votes = data.get("imdbVotes")
    if raw_votes and raw_votes != "N/A":
        try:
            imdb_votes = int(raw_votes.replace(",", ""))
        except (ValueError, TypeError):
            pass

    # Ratings 배열에서 RT/Metacritic 추출
    rt_score = None
    meta_score = None
    for r in (data.get("Ratings") or []):
        source = (r.get("Source") or "").strip()
        value = (r.get("Value") or "").strip()
        if source == "Rotten Tomatoes" and value:
            try:
                rt_score = int(value.rstrip("%"))
            except (ValueError, TypeError):
                pass
        elif source == "Metacritic" and value:
            try:
                # "74/100" → 74
                meta_score = int(value.split("/")[0])
            except (ValueError, IndexError, TypeError):
                pass

    awards = (data.get("Awards") or "").strip() or None
    if awards == "N/A":
        awards = None

    box_office = (data.get("BoxOffice") or "").strip() or None
    if box_office == "N/A":
        box_office = None

    rated = (data.get("Rated") or "").strip() or None
    if rated == "N/A":
        rated = None

    runtime_omdb = (data.get("Runtime") or "").strip() or None
    if runtime_omdb == "N/A":
        runtime_omdb = None

    return (
        movie_id,
        imdb_id,
        imdb_rating,
        imdb_votes,
        rt_score,
        meta_score,
        awards,
        box_office[:50] if box_office else None,
        rated[:20] if rated else None,
        runtime_omdb[:20] if runtime_omdb else None,
        datetime.now(),
    )


# ══════════════════════════════════════════════════════════════
# Qdrant 에서 대상 영화 추출 (popularity 내림차순)
# ══════════════════════════════════════════════════════════════


async def _fetch_target_movies(
    skip_movie_ids: set[str],
    max_count: int,
) -> list[tuple[str, str, float]]:
    """
    Qdrant `movies` 컬렉션에서 OMDb 적재 대상 영화 추출.

    필터:
        - imdb_id 가 있어야 함 (OMDb 는 imdb_id 로만 조회)
        - skip_movie_ids 에 없어야 함 (이미 적재된 영화 제외)

    정렬:
        - popularity_score DESC (인기 영화 우선)

    Returns:
        list[(movie_id, imdb_id, popularity)] — 최대 max_count
    """
    client = await get_qdrant()
    candidates: list[tuple[str, str, float]] = []
    offset = None

    print(f"  Qdrant scroll (popularity desc, imdb_id 있는 영화만)")

    # popularity DESC 로 scroll 하기는 query_points 가 더 효율적이지만
    # 여기서는 단순 scroll 후 메모리에서 정렬 (1.18M 메타만이라 가벼움)
    all_with_imdb: list[tuple[str, str, float]] = []

    while True:
        result = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=5000,
            offset=offset,
            with_vectors=False,
            with_payload=["id", "imdb_id", "popularity_score"],
        )
        points = result[0]
        next_offset = result[1]

        if not points:
            break

        for p in points:
            payload = p.payload or {}
            movie_id = str(payload.get("id") or p.id)
            imdb_id = (payload.get("imdb_id") or "").strip()
            popularity = float(payload.get("popularity_score") or 0.0)

            if not imdb_id:
                continue
            if movie_id in skip_movie_ids:
                continue

            all_with_imdb.append((movie_id, imdb_id, popularity))

        if next_offset is None:
            break
        offset = next_offset

    # popularity 내림차순 정렬
    all_with_imdb.sort(key=lambda x: x[2], reverse=True)
    candidates = all_with_imdb[:max_count]

    print(f"  imdb_id 있는 영화: {len(all_with_imdb):,} / 처리 대상: {len(candidates):,}")
    return candidates


# ══════════════════════════════════════════════════════════════
# MySQL 헬퍼
# ══════════════════════════════════════════════════════════════


async def _create_pool() -> aiomysql.Pool:
    return await aiomysql.create_pool(
        host=settings.MYSQL_HOST,
        port=int(getattr(settings, "MYSQL_PORT", 3306)),
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        db=settings.MYSQL_DATABASE,
        minsize=1,
        maxsize=5,
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=10,
    )


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


async def _load_existing_movie_ids(pool: aiomysql.Pool) -> set[str]:
    """이미 movie_external_ratings 에 있는 movie_id 셋 (재실행 skip)."""
    ids: set[str] = set()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(EXISTING_IDS_SQL)
            rows = await cur.fetchall()
            for (mid,) in rows:
                ids.add(str(mid))
    return ids


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
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════


async def run_omdb_ratings_load(
    max_calls: int = DEFAULT_MAX_CALLS,
    rps: int = DEFAULT_RPS,
    batch_insert: int = DEFAULT_BATCH_INSERT,
) -> None:
    """
    OMDb → MySQL movie_external_ratings 적재.

    Args:
        max_calls: 이번 실행에서 최대 OMDb API 호출 수 (무료 1,000/day 안전 마진)
        rps: 초당 호출 수
        batch_insert: MySQL executemany 배치
    """
    pipeline_start = time.time()

    omdb_api_key = os.environ.get("OMDB_API_KEY") or getattr(settings, "OMDB_API_KEY", None)
    if not omdb_api_key:
        print("[ERROR] OMDB_API_KEY 가 .env 에 설정되지 않았습니다.")
        print("        무료 가입: https://www.omdbapi.com/apikey.aspx")
        return

    print(f"[Step 0] DB 클라이언트 + MySQL 풀")
    await init_all_clients()
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

        # ── 1. 기존 적재 ID 로드 (중복 제거) ──
        print(f"\n[Step 1] 기존 적재 movie_id 로드 (skip 대상)")
        existing_ids = await _load_existing_movie_ids(pool)
        print(f"  기존 적재: {len(existing_ids):,}")

        # ── 2. Qdrant 에서 대상 영화 추출 (popularity desc, max_calls 만큼) ──
        print(f"\n[Step 2] Qdrant 에서 인기 영화 후보 추출 (최대 {max_calls:,}건)")
        candidates = await _fetch_target_movies(
            skip_movie_ids=existing_ids,
            max_count=max_calls,
        )

        if not candidates:
            print("  처리할 영화가 없습니다 (모두 적재됨 또는 imdb_id 없음).")
            return

        # ── 3. OMDb API 호출 + 적재 ──
        print(f"\n[Step 3] OMDb API 호출 + INSERT")
        print(f"  rps: {rps}, batch_insert: {batch_insert}")
        print(f"  예상 소요: ~{len(candidates) / rps / 60:.1f} 분")
        print()

        checkpoint = _load_checkpoint()
        checkpoint["phase"] = "fetching"
        rate_limiter = _RateLimiter(rps)

        batch_buffer: list[tuple] = []
        api_calls_in_run = 0

        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as http_client:
            for idx, (movie_id, imdb_id, popularity) in enumerate(candidates):
                # imdb_id 없으면 skip (방어적, _fetch 단계에서 이미 필터)
                if not imdb_id.startswith("tt"):
                    checkpoint["total_skipped_no_imdb"] += 1
                    continue

                # OMDb 호출
                data = await _call_omdb(
                    client=http_client,
                    api_key=omdb_api_key,
                    imdb_id=imdb_id,
                    rate_limiter=rate_limiter,
                )
                api_calls_in_run += 1
                checkpoint["total_api_calls"] += 1

                if not data:
                    checkpoint["total_failed"] += 1
                    continue

                row = _parse_omdb_response(data, movie_id)
                batch_buffer.append(row)
                checkpoint["last_processed_movie_id"] = movie_id

                # 배치 INSERT
                if len(batch_buffer) >= batch_insert:
                    try:
                        inserted = await _insert_batch(pool, batch_buffer)
                        checkpoint["total_inserted"] += inserted
                    except Exception as e:
                        logger.error(
                            "omdb_batch_insert_failed",
                            batch_size=len(batch_buffer),
                            error=str(e)[:200],
                        )
                        checkpoint["total_failed"] += len(batch_buffer)
                    batch_buffer = []
                    _save_checkpoint(checkpoint)

                # 진행률 (50건마다)
                if api_calls_in_run % 50 == 0:
                    elapsed = time.time() - pipeline_start
                    rate = api_calls_in_run / elapsed if elapsed > 0 else 0
                    remaining = max_calls - api_calls_in_run
                    eta_min = remaining / rate / 60 if rate > 0 else 0
                    print(
                        f"  [{api_calls_in_run:>4}/{max_calls}] "
                        f"inserted {checkpoint['total_inserted']:>4} | "
                        f"failed {checkpoint['total_failed']:>3} | "
                        f"속도 {rate:>4.1f}/s | "
                        f"ETA {eta_min:>5.1f}m"
                    )

                # max_calls 도달 시 중단 (무료 한도 보호)
                if api_calls_in_run >= max_calls:
                    print(f"\n  max_calls {max_calls} 도달 → 중단 (다음 실행에서 이어감)")
                    break

        # 마지막 남은 배치
        if batch_buffer:
            try:
                inserted = await _insert_batch(pool, batch_buffer)
                checkpoint["total_inserted"] += inserted
            except Exception as e:
                logger.error("omdb_final_batch_failed", error=str(e)[:200])

        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        after_count = await _count_table(pool)
        total_elapsed = time.time() - pipeline_start

        print(f"\n{'=' * 60}")
        print(f"[OMDb 평점 적재 완료]")
        print(f"  이번 실행 API 호출: {api_calls_in_run:>10,} / max {max_calls:,}")
        print(f"  inserted:           {checkpoint['total_inserted']:>10,}")
        print(f"  failed:             {checkpoint['total_failed']:>10,}")
        print(f"  before:             {before_count:>10,}")
        print(f"  after:              {after_count:>10,}")
        print(f"  diff:               {after_count - before_count:>10,}")
        print(f"  소요:               {total_elapsed / 60:>10.1f} 분")
        print(f"  💡 일일 cron 으로 매일 {max_calls}건씩 적재 권장")
        print(f"{'=' * 60}")

    finally:
        pool.close()
        await pool.wait_closed()
        await close_all_clients()


# ══════════════════════════════════════════════════════════════
# 상태 조회
# ══════════════════════════════════════════════════════════════


async def show_status() -> None:
    cp = _load_checkpoint()
    print("=" * 60)
    print(f"  OMDb Ratings 체크포인트 (누적)")
    print("=" * 60)
    print(f"  단계:           {cp.get('phase', '미시작')}")
    print(f"  누적 API 호출:  {cp.get('total_api_calls', 0):>10,}")
    print(f"  inserted:       {cp.get('total_inserted', 0):>10,}")
    print(f"  failed:         {cp.get('total_failed', 0):>10,}")
    print(f"  마지막 movie:   {cp.get('last_processed_movie_id', '-')}")
    print(f"  마지막 갱신:    {cp.get('last_updated', '-')}")
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
        description="OMDb 평점 → MySQL movie_external_ratings 적재 (인기 영화 우선)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 일일 cron (950건/일)
  PYTHONPATH=src uv run python scripts/run_omdb_ratings_load.py

  # 처음 100건만 (테스트)
  PYTHONPATH=src uv run python scripts/run_omdb_ratings_load.py --max-calls 100

  # 상태
  PYTHONPATH=src uv run python scripts/run_omdb_ratings_load.py --status
        """,
    )
    parser.add_argument(
        "--max-calls", type=int, default=DEFAULT_MAX_CALLS,
        help=f"이번 실행 최대 API 호출 (기본 {DEFAULT_MAX_CALLS}, 무료 1,000/day 안전 마진)",
    )
    parser.add_argument(
        "--rps", type=int, default=DEFAULT_RPS,
        help=f"초당 호출 수 (기본 {DEFAULT_RPS})",
    )
    parser.add_argument(
        "--batch-insert", type=int, default=DEFAULT_BATCH_INSERT,
        help=f"executemany 배치 (기본 {DEFAULT_BATCH_INSERT})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="(기본 동작) 기존 적재 movie_id skip — 항상 적용됨",
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
            run_omdb_ratings_load(
                max_calls=args.max_calls,
                rps=args.rps,
                batch_insert=args.batch_insert,
            )
        )
