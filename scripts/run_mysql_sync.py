"""
MySQL movies 테이블 동기화 스크립트 (Qdrant → MySQL).

Qdrant에 적재된 영화 데이터를 MySQL movies 테이블에 동기화한다.
임베딩은 불필요하며, Qdrant payload를 MySQL 컬럼에 매핑하여 upsert한다.

배경:
    run_full_reload.py는 Qdrant/Neo4j/ES 3개 DB만 적재하고 MySQL은 포함하지 않는다.
    이 스크립트로 Qdrant의 최신 데이터를 MySQL movies 테이블에 반영한다.

사용법:
    # 기본 실행 (Qdrant 전체 → MySQL upsert)
    PYTHONPATH=src uv run python scripts/run_mysql_sync.py

    # 배치 크기 조정
    PYTHONPATH=src uv run python scripts/run_mysql_sync.py --batch-size 1000

    # 현재 MySQL 건수 확인만
    PYTHONPATH=src uv run python scripts/run_mysql_sync.py --status

    # 특정 source만 동기화
    PYTHONPATH=src uv run python scripts/run_mysql_sync.py --source tmdb

소요 시간 추정:
    - Qdrant 조회: ~10분 (806K건 scroll)
    - MySQL upsert: ~30분 (배치 1000건, 806K건)
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog  # noqa: E402

from monglepick.db.clients import init_all_clients, close_all_clients, get_mysql  # noqa: E402
from monglepick.config import settings  # noqa: E402

logger = structlog.get_logger()

DEFAULT_BATCH_SIZE = 1000


# ============================================================
# Qdrant에서 전체 payload 스트리밍 조회
# ============================================================

def _scroll_qdrant_payloads(
    source_filter: str | None = None,
) -> "Generator[list[tuple[str, dict]], None, None]":
    """
    Qdrant에서 1000건씩 payload를 스트리밍 조회한다.

    Args:
        source_filter: 특정 소스만 필터링 (tmdb, kaggle, kobis, kmdb)

    Yields:
        [(point_id, payload), ...] 리스트 (1000건 단위)
    """
    from qdrant_client import QdrantClient, models as qmodels

    client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)

    # 필터 설정
    scroll_filter = None
    if source_filter:
        scroll_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="source",
                match=qmodels.MatchValue(value=source_filter),
            )]
        )

    # MySQL에 필요한 필드만 조회 (payload 전체 대신 필요 필드 지정)
    # Phase 3 재적재 보강 (2026-04-07): JPA Movie 엔티티 35컬럼 전체 커버
    # - cast_members 추가 (구 cast 컬럼은 레거시, JPA 엔티티에 없음)
    # - keywords / mood_tags / ott_platforms (JSON) 추가
    # - adult / kr_release_date 추가 (release_date 컬럼 생성용)
    payload_fields = [
        "id",
        "title", "title_en", "poster_path", "backdrop_path",
        "release_year", "runtime", "rating", "vote_count",
        "popularity_score", "genres", "director", "cast",
        "certification", "trailer_url", "overview", "tagline",
        "imdb_id", "original_language", "collection_name",
        "kobis_movie_cd", "sales_acc", "audience_count", "screen_count",
        "kobis_watch_grade", "kobis_open_dt", "kmdb_id", "awards",
        "filming_location", "source",
        # Phase 3 재적재 추가 필드
        "keywords", "mood_tags", "ott_platforms",
        "adult", "kr_release_date",
    ]

    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            scroll_filter=scroll_filter,
            with_vectors=False,
            with_payload=payload_fields,
        )
        if not points:
            break

        batch = []
        for p in points:
            batch.append((str(p.id), p.payload or {}))
        yield batch

        if next_offset is None:
            break
        offset = next_offset

    client.close()


# ============================================================
# payload → MySQL INSERT 값 변환
# ============================================================

def _resolve_release_date(payload: dict) -> str | None:
    """
    JPA Movie.releaseDate (DATE) 컬럼에 넣을 값을 우선순위로 선택한다.

    우선순위:
        1. kr_release_date (YYYY-MM-DD, TMDB release_dates.KR 에서 추출)
        2. kobis_open_dt (YYYY-MM-DD 또는 YYYYMMDD)
        3. f"{release_year}-01-01" (release_year > 1800 인 경우만)

    Returns:
        MySQL DATE 형식 문자열 ("YYYY-MM-DD") 또는 None
    """
    # 1. kr_release_date (이미 YYYY-MM-DD 포맷)
    kr = payload.get("kr_release_date", "") or ""
    if kr and len(kr) >= 10 and kr[4] == "-" and kr[7] == "-":
        return kr[:10]

    # 2. kobis_open_dt — YYYY-MM-DD 또는 YYYYMMDD
    ko = payload.get("kobis_open_dt", "") or ""
    if ko:
        if len(ko) >= 10 and ko[4] == "-" and ko[7] == "-":
            return ko[:10]
        if len(ko) == 8 and ko.isdigit():
            return f"{ko[:4]}-{ko[4:6]}-{ko[6:8]}"

    # 3. release_year 기반 대략값
    try:
        year = int(payload.get("release_year") or 0)
        if 1800 < year < 2100:
            return f"{year:04d}-01-01"
    except (TypeError, ValueError):
        pass

    return None


def _payload_to_mysql_row(point_id: str, payload: dict) -> tuple:
    """
    Qdrant payload를 MySQL movies 테이블 INSERT 값으로 변환한다.

    Phase 3 재적재 보강 (2026-04-07):
        기존 30 컬럼 → 36 컬럼으로 확장. JPA Movie 엔티티와 정합.
        - 레거시 `cast` (VARCHAR) 컬럼은 더 이상 사용하지 않음 (JPA 엔티티에 없음)
        - JPA가 쓰는 `cast_members` (JSON) 컬럼에 배우 리스트 저장
        - `keywords`, `mood_tags`, `ott_platforms` (JSON 3종) 추가
        - `tmdb_id` (BIGINT) 추가 — source='tmdb' 이고 movie_id가 정수일 때만
        - `adult` (BOOLEAN) 추가
        - `release_date` (DATE) 추가 — _resolve_release_date()로 우선순위 결정

    JSON 컬럼은 json.dumps(ensure_ascii=False)로 직렬화하여 한글 보존.
    NULL 가능 필드는 빈 값을 None으로 변환한다.

    Returns:
        MySQL INSERT에 사용할 값 튜플 (36개 컬럼 순서) — UPSERT_SQL과 일치해야 함
    """
    # cast 필드 정규화 (list[dict] → list[str])
    # Qdrant payload의 cast 는 Phase ML-2 이후 list[str] (한영 이중) 이지만,
    # 이전 적재본은 list[dict](cast_characters 형태)일 수 있으므로 둘 다 처리.
    cast_raw = payload.get("cast", [])
    if cast_raw and isinstance(cast_raw[0], dict):
        cast_list = [c.get("name", "") for c in cast_raw if isinstance(c, dict) and c.get("name")]
    elif cast_raw and isinstance(cast_raw[0], str):
        cast_list = cast_raw
    else:
        cast_list = []

    # movie_id: Qdrant payload["id"] 우선, 없으면 point_id 사용
    movie_id = str(payload.get("id", point_id))

    # tmdb_id: source='tmdb'이고 movie_id가 순수 정수 문자열일 때만 BIGINT로 변환
    # KOBIS/KMDb source는 uuid5 기반 문자열이므로 BIGINT 변환 불가 → NULL
    source = payload.get("source", "tmdb") or "tmdb"
    tmdb_id = None
    if source == "tmdb" and movie_id.isdigit():
        try:
            tmdb_id = int(movie_id)
        except ValueError:
            tmdb_id = None

    # JSON 컬럼 직렬화 (한글 보존 ensure_ascii=False)
    genres_json = (
        json.dumps(payload.get("genres", []), ensure_ascii=False)
        if payload.get("genres") else None
    )
    cast_members_json = (
        json.dumps(cast_list, ensure_ascii=False) if cast_list else None
    )
    keywords_json = (
        json.dumps(payload.get("keywords", []), ensure_ascii=False)
        if payload.get("keywords") else None
    )
    mood_tags_json = (
        json.dumps(payload.get("mood_tags", []), ensure_ascii=False)
        if payload.get("mood_tags") else None
    )
    ott_platforms_json = (
        json.dumps(payload.get("ott_platforms", []), ensure_ascii=False)
        if payload.get("ott_platforms") else None
    )

    # adult: Qdrant payload에 저장된 bool 값, 기본 False
    adult_value = bool(payload.get("adult", False))

    # release_date: 우선순위 기반 선택
    release_date = _resolve_release_date(payload)

    # UPSERT_SQL의 37개 컬럼 순서와 정확히 일치해야 함
    return (
        movie_id,                                                    # 1. movie_id
        tmdb_id,                                                     # 2. tmdb_id
        payload.get("title", "") or None,                            # 3. title
        payload.get("title_en", "") or None,                         # 4. title_en
        (payload.get("overview", "") or "")[:65535] or None,         # 5. overview (TEXT, 길이 제한)
        genres_json,                                                 # 6. genres (JSON)
        payload.get("release_year") or None,                         # 7. release_year
        payload.get("rating") or None,                               # 8. rating
        payload.get("poster_path", "") or None,                      # 9. poster_path
        cast_members_json,                                           # 10. cast_members (JSON)
        payload.get("director", "") or None,                         # 11. director
        keywords_json,                                               # 12. keywords (JSON)
        ott_platforms_json,                                          # 13. ott_platforms (JSON)
        mood_tags_json,                                              # 14. mood_tags (JSON)
        source,                                                      # 15. source
        release_date,                                                # 16. release_date
        payload.get("runtime") or None,                              # 17. runtime
        payload.get("vote_count") or None,                           # 18. vote_count
        payload.get("popularity_score") or None,                     # 19. popularity_score
        payload.get("certification", "") or None,                    # 20. certification
        payload.get("trailer_url", "") or None,                      # 21. trailer_url
        payload.get("tagline", "") or None,                          # 22. tagline
        payload.get("imdb_id", "") or None,                          # 23. imdb_id
        payload.get("original_language", "") or None,                # 24. original_language
        payload.get("collection_name", "") or None,                  # 25. collection_name
        payload.get("kobis_movie_cd", "") or None,                   # 26. kobis_movie_cd
        payload.get("sales_acc") or None,                            # 27. sales_acc
        payload.get("audience_count") or None,                       # 28. audience_count
        payload.get("screen_count") or None,                         # 29. screen_count
        payload.get("kobis_watch_grade", "") or None,                # 30. kobis_watch_grade
        payload.get("kobis_open_dt", "") or None,                    # 31. kobis_open_dt (원본 문자열 보존)
        payload.get("kmdb_id", "") or None,                          # 32. kmdb_id
        payload.get("backdrop_path", "") or None,                    # 33. backdrop_path
        adult_value,                                                 # 34. adult
        payload.get("awards", "") or None,                           # 35. awards
        payload.get("filming_location", "") or None,                 # 36. filming_location
    )


# ============================================================
# MySQL 배치 upsert
# ============================================================

# movies 테이블의 INSERT SQL (ON DUPLICATE KEY UPDATE)
# Phase 3 재적재 보강 (2026-04-07): 30 → 36 컬럼. JPA Movie 엔티티와 정합.
#
# 컬럼 순서는 _payload_to_mysql_row() 반환 튜플의 순서와 반드시 일치해야 한다.
# 변경점:
#   - `cast` (레거시 VARCHAR) 제거 → `cast_members` (JSON) 추가
#   - `tmdb_id` BIGINT 추가 (2번째 슬롯)
#   - `keywords`, `mood_tags`, `ott_platforms` (JSON 3종) 추가
#   - `release_date` DATE 추가
#   - `adult` BOOLEAN 추가
UPSERT_SQL = """
INSERT INTO movies (
    movie_id, tmdb_id, title, title_en, overview,
    genres, release_year, rating, poster_path, cast_members,
    director, keywords, ott_platforms, mood_tags, source,
    release_date, runtime, vote_count, popularity_score, certification,
    trailer_url, tagline, imdb_id, original_language, collection_name,
    kobis_movie_cd, sales_acc, audience_count, screen_count, kobis_watch_grade,
    kobis_open_dt, kmdb_id, backdrop_path, adult, awards,
    filming_location
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s
) ON DUPLICATE KEY UPDATE
    tmdb_id = VALUES(tmdb_id),
    title = VALUES(title),
    title_en = VALUES(title_en),
    overview = VALUES(overview),
    genres = VALUES(genres),
    release_year = VALUES(release_year),
    rating = VALUES(rating),
    poster_path = VALUES(poster_path),
    cast_members = VALUES(cast_members),
    director = VALUES(director),
    keywords = VALUES(keywords),
    ott_platforms = VALUES(ott_platforms),
    mood_tags = VALUES(mood_tags),
    source = VALUES(source),
    release_date = VALUES(release_date),
    runtime = VALUES(runtime),
    vote_count = VALUES(vote_count),
    popularity_score = VALUES(popularity_score),
    certification = VALUES(certification),
    trailer_url = VALUES(trailer_url),
    tagline = VALUES(tagline),
    imdb_id = VALUES(imdb_id),
    original_language = VALUES(original_language),
    collection_name = VALUES(collection_name),
    kobis_movie_cd = VALUES(kobis_movie_cd),
    sales_acc = VALUES(sales_acc),
    audience_count = VALUES(audience_count),
    screen_count = VALUES(screen_count),
    kobis_watch_grade = VALUES(kobis_watch_grade),
    kobis_open_dt = VALUES(kobis_open_dt),
    kmdb_id = VALUES(kmdb_id),
    backdrop_path = VALUES(backdrop_path),
    adult = VALUES(adult),
    awards = VALUES(awards),
    filming_location = VALUES(filming_location)
"""


async def _upsert_batch(rows: list[tuple]) -> int:
    """
    MySQL movies 테이블에 배치 upsert를 실행한다.

    ON DUPLICATE KEY UPDATE로 기존 레코드는 갱신, 신규는 삽입한다.

    Args:
        rows: _payload_to_mysql_row()로 변환된 값 튜플 리스트

    Returns:
        처리된 행 수
    """
    pool = await get_mysql()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                await cursor.executemany(UPSERT_SQL, rows)
                await conn.commit()
                return len(rows)
            except Exception:
                await conn.rollback()
                raise


# ============================================================
# 메인 파이프라인
# ============================================================

async def run_mysql_sync(
    batch_size: int = DEFAULT_BATCH_SIZE,
    source_filter: str | None = None,
) -> None:
    """
    Qdrant → MySQL movies 테이블 동기화.

    1. Qdrant에서 payload를 1000건씩 스트리밍 조회
    2. MySQL INSERT 값으로 변환
    3. 배치 upsert (ON DUPLICATE KEY UPDATE)

    Args:
        batch_size: MySQL upsert 배치 크기
        source_filter: 특정 소스만 동기화 (tmdb, kaggle, kobis, kmdb)
    """
    pipeline_start = time.time()

    # ── Step 0: DB 초기화 ──
    await init_all_clients()

    try:
        # ── Step 1: Qdrant → MySQL 동기화 ──
        filter_msg = f" (source: {source_filter})" if source_filter else ""
        print(f"[Step 1] Qdrant → MySQL 동기화{filter_msg}")

        total_processed = 0
        total_upserted = 0
        total_errors = 0
        batch_count = 0

        # Qdrant 스트리밍 조회 (1000건씩)
        for qdrant_batch in _scroll_qdrant_payloads(source_filter):
            # payload → MySQL 행 변환
            rows: list[tuple] = []
            for pid, payload in qdrant_batch:
                try:
                    row = _payload_to_mysql_row(pid, payload)
                    # title이 없는 행은 스킵
                    if row[1]:  # title 필드 (인덱스 1)
                        rows.append(row)
                except Exception as e:
                    total_errors += 1
                    if total_errors <= 10:
                        logger.warning("row_convert_failed", id=pid, error=str(e))

            # 배치 upsert (batch_size 단위로 분할)
            for i in range(0, len(rows), batch_size):
                sub_batch = rows[i:i + batch_size]
                try:
                    affected = await _upsert_batch(sub_batch)
                    total_upserted += len(sub_batch)
                except Exception as e:
                    total_errors += len(sub_batch)
                    logger.error("mysql_upsert_failed", batch_size=len(sub_batch), error=str(e))

            total_processed += len(qdrant_batch)
            batch_count += 1

            # 10배치마다 진행률 출력
            if batch_count % 10 == 0:
                elapsed = time.time() - pipeline_start
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(
                    f"  진행: {total_processed:>10,}건 | "
                    f"upsert: {total_upserted:>10,}건 | "
                    f"에러: {total_errors:>5,}건 | "
                    f"속도: {rate:,.0f}건/s"
                )

        # ── 완료 ──
        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[MySQL 동기화 완료]")
        print(f"  Qdrant 조회: {total_processed:>10,}건")
        print(f"  MySQL upsert: {total_upserted:>10,}건")
        print(f"  에러:         {total_errors:>10,}건")
        print(f"  소요:         {total_elapsed / 60:>10.1f}분")
        print(f"{'=' * 60}")

    finally:
        await close_all_clients()


# ============================================================
# 상태 조회
# ============================================================

async def show_status() -> None:
    """현재 MySQL movies 테이블 건수를 출력한다."""
    await init_all_clients()

    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # 전체 건수
                await cursor.execute("SELECT COUNT(*) FROM movies")
                (total,) = await cursor.fetchone()

                # source별 건수
                await cursor.execute(
                    "SELECT source, COUNT(*) FROM movies GROUP BY source ORDER BY COUNT(*) DESC"
                )
                sources = await cursor.fetchall()

        print("=" * 60)
        print("  MySQL movies 테이블 현황")
        print("=" * 60)
        print(f"  전체: {total:>10,}건")
        for src, cnt in sources:
            print(f"  {src or 'NULL':>10}: {cnt:>10,}건")
        print("=" * 60)

    finally:
        await close_all_clients()


# ============================================================
# 진입점
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MySQL movies 테이블 동기화 (Qdrant → MySQL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 동기화
  PYTHONPATH=src uv run python scripts/run_mysql_sync.py

  # TMDB 소스만 동기화
  PYTHONPATH=src uv run python scripts/run_mysql_sync.py --source tmdb

  # 현재 MySQL 건수 확인
  PYTHONPATH=src uv run python scripts/run_mysql_sync.py --status
        """,
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"MySQL upsert 배치 크기 (기본: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        choices=["tmdb", "kaggle", "kobis", "kmdb"],
        help="특정 소스만 동기화",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="현재 MySQL movies 건수 확인만",
    )
    args = parser.parse_args()

    if args.status:
        asyncio.run(show_status())
    else:
        asyncio.run(
            run_mysql_sync(
                batch_size=args.batch_size,
                source_filter=args.source,
            )
        )
