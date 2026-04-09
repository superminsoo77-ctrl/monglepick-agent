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

    # MySQL 에 필요한 필드 전체 (58 컬럼, init.sql 2026-04-09 확장 기준)
    # Phase 2026-04-09: Phase ML-1 한영이중 + KOBIS 풀필드 + KMDb 풀필드 + 재무/다국어 포함
    payload_fields = [
        "id",
        # 기본
        "title", "title_en", "poster_path", "backdrop_path",
        "release_year", "kr_release_date",
        "runtime", "rating", "vote_count", "popularity_score",
        "genres", "keywords", "mood_tags", "ott_platforms",
        # Phase ML-1 한영 이중
        "director", "director_original_name",
        "cast", "cast_characters",
        # 메타
        "certification", "trailer_url", "overview", "overview_en",
        "tagline", "imdb_id", "original_language",
        "spoken_languages", "origin_country",
        "production_countries", "production_country_names", "production_companies",
        "budget", "revenue", "homepage", "status", "adult",
        "collection_id", "collection_name",
        # KOBIS 풀필드
        "kobis_movie_cd", "sales_acc", "audience_count", "screen_count",
        "kobis_nation", "kobis_watch_grade", "kobis_open_dt", "kobis_type_nm",
        "kobis_directors", "kobis_actors", "kobis_companies", "kobis_staffs",
        "kobis_genres",
        # KMDb 풀필드
        "kmdb_id", "awards", "filming_location",
        "soundtrack", "theme_song",
        # 출처
        "source",
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


def _json_or_null(value) -> str | None:
    """list/dict 를 JSON 으로 직렬화 (ensure_ascii=False). 비어있으면 None."""
    if value is None:
        return None
    if isinstance(value, (list, dict)) and not value:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return None


def _payload_to_mysql_row(point_id: str, payload: dict) -> tuple:
    """
    Qdrant payload → MySQL movies 58 컬럼 INSERT 값 튜플.

    2026-04-09 전면 확장:
        - 36 → 58 컬럼 (init.sql 재확장 반영)
        - Phase ML-1 한영 이중: director_original_name + cast_original_names
          (cast_original_names 는 cast_characters[].name 에서 추출 — TMDB 는 name 이 원어)
        - KOBIS 풀필드: nation/type_nm + directors/actors/companies/staffs/genres JSON
        - KMDb 풀필드: soundtrack/theme_song 추가
        - 재무/다국어: budget/revenue/homepage/status/spoken_languages/production_*
        - 컬렉션: collection_id (int)

    JSON 컬럼 (한글 보존): genres, keywords, mood_tags, ott_platforms,
        cast_members, cast_original_names, spoken_languages, origin_country,
        production_countries, production_country_names, production_companies,
        kobis_directors, kobis_actors, kobis_companies, kobis_staffs, kobis_genres

    Returns:
        58 컬럼 순서 튜플 — UPSERT_SQL 과 반드시 일치
    """
    # ── cast 한영 이중 처리 ──
    # cast (list[str]): Phase ML-1 이후 한국어 번역이 있으면 한국어, 없으면 원어
    # cast_characters (list[dict]): {id, name, character, profile_path} — name 은 항상 원어
    cast_raw = payload.get("cast", []) or []
    if cast_raw and isinstance(cast_raw[0], dict):
        cast_members_list = [c.get("name", "") for c in cast_raw if isinstance(c, dict) and c.get("name")]
    elif cast_raw and isinstance(cast_raw[0], str):
        cast_members_list = cast_raw
    else:
        cast_members_list = []

    cast_characters_raw = payload.get("cast_characters", []) or []
    cast_original_list = [
        c.get("name", "") for c in cast_characters_raw
        if isinstance(c, dict) and c.get("name")
    ]

    # ── movie_id / tmdb_id ──
    movie_id = str(payload.get("id", point_id))
    source = payload.get("source", "tmdb") or "tmdb"
    tmdb_id = None
    if source == "tmdb" and movie_id.isdigit():
        try:
            tmdb_id = int(movie_id)
        except ValueError:
            tmdb_id = None

    # ── adult / release_date ──
    adult_value = bool(payload.get("adult", False))
    release_date = _resolve_release_date(payload)
    kr_release_date = (payload.get("kr_release_date") or "")[:10] or None

    # ── TEXT 길이 제한 ──
    overview = (payload.get("overview", "") or "")[:65535] or None
    overview_en = (payload.get("overview_en", "") or "")[:65535] or None
    awards = (payload.get("awards", "") or "")[:65535] or None
    filming_location = (payload.get("filming_location", "") or "")[:65535] or None
    soundtrack = (payload.get("soundtrack", "") or "")[:65535] or None
    theme_song = (payload.get("theme_song", "") or "")[:65535] or None

    # ── 58 컬럼 튜플 (init.sql 순서) ──
    return (
        movie_id,                                                              # 1. movie_id
        tmdb_id,                                                               # 2. tmdb_id
        payload.get("title", "") or None,                                      # 3. title
        payload.get("title_en", "") or None,                                   # 4. title_en
        payload.get("poster_path", "") or None,                                # 5. poster_path
        payload.get("backdrop_path", "") or None,                              # 6. backdrop_path
        payload.get("release_year") or None,                                   # 7. release_year
        release_date,                                                          # 8. release_date
        kr_release_date,                                                       # 9. kr_release_date
        payload.get("runtime") or None,                                        # 10. runtime
        payload.get("rating") or None,                                         # 11. rating
        payload.get("vote_count") or None,                                     # 12. vote_count
        payload.get("popularity_score") or None,                               # 13. popularity_score
        _json_or_null(payload.get("genres")),                                  # 14. genres
        _json_or_null(payload.get("keywords")),                                # 15. keywords
        _json_or_null(payload.get("mood_tags")),                               # 16. mood_tags
        _json_or_null(payload.get("ott_platforms")),                           # 17. ott_platforms
        payload.get("director", "") or None,                                   # 18. director
        payload.get("director_original_name", "") or None,                     # 19. director_original_name
        _json_or_null(cast_members_list),                                      # 20. cast_members
        _json_or_null(cast_original_list),                                     # 21. cast_original_names
        payload.get("certification", "") or None,                              # 22. certification
        payload.get("trailer_url", "") or None,                                # 23. trailer_url
        overview,                                                              # 24. overview
        overview_en,                                                           # 25. overview_en
        payload.get("tagline", "") or None,                                    # 26. tagline
        payload.get("imdb_id", "") or None,                                    # 27. imdb_id
        payload.get("original_language", "") or None,                          # 28. original_language
        _json_or_null(payload.get("spoken_languages")),                        # 29. spoken_languages
        _json_or_null(payload.get("origin_country")),                          # 30. origin_country
        _json_or_null(payload.get("production_countries")),                    # 31. production_countries
        _json_or_null(payload.get("production_country_names")),                # 32. production_country_names
        _json_or_null(payload.get("production_companies")),                    # 33. production_companies
        int(payload.get("budget") or 0),                                       # 34. budget
        int(payload.get("revenue") or 0),                                      # 35. revenue
        payload.get("homepage", "") or None,                                   # 36. homepage
        payload.get("status", "") or None,                                     # 37. status
        adult_value,                                                           # 38. adult
        int(payload.get("collection_id") or 0) or None,                        # 39. collection_id
        payload.get("collection_name", "") or None,                            # 40. collection_name
        payload.get("kobis_movie_cd", "") or None,                             # 41. kobis_movie_cd
        int(payload.get("sales_acc") or 0),                                    # 42. sales_acc
        int(payload.get("audience_count") or 0),                               # 43. audience_count
        int(payload.get("screen_count") or 0),                                 # 44. screen_count
        payload.get("kobis_nation", "") or None,                               # 45. kobis_nation
        payload.get("kobis_watch_grade", "") or None,                          # 46. kobis_watch_grade
        payload.get("kobis_open_dt", "") or None,                              # 47. kobis_open_dt
        payload.get("kobis_type_nm", "") or None,                              # 48. kobis_type_nm
        _json_or_null(payload.get("kobis_directors")),                         # 49. kobis_directors
        _json_or_null(payload.get("kobis_actors")),                            # 50. kobis_actors
        _json_or_null(payload.get("kobis_companies")),                         # 51. kobis_companies
        _json_or_null(payload.get("kobis_staffs")),                            # 52. kobis_staffs
        _json_or_null(payload.get("kobis_genres")),                            # 53. kobis_genres
        payload.get("kmdb_id", "") or None,                                    # 54. kmdb_id
        awards,                                                                # 55. awards
        filming_location,                                                      # 56. filming_location
        soundtrack,                                                            # 57. soundtrack
        theme_song,                                                            # 58. theme_song
        source,                                                                # 59. source
    )


# ============================================================
# MySQL 배치 upsert
# ============================================================

# movies 테이블 INSERT SQL (ON DUPLICATE KEY UPDATE)
# 2026-04-09 전면 확장: 36 → 59 컬럼 (init.sql 재확장 기준).
#
# 컬럼 순서는 _payload_to_mysql_row() 반환 튜플 순서와 정확히 일치해야 한다.
# 주요 추가 컬럼:
#   - Phase ML-1 한영이중: director_original_name, cast_original_names (JSON)
#   - 다국어: overview_en, spoken_languages, origin_country,
#            production_countries, production_country_names, production_companies (JSON)
#   - 재무: budget, revenue, homepage, status, collection_id
#   - 한국 개봉: kr_release_date
#   - KOBIS 풀필드: kobis_nation, kobis_type_nm,
#                   kobis_directors/actors/companies/staffs/genres (JSON 5종)
#   - KMDb 풀필드: soundtrack, theme_song
UPSERT_SQL = """
INSERT INTO movies (
    movie_id, tmdb_id, title, title_en, poster_path,
    backdrop_path, release_year, release_date, kr_release_date, runtime,
    rating, vote_count, popularity_score, genres, keywords,
    mood_tags, ott_platforms, director, director_original_name, cast_members,
    cast_original_names, certification, trailer_url, overview, overview_en,
    tagline, imdb_id, original_language, spoken_languages, origin_country,
    production_countries, production_country_names, production_companies, budget, revenue,
    homepage, status, adult, collection_id, collection_name,
    kobis_movie_cd, sales_acc, audience_count, screen_count, kobis_nation,
    kobis_watch_grade, kobis_open_dt, kobis_type_nm, kobis_directors, kobis_actors,
    kobis_companies, kobis_staffs, kobis_genres, kmdb_id, awards,
    filming_location, soundtrack, theme_song, source
) VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s
) ON DUPLICATE KEY UPDATE
    tmdb_id = VALUES(tmdb_id),
    title = VALUES(title),
    title_en = VALUES(title_en),
    poster_path = VALUES(poster_path),
    backdrop_path = VALUES(backdrop_path),
    release_year = VALUES(release_year),
    release_date = VALUES(release_date),
    kr_release_date = VALUES(kr_release_date),
    runtime = VALUES(runtime),
    rating = VALUES(rating),
    vote_count = VALUES(vote_count),
    popularity_score = VALUES(popularity_score),
    genres = VALUES(genres),
    keywords = VALUES(keywords),
    mood_tags = VALUES(mood_tags),
    ott_platforms = VALUES(ott_platforms),
    director = VALUES(director),
    director_original_name = VALUES(director_original_name),
    cast_members = VALUES(cast_members),
    cast_original_names = VALUES(cast_original_names),
    certification = VALUES(certification),
    trailer_url = VALUES(trailer_url),
    overview = VALUES(overview),
    overview_en = VALUES(overview_en),
    tagline = VALUES(tagline),
    imdb_id = VALUES(imdb_id),
    original_language = VALUES(original_language),
    spoken_languages = VALUES(spoken_languages),
    origin_country = VALUES(origin_country),
    production_countries = VALUES(production_countries),
    production_country_names = VALUES(production_country_names),
    production_companies = VALUES(production_companies),
    budget = VALUES(budget),
    revenue = VALUES(revenue),
    homepage = VALUES(homepage),
    status = VALUES(status),
    adult = VALUES(adult),
    collection_id = VALUES(collection_id),
    collection_name = VALUES(collection_name),
    kobis_movie_cd = VALUES(kobis_movie_cd),
    sales_acc = VALUES(sales_acc),
    audience_count = VALUES(audience_count),
    screen_count = VALUES(screen_count),
    kobis_nation = VALUES(kobis_nation),
    kobis_watch_grade = VALUES(kobis_watch_grade),
    kobis_open_dt = VALUES(kobis_open_dt),
    kobis_type_nm = VALUES(kobis_type_nm),
    kobis_directors = VALUES(kobis_directors),
    kobis_actors = VALUES(kobis_actors),
    kobis_companies = VALUES(kobis_companies),
    kobis_staffs = VALUES(kobis_staffs),
    kobis_genres = VALUES(kobis_genres),
    kmdb_id = VALUES(kmdb_id),
    awards = VALUES(awards),
    filming_location = VALUES(filming_location),
    soundtrack = VALUES(soundtrack),
    theme_song = VALUES(theme_song),
    source = VALUES(source)
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
                    # 2026-04-09: 59 컬럼 확장 후 title 인덱스가 2로 이동.
                    # row[0]=movie_id, row[1]=tmdb_id, row[2]=title
                    # 기존 row[1] 는 tmdb_id 였고 KOBIS/KMDb 는 None → 105K 건 필터되던 버그
                    if row[2]:  # title 필드 (인덱스 2)
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
