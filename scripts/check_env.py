"""
환경 변수 / API 키 / DB 연결 사전 검증 스크립트.

Task #5 (run_full_reload.py) 완료 후 Phase 2~9 순차 실행 전에
필요한 모든 환경 변수가 설정됐는지 + DB 가 연결 가능한지 검증한다.

중간에 키 누락 / 연결 실패로 장시간 작업이 날아가는 것을 방지한다.

설계 진실 원본:
    docs/Phase_ML4_재적재_진행상황_세션인계.md §12.7 Phase 2~9

검증 항목:
    1. 필수 API 키: TMDB, KOBIS, KMDB, UPSTAGE
    2. 선택 API 키: OMDB (없으면 Phase G-2 skip)
    3. MySQL 연결 (aiomysql)
    4. Qdrant 연결 (HTTP /collections)
    5. Neo4j 연결 (Bolt)
    6. Elasticsearch 연결 (HTTP /_cluster/health)
    7. Redis 연결
    8. 원본 데이터 파일 존재:
        - data/tmdb_full/tmdb_full_movies.jsonl
        - data/kaggle_movies/ratings.csv
        - data/kaggle_movies/links.csv

사용법:
    # 전체 검증
    PYTHONPATH=src uv run python scripts/check_env.py

    # 선택 항목 건너뛰기 (아직 준비 안 된 키)
    PYTHONPATH=src uv run python scripts/check_env.py --skip-omdb

    # 특정 카테고리만
    PYTHONPATH=src uv run python scripts/check_env.py --only api
    PYTHONPATH=src uv run python scripts/check_env.py --only db
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
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

from monglepick.config import settings  # noqa: E402


# ══════════════════════════════════════════════════════════════
# 출력 헬퍼
# ══════════════════════════════════════════════════════════════

SYM_OK = "✅"
SYM_WARN = "⚠️ "
SYM_FAIL = "❌"


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_check(symbol: str, name: str, detail: str = "") -> None:
    line = f"  {symbol} {name:30s}"
    if detail:
        line += f" — {detail}"
    print(line)


# ══════════════════════════════════════════════════════════════
# 1. API 키 검증
# ══════════════════════════════════════════════════════════════


def check_api_keys(skip_omdb: bool = False) -> dict:
    """필수/선택 API 키 존재 여부만 확인 (실제 호출 안 함)."""
    _print_header("1. API 키 검증")

    result = {
        "required_missing": [],
        "optional_missing": [],
    }

    required_keys = [
        ("TMDB_API_KEY", "TMDB API"),
        ("KOBIS_API_KEY", "KOBIS API"),
        ("KMDB_API_KEY", "KMDb API"),
        ("UPSTAGE_API_KEY", "Upstage Solar API"),
    ]
    for env, label in required_keys:
        value = os.environ.get(env, "")
        if value and len(value) > 5:
            _print_check(SYM_OK, label, f"{env} ({len(value)} chars)")
        else:
            _print_check(SYM_FAIL, label, f"{env} 없음 또는 너무 짧음")
            result["required_missing"].append(env)

    # 선택 API 키
    optional_keys = []
    if not skip_omdb:
        optional_keys.append(("OMDB_API_KEY", "OMDb API (Phase G-2)"))

    for env, label in optional_keys:
        value = os.environ.get(env, "")
        if value and len(value) > 5:
            _print_check(SYM_OK, label, f"{env} ({len(value)} chars)")
        else:
            _print_check(
                SYM_WARN, label,
                f"{env} 없음 — 무료 발급: https://www.omdbapi.com/apikey.aspx"
            )
            result["optional_missing"].append(env)

    return result


# ══════════════════════════════════════════════════════════════
# 2. DB 연결 검증
# ══════════════════════════════════════════════════════════════


async def check_mysql() -> bool:
    """aiomysql 로 MySQL 연결 + persons/movies/kaggle_watch_history 테이블 존재 확인."""
    try:
        import aiomysql
        conn = await aiomysql.connect(
            host=settings.MYSQL_HOST,
            port=int(getattr(settings, "MYSQL_PORT", 3306)),
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            db=settings.MYSQL_DATABASE,
            charset="utf8mb4",
            connect_timeout=5,
        )
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema=%s",
                    (settings.MYSQL_DATABASE,),
                )
                (table_count,) = await cur.fetchone()

                # 주요 테이블 존재 확인
                key_tables = [
                    "movies", "users",
                    "persons",                # Phase C-6 신규
                    "kaggle_watch_history",   # Phase B-1 리네임
                    "box_office_daily",       # Phase G-1 신규
                    "movie_external_ratings", # Phase G-2 신규
                ]
                await cur.execute(
                    f"SELECT table_name FROM information_schema.tables "
                    f"WHERE table_schema=%s AND table_name IN ({','.join(['%s'] * len(key_tables))})",
                    (settings.MYSQL_DATABASE, *key_tables),
                )
                rows = await cur.fetchall()
                existing = {r[0] for r in rows}

                _print_check(SYM_OK, "MySQL 연결", f"{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE} ({table_count} tables)")

                for t in key_tables:
                    if t in existing:
                        _print_check(SYM_OK, f"  table: {t}", "존재")
                    else:
                        _print_check(SYM_WARN, f"  table: {t}", "없음 (init.sql 실행 필요)")

            return True
        finally:
            conn.close()
    except Exception as e:
        _print_check(SYM_FAIL, "MySQL 연결", str(e)[:80])
        return False


async def check_qdrant() -> bool:
    """Qdrant HTTP /collections 로 연결 확인."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.QDRANT_URL}/collections")
            if response.status_code != 200:
                _print_check(SYM_FAIL, "Qdrant 연결", f"HTTP {response.status_code}")
                return False

            data = response.json()
            collections = [c["name"] for c in data.get("result", {}).get("collections", [])]
            _print_check(SYM_OK, "Qdrant 연결", f"{settings.QDRANT_URL} ({len(collections)} collections)")

            for coll in ["movies", "persons"]:
                if coll in collections:
                    # 포인트 수 조회
                    r2 = await client.get(f"{settings.QDRANT_URL}/collections/{coll}")
                    pts = r2.json().get("result", {}).get("points_count", 0)
                    _print_check(SYM_OK, f"  collection: {coll}", f"{pts:,} points")
                else:
                    if coll == "persons":
                        _print_check(SYM_WARN, f"  collection: {coll}", "없음 (Phase C 실행 후 자동 생성)")
                    else:
                        _print_check(SYM_FAIL, f"  collection: {coll}", "없음")

        return True
    except Exception as e:
        _print_check(SYM_FAIL, "Qdrant 연결", str(e)[:80])
        return False


async def check_neo4j() -> bool:
    """Neo4j Bolt 연결 + Movie/Person 노드 수."""
    try:
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        try:
            async with driver.session() as session:
                # 연결 테스트
                await session.run("RETURN 1")

                # Movie 노드 수
                r = await session.run("MATCH (m:Movie) RETURN count(m) AS cnt")
                movie_count = (await r.single())["cnt"]

                # Person 노드 수
                r2 = await session.run("MATCH (p:Person) RETURN count(p) AS cnt")
                person_count = (await r2.single())["cnt"]

                _print_check(SYM_OK, "Neo4j 연결", settings.NEO4J_URI)
                _print_check(SYM_OK, f"  Movie 노드", f"{movie_count:,}")
                _print_check(SYM_OK, f"  Person 노드", f"{person_count:,}")

            return True
        finally:
            await driver.close()
    except Exception as e:
        _print_check(SYM_FAIL, "Neo4j 연결", str(e)[:80])
        return False


async def check_elasticsearch() -> bool:
    """Elasticsearch HTTP 연결 + movies_bm25 인덱스."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{settings.ELASTICSEARCH_URL}/_cluster/health")
            if r.status_code != 200:
                _print_check(SYM_FAIL, "Elasticsearch 연결", f"HTTP {r.status_code}")
                return False

            status = r.json().get("status")
            _print_check(SYM_OK, "Elasticsearch 연결", f"{settings.ELASTICSEARCH_URL} (status={status})")

            # movies_bm25 인덱스
            r2 = await client.get(f"{settings.ELASTICSEARCH_URL}/movies_bm25/_count")
            if r2.status_code == 200:
                count = r2.json().get("count", 0)
                _print_check(SYM_OK, "  index: movies_bm25", f"{count:,} docs")
            else:
                _print_check(SYM_WARN, "  index: movies_bm25", "없음")

        return True
    except Exception as e:
        _print_check(SYM_FAIL, "Elasticsearch 연결", str(e)[:80])
        return False


async def check_redis() -> bool:
    """Redis 연결 + CF 캐시 키 존재 여부."""
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            await client.ping()
            dbsize = await client.dbsize()
            _print_check(SYM_OK, "Redis 연결", f"{settings.REDIS_URL} ({dbsize} keys)")

            # CF 캐시 키 확인
            for key_pattern in ["cf:similar_users:*", "cf:user_ratings:*", "cf:movie_avg_rating:*"]:
                # KEYS 는 큰 DB 에서 느릴 수 있지만 검증용이라 허용
                keys = []
                async for k in client.scan_iter(match=key_pattern, count=100):
                    keys.append(k)
                    if len(keys) >= 5:
                        break
                if keys:
                    _print_check(SYM_OK, f"  cache: {key_pattern}", f"{len(keys)}+ keys")
                else:
                    _print_check(SYM_WARN, f"  cache: {key_pattern}", "없음 (Task #5 Step 3 완료 후 생성)")

            return True
        finally:
            await client.aclose()
    except Exception as e:
        _print_check(SYM_FAIL, "Redis 연결", str(e)[:80])
        return False


async def check_dbs() -> dict:
    """모든 DB 연결 검증."""
    _print_header("2. DB 연결 검증")
    return {
        "mysql":         await check_mysql(),
        "qdrant":        await check_qdrant(),
        "neo4j":         await check_neo4j(),
        "elasticsearch": await check_elasticsearch(),
        "redis":         await check_redis(),
    }


# ══════════════════════════════════════════════════════════════
# 3. 원본 데이터 파일 검증
# ══════════════════════════════════════════════════════════════


def check_data_files() -> dict:
    """TMDB JSONL / Kaggle CSV / Person JSONL 존재 확인."""
    _print_header("3. 원본 데이터 파일 검증")

    files = {
        "tmdb_jsonl": {
            "path": Path("data/tmdb_full/tmdb_full_movies.jsonl"),
            "required": True,
            "label": "TMDB JSONL (Task #5 입력)",
        },
        "kaggle_ratings": {
            "path": Path("data/kaggle_movies/ratings.csv"),
            "required": True,
            "label": "Kaggle ratings.csv (Phase B-2 + Redis CF 입력)",
        },
        "kaggle_links": {
            "path": Path("data/kaggle_movies/links.csv"),
            "required": True,
            "label": "Kaggle links.csv (TMDB ID 매핑)",
        },
        "kaggle_metadata": {
            "path": Path("data/kaggle_movies/movies_metadata.csv"),
            "required": False,
            "label": "Kaggle movies_metadata.csv (Phase 2 Kaggle supplement)",
        },
        "kaggle_credits": {
            "path": Path("data/kaggle_movies/credits.csv"),
            "required": False,
            "label": "Kaggle credits.csv (Phase 2 Kaggle supplement)",
        },
        "kaggle_keywords": {
            "path": Path("data/kaggle_movies/keywords.csv"),
            "required": False,
            "label": "Kaggle keywords.csv (Phase 2 Kaggle supplement)",
        },
        "person_jsonl": {
            "path": Path("data/tmdb_persons/tmdb_persons.jsonl"),
            "required": False,
            "label": "TMDB Person JSONL (Phase 4 입력, Task #5 완료 후 생성)",
        },
    }

    result = {"missing_required": [], "missing_optional": []}

    for key, meta in files.items():
        path = meta["path"]
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            # 심볼릭 링크 여부
            is_symlink = path.is_symlink()
            extra = " (symlink)" if is_symlink else ""
            if size_mb > 1024:
                _print_check(SYM_OK, meta["label"], f"{size_mb / 1024:.1f} GB{extra}")
            else:
                _print_check(SYM_OK, meta["label"], f"{size_mb:.1f} MB{extra}")
        else:
            if meta["required"]:
                _print_check(SYM_FAIL, meta["label"], f"{path} 없음")
                result["missing_required"].append(str(path))
            else:
                _print_check(SYM_WARN, meta["label"], f"{path} 없음 (선택)")
                result["missing_optional"].append(str(path))

    return result


# ══════════════════════════════════════════════════════════════
# 4. 종합 결과
# ══════════════════════════════════════════════════════════════


def print_summary(
    api_result: dict | None,
    db_result: dict | None,
    file_result: dict | None,
) -> int:
    _print_header("종합 결과")

    total_issues = 0

    if api_result is not None:
        req_missing = api_result.get("required_missing", [])
        opt_missing = api_result.get("optional_missing", [])
        if req_missing:
            print(f"  {SYM_FAIL} 필수 API 키 누락: {', '.join(req_missing)}")
            total_issues += len(req_missing)
        else:
            print(f"  {SYM_OK} 필수 API 키 모두 설정")
        if opt_missing:
            print(f"  {SYM_WARN} 선택 API 키 누락: {', '.join(opt_missing)}")

    if db_result is not None:
        ok_count = sum(1 for v in db_result.values() if v)
        total_count = len(db_result)
        if ok_count == total_count:
            print(f"  {SYM_OK} 모든 DB 연결 성공 ({ok_count}/{total_count})")
        else:
            failed = [k for k, v in db_result.items() if not v]
            print(f"  {SYM_FAIL} DB 연결 실패: {', '.join(failed)}")
            total_issues += total_count - ok_count

    if file_result is not None:
        if file_result.get("missing_required"):
            print(f"  {SYM_FAIL} 필수 데이터 파일 누락: {len(file_result['missing_required'])}개")
            total_issues += len(file_result["missing_required"])
        else:
            print(f"  {SYM_OK} 필수 데이터 파일 모두 존재")
        if file_result.get("missing_optional"):
            print(f"  {SYM_WARN} 선택 데이터 파일 누락: {len(file_result['missing_optional'])}개")

    print()
    if total_issues == 0:
        print(f"  {SYM_OK} 검증 통과 — Phase 2~9 실행 가능")
        return 0
    else:
        print(f"  {SYM_FAIL} {total_issues}건 이슈 발견 — 수정 후 재실행 필요")
        return 1


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════


async def main_async(args: argparse.Namespace) -> int:
    api_result: dict | None = None
    db_result: dict | None = None
    file_result: dict | None = None

    if args.only in (None, "api"):
        api_result = check_api_keys(skip_omdb=args.skip_omdb)

    if args.only in (None, "db"):
        db_result = await check_dbs()

    if args.only in (None, "file"):
        file_result = check_data_files()

    return print_summary(api_result, db_result, file_result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="환경 변수 / API 키 / DB 연결 사전 검증",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 검증
  PYTHONPATH=src uv run python scripts/check_env.py

  # OMDb 제외 (키 아직 없음)
  PYTHONPATH=src uv run python scripts/check_env.py --skip-omdb

  # API 키만
  PYTHONPATH=src uv run python scripts/check_env.py --only api
        """,
    )
    parser.add_argument(
        "--only", choices=["api", "db", "file"], default=None,
        help="특정 카테고리만 검증",
    )
    parser.add_argument(
        "--skip-omdb", action="store_true",
        help="OMDB_API_KEY 검증 건너뛰기 (아직 발급 안 함)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)
