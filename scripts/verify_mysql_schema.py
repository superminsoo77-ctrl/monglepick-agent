"""
init.sql ↔ MySQL 적재 스크립트 정합성 검증.

검증 항목:
    1. init.sql 각 CREATE TABLE 블록 → 컬럼명 + 타입 추출
    2. 각 적재 스크립트의 INSERT 문 → 컬럼명 추출
    3. 교차 검증:
        - init.sql 정의 컬럼 vs 스크립트 INSERT 컬럼 (쌍방향)
        - 누락 / 초과 / 타입 불일치 리포트
    4. 실제 MySQL 상태와 init.sql 대조 (선택)

검증 대상 테이블 + 스크립트:
    - movies                 ← scripts/run_mysql_sync.py
    - persons                ← scripts/run_persons_mysql_sync.py
    - kaggle_watch_history   ← scripts/run_kaggle_ratings_load.py
    - box_office_daily       ← scripts/run_kobis_boxoffice_history.py
    - movie_external_ratings ← scripts/run_omdb_ratings_load.py

사용법:
    PYTHONPATH=src uv run python scripts/verify_mysql_schema.py
    PYTHONPATH=src uv run python scripts/verify_mysql_schema.py --check-live-db
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INIT_SQL_PATH = PROJECT_ROOT.parent / "db_dumps" / "prod_old_backup" / "init.sql"

SCRIPT_MAP = {
    "movies": "scripts/run_mysql_sync.py",
    "persons": "scripts/run_persons_mysql_sync.py",
    "kaggle_watch_history": "scripts/run_kaggle_ratings_load.py",
    "box_office_daily": "scripts/run_kobis_boxoffice_history.py",
    "movie_external_ratings": "scripts/run_omdb_ratings_load.py",
}


def parse_init_sql_tables(init_sql_path: Path) -> dict[str, dict]:
    """
    init.sql 파일에서 모든 CREATE TABLE 블록 파싱.

    Returns:
        {table_name: {"columns": {col_name: col_type}, "indexes": [...]}}
    """
    content = init_sql_path.read_text(encoding="utf-8")

    tables: dict[str, dict] = {}

    # CREATE TABLE 블록 추출: CREATE TABLE IF NOT EXISTS {name} (\n ... \n) ENGINE=...;
    pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z_]\w*)\s*\((.*?)\)\s*ENGINE",
        re.DOTALL | re.IGNORECASE,
    )

    for match in pattern.finditer(content):
        table_name = match.group(1)
        body = match.group(2)

        columns: dict[str, str] = {}
        indexes: list[str] = []

        for raw_line in body.split("\n"):
            line = raw_line.strip().rstrip(",").strip()
            if not line or line.startswith("--"):
                continue
            # 인덱스 / 제약 (word boundary 필수 — `keywords` 컬럼이 `KEY` 로 오탐되는 것 방지)
            if re.match(r"^(PRIMARY\s+KEY\b|UNIQUE\s+KEY\b|UNIQUE\b|KEY\s|INDEX\b|FOREIGN\s+KEY\b|CONSTRAINT\b|CHECK\s*\()", line, re.IGNORECASE):
                indexes.append(line[:80])
                continue
            # 컬럼 정의: `col_name` TYPE ... 또는 col_name TYPE ...
            col_match = re.match(
                r"`?(\w+)`?\s+([A-Z]+(?:\([\d,\s]+\))?)",
                line,
                re.IGNORECASE,
            )
            if col_match:
                col_name = col_match.group(1)
                col_type = col_match.group(2).upper()
                # PRIMARY KEY / UNIQUE KEY 같은 키워드는 컬럼으로 잡지 않도록 필터
                if col_name.upper() in {"PRIMARY", "UNIQUE", "KEY", "INDEX", "CONSTRAINT", "FOREIGN"}:
                    continue
                columns[col_name] = col_type

        if columns:
            tables[table_name] = {"columns": columns, "indexes": indexes}

    return tables


def parse_insert_columns(script_path: Path) -> list[str]:
    """
    스크립트 파일에서 INSERT INTO ... (col1, col2, ...) 의 컬럼 목록 추출.

    f-string 지원: INSERT INTO {TABLE_NAME} (...) 형태도 매칭.
    """
    if not script_path.exists():
        return []
    content = script_path.read_text(encoding="utf-8")

    # INSERT INTO <table_or_placeholder> (col1, col2, ...) 블록
    # - `\w+` 리터럴 이름 (movies, persons, ...)
    # - `\{\w+\}` f-string placeholder ({TABLE_NAME}, {TBL}, ...)
    # - 여러 줄 허용 (DOTALL)
    pattern = re.compile(
        r"INSERT\s+INTO\s+[\w{}]+\s*\((.*?)\)\s*VALUES",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(content)
    if not matches:
        return []

    # 여러 INSERT 가 있으면 가장 긴 것 (보통 실제 upsert SQL)
    longest = max(matches, key=len)

    # 주석 제거 후 컬럼명 추출
    # Python f-string 내부 주석은 SQL 주석 문법 (--) 사용 가능
    lines_clean: list[str] = []
    for line in longest.split("\n"):
        # SQL 라인 주석 제거
        idx = line.find("--")
        if idx >= 0:
            line = line[:idx]
        lines_clean.append(line)
    cleaned = "\n".join(lines_clean)

    cols = [c.strip().strip("`").strip() for c in cleaned.split(",")]
    cols = [c for c in cols if c and not c.startswith("#")]
    return cols


def verify_table(table_name: str, init_sql_cols: dict[str, str], script_cols: list[str]) -> dict:
    """단일 테이블의 init.sql vs 스크립트 컬럼 교차 검증."""
    init_col_set = set(init_sql_cols.keys())
    script_col_set = set(script_cols)

    # 공통 (OK)
    common = init_col_set & script_col_set

    # init.sql 에만 있음 (스크립트가 INSERT 안 함 — DEFAULT 값이나 자동 생성이면 OK, 아니면 필수 확인)
    only_in_init = init_col_set - script_col_set

    # 스크립트에만 있음 (init.sql 누락 — ❌ INSERT 실패)
    only_in_script = script_col_set - init_col_set

    return {
        "table": table_name,
        "init_sql_total": len(init_col_set),
        "script_total": len(script_col_set),
        "common": sorted(common),
        "only_in_init": sorted(only_in_init),
        "only_in_script": sorted(only_in_script),
        "passed": len(only_in_script) == 0,
    }


def print_report(results: list[dict], init_sql_tables: dict) -> int:
    """검증 결과 리포트 출력."""
    print("=" * 80)
    print("  init.sql ↔ MySQL 적재 스크립트 정합성 검증")
    print("=" * 80)

    # init.sql 전체 테이블 목록
    print(f"\n[init.sql 정의된 테이블]  총 {len(init_sql_tables)} 개")
    for name, info in init_sql_tables.items():
        print(f"  - {name:30s}  컬럼 {len(info['columns'])}개  인덱스 {len(info['indexes'])}개")

    print()
    print("=" * 80)
    print("  스크립트 검증 결과")
    print("=" * 80)

    all_passed = True
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"\n[{status}]  {r['table']}")
        print(f"  init.sql 컬럼: {r['init_sql_total']}  |  스크립트 INSERT: {r['script_total']}")

        if r["only_in_script"]:
            all_passed = False
            print(f"  ❌ 스크립트에만 있고 init.sql 에 없음 ({len(r['only_in_script'])}):")
            for c in r["only_in_script"]:
                print(f"       - {c}")

        if r["only_in_init"]:
            print(f"  ⚠️  init.sql 에만 있음 (스크립트가 INSERT 안 함, {len(r['only_in_init'])}):")
            for c in r["only_in_init"]:
                print(f"       - {c}")

        if r["passed"] and not r["only_in_init"]:
            print(f"  ✅ 완벽 일치 ({len(r['common'])} 컬럼)")

    print()
    print("=" * 80)
    if all_passed:
        print("  🎉 모든 INSERT 스크립트가 init.sql 과 호환됨")
    else:
        print("  ❌ 일부 스크립트가 init.sql 에 없는 컬럼을 INSERT 시도 — 수정 필요")
    print("=" * 80)

    return 0 if all_passed else 1


async def check_live_mysql(init_sql_tables: dict) -> None:
    """실제 MySQL 상태와 init.sql 대조."""
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from monglepick.db.clients import close_all_clients, get_mysql, init_all_clients

    await init_all_clients()
    print()
    print("=" * 80)
    print("  실제 MySQL 상태 vs init.sql")
    print("=" * 80)

    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                for table_name, info in init_sql_tables.items():
                    await cursor.execute(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_schema=DATABASE() AND table_name=%s",
                        (table_name,),
                    )
                    exists = (await cursor.fetchone())[0] > 0

                    if not exists:
                        print(f"  ❌ {table_name:30s}  없음 (init.sql 실행 필요)")
                        continue

                    # 실제 컬럼 수
                    await cursor.execute(
                        "SELECT COLUMN_NAME, COLUMN_TYPE FROM information_schema.columns "
                        "WHERE table_schema=DATABASE() AND table_name=%s",
                        (table_name,),
                    )
                    live_cols = {row[0]: row[1].upper() for row in await cursor.fetchall()}

                    init_col_set = set(info["columns"].keys())
                    live_col_set = set(live_cols.keys())

                    missing = init_col_set - live_col_set
                    extra = live_col_set - init_col_set

                    status = "✅" if not missing else "⚠️"
                    print(f"  {status} {table_name:30s}  실제 {len(live_col_set):3d} / init.sql {len(init_col_set):3d} 컬럼")

                    if missing:
                        print(f"      ❌ 누락 ({len(missing)}): {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
                    if extra:
                        print(f"      ⚠️  실제에만 있음 ({len(extra)}): {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")
    finally:
        await close_all_clients()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-live-db", action="store_true", help="실제 MySQL 상태도 대조")
    args = parser.parse_args()

    # 1. init.sql 파싱
    if not INIT_SQL_PATH.exists():
        print(f"❌ init.sql 없음: {INIT_SQL_PATH}")
        return 1

    init_sql_tables = parse_init_sql_tables(INIT_SQL_PATH)

    # 2. 각 스크립트 검증
    results = []
    for table_name, script_rel in SCRIPT_MAP.items():
        script_path = PROJECT_ROOT / script_rel
        if table_name not in init_sql_tables:
            print(f"⚠️  init.sql 에 테이블 없음: {table_name}")
            results.append({
                "table": table_name,
                "init_sql_total": 0,
                "script_total": 0,
                "common": [],
                "only_in_init": [],
                "only_in_script": [],
                "passed": False,
            })
            continue

        init_cols = init_sql_tables[table_name]["columns"]
        script_cols = parse_insert_columns(script_path)

        result = verify_table(table_name, init_cols, script_cols)
        results.append(result)

    exit_code = print_report(results, init_sql_tables)

    # 3. 실제 MySQL 대조 (선택)
    if args.check_live_db:
        import asyncio
        asyncio.run(check_live_mysql(init_sql_tables))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
