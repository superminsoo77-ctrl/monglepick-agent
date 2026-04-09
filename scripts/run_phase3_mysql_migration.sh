#!/usr/bin/env bash
#
# Phase 3 실행 자동화 스크립트.
#
# 절차:
#   1. mysqldump 백업
#   2. migration_2026-04-09_agent_tables_reset.sql 실행 (Agent 소유 5 테이블 DROP)
#   3. init.sql 실행 (61 컬럼 movies + 4 신규 테이블 CREATE)
#   4. verify_mysql_schema.py --check-live-db 로 DDL 검증
#   5. run_mysql_sync.py 실행 (Qdrant → MySQL)
#   6. run_kaggle_ratings_load.py 실행 (26M ratings)
#   7. 최종 카운트 검증
#
# 사용법:
#   bash scripts/run_phase3_mysql_migration.sh
#   bash scripts/run_phase3_mysql_migration.sh --dry-run   # 실행 없이 순서만 출력
#
# 중단 시: 각 단계는 멱등하므로 재실행 안전.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MONGLEPICK_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

cd "${PROJECT_ROOT}"

# .env 로드
set -a
source .env
set +a

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DUMP_DIR="${MONGLEPICK_ROOT}/db_dumps"
BACKUP_FILE="${DUMP_DIR}/monglepick_pre_phase3_${TIMESTAMP}.sql"
MIGRATION_FILE="${DUMP_DIR}/migration_2026-04-09_agent_tables_reset.sql"
INIT_SQL_FILE="${DUMP_DIR}/prod_old_backup/init.sql"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

echo "=========================================================="
echo "  Phase 3 MySQL 마이그레이션 + 재적재"
echo "  ${TIMESTAMP}"
echo "=========================================================="
echo
echo "  DRY_RUN: $DRY_RUN"
echo "  DUMP_DIR: ${DUMP_DIR}"
echo "  BACKUP: ${BACKUP_FILE}"
echo "  MIGRATION: ${MIGRATION_FILE}"
echo "  INIT_SQL: ${INIT_SQL_FILE}"
echo

_run() {
    echo ">>> $*"
    if [[ $DRY_RUN -eq 0 ]]; then
        eval "$@"
    fi
}

# ──────────────────────────────────────────────────────────────
# Step 1: 백업
# ──────────────────────────────────────────────────────────────

echo
echo "[Step 1/7] mysqldump 백업"
echo "─────────────────────────────"
_run "docker exec -e MYSQL_PWD=\"${MYSQL_PASSWORD}\" monglepick-mysql \
        mysqldump --single-transaction --quick --lock-tables=false --no-tablespaces \
        -u${MYSQL_USER} ${MYSQL_DATABASE} \
        > \"${BACKUP_FILE}\""

if [[ $DRY_RUN -eq 0 && -f "${BACKUP_FILE}" ]]; then
    BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
    echo "  ✅ 백업: ${BACKUP_FILE} (${BACKUP_SIZE})"
fi

# ──────────────────────────────────────────────────────────────
# Step 2: 마이그레이션 SQL 실행 (5 테이블 DROP)
# ──────────────────────────────────────────────────────────────

echo
echo "[Step 2/7] Agent 소유 5 테이블 DROP"
echo "─────────────────────────────"
_run "docker exec -i -e MYSQL_PWD=\"${MYSQL_PASSWORD}\" monglepick-mysql \
        mysql -u${MYSQL_USER} ${MYSQL_DATABASE} \
        < \"${MIGRATION_FILE}\""

# ──────────────────────────────────────────────────────────────
# Step 3: init.sql 실행 (5 테이블 재생성)
# ──────────────────────────────────────────────────────────────

echo
echo "[Step 3/7] init.sql 실행 — 61컬럼 movies + 4 신규 테이블"
echo "─────────────────────────────"
_run "docker exec -i -e MYSQL_PWD=\"${MYSQL_PASSWORD}\" monglepick-mysql \
        mysql -u${MYSQL_USER} ${MYSQL_DATABASE} \
        < \"${INIT_SQL_FILE}\""

# ──────────────────────────────────────────────────────────────
# Step 4: DDL 검증
# ──────────────────────────────────────────────────────────────

echo
echo "[Step 4/7] DDL 검증 (verify_mysql_schema.py --check-live-db)"
echo "─────────────────────────────"
_run "PYTHONPATH=src uv run python scripts/verify_mysql_schema.py --check-live-db 2>&1 \
        | grep -E '^\s+[✅❌⚠️]' | head -10"

# ──────────────────────────────────────────────────────────────
# Step 5: Qdrant → MySQL movies sync (59 col)
# ──────────────────────────────────────────────────────────────

echo
echo "[Step 5/7] run_mysql_sync.py — Qdrant → MySQL movies 59 col"
echo "─────────────────────────────"
_run "PYTHONPATH=src uv run python scripts/run_mysql_sync.py 2>&1 | tee logs/phase3_mysql_sync.log"

# ──────────────────────────────────────────────────────────────
# Step 6: Kaggle ratings 26M 적재
# ──────────────────────────────────────────────────────────────

echo
echo "[Step 6/7] run_kaggle_ratings_load.py — 26M ratings → kaggle_watch_history"
echo "─────────────────────────────"
_run "PYTHONPATH=src uv run python scripts/run_kaggle_ratings_load.py 2>&1 | tee logs/phase3_kaggle_ratings.log"

# ──────────────────────────────────────────────────────────────
# Step 7: 최종 카운트 검증
# ──────────────────────────────────────────────────────────────

echo
echo "[Step 7/7] 최종 카운트 검증"
echo "─────────────────────────────"
_run "docker exec -e MYSQL_PWD=\"${MYSQL_PASSWORD}\" monglepick-mysql \
        mysql -u${MYSQL_USER} ${MYSQL_DATABASE} -Nse \"
            SELECT 'movies' AS t, COUNT(*) FROM movies
            UNION SELECT 'persons', COUNT(*) FROM persons
            UNION SELECT 'kaggle_watch_history', COUNT(*) FROM kaggle_watch_history
            UNION SELECT 'box_office_daily', COUNT(*) FROM box_office_daily
            UNION SELECT 'movie_external_ratings', COUNT(*) FROM movie_external_ratings;
        \""

echo
echo "=========================================================="
echo "  Phase 3 완료"
echo "=========================================================="
echo "  백업: ${BACKUP_FILE}"
echo "  다음: Phase 4-A resume → Phase 4-B Person LLM"
