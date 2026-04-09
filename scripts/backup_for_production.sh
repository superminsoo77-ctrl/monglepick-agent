#!/usr/bin/env bash
#
# 운영 배포용 5DB 전체 덤프 스크립트.
#
# 로컬 Mac (Tailscale 100.73.239.117) 에서 수집/보강 완료된 5DB 를
# 운영 VM4 (10.20.0.10) 로 이관하기 위한 덤프 생성.
#
# 덤프 대상:
#   1. Qdrant — movies + persons 컬렉션 snapshot
#   2. Elasticsearch — movies_bm25 인덱스 snapshot (repository 설정 필요)
#   3. Neo4j — neo4j-admin database dump
#   4. MySQL — mysqldump (movies + persons + kaggle_watch_history +
#              box_office_daily + movie_external_ratings + 기타 운영 테이블)
#   5. Redis — BGSAVE + dump.rdb 복사 (CF 캐시 포함)
#
# 출력: db_dumps/prod_migration_YYYYMMDD_HHMMSS/
#
# 사용법:
#   bash scripts/backup_for_production.sh
#   bash scripts/backup_for_production.sh --skip-es         # ES snapshot 건너뛰기
#   bash scripts/backup_for_production.sh --skip-redis      # Redis 건너뛰기 (운영에서 재구축)
#
# 운영 이관:
#   scp -r db_dumps/prod_migration_<ts>/  \
#       ubuntu@210.109.15.187:~/monglepick-migration/
#   ssh -A -J ubuntu@210.109.15.187 ubuntu@10.20.0.10
#   # 각 DB 별 복원 절차는 docs/production_migration_2026-04-09.md 참조

set -euo pipefail

# ──────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f .env ]]; then
    echo "[ERROR] .env not found in ${PROJECT_ROOT}"
    exit 1
fi

# .env 로드
set -a
source .env
set +a

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DUMP_DIR="db_dumps/prod_migration_${TIMESTAMP}"
mkdir -p "${DUMP_DIR}"

LOG_FILE="${DUMP_DIR}/backup.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================================="
echo "  운영 배포 덤프 — ${TIMESTAMP}"
echo "  출력: ${DUMP_DIR}"
echo "=========================================================="
echo

# ──────────────────────────────────────────────────────────────
# 인자 파싱
# ──────────────────────────────────────────────────────────────

SKIP_ES=0
SKIP_REDIS=0
SKIP_QDRANT=0
SKIP_NEO4J=0
SKIP_MYSQL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-es) SKIP_ES=1; shift ;;
        --skip-redis) SKIP_REDIS=1; shift ;;
        --skip-qdrant) SKIP_QDRANT=1; shift ;;
        --skip-neo4j) SKIP_NEO4J=1; shift ;;
        --skip-mysql) SKIP_MYSQL=1; shift ;;
        -h|--help)
            grep '^#' "$0" | head -35
            exit 0
            ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# ──────────────────────────────────────────────────────────────
# 1. Qdrant — movies + persons snapshot
# ──────────────────────────────────────────────────────────────

if [[ ${SKIP_QDRANT} -eq 0 ]]; then
    echo "[1/5] Qdrant snapshot 생성"
    echo "─────────────────────────────────────────────"

    QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

    for COLLECTION in movies persons; do
        echo "  - Creating snapshot for '${COLLECTION}'..."

        RESPONSE=$(curl -s -X POST "${QDRANT_URL}/collections/${COLLECTION}/snapshots" \
            -H "Content-Type: application/json" || echo '{"error": "collection may not exist"}')

        SNAPSHOT_NAME=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('result', {}).get('name', ''))
except Exception:
    print('')
")

        if [[ -n "${SNAPSHOT_NAME}" ]]; then
            echo "    snapshot: ${SNAPSHOT_NAME}"
            # 다운로드
            curl -s -o "${DUMP_DIR}/qdrant_${COLLECTION}_${SNAPSHOT_NAME}" \
                "${QDRANT_URL}/collections/${COLLECTION}/snapshots/${SNAPSHOT_NAME}"
            SIZE_MB=$(du -m "${DUMP_DIR}/qdrant_${COLLECTION}_${SNAPSHOT_NAME}" | cut -f1)
            echo "    downloaded: ${SIZE_MB} MB"
        else
            echo "    ⚠️  snapshot 생성 실패 (컬렉션 없음?): ${COLLECTION}"
        fi
    done
    echo
else
    echo "[1/5] Qdrant SKIP"
fi

# ──────────────────────────────────────────────────────────────
# 2. Elasticsearch — movies_bm25 인덱스
# ──────────────────────────────────────────────────────────────

if [[ ${SKIP_ES} -eq 0 ]]; then
    echo "[2/5] Elasticsearch 인덱스 덤프"
    echo "─────────────────────────────────────────────"

    ES_URL="${ELASTICSEARCH_URL:-http://localhost:9200}"

    # 인덱스 설정/매핑 저장
    curl -s "${ES_URL}/movies_bm25/_settings?pretty" > "${DUMP_DIR}/es_settings.json"
    curl -s "${ES_URL}/movies_bm25/_mapping?pretty" > "${DUMP_DIR}/es_mapping.json"

    # docs.count 기록
    curl -s "${ES_URL}/_cat/indices/movies_bm25?h=docs.count,store.size" \
        > "${DUMP_DIR}/es_count.txt" || true

    # 전체 bulk export (elasticdump 없으면 Python 스트리밍)
    echo "  - Exporting via Python scroll API..."
    python3 << PYEOF > "${DUMP_DIR}/es_movies_bm25.jsonl" 2>/dev/null || true
import json, urllib.request

ES_URL = "${ES_URL}"
scroll_url = f"{ES_URL}/movies_bm25/_search?scroll=5m&size=1000"
req = urllib.request.Request(
    scroll_url,
    data=json.dumps({"query": {"match_all": {}}, "sort": ["_doc"]}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.loads(resp.read())

scroll_id = data.get("_scroll_id")
hits = data["hits"]["hits"]
total_written = 0
while hits:
    for h in hits:
        print(json.dumps({"id": h["_id"], "doc": h["_source"]}))
        total_written += 1

    # 다음 배치
    req2 = urllib.request.Request(
        f"{ES_URL}/_search/scroll",
        data=json.dumps({"scroll": "5m", "scroll_id": scroll_id}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req2, timeout=60) as resp2:
            data = json.loads(resp2.read())
    except Exception:
        break
    scroll_id = data.get("_scroll_id")
    hits = data["hits"]["hits"]

import sys
sys.stderr.write(f"ES exported: {total_written:,} docs\n")
PYEOF

    ES_LINES=$(wc -l < "${DUMP_DIR}/es_movies_bm25.jsonl" || echo 0)
    echo "  - Exported: ${ES_LINES} docs"
    echo
else
    echo "[2/5] ES SKIP"
fi

# ──────────────────────────────────────────────────────────────
# 3. Neo4j — neo4j-admin database dump
# ──────────────────────────────────────────────────────────────

if [[ ${SKIP_NEO4J} -eq 0 ]]; then
    echo "[3/5] Neo4j database dump"
    echo "─────────────────────────────────────────────"

    # Docker 컨테이너 이름
    NEO4J_CONTAINER="monglepick-neo4j"

    # 컨테이너 안에서 dump → /tmp → host 로 복사
    NEO4J_TMP_PATH="/tmp/neo4j_dump_${TIMESTAMP}"

    echo "  - Stopping Neo4j database (dump 동안만)..."
    # neo4j-admin dump 는 offline 모드 필요 (Community 버전)
    # 대신 online 모드로는 backup API (Enterprise 전용). Community 는 offline 만 지원.
    # → docker exec stop → dump → start 순서
    docker exec "${NEO4J_CONTAINER}" neo4j stop || true
    sleep 3

    echo "  - Running neo4j-admin database dump..."
    docker exec "${NEO4J_CONTAINER}" mkdir -p "${NEO4J_TMP_PATH}"
    docker exec "${NEO4J_CONTAINER}" neo4j-admin database dump neo4j \
        --to-path="${NEO4J_TMP_PATH}" --overwrite-destination=true

    echo "  - Copying dump to host..."
    docker cp "${NEO4J_CONTAINER}:${NEO4J_TMP_PATH}/neo4j.dump" "${DUMP_DIR}/neo4j.dump"

    echo "  - Cleanup + Neo4j restart..."
    docker exec "${NEO4J_CONTAINER}" rm -rf "${NEO4J_TMP_PATH}" || true
    docker exec -d "${NEO4J_CONTAINER}" neo4j start || true
    sleep 5

    if [[ -f "${DUMP_DIR}/neo4j.dump" ]]; then
        SIZE_MB=$(du -m "${DUMP_DIR}/neo4j.dump" | cut -f1)
        echo "  ✅ Neo4j dump: ${SIZE_MB} MB"
    else
        echo "  ❌ Neo4j dump 실패"
    fi
    echo
else
    echo "[3/5] Neo4j SKIP"
fi

# ──────────────────────────────────────────────────────────────
# 4. MySQL — mysqldump
# ──────────────────────────────────────────────────────────────

if [[ ${SKIP_MYSQL} -eq 0 ]]; then
    echo "[4/5] MySQL mysqldump"
    echo "─────────────────────────────────────────────"

    MYSQL_CONTAINER="monglepick-mysql"
    MYSQL_DB="${MYSQL_DB:-monglepick}"
    MYSQL_USER="${MYSQL_USER:-monglepick}"

    echo "  - Dumping database '${MYSQL_DB}'..."
    docker exec -e MYSQL_PWD="${MYSQL_PASSWORD:-${MYSQL_PWD:-}}" \
        "${MYSQL_CONTAINER}" mysqldump \
        --single-transaction \
        --quick \
        --lock-tables=false \
        --no-tablespaces \
        -u "${MYSQL_USER}" "${MYSQL_DB}" \
        > "${DUMP_DIR}/mysql_${MYSQL_DB}.sql" 2>"${DUMP_DIR}/mysql_dump_stderr.log" || {
            echo "  ⚠️  mysqldump 경고 — stderr: ${DUMP_DIR}/mysql_dump_stderr.log"
        }

    if [[ -f "${DUMP_DIR}/mysql_${MYSQL_DB}.sql" ]]; then
        SIZE_MB=$(du -m "${DUMP_DIR}/mysql_${MYSQL_DB}.sql" | cut -f1)
        LINES=$(wc -l < "${DUMP_DIR}/mysql_${MYSQL_DB}.sql")
        echo "  ✅ MySQL dump: ${SIZE_MB} MB, ${LINES} lines"

        # 테이블 목록 기록
        grep -oE 'CREATE TABLE [^ ]+' "${DUMP_DIR}/mysql_${MYSQL_DB}.sql" \
            > "${DUMP_DIR}/mysql_tables.txt" || true
        TABLE_COUNT=$(wc -l < "${DUMP_DIR}/mysql_tables.txt")
        echo "  - Tables: ${TABLE_COUNT}"
    else
        echo "  ❌ MySQL dump 실패"
    fi
    echo
else
    echo "[4/5] MySQL SKIP"
fi

# ──────────────────────────────────────────────────────────────
# 5. Redis — BGSAVE + dump.rdb 복사 (CF 캐시)
# ──────────────────────────────────────────────────────────────

if [[ ${SKIP_REDIS} -eq 0 ]]; then
    echo "[5/5] Redis BGSAVE + dump.rdb"
    echo "─────────────────────────────────────────────"

    REDIS_CONTAINER="monglepick-redis"

    echo "  - Triggering BGSAVE..."
    docker exec "${REDIS_CONTAINER}" redis-cli BGSAVE || true

    # BGSAVE 완료 대기
    for i in {1..30}; do
        STATUS=$(docker exec "${REDIS_CONTAINER}" redis-cli LASTSAVE 2>/dev/null || echo "0")
        sleep 2
        NEW_STATUS=$(docker exec "${REDIS_CONTAINER}" redis-cli LASTSAVE 2>/dev/null || echo "0")
        if [[ "${STATUS}" != "${NEW_STATUS}" ]]; then
            echo "  - BGSAVE 완료"
            break
        fi
        if [[ $i -eq 30 ]]; then
            echo "  - BGSAVE 대기 타임아웃 (60초) — 기존 dump 복사 시도"
        fi
    done

    # dump.rdb 복사
    docker cp "${REDIS_CONTAINER}:/data/dump.rdb" "${DUMP_DIR}/redis_dump.rdb" 2>/dev/null || {
        echo "  ⚠️  /data/dump.rdb 없음 — 경로 확인 필요"
    }

    if [[ -f "${DUMP_DIR}/redis_dump.rdb" ]]; then
        SIZE_MB=$(du -m "${DUMP_DIR}/redis_dump.rdb" | cut -f1)
        echo "  ✅ Redis dump: ${SIZE_MB} MB"

        # CF 키 카운트 기록
        CF_KEYS=$(docker exec "${REDIS_CONTAINER}" redis-cli --scan --pattern 'cf:*' | wc -l || echo 0)
        echo "  - CF keys: ${CF_KEYS}"
        echo "${CF_KEYS}" > "${DUMP_DIR}/redis_cf_key_count.txt"
    else
        echo "  ❌ Redis dump 실패"
    fi
    echo
else
    echo "[5/5] Redis SKIP"
fi

# ──────────────────────────────────────────────────────────────
# 메타데이터 + 체크섬
# ──────────────────────────────────────────────────────────────

echo "[메타데이터] 체크섬 + summary 생성"
echo "─────────────────────────────────────────────"

# 파일 목록 + 크기
ls -lh "${DUMP_DIR}" > "${DUMP_DIR}/files.txt"

# SHA256 체크섬
(cd "${DUMP_DIR}" && find . -type f ! -name "sha256.txt" ! -name "files.txt" -exec shasum -a 256 {} \; > sha256.txt)

# 수집 요약
cat > "${DUMP_DIR}/SUMMARY.md" <<SUMMARY
# 운영 배포 덤프 — ${TIMESTAMP}

## 포함 내용
- Qdrant: movies + persons snapshot
- Elasticsearch: movies_bm25 인덱스 (JSONL export)
- Neo4j: neo4j-admin dump
- MySQL: 전체 DB (monglepick)
- Redis: dump.rdb (CF 캐시 포함)

## 총 크기
$(du -sh "${DUMP_DIR}" | cut -f1)

## 파일 목록
$(cat "${DUMP_DIR}/files.txt")

## 운영 복원 절차
1. SCP 전송:
   scp -r ${DUMP_DIR}/ ubuntu@210.109.15.187:~/monglepick-migration/

2. VM4 (10.20.0.10) 복원:
   a) Qdrant: POST /collections/{name}/snapshots/{file}/recover
   b) ES: bulk insert from es_movies_bm25.jsonl
   c) Neo4j: neo4j-admin database load neo4j --from-path=...
   d) MySQL: mysql < mysql_monglepick.sql
   e) Redis: stop + cp dump.rdb /data/ + start

3. 운영 Agent 재기동 + 최종 검증:
   PYTHONPATH=src uv run python scripts/run_final_verification.py

## 생성 시각
${TIMESTAMP}
SUMMARY

echo "  - SUMMARY.md 생성"
echo "  - sha256.txt 생성"
echo "  - files.txt 생성"
echo

# ──────────────────────────────────────────────────────────────
# 최종 요약
# ──────────────────────────────────────────────────────────────

TOTAL_SIZE=$(du -sh "${DUMP_DIR}" | cut -f1)

echo "=========================================================="
echo "  덤프 완료"
echo "=========================================================="
echo "  위치: ${DUMP_DIR}"
echo "  총 크기: ${TOTAL_SIZE}"
echo
echo "  다음 단계:"
echo "    cat ${DUMP_DIR}/SUMMARY.md"
echo "    scp -r ${DUMP_DIR}/ ubuntu@210.109.15.187:~/monglepick-migration/"
echo "=========================================================="
