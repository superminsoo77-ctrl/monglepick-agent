#!/usr/bin/env bash
#
# Upstage 키 atomic swap 스크립트.
#
# .env 의 UPSTAGE_API_KEY 와 UPSTAGE_API_KEY2 값을 안전하게 교체한다.
#
# 절차:
#   1. .env 백업 (.env.bak.YYYYMMDD_HHMMSS)
#   2. KEY1 / KEY2 의 현재 값 추출
#   3. 둘이 모두 비어있지 않은지 검증
#   4. 임시 파일에 swap 결과 작성
#   5. 검증 (양쪽 모두 채워졌고 길이가 같은지)
#   6. 원본 .env 교체 (atomic mv)
#
# 사용법:
#   bash scripts/swap_upstage_keys.sh                  # swap 실행
#   bash scripts/swap_upstage_keys.sh --dry-run        # 미리보기
#   bash scripts/swap_upstage_keys.sh --restore <backup>  # 복원

set -euo pipefail

ENV_FILE=".env"

usage() {
  cat <<EOF
Usage: $0 [--dry-run|--restore <backup_file>]

옵션:
  (없음)              .env 의 UPSTAGE_API_KEY ↔ UPSTAGE_API_KEY2 swap
  --dry-run           swap 결과를 출력만 (파일 변경 없음)
  --restore <file>    백업 파일에서 .env 복원
  -h, --help          이 도움말
EOF
  exit 0
}

# 인자 파싱
DRY_RUN=0
RESTORE_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --restore) RESTORE_FILE="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "[ERROR] 알 수 없는 옵션: $1"; usage ;;
  esac
done

# 복원 모드
if [[ -n "$RESTORE_FILE" ]]; then
  if [[ ! -f "$RESTORE_FILE" ]]; then
    echo "[ERROR] 백업 파일을 찾을 수 없음: $RESTORE_FILE"
    exit 1
  fi
  cp "$RESTORE_FILE" "$ENV_FILE"
  echo "[INFO] $ENV_FILE 복원 완료 (from $RESTORE_FILE)"
  exit 0
fi

# .env 존재 확인
if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] $ENV_FILE 파일이 없습니다. 현재 디렉토리: $(pwd)"
  exit 1
fi

# 키 값 추출 (앞뒤 공백 제거)
KEY1_VALUE=$(grep -E "^UPSTAGE_API_KEY=" "$ENV_FILE" | head -1 | sed 's/^UPSTAGE_API_KEY=//' | tr -d '[:space:]')
KEY2_VALUE=$(grep -E "^UPSTAGE_API_KEY2=" "$ENV_FILE" | head -1 | sed 's/^UPSTAGE_API_KEY2=//' | tr -d '[:space:]')

# 검증
if [[ -z "$KEY1_VALUE" ]]; then
  echo "[ERROR] UPSTAGE_API_KEY 가 비어있음"
  exit 1
fi

if [[ -z "$KEY2_VALUE" ]]; then
  echo "[ERROR] UPSTAGE_API_KEY2 가 비어있음 — 백업 키가 .env 에 설정되지 않았습니다"
  exit 1
fi

if [[ "$KEY1_VALUE" == "$KEY2_VALUE" ]]; then
  echo "[ERROR] KEY1 과 KEY2 가 동일합니다 — swap 의미 없음"
  exit 1
fi

KEY1_PREFIX="${KEY1_VALUE:0:10}..."
KEY2_PREFIX="${KEY2_VALUE:0:10}..."

echo "[INFO] 현재 키 상태:"
echo "  UPSTAGE_API_KEY  = $KEY1_PREFIX (length ${#KEY1_VALUE})"
echo "  UPSTAGE_API_KEY2 = $KEY2_PREFIX (length ${#KEY2_VALUE})"
echo
echo "[INFO] swap 후 결과:"
echo "  UPSTAGE_API_KEY  = $KEY2_PREFIX (length ${#KEY2_VALUE})"
echo "  UPSTAGE_API_KEY2 = $KEY1_PREFIX (length ${#KEY1_VALUE})"

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "[DRY-RUN] 파일 변경 없음. 실제 swap 하려면 --dry-run 없이 실행하세요."
  exit 0
fi

# 백업
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${ENV_FILE}.bak.${TIMESTAMP}"
cp "$ENV_FILE" "$BACKUP_FILE"
echo
echo "[INFO] 백업: $BACKUP_FILE"

# 임시 파일에 swap 결과 작성
TMP_FILE="${ENV_FILE}.tmp.$$"

# Python 으로 안전 처리 (sed 의 특수문자 이슈 회피)
python3 - <<EOF > "$TMP_FILE"
import re

with open("$ENV_FILE", "r", encoding="utf-8") as f:
    lines = f.readlines()

key1 = "$KEY1_VALUE"
key2 = "$KEY2_VALUE"

new_lines = []
key1_swapped = False
key2_swapped = False

for line in lines:
    stripped = line.strip()
    if stripped.startswith("UPSTAGE_API_KEY=") and not stripped.startswith("UPSTAGE_API_KEY2="):
        new_lines.append(f"UPSTAGE_API_KEY={key2}\n")
        key1_swapped = True
    elif stripped.startswith("UPSTAGE_API_KEY2="):
        new_lines.append(f"UPSTAGE_API_KEY2={key1}\n")
        key2_swapped = True
    else:
        new_lines.append(line)

import sys
if not key1_swapped or not key2_swapped:
    sys.stderr.write(f"[ERROR] swap 실패: KEY1={key1_swapped}, KEY2={key2_swapped}\n")
    sys.exit(1)

sys.stdout.write("".join(new_lines))
EOF

# 검증: 임시 파일에서 키 추출
NEW_KEY1=$(grep -E "^UPSTAGE_API_KEY=" "$TMP_FILE" | head -1 | sed 's/^UPSTAGE_API_KEY=//' | tr -d '[:space:]')
NEW_KEY2=$(grep -E "^UPSTAGE_API_KEY2=" "$TMP_FILE" | head -1 | sed 's/^UPSTAGE_API_KEY2=//' | tr -d '[:space:]')

if [[ "$NEW_KEY1" != "$KEY2_VALUE" ]] || [[ "$NEW_KEY2" != "$KEY1_VALUE" ]]; then
  echo "[ERROR] swap 검증 실패!"
  echo "  expected NEW_KEY1=$KEY2_PREFIX, got=${NEW_KEY1:0:10}..."
  echo "  expected NEW_KEY2=$KEY1_PREFIX, got=${NEW_KEY2:0:10}..."
  rm -f "$TMP_FILE"
  exit 1
fi

# atomic mv
mv "$TMP_FILE" "$ENV_FILE"

echo
echo "[SUCCESS] swap 완료!"
echo "  새 UPSTAGE_API_KEY  = $KEY2_PREFIX"
echo "  새 UPSTAGE_API_KEY2 = $KEY1_PREFIX (백업 보존)"
echo
echo "[INFO] 복원하려면:"
echo "  bash $0 --restore $BACKUP_FILE"
