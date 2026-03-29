#!/usr/bin/env bash
# =============================================================================
# 몽글이 Hybrid LLM 모드 전환 스크립트 (M-LLM-7)
#
# 목적:
#   EXAONE 4.0 1.2B LoRA 파인튜닝이 완료된 몽글이 모델을 Ollama에 등록하고,
#   LLM_MODE=hybrid 로 전환하는 절차를 단계별로 안내한다.
#
# 전환 효과 (설계서 §1-4):
#   - 일반 대화 / 후속 질문: 몽글이 (1.2B, ~50+ tok/s) → 응답 속도 대폭 향상
#   - 의도 분류 / 선호 추출 / 추천 이유 / 이미지 분석: Solar API → 품질 유지
#   - Ollama VRAM 사용량: ~40GB (32B+35B 경합) → ~1GB (1.2B 단일)
#
# 전제 조건 (실행 전 확인 필수):
#   1. 몽글이 GGUF 파일이 준비되어 있어야 한다:
#      models/mongle-exaone4-q4km.gguf
#   2. Modelfile.mongle 이 프로젝트 루트에 있어야 한다
#   3. Ollama가 실행 중이어야 한다 (기본: localhost:11434)
#   4. evaluate_mongle.py 실행 결과가 PASS여야 한다
#   5. .env 파일이 존재해야 한다
#
# 사용법:
#   bash scripts/switch_to_hybrid.sh
#
# =============================================================================

set -euo pipefail  # 오류 즉시 종료, 미선언 변수 오류, 파이프 오류 전파

# ── 색상 코드 (터미널 가독성) ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

# ── 프로젝트 루트 자동 감지 ──
# 스크립트가 scripts/ 폴더 안에 있을 때 부모 디렉토리가 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 설정값 (변경 필요 시 이 부분만 수정) ──
MONGLE_MODEL_NAME="mongle"                                    # Ollama 모델 등록 이름
MODELFILE_PATH="${PROJECT_ROOT}/Modelfile.mongle"             # Ollama Modelfile 경로
GGUF_PATH="${PROJECT_ROOT}/models/mongle-exaone4-q4km.gguf"  # GGUF 파일 경로
ENV_FILE="${PROJECT_ROOT}/.env"                               # 환경변수 파일 경로
OLLAMA_URL="http://localhost:11434"                           # Ollama 서버 URL
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_mongle.py"                # 품질 평가 스크립트
EVAL_DATA="${PROJECT_ROOT}/data/finetune/mongle_eval.jsonl"   # 평가 데이터
EVAL_OUTPUT="${PROJECT_ROOT}/data/finetune/eval_results.json" # 평가 결과 저장

# =============================================================================
# 유틸리티 함수
# =============================================================================

# 헤더 출력
print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}==============================================================================${RESET}"
    echo -e "${BOLD}${BLUE}  $1${RESET}"
    echo -e "${BOLD}${BLUE}==============================================================================${RESET}"
}

# 단계 출력
print_step() {
    local step="$1"
    local desc="$2"
    echo ""
    echo -e "${BOLD}[${step}] ${desc}${RESET}"
}

# 성공 메시지
ok() {
    echo -e "  ${GREEN}OK${RESET}  $1"
}

# 경고 메시지
warn() {
    echo -e "  ${YELLOW}WARN${RESET} $1"
}

# 오류 메시지 후 종료
fail() {
    echo -e "  ${RED}FAIL${RESET} $1"
    echo ""
    echo -e "${RED}전환이 중단되었습니다. 위 오류를 해결하고 다시 실행하세요.${RESET}"
    exit 1
}

# 사용자 확인 (y/N)
confirm() {
    local msg="$1"
    echo -e "${YELLOW}  ? ${msg} [y/N]${RESET}"
    read -r answer
    [[ "${answer,,}" == "y" ]]
}

# =============================================================================
# STEP 0: 시작 안내
# =============================================================================

print_header "몽글이 Hybrid LLM 모드 전환 스크립트 (M-LLM-7)"
echo ""
echo "  이 스크립트는 다음 절차를 단계별로 수행합니다:"
echo "    1. Ollama 서버 및 모델 파일 존재 여부 확인"
echo "    2. mongle 모델 Ollama 등록 (이미 등록된 경우 생략)"
echo "    3. mongle 모델 응답 테스트"
echo "    4. 품질 평가 스크립트 실행 (evaluate_mongle.py)"
echo "    5. .env 파일에서 LLM_MODE=hybrid 전환"
echo "    6. 전환 완료 체크리스트 출력"
echo ""
echo "  프로젝트 루트: ${PROJECT_ROOT}"
echo ""

if ! confirm "전환 절차를 시작하시겠습니까?"; then
    echo "  전환을 취소했습니다."
    exit 0
fi

# =============================================================================
# STEP 1: 전제 조건 확인
# =============================================================================

print_step "1/6" "전제 조건 확인"

# 1-1. Ollama 실행 여부 확인
echo "  Ollama 서버 연결 확인 (${OLLAMA_URL})..."
if curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    ok "Ollama 서버 실행 중"
else
    fail "Ollama 서버에 연결할 수 없습니다.\n  ollama serve 를 먼저 실행하세요."
fi

# 1-2. .env 파일 존재 여부
if [[ -f "${ENV_FILE}" ]]; then
    ok ".env 파일 존재: ${ENV_FILE}"
else
    fail ".env 파일이 없습니다: ${ENV_FILE}\n  .env.example 을 복사하여 .env 를 만드세요."
fi

# 1-3. Modelfile.mongle 존재 여부
if [[ -f "${MODELFILE_PATH}" ]]; then
    ok "Modelfile.mongle 존재: ${MODELFILE_PATH}"
else
    fail "Modelfile.mongle 이 없습니다: ${MODELFILE_PATH}\n  설계서 §3-5를 참고하여 Modelfile.mongle 을 작성하세요."
fi

# 1-4. GGUF 파일 존재 여부
if [[ -f "${GGUF_PATH}" ]]; then
    GGUF_SIZE_MB=$(du -m "${GGUF_PATH}" | cut -f1)
    ok "GGUF 파일 존재: ${GGUF_PATH} (${GGUF_SIZE_MB} MB)"
    # Q4_K_M 기준 1.2B 모델은 약 700~900MB
    if [[ "${GGUF_SIZE_MB}" -lt 100 ]]; then
        warn "GGUF 파일 크기가 너무 작습니다 (${GGUF_SIZE_MB} MB). 변환이 정상적으로 완료되었는지 확인하세요."
    fi
else
    fail "GGUF 파일이 없습니다: ${GGUF_PATH}\n  설계서 §3-4의 Step 4 (GGUF 변환) 를 먼저 실행하세요:\n  python llama.cpp/convert_hf_to_gguf.py models/mongle-merged --outfile models/mongle-exaone4-f16.gguf\n  ./llama.cpp/llama-quantize models/mongle-exaone4-f16.gguf models/mongle-exaone4-q4km.gguf Q4_K_M"
fi

# 1-5. evaluate_mongle.py 존재 여부
if [[ -f "${EVAL_SCRIPT}" ]]; then
    ok "evaluate_mongle.py 존재"
else
    warn "evaluate_mongle.py 를 찾을 수 없습니다 (${EVAL_SCRIPT}). 품질 평가 단계를 건너뜁니다."
fi

# =============================================================================
# STEP 2: Ollama 모델 등록
# =============================================================================

print_step "2/6" "Ollama mongle 모델 등록 확인"

# 현재 등록된 모델 목록 조회
REGISTERED_MODELS=$(ollama list 2>/dev/null | awk '{print $1}' | tail -n +2 || echo "")

# mongle 모델이 이미 등록되어 있는지 확인 (태그 포함 부분 매칭)
if echo "${REGISTERED_MODELS}" | grep -q "^${MONGLE_MODEL_NAME}"; then
    ok "'${MONGLE_MODEL_NAME}' 모델이 이미 등록되어 있습니다."
    echo "  등록된 모델:"
    ollama list | grep "${MONGLE_MODEL_NAME}" | awk '{printf "    - %s\n", $0}'

    if confirm "기존 모델을 삭제하고 새로 등록하시겠습니까? (파인튜닝 재실행 후 업데이트 시)"; then
        echo "  기존 모델 삭제 중..."
        ollama rm "${MONGLE_MODEL_NAME}" && ok "기존 모델 삭제 완료" || warn "모델 삭제 실패 (무시하고 계속)"

        echo "  새 모델 등록 중 (GGUF: ${GGUF_SIZE_MB} MB, 시간이 걸릴 수 있습니다)..."
        ollama create "${MONGLE_MODEL_NAME}" -f "${MODELFILE_PATH}" \
            && ok "'${MONGLE_MODEL_NAME}' 모델 등록 완료" \
            || fail "모델 등록 실패. Modelfile.mongle 및 GGUF 경로를 확인하세요."
    else
        ok "기존 모델을 유지합니다."
    fi
else
    echo "  '${MONGLE_MODEL_NAME}' 모델이 등록되지 않았습니다. 새로 등록합니다..."
    echo "  GGUF 파일: ${GGUF_PATH} (${GGUF_SIZE_MB} MB)"
    echo "  등록 중... (시간이 걸릴 수 있습니다)"
    ollama create "${MONGLE_MODEL_NAME}" -f "${MODELFILE_PATH}" \
        && ok "'${MONGLE_MODEL_NAME}' 모델 등록 완료" \
        || fail "모델 등록 실패.\n  Modelfile.mongle의 FROM 경로가 GGUF 파일을 가리키는지 확인하세요.\n  FROM ${GGUF_PATH}"
fi

# 등록 후 최종 확인
echo ""
echo "  현재 Ollama 등록 모델 목록:"
ollama list | awk '{printf "    %s\n", $0}'

# =============================================================================
# STEP 3: 모델 응답 테스트
# =============================================================================

print_step "3/6" "mongle 모델 응답 테스트"

echo "  테스트 프롬프트: '안녕하세요! 영화 추천해줘'"
echo "  응답 (최초 실행 시 모델 로딩으로 시간이 걸릴 수 있습니다):"
echo "  ──────────────────────────────────────"

# Ollama REST API로 직접 테스트 (타임아웃 120초)
TEST_RESPONSE=$(curl -sf --max-time 120 "${OLLAMA_URL}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MONGLE_MODEL_NAME}\",
        \"prompt\": \"안녕하세요! 영화 추천해줘\",
        \"stream\": false
    }" 2>/dev/null || echo "")

if [[ -z "${TEST_RESPONSE}" ]]; then
    fail "모델 응답을 받지 못했습니다. Ollama 로그를 확인하세요: ollama logs"
fi

# jq가 있으면 response 필드만 추출, 없으면 전체 출력
if command -v jq &> /dev/null; then
    RESPONSE_TEXT=$(echo "${TEST_RESPONSE}" | jq -r '.response // empty' 2>/dev/null || echo "")
    EVAL_COUNT=$(echo "${TEST_RESPONSE}" | jq -r '.eval_count // 0' 2>/dev/null || echo "0")
    EVAL_DURATION_NS=$(echo "${TEST_RESPONSE}" | jq -r '.eval_duration // 0' 2>/dev/null || echo "0")
else
    # jq 없을 때 간단한 grep으로 추출 (완전하지 않을 수 있음)
    RESPONSE_TEXT=$(echo "${TEST_RESPONSE}" | grep -o '"response":"[^"]*"' | head -1 | sed 's/"response":"//;s/"$//' || echo "(응답 파싱 불가 — jq 설치 권장)")
    EVAL_COUNT="0"
    EVAL_DURATION_NS="0"
fi

if [[ -n "${RESPONSE_TEXT}" ]]; then
    echo "  ${RESPONSE_TEXT}" | fold -s -w 70 | sed 's/^/  /'
    echo "  ──────────────────────────────────────"

    # 응답 속도 계산 (eval_count / eval_duration_ns * 1e9)
    if [[ "${EVAL_COUNT}" -gt 0 ]] && [[ "${EVAL_DURATION_NS}" -gt 0 ]]; then
        TPS=$(echo "scale=1; ${EVAL_COUNT} * 1000000000 / ${EVAL_DURATION_NS}" | bc 2>/dev/null || echo "N/A")
        echo "  토큰 수: ${EVAL_COUNT}, 생성 속도: ~${TPS} tok/s"
    fi
    ok "응답 수신 확인"
else
    warn "응답 텍스트가 비어있습니다. 모델이 정상 작동하지 않을 수 있습니다."
fi

# =============================================================================
# STEP 4: 품질 평가
# =============================================================================

print_step "4/6" "품질 평가 (evaluate_mongle.py)"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
    warn "evaluate_mongle.py 가 없어 품질 평가를 건너뜁니다."
    EVAL_PASSED="skipped"
elif confirm "지금 품질 평가를 실행하시겠습니까? (--no-solar 옵션: 규칙 기반만, 빠름)"; then
    echo ""
    echo "  평가 방식 선택:"
    echo "    1) Solar API 채점 포함 (정확, UPSTAGE_API_KEY 필요)"
    echo "    2) 규칙 기반만 (빠름, API 키 불필요)"
    echo -e "${YELLOW}  ? 선택 [1/2]:${RESET}"
    read -r eval_mode

    SOLAR_FLAG=""
    if [[ "${eval_mode}" == "2" ]]; then
        SOLAR_FLAG="--no-solar"
        echo "  → 규칙 기반 모드로 평가합니다."
    else
        echo "  → Solar API 채점 모드로 평가합니다."
    fi

    echo ""
    echo "  평가 실행 중... (시간이 걸릴 수 있습니다)"
    echo ""

    # PYTHONPATH=src 로 evaluate_mongle.py 실행
    # uv가 없으면 python3 fallback
    EVAL_CMD="PYTHONPATH=${PROJECT_ROOT}/src"
    if command -v uv &> /dev/null; then
        EVAL_RUNNER="uv run python"
    else
        EVAL_RUNNER="python3"
    fi

    cd "${PROJECT_ROOT}"
    if PYTHONPATH="${PROJECT_ROOT}/src" ${EVAL_RUNNER} "${EVAL_SCRIPT}" \
        --model "${MONGLE_MODEL_NAME}" \
        --eval_data "${EVAL_DATA}" \
        --output "${EVAL_OUTPUT}" \
        ${SOLAR_FLAG}; then
        ok "품질 평가 PASS"
        EVAL_PASSED="pass"
    else
        EVAL_PASSED="fail"
        warn "품질 평가 FAIL. 결과: ${EVAL_OUTPUT}"
        echo ""
        echo "  FAIL 상태에서도 hybrid 전환을 계속할 수 있지만,"
        echo "  품질이 보장되지 않습니다."
        if ! confirm "품질 평가 FAIL에도 불구하고 hybrid 전환을 계속하시겠습니까?"; then
            echo ""
            echo "  evaluate_mongle.py 결과 파일: ${EVAL_OUTPUT}"
            echo "  FAIL 항목을 수정 후 다시 실행하세요."
            exit 1
        fi
        warn "FAIL 상태로 전환을 계속합니다."
    fi
else
    warn "품질 평가를 건너뜁니다."
    EVAL_PASSED="skipped"
fi

# =============================================================================
# STEP 5: .env 파일 LLM_MODE 전환
# =============================================================================

print_step "5/6" ".env 파일에서 LLM_MODE=hybrid 전환"

# 현재 LLM_MODE 값 확인
CURRENT_LLM_MODE=$(grep -E "^LLM_MODE=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"' || echo "")
CURRENT_MONGLE_MODEL=$(grep -E "^MONGLE_MODEL=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"' || echo "")

echo "  현재 LLM_MODE: '${CURRENT_LLM_MODE:-미설정}'"
echo "  현재 MONGLE_MODEL: '${CURRENT_MONGLE_MODEL:-미설정}'"
echo ""

if [[ "${CURRENT_LLM_MODE}" == "hybrid" ]]; then
    ok "LLM_MODE 가 이미 hybrid 입니다."
else
    echo "  .env 파일을 수정하기 전에 백업을 생성합니다..."
    # .env.bak 백업 생성 (기존 백업이 있으면 타임스탬프로 구분)
    BACKUP_PATH="${ENV_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
    cp "${ENV_FILE}" "${BACKUP_PATH}"
    ok "백업 생성: ${BACKUP_PATH}"

    # LLM_MODE 변경 처리:
    # - LLM_MODE= 라인이 이미 있으면 대체
    # - 없으면 .env 파일 끝에 추가
    if grep -qE "^LLM_MODE=" "${ENV_FILE}"; then
        # macOS와 Linux 모두 호환되는 sed 사용
        if [[ "$(uname)" == "Darwin" ]]; then
            sed -i '' "s|^LLM_MODE=.*|LLM_MODE=hybrid|" "${ENV_FILE}"
        else
            sed -i "s|^LLM_MODE=.*|LLM_MODE=hybrid|" "${ENV_FILE}"
        fi
        ok "LLM_MODE=hybrid 로 변경 완료"
    else
        echo "" >> "${ENV_FILE}"
        echo "# 몽글이 하이브리드 LLM 모드 (M-LLM-7)" >> "${ENV_FILE}"
        echo "LLM_MODE=hybrid" >> "${ENV_FILE}"
        ok "LLM_MODE=hybrid 추가 완료"
    fi
fi

# MONGLE_MODEL 확인 및 설정
if grep -qE "^MONGLE_MODEL=" "${ENV_FILE}"; then
    CURRENT_MONGLE=$(grep -E "^MONGLE_MODEL=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"')
    if [[ "${CURRENT_MONGLE}" != "${MONGLE_MODEL_NAME}" ]]; then
        warn "MONGLE_MODEL='${CURRENT_MONGLE}' 이 설정되어 있습니다. '${MONGLE_MODEL_NAME}' 과 다릅니다."
        if confirm "MONGLE_MODEL=${MONGLE_MODEL_NAME} 로 변경하시겠습니까?"; then
            if [[ "$(uname)" == "Darwin" ]]; then
                sed -i '' "s|^MONGLE_MODEL=.*|MONGLE_MODEL=${MONGLE_MODEL_NAME}|" "${ENV_FILE}"
            else
                sed -i "s|^MONGLE_MODEL=.*|MONGLE_MODEL=${MONGLE_MODEL_NAME}|" "${ENV_FILE}"
            fi
            ok "MONGLE_MODEL=${MONGLE_MODEL_NAME} 로 변경 완료"
        fi
    else
        ok "MONGLE_MODEL=${MONGLE_MODEL_NAME} 이미 설정됨"
    fi
else
    echo "MONGLE_MODEL=${MONGLE_MODEL_NAME}" >> "${ENV_FILE}"
    ok "MONGLE_MODEL=${MONGLE_MODEL_NAME} 추가 완료"
fi

# UPSTAGE_API_KEY 설정 여부 확인 (hybrid 모드에서 Solar API 필수)
UPSTAGE_KEY=$(grep -E "^UPSTAGE_API_KEY=" "${ENV_FILE}" | cut -d'=' -f2 | tr -d '"' || echo "")
if [[ -z "${UPSTAGE_KEY}" ]]; then
    warn "UPSTAGE_API_KEY 가 설정되지 않았습니다."
    warn "hybrid 모드에서 Solar API 체인(의도/감정/선호/추천이유/이미지)이 작동하지 않습니다."
    warn ".env 파일에 UPSTAGE_API_KEY=your_key_here 를 추가하세요."
else
    ok "UPSTAGE_API_KEY 설정 확인"
fi

# 변경 후 최종 확인
echo ""
echo "  변경된 .env 설정:"
grep -E "^(LLM_MODE|MONGLE_MODEL|SOLAR_API_MODEL|UPSTAGE_API_KEY)=" "${ENV_FILE}" | sed 's/^/    /'

# =============================================================================
# STEP 6: 서비스 재시작 안내
# =============================================================================

print_step "6/6" "서비스 재시작"

echo "  .env 변경사항을 반영하려면 uvicorn을 재시작해야 합니다."
echo ""
echo "  재시작 명령어:"
echo -e "  ${BOLD}  cd ${PROJECT_ROOT}${RESET}"
echo -e "  ${BOLD}  PYTHONPATH=src uv run uvicorn monglepick.main:app --reload${RESET}"
echo ""
echo "  재시작 후 /health 엔드포인트로 정상 동작을 확인하세요:"
echo -e "  ${BOLD}  curl http://localhost:8000/health${RESET}"
echo ""

if confirm "지금 uvicorn을 백그라운드에서 재시작하시겠습니까?"; then
    # 기존 uvicorn 프로세스 종료 (포트 8000 기준)
    EXISTING_PID=$(lsof -ti :8000 2>/dev/null || echo "")
    if [[ -n "${EXISTING_PID}" ]]; then
        echo "  기존 uvicorn 프로세스 종료 (PID: ${EXISTING_PID})..."
        kill "${EXISTING_PID}" 2>/dev/null && ok "기존 프로세스 종료" || warn "프로세스 종료 실패 (수동으로 종료하세요)"
        sleep 2  # 포트 해제 대기
    fi

    echo "  uvicorn 시작 중..."
    cd "${PROJECT_ROOT}"
    # 백그라운드 실행 후 PID 저장
    nohup env PYTHONPATH="${PROJECT_ROOT}/src" uv run uvicorn monglepick.main:app \
        --host 0.0.0.0 --port 8000 \
        > /tmp/monglepick-agent.log 2>&1 &
    SERVER_PID=$!
    echo "  uvicorn PID: ${SERVER_PID}"
    echo "  로그: tail -f /tmp/monglepick-agent.log"

    # 5초 대기 후 헬스체크
    sleep 5
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        ok "서버 정상 실행 확인 (http://localhost:8000/health)"
    else
        warn "헬스체크 실패. 로그를 확인하세요: tail -f /tmp/monglepick-agent.log"
    fi
else
    echo "  수동으로 재시작하세요:"
    echo "    PYTHONPATH=src uv run uvicorn monglepick.main:app --reload"
fi

# =============================================================================
# 최종 전환 완료 체크리스트
# =============================================================================

print_header "전환 완료 체크리스트"

# 각 항목 상태 확인
CHECK_OLLAMA="$(ollama list 2>/dev/null | grep -q "${MONGLE_MODEL_NAME}" && echo 'PASS' || echo 'FAIL')"
CHECK_ENV_MODE="$(grep -qE '^LLM_MODE=hybrid' "${ENV_FILE}" && echo 'PASS' || echo 'FAIL')"
CHECK_ENV_MODEL="$(grep -qE "^MONGLE_MODEL=${MONGLE_MODEL_NAME}" "${ENV_FILE}" && echo 'PASS' || echo 'FAIL')"
CHECK_EVAL="$([ "${EVAL_PASSED}" == "pass" ] && echo 'PASS' || ([ "${EVAL_PASSED}" == "skipped" ] && echo 'SKIP' || echo 'FAIL'))"
CHECK_HEALTH="$(curl -sf http://localhost:8000/health > /dev/null 2>&1 && echo 'PASS' || echo 'FAIL')"
CHECK_API_KEY="$([ -n "${UPSTAGE_KEY}" ] && echo 'PASS' || echo 'WARN')"

# 상태별 색상 함수
status_icon() {
    case "$1" in
        PASS) echo -e "${GREEN}[PASS]${RESET}" ;;
        FAIL) echo -e "${RED}[FAIL]${RESET}" ;;
        SKIP) echo -e "${YELLOW}[SKIP]${RESET}" ;;
        WARN) echo -e "${YELLOW}[WARN]${RESET}" ;;
        *)    echo -e "${YELLOW}[----]${RESET}" ;;
    esac
}

echo ""
echo "  $(status_icon "${CHECK_OLLAMA}") ollama list에서 '${MONGLE_MODEL_NAME}' 모델 등록 확인"
echo "  $(status_icon "${CHECK_ENV_MODE}") .env: LLM_MODE=hybrid"
echo "  $(status_icon "${CHECK_ENV_MODEL}") .env: MONGLE_MODEL=${MONGLE_MODEL_NAME}"
echo "  $(status_icon "${CHECK_API_KEY}") .env: UPSTAGE_API_KEY 설정 $([ -z "${UPSTAGE_KEY}" ] && echo '(미설정 — Solar API 체인 불가)')"
echo "  $(status_icon "${CHECK_EVAL}") evaluate_mongle.py 결과 PASS $([ "${EVAL_PASSED}" == "skipped" ] && echo '(건너뜀)')"
echo "  $(status_icon "${CHECK_HEALTH}") uvicorn 재시작 + /health 정상 응답"
echo ""

# 전체 성공 여부 (WARN/SKIP은 경고로 처리)
ALL_OK=true
for status in "${CHECK_OLLAMA}" "${CHECK_ENV_MODE}" "${CHECK_ENV_MODEL}" "${CHECK_HEALTH}"; do
    [[ "${status}" != "PASS" ]] && ALL_OK=false && break
done

if ${ALL_OK}; then
    echo -e "${BOLD}${GREEN}  *** 몽글이 Hybrid LLM 모드 전환 완료! ***${RESET}"
    echo ""
    echo "  이제 다음 체인은 몽글이(로컬, 빠른 응답)를 사용합니다:"
    echo "    - general_chain (일반 대화)"
    echo "    - question_chain (후속 질문 생성)"
    echo ""
    echo "  다음 체인은 Solar API(품질 보장)를 사용합니다:"
    echo "    - intent_emotion_chain (의도+감정 분류)"
    echo "    - preference_chain (선호 추출)"
    echo "    - explanation_chain (추천 이유 생성)"
    echo "    - image_analysis_chain (이미지 분석)"
else
    echo -e "${YELLOW}  일부 항목이 FAIL 상태입니다. 위 체크리스트를 확인하여 수동으로 완료하세요.${RESET}"
fi

# =============================================================================
# 참고: 모드 되돌리기
# =============================================================================

echo ""
echo "  ── 모드 되돌리기 ──────────────────────────────────────"
echo "  hybrid → local_only: .env에서 LLM_MODE=local_only 로 변경 후 재시작"
echo "  hybrid → api_only:   .env에서 LLM_MODE=api_only 로 변경 후 재시작"
echo "  백업 파일: ${ENV_FILE}.bak.*"
echo ""
echo "  ── 관련 명령어 ─────────────────────────────────────────"
echo "  품질 재평가:  PYTHONPATH=src uv run python scripts/evaluate_mongle.py --model ${MONGLE_MODEL_NAME}"
echo "  모델 목록:    ollama list"
echo "  모델 삭제:    ollama rm ${MONGLE_MODEL_NAME}"
echo "  LangSmith:    https://smith.langchain.com (트레이싱 확인)"
echo ""
