"""
몽글이 파인튜닝 학습 데이터 자동 생성 스크립트 (M-LLM-3).

Solar API(solar-pro)를 사용하여 몽글이(EXAONE 4.0 1.2B LoRA)의 학습 데이터를
자동 생성한다. 3가지 카테고리 × 시드 프롬프트 → JSONL 형식으로 저장.

─────────────────────────────────────────────────────────────────
데이터 구조 (JSONL):
  {"instruction": "시스템 프롬프트", "input": "사용자 입력", "output": "몽글이 응답"}

카테고리별 목표 건수:
  1. 몽글 페르소나 대화      300~500쌍 — 인사, 감정 공감, 영화 잡담
  2. 후속 질문 패턴          200~300쌍 — 부족 필드 → 자연스러운 질문
  3. 영화 도메인 응답        200~300쌍 — 장르/감독/배우 가벼운 대화

─────────────────────────────────────────────────────────────────
Solar API 스펙 (Upstage):
  - 엔드포인트: https://api.upstage.ai/v1/chat/completions
  - 모델: solar-pro (OpenAI 호환)
  - 인증: Bearer {UPSTAGE_API_KEY}
  - Rate Limit: ~100 RPM (보수적으로 30 RPM 사용)

비용 추정 (solar-pro 기준, 800쌍 생성 시):
  - API 호출 횟수: ~80회 (배치 10쌍/호출)
  - 입력: ~120K tokens × $0.50/1M = ~$0.06
  - 출력: ~200K tokens × $1.50/1M = ~$0.30
  - 총합: ~$0.36 (약 500원)

─────────────────────────────────────────────────────────────────
사용법:
  # 기본 (총 800쌍 생성)
  PYTHONPATH=src uv run python scripts/generate_training_data.py

  # 수량 지정
  PYTHONPATH=src uv run python scripts/generate_training_data.py --total 1000

  # 출력 경로 지정
  PYTHONPATH=src uv run python scripts/generate_training_data.py \\
    --output data/finetune/mongle_train.jsonl \\
    --eval-output data/finetune/mongle_eval.jsonl \\
    --eval-ratio 0.1

  # 중단 후 재개 (기존 체크포인트에서 이어서)
  PYTHONPATH=src uv run python scripts/generate_training_data.py --resume

  # 비용 추정만 (API 호출 없음)
  PYTHONPATH=src uv run python scripts/generate_training_data.py --estimate-only

관련 설계:
  - docs/몽글이_하이브리드_LLM_설계서.md §3-3 (학습 데이터 설계)
  - docs/몽글이_하이브리드_LLM_설계서.md §6-1 (디렉토리 구조)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── 프로젝트 루트를 sys.path에 추가 + .env 로드 ──
# PYTHONPATH=src 없이도 동작하도록 src 경로를 직접 추가
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# .env 파일에서 환경변수 로드 (config.py 초기화 전에 반드시 필요)
_env_file = _project_root / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

import structlog  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수 — 카테고리 비율 및 API 설정
# ══════════════════════════════════════════════════════════════

# 카테고리별 목표 비율 (합계 = 1.0)
CATEGORY_RATIOS: dict[str, float] = {
    "persona":  0.44,   # 몽글 페르소나 대화 (~44%)
    "question": 0.31,   # 후속 질문 패턴 (~31%)
    "domain":   0.25,   # 영화 도메인 응답 (~25%)
}

# Solar API 기본 설정
SOLAR_BASE_URL = "https://api.upstage.ai/v1"
SOLAR_MODEL = "solar-pro"

# 분당 최대 요청 수 (Upstage RPM 한도 100, 보수적으로 30 사용)
DEFAULT_RPM = 30
# 1회 API 호출에서 생성 요청할 데이터 쌍 수
BATCH_SIZE_PER_CALL = 10
# 최대 동시 API 호출 수
DEFAULT_CONCURRENCY = 3

# 체크포인트 파일 경로 (중단 재개용)
CHECKPOINT_FILE = _project_root / "data" / "finetune" / "checkpoint.json"

# 유사도 필터링 — input 문자열 최소 편집 거리 비율 (0.0~1.0)
# 두 input이 이 값 이하로 유사하면 중복으로 판정하여 버림
DEDUP_SIMILARITY_THRESHOLD = 0.85


# ══════════════════════════════════════════════════════════════
# 시드 프롬프트 — 카테고리별 다양성 확보용 시드
# ══════════════════════════════════════════════════════════════

# ── 카테고리 1: 몽글 페르소나 대화 ──
# Solar API에게 "이런 상황에서 몽글이 대화 10쌍을 만들어라"는 프롬프트의 씨앗.
# 각 시드는 상황(situation)과 힌트(hints)로 구성된다.
PERSONA_SEEDS: list[dict[str, str]] = [
    {
        "situation": "사용자가 처음 인사하며 영화 추천을 요청하는 상황",
        "hints": "반갑게 맞이하고, 어떤 영화를 원하는지 부드럽게 물어보는 대화",
    },
    {
        "situation": "사용자가 오늘 기분이 우울하다고 말하는 상황",
        "hints": "감정에 공감하고, 힐링 또는 기분 전환 영화 방향을 제안하는 대화",
    },
    {
        "situation": "사용자가 친구와 함께 볼 영화를 찾는 상황",
        "hints": "같이 보기 좋은 장르를 확인하고 상황에 맞는 추천 방향 제시",
    },
    {
        "situation": "사용자가 최근에 본 영화가 너무 좋았다고 말하는 상황",
        "hints": "그 영화에 대해 공감하며, 비슷한 취향의 영화를 찾아주겠다는 대화",
    },
    {
        "situation": "사용자가 오랜만에 영화를 보려고 하는 상황",
        "hints": "설레는 분위기에 공감하며, 오랜만에 보기 좋은 영화 스타일 확인",
    },
    {
        "situation": "사용자가 영화 추천해달라고 짧게 요청하는 상황",
        "hints": "간단한 인사 후 취향을 파악하기 위한 첫 질문을 던지는 대화",
    },
    {
        "situation": "사용자가 혼자 밤에 볼 영화를 찾는 상황",
        "hints": "혼자 보기 좋은 장르(스릴러, 공포, 드라마 등) 취향 확인",
    },
    {
        "situation": "사용자가 스트레스를 풀고 싶다고 말하는 상황",
        "hints": "통쾌한 액션, 유쾌한 코미디, 카타르시스 있는 드라마 방향 제안",
    },
    {
        "situation": "사용자가 가족과 함께 볼 영화를 찾는 상황",
        "hints": "온 가족이 볼 수 있는 연령대 고려한 장르 확인",
    },
    {
        "situation": "사용자가 데이트할 때 볼 영화를 추천해달라는 상황",
        "hints": "분위기, 두 사람의 취향 파악 후 로맨스/코미디/판타지 방향 제안",
    },
    {
        "situation": "사용자가 감동받고 싶다고 말하는 상황",
        "hints": "감동적인 실화, 가족 드라마, 우정 이야기 방향의 따뜻한 추천 대화",
    },
    {
        "situation": "사용자가 무서운 영화를 보고 싶다고 말하는 상황",
        "hints": "공포 영화 취향(심리적 공포 vs 충격적 고어 vs 귀신 등) 파악",
    },
    {
        "situation": "사용자가 시간 가는 줄 모르게 볼 영화를 찾는 상황",
        "hints": "몰입감 있는 장르(스릴러, SF, 범죄) 취향 확인",
    },
    {
        "situation": "사용자가 영화를 많이 봐서 뭘 봐야 할지 모르는 상황",
        "hints": "지금까지 좋아했던 영화나 장르 중심으로 취향 좁혀가는 대화",
    },
    {
        "situation": "사용자가 요즘 영화 취향이 바뀐 것 같다고 말하는 상황",
        "hints": "어떻게 바뀌었는지 공감하며 새로운 취향 탐색을 도와주는 대화",
    },
    {
        "situation": "사용자가 몽글이에게 자기 소개를 요청하는 상황",
        "hints": "몽글이가 자신을 소개하고 서비스를 친근하게 설명하는 대화",
    },
    {
        "situation": "사용자가 영화 추천받은 후 재미없었다고 하는 상황",
        "hints": "솔직한 피드백에 공감하고, 무엇이 맞지 않았는지 파악하는 대화",
    },
    {
        "situation": "사용자가 좋아하는 영화 장르가 여러 개라서 고민인 상황",
        "hints": "오늘 기분이나 상황에 따라 하나를 좁혀가는 대화",
    },
    {
        "situation": "사용자가 영화관에서 볼 영화를 고르는 상황",
        "hints": "현재 상영작 중 선택 도움, 선호 장르 확인",
    },
    {
        "situation": "사용자가 한국 영화를 보고 싶다고 말하는 상황",
        "hints": "한국 영화 내에서 장르와 분위기 취향 파악",
    },
]

# ── 카테고리 2: 후속 질문 패턴 ──
# 부족 필드 조합별로 자연스러운 후속 질문을 생성하기 위한 시드.
# missing_fields: 아직 파악되지 않은 선호 필드 목록
# known_context: 이미 알고 있는 맥락 (없으면 빈 문자열)
QUESTION_SEEDS: list[dict[str, Any]] = [
    {
        "missing_fields": ["장르", "분위기"],
        "known_context": "",
        "user_input": "영화 추천해줘",
    },
    {
        "missing_fields": ["분위기"],
        "known_context": "액션 영화 좋아함",
        "user_input": "액션 영화 추천해줘",
    },
    {
        "missing_fields": ["장르"],
        "known_context": "기분이 우울함",
        "user_input": "우울한데 뭐 볼까",
    },
    {
        "missing_fields": ["플랫폼"],
        "known_context": "스릴러 좋아함, 혼자 볼 예정",
        "user_input": "스릴러 추천해줘",
    },
    {
        "missing_fields": ["시청 상황"],
        "known_context": "로맨스 영화 원함",
        "user_input": "로맨스 영화 추천해줘",
    },
    {
        "missing_fields": ["참조 영화"],
        "known_context": "SF 좋아함, 넷플릭스 이용",
        "user_input": "넷플릭스에서 볼 SF 추천해줘",
    },
    {
        "missing_fields": ["시대/연도"],
        "known_context": "드라마 장르, 감동적인 분위기",
        "user_input": "감동적인 드라마 추천해줘",
    },
    {
        "missing_fields": ["제외 조건"],
        "known_context": "공포 영화 원함",
        "user_input": "무서운 영화 추천해줘",
    },
    {
        "missing_fields": ["장르", "시청 상황"],
        "known_context": "",
        "user_input": "주말에 볼 영화 추천",
    },
    {
        "missing_fields": ["분위기", "참조 영화"],
        "known_context": "코미디 좋아함",
        "user_input": "웃긴 영화 추천해줘",
    },
    {
        "missing_fields": ["장르", "분위기", "시청 상황"],
        "known_context": "",
        "user_input": "뭔가 보고 싶은데 뭘 볼지 모르겠어",
    },
    {
        "missing_fields": ["플랫폼", "시청 상황"],
        "known_context": "스트레스 풀고 싶음",
        "user_input": "스트레스 풀리는 영화 뭐 있어?",
    },
    {
        "missing_fields": ["장르"],
        "known_context": "가족과 함께 볼 예정",
        "user_input": "가족끼리 같이 볼 영화 추천해줘",
    },
    {
        "missing_fields": ["분위기", "플랫폼"],
        "known_context": "데이트용 영화 원함",
        "user_input": "데이트할 때 볼 영화 추천해줘",
    },
    {
        "missing_fields": ["참조 영화", "시대/연도"],
        "known_context": "범죄 스릴러 좋아함",
        "user_input": "범죄 스릴러 추천해줘",
    },
    {
        "missing_fields": ["장르", "제외 조건"],
        "known_context": "",
        "user_input": "오늘 저녁에 볼 영화 추천",
    },
    {
        "missing_fields": ["분위기"],
        "known_context": "한국 영화 원함",
        "user_input": "한국 영화 추천해줘",
    },
    {
        "missing_fields": ["장르", "분위기"],
        "known_context": "넷플릭스 이용, 혼자 볼 예정",
        "user_input": "넷플릭스에서 혼자 볼 영화 추천",
    },
    {
        "missing_fields": ["시청 상황", "플랫폼"],
        "known_context": "판타지 영화 좋아함",
        "user_input": "판타지 영화 추천해줘",
    },
    {
        "missing_fields": ["참조 영화"],
        "known_context": "감동 영화, 드라마 장르",
        "user_input": "감동적인 드라마 좀 추천해줘",
    },
]

# ── 카테고리 3: 영화 도메인 응답 ──
# 감독, 배우, 장르에 대한 가벼운 대화 시드.
# user_say: 사용자가 말하는 것 / topic: 대화 주제
DOMAIN_SEEDS: list[dict[str, str]] = [
    {
        "user_say": "봉준호 감독 영화가 너무 좋아",
        "topic": "봉준호 감독 작품 세계 — 사회 비판, 장르 혼합의 매력",
    },
    {
        "user_say": "마블 영화는 이제 좀 지루해",
        "topic": "마블 피로감 공감 — 대안 장르 소개",
    },
    {
        "user_say": "크리스토퍼 놀란 감독 영화 다 봤어",
        "topic": "놀란 팬에게 공감하며 비슷한 감독 소개",
    },
    {
        "user_say": "한국 영화가 요즘 너무 잘 나오는 것 같아",
        "topic": "한국 영화의 최근 흐름과 주목할 작품들",
    },
    {
        "user_say": "공포 영화는 잘 못 보는데 스릴러는 좋아해",
        "topic": "공포와 스릴러의 차이, 스릴러 추천 방향",
    },
    {
        "user_say": "애니메이션 영화도 어른이 봐도 돼?",
        "topic": "성인도 즐길 수 있는 애니메이션 영화 이야기",
    },
    {
        "user_say": "SF 영화는 너무 어려운 것 같아",
        "topic": "쉽게 즐길 수 있는 SF 영화 장르 안내",
    },
    {
        "user_say": "송강호 배우 연기가 정말 대단하더라",
        "topic": "송강호 배우의 매력과 대표 작품",
    },
    {
        "user_say": "히치콕 감독 영화 본 적 있어?",
        "topic": "히치콕 클래식 영화의 매력과 입문작 안내",
    },
    {
        "user_say": "요즘 드라마가 영화보다 재밌는 것 같아",
        "topic": "드라마와 영화의 차이, 영화만의 매력 안내",
    },
    {
        "user_say": "흑백 영화도 볼 만해?",
        "topic": "흑백 영화의 매력과 입문하기 좋은 작품",
    },
    {
        "user_say": "외국 영화는 자막이 불편해",
        "topic": "자막 영화에 익숙해지는 방법, 잘 만든 자막 영화 소개",
    },
    {
        "user_say": "4시간짜리 영화도 있어?",
        "topic": "장편 영화의 매력, 시간이 아깝지 않은 명작들",
    },
    {
        "user_say": "영화 배경음악이 정말 중요한 것 같아",
        "topic": "음악으로 기억되는 영화들, OST 명작 이야기",
    },
    {
        "user_say": "실화 기반 영화가 더 감동적인 것 같아",
        "topic": "실화 기반 영화의 매력과 추천 방향",
    },
    {
        "user_say": "독립 영화랑 상업 영화 차이가 뭐야?",
        "topic": "독립 영화의 특징과 입문 추천작",
    },
    {
        "user_say": "영화제 수상작이 꼭 재밌는 건 아닌 것 같아",
        "topic": "영화제 수상작과 대중 영화의 차이, 공감하며 설명",
    },
    {
        "user_say": "로버트 드 니로랑 알 파치노 둘 중에 누가 더 좋아?",
        "topic": "두 배우의 매력 비교, 각 대표작 소개",
    },
    {
        "user_say": "미야자키 하야오 감독 은퇴했다던데",
        "topic": "미야자키 하야오 감독의 작품 세계와 레거시",
    },
    {
        "user_say": "영화 평점이 낮아도 재밌는 영화가 있어?",
        "topic": "평점과 관계없이 매력 있는 영화, 취향의 다양성",
    },
]


# ══════════════════════════════════════════════════════════════
# 시스템 프롬프트 — Solar API가 데이터를 생성할 때 사용
# ══════════════════════════════════════════════════════════════

# 몽글 페르소나 설명 (모든 카테고리 공통으로 Solar API에게 전달)
MONGLE_PERSONA_DESC = """\
[몽글이 페르소나]
- 이름: 몽글이 (영화 추천 서비스 "몽글픽"의 AI 어시스턴트)
- 말투: ~요/~에요 존댓말, 친근하고 따뜻하게, 영화에 대한 열정 표현
- 이모지: 자연스러운 경우 1~2개 사용, 강요 금지
- 금지: 스포일러, 영화 비하, 미확인 정보 전달
- 핵심: 사용자 감정에 공감하며, 구체적이고 자연스럽게 대화
"""

# ── 카테고리 1: 페르소나 대화 생성 시스템 프롬프트 ──
PERSONA_GEN_SYSTEM = f"""\
당신은 AI 학습 데이터 생성 전문가입니다.
영화 추천 AI "몽글이"의 LoRA 파인튜닝을 위한 대화 데이터를 생성해주세요.

{MONGLE_PERSONA_DESC}

[출력 형식 — 반드시 다음 JSON 배열로만 응답]
[
  {{
    "instruction": "당신은 몽글이입니다. 영화 추천 서비스 몽글픽의 친근한 AI 어시스턴트로서 사용자와 자연스럽게 대화하세요.",
    "input": "사용자가 실제로 입력하는 문장",
    "output": "몽글이의 자연스럽고 친근한 응답"
  }}
]

[규칙]
1. 정확히 {BATCH_SIZE_PER_CALL}개의 JSON 객체를 배열에 담아 반환하세요.
2. instruction은 위 형식을 그대로 사용하세요 (변경 금지).
3. input은 실제 사용자가 입력할 법한 자연스러운 한국어 문장이어야 합니다.
4. output은 몽글이의 말투 규칙을 완벽히 따라야 합니다.
5. 다양한 상황을 고루 커버하여 중복이 없도록 하세요.
6. JSON 배열 외에 다른 텍스트는 절대 출력하지 마세요.
"""

# ── 카테고리 2: 후속 질문 생성 시스템 프롬프트 ──
QUESTION_GEN_SYSTEM = f"""\
당신은 AI 학습 데이터 생성 전문가입니다.
영화 추천 AI "몽글이"의 후속 질문 패턴 학습 데이터를 생성해주세요.

{MONGLE_PERSONA_DESC}

[후속 질문의 목적]
- 사용자의 영화 추천을 위해 아직 파악되지 않은 정보를 자연스럽게 물어보는 것
- 이미 아는 정보를 자연스럽게 언급하며 부드럽게 이어가는 것
- 한 번에 하나의 질문만 하는 것

[출력 형식 — 반드시 다음 JSON 배열로만 응답]
[
  {{
    "instruction": "부족한 선호 정보를 자연스럽게 질문하세요. 부족 필드: [필드 목록]",
    "input": "사용자가 입력한 문장",
    "output": "몽글이의 자연스러운 후속 질문"
  }}
]

[규칙]
1. 정확히 {BATCH_SIZE_PER_CALL}개의 JSON 객체를 배열에 담아 반환하세요.
2. instruction의 "부족 필드: [필드 목록]" 부분에는 실제 부족 필드를 작성하세요.
3. 질문은 하나만, 자연스럽고 친근하게, 이미 아는 정보를 살짝 언급하며.
4. output은 질문 하나만 포함 (추천 내용 포함 금지).
5. 다양한 부족 필드 조합을 커버하세요.
6. JSON 배열 외에 다른 텍스트는 절대 출력하지 마세요.
"""

# ── 카테고리 3: 도메인 응답 생성 시스템 프롬프트 ──
DOMAIN_GEN_SYSTEM = f"""\
당신은 AI 학습 데이터 생성 전문가입니다.
영화 추천 AI "몽글이"의 영화 도메인 대화 학습 데이터를 생성해주세요.

{MONGLE_PERSONA_DESC}

[영화 도메인 대화의 특징]
- 특정 추천이 아닌, 감독/배우/장르에 대한 가벼운 대화
- 사용자의 의견에 공감하며 영화 지식을 자연스럽게 나누는 것
- 대화를 통해 취향을 파악하고 추천 방향을 암시하는 것

[출력 형식 — 반드시 다음 JSON 배열로만 응답]
[
  {{
    "instruction": "당신은 몽글이입니다. 영화에 관한 사용자의 이야기에 공감하고 자연스럽게 대화하세요.",
    "input": "사용자가 영화에 대해 말하는 문장",
    "output": "몽글이의 공감하며 영화 지식을 나누는 응답"
  }}
]

[규칙]
1. 정확히 {BATCH_SIZE_PER_CALL}개의 JSON 객체를 배열에 담아 반환하세요.
2. instruction은 위 형식을 그대로 사용하세요.
3. 다양한 감독, 배우, 장르, 영화 관련 주제를 고루 커버하세요.
4. output은 공감 + 간결한 정보 제공 (3~5문장 이내).
5. JSON 배열 외에 다른 텍스트는 절대 출력하지 마세요.
"""


# ══════════════════════════════════════════════════════════════
# RPM 제한기 (슬라이딩 윈도우)
# ══════════════════════════════════════════════════════════════

class RPMLimiter:
    """
    분당 요청 수(RPM)를 제한하는 슬라이딩 윈도우 제한기.

    60초 윈도우 내의 요청 타임스탬프를 추적하고,
    RPM 한도 초과 시 필요한 만큼 대기한다.
    asyncio.Lock으로 동시 접근을 보호한다.
    """

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        # 최근 60초 내의 요청 타임스탬프 목록
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """요청 슬롯을 확보한다. RPM 한도 초과 시 충분히 대기."""
        async with self._lock:
            now = time.time()
            # 60초 이전 타임스탬프 제거 (슬라이딩 윈도우 유지)
            self._timestamps = [t for t in self._timestamps if now - t < 60.0]

            if len(self._timestamps) >= self.rpm:
                # 가장 오래된 요청이 60초를 넘길 때까지 대기
                wait_until = self._timestamps[0] + 60.0
                wait_sec = wait_until - now
                if wait_sec > 0:
                    logger.debug("rpm_wait", wait_sec=round(wait_sec, 1), rpm=self.rpm)
                    await asyncio.sleep(wait_sec)
                now = time.time()

            self._timestamps.append(now)


# ══════════════════════════════════════════════════════════════
# 중복 필터 — 유사 input 감지
# ══════════════════════════════════════════════════════════════

def _simple_similarity(a: str, b: str) -> float:
    """
    두 문자열의 단순 Jaccard 유사도를 계산한다.

    완전한 편집거리(Levenshtein)는 비용이 크므로,
    음절 수준의 집합 기반 Jaccard 유사도를 사용한다.
    짧은 문자열(30자 미만)은 글자 단위, 그 이상은 2-gram 단위.

    Args:
        a: 첫 번째 문자열
        b: 두 번째 문자열

    Returns:
        0.0(완전 상이) ~ 1.0(완전 일치) 사이의 유사도
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    # 2-gram 집합 생성
    def bigrams(s: str) -> set[str]:
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) > 1 else {s}

    sa = bigrams(a.strip())
    sb = bigrams(b.strip())
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union > 0 else 0.0


class DedupFilter:
    """
    기존에 생성된 input 문자열을 기반으로 중복 데이터를 필터링한다.

    DEDUP_SIMILARITY_THRESHOLD 이상 유사한 input이 있으면 중복으로 판정.
    성능을 위해 전체 비교 대신 최근 1,000개만 비교한다.
    """

    def __init__(self, threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> None:
        self.threshold = threshold
        # 기존 input 문자열 목록 (최근 1,000개 유지)
        self._seen: list[str] = []
        self._max_seen = 1000

    def seed(self, existing_inputs: list[str]) -> None:
        """기존 데이터의 input 목록을 초기 시드로 등록한다."""
        self._seen = existing_inputs[-self._max_seen:]

    def is_duplicate(self, input_text: str) -> bool:
        """
        input_text가 기존 데이터와 유사한지 판정한다.

        Returns:
            True이면 중복 (버려야 함)
        """
        for seen in self._seen[-500:]:  # 성능을 위해 최근 500개만 비교
            if _simple_similarity(input_text, seen) >= self.threshold:
                return True
        return False

    def add(self, input_text: str) -> None:
        """새 input_text를 seen 목록에 등록한다."""
        self._seen.append(input_text)
        if len(self._seen) > self._max_seen:
            # FIFO: 가장 오래된 항목 제거
            self._seen = self._seen[-self._max_seen:]


# ══════════════════════════════════════════════════════════════
# 체크포인트 — 중단 재개 지원
# ══════════════════════════════════════════════════════════════

def load_checkpoint() -> dict[str, Any]:
    """
    체크포인트 파일에서 이전 실행 상태를 불러온다.

    Returns:
        체크포인트 딕셔너리. 파일 없으면 빈 상태 반환.
    """
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
            logger.info("checkpoint_loaded", file=str(CHECKPOINT_FILE))
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("checkpoint_load_failed", error=str(e))
    return {}


def save_checkpoint(state: dict[str, Any]) -> None:
    """
    현재 실행 상태를 체크포인트 파일에 저장한다.

    Args:
        state: 저장할 상태 딕셔너리 (generated_count, category_counts 등)
    """
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["saved_at"] = datetime.now().isoformat()
    CHECKPOINT_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ══════════════════════════════════════════════════════════════
# Solar API 호출 — 배치 데이터 생성
# ══════════════════════════════════════════════════════════════

async def _call_solar_api(
    client: AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    rpm_limiter: RPMLimiter,
    semaphore: asyncio.Semaphore,
    attempt: int = 0,
) -> list[dict[str, str]]:
    """
    Solar API에 프롬프트를 보내 instruction-input-output 쌍 목록을 생성한다.

    RPM 제한기와 세마포어로 동시 요청을 제어한다.
    API 오류 또는 JSON 파싱 실패 시 재시도 (최대 3회).

    Args:
        client: AsyncOpenAI 클라이언트 (Solar API 백엔드)
        system_prompt: 카테고리별 시스템 프롬프트
        user_prompt: 시드 기반 사용자 지시 프롬프트
        rpm_limiter: 분당 요청 수 제한기
        semaphore: 동시 요청 수 제한 세마포어
        attempt: 현재 재시도 횟수 (0부터 시작)

    Returns:
        생성된 데이터 쌍 목록. 생성 실패 시 빈 리스트.
    """
    MAX_ATTEMPTS = 3

    await rpm_limiter.acquire()

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=SOLAR_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.8,   # 다양성 확보를 위해 약간 높게 설정
                max_tokens=4096,
            )
        except Exception as e:
            if attempt < MAX_ATTEMPTS - 1:
                wait_sec = 2 ** attempt   # 지수 백오프: 1s, 2s, 4s
                logger.warning(
                    "solar_api_error_retry",
                    error=str(e),
                    attempt=attempt + 1,
                    wait_sec=wait_sec,
                )
                await asyncio.sleep(wait_sec)
                return await _call_solar_api(
                    client, system_prompt, user_prompt,
                    rpm_limiter, semaphore, attempt + 1,
                )
            logger.error("solar_api_failed", error=str(e), attempts=MAX_ATTEMPTS)
            return []

    # ── 응답 파싱 ──
    raw_text = response.choices[0].message.content or ""

    # JSON 배열 추출 (응답에 앞뒤 텍스트가 포함될 경우를 대비)
    start = raw_text.find("[")
    end = raw_text.rfind("]") + 1
    if start == -1 or end == 0:
        logger.warning("json_array_not_found", raw_preview=raw_text[:200])
        # 재시도
        if attempt < MAX_ATTEMPTS - 1:
            await asyncio.sleep(1)
            return await _call_solar_api(
                client, system_prompt, user_prompt,
                rpm_limiter, semaphore, attempt + 1,
            )
        return []

    json_text = raw_text[start:end]
    try:
        items = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning("json_parse_failed", error=str(e), raw_preview=json_text[:200])
        if attempt < MAX_ATTEMPTS - 1:
            await asyncio.sleep(1)
            return await _call_solar_api(
                client, system_prompt, user_prompt,
                rpm_limiter, semaphore, attempt + 1,
            )
        return []

    # ── 필드 검증 ──
    valid_items: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction", "")).strip()
        inp = str(item.get("input", "")).strip()
        output = str(item.get("output", "")).strip()

        # 필수 필드가 모두 존재하고 비어있지 않아야 함
        if not instruction or not inp or not output:
            continue
        # 너무 짧은 output은 품질 불량으로 간주 (10자 미만)
        if len(output) < 10:
            continue
        # instruction에 "몽글" 또는 관련 키워드가 없으면 잘못된 데이터
        valid_items.append({"instruction": instruction, "input": inp, "output": output})

    return valid_items


# ══════════════════════════════════════════════════════════════
# 사용자 프롬프트 빌더 — 카테고리별 시드 → 프롬프트 텍스트
# ══════════════════════════════════════════════════════════════

def _build_persona_user_prompt(seeds: list[dict[str, str]]) -> str:
    """
    페르소나 대화 카테고리의 사용자 프롬프트를 생성한다.

    여러 시드를 무작위로 골라 하나의 요청 프롬프트로 조합한다.

    Args:
        seeds: PERSONA_SEEDS 목록 (전체 또는 일부)

    Returns:
        Solar API에 전달할 사용자 프롬프트 문자열
    """
    # 2~4개 시드를 무작위 선택하여 다양성 확보
    chosen = random.sample(seeds, min(len(seeds), random.randint(2, 4)))
    situations = "\n".join(
        f"- 상황 {i+1}: {s['situation']} (힌트: {s['hints']})"
        for i, s in enumerate(chosen)
    )
    return (
        f"다음 상황들을 참고하여 몽글이의 대화 데이터 {BATCH_SIZE_PER_CALL}쌍을 생성해주세요.\n"
        f"각 쌍은 서로 다른 상황이어야 하며, 총 {BATCH_SIZE_PER_CALL}개가 되도록 다양하게 만드세요.\n\n"
        f"[참고 상황]\n{situations}\n\n"
        f"JSON 배열로만 응답하세요."
    )


def _build_question_user_prompt(seeds: list[dict[str, Any]]) -> str:
    """
    후속 질문 카테고리의 사용자 프롬프트를 생성한다.

    Args:
        seeds: QUESTION_SEEDS 목록

    Returns:
        Solar API에 전달할 사용자 프롬프트 문자열
    """
    chosen = random.sample(seeds, min(len(seeds), random.randint(3, 6)))

    examples: list[str] = []
    for s in chosen:
        fields_str = ", ".join(s["missing_fields"])
        known = f"(이미 파악: {s['known_context']})" if s.get("known_context") else ""
        examples.append(
            f'- 부족 필드: [{fields_str}]{known}, 사용자 입력: "{s["user_input"]}"'
        )

    examples_str = "\n".join(examples)
    return (
        f"다음 예시를 참고하여 후속 질문 데이터 {BATCH_SIZE_PER_CALL}쌍을 생성해주세요.\n"
        f"각 쌍은 서로 다른 부족 필드 조합을 다루어야 합니다.\n\n"
        f"[참고 예시]\n{examples_str}\n\n"
        f"instruction의 '부족 필드' 부분은 실제 해당 쌍의 부족 필드 목록으로 채우세요.\n"
        f"JSON 배열로만 응답하세요."
    )


def _build_domain_user_prompt(seeds: list[dict[str, str]]) -> str:
    """
    영화 도메인 응답 카테고리의 사용자 프롬프트를 생성한다.

    Args:
        seeds: DOMAIN_SEEDS 목록

    Returns:
        Solar API에 전달할 사용자 프롬프트 문자열
    """
    chosen = random.sample(seeds, min(len(seeds), random.randint(3, 5)))
    examples = "\n".join(
        f'- 사용자: "{s["user_say"]}" (주제: {s["topic"]})'
        for s in chosen
    )
    return (
        f"다음 예시를 참고하여 영화 도메인 대화 데이터 {BATCH_SIZE_PER_CALL}쌍을 생성해주세요.\n"
        f"다양한 감독, 배우, 장르를 고루 다루어야 합니다.\n\n"
        f"[참고 예시]\n{examples}\n\n"
        f"JSON 배열로만 응답하세요."
    )


# ══════════════════════════════════════════════════════════════
# 카테고리별 생성 작업
# ══════════════════════════════════════════════════════════════

async def generate_category(
    category: str,
    target_count: int,
    client: AsyncOpenAI,
    rpm_limiter: RPMLimiter,
    semaphore: asyncio.Semaphore,
    dedup: DedupFilter,
    already_generated: int = 0,
) -> list[dict[str, str]]:
    """
    특정 카테고리의 학습 데이터를 목표 건수까지 생성한다.

    API 호출을 반복하여 목표 건수 - already_generated 만큼 추가 생성한다.

    Args:
        category: 카테고리명 ("persona" | "question" | "domain")
        target_count: 해당 카테고리의 목표 총 건수
        client: Solar API AsyncOpenAI 클라이언트
        rpm_limiter: RPM 제한기
        semaphore: 동시 호출 세마포어
        dedup: 중복 필터
        already_generated: 이미 생성된 건수 (재개 시 사용)

    Returns:
        이번 실행에서 새로 생성된 데이터 목록
    """
    # 카테고리별 시스템 프롬프트 + 사용자 프롬프트 빌더 선택
    if category == "persona":
        sys_prompt = PERSONA_GEN_SYSTEM
        def build_user_prompt() -> str:
            return _build_persona_user_prompt(PERSONA_SEEDS)
    elif category == "question":
        sys_prompt = QUESTION_GEN_SYSTEM
        def build_user_prompt() -> str:
            return _build_question_user_prompt(QUESTION_SEEDS)
    else:  # domain
        sys_prompt = DOMAIN_GEN_SYSTEM
        def build_user_prompt() -> str:
            return _build_domain_user_prompt(DOMAIN_SEEDS)

    results: list[dict[str, str]] = []
    remaining = target_count - already_generated

    logger.info(
        "category_start",
        category=category,
        target=target_count,
        already=already_generated,
        remaining=remaining,
    )

    while len(results) < remaining:
        # 현재 진행률 출력
        total_done = already_generated + len(results)
        pct = total_done / target_count * 100
        print(
            f"  [{category}] {total_done}/{target_count} ({pct:.1f}%) "
            f"| 이번 실행 +{len(results)}",
            end="\r",
            flush=True,
        )

        # API 호출
        user_prompt = build_user_prompt()
        batch = await _call_solar_api(
            client, sys_prompt, user_prompt, rpm_limiter, semaphore,
        )

        # 중복 필터링 후 결과에 추가
        added = 0
        for item in batch:
            if dedup.is_duplicate(item["input"]):
                continue
            dedup.add(item["input"])
            results.append(item)
            added += 1

        logger.debug(
            "batch_done",
            category=category,
            batch_size=len(batch),
            added=added,
            dedup_filtered=len(batch) - added,
            total_new=len(results),
        )

        # 모든 배치가 중복으로 걸러지면 무한루프 방지
        if not batch and len(results) < remaining:
            logger.warning("empty_batch_skip", category=category)
            await asyncio.sleep(2)

    print()  # 진행률 줄 마무리
    return results[:remaining]  # 목표 건수까지만 반환


# ══════════════════════════════════════════════════════════════
# 비용 추정 출력
# ══════════════════════════════════════════════════════════════

def print_cost_estimate(total: int) -> None:
    """
    Solar API 호출 비용을 추정하여 출력한다.

    solar-pro 가격 기준 (예상치, 실제와 다를 수 있음):
      - 입력: $0.50 / 1M tokens
      - 출력: $1.50 / 1M tokens

    Args:
        total: 목표 생성 건수
    """
    api_calls = total // BATCH_SIZE_PER_CALL + 1

    # 프롬프트당 평균 토큰 수 추정
    avg_input_tokens_per_call = 800    # 시스템 + 사용자 프롬프트
    avg_output_tokens_per_call = 1200  # 10쌍 × 평균 120 tokens/쌍

    total_input_tokens = api_calls * avg_input_tokens_per_call
    total_output_tokens = api_calls * avg_output_tokens_per_call

    cost_input = total_input_tokens / 1_000_000 * 0.50
    cost_output = total_output_tokens / 1_000_000 * 1.50
    cost_total = cost_input + cost_output
    cost_krw = cost_total * 1380  # 대략 환율 (1 USD = 1,380 KRW)

    print("\n[비용 추정] (solar-pro 기준, 예상치)")
    print(f"  - 목표 생성 건수: {total:,}쌍")
    print(f"  - API 호출 횟수:  ~{api_calls:,}회 (배치 {BATCH_SIZE_PER_CALL}건/호출)")
    print(f"  - 입력 토큰:     ~{total_input_tokens:,} tokens (${cost_input:.3f})")
    print(f"  - 출력 토큰:     ~{total_output_tokens:,} tokens (${cost_output:.3f})")
    print(f"  - 예상 비용:      ~${cost_total:.3f} (~{cost_krw:,.0f}원)")
    print(f"  - 예상 소요:      ~{api_calls // DEFAULT_RPM + 1}분 ({DEFAULT_RPM} RPM 기준)\n")


# ══════════════════════════════════════════════════════════════
# JSONL 파일 입출력
# ══════════════════════════════════════════════════════════════

def load_existing_jsonl(path: Path) -> list[dict[str, str]]:
    """
    기존 JSONL 파일에서 데이터를 로드한다.

    중단 재개 시 이미 생성된 데이터를 파악하는 데 사용한다.

    Args:
        path: JSONL 파일 경로

    Returns:
        파싱된 데이터 목록. 파일 없으면 빈 리스트.
    """
    if not path.exists():
        return []
    items: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return items


def append_to_jsonl(path: Path, items: list[dict[str, str]]) -> None:
    """
    데이터 목록을 JSONL 파일에 추가 저장(append)한다.

    파일이 없으면 새로 생성하고, 있으면 기존 내용 유지 후 이어서 쓴다.

    Args:
        path: 저장할 JSONL 파일 경로
        items: 저장할 데이터 목록
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════
# 메인 실행 함수
# ══════════════════════════════════════════════════════════════

async def main(args: argparse.Namespace) -> None:
    """
    학습 데이터 생성 메인 로직.

    1. 기존 데이터 로드 (재개 시)
    2. Solar API 클라이언트 초기화
    3. 카테고리별 목표 건수 계산
    4. 각 카테고리 병렬/순차 생성
    5. train/eval 분리 저장
    6. 체크포인트 저장

    Args:
        args: argparse.Namespace (CLI 인수)
    """
    total_target = args.total
    output_path = Path(args.output)
    eval_output_path = Path(args.eval_output)
    eval_ratio = args.eval_ratio

    print("\n" + "=" * 60)
    print("  몽글이 파인튜닝 학습 데이터 생성기 (M-LLM-3)")
    print("=" * 60)
    print_cost_estimate(total_target)

    if args.estimate_only:
        print("[--estimate-only] API 호출 없이 종료합니다.\n")
        return

    # ── API 키 확인 ──
    api_key = os.environ.get("UPSTAGE_API_KEY", "").strip()
    if not api_key:
        print("[오류] UPSTAGE_API_KEY가 설정되지 않았습니다.")
        print("       .env 파일에 UPSTAGE_API_KEY=up_xxx 를 추가하세요.")
        sys.exit(1)

    # ── Solar API 클라이언트 초기화 ──
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=SOLAR_BASE_URL,
    )

    # ── 공통 제어 객체 ──
    rpm_limiter = RPMLimiter(rpm=DEFAULT_RPM)
    semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY)
    dedup = DedupFilter(threshold=DEDUP_SIMILARITY_THRESHOLD)

    # ── 기존 데이터 로드 (재개 지원) ──
    existing_data: list[dict[str, str]] = []
    checkpoint: dict[str, Any] = {}

    if args.resume:
        existing_data = load_existing_jsonl(output_path)
        checkpoint = load_checkpoint()
        if existing_data:
            print(f"[재개] 기존 {len(existing_data):,}쌍 로드 완료 (체크포인트: {checkpoint})")
            # 중복 필터에 기존 input 등록
            dedup.seed([d["input"] for d in existing_data])

    # ── 카테고리별 목표 건수 및 기존 건수 계산 ──
    existing_by_cat: dict[str, int] = checkpoint.get("category_counts", {})

    category_targets: dict[str, int] = {
        cat: max(0, round(total_target * ratio))
        for cat, ratio in CATEGORY_RATIOS.items()
    }
    # 합계가 total_target과 맞지 않을 경우 persona에 나머지 배분
    diff = total_target - sum(category_targets.values())
    category_targets["persona"] += diff

    print(f"\n[카테고리별 목표]")
    for cat, tgt in category_targets.items():
        done = existing_by_cat.get(cat, 0)
        print(f"  - {cat}: {tgt}쌍 (기존 {done}쌍, 추가 필요 {max(0, tgt - done)}쌍)")

    # ── 생성 시작 ──
    all_new_data: list[dict[str, str]] = []
    category_new_counts: dict[str, int] = {}
    start_time = time.time()

    for cat, tgt in category_targets.items():
        already = existing_by_cat.get(cat, 0)
        if already >= tgt:
            print(f"\n[{cat}] 이미 목표 달성 — 건너뜀 ({already}/{tgt})")
            category_new_counts[cat] = 0
            continue

        print(f"\n[{cat}] 생성 시작 ({already} → {tgt}쌍)")
        new_items = await generate_category(
            category=cat,
            target_count=tgt,
            client=client,
            rpm_limiter=rpm_limiter,
            semaphore=semaphore,
            dedup=dedup,
            already_generated=already,
        )
        all_new_data.extend(new_items)
        category_new_counts[cat] = len(new_items)

        # 카테고리 완료 시 즉시 저장 (중단 대비)
        append_to_jsonl(output_path, new_items)
        logger.info(
            "category_done",
            category=cat,
            new=len(new_items),
            total=already + len(new_items),
        )

        # 체크포인트 갱신
        save_checkpoint({
            "total_target": total_target,
            "category_counts": {
                c: existing_by_cat.get(c, 0) + category_new_counts.get(c, 0)
                for c in CATEGORY_RATIOS
            },
            "output_path": str(output_path),
        })

    elapsed = time.time() - start_time

    # ── 최종 통계 ──
    all_data = existing_data + all_new_data
    total_generated = len(all_data)

    print(f"\n{'=' * 60}")
    print(f"  생성 완료!")
    print(f"  - 신규 생성:   {len(all_new_data):,}쌍")
    print(f"  - 총 데이터:   {total_generated:,}쌍")
    print(f"  - 소요 시간:   {elapsed:.1f}초")
    print(f"  - 저장 경로:   {output_path}")

    # ── train / eval 분리 ──
    # 전체 데이터를 섞어서 eval_ratio만큼 eval 파일로 분리
    random.shuffle(all_data)
    eval_count = max(1, round(total_generated * eval_ratio))
    eval_data = all_data[:eval_count]
    train_data = all_data[eval_count:]

    # eval 파일 덮어쓰기 (append 아닌 write)
    eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_output_path.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in eval_data) + "\n",
        encoding="utf-8",
    )

    # train 파일도 섞인 순서로 다시 쓰기 (이미 append되어 있으므로 덮어씀)
    output_path.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in train_data) + "\n",
        encoding="utf-8",
    )

    print(f"  - 학습 데이터: {len(train_data):,}쌍 → {output_path}")
    print(f"  - 검증 데이터: {len(eval_data):,}쌍 → {eval_output_path}")
    print(f"{'=' * 60}\n")

    # 체크포인트 초기화 (완료)
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("checkpoint_cleared")

    logger.info(
        "generation_complete",
        total=total_generated,
        train=len(train_data),
        eval=len(eval_data),
        elapsed_sec=round(elapsed, 1),
    )


# ══════════════════════════════════════════════════════════════
# CLI 인수 파서
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    CLI 인수를 파싱하여 Namespace로 반환한다.

    Returns:
        파싱된 인수 Namespace
    """
    parser = argparse.ArgumentParser(
        description="몽글이 파인튜닝 학습 데이터 자동 생성 (Solar API 사용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (800쌍 생성)
  PYTHONPATH=src uv run python scripts/generate_training_data.py

  # 1,000쌍 생성
  PYTHONPATH=src uv run python scripts/generate_training_data.py --total 1000

  # 출력 경로 지정
  PYTHONPATH=src uv run python scripts/generate_training_data.py \\
    --output data/finetune/mongle_train.jsonl \\
    --eval-output data/finetune/mongle_eval.jsonl

  # 중단 후 재개
  PYTHONPATH=src uv run python scripts/generate_training_data.py --resume

  # 비용 추정만 (API 호출 없음)
  PYTHONPATH=src uv run python scripts/generate_training_data.py --estimate-only
        """,
    )

    parser.add_argument(
        "--total",
        type=int,
        default=800,
        help="목표 생성 총 건수 (기본값: 800)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/finetune/mongle_train.jsonl",
        help="학습 데이터 출력 경로 (기본값: data/finetune/mongle_train.jsonl)",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default="data/finetune/mongle_eval.jsonl",
        help="검증 데이터 출력 경로 (기본값: data/finetune/mongle_eval.jsonl)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="검증 데이터 비율 (기본값: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="중단된 생성 작업을 이어서 진행",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="비용 추정만 출력하고 종료 (API 호출 없음)",
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
