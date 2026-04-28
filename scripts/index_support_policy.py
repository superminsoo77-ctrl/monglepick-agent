"""
고객센터 정책 RAG 인덱서 (v4, 2026-04-28).

마크다운 정책 문서를 읽어 청크 분할 → Solar 임베딩 → Qdrant `support_policy_v1` 에 UPSERT 한다.

설계 근거: docs/고객센터_AI에이전트_v4_재설계.md §6 (정책 RAG 스펙)

── 청킹 전략 ──────────────────────────────────────────────────────────────────
1. Level-2 헤딩(`^## `) 마다 청크 1차 분할
2. 코드블록(```) 안에 헤딩이 있어도 깨뜨리지 않음 — 코드블록 닫힌 후 다음 헤딩에서 분할
3. 청크 길이 1500자 초과 → 800자 슬라이딩 윈도우 (overlap 200) 로 2차 분할
4. 청크당 메타데이터: doc_id, doc_path, section, headings, policy_topic, doc_version,
   indexed_at, chunk_idx, text

── policy_topic 자동 추론 ──────────────────────────────────────────────────────
헤딩 + 본문 키워드 기반 우선순위 매칭:
  "등급" + ("혜택"|"테마"|"BRONZE"|"SILVER"|...) → grade_benefit
  "AI" + ("쿼터"|"한도"|"무료"|"3-소스")         → ai_quota
  "구독" + ("플랜"|"monthly"|"yearly")            → subscription
  "환불" + ("정책"|"기간"|"신청")                  → refund
  "리워드" + ("적립"|"활동"|"출석"|"리뷰")         → reward
  "결제" + ("Toss"|"카드"|"방법")                  → payment
  기타                                              → general

── CLI 옵션 ────────────────────────────────────────────────────────────────────
  --source <md_path>      인덱싱할 마크다운 파일 (반복 가능, 여러 문서 동시 처리)
  --clear-db              해당 doc_id 의 기존 청크를 삭제 후 재인덱싱
  --dry-run               청크만 출력, 임베딩/Qdrant 적재 없음
  --policy-topic <topic>  policy_topic 수동 지정 (미지정 시 자동 추론)

── 실행 방법 ────────────────────────────────────────────────────────────────────
  cd monglepick-agent
  PYTHONPATH=src uv run python scripts/index_support_policy.py \\
    --source docs/리워드_결제_설계서.md \\
    --source docs/결제_구독_시스템_설계_및_구현_보고서.md \\
    --clear-db

  # 드라이런 (청크 확인만, Qdrant/Solar 호출 없음)
  PYTHONPATH=src uv run python scripts/index_support_policy.py \\
    --source docs/리워드_결제_설계서.md \\
    --dry-run

── 사전 조건 ───────────────────────────────────────────────────────────────────
  - .env 에 QDRANT_URL, UPSTAGE_API_KEY 설정
  - Qdrant 서비스 실행 중 (보통 :6333)
  - 실제 Qdrant/Solar 호출 없이 청크 확인만 하려면 --dry-run 사용

── 주의 ────────────────────────────────────────────────────────────────────────
  - Solar embedding-passage: 100 RPM 제한 — 배치(50개)당 0.7초 딜레이 내장
  - --clear-db 는 해당 doc_id 만 삭제 (다른 문서 청크는 보존)
  - 여러 번 실행해도 point ID 가 deterministic(UUID5) 이므로 UPSERT — 멱등
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import NamedTuple

# ── 프로젝트 루트를 sys.path 에 추가 (PYTHONPATH=src 없이도 동작하도록 보조) ──
_AGENT_ROOT = Path(__file__).parent.parent
_SRC_PATH = _AGENT_ROOT / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

# ── 프로젝트 패키지 import — sys.path 설정 이후에 위치해야 한다 ──
# 모듈 상단에서 import 하면 `patch("index_support_policy.embed_texts", ...)` 패턴으로
# 단위 테스트에서 mock 대체가 가능하다.
# (함수 내부 지연 import 로 두면 patch 대상 경로가 다이버전트해 테스트 불가)
from monglepick.data_pipeline.embedder import embed_texts  # noqa: E402
from monglepick.db.clients import (  # noqa: E402
    ensure_support_policy_collection,
    get_qdrant,
)


# ============================================================
# 상수
# ============================================================

#: Qdrant 컬렉션명 (db/clients.py 의 SUPPORT_POLICY_COLLECTION 과 동일)
COLLECTION_NAME: str = "support_policy_v1"

#: 청킹 1차 기준 — Level-2 헤딩 (`## 제목`)
_HEADING_PATTERN: re.Pattern = re.compile(r"^## .+", re.MULTILINE)

#: 청크 길이 상한 (자). 초과 시 슬라이딩 윈도우로 2차 분할.
CHUNK_MAX_CHARS: int = 1500

#: 슬라이딩 윈도우 크기 및 오버랩 (자)
SLIDING_WINDOW_SIZE: int = 800
SLIDING_OVERLAP: int = 200

#: 한국 표준시 UTC+9
_KST = timezone(timedelta(hours=9))

#: UUID 네임스페이스 — 동일 (doc_id, chunk_idx) 는 항상 같은 point ID
_CHUNK_UUID_NS: uuid.UUID = uuid.UUID("d7a3c120-4f56-7890-abcd-ef0123456789")

# ── 청크 필터 — 구현 세부 청크 배제 키워드 목록 ──────────────────────────────
# 헤딩에 아래 키워드가 포함된 섹션은 정책 RAG 에 부적합한 구현 세부 섹션으로 판단한다.
# 설계 근거: docs/고객센터_AI에이전트_v4_재설계.md §6.3
#   "인덱싱 제외 — 구현 세부는 RAG 에 넣지 않는다."
_EXCLUDE_HEADING_KEYWORDS: tuple[str, ...] = (
    "구현", "코드", "트랜잭션", "AOP", "클래스 구조", "아키텍처",
    "부록", "머리말", "목차", "수정 이력",
    "Repository", "Service 레이어", "Bean 분리", "FOR UPDATE",
    "@Transactional", "JPA", "MyBatis",
)

# ── policy_topic 추론 규칙 (우선순위 순) ──────────────────────────────────────
# 각 항목: (topic_name, 필수 키워드 set A, 부가 키워드 set B)
# 텍스트(헤딩+본문)에 A 중 하나 AND B 중 하나가 있으면 해당 topic 으로 분류.
# 리스트 순서가 우선순위 — 위쪽 규칙이 먼저 매칭되면 아래 규칙은 평가하지 않음.
_TOPIC_RULES: list[tuple[str, set[str], set[str]]] = [
    (
        "grade_benefit",
        {"등급"},
        {"혜택", "테마", "BRONZE", "SILVER", "GOLD", "PLATINUM", "DIAMOND",
         "강냉이", "팝콘", "카라멜", "몽글팝콘", "몽아일체", "알갱이", "일일"},
    ),
    (
        "ai_quota",
        {"AI"},
        {"쿼터", "한도", "무료", "3-소스", "GRADE_FREE", "SUB_BONUS", "PURCHASED",
         "daily_ai", "purchased_ai", "monthly_coupon", "이용권"},
    ),
    (
        "subscription",
        {"구독"},
        {"플랜", "monthly", "yearly", "monthly_basic", "monthly_premium",
         "yearly_basic", "yearly_premium", "basic", "premium"},
    ),
    (
        "refund",
        {"환불"},
        {"정책", "기간", "신청", "취소", "부분", "전액"},
    ),
    (
        "reward",
        {"리워드"},
        {"적립", "활동", "출석", "리뷰", "보상", "경험치", "뱃지"},
    ),
    (
        "payment",
        {"결제"},
        {"Toss", "카드", "방법", "수단", "PG", "webhook", "idempotency",
         "confirm", "승인", "취소", "billing"},
    ),
]


# ============================================================
# 청크 데이터 클래스
# ============================================================

class PolicyChunk(NamedTuple):
    """
    정책 문서 청크 단위.

    Qdrant point 에 저장될 메타데이터 + 벡터 임베딩 입력 텍스트를 보관한다.
    """
    doc_id: str          # 문서 식별자 (파일명 기반, ex: "rewards_payment_v3")
    doc_path: str        # 원본 파일 상대/절대 경로
    section: str         # "§N 헤딩 텍스트" 형식의 섹션 레이블
    headings: list[str]  # 이 청크가 속한 헤딩 경로 (Level-2 이상)
    policy_topic: str    # grade_benefit / ai_quota / subscription / refund / reward / payment / general
    doc_version: str     # 문서 버전 (파일 내 "v3.4" 같은 태그 또는 "unknown")
    indexed_at: str      # ISO 8601 KST 타임스탬프
    chunk_idx: int       # 문서 내 순번 (0-based)
    text: str            # 청크 원문 (검색 결과 표시 + 임베딩 입력)


# ============================================================
# 유틸리티 함수
# ============================================================

def _doc_id_from_path(md_path: Path) -> str:
    """
    마크다운 파일 경로에서 doc_id 를 파생한다.

    파일명(확장자 제외)을 소문자 + 공백→언더스코어 변환.
    예: "docs/리워드_결제_설계서.md" → "리워드_결제_설계서"

    Args:
        md_path: 마크다운 파일 경로

    Returns:
        doc_id 문자열 (Qdrant payload 에 저장될 식별자)
    """
    return md_path.stem  # Path.stem = 파일명에서 마지막 확장자 제거


def _extract_doc_version(text: str) -> str:
    """
    문서 전체 텍스트에서 버전 태그를 추출한다.

    "v3.4", "v2.1", "v10.0" 형태의 첫 번째 매칭을 반환한다.
    없으면 "unknown" 반환.

    Args:
        text: 문서 전체 텍스트

    Returns:
        "v3.4" 형태 버전 문자열 또는 "unknown"
    """
    match = re.search(r"\bv(\d+\.\d+)\b", text)
    return f"v{match.group(1)}" if match else "unknown"


def _infer_policy_topic(text: str, override: str | None = None) -> str:
    """
    청크 텍스트(헤딩+본문)에서 policy_topic 을 자동 추론한다.

    `override` 가 지정되면 추론 없이 그 값을 반환 (CLI --policy-topic 옵션).

    추론 규칙 (우선순위 순, _TOPIC_RULES 리스트 참조):
      1. "등급" + 등급명/혜택 키워드 → grade_benefit
      2. "AI" + 쿼터/한도/소스 키워드 → ai_quota
      3. "구독" + 플랜/monthly/yearly  → subscription
      4. "환불" + 정책/기간/신청        → refund
      5. "리워드" + 적립/활동/출석      → reward
      6. "결제" + Toss/카드/방법        → payment
      7. 기타                            → general

    Args:
        text: 청크 텍스트 (헤딩 + 본문 포함)
        override: CLI 로 지정된 강제 topic. None 이면 자동 추론.

    Returns:
        topic 문자열 (grade_benefit / ai_quota / subscription / refund / reward / payment / general)
    """
    if override:
        return override

    for topic, must_set, bonus_set in _TOPIC_RULES:
        # must_set 중 하나 AND bonus_set 중 하나가 텍스트에 존재해야 매칭
        has_must = any(kw in text for kw in must_set)
        has_bonus = any(kw in text for kw in bonus_set)
        if has_must and has_bonus:
            return topic

    return "general"


def _is_code_heavy(text: str, threshold: float = 0.5) -> bool:
    """
    청크 본문에서 ``` 코드블록이 차지하는 비율이 threshold 이상이면 True 를 반환한다.

    정책 RAG 에는 코드 예시보다 사용자 안내 텍스트가 적합하다.
    코드블록 비중이 높은 청크는 구현 세부 설명일 가능성이 높으므로 제외한다.

    예시:
        - ```python\\n...\\n``` 이 전체의 60% 를 차지하는 청크 → True (제외)
        - 코드블록 없는 등급 혜택표 텍스트 → False (보존)

    Args:
        text: 청크 원문
        threshold: 코드블록 비중 임계값 (기본 0.5 = 50%)

    Returns:
        코드블록 비중 >= threshold 이면 True, 아니면 False
    """
    if not text:
        return False
    # ``` 로 시작해서 ``` 로 끝나는 코드블록 전체를 추출 (DOTALL 모드로 줄바꿈 포함)
    code_blocks = re.findall(r"```[\s\S]*?```", text)
    code_chars = sum(len(b) for b in code_blocks)
    return code_chars / len(text) >= threshold


def _is_implementation_section(headings: list[str]) -> bool:
    """
    청크의 headings 리스트(마지막 헤딩)에 구현 세부 키워드가 포함되면 True 를 반환한다.

    _EXCLUDE_HEADING_KEYWORDS 에 정의된 키워드 중 하나라도 마지막 헤딩에 있으면
    해당 청크는 구현/코드/아키텍처 섹션으로 간주하여 정책 RAG 에서 제외한다.

    빈 headings 리스트(Level-2 헤딩이 없는 전문 섹션 등)는 False 를 반환한다.

    예시:
        - headings=["§13.1 클래스 구조 및 책임 분리"] → True (제외)
        - headings=["§4.5 등급 혜택표"]               → False (보존)
        - headings=[]                                   → False (보존)

    Args:
        headings: PolicyChunk.headings 필드 (빈 리스트 가능)

    Returns:
        마지막 헤딩에 구현 세부 키워드가 있으면 True, 아니면 False
    """
    if not headings:
        return False
    # headings 의 마지막 항목이 현재 청크가 속한 섹션 헤딩
    last = headings[-1]
    return any(kw in last for kw in _EXCLUDE_HEADING_KEYWORDS)


def _is_too_short(text: str, min_chars: int = 200) -> bool:
    """
    청크 텍스트가 min_chars 미만이면 True 를 반환한다.

    작성자/날짜/링크 목차 등 메타 청크는 정책 RAG 검색 품질을 떨어뜨린다.
    공백 제거 후 200자 미만이면 실질적 정보가 없는 메타 청크로 간주한다.

    예시:
        - "작성자: 홍길동\\n날짜: 2026-04-28" (52자) → True (제외)
        - "BRONZE 등급의 일일 AI 한도는 3회입니다. ..." (500자) → False (보존)

    Args:
        text: 청크 원문
        min_chars: 최소 글자 수 임계값 (기본 200자)

    Returns:
        공백 제거 후 길이 < min_chars 이면 True, 아니면 False
    """
    return len(text.strip()) < min_chars


def _chunk_uuid(doc_id: str, chunk_idx: int) -> str:
    """
    (doc_id, chunk_idx) 조합으로 deterministic UUID5 를 생성한다.

    같은 입력은 항상 같은 UUID — 재인덱싱 시 같은 point ID 로 UPSERT 되어 중복 방지.
    Qdrant point ID 로 사용된다.

    Args:
        doc_id: 문서 식별자 (예: "리워드_결제_설계서")
        chunk_idx: 청크 순번 (0-based)

    Returns:
        UUID 문자열 (예: "a1b2c3d4-...")
    """
    key = f"{doc_id}::{chunk_idx}"
    return str(uuid.uuid5(_CHUNK_UUID_NS, key))


# ============================================================
# 청킹 로직
# ============================================================

def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """
    마크다운 텍스트를 Level-2 헤딩(`## `) 기준으로 1차 분할한다.

    코드블록(```) 안에 헤딩 패턴이 있어도 깨뜨리지 않는다.
    코드블록이 열린 상태에서 헤딩이 나오면 코드블록이 닫힌 후
    다음 헤딩에서 분할한다.

    구현 방식:
    - 줄 단위로 순회하며 ``` 토글로 코드블록 상태를 추적
    - 코드블록 밖에서 `## ` 로 시작하는 줄을 만나면 이전 청크를 확정하고 새 청크 시작

    Args:
        text: 마크다운 전체 텍스트

    Returns:
        [(heading, body_text)] 리스트.
        heading: "## 제목 텍스트" (첫 번째 헤딩 줄)
        body_text: 해당 헤딩 이후 다음 Level-2 헤딩 전까지의 텍스트 (헤딩 포함)
    """
    segments: list[tuple[str, str]] = []
    current_heading: str = ""          # 현재 청크의 헤딩 텍스트
    current_lines: list[str] = []      # 현재 청크의 줄 목록
    in_code_block: bool = False        # ``` 코드블록 안인지 여부

    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n\r")

        # ── 코드블록 토글 추적 ──
        # ``` 로 시작하는 줄(공백 없이 또는 언어 지정자 포함)을 코드블록 경계로 인식
        if stripped.lstrip().startswith("```"):
            in_code_block = not in_code_block

        # ── Level-2 헤딩 감지 (코드블록 밖에서만) ──
        if not in_code_block and stripped.startswith("## "):
            # 이전 청크가 있으면 확정
            if current_lines:
                body = "".join(current_lines)
                segments.append((current_heading, body))
            # 새 청크 시작
            current_heading = stripped
            current_lines = [line]
        else:
            current_lines.append(line)

    # 마지막 청크 확정
    if current_lines:
        body = "".join(current_lines)
        segments.append((current_heading, body))

    return segments


def _sliding_window_split(text: str, window: int, overlap: int) -> list[str]:
    """
    텍스트를 슬라이딩 윈도우로 분할한다.

    청크 길이가 `window` 를 초과하지 않도록 하되, 연속 청크 간에
    `overlap` 글자가 겹쳐 문맥 연속성을 유지한다.

    단어 경계(공백) 가 아닌 글자 단위로 분할한다.
    (한국어 텍스트는 단어 경계가 불명확하므로 글자 단위가 더 안전)

    Args:
        text: 분할할 텍스트
        window: 최대 청크 길이 (자)
        overlap: 이전 청크와 겹치는 글자 수

    Returns:
        분할된 텍스트 리스트 (각 항목 ≤ window 자)
    """
    if len(text) <= window:
        return [text]

    chunks: list[str] = []
    step = window - overlap  # 슬라이드 이동 폭
    start = 0

    while start < len(text):
        end = start + window
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step

    return chunks


def build_chunks(
    md_path: Path,
    *,
    policy_topic_override: str | None = None,
) -> list[PolicyChunk]:
    """
    마크다운 파일을 읽어 PolicyChunk 리스트를 생성한다.

    처리 단계:
    1. 파일 읽기
    2. Level-2 헤딩 단위 1차 분할 (`_split_by_heading`)
    3. 각 세그먼트가 CHUNK_MAX_CHARS 초과 시 슬라이딩 윈도우 2차 분할
    4. 각 청크에 메타데이터 부착
    5. 구현 세부 청크 필터링 3종 적용 (임베딩/적재 전 마지막 관문)
       - code_heavy    : 코드블록 비중 ≥ 50% 청크 제외
       - implementation: 구현 세부 헤딩 키워드 매칭 청크 제외
       - too_short     : 200자 미만 메타 청크 제외
    6. 보존 청크의 chunk_idx 를 0-based 로 재부여

    Args:
        md_path: 마크다운 파일 경로
        policy_topic_override: --policy-topic CLI 옵션 값. None 이면 자동 추론.

    Returns:
        필터 적용 후 PolicyChunk 리스트 (chunk_idx 는 0-based 연속 번호)

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        UnicodeDecodeError: 파일이 UTF-8 이 아닐 때
    """
    # ── 파일 읽기 ──
    raw_text = md_path.read_text(encoding="utf-8")

    # ── 문서 수준 메타데이터 추출 ──
    doc_id = _doc_id_from_path(md_path)
    doc_version = _extract_doc_version(raw_text)
    indexed_at = datetime.now(_KST).strftime("%Y-%m-%dT%H:%M:%S+09:00")

    # ── Level-2 헤딩 단위 1차 분할 ──
    segments = _split_by_heading(raw_text)

    chunks: list[PolicyChunk] = []
    chunk_idx = 0                    # 문서 전체 기준 청크 순번
    heading_counter = 0              # 섹션 번호 (§1, §2, ...)

    for heading, body in segments:
        heading_counter += 1

        # ── 섹션 레이블 (§N 헤딩텍스트) ──
        heading_clean = heading.lstrip("#").strip()
        section_label = f"§{heading_counter} {heading_clean}" if heading_clean else f"§{heading_counter}"

        # ── policy_topic 추론 ──
        topic = _infer_policy_topic(body, override=policy_topic_override)

        # ── 청크 길이 초과 시 슬라이딩 윈도우 2차 분할 ──
        if len(body) > CHUNK_MAX_CHARS:
            sub_texts = _sliding_window_split(body, SLIDING_WINDOW_SIZE, SLIDING_OVERLAP)
        else:
            sub_texts = [body]

        for sub_idx, sub_text in enumerate(sub_texts):
            # 서브청크별 topic 재추론 (sub_text 가 더 정확한 범위)
            sub_topic = _infer_policy_topic(sub_text, override=policy_topic_override)

            chunks.append(PolicyChunk(
                doc_id=doc_id,
                doc_path=str(md_path),
                section=section_label,
                headings=[heading_clean] if heading_clean else [],
                policy_topic=sub_topic,
                doc_version=doc_version,
                indexed_at=indexed_at,
                chunk_idx=chunk_idx,
                text=sub_text,
            ))
            chunk_idx += 1

    # ── 청크 필터링 — 구현 세부 청크 배제 (임베딩/적재 전 마지막 관문) ──────────
    # 설계 근거: docs/고객센터_AI에이전트_v4_재설계.md §6.3
    #   구현 세부(코드블록 과다, 구현 헤딩, 메타 전문)는 정책 RAG 에 부적합.
    #   필터링 통계는 dry-run/실제 적재 모두에서 동일하게 출력된다.
    raw_count = len(chunks)
    filter_stats: dict[str, int] = {
        "code_heavy": 0,        # 코드블록 비중 ≥ 50% 청크
        "implementation": 0,    # 구현 세부 헤딩 키워드 매칭 청크
        "too_short": 0,         # 200자 미만 메타 청크
    }
    filtered_chunks: list[PolicyChunk] = []

    for chunk in chunks:
        if _is_code_heavy(chunk.text):
            filter_stats["code_heavy"] += 1
            continue
        if _is_implementation_section(chunk.headings):
            filter_stats["implementation"] += 1
            continue
        if _is_too_short(chunk.text):
            filter_stats["too_short"] += 1
            continue
        filtered_chunks.append(chunk)

    # 필터링 결과 출력 (dry-run/실제 적재 모두 동일)
    kept_count = len(filtered_chunks)
    excluded_count = raw_count - kept_count
    if excluded_count > 0:
        print(
            f"  [필터] 제외 {excluded_count}개 "
            f"(code_heavy={filter_stats['code_heavy']}, "
            f"implementation={filter_stats['implementation']}, "
            f"too_short={filter_stats['too_short']})"
        )
    print(f"  [필터] 보존 {kept_count}개 / 원본 {raw_count}개")

    # chunk_idx 를 보존된 청크 기준으로 재부여
    # (삭제된 청크가 중간에 있으면 순번에 구멍이 생기므로 재정렬)
    final_chunks: list[PolicyChunk] = []
    for new_idx, chunk in enumerate(filtered_chunks):
        if chunk.chunk_idx != new_idx:
            # NamedTuple 은 불변이므로 _replace 로 새 인스턴스 생성
            chunk = chunk._replace(chunk_idx=new_idx)
        final_chunks.append(chunk)

    return final_chunks


# ============================================================
# 드라이런 출력
# ============================================================

def _print_dry_run(chunks: list[PolicyChunk], md_path: Path) -> None:
    """
    드라이런 모드에서 청크 목록을 사람이 읽기 좋게 출력한다.

    `chunks` 는 이미 필터 3종(code_heavy / implementation / too_short)이
    적용된 최종 목록이다.  필터 통계는 `build_chunks` 내부에서 먼저 출력된다.

    임베딩/Qdrant 호출은 수행하지 않는다.

    Args:
        chunks: 필터 적용 후 최종 PolicyChunk 리스트
        md_path: 원본 마크다운 파일 경로 (표시용)
    """
    print(f"\n{'=' * 70}")
    # 필터 적용 후 청크 수임을 명시한다
    print(f"[DRY-RUN] {md_path.name}  →  {len(chunks)}개 청크 (필터 적용 후)")
    print(f"{'=' * 70}")

    for c in chunks:
        # 청크 텍스트 미리보기 (앞 120자)
        preview = c.text.replace("\n", " ").strip()[:120]
        print(
            f"  [{c.chunk_idx:03d}] section={c.section!r}  "
            f"topic={c.policy_topic}  len={len(c.text)}자\n"
            f"         {preview}{'...' if len(c.text) > 120 else ''}"
        )

    print()

    # topic 분포 요약
    from collections import Counter
    topic_dist = Counter(c.policy_topic for c in chunks)
    print("  topic 분포 (필터 적용 후):")
    for topic, cnt in sorted(topic_dist.items(), key=lambda x: -x[1]):
        print(f"    {topic:<20} {cnt}개")
    print()


# ============================================================
# Qdrant UPSERT 로직
# ============================================================

async def _delete_by_doc_id(doc_id: str) -> int:
    """
    Qdrant 에서 doc_id 에 해당하는 모든 point 를 삭제한다.

    `--clear-db` 옵션 사용 시 재인덱싱 전에 호출한다.
    다른 doc_id 의 청크는 건드리지 않는다.

    Args:
        doc_id: 삭제할 문서 식별자

    Returns:
        삭제 요청 결과 (Qdrant UpdateResult, 실제 삭제 수는 알 수 없음)

    Raises:
        Exception: Qdrant 연결 실패 시 (호출자가 try/except 처리)
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = await get_qdrant()

    result = await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        ),
    )
    return result


async def _upsert_chunks(chunks: list[PolicyChunk], batch_size: int = 50) -> int:
    """
    PolicyChunk 리스트를 임베딩하여 Qdrant 에 UPSERT 한다.

    임베딩:
    - `embed_texts` (동기) 를 `asyncio.to_thread` 로 비동기화
    - batch_size=50 — Upstage 100 RPM + 0.7초 딜레이 (embedder.py 내장)

    UPSERT:
    - point ID: `_chunk_uuid(doc_id, chunk_idx)` — deterministic, 재실행 멱등
    - payload: PolicyChunk 의 모든 필드 (text 포함)
    - vector: 4096차원 float32 리스트

    Args:
        chunks: 적재할 PolicyChunk 리스트
        batch_size: 임베딩 API 배치 크기 (기본 50)

    Returns:
        upsert 완료된 청크 수

    Raises:
        Exception: Upstage API 실패 또는 Qdrant 연결 실패 (호출자가 try/except 처리)
    """
    from qdrant_client.models import PointStruct

    client = await get_qdrant()

    # ── 임베딩 텍스트 추출 ──
    # text 필드 그대로 사용 (Solar embedding-passage 는 한국어 긴 텍스트 최적화)
    texts = [c.text for c in chunks]

    print(f"  임베딩 시작: {len(texts)}개 청크 (배치={batch_size}, dim=4096)")
    embed_start = time.monotonic()

    # 동기 함수 embed_texts 를 to_thread 로 비동기화 — event loop 블로킹 방지
    vectors = await asyncio.to_thread(embed_texts, texts, batch_size)

    embed_elapsed = time.monotonic() - embed_start
    print(f"  임베딩 완료: {embed_elapsed:.1f}초")

    if vectors.shape[0] != len(chunks):
        raise RuntimeError(
            f"임베딩 개수 불일치: expected {len(chunks)}, got {vectors.shape[0]}"
        )

    # ── Qdrant PointStruct 구성 ──
    points = [
        PointStruct(
            id=_chunk_uuid(c.doc_id, c.chunk_idx),
            vector=vectors[i].tolist(),  # numpy float32 → Python float 리스트
            payload={
                "doc_id": c.doc_id,
                "doc_path": c.doc_path,
                "section": c.section,
                "headings": c.headings,
                "policy_topic": c.policy_topic,
                "doc_version": c.doc_version,
                "indexed_at": c.indexed_at,
                "chunk_idx": c.chunk_idx,
                "text": c.text,
            },
        )
        for i, c in enumerate(chunks)
    ]

    # ── Qdrant UPSERT (배치 분할) ──
    # 청크가 많으면 Qdrant 단일 upsert 호출로 처리.
    # 1000개 이상이면 Qdrant 권장 배치(100~500) 로 나눠 호출.
    qdrant_batch = 200
    upserted = 0
    for i in range(0, len(points), qdrant_batch):
        batch_points = points[i : i + qdrant_batch]
        await client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch_points,
        )
        upserted += len(batch_points)
        print(f"  UPSERT 진행: {upserted}/{len(points)}")

    return upserted


# ============================================================
# 메인 진입점
# ============================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    CLI 인수를 파싱하여 Namespace 로 반환한다.

    Args:
        argv: 파싱할 인수 리스트. None 이면 sys.argv[1:] 사용.

    Returns:
        argparse.Namespace (source, clear_db, dry_run, policy_topic 필드)
    """
    parser = argparse.ArgumentParser(
        prog="index_support_policy.py",
        description="고객센터 정책 RAG 인덱서 — 마크다운 → 청크 → Solar 임베딩 → Qdrant 적재",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 1차 인덱싱 (두 문서 동시)
  PYTHONPATH=src uv run python scripts/index_support_policy.py \\
    --source docs/리워드_결제_설계서.md \\
    --source docs/결제_구독_시스템_설계_및_구현_보고서.md \\
    --clear-db

  # 드라이런 — 청크 출력만 (Qdrant/Solar 호출 없음)
  PYTHONPATH=src uv run python scripts/index_support_policy.py \\
    --source docs/리워드_결제_설계서.md \\
    --dry-run

  # policy_topic 수동 지정
  PYTHONPATH=src uv run python scripts/index_support_policy.py \\
    --source docs/특수문서.md \\
    --policy-topic ai_quota
""",
    )
    parser.add_argument(
        "--source",
        dest="source",
        metavar="MD_PATH",
        action="append",
        required=True,
        help="인덱싱할 마크다운 파일 경로 (반복 가능)",
    )
    parser.add_argument(
        "--clear-db",
        dest="clear_db",
        action="store_true",
        default=False,
        help="해당 doc_id 의 기존 Qdrant 청크를 삭제 후 재인덱싱",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="청크만 출력, 임베딩/Qdrant 적재 없음",
    )
    parser.add_argument(
        "--policy-topic",
        dest="policy_topic",
        metavar="TOPIC",
        default=None,
        help=(
            "policy_topic 수동 지정 (미지정 시 헤딩+키워드 자동 추론). "
            "허용값: grade_benefit, ai_quota, subscription, refund, reward, payment, general"
        ),
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> None:
    """
    인덱서 메인 루틴.

    1. CLI 인수 파싱 및 파일 존재 확인
    2. 각 문서에 대해:
       a. 청크 생성 (build_chunks)
       b. --dry-run 이면 출력 후 종료
       c. --clear-db 이면 기존 doc_id 청크 삭제
       d. 컬렉션 부트스트랩 (ensure_support_policy_collection)
       e. 임베딩 + Qdrant UPSERT (_upsert_chunks)

    Args:
        argv: 파싱할 인수 리스트. None 이면 sys.argv[1:] 사용.
    """
    args = _parse_args(argv)

    print("=" * 70)
    print("몽글픽 고객센터 정책 RAG 인덱서 시작")
    print("=" * 70)

    # ── 입력 파일 목록 검증 ──
    md_paths: list[Path] = []
    for raw in args.source:
        p = Path(raw)
        if not p.exists():
            # 스크립트 위치 기준 상대 경로도 시도 (프로젝트 루트 기준)
            alt = _AGENT_ROOT / raw
            if alt.exists():
                p = alt
            else:
                print(f"[오류] 파일이 존재하지 않습니다: {raw}")
                print(f"       현재 작업 디렉토리: {Path.cwd()}")
                sys.exit(1)
        md_paths.append(p)

    print(f"\n처리 대상: {len(md_paths)}개 문서")
    for p in md_paths:
        print(f"  - {p}")

    if args.dry_run:
        print("\n[DRY-RUN 모드] 임베딩/Qdrant 적재 없이 청크만 출력합니다.\n")

    # ── 문서별 처리 ──
    total_chunks = 0
    overall_start = time.monotonic()

    for md_path in md_paths:
        print(f"\n{'─' * 70}")
        print(f"[문서] {md_path.name}")
        print(f"{'─' * 70}")

        # 1) 청크 생성
        chunk_start = time.monotonic()
        try:
            chunks = build_chunks(md_path, policy_topic_override=args.policy_topic)
        except (FileNotFoundError, UnicodeDecodeError) as e:
            print(f"[오류] 파일 읽기 실패: {e}")
            sys.exit(1)

        chunk_elapsed = time.monotonic() - chunk_start
        print(f"  청크 생성 완료: {len(chunks)}개 / {chunk_elapsed:.2f}초")

        # 2) 드라이런 — 출력만 하고 다음 문서로
        if args.dry_run:
            _print_dry_run(chunks, md_path)
            total_chunks += len(chunks)
            continue

        doc_id = _doc_id_from_path(md_path)

        # 3) 컬렉션 부트스트랩 (없으면 생성, 있으면 idempotent)
        print(f"\n  [1/3] 컬렉션 '{COLLECTION_NAME}' 확인/생성 중...")
        try:
            await ensure_support_policy_collection()
            print(f"        컬렉션 준비 완료")
        except Exception as e:
            print(f"[오류] 컬렉션 생성 실패: {e}")
            print("       Qdrant 서비스 실행 여부와 QDRANT_URL 환경변수를 확인하세요.")
            sys.exit(1)

        # 4) --clear-db: 해당 doc_id 의 기존 청크 삭제
        if args.clear_db:
            print(f"\n  [2/3] 기존 청크 삭제 (doc_id={doc_id!r}) ...")
            try:
                await _delete_by_doc_id(doc_id)
                print(f"        삭제 요청 완료")
            except Exception as e:
                print(f"[오류] 기존 청크 삭제 실패: {e}")
                print("       계속 진행합니다 (UPSERT 로 덮어씌워짐).")
        else:
            print(f"\n  [2/3] --clear-db 미지정 → 기존 청크 보존 (UPSERT 덮어쓰기)")

        # 5) 임베딩 + Qdrant UPSERT
        print(f"\n  [3/3] 임베딩 + UPSERT 시작...")
        upsert_start = time.monotonic()
        try:
            upserted = await _upsert_chunks(chunks)
        except Exception as e:
            print(f"[오류] UPSERT 실패: {e}")
            print("       UPSTAGE_API_KEY 환경변수와 Upstage 계정 quota 를 확인하세요.")
            sys.exit(1)

        upsert_elapsed = time.monotonic() - upsert_start
        print(
            f"  완료: {upserted}개 청크 UPSERT / {upsert_elapsed:.1f}초"
        )
        total_chunks += upserted

    # ── 전체 요약 ──
    overall_elapsed = time.monotonic() - overall_start
    print(f"\n{'=' * 70}")
    if args.dry_run:
        print(
            f"[DRY-RUN 완료] {len(md_paths)}개 문서 / 총 {total_chunks}개 청크 생성 예정"
        )
        print("               실제 적재하려면 --dry-run 옵션을 제거하고 다시 실행하세요.")
    else:
        print(
            f"완료: {len(md_paths)}개 문서 / 총 {total_chunks}개 청크 / "
            f"{overall_elapsed:.1f}초"
        )
        print(f"  컬렉션: {COLLECTION_NAME}")
        print(f"  다음 단계:")
        print(f"    - support_assistant v4 의 lookup_policy tool 로 검색 활용")
        print(f"    - 정책 문서 변경 시 --clear-db 로 재인덱싱")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
