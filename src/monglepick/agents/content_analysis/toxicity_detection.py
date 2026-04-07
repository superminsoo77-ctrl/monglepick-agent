"""
비속어/혐오 표현 검출 모듈 (§8-2 기능4).

처리 흐름:
1. 모듈 레벨 한국어 비속어 사전(약 50개 패턴)으로 정규식 매칭
2. toxicity_score = 검출 단어 수 / max(총 단어 수, 1) (0~1 클램핑)
3. content_type별 민감도 가중치 적용 (chat 느슨, review/comment 엄격)
4. 가중치 적용 점수 기준으로 action 결정 (pass/warning/blind/block)

LLM 없이 순수 규칙 기반으로 처리하여 응답 지연 없음.
에러 시 안전 기본값(pass)을 반환하고 에러를 전파하지 않는다.
"""

from __future__ import annotations

import re

import structlog

from monglepick.agents.content_analysis.models import (
    ProfanityCheckInput,
    ProfanityCheckOutput,
)

logger = structlog.get_logger()

# ============================================================
# 한국어 비속어/혐오표현 사전
# ============================================================
# 실제 운영에서는 외부 파일이나 DB에서 로드하는 방식으로 확장 가능.
# 정규식 패턴을 사용하여 변형 표현도 탐지한다 (예: 시*발, ㅅㅂ).
# 목록은 대표적인 단어로 구성하며, 실제 서비스에서는 보안팀과 협의하여 관리한다.

_RAW_PATTERNS: list[str] = [
    # ── 욕설 (기본형) ──
    r"시발",
    r"씨발",
    r"씨팔",
    r"ㅅㅂ",
    r"sibal",
    r"ssibal",
    r"개새끼",
    r"개색기",
    r"개세끼",
    r"ㄱㅅㄲ",
    r"병신",
    r"ㅂㅅ",
    r"byungsin",
    r"지랄",
    r"ㅈㄹ",
    r"미친놈",
    r"미친년",
    r"미친새끼",
    r"ㅁㅊ",
    r"fuck",
    r"shit",
    r"bitch",
    r"asshole",
    r"bastard",
    r"wtf",
    # ── 혐오 표현 (성별/지역/인종) ──
    r"된장녀",
    r"김치녀",
    r"한남충",
    r"페미나치",
    r"홍어",
    r"짱깨",
    r"쪽발이",
    r"니거",
    r"깜둥",
    # ── 성적 비하 ──
    r"창녀",
    r"걸레년",
    r"보지",
    r"자지",
    r"섹스",
    r"야동",
    r"포르노",
    # ── 폭력/위협 ──
    r"죽여버",
    r"뒤져",
    r"패버",
    r"박살",
    r"쳐죽",
    # ── 초성/변형 패턴 ──
    r"ㅆㅂ",
    r"ㅈㄴ",
    r"존나",
    r"개같은",
    r"썅",
]

# 컴파일된 정규식 패턴 목록 (대소문자 무시, 전체 단어가 아닌 부분 포함 검색)
_COMPILED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in _RAW_PATTERNS
]

# ============================================================
# content_type별 민감도 가중치
# ============================================================
# 높을수록 같은 독성 점수라도 더 강한 처리 액션이 적용됨.
# - chat: 실시간 대화라 어느 정도 허용 (느슨)
# - post/comment: 중간
# - review: 공개 콘텐츠라 엄격
_SENSITIVITY_WEIGHTS: dict[str, float] = {
    "chat":    0.7,   # 느슨 — 일상 대화에서 경미한 표현 허용
    "post":    1.0,   # 기본
    "comment": 1.1,   # 약간 엄격
    "review":  1.3,   # 엄격 — 공개 리뷰는 품질 관리 필요
}
_DEFAULT_WEIGHT = 1.0  # 정의되지 않은 content_type의 기본 가중치

# ============================================================
# 액션 결정 임계값 (가중치 적용 후 점수 기준)
# ============================================================
_THRESHOLD_WARNING = 0.0   # score > 0 이면 warning 이상
_THRESHOLD_BLIND   = 0.05  # score >= 0.05 이면 blind
_THRESHOLD_BLOCK   = 0.20  # score >= 0.20 이면 block


# ============================================================
# 내부 유틸 함수
# ============================================================

def _detect_toxic_words(text: str) -> list[str]:
    """
    텍스트에서 비속어/혐오표현 패턴을 찾아 매칭된 단어 목록을 반환한다.

    동일 패턴이 여러 번 등장해도 1회로 계산하여 중복을 제거한다.

    Args:
        text: 검사할 텍스트

    Returns:
        매칭된 고유 표현 목록
    """
    detected: list[str] = []
    seen: set[str] = set()

    for pattern in _COMPILED_PATTERNS:
        for m in pattern.finditer(text):
            word = m.group()
            if word not in seen:
                detected.append(word)
                seen.add(word)

    return detected


def _compute_toxicity_score(detected_count: int, word_count: int) -> float:
    """
    독성 점수를 계산한다.

    score = 검출 단어 수 / max(총 단어 수, 1)
    결과를 0.0~1.0으로 클램핑한다.

    Args:
        detected_count: 검출된 비속어 수
        word_count    : 텍스트 총 단어 수

    Returns:
        0.0~1.0 사이의 float
    """
    if detected_count == 0:
        return 0.0
    raw_score = detected_count / max(word_count, 1)
    return min(max(raw_score, 0.0), 1.0)


def _decide_action(weighted_score: float) -> str:
    """
    가중치 적용 독성 점수를 기반으로 처리 액션을 결정한다.

    액션 기준:
    - score == 0.0       → "pass"    (정상, 처리 불필요)
    - 0 < score < 0.05  → "warning" (경고 표시)
    - 0.05 <= score < 0.20 → "blind"   (콘텐츠 블라인드)
    - score >= 0.20     → "block"   (작성 차단)

    Args:
        weighted_score: 가중치 적용된 독성 점수 (0~1)

    Returns:
        "pass" | "warning" | "blind" | "block"
    """
    if weighted_score <= 0.0:
        return "pass"
    elif weighted_score < _THRESHOLD_BLIND:
        return "warning"
    elif weighted_score < _THRESHOLD_BLOCK:
        return "blind"
    else:
        return "block"


# ============================================================
# 공개 API
# ============================================================

async def check_profanity(inp: ProfanityCheckInput) -> ProfanityCheckOutput:
    """
    텍스트에서 비속어/혐오표현을 검출하고 처리 액션을 결정한다.

    처리 순서:
    1. 정규식 패턴 매칭으로 비속어 검출
    2. 기본 독성 점수 계산 (검출 수 / 단어 수)
    3. content_type 가중치 적용
    4. 가중 점수 기반 action 결정

    LLM 없이 동기 연산으로 처리되므로 응답이 빠르다.
    에러 시 안전 기본값("pass") 반환 (에러 전파 금지).

    Args:
        inp: ProfanityCheckInput (text, user_id, content_type)

    Returns:
        ProfanityCheckOutput (is_toxic, toxicity_score, detected_words, action)
    """
    try:
        text = inp.text or ""
        content_type = inp.content_type or "post"

        # ── Step 1: 비속어 패턴 매칭 ──
        detected_words = _detect_toxic_words(text)

        # ── Step 2: 기본 독성 점수 계산 ──
        # 단어 수는 공백 기준 분리 (한국어 특성상 어절 단위)
        word_count = len(text.split())
        base_score = _compute_toxicity_score(len(detected_words), word_count)

        # ── Step 3: content_type 가중치 적용 ──
        weight = _SENSITIVITY_WEIGHTS.get(content_type, _DEFAULT_WEIGHT)
        weighted_score = min(base_score * weight, 1.0)

        # ── Step 4: 처리 액션 결정 ──
        action = _decide_action(weighted_score)
        is_toxic = action != "pass"

        if is_toxic:
            logger.info(
                "toxicity_detected",
                user_id=inp.user_id,
                content_type=content_type,
                detected_words=detected_words,
                base_score=round(base_score, 4),
                weighted_score=round(weighted_score, 4),
                action=action,
            )
        else:
            logger.debug(
                "toxicity_check_pass",
                user_id=inp.user_id,
                content_type=content_type,
            )

        return ProfanityCheckOutput(
            is_toxic=is_toxic,
            toxicity_score=round(weighted_score, 4),
            detected_words=detected_words,
            action=action,
        )

    except Exception as e:
        logger.error(
            "check_profanity_fatal_error",
            user_id=getattr(inp, "user_id", "unknown"),
            error=str(e),
        )
        # 에러 시 안전 기본값 반환 (차단하지 않음 — 오탐보다 서비스 가용성 우선)
        return ProfanityCheckOutput(
            is_toxic=False,
            toxicity_score=0.0,
            detected_words=[],
            action="pass",
        )
