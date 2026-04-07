"""
사용자 시청 패턴 분석 + 업적 판정 모듈 (§8-2 기능3).

처리 흐름 (규칙 기반, LLM 없음):
1. 시청 이력에서 총 편수, 장르 분포, 평균 평점 계산
2. 사전 정의된 업적 판정 규칙으로 달성 여부 확인
3. existing_achievements에 없는 업적만 new_achievements로 반환
4. 장르 분포 기반 40차원 사용자 패턴 벡터 생성

에러 시 빈 결과를 반환하고 에러를 전파하지 않는다.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import structlog

from monglepick.agents.content_analysis.models import (
    Achievement,
    PatternAnalysisInput,
    PatternAnalysisOutput,
    WatchRecord,
)

logger = structlog.get_logger()

# ============================================================
# 업적 정의 상수 (§8-2, §14 기준)
# ============================================================

# (업적ID, 이름, 설명, 아이콘)
_ALL_ACHIEVEMENTS: list[tuple[str, str, str, str]] = [
    ("ACH_001", "영화 입문",    "영화를 10편 이상 시청했습니다.",          "🎬"),
    ("ACH_002", "영화 팬",      "영화를 50편 이상 시청했습니다.",          "🍿"),
    ("ACH_003", "영화 마니아",  "영화를 100편 이상 시청했습니다.",         "🎭"),
    ("ACH_004", "장르 전문가",  "동일 장르 영화를 20편 이상 시청했습니다.", "🏆"),
    ("ACH_005", "까다로운 심판","평균 평점 4.0 이상을 유지하고 있습니다.", "⭐"),
    ("ACH_006", "장르 탐험가",  "5개 이상 다른 장르를 시청했습니다.",      "🌍"),
]

# 패턴 벡터에 사용할 40개 장르 슬롯 (인덱스 고정)
# 정의되지 않은 장르는 마지막 슬롯(index 39: 기타)에 합산
_GENRE_SLOTS: list[str] = [
    "액션", "드라마", "코미디", "로맨스", "공포",
    "스릴러", "SF", "판타지", "애니메이션", "다큐멘터리",
    "범죄", "미스터리", "어드벤처", "가족", "뮤지컬",
    "전쟁", "서부", "역사", "스포츠", "공연",
    "음악", "요리", "여행", "자연", "과학",
    "철학", "심리", "종교", "예술", "건축",
    "패션", "게임", "뉴스", "단편", "실험",
    "에로틱", "성인", "교육", "청소년", "기타",
]
assert len(_GENRE_SLOTS) == 40, "패턴 벡터 차원은 반드시 40이어야 한다."

_GENRE_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(_GENRE_SLOTS)}


# ============================================================
# 내부 유틸 함수
# ============================================================

def _compute_stats(watch_history: list[WatchRecord]) -> dict[str, Any]:
    """
    시청 이력에서 핵심 통계를 계산한다.

    반환 dict 키:
    - total_watched   : 총 시청 편수 (int)
    - genre_counter   : {장르: 시청 횟수} Counter
    - unique_genres   : 고유 장르 수 (int)
    - top_genre       : 가장 많이 시청한 장르 (str)
    - top_genre_count : top_genre 시청 횟수 (int)
    - avg_rating      : 평점을 매긴 영화들의 평균 (float, 없으면 0.0)

    Args:
        watch_history: WatchRecord 목록

    Returns:
        통계 dict
    """
    total_watched = len(watch_history)

    # 장르 분포
    genre_counter: Counter[str] = Counter()
    for rec in watch_history:
        for genre in rec.genres:
            genre_counter[genre.strip()] += 1

    top_genre, top_genre_count = ("", 0)
    if genre_counter:
        top_genre, top_genre_count = genre_counter.most_common(1)[0]

    unique_genres = len(genre_counter)

    # 평균 평점 (평점 있는 기록만)
    ratings = [rec.rating for rec in watch_history if rec.rating is not None]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

    return {
        "total_watched": total_watched,
        "genre_counter": genre_counter,
        "unique_genres": unique_genres,
        "top_genre": top_genre,
        "top_genre_count": top_genre_count,
        "avg_rating": avg_rating,
    }


def _evaluate_achievements(
    stats: dict[str, Any],
    existing: set[str],
) -> list[Achievement]:
    """
    통계 기반 업적 달성 여부를 판정하고 신규 업적 목록을 반환한다.

    판정 규칙:
    - ACH_001: 총 시청 >= 10
    - ACH_002: 총 시청 >= 50
    - ACH_003: 총 시청 >= 100
    - ACH_004: 단일 장르 최다 시청 >= 20
    - ACH_005: 평균 평점 >= 4.0 (평점 기록 있는 경우만)
    - ACH_006: 고유 장르 수 >= 5

    Args:
        stats   : _compute_stats() 반환값
        existing: 이미 보유한 업적 ID 집합

    Returns:
        새로 달성한 Achievement 목록
    """
    new_achievements: list[Achievement] = []

    total = stats["total_watched"]
    top_genre_count = stats["top_genre_count"]
    avg_rating = stats["avg_rating"]
    unique_genres = stats["unique_genres"]

    # 업적 판정 조건 — (업적ID, 달성 여부 bool)
    conditions: list[tuple[str, bool]] = [
        ("ACH_001", total >= 10),
        ("ACH_002", total >= 50),
        ("ACH_003", total >= 100),
        ("ACH_004", top_genre_count >= 20),
        ("ACH_005", avg_rating >= 4.0),
        ("ACH_006", unique_genres >= 5),
    ]

    # 업적 메타데이터 빠른 조회용 dict
    ach_meta: dict[str, tuple[str, str, str, str]] = {
        aid: (aid, name, desc, icon)
        for aid, name, desc, icon in _ALL_ACHIEVEMENTS
    }

    for ach_id, achieved in conditions:
        if achieved and ach_id not in existing:
            meta = ach_meta.get(ach_id)
            if meta:
                new_achievements.append(
                    Achievement(
                        id=meta[0],
                        name=meta[1],
                        description=meta[2],
                        icon=meta[3],
                    )
                )

    return new_achievements


def _build_pattern_vector(genre_counter: Counter[str]) -> list[float]:
    """
    장르 분포 Counter를 40차원 정규화 벡터로 변환한다.

    - 각 인덱스 = _GENRE_SLOTS 순서에 대응하는 장르 시청 비율
    - 정의되지 않은 장르는 index 39 (기타) 슬롯에 합산
    - 총 시청 횟수가 0이면 모두 0.0인 벡터 반환

    Args:
        genre_counter: {장르: 시청 횟수} Counter

    Returns:
        40차원 float 리스트 (합계 = 1.0 또는 0.0)
    """
    vector = [0.0] * 40
    total = sum(genre_counter.values())

    if total == 0:
        return vector

    for genre, count in genre_counter.items():
        idx = _GENRE_TO_IDX.get(genre, 39)  # 미정의 장르 → 기타(39)
        vector[idx] += count / total  # 비율로 정규화

    # 부동소수점 반올림 (6자리)
    vector = [round(v, 6) for v in vector]
    return vector


# ============================================================
# 공개 API
# ============================================================

async def analyze_user_pattern(inp: PatternAnalysisInput) -> PatternAnalysisOutput:
    """
    사용자 시청 이력을 분석하여 업적 및 패턴 벡터를 반환한다.

    처리 순서:
    1. 시청 이력 통계 계산 (총 편수, 장르 분포, 평균 평점)
    2. 업적 판정 (규칙 기반, LLM 없음)
    3. 장르 분포 기반 40차원 패턴 벡터 생성

    모든 처리는 동기 연산으로 수행 (DB/LLM 호출 없음).
    에러 시 빈 결과 반환 (에러 전파 금지).

    Args:
        inp: PatternAnalysisInput (user_id, watch_history, existing_achievements)

    Returns:
        PatternAnalysisOutput (new_achievements, user_pattern_vector)
    """
    try:
        logger.info(
            "pattern_analysis_start",
            user_id=inp.user_id,
            watch_count=len(inp.watch_history),
            existing_achievement_count=len(inp.existing_achievements),
        )

        # ── Step 1: 통계 계산 ──
        stats = _compute_stats(inp.watch_history)

        logger.info(
            "pattern_stats_computed",
            user_id=inp.user_id,
            total_watched=stats["total_watched"],
            unique_genres=stats["unique_genres"],
            top_genre=stats["top_genre"],
            avg_rating=stats["avg_rating"],
        )

        # ── Step 2: 업적 판정 ──
        existing_set = set(inp.existing_achievements)
        new_achievements = _evaluate_achievements(stats, existing_set)

        if new_achievements:
            logger.info(
                "achievements_unlocked",
                user_id=inp.user_id,
                new_achievements=[a.id for a in new_achievements],
            )

        # ── Step 3: 패턴 벡터 생성 ──
        user_pattern_vector = _build_pattern_vector(stats["genre_counter"])

        logger.info(
            "pattern_analysis_complete",
            user_id=inp.user_id,
            new_achievement_count=len(new_achievements),
            vector_nonzero=sum(1 for v in user_pattern_vector if v > 0.0),
        )

        return PatternAnalysisOutput(
            new_achievements=new_achievements,
            user_pattern_vector=user_pattern_vector,
        )

    except Exception as e:
        logger.error(
            "analyze_user_pattern_fatal_error",
            user_id=getattr(inp, "user_id", "unknown"),
            error=str(e),
        )
        # 에러 전파 금지 → 빈 결과 반환
        return PatternAnalysisOutput(
            new_achievements=[],
            user_pattern_vector=[0.0] * 40,
        )
