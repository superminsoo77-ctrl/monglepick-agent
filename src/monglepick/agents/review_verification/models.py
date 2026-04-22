from __future__ import annotations

from typing import Optional

from typing_extensions import TypedDict


class ReviewVerificationState(TypedDict, total=False):
    """
    리뷰 검증 에이전트 그래프 전체 상태.

    total=False: 모든 키가 선택적 — 노드는 자신이 담당하는 키만 업데이트한다.
    입력 필드는 admin.py ReviewVerificationRequest와 1:1 대응한다.

    review_text / review_id 는 영화 상세 페이지 리뷰가 아닌
    course_review 테이블의 도장깨기 인증 리뷰다.
    """

    # ── 입력 (API에서 초기화, ReviewVerificationRequest와 동일) ──
    verification_id: int        # course_verification PK
    user_id: str
    course_id: str
    movie_id: str
    review_id: Optional[int]    # course_review PK (로깅용, 없어도 무방)
    review_text: str            # course_review.course_review_text
    movie_plot: str             # 비교 기준 영화 줄거리

    # ── preprocessor 출력 ─────────────────────────────────────
    clean_review: str           # HTML/마크다운 제거 + 1500자 truncate
    clean_plot: str
    early_exit: bool            # True: 리뷰 20자 미만 → 이후 노드 pass-through

    # ── embedding_similarity 출력 ─────────────────────────────
    similarity_score: float     # 코사인 유사도 [0.0, 1.0]

    # ── keyword_matcher 출력 ──────────────────────────────────
    matched_keywords: list[str] # 리뷰 ∩ 줄거리 공통 키워드
    keyword_score: float        # min(교집합 수 / 5, 1.0)

    # ── llm_revalidator 출력 ──────────────────────────────────
    confidence_draft: float     # 0.7*sim + 0.3*kw
    llm_adjustment: float       # yes=+0.1, no=-0.2, 스킵=0.0

    # ── threshold_decider 출력 (최종) ─────────────────────────
    confidence: float           # clip(draft + adj, 0.0, 1.0)
    review_status: str          # AUTO_VERIFIED | NEEDS_REVIEW | AUTO_REJECTED
    rationale: str              # 판정 근거 한 줄 요약
