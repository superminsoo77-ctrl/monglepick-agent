"""
Chat Agent Pydantic 모델 정의.

§6 Chat Agent 그래프의 각 노드가 사용하는 데이터 모델.
Phase 2에서 체인 입출력으로 사용하고, Phase 3에서 LangGraph State에 통합한다.

모델 목록:
- IntentResult: 의도 분류 결과 (6가지 intent + confidence)
- EmotionResult: 감정 분석 결과 (emotion + mood_tags)
- IntentEmotionResult: 의도+감정 통합 LLM 출력 모델 (1회 LLM 호출로 intent+emotion 동시 추출)
- ExtractedPreferences: 사용자 선호 조건 7개 필드
- ImageAnalysisResult: VLM 이미지 분석 결과 (장르/무드/시각요소/키워드/설명/포스터 여부)
- ClarificationHint: 후속 질문 힌트 옵션 (칩/버튼 UI용)
- ClarificationResponse: 후속 질문 + 힌트 구조화 응답
- SearchQuery: RAG 검색 쿼리 구조 (Phase 3 준비)
- CandidateMovie: 검색 결과 후보 영화 (Phase 3 준비)
- ScoreDetail: 추천 점수 분해 (Phase 4 준비)
- RankedMovie: 최종 추천 영화 (Phase 3 준비)
- ChatAgentState: LangGraph TypedDict State (Phase 3 준비)

유틸 함수:
- calculate_sufficiency: 가중치 기반 충분성 점수 계산
- is_sufficient: 추천 진행 가능 여부 판정
- merge_preferences: 이전 선호 + 현재 선호 병합
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from monglepick.config import settings as _settings

# ============================================================
# 의도 분류 결과
# ============================================================

# 6가지 의도 타입 (§6-2 Node 2)
IntentType = Literal["recommend", "search", "info", "theater", "booking", "general"]


class IntentResult(BaseModel):
    """
    의도 분류 LLM 출력 모델.

    intent: 6가지 중 하나 (recommend, search, info, theater, booking, general)
    confidence: 0.0~1.0 신뢰도 (< 0.6이면 general로 보정)
    """

    intent: IntentType = Field(
        default="general",
        description="분류된 사용자 의도 (6가지 중 하나)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="의도 분류 신뢰도 (0.0~1.0)",
    )


# ============================================================
# 감정 분석 결과
# ============================================================

class EmotionResult(BaseModel):
    """
    감정 분석 LLM 출력 모델.

    emotion: 감지된 감정 (happy, sad, excited, angry, calm, None)
    mood_tags: 감정→무드 매핑 결과 (25개 화이트리스트 한정)
    """

    emotion: str | None = Field(
        default=None,
        description="감지된 감정 (happy/sad/excited/angry/calm 또는 None)",
    )
    mood_tags: list[str] = Field(
        default_factory=list,
        description="감정에서 매핑된 무드 태그 목록 (25개 화이트리스트 한정)",
    )


# ============================================================
# 의도+감정 통합 결과 (1회 LLM 호출)
# ============================================================

class IntentEmotionResult(BaseModel):
    """
    의도 분류 + 감정 분석 통합 LLM 출력 모델.

    기존 IntentResult + EmotionResult를 1회 LLM 호출로 동시 추출한다.
    동일 모델(qwen3.5:35b-a3b)로 동일 입력을 분석하므로
    2번 호출할 필요 없이 통합하여 지연 시간을 ~45초 절감한다.
    """

    intent: IntentType = Field(
        default="general",
        description="분류된 사용자 의도 (6가지 중 하나)",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="의도 분류 신뢰도 (0.0~1.0)",
    )
    emotion: str | None = Field(
        default=None,
        description="감지된 감정 (happy/sad/excited/angry/calm 또는 None)",
    )
    mood_tags: list[str] = Field(
        default_factory=list,
        description="감정에서 매핑된 무드 태그 목록 (25개 화이트리스트 한정)",
    )


# ============================================================
# 이미지 분석 결과 (VLM)
# ============================================================

class ImageAnalysisResult(BaseModel):
    """
    VLM(Vision Language Model) 이미지 분석 결과.

    사용자가 업로드한 이미지(영화 포스터, 분위기 사진 등)를 분석하여
    장르/무드/시각요소/키워드/설명을 추출한다.
    이 결과는 선호 추출, 검색 쿼리 구성, 충분성 판정에 활용된다.
    """

    genre_cues: list[str] = Field(
        default_factory=list,
        description="이미지에서 감지된 장르 힌트 (예: ['SF', '모험'])",
    )
    mood_cues: list[str] = Field(
        default_factory=list,
        description="이미지에서 감지된 분위기/무드 (MOOD_WHITELIST 기준)",
    )
    visual_elements: list[str] = Field(
        default_factory=list,
        description="시각적 요소 (예: ['우주선', '폭발', '야경'])",
    )
    search_keywords: list[str] = Field(
        default_factory=list,
        description="검색용 키워드 (RAG 부스트에 활용)",
    )
    description: str = Field(
        default="",
        description="이미지 설명 텍스트 (시맨틱 검색 쿼리에 추가)",
    )
    is_movie_poster: bool = Field(
        default=False,
        description="영화 포스터 여부 (True이면 제목 인식 시도)",
    )
    detected_movie_title: str | None = Field(
        default=None,
        description="포스터에서 인식된 영화 제목 (포스터가 아니면 None)",
    )
    analyzed: bool = Field(
        default=False,
        description="분석 수행 여부 (False면 이미지 없음 또는 분석 실패)",
    )


# ============================================================
# 사용자 선호 조건
# ============================================================

class ExtractedPreferences(BaseModel):
    """
    사용자 선호 조건 7개 필드 (§6-2 Node 4).

    모든 필드는 Optional — 아직 파악되지 않은 선호는 None으로 유지.
    가중치를 기반으로 충분성을 판단하여 추천 진행 여부를 결정한다.

    가중치 테이블:
    - genre_preference: 2.0
    - mood: 2.0
    - viewing_context: 1.0
    - platform: 1.0
    - reference_movies: 1.5
    - era: 0.5
    - exclude: 0.5
    """

    genre_preference: str | None = Field(
        default=None,
        description="선호 장르 (예: 'SF', '액션 코미디')",
    )
    mood: str | None = Field(
        default=None,
        description="원하는 분위기 (예: '따뜻한', '긴장감 있는')",
    )
    viewing_context: str | None = Field(
        default=None,
        description="시청 상황 (예: '혼자', '연인과', '가족과')",
    )
    platform: str | None = Field(
        default=None,
        description="시청 플랫폼 (예: '넷플릭스', '극장')",
    )
    reference_movies: list[str] = Field(
        default_factory=list,
        description="참조 영화 제목 목록 (예: ['인셉션', '인터스텔라'])",
    )
    era: str | None = Field(
        default=None,
        description="선호 시대/연도 (예: '2020년대', '90년대')",
    )
    exclude: str | None = Field(
        default=None,
        description="제외 조건 (예: '공포는 빼주세요', '한국 영화 말고')",
    )


# ============================================================
# 선호 충분성 가중치 테이블 (§6-2 Node 4)
# ============================================================

PREFERENCE_WEIGHTS: dict[str, float] = {
    "genre_preference": 2.0,
    "mood": 2.0,
    "viewing_context": 1.0,
    "platform": 1.0,
    "reference_movies": 1.5,
    "era": 0.5,
    "exclude": 0.5,
}

# 충분성 판정 임계값 — config.py에서 환경변수로 설정 가능 (기본값 2.5)
# 장르+감정(4.0), 참조영화+감정(3.5) 등 2개 이상 정보가 있으면 바로 추천 진행
# 감정만 있으면(2.0 < 2.5) 추가 질문, 턴2 오버라이드(TURN_COUNT_OVERRIDE=2)로 보완
SUFFICIENCY_THRESHOLD: float = _settings.SUFFICIENCY_THRESHOLD
# 턴 카운트 오버라이드 임계값 (2턴째부터는 선호 부족해도 추천 진행)
TURN_COUNT_OVERRIDE: int = _settings.TURN_COUNT_OVERRIDE


# ============================================================
# 후속 질문 힌트 (Part 2: 구조화된 후속 질문)
# ============================================================

class ClarificationHint(BaseModel):
    """
    후속 질문 힌트 옵션 (칩/버튼 UI용).

    프론트엔드에서 사용자가 선택할 수 있는 옵션 칩으로 렌더링된다.
    field: 대응하는 ExtractedPreferences 필드명
    label: 사용자에게 표시할 한국어 레이블
    options: 선택 가능한 옵션 목록
    """

    field: str = Field(..., description="대응하는 선호 필드명 (예: 'genre_preference')")
    label: str = Field(..., description="사용자에게 표시할 한국어 레이블 (예: '장르')")
    options: list[str] = Field(default_factory=list, description="선택 가능한 옵션 목록")


class ClarificationResponse(BaseModel):
    """
    후속 질문 + 힌트 구조화 응답.

    question_generator 노드가 생성하여 SSE clarification 이벤트로 발행한다.
    기존 텍스트 질문에 구조화된 힌트를 추가하여 UX를 향상한다.
    """

    question: str = Field(default="", description="후속 질문 텍스트")
    hints: list[ClarificationHint] = Field(
        default_factory=list,
        description="부족 필드 상위 3개의 힌트 목록",
    )
    primary_field: str = Field(
        default="",
        description="가장 중요한 부족 필드명",
    )


# 필드별 힌트 옵션 상수 (UI 칩 렌더링용)
FIELD_HINTS: dict[str, dict[str, Any]] = {
    "genre_preference": {
        "label": "장르",
        "options": [
            "액션", "SF", "로맨스", "코미디", "드라마", "공포", "스릴러",
            "애니메이션", "판타지", "모험", "범죄", "미스터리", "다큐멘터리",
            "가족", "전쟁", "역사", "음악",
        ],
    },
    "mood": {
        "label": "분위기",
        "options": [
            "힐링", "감동", "유쾌", "웅장", "긴장감", "따뜻", "몰입",
            "스릴", "잔잔", "로맨틱", "철학적", "카타르시스", "공포", "다크",
        ],
    },
    "viewing_context": {
        "label": "시청 상황",
        "options": ["혼자", "연인과", "가족과", "친구와"],
    },
    "platform": {
        "label": "플랫폼",
        "options": [
            "넷플릭스", "왓챠", "디즈니+", "티빙", "웨이브",
            "쿠팡플레이", "애플TV+", "극장",
        ],
    },
    "era": {
        "label": "시대",
        "options": ["최신 (2020년대)", "2010년대", "2000년대", "90년대 이전", "상관없음"],
    },
    "exclude": {
        "label": "제외",
        "options": ["공포/호러 빼주세요", "한국 영화 제외", "19금 제외", "없음"],
    },
    "reference_movies": {
        "label": "참조 영화",
        "options": [],  # 자유 입력
    },
}


# ============================================================
# RAG 검색 품질 판정 임계값 (Part 3)
# ============================================================

# 최소 후보 수: 이 값 미만이면 검색 품질 미달 — config.py에서 환경변수로 설정 가능
RETRIEVAL_MIN_CANDIDATES: int = _settings.RETRIEVAL_MIN_CANDIDATES
# Top-1 RRF 점수 최소값: RRF(k=60)에서 단일 엔진 1위 = 1/61 ≈ 0.01639
# 0.015로 설정하면 최소 1개 엔진에서 Top-2 이내에 포함되어야 통과
RETRIEVAL_MIN_TOP_SCORE: float = _settings.RETRIEVAL_MIN_TOP_SCORE
# 상위 5개 평균 RRF 점수 최소값
RETRIEVAL_QUALITY_MIN_AVG: float = _settings.RETRIEVAL_QUALITY_MIN_AVG


# ============================================================
# RAG 검색 쿼리 (Phase 3 준비)
# ============================================================

class SearchQuery(BaseModel):
    """
    RAG 검색용 구조화된 쿼리 (§6-2 Node 6).

    query_builder 노드가 선호 조건을 기반으로 생성한다.
    rag_retriever 노드가 이 쿼리로 Qdrant + ES + Neo4j 하이브리드 검색을 수행한다.
    """

    semantic_query: str = Field(
        default="",
        description="벡터 검색용 자연어 쿼리 (Qdrant 임베딩 검색)",
    )
    keyword_query: str = Field(
        default="",
        description="BM25 키워드 검색용 쿼리 (Elasticsearch)",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="필터 조건 (장르, 연도, 플랫폼 등)",
    )
    boost_keywords: list[str] = Field(
        default_factory=list,
        description="가산점 부여 키워드 목록",
    )
    exclude_ids: list[str] = Field(
        default_factory=list,
        description="제외할 영화 ID 목록 (이미 추천/시청한 영화)",
    )
    limit: int = Field(
        default=15,
        description="검색 결과 최대 수 (10~15편)",
    )


# ============================================================
# 검색 결과 후보 영화 (Phase 3 준비)
# ============================================================

class CandidateMovie(BaseModel):
    """
    하이브리드 검색 결과 후보 영화 (§6-2 Node 7 출력).

    RRF 합산된 점수와 함께 영화 메타데이터를 포함한다.
    recommendation_ranker 노드의 입력으로 전달된다.
    """

    id: str = Field(..., description="영화 ID (TMDB/KOBIS/KMDb)")
    title: str = Field(default="", description="영화 제목")
    title_en: str = Field(default="", description="영문 제목")
    genres: list[str] = Field(default_factory=list, description="장르 목록")
    director: str = Field(default="", description="감독명")
    cast: list[str] = Field(default_factory=list, description="출연 배우 목록")
    rating: float = Field(default=0.0, description="TMDB 평점")
    release_year: int = Field(default=0, description="개봉 연도")
    overview: str = Field(default="", description="줄거리")
    mood_tags: list[str] = Field(default_factory=list, description="무드 태그")
    poster_path: str = Field(default="", description="포스터 경로")
    ott_platforms: list[str] = Field(default_factory=list, description="OTT 플랫폼 목록")
    certification: str = Field(default="", description="관람등급")
    trailer_url: str = Field(default="", description="트레일러 URL")
    rrf_score: float = Field(default=0.0, description="RRF 합산 점수")
    retrieval_source: str = Field(
        default="hybrid",
        description="검색 소스 (qdrant/es/neo4j/hybrid)",
    )


# ============================================================
# 추천 점수 분해 (Phase 4 준비)
# ============================================================

class ScoreDetail(BaseModel):
    """
    추천 점수 분해 (§7-2 Node 6 출력).

    추천 엔진 서브그래프가 생성하는 CF/CBF/하이브리드 점수 상세 정보.
    explanation_generator에서 추천 이유 생성 시 참조한다.
    """

    cf_score: float = Field(default=0.0, description="협업 필터링(CF) 점수")
    cbf_score: float = Field(default=0.0, description="컨텐츠 기반 필터링(CBF) 점수")
    hybrid_score: float = Field(default=0.0, description="하이브리드 합산 점수")
    genre_match: float = Field(default=0.0, description="장르 일치도 (0.0~1.0)")
    mood_match: float = Field(default=0.0, description="무드 일치도 (0.0~1.0)")
    similar_to: list[str] = Field(
        default_factory=list,
        description="유사 영화 제목 목록 (추천 이유 보조)",
    )


# ============================================================
# 최종 추천 영화 (Phase 3 준비)
# ============================================================

class RankedMovie(BaseModel):
    """
    최종 추천 영화 (§6-2 Node 8 출력).

    recommendation_ranker가 CandidateMovie를 점수별로 정렬하고
    ScoreDetail을 첨부하여 생성한다. explanation_generator의 입력.
    """

    id: str = Field(..., description="영화 ID")
    title: str = Field(default="", description="영화 제목")
    title_en: str = Field(default="", description="영문 제목")
    genres: list[str] = Field(default_factory=list, description="장르 목록")
    director: str = Field(default="", description="감독명")
    cast: list[str] = Field(default_factory=list, description="출연 배우 목록")
    rating: float = Field(default=0.0, description="TMDB 평점")
    release_year: int = Field(default=0, description="개봉 연도")
    overview: str = Field(default="", description="줄거리")
    mood_tags: list[str] = Field(default_factory=list, description="무드 태그")
    poster_path: str = Field(default="", description="포스터 경로")
    ott_platforms: list[str] = Field(default_factory=list, description="OTT 플랫폼 목록")
    certification: str = Field(default="", description="관람등급")
    trailer_url: str = Field(default="", description="트레일러 URL")
    rank: int = Field(default=0, description="추천 순위 (1부터)")
    score_detail: ScoreDetail = Field(
        default_factory=ScoreDetail,
        description="추천 점수 분해 상세 정보",
    )
    explanation: str = Field(default="", description="추천 이유 텍스트 (explanation_generator 생성)")


# ============================================================
# LangGraph State (Phase 3 준비)
# ============================================================

class ChatAgentState(TypedDict, total=False):
    """
    Chat Agent LangGraph TypedDict State (§6-1).

    11개 노드가 이 State를 읽고 쓰며 그래프를 진행한다.
    TypedDict를 사용하는 이유: LangGraph StateGraph 호환 (Pydantic X).
    total=False: 모든 키가 Optional (초기 State에 일부만 존재).
    """

    # ── 입력 ──
    user_id: str
    session_id: str
    current_input: str
    image_data: str | None  # base64 인코딩된 이미지 데이터 (None이면 이미지 없음)

    # ── context_loader 출력 ──
    user_profile: dict[str, Any]
    watch_history: list[dict[str, Any]]
    messages: list[dict[str, str]]

    # ── image_analyzer 출력 ──
    image_analysis: ImageAnalysisResult

    # ── intent_classifier 출력 ──
    intent: IntentResult

    # ── emotion_analyzer 출력 ──
    emotion: EmotionResult

    # ── preference_refiner 출력 ──
    preferences: ExtractedPreferences
    needs_clarification: bool
    turn_count: int

    # ── question_generator 출력 ──
    follow_up_question: str

    # ── query_builder 출력 ──
    search_query: SearchQuery

    # ── rag_retriever 출력 ──
    candidate_movies: list[CandidateMovie]

    # ── recommendation_ranker 출력 ──
    ranked_movies: list[RankedMovie]

    # ── response_formatter 출력 ──
    response: str

    # ── question_generator 출력 (Part 2: 구조화된 힌트) ──
    clarification: ClarificationResponse | None  # 후속 질문 힌트 (SSE clarification 이벤트)

    # ── RAG 검색 품질 판정 (Part 3) ──
    retrieval_quality_passed: bool  # 검색 품질 통과 여부
    retrieval_feedback: str  # 품질 미달 시 피드백 메시지

    # ── error_handler ──
    error: str | None


# ============================================================
# 유틸 함수: 선호 충분성 판정 + 병합
# ============================================================

def calculate_sufficiency(
    prefs: ExtractedPreferences,
    has_emotion: bool = False,
    has_image_analysis: bool = False,
) -> float:
    """
    선호 조건의 가중치 합산 점수를 계산한다 (§6-2 Node 4).

    채워진 필드의 가중치를 합산하여 충분성 점수를 반환.
    has_emotion이 True이면 mood 가중치(2.0)를 추가한다
    (감정 분석 결과가 있으면 무드가 암시적으로 파악된 것으로 간주).
    has_image_analysis가 True이면 +1.5 보너스를 추가한다
    (이미지 분석으로 장르/무드/키워드가 보강된 것으로 간주).

    Args:
        prefs: 현재까지 파악된 사용자 선호 조건
        has_emotion: 감정 분석 결과 존재 여부
        has_image_analysis: 이미지 분석 수행 여부

    Returns:
        가중치 합산 점수 (float)
    """
    score = 0.0
    # 각 필드가 채워져 있으면 해당 가중치를 합산
    if prefs.genre_preference:
        score += PREFERENCE_WEIGHTS["genre_preference"]
    if prefs.mood:
        score += PREFERENCE_WEIGHTS["mood"]
    elif has_emotion:
        # 감정이 감지되면 mood가 없어도 무드 가중치 부여
        score += PREFERENCE_WEIGHTS["mood"]
    if prefs.viewing_context:
        score += PREFERENCE_WEIGHTS["viewing_context"]
    if prefs.platform:
        score += PREFERENCE_WEIGHTS["platform"]
    if prefs.reference_movies:
        score += PREFERENCE_WEIGHTS["reference_movies"]
    if prefs.era:
        score += PREFERENCE_WEIGHTS["era"]
    if prefs.exclude:
        score += PREFERENCE_WEIGHTS["exclude"]
    # 이미지 분석 보너스: 이미지에서 장르/무드/키워드가 추출되면 +1.5
    if has_image_analysis:
        score += 1.5
    return score


def is_sufficient(
    prefs: ExtractedPreferences,
    turn_count: int = 0,
    has_emotion: bool = False,
    has_image_analysis: bool = False,
) -> bool:
    """
    추천 진행 가능 여부를 판정한다 (§6-2 Node 4).

    판정 기준 (OR 조건):
    1. 가중치 합산 >= 2.5 (SUFFICIENCY_THRESHOLD) — 장르+감정 등 2개 이상 정보로 추천 가능
    2. turn_count >= 2 (TURN_COUNT_OVERRIDE) — 2턴째부터 추천 진행

    Args:
        prefs: 현재까지 파악된 사용자 선호 조건
        turn_count: 현재 대화 턴 수
        has_emotion: 감정 분석 결과 존재 여부
        has_image_analysis: 이미지 분석 수행 여부 (True이면 +1.5 보너스)

    Returns:
        True면 추천 진행, False면 후속 질문 필요
    """
    # 턴 카운트 오버라이드: 3턴 이상이면 선호 부족해도 추천 진행
    if turn_count >= TURN_COUNT_OVERRIDE:
        return True
    # 가중치 합산 기반 판정
    return calculate_sufficiency(prefs, has_emotion, has_image_analysis) >= SUFFICIENCY_THRESHOLD


def merge_preferences(
    prev: ExtractedPreferences | None,
    curr: ExtractedPreferences,
) -> ExtractedPreferences:
    """
    이전 선호 조건과 현재 추출된 선호 조건을 병합한다.

    병합 규칙:
    - 새 값이 None이 아니면 덮어쓰기
    - 새 값이 None이면 이전 값 유지
    - reference_movies: 합집합 (중복 제거)

    Args:
        prev: 이전 턴까지 누적된 선호 조건 (None이면 빈 조건)
        curr: 현재 턴에서 추출된 선호 조건

    Returns:
        병합된 ExtractedPreferences
    """
    if prev is None:
        return curr

    return ExtractedPreferences(
        genre_preference=curr.genre_preference if curr.genre_preference is not None else prev.genre_preference,
        mood=curr.mood if curr.mood is not None else prev.mood,
        viewing_context=curr.viewing_context if curr.viewing_context is not None else prev.viewing_context,
        platform=curr.platform if curr.platform is not None else prev.platform,
        # reference_movies: 합집합 (이전 + 현재, 중복 제거, 순서 유지)
        reference_movies=list(dict.fromkeys(prev.reference_movies + curr.reference_movies)),
        era=curr.era if curr.era is not None else prev.era,
        exclude=curr.exclude if curr.exclude is not None else prev.exclude,
    )
