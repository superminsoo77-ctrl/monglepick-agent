"""
Chat Agent Pydantic 모델 정의.

§6 Chat Agent 그래프의 각 노드가 사용하는 데이터 모델.
Phase 2에서 체인 입출력으로 사용하고, Phase 3에서 LangGraph State에 통합한다.

모델 목록:
- IntentResult: 의도 분류 결과 (6가지 intent + confidence)
- EmotionResult: 감정 분석 결과 (emotion + mood_tags)
- IntentEmotionResult: 의도+감정 통합 LLM 출력 모델 (1회 LLM 호출로 intent+emotion 동시 추출)
- FilterCondition: 동적 필터 조건 (LLM이 사용자 요청에서 자유롭게 추출)
- ExtractedPreferences: 사용자 선호 조건 — 핵심 구조화 필드 + 유연한 동적 필터
- ImageAnalysisResult: VLM 이미지 분석 결과 (장르/무드/시각요소/키워드/설명/포스터 여부)
- ClarificationHint: 후속 질문 힌트 옵션 (칩/버튼 UI용)
- ClarificationResponse: 후속 질문 + 힌트 구조화 응답
- SearchQuery: RAG 검색 쿼리 구조 (Phase 3 준비)
- CandidateMovie: 검색 결과 후보 영화 (Phase 3 준비)
- ScoreDetail: 추천 점수 분해 (Phase 4 준비)
- RankedMovie: 최종 추천 영화 (Phase 3 준비)
- ChatAgentState: LangGraph TypedDict State (Phase 3 준비)

유틸 함수:
- calculate_sufficiency: Intent-First 기반 충분성 점수 계산
- is_sufficient: 추천 진행 가능 여부 판정 (의도/동적필터/핵심필드 기반)
- merge_preferences: 이전 선호 + 현재 선호 병합 (동적 필터 포함)
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from monglepick.config import settings as _settings

# ============================================================
# 외부 지도 연동 — 사용자 위치 (Phase 6 외부 지도 연동)
# ============================================================

class Location(BaseModel):
    """
    사용자 위치 정보 (theater 의도 처리 시 카카오 Local 검색에 사용).

    Client 가 navigator.geolocation 으로 좌표를 직접 보낼 수도 있고,
    사용자 메시지에 지명만 있으면 geocoding 도구가 좌표를 채워 넣는다.
    address 는 둘 다 있을 때 응답 자연어에 활용된다.
    """

    latitude: float = Field(..., description="위도 (예: 37.5665)")
    longitude: float = Field(..., description="경도 (예: 126.9780)")
    address: str | None = Field(
        default=None,
        description="원본 주소/지역명 (선택, 사용자 응답 자연어 생성에 활용)",
    )


# ============================================================
# 의도 분류 결과
# ============================================================

# 7가지 의도 타입 (§6-2 Node 2) — Phase M-Relation: "relation" 추가
# - relation: 인물/관계 기반 탐색 ("봉준호 감독이 찍은 스릴러에 나온 배우들의 영화" 등)
#   → graph_traversal_node로 라우팅 (Neo4j 멀티홉 탐색)
IntentType = Literal["recommend", "search", "info", "theater", "booking", "general", "relation"]


class IntentResult(BaseModel):
    """
    의도 분류 LLM 출력 모델.

    intent: 7가지 중 하나 (recommend, search, info, theater, booking, general, relation)
    confidence: 0.0~1.0 신뢰도 (< 0.6이면 general로 보정)
    """

    intent: IntentType = Field(
        default="general",
        description="분류된 사용자 의도 (7가지 중 하나)",
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

    intent: 7가지 중 하나 (recommend, search, info, theater, booking, general, relation)
    - relation: "봉준호 감독이 찍은 스릴러에 나온 배우들의 영화" 등 인물/관계 기반 탐색
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
# 동적 필터 조건 (Intent-First 아키텍처)
# ============================================================

class FilterCondition(BaseModel):
    """
    LLM이 사용자 요청에서 자유롭게 추출하는 동적 필터 조건.

    기존 7개 고정 필드로는 "평점 높은", "트레일러 있는", "2시간 이내" 등
    예측하지 못한 사용자 요구를 처리할 수 없었다.
    FilterCondition을 통해 DB에 존재하는 모든 필드에 대해 필터링이 가능하다.

    지원 필드 (DB payload 기준):
    - rating: 평점 (float, 0~10)
    - release_year: 개봉 연도 (int)
    - runtime: 상영시간 분 (int)
    - director: 감독명 (str)
    - certification: 관람등급 (str, 예: "15세", "전체")
    - trailer_url: 트레일러 URL (str, exists 연산으로 유무 확인)
    - popularity_score: 인기도 점수 (float)
    - vote_count: 투표 수 (int)

    지원 연산자:
    - gte: 이상 (>=)
    - lte: 이하 (<=)
    - eq: 일치 (==)
    - exists: 존재 여부 (값이 비어있지 않은지)
    - contains: 포함 (문자열/리스트 내 포함 여부)
    - not_eq: 불일치 (!=)
    """

    field: str = Field(
        ...,
        description="DB 필드명 (rating, release_year, runtime, director, trailer_url 등)",
    )
    operator: str = Field(
        ...,
        description="비교 연산자 (gte, lte, eq, exists, contains, not_eq)",
    )
    value: Any = Field(
        default=None,
        description="필터 값 (exists 연산자일 때는 true/false)",
    )


# 동적 필터로 지원하는 DB 필드 목록 — 프롬프트와 쿼리 빌더에서 참조
FILTERABLE_FIELDS: dict[str, dict[str, str]] = {
    "rating": {"type": "float", "description": "TMDB 평점 (0~10)"},
    "release_year": {"type": "int", "description": "개봉 연도"},
    "runtime": {"type": "int", "description": "상영시간 (분)"},
    "director": {"type": "str", "description": "감독명"},
    "certification": {"type": "str", "description": "관람등급 (전체, 12세, 15세, 청소년관람불가)"},
    "trailer_url": {"type": "str", "description": "트레일러/예고편 URL (exists로 유무 확인)"},
    "popularity_score": {"type": "float", "description": "인기도 점수"},
    "vote_count": {"type": "int", "description": "투표/평가 수"},
    # ── 국가/언어 필터 (한국영화, 일본 애니 등 국가 기반 추천 지원) ──
    "origin_country": {"type": "list[str]", "description": "창작 원산국 ISO 3166-1 코드 (예: KR, US, JP)"},
    "original_language": {"type": "str", "description": "원본 언어 ISO 639-1 소문자 코드 (예: ko, en, ja) — 반드시 소문자"},
    "production_countries": {"type": "list[str]", "description": "제작 국가 ISO 3166-1 코드 (예: KR, US)"},
}


# ============================================================
# 사용자 선호 조건 (Intent-First + Dynamic Filter)
# ============================================================

class ExtractedPreferences(BaseModel):
    """
    사용자 선호 조건 — 핵심 구조화 필드 + 유연한 동적 필터 (§6-2 Node 4).

    Intent-First 아키텍처:
    - user_intent: LLM이 요약한 사용자 의도 (시맨틱 검색의 핵심 입력)
    - dynamic_filters: 사용자 요청에서 추출한 정량적/불린 조건 (DB 필터링)
    - search_keywords: 검색 부스트용 키워드

    기존 구조화 필드는 검색 엔진이 잘 지원하는 핵심 4개만 유지:
    - genre_preference, mood, reference_movies, exclude

    충분성 판정:
    - user_intent가 있으면 → 충분 (사용자가 원하는 것을 알고 있음)
    - dynamic_filters가 있으면 → 충분 (구체적 필터 조건 존재)
    - 핵심 필드 중 하나라도 있으면 → 충분
    - 이전 방식의 가중치 합산은 fallback으로만 사용
    """

    # ── 핵심 구조화 필드 (검색 엔진이 잘 지원하는 것만 유지) ──
    genre_preference: str | None = Field(
        default=None,
        description="선호 장르 (예: 'SF', '액션 코미디')",
    )
    mood: str | None = Field(
        default=None,
        description="원하는 분위기 (예: '따뜻한', '긴장감 있는')",
    )
    reference_movies: list[str] = Field(
        default_factory=list,
        description="참조 영화 제목 목록 (예: ['인셉션', '인터스텔라'])",
    )
    exclude: str | None = Field(
        default=None,
        description="제외 조건 (예: '공포는 빼주세요', '한국 영화 말고')",
    )

    # ── 하위 호환용 (기존 코드에서 참조하는 필드, 점진적 제거 예정) ──
    viewing_context: str | None = Field(
        default=None,
        description="시청 상황 (예: '혼자', '연인과', '가족과')",
    )
    platform: str | None = Field(
        default=None,
        description="시청 플랫폼 (예: '넷플릭스', '극장')",
    )
    era: str | None = Field(
        default=None,
        description="선호 시대/연도 (예: '2020년대', '90년대')",
    )

    # ── 새로운 유연한 필드들 (Intent-First) ──
    user_intent: str = Field(
        default="",
        description="LLM이 요약한 사용자의 추천 의도 (시맨틱 검색 쿼리로 사용)",
    )
    dynamic_filters: list[FilterCondition] = Field(
        default_factory=list,
        description="사용자 요청에서 추출한 동적 필터 조건 (평점, 트레일러, 런타임 등)",
    )
    search_keywords: list[str] = Field(
        default_factory=list,
        description="검색 부스트용 키워드 (LLM이 추출한 핵심 검색어)",
    )

    # ── 사용자가 명시적으로 요청한 추천 편수 (2026-04-24) ──
    # "한 편만", "1편만", "딱 하나", "세 편" 등 구체적 숫자 표현이 있을 때만 채운다.
    # None = 사용자가 편수를 언급하지 않음 → 기본 TOP_K(5) 사용.
    # 유효 범위: 1~TOP_K(5). 범위 밖 값은 recommendation 엔진에서 clamp.
    requested_count: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description="사용자가 명시적으로 요청한 추천 편수 (1~5). 미지정 시 기본 TOP_K 사용",
    )


# ============================================================
# 선호 충분성 판정 (Intent-First 아키텍처)
# ============================================================

# 기존 가중치 테이블 (fallback 용도로 유지)
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


class SuggestedOption(BaseModel):
    """
    Claude Code 스타일 제안 옵션 카드 (2026-04-15 신설).

    question_generator 가 LLM 으로 생성하는 "한 번에 고를 수 있는" 자연어 옵션.
    ClarificationHint(필드별 칩 묶음)와 달리 이 구조는 문장 단위의 자연스러운
    제안이며, 사용자가 카드를 클릭하면 `value` 가 그대로 채팅 입력으로 전송된다.

    - text: 사용자에게 표시할 짧은 라벨 (예: "잔잔한 힐링 일본 영화")
    - value: 클릭 시 채팅 입력창에 삽입될 실제 문자열
    - reason: 왜 이 옵션인지 짧은 부제 (선택)
    - tags: 부족 필드명 등 디버깅/분석용 태그
    """

    text: str = Field(..., description="옵션 카드 타이틀 (짧은 라벨)")
    value: str = Field(..., description="클릭 시 입력으로 삽입될 문자열")
    reason: str = Field(default="", description="옵션 선택 근거 (선택적 부제)")
    tags: list[str] = Field(default_factory=list, description="부족 필드명 등 분석용 태그")


class ClarificationResponse(BaseModel):
    """
    후속 질문 + 힌트 + 제안 옵션 구조화 응답.

    question_generator 노드가 생성하여 SSE clarification 이벤트로 발행한다.
    - hints: 필드별 선택 칩 (기존 UX)
    - suggestions: Claude Code 스타일 문장 단위 제안 카드 (2026-04-15 추가)
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
    suggestions: list[SuggestedOption] = Field(
        default_factory=list,
        description="자연어 제안 카드 2~4개 (LLM 생성, Claude Code 스타일)",
    )
    allow_custom: bool = Field(
        default=True,
        description="사용자가 제안 외 자유 입력을 할 수 있는지 여부 (UI 힌트)",
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
# Soft-ambiguous 임계값 — 후보는 있지만 점수가 애매한 구간(top_score < 이 값)에서
# turn_count < TURN_COUNT_OVERRIDE 이면 similar_fallback_search 대신 question_generator 로 보낸다.
# "애매하면 재질문" UX 정책 (2026-04-15 추가).
RETRIEVAL_SOFT_AMBIGUOUS_TOP_SCORE: float = _settings.RETRIEVAL_SOFT_AMBIGUOUS_TOP_SCORE


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
    # ── 추가 메타데이터 필드 (설계서 기준 확장, 필터링·UI 렌더링 용도) ──
    runtime: int | None = Field(
        default=None,
        description="상영시간 (분) — max_runtime 후처리 필터 및 프론트엔드 표시용",
    )
    popularity_score: float | None = Field(
        default=None,
        description="TMDB 인기도 점수 — min_popularity 필터 및 랭킹 보조용",
    )
    vote_count: int | None = Field(
        default=None,
        description="평점 참여 수 — min_vote_count 필터 및 신뢰도 판단용",
    )
    backdrop_path: str | None = Field(
        default=None,
        description="배경 이미지 경로 — 프론트엔드 배너/상세 화면 표시용",
    )
    # ── 국가/언어 필드 (한국영화 필터링 및 재랭킹 시 국가 판별용) ──
    original_language: str = Field(
        default="",
        description="원본 언어 ISO 639-1 코드 (예: ko, en, ja)",
    )
    origin_country: list[str] = Field(
        default_factory=list,
        description="창작 원산국 ISO 3166-1 코드 목록 (예: ['KR'])",
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
    # ── 추가 메타데이터 필드 (CandidateMovie와 동기화, SSE movie_card 이벤트로 프론트엔드 전달) ──
    runtime: int | None = Field(
        default=None,
        description="상영시간 (분) — 프론트엔드 영화 카드에서 '2시간 3분' 등으로 표시",
    )
    popularity_score: float | None = Field(
        default=None,
        description="TMDB 인기도 점수 — 추천 점수 보조 참고값으로 프론트엔드 전달",
    )
    vote_count: int | None = Field(
        default=None,
        description="평점 참여 수 — 평점 신뢰도 표시용 (예: '평점 8.2 (1.2만 명 참여)')",
    )
    backdrop_path: str | None = Field(
        default=None,
        description="배경 이미지 경로 — 영화 상세/추천 카드 배너 이미지 표시용",
    )
    # 2026-04-15 신규: Backend `POST /api/v1/recommendations/internal/batch` 호출 후 채워지는
    # recommendation_log PK. SSE `movie_card` payload 로 Client 까지 실어 보내야
    # 마이픽 피드백(관심없음/좋아요) 버튼이 FK 로 사용할 수 있다.
    # 저장 실패/movieId FK 부재 시 None 유지 (graceful — 피드백 버튼만 비활성, 추천 노출은 유지).
    recommendation_log_id: int | None = Field(
        default=None,
        description="recommendation_log PK — 저장 후 채워짐 (Client 피드백 FK)",
    )


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
    # 사용자 위치 (외부 지도 연동) — Client 가 navigator.geolocation 으로 좌표를 보내거나
    # 사용자가 지명만 입력했을 때 tool_executor_node 가 geocoding 으로 채워 넣는다.
    # theater/booking 의도 외에는 미사용.
    location: Location | None

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

    # ── tool_executor_node 출력 (Phase 6 외부 지도 연동) ──
    # info/theater/booking 의도에서 INTENT_TOOL_MAP 의 도구들이 비동기 병렬 실행된 결과.
    # 키: 도구 이름 (예: "theater_search", "kobis_now_showing", "movie_detail")
    # 값: 도구 반환값 (list[dict] | dict | str)
    # response_formatter 가 이 값을 SSE theater_card / movie_card / token 으로 분기 송출.
    tool_results: dict[str, Any]

    # ── question_generator 출력 (Part 2: 구조화된 힌트 + 제안 옵션) ──
    # clarification.suggestions 에 Claude Code 스타일 제안 카드가 포함된다 (2026-04-15).
    clarification: ClarificationResponse | None  # 후속 질문 힌트 (SSE clarification 이벤트)

    # ── RAG 검색 품질 판정 (Part 3) ──
    retrieval_quality_passed: bool  # 검색 품질 통과 여부
    retrieval_feedback: str  # 품질 미달 시 피드백 메시지

    # ── Phase 3: 암시적 평점 (user_implicit_rating 테이블) ──
    implicit_ratings: dict[str, float]  # {movie_id: implicit_score} — CF fallback에 활용

    # ── Phase 4: 행동 프로필 (user_behavior_profile 테이블) ──
    user_behavior_profile: dict[str, Any]  # taste_consistency, genre_affinity 등

    # ── relation Intent 전용 (graph_traversal_node 출력) ──
    # graph_traversal_node가 LLM으로 추출한 Neo4j 탐색 계획.
    # query_type(chain/intersection/person_filmography), start_entity, hop_genre,
    # persons, relation_type 등 Cypher 생성에 필요한 파라미터를 포함한다.
    graph_query_plan: dict | None
    # Neo4j 멀티홉 탐색의 원시 결과 (metadata dict 목록).
    # explanation_generator 등에서 관계 정보를 추가 활용할 수 있도록 보존한다.
    traversal_results: list[dict]
    # graph_traversal_node 또는 다른 노드가 결과 없음/에러 시 직접 설정하는 응답.
    # response_formatter에서 final_answer가 있으면 LLM 생성 없이 그대로 반환한다.
    final_answer: str | None

    # ── error_handler ──
    error: str | None


# ============================================================
# 유틸 함수: 선호 충분성 판정 + 병합 (Intent-First)
# ============================================================

def calculate_sufficiency(
    prefs: ExtractedPreferences,
    has_emotion: bool = False,
    has_image_analysis: bool = False,
) -> float:
    """
    선호 조건의 충분성 점수를 계산한다 (§6-2 Node 4, Intent-First).

    Intent-First 아키텍처에서는 user_intent와 dynamic_filters를 우선 반영한다:
    - user_intent가 있으면: +3.0 (시맨틱 검색으로 즉시 추천 가능)
    - dynamic_filters가 있으면: 필터당 +1.5 (구체적 조건 존재, 최대 3.0)

    기존 구조화 필드 가중치는 보조 점수로 합산된다.
    has_emotion이 True이면 mood 가중치(2.0)를 추가한다.
    has_image_analysis가 True이면 +1.5 보너스를 추가한다.

    Args:
        prefs: 현재까지 파악된 사용자 선호 조건
        has_emotion: 감정 분석 결과 존재 여부
        has_image_analysis: 이미지 분석 수행 여부

    Returns:
        충분성 점수 (float)
    """
    score = 0.0

    # ── Intent-First 점수: user_intent가 있으면 즉시 충분 ──
    if prefs.user_intent:
        score += 3.0

    # ── 동적 필터: 각 필터 조건당 +1.5 (최대 3.0) ──
    if prefs.dynamic_filters:
        score += min(len(prefs.dynamic_filters) * 1.5, 3.0)

    # ── 기존 구조화 필드 가중치 (보조) ──
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

    # 이미지 분석 보너스
    if has_image_analysis:
        score += 1.5

    # 검색 키워드 보너스 (키워드가 있으면 +1.0)
    if prefs.search_keywords:
        score += 1.0

    return score


def is_sufficient(
    prefs: ExtractedPreferences,
    turn_count: int = 0,
    has_emotion: bool = False,
    has_image_analysis: bool = False,
) -> bool:
    """
    추천 진행 가능 여부를 판정한다 (§6-2 Node 4, Intent-First).

    Intent-First 판정 기준 (OR 조건, 우선순위 순):
    1. user_intent가 비어있지 않으면 → 충분 (사용자 의도 파악됨)
    2. dynamic_filters가 하나라도 있으면 → 충분 (구체적 필터 조건 존재)
    3. 핵심 필드 중 하나라도 있으면 → 충분 (genre/mood/reference_movies)
    4. turn_count >= TURN_COUNT_OVERRIDE → 충분 (턴 오버라이드)
    5. 가중치 합산 >= SUFFICIENCY_THRESHOLD → 충분 (기존 방식 fallback)

    Args:
        prefs: 현재까지 파악된 사용자 선호 조건
        turn_count: 현재 대화 턴 수
        has_emotion: 감정 분석 결과 존재 여부
        has_image_analysis: 이미지 분석 수행 여부

    Returns:
        True면 추천 진행, False면 후속 질문 필요
    """
    # Phase ML-3: user_intent만으로는 즉시 충분 판정하지 않는다.
    # "영화 추천해줘" 같은 모호한 요청에서도 user_intent가 채워지지만,
    # 구체적 선호 정보(장르/무드/참조영화/필터)가 하나도 없으면 재질문이 더 좋은 UX.

    # 구체적 선호 정보가 하나라도 있는지 확인
    has_specific_pref = bool(
        prefs.genre_preference
        or prefs.mood
        or prefs.reference_movies
        or prefs.dynamic_filters
        or prefs.search_keywords
    )

    # 1) user_intent + 구체적 선호 정보 → 충분
    #    "액션 영화 추천해줘" → intent + genre="액션" → 충분
    #    "영화 추천해줘" → intent + (없음) → 부족 → 재질문
    if prefs.user_intent and has_specific_pref:
        return True

    # 2) dynamic_filters만으로도 충분 (구체적 필터 조건 존재)
    if prefs.dynamic_filters:
        return True

    # 3) 핵심 구조화 필드 중 하나라도 있으면 → 충분
    if prefs.genre_preference or prefs.mood or prefs.reference_movies:
        return True

    # 4) 감정 감지 + user_intent → 충분
    #    "우울한데 영화 추천해줘" → emotion=sad + intent → 충분
    if prefs.user_intent and has_emotion:
        return True

    # 5) 턴 카운트 오버라이드 (강제 추천)
    if turn_count >= TURN_COUNT_OVERRIDE:
        return True

    # 6) 기존 가중치 합산 fallback
    return calculate_sufficiency(prefs, has_emotion, has_image_analysis) >= SUFFICIENCY_THRESHOLD


def _merge_comma_field(prev_val: str | None, curr_val: str | None) -> str | None:
    """
    쉼표로 구분된 문자열 필드를 합집합으로 병합한다.

    Phase ML-3: genre_preference, mood 등의 필드를 덮어쓰기 대신 누적한다.
    - 이전: "액션" + 현재: "코미디" → "액션, 코미디"
    - 이전: "액션, SF" + 현재: "액션" → "액션, SF" (중복 제거)
    - 이전: "액션" + 현재: None → "액션" (이전 유지)
    - 이전: None + 현재: "코미디" → "코미디"
    """
    if not prev_val and not curr_val:
        return None
    if not prev_val:
        return curr_val
    if not curr_val:
        return prev_val

    # 쉼표/공백으로 분리 후 합집합 (순서 유지, 중복 제거)
    prev_items = [item.strip() for item in prev_val.split(",") if item.strip()]
    curr_items = [item.strip() for item in curr_val.split(",") if item.strip()]
    merged = list(dict.fromkeys(prev_items + curr_items))
    return ", ".join(merged) if merged else None


def merge_preferences(
    prev: ExtractedPreferences | None,
    curr: ExtractedPreferences,
) -> ExtractedPreferences:
    """
    이전 선호 조건과 현재 추출된 선호 조건을 병합한다.

    병합 규칙:
    - genre_preference: 합집합 (Phase ML-3, "액션" + "코미디" = "액션, 코미디")
    - mood: 합집합 (Phase ML-3, "웅장" + "잔잔" = "웅장, 잔잔")
    - 기타 단일값 필드: 새 값이 있으면 덮어쓰기, 없으면 이전 유지
    - reference_movies: 합집합 (중복 제거)
    - dynamic_filters: 2026-04-15 replace 전환 — curr 가 필터를 하나라도 가지면
      curr 를 그대로 사용, 비어있으면 prev 유지 (아래 상세 참고)
    - search_keywords: 합집합 (중복 제거)
    - user_intent: 현재 턴 것이 있으면 덮어쓰기 (최신 의도 우선)

    Args:
        prev: 이전 턴까지 누적된 선호 조건 (None이면 빈 조건)
        curr: 현재 턴에서 추출된 선호 조건

    Returns:
        병합된 ExtractedPreferences
    """
    if prev is None:
        return curr

    # ── dynamic_filters 병합 (2026-04-15 replace 전환) ──
    # 이전에는 "curr 우선 + 서로 다른 field 의 prev 추가" 누적 방식이었으나,
    # 프롬프트가 "이미 파악된 조건은 생략하세요" 로 지시하기 때문에 prev 의 필터가
    # curr 에서 재확인 없이도 영원히 유지되어 턴이 쌓일수록 검색 범위가 과도하게
    # 좁아지는 버그가 있었다. 예: 턴1 "요즘 인기 SF" → release_year>=2025 +
    # popularity_score>=50 이 영구 고정 → 턴3 에서 무명 인디만 통과하는 상태가 됨.
    #
    # 새 정책:
    #   - curr 가 필터를 하나라도 뽑았다 → 그 필터셋을 "현재 유효한 전체"로 간주해 replace.
    #     LLM 은 existing_prefs 를 프롬프트로 받기 때문에 유효한 이전 필터를 계속
    #     유지하고 싶으면 재추출하도록 프롬프트 지시를 함께 조정한다 (preference.py).
    #   - curr 가 비었다 → prev 그대로 유지 (LLM 이 이번 턴에 필터 미추출).
    # 이렇게 하면 필터는 언제든 "LLM 이 본 최신 스냅샷" 이 되므로 stale 누적이 사라진다.
    if curr.dynamic_filters:
        merged_filters = list(curr.dynamic_filters)
    else:
        merged_filters = list(prev.dynamic_filters)

    # ── search_keywords 병합: 합집합 (중복 제거, 순서 유지) ──
    merged_keywords = list(dict.fromkeys(
        prev.search_keywords + curr.search_keywords
    ))

    # ── Phase ML-3: genre_preference/mood 합집합 누적 ──
    # 기존: 덮어쓰기 → 이전 턴 장르/무드 유실
    # 개선: 합집합 → 턴1 "액션" + 턴2 "코미디도" = "액션, 코미디"
    merged_genre = _merge_comma_field(prev.genre_preference, curr.genre_preference)
    merged_mood = _merge_comma_field(prev.mood, curr.mood)

    # ── requested_count 병합 (2026-04-24) ──
    # 사용자가 매 턴마다 원하는 편수를 바꿀 수 있으므로 curr 가 명시적으로 값을 가지면
    # 덮어쓰고, curr 가 None 이면 prev 를 유지한다. "이 중에서 한 편만" 같은 후속
    # 요청도 반영되도록 replace 정책.
    merged_requested_count = (
        curr.requested_count if curr.requested_count is not None else prev.requested_count
    )

    return ExtractedPreferences(
        genre_preference=merged_genre,
        mood=merged_mood,
        viewing_context=curr.viewing_context if curr.viewing_context is not None else prev.viewing_context,
        platform=curr.platform if curr.platform is not None else prev.platform,
        # reference_movies: 합집합 (이전 + 현재, 중복 제거, 순서 유지)
        reference_movies=list(dict.fromkeys(prev.reference_movies + curr.reference_movies)),
        era=curr.era if curr.era is not None else prev.era,
        exclude=curr.exclude if curr.exclude is not None else prev.exclude,
        # ── Intent-First 필드 ──
        user_intent=curr.user_intent if curr.user_intent else prev.user_intent,
        dynamic_filters=merged_filters,
        search_keywords=merged_keywords,
        requested_count=merged_requested_count,
    )
