"""
KMDb 데이터 → MovieDocument 변환 및 기존 데이터 보강.

KMDb API에서 수집한 한국영화 데이터를 두 가지 방식으로 활용한다:

1. **기존 영화 보강 (Enrichment)**:
   KMDb의 제목+연도로 Qdrant에서 기존 TMDB/Kaggle 영화를 매칭하여,
   수상내역(awards), 누적관객수(audiAcc), 촬영장소(fLocation),
   한국어 줄거리(plots), 스틸컷(stlls) 등 KMDb 고유 데이터를 추가한다.

2. **신규 영화 추가 (New)**:
   TMDB/Kaggle에 없는 한국영화를 새로운 MovieDocument로 생성하여 적재한다.

매칭 전략:
- 한국어 제목 + 제작연도 정확 매칭 (1차)
- 영문 제목 + 제작연도 정확 매칭 (2차)
- 제목 정규화: 공백/특수문자 제거 후 비교
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import structlog

from monglepick.data_pipeline.models import KMDbRawMovie, MovieDocument
from monglepick.data_pipeline.preprocessor import (
    GENRE_EN_TO_KR,
    build_embedding_text,
    get_fallback_mood_tags,
    validate_movie,
)

logger = structlog.get_logger()

# ============================================================
# KMDb 장르 → 한국어 매핑
# ============================================================

# KMDb 장르는 이미 한국어인 경우가 많으나, 일부 영문 장르도 존재
KMDB_GENRE_MAP: dict[str, str] = {
    "드라마": "드라마",
    "코미디": "코미디",
    "액션": "액션",
    "스릴러": "스릴러",
    "공포": "공포",
    "범죄": "범죄",
    "SF": "SF",
    "판타지": "판타지",
    "로맨스": "로맨스",
    "멜로": "로맨스",
    "멜로/로맨스": "로맨스",
    "모험": "모험",
    "애니메이션": "애니메이션",
    "다큐멘터리": "다큐멘터리",
    "가족": "가족",
    "역사": "역사",
    "전쟁": "전쟁",
    "음악": "음악",
    "뮤지컬": "음악",
    "미스터리": "미스터리",
    "서부": "서부",
    "공연실황": "다큐멘터리",
    "사극": "역사",
    "무협": "액션",
    "성인물(에로)": "드라마",
    "실험": "드라마",
    # 영문 fallback
    **GENRE_EN_TO_KR,
}

# KMDb 관람등급 → 표준 한국어 관람등급 매핑
KMDB_RATING_MAP: dict[str, str] = {
    "전체관람가": "전체 관람가",
    "전체 관람가": "전체 관람가",
    "ALL": "전체 관람가",
    "12세관람가": "12세 이상 관람가",
    "12세 관람가": "12세 이상 관람가",
    "12세이상관람가": "12세 이상 관람가",
    "12세 이상 관람가": "12세 이상 관람가",
    "15세관람가": "15세 이상 관람가",
    "15세 관람가": "15세 이상 관람가",
    "15세이상관람가": "15세 이상 관람가",
    "15세 이상 관람가": "15세 이상 관람가",
    "18세관람가": "청소년 관람불가",
    "18세 관람가": "청소년 관람불가",
    "청소년관람불가": "청소년 관람불가",
    "청소년 관람불가": "청소년 관람불가",
    "제한상영가": "제한상영가",
    "제한 상영가": "제한상영가",
    "미성년자관람불가": "청소년 관람불가",
}


def _normalize_title(title: str) -> str:
    """
    제목 정규화: 매칭 정확도를 높이기 위해 공백/특수문자/반각전각 차이를 제거한다.

    1. NFKC 유니코드 정규화 (전각→반각, 호환문자 통일)
    2. 소문자 변환
    3. 공백/특수문자 제거 (한글, 영문, 숫자만 남김)
    """
    # NFKC 정규화 (전각→반각 등)
    normalized = unicodedata.normalize("NFKC", title)
    # 소문자 변환
    normalized = normalized.lower()
    # 한글, 영문, 숫자만 남기고 제거
    normalized = re.sub(r"[^가-힣a-z0-9]", "", normalized)
    return normalized


def _parse_year(year_str: str) -> int:
    """연도 문자열을 정수로 변환한다. 실패 시 0 반환."""
    if not year_str:
        return 0
    try:
        # "2019" 또는 "20190530" 형태 모두 처리
        return int(year_str[:4])
    except (ValueError, IndexError):
        return 0


def _extract_plot_korean(plots: list[dict]) -> str:
    """
    KMDb plots 배열에서 한국어 줄거리를 추출한다.

    우선순위: 한국어 > 영어 > 첫 번째 줄거리
    """
    korean_plot = ""
    english_plot = ""
    first_plot = ""

    for plot in plots:
        lang = plot.get("plotLang", "")
        text = plot.get("plotText", "").strip()
        if not text:
            continue

        if not first_plot:
            first_plot = text

        if lang == "한국어":
            korean_plot = text
        elif lang in ("영어", "English"):
            english_plot = text

    return korean_plot or english_plot or first_plot


def _convert_kmdb_genres(genre_str: str) -> list[str]:
    """
    KMDb 장르 문자열(쉼표 구분)을 한국어 장르 리스트로 변환한다.

    예: "드라마,코미디,스릴러" → ["드라마", "코미디", "스릴러"]
    """
    if not genre_str:
        return []

    genres: list[str] = []
    for g in genre_str.split(","):
        g = g.strip()
        if not g:
            continue
        mapped = KMDB_GENRE_MAP.get(g, g)
        if mapped and mapped not in genres:
            genres.append(mapped)

    return genres


def _convert_kmdb_certification(rating_str: str) -> str:
    """KMDb 관람등급을 표준 한국어 관람등급으로 변환한다."""
    if not rating_str:
        return ""
    return KMDB_RATING_MAP.get(rating_str.strip(), rating_str.strip())


def _extract_trailer_url(vods: list[dict]) -> str:
    """KMDb VOD 목록에서 예고편 URL을 추출한다."""
    for vod in vods:
        vod_class = vod.get("vodClass", "")
        vod_url = vod.get("vodUrl", "").strip()
        if vod_class in ("예고편", "메인예고편", "티저") and vod_url:
            return vod_url
    return ""


# ============================================================
# 매칭 인덱스 구축
# ============================================================


def build_title_index(existing_docs: list[dict]) -> dict[str, list[dict]]:
    """
    기존 영화 데이터로부터 제목+연도 기반 매칭 인덱스를 구축한다.

    Qdrant에서 가져온 payload 목록을 정규화된 제목+연도 키로 인덱싱한다.
    동일 키에 여러 영화가 매핑될 수 있으므로 리스트로 저장한다.

    Args:
        existing_docs: Qdrant payload 딕셔너리 목록 (id, title, title_en, release_year 포함)

    Returns:
        {"정규화제목_연도": [{"id": ..., "title": ..., ...}, ...]} 형태의 인덱스
    """
    index: dict[str, list[dict]] = {}

    for doc in existing_docs:
        title = doc.get("title", "")
        title_en = doc.get("title_en", "")
        year = doc.get("release_year", 0)

        # 한국어 제목 + 연도 키
        if title:
            key = f"{_normalize_title(title)}_{year}"
            index.setdefault(key, []).append(doc)

        # 영문 제목 + 연도 키 (한국어 제목과 다른 경우만)
        if title_en and title_en != title:
            key_en = f"{_normalize_title(title_en)}_{year}"
            index.setdefault(key_en, []).append(doc)

    logger.info("title_index_built", total_keys=len(index), total_docs=len(existing_docs))
    return index


def match_kmdb_to_existing(
    kmdb_movie: KMDbRawMovie,
    title_index: dict[str, list[dict]],
) -> dict | None:
    """
    KMDb 영화를 기존 영화 인덱스에서 매칭한다.

    매칭 전략 (우선순위):
    1. 한국어 제목 + 연도 정확 매칭
    2. 영문 제목 + 연도 정확 매칭
    3. 원제 + 연도 정확 매칭

    Args:
        kmdb_movie: KMDb 영화 데이터
        title_index: build_title_index()로 구축한 인덱스

    Returns:
        매칭된 기존 영화 payload 딕셔너리, 없으면 None
    """
    year = _parse_year(kmdb_movie.prod_year)
    if not year:
        year = _parse_year(kmdb_movie.release_date)

    # 매칭 시도할 제목 목록
    titles_to_try = [kmdb_movie.title, kmdb_movie.title_eng, kmdb_movie.title_org]

    for title in titles_to_try:
        if not title:
            continue
        key = f"{_normalize_title(title)}_{year}"
        matches = title_index.get(key)
        if matches:
            return matches[0]  # 첫 번째 매칭 반환

    return None


# ============================================================
# KMDbRawMovie → MovieDocument 변환
# ============================================================


def kmdb_to_movie_document(raw: KMDbRawMovie) -> MovieDocument | None:
    """
    KMDb 영화 데이터를 새로운 MovieDocument로 변환한다.

    기존에 없는 한국영화를 새로 추가할 때 사용한다.
    TMDB ID가 없으므로 KMDb의 movieId+movieSeq를 조합하여 ID를 생성한다.

    Args:
        raw: KMDb API에서 파싱된 영화 데이터

    Returns:
        MovieDocument 또는 None (유효성 검증 실패 시)
    """
    if not raw.title:
        return None

    # ID 생성: KMDb movieId + movieSeq (예: "K_12345")
    kmdb_id = f"{raw.movie_id}_{raw.movie_seq}"
    # TMDB ID(1~7자리)와 범위가 겹치지 않도록 10,000,000 오프셋을 더한다.
    # movieSeq가 숫자가 아닌 경우(드물지만 존재) 문자열 연결로 fallback.
    try:
        doc_id = str(int(raw.movie_seq) + 10_000_000)
    except ValueError:
        doc_id = kmdb_id.replace("_", "")

    # 연도 파싱
    year = _parse_year(raw.prod_year)
    if not year:
        year = _parse_year(raw.release_date)

    # 장르 변환
    genres = _convert_kmdb_genres(raw.genre)

    # 감독 추출 (첫 번째 감독)
    director = ""
    if raw.directors:
        director = raw.directors[0].get("directorNm", "").strip()

    # 배우 추출 (상위 5명)
    cast: list[str] = []
    for actor in raw.actors[:5]:
        name = actor.get("actorNm", "").strip()
        if name:
            cast.append(name)

    # 줄거리
    overview = _extract_plot_korean(raw.plots)

    # 키워드 파싱 (쉼표 구분)
    keywords_list: list[str] = []
    if raw.keywords:
        keywords_list = [k.strip() for k in raw.keywords.split(",") if k.strip()]

    # 런타임
    runtime = 0
    if raw.runtime:
        try:
            runtime = int(raw.runtime)
        except ValueError:
            pass

    # 관람등급
    certification = _convert_kmdb_certification(raw.rating)

    # 트레일러 URL
    trailer_url = _extract_trailer_url(raw.vods)

    # 관객수
    audience_count = 0
    if raw.audi_acc:
        try:
            audience_count = int(raw.audi_acc)
        except ValueError:
            pass

    # 수상내역 합산
    awards = ""
    if raw.awards1:
        awards = raw.awards1
    if raw.awards2:
        awards = f"{awards} {raw.awards2}".strip()

    # 무드태그 (장르 기반 fallback)
    mood_tags = get_fallback_mood_tags(genres)

    # 줄거리가 없거나 너무 짧으면 장르/키워드로 대체 텍스트 생성
    # (임베딩 품질 유지를 위해 최소한의 맥락 정보 제공)
    if not overview or len(overview) < 10:
        overview = f"{', '.join(genres)} 장르의 한국영화."
        if keywords_list:
            overview += f" 키워드: {', '.join(keywords_list[:5])}"

    # 포스터 경로 (첫 번째 포스터 URL)
    poster_path = raw.posters[0] if raw.posters else ""

    doc = MovieDocument(
        id=doc_id,
        title=raw.title,
        title_en=raw.title_eng or raw.title_org or "",
        overview=overview,
        release_year=year,
        runtime=runtime,
        rating=0.0,  # KMDb는 TMDB 평점 없음
        popularity_score=0.0,
        poster_path=poster_path,
        genres=genres,
        keywords=keywords_list,
        director=director,
        cast=cast,
        ott_platforms=[],  # KMDb에는 OTT 정보 없음
        mood_tags=mood_tags,
        certification=certification,
        trailer_url=trailer_url,
        similar_movie_ids=[],
        # KMDb 고유 필드
        kmdb_id=kmdb_id,
        awards=awards,
        audience_count=audience_count,
        filming_location=raw.f_location,
        stills=raw.stills[:5],  # 스틸컷 최대 5장
        theme_song=raw.theme_song,
        soundtrack=raw.soundtrack_field,
        source="kmdb",
    )

    # 임베딩 텍스트 구성
    doc.embedding_text = build_embedding_text(doc)

    # 유효성 검증
    if not validate_movie(doc):
        return None

    return doc


# ============================================================
# 기존 영화 보강 데이터 추출
# ============================================================


def extract_enrichment_data(raw: KMDbRawMovie) -> dict[str, Any]:
    """
    KMDb 영화에서 기존 MovieDocument를 보강할 데이터를 추출한다.

    기존 TMDB/Kaggle 영화에 덮어쓰지 않고 **추가**할 데이터만 반환한다.
    이미 값이 있는 필드는 건너뛴다 (KMDb가 우선하지 않음).

    Args:
        raw: KMDb API에서 파싱된 영화 데이터

    Returns:
        보강 데이터 딕셔너리:
        - kmdb_id: KMDb 영화 ID
        - awards: 수상내역
        - audience_count: 누적관객수
        - filming_location: 촬영장소
        - stills: 스틸컷 URL 목록
        - theme_song: 주제곡
        - soundtrack: 삽입곡
        - plot_korean: 한국어 줄거리 (overview 보강용)
        - certification_kmdb: KMDb 관람등급 (certification 없을 때 보강용)
        - trailer_url_kmdb: KMDb 예고편 URL (trailer_url 없을 때 보강용)
    """
    kmdb_id = f"{raw.movie_id}_{raw.movie_seq}"

    # 수상내역 합산
    awards = ""
    if raw.awards1:
        awards = raw.awards1
    if raw.awards2:
        awards = f"{awards} {raw.awards2}".strip()

    # 관객수
    audience_count = 0
    if raw.audi_acc:
        try:
            audience_count = int(raw.audi_acc)
        except ValueError:
            pass

    return {
        "kmdb_id": kmdb_id,
        "awards": awards,
        "audience_count": audience_count,
        "filming_location": raw.f_location,
        "stills": raw.stills[:5],
        "theme_song": raw.theme_song,
        "soundtrack": raw.soundtrack_field,
        "plot_korean": _extract_plot_korean(raw.plots),
        "certification_kmdb": _convert_kmdb_certification(raw.rating),
        "trailer_url_kmdb": _extract_trailer_url(raw.vods),
    }


# ============================================================
# 배치 처리
# ============================================================


def process_kmdb_batch(
    kmdb_movies: list[KMDbRawMovie],
    title_index: dict[str, list[dict]],
) -> tuple[list[dict], list[MovieDocument]]:
    """
    KMDb 영화 배치를 처리하여 보강 데이터와 신규 MovieDocument를 분류한다.

    Args:
        kmdb_movies: KMDb에서 수집한 영화 목록
        title_index: 기존 영화 매칭 인덱스

    Returns:
        (enrichments, new_documents) 튜플
        - enrichments: 기존 영화 보강 데이터 리스트
          [{"existing_id": "157336", "data": {...}}, ...]
        - new_documents: 신규 MovieDocument 리스트
    """
    enrichments: list[dict] = []
    new_documents: list[MovieDocument] = []
    matched_count = 0
    new_count = 0
    failed_count = 0

    for raw in kmdb_movies:
        try:
            # 제목+연도로 기존 DB 영화와 매칭 시도
            match = match_kmdb_to_existing(raw, title_index)

            if match:
                # 매칭 성공 → KMDb 고유 데이터(수상내역/관객수/촬영장소 등)를 추출
                existing_id = match.get("id") or match.get("_id", "")
                enrichment_data = extract_enrichment_data(raw)
                enrichments.append({
                    "existing_id": str(existing_id),
                    "data": enrichment_data,
                })
                matched_count += 1
            else:
                # 매칭 실패 → TMDB/Kaggle에 없는 한국영화이므로 신규 생성
                doc = kmdb_to_movie_document(raw)
                if doc:
                    new_documents.append(doc)
                    new_count += 1
                else:
                    failed_count += 1
        except Exception as e:
            failed_count += 1
            # 에러 로그 폭주 방지: 처음 10건만 상세 로깅
            if failed_count <= 10:
                logger.warning(
                    "kmdb_process_failed",
                    title=raw.title,
                    error=str(e),
                )

    logger.info(
        "kmdb_batch_processed",
        total=len(kmdb_movies),
        matched=matched_count,
        new=new_count,
        failed=failed_count,
    )

    return enrichments, new_documents
