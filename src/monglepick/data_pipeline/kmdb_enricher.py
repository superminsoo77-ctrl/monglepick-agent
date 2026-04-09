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
    _KEYWORD_EN_TO_KR_LOWER,  # Phase ML-2: 영문→한국어 키워드 매핑 (200개)
    build_embedding_text,
    get_fallback_mood_tags,
    validate_movie,
)

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# Phase ML-2 일관성 헬퍼 (KMDb 한영 이중 + 키워드 매핑)
# ══════════════════════════════════════════════════════════════


def _extract_director_bilingual_kmdb(directors: list[dict]) -> tuple[str, str]:
    """
    KMDb directors 배열에서 감독의 한글 + 영문 이름을 튜플로 반환한다.

    KMDb API는 `directors[].directorNm` (한글) + `directorEnNm` (영문)을 제공한다.
    한영이 같으면 영문은 빈 문자열 반환 (TMDB extract_director_names와 동일 정책).

    Args:
        directors: KMDb API의 directors 배열
            예: [{"directorNm": "봉준호", "directorEnNm": "BONG Joon-ho", "directorId": "..."}]

    Returns:
        (director_kr, director_original_name) 튜플.
    """
    if not directors:
        return ("", "")
    name_kr = (directors[0].get("directorNm", "") or "").strip()
    name_en = (directors[0].get("directorEnNm", "") or "").strip()
    if not name_kr:
        return ("", "")
    if name_en == name_kr:
        return (name_kr, "")
    return (name_kr, name_en)


def _extract_cast_bilingual_kmdb(actors: list[dict], top_n: int = 5) -> list[str]:
    """
    KMDb actors 배열에서 배우 상위 N명을 한영 이중 리스트로 반환한다.

    KMDb API는 `actors[].actorNm` (한글) + `actorEnNm` (영문)을 제공한다.
    중복 제거된 리스트를 반환하여 한국어/영문 검색 양방향 매칭을 지원한다.

    Args:
        actors: KMDb API의 actors 배열
            예: [{"actorNm": "송강호", "actorEnNm": "SONG Kang-ho"}]
        top_n: 상위 N명 (기본 5)

    Returns:
        한영 이중 리스트 (최대 top_n*2개)
    """
    result: list[str] = []
    seen: set[str] = set()

    for actor in actors[:top_n]:
        name_kr = (actor.get("actorNm", "") or "").strip()
        name_en = (actor.get("actorEnNm", "") or "").strip()

        if name_kr and name_kr not in seen:
            result.append(name_kr)
            seen.add(name_kr)
        if name_en and name_en != name_kr and name_en not in seen:
            result.append(name_en)
            seen.add(name_en)

    return result


def _apply_korean_mapping_to_keywords_kmdb(keywords: list[str]) -> list[str]:
    """
    KMDb 키워드에 Phase ML-2 한국어 매핑을 적용한다.

    KMDb는 대부분 한국어 키워드를 제공하지만 일부 영문 키워드도 섞일 수 있다.
    `preprocessor._KEYWORD_EN_TO_KR_LOWER` (200개 매핑 사전)으로 영문 키워드만
    한국어로 변환하고, 한국어 키워드는 그대로 유지한다.

    TMDB collector의 extract_keywords()와 동일 정책.

    Args:
        keywords: KMDb 키워드 리스트 (한글/영문 혼재 가능)

    Returns:
        한국어 우선 + 영문 보존 리스트 (중복 제거)
    """
    result: list[str] = []
    seen: set[str] = set()

    for kw in keywords:
        if not kw or not isinstance(kw, str):
            continue
        kw_stripped = kw.strip()
        if not kw_stripped:
            continue
        # 한국어 매핑 시도 (영문 키워드만 매칭됨)
        kr_name = _KEYWORD_EN_TO_KR_LOWER.get(kw_stripped.lower())
        if kr_name and kr_name not in seen:
            result.append(kr_name)
            seen.add(kr_name)
        # 원본도 추가 (한국어 키워드는 여기서 들어감)
        if kw_stripped not in seen:
            result.append(kw_stripped)
            seen.add(kw_stripped)

    return result

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

    # 감독 추출 — Phase ML-2 일관성: 한영 이중 (directorNm + directorEnNm)
    director, director_original_name = _extract_director_bilingual_kmdb(raw.directors)

    # 배우 추출 — Phase ML-2 일관성: 한영 이중 (actorNm + actorEnNm)
    cast = _extract_cast_bilingual_kmdb(raw.actors, top_n=5)

    # 줄거리
    overview = _extract_plot_korean(raw.plots)

    # 키워드 파싱 (쉼표 구분) + Phase ML-2 한국어 매핑
    keywords_list: list[str] = []
    if raw.keywords:
        raw_keywords = [k.strip() for k in raw.keywords.split(",") if k.strip()]
        # KMDb 키워드는 대부분 한글이지만 일부 영문 매칭으로 한국어 보강
        keywords_list = _apply_korean_mapping_to_keywords_kmdb(raw_keywords)

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
        director_original_name=director_original_name,  # Phase ML-2: 한영 이중
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


def build_kmdb_full_enrichment_payload(raw: KMDbRawMovie) -> dict[str, Any]:
    """
    KMDb 영화의 **모든 풍부 필드** 를 기존 영화 enrichment payload 로 변환.

    extract_enrichment_data() 의 확장판. 사용자의 "모든 데이터 컬럼 수집"
    원칙에 따라 KMDb 가 제공하는 47 필드 중 Qdrant/ES/Neo4j/MySQL 에
    저장 가능한 모든 항목을 포함한다.

    TMDB 이미 있는 필드 우선순위:
        - 덮어쓰지 않음: title, overview, poster_path, runtime (기본)
        - 보강 (기존 비어있을 때만): director_original_name, cast_original_names,
          certification, trailer_url
        - KMDb 고유 (항상 추가): awards, filming_location, soundtrack, theme_song,
          keywords(한국어), kmdb_id, audience_count

    Args:
        raw: KMDbRawMovie (KMDb API 응답 파싱 결과)

    Returns:
        Qdrant set_payload / ES update / Neo4j SET 에 바로 쓸 수 있는 dict.
        loader 가 자체 직렬화로 알맞게 처리한다.
    """
    out: dict[str, Any] = {}

    # ── 1. KMDb 식별자 ──
    kmdb_id = f"{raw.movie_id}_{raw.movie_seq}"
    if kmdb_id and kmdb_id != "_":
        out["kmdb_id"] = kmdb_id

    # ── 1-B. Phase ML-1 한영 이중 핵심: 영문 제목 ──
    # KMDb titleEng 는 종종 "Parasite (Gi-saeng-chung)" 형태.
    # 괄호 앞부분만 추출하여 깔끔한 영문 제목으로 보강.
    title_eng_raw = (raw.title_eng or "").strip()
    if title_eng_raw:
        # 괄호 제거 (예: "Parasite (Gi-saeng-chung)" → "Parasite")
        paren_idx = title_eng_raw.find("(")
        title_en = (title_eng_raw[:paren_idx] if paren_idx > 0 else title_eng_raw).strip()
        if title_en:
            out["title_en"] = title_en

    # 원제 (titleOrg — 외국 영화의 원제)
    title_org = (raw.title_org or "").strip()
    if title_org:
        out["title_original"] = title_org

    # ── 2. 수상내역 (awards1 + awards2 병합) ──
    awards_parts = []
    if raw.awards1:
        awards_parts.append(raw.awards1)
    if raw.awards2:
        awards_parts.append(raw.awards2)
    if awards_parts:
        out["awards"] = " ".join(awards_parts)

    # ── 3. 누적 관객수 ──
    if raw.audi_acc:
        try:
            out["audience_count"] = int(raw.audi_acc)
        except ValueError:
            pass

    # ── 4. 촬영장소 ──
    if raw.f_location:
        out["filming_location"] = raw.f_location

    # ── 5. OST / 주제곡 / 삽입곡 ──
    if raw.theme_song:
        out["theme_song"] = raw.theme_song
    if raw.soundtrack_field:
        out["soundtrack"] = raw.soundtrack_field

    # ── 6. 한국어 plot (기존 overview 비어있을 때만 보강) ──
    plot_ko = _extract_plot_korean(raw.plots)
    if plot_ko:
        out["plot_korean"] = plot_ko  # 호출 측에서 overview 가 비었을 때만 적용

    # 영어 plot
    for p in (raw.plots or []):
        lang = (p.get("plotLang", "") or "").strip().lower()
        text = (p.get("plotText", "") or "").strip()
        if text and ("eng" in lang or "영어" in lang):
            out["overview_en_kmdb"] = text  # 호출 측 결정
            break

    # ── 7. Phase ML-1 한영 이중 감독/배우 (기존 비어있을 때만 보강) ──
    directors = raw.directors or []
    if directors:
        d0 = directors[0]
        d_en = (d0.get("directorEnNm", "") or "").strip()
        if d_en:
            out["director_original_name"] = d_en

    actors = raw.actors or []
    if actors:
        # Phase ML-1 핵심: 주요 배우 영문명 추출 (기생충 등 한국 영화 한영 이중 보강).
        # TMDB preprocessor 가 한국 영화 cast 를 한국어로만 저장한 버그 수정용.
        # (예: 기생충 → ['Song Kang-ho', 'Lee Sun-kyun', 'Cho Yeo-jeong', ...])
        cast_original = [
            (a.get("actorEnNm", "") or "").strip()
            for a in actors[:10]
            if (a.get("actorEnNm", "") or "").strip()
        ]
        if cast_original:
            out["cast_original_names"] = cast_original

    # ── 8. KMDb 한국어 keywords (기존 keyword 리스트에 병합) ──
    if raw.keywords:
        kw_list = [k.strip() for k in raw.keywords.split(",") if k.strip()]
        if kw_list:
            out["kmdb_keywords"] = kw_list

    # ── 9. KMDb 전체 스태프 (JSON 보존) ──
    if raw.staffs:
        out["kmdb_staffs"] = [
            {
                "peopleNm": (s.get("staffNm", "") or s.get("peopleNm", "") or "").strip(),
                "peopleNmEn": (s.get("staffEnNm", "") or s.get("peopleNmEn", "") or "").strip(),
                "staffRoleNm": (s.get("staffRoleNm", "") or s.get("staffRoleGroup", "") or "").strip(),
            }
            for s in raw.staffs
        ]

    # ── 10. KMDb 관람등급 / 예고편 (기존 비어있을 때만) ──
    cert_kmdb = _convert_kmdb_certification(raw.rating)
    if cert_kmdb:
        out["certification_kmdb"] = cert_kmdb
    trailer_kmdb = _extract_trailer_url(raw.vods)
    if trailer_kmdb:
        out["trailer_url_kmdb"] = trailer_kmdb

    # ── 11. KMDb 포스터 / 스틸컷 ──
    if raw.posters:
        out["kmdb_posters"] = raw.posters[:10]
    if raw.stills:
        out["stills"] = raw.stills[:10]

    return out


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
                # 매칭 성공 → KMDb 풀필드 데이터 추출 (2026-04-09 확장)
                # 기존 extract_enrichment_data 는 10개 필드만 추출했으나
                # build_kmdb_full_enrichment_payload 는 47 필드 전부 (awards,
                # filming_location, soundtrack, theme_song, 한국어 keywords,
                # 한영이중 director/actors, 전체 staffs, 포스터/스틸 등) 추출.
                existing_id = match.get("id") or match.get("_id", "")
                enrichment_data = build_kmdb_full_enrichment_payload(raw)
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
