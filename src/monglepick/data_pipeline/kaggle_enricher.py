"""
Kaggle 데이터 → MovieDocument 변환기.

TMDB API로 수집되지 않은 영화를 Kaggle CSV에서 직접 변환하여 보강한다.
Kaggle에는 45,000+건의 영화가 있으나, TMDB API로는 ~3,700건만 수집 가능하므로
나머지 ~44,000건을 Kaggle 데이터로 채운다.

Kaggle CSV 구조:
- movies_metadata.csv: id, title, overview, genres, release_date, vote_average, popularity, poster_path, runtime
- credits.csv: id, cast(JSON), crew(JSON)
- keywords.csv: id, keywords(JSON)

Kaggle에서 제공되지 않는 필드:
- OTT 플랫폼 (watch_providers): 빈 리스트로 설정
- 한국어 제목: original_title 사용 (대부분 영문)
"""

from __future__ import annotations

import pandas as pd
import structlog

from monglepick.data_pipeline.kaggle_loader import KaggleLoader
from monglepick.data_pipeline.models import MovieDocument
from monglepick.data_pipeline.preprocessor import (
    build_embedding_text,
    convert_genres,
    get_fallback_mood_tags,
    validate_movie,
)

logger = structlog.get_logger()


def load_kaggle_movies(
    kaggle_dir: str,
    exclude_ids: set[int] | None = None,
) -> list[MovieDocument]:
    """
    Kaggle CSV에서 MovieDocument 리스트를 생성한다.

    이미 적재된 ID는 제외하고, 유효한 영화만 반환한다.

    Args:
        kaggle_dir: Kaggle 데이터 디렉토리 경로
        exclude_ids: 제외할 TMDB ID 집합 (이미 적재된 영화)

    Returns:
        list[MovieDocument]: 변환된 영화 문서 리스트
    """
    exclude_ids = exclude_ids or set()
    loader = KaggleLoader(kaggle_dir)

    # 1. 메타데이터 로드
    logger.info("kaggle_enricher_loading_metadata")
    metadata_df = loader.load_movies_metadata()

    # 2. 크레딧 로드 (감독/배우)
    logger.info("kaggle_enricher_loading_credits")
    credits_df = loader.load_credits()

    # 3. 키워드 로드
    logger.info("kaggle_enricher_loading_keywords")
    keywords_df = loader.load_keywords()

    # 4. 크레딧, 키워드를 메타데이터에 조인 (TMDB ID 기준)
    merged = metadata_df.merge(credits_df, on="id", how="left")
    merged = merged.merge(keywords_df, on="id", how="left")

    # 이미 적재된 ID 제외
    if exclude_ids:
        before = len(merged)
        merged = merged[~merged["id"].isin(exclude_ids)]
        excluded = before - len(merged)
        logger.info("kaggle_enricher_excluded_existing", excluded=excluded, remaining=len(merged))

    # 5. MovieDocument로 변환
    logger.info("kaggle_enricher_converting", count=len(merged))
    documents: list[MovieDocument] = []
    failed = 0

    for _, row in merged.iterrows():
        try:
            doc = _row_to_movie_document(row)
            # 변환 성공 + 유효성 검증 통과 시에만 결과에 포함
            if doc and validate_movie(doc):
                documents.append(doc)
            else:
                failed += 1
        except Exception as e:
            failed += 1
            # 에러 로그 폭증 방지: 처음 10건만 상세 로그 출력
            if failed <= 10:
                logger.warning("kaggle_enricher_convert_failed", id=row.get("id"), error=str(e))

    logger.info(
        "kaggle_enricher_complete",
        total=len(merged),
        success=len(documents),
        failed=failed,
    )

    return documents


def _row_to_movie_document(row: pd.Series) -> MovieDocument | None:
    """
    Kaggle 메타데이터 DataFrame 행을 MovieDocument로 변환한다.

    Args:
        row: merged DataFrame의 한 행
             (metadata + credits + keywords 조인 결과)

    Returns:
        MovieDocument 또는 None (변환 불가 시)
    """
    tmdb_id = int(row["id"])

    # 제목 (한국어 제목이 없으면 원제 사용)
    title = row.get("title", "") or row.get("original_title", "")
    title_en = row.get("original_title", "") or title
    if not title:
        return None

    # 개봉 연도 추출
    release_date = str(row.get("release_date", ""))
    release_year = 0
    if release_date and len(release_date) >= 4:
        try:
            release_year = int(release_date[:4])
        except ValueError:
            pass

    # 장르 변환 (Kaggle의 genres_parsed는 [{"id": 28, "name": "Action"}, ...] 형태)
    genres_parsed = row.get("genres_parsed", [])
    if not isinstance(genres_parsed, list):
        genres_parsed = []
    genres = convert_genres(genres_parsed)

    # 감독/배우 (credits_df에서 조인된 컬럼)
    director = row.get("director", "") or ""
    cast_names = row.get("cast_names", [])
    if not isinstance(cast_names, list):
        cast_names = []

    # 키워드 (keywords_df에서 조인된 컬럼)
    keywords_list = row.get("keywords_list", [])
    if not isinstance(keywords_list, list):
        keywords_list = []

    # 기본 필드
    overview = str(row.get("overview", "")) if pd.notna(row.get("overview")) else ""
    rating = float(row.get("vote_average", 0)) if pd.notna(row.get("vote_average")) else 0.0
    popularity = float(row.get("popularity", 0)) if pd.notna(row.get("popularity")) else 0.0
    runtime = int(row.get("runtime", 0)) if pd.notna(row.get("runtime")) else 0
    poster_path = str(row.get("poster_path", "")) if pd.notna(row.get("poster_path")) else ""

    # ── Phase B: 재무 정보 ──
    budget = int(row.get("budget", 0)) if pd.notna(row.get("budget")) else 0
    revenue = int(row.get("revenue", 0)) if pd.notna(row.get("revenue")) else 0
    vote_count = int(row.get("vote_count", 0)) if pd.notna(row.get("vote_count")) else 0

    # ── Phase B: 텍스트 메타데이터 ──
    tagline = str(row.get("tagline", "")) if pd.notna(row.get("tagline")) else ""
    homepage = str(row.get("homepage", "")) if pd.notna(row.get("homepage")) else ""

    # ── Phase B: 컬렉션/프랜차이즈 ──
    # KaggleLoader에서 belongs_to_collection을 파싱한 결과 (dict 또는 None)
    collection_parsed = row.get("collection_parsed")
    collection_id = 0
    collection_name = ""
    if isinstance(collection_parsed, dict):
        collection_id = int(collection_parsed.get("id", 0) or 0)
        collection_name = str(collection_parsed.get("name", "") or "")

    # ── Phase B: 제작사 ──
    # KaggleLoader에서 production_companies를 파싱한 결과 (list[dict])
    # Phase C에서 logo_path/origin_country를 포함한 production_companies_full로 확장하므로
    # 여기서는 파싱 원본(production_companies_parsed)만 보존한다.
    production_companies_parsed = row.get("production_companies_parsed", [])
    if not isinstance(production_companies_parsed, list):
        production_companies_parsed = []

    # ── Phase B: 제작 국가 ──
    production_countries_parsed = row.get("production_countries_parsed", [])
    if not isinstance(production_countries_parsed, list):
        production_countries_parsed = []
    production_countries = [
        str(c.get("iso_3166_1", ""))
        for c in production_countries_parsed
        if c.get("iso_3166_1")
    ]

    # ── Phase B: 언어 정보 ──
    original_language = str(row.get("original_language", "")) if pd.notna(row.get("original_language")) else ""
    spoken_languages_parsed = row.get("spoken_languages_parsed", [])
    if not isinstance(spoken_languages_parsed, list):
        spoken_languages_parsed = []
    spoken_languages = [
        str(lang.get("iso_639_1", ""))
        for lang in spoken_languages_parsed
        if lang.get("iso_639_1")
    ]

    # ── Phase B: 외부 ID 및 메타 ──
    imdb_id = str(row.get("imdb_id", "")) if pd.notna(row.get("imdb_id")) else ""
    adult = bool(row.get("adult", False))
    status = str(row.get("status", "")) if pd.notna(row.get("status")) else ""

    # ── Phase B: 확장 크레딧 (credits_df에서 조인된 컬럼) ──
    # DataFrame 조인 시 NaN이 들어올 수 있으므로 타입 방어 필수
    cast_characters = row.get("cast_characters", [])
    if not isinstance(cast_characters, list):
        cast_characters = []
    cinematographer = str(row.get("cinematographer", "")) if pd.notna(row.get("cinematographer")) else ""
    composer = str(row.get("composer", "")) if pd.notna(row.get("composer")) else ""
    screenwriters = row.get("screenwriters", [])
    if not isinstance(screenwriters, list):
        screenwriters = []
    producers = row.get("producers", [])
    if not isinstance(producers, list):
        producers = []
    editor = str(row.get("editor", "")) if pd.notna(row.get("editor")) else ""

    # ── Phase C: 추가 크루 ──
    executive_producers = row.get("executive_producers", [])
    if not isinstance(executive_producers, list):
        executive_producers = []
    production_designer_val = str(row.get("production_designer", "")) if pd.notna(row.get("production_designer")) else ""
    costume_designer_val = str(row.get("costume_designer", "")) if pd.notna(row.get("costume_designer")) else ""
    source_author_val = str(row.get("source_author", "")) if pd.notna(row.get("source_author")) else ""

    # ── Phase C: 감독 상세 정보 ──
    director_details = row.get("director_details", {})
    if not isinstance(director_details, dict):
        director_details = {}
    director_id = int(director_details.get("id", 0) or 0)
    director_profile_path = str(director_details.get("profile_path", "") or "")

    # Phase C: 비디오 플래그
    video_flag = bool(row.get("video_flag", False))

    # Phase C: 컬렉션 이미지 추출
    collection_poster_path = ""
    collection_backdrop_path = ""
    if isinstance(collection_parsed, dict):
        collection_poster_path = str(collection_parsed.get("poster_path") or "")
        collection_backdrop_path = str(collection_parsed.get("backdrop_path") or "")

    # Phase C: 제작사 확장 (logo_path, origin_country 포함)
    production_companies_full = [
        {
            "id": int(c.get("id", 0) or 0),
            "name": str(c.get("name", "")),
            "logo_path": str(c.get("logo_path") or ""),
            "origin_country": str(c.get("origin_country", "")),
        }
        for c in production_companies_parsed
        if c.get("name")
    ]

    # Phase C: 국가/언어 전체 이름
    production_country_names = [
        str(c.get("name", ""))
        for c in production_countries_parsed
        if c.get("name")
    ]
    spoken_language_names = [
        str(lang.get("name") or lang.get("english_name", ""))
        for lang in spoken_languages_parsed
        if lang.get("name") or lang.get("english_name")
    ]

    # 무드태그 (장르 기반 fallback)
    mood_tags = get_fallback_mood_tags(genres)

    # 줄거리 빈 문자열 대체
    if not overview or len(overview) < 10:
        overview = f"{', '.join(genres)} 장르의 영화. 키워드: {', '.join(keywords_list[:5])}"

    # MovieDocument 생성
    doc = MovieDocument(
        id=str(tmdb_id),
        title=title,
        title_en=title_en,
        overview=overview,
        release_year=release_year,
        runtime=runtime,
        rating=rating,
        popularity_score=popularity,
        poster_path=poster_path,
        genres=genres,
        keywords=keywords_list,
        director=director,
        cast=cast_names,
        ott_platforms=[],  # Kaggle에는 OTT 정보 없음
        mood_tags=mood_tags,
        # Phase B: 재무/텍스트 메타데이터
        budget=budget,
        revenue=revenue,
        vote_count=vote_count,
        tagline=tagline,
        homepage=homepage,
        # Phase B: 컬렉션/제작사
        collection_id=collection_id,
        collection_name=collection_name,
        production_companies=production_companies_full,
        production_countries=production_countries,
        original_language=original_language,
        spoken_languages=spoken_languages,
        imdb_id=imdb_id,
        adult=adult,
        status=status,
        # Phase B: 확장 크레딧
        cast_characters=cast_characters,
        cinematographer=cinematographer,
        composer=composer,
        screenwriters=screenwriters,
        producers=producers,
        editor=editor,
        # Phase C: 완전 데이터 추출
        director_id=director_id,
        director_profile_path=director_profile_path,
        video_flag=video_flag,
        executive_producers=executive_producers,
        production_designer=production_designer_val,
        costume_designer=costume_designer_val,
        source_author=source_author_val,
        collection_poster_path=collection_poster_path,
        collection_backdrop_path=collection_backdrop_path,
        production_country_names=production_country_names,
        spoken_language_names=spoken_language_names,
        source="kaggle",
    )

    # 임베딩 텍스트 구성
    doc.embedding_text = build_embedding_text(doc)

    return doc
