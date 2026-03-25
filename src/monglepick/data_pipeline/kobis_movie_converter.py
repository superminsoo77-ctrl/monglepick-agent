"""
KOBIS 영화 목록 데이터 → MovieDocument 변환기.

KOBIS searchMovieList API 응답 데이터를 MovieDocument로 변환한다.
KOBIS는 TMDB와 달리 줄거리(overview), 키워드(keywords), 무드태그, OTT 정보가 없으므로
사용 가능한 필드만으로 MovieDocument를 구성한다.

변환 가능한 필드 (목록 API):
  - title, title_en, release_year, genres, director, nation, companies
  - kobis_movie_cd, kobis_genres, kobis_nation, kobis_open_dt, kobis_type_nm

상세 API 보강 시 추가되는 필드:
  - cast, runtime, certification, staffs, show_types

중복 제거:
  KOBIS 영화 ID(8자리)와 TMDB ID(1~7자리)는 범위가 다르므로 ID만으로는
  동일 영화를 식별할 수 없다. 따라서 정규화된 제목+연도(±1) 매칭으로
  기존 DB 영화와의 중복을 제거한다 (dedup_kobis_movies 함수).
"""

from __future__ import annotations

import structlog

from monglepick.data_pipeline.kobis_collector import _normalize_title
from monglepick.data_pipeline.models import MovieDocument

logger = structlog.get_logger()

# ── KOBIS 장르 → TMDB 스타일 한국어 장르 매핑 ──
# KOBIS 장르는 TMDB와 분류 체계가 다름 (예: '멜로/로맨스' → '로맨스')
KOBIS_GENRE_NORMALIZE: dict[str, str] = {
    "멜로/로맨스": "로맨스",
    "사극": "역사",
    "공포(호러)": "공포",
    "뮤지컬": "음악",
    "기타": "",  # 기타는 제외
    "성인물(에로)": "",  # 제외
}


def _parse_genres(genre_alt: str) -> list[str]:
    """
    KOBIS genre_alt 문자열을 장르 리스트로 변환한다.

    '드라마,멜로/로맨스' → ['드라마', '로맨스']
    """
    if not genre_alt:
        return []
    genres = []
    for g in genre_alt.split(","):
        g = g.strip()
        if not g:
            continue
        # 정규화 매핑 적용
        normalized = KOBIS_GENRE_NORMALIZE.get(g, g)
        if normalized and normalized not in genres:
            genres.append(normalized)
    return genres


def _extract_year(open_dt: str, prdt_year: str) -> int:
    """
    KOBIS 개봉일/제작년도에서 연도를 추출한다.

    open_dt 우선 (YYYYMMDD → YYYY), prdt_year 보조 (YYYY).
    """
    if open_dt and len(open_dt) >= 4:
        try:
            year = int(open_dt[:4])
            if 1900 <= year <= 2100:
                return year
        except ValueError:
            pass
    if prdt_year:
        try:
            year = int(prdt_year)
            if 1900 <= year <= 2100:
                return year
        except ValueError:
            pass
    return 0


def _extract_director(directors: list[dict]) -> str:
    """감독 목록에서 첫 번째 감독명을 추출한다."""
    if directors and directors[0].get("peopleNm"):
        return directors[0]["peopleNm"]
    return ""


def _build_embedding_text(
    title: str,
    genres: list[str],
    director: str,
    actors: list[str] | None = None,
    nation: str = "",
) -> str:
    """
    KOBIS 영화용 임베딩 텍스트를 생성한다.

    TMDB와 달리 overview/keywords/mood_tags가 없으므로
    제목, 장르, 감독, 배우, 국가만으로 구성한다.

    형식: [제목] {title} [장르] {genres} [감독] {director} [출연] {actors} [국가] {nation}
    """
    parts = [f"[제목] {title}"]
    if genres:
        parts.append(f"[장르] {', '.join(genres)}")
    if director:
        parts.append(f"[감독] {director}")
    if actors:
        parts.append(f"[출연] {', '.join(actors[:5])}")
    if nation:
        parts.append(f"[국가] {nation}")
    return " ".join(parts)


def kobis_list_to_movie_document(
    kobis_movie: dict,
    detail_data: dict | None = None,
    boxoffice_data: dict | None = None,
) -> MovieDocument | None:
    """
    KOBIS 영화 목록 데이터를 MovieDocument로 변환한다.

    Args:
        kobis_movie: KOBIS searchMovieList API 응답의 단일 영화 dict
            필수: movieCd, movieNm
            선택: movieNmEn, openDt, prdtYear, genreAlt, nationAlt, directors, companys
        detail_data: KOBIS searchMovieInfo API 보강 데이터 (선택)
        boxoffice_data: 박스오피스 보강 데이터 (선택)

    Returns:
        MovieDocument 또는 None (유효성 실패 시)
    """
    movie_cd = kobis_movie.get("movieCd", "")
    movie_nm = kobis_movie.get("movieNm", "")

    # 필수 필드 검증
    if not movie_cd or not movie_nm:
        return None

    # 제목이 너무 짧으면 스킵 (1자 이하)
    if len(movie_nm.strip()) <= 1:
        return None

    # ── 기본 필드 추출 (목록 API) ──
    title = movie_nm.strip()
    title_en = kobis_movie.get("movieNmEn", "").strip()
    open_dt = kobis_movie.get("openDt", "").replace("-", "")
    prdt_year = kobis_movie.get("prdtYear", "")
    release_year = _extract_year(open_dt, prdt_year)
    genre_alt = kobis_movie.get("genreAlt", "")
    genres = _parse_genres(genre_alt)
    nation_alt = kobis_movie.get("nationAlt", "")
    rep_nation = kobis_movie.get("repNationNm", "")
    type_nm = kobis_movie.get("typeNm", "")

    # 감독 추출
    directors_raw = kobis_movie.get("directors", [])
    director = _extract_director(directors_raw)

    # 제작사 추출
    companies_raw = kobis_movie.get("companys", [])
    production_companies = [
        {"name": c.get("companyNm", ""), "id": 0}
        for c in companies_raw
        if c.get("companyNm")
    ]

    # KOBIS 감독 정보 (상세)
    kobis_directors = [
        {"peopleNm": d.get("peopleNm", ""), "peopleNmEn": d.get("peopleNmEn", "")}
        for d in directors_raw
    ]

    # ── 상세 API(searchMovieInfo) 보강 데이터 ──
    # 목록 API만으로는 배우/런타임/관람등급 정보가 없으므로,
    # 상세 API 응답(detail_data)이 있을 때만 보강한다.
    cast: list[str] = []
    cast_characters: list[dict] = []
    runtime = 0
    certification = ""
    kobis_actors: list[dict] = []
    kobis_companies: list[dict] = []
    kobis_staffs: list[dict] = []
    kobis_watch_grade = ""
    kobis_genres_detail: list[str] = []

    if detail_data:
        # 배우 추출
        actors_raw = detail_data.get("actors", [])
        cast = [a.get("peopleNm", "") for a in actors_raw[:5] if a.get("peopleNm")]
        cast_characters = [
            {
                "name": a.get("peopleNm", ""),
                "character": a.get("cast", ""),
            }
            for a in actors_raw[:10]
            if a.get("peopleNm")
        ]
        kobis_actors = [
            {
                "peopleNm": a.get("peopleNm", ""),
                "peopleNmEn": a.get("peopleNmEn", ""),
                "cast": a.get("cast", ""),
            }
            for a in actors_raw[:10]
        ]

        # 상영시간
        show_tm = detail_data.get("show_tm", "")
        if show_tm:
            try:
                runtime = int(show_tm)
            except ValueError:
                pass

        # 관람등급
        kobis_watch_grade = detail_data.get("watch_grade_nm", "")
        certification = kobis_watch_grade

        # 회사 (제작사/배급사 구분)
        companys_detail = detail_data.get("companys", [])
        if companys_detail:
            kobis_companies = [
                {
                    "companyNm": c.get("companyNm", ""),
                    "companyPartNm": c.get("companyPartNm", ""),
                }
                for c in companys_detail
            ]
            # 제작사만 production_companies에 반영
            production_companies = [
                {"name": c.get("companyNm", ""), "id": 0}
                for c in companys_detail
                if c.get("companyNm")
            ]

        # 스태프
        staffs_raw = detail_data.get("staffs", [])
        if staffs_raw:
            kobis_staffs = [
                {
                    "peopleNm": s.get("peopleNm", ""),
                    "peopleNmEn": s.get("peopleNmEn", ""),
                    "staffRoleNm": s.get("staffRoleNm", ""),
                }
                for s in staffs_raw
            ]

        # 장르 (상세 API)
        genres_detail = detail_data.get("genres", [])
        if genres_detail:
            kobis_genres_detail = [g.get("genreNm", "") for g in genres_detail if g.get("genreNm")]
            # 상세 장르로 기본 장르 보강
            if not genres:
                genres = _parse_genres(",".join(kobis_genres_detail))

        # 감독 (상세 API에서 더 정확한 정보)
        directors_detail = detail_data.get("directors", [])
        if directors_detail:
            director = directors_detail[0].get("peopleNm", director)
            kobis_directors = [
                {"peopleNm": d.get("peopleNm", ""), "peopleNmEn": d.get("peopleNmEn", "")}
                for d in directors_detail
            ]

    # ── 박스오피스 보강 ──
    audience_count = 0
    sales_acc = 0
    screen_count = 0

    if boxoffice_data:
        audience_count = boxoffice_data.get("audi_acc", 0)
        sales_acc = boxoffice_data.get("sales_acc", 0)
        screen_count = boxoffice_data.get("scrn_cnt", 0)

    # ── 임베딩 텍스트 생성 ──
    embedding_text = _build_embedding_text(
        title=title,
        genres=genres,
        director=director,
        actors=cast,
        nation=nation_alt or rep_nation,
    )

    # ── KOBIS 장르 (원본 보존) ──
    # 상세 API 장르가 있으면 우선 사용, 없으면 목록 API의 genreAlt를 파싱
    kobis_genres = kobis_genres_detail if kobis_genres_detail else (
        [g.strip() for g in genre_alt.split(",") if g.strip()] if genre_alt else []
    )

    # ── MovieDocument 생성 ──
    try:
        doc = MovieDocument(
            id=movie_cd,  # KOBIS 영화 코드를 ID로 사용
            title=title,
            title_en=title_en,
            release_year=release_year,
            runtime=runtime,
            genres=genres,
            director=director,
            cast=cast,
            cast_characters=cast_characters,
            certification=certification,
            production_companies=production_companies,
            original_language="ko" if "한국" in (nation_alt or rep_nation or "") else "",
            production_countries=_extract_country_codes(nation_alt),
            production_country_names=[n.strip() for n in nation_alt.split(",") if n.strip()] if nation_alt else [],
            embedding_text=embedding_text,
            # KOBIS 전용 필드
            kobis_movie_cd=movie_cd,
            kobis_genres=kobis_genres,
            kobis_directors=kobis_directors,
            kobis_actors=kobis_actors,
            kobis_companies=kobis_companies,
            kobis_staffs=kobis_staffs,
            kobis_nation=nation_alt or rep_nation,
            kobis_watch_grade=kobis_watch_grade,
            kobis_open_dt=open_dt,
            kobis_type_nm=type_nm,
            # 박스오피스 데이터
            audience_count=audience_count,
            sales_acc=sales_acc,
            screen_count=screen_count,
            # 출처
            source="kobis",
        )
        return doc
    except Exception as e:
        logger.debug("kobis_movie_conversion_failed", movie_cd=movie_cd, error=str(e))
        return None


def _extract_country_codes(nation_alt: str) -> list[str]:
    """
    KOBIS 국가명을 ISO 코드로 변환한다 (주요 국가만).

    '한국' → ['KR'], '미국' → ['US'], '한국,미국' → ['KR', 'US']
    """
    NATION_TO_ISO: dict[str, str] = {
        "한국": "KR", "미국": "US", "일본": "JP", "중국": "CN",
        "영국": "GB", "프랑스": "FR", "독일": "DE", "이탈리아": "IT",
        "스페인": "ES", "캐나다": "CA", "호주": "AU", "인도": "IN",
        "홍콩": "HK", "대만": "TW", "태국": "TH", "러시아": "RU",
        "브라질": "BR", "멕시코": "MX", "아르헨티나": "AR",
        "덴마크": "DK", "스웨덴": "SE", "노르웨이": "NO", "핀란드": "FI",
        "뉴질랜드": "NZ", "벨기에": "BE", "네덜란드": "NL",
        "스위스": "CH", "폴란드": "PL", "터키": "TR", "이란": "IR",
    }
    if not nation_alt:
        return []
    codes = []
    for nation in nation_alt.split(","):
        nation = nation.strip()
        code = NATION_TO_ISO.get(nation, "")
        if code and code not in codes:
            codes.append(code)
    return codes


def dedup_kobis_movies(
    kobis_movies: list[dict],
    db_movies: list[dict],
    exclude_ids: set[str | int] | None = None,
) -> list[dict]:
    """
    KOBIS 영화 목록에서 기존 DB 영화와 제목+연도로 중복되는 영화를 제거한다.

    KOBIS ID(8자리)와 TMDB ID(1~7자리)는 범위가 다르므로 ID만으로는
    동일 영화를 식별할 수 없다. 따라서 정규화된 제목(한국어/영어) + 연도(±1)
    매칭으로 기존 DB에 이미 존재하는 영화를 필터링한다.

    매칭 순서:
      1. 정규화된 한국어 제목 + 연도 ±1 (KOBIS movieNm ↔ DB title)
      2. 정규화된 영문 제목 + 연도 ±1 (KOBIS movieNmEn ↔ DB title_en)

    Args:
        kobis_movies: KOBIS searchMovieList 응답 리스트
        db_movies: Qdrant에서 로드한 기존 DB 영화
            각 항목: {'id': '157336', 'title': '인터스텔라', 'title_en': 'Interstellar', 'release_year': 2014}
        exclude_ids: 추가 제외 ID 집합 (ID 기반 중복 제거, KOBIS movieCd 또는 TMDB ID)

    Returns:
        list[dict]: 중복이 제거된 KOBIS 영화 리스트 (DB에 없는 영화만)
    """
    # ── DB 영화 인덱스 구축 ──
    # set으로 (정규화된_제목, 연도) 쌍을 저장하여 O(1) 중복 검사 수행
    db_index_kr: set[tuple[str, int]] = set()
    db_index_en: set[tuple[str, int]] = set()

    for movie in db_movies:
        year = int(movie.get("release_year", 0)) if movie.get("release_year") else 0
        if year < 1900:
            continue

        # 한국어 제목 인덱스
        title_kr = _normalize_title(movie.get("title", ""))
        if title_kr:
            db_index_kr.add((title_kr, year))

        # 영문 제목 인덱스
        title_en = _normalize_title(movie.get("title_en", ""))
        if title_en:
            db_index_en.add((title_en, year))

    logger.info(
        "dedup_index_built",
        kr_index_size=len(db_index_kr),
        en_index_size=len(db_index_en),
        db_movies_total=len(db_movies),
    )

    # ── KOBIS 영화 3단계 중복 검사 ──
    # 1단계: ID 기반 제외 → 2단계: 한국어 제목+연도 매칭 → 3단계: 영문 제목+연도 매칭
    exclude = set(str(x) for x in (exclude_ids or set()))
    deduped: list[dict] = []
    id_excluded = 0
    title_duplicates = 0

    for kobis in kobis_movies:
        movie_cd = kobis.get("movieCd", "")

        # 1차: ID 기반 중복 제거 (이미 DB에 있는 ID)
        if movie_cd in exclude:
            id_excluded += 1
            continue

        # KOBIS 개봉 연도 추출 (openDt 우선, prdtYear 보조)
        open_dt = kobis.get("openDt", "")
        prdt_year = kobis.get("prdtYear", "")
        year = 0
        if open_dt and len(open_dt) >= 4:
            try:
                year = int(open_dt[:4])
            except ValueError:
                pass
        if not year and prdt_year:
            try:
                year = int(prdt_year)
            except ValueError:
                pass

        # 연도가 없으면 중복 검사 불가 → 그대로 포함 (변환 단계에서 필터링)
        if year < 1900 or year > 2100:
            deduped.append(kobis)
            continue

        # 2차: 한국어 제목 + 연도 ±1 매칭
        kobis_title_kr = _normalize_title(kobis.get("movieNm", ""))
        found_match = False

        if kobis_title_kr:
            for y_offset in [0, -1, 1]:
                if (kobis_title_kr, year + y_offset) in db_index_kr:
                    title_duplicates += 1
                    found_match = True
                    break

        # 3차: 영문 제목 + 연도 ±1 매칭 (한국어 제목 매칭 실패 시)
        if not found_match:
            kobis_title_en = _normalize_title(kobis.get("movieNmEn", ""))
            if kobis_title_en:
                for y_offset in [0, -1, 1]:
                    if (kobis_title_en, year + y_offset) in db_index_en:
                        title_duplicates += 1
                        found_match = True
                        break

        # 중복이 아닌 영화만 포함
        if not found_match:
            deduped.append(kobis)

    logger.info(
        "kobis_dedup_complete",
        total_kobis=len(kobis_movies),
        id_excluded=id_excluded,
        title_duplicates=title_duplicates,
        remaining=len(deduped),
    )

    return deduped


def convert_kobis_movies(
    kobis_movies: list[dict],
    exclude_ids: set[str | int] | None = None,
    detail_map: dict[str, dict] | None = None,
    boxoffice_map: dict[str, dict] | None = None,
) -> list[MovieDocument]:
    """
    KOBIS 영화 목록을 MovieDocument 리스트로 변환한다.

    Args:
        kobis_movies: KOBIS searchMovieList API 응답 리스트
        exclude_ids: 제외할 ID 집합 (이미 DB에 있는 영화, TMDB ID 또는 KOBIS 코드)
        detail_map: {movieCd: detail_dict} 상세정보 매핑 (선택)
        boxoffice_map: {movieCd: boxoffice_dict} 박스오피스 매핑 (선택)

    Returns:
        list[MovieDocument]: 변환 성공한 영화 리스트
    """
    exclude = set(str(x) for x in (exclude_ids or set()))
    detail_map = detail_map or {}
    boxoffice_map = boxoffice_map or {}

    documents: list[MovieDocument] = []
    skipped = 0
    failed = 0

    for movie in kobis_movies:
        movie_cd = movie.get("movieCd", "")

        # 이미 DB에 있는 영화 스킵
        if movie_cd in exclude:
            skipped += 1
            continue

        # 변환
        detail = detail_map.get(movie_cd)
        boxoffice = boxoffice_map.get(movie_cd)
        doc = kobis_list_to_movie_document(movie, detail, boxoffice)

        if doc:
            documents.append(doc)
        else:
            failed += 1

    logger.info(
        "kobis_movie_conversion_complete",
        total=len(kobis_movies),
        converted=len(documents),
        skipped=skipped,
        failed=failed,
    )

    return documents
