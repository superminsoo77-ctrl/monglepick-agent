"""
데이터 파이프라인 Pydantic 모델 정의

§11 데이터 파이프라인에서 사용하는 핵심 데이터 모델.
TMDB/KOBIS/Kaggle에서 수집한 원본 데이터를 정규화하여
Qdrant, Neo4j, Elasticsearch에 적재하는 중간 표현(MovieDocument)을 정의한다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MovieDocument(BaseModel):
    """
    정규화된 영화 문서 모델.

    TMDB + KOBIS + Kaggle 데이터를 병합·전처리한 결과물.
    이 모델이 Qdrant(벡터), Neo4j(그래프), Elasticsearch(BM25)에 적재되는 공통 입력이다.

    §11-1 전체 흐름: raw data → preprocessor → MovieDocument → 4개 저장소 적재
    """

    # ── 식별자 ──
    id: str = Field(..., description="TMDB ID (문자열, 예: '157336')")

    # ── 기본 메타데이터 ──
    title: str = Field(..., description="한국어 제목 (TMDB 우선, KOBIS 보강)")
    title_en: str = Field(default="", description="영문 제목")
    overview: str = Field(default="", description="줄거리 (TMDB, 한국어 우선)")
    release_year: int = Field(default=0, description="개봉 연도 (1900~현재)")
    runtime: int = Field(default=0, description="러닝타임 (분, KOBIS 우선)")
    rating: float = Field(default=0.0, description="TMDB 평점 (0~10)")
    popularity_score: float = Field(default=0.0, description="인기도 (TMDB + KOBIS 관객수 보정)")
    poster_path: str = Field(default="", description="포스터 이미지 경로")

    # ── 분류 정보 ──
    genres: list[str] = Field(default_factory=list, description="장르 한국어 배열 (예: ['SF', '드라마'])")
    keywords: list[str] = Field(default_factory=list, description="키워드 배열")

    # ── 인물 정보 ──
    director: str = Field(default="", description="감독명 (KOBIS 우선, TMDB 보강)")
    cast: list[str] = Field(default_factory=list, description="출연 배우 상위 5명")

    # ── AI 생성 필드 ──
    mood_tags: list[str] = Field(
        default_factory=list,
        description="무드 태그 3~5개 (GPT-4o-mini 생성, 25개 화이트리스트 한정)",
    )

    # ── 플랫폼 정보 ──
    ott_platforms: list[str] = Field(
        default_factory=list,
        description="OTT 플랫폼 한국어 배열 (예: ['넷플릭스', '왓챠'])",
    )

    # ── 임베딩 입력 텍스트 ──
    embedding_text: str = Field(
        default="",
        description=(
            "임베딩 모델 입력용 구조화 텍스트. "
            "형식: [제목] {title} [장르] {genres} [감독] {director} "
            "[키워드] {keywords} [무드] {mood_tags} [줄거리] {overview[:200]}"
        ),
    )

    # ── TMDB 보강 필드 (Phase A) ──
    vote_count: int = Field(default=0, description="TMDB 투표 수")
    reviews: list[str] = Field(
        default_factory=list,
        description="리뷰 텍스트 상위 5개 (각 500자 제한)",
    )
    trailer_url: str = Field(default="", description="대표 트레일러 YouTube URL")
    behind_the_scenes: list[str] = Field(
        default_factory=list,
        description="비하인드/피처렛 YouTube URL 목록",
    )
    certification: str = Field(default="", description="한국 관람등급 (예: '15세 이상 관람가')")
    similar_movie_ids: list[str] = Field(
        default_factory=list,
        description="TMDB 유사 영화 ID 목록 (문자열)",
    )

    # ── Phase B: 재무/흥행 정보 ──
    budget: int = Field(default=0, description="제작 예산 (USD)")
    revenue: int = Field(default=0, description="총 수익 (USD)")

    # ── Phase B: 텍스트 메타데이터 ──
    tagline: str = Field(default="", description="영화 캐치프레이즈/태그라인")
    homepage: str = Field(default="", description="공식 웹사이트 URL")

    # ── Phase B: 프랜차이즈/컬렉션 정보 ──
    collection_id: int = Field(default=0, description="소속 컬렉션/프랜차이즈 ID (TMDB)")
    collection_name: str = Field(default="", description="소속 컬렉션명 (예: 'Toy Story Collection')")

    # ── Phase B: 제작사 정보 ──
    production_companies: list[dict] = Field(
        default_factory=list,
        description="제작사 목록 [{'id': 3, 'name': 'Pixar Animation Studios'}, ...]",
    )

    # ── Phase B: 국가/언어 정보 ──
    production_countries: list[str] = Field(
        default_factory=list,
        description="제작 국가 ISO 코드 목록 (예: ['US', 'KR'])",
    )
    original_language: str = Field(default="", description="원본 언어 ISO 코드 (예: 'en', 'ko')")
    spoken_languages: list[str] = Field(
        default_factory=list,
        description="사용 언어 ISO 코드 목록 (예: ['en', 'fr'])",
    )

    # ── Phase B: 외부 ID 및 미디어 ──
    imdb_id: str = Field(default="", description="IMDb ID (예: 'tt0114709')")
    backdrop_path: str = Field(default="", description="배경 이미지 경로 (TMDB)")
    adult: bool = Field(default=False, description="성인물 여부")
    status: str = Field(default="", description="영화 상태 ('Released', 'Post Production' 등)")

    # ── Phase B: 확장 크레딧 정보 ──
    cast_characters: list[dict] = Field(
        default_factory=list,
        description="배우-캐릭터 매핑 [{'name': 'Tom Hanks', 'character': 'Woody'}, ...]",
    )
    cinematographer: str = Field(default="", description="촬영감독명")
    composer: str = Field(default="", description="음악감독/작곡가명")
    screenwriters: list[str] = Field(default_factory=list, description="각본가 목록")
    producers: list[str] = Field(default_factory=list, description="프로듀서 목록 (Producer 직급만)")
    editor: str = Field(default="", description="편집자명")

    # ── Phase C: 완전 데이터 추출 ──
    # 창작 원산국 (TMDB origin_country, production_countries와 다름)
    origin_country: list[str] = Field(
        default_factory=list,
        description="창작 원산국 ISO 코드 (예: 미국 제작 한국 영화라도 ['KR'])",
    )
    # 감독 확장 정보 (Person 노드 중복 방지 + UI용)
    director_id: int = Field(default=0, description="TMDB 감독 person ID (Neo4j 유니크 키)")
    director_profile_path: str = Field(default="", description="감독 프로필 사진 경로")
    director_original_name: str = Field(default="", description="감독 원어 이름")
    # 대체 제목 (검색 recall 개선용)
    alternative_titles: list[dict] = Field(
        default_factory=list,
        description="대체 제목 [{'iso_3166_1': 'KR', 'title': '겨울왕국', 'type': ''}, ...]",
    )
    # TMDB 추천 영화 (similar_movie_ids와 다른 알고리즘 기반)
    recommendation_ids: list[str] = Field(
        default_factory=list,
        description="TMDB 추천 영화 ID 목록 (similar와 다른 알고리즘)",
    )
    # 다중 이미지 (poster_path 1개 → 다수)
    images_posters: list[str] = Field(
        default_factory=list,
        description="추가 포스터 이미지 경로 목록 (최대 10개)",
    )
    images_backdrops: list[str] = Field(
        default_factory=list,
        description="추가 배경 이미지 경로 목록 (최대 10개)",
    )
    # 컬렉션/프랜차이즈 이미지
    collection_poster_path: str = Field(default="", description="컬렉션 포스터 이미지 경로")
    collection_backdrop_path: str = Field(default="", description="컬렉션 배경 이미지 경로")
    # 한국 개봉일 (release_dates에서 추출, base release_date와 다를 수 있음)
    kr_release_date: str = Field(default="", description="한국 개봉일 (YYYY-MM-DD)")
    # 직접 비디오 플래그 (비극장 개봉 콘텐츠)
    video_flag: bool = Field(default=False, description="직접 비디오/홈 비디오 콘텐츠 여부")
    # 추가 크루 직군
    executive_producers: list[str] = Field(
        default_factory=list,
        description="총괄 프로듀서 목록 (Executive Producer)",
    )
    production_designer: str = Field(default="", description="프로덕션 디자이너명")
    costume_designer: str = Field(default="", description="의상 디자이너명")
    source_author: str = Field(default="", description="원작 작가 (Novel/Characters)")
    # 제작 국가/언어 전체 이름 (ISO 코드 외 보충)
    production_country_names: list[str] = Field(
        default_factory=list,
        description="제작 국가 전체 이름 (예: ['United States of America'])",
    )
    spoken_language_names: list[str] = Field(
        default_factory=list,
        description="사용 언어 전체 이름 (예: ['English', '한국어'])",
    )

    # ── KMDb 보강 필드 ──
    kmdb_id: str = Field(default="", description="KMDb 영화 고유 ID (movieId + movieSeq)")
    awards: str = Field(default="", description="수상 내역 (KMDb Awards1 + Awards2)")
    audience_count: int = Field(default=0, description="누적 관객수 (KOBIS audiAcc / KMDb)")
    filming_location: str = Field(default="", description="촬영 장소 (KMDb fLocation)")
    stills: list[str] = Field(default_factory=list, description="스틸컷 이미지 URL 목록 (KMDb)")
    theme_song: str = Field(default="", description="주제곡 (KMDb)")
    soundtrack: str = Field(default="", description="삽입곡 (KMDb)")

    # ── KOBIS 보강 필드 ──
    kobis_movie_cd: str = Field(default="", description="KOBIS 영화 코드 (추적/연동용)")
    sales_acc: int = Field(default=0, description="누적 매출액 (KRW, KOBIS 박스오피스)")
    screen_count: int = Field(default=0, description="최대 상영 스크린 수 (KOBIS)")
    kobis_genres: list[str] = Field(
        default_factory=list,
        description="KOBIS 장르 분류 (예: ['드라마', '멜로/로맨스'], TMDB 장르와 다를 수 있음)",
    )
    kobis_directors: list[dict] = Field(
        default_factory=list,
        description="KOBIS 감독 정보 [{'peopleNm': '봉준호', 'peopleNmEn': 'BONG Joon-ho'}]",
    )
    kobis_actors: list[dict] = Field(
        default_factory=list,
        description="KOBIS 배우 정보 [{'peopleNm': '송강호', 'cast': '기택'}] (캐릭터명 포함)",
    )
    kobis_companies: list[dict] = Field(
        default_factory=list,
        description="KOBIS 제작/배급사 [{'companyNm': 'CJ ENM', 'companyPartNm': '배급사'}]",
    )
    kobis_staffs: list[dict] = Field(
        default_factory=list,
        description="KOBIS 스태프 [{'peopleNm': '한스 짐머', 'staffRoleNm': '음악'}]",
    )
    kobis_nation: str = Field(default="", description="KOBIS 제작국가 (예: '한국', '미국')")
    kobis_watch_grade: str = Field(default="", description="KOBIS 관람등급 (예: '12세이상관람가')")
    kobis_open_dt: str = Field(default="", description="KOBIS 개봉일 (YYYYMMDD)")
    kobis_type_nm: str = Field(default="", description="KOBIS 영화 유형 (예: '장편', '단편', '애니메이션')")

    # ── Phase D: TMDB 전체 수집 보강 필드 ──
    # translations에서 추출한 다국어 줄거리 (overview 빈값 보강용)
    overview_en: str = Field(default="", description="영문 줄거리 (translations에서 추출)")
    overview_ja: str = Field(default="", description="일본어 줄거리 (translations에서 추출)")
    # 외부 서비스 ID (소셜 미디어 연동, 외부 DB 크로스레퍼런스)
    facebook_id: str = Field(default="", description="Facebook 페이지 ID")
    instagram_id: str = Field(default="", description="Instagram 계정 ID")
    twitter_id: str = Field(default="", description="Twitter/X 계정 ID")
    wikidata_id: str = Field(default="", description="Wikidata 항목 ID (예: 'Q12345')")
    # TMDB 사용자 리스트 포함 수 (인기도 보조 지표)
    tmdb_list_count: int = Field(default=0, description="이 영화가 포함된 TMDB 사용자 리스트 수")
    # 로고 이미지 (UI 표시용)
    images_logos: list[str] = Field(
        default_factory=list,
        description="로고 이미지 경로 목록",
    )

    # ── 데이터 출처 추적 ──
    source: str = Field(default="tmdb", description="데이터 출처 ('tmdb', 'kobis', 'kaggle', 'kmdb', 'merged')")


class TMDBRawMovie(BaseModel):
    """
    TMDB API에서 수집한 원본 영화 데이터.

    /movie/{id}?append_to_response=credits,keywords,reviews,videos,similar_movies,release_dates,images,alternative_titles,recommendations
    응답을 파싱한 구조. 전처리기(preprocessor)가 이 모델을 MovieDocument로 변환한다.
    """

    id: int
    title: str = ""
    original_title: str = ""
    overview: str = ""
    release_date: str = ""
    vote_average: float = 0.0
    vote_count: int = 0  # Phase A: 투표 수
    popularity: float = 0.0
    poster_path: str | None = None
    runtime: int | None = None
    # Phase D: video 플래그 (TMDB 기본 상세에 포함, 비극장 개봉 콘텐츠 식별)
    video: bool = False

    # 장르 (TMDB genre 객체 배열)
    genres: list[dict] = Field(default_factory=list, description="[{'id': 28, 'name': 'Action'}, ...]")

    # 크레딧 (감독 + 배우)
    credits: dict = Field(default_factory=dict, description="{'crew': [...], 'cast': [...]}")

    # 키워드
    keywords: dict = Field(default_factory=dict, description="{'keywords': [{'id': 1, 'name': '...'}]}")

    # OTT 제공 정보 (watch/providers)
    watch_providers: dict = Field(default_factory=dict, description="{'KR': {'flatrate': [...]}}")

    # ── Phase A: TMDB append_to_response 보강 필드 ──
    # Phase D: reviews 전체 dict 저장 (기존 3개 필드만 → 원본 응답 전체 보존)
    reviews: list[dict] = Field(
        default_factory=list,
        description="리뷰 원본 전체 [{'author': '...', 'content': '...', 'author_details': {...}, 'created_at': '...', 'updated_at': '...', 'url': '...', 'id': '...'}]",
    )
    videos: list[dict] = Field(
        default_factory=list,
        description="[{'key': 'dQw4...', 'type': 'Trailer', 'site': 'YouTube'}]",
    )
    similar_movie_ids: list[int] = Field(
        default_factory=list,
        description="TMDB 유사 영화 ID 목록 (하위 호환용)",
    )
    release_dates: list[dict] = Field(
        default_factory=list,
        description="국가별 개봉일 + 관람등급 (release_dates.results)",
    )

    # ── Phase B: TMDB 보강 필드 ──
    budget: int = 0
    revenue: int = 0
    tagline: str = ""
    homepage: str = ""
    belongs_to_collection: dict | None = Field(
        default=None,
        description="컬렉션 정보 {'id': 10194, 'name': 'Toy Story Collection', 'poster_path': '...', 'backdrop_path': '...'}",
    )
    production_companies: list[dict] = Field(
        default_factory=list,
        description="제작사 [{'id': 3, 'name': 'Pixar', 'logo_path': '...', 'origin_country': 'US'}, ...]",
    )
    production_countries: list[dict] = Field(
        default_factory=list,
        description="제작국가 [{'iso_3166_1': 'US', 'name': 'United States'}, ...]",
    )
    original_language: str = ""
    spoken_languages: list[dict] = Field(
        default_factory=list,
        description="사용언어 [{'iso_639_1': 'en', 'english_name': 'English', 'name': 'English'}, ...]",
    )
    imdb_id: str = ""
    backdrop_path: str | None = None
    adult: bool = False
    status: str = ""

    # ── Phase C: TMDB 추가 보강 필드 ──
    origin_country: list[str] = Field(
        default_factory=list,
        description="창작 원산국 ISO 코드 (production_countries와 다름)",
    )
    alternative_titles: list[dict] = Field(
        default_factory=list,
        description="대체 제목 [{'iso_3166_1': 'KR', 'title': '겨울왕국', 'type': ''}, ...]",
    )
    # Phase D: recommendations 전체 메타데이터 (기존 ID만 추출 → dict 전체 저장)
    recommendations: list[dict] = Field(
        default_factory=list,
        description="TMDB 추천 영화 전체 메타데이터 [{'id': 123, 'title': '...', 'overview': '...', 'poster_path': '...'}, ...]",
    )
    images: dict = Field(
        default_factory=dict,
        description="다중 이미지 {'posters': [...], 'backdrops': [...], 'logos': [...]}",
    )

    # ── Phase D: TMDB 전체 데이터 수집 (Full Collection) ──
    # Phase D: translations 전체 dict 저장 (기존 5개 필드만 → data dict 전체 보존)
    translations: list[dict] = Field(
        default_factory=list,
        description="다국어 번역 원본 전체 [{'iso_3166_1': 'KR', 'iso_639_1': 'ko', 'name': '...', 'english_name': '...', 'data': {'title': '...', 'overview': '...', 'tagline': '...', 'homepage': '...', 'runtime': 0}}]",
    )
    # Phase D: external_ids raw dict 전체 저장 (기존 5개만 → 모든 외부 ID 보존)
    external_ids: dict = Field(
        default_factory=dict,
        description="외부 ID 원본 전체 (TMDB 응답 그대로) {'imdb_id': 'tt...', 'facebook_id': '...', 'instagram_id': '...', 'twitter_id': '...', 'wikidata_id': 'Q...', ...}",
    )
    # lists: 이 영화가 포함된 TMDB 사용자 리스트
    lists: dict = Field(
        default_factory=dict,
        description="포함된 리스트 {'total_results': 100, 'results': [{'id': 1, 'name': '...'}]}",
    )


class KOBISRawMovie(BaseModel):
    """
    KOBIS API에서 수집한 원본 한국 영화 데이터.

    영화 목록 API (/movie/searchMovieList) + 상세정보 API (/movie/searchMovieInfo) 응답을 파싱.
    목록 API에서 기본 정보를 수집하고, 상세 API에서 배우/스태프/관람등급 등을 보강한다.
    """

    # ── 기본 정보 (목록 + 상세 공통) ──
    movie_cd: str = Field(..., description="KOBIS 영화 코드")
    movie_nm: str = Field(default="", description="한국어 영화명")
    movie_nm_en: str = Field(default="", description="영문 영화명")
    movie_nm_og: str = Field(default="", description="원제 (상세 API)")
    open_dt: str = Field(default="", description="개봉일 (YYYYMMDD)")
    prdt_year: str = Field(default="", description="제작년도 (YYYY)")
    show_tm: str = Field(default="", description="상영시간 (분, 상세 API)")
    type_nm: str = Field(default="", description="영화유형 ('장편', '단편', '애니메이션' 등)")
    prdt_stat_nm: str = Field(default="", description="제작상태 ('개봉', '개봉예정' 등)")
    genre_alt: str = Field(default="", description="장르 (쉼표 구분, 목록 API)")
    nation_alt: str = Field(default="", description="제작국가 (쉼표 구분, 목록 API)")
    rep_nation_nm: str = Field(default="", description="대표 제작국가명")
    rep_genre_nm: str = Field(default="", description="대표 장르명")

    # ── 상세 정보 (searchMovieInfo) ──
    nations: list[dict] = Field(
        default_factory=list, description="제작국가 [{'nationNm': '한국'}]",
    )
    genres: list[dict] = Field(
        default_factory=list, description="장르 [{'genreNm': 'SF'}]",
    )
    directors: list[dict] = Field(
        default_factory=list,
        description="감독 [{'peopleNm': '봉준호', 'peopleNmEn': 'BONG Joon-ho'}]",
    )
    actors: list[dict] = Field(
        default_factory=list,
        description="배우 [{'peopleNm': '송강호', 'peopleNmEn': '', 'cast': '기택'}]",
    )
    audits: list[dict] = Field(
        default_factory=list,
        description="심의 [{'auditNo': '2019-MF...', 'watchGradeNm': '15세이상관람가'}]",
    )
    companys: list[dict] = Field(
        default_factory=list,
        description="회사 [{'companyCd': '', 'companyNm': 'CJ ENM', 'companyPartNm': '배급사'}]",
    )
    staffs: list[dict] = Field(
        default_factory=list,
        description="스태프 [{'peopleNm': '한스 짐머', 'staffRoleNm': '음악'}]",
    )
    show_types: list[dict] = Field(
        default_factory=list,
        description="상영형태 [{'showTypeGroupNm': '일반', 'showTypeNm': '2D'}]",
    )

    # ── 박스오피스 보강 (별도 API에서 누적) ──
    audi_acc: int = Field(default=0, description="누적관객수 (박스오피스 API)")
    sales_acc: int = Field(default=0, description="누적매출액 (KRW, 박스오피스 API)")
    scrn_cnt: int = Field(default=0, description="최대 스크린수 (박스오피스 API)")
    watch_grade_nm: str = Field(default="", description="관람등급 (audits에서 추출)")

    # ── 상세 정보 수집 여부 플래그 ──
    detail_fetched: bool = Field(default=False, description="상세 API 호출 완료 여부")


class KOBISBoxOffice(BaseModel):
    """KOBIS 박스오피스 데이터 (트렌딩 분석용)."""

    movie_cd: str
    movie_nm: str = ""
    rank: int = 0
    rank_inten: int = Field(default=0, description="전일 대비 순위 변동")
    rank_old_and_new: str = Field(default="OLD", description="'NEW' or 'OLD'")
    audi_cnt: int = Field(default=0, description="해당일 관객수")
    audi_acc: int = Field(default=0, description="누적 관객수")
    sales_amt: int = Field(default=0, description="해당일 매출액 (KRW)")
    sales_acc: int = Field(default=0, description="누적 매출액 (KRW)")
    scrn_cnt: int = Field(default=0, description="상영 스크린 수")
    show_cnt: int = Field(default=0, description="상영 횟수")
    open_dt: str = ""


class KMDbRawMovie(BaseModel):
    """
    KMDb API에서 수집한 원본 한국 영화 데이터.

    KMDb 영화상세정보 API 응답(Data[0].Result[i])을 파싱한 구조.
    title 필드에 포함되는 !HS/!HE 하이라이트 마크업은 수집기에서 제거한다.
    posters/stlls 필드는 파이프(|) 구분 문자열이며, 수집기에서 리스트로 변환한다.
    """

    # ── 식별자 ──
    doc_id: str = Field(default="", description="KMDb 문서 고유 ID (DOCID)")
    movie_id: str = Field(default="", description="KMDb 영화 등록 ID (movieId)")
    movie_seq: str = Field(default="", description="KMDb 영화 등록 SEQ (movieSeq)")

    # ── 기본 정보 ──
    title: str = Field(default="", description="한국어 제목 (!HS/!HE 제거 후)")
    title_eng: str = Field(default="", description="영문 제목")
    title_org: str = Field(default="", description="원제")
    prod_year: str = Field(default="", description="제작년도 (YYYY)")
    nation: str = Field(default="", description="제작국가")
    company: str = Field(default="", description="제작사")
    runtime: str = Field(default="", description="상영시간 (분)")
    genre: str = Field(default="", description="장르 (쉼표 구분)")
    rating: str = Field(default="", description="관람등급")
    type_name: str = Field(default="", description="유형 (극영화/애니메이션/다큐멘터리 등)")
    use: str = Field(default="", description="용도 구분")
    keywords: str = Field(default="", description="키워드 (쉼표 구분)")

    # ── 개봉 정보 ──
    release_date: str = Field(default="", description="개봉일 (YYYYMMDD)")
    rep_rls_date: str = Field(default="", description="대표 개봉일")

    # ── 줄거리 ──
    plots: list[dict] = Field(
        default_factory=list,
        description="줄거리 배열 [{'plotLang': '한국어', 'plotText': '...'}, ...]",
    )

    # ── 인물 정보 ──
    directors: list[dict] = Field(
        default_factory=list,
        description="감독 배열 [{'directorNm': '봉준호', 'directorEnNm': 'BONG Joon-ho', 'directorId': '...'}, ...]",
    )
    actors: list[dict] = Field(
        default_factory=list,
        description="배우 배열 [{'actorNm': '송강호', 'actorEnNm': '...', 'actorId': '...'}, ...]",
    )
    staffs: list[dict] = Field(
        default_factory=list,
        description="스태프 배열 [{'staffNm': '...', 'staffRoleGroup': '촬영', 'staffRole': '...'}, ...]",
    )

    # ── 미디어 ──
    posters: list[str] = Field(default_factory=list, description="포스터 URL 목록 (파이프 분리 후)")
    stills: list[str] = Field(default_factory=list, description="스틸컷 URL 목록 (파이프 분리 후)")
    vods: list[dict] = Field(
        default_factory=list,
        description="VOD 배열 [{'vodClass': '예고편', 'vodUrl': '...'}, ...]",
    )

    # ── 수상/흥행 ──
    awards1: str = Field(default="", description="영화제 수상내역")
    awards2: str = Field(default="", description="수상내역 기타")
    sales_acc: str = Field(default="", description="누적매출액")
    audi_acc: str = Field(default="", description="누적관람인원")

    # ── 촬영/음악 ──
    f_location: str = Field(default="", description="촬영장소")
    theme_song: str = Field(default="", description="주제곡")
    soundtrack_field: str = Field(default="", description="삽입곡")

    # ── 메타 ──
    kmdb_url: str = Field(default="", description="KMDb 상세페이지 URL")
    codes: list[dict] = Field(default_factory=list, description="코드 배열")


class PipelineState(BaseModel):
    """
    배치 파이프라인 진행 상태 (data/pipeline_state.json).

    §11-9: 중단점 재개를 위해 마지막 처리 지점을 기록한다.
    """

    last_movie_id: str = Field(default="", description="마지막 처리 완료한 TMDB ID")
    current_step: str = Field(default="", description="현재 진행 단계 (collect/preprocess/embed/load)")
    total_collected: int = Field(default=0, description="수집 완료 영화 수")
    total_processed: int = Field(default=0, description="전처리 완료 영화 수")
    total_loaded: int = Field(default=0, description="적재 완료 영화 수")
    failed_ids: list[str] = Field(default_factory=list, description="처리 실패한 TMDB ID 목록")
    timestamp: str = Field(default="", description="마지막 업데이트 시각 (ISO 8601)")
