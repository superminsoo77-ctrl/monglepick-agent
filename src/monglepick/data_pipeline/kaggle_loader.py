"""
Kaggle The Movies Dataset 로더.

§11-4 Kaggle 데이터 로딩 명세:
- movies_metadata.csv: TMDB ID 정규화, 결측치 처리 → 보강용 DataFrame
- ratings.csv: userId·movieId·rating·timestamp → CF 매트릭스 입력
- credits.csv: JSON 파싱 (cast/crew) → 감독·배우 추출
- keywords.csv: JSON 파싱 → 키워드 목록
- links.csv: movieId → tmdbId → imdbId 매핑 테이블

§11-2: MovieLens movieId ↔ TMDB tmdbId 매핑은 links.csv로 수행.
매핑 실패 레코드는 제외하고 로그 기록 (~5% 예상).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import structlog

logger = structlog.get_logger()

# 기본 Kaggle 데이터 경로
DEFAULT_DATA_DIR = Path("data/kaggle_movies")


class KaggleLoader:
    """
    Kaggle The Movies Dataset CSV 파일 로더.

    사용 예:
        loader = KaggleLoader("data/kaggle_movies")
        id_map = loader.load_links()
        metadata_df = loader.load_movies_metadata()
        ratings_df = loader.load_ratings(id_map)
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        """
        Args:
            data_dir: Kaggle CSV 파일들이 위치한 디렉토리 경로
        """
        self.data_dir = Path(data_dir)

    def _path(self, filename: str) -> Path:
        """데이터 디렉토리 내 파일의 전체 경로를 반환한다."""
        return self.data_dir / filename

    # ── links.csv: ID 매핑 테이블 ──

    def load_links(self) -> dict[int, int]:
        """
        MovieLens movieId → TMDB tmdbId 매핑 딕셔너리를 생성한다.

        §11-2: links.csv의 movieId를 키로 tmdbId를 조회.
        tmdbId가 NaN이거나 숫자가 아닌 경우 제외.

        Returns:
            dict[int, int]: {movieLens_id: tmdb_id}
        """
        df = pd.read_csv(self._path("links.csv"))

        # tmdbId가 유효한 숫자인 행만 사용
        df = df.dropna(subset=["tmdbId"])
        df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce")
        df = df.dropna(subset=["tmdbId"])

        id_map = dict(zip(df["movieId"].astype(int), df["tmdbId"].astype(int)))
        logger.info("kaggle_links_loaded", count=len(id_map))
        return id_map

    # ── movies_metadata.csv: 영화 메타데이터 ──

    def load_movies_metadata(self) -> pd.DataFrame:
        """
        영화 메타데이터를 로드하고 정규화한다.

        §11-4: TMDB ID 정규화 (숫자가 아닌 id 제외), 결측치 처리.

        Returns:
            DataFrame: id(int), title, overview, genres(list), release_date, vote_average, popularity 등
        """
        df = pd.read_csv(
            self._path("movies_metadata.csv"),
            low_memory=False,
        )

        # TMDB ID 정규화: 숫자가 아닌 id 제외
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df = df.dropna(subset=["id"])
        df["id"] = df["id"].astype(int)

        # 중복 TMDB ID 제거 (첫 번째 유지)
        df = df.drop_duplicates(subset=["id"], keep="first")

        # genres 컬럼 파싱 (문자열 → 리스트)
        df["genres_parsed"] = df["genres"].apply(self._parse_json_column)

        # 결측치 처리
        df["overview"] = df["overview"].fillna("")
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(0).astype(int)

        # Phase B: 추가 컬럼 파싱 (재무/텍스트/분류 정보)
        df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0).astype(int)
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0).astype(int)
        df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)
        df["tagline"] = df["tagline"].fillna("") if "tagline" in df.columns else ""
        df["homepage"] = df["homepage"].fillna("") if "homepage" in df.columns else ""
        df["original_language"] = df["original_language"].fillna("") if "original_language" in df.columns else ""
        df["imdb_id"] = df["imdb_id"].fillna("") if "imdb_id" in df.columns else ""
        df["status"] = df["status"].fillna("") if "status" in df.columns else ""
        # adult 컬럼은 "True"/"False" 문자열 → bool 변환
        if "adult" in df.columns:
            df["adult"] = df["adult"].map(
                {"True": True, "False": False, True: True, False: False}
            ).fillna(False)
        else:
            df["adult"] = False

        # Phase C: video 컬럼 (직접 비디오/홈 비디오 플래그)
        if "video" in df.columns:
            df["video_flag"] = df["video"].map(
                {"True": True, "False": False, True: True, False: False}
            ).fillna(False)
        else:
            df["video_flag"] = False

        # Phase B: JSON 컬럼 파싱 (컬렉션/제작사/국가/언어)
        if "belongs_to_collection" in df.columns:
            df["collection_parsed"] = df["belongs_to_collection"].apply(self._parse_json_single)
        else:
            df["collection_parsed"] = None
        if "production_companies" in df.columns:
            df["production_companies_parsed"] = df["production_companies"].apply(self._parse_json_column)
        else:
            df["production_companies_parsed"] = [[] for _ in range(len(df))]
        if "production_countries" in df.columns:
            df["production_countries_parsed"] = df["production_countries"].apply(self._parse_json_column)
        else:
            df["production_countries_parsed"] = [[] for _ in range(len(df))]
        if "spoken_languages" in df.columns:
            df["spoken_languages_parsed"] = df["spoken_languages"].apply(self._parse_json_column)
        else:
            df["spoken_languages_parsed"] = [[] for _ in range(len(df))]

        logger.info("kaggle_metadata_loaded", count=len(df))
        return df

    # ── ratings.csv: 유저-영화 평점 ──

    def load_ratings(self, id_map: dict[int, int] | None = None) -> pd.DataFrame:
        """
        평점 데이터를 로드하고 TMDB ID로 매핑한다.

        §11-4: userId·movieId·rating·timestamp 로드, links.csv로 TMDB ID 매핑.
        매핑 실패 레코드는 제외.

        Args:
            id_map: MovieLens movieId → TMDB tmdbId 매핑. None이면 자동 로드.

        Returns:
            DataFrame: userId, tmdbId, rating, timestamp
        """
        if id_map is None:
            id_map = self.load_links()

        df = pd.read_csv(self._path("ratings.csv"))

        # MovieLens movieId → TMDB tmdbId 매핑
        df["tmdbId"] = df["movieId"].map(id_map)
        original_count = len(df)

        # 매핑 실패 레코드 제외
        df = df.dropna(subset=["tmdbId"])
        df["tmdbId"] = df["tmdbId"].astype(int)

        dropped = original_count - len(df)
        logger.info(
            "kaggle_ratings_loaded",
            total=len(df),
            dropped=dropped,
            drop_rate=f"{dropped / original_count * 100:.1f}%",
        )
        return df[["userId", "tmdbId", "rating", "timestamp"]]

    # ── credits.csv: 감독/배우 정보 ──

    def load_credits(self) -> pd.DataFrame:
        """
        크레딧 데이터에서 감독, 배우, 촬영감독, 작곡가, 각본가, 프로듀서, 편집자를 추출한다.

        §11-4: JSON 파싱 (cast/crew 컬럼).
        Phase B 확장: crew에서 추가 직군 (촬영감독/작곡가/각본가/프로듀서/편집자) 추출.

        Returns:
            DataFrame: id, director, cast_names, cast_characters,
                       cinematographer, composer, screenwriters, producers, editor
        """
        df = pd.read_csv(self._path("credits.csv"))

        # id 정규화
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df = df.dropna(subset=["id"])
        df["id"] = df["id"].astype(int)

        # crew에서 감독 추출
        df["director"] = df["crew"].apply(self._extract_director)

        # cast에서 상위 5명 배우 추출
        df["cast_names"] = df["cast"].apply(lambda x: self._extract_cast(x, top_n=5))

        # Phase B: 캐릭터명-배우명 매핑 (상위 5명)
        df["cast_characters"] = df["cast"].apply(self._extract_cast_with_characters)

        # Phase B: 확장 크루 추출
        df["cinematographer"] = df["crew"].apply(
            lambda x: self._extract_crew_by_job(x, "Director of Photography")
        )
        df["composer"] = df["crew"].apply(
            lambda x: self._extract_crew_by_job(x, "Original Music Composer")
        )
        df["screenwriters"] = df["crew"].apply(
            lambda x: self._extract_crew_list_by_jobs(x, ["Screenplay", "Writer"])
        )
        df["producers"] = df["crew"].apply(
            lambda x: self._extract_crew_list_by_jobs(x, ["Producer"])
        )
        df["editor"] = df["crew"].apply(
            lambda x: self._extract_crew_by_job(x, "Editor")
        )

        # Phase C: 추가 크루 직군 추출
        df["executive_producers"] = df["crew"].apply(
            lambda x: self._extract_crew_list_by_jobs(x, ["Executive Producer"])
        )
        df["production_designer"] = df["crew"].apply(
            lambda x: self._extract_crew_by_job(x, "Production Design")
        )
        df["costume_designer"] = df["crew"].apply(
            lambda x: self._extract_crew_by_jobs(x, ["Costume Design", "Costume Designer"])
        )
        df["source_author"] = df["crew"].apply(
            lambda x: self._extract_crew_by_jobs(x, ["Novel", "Characters", "Original Story", "Story"])
        )

        # Phase C: 감독 상세 정보 (ID, 프로필 사진)
        df["director_details"] = df["crew"].apply(self._extract_director_details)

        logger.info("kaggle_credits_loaded", count=len(df))
        return df[["id", "director", "cast_names", "cast_characters",
                    "cinematographer", "composer", "screenwriters", "producers", "editor",
                    "executive_producers", "production_designer", "costume_designer",
                    "source_author", "director_details"]]

    # ── keywords.csv: 키워드 목록 ──

    def load_keywords(self) -> pd.DataFrame:
        """
        키워드 데이터를 파싱한다.

        §11-4: JSON 파싱 (keywords 컬럼), 키워드 목록 추출.

        Returns:
            DataFrame: id(int), keywords(list[str])
        """
        df = pd.read_csv(self._path("keywords.csv"))

        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df = df.dropna(subset=["id"])
        df["id"] = df["id"].astype(int)

        df["keywords_list"] = df["keywords"].apply(
            lambda x: [item["name"] for item in self._parse_json_column(x)]
        )

        logger.info("kaggle_keywords_loaded", count=len(df))
        return df[["id", "keywords_list"]]

    # ── JSON 파싱 헬퍼 ──

    @staticmethod
    def _parse_json_column(value: str) -> list[dict]:
        """문자열로 저장된 JSON/Python 리터럴을 파싱한다."""
        if pd.isna(value) or not value:
            return []
        try:
            return json.loads(value.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []

    @staticmethod
    def _extract_director(crew_str: str) -> str:
        """crew JSON에서 job='Director'인 사람의 이름을 추출한다."""
        crew = KaggleLoader._parse_json_column(crew_str)
        for person in crew:
            if person.get("job") == "Director":
                return person.get("name", "")
        return ""

    @staticmethod
    def _extract_cast(cast_str: str, top_n: int = 5) -> list[str]:
        """cast JSON에서 상위 N명의 배우 이름을 추출한다."""
        cast = KaggleLoader._parse_json_column(cast_str)
        return [person.get("name", "") for person in cast[:top_n] if person.get("name")]

    @staticmethod
    def _extract_cast_with_characters(cast_str: str, top_n: int = 5) -> list[dict]:
        """
        cast JSON에서 상위 N명 배우의 상세 정보를 추출한다.

        Phase C 확장: id, profile_path, gender 추가.
        Kaggle 데이터에는 popularity/original_name이 없으므로 기본값으로 채운다.

        Args:
            cast_str: JSON/Python 리터럴 형태의 cast 문자열
            top_n: 추출할 최대 배우 수 (기본 5)

        Returns:
            배우 상세 정보 딕셔너리 리스트
        """
        cast = KaggleLoader._parse_json_column(cast_str)
        return [
            {
                "id": p.get("id", 0),
                "name": p.get("name", ""),
                "character": p.get("character", ""),
                "profile_path": p.get("profile_path") or "",
                "gender": p.get("gender", 0),
                "popularity": 0.0,  # Kaggle에는 인기도 없음
                "original_name": "",  # Kaggle에는 원어 이름 없음
                "order": p.get("order", i),
            }
            for i, p in enumerate(cast[:top_n])
            if p.get("name")
        ]

    @staticmethod
    def _extract_crew_by_job(crew_str: str, job: str) -> str:
        """
        crew JSON에서 특정 job의 첫 번째 사람 이름을 추출한다.

        Args:
            crew_str: JSON/Python 리터럴 형태의 crew 문자열
            job: 추출할 직군명 (예: "Director of Photography")

        Returns:
            매칭된 사람 이름 또는 빈 문자열
        """
        crew = KaggleLoader._parse_json_column(crew_str)
        for person in crew:
            if person.get("job") == job:
                return person.get("name", "")
        return ""

    @staticmethod
    def _extract_crew_list_by_jobs(crew_str: str, jobs: list[str]) -> list[str]:
        """
        crew JSON에서 특정 job 목록에 해당하는 모든 사람 이름을 추출한다.

        Args:
            crew_str: JSON/Python 리터럴 형태의 crew 문자열
            jobs: 추출할 직군명 리스트 (예: ["Screenplay", "Writer"])

        Returns:
            중복 제거된 이름 리스트
        """
        crew = KaggleLoader._parse_json_column(crew_str)
        names: list[str] = []
        for person in crew:
            if person.get("job") in jobs and person.get("name"):
                name = person["name"]
                if name not in names:
                    names.append(name)
        return names

    @staticmethod
    def _extract_crew_by_jobs(crew_str: str, jobs: list[str]) -> str:
        """
        crew JSON에서 특정 job 목록 중 첫 번째 매칭하는 사람 이름을 추출한다.

        Args:
            crew_str: JSON/Python 리터럴 형태의 crew 문자열
            jobs: 우선순위별 직군명 리스트 (예: ["Costume Design", "Costume Designer"])

        Returns:
            첫 번째 매칭된 사람 이름 또는 빈 문자열
        """
        crew = KaggleLoader._parse_json_column(crew_str)
        for person in crew:
            if person.get("job") in jobs and person.get("name"):
                return person["name"]
        return ""

    @staticmethod
    def _extract_director_details(crew_str: str) -> dict:
        """
        crew JSON에서 감독의 상세 정보를 추출한다.

        Args:
            crew_str: JSON/Python 리터럴 형태의 crew 문자열

        Returns:
            {"id": int, "profile_path": str, "original_name": str}
            (Kaggle에는 original_name이 없으므로 빈 문자열)
        """
        crew = KaggleLoader._parse_json_column(crew_str)
        for person in crew:
            if person.get("job") == "Director":
                return {
                    "id": person.get("id", 0),
                    "profile_path": person.get("profile_path") or "",
                    "original_name": "",  # Kaggle에는 original_name 없음
                }
        return {"id": 0, "profile_path": "", "original_name": ""}

    @staticmethod
    def _parse_json_single(value: str) -> dict | None:
        """
        문자열로 저장된 단일 JSON 객체를 파싱한다.

        belongs_to_collection 컬럼 전용. json.loads 실패 시 ast.literal_eval로 재시도한다.

        Args:
            value: JSON/Python 리터럴 형태의 문자열

        Returns:
            파싱된 딕셔너리 또는 None (파싱 실패 시)
        """
        if pd.isna(value) or not value:
            return None
        try:
            return json.loads(value.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            try:
                result = ast.literal_eval(value)
                return result if isinstance(result, dict) else None
            except (ValueError, SyntaxError):
                return None
