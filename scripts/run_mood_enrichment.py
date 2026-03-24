"""
무드태그 보강 스크립트 (Upstage Solar / OpenAI 호환 API).

기존 DB에 적재된 영화의 무드태그를 LLM으로 정밀 분석하여 갱신한다.
장르 기반 fallback(단순 매핑) 대신, 제목+장르+키워드+줄거리를 종합 분석하여
25개 무드태그 중 3~5개를 선택한다.

기본 Provider: Upstage Solar Pro 3 (OpenAI 호환 API)

비용 추정 (Solar Pro 3, ~910,000건, 배치 10건/요청):
    - API 호출: ~91,000회
    - 입력: ~86.5M 토큰 × $0.15/1M = ~$13
    - 출력: ~18.2M 토큰 × $0.60/1M = ~$11
    - 총합: ~$24 (약 3.3만원)

사용법:
    # Upstage Solar Pro 3으로 실행 (기본, .env의 UPSTAGE_API_KEY 사용)
    PYTHONPATH=src uv run python scripts/run_mood_enrichment.py

    # API 키 직접 지정
    PYTHONPATH=src uv run python scripts/run_mood_enrichment.py --api-key up_xxx

    # 이전 중단점에서 재개 (API 키 교체 가능)
    PYTHONPATH=src uv run python scripts/run_mood_enrichment.py --resume
    PYTHONPATH=src uv run python scripts/run_mood_enrichment.py --resume --api-key up_새키

    # OpenAI GPT-4o-mini로 실행
    PYTHONPATH=src uv run python scripts/run_mood_enrichment.py --provider openai --api-key sk-xxx

    # 비용 추정만 (API 호출 없이 예상 비용 출력)
    PYTHONPATH=src uv run python scripts/run_mood_enrichment.py --estimate-only

    # JSONL에서 읽기 (DB 미적재 상태에서도 사용 가능)
    PYTHONPATH=src uv run python scripts/run_mood_enrichment.py --source jsonl

소요 시간 추정:
    - Upstage Solar Pro 3 (RPM ~100): ~2~4시간
    - OpenAI Tier 1 (500 RPM), concurrency=10: ~4시간
    - OpenAI Tier 2 (5,000 RPM), concurrency=50: ~30분
    - DB 갱신: ~30분 (Qdrant + Neo4j + ES 동시)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 프로젝트 루트를 sys.path에 추가 + .env 로드 ──
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# .env 파일에서 환경변수 로드 (UPSTAGE_API_KEY 등)
_env_file = _project_root / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import structlog  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from monglepick.db.clients import (  # noqa: E402
    init_all_clients,
    close_all_clients,
    get_qdrant,
    get_neo4j,
    get_elasticsearch,
    ES_INDEX_NAME,
)

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

# 25개 유효 무드태그 화이트리스트 (§11-6 정의, preprocessor.py와 동일)
MOOD_WHITELIST: set[str] = {
    "몰입", "감동", "웅장", "긴장감", "힐링", "유쾌", "따뜻", "슬픔",
    "공포", "잔잔", "스릴", "카타르시스", "청춘", "우정", "가족애",
    "로맨틱", "미스터리", "반전", "철학적", "사회비판", "모험", "판타지",
    "레트로", "다크", "유머",
}

# 장르→무드 fallback 매핑 (API 실패 시 사용, preprocessor.py와 동일)
GENRE_TO_MOOD: dict[str, list[str]] = {
    "액션": ["몰입", "스릴"], "모험": ["모험", "몰입"], "애니메이션": ["따뜻", "판타지"],
    "코미디": ["유쾌", "유머"], "범죄": ["긴장감", "다크"], "다큐멘터리": ["철학적", "사회비판"],
    "드라마": ["감동", "잔잔"], "가족": ["가족애", "따뜻"], "판타지": ["판타지", "모험"],
    "역사": ["웅장", "감동"], "공포": ["공포", "다크"], "음악": ["감동", "힐링"],
    "미스터리": ["미스터리", "긴장감"], "로맨스": ["로맨틱", "따뜻"], "SF": ["몰입", "웅장"],
    "TV 영화": ["잔잔"], "스릴러": ["스릴", "긴장감"], "전쟁": ["웅장", "카타르시스"],
    "서부": ["모험", "레트로"],
}

# 파일 경로
CHECKPOINT_FILE = Path("data/mood_checkpoint.json")
DEFAULT_JSONL_PATH = Path("data/tmdb_full/tmdb_full_movies.jsonl")

# 기본 설정
DEFAULT_CONCURRENCY = 20
DEFAULT_BATCH_SIZE = 10  # 1회 API 호출당 영화 수
DEFAULT_RPM = 400         # 분당 최대 요청 수 (Tier 1: 500 RPM, 여유 확보)
MEGA_CHUNK_SIZE = 1000    # DB 갱신 및 체크포인트 저장 단위

# ── Provider별 기본 설정 ──
# Upstage Solar: OpenAI 호환 API, base_url만 변경하면 동일 AsyncOpenAI 클라이언트 사용 가능
PROVIDER_CONFIGS: dict[str, dict] = {
    "upstage": {
        "base_url": "https://api.upstage.ai/v1",
        "model": "solar-pro3",       # 102B MoE, 한국어 최적, 구조화 출력 지원
        "rpm": 30,                    # Upstage RPM (429 방지, 보수적)
        "concurrency": 3,
        "env_key": "UPSTAGE_API_KEY",
    },
    "openai": {
        "base_url": None,             # OpenAI 기본 URL
        "model": "gpt-4o-mini",
        "rpm": 400,
        "concurrency": 20,
        "env_key": "OPENAI_API_KEY",
    },
}

# ── LLM 프롬프트 ──
# 시스템 프롬프트: 무드태그 분석 전문가 역할 정의 + 25개 태그 목록 + 규칙
SYSTEM_PROMPT = """당신은 영화 분위기 분석 전문가입니다.
주어진 영화들의 분위기를 분석하여 각각 무드 태그를 3~5개 선택해주세요.

[사용 가능한 무드 태그 - 반드시 이 목록에서만 선택]
몰입, 감동, 웅장, 긴장감, 힐링, 유쾌, 따뜻, 슬픔, 공포, 잔잔,
스릴, 카타르시스, 청춘, 우정, 가족애, 로맨틱, 미스터리, 반전,
철학적, 사회비판, 모험, 판타지, 레트로, 다크, 유머

[규칙]
1. 반드시 위 25개 태그에서만 선택하세요
2. 각 영화당 3~5개를 선택하세요
3. 장르, 키워드, 줄거리를 종합적으로 분석하세요
4. JSON 객체로만 응답하세요 (다른 텍스트 금지)"""


# ══════════════════════════════════════════════════════════════
# RPM 제한기 (슬라이딩 윈도우)
# ══════════════════════════════════════════════════════════════

class RPMLimiter:
    """
    분당 요청 수(RPM)를 제한하는 슬라이딩 윈도우 제한기.

    asyncio.Semaphore와 함께 사용하여 동시 요청 수와 RPM을 동시에 제어한다.
    60초 윈도우 내의 요청 타임스탬프를 추적하고, 한도 초과 시 대기한다.
    """

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.timestamps: list[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """요청 슬롯을 확보한다. RPM 초과 시 대기."""
        async with self.lock:
            now = time.time()
            # 60초 이전 타임스탬프 제거 (슬라이딩 윈도우)
            self.timestamps = [t for t in self.timestamps if now - t < 60]

            if len(self.timestamps) >= self.rpm:
                # 가장 오래된 요청이 60초 경과할 때까지 대기
                wait_until = self.timestamps[0] + 60
                wait_time = wait_until - now
                if wait_time > 0:
                    logger.debug("rpm_limit_wait", wait_seconds=round(wait_time, 1))
                    await asyncio.sleep(wait_time)
                # 대기 후 만료된 타임스탬프 다시 제거
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < 60]

            self.timestamps.append(time.time())


# ══════════════════════════════════════════════════════════════
# 체크포인트 (중단 후 재개 지원)
# ══════════════════════════════════════════════════════════════

def load_checkpoint() -> dict:
    """체크포인트 파일을 로드한다. 없으면 빈 dict 반환."""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data: dict) -> None:
    """체크포인트를 파일에 저장한다."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════
# LLM API 호출
# ══════════════════════════════════════════════════════════════

def build_user_prompt(movies: list[dict]) -> str:
    """
    배치 영화 정보를 사용자 프롬프트로 구성한다.

    각 영화를 번호와 함께 한 줄로 요약하고,
    JSON 객체 형태의 응답을 요청한다.

    Args:
        movies: [{"movie_id": 123, "title": "...", "genres": [...], ...}, ...]

    Returns:
        사용자 프롬프트 문자열
    """
    lines = []
    for i, m in enumerate(movies, 1):
        title = m.get("title", "제목 없음")
        genres = ", ".join(m.get("genres", []))
        keywords = ", ".join(m.get("keywords", [])[:10])
        # 줄거리는 200자로 제한 (토큰 절약)
        overview = (m.get("overview") or "")[:200]
        lines.append(
            f"영화 {i}: {title} | 장르: {genres} | 키워드: {keywords} | 줄거리: {overview}"
        )

    movies_block = "\n".join(lines)
    return (
        f"{movies_block}\n\n"
        f"위 {len(movies)}개 영화의 무드 태그를 JSON 객체로 응답하세요.\n"
        f'키는 영화 번호(문자열), 값은 무드 태그 배열.\n'
        f'예: {{"1": ["몰입", "감동"], "2": ["유쾌", "따뜻"]}}'
    )


async def generate_mood_batch(
    client: AsyncOpenAI,
    movies: list[dict],
    model: str,
    rpm_limiter: RPMLimiter,
) -> dict[str, list[str]]:
    """
    LLM으로 배치 영화의 무드태그를 생성한다.

    N개 영화를 하나의 API 호출로 처리하여 비용을 최소화한다.
    response_format=json_object를 시도하고, 미지원 모델이면 텍스트에서 JSON 추출한다.
    실패 시 3회 재시도 후 장르 기반 fallback을 사용한다.

    Args:
        client: AsyncOpenAI 클라이언트 (Upstage/OpenAI 호환)
        movies: [{"movie_id": 123, "title": "...", "genres": [...], ...}, ...]
        model: 사용할 모델명 (solar-pro3, gpt-4o-mini 등)
        rpm_limiter: RPM 제한기

    Returns:
        {doc_id(str): ["몰입", "감동", ...], ...}
    """
    import re

    user_prompt = build_user_prompt(movies)

    for attempt in range(5):  # 최대 5회 시도 (429 대응)
        try:
            # RPM 제한 적용
            await rpm_limiter.acquire()

            # API 호출 — response_format=json_object 시도, 미지원 시 일반 호출
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )
            except Exception as inner_e:
                # response_format 미지원 에러만 fallback (429 등은 외부 except에서 처리)
                if "response_format" in str(inner_e).lower() or "json" in str(inner_e).lower():
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.3,
                        max_tokens=500,
                    )
                else:
                    raise  # 429 등 다른 에러는 외부 except로 전파

            content = response.choices[0].message.content or "{}"

            # JSON 추출: 텍스트에 JSON이 포함된 경우 { } 블록만 추출
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
            if json_match:
                content = json_match.group()
            parsed = json.loads(content)

            # 결과 매핑: 영화 번호(1-indexed) → movie_id (문자열 ID)
            result: dict[str, list[str]] = {}
            for i, m in enumerate(movies, 1):
                tags = parsed.get(str(i), [])
                # 화이트리스트 필터링 (유효하지 않은 태그 제거)
                valid_tags = [t for t in tags if t in MOOD_WHITELIST]
                if valid_tags:
                    result[m["movie_id"]] = valid_tags[:5]
                else:
                    # LLM이 유효 태그를 반환하지 못한 경우 → fallback
                    result[m["movie_id"]] = _genre_fallback(m.get("genres", []))

            return result

        except json.JSONDecodeError as e:
            logger.warning("mood_json_parse_error", attempt=attempt + 1, error=str(e))
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "too_many_requests" in error_str
            if is_rate_limit:
                # 429 Rate Limit → 긴 대기 (30s, 60s, 90s, 120s, 150s)
                wait = 30 * (attempt + 1)
                logger.warning("rate_limit_backoff", attempt=attempt + 1, wait_seconds=wait)
                await asyncio.sleep(wait)
            else:
                logger.warning("mood_api_error", attempt=attempt + 1, error=error_str)
                if attempt < 4:
                    await asyncio.sleep(2 ** (attempt + 1))

    # 모든 시도 실패 → 장르 기반 fallback
    logger.warning("mood_batch_all_attempts_failed", movie_ids=[m["movie_id"] for m in movies])
    return {
        m["movie_id"]: _genre_fallback(m.get("genres", []))
        for m in movies
    }


def _genre_fallback(genres: list[str]) -> list[str]:
    """장르 기반 기본 무드태그 (API 실패 시 fallback)."""
    moods: set[str] = set()
    for g in genres:
        moods.update(GENRE_TO_MOOD.get(g, []))
    return list(moods)[:5] if moods else ["잔잔"]


# ══════════════════════════════════════════════════════════════
# DB 갱신 (Qdrant, Neo4j, Elasticsearch)
# ══════════════════════════════════════════════════════════════

async def update_qdrant_moods(updates: dict[str, list[str]]) -> int:
    """
    Qdrant payload의 mood_tags 필드를 갱신한다.

    set_payload()로 해당 필드만 업데이트하며, 다른 payload 필드는 유지된다.
    Qdrant point ID는 int (TMDB 숫자 ID) 또는 UUID5 (KOBIS 영문 코드) 형식이다.
    MovieDocument.id (문자열)를 qdrant_loader._to_point_id()와 동일하게 변환한다.

    Args:
        updates: {doc_id(str): ["몰입", "감동", ...], ...}

    Returns:
        성공적으로 갱신된 건수
    """
    qdrant = await get_qdrant()
    updated = 0
    for doc_id, mood_tags in updates.items():
        try:
            # Qdrant point ID 변환: 숫자 문자열 → int, 그 외 → UUID5
            # (qdrant_loader._to_point_id와 동일 로직)
            if str(doc_id).isdigit():
                point_id: int | str = int(doc_id)
            else:
                import uuid
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"kobis:{doc_id}"))

            await qdrant.set_payload(
                collection_name="movies",
                payload={"mood_tags": mood_tags},
                points=[point_id],
            )
            updated += 1
        except Exception as e:
            logger.debug("qdrant_mood_update_skip", doc_id=doc_id, error=str(e))
    return updated


async def update_neo4j_moods(updates: dict[str, list[str]]) -> int:
    """
    Neo4j의 HAS_MOOD 관계를 갱신한다.

    1단계: 기존 HAS_MOOD 관계 삭제 (해당 영화에 대해서만)
    2단계: 새 MoodTag 노드 MERGE + HAS_MOOD 관계 생성

    ⚠️ Neo4j Movie 노드 식별자는 `id` 속성이다 (MovieDocument.id, 문자열).
       neo4j_loader.py: MERGE (movie:Movie {id: m.id})

    500건씩 배치 처리하여 Neo4j 트랜잭션 크기를 제한한다.

    Args:
        updates: {doc_id(str): ["몰입", "감동", ...], ...}

    Returns:
        성공적으로 갱신된 건수
    """
    driver = await get_neo4j()

    # Cypher 1: 기존 HAS_MOOD 관계 삭제
    # ⚠️ Movie 노드 식별자는 `id` 속성 (neo4j_loader.py: MERGE (movie:Movie {id: m.id}))
    delete_query = """
    UNWIND $movie_ids AS mid
    MATCH (m:Movie {id: mid})-[r:HAS_MOOD]->()
    DELETE r
    """

    # Cypher 2: 새 HAS_MOOD 관계 생성
    create_query = """
    UNWIND $updates AS u
    MATCH (m:Movie {id: u.doc_id})
    UNWIND u.mood_tags AS tag
    MERGE (mt:MoodTag {name: tag})
    MERGE (m)-[:HAS_MOOD]->(mt)
    """

    params = [
        {"doc_id": mid, "mood_tags": tags}
        for mid, tags in updates.items()
    ]
    movie_ids = list(updates.keys())

    # 500건씩 배치 처리
    batch_size = 500
    updated = 0
    async with driver.session() as session:
        for i in range(0, len(params), batch_size):
            batch_params = params[i:i + batch_size]
            batch_ids = movie_ids[i:i + batch_size]
            try:
                # 1단계: 기존 관계 삭제
                await session.run(delete_query, movie_ids=batch_ids)
                # 2단계: 새 관계 생성
                await session.run(create_query, updates=batch_params)
                updated += len(batch_params)
            except Exception as e:
                logger.warning(
                    "neo4j_mood_update_failed",
                    batch_start=i,
                    batch_size=len(batch_params),
                    error=str(e),
                )
    return updated


async def update_es_moods(updates: dict[str, list[str]]) -> int:
    """
    Elasticsearch 문서의 mood_tags 필드를 갱신한다.

    _bulk API의 update 액션으로 부분 업데이트를 수행한다.
    다른 필드는 영향받지 않으며, mood_tags 필드만 교체된다.

    Args:
        updates: {movie_id: ["몰입", "감동", ...], ...}

    Returns:
        성공적으로 갱신된 건수
    """
    from elasticsearch.helpers import async_bulk as es_async_bulk

    es = await get_elasticsearch()

    # bulk update 액션 생성
    actions = [
        {
            "_op_type": "update",
            "_index": ES_INDEX_NAME,
            "_id": str(movie_id),
            "doc": {"mood_tags": mood_tags},
        }
        for movie_id, mood_tags in updates.items()
    ]

    if not actions:
        return 0

    try:
        success_count, errors = await es_async_bulk(
            es,
            actions,
            raise_on_error=False,
            stats_only=True,
        )
        if errors:
            logger.warning("es_mood_update_partial_failure", errors=errors)
        return success_count
    except Exception as e:
        logger.warning("es_mood_update_failed", error=str(e))
        return 0


# ══════════════════════════════════════════════════════════════
# 데이터 소스 (ES 또는 JSONL)
# ══════════════════════════════════════════════════════════════

async def read_movies_from_es() -> list[dict]:
    """
    Elasticsearch에서 전체 영화 목록을 읽어온다 (scroll API).

    id 기준 오름차순으로 정렬하여 일관된 순서를 보장한다.
    필요한 필드만 가져와 메모리를 절약한다.

    ⚠️ ES 문서 필드명 주의:
       - `id`: MovieDocument.id (문자열, TMDB ID), _id와 동일
       - `keywords`: 공백 구분 문자열 (es_loader에서 " ".join(doc.keywords))
       - `genres`: 문자열 리스트 (한국어)
       - ES에는 `movie_id` 필드가 없음 (es_loader.py 참조)

    Returns:
        [{"movie_id": "157336", "title": "...", "genres": [...], ...}, ...]
    """
    es = await get_elasticsearch()
    movies: list[dict] = []

    # 초기 검색 (scroll 시작)
    # ES 문서의 필드명은 `id` (MovieDocument.id), `movie_id`는 존재하지 않음
    response = await es.search(
        index=ES_INDEX_NAME,
        body={
            "query": {"match_all": {}},
            "sort": [{"_doc": "asc"}],  # 내부 순서 (가장 빠른 scroll)
            "_source": ["id", "title", "genres", "keywords", "overview"],
            "size": 5000,
        },
        scroll="5m",
    )

    scroll_id = response["_scroll_id"]
    hits = response["hits"]["hits"]

    while hits:
        for hit in hits:
            src = hit["_source"]
            # keywords: ES에서는 공백 구분 문자열로 저장됨 → 리스트로 변환
            raw_keywords = src.get("keywords", "")
            if isinstance(raw_keywords, str):
                keywords = raw_keywords.split() if raw_keywords else []
            else:
                keywords = raw_keywords  # 이미 리스트인 경우

            movies.append({
                "movie_id": src.get("id", hit["_id"]),  # MovieDocument.id (문자열)
                "title": src.get("title", ""),
                "genres": src.get("genres", []),
                "keywords": keywords,
                "overview": src.get("overview", ""),
            })

        # 다음 scroll 페이지
        response = await es.scroll(scroll_id=scroll_id, scroll="5m")
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]

    # scroll context 정리 (리소스 반환)
    try:
        await es.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    logger.info("movies_loaded_from_es", total=len(movies))
    return movies


def read_movies_from_jsonl(jsonl_path: Path) -> list[dict]:
    """
    JSONL 파일에서 영화 목록을 읽어온다.

    DB가 아직 적재되지 않은 상태에서도 사용 가능하다.
    전처리 없이 최소 필드만 추출한다 (movie_id, title, genres, keywords, overview).

    Args:
        jsonl_path: JSONL 파일 경로

    Returns:
        [{"movie_id": 123, "title": "...", "genres": [...], ...}, ...]
    """
    # 장르 영문→한국어 변환 테이블 (preprocessor.py의 GENRE_ID_TO_KR 축약)
    genre_en_to_kr: dict[str, str] = {
        "Action": "액션", "Adventure": "모험", "Animation": "애니메이션",
        "Comedy": "코미디", "Crime": "범죄", "Documentary": "다큐멘터리",
        "Drama": "드라마", "Family": "가족", "Fantasy": "판타지",
        "History": "역사", "Horror": "공포", "Music": "음악",
        "Mystery": "미스터리", "Romance": "로맨스", "Science Fiction": "SF",
        "TV Movie": "TV 영화", "Thriller": "스릴러", "War": "전쟁", "Western": "서부",
    }

    movies: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # 장르 변환
                raw_genres = data.get("genres") or []
                genres = []
                for g in raw_genres:
                    name = g.get("name", "") if isinstance(g, dict) else str(g)
                    genres.append(genre_en_to_kr.get(name, name))

                # 키워드 추출
                raw_keywords = data.get("keywords") or []
                keywords = []
                for kw in raw_keywords:
                    if isinstance(kw, dict):
                        keywords.append(kw.get("name", ""))
                    elif isinstance(kw, str):
                        keywords.append(kw)

                movies.append({
                    "movie_id": str(data.get("id", "")),  # 문자열로 통일 (ES의 id 필드와 일치)
                    "title": data.get("title") or data.get("original_title", ""),
                    "genres": genres,
                    "keywords": keywords,
                    "overview": data.get("overview") or "",
                })
            except (json.JSONDecodeError, Exception):
                continue

    logger.info("movies_loaded_from_jsonl", total=len(movies))
    return movies


# ══════════════════════════════════════════════════════════════
# 비용 추정
# ══════════════════════════════════════════════════════════════

def estimate_cost(
    total_movies: int,
    batch_size: int,
    model: str,
) -> None:
    """
    예상 비용을 출력한다 (API 호출 없음).

    모델별 토큰 단가 기준으로 입력/출력 토큰을 추정한다.
    """
    # 모델별 단가 ($/1M tokens) — (입력, 출력)
    pricing: dict[str, tuple[float, float]] = {
        # Upstage Solar 모델
        "solar-pro3": (0.15, 0.60),
        "solar-pro2": (0.15, 0.60),
        "solar-mini": (0.15, 0.15),
        # OpenAI 모델
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1-nano": (0.10, 0.40),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4o": (2.50, 10.00),
    }

    input_price, output_price = pricing.get(model, (0.15, 0.60))
    api_calls = (total_movies + batch_size - 1) // batch_size

    # 토큰 추정: 시스템 ~150 + 영화당 ~80 입력, 영화당 ~20 출력
    input_tokens = api_calls * (150 + 80 * batch_size)
    output_tokens = api_calls * (20 * batch_size)

    input_cost = input_tokens / 1_000_000 * input_price
    output_cost = output_tokens / 1_000_000 * output_price
    total_cost = input_cost + output_cost
    total_krw = total_cost * 1370  # 환율 추정

    print(f"\n{'=' * 60}")
    print(f"  비용 추정 (model={model}, batch={batch_size})")
    print(f"{'=' * 60}")
    print(f"  대상 영화: {total_movies:,}건")
    print(f"  API 호출: {api_calls:,}회")
    print(f"  입력 토큰: ~{input_tokens / 1_000_000:.1f}M × ${input_price}/1M = ${input_cost:.2f}")
    print(f"  출력 토큰: ~{output_tokens / 1_000_000:.1f}M × ${output_price}/1M = ${output_cost:.2f}")
    print(f"  ────────────────────────────")
    print(f"  총 예상 비용: ${total_cost:.2f} (약 {total_krw:,.0f}원)")
    print(f"{'=' * 60}\n")


# ══════════════════════════════════════════════════════════════
# 메인 처리
# ══════════════════════════════════════════════════════════════

async def run_mood_enrichment(
    api_key: str,
    provider: str = "upstage",
    model: str | None = None,
    base_url: str | None = None,
    concurrency: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rpm: int | None = None,
    source: str = "es",
    jsonl_path: Path | None = None,
    resume: bool = False,
    estimate_only: bool = False,
) -> None:
    """
    무드태그 보강 메인 함수.

    1. DB/JSONL에서 영화 목록 로드
    2. 체크포인트 기반 오프셋 적용 (resume 모드)
    3. MEGA_CHUNK_SIZE(1000건) 단위로 반복:
       a. batch_size(10건) 단위로 LLM 호출 (비동기 동시 실행)
       b. 3개 DB 동시 갱신 (Qdrant + Neo4j + ES)
       c. 체크포인트 저장
    4. 완료 요약 출력

    Args:
        api_key: LLM API 키 (Upstage/OpenAI)
        provider: API 제공자 ("upstage" 또는 "openai")
        model: 사용할 모델 (미지정 시 provider별 기본값)
        base_url: 커스텀 API 엔드포인트 (미지정 시 provider별 기본값)
        concurrency: 동시 LLM 요청 수 (미지정 시 provider별 기본값)
        batch_size: 1회 API 호출당 영화 수 (기본: 10)
        rpm: 분당 최대 요청 수 (미지정 시 provider별 기본값)
        source: 데이터 소스 ("es" 또는 "jsonl")
        jsonl_path: JSONL 파일 경로 (source="jsonl" 시)
        resume: 이전 체크포인트에서 재개 여부
        estimate_only: 비용 추정만 출력 (API 호출 없음)
    """
    # ── Provider별 기본값 적용 ──
    prov_cfg = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["upstage"])
    model = model or prov_cfg["model"]
    base_url = base_url or prov_cfg["base_url"]
    rpm = rpm or prov_cfg["rpm"]
    concurrency = concurrency or prov_cfg["concurrency"]
    # ── 1. 체크포인트 로드 ──
    checkpoint = load_checkpoint() if resume else {}
    offset = checkpoint.get("total_processed", 0) if resume else 0
    total_updated = checkpoint.get("total_updated", 0) if resume else 0
    total_failed = checkpoint.get("total_failed", 0) if resume else 0
    total_fallback = checkpoint.get("total_fallback", 0) if resume else 0

    if resume and offset > 0:
        print(f"[RESUME] 이전 체크포인트에서 재개: offset={offset:,}, updated={total_updated:,}")

    # ── 2. DB 초기화 ──
    print("[Step 0] DB 클라이언트 초기화...")
    await init_all_clients()

    # ── 3. 데이터 읽기 ──
    print(f"[Step 1] 영화 데이터 로드 (source={source})...")
    if source == "es":
        all_movies = await read_movies_from_es()
    else:
        path = jsonl_path or DEFAULT_JSONL_PATH
        all_movies = read_movies_from_jsonl(path)

    # offset 적용 (resume 시 이미 처리된 영화 건너뛰기)
    movies = all_movies[offset:]
    total_movies = len(movies)
    print(f"         전체: {len(all_movies):,}건 | 처리 대상: {total_movies:,}건 (offset={offset:,})")

    if total_movies == 0:
        print("[완료] 처리할 영화가 없습니다.")
        await close_all_clients()
        return

    # ── 비용 추정 모드 ──
    if estimate_only:
        estimate_cost(total_movies, batch_size, model)
        await close_all_clients()
        return

    # ── 4. LLM 클라이언트 초기화 (Upstage/OpenAI 호환) ──
    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)
    print(f"         provider={provider} | base_url={base_url or 'default'}")
    semaphore = asyncio.Semaphore(concurrency)
    rpm_limiter = RPMLimiter(rpm)

    # ── 5. 메가 청크 단위 처리 ──
    start_time = time.time()
    processed = 0

    print(f"\n[Step 2] 무드태그 생성 시작")
    print(f"         model={model} | concurrency={concurrency} | batch={batch_size} | rpm={rpm}")
    print(f"         예상 API 호출: {(total_movies + batch_size - 1) // batch_size:,}회")
    estimate_cost(total_movies, batch_size, model)
    print("=" * 70)

    for mega_start in range(0, total_movies, MEGA_CHUNK_SIZE):
        mega_chunk = movies[mega_start:mega_start + MEGA_CHUNK_SIZE]

        # ── LLM 배치 호출 (비동기 동시 실행) ──
        all_updates: dict[str, list[str]] = {}

        async def process_single_batch(batch_movies: list[dict]) -> dict[str, list[str]]:
            """단일 배치의 무드태그를 생성한다 (세마포어 + RPM 제한 적용)."""
            async with semaphore:
                return await generate_mood_batch(client, batch_movies, model, rpm_limiter)

        # 배치 분할 및 동시 실행
        tasks = []
        for i in range(0, len(mega_chunk), batch_size):
            batch = mega_chunk[i:i + batch_size]
            tasks.append(process_single_batch(batch))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 수집
        chunk_fallback = 0
        for result in results:
            if isinstance(result, Exception):
                logger.warning("batch_exception", error=str(result))
                total_failed += batch_size
                continue
            all_updates.update(result)

        # ── DB 갱신 (3개 DB 동시) ──
        if all_updates:
            db_results = await asyncio.gather(
                update_qdrant_moods(all_updates),
                update_neo4j_moods(all_updates),
                update_es_moods(all_updates),
                return_exceptions=True,
            )

            # DB 갱신 결과 로깅
            db_names = ["qdrant", "neo4j", "es"]
            for name, result in zip(db_names, db_results):
                if isinstance(result, Exception):
                    logger.warning(f"{name}_update_exception", error=str(result))

        # ── 진행 상황 계산 ──
        processed += len(mega_chunk)
        total_updated += len(all_updates)

        elapsed = time.time() - start_time
        speed = processed / elapsed if elapsed > 0 else 0
        eta_seconds = (total_movies - processed) / speed if speed > 0 else 0
        eta_h = int(eta_seconds) // 3600
        eta_m = (int(eta_seconds) % 3600) // 60

        # ── 체크포인트 저장 ──
        save_checkpoint({
            "total_processed": offset + processed,
            "total_updated": total_updated,
            "total_failed": total_failed,
            "total_fallback": total_fallback,
            "total_movies": len(all_movies),
            "model": model,
            "start_time": checkpoint.get("start_time", datetime.now().isoformat()),
        })

        # ── 진행 상황 출력 ──
        print(
            f"  [Chunk {mega_start // MEGA_CHUNK_SIZE + 1:>5}] "
            f"{offset + processed:>10,}/{len(all_movies):,} "
            f"({processed / total_movies * 100:5.1f}%) | "
            f"갱신 {len(all_updates):>4} | "
            f"속도 {speed:.0f}/s | "
            f"ETA {eta_h}h{eta_m:02d}m"
        )

    # ── 6. 완료 ──
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[완료] 무드태그 보강 완료")
    print(f"  처리: {processed:,}건 | 갱신: {total_updated:,}건 | 실패: {total_failed:,}건")
    print(f"  소요: {elapsed / 3600:.1f}시간 ({elapsed:.0f}초)")

    # 체크포인트 최종 저장
    save_checkpoint({
        "total_processed": offset + processed,
        "total_updated": total_updated,
        "total_failed": total_failed,
        "total_fallback": total_fallback,
        "total_movies": len(all_movies),
        "model": model,
        "status": "completed",
        "start_time": checkpoint.get("start_time", datetime.now().isoformat()),
        "end_time": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed),
    })

    await close_all_clients()


# ══════════════════════════════════════════════════════════════
# CLI 인터페이스
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """커맨드라인 인수를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="무드태그 보강 스크립트 (Upstage Solar / OpenAI 호환 API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Upstage Solar Pro 3으로 실행 (기본, .env의 UPSTAGE_API_KEY 사용)
  PYTHONPATH=src uv run python scripts/run_mood_enrichment.py

  # API 키 직접 지정
  PYTHONPATH=src uv run python scripts/run_mood_enrichment.py --api-key up_xxx

  # 이전 중단점에서 재개 (API 키 교체 가능)
  PYTHONPATH=src uv run python scripts/run_mood_enrichment.py --resume --api-key up_새키

  # OpenAI GPT-4o-mini로 실행
  PYTHONPATH=src uv run python scripts/run_mood_enrichment.py --provider openai --api-key sk-xxx

  # 비용 추정만
  PYTHONPATH=src uv run python scripts/run_mood_enrichment.py --estimate-only
        """,
    )
    parser.add_argument(
        "--provider", choices=["upstage", "openai"], default="upstage",
        help="API 제공자 (기본: upstage)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API 키 (미지정 시 .env에서 provider별 환경변수 사용)",
    )
    parser.add_argument(
        "--model", default=None,
        help="사용할 모델 (미지정 시 provider별 기본값: upstage=solar-pro3, openai=gpt-4o-mini)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="커스텀 API 엔드포인트 (미지정 시 provider별 기본값)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="동시 요청 수 (미지정 시 provider별 기본값: upstage=10, openai=20)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"1회 API 호출당 영화 수 (기본: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--rpm", type=int, default=None,
        help="분당 최대 요청 수 (미지정 시 provider별 기본값: upstage=100, openai=400)",
    )
    parser.add_argument(
        "--source", choices=["es", "jsonl"], default="es",
        help="데이터 소스 (기본: es)",
    )
    parser.add_argument(
        "--jsonl-path", type=Path, default=None,
        help="JSONL 파일 경로 (source=jsonl 시)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="이전 체크포인트에서 재개",
    )
    parser.add_argument(
        "--estimate-only", action="store_true",
        help="비용 추정만 출력 (API 호출 없음)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Provider별 환경변수에서 API 키 로드
    provider = args.provider
    prov_cfg = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["upstage"])
    api_key = args.api_key or os.environ.get(prov_cfg["env_key"], "")

    if not api_key and not args.estimate_only:
        env_key = prov_cfg["env_key"]
        print(f"ERROR: --api-key 또는 {env_key} 환경변수를 설정해주세요.")
        print(f"  예: PYTHONPATH=src uv run python scripts/run_mood_enrichment.py --api-key up_xxx")
        sys.exit(1)

    asyncio.run(run_mood_enrichment(
        api_key=api_key,
        provider=provider,
        model=args.model,
        base_url=args.base_url,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        rpm=args.rpm,
        source=args.source,
        jsonl_path=args.jsonl_path,
        resume=args.resume,
        estimate_only=args.estimate_only,
    ))
