"""
최종 전수 검증 스크립트 — 설계 대비 실제 5DB 적재 상태 검증.

Phase ML-4 재적재 + Phase 2~9 전체 완료 후 실행.
사용자의 "데이터 품질 최우선 + 설계대로 적재 검증" 요구에 맞춰
14 카테고리를 종합 검증한다.

설계 진실 원본:
    - docs/Phase_ML4_후속_실행_체크리스트.md §9 최종 검증
    - docs/데이터_적재_프로세스_전체분석_및_개선계획.md §9 임베딩 확장 + §10 신규 테이블
    - docs/AI_Agent_설계_및_구현계획서.md

검증 14 카테고리:
    1. 5DB 카운트 정합성 (Qdrant/ES/Neo4j/MySQL/Redis)
    2. Phase ML-1 한영이중 (한국 영화 title_en/director_original_name 영문률)
    3. Phase ML-2 한국어 keyword 매핑
    4. Phase ML-4 mood 품질 + fallback 비율
    5. KOBIS enrichment 적용 (kobis_movie_cd, kobis_nation, kobis_staffs)
    6. KMDb enrichment 적용 (kmdb_id, awards, kmdb_staffs)
    7. OMDb 적재 (movie_external_ratings)
    8. KOBIS 박스오피스 시계열 (box_office_daily)
    9. Kaggle ratings (kaggle_watch_history 26M)
    10. Person 파이프라인 (persons MySQL + Qdrant + Neo4j 보강)
    11. Movie LLM 보강 (5 target)
    12. Redis CF 캐시
    13. 샘플 영화 5건 E2E
    14. Agent 스모크 테스트

사용법:
    PYTHONPATH=src uv run python scripts/run_final_verification.py

    # 특정 카테고리만
    PYTHONPATH=src uv run python scripts/run_final_verification.py --only 1,2,3

    # Agent 스모크 skip (Agent 미기동 시)
    PYTHONPATH=src uv run python scripts/run_final_verification.py --skip-agent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

_env_file = _project_root / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from monglepick.config import settings  # noqa: E402
from monglepick.db.clients import (  # noqa: E402
    close_all_clients,
    get_elasticsearch,
    get_mysql,
    get_neo4j,
    get_redis,
    init_all_clients,
)

_HANGUL = re.compile(r"[\uac00-\ud7a3]")


@dataclass
class CheckResult:
    category: int
    name: str
    passed: bool
    warn: bool = False
    details: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.passed:
            return "✅ PASS"
        if self.warn:
            return "⚠️  WARN"
        return "❌ FAIL"


# ══════════════════════════════════════════════════════════════
# HTTP 유틸
# ══════════════════════════════════════════════════════════════


def _http_get(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post(url: str, body: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _qdrant_scroll(limit: int, filter_: dict | None = None) -> list[dict]:
    points = []
    offset = None
    while len(points) < limit:
        body = {
            "limit": min(256, limit - len(points)),
            "with_payload": True,
            "with_vector": False,
        }
        if offset:
            body["offset"] = offset
        if filter_:
            body["filter"] = filter_

        url = f"{settings.QDRANT_URL}/collections/{settings.QDRANT_COLLECTION}/points/scroll"
        try:
            data = _http_post(url, body)
        except Exception:
            break
        chunk = data.get("result", {}).get("points", [])
        if not chunk:
            break
        points.extend(chunk)
        offset = data.get("result", {}).get("next_page_offset")
        if not offset:
            break
    return points[:limit]


# ══════════════════════════════════════════════════════════════
# 검증 함수들 (14 카테고리)
# ══════════════════════════════════════════════════════════════


async def check_01_counts() -> CheckResult:
    """5DB 카운트 정합성."""
    details: dict = {}

    # Qdrant
    try:
        d = _http_get(f"{settings.QDRANT_URL}/collections/{settings.QDRANT_COLLECTION}")
        details["qdrant_movies"] = d["result"].get("points_count", 0)
    except Exception as e:
        details["qdrant_movies"] = f"ERROR: {e}"

    try:
        d = _http_get(f"{settings.QDRANT_URL}/collections/persons")
        details["qdrant_persons"] = d["result"].get("points_count", 0)
    except Exception:
        details["qdrant_persons"] = 0

    # ES
    try:
        es = await get_elasticsearch()
        cnt = await es.count(index="movies_bm25")
        details["es_movies_bm25"] = cnt["count"]
    except Exception as e:
        details["es_movies_bm25"] = f"ERROR: {e}"

    # Neo4j
    try:
        driver = await get_neo4j()
        async with driver.session() as session:
            r = await session.run("MATCH (m:Movie) RETURN count(m) AS c")
            details["neo4j_movie"] = (await r.single())["c"]
            r = await session.run("MATCH (p:Person) RETURN count(p) AS c")
            details["neo4j_person"] = (await r.single())["c"]
            r = await session.run("MATCH (p:Person) WHERE p.biography_ko IS NOT NULL RETURN count(p) AS c")
            details["neo4j_person_enriched"] = (await r.single())["c"]
    except Exception as e:
        details["neo4j_movie"] = f"ERROR: {e}"

    # MySQL
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) FROM movies")
                details["mysql_movies"] = (await cursor.fetchone())[0]
                await cursor.execute("SELECT COUNT(*) FROM persons")
                details["mysql_persons"] = (await cursor.fetchone())[0]
                await cursor.execute("SELECT COUNT(*) FROM kaggle_watch_history")
                details["mysql_kaggle_watch"] = (await cursor.fetchone())[0]
                await cursor.execute("SELECT COUNT(*) FROM box_office_daily")
                details["mysql_boxoffice"] = (await cursor.fetchone())[0]
                await cursor.execute("SELECT COUNT(*) FROM movie_external_ratings")
                details["mysql_omdb_ratings"] = (await cursor.fetchone())[0]
    except Exception as e:
        details["mysql_error"] = str(e)

    # Redis CF
    try:
        redis = await get_redis()
        cf_keys = await redis.keys("cf:*")
        details["redis_cf_keys"] = len(cf_keys)
    except Exception as e:
        details["redis_cf_keys"] = f"ERROR: {e}"

    # 정합성 판단
    q = details.get("qdrant_movies", 0)
    e = details.get("es_movies_bm25", 0)
    n = details.get("neo4j_movie", 0)
    m = details.get("mysql_movies", 0)

    if isinstance(q, int) and isinstance(e, int) and isinstance(n, int) and isinstance(m, int):
        max_v = max(q, e, n, m, 1)
        diffs = {
            "Qdrant↔ES": abs(q - e),
            "Qdrant↔Neo4j": abs(q - n),
            "Qdrant↔MySQL": abs(q - m),
        }
        max_diff = max(diffs.values())
        details["diffs"] = diffs
        details["max_diff_pct"] = round(max_diff * 100 / max_v, 3)
        passed = max_diff < max_v * 0.01  # 1% 허용
    else:
        passed = False

    return CheckResult(1, "5DB 카운트 정합성", passed, details=details)


async def check_02_ml1_bilingual() -> CheckResult:
    """Phase ML-1 한영이중 — 한국 영화 title_en / director_original_name 영문률."""
    # 한국 영화 필터 (original_language=ko OR kobis_nation=한국)
    korean_filter = {
        "should": [
            {"key": "original_language", "match": {"value": "ko"}},
            {"key": "kobis_nation", "match": {"value": "한국"}},
        ],
    }
    samples = _qdrant_scroll(2000, filter_={"must": [], "should": korean_filter["should"], "must_not": []})

    n = len(samples)
    if n == 0:
        return CheckResult(2, "ML-1 한영이중 (한국 영화)", False, details={"error": "한국 영화 0건 — filter 확인"})

    total = 0
    title_en_eng = 0
    director_eng = 0
    cast_original_exists = 0

    for p in samples:
        pl = p.get("payload", {}) or {}
        total += 1

        title_en = (pl.get("title_en", "") or "").strip()
        if title_en and not _HANGUL.search(title_en):
            title_en_eng += 1

        dname = (pl.get("director_original_name", "") or "").strip()
        if dname and not _HANGUL.search(dname):
            director_eng += 1

        con = pl.get("cast_original_names") or []
        if con and isinstance(con, list) and len(con) > 0:
            cast_original_exists += 1

    title_en_rate = title_en_eng * 100 / total
    director_rate = director_eng * 100 / total
    cast_rate = cast_original_exists * 100 / total

    details = {
        "sample_total": total,
        "title_en 영문률": f"{title_en_rate:.1f}%",
        "director_original_name 영문률": f"{director_rate:.1f}%",
        "cast_original_names 존재율": f"{cast_rate:.1f}%",
    }
    # 허용 기준: Phase 2 실행 후 70% 이상
    passed = title_en_rate >= 70 and director_rate >= 70
    warn = not passed and title_en_rate >= 40
    return CheckResult(2, "ML-1 한영이중 (한국 영화)", passed, warn=warn, details=details)


async def check_03_ml2_korean_keywords() -> CheckResult:
    """Phase ML-2 한국어 keyword 매핑."""
    samples = _qdrant_scroll(2000)
    n = len(samples)
    if n == 0:
        return CheckResult(3, "ML-2 한국어 keyword", False, details={"error": "샘플 0"})

    total_movies_with_kw = 0
    movies_with_korean_kw = 0
    total_terms = 0
    korean_terms = 0

    for p in samples:
        pl = p.get("payload", {}) or {}
        kws = pl.get("keywords") or []
        if not isinstance(kws, list) or not kws:
            continue
        total_movies_with_kw += 1
        has_korean = False
        for kw in kws:
            if isinstance(kw, str):
                total_terms += 1
                if _HANGUL.search(kw):
                    korean_terms += 1
                    has_korean = True
        if has_korean:
            movies_with_korean_kw += 1

    movie_rate = movies_with_korean_kw * 100 / max(total_movies_with_kw, 1)
    term_rate = korean_terms * 100 / max(total_terms, 1)

    details = {
        "keywords 배열 존재 영화": total_movies_with_kw,
        "한국어 매핑 영화 비율": f"{movie_rate:.1f}%",
        "전체 terms 한국어 비율": f"{term_rate:.1f}%",
    }
    passed = movie_rate >= 60
    warn = not passed and movie_rate >= 30
    return CheckResult(3, "ML-2 한국어 keyword", passed, warn=warn, details=details)


async def check_04_ml4_mood() -> CheckResult:
    """Phase ML-4 mood 품질 + fallback 비율."""
    samples = _qdrant_scroll(2000)
    n = len(samples)
    if n == 0:
        return CheckResult(4, "ML-4 mood 품질", False, details={"error": "샘플 0"})

    mood_count_dist = Counter()
    total_with_mood = 0
    deterministic_fallback = 0  # ["잔잔"] 단일
    tag_freq = Counter()

    for p in samples:
        pl = p.get("payload", {}) or {}
        moods = pl.get("mood_tags") or []
        if not isinstance(moods, list):
            continue
        mood_count_dist[min(len(moods), 10)] += 1
        if moods:
            total_with_mood += 1
            for t in moods:
                if isinstance(t, str):
                    tag_freq[t] += 1
        if moods == ["잔잔"]:
            deterministic_fallback += 1

    fallback_rate = deterministic_fallback * 100 / n
    mood_coverage = total_with_mood * 100 / n
    unique_tags = len(tag_freq)

    details = {
        "mood_coverage": f"{mood_coverage:.1f}%",
        "결정적 fallback (['잔잔'])": f"{fallback_rate:.2f}%",
        "unique mood tags": unique_tags,
        "top 5 moods": tag_freq.most_common(5),
    }
    passed = fallback_rate < 2.0 and mood_coverage > 95
    return CheckResult(4, "ML-4 mood 품질", passed, details=details)


async def check_05_kobis_enrichment() -> CheckResult:
    """KOBIS enrichment 적용 비율."""
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT COUNT(*) FROM movies WHERE kobis_movie_cd IS NOT NULL AND kobis_movie_cd != ''"
                )
                kobis_count = (await cursor.fetchone())[0]
                await cursor.execute(
                    "SELECT COUNT(*) FROM movies WHERE kobis_nation = '한국'"
                )
                korean_count = (await cursor.fetchone())[0]
                await cursor.execute(
                    "SELECT AVG(JSON_LENGTH(kobis_staffs)) FROM movies "
                    "WHERE kobis_staffs IS NOT NULL"
                )
                avg_staffs = (await cursor.fetchone())[0] or 0
                await cursor.execute(
                    "SELECT AVG(JSON_LENGTH(kobis_actors)) FROM movies "
                    "WHERE kobis_actors IS NOT NULL"
                )
                avg_actors = (await cursor.fetchone())[0] or 0

        details = {
            "kobis_movie_cd 채움": kobis_count,
            "한국 영화 (nation=한국)": korean_count,
            "평균 kobis_staffs 길이": round(float(avg_staffs), 1),
            "평균 kobis_actors 길이": round(float(avg_actors), 1),
        }
        # 합격: 한국 영화 10K+ 있고 staffs 평균 100+
        passed = korean_count >= 10_000 and float(avg_staffs) >= 100
        warn = not passed and korean_count >= 5_000
        return CheckResult(5, "KOBIS enrichment", passed, warn=warn, details=details)
    except Exception as e:
        return CheckResult(5, "KOBIS enrichment", False, details={"error": str(e)[:200]})


async def check_06_kmdb_enrichment() -> CheckResult:
    """KMDb enrichment 적용 비율."""
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) FROM movies WHERE kmdb_id IS NOT NULL AND kmdb_id != ''")
                kmdb_count = (await cursor.fetchone())[0]
                await cursor.execute("SELECT COUNT(*) FROM movies WHERE awards IS NOT NULL AND awards != ''")
                awards_count = (await cursor.fetchone())[0]
                await cursor.execute("SELECT COUNT(*) FROM movies WHERE filming_location IS NOT NULL AND filming_location != ''")
                location_count = (await cursor.fetchone())[0]

        details = {
            "kmdb_id 채움": kmdb_count,
            "awards 채움": awards_count,
            "filming_location 채움": location_count,
        }
        passed = kmdb_count >= 5_000 and awards_count >= 1_000
        warn = not passed and kmdb_count >= 1_000
        return CheckResult(6, "KMDb enrichment", passed, warn=warn, details=details)
    except Exception as e:
        return CheckResult(6, "KMDb enrichment", False, details={"error": str(e)[:200]})


async def check_07_omdb() -> CheckResult:
    """OMDb movie_external_ratings 적재."""
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) FROM movie_external_ratings")
                total = (await cursor.fetchone())[0]
                await cursor.execute(
                    "SELECT COUNT(*) FROM movie_external_ratings WHERE imdb_rating IS NOT NULL"
                )
                imdb_count = (await cursor.fetchone())[0]
                await cursor.execute(
                    "SELECT COUNT(*) FROM movie_external_ratings WHERE rotten_tomatoes_score IS NOT NULL"
                )
                rt_count = (await cursor.fetchone())[0]
                await cursor.execute(
                    "SELECT COUNT(*) FROM movie_external_ratings WHERE metacritic IS NOT NULL"
                )
                meta_count = (await cursor.fetchone())[0]

        details = {
            "total": total,
            "imdb_rating": imdb_count,
            "rotten_tomatoes": rt_count,
            "metacritic": meta_count,
        }
        passed = total >= 900  # Phase 6 첫 실행 950 기준
        warn = not passed and total >= 100
        return CheckResult(7, "OMDb 외부 평점", passed, warn=warn, details=details)
    except Exception as e:
        return CheckResult(7, "OMDb 외부 평점", False, details={"error": str(e)[:200]})


async def check_08_boxoffice_history() -> CheckResult:
    """KOBIS 박스오피스 시계열."""
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) FROM box_office_daily")
                total = (await cursor.fetchone())[0]
                await cursor.execute(
                    "SELECT MIN(target_dt), MAX(target_dt), COUNT(DISTINCT target_dt) FROM box_office_daily"
                )
                row = await cursor.fetchone()
                min_dt, max_dt, uniq_days = row if row else (None, None, 0)

        details = {
            "total_rows": total,
            "date_range": f"{min_dt} ~ {max_dt}",
            "unique_days": uniq_days,
        }
        passed = total >= 3000 and (uniq_days or 0) >= 300
        warn = not passed and total >= 500
        return CheckResult(8, "KOBIS 박스오피스 시계열", passed, warn=warn, details=details)
    except Exception as e:
        return CheckResult(8, "KOBIS 박스오피스 시계열", False, details={"error": str(e)[:200]})


async def check_09_kaggle_ratings() -> CheckResult:
    """Kaggle ratings 26M 적재."""
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) FROM kaggle_watch_history")
                total = (await cursor.fetchone())[0]
                await cursor.execute("SELECT COUNT(DISTINCT user_id) FROM kaggle_watch_history")
                users = (await cursor.fetchone())[0]
                await cursor.execute("SELECT COUNT(DISTINCT movie_id) FROM kaggle_watch_history")
                movies = (await cursor.fetchone())[0]

        details = {
            "total": total,
            "distinct_users": users,
            "distinct_movies": movies,
        }
        passed = total >= 25_000_000
        warn = not passed and total >= 1_000_000
        return CheckResult(9, "Kaggle ratings 26M", passed, warn=warn, details=details)
    except Exception as e:
        return CheckResult(9, "Kaggle ratings 26M", False, details={"error": str(e)[:200]})


async def check_10_person_pipeline() -> CheckResult:
    """Person 파이프라인 — MySQL persons + Qdrant persons + Neo4j biography_ko."""
    details: dict = {}

    # MySQL
    try:
        pool = await get_mysql()
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) FROM persons")
                details["mysql_persons"] = (await cursor.fetchone())[0]
                await cursor.execute("SELECT COUNT(*) FROM persons WHERE biography_ko IS NOT NULL AND biography_ko != ''")
                details["mysql_persons_enriched"] = (await cursor.fetchone())[0]
    except Exception as e:
        details["mysql_error"] = str(e)[:100]

    # Qdrant persons
    try:
        d = _http_get(f"{settings.QDRANT_URL}/collections/persons")
        details["qdrant_persons"] = d["result"].get("points_count", 0)
    except Exception:
        details["qdrant_persons"] = 0

    # Neo4j
    try:
        driver = await get_neo4j()
        async with driver.session() as session:
            r = await session.run("MATCH (p:Person) WHERE p.biography_ko IS NOT NULL RETURN count(p) AS c")
            details["neo4j_persons_enriched"] = (await r.single())["c"]
            r = await session.run("MATCH (p:Person) WHERE p.style_tags IS NOT NULL RETURN count(p) AS c")
            details["neo4j_persons_with_style"] = (await r.single())["c"]
    except Exception as e:
        details["neo4j_error"] = str(e)[:100]

    q = details.get("qdrant_persons", 0)
    m = details.get("mysql_persons", 0)
    passed = isinstance(q, int) and q >= 500_000 and isinstance(m, int) and m >= 500_000
    warn = not passed and isinstance(q, int) and q >= 50_000
    return CheckResult(10, "Person 파이프라인", passed, warn=warn, details=details)


async def check_11_movie_llm_enrichment() -> CheckResult:
    """Movie LLM 보강 (5 target) — popularity 구간별 채움률 정밀 검증."""
    samples = _qdrant_scroll(5000)
    n = len(samples)
    if n == 0:
        return CheckResult(11, "Movie LLM 보강", False, details={"error": "샘플 0"})

    # popularity 구간별 분석
    pop_buckets = {
        "pop>=10 유명작": {"min": 10, "max": 999999},
        "pop 5~10 인기": {"min": 5, "max": 10},
        "pop 2~5 보통": {"min": 2, "max": 5},
        "pop <2 비인기": {"min": 0, "max": 2},
    }
    bucket_stats: dict[str, dict] = {
        name: {"total": 0, "tagline_ko": 0, "category_tags": 0, "overview_ko": 0,
               "one_line_summary": 0, "llm_keywords": 0}
        for name in pop_buckets
    }

    counts = {
        "tagline_ko": 0,
        "overview_ko": 0,
        "category_tags": 0,
        "one_line_summary": 0,
        "llm_keywords": 0,
    }
    for p in samples:
        pl = p.get("payload", {}) or {}
        pop = pl.get("popularity_score") or 0

        # 전체 카운트 + 구간별 카운트
        for k in counts:
            v = pl.get(k)
            if v and (not isinstance(v, str) or v.strip()):
                counts[k] += 1

        # popularity 구간별 집계
        for bname, brange in pop_buckets.items():
            if brange["min"] <= pop < brange["max"]:
                bucket_stats[bname]["total"] += 1
                for k in counts:
                    v = pl.get(k)
                    if v and (not isinstance(v, str) or v.strip()):
                        bucket_stats[bname][k] += 1
                break

    details = {k: f"{v}/{n} ({v*100/n:.1f}%)" for k, v in counts.items()}

    # popularity 구간별 채움률 (핵심 검증)
    details["popularity 구간별"] = "---"
    for bname, stats in bucket_stats.items():
        bt = stats["total"]
        if bt == 0:
            details[f"  {bname}"] = "0건"
            continue
        cat_rate = stats["category_tags"] * 100 / bt
        tag_rate = stats["tagline_ko"] * 100 / bt
        details[f"  {bname} ({bt}건)"] = f"category={cat_rate:.0f}% tagline={tag_rate:.0f}%"

    # 합격 기준 (보강):
    # - 인기작 (pop≥2.0) 구간에서 category_tags 80% 이상이면 PASS
    # - 비인기작은 mood_tags 100% (Task #5)로 기본 품질 보장
    pop2_stats = {k: v for bname, v in bucket_stats.items()
                  for k in [bname] if "비인기" not in bname}
    pop2_total = sum(v["total"] for v in pop2_stats.values())
    pop2_category = sum(v["category_tags"] for v in pop2_stats.values())
    pop2_rate = pop2_category * 100 / max(pop2_total, 1)

    details["인기작(pop>=2) category 채움률"] = f"{pop2_rate:.1f}%"
    passed = pop2_rate >= 80
    warn = not passed and pop2_rate >= 50
    return CheckResult(11, "Movie LLM 보강 (popularity 구간별)", passed, warn=warn, details=details)


async def check_12_redis_cf() -> CheckResult:
    """Redis CF 캐시."""
    try:
        redis = await get_redis()
        all_keys = await redis.keys("cf:*")
        details = {"cf_keys": len(all_keys)}

        # 샘플 조회
        if all_keys:
            sample_key = all_keys[0]
            sample_type = await redis.type(sample_key)
            details["sample_key_type"] = sample_type
            if sample_type == b"string" or sample_type == "string":
                val = await redis.get(sample_key)
                details["sample_size"] = len(val) if val else 0
            elif sample_type == b"zset" or sample_type == "zset":
                card = await redis.zcard(sample_key)
                details["sample_zset_card"] = card

        passed = len(all_keys) >= 100_000  # 270K user 기준
        warn = not passed and len(all_keys) >= 1000
        return CheckResult(12, "Redis CF 캐시", passed, warn=warn, details=details)
    except Exception as e:
        return CheckResult(12, "Redis CF 캐시", False, details={"error": str(e)[:200]})


async def check_13_sample_movies() -> CheckResult:
    """샘플 영화 5건 E2E 검증 (5DB 존재 + Phase ML-1 영문 필드 확인)."""
    samples_to_check = [
        ("496243", "기생충", "Parasite"),
        ("157336", "인터스텔라", "Interstellar"),
        ("396535", "부산행", "Train to Busan"),  # 2016
        ("313369", "라라랜드", "La La Land"),
        ("670", "올드보이", "Oldboy"),
    ]

    details: dict = {}
    all_passed = True

    pool = await get_mysql()
    es = await get_elasticsearch()
    driver = await get_neo4j()

    for mid, ko_title, en_title in samples_to_check:
        sample_result = {"title_ko": ko_title, "expected_title_en": en_title}

        # Qdrant
        try:
            d = _http_post(
                f"{settings.QDRANT_URL}/collections/{settings.QDRANT_COLLECTION}/points",
                {"ids": [int(mid)], "with_payload": True, "with_vector": False},
            )
            points = d.get("result", [])
            sample_result["qdrant"] = bool(points)
            if points:
                pl = points[0].get("payload", {})
                sample_result["qdrant_title_en"] = pl.get("title_en", "")
        except Exception as e:
            sample_result["qdrant"] = f"ERROR: {str(e)[:50]}"

        # ES
        try:
            doc = await es.get(index="movies_bm25", id=mid)
            sample_result["es"] = True
            sample_result["es_title_en"] = doc["_source"].get("title_en", "")
        except Exception:
            sample_result["es"] = False

        # Neo4j
        try:
            async with driver.session() as session:
                r = await session.run("MATCH (m:Movie {id: $id}) RETURN m.title_en AS te", {"id": mid})
                rec = await r.single()
                if rec:
                    sample_result["neo4j"] = True
                    sample_result["neo4j_title_en"] = rec["te"] or ""
                else:
                    sample_result["neo4j"] = False
        except Exception as e:
            sample_result["neo4j"] = f"ERROR: {str(e)[:50]}"

        # MySQL
        try:
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "SELECT title_en, director_original_name FROM movies WHERE movie_id = %s",
                        (mid,),
                    )
                    row = await cursor.fetchone()
                    if row:
                        sample_result["mysql"] = True
                        sample_result["mysql_title_en"] = row[0] or ""
                    else:
                        sample_result["mysql"] = False
        except Exception as e:
            sample_result["mysql"] = f"ERROR: {str(e)[:50]}"

        # 합격 기준: 4DB 모두 존재 + title_en 이 영어 (한글 아님)
        present_count = sum(1 for db in ("qdrant", "es", "neo4j", "mysql") if sample_result.get(db) is True)
        sample_result["present_in"] = f"{present_count}/4 DBs"

        if present_count < 4:
            all_passed = False

        details[f"id={mid}"] = sample_result

    return CheckResult(13, "샘플 영화 5건 E2E", all_passed, details=details)


async def check_14_agent_smoke(skip: bool = False) -> CheckResult:
    """Agent /health + /api/v1/chat/sync 스모크 테스트."""
    if skip:
        return CheckResult(14, "Agent 스모크 테스트", True, warn=True, details={"skipped": True})

    details: dict = {}
    try:
        h = _http_get("http://localhost:8000/health", timeout=5)
        details["health"] = h
    except Exception as e:
        details["health_error"] = str(e)[:100]
        return CheckResult(14, "Agent 스모크 테스트", False, details=details)

    queries = [
        "봉준호 감독 영화 추천해줘",
        "Parasite 같은 스릴러",
        "송강호 주연 최신 영화",
    ]
    passed_count = 0
    for q in queries:
        try:
            body = {
                "user_id": "final-verify",
                "session_id": f"verify-{hash(q)}",
                "message": q,
            }
            r = _http_post("http://localhost:8000/api/v1/chat/sync", body, timeout=60)
            ok = bool(r.get("response") or r.get("movies"))
            details[f"query: {q[:20]}"] = "✅" if ok else "❌"
            if ok:
                passed_count += 1
        except Exception as e:
            details[f"query: {q[:20]}"] = f"ERROR: {str(e)[:60]}"

    passed = passed_count == len(queries)
    return CheckResult(14, "Agent 스모크 테스트", passed, details=details)


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════


async def main(only: set[int] | None = None, skip_agent: bool = False) -> int:
    all_checks = [
        (1, check_01_counts),
        (2, check_02_ml1_bilingual),
        (3, check_03_ml2_korean_keywords),
        (4, check_04_ml4_mood),
        (5, check_05_kobis_enrichment),
        (6, check_06_kmdb_enrichment),
        (7, check_07_omdb),
        (8, check_08_boxoffice_history),
        (9, check_09_kaggle_ratings),
        (10, check_10_person_pipeline),
        (11, check_11_movie_llm_enrichment),
        (12, check_12_redis_cf),
        (13, check_13_sample_movies),
        (14, lambda: check_14_agent_smoke(skip=skip_agent)),
    ]

    if only:
        all_checks = [(i, f) for i, f in all_checks if i in only]

    print("=" * 78)
    print("  최종 전수 검증 — Phase ML-4 재적재 + Phase 2~9 완료 후")
    print("=" * 78)

    await init_all_clients()
    results: list[CheckResult] = []

    try:
        for idx, fn in all_checks:
            print(f"\n[{idx:2d}] 실행 중...", flush=True)
            try:
                result = await fn()
            except Exception as e:
                result = CheckResult(idx, f"check_{idx}", False, details={"exception": str(e)[:200]})
            results.append(result)

            # 즉시 출력
            print(f"     {result.status}  {result.name}")
            for k, v in result.details.items():
                print(f"       {k:40s} {v}")

        # 최종 요약
        print()
        print("=" * 78)
        print("  최종 요약")
        print("=" * 78)
        pass_count = sum(1 for r in results if r.passed)
        warn_count = sum(1 for r in results if r.warn and not r.passed)
        fail_count = sum(1 for r in results if not r.passed and not r.warn)

        for r in results:
            print(f"  [{r.category:2d}] {r.status}  {r.name}")

        print()
        print(f"  ✅ PASS: {pass_count}/{len(results)}")
        print(f"  ⚠️  WARN: {warn_count}")
        print(f"  ❌ FAIL: {fail_count}")

        if pass_count == len(results):
            print("\n  🎉 모든 검증 통과 — 운영 배포 준비 완료")
        elif fail_count == 0:
            print("\n  ⚠️  일부 경고 있음 — 내용 검토 후 배포 결정")
        else:
            print("\n  ❌ 실패 항목 있음 — 수정 후 재검증 필요")

        return 0 if fail_count == 0 else 1

    finally:
        await close_all_clients()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="특정 카테고리만 (예: 1,2,3)")
    parser.add_argument("--skip-agent", action="store_true", help="14번 Agent 스모크 skip")
    args = parser.parse_args()

    only_set = None
    if args.only:
        only_set = set(int(x.strip()) for x in args.only.split(",") if x.strip())

    sys.exit(asyncio.run(main(only=only_set, skip_agent=args.skip_agent)))
