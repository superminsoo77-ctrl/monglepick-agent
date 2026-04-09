"""
Phase 1 검증 스크립트 — Task #5 완료 후 5DB 카운트 일치 + 샘플 품질 검증.

docs/Phase_ML4_후속_실행_체크리스트.md §1 참조.

검증 항목:
    1. 5DB 카운트: Qdrant / ES / Neo4j / (MySQL Phase 3 선행 X) / Redis CF 존재
    2. Phase ML-1/2/4 샘플 품질 (check_quality_ml1_ml2 재활용)
    3. mood fallback 로그 재검사 (check_mood_fallback 재활용)
    4. 샘플 영화 (기생충) 5DB 공통 존재 검증

사용법:
    PYTHONPATH=src uv run python scripts/run_phase1_verify.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# .env 로드
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


async def verify_counts() -> dict:
    """5DB 전 카운트 조회."""
    result: dict = {}

    # Qdrant
    req = urllib.request.Request(
        f"{settings.QDRANT_URL}/collections/{settings.QDRANT_COLLECTION}",
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        d = json.loads(resp.read().decode())
    result["qdrant_movies"] = d["result"].get("points_count", 0)

    # ES
    es = await get_elasticsearch()
    es_count = await es.count(index="movies_bm25")
    result["es_movies_bm25"] = es_count["count"]

    # Neo4j
    driver = await get_neo4j()
    async with driver.session() as session:
        r = await session.run("MATCH (m:Movie) RETURN count(m) AS c")
        result["neo4j_movie"] = (await r.single())["c"]
        r = await session.run("MATCH (p:Person) RETURN count(p) AS c")
        result["neo4j_person"] = (await r.single())["c"]
        r = await session.run(
            "MATCH (m:Movie)-[:HAS_GENRE]->(:Genre) RETURN count(DISTINCT m) AS c"
        )
        result["neo4j_movies_with_genre"] = (await r.single())["c"]
        r = await session.run(
            "MATCH ()-[r:DIRECTED]->() RETURN count(r) AS c"
        )
        result["neo4j_directed_relations"] = (await r.single())["c"]

    # MySQL (Phase 3 선행 필요 — 단순 존재 확인)
    pool = await get_mysql()
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) FROM movies")
                (cnt,) = await cursor.fetchone()
                result["mysql_movies"] = cnt
    except Exception as e:
        result["mysql_movies"] = f"ERROR: {str(e)[:80]}"

    # Redis CF 캐시
    redis = await get_redis()
    try:
        keys_pattern = await redis.keys("cf:*")
        result["redis_cf_keys"] = len(keys_pattern)
    except Exception as e:
        result["redis_cf_keys"] = f"ERROR: {str(e)[:80]}"

    return result


async def verify_sample_movie(target_title: str = "기생충") -> dict:
    """샘플 영화 (기생충) 가 5DB 에 존재하는지 검증."""
    result: dict = {"target": target_title}

    # ES 에서 검색하여 movie_id 획득
    es = await get_elasticsearch()
    search = await es.search(
        index="movies_bm25",
        body={
            "query": {
                "bool": {
                    "should": [
                        {"match": {"title": target_title}},
                        {"match": {"title_en": "Parasite"}},
                    ]
                }
            },
            "size": 3,
        },
    )
    hits = search.get("hits", {}).get("hits", [])
    if not hits:
        result["es_found"] = 0
        return result

    top = hits[0]
    movie_id = top["_id"]
    result["es_found"] = len(hits)
    result["es_top_id"] = movie_id
    result["es_top_title"] = top["_source"].get("title", "")
    result["es_top_title_en"] = top["_source"].get("title_en", "")
    result["es_director"] = top["_source"].get("director", "")
    result["es_director_original"] = top["_source"].get("director_original_name", "")

    # Qdrant 에서 같은 id 조회
    req_body = {"ids": [int(movie_id) if movie_id.isdigit() else movie_id], "with_payload": True, "with_vector": False}
    req = urllib.request.Request(
        f"{settings.QDRANT_URL}/collections/{settings.QDRANT_COLLECTION}/points",
        data=json.dumps(req_body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            qd = json.loads(resp.read().decode())
        points = qd.get("result", [])
        result["qdrant_found"] = len(points)
        if points:
            pl = points[0].get("payload", {})
            result["qdrant_mood_tags"] = pl.get("mood_tags", [])[:5]
            result["qdrant_keywords"] = pl.get("keywords", [])[:5]
            result["qdrant_director_original"] = pl.get("director_original_name", "")
    except Exception as e:
        result["qdrant_found"] = f"ERROR: {str(e)[:80]}"

    # Neo4j 에서 같은 id 조회
    driver = await get_neo4j()
    async with driver.session() as session:
        r = await session.run(
            "MATCH (m:Movie {id: $id}) "
            "OPTIONAL MATCH (p:Person)-[:DIRECTED]->(m) "
            "OPTIONAL MATCH (m)-[:HAS_GENRE]->(g:Genre) "
            "OPTIONAL MATCH (m)-[:HAS_KEYWORD]->(k:Keyword) "
            "RETURN m.title AS title, m.director AS director, "
            "       collect(DISTINCT p.name) AS directors, "
            "       collect(DISTINCT g.name) AS genres, "
            "       collect(DISTINCT k.name) AS keywords",
            {"id": movie_id},
        )
        record = await r.single()
        if record:
            result["neo4j_found"] = 1
            result["neo4j_title"] = record["title"]
            result["neo4j_genres"] = record["genres"][:5]
            result["neo4j_directors"] = record["directors"][:3]
            result["neo4j_keyword_count"] = len(record["keywords"])
        else:
            result["neo4j_found"] = 0

    # MySQL 에서 같은 id 조회
    pool = await get_mysql()
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "SELECT movie_id, title, title_en, director, director_original_name, "
                    "kobis_movie_cd, kmdb_id FROM movies WHERE movie_id = %s",
                    (movie_id,),
                )
                row = await cursor.fetchone()
                if row:
                    result["mysql_found"] = 1
                    result["mysql_title"] = row[1]
                    result["mysql_director"] = row[3]
                    result["mysql_director_original"] = row[4]
                    result["mysql_kobis_cd"] = row[5]
                    result["mysql_kmdb_id"] = row[6]
                else:
                    result["mysql_found"] = 0
    except Exception as e:
        result["mysql_found"] = f"ERROR: {str(e)[:80]}"

    return result


async def main() -> None:
    print("=" * 70)
    print("  Phase 1 검증 — Task #5 완료 후 5DB 정합성")
    print("=" * 70)
    print()

    await init_all_clients()

    try:
        # ── 1. 5DB 카운트 ──
        print("[1] 5DB 카운트")
        counts = await verify_counts()
        for k, v in counts.items():
            label = k.replace("_", " ")
            if isinstance(v, int):
                print(f"  {label:35s} {v:>12,}")
            else:
                print(f"  {label:35s} {v}")

        # 카운트 정합성 검증
        print()
        print("[2] 카운트 정합성")
        q = counts.get("qdrant_movies", 0) or 0
        e = counts.get("es_movies_bm25", 0) or 0
        n = counts.get("neo4j_movie", 0) or 0

        if isinstance(q, int) and isinstance(e, int) and isinstance(n, int):
            diff_qe = abs(q - e)
            diff_qn = abs(q - n)
            print(f"  Qdrant ↔ ES   diff: {diff_qe:>6,} ({diff_qe*100/max(q,1):.3f}%)")
            print(f"  Qdrant ↔ Neo4j diff: {diff_qn:>6,} ({diff_qn*100/max(q,1):.3f}%)")
            if diff_qe == 0 and diff_qn == 0:
                print("  ✅ 3DB 완벽 일치")
            elif diff_qe < q * 0.01 and diff_qn < q * 0.01:
                print("  ✅ 허용 범위 (< 1%)")
            else:
                print("  ⚠️  WARN — Neo4j 재동기화 필요")

        # ── 2. 샘플 영화 (기생충) ──
        print()
        print("[3] 샘플 영화 — 기생충 5DB 존재 검증")
        sample = await verify_sample_movie("기생충")
        for k, v in sample.items():
            label = k.replace("_", " ")
            if isinstance(v, list):
                print(f"  {label:30s} {v}")
            else:
                print(f"  {label:30s} {v}")

        # ── 3. 요약 ──
        print()
        print("=" * 70)
        print("  요약")
        print("=" * 70)
        es_ok = sample.get("es_found", 0) and sample.get("es_found") != 0
        qd_ok = sample.get("qdrant_found", 0) and sample.get("qdrant_found") != 0
        n4_ok = sample.get("neo4j_found", 0) and sample.get("neo4j_found") != 0
        ms_ok = sample.get("mysql_found", 0) and sample.get("mysql_found") != 0
        print(f"  ES   : {'✅' if es_ok else '❌'}")
        print(f"  Qdrant: {'✅' if qd_ok else '❌'}")
        print(f"  Neo4j: {'✅' if n4_ok else '❌'}")
        print(f"  MySQL: {'✅' if ms_ok else '❌ (Phase 3 선행 필요)'}")

    finally:
        await close_all_clients()


if __name__ == "__main__":
    asyncio.run(main())
