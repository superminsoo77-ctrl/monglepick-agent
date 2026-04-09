"""
기생충 1건 E2E 테스트 — Phase 2 KOBIS/KMDb enrichment 5DB 전파 검증.

Phase 2 전체 실행 전 단일 영화로 enrichment 파이프라인이 5DB 에 올바르게
반영되는지 검증한다. 기생충 (TMDB ID 496243) 이 한국 영화 Phase ML-1 한영
이중 버그의 대표 케이스이므로 테스트 대상으로 선택.

검증 절차:
    1. Before: 기생충 5DB 상태 스냅샷 (title_en, director_original_name, ...)
    2. KOBIS API 실 호출: movieCd=20183782 (기생충 2019)
    3. KMDb API 실 호출: 봉준호 감독 + 2019년
    4. build_kobis_enrichment_payload + build_kmdb_full_enrichment_payload
    5. Qdrant set_payload + ES update + Neo4j SET 속성
    6. After: 재조회하여 title_en="Parasite", director_original_name 영문 확인
    7. 5DB 일치 PASS/FAIL 판정

사용법:
    PYTHONPATH=src uv run python scripts/test_parasite_e2e.py
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
from monglepick.data_pipeline.es_loader import update_movie_partial  # noqa: E402
from monglepick.data_pipeline.kmdb_collector import KMDbCollector  # noqa: E402
from monglepick.data_pipeline.kmdb_enricher import (  # noqa: E402
    build_kmdb_full_enrichment_payload,
)
from monglepick.data_pipeline.kobis_collector import KOBISCollector  # noqa: E402
from monglepick.data_pipeline.kobis_movie_converter import (  # noqa: E402
    build_kobis_enrichment_payload,
)
from monglepick.data_pipeline.neo4j_loader import update_movie_properties  # noqa: E402
from monglepick.db.clients import (  # noqa: E402
    close_all_clients,
    get_elasticsearch,
    get_neo4j,
    init_all_clients,
)

PARASITE_TMDB_ID = "496243"
PARASITE_KOBIS_CD = "20183782"


def _snapshot_qdrant(movie_id: str) -> dict:
    """Qdrant 에서 기생충 payload 조회."""
    req_body = {
        "ids": [int(movie_id) if movie_id.isdigit() else movie_id],
        "with_payload": True,
        "with_vector": False,
    }
    req = urllib.request.Request(
        f"{settings.QDRANT_URL}/collections/{settings.QDRANT_COLLECTION}/points",
        data=json.dumps(req_body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        d = json.loads(resp.read().decode())
    points = d.get("result", [])
    if not points:
        return {}
    return points[0].get("payload", {}) or {}


def _qdrant_set_payload(movie_id: str, update: dict) -> bool:
    """Qdrant 포인트 payload 부분 업데이트."""
    try:
        point_id: int | str = int(movie_id) if movie_id.isdigit() else movie_id
    except (ValueError, TypeError):
        point_id = movie_id

    body = {"payload": update, "points": [point_id]}
    req = urllib.request.Request(
        f"{settings.QDRANT_URL}/collections/{settings.QDRANT_COLLECTION}/points/payload",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            d = json.loads(resp.read().decode())
        return d.get("status") == "ok"
    except Exception as e:
        print(f"  [ERROR] Qdrant set_payload: {e}")
        return False


async def _snapshot_neo4j(movie_id: str) -> dict:
    """Neo4j 에서 기생충 Movie 노드 속성 조회."""
    driver = await get_neo4j()
    async with driver.session() as session:
        r = await session.run(
            "MATCH (m:Movie {id: $id}) RETURN m",
            {"id": str(movie_id)},
        )
        record = await r.single()
        if not record:
            return {}
        node = record["m"]
        return dict(node.items())


async def _snapshot_es(movie_id: str) -> dict:
    """ES 에서 기생충 문서 _source 조회."""
    es = await get_elasticsearch()
    try:
        doc = await es.get(index="movies_bm25", id=movie_id)
        return doc.get("_source", {})
    except Exception as e:
        print(f"  [ERROR] ES get: {e}")
        return {}


def _print_snapshot(label: str, snaps: dict) -> None:
    print(f"\n  [{label}]")
    for db, data in snaps.items():
        print(f"    {db}:")
        for k in ("title", "title_en", "director", "director_original_name",
                  "kobis_movie_cd", "kmdb_id", "kobis_nation", "awards",
                  "filming_location", "kobis_watch_grade"):
            v = data.get(k)
            if v:
                vs = str(v)[:60]
                print(f"      {k:30s} {vs}")
        cast_en = data.get("cast_original_names") or []
        if cast_en:
            print(f"      cast_original_names           {cast_en[:5]}")
        staffs = data.get("kobis_staffs") or []
        if staffs:
            print(f"      kobis_staffs count            {len(staffs)}")


async def main() -> None:
    print("=" * 70)
    print("  C-1 기생충 E2E 테스트 — KOBIS + KMDb → 5DB enrichment 검증")
    print("=" * 70)

    if not settings.KOBIS_API_KEY:
        print("❌ KOBIS_API_KEY 없음")
        return
    if not settings.KMDB_API_KEY:
        print("❌ KMDB_API_KEY 없음")
        return

    await init_all_clients()

    try:
        # ── Step 1: Before 스냅샷 ──
        print("\n[Step 1] Before 스냅샷 — 5DB 기생충 현재 상태")
        before = {
            "Qdrant": _snapshot_qdrant(PARASITE_TMDB_ID),
            "Neo4j": await _snapshot_neo4j(PARASITE_TMDB_ID),
            "ES": await _snapshot_es(PARASITE_TMDB_ID),
        }
        _print_snapshot("BEFORE", before)

        # ── Step 2: KOBIS 실 API 수집 ──
        print("\n[Step 2] KOBIS API 수집 — movieCd=20183782")
        async with KOBISCollector() as kobis:
            kobis_detail = await kobis.collect_movie_detail(PARASITE_KOBIS_CD)
            print(f"  movie_nm: {kobis_detail.movie_nm}")
            print(f"  movie_nm_en: {kobis_detail.movie_nm_en}")
            print(f"  directors: {[(d.get('peopleNm'), d.get('peopleNmEn')) for d in kobis_detail.directors[:2]]}")
            print(f"  actors count: {len(kobis_detail.actors)}")
            print(f"  staffs count: {len(kobis_detail.staffs)}")
            print(f"  watch_grade_nm: {kobis_detail.watch_grade_nm}")

        # KOBIS 목록 API 응답 dict 형태로 wrapping
        kobis_raw = {
            "movieCd": kobis_detail.movie_cd,
            "movieNm": kobis_detail.movie_nm,
            "movieNmEn": kobis_detail.movie_nm_en,
            "openDt": kobis_detail.open_dt,
            "nationAlt": "한국",  # 목록 API 필드
            "typeNm": kobis_detail.type_nm,
        }
        kobis_detail_dict = {
            "actors": kobis_detail.actors,
            "directors": kobis_detail.directors,
            "staffs": kobis_detail.staffs,
            "companys": kobis_detail.companys,
            "audits": kobis_detail.audits,
            "show_tm": kobis_detail.show_tm,
            "genres": kobis_detail.genres,
        }

        kobis_enrichment = build_kobis_enrichment_payload(
            kobis_raw=kobis_raw,
            detail_data=kobis_detail_dict,
            boxoffice_data=None,
        )
        print(f"\n  → build_kobis_enrichment_payload 결과 키: {sorted(kobis_enrichment.keys())}")
        print(f"  title_en: {kobis_enrichment.get('title_en')}")
        print(f"  director_original_name: {kobis_enrichment.get('director_original_name')}")
        print(f"  cast_original_names: {kobis_enrichment.get('cast_original_names', [])[:5]}")

        # ── Step 3: KMDb 실 API 수집 ──
        print("\n[Step 3] KMDb API 수집 — 기생충 2019")
        async with KMDbCollector() as kmdb:
            kmdb_movies = await kmdb.collect_all_movies(start_year=2019, end_year=2019)
            parasite_kmdb = None
            for m in kmdb_movies:
                if m.title.strip() == "기생충" and any(
                    "봉준호" in d.get("directorNm", "") for d in m.directors
                ):
                    parasite_kmdb = m
                    break
        if not parasite_kmdb:
            print("  ❌ KMDb 기생충 찾기 실패 (사전 수집 한도 초과 가능)")
            kmdb_enrichment = {}
        else:
            print(f"  title: {parasite_kmdb.title}")
            print(f"  titleEng: {parasite_kmdb.title_eng}")
            print(f"  directors: {[(d.get('directorNm'), d.get('directorEnNm')) for d in parasite_kmdb.directors[:2]]}")
            print(f"  actors count: {len(parasite_kmdb.actors)}")
            print(f"  staffs count: {len(parasite_kmdb.staffs)}")

            kmdb_enrichment = build_kmdb_full_enrichment_payload(parasite_kmdb)
            print(f"\n  → build_kmdb_full_enrichment_payload 결과 키: {sorted(kmdb_enrichment.keys())}")
            print(f"  title_en: {kmdb_enrichment.get('title_en')}")
            print(f"  director_original_name: {kmdb_enrichment.get('director_original_name')}")
            print(f"  cast_original_names: {kmdb_enrichment.get('cast_original_names', [])[:5]}")
            print(f"  awards len: {len(kmdb_enrichment.get('awards', ''))}")
            print(f"  kmdb_staffs count: {len(kmdb_enrichment.get('kmdb_staffs', []))}")

        # ── Step 4: 병합 (KOBIS + KMDb → 최종 enrichment) ──
        # KMDb 가 더 정확한 영문명 (Bong Joon-ho) 을 제공하므로 우선
        print("\n[Step 4] enrichment 병합 (KMDb 우선)")
        merged = {**kobis_enrichment, **kmdb_enrichment}
        # runtime_kobis 제거
        merged.pop("runtime_kobis", None)
        print(f"  병합 후 키 {len(merged)}: {sorted(merged.keys())}")

        # ── Step 5: 5DB 동기화 ──
        print("\n[Step 5] 5DB 동기화 적용")

        # 5-1. Qdrant set_payload
        print("  Qdrant...")
        q_ok = _qdrant_set_payload(PARASITE_TMDB_ID, merged)
        print(f"  Qdrant: {'✅' if q_ok else '❌'}")

        # 5-2. ES update
        print("  ES...")
        es_ok = await update_movie_partial(PARASITE_TMDB_ID, merged)
        print(f"  ES: {'✅' if es_ok else '❌'}")

        # 5-3. Neo4j SET 속성
        print("  Neo4j...")
        n_ok = await update_movie_properties(PARASITE_TMDB_ID, merged)
        print(f"  Neo4j: {'✅' if n_ok else '❌'}")

        # MySQL 은 Phase 3 의 run_mysql_sync 에서 전파 (Qdrant → MySQL)
        print("  MySQL: (Phase 3 run_mysql_sync 에서 전파 예정)")

        # ── Step 6: After 스냅샷 ──
        print("\n[Step 6] After 스냅샷 — 동기화 결과 검증")
        after = {
            "Qdrant": _snapshot_qdrant(PARASITE_TMDB_ID),
            "Neo4j": await _snapshot_neo4j(PARASITE_TMDB_ID),
            "ES": await _snapshot_es(PARASITE_TMDB_ID),
        }
        _print_snapshot("AFTER", after)

        # ── Step 7: PASS/FAIL 판정 ──
        print("\n[Step 7] PASS/FAIL 판정")
        checks = []

        # title_en 영문 전환 확인
        for db in ("Qdrant", "Neo4j", "ES"):
            before_te = before[db].get("title_en", "") or ""
            after_te = after[db].get("title_en", "") or ""
            passed = "Parasite" in after_te or "parasite" in after_te.lower()
            checks.append((f"{db} title_en = Parasite*", passed, before_te, after_te))

        # director_original_name 영문
        for db in ("Qdrant", "Neo4j", "ES"):
            before_d = before[db].get("director_original_name", "") or ""
            after_d = after[db].get("director_original_name", "") or ""
            passed = "Bong" in after_d or "BONG" in after_d
            checks.append((f"{db} director_original = Bong*", passed, before_d, after_d))

        # kobis_movie_cd 적용
        for db in ("Qdrant", "Neo4j", "ES"):
            after_cd = after[db].get("kobis_movie_cd", "") or ""
            passed = after_cd == PARASITE_KOBIS_CD
            checks.append((f"{db} kobis_movie_cd = {PARASITE_KOBIS_CD}", passed, "", after_cd))

        pass_count = sum(1 for _, p, _, _ in checks if p)
        fail_count = len(checks) - pass_count

        for name, passed, before_v, after_v in checks:
            icon = "✅" if passed else "❌"
            print(f"  {icon} {name}")
            if not passed:
                print(f"      before: '{before_v}' → after: '{after_v}'")

        print()
        print("=" * 70)
        print(f"  결과: {pass_count}/{len(checks)} PASS, {fail_count} FAIL")
        print("=" * 70)

    finally:
        await close_all_clients()


if __name__ == "__main__":
    asyncio.run(main())
