"""
Neo4j 재동기화 스크립트 — Qdrant 기준으로 누락된 Movie 노드 복구.

사용 시나리오:
    Task #5 (run_full_reload.py) 후반부에 Neo4j 컨테이너가 OOM 등으로 다운
    되어 마지막 청크들의 Neo4j 적재가 실패한 경우, Qdrant/ES 에는 적재됐지만
    Neo4j Movie 카운트가 부족해진다. 이 스크립트는:

    1. Neo4j 에서 기존 Movie id 전체 추출 (set)
    2. Qdrant 에서 전체 point scroll → Neo4j 에 없는 id 식별
    3. 누락 영화의 payload 를 MovieDocument 로 복원
    4. load_to_neo4j() 로 배치 단위 재적재 (9 노드 + 19 관계 MERGE)

    MERGE 기반이므로 이미 있는 노드/관계는 중복 생성되지 않는다.

설계 진실 원본:
    docs/Phase_ML4_재적재_진행상황_세션인계.md §8.1 Neo4j OOM 대응

사용법:
    # 전체 누락분 복구 (자동)
    PYTHONPATH=src uv run python scripts/run_neo4j_resync.py

    # 배치 크기 조정
    PYTHONPATH=src uv run python scripts/run_neo4j_resync.py --batch-size 1000

    # 미리보기 (Qdrant scroll + 차집합만)
    PYTHONPATH=src uv run python scripts/run_neo4j_resync.py --dry-run

    # 특정 수만 복구 (테스트)
    PYTHONPATH=src uv run python scripts/run_neo4j_resync.py --limit 100

안전성:
    - load_to_neo4j 가 모두 MERGE 기반이므로 중복 생성 불가
    - 실패 시 --resume 없이 재실행 해도 idempotent
    - Task #5 가 아직 진행 중이면 Neo4j 쓰기 경합 주의 (본 스크립트는 Task #5
      종료 후 실행 권장)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트 sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# .env 명시 로드
_env_file = _project_root / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import structlog  # noqa: E402

from monglepick.config import settings  # noqa: E402
from monglepick.data_pipeline.models import MovieDocument  # noqa: E402
from monglepick.data_pipeline.neo4j_loader import load_to_neo4j  # noqa: E402
from monglepick.db.clients import close_all_clients, get_neo4j, init_all_clients  # noqa: E402

logger = structlog.get_logger()

DEFAULT_BATCH_SIZE = 500


# ══════════════════════════════════════════════════════════════
# Neo4j 기존 Movie id 추출
# ══════════════════════════════════════════════════════════════


async def fetch_neo4j_movie_ids() -> set[str]:
    """
    Neo4j (:Movie) 노드의 id 전체를 set 으로 추출.

    수만~수백만 건이므로 한 번에 RETURN. Neo4j Community 5 에서 1M 건도
    Python set 에 들어가면 메모리 OK.
    """
    driver = await get_neo4j()
    ids: set[str] = set()

    async with driver.session() as session:
        result = await session.run(
            "MATCH (m:Movie) WHERE m.id IS NOT NULL RETURN m.id AS id"
        )
        async for record in result:
            mid = record["id"]
            if mid is not None:
                ids.add(str(mid))

    logger.info("neo4j_movie_ids_loaded", count=len(ids))
    return ids


# ══════════════════════════════════════════════════════════════
# Qdrant → MovieDocument 복원
# ══════════════════════════════════════════════════════════════


def _payload_to_movie_document(point_id: int | str, payload: dict) -> MovieDocument | None:
    """
    Qdrant payload 를 MovieDocument 로 복원.

    qdrant_loader._movie_to_point 과 역변환 관계.
    id 필드는 payload 에 없으므로 point_id 에서 복원.

    Returns:
        MovieDocument 또는 None (필수 필드 누락 시)
    """
    # 필수 필드 확인
    title = payload.get("title", "") or ""
    if not title:
        return None

    # Pydantic model_validate 는 extra='ignore' 기본
    # payload 에 MovieDocument 에 없는 키가 있어도 무시됨.
    data = dict(payload)
    data["id"] = str(point_id)
    data["title"] = title

    # 기본값으로 채워야 하는 필드들
    data.setdefault("embedding_text", "")

    # int/float 타입 보정 (Qdrant 가 일부 숫자를 문자열로 반환할 수 있음)
    for int_field in ("release_year", "runtime", "vote_count",
                      "collection_id", "director_id", "budget", "revenue",
                      "audience_count", "sales_acc", "screen_count",
                      "tmdb_list_count"):
        v = data.get(int_field)
        if v is not None and not isinstance(v, int):
            try:
                data[int_field] = int(v) if v not in ("", None) else 0
            except (ValueError, TypeError):
                data[int_field] = 0

    for float_field in ("rating", "popularity_score"):
        v = data.get(float_field)
        if v is not None and not isinstance(v, (int, float)):
            try:
                data[float_field] = float(v) if v not in ("", None) else 0.0
            except (ValueError, TypeError):
                data[float_field] = 0.0

    # list/dict 기본값 보정
    for list_field in ("genres", "cast", "keywords", "mood_tags", "ott_platforms",
                       "cast_characters", "production_companies", "production_countries",
                       "production_country_names", "spoken_languages", "spoken_language_names",
                       "origin_country", "alternative_titles", "recommendation_ids",
                       "similar_movie_ids", "reviews", "behind_the_scenes",
                       "kobis_directors", "kobis_actors", "kobis_companies",
                       "kobis_staffs", "kobis_genres", "images_posters",
                       "images_backdrops", "images_logos", "executive_producers",
                       "screenwriters", "producers", "stills"):
        if data.get(list_field) is None:
            data[list_field] = []

    try:
        return MovieDocument(**data)
    except Exception as e:
        logger.debug(
            "payload_to_movie_doc_failed",
            id=point_id,
            title=title[:40],
            error=str(e)[:100],
        )
        return None


# ══════════════════════════════════════════════════════════════
# Qdrant scroll → 누락 ID 추출 → 배치 재적재
# ══════════════════════════════════════════════════════════════


async def run_resync(
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    """
    Neo4j 재동기화 메인.

    Args:
        batch_size: load_to_neo4j 배치 크기 (기본 500)
        limit: 최대 처리 건수 (테스트용)
        dry_run: True 이면 차집합만 출력하고 종료
    """
    pipeline_start = time.time()

    # ── 초기화 ──
    await init_all_clients()

    try:
        # ── Step 1: Neo4j 기존 Movie id 추출 ──
        print("[Step 1] Neo4j 기존 Movie id 로드")
        neo4j_ids = await fetch_neo4j_movie_ids()
        print(f"  Neo4j Movie 노드: {len(neo4j_ids):,}건")

        # ── Step 2: Qdrant scroll → 누락 ID 식별 ──
        print("\n[Step 2] Qdrant scroll → 누락 ID 식별")
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
        missing_points: list[tuple[int | str, dict]] = []
        qdrant_total = 0
        offset = None

        while True:
            points, next_offset = client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=1000,
                offset=offset,
                with_vectors=False,
                with_payload=True,
            )
            if not points:
                break

            for p in points:
                qdrant_total += 1
                pid_str = str(p.id)
                if pid_str not in neo4j_ids:
                    missing_points.append((p.id, p.payload or {}))
                    if limit and len(missing_points) >= limit:
                        break

            if limit and len(missing_points) >= limit:
                break
            if next_offset is None:
                break
            offset = next_offset

            # 진행률 10만 건마다 출력
            if qdrant_total % 100000 == 0:
                print(
                    f"  scroll 진행: {qdrant_total:,}건 / 누락 {len(missing_points):,}건"
                )

        client.close()

        print(f"\n  Qdrant 전체 scroll: {qdrant_total:,}건")
        print(f"  Neo4j 누락 (차집합): {len(missing_points):,}건")
        print(f"  Neo4j 재적재 대상률: {len(missing_points) * 100 / max(qdrant_total, 1):.2f}%")

        if not missing_points:
            print("\n  ✅ 누락 없음 — 재동기화 불필요.")
            return

        if dry_run:
            print(f"\n[DRY-RUN] 실제 재적재 하지 않음. 샘플 5건:")
            for pid, payload in missing_points[:5]:
                print(f"  - id={pid} title={payload.get('title', '')[:40]}")
            return

        # ── Step 3: payload → MovieDocument 복원 ──
        print(f"\n[Step 3] MovieDocument 복원 ({len(missing_points):,}건)")
        documents: list[MovieDocument] = []
        failed = 0
        for pid, payload in missing_points:
            doc = _payload_to_movie_document(pid, payload)
            if doc:
                documents.append(doc)
            else:
                failed += 1

        print(f"  복원 성공: {len(documents):,}건")
        print(f"  복원 실패: {failed:,}건")

        if not documents:
            print("  ❌ 복원 가능한 문서가 없습니다.")
            return

        # ── Step 4: Neo4j 배치 적재 ──
        print(f"\n[Step 4] Neo4j 재적재 (배치: {batch_size})")
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch_idx = i // batch_size + 1
            batch = documents[i : i + batch_size]
            batch_start = time.time()

            try:
                await load_to_neo4j(batch)
            except Exception as e:
                logger.error(
                    "neo4j_batch_load_failed",
                    batch=batch_idx,
                    error=str(e)[:200],
                )
                continue

            elapsed = time.time() - batch_start
            print(
                f"  [Batch {batch_idx:>4}/{total_batches}] "
                f"적재 {len(batch):>5}건 | 소요 {elapsed:>5.1f}s"
            )

        # ── Step 5: 최종 검증 ──
        print("\n[Step 5] 최종 Neo4j 카운트 검증")
        final_ids = await fetch_neo4j_movie_ids()
        print(f"  Neo4j Movie (재적재 후): {len(final_ids):,}건")
        print(f"  증분: +{len(final_ids) - len(neo4j_ids):,}건")
        print(f"  Qdrant 대비: {len(final_ids):,} / {qdrant_total:,} ({len(final_ids)*100/qdrant_total:.2f}%)")

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[Neo4j 재동기화 완료] 소요: {total_elapsed / 60:.1f}분")
        print(f"{'=' * 60}")

    finally:
        await close_all_clients()


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neo4j 재동기화 (Qdrant 기준으로 누락 Movie 복구)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"load_to_neo4j 배치 크기 (기본: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="최대 처리 건수 (테스트용)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="차집합만 출력하고 종료 (재적재 안 함)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_resync(
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    )
