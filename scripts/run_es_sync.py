"""
Elasticsearch 동기화 스크립트 (Qdrant → ES 보충 적재).

Qdrant에는 있지만 ES에는 누락된 문서를 찾아 ES에 적재한다.
임베딩은 불필요하며, Qdrant payload를 MovieDocument로 변환하여 ES에 넣는다.

배경:
    ES 컨테이너가 중간에 종료되면서 일부 배치 적재가 유실될 수 있다.
    Qdrant와 Neo4j는 정상(806K)인데 ES만 764K인 경우 이 스크립트로 보충한다.

사용법:
    # 기본 실행 (Qdrant vs ES 비교 → 누락분 적재)
    PYTHONPATH=src uv run python scripts/run_es_sync.py

    # 배치 크기 조정
    PYTHONPATH=src uv run python scripts/run_es_sync.py --batch-size 500

    # 건수 확인만 (적재 없이 누락 건수만 출력)
    PYTHONPATH=src uv run python scripts/run_es_sync.py --dry-run
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog  # noqa: E402

from monglepick.data_pipeline.es_loader import load_to_elasticsearch  # noqa: E402
from monglepick.data_pipeline.models import MovieDocument  # noqa: E402
from monglepick.db.clients import (  # noqa: E402
    init_all_clients,
    close_all_clients,
    get_elasticsearch,
    ES_INDEX_NAME,
)
from monglepick.config import settings  # noqa: E402

logger = structlog.get_logger()

DEFAULT_BATCH_SIZE = 500


# ============================================================
# Qdrant 전체 ID 목록 조회
# ============================================================

def _get_qdrant_ids() -> set[str]:
    """Qdrant에서 모든 포인트 ID를 수집한다."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
    ids: set[str] = set()
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_vectors=False,
            with_payload=False,
        )
        if not points:
            break

        for p in points:
            ids.add(str(p.id))

        if next_offset is None:
            break
        offset = next_offset

    client.close()
    return ids


# ============================================================
# ES 전체 ID 목록 조회
# ============================================================

async def _get_es_ids() -> set[str]:
    """ES에서 모든 문서 ID를 수집한다 (scroll API 사용)."""
    es = await get_elasticsearch()
    ids: set[str] = set()
    scroll_id: str | None = None

    try:
        # scroll API로 전체 ID 조회 (source 없이 ID만)
        resp = await es.search(
            index=ES_INDEX_NAME,
            query={"match_all": {}},
            _source=False,
            scroll="2m",
            size=5000,
        )

        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]

        while hits:
            for hit in hits:
                ids.add(str(hit["_id"]))

            resp = await es.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = resp["_scroll_id"]
            hits = resp["hits"]["hits"]

    finally:
        # scroll 컨텍스트 정리 (scroll_id가 있을 때만)
        if scroll_id:
            try:
                await es.clear_scroll(scroll_id=scroll_id)
            except Exception:
                pass

    return ids


# ============================================================
# Qdrant payload → MovieDocument 변환
# ============================================================

def _payload_to_movie_document(point_id: str, payload: dict) -> MovieDocument:
    """
    Qdrant 포인트 payload를 MovieDocument로 변환한다.

    Phase ML-4 보강 (2026-04-07):
        기존 구현은 명시적으로 열거한 필드(약 35개)만 MovieDocument에 매핑했고,
        overview_en / alternative_titles / screenwriters / director_original_name /
        cast_characters / reviews / kmdb_* / kobis_directors / images_* 등 약 40개
        필드를 누락했다. 이로 인해 ES 재동기화 시 Phase ML-1/ML-2에서 추가된
        다국어 검색 필드(alternative_titles standard analyzer, overview_en
        multi_match 등)가 전부 빈 값으로 들어가 검색 품질이 오히려 저하된다.

        본 구현은 Pydantic v2의 model_validate()를 사용하여 payload dict를
        MovieDocument로 통째 매핑한다. Qdrant payload(qdrant_loader._movie_to_point)
        는 MovieDocument의 모든 필드를 저장하므로, payload를 그대로 모델에
        주입하면 누락 없이 복원된다. MovieDocument에 정의되지 않은 키는
        Pydantic 기본 설정(extra="ignore")으로 무시된다.

    처리 흐름:
        1. payload dict 복사 (원본 불변)
        2. id 보충 (payload["id"]가 없으면 point_id로 대체)
        3. cast 타입 정규화 (list[dict] → list[str], 구버전 호환)
        4. MovieDocument.model_validate()로 자동 매핑 + 검증
        5. 실패 시 최소 필드 fallback 후 warning 로깅

    Args:
        point_id: Qdrant PointStruct의 id (int 또는 uuid5 문자열)
        payload: Qdrant PointStruct의 payload dict

    Returns:
        완전히 복원된 MovieDocument (40+ 필드 보존)
    """
    # 1) payload 복사 — 원본 dict 수정 방지
    data = dict(payload)

    # 2) id 보충 — Qdrant payload에는 "id" 키가 없을 수 있으므로 point_id로 보완
    if not data.get("id"):
        data["id"] = point_id
    else:
        data["id"] = str(data["id"])

    # 3) cast 타입 정규화
    # Phase ML-2 이전 적재본은 cast를 list[dict](cast_characters 형태)로
    # 저장했을 수 있다. MovieDocument.cast는 list[str]이므로 dict인 경우
    # name만 추출한다. Phase ML-2 이후는 list[str]이므로 그대로 사용한다.
    cast_raw = data.get("cast") or []
    if isinstance(cast_raw, list) and cast_raw and isinstance(cast_raw[0], dict):
        data["cast"] = [
            c.get("name", "")
            for c in cast_raw
            if isinstance(c, dict) and c.get("name")
        ]
    elif not isinstance(cast_raw, list):
        # 예상치 못한 타입은 빈 리스트로 정규화
        data["cast"] = []

    # 4) Pydantic model_validate로 자동 매핑
    # MovieDocument는 extra="ignore" 기본 설정이므로 Qdrant payload에만 존재하는
    # 키(point_id 등)는 자동 무시된다. 필드 검증 실패 시 ValidationError 발생.
    try:
        return MovieDocument.model_validate(data)
    except Exception as e:
        # 5) fallback — 검증 실패 시 최소 필드만으로 생성하여 배치 진행 유지
        logger.warning(
            "payload_validate_failed_fallback",
            id=data.get("id"),
            error=str(e)[:200],
        )
        return MovieDocument(
            id=str(data.get("id", point_id)),
            title=data.get("title", ""),
        )


# ============================================================
# Qdrant에서 누락 ID의 payload 배치 조회
# ============================================================

def _get_qdrant_payloads(missing_ids: list[str], batch_size: int = 100) -> list[tuple[str, dict]]:
    """
    Qdrant에서 지정된 ID들의 payload를 배치 조회한다.

    Args:
        missing_ids: 조회할 ID 목록
        batch_size: 한 번에 조회할 건수

    Returns:
        [(point_id, payload), ...] 리스트
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
    results: list[tuple[str, dict]] = []

    for i in range(0, len(missing_ids), batch_size):
        batch_ids = missing_ids[i:i + batch_size]

        # ID 타입 변환 (숫자면 int, 아니면 str)
        typed_ids = []
        for pid in batch_ids:
            try:
                typed_ids.append(int(pid))
            except ValueError:
                typed_ids.append(pid)

        points = client.retrieve(
            collection_name=settings.QDRANT_COLLECTION,
            ids=typed_ids,
            with_vectors=False,
            with_payload=True,
        )

        for p in points:
            results.append((str(p.id), p.payload or {}))

    client.close()
    return results


# ============================================================
# 메인 파이프라인
# ============================================================

async def run_es_sync(
    batch_size: int = DEFAULT_BATCH_SIZE,
    dry_run: bool = False,
) -> None:
    """
    Qdrant → ES 동기화.

    1. Qdrant 전체 ID 수집
    2. ES 전체 ID 수집
    3. 차집합 (Qdrant에만 있는 ID) 계산
    4. 누락 ID의 payload를 Qdrant에서 조회
    5. MovieDocument 변환 → ES 적재

    Args:
        batch_size: ES 적재 배치 크기
        dry_run: True이면 누락 건수만 출력 (적재 안 함)
    """
    pipeline_start = time.time()

    # ── Step 0: 초기화 ──
    await init_all_clients()

    try:
        # ── Step 1: Qdrant ID 수집 ──
        print("[Step 1] Qdrant 전체 ID 수집")
        qdrant_ids = _get_qdrant_ids()
        print(f"  Qdrant: {len(qdrant_ids):,}건")

        # ── Step 2: ES ID 수집 ──
        print("\n[Step 2] ES 전체 ID 수집")
        es_ids = await _get_es_ids()
        print(f"  ES: {len(es_ids):,}건")

        # ── Step 3: 차집합 계산 ──
        missing_ids = qdrant_ids - es_ids
        extra_ids = es_ids - qdrant_ids

        print(f"\n[Step 3] 비교 결과")
        print(f"  ES 누락 (Qdrant에만 있음): {len(missing_ids):,}건")
        print(f"  ES 초과 (Qdrant에 없음):   {len(extra_ids):,}건")

        if not missing_ids:
            print("\n  ES와 Qdrant가 동기화되어 있습니다.")
            return

        if dry_run:
            print(f"\n  [Dry Run] {len(missing_ids):,}건 누락 확인 (적재 스킵)")
            return

        # ── Step 4 & 5: payload 조회 → ES 적재 ──
        missing_list = sorted(missing_ids)
        total_loaded = 0
        total_batches = (len(missing_list) + batch_size - 1) // batch_size

        print(f"\n[Step 4-5] 누락분 ES 적재 ({len(missing_list):,}건, 배치: {batch_size})")

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(missing_list))
            batch_ids = missing_list[batch_start:batch_end]
            batch_start_time = time.time()

            # Qdrant에서 payload 조회
            payloads = _get_qdrant_payloads(batch_ids)

            # MovieDocument 변환
            documents = []
            for pid, payload in payloads:
                try:
                    doc = _payload_to_movie_document(pid, payload)
                    documents.append(doc)
                except Exception as e:
                    logger.warning("payload_convert_failed", id=pid, error=str(e))

            # ES 적재
            if documents:
                es_count = await load_to_elasticsearch(documents)
                total_loaded += es_count
            else:
                es_count = 0

            batch_elapsed = time.time() - batch_start_time
            remaining = total_batches - batch_idx - 1
            eta = batch_elapsed * remaining if remaining > 0 else 0

            print(
                f"  [Batch {batch_idx + 1:>4}/{total_batches}] "
                f"{es_count:,}건 적재 | "
                f"소요: {batch_elapsed:.1f}s | ETA: {eta / 60:.1f}m"
            )

        # ── 완료 ──
        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[ES 동기화 완료]")
        print(f"  누락 발견: {len(missing_ids):>10,}건")
        print(f"  적재 완료: {total_loaded:>10,}건")
        print(f"  소요:      {total_elapsed / 60:>10.1f}분")
        print(f"{'=' * 60}")

    finally:
        await close_all_clients()


# ============================================================
# 진입점
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Elasticsearch 동기화 (Qdrant → ES 누락분 보충)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 동기화 실행
  PYTHONPATH=src uv run python scripts/run_es_sync.py

  # 누락 건수 확인만
  PYTHONPATH=src uv run python scripts/run_es_sync.py --dry-run
        """,
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"ES 적재 배치 크기 (기본: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="누락 건수만 확인 (적재 안 함)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_es_sync(
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    )
