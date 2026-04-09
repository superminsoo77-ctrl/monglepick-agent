"""
KMDb (한국영화데이터베이스) 데이터 수집 + 적재 스크립트.

KMDb API에서 한국영화를 수집하고, 기존 DB 영화와 매칭하여:
  - 매칭 성공: 기존 영화에 KMDb 고유 데이터 보강 (수상내역, 관객수, 촬영장소 등)
  - 매칭 실패: 신규 MovieDocument로 생성하여 3DB 적재

사용법:
    # 기본 실행 (2000~현재년도 수집)
    PYTHONPATH=src uv run python scripts/run_kmdb_load.py

    # 연도 범위 지정
    PYTHONPATH=src uv run python scripts/run_kmdb_load.py --start-year 1960 --end-year 2025

    # 캐시 사용 (이전 수집 결과 재활용)
    PYTHONPATH=src uv run python scripts/run_kmdb_load.py --use-cache

    # 적재 배치 크기 조정
    PYTHONPATH=src uv run python scripts/run_kmdb_load.py --batch-size 1000

    # 현재 진행 상태 확인
    PYTHONPATH=src uv run python scripts/run_kmdb_load.py --status

소요 시간 추정:
    - 수집: ~1시간 (43K건, KMDb API 일일 1,000건 제한)
    - 매칭: ~5분 (타이틀 인덱스 구축 + 매칭)
    - 임베딩: ~10분 (신규 ~36K건, Upstage 100 RPM)
    - 적재: ~10분 (3DB 병렬)
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog  # noqa: E402

from monglepick.data_pipeline.embedder import embed_texts  # noqa: E402
from monglepick.data_pipeline.es_loader import (  # noqa: E402
    load_to_elasticsearch,
    update_movies_partial_bulk,
)
from monglepick.data_pipeline.kmdb_collector import KMDbCollector  # noqa: E402
from monglepick.data_pipeline.kmdb_enricher import (  # noqa: E402
    build_kmdb_full_enrichment_payload,
    build_title_index,
    process_kmdb_batch,
)
# Phase ML-4 일관성: Solar Pro 3 배치 무드태그 + embedding_text 재생성
from monglepick.data_pipeline.mood_batch import enrich_documents_with_solar_mood  # noqa: E402
from monglepick.data_pipeline.neo4j_loader import (  # noqa: E402
    load_to_neo4j,
    update_movies_properties_bulk,
)
from monglepick.data_pipeline.qdrant_loader import load_to_qdrant  # noqa: E402
from monglepick.db.clients import init_all_clients, close_all_clients  # noqa: E402
from monglepick.config import settings  # noqa: E402

logger = structlog.get_logger()

# ── 경로 및 상수 ──
CACHE_DIR = Path("data")
KMDB_CACHE_PATH = CACHE_DIR / "kmdb_movies_cache.json"
CHECKPOINT_PATH = CACHE_DIR / "kmdb_load_checkpoint.json"
DEFAULT_BATCH_SIZE = 2000
DEFAULT_EMBED_BATCH = 50


# ============================================================
# 체크포인트 관리
# ============================================================

def _new_checkpoint() -> dict:
    """새 체크포인트를 생성한다."""
    return {
        "phase": "",               # collect / match / enrich / embed / load / done
        "total_collected": 0,      # KMDb 수집 총 건수
        "total_matched": 0,        # 기존 영화 매칭 건수 (보강 대상)
        "total_new": 0,            # 신규 영화 건수 (적재 대상)
        "total_enriched": 0,       # 보강 완료 건수
        "total_loaded": 0,         # 신규 적재 완료 건수
        "batch_offset": 0,         # 현재 적재 배치 오프셋
        "failed_ids": [],
        "start_time": datetime.now().isoformat(),
        "last_updated": "",
    }


def _load_checkpoint() -> dict:
    """체크포인트 파일을 로드한다."""
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return _new_checkpoint()


def _save_checkpoint(state: dict) -> None:
    """체크포인트 파일에 진행 상태를 저장한다."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))


# ============================================================
# Qdrant에서 기존 영화 목록 조회 (매칭용)
# ============================================================

def _get_existing_movies_from_qdrant() -> list[dict]:
    """
    Qdrant에서 기존 영화 payload를 조회한다.

    KMDb 매칭에 필요한 필드: id, title, title_en, release_year
    build_title_index()에 전달하기 위한 형식으로 반환한다.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
    db_movies: list[dict] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=1000,
            offset=offset,
            with_vectors=False,
            with_payload=["title", "title_en", "release_year", "source"],
        )
        if not points:
            break

        for p in points:
            payload = p.payload or {}
            db_movies.append({
                "id": str(p.id),
                "title": payload.get("title", ""),
                "title_en": payload.get("title_en", ""),
                "release_year": payload.get("release_year", 0),
                "source": payload.get("source", "tmdb"),
            })

        if next_offset is None:
            break
        offset = next_offset

    client.close()
    logger.info("existing_movies_loaded", count=len(db_movies))
    return db_movies


# ============================================================
# Qdrant payload 보강 (기존 영화에 KMDb 데이터 추가)
# ============================================================

async def _apply_enrichments_3db(enrichments: list[dict]) -> int:
    """
    기존 영화에 KMDb 풍부 데이터를 Qdrant + Elasticsearch + Neo4j 3DB 에
    partial update 한다 (2026-04-09 확장).

    기존 `_apply_enrichments` 는 Qdrant 만 갱신했으나, ES/Neo4j 가 뒤처져
    3DB 불일치가 발생했다. 본 함수는 3DB 를 동시에 갱신하여 일관성 보장.

    MySQL 은 Phase 3 의 run_mysql_sync.py 가 Qdrant → MySQL 로 sync 하므로
    여기서는 제외 (Qdrant payload 갱신 → MySQL 자동 전파).

    Args:
        enrichments: [{"existing_id": str, "data": {...}}, ...]
                     process_kmdb_batch 가 반환하는 형식.
                     data 는 extract_enrichment_data 또는
                     build_kmdb_full_enrichment_payload 의 결과.

    Returns:
        Qdrant 에서 실제 업데이트된 건수 (3DB 중 가장 보수적)
    """
    if not enrichments:
        return 0

    # ── 1. enrichment item 을 (movie_id, enrichment_dict) 형태로 정규화 ──
    bulk_updates: list[tuple[str, dict]] = []
    for item in enrichments:
        existing_id = item["existing_id"]
        data = item["data"] or {}

        # 빈 값 필터링
        clean_data = {
            k: v for k, v in data.items()
            if v not in (None, "", [], {}) and v != 0
        }
        if not clean_data:
            continue

        # 특수 키 재매핑 (_kmdb 접미어 등 호출 측 보강)
        # - plot_korean / overview_en_kmdb / cast_original_names_kmdb /
        #   certification_kmdb / trailer_url_kmdb 는 호출 측에서
        #   기존 값이 비어있을 때만 적용해야 하므로 여기서 결정하지 않음.
        #   Qdrant payload 에 _kmdb 필드로 그대로 저장한다.

        bulk_updates.append((str(existing_id), clean_data))

    if not bulk_updates:
        return 0

    # ── 2. Qdrant set_payload (sync executor) ──
    def _sync_qdrant_apply() -> int:
        from qdrant_client import QdrantClient

        sync_client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
        ok = 0

        for existing_id, payload_update in bulk_updates:
            try:
                try:
                    point_id: int | str = int(existing_id)
                except (ValueError, TypeError):
                    point_id = existing_id

                sync_client.set_payload(
                    collection_name=settings.QDRANT_COLLECTION,
                    payload=payload_update,
                    points=[point_id],
                )
                ok += 1

            except Exception as e:
                logger.debug(
                    "kmdb_qdrant_enrich_failed",
                    id=existing_id, error=str(e)[:100],
                )

            if ok > 0 and ok % 1000 == 0:
                logger.info(
                    "kmdb_qdrant_enrich_progress",
                    enriched=ok, total=len(bulk_updates),
                )

        sync_client.close()
        return ok

    loop = asyncio.get_event_loop()
    qdrant_ok = await loop.run_in_executor(None, _sync_qdrant_apply)
    logger.info("kmdb_qdrant_enriched", count=qdrant_ok)

    # ── 3. Elasticsearch bulk partial update ──
    try:
        es_ok = await update_movies_partial_bulk(bulk_updates)
        logger.info("kmdb_es_enriched", count=es_ok)
    except Exception as e:
        logger.warning("kmdb_es_enrich_failed", error=str(e)[:200])

    # ── 4. Neo4j bulk SET 속성 ──
    try:
        neo4j_ok = await update_movies_properties_bulk(bulk_updates)
        logger.info("kmdb_neo4j_enriched", count=neo4j_ok)
    except Exception as e:
        logger.warning("kmdb_neo4j_enrich_failed", error=str(e)[:200])

    logger.info(
        "kmdb_enrichments_applied_3db",
        qdrant=qdrant_ok,
        total=len(bulk_updates),
    )
    return qdrant_ok


# Backwards-compat alias — 기존 호출부 보존
async def _apply_enrichments(enrichments: list[dict]) -> int:
    """Legacy wrapper — 이제 3DB sync 를 수행한다 (2026-04-09)."""
    return await _apply_enrichments_3db(enrichments)


# ============================================================
# KMDb 캐시 저장/로드
# ============================================================

def _save_kmdb_cache(movies: list, cache_path: Path) -> None:
    """KMDb 수집 결과를 JSON 캐시로 저장한다."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = [m.model_dump() if hasattr(m, "model_dump") else m for m in movies]
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("kmdb_cache_saved", count=len(data), path=str(cache_path))


def _load_kmdb_cache(cache_path: Path):
    """캐시된 KMDb 데이터를 로드한다."""
    from monglepick.data_pipeline.models import KMDbRawMovie

    if not cache_path.exists():
        return None

    data = json.loads(cache_path.read_text())
    movies = []
    for item in data:
        try:
            movies.append(KMDbRawMovie(**item))
        except Exception as e:
            logger.warning("cache_parse_error", error=str(e))
    logger.info("kmdb_cache_loaded", count=len(movies))
    return movies


# ============================================================
# 메인 파이프라인
# ============================================================

async def run_kmdb_load(
    start_year: int = 2000,
    end_year: int | None = None,
    use_cache: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embed_batch_size: int = DEFAULT_EMBED_BATCH,
    mood_provider: str = "upstage",
    mood_model: str = "solar-pro3",
    mood_rpm: int = 100,
    mood_concurrency: int = 20,
    mood_batch_size: int = 10,
) -> None:
    """
    KMDb 데이터 수집 → 매칭 → 보강/신규 적재.

    흐름:
    1. KMDb API 수집 (또는 캐시 로드)
    2. Qdrant에서 기존 영화 로드 → 타이틀 인덱스 구축
    3. KMDb ↔ 기존 영화 매칭 (process_kmdb_batch)
       - 매칭 성공 → enrichments (Qdrant payload 갱신)
       - 매칭 실패 → new_documents (신규 적재)
    4. 보강 적용 (Qdrant set_payload)
    5. 신규 영화: 임베딩 → 3DB 적재

    Args:
        start_year: 수집 시작 연도 (기본: 2000)
        end_year: 수집 종료 연도 (기본: 현재 연도)
        use_cache: True이면 기존 캐시 사용
        batch_size: 적재 배치 크기
        embed_batch_size: Upstage 임베딩 배치 크기
    """
    from monglepick.data_pipeline.models import KMDbRawMovie

    if end_year is None:
        end_year = datetime.now().year

    pipeline_start = time.time()
    checkpoint = _load_checkpoint()

    # ── Step 0: DB 클라이언트 초기화 ──
    await init_all_clients()

    try:
        # ── Step 1: KMDb 수집 ──
        print(f"[Step 1] KMDb 영화 수집 ({start_year}~{end_year})")

        kmdb_movies: list[KMDbRawMovie] = []

        if use_cache:
            cached = _load_kmdb_cache(KMDB_CACHE_PATH)
            if cached:
                kmdb_movies = cached
                print(f"  캐시 로드: {len(kmdb_movies):,}건")

        if not kmdb_movies:
            if not settings.KMDB_API_KEY:
                print("  [오류] KMDB_API_KEY가 .env에 설정되지 않았습니다.")
                return

            async with KMDbCollector() as collector:
                kmdb_movies = await collector.collect_all_movies(
                    start_year=start_year,
                    end_year=end_year,
                )
                # KMDbCollector에 public 프로퍼티가 없으므로 내부 카운터 직접 참조
                print(f"  API 수집: {len(kmdb_movies):,}건 (호출: {collector._request_count}회)")

                # 캐시 저장
                _save_kmdb_cache(kmdb_movies, KMDB_CACHE_PATH)

        checkpoint["total_collected"] = len(kmdb_movies)
        checkpoint["phase"] = "collect"
        _save_checkpoint(checkpoint)

        if not kmdb_movies:
            print("  수집된 영화가 없습니다.")
            return

        # ── Step 2: 기존 영화 로드 + 타이틀 인덱스 구축 ──
        print("\n[Step 2] 기존 영화 로드 + 매칭 인덱스 구축")

        db_movies = _get_existing_movies_from_qdrant()
        title_index = build_title_index(db_movies)

        print(f"  기존 영화: {len(db_movies):,}건")
        print(f"  타이틀 인덱스: {len(title_index):,}개 키")

        # ── Step 3: KMDb ↔ 기존 영화 매칭 ──
        print("\n[Step 3] KMDb ↔ 기존 영화 매칭")

        enrichments, new_documents = process_kmdb_batch(kmdb_movies, title_index)

        checkpoint["total_matched"] = len(enrichments)
        checkpoint["total_new"] = len(new_documents)
        checkpoint["phase"] = "match"
        _save_checkpoint(checkpoint)

        print(f"  매칭 성공 (보강 대상): {len(enrichments):,}건")
        print(f"  매칭 실패 (신규 적재): {len(new_documents):,}건")
        print(f"  총 처리:               {len(enrichments) + len(new_documents):,} / {len(kmdb_movies):,}건")

        # ── Step 4: 기존 영화 보강 (Qdrant payload 갱신) ──
        if enrichments:
            print(f"\n[Step 4] 기존 영화 보강 ({len(enrichments):,}건)")

            enriched_count = await _apply_enrichments(enrichments)

            checkpoint["total_enriched"] = enriched_count
            checkpoint["phase"] = "enrich"
            _save_checkpoint(checkpoint)

            print(f"  보강 완료: {enriched_count:,}건")
        else:
            print("\n[Step 4] 보강 대상 없음")

        # ── Step 5 & 6: 신규 영화 mood 보강 + 임베딩 + 적재 ──
        # Phase ML-4 일관성 (2026-04-08): 배치 단위 Solar Pro 3 정밀 mood 적용 후
        # build_embedding_text 재생성 → Solar embedding → 적재. TMDB run_full_reload와 동일.
        if new_documents:
            print(f"\n[Step 5-6] 신규 영화 mood 보강 + 임베딩 + 적재 ({len(new_documents):,}건)")

            # mood_provider 사전 검증: API 키 없으면 fallback
            upstage_api_key: str | None = None
            if mood_provider == "upstage":
                upstage_api_key = (
                    settings.UPSTAGE_API_KEY
                    if hasattr(settings, "UPSTAGE_API_KEY") and settings.UPSTAGE_API_KEY
                    else None
                )
                if upstage_api_key:
                    print(
                        f"  무드 모드: upstage ({mood_model}, "
                        f"rpm={mood_rpm}, concurrency={mood_concurrency}, batch={mood_batch_size})"
                    )
                else:
                    logger.warning("kmdb_upstage_api_key_missing_fallback")
                    mood_provider = "fallback"

            total_loaded = 0
            start_offset = checkpoint.get("batch_offset", 0)
            total_batches = (len(new_documents) - start_offset + batch_size - 1) // batch_size

            for batch_idx, batch_start in enumerate(
                range(start_offset, len(new_documents), batch_size)
            ):
                batch_end = min(batch_start + batch_size, len(new_documents))
                batch = new_documents[batch_start:batch_end]
                batch_start_time = time.time()

                # ── Step 5a (신규): Solar Pro 3 배치 mood 보강 ──
                # kmdb_to_movie_document()가 fallback mood로 채운 mood_tags를
                # Solar Pro 3로 덮어쓰고 embedding_text를 재생성한다.
                if mood_provider == "upstage" and upstage_api_key:
                    try:
                        mood_stats = await enrich_documents_with_solar_mood(
                            documents=batch,
                            api_key=upstage_api_key,
                            model=mood_model,
                            rpm=mood_rpm,
                            concurrency=mood_concurrency,
                            batch_size=mood_batch_size,
                            rebuild_embedding_text=True,
                        )
                        logger.info(
                            "kmdb_batch_mood_enriched",
                            batch_idx=batch_idx,
                            total=mood_stats["total"],
                            enriched=mood_stats["enriched"],
                            elapsed_s=mood_stats["elapsed_s"],
                        )
                    except Exception as e:
                        # mood 실패는 치명적 X — fallback mood로 진행
                        logger.error(
                            "kmdb_batch_mood_failed_continue_with_fallback",
                            batch_idx=batch_idx,
                            error=str(e)[:200],
                        )

                # ── Step 5b: 임베딩 (정밀 mood 반영된 embedding_text 사용) ──
                texts = [doc.embedding_text for doc in batch]
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, embed_texts, texts, embed_batch_size
                )

                # ── Step 6: 3DB 적재 ──
                qdrant_count = await load_to_qdrant(batch, embeddings)
                await load_to_neo4j(batch)
                es_count = await load_to_elasticsearch(batch)

                total_loaded += len(batch)
                batch_elapsed = time.time() - batch_start_time

                # 체크포인트 저장
                checkpoint["batch_offset"] = batch_end
                checkpoint["total_loaded"] = total_loaded
                checkpoint["phase"] = "load"
                _save_checkpoint(checkpoint)

                remaining = total_batches - batch_idx - 1
                eta = batch_elapsed * remaining if remaining > 0 else 0
                print(
                    f"  [Batch {batch_idx + 1:>3}/{total_batches}] "
                    f"{len(batch):,}건 | "
                    f"Qdrant: {qdrant_count} | ES: {es_count} | "
                    f"소요: {batch_elapsed:.1f}s | ETA: {eta / 60:.1f}m"
                )
        else:
            print("\n[Step 5-6] 신규 적재 대상 없음")

        # ── 완료 ──
        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[KMDb 적재 완료]")
        print(f"  수집:      {checkpoint['total_collected']:>10,}건")
        print(f"  보강:      {checkpoint.get('total_enriched', 0):>10,}건")
        print(f"  신규 적재: {checkpoint.get('total_loaded', 0):>10,}건")
        print(f"  소요:      {total_elapsed / 60:>10.1f}분")
        print(f"{'=' * 60}")

    finally:
        await close_all_clients()


# ============================================================
# 상태 조회
# ============================================================

def show_status() -> None:
    """현재 체크포인트 상태를 출력한다."""
    checkpoint = _load_checkpoint()

    print("=" * 60)
    print("  KMDb 적재 파이프라인 상태")
    print("=" * 60)
    print(f"  현재 단계:     {checkpoint.get('phase', '미시작')}")
    print(f"  수집:          {checkpoint.get('total_collected', 0):>10,}건")
    print(f"  매칭 (보강):   {checkpoint.get('total_matched', 0):>10,}건")
    print(f"  신규:          {checkpoint.get('total_new', 0):>10,}건")
    print(f"  보강 완료:     {checkpoint.get('total_enriched', 0):>10,}건")
    print(f"  신규 적재:     {checkpoint.get('total_loaded', 0):>10,}건")
    print(f"  마지막 업데이트: {checkpoint.get('last_updated', '-')}")
    print("=" * 60)


# ============================================================
# 진입점
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KMDb 데이터 수집 + 적재 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (2000~현재)
  PYTHONPATH=src uv run python scripts/run_kmdb_load.py

  # 연도 범위 지정 + 캐시 사용
  PYTHONPATH=src uv run python scripts/run_kmdb_load.py --start-year 1960 --end-year 2025 --use-cache
        """,
    )
    parser.add_argument(
        "--start-year", type=int, default=2000,
        help="수집 시작 연도 (기본: 2000)",
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="수집 종료 연도 (기본: 현재 연도)",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="이전 수집 캐시 사용",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"적재 배치 크기 (기본: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH,
        help=f"Upstage 임베딩 배치 크기 (기본: {DEFAULT_EMBED_BATCH})",
    )
    # Phase ML-4 일관성: 청크 단위 Solar Pro 3 무드태그 통합
    parser.add_argument(
        "--mood-provider",
        choices=["upstage", "fallback"],
        default="upstage",
        help=(
            "청크 단위 무드태그 생성 방식. "
            "'upstage' (기본): Solar Pro 3 배치 API + embedding_text 재생성, "
            "'fallback': kmdb_to_movie_document()의 fallback mood 유지"
        ),
    )
    parser.add_argument(
        "--mood-model", type=str, default="solar-pro3",
        help="Upstage 모델명 (기본: solar-pro3)",
    )
    parser.add_argument(
        "--mood-rpm", type=int, default=100,
        help="Solar API 분당 호출 한도 (기본: 100)",
    )
    parser.add_argument(
        "--mood-concurrency", type=int, default=20,
        help="Solar API 동시 호출 수 (기본: 20)",
    )
    parser.add_argument(
        "--mood-batch-size", type=int, default=10,
        help="1회 Solar API 호출당 영화 수 (기본: 10)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="현재 진행 상태만 확인",
    )
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        asyncio.run(
            run_kmdb_load(
                start_year=args.start_year,
                end_year=args.end_year,
                use_cache=args.use_cache,
                batch_size=args.batch_size,
                embed_batch_size=args.embed_batch_size,
                mood_provider=args.mood_provider,
                mood_model=args.mood_model,
                mood_rpm=args.mood_rpm,
                mood_concurrency=args.mood_concurrency,
                mood_batch_size=args.mood_batch_size,
            )
        )
