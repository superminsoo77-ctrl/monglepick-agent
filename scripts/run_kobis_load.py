"""
KOBIS (영화진흥위원회) 데이터 수집 + 적재 스크립트.

KOBIS API에서 영화 목록/상세/박스오피스를 수집하고,
기존 DB(TMDB/Kaggle)와 중복 제거 후 Qdrant/Neo4j/ES에 적재한다.

사용법:
    # 기본 실행 (캐시 없으면 API 수집, 있으면 캐시 사용)
    PYTHONPATH=src uv run python scripts/run_kobis_load.py

    # 캐시 무시하고 API 재수집
    PYTHONPATH=src uv run python scripts/run_kobis_load.py --no-cache

    # 상세정보 수집 제한 (일일 API 한도 고려)
    PYTHONPATH=src uv run python scripts/run_kobis_load.py --detail-limit 2500

    # 박스오피스 히스토리 수집 (최근 N일)
    PYTHONPATH=src uv run python scripts/run_kobis_load.py --boxoffice-days 365

    # 적재 배치 크기 조정
    PYTHONPATH=src uv run python scripts/run_kobis_load.py --batch-size 1000

    # 현재 진행 상태 확인
    PYTHONPATH=src uv run python scripts/run_kobis_load.py --status

소요 시간 추정:
    - 목록 수집: ~10분 (117K건, 1,170 페이지)
    - 상세 수집: ~20분/2,500건 (일일 한도)
    - 임베딩: ~15분 (77K건, Upstage 100 RPM)
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
from monglepick.data_pipeline.es_loader import load_to_elasticsearch  # noqa: E402
from monglepick.data_pipeline.kobis_collector import (  # noqa: E402
    KOBISCollector,
    save_kobis_cache,
    load_kobis_cache,
)
from monglepick.data_pipeline.kobis_movie_converter import (  # noqa: E402
    build_kobis_enrichment_payload,
    convert_kobis_movies,
    dedup_kobis_movies,  # backwards compat (미사용)
    split_kobis_movies,
)
# Phase ML-4 일관성: Solar Pro 3 배치 무드태그 + embedding_text 재생성
from monglepick.data_pipeline.mood_batch import enrich_documents_with_solar_mood  # noqa: E402
from monglepick.data_pipeline.neo4j_loader import (  # noqa: E402
    load_to_neo4j,
    update_movies_properties_bulk,
)
from monglepick.data_pipeline.qdrant_loader import load_to_qdrant  # noqa: E402
from monglepick.data_pipeline.es_loader import update_movies_partial_bulk  # noqa: E402
from monglepick.db.clients import init_all_clients, close_all_clients  # noqa: E402
from monglepick.config import settings  # noqa: E402

logger = structlog.get_logger()

# ── 경로 및 상수 ──
CACHE_DIR = Path("data")
KOBIS_CACHE_PATH = CACHE_DIR / "kobis_movies_cache.json"
CHECKPOINT_PATH = CACHE_DIR / "kobis_load_checkpoint.json"
DEFAULT_BATCH_SIZE = 2000
DEFAULT_EMBED_BATCH = 50
DEFAULT_DETAIL_LIMIT = 2500  # KOBIS 일일 API 한도 고려


# ============================================================
# 체크포인트 관리
# ============================================================

def _new_checkpoint() -> dict:
    """새 체크포인트를 생성한다."""
    return {
        "phase": "",                # collect / dedup / detail / boxoffice / embed / load / done
        "total_collected": 0,       # 수집된 총 영화 수
        "total_after_dedup": 0,     # 중복 제거 후 영화 수
        "detail_fetched": 0,        # 상세정보 수집 완료 수
        "total_converted": 0,       # MovieDocument 변환 완료 수
        "total_loaded": 0,          # DB 적재 완료 수
        "batch_offset": 0,          # 현재 적재 배치 오프셋
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
# Qdrant에서 기존 영화 목록 조회 (중복 제거용)
# ============================================================

def _get_existing_movies_from_qdrant() -> list[dict]:
    """
    Qdrant에서 기존 영화 payload를 조회한다 (id, title, title_en, release_year, kobis_movie_cd).

    split_kobis_movies() 에 전달할 최소 필드.
    kobis_movie_cd 는 ID 기반 1차 매칭 (이미 KOBIS 적재된 영화 재보강) 용도.
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
            with_payload=["title", "title_en", "release_year", "source", "kobis_movie_cd"],
        )
        if not points:
            break

        for p in points:
            payload = p.payload or {}
            db_movies.append({
                "id": p.id,
                "title": payload.get("title", ""),
                "title_en": payload.get("title_en", ""),
                "release_year": payload.get("release_year", 0),
                "source": payload.get("source", "tmdb"),
                "kobis_movie_cd": payload.get("kobis_movie_cd", ""),
            })

        if next_offset is None:
            break
        offset = next_offset

    client.close()
    logger.info("existing_movies_loaded", count=len(db_movies))
    return db_movies


# ============================================================
# KOBIS 기존 영화 3DB enrichment (2026-04-09)
# ============================================================

async def _apply_kobis_enrichments_3db(
    enrichment_targets: list[tuple[str, dict]],
    detail_map: dict[str, dict],
    boxoffice_map: dict[str, dict],
) -> int:
    """
    split_kobis_movies 가 분류한 enrichment 대상에 KOBIS 풍부 데이터를
    Qdrant + Elasticsearch + Neo4j 3DB 에 partial update 한다.

    MySQL 은 Phase 3 의 run_mysql_sync.py 가 Qdrant → MySQL 로 sync
    하므로 여기서는 Qdrant payload 갱신으로 충분 (MySQL 전파 자동).

    Args:
        enrichment_targets: [(existing_movie_id, kobis_raw_dict), ...]
        detail_map: {movieCd: detail_dict} — searchMovieInfo 응답
        boxoffice_map: {movieCd: boxoffice_dict} — 박스오피스 집계

    Returns:
        Qdrant 에서 실제 업데이트된 건수 (3DB 중 가장 보수적)
    """
    from qdrant_client import QdrantClient

    if not enrichment_targets:
        return 0

    # ── 1. 각 enrichment 대상의 payload 생성 ──
    qdrant_updates: list[tuple[str, dict]] = []  # (point_id, payload)
    bulk_updates: list[tuple[str, dict]] = []     # (movie_id, enrichment_dict) ES/Neo4j 공용

    for existing_id, kobis_raw in enrichment_targets:
        movie_cd = kobis_raw.get("movieCd", "")
        detail_data = detail_map.get(movie_cd)
        boxoffice_data = boxoffice_map.get(movie_cd)

        enrichment = build_kobis_enrichment_payload(
            kobis_raw=kobis_raw,
            detail_data=detail_data,
            boxoffice_data=boxoffice_data,
        )
        if not enrichment:
            continue

        # runtime_kobis 는 내부 용도 — Qdrant payload 로는 제외
        # (기존 영화의 runtime 을 덮어쓰지 않음 — 원본 보존)
        enrichment_for_qdrant = {
            k: v for k, v in enrichment.items() if k != "runtime_kobis"
        }

        qdrant_updates.append((existing_id, enrichment_for_qdrant))
        bulk_updates.append((existing_id, enrichment_for_qdrant))

    if not qdrant_updates:
        return 0

    # ── 2. Qdrant set_payload (sync executor) ──
    def _sync_qdrant_apply() -> int:
        client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)
        ok = 0
        for existing_id, payload_update in qdrant_updates:
            # 빈 값 필터링 (기존 값 덮어쓰기 방지)
            clean = {
                k: v for k, v in payload_update.items()
                if v not in (None, "", [], {}) and v != 0
            }
            if not clean:
                continue

            # Qdrant point_id 타입 추론 (숫자면 int, 아니면 str)
            try:
                pid: int | str = int(existing_id)
            except (ValueError, TypeError):
                pid = str(existing_id)

            try:
                client.set_payload(
                    collection_name=settings.QDRANT_COLLECTION,
                    payload=clean,
                    points=[pid],
                )
                ok += 1
            except Exception as e:
                logger.debug(
                    "kobis_qdrant_enrich_failed",
                    id=existing_id, error=str(e)[:100],
                )

        client.close()
        return ok

    loop = asyncio.get_event_loop()
    qdrant_ok = await loop.run_in_executor(None, _sync_qdrant_apply)
    logger.info("kobis_qdrant_enriched", count=qdrant_ok)

    # ── 3. Elasticsearch bulk partial update ──
    try:
        es_ok = await update_movies_partial_bulk(bulk_updates)
        logger.info("kobis_es_enriched", count=es_ok)
    except Exception as e:
        logger.warning("kobis_es_enrich_failed", error=str(e)[:200])

    # ── 4. Neo4j bulk SET 속성 ──
    try:
        neo4j_ok = await update_movies_properties_bulk(bulk_updates)
        logger.info("kobis_neo4j_enriched", count=neo4j_ok)
    except Exception as e:
        logger.warning("kobis_neo4j_enrich_failed", error=str(e)[:200])

    return qdrant_ok


# ============================================================
# 메인 파이프라인
# ============================================================

async def run_kobis_load(
    use_cache: bool = True,
    detail_limit: int = DEFAULT_DETAIL_LIMIT,
    boxoffice_days: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embed_batch_size: int = DEFAULT_EMBED_BATCH,
    mood_provider: str = "upstage",
    mood_model: str = "solar-pro3",
    mood_rpm: int = 100,
    mood_concurrency: int = 20,
    mood_batch_size: int = 10,
) -> None:
    """
    KOBIS 데이터 수집 → 중복 제거 → 변환 → 임베딩 → 3DB 적재.

    흐름:
    1. KOBIS 목록 수집 (또는 캐시 로드)
    2. 기존 DB(Qdrant)와 중복 제거
    3. 상세정보 수집 (선택, 일일 한도 있음)
    4. 박스오피스 수집 (선택)
    5. MovieDocument 변환
    6. 임베딩 (Upstage API)
    7. Qdrant/Neo4j/ES 적재

    Args:
        use_cache: True이면 기존 캐시 사용 (기본: True)
        detail_limit: 상세정보 수집 건수 제한 (0이면 스킵)
        boxoffice_days: 박스오피스 히스토리 수집 일수 (0이면 스킵)
        batch_size: 적재 배치 크기
        embed_batch_size: Upstage 임베딩 API 배치 크기
    """
    pipeline_start = time.time()
    checkpoint = _load_checkpoint()

    # ── Step 0: DB 클라이언트 초기화 ──
    await init_all_clients()

    try:
        # ── Step 1: KOBIS 목록 수집 ──
        print("[Step 1] KOBIS 영화 목록 수집")

        kobis_movies: list[dict] = []

        if use_cache and KOBIS_CACHE_PATH.exists():
            cached = load_kobis_cache(str(KOBIS_CACHE_PATH))
            if cached:
                kobis_movies = cached
                print(f"  캐시 로드: {len(kobis_movies):,}건")
                logger.info("kobis_cache_loaded", count=len(kobis_movies))

        if not kobis_movies:
            if not settings.KOBIS_API_KEY:
                print("  [오류] KOBIS_API_KEY가 .env에 설정되지 않았습니다.")
                return

            async with KOBISCollector() as collector:
                kobis_movies = await collector.collect_all_movie_list()
                print(f"  API 수집: {len(kobis_movies):,}건")

                # 캐시 저장
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                save_kobis_cache(kobis_movies, str(KOBIS_CACHE_PATH))
                print(f"  캐시 저장: {KOBIS_CACHE_PATH}")

        checkpoint["total_collected"] = len(kobis_movies)
        checkpoint["phase"] = "collect"
        _save_checkpoint(checkpoint)

        # ── Step 2: 기존 DB 매칭 (enrichment vs 신규 분리) ──
        # 2026-04-09: 중복 skip → split 로 변경.
        # 기존 영화에도 KOBIS 풍부 데이터를 병합하여 데이터 유실 차단.
        print("\n[Step 2] 기존 DB 매칭 — enrichment 대상 / 신규 영화 분리")

        db_movies = _get_existing_movies_from_qdrant()

        enrichment_targets, new_movies = split_kobis_movies(
            kobis_movies=kobis_movies,
            db_movies=db_movies,
            exclude_ids=None,
        )

        checkpoint["total_enrichment_targets"] = len(enrichment_targets)
        checkpoint["total_after_dedup"] = len(new_movies)  # backwards compat: 신규 적재 대상
        checkpoint["phase"] = "split"
        _save_checkpoint(checkpoint)

        print(f"  원본:           {len(kobis_movies):,}건")
        print(f"  enrichment 대상: {len(enrichment_targets):,}건 (기존 영화에 KOBIS 병합)")
        print(f"  신규 적재 대상:  {len(new_movies):,}건")

        # ── Step 3: 상세정보 수집 (enrichment + 신규 모두) ──
        # enrichment 대상에도 상세정보가 필요 — actors/staffs/watch_grade 등이 핵심.
        detail_map: dict[str, dict] = {}

        if detail_limit > 0 and settings.KOBIS_API_KEY:
            print(f"\n[Step 3] 상세정보 수집 (최대 {detail_limit}건)")

            # enrichment + 신규 합쳐서 모두 상세 수집 (enrichment 우선)
            all_movie_cds: list[str] = []
            for _, kobis_raw in enrichment_targets:
                cd = kobis_raw.get("movieCd", "")
                if cd:
                    all_movie_cds.append(cd)
            for kobis_raw in new_movies:
                cd = kobis_raw.get("movieCd", "")
                if cd:
                    all_movie_cds.append(cd)
            target_cds = all_movie_cds[:detail_limit]

            async with KOBISCollector() as collector:
                details = await collector.collect_movie_details_batch(target_cds)
                for d in details:
                    detail_map[d.movie_cd] = {
                        "actors": d.actors,
                        "staffs": d.staffs,
                        "audits": d.audits,
                        "show_tm": d.show_tm,
                        "companys": d.companys,
                        "directors": getattr(d, "directors", []),
                        "genres": getattr(d, "genres", []),
                    }
                print(f"  상세 수집 완료: {len(detail_map):,}건 (API 호출: {collector.call_count}회)")

            checkpoint["detail_fetched"] = len(detail_map)
            checkpoint["phase"] = "detail"
            _save_checkpoint(checkpoint)
        else:
            print("\n[Step 3] 상세정보 수집 스킵")

        # ── Step 4: 박스오피스 수집 (선택) ──
        boxoffice_map: dict[str, dict] = {}

        if boxoffice_days > 0 and settings.KOBIS_API_KEY:
            print(f"\n[Step 4] 박스오피스 히스토리 수집 ({boxoffice_days}일)")

            async with KOBISCollector() as collector:
                bo_data = await collector.collect_boxoffice_history(days=boxoffice_days)
                # KOBISBoxOffice → dict 변환
                for movie_cd, bo in bo_data.items():
                    boxoffice_map[movie_cd] = {
                        "audi_acc": bo.audi_acc,
                        "sales_acc": bo.sales_acc,
                        "scrn_cnt": bo.scrn_cnt,
                    }
                print(f"  박스오피스 수집: {len(boxoffice_map):,}건")

            checkpoint["phase"] = "boxoffice"
            _save_checkpoint(checkpoint)
        else:
            print("\n[Step 4] 박스오피스 수집 스킵")

        # ── Step 5-A: 기존 영화 KOBIS enrichment 3DB 동기화 ──
        # 2026-04-09 신규: Qdrant/ES/Neo4j 3DB partial update.
        # 기존 TMDB 영화에 KOBIS 의 한국 관객/매출/등급/감독/배우/스태프/제작사 병합.
        if enrichment_targets:
            print(f"\n[Step 5-A] 기존 영화 KOBIS enrichment ({len(enrichment_targets):,}건)")
            enrichment_count = await _apply_kobis_enrichments_3db(
                enrichment_targets=enrichment_targets,
                detail_map=detail_map,
                boxoffice_map=boxoffice_map,
            )
            checkpoint["total_enriched"] = enrichment_count
            checkpoint["phase"] = "enriched"
            _save_checkpoint(checkpoint)
            print(f"  3DB enrichment 완료: {enrichment_count:,}건")
        else:
            print("\n[Step 5-A] enrichment 대상 없음")

        # ── Step 5-B: 신규 영화 MovieDocument 변환 ──
        print(f"\n[Step 5-B] 신규 영화 MovieDocument 변환 ({len(new_movies):,}건)")

        documents = convert_kobis_movies(
            kobis_movies=new_movies,
            detail_map=detail_map,
            boxoffice_map=boxoffice_map,
        )

        checkpoint["total_converted"] = len(documents)
        checkpoint["phase"] = "convert"
        _save_checkpoint(checkpoint)

        print(f"  변환 성공: {len(documents):,}건 / {len(new_movies):,}건")

        if not documents:
            print("  신규 적재 영화가 없습니다. (enrichment 는 완료됨)")
            return

        # ── Step 6 & 7: 배치 단위 mood 보강 → 임베딩 → 적재 ──
        # Phase ML-4 일관성 (2026-04-08): 배치 단위로 Solar Pro 3 정밀 mood 적용 후
        # build_embedding_text 재생성 → Solar embedding → 적재. TMDB run_full_reload와 동일 패턴.
        print(f"\n[Step 6-7] 배치 mood + 임베딩 + 3DB 적재 (배치: {batch_size}건)")

        # mood_provider 사전 검증: API 키 없으면 fallback (기존 fallback mood 유지)
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
                logger.warning("kobis_upstage_api_key_missing_fallback")
                mood_provider = "fallback"

        total_loaded = 0
        start_offset = checkpoint.get("batch_offset", 0)
        total_batches = (len(documents) - start_offset + batch_size - 1) // batch_size

        for batch_idx, batch_start in enumerate(
            range(start_offset, len(documents), batch_size)
        ):
            batch_end = min(batch_start + batch_size, len(documents))
            batch = documents[batch_start:batch_end]
            batch_start_time = time.time()

            # ── Step 6a (신규): Solar Pro 3 배치 mood 보강 ──
            # convert_kobis_movies()가 fallback mood로 채운 mood_tags를
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
                        "kobis_batch_mood_enriched",
                        batch_idx=batch_idx,
                        total=mood_stats["total"],
                        enriched=mood_stats["enriched"],
                        elapsed_s=mood_stats["elapsed_s"],
                    )
                except Exception as e:
                    # mood 실패는 치명적 X — fallback mood로 진행
                    logger.error(
                        "kobis_batch_mood_failed_continue_with_fallback",
                        batch_idx=batch_idx,
                        error=str(e)[:200],
                    )

            # ── Step 6b: 임베딩 (Solar embedding 4096차원) ──
            # mood 보강 후 재생성된 embedding_text를 사용 (정밀 mood 반영)
            texts = [doc.embedding_text for doc in batch]
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, embed_texts, texts, embed_batch_size
            )

            # ── Step 7: 3DB 적재 ──
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

            # 진행률 출력
            remaining = total_batches - batch_idx - 1
            eta = batch_elapsed * remaining if remaining > 0 else 0
            print(
                f"  [Batch {batch_idx + 1:>3}/{total_batches}] "
                f"{len(batch):,}건 적재 | "
                f"Qdrant: {qdrant_count} | ES: {es_count} | "
                f"소요: {batch_elapsed:.1f}s | ETA: {eta / 60:.1f}m"
            )

        # ── 완료 ──
        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[KOBIS 적재 완료]")
        print(f"  수집:      {checkpoint['total_collected']:>10,}건")
        print(f"  중복 제거: {checkpoint['total_after_dedup']:>10,}건")
        print(f"  변환:      {checkpoint['total_converted']:>10,}건")
        print(f"  적재:      {total_loaded:>10,}건")
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
    print("  KOBIS 적재 파이프라인 상태")
    print("=" * 60)
    print(f"  현재 단계:     {checkpoint.get('phase', '미시작')}")
    print(f"  수집:          {checkpoint.get('total_collected', 0):>10,}건")
    print(f"  중복 제거 후:  {checkpoint.get('total_after_dedup', 0):>10,}건")
    print(f"  상세 수집:     {checkpoint.get('detail_fetched', 0):>10,}건")
    print(f"  변환:          {checkpoint.get('total_converted', 0):>10,}건")
    print(f"  적재:          {checkpoint.get('total_loaded', 0):>10,}건")
    print(f"  마지막 업데이트: {checkpoint.get('last_updated', '-')}")
    print("=" * 60)


# ============================================================
# 진입점
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KOBIS 데이터 수집 + 적재 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (캐시 사용)
  PYTHONPATH=src uv run python scripts/run_kobis_load.py

  # 상세정보 + 박스오피스 포함
  PYTHONPATH=src uv run python scripts/run_kobis_load.py --detail-limit 2500 --boxoffice-days 365
        """,
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="캐시 무시하고 API 재수집",
    )
    parser.add_argument(
        "--detail-limit", type=int, default=DEFAULT_DETAIL_LIMIT,
        help=f"상세정보 수집 건수 제한 (기본: {DEFAULT_DETAIL_LIMIT}, 0이면 스킵)",
    )
    parser.add_argument(
        "--boxoffice-days", type=int, default=0,
        help="박스오피스 히스토리 수집 일수 (기본: 0=스킵)",
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
            "'fallback': convert_kobis_movies()의 fallback mood 유지"
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
            run_kobis_load(
                use_cache=not args.no_cache,
                detail_limit=args.detail_limit,
                boxoffice_days=args.boxoffice_days,
                batch_size=args.batch_size,
                embed_batch_size=args.embed_batch_size,
                mood_provider=args.mood_provider,
                mood_model=args.mood_model,
                mood_rpm=args.mood_rpm,
                mood_concurrency=args.mood_concurrency,
                mood_batch_size=args.mood_batch_size,
            )
        )
