"""
전체 데이터 재적재 스크립트 (스트리밍 배치 파이프라인).

1.17M건 TMDB JSONL을 메모리 효율적으로 처리한다.
청크 단위(CHUNK_SIZE)로 읽기 → 전처리 → 임베딩 → 적재를 반복하며,
체크포인트를 통해 중단 후 재개가 가능하다.

사용법:
    # 전체 재적재 (TMDB JSONL + Kaggle 보강 + CF 재구축)
    PYTHONPATH=src uv run python scripts/run_full_reload.py

    # TMDB JSONL만 적재 (Kaggle/CF 스킵)
    PYTHONPATH=src uv run python scripts/run_full_reload.py --skip-kaggle --skip-cf

    # 기존 DB 초기화 후 적재 (주의: 기존 데이터 삭제)
    PYTHONPATH=src uv run python scripts/run_full_reload.py --clear-db

    # 이전 중단점에서 재개 (체크포인트 파일 자동 참조)
    PYTHONPATH=src uv run python scripts/run_full_reload.py --resume

    # 청크 크기/임베딩 배치 크기 조정
    PYTHONPATH=src uv run python scripts/run_full_reload.py --chunk-size 5000 --embed-batch 50

    # 무드태그 Ollama 생성 활성화 (기본: fallback 사용)
    PYTHONPATH=src uv run python scripts/run_full_reload.py --generate-mood

    # 커스텀 JSONL 경로 지정
    PYTHONPATH=src uv run python scripts/run_full_reload.py --jsonl-path data/tmdb_full/tmdb_full_movies.jsonl

소요 시간 추정 (1.17M건 기준):
    - 전처리: ~30분 (CPU bound)
    - 임베딩: ~4~8시간 (Upstage API 100 RPM 제한)
    - Qdrant 적재: ~30분
    - Neo4j 적재: ~1시간
    - ES 적재: ~20분
    - 총합: ~6~10시간
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

from monglepick.data_pipeline.models import TMDBRawMovie, MovieDocument  # noqa: E402
from monglepick.data_pipeline.preprocessor import process_raw_movie  # noqa: E402
from monglepick.data_pipeline.embedder import embed_texts  # noqa: E402
from monglepick.data_pipeline.qdrant_loader import load_to_qdrant  # noqa: E402
from monglepick.data_pipeline.neo4j_loader import load_to_neo4j  # noqa: E402
from monglepick.data_pipeline.es_loader import load_to_elasticsearch  # noqa: E402
from monglepick.data_pipeline.cf_builder import build_cf_matrix, cache_cf_to_redis  # noqa: E402
from monglepick.data_pipeline.kaggle_loader import KaggleLoader  # noqa: E402
# Phase ML-4 품질 개선: 청크 단위 Solar Pro 3 배치 무드태그 생성
from monglepick.data_pipeline.mood_batch import enrich_documents_with_solar_mood  # noqa: E402
from monglepick.db.clients import (  # noqa: E402
    init_all_clients,
    close_all_clients,
    get_qdrant,
    get_neo4j,
    get_elasticsearch,
    ES_INDEX_NAME,
)
from monglepick.config import settings  # noqa: E402

logger = structlog.get_logger()

# ── 기본 경로 ──
DEFAULT_JSONL_PATH = Path("data/tmdb_full/tmdb_full_movies.jsonl")
CHECKPOINT_FILE = Path("data/reload_checkpoint.json")

# ── 청크 중간 캐시 경로 ──
# 전처리 완료 후 문서를 캐시하여, 임베딩/적재 중 중단 시 전처리를 다시 하지 않도록 한다.
CHUNK_DOCS_CACHE = Path("data/reload_chunk_docs.jsonl")
# 임베딩 완료 후 벡터를 캐시하여, 적재 중 중단 시 임베딩을 다시 하지 않도록 한다.
CHUNK_EMBEDDINGS_CACHE = Path("data/reload_chunk_embeddings.npy")

# ── 기본 청크/배치 크기 ──
DEFAULT_CHUNK_SIZE = 2000       # JSONL 읽기 청크 (메모리 효율)
DEFAULT_EMBED_BATCH = 50        # Upstage API 배치 (최대 100, Rate Limit 고려)


# ============================================================
# 체크포인트 관리
# ============================================================
#
# 체크포인트 구조:
# {
#   "last_line": 4000,              — 마지막으로 읽은 JSONL 라인 번호
#   "last_completed_line": 2000,    — 전처리+임베딩+적재까지 완료된 마지막 라인
#   "chunk_step": "embedding",      — 현재 청크 진행 단계 ("" | "preprocessed" | "embedded" | "loaded")
#   "embed_batch_done": 15,         — 임베딩 완료된 배치 수 (배치 단위 재개용)
#   "total_processed": 1234,
#   "total_loaded": 1200,
#   "total_skipped": 34,
#   "failed_ids": [],
#   "start_time": "...",
#   "last_updated": "..."
# }


def _new_checkpoint() -> dict:
    """새 체크포인트를 생성한다."""
    return {
        "last_line": 0,
        "last_completed_line": 0,
        "chunk_step": "",
        "embed_batch_done": 0,
        "total_processed": 0,
        "total_loaded": 0,
        "total_skipped": 0,
        "failed_ids": [],
        "start_time": datetime.now().isoformat(),
    }


def _load_checkpoint() -> dict:
    """
    체크포인트 파일을 로드한다.
    중단 후 재개 시 마지막 처리 완료 지점을 참조한다.
    """
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text())
        # 하위 호환: 이전 형식 체크포인트에 새 필드 보충
        data.setdefault("last_completed_line", data.get("last_line", 0))
        data.setdefault("chunk_step", "")
        data.setdefault("embed_batch_done", 0)
        return data
    return _new_checkpoint()


def _save_checkpoint(state: dict) -> None:
    """체크포인트 파일에 진행 상태를 저장한다."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def _save_docs_cache(documents: list[MovieDocument]) -> None:
    """전처리 완료 문서를 JSONL 캐시로 저장한다 (임베딩 중 중단 복구용)."""
    CHUNK_DOCS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNK_DOCS_CACHE, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json() + "\n")
    logger.info("chunk_docs_cached", count=len(documents), path=str(CHUNK_DOCS_CACHE))


def _load_docs_cache() -> list[MovieDocument] | None:
    """캐시된 전처리 문서를 로드한다. 파일이 없으면 None 반환."""
    if not CHUNK_DOCS_CACHE.exists():
        return None
    docs: list[MovieDocument] = []
    with open(CHUNK_DOCS_CACHE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(MovieDocument.model_validate_json(line))
    logger.info("chunk_docs_loaded_from_cache", count=len(docs))
    return docs


def _save_embeddings_cache(embeddings: "np.ndarray") -> None:
    """임베딩 벡터를 numpy 파일로 캐시한다 (적재 중 중단 복구용)."""
    import numpy as np
    CHUNK_EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(CHUNK_EMBEDDINGS_CACHE), embeddings)
    logger.info("chunk_embeddings_cached", shape=embeddings.shape, path=str(CHUNK_EMBEDDINGS_CACHE))


def _load_embeddings_cache() -> "np.ndarray | None":
    """캐시된 임베딩 벡터를 로드한다. 파일이 없으면 None 반환."""
    import numpy as np
    if not CHUNK_EMBEDDINGS_CACHE.exists():
        return None
    embeddings = np.load(str(CHUNK_EMBEDDINGS_CACHE))
    logger.info("chunk_embeddings_loaded_from_cache", shape=embeddings.shape)
    return embeddings


def _clear_chunk_cache() -> None:
    """청크 중간 캐시 파일을 삭제한다 (청크 완료 후 정리)."""
    if CHUNK_DOCS_CACHE.exists():
        CHUNK_DOCS_CACHE.unlink()
    if CHUNK_EMBEDDINGS_CACHE.exists():
        CHUNK_EMBEDDINGS_CACHE.unlink()


# ============================================================
# 스트리밍 JSONL 청크 리더
# ============================================================


def read_jsonl_chunks(
    jsonl_path: Path,
    chunk_size: int,
    skip_lines: int = 0,
) -> "Generator[tuple[list[TMDBRawMovie], int], None, None]":
    """
    JSONL 파일을 chunk_size 단위로 읽어 TMDBRawMovie 리스트를 yield한다.

    메모리 효율적으로 한 줄씩 파싱하며, skip_lines으로 이미 처리한 라인을 건너뛴다.
    파싱 실패 라인은 경고 로그 후 건너뛴다.

    Args:
        jsonl_path: JSONL 파일 경로
        chunk_size: 한 번에 yield할 레코드 수
        skip_lines: 건너뛸 라인 수 (체크포인트 재개용)

    Yields:
        (TMDBRawMovie 리스트, 현재까지 읽은 총 라인 수) 튜플
    """
    chunk: list[TMDBRawMovie] = []
    line_count = 0
    parse_errors = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1

            # 체크포인트 재개: 이미 처리한 라인 건너뛰기
            if line_count <= skip_lines:
                continue

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # 하위 호환: recommendations가 list[int]인 경우 list[dict]로 변환
                recs = data.get("recommendations", [])
                if recs and isinstance(recs[0], int):
                    data["recommendations"] = [{"id": r} for r in recs]

                # 삭제된 필드 제거 (기존 캐시/JSONL에 남아있을 수 있음)
                data.pop("recommendation_ids_raw", None)
                data.pop("similar_movies_full", None)
                data.pop("changes", None)

                chunk.append(TMDBRawMovie(**data))

            except (json.JSONDecodeError, Exception) as e:
                parse_errors += 1
                if parse_errors <= 20:
                    logger.warning("jsonl_parse_error", line_no=line_count, error=str(e))
                continue

            # 청크가 가득 차면 yield
            if len(chunk) >= chunk_size:
                yield chunk, line_count
                chunk = []

    # 마지막 남은 청크
    if chunk:
        yield chunk, line_count

    if parse_errors > 0:
        logger.warning("jsonl_parse_errors_total", total=parse_errors)


# ============================================================
# 청크 단위 처리 함수
# ============================================================


async def process_chunk(
    raw_movies: list[TMDBRawMovie],
    generate_mood: bool = False,
) -> tuple[list[MovieDocument], int]:
    """
    TMDBRawMovie 청크를 MovieDocument로 변환한다 (전처리).

    개별 영화 실패 시 로그 후 건너뛰며, 유효성 검증(validate_movie)을 통과한
    문서만 반환한다.

    Args:
        raw_movies: 원본 TMDB 영화 데이터 청크
        generate_mood: True이면 Ollama로 무드태그 생성 (느림, 기본 False)

    Returns:
        (성공한 MovieDocument 리스트, 건너뛴 레코드 수)
    """
    documents: list[MovieDocument] = []
    skipped = 0

    for raw in raw_movies:
        try:
            doc = await process_raw_movie(raw, generate_mood=generate_mood)
            if doc:
                documents.append(doc)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            logger.warning("preprocess_failed", movie_id=raw.id, error=str(e))

    return documents, skipped


async def embed_chunk(
    documents: list[MovieDocument],
    embed_batch_size: int = DEFAULT_EMBED_BATCH,
) -> "np.ndarray":
    """
    MovieDocument 청크를 임베딩한다.

    embed_texts는 동기 함수(내부 time.sleep)이므로 executor에서 실행하여
    이벤트 루프 블록을 방지한다.

    Args:
        documents: 임베딩할 MovieDocument 리스트
        embed_batch_size: Upstage API 배치 크기

    Returns:
        np.ndarray: shape (len(documents), 4096)
    """
    texts = [doc.embedding_text for doc in documents]
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, embed_texts, texts, embed_batch_size)
    return embeddings


async def load_chunk_to_dbs(
    documents: list[MovieDocument],
    embeddings: "np.ndarray",
) -> int:
    """
    임베딩된 MovieDocument 청크를 3개 DB에 동시 적재한다.

    return_exceptions=True로 하나가 실패해도 나머지는 계속 적재한다.

    Args:
        documents: 적재할 MovieDocument 리스트
        embeddings: 임베딩 벡터 배열

    Returns:
        적재 완료 건수
    """
    results = await asyncio.gather(
        load_to_qdrant(documents, embeddings),
        load_to_neo4j(documents),
        load_to_elasticsearch(documents),
        return_exceptions=True,
    )

    # 개별 DB 에러 로깅
    loaded = 0
    for i, result in enumerate(results):
        db_name = ["qdrant", "neo4j", "elasticsearch"][i]
        if isinstance(result, Exception):
            logger.error(f"{db_name}_chunk_load_failed", error=str(result))
        elif isinstance(result, int):
            loaded = max(loaded, result)

    return loaded


# ============================================================
# DB 초기화 (기존 데이터 삭제)
# ============================================================


async def clear_databases() -> None:
    """
    Qdrant 컬렉션, Neo4j 전체 노드/관계, ES 인덱스를 초기화한다.

    --clear-db 옵션으로 호출되며, 기존 데이터를 완전히 삭제한 후
    컬렉션/인덱스를 재생성한다.

    주의: 이 작업은 되돌릴 수 없다.
    """
    logger.info("clear_databases_start")

    # Qdrant: 컬렉션 삭제 → init_all_clients()에서 재생성
    try:
        client = await get_qdrant()
        collections = await client.get_collections()
        existing = [c.name for c in collections.collections]
        if settings.QDRANT_COLLECTION in existing:
            await client.delete_collection(settings.QDRANT_COLLECTION)
            logger.info("qdrant_collection_deleted", name=settings.QDRANT_COLLECTION)
    except Exception as e:
        logger.error("qdrant_clear_error", error=str(e))

    # Neo4j: 전체 노드/관계 삭제 (소규모 배치 + 자동 축소로 메모리 보호)
    # 노드당 관계가 많은 경우(SIMILAR_TO, RECOMMENDED 등) DETACH DELETE의
    # 트랜잭션 메모리 사용량이 급증하므로 배치 크기를 보수적으로 설정한다.
    # 실패 시 배치 크기를 절반으로 줄여 재시도한다 (최소 100건).
    try:
        driver = await get_neo4j()
        deleted_total = 0
        batch_size = 2000  # 초기 배치 크기 (10,000 → 2,000으로 축소)

        while True:
            try:
                async with driver.session() as session:
                    result = await session.run(
                        f"MATCH (n) WITH n LIMIT {batch_size} DETACH DELETE n RETURN count(*) as cnt"
                    )
                    record = await result.single()
                    cnt = record["cnt"] if record else 0

                if cnt == 0:
                    break
                deleted_total += cnt
                if deleted_total % 10000 == 0 or cnt < batch_size:
                    logger.info("neo4j_batch_deleted", deleted=cnt, total=deleted_total, batch_size=batch_size)

            except Exception as batch_err:
                # 트랜잭션 메모리 초과 시 배치 크기를 절반으로 줄여 재시도
                if batch_size > 100:
                    old_size = batch_size
                    batch_size = max(100, batch_size // 2)
                    logger.warning(
                        "neo4j_batch_size_reduced",
                        old_size=old_size,
                        new_size=batch_size,
                        error=str(batch_err)[:200],
                    )
                else:
                    # 최소 배치 크기(100)에서도 실패하면 에러 전파
                    raise

        logger.info("neo4j_all_deleted", total=deleted_total)
    except Exception as e:
        logger.error("neo4j_clear_error", error=str(e))
        raise RuntimeError(f"Neo4j 초기화 실패: {e}") from e

    # Elasticsearch: 인덱스 삭제 → init_all_clients()에서 재생성
    try:
        es = await get_elasticsearch()
        if await es.indices.exists(index=ES_INDEX_NAME):
            await es.indices.delete(index=ES_INDEX_NAME)
            logger.info("es_index_deleted", index=ES_INDEX_NAME)
    except Exception as e:
        logger.error("es_clear_error", error=str(e))

    logger.info("clear_databases_complete")


# ============================================================
# 메인 파이프라인
# ============================================================


async def run_full_reload(
    jsonl_path: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    embed_batch_size: int = DEFAULT_EMBED_BATCH,
    generate_mood: bool = False,
    clear_db: bool = False,
    resume: bool = False,
    skip_kaggle: bool = False,
    skip_cf: bool = False,
    kaggle_dir: str = "data/kaggle_movies",
    mood_provider: str = "upstage",
    mood_model: str = "solar-pro3",
    mood_rpm: int = 100,
    mood_concurrency: int = 20,
    mood_batch_size: int = 10,
) -> None:
    """
    전체 재적재 파이프라인을 실행한다.

    스트리밍 배치 방식으로 JSONL을 청크 단위로 처리하여 메모리 사용을 최소화한다.

    흐름:
    1. DB 클라이언트 초기화 (+ 선택적 기존 데이터 삭제)
    2. JSONL 스트리밍 읽기 → 청크 단위 전처리 → [Solar Pro 3 배치 무드] →
       임베딩 텍스트 재생성 → 벡터 임베딩 → 3DB 적재
    3. (선택) Kaggle 보강 데이터 적재
    4. (선택) CF 매트릭스 재구축 (Redis)

    Phase ML-4 품질 개선 (2026-04-08):
        청크 단위로 Solar Pro 3 배치 API 를 호출하여 정밀 무드태그를 생성하고,
        그 결과로 build_embedding_text 를 재실행하여 임베딩 벡터에 정밀 무드가
        반영되도록 한다 (mood_batch.enrich_documents_with_solar_mood).
        기존 방식(fallback mood → 임베딩 → 적재 → run_mood_enrichment 별도 실행)
        은 벡터에 정밀 무드를 반영하지 못했다.

    Args:
        jsonl_path: TMDB JSONL 파일 경로 (None이면 기본 경로)
        chunk_size: 청크당 레코드 수 (메모리 vs 오버헤드 트레이드오프)
        embed_batch_size: Upstage API 배치 크기 (최대 100)
        generate_mood: Ollama 무드태그 생성 여부 (True: 느림, False: 장르 기반 fallback)
            ※ mood_provider="upstage" 일 때는 청크 단위로 Solar Pro 3 이 덮어쓰므로
               이 값은 실질적으로 무시된다.
        clear_db: 기존 DB 데이터 삭제 후 적재 여부
        resume: 체크포인트에서 재개 여부
        skip_kaggle: Kaggle 보강 스킵 여부
        skip_cf: CF 매트릭스 재구축 스킵 여부
        kaggle_dir: Kaggle 데이터 디렉토리 경로
        mood_provider: 청크 단위 무드태그 제공자.
            - "upstage" (기본): Solar Pro 3 배치 API 로 정밀 분석
            - "fallback": 장르 기반 fallback (process_raw_movie 가 생성한 값 유지)
        mood_model: Upstage 모델명 (기본 "solar-pro3", 102B MoE)
        mood_rpm: Solar API 분당 호출 한도 (기본 100)
        mood_concurrency: Solar API 동시 호출 수 (기본 20)
        mood_batch_size: 1회 API 호출당 영화 수 (기본 10)
    """
    path = Path(jsonl_path) if jsonl_path else DEFAULT_JSONL_PATH

    # JSONL 파일 존재 확인
    if not path.exists():
        logger.error("jsonl_file_not_found", path=str(path))
        print(f"[ERROR] JSONL 파일을 찾을 수 없습니다: {path}")
        return

    # JSONL 전체 라인 수 (진행률 표시용)
    print(f"[INFO] JSONL 파일 라인 수 카운트 중... ({path})")
    with open(path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"[INFO] 총 {total_lines:,}건")

    # --clear-db와 --resume 동시 사용 방지
    if clear_db and resume:
        print("[ERROR] --clear-db와 --resume은 함께 사용할 수 없습니다.")
        return

    # 체크포인트 로드/초기화
    if resume:
        checkpoint = _load_checkpoint()
        # 미완료 청크가 있으면 해당 청크의 시작 라인으로 되돌린다
        if checkpoint.get("chunk_step") and checkpoint["chunk_step"] != "loaded":
            skip_lines = checkpoint.get("last_completed_line", 0)
            print(f"[RESUME] 미완료 청크 감지 (단계: {checkpoint['chunk_step']})")
            print(f"         {skip_lines:,}번째 라인부터 재개")
        else:
            skip_lines = checkpoint.get("last_completed_line", checkpoint.get("last_line", 0))
            print(f"[RESUME] 체크포인트에서 재개: {skip_lines:,}번째 라인부터")
    else:
        checkpoint = _new_checkpoint()
        skip_lines = 0
        _clear_chunk_cache()  # 이전 실행의 중간 캐시 정리

    # ── Step 0: DB 초기화 ──
    print("[Step 0] DB 클라이언트 초기화...")
    await init_all_clients()

    try:
        # 기존 데이터 삭제 (--clear-db)
        if clear_db:
            print("[Step 0] 기존 DB 데이터 삭제 중... (되돌릴 수 없습니다)")
            await clear_databases()
            # 컬렉션/인덱스 재생성을 위해 다시 초기화
            await init_all_clients()

        # ── Step 1: 스트리밍 배치 처리 ──
        pipeline_start = time.time()
        chunk_count = 0
        total_processed = checkpoint["total_processed"]
        total_loaded = checkpoint["total_loaded"]
        total_skipped = checkpoint["total_skipped"]

        print(f"\n[Step 1] TMDB JSONL 스트리밍 적재 시작 (chunk_size={chunk_size})")
        print(f"         임베딩 배치: {embed_batch_size} | 무드 생성: {generate_mood}")
        print(f"         예상 청크 수: {(total_lines - skip_lines) // chunk_size + 1}")
        print("=" * 70)

        # ── 미완료 청크 복구 (resume 시) ──
        # 이전 실행에서 전처리/임베딩 완료 후 중단된 청크가 있으면 캐시에서 복구
        if resume and checkpoint.get("chunk_step") in ("preprocessed", "embedded"):
            chunk_count += 1
            chunk_start = time.time()
            step = checkpoint["chunk_step"]

            if step == "preprocessed":
                # 전처리 완료, 임베딩부터 재개
                documents = _load_docs_cache()
                if documents:
                    print(f"  [복구] 캐시에서 {len(documents)}건 로드 → 임베딩부터 재개")
                    embeddings = await embed_chunk(documents, embed_batch_size)
                    _save_embeddings_cache(embeddings)
                    checkpoint["chunk_step"] = "embedded"
                    _save_checkpoint(checkpoint)

                    loaded = await load_chunk_to_dbs(documents, embeddings)
                    total_processed += len(documents)
                    total_loaded += loaded
                else:
                    print("  [복구] 캐시 파일 없음, 청크 재처리 필요")

            elif step == "embedded":
                # 임베딩 완료, 적재부터 재개
                documents = _load_docs_cache()
                embeddings = _load_embeddings_cache()
                if documents and embeddings is not None:
                    print(f"  [복구] 캐시에서 {len(documents)}건 + 임베딩 로드 → 적재부터 재개")
                    loaded = await load_chunk_to_dbs(documents, embeddings)
                    total_processed += len(documents)
                    total_loaded += loaded
                else:
                    print("  [복구] 캐시 파일 없음, 청크 재처리 필요")

            # 복구 완료: 체크포인트 갱신
            checkpoint["chunk_step"] = "loaded"
            checkpoint["last_completed_line"] = checkpoint.get("last_line", skip_lines)
            skip_lines = checkpoint["last_completed_line"]
            _clear_chunk_cache()
            _save_checkpoint(checkpoint)
            print(f"  [복구 완료] {skip_lines:,}번째 라인까지 처리 완료")

        # ── Step Mood 사전 검증: Upstage API 키 확인 ──
        # mood_provider="upstage" 인데 키가 없으면 안전하게 fallback 으로 강등
        upstage_api_key: str | None = None
        if mood_provider == "upstage":
            upstage_api_key = (
                settings.UPSTAGE_API_KEY
                if hasattr(settings, "UPSTAGE_API_KEY") and settings.UPSTAGE_API_KEY
                else None
            )
            if not upstage_api_key:
                logger.warning(
                    "upstage_api_key_missing_fallback_to_local",
                    message="UPSTAGE_API_KEY 미설정 → fallback mood 로 진행",
                )
                mood_provider = "fallback"
            else:
                print(
                    f"         무드 모드: upstage ({mood_model}, "
                    f"rpm={mood_rpm}, concurrency={mood_concurrency}, batch={mood_batch_size})"
                )

        # ── 메인 청크 루프 ──
        for raw_movies, current_line in read_jsonl_chunks(path, chunk_size, skip_lines):
            chunk_count += 1
            chunk_start = time.time()

            # ── Step A: 전처리 (fallback mood 로 일단 채움) ──
            documents, skipped = await process_chunk(raw_movies, generate_mood)
            total_skipped += skipped

            if not documents:
                logger.warning("chunk_all_skipped", chunk=chunk_count)
                # 빈 청크도 완료 처리
                checkpoint["last_line"] = current_line
                checkpoint["last_completed_line"] = current_line
                checkpoint["chunk_step"] = "loaded"
                checkpoint["total_skipped"] = total_skipped
                _save_checkpoint(checkpoint)
                continue

            # ── Step A2 (신규): Solar Pro 3 배치 무드 보강 ──
            # process_chunk 가 만든 fallback mood 를 Solar Pro 3 로 덮어쓰고,
            # embedding_text 를 재생성하여 정밀 무드가 벡터에 반영되도록 한다.
            # 내부적으로 RPM/concurrency 제한 및 429 재시도 + fallback 처리됨.
            if mood_provider == "upstage" and upstage_api_key:
                try:
                    mood_stats = await enrich_documents_with_solar_mood(
                        documents=documents,
                        api_key=upstage_api_key,
                        model=mood_model,
                        rpm=mood_rpm,
                        concurrency=mood_concurrency,
                        batch_size=mood_batch_size,
                        rebuild_embedding_text=True,
                    )
                    logger.info(
                        "chunk_mood_enrichment_done",
                        chunk=chunk_count,
                        total=mood_stats["total"],
                        enriched=mood_stats["enriched"],
                        batches=mood_stats["batches"],
                        elapsed_s=mood_stats["elapsed_s"],
                    )
                except Exception as e:
                    # 청크 단위 mood 실패는 치명적이지 않음 — fallback mood 로 진행
                    logger.error(
                        "chunk_mood_enrichment_failed_continue_with_fallback",
                        chunk=chunk_count,
                        error=str(e)[:200],
                    )

            # 전처리+mood 완료 → 캐시 저장 + 체크포인트
            _save_docs_cache(documents)
            checkpoint["last_line"] = current_line
            checkpoint["chunk_step"] = "preprocessed"
            _save_checkpoint(checkpoint)

            # ── Step B: 임베딩 (정밀 mood 가 반영된 embedding_text 로 벡터 생성) ──
            embeddings = await embed_chunk(documents, embed_batch_size)

            # 임베딩 완료 → 캐시 저장 + 체크포인트
            _save_embeddings_cache(embeddings)
            checkpoint["chunk_step"] = "embedded"
            _save_checkpoint(checkpoint)

            # ── Step C: 3DB 적재 ──
            loaded = await load_chunk_to_dbs(documents, embeddings)
            total_processed += len(documents)
            total_loaded += loaded

            # 청크 완료 → 캐시 삭제 + 체크포인트 갱신
            _clear_chunk_cache()
            checkpoint["chunk_step"] = "loaded"
            checkpoint["last_completed_line"] = current_line
            checkpoint["total_processed"] = total_processed
            checkpoint["total_loaded"] = total_loaded
            checkpoint["total_skipped"] = total_skipped
            _save_checkpoint(checkpoint)

            # 진행률 표시
            chunk_elapsed = time.time() - chunk_start
            total_elapsed = time.time() - pipeline_start
            progress_pct = (current_line / total_lines) * 100
            remaining_chunks = ((total_lines - current_line) // chunk_size) + 1
            eta_seconds = (total_elapsed / max(chunk_count, 1)) * remaining_chunks

            print(
                f"  [Chunk {chunk_count:>5}] "
                f"라인 {current_line:>9,}/{total_lines:,} ({progress_pct:5.1f}%) | "
                f"적재 {len(documents):>5} | 스킵 {skipped:>4} | "
                f"소요 {chunk_elapsed:>6.1f}s | "
                f"ETA {eta_seconds/3600:>5.1f}h"
            )

        print("=" * 70)
        total_elapsed = time.time() - pipeline_start
        print(
            f"\n[Step 1 완료] "
            f"처리: {total_processed:,} | 적재: {total_loaded:,} | "
            f"스킵: {total_skipped:,} | "
            f"소요: {total_elapsed/3600:.1f}시간"
        )

        # ── Step 2: Kaggle 보강 (선택) ──
        if not skip_kaggle:
            kaggle_path = Path(kaggle_dir)
            if kaggle_path.exists() and (kaggle_path / "movies_metadata.csv").exists():
                print(f"\n[Step 2] Kaggle 보강 데이터 적재 ({kaggle_dir})...")
                # Kaggle 보강은 기존 스크립트 로직을 재사용
                # (run_kaggle_supplement.py와 동일한 흐름)
                print("  [INFO] Kaggle 보강은 별도 스크립트로 실행하세요:")
                print(f"    PYTHONPATH=src uv run python scripts/run_kaggle_supplement.py --kaggle-dir {kaggle_dir}")
            else:
                print(f"\n[Step 2] Kaggle 디렉토리 없음, 건너뜀: {kaggle_dir}")
        else:
            print("\n[Step 2] Kaggle 보강 스킵 (--skip-kaggle)")

        # ── Step 3: CF 매트릭스 재구축 (선택) ──
        if not skip_cf:
            kaggle_path = Path(kaggle_dir)
            ratings_file = kaggle_path / "ratings.csv"
            if ratings_file.exists():
                print("\n[Step 3] CF 매트릭스 재구축 (Redis)...")
                kaggle = KaggleLoader(kaggle_dir)
                id_map = kaggle.load_links()
                ratings_df = kaggle.load_ratings(id_map)
                similar_users, user_ratings, movie_avg = await build_cf_matrix(ratings_df)
                await cache_cf_to_redis(similar_users, user_ratings, movie_avg)
                print("  [완료] CF 매트릭스 Redis 캐시 완료")
            else:
                print(f"\n[Step 3] ratings.csv 없음, CF 스킵: {ratings_file}")
        else:
            print("\n[Step 3] CF 재구축 스킵 (--skip-cf)")

        # ── 완료 ──
        checkpoint["current_step"] = "complete"
        _save_checkpoint(checkpoint)

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 70}")
        print(f"[전체 완료]")
        print(f"  총 처리: {total_processed:,}건")
        print(f"  총 적재: {total_loaded:,}건")
        print(f"  총 스킵: {total_skipped:,}건 (검증 실패)")
        print(f"  총 소요: {total_elapsed/3600:.1f}시간 ({total_elapsed:.0f}초)")
        print(f"  체크포인트: {CHECKPOINT_FILE}")
        print(f"{'=' * 70}")

    finally:
        await close_all_clients()


# ============================================================
# CLI 엔트리포인트
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="몽글픽 전체 데이터 재적재 (스트리밍 배치 파이프라인)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (TMDB JSONL 적재)
  PYTHONPATH=src uv run python scripts/run_full_reload.py

  # DB 초기화 후 전체 재적재
  PYTHONPATH=src uv run python scripts/run_full_reload.py --clear-db

  # 중단 후 재개
  PYTHONPATH=src uv run python scripts/run_full_reload.py --resume

  # 빠른 테스트 (작은 청크)
  PYTHONPATH=src uv run python scripts/run_full_reload.py --chunk-size 100
        """,
    )

    parser.add_argument(
        "--jsonl-path",
        type=str,
        default=None,
        help="TMDB JSONL 파일 경로 (기본: data/tmdb_full/tmdb_full_movies.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"청크당 레코드 수 (기본: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--embed-batch",
        type=int,
        default=DEFAULT_EMBED_BATCH,
        help=f"임베딩 API 배치 크기 (기본: {DEFAULT_EMBED_BATCH}, 최대 100)",
    )
    parser.add_argument(
        "--generate-mood",
        action="store_true",
        help="Ollama로 무드태그 생성 (기본: 장르 기반 fallback 사용)",
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="기존 DB 데이터 삭제 후 적재 (주의: 되돌릴 수 없음)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="체크포인트에서 재개 (이전 중단점부터 계속)",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Kaggle 보강 데이터 적재 건너뛰기",
    )
    parser.add_argument(
        "--skip-cf",
        action="store_true",
        help="CF 매트릭스 재구축 건너뛰기",
    )
    parser.add_argument(
        "--kaggle-dir",
        type=str,
        default="data/kaggle_movies",
        help="Kaggle 데이터 디렉토리 경로 (기본: data/kaggle_movies)",
    )
    # ── Phase ML-4 품질 개선: 청크 단위 Solar Pro 3 무드태그 통합 ──
    parser.add_argument(
        "--mood-provider",
        choices=["upstage", "fallback"],
        default="upstage",
        help=(
            "청크 단위 무드태그 생성 방식. "
            "'upstage' (기본): Solar Pro 3 배치 API 로 정밀 분석 + embedding_text 재생성, "
            "'fallback': 장르 기반 기본값만 사용 (process_raw_movie 결과 유지)"
        ),
    )
    parser.add_argument(
        "--mood-model",
        type=str,
        default="solar-pro3",
        help="Upstage 모델명 (기본: solar-pro3, 102B MoE)",
    )
    parser.add_argument(
        "--mood-rpm",
        type=int,
        default=100,
        help="Solar API 분당 호출 한도 (기본: 100, 429 발생 시 낮춤)",
    )
    parser.add_argument(
        "--mood-concurrency",
        type=int,
        default=20,
        help="Solar API 동시 호출 수 (기본: 20)",
    )
    parser.add_argument(
        "--mood-batch-size",
        type=int,
        default=10,
        help="1회 Solar API 호출당 영화 수 (기본: 10)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="--clear-db 확인 프롬프트 자동 승인 (비대화형 실행용)",
    )

    args = parser.parse_args()

    # --clear-db 안전 확인 (--yes 로 비대화형 우회 가능)
    if args.clear_db and not args.yes:
        confirm = input(
            "\n⚠️  --clear-db: 기존 Qdrant/Neo4j/Elasticsearch 데이터가 모두 삭제됩니다.\n"
            "   정말 진행하시겠습니까? (yes/no): "
        )
        if confirm.lower() not in ("yes", "y"):
            print("취소되었습니다.")
            return

    asyncio.run(
        run_full_reload(
            jsonl_path=args.jsonl_path,
            chunk_size=args.chunk_size,
            embed_batch_size=args.embed_batch,
            generate_mood=args.generate_mood,
            clear_db=args.clear_db,
            resume=args.resume,
            skip_kaggle=args.skip_kaggle,
            skip_cf=args.skip_cf,
            kaggle_dir=args.kaggle_dir,
            mood_provider=args.mood_provider,
            mood_model=args.mood_model,
            mood_rpm=args.mood_rpm,
            mood_concurrency=args.mood_concurrency,
            mood_batch_size=args.mood_batch_size,
        )
    )


if __name__ == "__main__":
    main()
