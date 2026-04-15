"""
Qdrant 벡터를 TensorFlow Embedding Projector(https://projector.tensorflow.org/)
형식으로 내보내는 스크립트.

출력 파일 (2개):
  1. vectors.tsv   — 각 행이 하나의 임베딩 벡터 (탭 구분, 헤더 없음)
  2. metadata.tsv  — 각 행이 vectors.tsv 의 동일 순번 벡터의 메타데이터
                      (1행은 헤더: title, year, genres, director, ...)

경량화 전략 (기본값):
  - 인기작 필터: `vote_count >= --popular-threshold` (기본 500) 만 스크롤
  - 상위 N건 선별: `--limit` 건까지만 수집 (기본 3,000건)
  - 차원 축소(선택): `--pca-dim` 지정 시 4096 → N차원 PCA 사전 축소
    (Projector 내부에서도 PCA/UMAP/t-SNE 하지만 파일 크기/로드 속도 단축용)

예상 파일 크기:
  - 3,000건 × 4096-dim (원본)                ≈ 85 MB
  - 3,000건 × 256-dim  (PCA 축소)           ≈ 5 MB   (권장)
  - 5,000건 × 128-dim  (PCA 축소)           ≈ 4 MB

사용법:
  # 기본 (인기작 3,000건, 원본 4096차원)
  uv run python scripts/export_embeddings_for_projector.py

  # 경량 데모용 (인기작 3,000건, PCA 256차원)
  uv run python scripts/export_embeddings_for_projector.py --pca-dim 256

  # 더 많은 샘플 + 더 가벼움
  uv run python scripts/export_embeddings_for_projector.py \
      --limit 5000 --pca-dim 128 --popular-threshold 300

시각화:
  1. https://projector.tensorflow.org/ 접속
  2. 좌측 "Load" → vectors.tsv / metadata.tsv 각각 업로드
  3. 우측 상단에서 PCA / t-SNE / UMAP 선택
  4. "Color by" 에 original_language, release_year 등 지정 → 군집 색칠
"""

import argparse
import asyncio
import csv
import sys
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path 에 추가 (config import 용)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monglepick.config import settings  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import FieldCondition, Filter, Range  # noqa: E402


# ────────────────────────────────────────────────────────────────────────
# 메타데이터 스키마
# ────────────────────────────────────────────────────────────────────────
# Projector "Color by" 드롭다운에 노출될 컬럼.
# - title 은 label 로 활용 (hover/search)
# - release_year / original_language / vote_count 는 색칠용으로 유용
META_HEADERS = [
    "title",
    "release_year",
    "genres",
    "director",
    "original_language",
    "rating",
    "vote_count",
    "mood_tags",
]


# ────────────────────────────────────────────────────────────────────────
# Qdrant 에서 인기작만 수집
# ────────────────────────────────────────────────────────────────────────
def _build_popular_filter(min_vote_count: int) -> Filter:
    """vote_count >= threshold 필터. Qdrant payload index 활용."""
    return Filter(
        must=[
            FieldCondition(
                key="vote_count",
                range=Range(gte=min_vote_count),
            ),
        ]
    )


def _fmt_list(val: Any, max_items: int = 5) -> str:
    """payload 의 list 값을 콤마로 합치되 너무 길면 잘라낸다."""
    if not isinstance(val, list):
        return str(val or "")
    # 탭/개행 제거 (TSV 안전)
    cleaned = [str(v).replace("\t", " ").replace("\n", " ").strip() for v in val[:max_items]]
    return ", ".join(c for c in cleaned if c)


def _fmt_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val).replace("\t", " ").replace("\n", " ").strip()


def collect_points(
    client: QdrantClient,
    collection: str,
    limit: int,
    min_vote_count: int,
) -> tuple[list[list[float]], list[dict]]:
    """
    Qdrant 스크롤로 `min_vote_count` 이상 인기작을 `limit` 건 수집.
    반환: (벡터 리스트, payload 리스트)
    """
    vectors: list[list[float]] = []
    payloads: list[dict] = []

    offset = None
    batch_size = 256

    scroll_filter = _build_popular_filter(min_vote_count) if min_vote_count > 0 else None

    while len(vectors) < limit:
        need = min(batch_size, limit - len(vectors))

        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=need,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )

        if not points:
            break

        for p in points:
            if p.vector is None:
                continue
            vectors.append(list(p.vector))
            payloads.append(p.payload or {})
            if len(vectors) >= limit:
                break

        # 진행 표시
        if len(vectors) % 500 == 0 or len(vectors) >= limit:
            print(f"  수집 진행: {len(vectors)}/{limit}")

        if next_offset is None:
            break
        offset = next_offset

    return vectors, payloads


# ────────────────────────────────────────────────────────────────────────
# PCA (선택) — 파일 크기 축소
# ────────────────────────────────────────────────────────────────────────
def apply_pca(vectors: list[list[float]], target_dim: int) -> list[list[float]]:
    """4096-dim 을 target_dim 으로 PCA 축소. scikit-learn 사용."""
    import numpy as np
    from sklearn.decomposition import PCA

    X = np.asarray(vectors, dtype=np.float32)
    print(f"  PCA: {X.shape} → (*, {target_dim})")
    pca = PCA(n_components=target_dim, svd_solver="randomized", random_state=42)
    reduced = pca.fit_transform(X)
    print(f"  PCA 설명 분산비 합계: {pca.explained_variance_ratio_.sum():.3f}")
    return reduced.tolist()


# ────────────────────────────────────────────────────────────────────────
# TSV 저장
# ────────────────────────────────────────────────────────────────────────
def write_tsv(
    output_dir: Path,
    vectors: list[list[float]],
    payloads: list[dict],
) -> tuple[Path, Path]:
    vectors_file = output_dir / "vectors.tsv"
    metadata_file = output_dir / "metadata.tsv"

    with open(vectors_file, "w", newline="") as vf, \
         open(metadata_file, "w", newline="", encoding="utf-8") as mf:

        vec_writer = csv.writer(vf, delimiter="\t", lineterminator="\n")
        meta_writer = csv.writer(mf, delimiter="\t", lineterminator="\n")

        # 메타데이터 헤더
        meta_writer.writerow(META_HEADERS)

        for vec, payload in zip(vectors, payloads):
            # 벡터: 소수점 6자리 (용량 절약)
            vec_writer.writerow([f"{v:.6f}" for v in vec])

            # 메타데이터: 컬럼 순서 = META_HEADERS
            meta_writer.writerow([
                _fmt_str(payload.get("title")),
                _fmt_str(payload.get("release_year")),
                _fmt_list(payload.get("genres")),
                _fmt_str(payload.get("director")),
                _fmt_str(payload.get("original_language")),
                _fmt_str(payload.get("rating")),
                _fmt_str(payload.get("vote_count")),
                _fmt_list(payload.get("mood_tags")),
            ])

    return vectors_file, metadata_file


# ────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────
def export_embeddings(
    output_dir: str,
    limit: int,
    min_vote_count: int,
    pca_dim: int,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(url=settings.QDRANT_URL, check_compatibility=False)

    info = client.get_collection(settings.QDRANT_COLLECTION)
    print(f"컬렉션: {settings.QDRANT_COLLECTION}")
    print(f"  총 포인트: {info.points_count:,} / 차원: {info.config.params.vectors.size}")
    print(f"  필터: vote_count >= {min_vote_count}")
    print(f"  목표 수집량: {limit:,} 건")
    if pca_dim > 0:
        print(f"  PCA 축소 대상 차원: {pca_dim}")
    print()

    # 1. 수집
    vectors, payloads = collect_points(
        client=client,
        collection=settings.QDRANT_COLLECTION,
        limit=limit,
        min_vote_count=min_vote_count,
    )
    print(f"\n수집 완료: {len(vectors):,} 건")
    if not vectors:
        print("⚠️  수집된 벡터가 없습니다. --popular-threshold 를 낮춰 재시도하세요.")
        return

    # 2. (선택) PCA 축소
    if pca_dim > 0 and pca_dim < len(vectors[0]):
        vectors = apply_pca(vectors, pca_dim)

    # 3. 저장
    vec_file, meta_file = write_tsv(output_path, vectors, payloads)

    # 4. 요약
    vec_size_mb = vec_file.stat().st_size / 1024 / 1024
    meta_size_kb = meta_file.stat().st_size / 1024

    print("\n✅ 내보내기 완료")
    print(f"  {vec_file}  ({len(vectors):,} × {len(vectors[0])}, {vec_size_mb:.1f} MB)")
    print(f"  {meta_file}  ({len(payloads):,} 행 + 헤더, {meta_size_kb:.1f} KB)")
    print()
    print("📊 Projector 업로드 방법:")
    print("  1. https://projector.tensorflow.org/ 접속")
    print("  2. 좌측 상단 'Load' 버튼")
    print(f"  3. Step 1: Load vectors → {vec_file.name}")
    print(f"  4. Step 2: Load metadata → {meta_file.name}")
    print("  5. 우측 상단 'Color by' → original_language / release_year 등 선택")
    print("  6. 우측 하단 시각화 방법 → PCA / t-SNE / UMAP 선택")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qdrant 벡터를 TensorFlow Projector 형식으로 내보내기",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/projector",
        help="출력 디렉토리 (기본: data/projector)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help="내보낼 최대 벡터 수 (기본: 3000)",
    )
    parser.add_argument(
        "--popular-threshold",
        type=int,
        default=500,
        help="인기작 필터 최소 vote_count (기본: 500). 0 이면 필터 없이 앞에서부터 수집.",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help="PCA 축소 목표 차원 (기본: 0 = 원본 4096차원 유지). 권장: 128 또는 256.",
    )
    args = parser.parse_args()

    export_embeddings(
        output_dir=args.output_dir,
        limit=args.limit,
        min_vote_count=args.popular_threshold,
        pca_dim=args.pca_dim,
    )
