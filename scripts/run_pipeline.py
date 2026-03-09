"""
데이터 파이프라인 실행 스크립트.

사용법:
    # 전체 파이프라인 실행
    uv run python scripts/run_pipeline.py

    # TMDB 수집 건너뛰기 (기존 데이터 재적재)
    uv run python scripts/run_pipeline.py --skip-collect

    # 캐시에서 TMDB 데이터 로드 (API 재호출 없이 재처리)
    uv run python scripts/run_pipeline.py --use-cache --skip-mood

    # 무드태그 GPT 생성 건너뛰기 (fallback 사용, API 비용 절약)
    uv run python scripts/run_pipeline.py --skip-mood

    # GPU 사용 시 배치 크기 증가
    uv run python scripts/run_pipeline.py --batch-size 128
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monglepick.data_pipeline.pipeline import run_full_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="몽글픽 데이터 파이프라인 실행")

    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="TMDB 수집 단계를 건너뜁니다 (기존 데이터 재적재)",
    )
    parser.add_argument(
        "--skip-mood",
        action="store_true",
        help="GPT 무드태그 생성을 건너뜁니다 (장르 기반 fallback 사용)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="TMDB API 대신 캐시 파일에서 로드합니다 (data/tmdb/tmdb_raw_movies.json)",
    )
    parser.add_argument(
        "--kaggle-dir",
        type=str,
        default="data/kaggle_movies",
        help="Kaggle 데이터 디렉토리 경로 (기본: data/kaggle_movies)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="임베딩 배치 크기 (CPU: 32, GPU: 128)",
    )
    parser.add_argument(
        "--use-jsonl",
        action="store_true",
        help="Phase D 전체 수집 JSONL 파일에서 로드합니다 (data/tmdb_full/tmdb_full_movies.jsonl)",
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default=None,
        help="JSONL 파일 경로를 지정합니다 (--use-jsonl과 함께 사용)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_full_pipeline(
            skip_collect=args.skip_collect,
            skip_mood_generation=args.skip_mood,
            kaggle_data_dir=args.kaggle_dir,
            embedding_batch_size=args.batch_size,
            use_cache=args.use_cache,
            use_jsonl=args.use_jsonl,
            jsonl_path=args.jsonl_path,
        )
    )


if __name__ == "__main__":
    main()
