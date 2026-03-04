"""
파이프라인 체크포인트 관리.

어디까지 수집/적재했는지 추적하여 중단 시 이어서 진행할 수 있게 한다.

체크포인트 파일: data/checkpoint.json
{
    "tmdb_api_loaded_ids": [11, 12, 13, ...],     # TMDB API로 수집하여 적재 완료된 ID
    "kaggle_loaded_ids": [862, 1893, ...],          # Kaggle CSV에서 적재 완료된 ID
    "embedded_ids": [11, 12, 862, ...],             # 임베딩 완료된 ID
    "failed_ids": [99999, ...],                     # 처리 실패한 ID
    "last_updated": "2026-02-25T14:30:00",
    "tmdb_api_total_collected": 3727,               # TMDB API 수집 시도 총 수
    "kaggle_total_available": 44176,                # Kaggle에서 보강 가능한 총 수
}
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

CHECKPOINT_FILE = Path("data/checkpoint.json")


class PipelineCheckpoint:
    """
    파이프라인 체크포인트 관리자.

    TMDB API 수집과 Kaggle 보강 파이프라인의 진행 상태를 JSON 파일로 관리한다.
    파이프라인이 중단되면 마지막 체크포인트부터 이어서 진행할 수 있다.

    각 ID 집합은 set으로 관리되어 O(1) 멤버십 검사가 가능하다.
    save() 시 sorted list로 변환하여 JSON 직렬화한다.
    """

    def __init__(self, filepath: Path = CHECKPOINT_FILE) -> None:
        self.filepath = filepath
        self.tmdb_api_loaded_ids: set[int] = set()   # TMDB API로 수집+적재 완료된 영화 ID
        self.kaggle_loaded_ids: set[int] = set()     # Kaggle CSV에서 적재 완료된 영화 ID
        self.embedded_ids: set[int] = set()          # 임베딩 벡터 생성 완료된 영화 ID
        self.failed_ids: set[int] = set()            # 처리 실패한 영화 ID (디버깅용)
        self.last_updated: str = ""
        self.tmdb_api_total_collected: int = 0       # TMDB API 수집 시도 총 수
        self.kaggle_total_available: int = 0         # Kaggle에서 보강 가능한 총 수

    def load(self) -> None:
        """
        체크포인트 파일에서 상태를 로드한다.

        파일이 존재하지 않으면 초기 상태를 유지하고 새로운 체크포인트를 생성한다.
        """
        if self.filepath.exists():
            data = json.loads(self.filepath.read_text())
            self.tmdb_api_loaded_ids = set(data.get("tmdb_api_loaded_ids", []))
            self.kaggle_loaded_ids = set(data.get("kaggle_loaded_ids", []))
            self.embedded_ids = set(data.get("embedded_ids", []))
            self.failed_ids = set(data.get("failed_ids", []))
            self.last_updated = data.get("last_updated", "")
            self.tmdb_api_total_collected = data.get("tmdb_api_total_collected", 0)
            self.kaggle_total_available = data.get("kaggle_total_available", 0)
            logger.info(
                "checkpoint_loaded",
                tmdb_api=len(self.tmdb_api_loaded_ids),
                kaggle=len(self.kaggle_loaded_ids),
                embedded=len(self.embedded_ids),
                failed=len(self.failed_ids),
            )
        else:
            logger.info("checkpoint_not_found_creating_new")

    def save(self) -> None:
        """
        현재 체크포인트 상태를 JSON 파일에 저장한다.

        set을 sorted list로 변환하여 직렬화한다.
        배치 처리 완료 시마다 호출하여 중단 복구를 보장한다.
        """
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now().isoformat()
        data = {
            "tmdb_api_loaded_ids": sorted(self.tmdb_api_loaded_ids),
            "kaggle_loaded_ids": sorted(self.kaggle_loaded_ids),
            "embedded_ids": sorted(self.embedded_ids),
            "failed_ids": sorted(self.failed_ids),
            "last_updated": self.last_updated,
            "tmdb_api_total_collected": self.tmdb_api_total_collected,
            "kaggle_total_available": self.kaggle_total_available,
        }
        self.filepath.write_text(json.dumps(data))
        logger.info(
            "checkpoint_saved",
            tmdb_api=len(self.tmdb_api_loaded_ids),
            kaggle=len(self.kaggle_loaded_ids),
            total=len(self.all_loaded_ids),
        )

    @property
    def all_loaded_ids(self) -> set[int]:
        """적재 완료된 전체 ID (TMDB API + Kaggle)."""
        return self.tmdb_api_loaded_ids | self.kaggle_loaded_ids

    def summary(self) -> str:
        """현재 상태 요약 문자열."""
        total = len(self.all_loaded_ids)
        return (
            f"TMDB API: {len(self.tmdb_api_loaded_ids)} | "
            f"Kaggle: {len(self.kaggle_loaded_ids)} | "
            f"총 적재: {total} | "
            f"실패: {len(self.failed_ids)} | "
            f"최종 업데이트: {self.last_updated}"
        )
