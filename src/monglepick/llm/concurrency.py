"""
모델별 동시성 제어 모듈.

Ollama는 GPU 추론을 모델당 직렬 처리하므로,
동일 모델에 대한 동시 LLM 호출 수를 세마포어로 제한한다.

모델명을 키로 asyncio.Semaphore를 관리하며,
acquire 시 대기 시간을 측정하여 structlog로 로깅한다.

사용법:
    from monglepick.llm.concurrency import acquire_model_slot, release_model_slot

    await acquire_model_slot("exaone-32b:latest", request_id="req-123")
    try:
        result = await llm.ainvoke(prompt)
    finally:
        release_model_slot("exaone-32b:latest")
"""

from __future__ import annotations

import asyncio
import time

import structlog

from monglepick.config import settings

logger = structlog.get_logger()

# ============================================================
# 모델명 → asyncio.Semaphore 매핑 (모듈 레벨 싱글턴)
# 모델별로 동시에 활성화할 수 있는 LLM 호출 수를 제한한다.
# ============================================================
_model_semaphores: dict[str, asyncio.Semaphore] = {}


def _get_semaphore(model: str) -> asyncio.Semaphore:
    """
    모델명에 해당하는 세마포어를 반환한다 (없으면 생성).

    LLM_PER_MODEL_CONCURRENCY 설정값으로 세마포어를 초기화한다.
    동일 모델명에 대해 항상 같은 세마포어 인스턴스를 반환한다.

    Args:
        model: Ollama 모델명 (예: "exaone-32b:latest", "qwen3.5:35b-a3b")

    Returns:
        해당 모델의 asyncio.Semaphore 인스턴스
    """
    if model not in _model_semaphores:
        _model_semaphores[model] = asyncio.Semaphore(
            settings.LLM_PER_MODEL_CONCURRENCY,
        )
        logger.info(
            "model_semaphore_created",
            model=model,
            concurrency_limit=settings.LLM_PER_MODEL_CONCURRENCY,
        )
    return _model_semaphores[model]


async def acquire_model_slot(model: str, request_id: str = "") -> float:
    """
    모델별 세마포어 슬롯을 획득한다 (대기 가능).

    슬롯이 모두 사용 중이면 대기하며, 대기 시간을 측정하여 로깅한다.
    대기 시간이 100ms 이상이면 INFO, 미만이면 DEBUG 레벨로 로깅한다.

    Args:
        model: Ollama 모델명
        request_id: 요청 식별자 (로깅용, 빈 문자열이면 생략)

    Returns:
        대기 시간 (초 단위, float)
    """
    sem = _get_semaphore(model)

    # 대기 시간 측정 시작
    wait_start = time.perf_counter()
    await sem.acquire()
    wait_sec = time.perf_counter() - wait_start

    # 대기 시간에 따라 로그 레벨 분기
    log_data = {
        "model": model,
        "wait_ms": round(wait_sec * 1000, 1),
        "request_id": request_id,
    }
    if wait_sec >= 0.1:
        # 100ms 이상 대기 → 큐 경합 발생, INFO 레벨
        logger.info("model_slot_acquired_after_wait", **log_data)
    else:
        # 즉시 획득 → DEBUG 레벨 (과다 로깅 방지)
        logger.debug("model_slot_acquired", **log_data)

    return wait_sec


def release_model_slot(model: str) -> None:
    """
    모델별 세마포어 슬롯을 반환한다.

    반드시 acquire_model_slot()과 쌍으로 호출해야 한다.
    try/finally 블록에서 사용을 권장한다.

    Args:
        model: Ollama 모델명
    """
    sem = _get_semaphore(model)
    sem.release()
    logger.debug("model_slot_released", model=model)


def reset_semaphores() -> None:
    """
    모든 모델 세마포어를 초기화한다 (테스트용).

    테스트 간 격리를 위해 세마포어 상태를 완전히 리셋한다.
    프로덕션 코드에서는 호출하지 않는다.
    """
    _model_semaphores.clear()
    logger.debug("model_semaphores_reset")
