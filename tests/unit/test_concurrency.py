"""
동시성 제어 모듈 단위 테스트.

테스트 대상:
- 모델별 세마포어 acquire/release 정상 동작
- 세마포어 동시 제한 (LLM_PER_MODEL_CONCURRENCY 초과 시 대기)
- 대기 시간 측정
- guarded_ainvoke 래퍼 정상 동작
- guarded_ainvoke 에러 시 세마포어 반환 보장
- reset_semaphores 테스트 격리
- generate_explanations_batch 직렬화 + MAX_EXPLANATION_MOVIES 제한
- API 글로벌 세마포어 설정값 확인
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monglepick.config import settings
from monglepick.llm.concurrency import (
    _model_semaphores,
    acquire_model_slot,
    release_model_slot,
    reset_semaphores,
)


@pytest.fixture(autouse=True)
def _clean_semaphores():
    """각 테스트 전후로 세마포어 상태를 초기화한다."""
    reset_semaphores()
    yield
    reset_semaphores()


# ============================================================
# acquire / release 기본 동작
# ============================================================

class TestModelSlotAcquireRelease:
    """모델별 세마포어 acquire/release 기본 동작 테스트."""

    @pytest.mark.asyncio
    async def test_acquire_creates_semaphore_on_first_call(self):
        """첫 acquire 시 세마포어가 자동 생성된다."""
        assert "test-model" not in _model_semaphores
        await acquire_model_slot("test-model")
        assert "test-model" in _model_semaphores
        release_model_slot("test-model")

    @pytest.mark.asyncio
    async def test_acquire_returns_wait_time(self):
        """acquire는 대기 시간(초)을 반환한다."""
        wait_sec = await acquire_model_slot("test-model")
        assert isinstance(wait_sec, float)
        assert wait_sec >= 0.0
        release_model_slot("test-model")

    @pytest.mark.asyncio
    async def test_release_allows_next_acquire(self):
        """release 후 다른 코루틴이 acquire할 수 있다."""
        # 세마포어 한도(기본 2)까지 acquire
        await acquire_model_slot("test-model")
        await acquire_model_slot("test-model")

        # 한도 초과 → 대기 발생 확인을 위해 release 1개
        release_model_slot("test-model")

        # release 후 즉시 acquire 가능
        wait_sec = await acquire_model_slot("test-model")
        assert wait_sec < 0.5  # 즉시 획득

        # 정리
        release_model_slot("test-model")
        release_model_slot("test-model")

    @pytest.mark.asyncio
    async def test_different_models_use_different_semaphores(self):
        """모델별로 독립적인 세마포어가 생성된다."""
        await acquire_model_slot("model-a")
        await acquire_model_slot("model-b")

        assert "model-a" in _model_semaphores
        assert "model-b" in _model_semaphores
        assert _model_semaphores["model-a"] is not _model_semaphores["model-b"]

        release_model_slot("model-a")
        release_model_slot("model-b")


# ============================================================
# 동시 제한 (대기 발생)
# ============================================================

class TestConcurrencyLimit:
    """세마포어 동시 제한 테스트."""

    @pytest.mark.asyncio
    async def test_blocks_when_limit_reached(self):
        """
        LLM_PER_MODEL_CONCURRENCY(2)까지 acquire 후,
        추가 acquire는 대기한다.
        """
        model = "blocking-test-model"

        # 한도(2)까지 acquire
        await acquire_model_slot(model)
        await acquire_model_slot(model)

        # 3번째 acquire → 대기 발생
        acquired = asyncio.Event()

        async def try_acquire():
            await acquire_model_slot(model)
            acquired.set()

        task = asyncio.create_task(try_acquire())

        # 잠시 대기 후 아직 acquire 안 됨 확인
        await asyncio.sleep(0.05)
        assert not acquired.is_set(), "세마포어 한도 초과 시 대기해야 한다"

        # 1개 release → 대기 중인 acquire 성공
        release_model_slot(model)
        await asyncio.wait_for(task, timeout=1.0)
        assert acquired.is_set()

        # 정리
        release_model_slot(model)
        release_model_slot(model)


# ============================================================
# guarded_ainvoke 래퍼
# ============================================================

class TestGuardedAinvoke:
    """guarded_ainvoke 래퍼 테스트."""

    @pytest.mark.asyncio
    async def test_returns_llm_response(self):
        """guarded_ainvoke가 LLM 응답을 정상 반환한다."""
        from monglepick.llm.factory import guarded_ainvoke

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "테스트 응답"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await guarded_ainvoke(mock_llm, "프롬프트", model="test-model")
        assert result is mock_response
        mock_llm.ainvoke.assert_awaited_once_with("프롬프트")

    @pytest.mark.asyncio
    async def test_releases_slot_on_error(self):
        """LLM 에러 시에도 세마포어 슬롯이 반환된다."""
        from monglepick.llm.factory import guarded_ainvoke

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM 에러"))

        model = "error-test-model"
        sem = _model_semaphores.get(model)

        with pytest.raises(RuntimeError, match="LLM 에러"):
            await guarded_ainvoke(mock_llm, "프롬프트", model=model)

        # 에러 후 세마포어가 정상 반환되었는지 확인 (즉시 acquire 가능)
        wait_sec = await acquire_model_slot(model)
        assert wait_sec < 0.1
        release_model_slot(model)

    @pytest.mark.asyncio
    async def test_passes_request_id_for_logging(self):
        """request_id가 acquire_model_slot에 전달된다."""
        from monglepick.llm.factory import guarded_ainvoke

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock())

        with patch("monglepick.llm.factory.acquire_model_slot", new_callable=AsyncMock) as mock_acquire:
            mock_acquire.return_value = 0.0
            await guarded_ainvoke(
                mock_llm, "프롬프트", model="test-model", request_id="req-123",
            )
            mock_acquire.assert_awaited_once_with("test-model", "req-123")


# ============================================================
# reset_semaphores (테스트 격리)
# ============================================================

class TestResetSemaphores:
    """reset_semaphores 함수 테스트."""

    @pytest.mark.asyncio
    async def test_clears_all_semaphores(self):
        """reset_semaphores 호출 시 모든 세마포어가 제거된다."""
        await acquire_model_slot("model-x")
        await acquire_model_slot("model-y")
        release_model_slot("model-x")
        release_model_slot("model-y")

        assert len(_model_semaphores) == 2
        reset_semaphores()
        assert len(_model_semaphores) == 0


# ============================================================
# explanation_chain 배치 직렬화 + 개수 제한
# ============================================================

class TestExplanationBatchSerial:
    """explanation_chain 배치 직렬화 + MAX_EXPLANATION_MOVIES 제한 테스트."""

    @pytest.mark.asyncio
    async def test_batch_respects_max_explanation_movies(self, mock_ollama):
        """MAX_EXPLANATION_MOVIES(3) 초과 영화는 fallback 설명을 사용한다."""
        from monglepick.chains.explanation_chain import generate_explanations_batch

        mock_ollama.set_response("LLM으로 생성한 추천 이유")

        movies = [
            {"title": f"영화{i}", "genres": ["액션"], "rating": 7.0 + i * 0.5}
            for i in range(5)
        ]

        with patch.object(settings, "MAX_EXPLANATION_MOVIES", 3):
            results = await generate_explanations_batch(movies)

        assert len(results) == 5
        # 처음 3편은 LLM 응답 (mock_ollama가 반환하는 내용)
        for i in range(3):
            assert "LLM으로 생성한 추천 이유" in results[i]
        # 4~5편은 fallback 템플릿 (장르 기반)
        for i in range(3, 5):
            assert "액션" in results[i]

    @pytest.mark.asyncio
    async def test_batch_serial_execution_order(self, mock_ollama):
        """배치가 순차 실행되며 올바른 순서로 결과를 반환한다."""
        from monglepick.chains.explanation_chain import generate_explanations_batch

        # 각 영화별로 다른 응답을 순차 반환
        mock_ollama.set_response_sequence([
            "첫 번째 영화 추천 이유",
            "두 번째 영화 추천 이유",
            "세 번째 영화 추천 이유",
        ])

        movies = [
            {"title": "영화A", "genres": ["드라마"], "rating": 8.0},
            {"title": "영화B", "genres": ["코미디"], "rating": 7.5},
            {"title": "영화C", "genres": ["액션"], "rating": 9.0},
        ]

        results = await generate_explanations_batch(movies)
        assert len(results) == 3
        assert "첫 번째" in results[0]
        assert "두 번째" in results[1]
        assert "세 번째" in results[2]


# ============================================================
# API 글로벌 세마포어 설정값 확인
# ============================================================

class TestGraphSemaphore:
    """API 글로벌 세마포어 설정 테스트."""

    def test_max_concurrent_requests_default(self):
        """MAX_CONCURRENT_REQUESTS 기본값이 3이다."""
        assert settings.MAX_CONCURRENT_REQUESTS == 3

    def test_llm_per_model_concurrency_default(self):
        """LLM_PER_MODEL_CONCURRENCY 기본값이 2이다."""
        assert settings.LLM_PER_MODEL_CONCURRENCY == 2

    def test_max_explanation_movies_default(self):
        """MAX_EXPLANATION_MOVIES 기본값이 3이다."""
        assert settings.MAX_EXPLANATION_MOVIES == 3

    def test_graph_semaphore_exists_in_chat_module(self):
        """chat.py에 _graph_semaphore가 존재한다."""
        from monglepick.api.chat import _graph_semaphore
        assert isinstance(_graph_semaphore, asyncio.Semaphore)
