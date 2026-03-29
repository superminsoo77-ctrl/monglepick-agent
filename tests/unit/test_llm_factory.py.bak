"""
LLM 팩토리 단위 테스트 — 하이브리드 라우팅 (Ollama + Solar API) 대응.

테스트 대상:
- ChatOllama 생성자 파라미터 전달 검증
- 동일 파라미터 캐시 히트 (싱글턴)
- 다른 파라미터 별도 인스턴스
- LLM_MODE별 라우팅 (local_only / hybrid / api_only)
- get_intent_llm()이 모드별로 올바른 백엔드를 사용하는지 확인
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import monglepick.llm.factory as factory_module
from monglepick.agents.chat.models import IntentResult
from monglepick.config import settings


def _clear_all_caches():
    """모든 LLM 캐시를 초기화한다 (테스트 간 격리)."""
    factory_module._ollama_cache.clear()
    factory_module._solar_cache.clear()
    factory_module._structured_cache.clear()


class TestGetLlm:
    """get_llm() / get_ollama_llm() 함수 테스트."""

    def setup_method(self):
        """각 테스트 전 캐시 초기화."""
        _clear_all_caches()

    def test_creates_chat_ollama_with_correct_params(self):
        """ChatOllama가 올바른 파라미터로 생성된다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            factory_module.get_llm(
                model="test-model",
                temperature=0.3,
                format="json",
            )
            mock_cls.assert_called_once_with(
                model="test-model",
                temperature=0.3,
                base_url=settings.OLLAMA_BASE_URL,
                format="json",
            )

    def test_cache_hit_same_params(self):
        """동일 파라미터로 호출하면 캐시에서 동일 인스턴스를 반환한다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            llm1 = factory_module.get_llm(model="test", temperature=0.5)
            llm2 = factory_module.get_llm(model="test", temperature=0.5)

            # 생성자는 1번만 호출
            assert mock_cls.call_count == 1
            # 동일 인스턴스
            assert llm1 is llm2

    def test_different_params_separate_instances(self):
        """다른 파라미터로 호출하면 별도 인스턴스를 생성한다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_cls:
            mock_cls.side_effect = [MagicMock(), MagicMock()]

            llm1 = factory_module.get_llm(model="model-a", temperature=0.3)
            llm2 = factory_module.get_llm(model="model-b", temperature=0.7)

            assert mock_cls.call_count == 2
            assert llm1 is not llm2

    def test_default_values(self):
        """기본값이 settings.CONVERSATION_MODEL, temp=0.5이다."""
        with patch("monglepick.llm.factory.ChatOllama") as mock_cls:
            mock_cls.return_value = MagicMock()
            factory_module.get_llm()
            mock_cls.assert_called_once_with(
                model=settings.CONVERSATION_MODEL,
                temperature=0.5,
                base_url=settings.OLLAMA_BASE_URL,
            )


class TestGetStructuredLlm:
    """get_structured_llm() 함수 테스트."""

    def setup_method(self):
        """각 테스트 전 캐시 초기화."""
        _clear_all_caches()

    def test_with_structured_output_called(self):
        """local_only 모드에서 with_structured_output이 schema와 함께 호출된다."""
        with (
            patch("monglepick.llm.factory.ChatOllama") as mock_cls,
            patch.object(settings, "LLM_MODE", "local_only"),
        ):
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            factory_module.get_structured_llm(
                schema=IntentResult,
                model="test-model",
                temperature=0.1,
            )

            mock_instance.with_structured_output.assert_called_once_with(
                IntentResult, method="json_schema",
            )

    def test_structured_cache_hit(self):
        """동일 schema+model+temp로 호출하면 캐시에서 반환한다."""
        with (
            patch("monglepick.llm.factory.ChatOllama") as mock_cls,
            patch.object(settings, "LLM_MODE", "local_only"),
        ):
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            r1 = factory_module.get_structured_llm(IntentResult, "test", 0.1)
            r2 = factory_module.get_structured_llm(IntentResult, "test", 0.1)

            assert mock_instance.with_structured_output.call_count == 1
            assert r1 is r2


class TestConvenienceFunctions:
    """용도별 편의 함수 테스트 — LLM_MODE별 라우팅 검증."""

    def setup_method(self):
        _clear_all_caches()

    def test_get_intent_llm_local_only(self):
        """local_only 모드에서 get_intent_llm()이 settings.INTENT_MODEL(Ollama)을 사용한다."""
        with (
            patch("monglepick.llm.factory.ChatOllama") as mock_cls,
            patch.object(settings, "LLM_MODE", "local_only"),
        ):
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            factory_module.get_intent_llm()

            # Ollama가 INTENT_MODEL로 호출됨
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == settings.INTENT_MODEL
            assert call_kwargs["temperature"] == 0.1

    def test_get_intent_llm_hybrid_uses_solar(self):
        """hybrid 모드에서 get_intent_llm()이 Solar API를 사용한다."""
        with (
            patch("monglepick.llm.factory.ChatOpenAI") as mock_solar_cls,
            patch.object(settings, "LLM_MODE", "hybrid"),
        ):
            mock_solar_instance = MagicMock()
            mock_solar_cls.return_value = mock_solar_instance

            factory_module.get_intent_llm()

            # Solar API가 호출됨
            call_kwargs = mock_solar_cls.call_args[1]
            assert call_kwargs["model"] == settings.SOLAR_API_MODEL
            assert call_kwargs["base_url"] == settings.SOLAR_API_BASE_URL

    def test_get_conversation_llm_local_only(self):
        """local_only 모드에서 get_conversation_llm()이 settings.CONVERSATION_MODEL을 사용한다."""
        with (
            patch("monglepick.llm.factory.ChatOllama") as mock_cls,
            patch.object(settings, "LLM_MODE", "local_only"),
        ):
            mock_cls.return_value = MagicMock()

            factory_module.get_conversation_llm()

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == settings.CONVERSATION_MODEL
            assert call_kwargs["temperature"] == 0.5

    def test_get_conversation_llm_hybrid_uses_mongle(self):
        """hybrid 모드에서 get_conversation_llm()이 몽글이(MONGLE_MODEL)를 사용한다."""
        with (
            patch("monglepick.llm.factory.ChatOllama") as mock_cls,
            patch.object(settings, "LLM_MODE", "hybrid"),
        ):
            mock_cls.return_value = MagicMock()

            factory_module.get_conversation_llm()

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == settings.MONGLE_MODEL
            assert call_kwargs["temperature"] == settings.MONGLE_TEMPERATURE

    def test_get_conversation_llm_api_only_uses_solar(self):
        """api_only 모드에서 get_conversation_llm()이 Solar API를 사용한다."""
        with (
            patch("monglepick.llm.factory.ChatOpenAI") as mock_solar_cls,
            patch.object(settings, "LLM_MODE", "api_only"),
        ):
            mock_solar_cls.return_value = MagicMock()

            factory_module.get_conversation_llm()

            call_kwargs = mock_solar_cls.call_args[1]
            assert call_kwargs["model"] == settings.SOLAR_API_MODEL

    def test_get_explanation_llm_local_only(self):
        """local_only 모드에서 get_explanation_llm()이 settings.EXPLANATION_MODEL을 사용한다."""
        with (
            patch("monglepick.llm.factory.ChatOllama") as mock_cls,
            patch.object(settings, "LLM_MODE", "local_only"),
        ):
            mock_cls.return_value = MagicMock()

            factory_module.get_explanation_llm()

            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == settings.EXPLANATION_MODEL
