"""
관리자 Tool RAG 단위 테스트 (Step 7a, 2026-04-27).

대상 모듈: `monglepick.tools.admin_tools.tool_rag`

테스트 영역:
1. `_classify_tool_kind` — tool 이름 패턴 → kind 분류 규칙
2. `_tool_name_to_uuid` — deterministic UUID5 (동일 이름 → 동일 ID)
3. `_build_embedding_text` — 임베딩 텍스트 합성
4. `ensure_admin_tool_collection` — 컬렉션 미존재 시 생성, 존재하면 멱등
5. `upsert_admin_tool_embeddings` — 임베딩 + UPSERT 정상 호출
6. `search_similar_tools` — 빈 query / Qdrant 장애 / 임베딩 장애 / role 필터 / score_threshold

Qdrant / Upstage API 실제 호출은 모두 mock 처리 (CI 환경에 인프라 의존성 없음).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from monglepick.tools.admin_tools import ADMIN_TOOL_REGISTRY, ToolSpec
from monglepick.tools.admin_tools.tool_rag import (
    ADMIN_TOOL_COLLECTION,
    _build_embedding_text,
    _classify_tool_kind,
    _tool_name_to_uuid,
    build_admin_tool_candidates_async,
    ensure_admin_tool_collection,
    search_similar_tools,
    upsert_admin_tool_embeddings,
)


# ============================================================
# 1) _classify_tool_kind — kind 분류 규칙
# ============================================================

class TestClassifyToolKind:
    """우선순위: draft > navigate > stats > read(기본)."""

    def test_default_read(self):
        """일반 조회 tool → 'read'."""
        assert _classify_tool_kind("users_list") == "read"
        assert _classify_tool_kind("user_detail") == "read"
        assert _classify_tool_kind("orders_list") == "read"
        assert _classify_tool_kind("faqs_list") == "read"
        assert _classify_tool_kind("system_services_status") == "read"

    def test_draft_suffix(self):
        """*_draft 접미사 → 'draft' (Backend 미호출, form_prefill)."""
        assert _classify_tool_kind("notice_draft") == "draft"
        assert _classify_tool_kind("faq_draft") == "draft"
        assert _classify_tool_kind("banner_draft") == "draft"
        assert _classify_tool_kind("help_article_draft") == "draft"

    def test_goto_prefix(self):
        """goto_* 접두사 → 'navigate' (Backend GET + 화면 이동)."""
        assert _classify_tool_kind("goto_user_detail") == "navigate"
        assert _classify_tool_kind("goto_user_suspend") == "navigate"
        assert _classify_tool_kind("goto_order_refund") == "navigate"
        assert _classify_tool_kind("goto_audit_log") == "navigate"

    def test_stats_prefix(self):
        """stats_* / dashboard_* 접두사 → 'stats'."""
        assert _classify_tool_kind("stats_overview") == "stats"
        assert _classify_tool_kind("stats_trends") == "stats"
        assert _classify_tool_kind("stats_revenue") == "stats"
        assert _classify_tool_kind("dashboard_kpi") == "stats"
        assert _classify_tool_kind("dashboard_recent") == "stats"

    def test_priority_draft_over_goto_or_stats(self):
        """가상의 충돌 케이스 — draft 가 우선 (현재 레지스트리에는 충돌 없음)."""
        assert _classify_tool_kind("goto_something_draft") == "draft"

    def test_priority_goto_over_stats(self):
        """가상의 충돌 케이스 — goto 가 stats 보다 우선."""
        assert _classify_tool_kind("goto_stats_revenue") == "navigate"


# ============================================================
# 2) _tool_name_to_uuid — deterministic
# ============================================================

class TestToolNameToUuid:
    def test_deterministic(self):
        """같은 이름은 항상 같은 UUID."""
        a = _tool_name_to_uuid("users_list")
        b = _tool_name_to_uuid("users_list")
        assert a == b

    def test_different_names_different_uuids(self):
        a = _tool_name_to_uuid("users_list")
        b = _tool_name_to_uuid("user_detail")
        assert a != b

    def test_uuid_format(self):
        """UUID 표준 문자열 형식 (36자, 4 hyphens)."""
        u = _tool_name_to_uuid("users_list")
        assert len(u) == 36
        assert u.count("-") == 4


# ============================================================
# 3) _build_embedding_text — 텍스트 합성
# ============================================================

class TestBuildEmbeddingText:
    def _make_spec(
        self,
        name: str = "fake_tool",
        description: str = "이 tool 은 무엇을 한다",
        examples: list[str] | None = None,
    ) -> ToolSpec:
        from pydantic import BaseModel

        class _Args(BaseModel):
            pass

        async def _handler(*args, **kwargs):  # pragma: no cover
            raise NotImplementedError

        return ToolSpec(
            name=name,
            tier=0,
            required_roles={"SUPER_ADMIN"},
            description=description,
            example_questions=examples or [],
            args_schema=_Args,
            handler=_handler,
        )

    def test_includes_name_and_description(self):
        spec = self._make_spec(name="my_tool", description="설명")
        text = _build_embedding_text(spec)
        assert "my_tool" in text
        assert "설명" in text

    def test_with_examples(self):
        spec = self._make_spec(
            description="설명",
            examples=["예시 1", "예시 2"],
        )
        text = _build_embedding_text(spec)
        assert "예시 1" in text
        assert "예시 2" in text
        assert "예:" in text

    def test_without_examples(self):
        spec = self._make_spec(description="설명", examples=[])
        text = _build_embedding_text(spec)
        assert "예:" not in text

    def test_strips_blank_examples(self):
        spec = self._make_spec(
            description="설명",
            examples=["", "  ", "유효 예시"],
        )
        text = _build_embedding_text(spec)
        assert "유효 예시" in text
        # 공백 예시 자체가 출력에 들어가도 무해하지만, 합쳐진 결과에 빈 토큰이 없어야 함
        assert "/  /" not in text


# ============================================================
# 4) ensure_admin_tool_collection — 컬렉션 보장
# ============================================================

@pytest.mark.asyncio
class TestEnsureCollection:
    async def test_creates_when_missing(self):
        """컬렉션이 없으면 create_collection 호출."""
        mock_client = MagicMock()
        mock_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[MagicMock(name="other_collection")])
        )
        # MagicMock(name=...) 는 .name 을 자동 설정하지 못하므로 spec 으로 우회
        existing = MagicMock()
        existing.name = "other_collection"
        mock_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[existing])
        )
        mock_client.create_collection = AsyncMock()
        mock_client.create_payload_index = AsyncMock()

        with patch(
            "monglepick.tools.admin_tools.tool_rag.get_qdrant",
            new=AsyncMock(return_value=mock_client),
        ):
            await ensure_admin_tool_collection()

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == ADMIN_TOOL_COLLECTION

    async def test_skips_when_exists(self):
        """이미 존재하면 create_collection 미호출 (멱등)."""
        existing = MagicMock()
        existing.name = ADMIN_TOOL_COLLECTION
        mock_client = MagicMock()
        mock_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[existing])
        )
        mock_client.create_collection = AsyncMock()
        mock_client.create_payload_index = AsyncMock()

        with patch(
            "monglepick.tools.admin_tools.tool_rag.get_qdrant",
            new=AsyncMock(return_value=mock_client),
        ):
            await ensure_admin_tool_collection()

        mock_client.create_collection.assert_not_called()

    async def test_payload_index_errors_swallowed(self):
        """payload_index 가 이미 존재해서 에러 던져도 전파하지 않는다."""
        existing = MagicMock()
        existing.name = ADMIN_TOOL_COLLECTION
        mock_client = MagicMock()
        mock_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[existing])
        )
        mock_client.create_collection = AsyncMock()
        mock_client.create_payload_index = AsyncMock(
            side_effect=Exception("index already exists")
        )

        with patch(
            "monglepick.tools.admin_tools.tool_rag.get_qdrant",
            new=AsyncMock(return_value=mock_client),
        ):
            # 예외 전파 없어야 함
            await ensure_admin_tool_collection()


# ============================================================
# 5) upsert_admin_tool_embeddings — 임베딩 + UPSERT
# ============================================================

@pytest.mark.asyncio
class TestUpsertEmbeddings:
    async def test_upserts_full_registry(self):
        """tool_specs=None 이면 ADMIN_TOOL_REGISTRY 전체 upsert."""
        n = len(ADMIN_TOOL_REGISTRY)
        # 가짜 임베딩 — shape (n, 4096)
        fake_vectors = np.zeros((n, 4096), dtype=np.float32)

        mock_client = MagicMock()
        mock_client.upsert = AsyncMock()

        with (
            patch(
                "monglepick.tools.admin_tools.tool_rag.embed_texts",
                return_value=fake_vectors,
            ),
            patch(
                "monglepick.tools.admin_tools.tool_rag.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            count = await upsert_admin_tool_embeddings()

        assert count == n
        mock_client.upsert.assert_called_once()
        upsert_kwargs = mock_client.upsert.call_args.kwargs
        assert upsert_kwargs["collection_name"] == ADMIN_TOOL_COLLECTION
        assert len(upsert_kwargs["points"]) == n

    async def test_upserts_subset(self):
        """일부 tool 만 넘겨도 정상 동작."""
        specs = list(ADMIN_TOOL_REGISTRY.values())[:3]
        fake_vectors = np.zeros((3, 4096), dtype=np.float32)

        mock_client = MagicMock()
        mock_client.upsert = AsyncMock()

        with (
            patch(
                "monglepick.tools.admin_tools.tool_rag.embed_texts",
                return_value=fake_vectors,
            ),
            patch(
                "monglepick.tools.admin_tools.tool_rag.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            count = await upsert_admin_tool_embeddings(specs)

        assert count == 3

    async def test_empty_registry_returns_zero(self):
        """빈 입력은 0 반환."""
        count = await upsert_admin_tool_embeddings([])
        assert count == 0

    async def test_embedding_count_mismatch_raises(self):
        """임베딩 개수가 입력과 다르면 RuntimeError."""
        specs = list(ADMIN_TOOL_REGISTRY.values())[:3]
        bad_vectors = np.zeros((2, 4096), dtype=np.float32)  # 3 expected, 2 returned

        with patch(
            "monglepick.tools.admin_tools.tool_rag.embed_texts",
            return_value=bad_vectors,
        ):
            with pytest.raises(RuntimeError, match="embedding count mismatch"):
                await upsert_admin_tool_embeddings(specs)


# ============================================================
# 6) search_similar_tools
# ============================================================

@pytest.mark.asyncio
class TestSearchSimilarTools:
    async def test_empty_query_returns_empty(self):
        """빈 쿼리는 즉시 빈 리스트."""
        result = await search_similar_tools("")
        assert result == []
        result = await search_similar_tools("   ")
        assert result == []

    async def test_embedding_failure_returns_empty(self):
        """Upstage 장애 시 빈 리스트 (호출측이 fallback)."""
        with patch(
            "monglepick.tools.admin_tools.tool_rag.embed_query_async",
            new=AsyncMock(side_effect=Exception("upstage down")),
        ):
            result = await search_similar_tools("아무 쿼리")
            assert result == []

    async def test_qdrant_failure_returns_empty(self):
        """Qdrant 장애 시 빈 리스트."""
        with (
            patch(
                "monglepick.tools.admin_tools.tool_rag.embed_query_async",
                new=AsyncMock(return_value=np.zeros(4096, dtype=np.float32)),
            ),
            patch(
                "monglepick.tools.admin_tools.tool_rag.get_qdrant",
                new=AsyncMock(side_effect=Exception("qdrant down")),
            ),
        ):
            result = await search_similar_tools("아무 쿼리")
            assert result == []

    async def test_returns_top_k_sorted(self):
        """Qdrant 결과를 score 내림차순으로 반환."""
        hits = [
            MagicMock(payload={"name": "users_list"}, score=0.85),
            MagicMock(payload={"name": "user_detail"}, score=0.95),
            MagicMock(payload={"name": "user_activity"}, score=0.55),
        ]
        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(return_value=MagicMock(points=hits))

        with (
            patch(
                "monglepick.tools.admin_tools.tool_rag.embed_query_async",
                new=AsyncMock(return_value=np.zeros(4096, dtype=np.float32)),
            ),
            patch(
                "monglepick.tools.admin_tools.tool_rag.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            result = await search_similar_tools("사용자 정보")

        assert result == ["user_detail", "users_list", "user_activity"]

    async def test_filters_by_allowed_names(self):
        """allowed_tool_names 교집합 — role 매트릭스 필터."""
        hits = [
            MagicMock(payload={"name": "users_list"}, score=0.95),
            MagicMock(payload={"name": "goto_user_suspend"}, score=0.80),
            MagicMock(payload={"name": "user_detail"}, score=0.70),
        ]
        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(return_value=MagicMock(points=hits))

        with (
            patch(
                "monglepick.tools.admin_tools.tool_rag.embed_query_async",
                new=AsyncMock(return_value=np.zeros(4096, dtype=np.float32)),
            ),
            patch(
                "monglepick.tools.admin_tools.tool_rag.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            # SUPPORT_ADMIN 권한이 있어 goto_user_suspend 는 막힌다고 가정
            result = await search_similar_tools(
                "사용자 관리",
                allowed_tool_names={"users_list", "user_detail"},
            )

        assert "goto_user_suspend" not in result
        assert result == ["users_list", "user_detail"]

    async def test_skips_payload_without_name(self):
        """payload 에 name 이 없는 비정상 point 는 스킵."""
        hits = [
            MagicMock(payload={"name": "users_list"}, score=0.90),
            MagicMock(payload={}, score=0.80),  # malformed
            MagicMock(payload=None, score=0.70),  # null payload
        ]
        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(return_value=MagicMock(points=hits))

        with (
            patch(
                "monglepick.tools.admin_tools.tool_rag.embed_query_async",
                new=AsyncMock(return_value=np.zeros(4096, dtype=np.float32)),
            ),
            patch(
                "monglepick.tools.admin_tools.tool_rag.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            result = await search_similar_tools("아무 쿼리")

        assert result == ["users_list"]


# ============================================================
# 7) build_admin_tool_candidates_async — Step 7b 하이브리드 머지
# ============================================================

@pytest.mark.asyncio
class TestBuildCandidates:
    """`tool_filter` + (옵션) `tool_rag` 머지 — `tool_selector` 진입점."""

    async def test_rag_disabled_returns_filter_only(self, monkeypatch):
        """ADMIN_TOOL_RAG_ENABLED 미설정 → filter 결과만 반환, RAG 미호출."""
        monkeypatch.delenv("ADMIN_TOOL_RAG_ENABLED", raising=False)

        with (
            patch(
                "monglepick.tools.admin_tools.tool_filter.shortlist_tools_by_category",
                return_value=["users_list", "user_detail"],
            ) as mock_filter,
            patch(
                "monglepick.tools.admin_tools.tool_rag.search_similar_tools",
                new=AsyncMock(return_value=["should_not_appear"]),
            ) as mock_rag,
        ):
            result = await build_admin_tool_candidates_async(
                user_message="사용자 목록",
                admin_role="ADMIN",
                intent_kind="query",
                max_tools=30,
            )

        assert result == ["users_list", "user_detail"]
        mock_filter.assert_called_once()
        mock_rag.assert_not_called()

    async def test_rag_enabled_merges_with_dedup(self, monkeypatch):
        """RAG 활성 → filter 우선 + RAG 보강 + 중복 제거 + 상한 절단."""
        monkeypatch.setenv("ADMIN_TOOL_RAG_ENABLED", "true")

        with (
            patch(
                "monglepick.tools.admin_tools.tool_filter.shortlist_tools_by_category",
                return_value=["users_list", "user_detail"],
            ),
            patch(
                "monglepick.tools.admin_tools.tool_rag.search_similar_tools",
                new=AsyncMock(return_value=["users_list", "goto_user_suspend", "user_activity"]),
            ),
        ):
            result = await build_admin_tool_candidates_async(
                user_message="사용자 정지 처리",
                admin_role="ADMIN",
                intent_kind="action",
                max_tools=30,
            )

        # filter 결과가 앞, RAG-only 가 뒤. users_list 중복 제거.
        assert result == ["users_list", "user_detail", "goto_user_suspend", "user_activity"]

    async def test_rag_enabled_max_tools_truncates(self, monkeypatch):
        """RAG 머지 후 max_tools 로 절단."""
        monkeypatch.setenv("ADMIN_TOOL_RAG_ENABLED", "true")

        with (
            patch(
                "monglepick.tools.admin_tools.tool_filter.shortlist_tools_by_category",
                return_value=["a", "b"],
            ),
            patch(
                "monglepick.tools.admin_tools.tool_rag.search_similar_tools",
                new=AsyncMock(return_value=["c", "d", "e", "f"]),
            ),
        ):
            result = await build_admin_tool_candidates_async(
                user_message="아무 쿼리",
                admin_role="ADMIN",
                intent_kind="query",
                max_tools=3,
            )

        # 머지 ["a","b","c","d","e","f"] → 상한 3 절단
        assert result == ["a", "b", "c"]

    async def test_rag_enabled_failure_keeps_filter(self, monkeypatch):
        """RAG 가 빈 리스트(장애/없음) 반환해도 filter 결과 보존."""
        monkeypatch.setenv("ADMIN_TOOL_RAG_ENABLED", "1")

        with (
            patch(
                "monglepick.tools.admin_tools.tool_filter.shortlist_tools_by_category",
                return_value=["users_list", "user_detail"],
            ),
            patch(
                "monglepick.tools.admin_tools.tool_rag.search_similar_tools",
                new=AsyncMock(return_value=[]),  # Qdrant/Upstage 장애 시뮬레이션
            ),
        ):
            result = await build_admin_tool_candidates_async(
                user_message="아무 쿼리",
                admin_role="ADMIN",
                intent_kind="query",
                max_tools=30,
            )

        assert result == ["users_list", "user_detail"]

    async def test_empty_role_skips_rag(self, monkeypatch):
        """admin_role 매트릭스 결과가 0 이면 RAG 건너뛰고 filter 그대로 반환."""
        monkeypatch.setenv("ADMIN_TOOL_RAG_ENABLED", "true")

        with (
            patch(
                "monglepick.tools.admin_tools.tool_filter.shortlist_tools_by_category",
                return_value=[],
            ),
            patch(
                "monglepick.tools.admin_tools.list_tools_for_role",
                return_value=[],
            ),
            patch(
                "monglepick.tools.admin_tools.tool_rag.search_similar_tools",
                new=AsyncMock(return_value=["x"]),
            ) as mock_rag,
        ):
            result = await build_admin_tool_candidates_async(
                user_message="아무 쿼리",
                admin_role="",  # 권한 없음 가정
                intent_kind="query",
                max_tools=30,
            )

        assert result == []
        mock_rag.assert_not_called()
