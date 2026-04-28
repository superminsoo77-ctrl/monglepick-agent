"""
고객센터 정책 RAG 검색 체인 + lookup_policy tool 단위 테스트 (2026-04-28).

대상 모듈:
- `src/monglepick/chains/support_policy_rag_chain.py` — search_policy()
- `src/monglepick/tools/support_tools/__init__.py`     — 레지스트리
- `src/monglepick/tools/support_tools/policy.py`       — lookup_policy tool

테스트 영역:
1. search_policy() 정상 동작 — Qdrant 모킹, 결과 리스트 길이·필드 매핑
2. topic_filter 미지정 시 Filter 없음
3. topic_filter 지정 시 FieldCondition(policy_topic) 적용
4. Qdrant 예외 시 빈 리스트 반환 (에러 전파 X)
5. 임베딩 타임아웃 시 빈 리스트 반환
6. lookup_policy handler PolicyChunk → dict 변환 정확성
7. requires_login=False — 게스트(ToolContext.is_guest=True) 에서도 정상 응답
8. 레지스트리 등록 확인 — SUPPORT_TOOL_REGISTRY["lookup_policy"] 존재

Qdrant / Upstage API 실제 호출은 모두 mock 처리 (CI 환경 인프라 의존성 없음).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from monglepick.chains.support_policy_rag_chain import PolicyChunk, search_policy
from monglepick.tools.support_tools import SUPPORT_TOOL_REGISTRY, ToolContext
from monglepick.tools.support_tools.policy import (
    LookupPolicyArgs,
    _handle_lookup_policy,
)


# ============================================================
# 헬퍼 — Qdrant 히트(scored point) mock 생성
# ============================================================

def _make_qdrant_hit(
    score: float = 0.87,
    doc_id: str = "리워드_결제_설계서",
    section: str = "§4.5 AI 쿼터 정책",
    headings: list[str] | None = None,
    policy_topic: str = "ai_quota",
    text: str = "BRONZE 등급은 하루 AI 추천을 3회 사용할 수 있습니다.",
) -> MagicMock:
    """Qdrant scored point mock 을 생성한다."""
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "doc_id": doc_id,
        "doc_path": f"docs/{doc_id}.md",
        "section": section,
        "headings": headings or ["## AI 쿼터", "### BRONZE"],
        "policy_topic": policy_topic,
        "text": text,
    }
    return hit


def _make_qdrant_response(hits: list[MagicMock]) -> MagicMock:
    """query_points 반환값 mock 을 생성한다."""
    resp = MagicMock()
    resp.points = hits
    return resp


# ============================================================
# 1. search_policy() 정상 동작
# ============================================================

@pytest.mark.asyncio
class TestSearchPolicyNormal:
    """search_policy() 가 Qdrant 결과를 PolicyChunk 리스트로 올바르게 변환한다."""

    async def test_returns_correct_chunk_count(self):
        """top_k=3 요청 시 3개 청크가 반환된다."""
        hits = [_make_qdrant_hit(score=0.9 - i * 0.1) for i in range(3)]
        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(
            return_value=_make_qdrant_response(hits)
        )
        mock_vec = np.zeros(4096, dtype=np.float32)

        with (
            patch(
                "monglepick.chains.support_policy_rag_chain.embed_query_async",
                new=AsyncMock(return_value=mock_vec),
            ),
            patch(
                "monglepick.chains.support_policy_rag_chain.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            result = await search_policy("AI 추천 횟수", top_k=3)

        assert len(result) == 3

    async def test_chunk_fields_mapped_correctly(self):
        """Qdrant payload 필드가 PolicyChunk 속성에 정확히 매핑된다."""
        hit = _make_qdrant_hit(
            score=0.92,
            doc_id="리워드_결제_설계서",
            section="§4.5 AI 쿼터",
            headings=["## 등급 혜택", "### BRONZE"],
            policy_topic="grade_benefit",
            text="BRONZE 등급은 AI 3회입니다.",
        )
        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(
            return_value=_make_qdrant_response([hit])
        )
        mock_vec = np.zeros(4096, dtype=np.float32)

        with (
            patch(
                "monglepick.chains.support_policy_rag_chain.embed_query_async",
                new=AsyncMock(return_value=mock_vec),
            ),
            patch(
                "monglepick.chains.support_policy_rag_chain.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            result = await search_policy("브론즈 등급 혜택")

        assert len(result) == 1
        chunk = result[0]
        assert isinstance(chunk, PolicyChunk)
        assert chunk.doc_id == "리워드_결제_설계서"
        assert chunk.section == "§4.5 AI 쿼터"
        assert chunk.headings == ["## 등급 혜택", "### BRONZE"]
        assert chunk.policy_topic == "grade_benefit"
        assert chunk.text == "BRONZE 등급은 AI 3회입니다."
        assert chunk.score == pytest.approx(0.92, abs=1e-6)

    async def test_returns_empty_list_when_no_hits(self):
        """Qdrant 결과가 0건이면 빈 리스트를 반환한다."""
        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(
            return_value=_make_qdrant_response([])
        )
        mock_vec = np.zeros(4096, dtype=np.float32)

        with (
            patch(
                "monglepick.chains.support_policy_rag_chain.embed_query_async",
                new=AsyncMock(return_value=mock_vec),
            ),
            patch(
                "monglepick.chains.support_policy_rag_chain.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            result = await search_policy("알 수 없는 질문")

        assert result == []


# ============================================================
# 2. topic_filter 미지정 시 Filter 없음
# ============================================================

@pytest.mark.asyncio
class TestSearchPolicyNoTopicFilter:
    """topic_filter=None 이면 query_filter=None 으로 전체 컬렉션 검색한다."""

    async def test_no_filter_when_topic_is_none(self):
        """topic_filter 미지정 시 query_points 에 query_filter=None 전달된다."""
        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(
            return_value=_make_qdrant_response([])
        )
        mock_vec = np.zeros(4096, dtype=np.float32)

        with (
            patch(
                "monglepick.chains.support_policy_rag_chain.embed_query_async",
                new=AsyncMock(return_value=mock_vec),
            ),
            patch(
                "monglepick.chains.support_policy_rag_chain.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            await search_policy("정책 질문", topic_filter=None)

        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is None


# ============================================================
# 3. topic_filter 지정 시 FieldCondition 적용
# ============================================================

@pytest.mark.asyncio
class TestSearchPolicyWithTopicFilter:
    """topic_filter 가 주어지면 FieldCondition(policy_topic) 을 must 에 포함한다."""

    async def test_filter_has_policy_topic_field_condition(self):
        """topic_filter='grade_benefit' → Filter.must 에 FieldCondition(policy_topic) 존재."""
        from qdrant_client.models import FieldCondition, Filter

        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(
            return_value=_make_qdrant_response([])
        )
        mock_vec = np.zeros(4096, dtype=np.float32)

        with (
            patch(
                "monglepick.chains.support_policy_rag_chain.embed_query_async",
                new=AsyncMock(return_value=mock_vec),
            ),
            patch(
                "monglepick.chains.support_policy_rag_chain.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            await search_policy("등급 혜택", topic_filter="grade_benefit")

        call_kwargs = mock_client.query_points.call_args.kwargs
        qfilter = call_kwargs["query_filter"]

        # Filter 인스턴스여야 한다
        assert isinstance(qfilter, Filter)
        # must 리스트에 FieldCondition 이 있어야 한다
        assert qfilter.must is not None
        assert len(qfilter.must) == 1
        condition = qfilter.must[0]
        assert isinstance(condition, FieldCondition)
        assert condition.key == "policy_topic"

    async def test_filter_match_value_equals_topic(self):
        """Filter.must[0].match.value 가 전달된 topic_filter 값과 일치한다."""
        from qdrant_client.models import MatchValue

        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(
            return_value=_make_qdrant_response([])
        )
        mock_vec = np.zeros(4096, dtype=np.float32)

        with (
            patch(
                "monglepick.chains.support_policy_rag_chain.embed_query_async",
                new=AsyncMock(return_value=mock_vec),
            ),
            patch(
                "monglepick.chains.support_policy_rag_chain.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            await search_policy("AI 구독", topic_filter="subscription")

        call_kwargs = mock_client.query_points.call_args.kwargs
        condition = call_kwargs["query_filter"].must[0]
        assert isinstance(condition.match, MatchValue)
        assert condition.match.value == "subscription"

    @pytest.mark.parametrize("topic", [
        "grade_benefit", "ai_quota", "subscription",
        "refund", "reward", "payment", "general",
    ])
    async def test_all_valid_topics_pass_as_filter(self, topic: str):
        """7개 유효 topic 모두 필터로 정상 전달된다."""
        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(
            return_value=_make_qdrant_response([])
        )
        mock_vec = np.zeros(4096, dtype=np.float32)

        with (
            patch(
                "monglepick.chains.support_policy_rag_chain.embed_query_async",
                new=AsyncMock(return_value=mock_vec),
            ),
            patch(
                "monglepick.chains.support_policy_rag_chain.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            result = await search_policy(f"{topic} 질문", topic_filter=topic)

        # 에러 없이 리스트 반환 (빈 것도 OK)
        assert isinstance(result, list)


# ============================================================
# 4. Qdrant 예외 시 빈 리스트 반환
# ============================================================

@pytest.mark.asyncio
class TestSearchPolicyQdrantError:
    """Qdrant 연결 실패·API 오류 시 예외를 전파하지 않고 빈 리스트를 반환한다."""

    async def test_qdrant_exception_returns_empty_list(self):
        """query_points 에서 예외 발생 시 빈 리스트 반환."""
        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(
            side_effect=Exception("Qdrant connection refused")
        )
        mock_vec = np.zeros(4096, dtype=np.float32)

        with (
            patch(
                "monglepick.chains.support_policy_rag_chain.embed_query_async",
                new=AsyncMock(return_value=mock_vec),
            ),
            patch(
                "monglepick.chains.support_policy_rag_chain.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            result = await search_policy("환불 정책")

        assert result == []

    async def test_get_qdrant_exception_returns_empty_list(self):
        """get_qdrant() 자체가 실패해도 빈 리스트 반환."""
        mock_vec = np.zeros(4096, dtype=np.float32)

        with (
            patch(
                "monglepick.chains.support_policy_rag_chain.embed_query_async",
                new=AsyncMock(return_value=mock_vec),
            ),
            patch(
                "monglepick.chains.support_policy_rag_chain.get_qdrant",
                new=AsyncMock(side_effect=RuntimeError("Qdrant init failed")),
            ),
        ):
            result = await search_policy("포인트 정책")

        assert result == []


# ============================================================
# 5. 임베딩 타임아웃 시 빈 리스트 반환
# ============================================================

@pytest.mark.asyncio
class TestSearchPolicyEmbedTimeout:
    """Upstage 임베딩 API 타임아웃 시 빈 리스트를 반환한다 (에러 전파 X)."""

    async def test_embed_timeout_returns_empty_list(self):
        """embed_query_async 가 TimeoutError 를 발생시키면 빈 리스트 반환."""

        async def _slow_embed(query: str) -> np.ndarray:
            """30초보다 긴 지연을 시뮬레이션한다."""
            raise asyncio.TimeoutError()

        with (
            patch(
                "monglepick.chains.support_policy_rag_chain.embed_query_async",
                side_effect=_slow_embed,
            ),
        ):
            result = await search_policy("출석 포인트")

        assert result == []

    async def test_embed_general_exception_returns_empty_list(self):
        """embed_query_async 가 일반 예외를 던져도 빈 리스트 반환."""
        with (
            patch(
                "monglepick.chains.support_policy_rag_chain.embed_query_async",
                new=AsyncMock(side_effect=ConnectionError("Upstage API unreachable")),
            ),
        ):
            result = await search_policy("구독 플랜")

        assert result == []


# ============================================================
# 6. lookup_policy handler — PolicyChunk → dict 변환 정확성
# ============================================================

@pytest.mark.asyncio
class TestLookupPolicyHandler:
    """_handle_lookup_policy 가 PolicyChunk 를 올바른 dict 스키마로 변환한다."""

    def _make_ctx(self, is_guest: bool = False) -> ToolContext:
        """테스트용 ToolContext 를 생성한다."""
        return ToolContext(
            user_id="user_001" if not is_guest else "",
            is_guest=is_guest,
            session_id="sess_abc",
            request_id="req_xyz",
        )

    def _make_chunks(self, n: int = 2) -> list[PolicyChunk]:
        """테스트용 PolicyChunk 리스트를 생성한다."""
        return [
            PolicyChunk(
                doc_id=f"doc_{i}",
                doc_path=f"docs/doc_{i}.md",
                section=f"§{i+1} 섹션",
                headings=[f"## 섹션 {i+1}"],
                policy_topic="ai_quota",
                text=f"청크 {i} 내용",
                score=0.9 - i * 0.1,
            )
            for i in range(n)
        ]

    async def test_ok_true_on_success(self):
        """정상 동작 시 ok=True 가 반환된다."""
        chunks = self._make_chunks(2)

        with patch(
            "monglepick.tools.support_tools.policy.search_policy",
            new=AsyncMock(return_value=chunks),
        ):
            result = await _handle_lookup_policy(
                ctx=self._make_ctx(), query="AI 쿼터"
            )

        assert result["ok"] is True

    async def test_chunks_list_length(self):
        """반환 dict 의 data.chunks 길이가 search_policy 결과와 일치한다."""
        chunks = self._make_chunks(3)

        with patch(
            "monglepick.tools.support_tools.policy.search_policy",
            new=AsyncMock(return_value=chunks),
        ):
            result = await _handle_lookup_policy(
                ctx=self._make_ctx(), query="등급 혜택"
            )

        assert len(result["data"]["chunks"]) == 3

    async def test_chunk_dict_schema(self):
        """청크 dict 가 narrator 가 기대하는 6개 필드를 포함한다."""
        chunks = self._make_chunks(1)

        with patch(
            "monglepick.tools.support_tools.policy.search_policy",
            new=AsyncMock(return_value=chunks),
        ):
            result = await _handle_lookup_policy(
                ctx=self._make_ctx(), query="환불 정책"
            )

        chunk_dict = result["data"]["chunks"][0]
        # 필수 6개 필드 존재 확인
        for field in ("doc_id", "section", "headings", "policy_topic", "text", "score"):
            assert field in chunk_dict, f"필드 '{field}' 누락"

    async def test_score_rounded_to_3_decimal(self):
        """score 가 소수점 3자리로 반올림된다."""
        chunk = PolicyChunk(
            doc_id="d",
            doc_path="d.md",
            section="§1",
            headings=[],
            policy_topic="general",
            text="내용",
            score=0.876543,  # 반올림 전
        )

        with patch(
            "monglepick.tools.support_tools.policy.search_policy",
            new=AsyncMock(return_value=[chunk]),
        ):
            result = await _handle_lookup_policy(
                ctx=self._make_ctx(), query="테스트"
            )

        assert result["data"]["chunks"][0]["score"] == 0.877  # round(0.876543, 3)

    async def test_field_values_match_source_chunk(self):
        """반환 dict 의 각 필드 값이 원본 PolicyChunk 값과 일치한다."""
        chunk = PolicyChunk(
            doc_id="리워드_결제_설계서",
            doc_path="docs/리워드_결제_설계서.md",
            section="§4.5 AI 쿼터 정책",
            headings=["## AI 쿼터", "### BRONZE"],
            policy_topic="ai_quota",
            text="BRONZE 등급은 하루 3회 AI 추천을 받을 수 있습니다.",
            score=0.931,
        )

        with patch(
            "monglepick.tools.support_tools.policy.search_policy",
            new=AsyncMock(return_value=[chunk]),
        ):
            result = await _handle_lookup_policy(
                ctx=self._make_ctx(), query="BRONZE AI 횟수"
            )

        d = result["data"]["chunks"][0]
        assert d["doc_id"] == "리워드_결제_설계서"
        assert d["section"] == "§4.5 AI 쿼터 정책"
        assert d["headings"] == ["## AI 쿼터", "### BRONZE"]
        assert d["policy_topic"] == "ai_quota"
        assert d["text"] == "BRONZE 등급은 하루 3회 AI 추천을 받을 수 있습니다."
        assert d["score"] == 0.931

    async def test_search_policy_receives_correct_args(self):
        """handler 가 search_policy 에 올바른 인자를 전달한다."""
        with patch(
            "monglepick.tools.support_tools.policy.search_policy",
            new=AsyncMock(return_value=[]),
        ) as mock_search:
            await _handle_lookup_policy(
                ctx=self._make_ctx(),
                query="구독 해지",
                topic="subscription",
            )

        mock_search.assert_awaited_once_with(
            query="구독 해지",
            top_k=5,
            topic_filter="subscription",
            request_id="req_xyz",
        )

    async def test_ok_false_on_exception(self):
        """search_policy 가 예외를 던지면 ok=False 와 error 메시지를 반환한다."""
        with patch(
            "monglepick.tools.support_tools.policy.search_policy",
            new=AsyncMock(side_effect=RuntimeError("예상치 못한 오류")),
        ):
            result = await _handle_lookup_policy(
                ctx=self._make_ctx(), query="포인트 환불"
            )

        assert result["ok"] is False
        assert "error" in result
        assert "예상치 못한 오류" in result["error"]

    async def test_empty_chunks_returns_ok_true(self):
        """검색 결과가 0건이어도 ok=True, chunks=[] 반환한다."""
        with patch(
            "monglepick.tools.support_tools.policy.search_policy",
            new=AsyncMock(return_value=[]),
        ):
            result = await _handle_lookup_policy(
                ctx=self._make_ctx(), query="없는 내용"
            )

        assert result["ok"] is True
        assert result["data"]["chunks"] == []


# ============================================================
# 7. requires_login=False — 게스트도 정상 응답
# ============================================================

@pytest.mark.asyncio
class TestLookupPolicyGuestAccess:
    """requires_login=False 이므로 게스트 ToolContext 에서도 정상 동작한다."""

    async def test_guest_context_returns_ok_true(self):
        """is_guest=True 인 ToolContext 에서도 ok=True 가 반환된다."""
        guest_ctx = ToolContext(
            user_id="",
            is_guest=True,
            session_id="guest_sess",
            request_id="guest_req",
        )
        chunk = PolicyChunk(
            doc_id="d",
            doc_path="d.md",
            section="§1",
            headings=[],
            policy_topic="general",
            text="공개 정책 내용",
            score=0.85,
        )

        with patch(
            "monglepick.tools.support_tools.policy.search_policy",
            new=AsyncMock(return_value=[chunk]),
        ):
            result = await _handle_lookup_policy(
                ctx=guest_ctx, query="이용 안내"
            )

        assert result["ok"] is True
        assert len(result["data"]["chunks"]) == 1

    async def test_requires_login_is_false(self):
        """SupportToolSpec.requires_login 이 False 임을 직접 확인한다."""
        spec = SUPPORT_TOOL_REGISTRY["lookup_policy"]
        assert spec.requires_login is False


# ============================================================
# 8. 레지스트리 등록 확인
# ============================================================

class TestSupportToolRegistry:
    """SUPPORT_TOOL_REGISTRY 에 lookup_policy 가 올바르게 등록된다."""

    def test_lookup_policy_registered(self):
        """lookup_policy 가 레지스트리에 존재한다."""
        assert "lookup_policy" in SUPPORT_TOOL_REGISTRY

    def test_spec_name_matches_key(self):
        """SupportToolSpec.name 이 레지스트리 키와 동일하다."""
        spec = SUPPORT_TOOL_REGISTRY["lookup_policy"]
        assert spec.name == "lookup_policy"

    def test_spec_args_schema_is_lookup_policy_args(self):
        """args_schema 가 LookupPolicyArgs 클래스이다."""
        spec = SUPPORT_TOOL_REGISTRY["lookup_policy"]
        assert spec.args_schema is LookupPolicyArgs

    def test_spec_handler_is_callable(self):
        """handler 가 callable 이다."""
        spec = SUPPORT_TOOL_REGISTRY["lookup_policy"]
        assert callable(spec.handler)

    def test_spec_description_is_nonempty(self):
        """description 이 비어있지 않다."""
        spec = SUPPORT_TOOL_REGISTRY["lookup_policy"]
        assert spec.description.strip()

    def test_lookup_policy_args_query_required(self):
        """LookupPolicyArgs 에서 query 는 필수 필드이다."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LookupPolicyArgs()  # query 없이 생성 → 검증 오류

    def test_lookup_policy_args_topic_optional(self):
        """LookupPolicyArgs 에서 topic 은 Optional (기본 None) 이다."""
        args = LookupPolicyArgs(query="테스트")
        assert args.topic is None

    def test_lookup_policy_args_topic_accepts_all_valid_topics(self):
        """7개 유효 topic 값 모두 ValidationError 없이 생성된다."""
        valid_topics = [
            "grade_benefit", "ai_quota", "subscription",
            "refund", "reward", "payment", "general",
        ]
        for topic in valid_topics:
            args = LookupPolicyArgs(query="질문", topic=topic)
            assert args.topic == topic
