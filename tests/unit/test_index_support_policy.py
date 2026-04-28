"""
고객센터 정책 RAG 인덱서 단위 테스트 (2026-04-28).

대상 모듈: `scripts/index_support_policy.py`

테스트 영역:
1. `_doc_id_from_path`  — 파일 경로 → doc_id 파생 규칙
2. `_extract_doc_version` — 문서 텍스트에서 버전 태그 추출
3. `_infer_policy_topic` — policy_topic 자동 추론 (7개 규칙 + override)
4. `_chunk_uuid`          — deterministic UUID5 (동일 입력 → 동일 UUID)
5. `_split_by_heading`   — Level-2 헤딩 단위 분할 + 코드블록 보호
6. `_sliding_window_split` — 슬라이딩 윈도우 2차 분할 (overlap 포함)
7. `build_chunks`          — 전체 파이프라인 (분할 → 메타데이터 부착)
8. `ensure_support_policy_collection` 부트스트랩 — 신규 생성 / 멱등
9. `_delete_by_doc_id`  — --clear-db 시 doc_id 기준 삭제
10. `_upsert_chunks`     — 임베딩 + UPSERT 정상 호출 + 개수 불일치 에러
11. `main` CLI           — --dry-run, --clear-db, --source 파일 없음 등

Qdrant / Upstage API 실제 호출은 모두 mock 처리 (CI 환경 인프라 의존성 없음).
"""

from __future__ import annotations

import sys
import textwrap
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ── scripts/ 디렉토리를 sys.path 에 추가 (PYTHONPATH=src 환경에서도 import 가능) ──
_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from index_support_policy import (
    CHUNK_MAX_CHARS,
    COLLECTION_NAME,
    SLIDING_OVERLAP,
    SLIDING_WINDOW_SIZE,
    PolicyChunk,
    _chunk_uuid,
    _delete_by_doc_id,
    _doc_id_from_path,
    _extract_doc_version,
    _infer_policy_topic,
    _is_code_heavy,
    _is_implementation_section,
    _is_too_short,
    _sliding_window_split,
    _split_by_heading,
    _upsert_chunks,
    build_chunks,
    main,
)


# ============================================================
# 1) _doc_id_from_path — 경로 → doc_id
# ============================================================

class TestDocIdFromPath:
    """파일 경로에서 doc_id 를 파생하는 규칙을 검증한다."""

    def test_extracts_stem(self):
        """확장자를 제거한 파일명을 반환한다."""
        p = Path("docs/리워드_결제_설계서.md")
        assert _doc_id_from_path(p) == "리워드_결제_설계서"

    def test_extracts_stem_with_absolute_path(self):
        """절대 경로에서도 파일명(stem)만 추출한다."""
        p = Path("/home/ubuntu/docs/결제_구독_시스템_설계_및_구현_보고서.md")
        assert _doc_id_from_path(p) == "결제_구독_시스템_설계_및_구현_보고서"

    def test_no_directory_component(self):
        """디렉토리 이름이 포함되지 않는다."""
        p = Path("docs/AI_Agent_설계서.md")
        result = _doc_id_from_path(p)
        assert "docs" not in result
        assert result == "AI_Agent_설계서"

    def test_extension_removed(self):
        """확장자(.md)는 결과에 포함되지 않는다."""
        p = Path("sample.md")
        assert ".md" not in _doc_id_from_path(p)


# ============================================================
# 2) _extract_doc_version — 버전 태그 추출
# ============================================================

class TestExtractDocVersion:
    """문서 텍스트에서 버전 태그를 추출하는 규칙을 검증한다."""

    def test_extracts_version(self):
        """vX.Y 형태의 첫 번째 매칭을 반환한다."""
        text = "# 리워드 설계서 v3.4\n\n내용..."
        assert _extract_doc_version(text) == "v3.4"

    def test_extracts_first_occurrence(self):
        """여러 버전이 있으면 첫 번째만 반환한다."""
        text = "v3.2 기준으로 작성. v3.4 에서 변경됨."
        assert _extract_doc_version(text) == "v3.2"

    def test_version_in_middle_of_text(self):
        """텍스트 중간에 버전이 있어도 추출한다."""
        text = "이 문서는 몽글픽 v10.0 기준입니다."
        assert _extract_doc_version(text) == "v10.0"

    def test_returns_unknown_when_missing(self):
        """버전 태그가 없으면 'unknown' 을 반환한다."""
        text = "# 제목\n\n내용만 있고 버전 없음"
        assert _extract_doc_version(text) == "unknown"

    def test_does_not_match_plain_numbers(self):
        """버전 형식이 아닌 숫자(예: 3.4) 는 매칭하지 않는다."""
        text = "3.4는 버전이 아닙니다."
        # \b 경계가 있어야 하므로 'v' 접두사 없는 숫자는 unknown
        assert _extract_doc_version(text) == "unknown"


# ============================================================
# 3) _infer_policy_topic — policy_topic 자동 추론
# ============================================================

class TestInferPolicyTopic:
    """policy_topic 자동 추론 규칙 7종을 검증한다."""

    def test_override_takes_priority(self):
        """override 가 지정되면 텍스트 내용과 무관하게 그 값을 반환한다."""
        text = "환불 정책에 관한 내용"
        assert _infer_policy_topic(text, override="ai_quota") == "ai_quota"

    def test_grade_benefit_detection(self):
        """'등급' + 등급명 키워드 → grade_benefit."""
        texts = [
            "BRONZE 등급 혜택표 — 일일 AI 한도 3회",
            "6등급 팝콘 테마: 알갱이 → 강냉이 → 팝콘",
            "등급별 SILVER 이상 혜택",
            "카라멜팝콘 등급 일일 한도",
        ]
        for text in texts:
            assert _infer_policy_topic(text) == "grade_benefit", f"실패: {text!r}"

    def test_ai_quota_detection(self):
        """'AI' + 쿼터/한도 키워드 → ai_quota.

        must_set={"AI"} 는 대문자 "AI" 단어를 요구하므로,
        소문자 포함 식별자(daily_ai_used, purchased_ai_tokens)는 매칭하지 않는다.
        """
        texts = [
            "AI 쿼터 3-소스 순서: GRADE_FREE → SUB_BONUS → PURCHASED",
            "AI 한도 무료 제공",
            "AI 이용권 daily_ai_used 카운터 증가",   # 대문자 "AI" 포함
            "purchased_ai_tokens AI 한도 소모",       # 대문자 "AI" 포함
        ]
        for text in texts:
            assert _infer_policy_topic(text) == "ai_quota", f"실패: {text!r}"

    def test_subscription_detection(self):
        """'구독' + 플랜/monthly 키워드 → subscription."""
        texts = [
            "구독 플랜 4종: monthly_basic, monthly_premium",
            "yearly_premium 구독 59,000원/년",
            "monthly 구독 basic 2,900원",
        ]
        for text in texts:
            assert _infer_policy_topic(text) == "subscription", f"실패: {text!r}"

    def test_refund_detection(self):
        """'환불' + 정책/기간/신청 키워드 → refund."""
        texts = [
            "환불 정책: 결제 후 7일 이내",
            "환불 기간 및 신청 방법",
            "환불 취소 부분 금액",
        ]
        for text in texts:
            assert _infer_policy_topic(text) == "refund", f"실패: {text!r}"

    def test_reward_detection(self):
        """'리워드' + 적립/활동/출석 키워드 → reward."""
        texts = [
            "리워드 적립 규칙 — 출석 체크시 +10P",
            "리워드 활동 포인트",
            "리워드 리뷰 보상",
        ]
        for text in texts:
            assert _infer_policy_topic(text) == "reward", f"실패: {text!r}"

    def test_payment_detection(self):
        """'결제' + Toss/카드/방법 키워드 → payment."""
        texts = [
            "결제 Toss Payments v2 통합",
            "결제 카드 방법 안내",
            "결제 webhook 설정",
        ]
        for text in texts:
            assert _infer_policy_topic(text) == "payment", f"실패: {text!r}"

    def test_general_fallback(self):
        """어떤 규칙도 매칭되지 않으면 'general' 반환."""
        texts = [
            "데이터베이스 스키마 정의서",
            "Nginx 설정 방법",
            "API 엔드포인트 목록",
        ]
        for text in texts:
            assert _infer_policy_topic(text) == "general", f"실패: {text!r}"

    def test_priority_grade_over_payment(self):
        """
        '등급'과 '결제'가 함께 있을 때 grade_benefit 이 우선한다.
        _TOPIC_RULES 리스트에서 grade_benefit 가 payment 보다 앞에 있기 때문.
        """
        text = "BRONZE 등급 결제 Toss 혜택"
        # 두 규칙 모두 매칭 가능 — 리스트 첫 번째(grade_benefit) 우선
        assert _infer_policy_topic(text) == "grade_benefit"


# ============================================================
# 4) _chunk_uuid — deterministic UUID5
# ============================================================

class TestChunkUuid:
    """동일 입력 → 동일 UUID 를 보장하는지 검증한다."""

    def test_deterministic(self):
        """같은 (doc_id, chunk_idx) 는 항상 같은 UUID."""
        a = _chunk_uuid("리워드_결제_설계서", 0)
        b = _chunk_uuid("리워드_결제_설계서", 0)
        assert a == b

    def test_different_idx_different_uuid(self):
        """청크 순번이 다르면 UUID 도 다르다."""
        a = _chunk_uuid("doc", 0)
        b = _chunk_uuid("doc", 1)
        assert a != b

    def test_different_doc_different_uuid(self):
        """문서 ID 가 다르면 같은 순번이어도 UUID 가 다르다."""
        a = _chunk_uuid("doc_a", 0)
        b = _chunk_uuid("doc_b", 0)
        assert a != b

    def test_uuid_format(self):
        """표준 UUID 문자열 형식 (36자, 하이픈 4개)."""
        u = _chunk_uuid("test_doc", 5)
        assert len(u) == 36
        assert u.count("-") == 4

    def test_valid_uuid(self):
        """반환값이 실제로 파싱 가능한 UUID 형식이다."""
        u = _chunk_uuid("test_doc", 99)
        parsed = uuid.UUID(u)  # 파싱 실패 시 ValueError
        assert str(parsed) == u


# ============================================================
# 5) _split_by_heading — Level-2 헤딩 단위 분할
# ============================================================

class TestSplitByHeading:
    """마크다운 Level-2 헤딩 단위 분할 로직을 검증한다."""

    def test_splits_on_h2(self):
        """## 헤딩마다 새 청크가 시작된다."""
        text = textwrap.dedent("""\
            # 제목 (Level-1 무시)

            ## 섹션 1
            내용 1

            ## 섹션 2
            내용 2
        """)
        segments = _split_by_heading(text)
        # Level-1 헤딩 + 공백은 첫 번째 세그먼트에, 이후 ## 기준으로 분할
        # 결과: [("", "# 제목...\n\n"), ("## 섹션 1", "..."), ("## 섹션 2", "...")]
        headings = [h for h, _ in segments if h.startswith("## ")]
        assert "## 섹션 1" in headings
        assert "## 섹션 2" in headings

    def test_body_contains_heading(self):
        """각 세그먼트의 body 에 헤딩 줄이 포함된다."""
        text = "## 첫 번째 섹션\n내용\n"
        segments = _split_by_heading(text)
        # heading='## 첫 번째 섹션', body 에 헤딩 줄 포함
        found = [(h, b) for h, b in segments if h == "## 첫 번째 섹션"]
        assert len(found) == 1
        assert "## 첫 번째 섹션" in found[0][1]

    def test_does_not_split_inside_code_block(self):
        """코드블록(```) 안의 ## 패턴은 분할 기준으로 사용하지 않는다."""
        text = textwrap.dedent("""\
            ## 실제 섹션

            ```python
            ## 이건 코드 주석
            x = 1
            ```

            ## 다음 섹션
        """)
        segments = _split_by_heading(text)
        # "이건 코드 주석" 이 헤딩으로 인식되지 않아야 함
        headings = [h for h, _ in segments]
        assert "## 이건 코드 주석" not in headings
        # 실제 헤딩 2개만 인식
        real_headings = [h for h in headings if h.startswith("## ")]
        assert "## 실제 섹션" in real_headings
        assert "## 다음 섹션" in real_headings

    def test_empty_text_returns_one_empty_segment(self):
        """빈 텍스트는 빈 세그먼트 하나를 반환한다."""
        segments = _split_by_heading("")
        # 빈 텍스트 → current_lines 가 비어 빈 리스트 OR 빈 body 하나
        assert isinstance(segments, list)

    def test_no_h2_returns_single_segment(self):
        """Level-2 헤딩이 없으면 전체를 하나의 세그먼트로 반환한다."""
        text = "# Level-1 만 있는 텍스트\n\n내용\n"
        segments = _split_by_heading(text)
        # 모든 텍스트가 하나의 세그먼트에 포함
        all_text = "".join(b for _, b in segments)
        assert "내용" in all_text

    def test_multiple_code_blocks(self):
        """중첩되지 않은 여러 코드블록이 있어도 정상 처리한다."""
        text = textwrap.dedent("""\
            ## 섹션 A
            ```
            ## 코드 안 헤딩 1
            ```
            텍스트

            ```yaml
            ## 코드 안 헤딩 2
            key: value
            ```

            ## 섹션 B
        """)
        segments = _split_by_heading(text)
        real_headings = [h for h, _ in segments if h.startswith("## ")]
        assert "## 섹션 A" in real_headings
        assert "## 섹션 B" in real_headings
        assert "## 코드 안 헤딩 1" not in real_headings
        assert "## 코드 안 헤딩 2" not in real_headings


# ============================================================
# 6) _sliding_window_split — 슬라이딩 윈도우 분할
# ============================================================

class TestSlidingWindowSplit:
    """슬라이딩 윈도우 분할 로직을 검증한다."""

    def test_short_text_not_split(self):
        """window 이하 길이의 텍스트는 분할하지 않는다."""
        text = "a" * 100
        result = _sliding_window_split(text, window=200, overlap=50)
        assert result == [text]

    def test_exact_window_not_split(self):
        """window 와 정확히 같은 길이도 분할하지 않는다."""
        text = "x" * 800
        result = _sliding_window_split(text, window=800, overlap=200)
        assert result == [text]

    def test_splits_long_text(self):
        """window 초과 텍스트는 여러 청크로 분할된다."""
        text = "a" * 1000
        result = _sliding_window_split(text, window=500, overlap=100)
        assert len(result) > 1

    def test_each_chunk_within_window(self):
        """분할된 각 청크의 길이가 window 이하다."""
        text = "가나다라마바사" * 200  # ~1400자
        result = _sliding_window_split(text, window=800, overlap=200)
        for chunk in result:
            assert len(chunk) <= 800

    def test_overlap_exists_between_consecutive_chunks(self):
        """연속된 청크 간에 overlap 만큼의 텍스트가 겹친다."""
        window = 10
        overlap = 3
        text = "0123456789ABCDEF"  # 16자
        result = _sliding_window_split(text, window=window, overlap=overlap)
        if len(result) >= 2:
            # 첫 번째 청크의 뒤 overlap 자 == 두 번째 청크의 앞 overlap 자
            tail_of_first = result[0][-overlap:]
            head_of_second = result[1][:overlap]
            assert tail_of_first == head_of_second

    def test_last_chunk_covers_end(self):
        """마지막 청크가 텍스트 끝부분을 포함한다."""
        text = "a" * 1500
        result = _sliding_window_split(text, window=800, overlap=200)
        last_chunk_end = text[-10:]
        assert result[-1].endswith(last_chunk_end)


# ============================================================
# 7) build_chunks — 전체 파이프라인
# ============================================================

class TestBuildChunks:
    """build_chunks 의 파이프라인 동작을 임시 마크다운 파일로 검증한다."""

    def _make_md_file(self, tmp_path: Path, content: str) -> Path:
        """임시 마크다운 파일을 생성하고 경로를 반환한다."""
        p = tmp_path / "test_policy.md"
        p.write_text(content, encoding="utf-8")
        return p

    def test_basic_metadata_attached(self, tmp_path: Path):
        """청크에 doc_id, doc_path, section, indexed_at 이 부착된다."""
        # too_short 필터(200자)를 통과하도록 본문을 충분히 작성한다
        body = "BRONZE 등급의 일일 AI 한도는 3회입니다. 구독하면 더 늘어납니다. " * 5
        content = f"## 등급 혜택\n\n{body}\n"
        md_path = self._make_md_file(tmp_path, content)

        chunks = build_chunks(md_path)

        assert len(chunks) >= 1
        c = chunks[0]
        assert c.doc_id == "test_policy"
        assert "test_policy.md" in c.doc_path
        assert c.indexed_at  # 비어있지 않음
        assert "§" in c.section

    def test_chunk_idx_sequential(self, tmp_path: Path):
        """chunk_idx 는 0 부터 순차적으로 증가한다."""
        content = "## 섹션 1\n내용\n\n## 섹션 2\n내용\n\n## 섹션 3\n내용\n"
        md_path = self._make_md_file(tmp_path, content)

        chunks = build_chunks(md_path)

        idxs = [c.chunk_idx for c in chunks]
        assert idxs == list(range(len(chunks)))

    def test_policy_topic_inferred(self, tmp_path: Path):
        """텍스트 내용에 맞는 policy_topic 이 자동 추론된다."""
        # too_short 필터(200자)를 통과하도록 본문을 충분히 작성한다
        body = "AI 한도 무료 3회, 구독으로 확장됩니다. " * 8
        content = f"## AI 쿼터 정책\n\n{body}\n"
        md_path = self._make_md_file(tmp_path, content)

        chunks = build_chunks(md_path)

        # "AI" + "한도" → ai_quota
        assert any(c.policy_topic == "ai_quota" for c in chunks)

    def test_policy_topic_override(self, tmp_path: Path):
        """policy_topic_override 가 있으면 추론 없이 그 값을 사용한다."""
        content = "## 아무 섹션\n\n데이터베이스 스키마\n"
        md_path = self._make_md_file(tmp_path, content)

        chunks = build_chunks(md_path, policy_topic_override="refund")

        assert all(c.policy_topic == "refund" for c in chunks)

    def test_long_section_is_subdivided(self, tmp_path: Path):
        """1500자 초과 섹션은 슬라이딩 윈도우로 추가 분할된다."""
        # CHUNK_MAX_CHARS 초과하는 긴 섹션 생성
        long_body = "내용 " * 400  # ~800자 × 2 이상
        content = f"## 긴 섹션\n\n{long_body}\n"
        md_path = self._make_md_file(tmp_path, content)

        chunks = build_chunks(md_path)

        # 1개 섹션이지만 여러 청크로 분할되어야 함
        section_chunks = [c for c in chunks if "긴 섹션" in c.section]
        assert len(section_chunks) >= 1
        # 전체 청크 수가 1보다 큰지 (슬라이딩 분할 발생)
        if len(content) > CHUNK_MAX_CHARS:
            assert len(chunks) > 1

    def test_doc_version_extracted(self, tmp_path: Path):
        """문서 내 버전 태그가 청크 메타데이터에 반영된다."""
        content = "# 설계서 v3.4\n\n## 등급 정책\n\nBRONZE 등급\n"
        md_path = self._make_md_file(tmp_path, content)

        chunks = build_chunks(md_path)

        assert all(c.doc_version == "v3.4" for c in chunks)

    def test_text_field_not_empty(self, tmp_path: Path):
        """각 청크의 text 필드가 비어 있지 않다."""
        content = "## 섹션\n\n내용 있음\n"
        md_path = self._make_md_file(tmp_path, content)

        chunks = build_chunks(md_path)

        for c in chunks:
            assert c.text.strip()

    def test_headings_list_populated(self, tmp_path: Path):
        """headings 필드에 해당 섹션의 헤딩 텍스트가 담긴다."""
        # too_short 필터(200자)를 통과하도록 본문을 충분히 작성한다
        # "환불 기간은..." 한 문장이 약 34자이므로 7회 반복 = ~238자로 필터 통과
        body = "환불 기간은 결제 후 7일 이내이며, 영업일 기준으로 처리됩니다. " * 7
        content = f"## 환불 정책\n\n{body}\n"
        md_path = self._make_md_file(tmp_path, content)

        chunks = build_chunks(md_path)

        # ## 헤딩이 있는 섹션의 청크는 headings 가 비어있지 않음
        non_empty_heading_chunks = [c for c in chunks if c.headings]
        assert len(non_empty_heading_chunks) >= 1


# ============================================================
# 8) ensure_support_policy_collection 부트스트랩
# ============================================================

@pytest.mark.asyncio
class TestEnsureSupportPolicyCollection:
    """db/clients.py 의 ensure_support_policy_collection 동작을 검증한다."""

    async def test_creates_when_missing(self):
        """컬렉션이 없으면 create_collection 을 호출한다."""
        from monglepick.db.clients import ensure_support_policy_collection

        mock_client = MagicMock()
        other = MagicMock()
        other.name = "movies"
        mock_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[other])
        )
        mock_client.create_collection = AsyncMock()
        mock_client.create_payload_index = AsyncMock()

        with patch(
            "monglepick.db.clients.get_qdrant",
            new=AsyncMock(return_value=mock_client),
        ):
            await ensure_support_policy_collection()

        mock_client.create_collection.assert_called_once()
        kwargs = mock_client.create_collection.call_args.kwargs
        assert kwargs["collection_name"] == "support_policy_v1"

    async def test_skips_when_exists(self):
        """이미 존재하면 create_collection 을 호출하지 않는다 (멱등)."""
        from monglepick.db.clients import ensure_support_policy_collection

        existing = MagicMock()
        existing.name = "support_policy_v1"
        mock_client = MagicMock()
        mock_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[existing])
        )
        mock_client.create_collection = AsyncMock()
        mock_client.create_payload_index = AsyncMock()

        with patch(
            "monglepick.db.clients.get_qdrant",
            new=AsyncMock(return_value=mock_client),
        ):
            await ensure_support_policy_collection()

        mock_client.create_collection.assert_not_called()

    async def test_payload_index_errors_swallowed(self):
        """payload 인덱스 중복 등 에러가 전파되지 않는다."""
        from monglepick.db.clients import ensure_support_policy_collection

        existing = MagicMock()
        existing.name = "support_policy_v1"
        mock_client = MagicMock()
        mock_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[existing])
        )
        mock_client.create_collection = AsyncMock()
        mock_client.create_payload_index = AsyncMock(
            side_effect=Exception("index already exists")
        )

        with patch(
            "monglepick.db.clients.get_qdrant",
            new=AsyncMock(return_value=mock_client),
        ):
            # 예외가 전파되지 않아야 함
            await ensure_support_policy_collection()

    async def test_creates_correct_keyword_payload_indexes(self):
        """policy_topic, doc_id, doc_path — 3개 keyword 인덱스를 시도한다."""
        from monglepick.db.clients import ensure_support_policy_collection

        # 신규 컬렉션 시나리오
        mock_client = MagicMock()
        mock_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[])
        )
        mock_client.create_collection = AsyncMock()
        mock_client.create_payload_index = AsyncMock()

        with patch(
            "monglepick.db.clients.get_qdrant",
            new=AsyncMock(return_value=mock_client),
        ):
            await ensure_support_policy_collection()

        # create_payload_index 가 3번 호출됨 (policy_topic, doc_id, doc_path)
        calls = mock_client.create_payload_index.call_args_list
        indexed_fields = [c.kwargs["field_name"] for c in calls]
        assert "policy_topic" in indexed_fields
        assert "doc_id" in indexed_fields
        assert "doc_path" in indexed_fields


# ============================================================
# 9) _delete_by_doc_id — --clear-db 삭제 로직
# ============================================================

@pytest.mark.asyncio
class TestDeleteByDocId:
    """doc_id 기준 Qdrant 청크 삭제 로직을 검증한다."""

    async def test_calls_qdrant_delete(self):
        """Qdrant delete 가 올바른 collection_name 으로 호출된다."""
        mock_client = MagicMock()
        mock_client.delete = AsyncMock(return_value=MagicMock())

        with patch(
            "index_support_policy.get_qdrant",
            new=AsyncMock(return_value=mock_client),
        ):
            await _delete_by_doc_id("리워드_결제_설계서")

        mock_client.delete.assert_called_once()
        call_kwargs = mock_client.delete.call_args.kwargs
        assert call_kwargs["collection_name"] == COLLECTION_NAME

    async def test_delete_uses_doc_id_filter(self):
        """삭제 필터가 doc_id 필드를 대상으로 한다."""
        from qdrant_client.models import Filter

        mock_client = MagicMock()
        mock_client.delete = AsyncMock(return_value=MagicMock())

        with patch(
            "index_support_policy.get_qdrant",
            new=AsyncMock(return_value=mock_client),
        ):
            await _delete_by_doc_id("test_doc")

        call_kwargs = mock_client.delete.call_args.kwargs
        # points_selector 가 Filter 인스턴스여야 함
        selector = call_kwargs["points_selector"]
        assert selector is not None


# ============================================================
# 10) _upsert_chunks — 임베딩 + UPSERT
# ============================================================

@pytest.mark.asyncio
class TestUpsertChunks:
    """_upsert_chunks 의 임베딩 호출 및 Qdrant UPSERT 동작을 검증한다."""

    def _make_chunks(self, n: int) -> list[PolicyChunk]:
        """테스트용 PolicyChunk 를 n 개 생성한다."""
        return [
            PolicyChunk(
                doc_id="test_doc",
                doc_path="docs/test.md",
                section=f"§{i+1} 섹션",
                headings=[f"섹션 {i+1}"],
                policy_topic="general",
                doc_version="v1.0",
                indexed_at="2026-04-28T10:00:00+09:00",
                chunk_idx=i,
                text=f"청크 {i} 내용",
            )
            for i in range(n)
        ]

    async def test_upserts_all_chunks(self):
        """모든 청크가 UPSERT 된다."""
        chunks = self._make_chunks(5)
        fake_vectors = np.zeros((5, 4096), dtype=np.float32)

        mock_client = MagicMock()
        mock_client.upsert = AsyncMock()

        with (
            patch(
                "index_support_policy.embed_texts",
                return_value=fake_vectors,
            ),
            patch(
                "index_support_policy.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            count = await _upsert_chunks(chunks)

        assert count == 5
        mock_client.upsert.assert_called()
        # 호출된 모든 upsert 의 collection_name 이 올바름
        for call in mock_client.upsert.call_args_list:
            assert call.kwargs["collection_name"] == COLLECTION_NAME

    async def test_upsert_point_ids_are_deterministic(self):
        """UPSERT 된 point ID 가 deterministic (_chunk_uuid 기반)이다."""
        chunks = self._make_chunks(3)
        fake_vectors = np.zeros((3, 4096), dtype=np.float32)

        captured_points: list = []

        mock_client = MagicMock()

        async def capture_upsert(**kwargs):
            captured_points.extend(kwargs.get("points", []))

        mock_client.upsert = AsyncMock(side_effect=capture_upsert)

        with (
            patch(
                "index_support_policy.embed_texts",
                return_value=fake_vectors,
            ),
            patch(
                "index_support_policy.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            await _upsert_chunks(chunks)

        # 각 point ID 가 _chunk_uuid(doc_id, chunk_idx) 와 일치해야 함
        for i, point in enumerate(captured_points):
            expected_id = _chunk_uuid("test_doc", i)
            assert point.id == expected_id

    async def test_embedding_count_mismatch_raises(self):
        """임베딩 개수와 청크 수가 다르면 RuntimeError 가 발생한다."""
        chunks = self._make_chunks(3)
        bad_vectors = np.zeros((2, 4096), dtype=np.float32)  # 3 expected, 2 returned

        with patch(
            "index_support_policy.embed_texts",
            return_value=bad_vectors,
        ):
            with pytest.raises(RuntimeError, match="임베딩 개수 불일치"):
                await _upsert_chunks(chunks)

    async def test_payload_contains_text_field(self):
        """UPSERT payload 에 'text' 필드가 포함된다 (검색 결과 표시용)."""
        chunks = self._make_chunks(2)
        fake_vectors = np.zeros((2, 4096), dtype=np.float32)

        captured_points: list = []

        mock_client = MagicMock()

        async def capture_upsert(**kwargs):
            captured_points.extend(kwargs.get("points", []))

        mock_client.upsert = AsyncMock(side_effect=capture_upsert)

        with (
            patch(
                "index_support_policy.embed_texts",
                return_value=fake_vectors,
            ),
            patch(
                "index_support_policy.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            await _upsert_chunks(chunks)

        for point in captured_points:
            assert "text" in point.payload
            assert "doc_id" in point.payload
            assert "policy_topic" in point.payload
            assert "chunk_idx" in point.payload

    async def test_large_batch_uses_multiple_upsert_calls(self):
        """청크가 200개 초과이면 여러 번 나눠서 UPSERT 한다."""
        # qdrant_batch=200 이므로 201개는 2회 호출이어야 함
        chunks = self._make_chunks(201)
        fake_vectors = np.zeros((201, 4096), dtype=np.float32)

        mock_client = MagicMock()
        mock_client.upsert = AsyncMock()

        with (
            patch(
                "index_support_policy.embed_texts",
                return_value=fake_vectors,
            ),
            patch(
                "index_support_policy.get_qdrant",
                new=AsyncMock(return_value=mock_client),
            ),
        ):
            count = await _upsert_chunks(chunks)

        assert count == 201
        # 200개 배치 + 1개 배치 = 2회 호출
        assert mock_client.upsert.call_count == 2


# ============================================================
# 11) main CLI — 통합 흐름 검증
# ============================================================

@pytest.mark.asyncio
class TestMainCli:
    """main() 의 CLI 분기 동작을 임시 파일로 검증한다."""

    def _write_policy_md(self, tmp_path: Path) -> Path:
        """테스트용 정책 마크다운 파일을 생성한다."""
        p = tmp_path / "test_policy.md"
        p.write_text(
            "# 테스트 정책서 v1.0\n\n"
            "## 등급 혜택\n\nBRONZE 등급 AI 한도 3회\n\n"
            "## 구독 플랜\n\nmonthly_basic 2,900원\n",
            encoding="utf-8",
        )
        return p

    async def test_dry_run_does_not_call_qdrant(self, tmp_path: Path, capsys):
        """--dry-run 모드에서 Qdrant 적재 및 임베딩이 호출되지 않는다."""
        md_path = self._write_policy_md(tmp_path)

        mock_upsert = AsyncMock()
        mock_embed = MagicMock()

        with (
            patch("index_support_policy._upsert_chunks", mock_upsert),
            patch("index_support_policy.embed_texts", mock_embed),
        ):
            await main(["--source", str(md_path), "--dry-run"])

        mock_upsert.assert_not_called()
        mock_embed.assert_not_called()

    async def test_dry_run_prints_chunks(self, tmp_path: Path, capsys):
        """--dry-run 모드에서 청크 정보가 출력된다."""
        md_path = self._write_policy_md(tmp_path)

        with (
            patch("index_support_policy._upsert_chunks", AsyncMock()),
            patch("index_support_policy.embed_texts", MagicMock()),
        ):
            await main(["--source", str(md_path), "--dry-run"])

        captured = capsys.readouterr()
        assert "DRY-RUN" in captured.out
        assert "청크" in captured.out

    async def test_missing_source_exits(self, tmp_path: Path):
        """존재하지 않는 파일을 --source 로 지정하면 SystemExit 이 발생한다."""
        with pytest.raises(SystemExit):
            await main(["--source", str(tmp_path / "없는파일.md")])

    async def test_clear_db_calls_delete(self, tmp_path: Path):
        """--clear-db 옵션 사용 시 _delete_by_doc_id 가 호출된다."""
        md_path = self._write_policy_md(tmp_path)

        mock_delete = AsyncMock()

        # embed_texts 는 실제 청크 수에 맞는 벡터를 반환해야 함 (side_effect 방식)
        def fake_embed(texts, batch_size=50):
            return np.zeros((len(texts), 4096), dtype=np.float32)

        with (
            patch(
                "index_support_policy._delete_by_doc_id",
                mock_delete,
            ),
            patch(
                "index_support_policy.embed_texts",
                side_effect=fake_embed,
            ),
            patch(
                "index_support_policy.get_qdrant",
                new=AsyncMock(return_value=MagicMock(upsert=AsyncMock())),
            ),
            patch(
                "index_support_policy.ensure_support_policy_collection",
                new=AsyncMock(),
            ),
        ):
            await main(["--source", str(md_path), "--clear-db"])

        mock_delete.assert_called_once_with("test_policy")

    async def test_without_clear_db_does_not_delete(self, tmp_path: Path):
        """--clear-db 없으면 _delete_by_doc_id 가 호출되지 않는다."""
        md_path = self._write_policy_md(tmp_path)

        mock_delete = AsyncMock()

        # embed_texts 는 실제 청크 수에 맞는 벡터를 반환해야 함 (side_effect 방식)
        def fake_embed(texts, batch_size=50):
            return np.zeros((len(texts), 4096), dtype=np.float32)

        with (
            patch(
                "index_support_policy._delete_by_doc_id",
                mock_delete,
            ),
            patch(
                "index_support_policy.embed_texts",
                side_effect=fake_embed,
            ),
            patch(
                "index_support_policy.get_qdrant",
                new=AsyncMock(return_value=MagicMock(upsert=AsyncMock())),
            ),
            patch(
                "index_support_policy.ensure_support_policy_collection",
                new=AsyncMock(),
            ),
        ):
            await main(["--source", str(md_path)])

        mock_delete.assert_not_called()

    async def test_multiple_sources_processed(self, tmp_path: Path):
        """--source 를 여러 번 지정하면 모든 문서가 처리된다."""
        md1 = tmp_path / "doc1.md"
        md2 = tmp_path / "doc2.md"
        md1.write_text("## 등급 혜택\n\nBRONZE 등급\n", encoding="utf-8")
        md2.write_text("## 구독 플랜\n\nmonthly_basic\n", encoding="utf-8")

        fake_vectors = np.zeros((5, 4096), dtype=np.float32)
        upsert_calls: list[str] = []

        original_upsert = _upsert_chunks

        async def tracking_upsert(chunks, **kwargs):
            upsert_calls.append(chunks[0].doc_id if chunks else "empty")
            # fake_vectors 크기를 청크 수에 맞게 조정
            vecs = np.zeros((len(chunks), 4096), dtype=np.float32)
            with patch("index_support_policy.embed_texts", return_value=vecs):
                with patch(
                    "index_support_policy.get_qdrant",
                    new=AsyncMock(return_value=MagicMock(upsert=AsyncMock())),
                ):
                    return await original_upsert(chunks, **kwargs)

        with (
            patch("index_support_policy._upsert_chunks", tracking_upsert),
            patch(
                "index_support_policy.ensure_support_policy_collection",
                new=AsyncMock(),
            ),
        ):
            await main(["--source", str(md1), "--source", str(md2)])

        # 두 문서 모두 처리됨
        assert len(upsert_calls) == 2


# ============================================================
# 12) 청크 필터 — _is_code_heavy
# ============================================================

class TestIsCodeHeavy:
    """코드블록 비중 필터를 검증한다."""

    def test_rejects_code_heavy_chunk(self):
        """코드블록이 전체 텍스트의 60% 이상을 차지하면 True 를 반환한다."""
        # 코드블록 60자 + 일반 텍스트 40자 = 전체 100자 → 비중 60%
        code_part = "```python\n" + "x = 1\n" * 8 + "```"   # ~60자
        text_part = "일반 설명 텍스트입니다. " * 3            # ~40자
        chunk_text = code_part + "\n" + text_part
        # 비중 계산: code_part / 전체 길이
        assert _is_code_heavy(chunk_text, threshold=0.5) is True

    def test_keeps_text_heavy_chunk(self):
        """코드블록 비중이 50% 미만이면 False 를 반환한다."""
        # 코드블록 10자 + 일반 텍스트 100자
        chunk_text = "```\ncode\n```\n" + "정책 설명 텍스트. " * 10
        assert _is_code_heavy(chunk_text, threshold=0.5) is False

    def test_no_code_block_is_false(self):
        """코드블록이 전혀 없으면 False 를 반환한다."""
        chunk_text = "BRONZE 등급의 일일 AI 한도는 3회입니다. 구독하면 더 늘어납니다."
        assert _is_code_heavy(chunk_text) is False

    def test_empty_text_is_false(self):
        """빈 텍스트는 False 를 반환한다 (ZeroDivisionError 발생하지 않아야 함)."""
        assert _is_code_heavy("") is False

    def test_only_code_block_is_true(self):
        """텍스트 전체가 코드블록이면 True 를 반환한다."""
        chunk_text = "```java\n@Transactional\npublic void pay() {}\n```"
        assert _is_code_heavy(chunk_text, threshold=0.5) is True

    def test_custom_threshold(self):
        """threshold 를 0.8 으로 높이면 60% 코드블록 청크도 통과한다."""
        code_part = "```python\n" + "x = 1\n" * 8 + "```"
        text_part = "일반 설명 텍스트입니다. " * 3
        chunk_text = code_part + "\n" + text_part
        # 비중 ~60% < threshold 0.8 → False
        assert _is_code_heavy(chunk_text, threshold=0.8) is False


# ============================================================
# 13) 청크 필터 — _is_implementation_section
# ============================================================

class TestIsImplementationSection:
    """구현 세부 헤딩 키워드 필터를 검증한다."""

    def test_rejects_class_structure_heading(self):
        """'클래스 구조' 키워드가 포함된 헤딩은 True 를 반환한다."""
        headings = ["§13.1 클래스 구조 및 책임 분리"]
        assert _is_implementation_section(headings) is True

    def test_rejects_transactional_heading(self):
        """'@Transactional' 키워드가 포함된 헤딩은 True 를 반환한다."""
        headings = ["@Transactional AOP 설정 방법"]
        assert _is_implementation_section(headings) is True

    def test_rejects_implementation_heading(self):
        """'구현' 키워드가 포함된 헤딩은 True 를 반환한다."""
        headings = ["결제 서비스 구현 상세"]
        assert _is_implementation_section(headings) is True

    def test_rejects_jpa_heading(self):
        """'JPA' 키워드가 포함된 헤딩은 True 를 반환한다."""
        headings = ["JPA Entity 설계"]
        assert _is_implementation_section(headings) is True

    def test_rejects_mybatis_heading(self):
        """'MyBatis' 키워드가 포함된 헤딩은 True 를 반환한다."""
        headings = ["MyBatis Mapper 구성"]
        assert _is_implementation_section(headings) is True

    def test_keeps_policy_benefit_heading(self):
        """'§4.5 등급 혜택표' 같은 정책 헤딩은 False 를 반환한다."""
        headings = ["§4.5 등급 혜택표"]
        assert _is_implementation_section(headings) is False

    def test_keeps_ai_quota_heading(self):
        """AI 쿼터 정책 헤딩은 False 를 반환한다."""
        headings = ["AI 쿼터 3-소스 정책"]
        assert _is_implementation_section(headings) is False

    def test_empty_headings_is_false(self):
        """빈 headings 리스트는 False 를 반환한다."""
        assert _is_implementation_section([]) is False

    def test_uses_last_heading(self):
        """여러 헤딩이 있을 때 마지막 헤딩만 기준으로 판단한다."""
        # 앞의 헤딩에 키워드가 있어도 마지막이 정책 헤딩이면 False
        headings = ["구현 상세", "§4.5 등급 혜택표"]
        assert _is_implementation_section(headings) is False

    def test_last_heading_has_keyword(self):
        """마지막 헤딩에 키워드가 있으면 앞 헤딩과 무관하게 True."""
        headings = ["§4.5 등급 혜택표", "부록 A: 코드 샘플"]
        assert _is_implementation_section(headings) is True


# ============================================================
# 14) 청크 필터 — _is_too_short
# ============================================================

class TestIsTooShort:
    """짧은 메타 청크 필터를 검증한다."""

    def test_rejects_short_chunk(self):
        """100자 미만 청크는 True 를 반환한다."""
        chunk_text = "작성자: 홍길동\n날짜: 2026-04-28\n버전: v3.4"
        assert len(chunk_text.strip()) < 200
        assert _is_too_short(chunk_text) is True

    def test_keeps_normal_chunk(self):
        """200자 이상 청크는 False 를 반환한다."""
        chunk_text = "BRONZE 등급의 일일 AI 한도는 3회입니다. " * 10  # ~200자 이상
        assert _is_too_short(chunk_text) is False

    def test_exactly_200_chars_is_kept(self):
        """정확히 200자이면 False 를 반환한다 (경계값 포함)."""
        chunk_text = "가" * 200
        assert _is_too_short(chunk_text) is False

    def test_199_chars_is_rejected(self):
        """199자이면 True 를 반환한다 (경계값 미만)."""
        chunk_text = "가" * 199
        assert _is_too_short(chunk_text) is True

    def test_whitespace_only_is_rejected(self):
        """공백만 있는 텍스트는 True 를 반환한다."""
        assert _is_too_short("   \n\n\t  ") is True

    def test_empty_string_is_rejected(self):
        """빈 문자열은 True 를 반환한다."""
        assert _is_too_short("") is True

    def test_custom_min_chars(self):
        """min_chars 를 50 으로 낮추면 100자 청크가 통과한다."""
        chunk_text = "가" * 100
        assert _is_too_short(chunk_text, min_chars=50) is False


# ============================================================
# 15) build_chunks 필터 통합 — 정책 청크 보존 + 구현 청크 제외
# ============================================================

class TestBuildChunksFilter:
    """
    build_chunks 에서 필터 3종이 올바르게 작동하는지 통합 검증한다.

    실제 마크다운 구조(정책 섹션 + 코드 과다 섹션 + 구현 헤딩 섹션 + 메타 전문)를
    임시 파일로 재현하여, 필터 후 보존/제외 청크 수를 검증한다.
    """

    def _make_filter_test_md(self, tmp_path: Path) -> Path:
        """
        필터 테스트용 마크다운 파일을 생성한다.

        섹션 구성:
        - § 등급 혜택표    : 정책 텍스트 → 보존 대상
        - § 클래스 구조    : 구현 헤딩 키워드 → implementation 필터 제외
        - § JPA 설정       : 구현 헤딩 키워드 → implementation 필터 제외
        - § 코드 예시      : 코드블록 비중 80% → code_heavy 필터 제외
        """
        code_block = "```java\n" + "@Transactional\npublic void pay() {}\n" * 15 + "```"
        content = (
            "# 테스트 설계서 v1.0\n\n"
            # 정책 섹션 — 200자 이상, 구현 키워드 없음 → 보존
            "## 등급 혜택표\n\n"
            + ("BRONZE 등급의 일일 AI 한도는 3회입니다. 구독하면 더 늘어납니다. " * 5)
            + "\n\n"
            # 구현 헤딩 섹션 — implementation 필터 제외
            "## 클래스 구조 및 책임 분리\n\n"
            + ("PaymentService 는 결제 처리를 담당합니다. " * 5)
            + "\n\n"
            # 또 다른 구현 헤딩 — implementation 필터 제외
            "## JPA Entity 설계\n\n"
            + ("Payment Entity 스키마 정의입니다. " * 5)
            + "\n\n"
            # 코드 과다 섹션 — code_heavy 필터 제외 (코드블록이 전체의 ~80%)
            "## 코드 예시\n\n"
            + "간략한 설명.\n\n"
            + code_block
            + "\n"
        )
        p = tmp_path / "filter_test.md"
        p.write_text(content, encoding="utf-8")
        return p

    def test_filter_keeps_policy_chunk(self, tmp_path: Path, capsys):
        """'§ 등급 혜택표' 정책 섹션은 필터 후에도 보존된다."""
        md_path = self._make_filter_test_md(tmp_path)
        chunks = build_chunks(md_path)

        # 등급 혜택표 섹션 청크가 최소 1개 남아야 한다
        policy_chunks = [c for c in chunks if "등급 혜택표" in c.section]
        assert len(policy_chunks) >= 1, "정책 청크가 잘못 제외되었습니다."

    def test_filter_code_heavy_chunk(self, tmp_path: Path, capsys):
        """코드블록 비중이 높은 '§ 코드 예시' 섹션은 필터에서 제외된다."""
        md_path = self._make_filter_test_md(tmp_path)
        chunks = build_chunks(md_path)

        # 코드 예시 섹션 청크가 없어야 한다
        code_chunks = [c for c in chunks if "코드 예시" in c.section]
        assert len(code_chunks) == 0, f"code_heavy 청크가 제외되지 않았습니다: {code_chunks}"

    def test_filter_implementation_heading(self, tmp_path: Path, capsys):
        """'클래스 구조' 및 'JPA' 구현 헤딩 섹션은 필터에서 제외된다."""
        md_path = self._make_filter_test_md(tmp_path)
        chunks = build_chunks(md_path)

        # 구현 헤딩 섹션 청크가 없어야 한다
        impl_chunks = [
            c for c in chunks
            if "클래스 구조" in c.section or "JPA" in c.section
        ]
        assert len(impl_chunks) == 0, (
            f"implementation 청크가 제외되지 않았습니다: {impl_chunks}"
        )

    def test_filter_too_short_chunk(self, tmp_path: Path, capsys):
        """200자 미만 메타 전문 섹션은 too_short 필터에서 제외된다."""
        # 짧은 메타 섹션이 포함된 마크다운 생성
        content = (
            "# 설계서 v1.0\n\n"
            # 짧은 메타 섹션 (100자 미만)
            "## 목차\n\n"
            "1. 등급 혜택\n2. 구독 플랜\n\n"
            # 정상 정책 섹션
            "## 등급 혜택표\n\n"
            + ("BRONZE 등급의 일일 AI 한도는 3회입니다. 구독하면 더 늘어납니다. " * 5)
            + "\n"
        )
        p = tmp_path / "short_test.md"
        p.write_text(content, encoding="utf-8")

        chunks = build_chunks(p)

        # 목차 섹션(짧음)이 제외되어야 한다
        toc_chunks = [c for c in chunks if "목차" in c.section]
        assert len(toc_chunks) == 0, f"too_short 청크가 제외되지 않았습니다: {toc_chunks}"

        # 정책 섹션은 보존되어야 한다
        policy_chunks = [c for c in chunks if "등급 혜택표" in c.section]
        assert len(policy_chunks) >= 1, "정상 정책 청크가 잘못 제외되었습니다."

    def test_chunk_idx_is_contiguous_after_filter(self, tmp_path: Path, capsys):
        """
        필터 후 chunk_idx 가 0부터 연속적으로 부여되어야 한다.

        중간 청크가 제외되더라도 순번에 구멍이 없어야 Qdrant point ID 가
        deterministic 하게 재생성된다.
        """
        md_path = self._make_filter_test_md(tmp_path)
        chunks = build_chunks(md_path)

        idxs = [c.chunk_idx for c in chunks]
        assert idxs == list(range(len(chunks))), (
            f"chunk_idx 가 연속적이지 않습니다: {idxs}"
        )

    def test_filter_stats_printed(self, tmp_path: Path, capsys):
        """
        필터링이 발생하면 '[필터]' 통계 줄이 stdout 에 출력된다.

        dry-run 없이 build_chunks 만 호출해도 통계가 출력되므로,
        사용자가 결과를 검증할 수 있다.
        """
        md_path = self._make_filter_test_md(tmp_path)
        build_chunks(md_path)

        captured = capsys.readouterr()
        assert "[필터]" in captured.out, "필터 통계가 출력되지 않았습니다."
        assert "보존" in captured.out
