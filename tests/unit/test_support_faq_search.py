"""
support_assistant faq_search.py 단위 테스트 (v3.3).

외부 의존성(ES 클라이언트) 을 monkeypatch 로 완전히 대체해
네트워크 없이 로직만 검증한다.

검증 범위:
  1. ES hit 3건 → FaqCandidate 3개 반환 (score 내림차순)
  2. ES hit 0건 → 빈 리스트 반환
  3. ES timeout/오류 → 빈 리스트 반환 + warn 로그 (에러 전파 금지)
  4. 쿼리 빌드 — is_published=true 필터 포함 여부
"""

from __future__ import annotations

import pytest

from monglepick.agents.support_assistant import faq_search as faq_mod
from monglepick.agents.support_assistant.faq_search import (
    FaqCandidate,
    search_faq_candidates,
)


# =============================================================================
# 헬퍼 — ES 응답 mock 구조 생성
# =============================================================================


def _make_es_response(hits: list[dict]) -> dict:
    """
    AsyncElasticsearch.search() 가 반환하는 응답 구조를 흉내낸다.

    hits 각 요소: {"_id": str, "_score": float, "_source": {...}}
    """
    return {
        "hits": {
            "total": {"value": len(hits)},
            "hits": hits,
        }
    }


def _make_hit(
    faq_id: int,
    score: float,
    category: str = "GENERAL",
    question: str = "질문",
    answer: str = "답변",
    keywords: str | None = None,
) -> dict:
    """단일 ES hit dict 을 생성한다."""
    source: dict = {
        "faq_id": faq_id,
        "category": category,
        "question": question,
        "answer": answer,
        "is_published": True,
    }
    if keywords is not None:
        source["keywords"] = keywords
    return {
        "_id": str(faq_id),
        "_score": score,
        "_source": source,
    }


# =============================================================================
# ES 클라이언트 stub — get_elasticsearch() 를 교체
# =============================================================================


class _FakeEsClient:
    """
    AsyncElasticsearch 의 .search() 만 흉내낸 stub.

    `side_effect` 에 예외를 지정하면 search() 호출 시 raise 한다.
    `response` 에 dict 를 지정하면 그 값을 반환한다.
    `last_body` 에 마지막으로 전달된 쿼리 body 가 저장된다.
    """

    def __init__(self, response: dict | None = None, side_effect: Exception | None = None):
        self.response = response or _make_es_response([])
        self.side_effect = side_effect
        self.last_body: dict | None = None  # 쿼리 검증용

    async def search(self, index: str, body: dict, **kwargs) -> dict:
        """마지막 body 를 기록하고 응답 또는 예외를 반환한다."""
        self.last_body = body
        if self.side_effect is not None:
            raise self.side_effect
        return self.response


def _install_es_stub(monkeypatch, client: _FakeEsClient) -> _FakeEsClient:
    """
    faq_search 모듈 내부에서 호출하는 `get_elasticsearch` 를
    주어진 stub 클라이언트를 반환하는 코루틴으로 교체한다.
    """
    async def _fake_get_es():
        return client

    monkeypatch.setattr(faq_mod, "get_elasticsearch", _fake_get_es)
    return client


# =============================================================================
# 1) ES hit 3건 → FaqCandidate 3개 반환 (score 내림차순)
# =============================================================================


@pytest.mark.asyncio
async def test_returns_candidates_in_score_order(monkeypatch):
    """
    ES 가 3건의 hit 을 반환할 때:
    - FaqCandidate 3개가 반환된다
    - 반환 순서는 ES 응답 순서와 동일 (score 내림차순으로 이미 정렬된 상태)
    - 각 FaqCandidate 의 필드(faq_id, category, question, answer, score) 가 정확히 매핑된다
    """
    hits = [
        _make_hit(faq_id=10, score=18.5, category="PAYMENT", question="환불 방법", answer="환불 답변"),
        _make_hit(faq_id=7, score=9.2, category="ACCOUNT", question="비밀번호 재설정", answer="비밀번호 답변"),
        _make_hit(faq_id=3, score=4.1, category="GENERAL", question="연락처", answer="이메일"),
    ]
    stub = _install_es_stub(monkeypatch, _FakeEsClient(response=_make_es_response(hits)))

    result = await search_faq_candidates("환불 하고 싶어요", top_k=5)

    assert len(result) == 3

    # 첫 번째 후보 — 가장 높은 점수
    assert result[0].faq_id == 10
    assert result[0].score == pytest.approx(18.5)
    assert result[0].category == "PAYMENT"
    assert result[0].question == "환불 방법"
    assert result[0].answer == "환불 답변"

    # 두 번째 후보
    assert result[1].faq_id == 7
    assert result[1].score == pytest.approx(9.2)

    # 세 번째 후보
    assert result[2].faq_id == 3
    assert result[2].score == pytest.approx(4.1)


# =============================================================================
# 2) ES hit 0건 → 빈 리스트 반환
# =============================================================================


@pytest.mark.asyncio
async def test_returns_empty_list_when_no_hits(monkeypatch):
    """
    ES 검색 결과가 0건일 때 빈 리스트를 반환하며 예외를 던지지 않는다.
    """
    _install_es_stub(monkeypatch, _FakeEsClient(response=_make_es_response([])))

    result = await search_faq_candidates("아무것도 없는 질문", top_k=5)

    assert result == []


# =============================================================================
# 3) ES timeout/오류 → 빈 리스트 반환 (에러 전파 금지)
# =============================================================================


@pytest.mark.asyncio
async def test_returns_empty_list_on_es_timeout(monkeypatch):
    """
    ES 호출 중 타임아웃/네트워크 예외가 발생해도 빈 리스트를 반환한다 (에러 전파 금지).

    structlog warning 로그 출력 자체는 faq_search.py 코드에서 이미 보장하므로
    여기서는 반환값만 검증한다. structlog 는 테스트 실행 환경에 따라
    stdout 라우팅이 달라져 capsys/caplog 로 안정적 캡처가 어렵다.
    """
    # ConnectionError 는 실제 타임아웃/네트워크 장애를 대표하는 예외
    side_effect = ConnectionError("ES connection timed out")
    _install_es_stub(monkeypatch, _FakeEsClient(side_effect=side_effect))

    result = await search_faq_candidates("타임아웃 테스트", top_k=5)

    # 에러 전파 없이 빈 리스트를 반환해야 한다
    assert result == []


@pytest.mark.asyncio
async def test_returns_empty_list_on_generic_exception(monkeypatch):
    """
    RuntimeError 등 예상치 못한 예외에서도 빈 리스트를 반환하며 에러를 전파하지 않는다.
    """
    _install_es_stub(monkeypatch, _FakeEsClient(side_effect=RuntimeError("unexpected")))

    result = await search_faq_candidates("예외 테스트", top_k=5)

    assert result == []


# =============================================================================
# 4) is_published filter 포함 검증 (쿼리 빌드 단위)
# =============================================================================


@pytest.mark.asyncio
async def test_query_includes_is_published_filter(monkeypatch):
    """
    search_faq_candidates 가 ES 에 전달하는 쿼리 body 에
    is_published=true 필터가 반드시 포함되어야 한다.

    미공개 FAQ 가 검색 결과에 노출되는 것을 방지하는 핵심 안전장치이므로
    쿼리 레벨에서 강제 검증한다.
    """
    stub = _install_es_stub(monkeypatch, _FakeEsClient(response=_make_es_response([])))

    await search_faq_candidates("테스트 질문", top_k=5)

    # ES 에 실제로 전달된 쿼리 body 를 직접 검사
    body = stub.last_body
    assert body is not None, "ES search 가 호출되지 않았습니다"

    # bool.filter 에 is_published=true term 이 있어야 한다
    filters = body.get("query", {}).get("bool", {}).get("filter", [])
    published_filter_found = any(
        f.get("term", {}).get("is_published") is True
        for f in filters
    )
    assert published_filter_found, (
        "쿼리 body 에 is_published=true 필터가 없습니다. "
        "미공개 FAQ 가 노출될 수 있습니다."
    )


@pytest.mark.asyncio
async def test_query_uses_multi_match_with_correct_fields(monkeypatch):
    """
    multi_match 쿼리가 question^3, keywords^2, answer 필드를 포함하는지 검증한다.
    """
    stub = _install_es_stub(monkeypatch, _FakeEsClient(response=_make_es_response([])))

    await search_faq_candidates("비밀번호 찾기", top_k=3)

    body = stub.last_body
    assert body is not None

    must_clauses = body.get("query", {}).get("bool", {}).get("must", [])
    assert len(must_clauses) > 0

    # must 절 중 multi_match 가 있어야 한다
    multi_match = None
    for clause in must_clauses:
        if "multi_match" in clause:
            multi_match = clause["multi_match"]
            break

    assert multi_match is not None, "must 절에 multi_match 쿼리가 없습니다"

    fields = multi_match.get("fields", [])
    # question^3, keywords^2, answer 필드 포함 여부 확인
    assert "question^3" in fields
    assert "keywords^2" in fields
    assert "answer" in fields

    # 사용자 발화가 query 에 포함되어야 한다
    assert multi_match.get("query") == "비밀번호 찾기"


# =============================================================================
# 5) keywords 필드 None 처리
# =============================================================================


@pytest.mark.asyncio
async def test_keywords_field_is_none_when_absent(monkeypatch):
    """
    ES 문서에 keywords 필드가 없을 때 FaqCandidate.keywords 가 None 으로 설정된다.
    """
    hit = _make_hit(faq_id=1, score=5.0, keywords=None)
    # _source 에서 keywords 키 자체를 제거
    if "keywords" in hit["_source"]:
        del hit["_source"]["keywords"]

    _install_es_stub(monkeypatch, _FakeEsClient(response=_make_es_response([hit])))

    result = await search_faq_candidates("질문", top_k=5)

    assert len(result) == 1
    assert result[0].keywords is None


@pytest.mark.asyncio
async def test_keywords_field_is_set_when_present(monkeypatch):
    """
    ES 문서에 keywords 필드가 있으면 FaqCandidate.keywords 에 올바르게 매핑된다.
    """
    hit = _make_hit(faq_id=2, score=7.0, keywords="환불,반환,취소")

    _install_es_stub(monkeypatch, _FakeEsClient(response=_make_es_response([hit])))

    result = await search_faq_candidates("환불", top_k=5)

    assert len(result) == 1
    assert result[0].keywords == "환불,반환,취소"


# =============================================================================
# 6) 빈 발화 → ES 호출 없이 빈 리스트 반환
# =============================================================================


@pytest.mark.asyncio
async def test_empty_user_message_skips_es_call(monkeypatch):
    """
    빈 발화가 들어오면 ES 를 호출하지 않고 즉시 빈 리스트를 반환한다.
    불필요한 네트워크 트래픽과 인덱스 부하를 방지하는 조기 종료 로직 검증.
    """
    stub = _install_es_stub(monkeypatch, _FakeEsClient(response=_make_es_response([])))

    result_empty = await search_faq_candidates("", top_k=5)
    result_whitespace = await search_faq_candidates("   ", top_k=5)

    assert result_empty == []
    assert result_whitespace == []
    # 빈 발화이므로 ES search 가 아예 호출되지 않아야 한다
    assert stub.last_body is None, "빈 발화에서 ES 가 호출되었습니다"
