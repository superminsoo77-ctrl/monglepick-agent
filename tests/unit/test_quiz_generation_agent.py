"""
영화 퀴즈 생성 에이전트 단위 테스트 (2026-04-28).

대상 모듈: `monglepick.agents.quiz_generation.{nodes, graph, models, prompts}`

테스트 영역 (7 노드 + 헬퍼):
    1. 헬퍼: _parse_json_array / _parse_quiz_json / _is_valid_options / _contains_spoiler
    2. 헬퍼: _build_fallback_draft / _select_category
    3. movie_selector       — 빈 DB / 후보 매핑
    4. metadata_enricher    — 메타 보강 / 빈 후보 pass
    5. question_generator   — LLM 정상 / 스키마 미달 → fallback
    6. quality_validator    — 정상 / 정답 불일치 / 스포일러 / fallback 자동 통과
    7. diversity_checker    — 동일 영화·카테고리 중복 제거
    8. fallback_filler      — 실패 영화에만 fallback 보충
    9. persistence          — INSERT 성공 카운트 / 빈 입력 메시지
   10. graph end-to-end     — 빈 DB 시 success=False 메시지 반환

MySQL pool / LLM 은 모두 unittest.mock — CI 환경에서 인프라 의존성 0.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monglepick.agents.quiz_generation import nodes as qg_nodes
from monglepick.agents.quiz_generation.models import (
    CandidateMovie,
    QuizDraft,
    QuizGenerationState,
)
from monglepick.agents.quiz_generation.nodes import (
    _build_fallback_draft,
    _contains_spoiler,
    _is_valid_options,
    _parse_json_array,
    _parse_quiz_json,
    _select_category,
    diversity_checker,
    fallback_filler,
    metadata_enricher,
    movie_selector,
    persistence,
    quality_validator,
    question_generator,
)


# ============================================================
# 헬퍼: MySQL pool/cursor 모킹 컨텍스트 매니저
# ============================================================


def _build_mysql_pool_mock(rows: list[tuple]) -> MagicMock:
    """
    aiomysql.Pool 의 acquire() → conn → cursor 체인을 모킹한다.

    cursor.fetchall() 가 주어진 rows 를 반환한다. lastrowid 는 1 부터 +1 씩 증가.
    """
    cursor = MagicMock()
    cursor.fetchall = AsyncMock(return_value=rows)
    cursor.execute = AsyncMock()

    # lastrowid 카운터 (persistence 테스트용)
    cursor._counter = {"value": 0}

    def _next_lastrowid():
        cursor._counter["value"] += 1
        return cursor._counter["value"]

    # MagicMock 의 property 설정 — 매 접근마다 +1
    type(cursor).lastrowid = property(lambda self: _next_lastrowid())

    @asynccontextmanager
    async def _cur_cm():
        yield cursor

    conn = MagicMock()
    conn.cursor = lambda: _cur_cm()
    conn.commit = AsyncMock()

    @asynccontextmanager
    async def _conn_cm():
        yield conn

    pool = MagicMock()
    pool.acquire = lambda: _conn_cm()
    pool._cursor = cursor
    return pool


# ============================================================
# 1) 헬퍼: _parse_json_array
# ============================================================


class TestParseJsonArray:
    def test_json_array_string(self):
        assert _parse_json_array('["액션", "SF"]') == ["액션", "SF"]

    def test_already_list(self):
        assert _parse_json_array(["a", "b"]) == ["a", "b"]

    def test_comma_fallback(self):
        # 깨진 JSON 은 콤마 분리로 fallback
        assert _parse_json_array("드라마, 코미디") == ["드라마", "코미디"]

    def test_empty_returns_empty_list(self):
        assert _parse_json_array("") == []
        assert _parse_json_array(None) == []

    def test_whitespace_strip(self):
        assert _parse_json_array('[" 액션 ", "SF"]') == ["액션", "SF"]


# ============================================================
# 2) 헬퍼: _parse_quiz_json
# ============================================================


class TestParseQuizJson:
    def test_plain_json(self):
        text = '{"question": "Q", "options": ["A","B","C","D"], "correctAnswer": "A"}'
        parsed = _parse_quiz_json(text)
        assert parsed["question"] == "Q"
        assert parsed["correctAnswer"] == "A"

    def test_markdown_codeblock_stripped(self):
        text = '```json\n{"question": "Q"}\n```'
        parsed = _parse_quiz_json(text)
        assert parsed["question"] == "Q"

    def test_brace_block_extraction(self):
        text = '잡담 {"question": "Q", "x": 1} 잡담'
        parsed = _parse_quiz_json(text)
        assert parsed["question"] == "Q"

    def test_invalid_returns_empty(self):
        assert _parse_quiz_json("그냥 텍스트") == {}


# ============================================================
# 3) 헬퍼: _is_valid_options
# ============================================================


class TestIsValidOptions:
    def test_valid_four_options(self):
        assert _is_valid_options(["A", "B", "C", "D"]) is True

    def test_three_options_fail(self):
        assert _is_valid_options(["A", "B", "C"]) is False

    def test_duplicate_fail(self):
        assert _is_valid_options(["A", "A", "B", "C"]) is False

    def test_empty_string_fail(self):
        assert _is_valid_options(["A", "", "C", "D"]) is False

    def test_non_list_fail(self):
        assert _is_valid_options("ABCD") is False


# ============================================================
# 4) 헬퍼: _contains_spoiler
# ============================================================


class TestContainsSpoiler:
    def test_clean_text(self):
        assert _contains_spoiler("이 영화의 감독은 누구인가요?") is False

    def test_ending_word(self):
        assert _contains_spoiler("결말이 어떻게 되나요?") is True

    def test_death_word(self):
        assert _contains_spoiler("주인공이 죽는다") is True

    def test_empty(self):
        assert _contains_spoiler("") is False


# ============================================================
# 5) 헬퍼: _build_fallback_draft
# ============================================================


class TestBuildFallbackDraft:
    def test_with_genre(self):
        movie = CandidateMovie(
            movie_id="m1", title="기생충", genres=["드라마"], release_year="2019",
        )
        draft = _build_fallback_draft(movie)
        assert draft.is_fallback is True
        assert draft.valid is True
        assert draft.correct_answer == "드라마"
        assert "드라마" in draft.options
        assert len(draft.options) == 4
        assert len(set(draft.options)) == 4  # 모두 서로 다른 값

    def test_empty_genres_default_drama(self):
        movie = CandidateMovie(movie_id="m2", title="X", genres=[])
        draft = _build_fallback_draft(movie)
        assert draft.correct_answer == "드라마"


# ============================================================
# 6) 헬퍼: _select_category
# ============================================================


class TestSelectCategory:
    def test_round_robin_index_zero_genre(self):
        m = CandidateMovie(movie_id="m1", title="X", genres=["드라마"])
        assert _select_category(m, 0) == "genre"

    def test_director_downgrade_when_empty(self):
        # 인덱스 1 = director 카테고리. director 비어 있으면 general 로 강등.
        m = CandidateMovie(movie_id="m1", title="X", director="")
        assert _select_category(m, 1) == "general"

    def test_director_kept_when_present(self):
        m = CandidateMovie(movie_id="m1", title="X", director="봉준호")
        assert _select_category(m, 1) == "director"

    def test_year_downgrade(self):
        m = CandidateMovie(movie_id="m1", title="X", release_year="")
        assert _select_category(m, 2) == "general"

    def test_cast_downgrade(self):
        m = CandidateMovie(movie_id="m1", title="X", cast_members=[])
        assert _select_category(m, 3) == "general"

    def test_plot_downgrade(self):
        m = CandidateMovie(movie_id="m1", title="X", overview="")
        assert _select_category(m, 4) == "general"


# ============================================================
# 7) movie_selector
# ============================================================


class TestMovieSelector:
    @pytest.mark.asyncio
    async def test_empty_db_returns_message(self):
        pool = _build_mysql_pool_mock(rows=[])
        with patch.object(qg_nodes, "get_mysql", AsyncMock(return_value=pool)):
            result = await movie_selector({"count": 5, "exclude_recent_days": 7})
        assert result["candidates"] == []
        assert "후보 영화가 없습니다" in result["selector_message"]

    @pytest.mark.asyncio
    async def test_maps_rows_to_candidates(self):
        rows = [
            ("mv1", "기생충", "Parasite", '["드라마","스릴러"]', 2019, None),
            ("mv2", "올드보이", "Oldboy", '["스릴러"]', 2003, None),
        ]
        pool = _build_mysql_pool_mock(rows=rows)
        with patch.object(qg_nodes, "get_mysql", AsyncMock(return_value=pool)):
            result = await movie_selector({"count": 2, "exclude_recent_days": 7})

        cands = result["candidates"]
        assert len(cands) == 2
        assert cands[0].movie_id == "mv1"
        assert cands[0].title == "기생충"
        assert cands[0].genres == ["드라마", "스릴러"]
        assert cands[0].release_year == "2019"
        assert result["selector_message"] == ""

    @pytest.mark.asyncio
    async def test_pool_failure_swallowed(self):
        # get_mysql 자체가 raise — 노드는 빈 결과 + 메시지 반환해야 함
        with patch.object(qg_nodes, "get_mysql", AsyncMock(side_effect=RuntimeError("db down"))):
            result = await movie_selector({"count": 5})
        assert result["candidates"] == []
        assert "오류" in result["selector_message"]


# ============================================================
# 8) metadata_enricher
# ============================================================


class TestMetadataEnricher:
    @pytest.mark.asyncio
    async def test_empty_candidates_pass_through(self):
        result = await metadata_enricher({"candidates": []})
        assert result["enriched_candidates"] == []

    @pytest.mark.asyncio
    async def test_enriches_overview_director_cast(self):
        candidates = [
            CandidateMovie(movie_id="mv1", title="기생충", genres=["드라마"]),
        ]
        rows = [
            ("mv1", "줄거리 본문", "봉준호", '["송강호","이선균"]', '["가족","계급"]', "한 가족의 이야기"),
        ]
        pool = _build_mysql_pool_mock(rows=rows)
        with patch.object(qg_nodes, "get_mysql", AsyncMock(return_value=pool)):
            result = await metadata_enricher({"candidates": candidates})

        out = result["enriched_candidates"]
        assert len(out) == 1
        assert out[0].overview == "줄거리 본문"
        assert out[0].director == "봉준호"
        assert "송강호" in out[0].cast_members
        assert "가족" in out[0].keywords

    @pytest.mark.asyncio
    async def test_db_failure_returns_original_candidates(self):
        candidates = [CandidateMovie(movie_id="mv1", title="X")]
        with patch.object(qg_nodes, "get_mysql", AsyncMock(side_effect=RuntimeError("boom"))):
            result = await metadata_enricher({"candidates": candidates})
        # 에러 전파 금지 — 보강 실패해도 후보 그대로 반환
        assert len(result["enriched_candidates"]) == 1


# ============================================================
# 9) question_generator
# ============================================================


class TestQuestionGenerator:
    @pytest.mark.asyncio
    async def test_empty_candidates(self):
        result = await question_generator({"enriched_candidates": []})
        assert result["drafts"] == []

    @pytest.mark.asyncio
    async def test_llm_success_returns_valid_draft(self):
        movies = [CandidateMovie(movie_id="mv1", title="X", genres=["드라마"])]
        llm_response = MagicMock()
        llm_response.content = json.dumps({
            "question": "이 영화의 주요 장르는?",
            "options": ["드라마", "액션", "코미디", "공포"],
            "correctAnswer": "드라마",
            "explanation": "드라마 장르입니다.",
            "category": "genre",
        }, ensure_ascii=False)

        with patch.object(qg_nodes, "get_conversation_llm", MagicMock(return_value=MagicMock())), \
             patch.object(qg_nodes, "guarded_ainvoke", AsyncMock(return_value=llm_response)):
            result = await question_generator({
                "enriched_candidates": movies,
                "difficulty": "medium",
            })

        drafts = result["drafts"]
        assert len(drafts) == 1
        assert drafts[0].is_fallback is False
        assert drafts[0].valid is True
        assert drafts[0].correct_answer == "드라마"

    @pytest.mark.asyncio
    async def test_llm_invalid_schema_falls_back(self):
        movies = [CandidateMovie(movie_id="mv1", title="X", genres=["스릴러"])]
        bad_response = MagicMock()
        bad_response.content = "그냥 잡담"  # JSON 아님

        with patch.object(qg_nodes, "get_conversation_llm", MagicMock(return_value=MagicMock())), \
             patch.object(qg_nodes, "guarded_ainvoke", AsyncMock(return_value=bad_response)):
            result = await question_generator({
                "enriched_candidates": movies,
                "difficulty": "medium",
            })

        drafts = result["drafts"]
        assert len(drafts) == 1
        assert drafts[0].is_fallback is True
        assert drafts[0].valid is True


# ============================================================
# 10) quality_validator
# ============================================================


class TestQualityValidator:
    @pytest.mark.asyncio
    async def test_valid_draft_passes(self):
        d = QuizDraft(
            movie_id="m1", movie_title="X",
            question="이 영화의 주요 장르는 무엇인가요?",
            options=["드라마", "액션", "코미디", "공포"],
            correct_answer="드라마",
            explanation="드라마 장르입니다.",
        )
        result = await quality_validator({"drafts": [d]})
        assert result["validated_drafts"][0].valid is True

    @pytest.mark.asyncio
    async def test_answer_not_in_options_rejected(self):
        d = QuizDraft(
            movie_id="m1", movie_title="X",
            question="장르는 무엇인가요?",
            options=["A", "B", "C", "D"],
            correct_answer="E",  # options 에 없음
        )
        result = await quality_validator({"drafts": [d]})
        rejected = result["validated_drafts"][0]
        assert rejected.valid is False
        assert "ANSWER_NOT_IN_OPTIONS" in rejected.reject_reason

    @pytest.mark.asyncio
    async def test_spoiler_rejected(self):
        d = QuizDraft(
            movie_id="m1", movie_title="X",
            question="이 영화의 결말은 어떻게 되나요?",  # 스포일러
            options=["A", "B", "C", "D"],
            correct_answer="A",
        )
        result = await quality_validator({"drafts": [d]})
        rejected = result["validated_drafts"][0]
        assert rejected.valid is False
        assert "SPOILER_DETECTED" in rejected.reject_reason

    @pytest.mark.asyncio
    async def test_fallback_draft_always_passes(self):
        # fallback 은 검증 우회 — 비록 question 이 비어도 통과
        d = QuizDraft(
            movie_id="m1", movie_title="X",
            question="짧",  # < 10자
            options=["A", "B", "C", "D"],
            correct_answer="A",
            is_fallback=True,
        )
        result = await quality_validator({"drafts": [d]})
        assert result["validated_drafts"][0].valid is True

    @pytest.mark.asyncio
    async def test_question_too_short(self):
        d = QuizDraft(
            movie_id="m1", movie_title="X",
            question="짧다",  # 2자
            options=["A", "B", "C", "D"],
            correct_answer="A",
        )
        result = await quality_validator({"drafts": [d]})
        assert result["validated_drafts"][0].valid is False


# ============================================================
# 11) diversity_checker
# ============================================================


class TestDiversityChecker:
    @pytest.mark.asyncio
    async def test_dedup_same_movie_category(self):
        d1 = QuizDraft(
            movie_id="m1", movie_title="X",
            question="장르?", options=["A","B","C","D"], correct_answer="A",
            category="genre",
        )
        d2 = QuizDraft(
            movie_id="m1", movie_title="X",
            question="다른 질문이지만 같은 카테고리",
            options=["A","B","C","D"], correct_answer="A",
            category="genre",
        )
        result = await diversity_checker({"validated_drafts": [d1, d2]})
        out = result["diversified_drafts"]
        # 첫 번째는 통과, 두 번째는 중복 제거
        assert out[0].valid is True
        assert out[1].valid is False
        assert "DUPLICATE" in out[1].reject_reason

    @pytest.mark.asyncio
    async def test_already_invalid_pass_through(self):
        d = QuizDraft(
            movie_id="m1", movie_title="X",
            question="Q", options=["A","B","C","D"], correct_answer="A",
            valid=False, reject_reason="PRIOR",
        )
        result = await diversity_checker({"validated_drafts": [d]})
        # 이미 invalid 한 건 그대로 유지
        assert result["diversified_drafts"][0].valid is False
        assert result["diversified_drafts"][0].reject_reason == "PRIOR"


# ============================================================
# 12) fallback_filler
# ============================================================


class TestFallbackFiller:
    @pytest.mark.asyncio
    async def test_invalid_draft_replaced_with_fallback(self):
        d = QuizDraft(
            movie_id="m1", movie_title="X",
            question="bad", options=["A","B","C","D"], correct_answer="Z",
            valid=False, reject_reason="ANSWER_NOT_IN_OPTIONS",
        )
        cand = CandidateMovie(movie_id="m1", title="X", genres=["코미디"])
        result = await fallback_filler({
            "diversified_drafts": [d],
            "enriched_candidates": [cand],
        })
        out = result["final_drafts"]
        assert len(out) == 1
        assert out[0].is_fallback is True
        assert out[0].correct_answer == "코미디"

    @pytest.mark.asyncio
    async def test_skip_fallback_when_movie_already_has_valid(self):
        valid = QuizDraft(
            movie_id="m1", movie_title="X",
            question="Q1", options=["A","B","C","D"], correct_answer="A",
        )
        invalid = QuizDraft(
            movie_id="m1", movie_title="X",
            question="Q2bad", options=["A","B","C","D"], correct_answer="Z",
            valid=False, reject_reason="DUPLICATE",
        )
        cand = CandidateMovie(movie_id="m1", title="X", genres=["드라마"])
        result = await fallback_filler({
            "diversified_drafts": [valid, invalid],
            "enriched_candidates": [cand],
        })
        # 같은 영화에 valid 가 이미 있으므로 fallback 추가 X
        assert len(result["final_drafts"]) == 1
        assert result["final_drafts"][0] is valid


# ============================================================
# 13) persistence
# ============================================================


class TestPersistence:
    @pytest.mark.asyncio
    async def test_inserts_only_valid_drafts(self):
        d_valid = QuizDraft(
            movie_id="m1", movie_title="X",
            question="이 영화 장르는?", options=["드라마","액션","코미디","공포"],
            correct_answer="드라마",
        )
        d_invalid = QuizDraft(
            movie_id="m2", movie_title="Y",
            question="bad", options=["A","B","C","D"], correct_answer="Z",
            valid=False, reject_reason="X",
        )
        pool = _build_mysql_pool_mock(rows=[])

        with patch.object(qg_nodes, "get_mysql", AsyncMock(return_value=pool)):
            result = await persistence({
                "final_drafts": [d_valid, d_invalid],
                "reward_point": 10,
            })

        assert len(result["persisted"]) == 1
        assert result["persisted"][0].movie_id == "m1"
        assert result["success"] is True
        assert "1개" in result["final_message"]

    @pytest.mark.asyncio
    async def test_empty_drafts_uses_selector_message(self):
        result = await persistence({
            "final_drafts": [],
            "selector_message": "후보 영화가 없습니다.",
            "reward_point": 10,
        })
        assert result["success"] is False
        assert result["persisted"] == []
        assert "후보 영화가 없습니다" in result["final_message"]


# ============================================================
# 14) graph end-to-end (빈 DB)
# ============================================================


class TestGraphEndToEnd:
    @pytest.mark.asyncio
    async def test_empty_db_short_circuit(self):
        """빈 DB → movie_selector 가 빈 결과 반환 → 그래프 끝까지 흘러서 success=False."""
        from monglepick.agents.quiz_generation.graph import quiz_generation_graph

        pool = _build_mysql_pool_mock(rows=[])
        with patch.object(qg_nodes, "get_mysql", AsyncMock(return_value=pool)):
            final = await quiz_generation_graph.ainvoke({
                "genre": None,
                "difficulty": "medium",
                "count": 3,
                "exclude_recent_days": 7,
                "reward_point": 10,
            })

        assert final["success"] is False
        assert final["persisted"] == []
        assert "후보 영화가 없습니다" in final["final_message"]
