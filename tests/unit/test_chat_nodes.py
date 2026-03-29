"""
Chat Agent 노드 단위 테스트 (Phase 3 + 의도+감정 통합 + 구조화 힌트 + RAG 품질).

13개 노드 함수를 개별 테스트한다.
모든 테스트는 mock LLM + mock DB로 실행 (Ollama/DB 서버 불필요).

테스트 시나리오:
- 각 노드의 정상 동작
- 익명 사용자 처리
- 에러 시 기본값 반환
- 시대 파싱 헬퍼
- 검색 결과 → CandidateMovie 변환
- 추천 순위 정렬 (Phase 4 서브그래프)
- 의도+감정 통합 분류 (intent_emotion_classifier)
- 구조화된 후속 질문 힌트 (clarification)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from monglepick.agents.chat.models import (
    CandidateMovie,
    ChatAgentState,
    ClarificationResponse,
    EmotionResult,
    ExtractedPreferences,
    ImageAnalysisResult,
    IntentEmotionResult,
    IntentResult,
    RankedMovie,
    ScoreDetail,
    SearchQuery,
)
from monglepick.agents.chat.nodes import (
    _parse_era,
    _search_result_to_candidate,
    context_loader,
    error_handler,
    explanation_generator,
    general_responder,
    intent_emotion_classifier,
    preference_refiner,
    query_builder,
    question_generator,
    rag_retriever,
    recommendation_ranker,
    response_formatter,
    tool_executor_node,
)
from monglepick.rag.hybrid_search import SearchResult


# ============================================================
# 헬퍼 함수 테스트
# ============================================================

class TestParseEra:
    """_parse_era 시대 파싱 헬퍼 테스트."""

    def test_four_digit_decade(self):
        """'2020년대' → (2020, 2029)"""
        assert _parse_era("2020년대") == (2020, 2029)

    def test_two_digit_decade(self):
        """'90년대' → (1990, 1999)"""
        assert _parse_era("90년대") == (1990, 1999)

    def test_specific_year(self):
        """'2020' → (2020, 2020)"""
        assert _parse_era("2020") == (2020, 2020)

    def test_empty_string(self):
        """빈 문자열 → None"""
        assert _parse_era("") is None

    def test_none_string(self):
        """파싱 불가 문자열 → None"""
        assert _parse_era("최신") is None

    def test_80_decade(self):
        """'80년대' → (1980, 1989)"""
        assert _parse_era("80년대") == (1980, 1989)


class TestSearchResultToCandidate:
    """_search_result_to_candidate 변환 테스트."""

    def test_basic_conversion(self):
        """기본 SearchResult → CandidateMovie 변환."""
        result = SearchResult(
            movie_id="157336",
            title="인터스텔라",
            score=0.95,
            source="rrf",
            metadata={
                "title": "인터스텔라",
                "title_en": "Interstellar",
                "genres": ["SF", "드라마"],
                "director": "놀란",
                "rating": 8.7,
                "release_year": 2014,
            },
        )
        candidate = _search_result_to_candidate(result, 0)
        assert candidate.id == "157336"
        assert candidate.title == "인터스텔라"
        assert candidate.genres == ["SF", "드라마"]
        assert candidate.rrf_score == 0.95

    def test_empty_metadata(self):
        """메타데이터 없는 SearchResult → 기본값 CandidateMovie."""
        result = SearchResult(movie_id="1", title="테스트", score=0.5, source="rrf")
        candidate = _search_result_to_candidate(result, 0)
        assert candidate.id == "1"
        assert candidate.genres == []
        assert candidate.rating == 0.0


# ============================================================
# context_loader 테스트
# ============================================================

class TestContextLoader:
    """context_loader 노드 테스트."""

    @pytest.mark.asyncio
    async def test_anonymous_user(self):
        """익명 사용자(user_id 없음) → 빈 기본값."""
        state: ChatAgentState = {
            "user_id": "",
            "session_id": "test-session",
            "current_input": "안녕",
        }
        result = await context_loader(state)
        assert result["user_profile"] == {}
        assert result["watch_history"] == []
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["turn_count"] == 1

    @pytest.mark.asyncio
    async def test_with_user_profile(self, mock_mysql):
        """유저 프로필이 있는 경우 → MySQL에서 로드."""
        mock_mysql.set_user({"user_id": "test123", "nickname": "테스터"})
        mock_mysql.set_watch_history([
            {"movie_id": "157336", "title": "인터스텔라", "rating": 5.0, "watched_at": None},
        ])

        state: ChatAgentState = {
            "user_id": "test123",
            "session_id": "test-session",
            "current_input": "영화 추천해줘",
        }
        result = await context_loader(state)
        assert result["user_profile"]["user_id"] == "test123"
        assert len(result["watch_history"]) == 1
        assert result["turn_count"] == 1

    @pytest.mark.asyncio
    async def test_existing_messages(self):
        """기존 메시지가 있는 경우 → 추가 + turn_count 증가."""
        state: ChatAgentState = {
            "user_id": "",
            "session_id": "test-session",
            "current_input": "SF 좋아해",
            "messages": [
                {"role": "user", "content": "안녕"},
                {"role": "assistant", "content": "안녕하세요!"},
            ],
        }
        result = await context_loader(state)
        assert len(result["messages"]) == 3
        assert result["turn_count"] == 2  # user 메시지 2개

    @pytest.mark.asyncio
    async def test_db_error_fallback(self):
        """MySQL 연결 에러 시 → 빈 기본값으로 진행."""
        with patch(
            "monglepick.agents.chat.nodes.get_mysql",
            side_effect=Exception("DB 연결 실패"),
        ):
            state: ChatAgentState = {
                "user_id": "test123",
                "session_id": "test-session",
                "current_input": "추천해줘",
            }
            result = await context_loader(state)
            # DB 에러여도 빈 기본값으로 정상 진행
            assert result["user_profile"] == {}
            assert result["watch_history"] == []
            assert result["turn_count"] == 1


# ============================================================
# intent_emotion_classifier 테스트 (통합 의도+감정 분류)
# ============================================================

class TestIntentEmotionClassifier:
    """intent_emotion_classifier 통합 노드 테스트."""

    @pytest.mark.asyncio
    async def test_recommend_intent_with_emotion(self, mock_ollama):
        """추천 의도 + 감정 동시 분류 테스트."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="recommend", confidence=0.95,
                emotion="sad", mood_tags=["힐링", "감동"],
            ),
        )
        state: ChatAgentState = {
            "current_input": "우울한데 영화 추천해줘",
            "messages": [{"role": "user", "content": "우울한데 영화 추천해줘"}],
        }
        result = await intent_emotion_classifier(state)
        # intent와 emotion이 동시에 반환된다
        assert result["intent"].intent == "recommend"
        assert result["intent"].confidence == 0.95
        assert result["emotion"].emotion == "sad"
        assert "힐링" in result["emotion"].mood_tags

    @pytest.mark.asyncio
    async def test_general_intent_no_emotion(self, mock_ollama):
        """일반 대화: 의도=general, 감정=None."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="general", confidence=0.8,
                emotion=None, mood_tags=[],
            ),
        )
        state: ChatAgentState = {
            "current_input": "안녕하세요",
            "messages": [{"role": "user", "content": "안녕하세요"}],
        }
        result = await intent_emotion_classifier(state)
        assert result["intent"].intent == "general"
        assert result["emotion"].emotion is None

    @pytest.mark.asyncio
    async def test_image_boost_general_to_recommend(self, mock_ollama):
        """이미지 부스트: general + 이미지 분석 → recommend."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="general", confidence=0.5,
                emotion=None, mood_tags=[],
            ),
        )
        state: ChatAgentState = {
            "current_input": "이런 느낌의 영화",
            "messages": [{"role": "user", "content": "이런 느낌의 영화"}],
            "image_analysis": ImageAnalysisResult(
                genre_cues=["SF"],
                mood_cues=["웅장"],
                analyzed=True,
            ),
        }
        result = await intent_emotion_classifier(state)
        # 이미지 분석이 있으면 general → recommend로 부스트
        assert result["intent"].intent == "recommend"
        assert result["intent"].confidence >= 0.7

    @pytest.mark.asyncio
    async def test_no_image_boost_for_non_general(self, mock_ollama):
        """이미지 부스트: recommend는 그대로 유지 (general만 부스트)."""
        mock_ollama.set_structured_response(
            IntentEmotionResult(
                intent="recommend", confidence=0.9,
                emotion="excited", mood_tags=["스릴"],
            ),
        )
        state: ChatAgentState = {
            "current_input": "액션 영화 추천",
            "messages": [],
            "image_analysis": ImageAnalysisResult(
                genre_cues=["액션"],
                analyzed=True,
            ),
        }
        result = await intent_emotion_classifier(state)
        # 이미 recommend이므로 부스트 없음
        assert result["intent"].intent == "recommend"
        assert result["intent"].confidence == 0.9

    @pytest.mark.asyncio
    async def test_error_fallback(self, mock_ollama):
        """LLM 에러 시 → general fallback + 감정 None."""
        mock_ollama.set_error(RuntimeError("LLM 에러"))
        state: ChatAgentState = {
            "current_input": "테스트",
            "messages": [],
        }
        result = await intent_emotion_classifier(state)
        assert result["intent"].intent == "general"
        assert result["intent"].confidence == 0.0
        assert result["emotion"].emotion is None
        assert result["emotion"].mood_tags == []


# ============================================================
# preference_refiner 테스트
# ============================================================

class TestPreferenceRefiner:
    """preference_refiner 노드 테스트."""

    @pytest.mark.asyncio
    async def test_sufficient_preferences(self, mock_ollama):
        """충분한 선호 → needs_clarification=False."""
        mock_ollama.set_structured_response(
            ExtractedPreferences(
                genre_preference="SF",
                mood="웅장한",
                reference_movies=["인터스텔라"],
            ),
        )
        state: ChatAgentState = {
            "current_input": "인터스텔라 같은 웅장한 SF 영화",
            "emotion": EmotionResult(emotion="excited", mood_tags=["웅장"]),
            "turn_count": 1,
        }
        result = await preference_refiner(state)
        # genre(2.0) + mood(2.0, 감정 있으므로) + reference(1.5) = 5.5 ≥ 3.0
        assert result["needs_clarification"] is False

    @pytest.mark.asyncio
    async def test_insufficient_preferences(self, mock_ollama):
        """핵심 필드/의도/동적필터 모두 없으면 → needs_clarification=True (Intent-First)."""
        # Intent-First: user_intent, dynamic_filters, genre/mood/reference 모두 없어야 불충분
        mock_ollama.set_structured_response(
            ExtractedPreferences(
                genre_preference=None, mood=None,
                user_intent="", dynamic_filters=[], search_keywords=[],
                reference_movies=[],
            ),
        )
        state: ChatAgentState = {
            "current_input": "영화 추천해줘",
            "turn_count": 1,
        }
        result = await preference_refiner(state)
        assert result["needs_clarification"] is True

    @pytest.mark.asyncio
    async def test_turn_count_override(self, mock_ollama):
        """turn_count ≥ 3 → 선호 부족해도 추천 진행."""
        mock_ollama.set_structured_response(
            ExtractedPreferences(),  # 빈 선호
        )
        state: ChatAgentState = {
            "current_input": "아무거나 추천해줘",
            "turn_count": 3,  # 오버라이드 임계값
        }
        result = await preference_refiner(state)
        assert result["needs_clarification"] is False

    @pytest.mark.asyncio
    async def test_reference_movie_auto_enrichment(self, mock_ollama, mock_reference_lookup):
        """참조 영화만 있으면 DB에서 장르/무드를 자동 보강하여 바로 추천 진행."""
        # LLM이 reference_movies만 추출 (장르/무드는 빈 상태)
        mock_ollama.set_structured_response(
            ExtractedPreferences(
                genre_preference=None,
                mood=None,
                reference_movies=["인터스텔라"],
            ),
        )
        # DB 조회 결과: 인터스텔라의 장르/무드
        mock_reference_lookup.set_result({
            "genres": ["SF", "드라마", "모험"],
            "mood_tags": ["웅장", "감동", "몰입"],
        })
        state: ChatAgentState = {
            "current_input": "인터스텔라 같은 영화 보고 싶어",
            "turn_count": 1,
        }
        result = await preference_refiner(state)
        # reference(1.5) + genre(2.0, DB 보강) + mood(2.0, DB 보강) = 5.5 ≥ 3.0
        assert result["needs_clarification"] is False
        assert result["preferences"].genre_preference is not None
        assert result["preferences"].mood is not None
        assert "인터스텔라" in result["preferences"].reference_movies

    @pytest.mark.asyncio
    async def test_reference_movie_db_miss(self, mock_ollama, mock_reference_lookup):
        """참조 영화가 DB에 없어도 reference_movies는 핵심 필드 → 추천 진행 (Intent-First)."""
        mock_ollama.set_structured_response(
            ExtractedPreferences(
                genre_preference=None,
                mood=None,
                reference_movies=["존재하지않는영화"],
                user_intent="",  # intent 없음
            ),
        )
        # DB에 해당 영화 없음
        mock_reference_lookup.set_empty()
        state: ChatAgentState = {
            "current_input": "존재하지않는영화 같은 영화 보고 싶어",
            "turn_count": 1,
        }
        result = await preference_refiner(state)
        # Intent-First: reference_movies가 핵심 필드이므로 충분 → 추천 진행
        assert result["needs_clarification"] is False
        assert "존재하지않는영화" in result["preferences"].reference_movies

    @pytest.mark.asyncio
    async def test_reference_movie_genre_already_set(self, mock_ollama, mock_reference_lookup):
        """장르가 이미 있으면 DB에서 무드만 보강."""
        mock_ollama.set_structured_response(
            ExtractedPreferences(
                genre_preference="SF",  # 이미 추출됨
                mood=None,
                reference_movies=["인터스텔라"],
            ),
        )
        mock_reference_lookup.set_result({
            "genres": ["SF", "드라마", "모험"],
            "mood_tags": ["웅장", "감동"],
        })
        state: ChatAgentState = {
            "current_input": "인터스텔라 같은 SF 영화",
            "turn_count": 1,
        }
        result = await preference_refiner(state)
        # genre(2.0, 기존) + mood(2.0, DB 보강) + reference(1.5) = 5.5 ≥ 3.0
        assert result["needs_clarification"] is False
        assert result["preferences"].genre_preference == "SF"  # 기존 값 유지
        assert result["preferences"].mood is not None  # DB에서 보강됨


# ============================================================
# question_generator 테스트
# ============================================================

class TestQuestionGenerator:
    """question_generator 노드 테스트."""

    @pytest.mark.asyncio
    async def test_generates_question(self, mock_ollama):
        """후속 질문 생성 테스트."""
        mock_ollama.set_response("어떤 장르를 좋아하세요?")
        state: ChatAgentState = {
            "preferences": ExtractedPreferences(),
            "turn_count": 1,
        }
        result = await question_generator(state)
        assert result["follow_up_question"]
        assert result["response"]  # response에도 질문 설정

    @pytest.mark.asyncio
    async def test_generates_clarification_hints(self, mock_ollama):
        """구조화된 힌트(ClarificationResponse) 생성 테스트."""
        mock_ollama.set_response("어떤 장르를 좋아하세요?")
        state: ChatAgentState = {
            "preferences": ExtractedPreferences(),  # 빈 선호 → 모든 필드 부족
            "turn_count": 1,
        }
        result = await question_generator(state)
        # clarification 필드가 반환되어야 함
        clarification = result.get("clarification")
        assert clarification is not None
        assert isinstance(clarification, ClarificationResponse)
        assert clarification.question  # 질문 텍스트 존재
        assert len(clarification.hints) > 0  # 힌트 1개 이상
        assert len(clarification.hints) <= 3  # 최대 3개
        assert clarification.primary_field  # 1순위 부족 필드 존재

    @pytest.mark.asyncio
    async def test_clarification_hints_content(self, mock_ollama):
        """힌트 옵션이 FIELD_HINTS에서 올바르게 매핑되는지 확인."""
        mock_ollama.set_response("테스트 질문")
        state: ChatAgentState = {
            "preferences": ExtractedPreferences(),  # 빈 선호
            "turn_count": 1,
        }
        result = await question_generator(state)
        clarification = result["clarification"]
        # 각 힌트의 필드가 유효한 필드명인지 확인
        valid_fields = {"genre_preference", "mood", "viewing_context", "platform", "era", "exclude", "reference_movies"}
        for hint in clarification.hints:
            assert hint.field in valid_fields
            assert hint.label  # 레이블이 비어있지 않음

    @pytest.mark.asyncio
    async def test_retrieval_feedback_question(self, mock_ollama):
        """검색 품질 미달 시 피드백 메시지 포함 질문 생성."""
        state: ChatAgentState = {
            "preferences": ExtractedPreferences(genre_preference="다큐멘터리"),
            "turn_count": 1,
            "retrieval_feedback": "조건에 맞는 영화를 찾지 못했어요.",
        }
        result = await question_generator(state)
        # 피드백 메시지가 질문에 포함되어야 함
        assert "찾지 못했어요" in result["follow_up_question"]
        assert "구체적" in result["follow_up_question"]

    @pytest.mark.asyncio
    async def test_error_fallback(self, mock_ollama):
        """LLM 에러 시 → 기본 질문 반환 (generate_question 내부 fallback)."""
        mock_ollama.set_error(RuntimeError("LLM 에러"))
        state: ChatAgentState = {
            "preferences": ExtractedPreferences(),
            "turn_count": 0,
        }
        result = await question_generator(state)
        assert result["follow_up_question"]
        assert "영화" in result["follow_up_question"] or "알려" in result["follow_up_question"]
        # generate_question 내부에서 에러를 잡고 fallback 반환하므로
        # question_generator 노드의 try 블록은 정상 진행 → clarification 생성됨
        clarification = result.get("clarification")
        assert clarification is not None
        assert len(clarification.hints) > 0


# ============================================================
# query_builder 테스트
# ============================================================

class TestQueryBuilder:
    """query_builder 노드 테스트."""

    @pytest.mark.asyncio
    async def test_builds_search_query(self):
        """선호 조건 기반 검색 쿼리 구성."""
        state: ChatAgentState = {
            "current_input": "웅장한 SF 영화",
            "preferences": ExtractedPreferences(
                genre_preference="SF",
                mood="웅장한",
                era="2020년대",
            ),
            "emotion": EmotionResult(emotion="excited", mood_tags=["웅장", "스릴"]),
            "watch_history": [{"movie_id": "100", "title": "이미 본 영화"}],
        }
        result = await query_builder(state)
        sq = result["search_query"]
        assert isinstance(sq, SearchQuery)
        assert "SF" in sq.semantic_query
        assert sq.filters.get("genres") == ["SF"]
        assert sq.filters.get("year_range") == (2020, 2029)
        assert "100" in sq.exclude_ids
        assert "웅장" in sq.boost_keywords

    @pytest.mark.asyncio
    async def test_empty_preferences(self):
        """빈 선호 → 기본 쿼리."""
        state: ChatAgentState = {
            "current_input": "영화 추천",
            "preferences": ExtractedPreferences(),
            "emotion": EmotionResult(),
            "watch_history": [],
        }
        result = await query_builder(state)
        sq = result["search_query"]
        assert sq.semantic_query == "영화 추천"
        assert sq.filters == {}


# ============================================================
# rag_retriever 테스트
# ============================================================

class TestRagRetriever:
    """rag_retriever 노드 테스트."""

    @pytest.mark.asyncio
    async def test_returns_candidates(self, mock_hybrid_search):
        """검색 결과 → CandidateMovie 리스트 변환."""
        mock_hybrid_search.set_results([
            SearchResult(
                movie_id="157336",
                title="인터스텔라",
                score=0.95,
                source="rrf",
                metadata={"genres": ["SF"], "director": "놀란", "rating": 8.7, "release_year": 2014},
            ),
            SearchResult(
                movie_id="27205",
                title="인셉션",
                score=0.90,
                source="rrf",
                metadata={"genres": ["SF", "액션"], "director": "놀란", "rating": 8.4, "release_year": 2010},
            ),
        ])
        state: ChatAgentState = {
            "search_query": SearchQuery(semantic_query="SF 영화", keyword_query="SF 영화"),
            "emotion": EmotionResult(),
        }
        result = await rag_retriever(state)
        assert len(result["candidate_movies"]) == 2
        assert result["candidate_movies"][0].title == "인터스텔라"

    @pytest.mark.asyncio
    async def test_excludes_watched_movies(self, mock_hybrid_search):
        """시청한 영화 제외."""
        mock_hybrid_search.set_results([
            SearchResult(movie_id="100", title="이미 본 영화", score=0.9, source="rrf", metadata={}),
            SearchResult(movie_id="200", title="안 본 영화", score=0.8, source="rrf", metadata={}),
        ])
        state: ChatAgentState = {
            "search_query": SearchQuery(
                semantic_query="영화",
                keyword_query="영화",
                exclude_ids=["100"],
            ),
            "emotion": EmotionResult(),
        }
        result = await rag_retriever(state)
        assert len(result["candidate_movies"]) == 1
        assert result["candidate_movies"][0].id == "200"

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_hybrid_search):
        """검색 결과 없음 → 빈 리스트."""
        mock_hybrid_search.set_results([])
        state: ChatAgentState = {
            "search_query": SearchQuery(semantic_query="없는 영화"),
        }
        result = await rag_retriever(state)
        assert result["candidate_movies"] == []


# ============================================================
# recommendation_ranker 테스트
# ============================================================

class TestRecommendationRanker:
    """recommendation_ranker 노드 테스트 (Phase 4 스텁)."""

    @pytest.mark.asyncio
    async def test_ranks_by_rrf_score(self):
        """RRF 점수 기준 정렬."""
        candidates = [
            CandidateMovie(id="1", title="영화A", rrf_score=0.5),
            CandidateMovie(id="2", title="영화B", rrf_score=0.9),
            CandidateMovie(id="3", title="영화C", rrf_score=0.7),
        ]
        state: ChatAgentState = {"candidate_movies": candidates}
        result = await recommendation_ranker(state)
        ranked = result["ranked_movies"]
        assert len(ranked) == 3
        assert ranked[0].title == "영화B"
        assert ranked[0].rank == 1
        assert ranked[1].title == "영화C"
        assert ranked[2].title == "영화A"

    @pytest.mark.asyncio
    async def test_limits_to_five(self):
        """상위 5편 제한."""
        candidates = [
            CandidateMovie(id=str(i), title=f"영화{i}", rrf_score=float(i) / 10)
            for i in range(10)
        ]
        state: ChatAgentState = {"candidate_movies": candidates}
        result = await recommendation_ranker(state)
        assert len(result["ranked_movies"]) == 5

    @pytest.mark.asyncio
    async def test_empty_candidates(self):
        """후보 없음 → 빈 결과."""
        state: ChatAgentState = {"candidate_movies": []}
        result = await recommendation_ranker(state)
        assert result["ranked_movies"] == []

    @pytest.mark.asyncio
    async def test_score_detail_populated(self):
        """Phase 4: ScoreDetail이 올바르게 채워진다 (Cold Start → CF/CBF=0.0)."""
        candidates = [CandidateMovie(id="1", title="테스트", rrf_score=0.8)]
        state: ChatAgentState = {"candidate_movies": candidates}
        result = await recommendation_ranker(state)
        sd = result["ranked_movies"][0].score_detail
        # Cold Start (시청이력 없음): CF/CBF=0.0
        assert sd.cf_score == 0.0
        assert sd.cbf_score == 0.0
        # hybrid_score는 popularity_fallback 기반 (정규화됨)
        assert sd.hybrid_score >= 0.0


# ============================================================
# explanation_generator 테스트
# ============================================================

class TestExplanationGenerator:
    """explanation_generator 노드 테스트."""

    @pytest.mark.asyncio
    async def test_generates_explanations(self, mock_ollama):
        """추천 이유 생성 테스트."""
        mock_ollama.set_response("우주를 배경으로 한 감동적인 SF 영화입니다.")
        ranked = [
            RankedMovie(id="1", title="인터스텔라", rank=1),
        ]
        state: ChatAgentState = {
            "ranked_movies": ranked,
            "emotion": EmotionResult(emotion="excited"),
            "preferences": ExtractedPreferences(genre_preference="SF"),
            "watch_history": [],
        }
        result = await explanation_generator(state)
        assert len(result["ranked_movies"]) == 1
        assert result["ranked_movies"][0].explanation != ""

    @pytest.mark.asyncio
    async def test_empty_ranked(self):
        """추천 영화 없음 → 빈 결과."""
        state: ChatAgentState = {"ranked_movies": []}
        result = await explanation_generator(state)
        assert result["ranked_movies"] == []


# ============================================================
# response_formatter 테스트
# ============================================================

class TestResponseFormatter:
    """response_formatter 노드 테스트."""

    @pytest.mark.asyncio
    async def test_recommendation_format(self):
        """추천 응답 — 몽글이 LLM이 자연스러운 대화체로 응답을 생성한다."""
        ranked = [
            RankedMovie(
                id="1",
                title="인터스텔라",
                genres=["SF", "드라마"],
                director="놀란",
                rating=8.7,
                release_year=2014,
                rank=1,
                explanation="웅장한 우주 SF",
            ),
        ]
        state: ChatAgentState = {
            "ranked_movies": ranked,
            "messages": [{"role": "user", "content": "추천해줘"}],
        }
        result = await response_formatter(state)
        # 몽글이가 생성한 응답은 유효한 텍스트여야 함
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 10
        assert result["messages"][-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_question_format(self):
        """질문 응답 포맷 (기존 response 사용)."""
        state: ChatAgentState = {
            "ranked_movies": [],
            "response": "어떤 장르를 좋아하세요?",
            "messages": [{"role": "user", "content": "추천해줘"}],
        }
        result = await response_formatter(state)
        assert result["response"] == "어떤 장르를 좋아하세요?"

    @pytest.mark.asyncio
    async def test_error_format(self):
        """에러 응답 포맷."""
        state: ChatAgentState = {
            "ranked_movies": [],
            "error": "테스트 에러",
            "messages": [],
        }
        result = await response_formatter(state)
        assert "죄송" in result["response"]

    @pytest.mark.asyncio
    async def test_default_format(self):
        """기본 응답 (아무것도 설정 안 됨)."""
        state: ChatAgentState = {
            "ranked_movies": [],
            "messages": [],
        }
        result = await response_formatter(state)
        assert result["response"]  # 빈 문자열이 아님


# ============================================================
# error_handler 테스트
# ============================================================

class TestErrorHandler:
    """error_handler 노드 테스트."""

    @pytest.mark.asyncio
    async def test_returns_friendly_message(self):
        """에러 시 친절한 안내 메시지."""
        state: ChatAgentState = {
            "error": "테스트 에러",
            "intent": IntentResult(intent="recommend", confidence=0.9),
        }
        result = await error_handler(state)
        assert "죄송" in result["response"]
        assert "다시" in result["response"]


# ============================================================
# general_responder 테스트
# ============================================================

class TestGeneralResponder:
    """general_responder 노드 테스트."""

    @pytest.mark.asyncio
    async def test_generates_response(self, mock_ollama):
        """일반 대화 응답 생성."""
        mock_ollama.set_response("안녕하세요! 반갑습니다 😊")
        state: ChatAgentState = {
            "current_input": "안녕",
            "messages": [{"role": "user", "content": "안녕"}],
        }
        result = await general_responder(state)
        assert result["response"] == "안녕하세요! 반갑습니다 😊"

    @pytest.mark.asyncio
    async def test_error_fallback(self, mock_ollama):
        """LLM 에러 시 기본 응답 (general_chat_chain.DEFAULT_ERROR_MESSAGE 또는 노드 fallback)."""
        mock_ollama.set_error(RuntimeError("LLM 에러"))
        state: ChatAgentState = {
            "current_input": "안녕",
            "messages": [],
        }
        result = await general_responder(state)
        # generate_general_response 체인이 에러 시 DEFAULT_ERROR_MESSAGE 반환
        # 또는 노드 fallback 반환 — 둘 다 빈 문자열이 아님
        assert result["response"]
        assert len(result["response"]) > 0


# ============================================================
# tool_executor_node 테스트
# ============================================================

class TestToolExecutorNode:
    """tool_executor_node 노드 테스트 (Phase 6 스텁)."""

    @pytest.mark.asyncio
    async def test_info_intent(self):
        """info 의도 → 안내 메시지."""
        state: ChatAgentState = {
            "intent": IntentResult(intent="info", confidence=0.9),
        }
        result = await tool_executor_node(state)
        assert "준비" in result["response"]

    @pytest.mark.asyncio
    async def test_theater_intent(self):
        """theater 의도 → 안내 메시지."""
        state: ChatAgentState = {
            "intent": IntentResult(intent="theater", confidence=0.8),
        }
        result = await tool_executor_node(state)
        assert "영화관" in result["response"]

    @pytest.mark.asyncio
    async def test_booking_intent(self):
        """booking 의도 → 안내 메시지."""
        state: ChatAgentState = {
            "intent": IntentResult(intent="booking", confidence=0.7),
        }
        result = await tool_executor_node(state)
        assert "예매" in result["response"]
