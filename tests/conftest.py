"""
테스트 공통 Fixture (Phase 2 LLM 체인 + Phase 3 Chat Agent + 의도+감정 통합 테스트용).

모든 테스트는 Ollama/DB 서버 없이 mock 기반으로 실행된다.
ChatOllama(Ollama), MySQL, hybrid_search, image_analysis, classify_intent_and_emotion을
패치하여 사전 정의된 응답을 반환한다.

Fixture 목록:
- mock_ollama: ChatOllama(Ollama) 패치 (ainvoke가 preset 응답 반환)
- mock_mysql: aiomysql Pool/Connection/Cursor mock (get_mysql 패치)
- mock_hybrid_search: hybrid_search 함수 mock (기본 빈 결과)
- mock_image_analysis: analyze_image 함수 mock (이미지 분석 결과)
- mock_intent_emotion: classify_intent_and_emotion 함수 mock (통합 의도+감정)
- sample_preferences: 미리 채워진 ExtractedPreferences
- sample_candidate_movie: CandidateMovie 인스턴스
- sample_ranked_movie: RankedMovie 인스턴스
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monglepick.agents.chat.models import (
    CandidateMovie,
    ExtractedPreferences,
    RankedMovie,
    ScoreDetail,
)


@pytest.fixture
def sample_preferences() -> ExtractedPreferences:
    """미리 채워진 사용자 선호 조건 fixture."""
    return ExtractedPreferences(
        genre_preference="SF",
        mood="웅장한",
        viewing_context="혼자",
        platform="넷플릭스",
        reference_movies=["인터스텔라", "인셉션"],
        era="2020년대",
        exclude="공포 제외",
    )


@pytest.fixture
def sample_candidate_movie() -> CandidateMovie:
    """검색 결과 후보 영화 fixture."""
    return CandidateMovie(
        id="157336",
        title="인터스텔라",
        title_en="Interstellar",
        genres=["SF", "드라마", "모험"],
        director="크리스토퍼 놀란",
        cast=["매튜 매커너히", "앤 해서웨이", "제시카 차스테인"],
        rating=8.7,
        release_year=2014,
        overview="세계 각국의 정부와 경제가 완전히 붕괴된 미래가 다가온다.",
        mood_tags=["웅장", "감동", "몰입"],
        poster_path="/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
        ott_platforms=["넷플릭스"],
        certification="12세 이상 관람가",
        trailer_url="https://www.youtube.com/watch?v=LEEtpP6dkIY",
        rrf_score=0.95,
    )


@pytest.fixture
def sample_ranked_movie(sample_candidate_movie: CandidateMovie) -> RankedMovie:
    """최종 추천 영화 fixture (CandidateMovie 기반)."""
    return RankedMovie(
        id=sample_candidate_movie.id,
        title=sample_candidate_movie.title,
        title_en=sample_candidate_movie.title_en,
        genres=sample_candidate_movie.genres,
        director=sample_candidate_movie.director,
        cast=sample_candidate_movie.cast,
        rating=sample_candidate_movie.rating,
        release_year=sample_candidate_movie.release_year,
        overview=sample_candidate_movie.overview,
        mood_tags=sample_candidate_movie.mood_tags,
        poster_path=sample_candidate_movie.poster_path,
        ott_platforms=sample_candidate_movie.ott_platforms,
        certification=sample_candidate_movie.certification,
        trailer_url=sample_candidate_movie.trailer_url,
        rank=1,
        score_detail=ScoreDetail(
            cf_score=0.3,
            cbf_score=0.85,
            hybrid_score=0.72,
            genre_match=0.9,
            mood_match=0.8,
            similar_to=["인셉션", "그래비티"],
        ),
        explanation="",
    )


def _create_mock_llm_response(content: str) -> MagicMock:
    """Mock LLM 응답 BaseMessage를 생성한다."""
    msg = MagicMock()
    msg.content = content
    return msg


@pytest.fixture
def mock_ollama():
    """
    ChatOllama(Ollama)를 패치하여 Ollama 서버 없이 테스트한다.

    사용법:
        def test_something(mock_ollama):
            mock_ollama.set_response("응답 텍스트")
            # 또는
            mock_ollama.set_structured_response(IntentResult(intent="recommend", confidence=0.95))

    내부적으로 ChatOllama 생성자, ainvoke, with_structured_output을 모두 패치한다.
    """

    class MockOllamaController:
        """Mock ChatOllama(Ollama) 컨트롤러."""

        def __init__(self):
            self._response_content = "mock response"
            self._structured_response = None
            self._error = None
            self._response_queue: list = []  # 순차 응답 큐
            self._structured_queue: list = []  # 순차 구조화 응답 큐
            self._mock_instance = MagicMock()
            self._mock_structured = MagicMock()

            # ainvoke mock 설정
            self._mock_instance.ainvoke = AsyncMock(
                side_effect=self._get_response,
            )
            # with_structured_output mock 설정
            self._mock_structured.ainvoke = AsyncMock(
                side_effect=self._get_structured_response,
            )
            self._mock_instance.with_structured_output = MagicMock(
                return_value=self._mock_structured,
            )

        async def _get_response(self, *args, **kwargs):
            if self._error:
                raise self._error
            # 큐에 응답이 있으면 순차 반환
            if self._response_queue:
                content = self._response_queue.pop(0)
                return _create_mock_llm_response(content)
            return _create_mock_llm_response(self._response_content)

        async def _get_structured_response(self, *args, **kwargs):
            if self._error:
                raise self._error
            # 큐에 구조화 응답이 있으면 순차 반환
            if self._structured_queue:
                return self._structured_queue.pop(0)
            if self._structured_response is not None:
                return self._structured_response
            return _create_mock_llm_response(self._response_content)

        def set_response(self, content: str):
            """자유 텍스트 응답을 설정한다."""
            self._response_content = content
            self._error = None

        def set_structured_response(self, response):
            """구조화 출력 응답을 설정한다 (Pydantic 모델 인스턴스)."""
            self._structured_response = response
            self._error = None

        def set_error(self, error: Exception):
            """에러를 설정한다 (에러 복원력 테스트용)."""
            self._error = error
            self._structured_response = None

        def set_response_sequence(self, responses: list[str]):
            """순차 응답 목록을 설정한다 (호출 순서대로 반환)."""
            self._response_queue = list(responses)
            self._error = None

        def set_structured_sequence(self, responses: list):
            """순차 구조화 응답 목록을 설정한다 (호출 순서대로 반환)."""
            self._structured_queue = list(responses)
            self._error = None

    controller = MockOllamaController()

    # ChatOllama + ChatOpenAI(Solar API) 생성자를 모두 패치하여 서버 불필요
    # LLM_MODE=local_only, VLLM_ENABLED=False로 강제하여 외부 API 호출 방지
    import monglepick.llm.factory as factory_module
    from monglepick.config import settings as _settings

    with (
        patch("monglepick.llm.factory.ChatOllama", return_value=controller._mock_instance),
        patch("monglepick.llm.factory.ChatOpenAI", return_value=controller._mock_instance),
        patch.object(_settings, "LLM_MODE", "local_only"),
        patch.object(_settings, "VLLM_ENABLED", False),
    ):
        # LLM 캐시 초기화 (테스트 간 격리)
        factory_module._ollama_cache.clear()
        factory_module._solar_cache.clear()
        factory_module._vllm_cache.clear()
        factory_module._structured_cache.clear()

        yield controller

        # 테스트 후 캐시 정리
        factory_module._ollama_cache.clear()
        factory_module._solar_cache.clear()
        factory_module._vllm_cache.clear()
        factory_module._structured_cache.clear()


# ============================================================
# Phase 3 추가 Fixture: MySQL Mock
# ============================================================

@pytest.fixture
def mock_mysql():
    """
    aiomysql Pool/Connection/Cursor를 mock하여 DB 서버 없이 테스트한다.

    기본 동작:
    - fetchone() → None (유저 없음)
    - fetchall() → [] (시청 이력 없음)

    사용법:
        def test_something(mock_mysql):
            mock_mysql.set_user({"user_id": "test", "nickname": "테스트"})
            mock_mysql.set_watch_history([{"movie_id": "1", "title": "인셉션"}])
    """

    class MockMySQLController:
        """Mock MySQL 컨트롤러."""

        def __init__(self):
            self._user_row = None
            self._watch_history_rows = []
            self._call_count = 0  # execute 호출 카운터

        def set_user(self, row: dict | None):
            """유저 프로필 응답을 설정한다."""
            self._user_row = row

        def set_watch_history(self, rows: list[dict]):
            """시청 이력 응답을 설정한다."""
            self._watch_history_rows = rows

    controller = MockMySQLController()

    # cursor mock 생성
    mock_cursor = AsyncMock()

    async def mock_execute(query, params=None):
        """SQL 쿼리를 기반으로 적절한 mock 결과를 설정한다."""
        controller._call_count += 1

    mock_cursor.execute = AsyncMock(side_effect=mock_execute)

    async def mock_fetchone():
        """첫 번째 쿼리 → 유저 프로필 반환."""
        return controller._user_row

    async def mock_fetchall():
        """두 번째 쿼리 → 시청 이력 반환."""
        return controller._watch_history_rows

    mock_cursor.fetchone = AsyncMock(side_effect=mock_fetchone)
    mock_cursor.fetchall = AsyncMock(side_effect=mock_fetchall)

    # cursor를 async context manager로 구성
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=False)

    # connection mock 생성
    mock_conn = AsyncMock()
    mock_conn.cursor = MagicMock(return_value=mock_cursor)

    # connection을 async context manager로 구성
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    # pool mock 생성
    mock_pool = AsyncMock()
    mock_pool.acquire = MagicMock(return_value=mock_conn)

    # get_mysql 패치
    with patch("monglepick.agents.chat.nodes.get_mysql", return_value=mock_pool):
        yield controller


# ============================================================
# Phase 3 추가 Fixture: Hybrid Search Mock
# ============================================================

@pytest.fixture
def mock_hybrid_search():
    """
    hybrid_search 함수를 mock하여 DB 서버 없이 검색 테스트한다.

    기본 동작: 빈 결과 반환

    사용법:
        def test_something(mock_hybrid_search):
            from monglepick.rag.hybrid_search import SearchResult
            mock_hybrid_search.set_results([
                SearchResult(movie_id="1", title="인셉션", score=0.9, source="rrf", metadata={...})
            ])
    """
    from monglepick.rag.hybrid_search import SearchResult

    class MockHybridSearchController:
        """Mock 하이브리드 검색 컨트롤러."""

        def __init__(self):
            self._results: list[SearchResult] = []

        def set_results(self, results: list[SearchResult]):
            """검색 결과를 설정한다."""
            self._results = results

    controller = MockHybridSearchController()

    async def mock_search(*args, **kwargs):
        return controller._results

    with patch("monglepick.agents.chat.nodes.hybrid_search", side_effect=mock_search):
        yield controller


# ============================================================
# Phase 4 추가 Fixture: Redis CF 캐시 Mock
# ============================================================

@pytest.fixture
def mock_redis_cf():
    """
    Redis CF 캐시를 mock하여 Redis 서버 없이 추천 엔진을 테스트한다.

    get_redis()를 패치하여 인메모리 딕셔너리 기반 mock을 반환한다.

    사용법:
        def test_something(mock_redis_cf):
            mock_redis_cf.set_similar_users("user1", [("user2", 0.9), ("user3", 0.8)])
            mock_redis_cf.set_user_ratings("user2", {"movie1": "4.5", "movie2": "3.0"})
    """

    class MockRedisCFController:
        """Mock Redis CF 캐시 컨트롤러."""

        def __init__(self):
            # Sorted Set: {key: {member: score}}
            self._sorted_sets: dict[str, dict[str, float]] = {}
            # Hash: {key: {field: value}}
            self._hashes: dict[str, dict[str, str]] = {}

        def set_similar_users(
            self,
            user_id: str,
            similar_users: list[tuple[str, float]],
        ):
            """유사 유저 Sorted Set을 설정한다."""
            key = f"cf:similar_users:{user_id}"
            self._sorted_sets[key] = {uid: score for uid, score in similar_users}

        def set_user_ratings(
            self,
            user_id: str,
            ratings: dict[str, str],
        ):
            """유저별 평점 Hash를 설정한다."""
            key = f"cf:user_ratings:{user_id}"
            self._hashes[key] = ratings

    controller = MockRedisCFController()

    # Redis mock 객체 생성
    mock_redis = AsyncMock()

    # zrevrangebyscore: Sorted Set에서 내림차순 조회
    async def mock_zrevrangebyscore(key, **kwargs):
        data = controller._sorted_sets.get(key, {})
        if not data:
            return []
        # score 내림차순 정렬하여 (member, score) 튜플 반환
        sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
        num = kwargs.get("num", 50)
        return sorted_items[:num]

    mock_redis.zrevrangebyscore = AsyncMock(side_effect=mock_zrevrangebyscore)

    # hgetall: Hash 전체 조회
    async def mock_hgetall(key):
        return controller._hashes.get(key, {})

    mock_redis.hgetall = AsyncMock(side_effect=mock_hgetall)

    # pipeline: 배치 조회 지원
    class MockPipeline:
        """Mock Redis pipeline."""

        def __init__(self):
            self._commands: list[tuple[str, tuple]] = []

        def hgetall(self, key):
            self._commands.append(("hgetall", (key,)))

        async def execute(self):
            results = []
            for cmd, args in self._commands:
                if cmd == "hgetall":
                    results.append(controller._hashes.get(args[0], {}))
            self._commands.clear()
            return results

    def mock_pipeline():
        return MockPipeline()

    mock_redis.pipeline = MagicMock(side_effect=mock_pipeline)

    # get_redis 패치 (recommendation nodes에서 사용)
    with patch(
        "monglepick.agents.recommendation.nodes.get_redis",
        return_value=mock_redis,
    ):
        yield controller


# ============================================================
# 참조 영화 DB 조회 Mock Fixture
# ============================================================

@pytest.fixture
def mock_reference_lookup():
    """
    _lookup_reference_movie_info 함수를 mock하여 ES 서버 없이 참조 영화 조회를 테스트한다.

    기본 동작: 인터스텔라의 장르(SF, 드라마, 모험)와 무드태그(웅장, 감동, 몰입) 반환

    사용법:
        def test_something(mock_reference_lookup):
            mock_reference_lookup.set_result({"genres": ["로맨스"], "mood_tags": ["로맨틱"]})
    """

    class MockReferenceLookupController:
        """Mock 참조 영화 DB 조회 컨트롤러."""

        def __init__(self):
            # 기본값: 인터스텔라 기준
            self._result: dict[str, list[str]] = {
                "genres": ["SF", "드라마", "모험"],
                "mood_tags": ["웅장", "감동", "몰입"],
            }

        def set_result(self, result: dict[str, list[str]]):
            """조회 결과를 설정한다."""
            self._result = result

        def set_empty(self):
            """빈 결과를 설정한다 (DB에 영화 없음)."""
            self._result = {"genres": [], "mood_tags": []}

    controller = MockReferenceLookupController()

    async def mock_lookup(movie_titles: list[str]) -> dict[str, list[str]]:
        return controller._result

    with patch(
        "monglepick.agents.chat.nodes._lookup_reference_movie_info",
        side_effect=mock_lookup,
    ):
        yield controller


# ============================================================
# 이미지 분석 Mock Fixture
# ============================================================

@pytest.fixture
def mock_image_analysis():
    """
    analyze_image 함수를 mock하여 VLM 서버 없이 이미지 분석을 테스트한다.

    기본 동작: SF 장르 + 웅장 무드 + 영화 포스터 분석 결과 반환

    사용법:
        def test_something(mock_image_analysis):
            mock_image_analysis.set_result(ImageAnalysisResult(
                genre_cues=["로맨스"],
                mood_cues=["로맨틱"],
                analyzed=True,
            ))
    """
    from monglepick.agents.chat.models import ImageAnalysisResult

    class MockImageAnalysisController:
        """Mock 이미지 분석 컨트롤러."""

        def __init__(self):
            # 기본값: SF 영화 포스터 분석 결과
            self._result = ImageAnalysisResult(
                genre_cues=["SF", "모험"],
                mood_cues=["웅장", "몰입"],
                visual_elements=["우주선", "별", "행성"],
                search_keywords=["우주", "탐험"],
                description="우주를 배경으로 한 SF 영화 포스터입니다.",
                is_movie_poster=True,
                detected_movie_title="인터스텔라",
                analyzed=True,
            )

        def set_result(self, result: ImageAnalysisResult):
            """분석 결과를 설정한다."""
            self._result = result

    controller = MockImageAnalysisController()

    async def mock_analyze(*args, **kwargs):
        return controller._result

    with patch("monglepick.agents.chat.nodes.analyze_image", side_effect=mock_analyze):
        yield controller


# ============================================================
# 의도+감정 통합 Mock Fixture
# ============================================================

@pytest.fixture
def mock_intent_emotion():
    """
    classify_intent_and_emotion 함수를 mock하여 vLLM 서버 없이 통합 분류를 테스트한다.

    기본 동작: recommend 의도 + happy 감정 + 유쾌 무드 태그

    사용법:
        def test_something(mock_intent_emotion):
            from monglepick.agents.chat.models import IntentEmotionResult
            mock_intent_emotion.set_result(IntentEmotionResult(
                intent="general", confidence=0.8,
                emotion=None, mood_tags=[],
            ))
    """
    from monglepick.agents.chat.models import IntentEmotionResult

    class MockIntentEmotionController:
        """Mock 의도+감정 통합 컨트롤러."""

        def __init__(self):
            # 기본값: 추천 의도 + happy 감정
            self._result = IntentEmotionResult(
                intent="recommend",
                confidence=0.9,
                emotion="happy",
                mood_tags=["유쾌", "따뜻"],
            )

        def set_result(self, result: IntentEmotionResult):
            """분류 결과를 설정한다."""
            self._result = result

    controller = MockIntentEmotionController()

    async def mock_classify(*args, **kwargs):
        return controller._result

    with patch(
        "monglepick.agents.chat.nodes.classify_intent_and_emotion",
        side_effect=mock_classify,
    ):
        yield controller


# ============================================================
# 세션 저장소 Mock Fixture
# ============================================================

@pytest.fixture
def mock_session_store():
    """
    Redis 세션 저장소를 mock하여 Redis 서버 없이 세션 영속화를 테스트한다.

    기본 동작:
    - load_session() → None (신규 세션)
    - save_session() → no-op

    사용법:
        def test_something(mock_session_store):
            mock_session_store.set_session({
                "messages": [{"role": "user", "content": "안녕"}],
                "preferences": None,
                "emotion": None,
                "turn_count": 1,
                "user_profile": {},
                "watch_history": [],
            })
    """

    class MockSessionStoreController:
        """Mock 세션 저장소 컨트롤러."""

        def __init__(self):
            self._session_data: dict | None = None
            self._saved_sessions: dict[str, dict] = {}

        def set_session(self, data: dict | None):
            """세션 로드 시 반환할 데이터를 설정한다."""
            self._session_data = data

        def get_saved(self, session_id: str) -> dict | None:
            """저장된 세션 데이터를 조회한다 (테스트 검증용)."""
            return self._saved_sessions.get(session_id)

    controller = MockSessionStoreController()

    async def mock_load(user_id: str, session_id: str):
        return controller._session_data

    async def mock_save(user_id: str, session_id: str, state: dict):
        # Pydantic 모델을 dict로 변환하여 저장
        save_data = {}
        for key, val in state.items():
            if hasattr(val, "model_dump"):
                save_data[key] = val.model_dump()
            else:
                save_data[key] = val
        controller._saved_sessions[session_id] = save_data

    with patch(
        "monglepick.agents.chat.graph.load_session",
        side_effect=mock_load,
    ), patch(
        "monglepick.agents.chat.graph.save_session",
        side_effect=mock_save,
    ):
        yield controller
