"""
Chat Agent 노드 함수 (§6-2 Node 1~11 + image_analyzer + general_responder + tool_executor_node).

LangGraph StateGraph의 각 노드로 등록되는 13개 async 함수.
시그니처: async def node_name(state: ChatAgentState) -> dict

모든 노드는 try/except로 감싸고, 에러 시 유효한 기본값을 반환한다 (에러 전파 금지).
반환값은 dict — LangGraph 컨벤션 (TypedDict State 일부 업데이트).

노드 목록:
1. context_loader              — 유저 프로필/시청이력/대화이력 로드 (MySQL)
2. image_analyzer              — 이미지 분석 (VLM, 이미지 없으면 패스스루)
3. intent_emotion_classifier   — 의도+감정 통합 분류 (1회 LLM) + 이미지 부스트
4. preference_refiner          — 선호 추출 + 이미지 보강 + 누적 병합 + 충분성 판정
5. question_generator          — 부족 정보 후속 질문 생성 + 구조화 힌트 + 검색 피드백
6. query_builder               — RAG 검색 쿼리 구성 (규칙 기반, LLM 없음) + 이미지 키워드
7. rag_retriever               — 하이브리드 검색 (Qdrant+ES+Neo4j RRF)
8. recommendation_ranker       — 추천 순위 정렬 (Phase 4 서브그래프)
9. explanation_generator       — 영화별 추천 이유 생성
10. response_formatter         — 응답 포맷팅 (추천/질문/일반/에러)
11. error_handler              — 에러 처리 + 친절한 안내 메시지
12. general_responder          — 일반 대화 응답 (몽글 페르소나)
13. tool_executor_node         — 도구 실행 
"""

from __future__ import annotations

import re
import time
import traceback
from typing import Any

import aiomysql
import structlog
from langsmith import traceable

from monglepick.agents.chat.models import (
    FIELD_HINTS,
    CandidateMovie,
    ChatAgentState,
    ClarificationHint,
    ClarificationResponse,
    EmotionResult,
    ExtractedPreferences,
    ImageAnalysisResult,
    IntentResult,
    RankedMovie,
    ScoreDetail,
    SearchQuery,
    is_sufficient,
)
from monglepick.chains import (
    analyze_image,
    classify_intent_and_emotion,
    extract_preferences,
    generate_explanations_batch,
    generate_general_response,
    generate_question,
)
from monglepick.chains.question_chain import _get_missing_fields
from monglepick.db.clients import ES_INDEX_NAME, get_elasticsearch, get_mysql
from monglepick.rag.hybrid_search import SearchResult, hybrid_search

logger = structlog.get_logger()


# ============================================================
# 1. context_loader — 유저 프로필/시청이력/대화이력 로드
# ============================================================

@traceable(name="context_loader", run_type="chain", metadata={"node": "1/13", "db": "mysql"})
async def context_loader(state: ChatAgentState) -> dict:
    """
    MySQL에서 유저 프로필과 시청 이력을 로드하고, 메시지 리스트를 구성한다.

    세션 캐싱 최적화:
    - 세션에서 user_profile/watch_history가 이미 로드되어 있으면 MySQL 쿼리를 스킵한다.
    - 2턴 이후에는 MySQL 쿼리 0회 (세션에서 캐싱된 프로필/시청이력 재사용).

    - user_id가 비어있으면(익명 사용자) 빈 기본값을 반환한다.
    - messages에 현재 입력을 user 메시지로 추가한다.
    - turn_count는 기존 user 메시지 수 + 1로 계산한다.

    Args:
        state: ChatAgentState (user_id, session_id, current_input 필수)

    Returns:
        dict: user_profile, watch_history, messages, turn_count 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        current_input = state.get("current_input", "")

        # 기존 메시지 복사 + 현재 입력 추가
        messages: list[dict[str, str]] = list(state.get("messages", []))
        messages.append({"role": "user", "content": current_input})

        # 턴 카운트: user 메시지 수
        turn_count = sum(1 for m in messages if m.get("role") == "user")

        # 익명 사용자: 빈 기본값
        if not user_id:
            logger.info("context_loader_anonymous")
            return {
                "user_profile": state.get("user_profile", {}),
                "watch_history": state.get("watch_history", []),
                "messages": messages,
                "turn_count": turn_count,
            }

        # 세션에서 이미 프로필이 로드되어 있으면 MySQL 조회 스킵
        existing_profile = state.get("user_profile", {})
        existing_history = state.get("watch_history", [])

        if existing_profile:
            # 2턴 이후: 세션에서 캐싱된 프로필/시청이력 재사용 → MySQL 0회
            elapsed_ms = (time.perf_counter() - node_start) * 1000
            logger.info(
                "context_loaded_from_session",
                user_id=user_id,
                history_count=len(existing_history),
                turn_count=turn_count,
                elapsed_ms=round(elapsed_ms, 1),
                session_id=session_id,
            )
            return {
                "user_profile": existing_profile,
                "watch_history": existing_history,
                "messages": messages,
                "turn_count": turn_count,
            }

        # 첫 턴: MySQL에서 유저 프로필 + 시청 이력 로드
        user_profile: dict[str, Any] = {}
        watch_history: list[dict[str, Any]] = []

        try:
            pool = await get_mysql()
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    # 유저 프로필 조회 — 필요 컬럼만 명시 (W-4: password_hash 노출 방지)
                    await cursor.execute(
                        "SELECT user_id, nickname, email, profile_image, user_role, user_birth FROM users WHERE user_id = %s LIMIT 1",
                        (user_id,),
                    )
                    row = await cursor.fetchone()
                    if row:
                        user_profile = dict(row)

                    # 시청 이력 조회 (최근 50건, 영화 제목 포함)
                    await cursor.execute(
                        """
                        SELECT wh.movie_id, m.title, wh.rating, wh.watched_at
                        FROM watch_history wh
                        LEFT JOIN movies m ON wh.movie_id = m.movie_id
                        WHERE wh.user_id = %s
                        ORDER BY wh.watched_at DESC
                        LIMIT 50
                        """,
                        (user_id,),
                    )
                    rows = await cursor.fetchall()
                    watch_history = [dict(r) for r in rows]
        except Exception as db_err:
            # DB 에러 시에도 빈 기본값으로 계속 진행
            logger.warning("context_loader_db_error", error=str(db_err))

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "context_loaded",
            user_id=user_id,
            profile_exists=bool(user_profile),
            history_count=len(watch_history),
            turn_count=turn_count,
            recent_watched=[wh.get("title", "") for wh in watch_history[:5]],
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
        )

        return {
            "user_profile": user_profile,
            "watch_history": watch_history,
            "messages": messages,
            "turn_count": turn_count,
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("context_loader_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        return {
            "user_profile": {},
            "watch_history": [],
            "messages": [{"role": "user", "content": state.get("current_input", "")}],
            "turn_count": 1,
            "error": str(e),
        }


# ============================================================
# 2. image_analyzer — 이미지 분석 (VLM)
# ============================================================

@traceable(name="image_analyzer", run_type="chain", metadata={"node": "2/13", "llm": "qwen3.5:35b-a3b"})
async def image_analyzer(state: ChatAgentState) -> dict:
    """
    사용자가 업로드한 이미지를 VLM으로 분석한다.

    이 노드는 conditional edge(route_has_image)에 의해 image_data가 있을 때만 실행된다.
    analyze_image() 체인을 호출하여 장르/무드/시각요소 등을 추출한다.

    Args:
        state: ChatAgentState (image_data, current_input 필요)

    Returns:
        dict: image_analysis(ImageAnalysisResult) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        image_data = state.get("image_data", "")
        current_input = state.get("current_input", "")

        logger.info(
            "image_analyzer_started",
            image_data_length=len(image_data),
            has_user_message=bool(current_input),
        )

        # VLM 이미지 분석 체인 호출
        result = await analyze_image(
            image_data=image_data,
            current_input=current_input,
        )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "image_analyzer_completed",
            analyzed=result.analyzed,
            genre_cues=result.genre_cues,
            mood_cues=result.mood_cues,
            is_poster=result.is_movie_poster,
            detected_title=result.detected_movie_title,
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {"image_analysis": result}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("image_analyzer_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        return {"image_analysis": ImageAnalysisResult(analyzed=False)}


# ============================================================
# 3. intent_emotion_classifier — 의도+감정 통합 분류 (1회 LLM)
# ============================================================

@traceable(name="intent_emotion_classifier", run_type="chain", metadata={"node": "3/13", "llm": "qwen3.5:35b-a3b"})
async def intent_emotion_classifier(state: ChatAgentState) -> dict:
    """
    사용자 메시지의 의도와 감정을 동시에 분류한다 (1회 LLM 호출).

    기존 intent_classifier + emotion_analyzer 2노드를 통합하여
    동일 모델(qwen3.5:35b-a3b)로 동일 입력을 1번만 분석한다.
    결과를 IntentResult + EmotionResult로 분해하여 state에 기록한다.

    이미지 부스트: 이미지 분석 결과가 있고 intent가 general이면 → recommend로 부스트.
    (이미지를 업로드한 사용자는 추천 의도가 높다고 간주)

    Args:
        state: ChatAgentState (current_input, messages, image_analysis 필요)

    Returns:
        dict: intent(IntentResult), emotion(EmotionResult) 동시 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        current_input = state.get("current_input", "")
        messages = state.get("messages", [])
        image_analysis = state.get("image_analysis")

        # 최근 6개 메시지를 포맷 (현재 입력 제외)
        recent = messages[-7:-1] if len(messages) > 1 else []
        recent_messages = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in recent[-6:]
        )

        # 통합 체인 호출 (1회 LLM)
        result = await classify_intent_and_emotion(
            current_input=current_input,
            recent_messages=recent_messages,
        )

        # IntentResult + EmotionResult로 분해
        intent_result = IntentResult(
            intent=result.intent,
            confidence=result.confidence,
        )
        emotion_result = EmotionResult(
            emotion=result.emotion,
            mood_tags=result.mood_tags,
        )

        # 이미지 부스트: 이미지 분석 결과가 있고 intent가 general이면 recommend로 부스트
        if (
            image_analysis is not None
            and image_analysis.analyzed
            and intent_result.intent == "general"
        ):
            logger.info(
                "intent_image_boost",
                original_intent=intent_result.intent,
                boosted_to="recommend",
                image_genre_cues=image_analysis.genre_cues,
            )
            intent_result = IntentResult(
                intent="recommend",
                confidence=max(intent_result.confidence, 0.7),
            )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "intent_emotion_classified_node",
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            emotion=emotion_result.emotion,
            mood_tags=emotion_result.mood_tags,
            image_boosted=bool(image_analysis and image_analysis.analyzed),
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {
            "intent": intent_result,
            "emotion": emotion_result,
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("intent_emotion_classifier_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        return {
            "intent": IntentResult(intent="general", confidence=0.0),
            "emotion": EmotionResult(emotion=None, mood_tags=[]),
        }


# ============================================================
# 참조 영화 DB 조회 헬퍼 (preference_refiner에서 사용)
# ============================================================

async def _lookup_reference_movie_info(movie_titles: list[str]) -> dict[str, list[str]]:
    """
    참조 영화 제목으로 Elasticsearch를 검색하여 장르/무드태그 정보를 조회한다.

    사용자가 "인터스텔라 같은 영화"라고 하면, 인터스텔라의 장르(SF, 모험, 드라마)와
    무드태그(몰입, 웅장 등)를 DB에서 가져와 선호 조건을 자동 보강한다.
    이를 통해 reference_movies만으로도 충분성 임계값(3.0)을 넘길 수 있다.

    Args:
        movie_titles: 참조 영화 제목 리스트 (예: ["인터스텔라"])

    Returns:
        {"genres": [...], "mood_tags": [...]} — 합산된 장르/무드 정보 (중복 제거)
    """
    try:
        es = await get_elasticsearch()
    except Exception:
        return {"genres": [], "mood_tags": []}

    all_genres: list[str] = []
    all_mood_tags: list[str] = []

    # 최대 3개 영화만 조회 (과다 조회 방지)
    for title in movie_titles[:3]:
        try:
            resp = await es.search(
                index=ES_INDEX_NAME,
                body={
                    "query": {
                        "match": {
                            "title": {
                                "query": title,
                                "analyzer": "korean_analyzer",
                            }
                        }
                    },
                    "size": 1,
                },
            )
            hits = resp["hits"]["hits"]
            if hits:
                source = hits[0]["_source"]
                genres = source.get("genres", [])
                mood_tags = source.get("mood_tags", [])
                if isinstance(genres, list):
                    all_genres.extend(genres)
                if isinstance(mood_tags, list):
                    all_mood_tags.extend(mood_tags)
                logger.info(
                    "reference_movie_lookup_hit",
                    query_title=title,
                    matched_title=source.get("title", ""),
                    genres=genres,
                    mood_tags=mood_tags[:5] if mood_tags else [],
                )
            else:
                logger.info("reference_movie_lookup_miss", query_title=title)
        except Exception as e:
            logger.warning("reference_movie_lookup_error", query_title=title, error=str(e))
            continue

    # 중복 제거 (순서 유지)
    return {
        "genres": list(dict.fromkeys(all_genres)),
        "mood_tags": list(dict.fromkeys(all_mood_tags)),
    }


# ============================================================
# 4. preference_refiner — 선호 추출 + 충분성 판정
# ============================================================

@traceable(name="preference_refiner", run_type="chain", metadata={"node": "4/13", "llm": "exaone-32b"})
async def preference_refiner(state: ChatAgentState) -> dict:
    """
    사용자 메시지에서 선호 조건을 추출하고, 이전 선호와 병합한 후 충분성을 판정한다.

    - extract_preferences 체인이 추출 + 병합을 수행한다.
    - 이미지 분석 결과가 있으면 genre_cues→genre_preference, mood_cues→mood,
      detected_movie_title→reference_movies에 보강한다.
    - is_sufficient()로 가중치 합산 ≥ 3.0 또는 turn_count ≥ 3 판정.
    - needs_clarification=True면 후속 질문 필요, False면 추천 진행.

    Args:
        state: ChatAgentState (current_input, preferences, emotion, turn_count,
               image_analysis 필요)

    Returns:
        dict: preferences(ExtractedPreferences), needs_clarification(bool) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        current_input = state.get("current_input", "")
        prev_prefs = state.get("preferences")
        emotion = state.get("emotion")
        turn_count = state.get("turn_count", 0)
        image_analysis = state.get("image_analysis")

        # 선호 추출 + 병합
        merged = await extract_preferences(
            current_input=current_input,
            previous_preferences=prev_prefs,
        )

        # 이미지 분석 결과로 선호 조건 보강
        has_image = False
        if image_analysis is not None and image_analysis.analyzed:
            has_image = True
            # genre_cues → genre_preference 보강 (기존 선호가 없을 때만)
            if not merged.genre_preference and image_analysis.genre_cues:
                merged = merged.model_copy(
                    update={"genre_preference": ", ".join(image_analysis.genre_cues[:3])}
                )
                logger.info(
                    "preference_image_genre_boost",
                    genre_cues=image_analysis.genre_cues,
                )
            # mood_cues → mood 보강 (기존 무드가 없을 때만)
            if not merged.mood and image_analysis.mood_cues:
                merged = merged.model_copy(
                    update={"mood": ", ".join(image_analysis.mood_cues[:3])}
                )
                logger.info(
                    "preference_image_mood_boost",
                    mood_cues=image_analysis.mood_cues,
                )
            # detected_movie_title → reference_movies에 추가
            if image_analysis.detected_movie_title:
                ref_movies = list(merged.reference_movies)
                if image_analysis.detected_movie_title not in ref_movies:
                    ref_movies.append(image_analysis.detected_movie_title)
                    merged = merged.model_copy(update={"reference_movies": ref_movies})
                    logger.info(
                        "preference_image_reference_boost",
                        detected_title=image_analysis.detected_movie_title,
                    )

        # ── 참조 영화 DB 조회로 선호 조건 자동 보강 ──
        # "인터스텔라 같은 영화"처럼 참조 영화가 있으면 해당 영화의 장르/무드를
        # DB에서 조회하여 빈 필드를 채운다. 이를 통해 reference_movies(1.5) +
        # genre(2.0) + mood(2.0) = 5.5 ≥ 3.0으로 바로 추천 진행이 가능해진다.
        if merged.reference_movies and (not merged.genre_preference or not merged.mood):
            ref_info = await _lookup_reference_movie_info(merged.reference_movies)
            if ref_info["genres"] and not merged.genre_preference:
                merged = merged.model_copy(
                    update={"genre_preference": ", ".join(ref_info["genres"][:5])}
                )
                logger.info(
                    "preference_reference_genre_enriched",
                    reference_movies=merged.reference_movies,
                    enriched_genres=ref_info["genres"][:5],
                )
            if ref_info["mood_tags"] and not merged.mood:
                merged = merged.model_copy(
                    update={"mood": ", ".join(ref_info["mood_tags"][:3])}
                )
                logger.info(
                    "preference_reference_mood_enriched",
                    reference_movies=merged.reference_movies,
                    enriched_moods=ref_info["mood_tags"][:3],
                )

        # 감정 존재 여부 확인 (무드 가중치 부여용)
        has_emotion = emotion is not None and emotion.emotion is not None

        # 충분성 판정 (이미지 분석 시 +1.5 보너스)
        sufficient = is_sufficient(
            prefs=merged,
            turn_count=turn_count,
            has_emotion=has_emotion,
            has_image_analysis=has_image,
        )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "preference_refined_node",
            needs_clarification=not sufficient,
            turn_count=turn_count,
            genre=merged.genre_preference,
            mood=merged.mood,
            has_image_analysis=has_image,
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {
            "preferences": merged,
            "needs_clarification": not sufficient,
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("preference_refiner_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        return {
            "preferences": state.get("preferences", ExtractedPreferences()),
            "needs_clarification": True,
        }


# ============================================================
# 5. question_generator — 후속 질문 생성
# ============================================================

@traceable(name="question_generator", run_type="chain", metadata={"node": "5/13", "llm": "exaone-32b"})
async def question_generator(state: ChatAgentState) -> dict:
    """
    부족한 선호 정보를 파악하기 위한 후속 질문을 생성한다.

    needs_clarification=True 또는 검색 품질 미달 시 호출된다.
    response 필드에도 질문 텍스트를 설정하여 response_formatter에서 바로 사용한다.
    구조화된 힌트(ClarificationResponse)를 함께 반환하여 UI에서 칩/버튼으로 표시한다.

    검색 품질 미달로 호출된 경우(retrieval_feedback 존재 시):
    - 피드백 메시지를 질문에 포함하여 사용자에게 안내한다.
    - 검색된 후보의 장르 분포를 참고하여 힌트를 구성한다.

    Args:
        state: ChatAgentState (preferences, emotion, turn_count, retrieval_feedback 필요)

    Returns:
        dict: follow_up_question, response, clarification 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        prefs = state.get("preferences", ExtractedPreferences())
        emotion = state.get("emotion")
        turn_count = state.get("turn_count", 0)
        retrieval_feedback = state.get("retrieval_feedback", "")

        emotion_str = emotion.emotion if emotion else None

        # 검색 품질 미달로 호출된 경우: 피드백 메시지 포함
        if retrieval_feedback:
            question = (
                f"{retrieval_feedback} "
                "좀 더 구체적으로 알려주시면 더 좋은 영화를 찾아드릴 수 있어요!"
            )
        else:
            question = await generate_question(
                extracted_preferences=prefs,
                emotion=emotion_str,
                turn_count=turn_count,
            )

        # ── 구조화된 힌트 구성 (부족 필드 상위 3개) ──
        missing_fields = _get_missing_fields(prefs)
        hints: list[ClarificationHint] = []
        for field_name, _weight in missing_fields[:3]:
            hint_info = FIELD_HINTS.get(field_name)
            if hint_info:
                hints.append(ClarificationHint(
                    field=field_name,
                    label=hint_info["label"],
                    options=hint_info["options"],
                ))

        # primary_field: 가장 중요한 부족 필드
        primary_field = missing_fields[0][0] if missing_fields else ""

        clarification = ClarificationResponse(
            question=question,
            hints=hints,
            primary_field=primary_field,
        )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "question_generated_node",
            question_preview=question[:50],
            hint_count=len(hints),
            primary_field=primary_field,
            retrieval_feedback=bool(retrieval_feedback),
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {
            "follow_up_question": question,
            "response": question,
            "clarification": clarification,
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("question_generator_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        fallback = "어떤 영화를 찾으시는지 좀 더 알려주세요!"
        return {
            "follow_up_question": fallback,
            "response": fallback,
            "clarification": None,
        }


# ============================================================
# 6. query_builder — RAG 검색 쿼리 구성 (규칙 기반)
# ============================================================

def _parse_era(era: str) -> tuple[int, int] | None:
    """
    시대/연도 문자열을 (시작연도, 끝연도) 튜플로 변환한다.

    지원 포맷:
    - "2020년대" → (2020, 2029)
    - "90년대" → (1990, 1999)
    - "2020" → (2020, 2020)

    Args:
        era: 시대/연도 문자열

    Returns:
        (시작연도, 끝연도) 튜플, 파싱 실패 시 None
    """
    if not era:
        return None

    # "2020년대", "90년대" 패턴
    match = re.match(r"(\d{2,4})년대", era)
    if match:
        year_str = match.group(1)
        if len(year_str) == 2:
            # "10년대" → 2010, "90년대" → 1990 (W-2: 50 기준으로 세기 판별)
            year_val = int(year_str)
            base = (2000 if year_val < 50 else 1900) + year_val
        else:
            # "2020년대" → 2020
            base = int(year_str)
        return (base, base + 9)

    # 단순 연도 "2020"
    match = re.match(r"(\d{4})", era)
    if match:
        year = int(match.group(1))
        return (year, year)

    return None


@traceable(name="query_builder", run_type="chain", metadata={"node": "6/13", "llm": "none"})
async def query_builder(state: ChatAgentState) -> dict:
    """
    선호 조건과 감정/이미지 분석 결과를 기반으로 RAG 검색 쿼리를 구성한다.

    규칙 기반 (LLM 없음):
    - semantic_query: 사용자 입력 + 장르 + 무드 + 참조 영화 + 이미지 설명 결합
    - keyword_query: 사용자 원문 입력
    - filters: 장르, 무드태그, OTT, 연도 범위
    - boost_keywords: 무드태그 + 참조영화 + 이미지 키워드 + 시각 요소
    - exclude_ids: 시청 이력 영화 ID

    Args:
        state: ChatAgentState (current_input, preferences, emotion, watch_history,
               image_analysis 필요)

    Returns:
        dict: search_query(SearchQuery) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        current_input = state.get("current_input", "")
        prefs = state.get("preferences", ExtractedPreferences())
        emotion = state.get("emotion", EmotionResult())
        watch_history = state.get("watch_history", [])
        image_analysis = state.get("image_analysis")

        # semantic_query 구성: 사용자 입력 + 장르 + 무드 + 참조 영화
        query_parts = [current_input]
        if prefs.genre_preference:
            query_parts.append(prefs.genre_preference)
        if prefs.mood:
            query_parts.append(prefs.mood)
        if prefs.reference_movies:
            query_parts.append(" ".join(prefs.reference_movies))
        # 이미지 분석 결과의 description을 semantic_query에 추가
        if image_analysis and image_analysis.analyzed and image_analysis.description:
            query_parts.append(image_analysis.description)
        semantic_query = " ".join(query_parts)

        # filters 구성
        filters: dict[str, Any] = {}
        if prefs.genre_preference:
            # 쉼표나 공백으로 구분된 장르를 리스트로 변환
            genres = [g.strip() for g in re.split(r"[,\s]+", prefs.genre_preference) if g.strip()]
            if genres:
                filters["genres"] = genres
        if prefs.platform:
            filters["platform"] = prefs.platform

        # 연도 범위 파싱
        year_range = _parse_era(prefs.era) if prefs.era else None
        if year_range:
            filters["year_range"] = year_range

        # boost_keywords: 무드태그 + 참조영화 + 이미지 키워드
        boost_keywords: list[str] = []
        if emotion and emotion.mood_tags:
            boost_keywords.extend(emotion.mood_tags)
        if prefs.reference_movies:
            boost_keywords.extend(prefs.reference_movies)
        # 이미지 분석 결과의 search_keywords + visual_elements를 boost에 추가
        if image_analysis and image_analysis.analyzed:
            boost_keywords.extend(image_analysis.search_keywords)
            boost_keywords.extend(image_analysis.visual_elements[:3])

        # exclude_ids: 시청 이력 영화 ID
        exclude_ids = [str(wh.get("movie_id", "")) for wh in watch_history if wh.get("movie_id")]

        search_query = SearchQuery(
            semantic_query=semantic_query,
            keyword_query=current_input,
            filters=filters,
            boost_keywords=boost_keywords,
            exclude_ids=exclude_ids,
            limit=15,
        )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "query_built_node",
            semantic_query=semantic_query[:200],
            keyword_query=current_input[:100],
            filters=filters,
            boost_keywords=boost_keywords[:10],
            exclude_count=len(exclude_ids),
            image_enhanced=bool(image_analysis and image_analysis.analyzed),
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {"search_query": search_query}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("query_builder_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        # 최소한 사용자 입력으로 검색 쿼리 구성
        return {
            "search_query": SearchQuery(
                semantic_query=state.get("current_input", "영화 추천"),
                keyword_query=state.get("current_input", "영화 추천"),
            )
        }


# ============================================================
# 7. rag_retriever — 하이브리드 검색
# ============================================================

def _search_result_to_candidate(result: SearchResult, rank: int) -> CandidateMovie:
    """
    SearchResult를 CandidateMovie로 변환한다.

    메타데이터에서 영화 정보를 추출하여 CandidateMovie 필드를 채운다.

    Args:
        result: 하이브리드 검색 결과
        rank: 검색 순위 (0-based)

    Returns:
        CandidateMovie 인스턴스
    """
    meta = result.metadata or {}
    return CandidateMovie(
        id=result.movie_id,
        title=result.title or meta.get("title", ""),
        title_en=meta.get("title_en", ""),
        genres=meta.get("genres", []) if isinstance(meta.get("genres"), list) else [],
        director=meta.get("director", ""),
        cast=meta.get("cast", []) if isinstance(meta.get("cast"), list) else [],
        rating=float(meta.get("rating", 0.0) or 0.0),
        release_year=int(meta.get("release_year", 0) or 0),
        overview=meta.get("overview", ""),
        mood_tags=meta.get("mood_tags", []) if isinstance(meta.get("mood_tags"), list) else [],
        poster_path=meta.get("poster_path", ""),
        ott_platforms=meta.get("ott_platforms", []) if isinstance(meta.get("ott_platforms"), list) else [],
        certification=meta.get("certification", ""),
        trailer_url=meta.get("trailer_url", ""),
        rrf_score=result.score,
        retrieval_source=result.source,
    )


@traceable(name="rag_retriever", run_type="retriever", metadata={"node": "7/13", "fusion": "RRF"})
async def rag_retriever(state: ChatAgentState) -> dict:
    """
    하이브리드 검색(Qdrant+ES+Neo4j RRF)을 실행하여 후보 영화를 검색한다.

    SearchQuery의 필터와 부스트를 hybrid_search()에 전달한다.
    결과를 CandidateMovie 리스트로 변환한다.

    Args:
        state: ChatAgentState (search_query 필요)

    Returns:
        dict: candidate_movies(list[CandidateMovie]) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        search_query = state.get("search_query", SearchQuery())
        emotion = state.get("emotion", EmotionResult())

        # 필터 파라미터 추출
        filters = search_query.filters
        genre_filter = filters.get("genres")
        mood_tags = emotion.mood_tags if emotion else None
        ott_filter = [filters["platform"]] if filters.get("platform") else None
        year_range = filters.get("year_range")

        # 하이브리드 검색 실행
        results = await hybrid_search(
            query=search_query.semantic_query or search_query.keyword_query,
            top_k=search_query.limit,
            genre_filter=genre_filter,
            mood_tags=mood_tags,
            ott_filter=ott_filter,
            year_range=year_range,
        )

        # SearchResult → CandidateMovie 변환
        candidates = [
            _search_result_to_candidate(r, i)
            for i, r in enumerate(results)
        ]

        # exclude_ids로 시청한 영화 제외
        if search_query.exclude_ids:
            exclude_set = set(search_query.exclude_ids)
            candidates = [c for c in candidates if c.id not in exclude_set]

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "rag_retrieved_node",
            candidate_count=len(candidates),
            query_preview=search_query.semantic_query[:80] if search_query.semantic_query else "",
            candidates=[
                {
                    "rank": i + 1,
                    "title": c.title,
                    "rrf_score": round(c.rrf_score, 6),
                    "genres": c.genres[:3],
                    "source": c.retrieval_source,
                }
                for i, c in enumerate(candidates[:10])
            ],
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {"candidate_movies": candidates}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("rag_retriever_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        return {"candidate_movies": []}


# ============================================================
# 7.5. retrieval_quality_checker — 검색 결과 품질 판정
# ============================================================

@traceable(name="retrieval_quality_checker", run_type="chain", metadata={"node": "7.5/14"})
async def retrieval_quality_checker(state: ChatAgentState) -> dict:
    """
    RAG 검색 결과의 품질을 판정하여 state에 기록한다.

    검색 결과(candidate_movies)의 건수, Top-1 점수, 평균 점수를 분석하여
    retrieval_quality_passed(bool)와 retrieval_feedback(str)을 설정한다.

    판정 기준 (models.py 상수 참조):
    - 후보 >= RETRIEVAL_MIN_CANDIDATES (3편)
    - Top-1 RRF >= RETRIEVAL_MIN_TOP_SCORE (0.015)
    - 평균 RRF >= RETRIEVAL_QUALITY_MIN_AVG (0.01)

    세 조건 모두 만족 → passed=True, 하나라도 미달 → passed=False + 피드백 메시지.

    Args:
        state: ChatAgentState (candidate_movies 필요)

    Returns:
        dict: retrieval_quality_passed(bool), retrieval_feedback(str) 업데이트
    """
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    try:
        candidates = state.get("candidate_movies", [])

        # 판정 상수 import
        from monglepick.agents.chat.models import (
            RETRIEVAL_MIN_CANDIDATES,
            RETRIEVAL_MIN_TOP_SCORE,
            RETRIEVAL_QUALITY_MIN_AVG,
        )

        # 빈 결과
        if not candidates:
            elapsed_ms = (time.perf_counter() - node_start) * 1000
            logger.info("retrieval_quality_checked", passed=False, reason="empty",
                        candidate_count=0, elapsed_ms=round(elapsed_ms, 1), session_id=session_id)
            return {
                "retrieval_quality_passed": False,
                "retrieval_feedback": "조건에 맞는 영화를 찾지 못했어요. 다른 키워드로 시도해볼까요?",
            }

        # 후보 수 부족
        if len(candidates) < RETRIEVAL_MIN_CANDIDATES:
            elapsed_ms = (time.perf_counter() - node_start) * 1000
            logger.info("retrieval_quality_checked", passed=False, reason="insufficient_candidates",
                        candidate_count=len(candidates), elapsed_ms=round(elapsed_ms, 1), session_id=session_id)
            return {
                "retrieval_quality_passed": False,
                "retrieval_feedback": "검색 결과가 부족해요. 조건을 조금 넓혀볼까요?",
            }

        # 점수 계산
        top_score = max(c.rrf_score for c in candidates)
        avg_score = sum(c.rrf_score for c in candidates) / len(candidates)

        # Top-1 점수 미달
        if top_score < RETRIEVAL_MIN_TOP_SCORE:
            elapsed_ms = (time.perf_counter() - node_start) * 1000
            logger.info("retrieval_quality_checked", passed=False, reason="low_top_score",
                        top_score=round(top_score, 6), avg_score=round(avg_score, 6),
                        candidate_count=len(candidates), elapsed_ms=round(elapsed_ms, 1), session_id=session_id)
            return {
                "retrieval_quality_passed": False,
                "retrieval_feedback": "조건과 딱 맞는 영화를 찾기 어려웠어요. 좀 더 구체적으로 알려주시면 더 잘 찾아볼게요!",
            }

        # 평균 점수 미달
        if avg_score < RETRIEVAL_QUALITY_MIN_AVG:
            elapsed_ms = (time.perf_counter() - node_start) * 1000
            logger.info("retrieval_quality_checked", passed=False, reason="low_avg_score",
                        top_score=round(top_score, 6), avg_score=round(avg_score, 6),
                        candidate_count=len(candidates), elapsed_ms=round(elapsed_ms, 1), session_id=session_id)
            return {
                "retrieval_quality_passed": False,
                "retrieval_feedback": "검색 결과의 전반적인 품질이 부족해요. 장르나 분위기를 더 알려주시겠어요?",
            }

        # 모든 조건 통과
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info("retrieval_quality_checked", passed=True,
                    top_score=round(top_score, 6), avg_score=round(avg_score, 6),
                    candidate_count=len(candidates), elapsed_ms=round(elapsed_ms, 1), session_id=session_id)
        return {
            "retrieval_quality_passed": True,
            "retrieval_feedback": "",
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("retrieval_quality_checker_error", error=str(e),
                      elapsed_ms=round(elapsed_ms, 1), session_id=session_id)
        # 에러 시 통과 처리하여 추천 흐름 계속 진행
        return {"retrieval_quality_passed": True, "retrieval_feedback": ""}


# ============================================================
# 8. recommendation_ranker — 추천 엔진 서브그래프 호출 (Phase 4)
# ============================================================

@traceable(name="recommendation_ranker", run_type="chain", metadata={"node": "8/13"})
async def recommendation_ranker(state: ChatAgentState) -> dict:
    """
    추천 엔진 서브그래프(§7)를 호출하여 CF+CBF 하이브리드 추천을 수행한다.

    서브그래프 흐름:
    - Cold Start 판정 → (정상: CF→CBF→hybrid / Cold: popularity_fallback)
    - MMR 다양성 재정렬 → ScoreDetail 첨부

    에러 시 RRF 점수 기반 fallback으로 복원한다.

    Args:
        state: ChatAgentState (candidate_movies, user_id, user_profile,
               watch_history, emotion, preferences 필요)

    Returns:
        dict: ranked_movies(list[RankedMovie]) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        candidates = state.get("candidate_movies", [])

        if not candidates:
            logger.warning("recommendation_ranker_no_candidates")
            return {"ranked_movies": []}

        # 추천 엔진 서브그래프 호출
        from monglepick.agents.recommendation.graph import run_recommendation_engine

        ranked = await run_recommendation_engine(
            candidate_movies=candidates,
            user_id=state.get("user_id", ""),
            user_profile=state.get("user_profile", {}),
            watch_history=state.get("watch_history", []),
            emotion=state.get("emotion"),
            preferences=state.get("preferences"),
        )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "recommendation_ranked_node",
            ranked_count=len(ranked),
            ranked_movies=[
                {
                    "rank": m.rank,
                    "title": m.title,
                    "hybrid_score": round(m.score_detail.hybrid_score, 4),
                    "cf_score": round(m.score_detail.cf_score, 4),
                    "cbf_score": round(m.score_detail.cbf_score, 4),
                    "genre_match": round(m.score_detail.genre_match, 4),
                    "mood_match": round(m.score_detail.mood_match, 4),
                }
                for m in ranked
            ],
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {"ranked_movies": ranked}

    except Exception as e:
        # fallback: RRF 점수 기준 정렬 (서브그래프 에러 시 기존 스텁 로직)
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("recommendation_ranker_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        candidates = state.get("candidate_movies", [])
        if not candidates:
            return {"ranked_movies": []}

        sorted_candidates = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)
        ranked: list[RankedMovie] = []
        for i, c in enumerate(sorted_candidates[:5]):
            ranked.append(RankedMovie(
                id=c.id,
                title=c.title,
                title_en=c.title_en,
                genres=c.genres,
                director=c.director,
                cast=c.cast,
                rating=c.rating,
                release_year=c.release_year,
                overview=c.overview,
                mood_tags=c.mood_tags,
                poster_path=c.poster_path,
                ott_platforms=c.ott_platforms,
                certification=c.certification,
                trailer_url=c.trailer_url,
                rank=i + 1,
                score_detail=ScoreDetail(
                    cf_score=0.0,
                    cbf_score=0.0,
                    hybrid_score=c.rrf_score,
                    genre_match=0.0,
                    mood_match=0.0,
                    similar_to=[],
                ),
                explanation="",
            ))
        return {"ranked_movies": ranked}


# ============================================================
# 9. explanation_generator — 추천 이유 생성
# ============================================================

@traceable(name="explanation_generator", run_type="chain", metadata={"node": "9/13", "llm": "exaone-32b"})
async def explanation_generator(state: ChatAgentState) -> dict:
    """
    각 추천 영화에 대해 사용자 맞춤 추천 이유를 생성한다.

    generate_explanations_batch()로 병렬 생성하고, 각 RankedMovie.explanation에 할당한다.

    Args:
        state: ChatAgentState (ranked_movies, emotion, preferences, watch_history 필요)

    Returns:
        dict: ranked_movies(list[RankedMovie], explanation 채워짐) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        ranked = state.get("ranked_movies", [])
        emotion = state.get("emotion")
        prefs = state.get("preferences")
        watch_history = state.get("watch_history", [])

        if not ranked:
            return {"ranked_movies": []}

        # 시청 이력 제목 목록 (상위 5개)
        watch_titles = [
            wh.get("title", "") for wh in watch_history[:5] if wh.get("title")
        ]

        emotion_str = emotion.emotion if emotion else None

        # 배치 병렬 생성
        explanations = await generate_explanations_batch(
            movies=ranked,
            emotion=emotion_str,
            preferences=prefs,
            watch_history_titles=watch_titles,
        )

        # 각 RankedMovie에 explanation 할당 (불변 모델이므로 새 인스턴스 생성)
        updated_ranked: list[RankedMovie] = []
        for movie, explanation in zip(ranked, explanations):
            updated = movie.model_copy(update={"explanation": explanation})
            updated_ranked.append(updated)

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "explanations_generated_node",
            count=len(updated_ranked),
            explanations=[
                {"title": m.title, "explanation_preview": m.explanation[:80]}
                for m in updated_ranked
            ],
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {"ranked_movies": updated_ranked}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("explanation_generator_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        return {"ranked_movies": state.get("ranked_movies", [])}


# ============================================================
# 10. response_formatter — 응답 포맷팅
# ============================================================

@traceable(name="response_formatter", run_type="chain", metadata={"node": "10/13"})
async def response_formatter(state: ChatAgentState) -> dict:
    """
    응답 유형별로 최종 텍스트를 포맷팅하고, messages에 assistant 메시지를 추가한다.

    응답 유형:
    - 추천: ranked_movies가 있으면 영화 카드 포맷
    - 질문: follow_up_question / response가 이미 설정된 경우
    - 에러: error가 설정된 경우
    - 일반: response가 이미 설정된 경우

    포맷:
    - 추천: "{rank}. **{title}** ({release_year})\n- 장르: ...\n- 감독: ...\n- 평점: ...\n{explanation}"
    - 질문/일반/에러: 텍스트 그대로

    Args:
        state: ChatAgentState (ranked_movies, response, error, messages 필요)

    Returns:
        dict: response, messages 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        ranked = state.get("ranked_movies", [])
        existing_response = state.get("response", "")
        error = state.get("error")
        messages = list(state.get("messages", []))

        # 에러 응답
        if error and not existing_response:
            response = "죄송해요, 지금은 추천이 어려워요. 다시 시도해주세요!"
        # 추천 응답: ranked_movies가 있으면 영화 카드 포맷
        elif ranked:
            parts = ["추천 영화를 찾았어요! 🎬\n"]
            for movie in ranked:
                genres_str = ", ".join(movie.genres[:3]) if movie.genres else "-"
                year_str = f" ({movie.release_year})" if movie.release_year else ""
                card = (
                    f"{movie.rank}. **{movie.title}**{year_str}\n"
                    f"   - 장르: {genres_str}\n"
                    f"   - 감독: {movie.director or '-'}\n"
                    f"   - 평점: {movie.rating:.1f}\n"
                )
                if movie.explanation:
                    card += f"   > {movie.explanation}\n"
                parts.append(card)
            response = "\n".join(parts)
        # 기존 response 사용 (질문/일반 대화)
        elif existing_response:
            response = existing_response
        else:
            response = "무엇을 도와드릴까요? 영화 추천이 필요하시면 말씀해주세요!"

        # assistant 메시지 추가
        messages.append({"role": "assistant", "content": response})

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "response_formatted_node",
            response_length=len(response),
            has_movies=bool(ranked),
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {
            "response": response,
            "messages": messages,
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("response_formatter_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        fallback = "죄송해요, 응답을 구성하는 중 문제가 생겼어요."
        messages = list(state.get("messages", []))
        messages.append({"role": "assistant", "content": fallback})
        return {
            "response": fallback,
            "messages": messages,
        }


# ============================================================
# 11. error_handler — 에러 처리
# ============================================================

@traceable(name="error_handler", run_type="chain", metadata={"node": "11/13"})
async def error_handler(state: ChatAgentState) -> dict:
    """
    에러 로깅 후 친절한 안내 메시지를 설정한다.

    그래프에서 unknown/None 의도가 감지되었을 때 호출된다.

    Args:
        state: ChatAgentState

    Returns:
        dict: response, error 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        error_msg = state.get("error", "알 수 없는 오류")
        intent = state.get("intent")
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "error_handler_node",
            error=error_msg,
            intent=intent.intent if intent else None,
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {
            "response": "죄송해요, 잠시 문제가 생겼어요. 다시 한번 말씀해주세요! 🙏",
            "error": error_msg,
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("error_handler_inner_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        return {
            "response": "죄송해요, 잠시 문제가 생겼어요. 다시 한번 말씀해주세요! 🙏",
            "error": str(e),
        }


# ============================================================
# 12. general_responder — 일반 대화 응답
# ============================================================

@traceable(name="general_responder", run_type="chain", metadata={"node": "12/13", "llm": "exaone-32b"})
async def general_responder(state: ChatAgentState) -> dict:
    """
    일반 대화(intent=general)에 대해 몽글 페르소나로 응답한다.

    generate_general_response 체인을 호출하고 response에 설정한다.

    Args:
        state: ChatAgentState (current_input, messages 필요)

    Returns:
        dict: response 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        current_input = state.get("current_input", "")
        messages = state.get("messages", [])

        # 최근 6개 메시지 포맷
        recent = messages[-7:-1] if len(messages) > 1 else []
        recent_messages = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in recent[-6:]
        )

        response = await generate_general_response(
            current_input=current_input,
            recent_messages=recent_messages,
        )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "general_response_node",
            response_preview=response[:50],
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {"response": response}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("general_responder_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        return {"response": "안녕하세요! 영화 추천이 필요하시면 말씀해주세요 😊"}


# ============================================================
# 13. tool_executor_node — 도구 실행 (Phase 6 스텁)
# ============================================================

@traceable(name="tool_executor_node", run_type="tool", metadata={"node": "13/13"})
async def tool_executor_node(state: ChatAgentState) -> dict:
    """
    도구 실행 노드 (Phase 6 스텁).

    info/theater/booking 의도에 대해 아직 구현되지 않은 기능임을 안내한다.
    NotImplementedError를 호출하지 않고, 친절한 안내 메시지를 반환한다.

    Args:
        state: ChatAgentState (intent 필요)

    Returns:
        dict: response 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        intent = state.get("intent")
        intent_str = intent.intent if intent else "unknown"

        # 의도별 안내 메시지
        messages_map = {
            "info": "영화 상세 정보 조회 기능은 곧 준비될 예정이에요! 🎬 "
                    "궁금한 영화가 있으시면 제목을 알려주세요, 아는 범위에서 추천해드릴게요!",
            "theater": "가까운 영화관 검색 기능은 아직 준비 중이에요! 🏢 "
                       "대신 보고 싶은 영화를 추천해드릴까요?",
            "booking": "예매 링크 연결 기능은 아직 준비 중이에요! 🎟️ "
                       "대신 보고 싶은 영화를 추천해드릴까요?",
        }
        response = messages_map.get(
            intent_str,
            "해당 기능은 아직 준비 중이에요. 영화 추천이 필요하시면 말씀해주세요! 🎬",
        )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "tool_executor_stub_node",
            intent=intent_str,
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {"response": response}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("tool_executor_node_error", error=str(e), error_type=type(e).__name__,
                      stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
                      session_id=session_id, user_id=user_id)
        return {"response": "해당 기능은 아직 준비 중이에요. 영화 추천이 필요하시면 말씀해주세요!"}
