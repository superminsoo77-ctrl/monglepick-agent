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

import json
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import aiomysql
import httpx
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
    FilterCondition,
    ImageAnalysisResult,
    IntentResult,
    Location,
    RankedMovie,
    ScoreDetail,
    SearchQuery,
    SuggestedOption,
    is_sufficient,
)
from monglepick.chains import (
    analyze_image,
    classify_intent_and_emotion,
    execute_tool,
    extract_preferences,
    generate_clarification,
    generate_explanations_batch,
    generate_general_response,
    generate_question,
)
from monglepick.tools.geocoding import geocoding
from monglepick.chains.question_chain import _get_missing_fields
from monglepick.db.clients import ES_INDEX_NAME, get_elasticsearch, get_mysql
from monglepick.metrics import external_map_location_source_total
from monglepick.rag.hybrid_search import SearchResult, hybrid_search
from monglepick.utils.movie_info_enricher import (
    enrich_movies_batch,
    search_external_movies,
)

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
        messages.append({
            "role": "user",
            "content": current_input,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

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

        # 첫 턴: MySQL에서 유저 프로필 + 시청 이력 + 암시적 평점 + 행동 프로필 로드
        user_profile: dict[str, Any] = {}
        watch_history: list[dict[str, Any]] = []
        implicit_ratings: dict[str, float] = {}  # Phase 3
        user_behavior_profile: dict[str, Any] = {}  # Phase 4

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

                    # 시청 이력 조회 (최근 50건, 영화 메타데이터 포함)
                    # — 2026-04-07: watch_history → reviews 로 교체
                    #   근거: 몽글픽은 영상 스트리밍 미제공. "봤다" = 리뷰 작성.
                    #         reviews 테이블이 단일 진실 원본이며,
                    #         watch_history 는 Kaggle 26M CF 학습용 시드 데이터 전용.
                    # — review_source: 어떤 경로(AI추천/검색/위시리스트 등)로 시청 후 리뷰했는지
                    # — CBF에서 장르/감독/배우/무드 프로필 구축에 활용
                    await cursor.execute(
                        """
                        SELECT r.movie_id, m.title, r.rating, r.created_at AS watched_at,
                               m.genres, m.director, m.cast_members AS `cast`,
                               m.mood_tags, m.popularity_score, m.rating AS movie_rating,
                               r.review_source
                        FROM reviews r
                        LEFT JOIN movies m ON r.movie_id = m.movie_id
                        WHERE r.user_id = %s
                          AND r.is_deleted = false
                        ORDER BY r.created_at DESC
                        LIMIT 50
                        """,
                        (user_id,),
                    )
                    rows = await cursor.fetchall()
                    watch_history = [dict(r) for r in rows]

                    # JSON 문자열 → list 파싱 (genres, cast, mood_tags)
                    for wh in watch_history:
                        for key in ("genres", "cast", "mood_tags"):
                            val = wh.get(key)
                            if isinstance(val, str):
                                try:
                                    parsed = json.loads(val)
                                    wh[key] = parsed if isinstance(parsed, list) else []
                                except (json.JSONDecodeError, TypeError):
                                    wh[key] = []
                            elif not isinstance(val, list):
                                wh[key] = []

                    # Phase 3: 암시적 평점 조회 (user_implicit_rating 테이블)
                    # CF 캐시 미스 시 fallback 점수로 활용
                    await cursor.execute(
                        """
                        SELECT movie_id, implicit_score
                        FROM user_implicit_rating
                        WHERE user_id = %s AND implicit_score > 0
                        ORDER BY implicit_score DESC
                        LIMIT 200
                        """,
                        (user_id,),
                    )
                    ir_rows = await cursor.fetchall()
                    implicit_ratings = {r["movie_id"]: float(r["implicit_score"]) for r in ir_rows}

                    # Phase 4: 행동 프로필 조회 (user_behavior_profile 테이블)
                    # hybrid_merger에서 taste_consistency 기반 동적 가중치에 활용
                    await cursor.execute(
                        """
                        SELECT genre_affinity, mood_affinity, director_affinity,
                               taste_consistency, recommendation_acceptance_rate,
                               avg_exploration_depth, activity_level
                        FROM user_behavior_profile
                        WHERE user_id = %s
                        LIMIT 1
                        """,
                        (user_id,),
                    )
                    bp_row = await cursor.fetchone()
                    if bp_row:
                        user_behavior_profile = dict(bp_row)
                        # JSON 문자열 → dict 파싱
                        for key in ("genre_affinity", "mood_affinity", "director_affinity"):
                            val = user_behavior_profile.get(key)
                            if isinstance(val, str):
                                try:
                                    user_behavior_profile[key] = json.loads(val)
                                except (json.JSONDecodeError, TypeError):
                                    user_behavior_profile[key] = {}
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
            "implicit_ratings": implicit_ratings,
            "user_behavior_profile": user_behavior_profile,
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
    titles_to_search = movie_titles[:3]
    if not titles_to_search:
        return {"genres": [], "mood_tags": []}

    try:
        # ES msearch API로 배치 조회 (N+1 → 1회 요청으로 개선)
        search_body: list[dict] = []
        for title in titles_to_search:
            # msearch 헤더: 인덱스 지정
            search_body.append({"index": ES_INDEX_NAME})
            # msearch 본문: 개별 쿼리
            search_body.append({
                "query": {
                    "match": {
                        "title": {
                            "query": title,
                            "analyzer": "korean_analyzer",
                        }
                    }
                },
                "size": 1,
            })

        resp = await es.msearch(body=search_body)

        # msearch 응답에서 각 영화의 장르/무드 추출
        for i, sub_resp in enumerate(resp["responses"]):
            title = titles_to_search[i]
            if "error" in sub_resp:
                logger.warning("reference_movie_lookup_error", query_title=title, error=str(sub_resp["error"]))
                continue

            hits = sub_resp.get("hits", {}).get("hits", [])
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
        # msearch 실패 시 빈 결과 반환 (기존 개별 호출도 에러 시 continue였으므로 동일한 graceful degradation)
        logger.warning("reference_movie_msearch_error", error=str(e))

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
        # 이미지 보너스(+1.5)는 genre_cues 또는 mood_cues가 실제로 추출된 경우에만 부여.
        # 비영화 이미지(음식/풍경 등)는 analyzed=True이더라도 cues가 비어 있으므로 보너스 미적용.
        has_image = False
        if image_analysis is not None and image_analysis.analyzed:
            has_image = bool(image_analysis.genre_cues or image_analysis.mood_cues)
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

@traceable(name="question_generator", run_type="chain", metadata={"node": "5/13", "llm": "solar-pro"})
async def question_generator(state: ChatAgentState) -> dict:
    """
    부족한 선호 정보를 파악하기 위한 후속 질문 + 제안 카드를 생성한다.

    needs_clarification=True 또는 검색 품질 미달 시 호출된다.
    - question: 자연스러운 후속 질문 텍스트 (response 필드에도 설정)
    - hints: 필드별 정적 옵션 칩 (기존 UX 유지)
    - suggestions: Claude Code 스타일 AI 생성 제안 카드 2~4개 (2026-04-15 추가)

    검색 품질 미달로 호출된 경우(retrieval_feedback 존재 시):
    - 피드백 메시지를 candidate_hint 로 LLM 에 전달해 더 구체적인 제안을 유도한다.

    Args:
        state: ChatAgentState (preferences, emotion, turn_count, retrieval_feedback,
               candidate_movies 필요)

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
        candidates = state.get("candidate_movies", []) or []

        emotion_str = emotion.emotion if emotion else None

        # ── 최근 후보 요약 (LLM 제안 힌트용) ──
        # 품질 미달 fallback 경로에서 LLM 이 "이런 영화들이 나왔는데 취향이 맞나요?"
        # 스타일의 제안을 만들 수 있도록 상위 3편 제목만 넘긴다.
        candidate_hint = ""
        if candidates:
            titles = []
            for c in candidates[:3]:
                title = getattr(c, "title", None) or getattr(c, "name", None)
                if title:
                    titles.append(str(title))
            if titles:
                candidate_hint = ", ".join(titles)

        # ── LLM 구조화 출력: question + suggestions 동시 생성 ──
        llm_output = await generate_clarification(
            extracted_preferences=prefs,
            emotion=emotion_str,
            turn_count=turn_count,
            retrieval_feedback=retrieval_feedback,
            candidate_hint=candidate_hint,
        )
        question = llm_output.question
        suggestions: list[SuggestedOption] = list(llm_output.suggestions or [])

        # 검색 품질 미달 시 피드백 메시지를 질문 앞에 덧붙여 사용자 인지 강화
        if retrieval_feedback and retrieval_feedback not in question:
            question = (
                f"{retrieval_feedback} "
                f"{question}"
            ).strip()

        # ── 정적 힌트(칩 UI) 유지 — 부족 필드 상위 3개 ──
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
            suggestions=suggestions,
            allow_custom=True,
        )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "question_generated_node",
            question_preview=question[:50],
            hint_count=len(hints),
            suggestion_count=len(suggestions),
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

    try:
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
    except (ValueError, OverflowError):
        # 방어적 처리: 정규식이 digits를 보장하지만, 예상치 못한 변환 실패 시 None 반환
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

        # ── semantic_query 구성 (Intent-First) ──
        # user_intent가 있으면 시맨틱 검색의 핵심 입력으로 사용
        # 없으면 기존 방식 (사용자 입력 + 장르 + 무드 + 참조영화)으로 fallback
        query_parts = []
        if prefs.user_intent:
            # Intent-First: LLM이 요약한 의도가 가장 강력한 시맨틱 쿼리
            query_parts.append(prefs.user_intent)
        query_parts.append(current_input)
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

        # ── filters 구성 (기존 구조화 필드 + 동적 필터 통합) ──
        filters: dict[str, Any] = {}
        if prefs.genre_preference:
            # 쉼표나 공백으로 구분된 장르를 리스트로 변환
            genres = [g.strip() for g in re.split(r"[,\s]+", prefs.genre_preference) if g.strip()]
            if genres:
                filters["genres"] = genres
        if prefs.platform:
            filters["platform"] = prefs.platform

        # 연도 범위 파싱 (era 필드에서)
        year_range = _parse_era(prefs.era) if prefs.era else None
        if year_range:
            filters["year_range"] = year_range

        # ── 동적 필터를 filters 딕셔너리에 변환 ──
        # LLM이 추출한 FilterCondition들을 검색 엔진이 이해하는 형태로 변환
        for fc in prefs.dynamic_filters:
            if fc.field == "rating" and fc.operator == "gte":
                filters["min_rating"] = float(fc.value)
            elif fc.field == "rating" and fc.operator == "lte":
                filters["max_rating"] = float(fc.value)
            elif fc.field == "trailer_url" and fc.operator == "exists":
                filters["has_trailer"] = bool(fc.value)
            elif fc.field == "runtime" and fc.operator == "lte":
                filters["max_runtime"] = int(fc.value)
            elif fc.field == "runtime" and fc.operator == "gte":
                filters["min_runtime"] = int(fc.value)
            elif fc.field == "director" and fc.operator == "eq":
                filters["director"] = str(fc.value)
            elif fc.field == "certification" and fc.operator == "eq":
                filters["certification"] = str(fc.value)
            elif fc.field == "release_year" and fc.operator == "gte":
                # 동적 필터의 release_year가 era보다 우선
                existing_range = filters.get("year_range")
                start = int(fc.value)
                end = existing_range[1] if existing_range else 2030
                filters["year_range"] = (start, end)
            elif fc.field == "release_year" and fc.operator == "lte":
                existing_range = filters.get("year_range")
                start = existing_range[0] if existing_range else 1900
                end = int(fc.value)
                filters["year_range"] = (start, end)
            elif fc.field == "popularity_score" and fc.operator == "gte":
                filters["min_popularity"] = float(fc.value)
            elif fc.field == "vote_count" and fc.operator == "gte":
                filters["min_vote_count"] = int(fc.value)
            # ── 국가/언어 동적 필터 (한국영화, 일본 애니 등 국가 기반 추천) ──
            elif fc.field == "origin_country" and fc.operator == "contains":
                # origin_country는 리스트로 누적 (여러 국가 OR 조건 가능)
                existing = filters.get("origin_country", [])
                existing.append(str(fc.value).upper())
                filters["origin_country"] = existing
            elif fc.field == "original_language" and fc.operator == "eq":
                filters["original_language"] = str(fc.value).lower()
            elif fc.field == "production_countries" and fc.operator == "contains":
                existing = filters.get("production_countries", [])
                existing.append(str(fc.value).upper())
                filters["production_countries"] = existing

        # ── boost_keywords 구성 (기존 + 새 search_keywords 통합) ──
        boost_keywords: list[str] = []
        # LLM이 추출한 search_keywords를 최우선 부스트로 추가
        if prefs.search_keywords:
            boost_keywords.extend(prefs.search_keywords)
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
            dynamic_filter_count=len(prefs.dynamic_filters),
            user_intent_used=bool(prefs.user_intent),
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

    # runtime: int 변환 — None 이면 None 유지 (후처리 필터에서 None은 통과 처리)
    raw_runtime = meta.get("runtime")
    runtime_val: int | None = int(raw_runtime) if raw_runtime is not None else None

    # popularity_score: float 변환 — None 이면 None 유지
    raw_popularity = meta.get("popularity_score")
    popularity_val: float | None = float(raw_popularity) if raw_popularity is not None else None

    # vote_count: int 변환 — None 이면 None 유지
    raw_vote_count = meta.get("vote_count")
    vote_count_val: int | None = int(raw_vote_count) if raw_vote_count is not None else None

    # backdrop_path: 빈 문자열보다 None이 명시적이므로 빈 문자열은 None으로 정규화
    raw_backdrop = meta.get("backdrop_path", "")
    backdrop_val: str | None = raw_backdrop if raw_backdrop else None

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
        # ── 확장 메타데이터 필드 (설계서 기준 추가, 필터링·UI 렌더링용) ──
        runtime=runtime_val,
        popularity_score=popularity_val,
        vote_count=vote_count_val,
        backdrop_path=backdrop_val,
        # ── 국가/언어 필드 (한국영화 필터링 및 재랭킹 시 국가 판별용) ──
        original_language=meta.get("original_language", ""),
        origin_country=meta.get("origin_country", []) if isinstance(meta.get("origin_country"), list) else [],
    )


# ── TMDB 포스터 보강 헬퍼 (Phase Q-3) ──
# 포스터 없는 후보 영화의 poster_path를 TMDB API로 가져온다.
# 포스터 없다고 추천에서 제외하지 않고, 외부에서 데이터를 보강하는 전략.
_TMDB_POSTER_TIMEOUT = 3.0  # 개별 API 호출 타임아웃 (초)


async def _enrich_missing_posters(candidates: list) -> None:
    """
    poster_path가 빈 후보 영화에 대해 TMDB API로 포스터를 가져와 보강한다.

    TMDB movie ID로 /movie/{id} 를 호출하여 poster_path를 채운다.
    실패 시 무시 (best-effort). 원본 candidates 리스트를 in-place 수정.
    """
    from monglepick.config import settings
    import asyncio

    if not settings.TMDB_API_KEY:
        return

    # 포스터가 없는 영화만 대상 (최대 5편으로 제한하여 지연 최소화)
    missing = [c for c in candidates if not c.poster_path or not c.poster_path.strip()]
    if not missing:
        return

    missing = missing[:5]

    async def _fetch_poster(movie) -> None:
        """단일 영화의 포스터를 TMDB API에서 가져온다."""
        try:
            # movie.id가 TMDB 숫자 ID인 경우에만 호출
            movie_id = movie.id
            if not movie_id or not str(movie_id).isdigit():
                return

            url = f"{settings.TMDB_BASE_URL}/movie/{movie_id}"
            params = {"api_key": settings.TMDB_API_KEY, "language": "ko-KR"}

            async with httpx.AsyncClient(timeout=_TMDB_POSTER_TIMEOUT) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    poster = data.get("poster_path", "")
                    if poster:
                        movie.poster_path = poster
                        logger.info(
                            "tmdb_poster_enriched",
                            movie_id=movie_id,
                            title=movie.title,
                            poster_path=poster,
                        )
        except Exception:
            pass  # best-effort: 실패해도 추천 흐름에 영향 없음

    await asyncio.gather(*[_fetch_poster(m) for m in missing])


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

        # ── 동적 필터 파라미터 추출 (Intent-First) ──
        min_rating = filters.get("min_rating")           # float | None
        has_trailer = filters.get("has_trailer")          # bool | None
        director_name = filters.get("director")           # str | None (동적 필터에서 추출)
        min_popularity = filters.get("min_popularity")    # float | None — TMDB 인기도 최소값
        max_runtime = filters.get("max_runtime")          # int | None — 최대 상영시간(분)
        min_vote_count = filters.get("min_vote_count")    # int | None — 최소 투표수
        # ── 국가/언어 필터 파라미터 추출 (한국영화 필터링) ──
        origin_country = filters.get("origin_country")            # list[str] | None — 예: ["KR"]
        original_language = filters.get("original_language")      # str | None — 예: "ko"
        production_countries = filters.get("production_countries") # list[str] | None — 예: ["US"]

        # 선호에서 참조영화 ID 추출 (Neo4j 검색용)
        preferences = state.get("preferences")
        similar_movie_id = None
        if preferences and preferences.reference_movies:
            ref_info = filters.get("reference_movie_id")
            if ref_info:
                similar_movie_id = ref_info

        # 하이브리드 검색 실행 — 동적 필터(min_rating, has_trailer, min_popularity, max_runtime, min_vote_count, 국가/언어) 전달
        results = await hybrid_search(
            query=search_query.semantic_query or search_query.keyword_query,
            top_k=search_query.limit,
            genre_filter=genre_filter,
            mood_tags=mood_tags,
            ott_filter=ott_filter,
            min_rating=min_rating,
            year_range=year_range,
            director=director_name,
            similar_to_movie_id=similar_movie_id,
            exclude_ids=search_query.exclude_ids,
            has_trailer=has_trailer,
            min_popularity=min_popularity,
            max_runtime=max_runtime,
            min_vote_count=min_vote_count,
            origin_country_filter=origin_country,
            language_filter=original_language,
            production_countries_filter=production_countries,
        )

        # SearchResult → CandidateMovie 변환
        candidates = [
            _search_result_to_candidate(r, i)
            for i, r in enumerate(results)
        ]

        # 안전망: hybrid_search 내부에서 이미 exclude 했지만, 혹시 누락된 경우 2차 필터
        if search_query.exclude_ids:
            exclude_set = set(search_query.exclude_ids)
            candidates = [c for c in candidates if c.id not in exclude_set]

        # ── 후처리 필터링 (DB 검색에서 적용 못한 동적 필터 2차 적용) ──
        # Qdrant/ES/Neo4j 모두에서 지원하지 않는 필터는 후처리로 적용
        pre_filter_count = len(candidates)

        # has_trailer 후처리: 트레일러 URL이 실제로 있는 영화만 필터 (Qdrant 필터 누락 대비)
        if has_trailer and candidates:
            filtered = [c for c in candidates if c.trailer_url]
            # 필터 후 후보가 2편 미만이면 필터 완화 (추천 실패 방지)
            if len(filtered) >= 2:
                candidates = filtered

        # min_rating 후처리: 평점 조건 2차 검증
        if min_rating is not None and candidates:
            filtered = [c for c in candidates if c.rating >= min_rating]
            if len(filtered) >= 2:
                candidates = filtered

        # max_runtime 후처리: CandidateMovie.runtime 필드로 상영시간 조건 2차 검증
        # hybrid_search 내부(Qdrant/ES)에서 이미 필터를 적용했지만,
        # Neo4j 결과에는 runtime 필터가 적용되지 않으므로 후처리로 보완한다.
        # runtime이 None인 영화는 데이터 누락으로 간주하여 제외하지 않는다 (false negative 방지).
        if max_runtime is not None and candidates:
            filtered = [
                c for c in candidates
                if c.runtime is None or c.runtime <= max_runtime
            ]
            # 필터 후 후보가 2편 이상이면 적용, 미만이면 필터 완화 (추천 실패 방지)
            if len(filtered) >= 2:
                candidates = filtered

        # ── 국가/언어 후처리: CandidateMovie.origin_country/original_language로 2차 검증 ──
        # hybrid_search 내부에서 이미 DB 레벨 필터를 적용했지만,
        # 메타데이터 불일치나 Neo4j 결과 누락을 보완하기 위해 후처리로 재검증한다.
        if origin_country and candidates:
            filter_set = set(origin_country)
            filtered = [
                c for c in candidates
                if not c.origin_country  # 데이터 없으면 통과 (false negative 방지)
                or any(cc in filter_set for cc in c.origin_country)
            ]
            if len(filtered) >= 2:
                candidates = filtered

        if original_language and candidates:
            filtered = [
                c for c in candidates
                if not c.original_language  # 데이터 없으면 통과
                or c.original_language == original_language
            ]
            if len(filtered) >= 2:
                candidates = filtered

        # ── year_range 후처리: ES/Neo4j 필터 누락 대비 2차 검증 ──
        if year_range and candidates:
            filtered = [
                c for c in candidates
                if not c.release_year or (year_range[0] <= c.release_year <= year_range[1])
            ]
            if len(filtered) >= 2:
                candidates = filtered

        # ── min_popularity 후처리: Qdrant 필터 누락 대비 2차 검증 ──
        if min_popularity is not None and candidates:
            filtered = [
                c for c in candidates
                if c.popularity_score is None or c.popularity_score >= min_popularity
            ]
            if len(filtered) >= 2:
                candidates = filtered

        if pre_filter_count != len(candidates):
            logger.info(
                "rag_post_filter_applied",
                before=pre_filter_count,
                after=len(candidates),
                has_trailer=has_trailer,
                min_rating=min_rating,
                origin_country=origin_country,
                original_language=original_language,
                year_range=year_range,
                min_popularity=min_popularity,
            )

        # ── 데이터 품질 필터링 (Phase Q-2: 충분한 메타데이터가 있는 영화만 추천) ──
        # 문제: 포스터/줄거리/평점이 없는 무명 영화가 검색 관련성만으로 추천되는 현상
        # 해결: 최소 데이터 품질 기준을 충족하지 못하는 후보를 사전 제거한다.
        # 단, 필터 후 후보가 너무 적어지면 완화하여 추천 실패를 방지한다.
        if candidates:
            pre_quality_count = len(candidates)

            def _has_sufficient_data(movie: CandidateMovie) -> bool:
                """
                영화가 추천에 충분한 메타데이터를 갖추고 있는지 판정한다.

                필수 조건 (Phase Q-3 내용 기반):
                - release_year > 0 (필수 — 0이면 TMDB에 개봉일 데이터 자체가 없는 항목)
                - 줄거리 20자 이상 OR 평점 1.0 이상 (내용/평가 중 하나는 있어야 함)
                포스터 없어도 내용이 충분하면 통과 — 포스터는 후속 TMDB 보강으로 해결.
                """
                # release_year 필수: 0이면 TMDB 메타데이터가 극히 불완전한 항목
                if not movie.release_year or movie.release_year < 1900:
                    return False
                # 줄거리 또는 평점 중 1개 이상 존재
                has_overview = movie.overview and len(movie.overview.strip()) >= 20
                has_rating = movie.rating and movie.rating >= 1.0
                return has_overview or has_rating

            quality_filtered = [c for c in candidates if _has_sufficient_data(c)]

            # 품질 필터 후 후보 수에 따른 적용
            MIN_QUALITY_KEEP = 3
            if len(quality_filtered) >= MIN_QUALITY_KEEP:
                candidates = quality_filtered
            elif quality_filtered:
                # 1~2편이라도 양질이면 그것만 사용
                candidates = quality_filtered
            # 양질 0편이면 원본 유지 (후속 TMDB 보강 + 저품질 안내 메시지로 대응)

            if pre_quality_count != len(candidates):
                logger.info(
                    "data_quality_filter_applied",
                    before=pre_quality_count,
                    after=len(candidates),
                    removed_count=pre_quality_count - len(candidates),
                    session_id=session_id,
                )

        # ── 포스터 보유 영화 우선 정렬 (Phase Q-2.2) ──
        # 동일 RRF 점수 대역에서 포스터가 있는 영화를 앞으로 배치하여
        # 프론트엔드 movie_card UI에서 "No Poster" 카드가 상위에 노출되는 것을 방지.
        if candidates:
            candidates.sort(
                key=lambda c: (
                    1 if (c.poster_path and c.poster_path.strip()) else 0,
                    c.rrf_score,
                ),
                reverse=True,
            )

        # ── TMDB 포스터 보강: 포스터 없는 영화에 대해 TMDB API로 poster_path 채움 ──
        # 포스터 없다고 추천에서 제외하지 않고, 외부에서 데이터를 보강하는 전략.
        await _enrich_missing_posters(candidates)

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "rag_retrieved_node",
            candidate_count=len(candidates),
            query_preview=search_query.semantic_query[:80] if search_query.semantic_query else "",
            dynamic_filters_applied={
                "min_rating": min_rating,
                "has_trailer": has_trailer,
                "director": director_name,
            },
            candidates=[
                {
                    "rank": i + 1,
                    "title": c.title,
                    "rating": c.rating,
                    "rrf_score": round(c.rrf_score, 6),
                    "genres": c.genres[:3],
                    "has_trailer": bool(c.trailer_url),
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

        # ── 개별 영화 데이터 품질 필터링 (Phase Q-3 내용 기반) ──
        # release_year 필수 + (줄거리 or 평점) 기준. rag_retriever와 동일.
        # 포스터 없어도 내용이 충분하면 통과 — 포스터는 TMDB 보강으로 해결.
        high_quality: list = []
        low_quality: list = []
        for c in candidates:
            has_year = bool(c.release_year and c.release_year >= 1900)
            has_overview = bool(c.overview and len(c.overview.strip()) >= 20)
            has_rating = bool(c.rating and c.rating >= 1.0)
            # 연도 필수, 줄거리/평점 중 1개 이상
            if has_year and (has_overview or has_rating):
                high_quality.append(c)
            else:
                low_quality.append(c)

        low_quality_count = len(low_quality)

        # 양질 후보가 최소 기준 이상이면 저품질 제거, 아니면 부족분만큼 저품질에서 보충
        if len(high_quality) >= RETRIEVAL_MIN_CANDIDATES:
            filtered_candidates = high_quality
        else:
            # 양질 후보 부족: 저품질 중 RRF 점수 상위로 보충하여 최소 기준 충족
            need = RETRIEVAL_MIN_CANDIDATES - len(high_quality)
            low_quality_sorted = sorted(low_quality, key=lambda c: c.rrf_score, reverse=True)
            filtered_candidates = high_quality + low_quality_sorted[:need]

        if low_quality_count > 0:
            logger.info(
                "retrieval_quality_filtered",
                original_count=len(candidates),
                filtered_count=len(filtered_candidates),
                removed_count=len(candidates) - len(filtered_candidates),
                low_quality_titles=[c.title for c in low_quality],
                session_id=session_id,
            )

        # 모든 조건 통과
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info("retrieval_quality_checked", passed=True,
                    top_score=round(top_score, 6), avg_score=round(avg_score, 6),
                    candidate_count=len(filtered_candidates),
                    low_quality_removed=len(candidates) - len(filtered_candidates),
                    elapsed_ms=round(elapsed_ms, 1), session_id=session_id)
        return {
            "retrieval_quality_passed": True,
            "retrieval_feedback": "",
            "candidate_movies": filtered_candidates,
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error("retrieval_quality_checker_error", error=str(e),
                      elapsed_ms=round(elapsed_ms, 1), session_id=session_id)
        # 에러 시 통과 처리하여 추천 흐름 계속 진행
        return {"retrieval_quality_passed": True, "retrieval_feedback": ""}


# ============================================================
# 7.6. similar_fallback_search — 품질 미달 시 비슷한 영화 확장 검색 (Phase Q-3)
# ============================================================

@traceable(name="similar_fallback_search", run_type="retriever", metadata={"node": "7.6/16"})
async def similar_fallback_search(state: ChatAgentState) -> dict:
    """
    검색 품질 미달 시, 기존 후보 영화의 장르/무드를 활용해 비슷한 영화를 확장 검색한다.

    기존 후보가 있지만 품질이 낮을 때(데이터 부족, 무명 영화 등),
    상위 후보의 장르·무드태그를 기반으로 hybrid_search를 다시 수행하여
    충분한 데이터를 가진 유사 영화로 후보를 보강한다.

    전략:
    1. 기존 후보 상위 3편에서 장르/무드태그 추출
    2. 추출한 장르·무드를 필터로 hybrid_search 재실행
    3. 기존 후보의 ID는 exclude하여 중복 방지
    4. 새 결과 + 기존 고품질 후보를 합산하여 candidate_movies 업데이트

    Args:
        state: ChatAgentState (candidate_movies, search_query 필요)

    Returns:
        dict: candidate_movies(확장된 list[CandidateMovie]) 업데이트
    """
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        candidates = state.get("candidate_movies", [])
        search_query = state.get("search_query")

        if not candidates:
            return {"candidate_movies": []}

        # ── 1. 기존 후보 상위 3편에서 장르/무드 추출 ──
        top_candidates = candidates[:3]
        fallback_genres: list[str] = []
        fallback_moods: list[str] = []
        exclude_ids: list[str] = [c.id for c in candidates]

        for c in top_candidates:
            fallback_genres.extend(c.genres[:2])
            if c.mood_tags:
                fallback_moods.extend(c.mood_tags[:2])

        # 중복 제거
        fallback_genres = list(dict.fromkeys(fallback_genres))[:5]
        fallback_moods = list(dict.fromkeys(fallback_moods))[:4]

        # ── 2. 장르/무드 기반으로 확장 검색 ──
        # 기존 검색어 대신 장르+무드를 의미적 쿼리로 구성
        semantic_parts = []
        if fallback_genres:
            semantic_parts.append(" ".join(fallback_genres))
        if fallback_moods:
            semantic_parts.append(" ".join(fallback_moods))
        # 원래 사용자 입력도 포함하여 관련성 유지
        original_query = state.get("current_input", "")
        if original_query:
            semantic_parts.append(original_query[:50])

        fallback_query = " ".join(semantic_parts) if semantic_parts else "인기 영화 추천"

        logger.info(
            "similar_fallback_search_start",
            original_candidates=len(candidates),
            fallback_genres=fallback_genres,
            fallback_moods=fallback_moods,
            fallback_query=fallback_query[:80],
            exclude_count=len(exclude_ids),
            session_id=session_id,
        )

        from monglepick.rag.hybrid_search import hybrid_search

        # ── 원본 동적 필터 보존 (Phase Q-2.2) ──
        # fallback 검색에서도 원래 사용자 쿼리의 국가/연도/인기도 필터를 유지하여
        # "한국 영화" 요청 시 비한국 영화가 fallback으로 추천되는 것을 방지.
        orig_filters = search_query.filters if search_query else {}

        results = await hybrid_search(
            query=fallback_query,
            top_k=15,
            genre_filter=fallback_genres if fallback_genres else None,
            mood_tags=fallback_moods if fallback_moods else None,
            exclude_ids=exclude_ids,
            origin_country_filter=orig_filters.get("origin_country"),
            language_filter=orig_filters.get("original_language"),
            year_range=orig_filters.get("year_range"),
            min_popularity=orig_filters.get("min_popularity"),
            min_vote_count=orig_filters.get("min_vote_count"),
            min_rating=orig_filters.get("min_rating"),
        )

        # ── 3. SearchResult → CandidateMovie 변환 ──
        new_candidates = [
            _search_result_to_candidate(r, i)
            for i, r in enumerate(results)
        ]

        # ── 4. 데이터 품질 필터 적용 (Phase Q-3 내용 기반, rag_retriever와 동일 기준) ──
        def _has_sufficient_data(movie) -> bool:
            # release_year 필수: 0이면 TMDB 메타데이터 극히 불완전
            if not movie.release_year or movie.release_year < 1900:
                return False
            # 줄거리 또는 평점 중 1개 이상
            has_overview = movie.overview and len(movie.overview.strip()) >= 20
            has_rating = movie.rating and movie.rating >= 1.0
            return has_overview or has_rating

        quality_new = [c for c in new_candidates if _has_sufficient_data(c)]

        # ── 5. 기존 고품질 후보 + 새 결과 합산 ──
        # 기존 후보 중 데이터 품질이 좋은 것은 유지
        quality_existing = [c for c in candidates if _has_sufficient_data(c)]

        # 합산: 기존 고품질 + 새 결과 (중복 ID 제거)
        seen_ids: set[str] = set()
        merged: list[CandidateMovie] = []
        for c in quality_existing + quality_new:
            if c.id not in seen_ids:
                seen_ids.add(c.id)
                merged.append(c)

        # 최소 5편 확보 못하면 기존 전체 + 새 결과로 완화
        if len(merged) < 5:
            for c in candidates + new_candidates:
                if c.id not in seen_ids:
                    seen_ids.add(c.id)
                    merged.append(c)

        # ── 5편 보장: 아직 부족하면 장르 필터 없이 추가 검색 ──
        if len(merged) < 5:
            logger.info(
                "similar_fallback_extra_search",
                current_count=len(merged),
                reason="still_under_5_after_genre_search",
                session_id=session_id,
            )
            extra_exclude = list(seen_ids)
            # 추가 검색에도 원본 국가/언어 필터 보존 (장르 필터는 해제하여 범위 확장)
            extra_results = await hybrid_search(
                query=original_query or "인기 영화 추천",
                top_k=15,
                exclude_ids=extra_exclude,
                origin_country_filter=orig_filters.get("origin_country"),
                language_filter=orig_filters.get("original_language"),
            )
            extra_candidates = [
                _search_result_to_candidate(r, i)
                for i, r in enumerate(extra_results)
            ]
            # 품질 좋은 것 우선, 부족하면 전부 추가
            extra_quality = [c for c in extra_candidates if _has_sufficient_data(c)]
            for c in extra_quality + extra_candidates:
                if len(merged) >= 10:
                    break
                if c.id not in seen_ids:
                    seen_ids.add(c.id)
                    merged.append(c)

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "similar_fallback_search_done",
            original_count=len(candidates),
            new_found=len(quality_new),
            merged_count=len(merged),
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
        )

        return {"candidate_movies": merged}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "similar_fallback_search_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
        )
        # 에러 시 기존 후보 그대로 유지
        return {"candidate_movies": state.get("candidate_movies", [])}


# ============================================================
# 7.65. external_search_node — DB 후보 0건 & 최신 시그널 존재 시 외부 웹 검색
# ============================================================
#
# 언제 호출되나?
#   route_after_retrieval() 가 num_candidates==0 AND preferences.dynamic_filters 의
#   release_year 하한이 (current_year - 1) 이상인 경우에만 이 노드로 분기한다.
#   (graph.py:_has_recency_signal() 참조)
#
# 왜 필요한가?
#   내부 DB(Qdrant/ES/Neo4j) 는 "수집 시점 기준" 영화만 가진다. 사용자가
#   "2026년 개봉 영화" / "올해 나온 영화" 같은 시기 질의를 던졌을 때 DB 에
#   해당 후보가 0 건이면 기존에는 question_generator 로 보내 재질문만 반복
#   했다. 실제 신작 정보는 Wikipedia/나무위키 등에 이미 있으므로 DuckDuckGo
#   웹 검색으로 보강해 "DB 외" 카드로라도 응답한다.
#
# 출력 계약:
#   - candidate_movies 는 건드리지 않는다. (recommendation_ranker 를 건너뛰므로
#     CF/CBF 점수 계산이 불가능하고, 0 건이면 ranker 가 바로 빈 리스트 반환한다.)
#   - ranked_movies 에 RankedMovie 스텁을 직접 채워 넣는다.
#     * id = "external_{i}" 접두사로 외부 출처 구분 가능
#     * score_detail.hybrid_score = 0.0 (정렬 기준 없음 — 웹 검색 순서 유지)
#     * explanation = "DB 외 외부 웹 검색 결과" 고정 문구
#   - 그래프 엣지는 external_search_node → response_formatter (직결) 로,
#     explanation_generator 의 enrich_movies_batch() 로 다시 검색하지 않는다
#     (이미 외부 검색 결과를 담고 있음).
# ============================================================

@traceable(name="external_search_node", run_type="retriever", metadata={"node": "7.65/17"})
async def external_search_node(state: ChatAgentState) -> dict:
    """
    DB 후보 0건 + 최신 시그널 있을 때 DuckDuckGo 로 외부 영화 정보를 검색한다.

    preferences 에서 dynamic_filters[release_year>=N] 의 하한값을 뽑아
    search_external_movies() 로 웹 검색하고, 결과를 RankedMovie 스텁으로
    변환해 ranked_movies 에 직접 담는다. 이후 그래프는 recommendation_ranker /
    explanation_generator 를 모두 건너뛰고 response_formatter 로 직행한다.

    에러·타임아웃·결과 0 건 모두 ranked_movies=[] 로 graceful degrade 되어
    response_formatter 의 "조건에 맞는 영화를 찾지 못했어요" 안내가 나온다.

    Args:
        state: ChatAgentState (preferences, current_input, session_id)

    Returns:
        dict: ranked_movies (list[RankedMovie] 스텁) 업데이트
    """
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")

    try:
        preferences: ExtractedPreferences | None = state.get("preferences")
        current_input = state.get("current_input", "")

        # ── 1. release_year 하한값 추출 (dynamic_filters 우선) ──
        release_year_gte: int | None = None
        user_intent = ""
        if preferences:
            user_intent = preferences.user_intent or ""
            for fc in preferences.dynamic_filters:
                if fc.field == "release_year" and fc.operator == "gte":
                    try:
                        release_year_gte = int(fc.value)
                    except (TypeError, ValueError):
                        pass
                    break

        # ── 2. DuckDuckGo 외부 검색 실행 ──
        external_movies = await search_external_movies(
            user_intent=user_intent,
            current_input=current_input,
            release_year_gte=release_year_gte,
            max_movies=5,
        )

        if not external_movies:
            elapsed_ms = (time.perf_counter() - node_start) * 1000
            logger.info(
                "external_search_no_results",
                release_year_gte=release_year_gte,
                user_intent=user_intent[:80],
                elapsed_ms=round(elapsed_ms, 1),
                session_id=session_id,
                user_id=user_id,
            )
            return {"ranked_movies": []}

        # ── 3. 외부 영화 스텁 → RankedMovie 변환 ──
        # 주의: 내부 DB PK 가 아니므로 recommendation_log 저장 경로는 건너뛴다.
        # Client 는 id 접두사 "external_" 로 이 카드를 "DB 외 정보" 로 표시해야 한다.
        ranked: list[RankedMovie] = []
        for i, m in enumerate(external_movies):
            overview_text = m.get("overview", "") or ""
            source_url = m.get("source_url", "") or ""

            # 출처 URL 을 overview 뒤에 부착해 Client 가 "더 보기" 링크로 활용 가능
            enriched_overview = overview_text
            if source_url:
                enriched_overview = (
                    f"{overview_text}\n[외부 출처] {source_url}"
                    if overview_text
                    else f"[외부 출처] {source_url}"
                )

            ranked.append(RankedMovie(
                id=m.get("id", f"external_{i}"),
                title=m.get("title", "") or "",
                title_en="",
                genres=[],
                director="",
                cast=[],
                rating=0.0,
                release_year=int(m.get("release_year") or 0),
                overview=enriched_overview,
                mood_tags=[],
                poster_path="",
                ott_platforms=[],
                certification="",
                trailer_url="",
                rank=i + 1,
                score_detail=ScoreDetail(
                    cf_score=0.0,
                    cbf_score=0.0,
                    hybrid_score=0.0,
                    genre_match=0.0,
                    mood_match=0.0,
                    similar_to=[],
                ),
                # 고정 문구: response_formatter 의 몽글이 LLM 이 이 설명을 보고
                # "최신 정보라 DB 에는 없지만 웹에서 찾아왔어요" 풍으로 자연스럽게 변환.
                explanation="DB 에 없는 최신 영화 정보를 외부 웹에서 찾아왔어요.",
                recommendation_log_id=None,
            ))

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "external_search_node_completed",
            release_year_gte=release_year_gte,
            user_intent=user_intent[:80],
            ranked_count=len(ranked),
            ranked_titles=[m.title for m in ranked],
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {"ranked_movies": ranked}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "external_search_node_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        # 에러 시 빈 ranked_movies 반환 → response_formatter 가 친절한 fallback 응답
        return {"ranked_movies": []}


# ============================================================
# 7.7. llm_reranker — LLM 기반 후처리 재랭킹 (Phase Q)
# ============================================================

@traceable(name="llm_reranker", run_type="chain", metadata={"node": "7.7/15", "llm": "solar-pro"})
async def llm_reranker(state: ChatAgentState) -> dict:
    """
    LLM 기반으로 후보 영화를 사용자 의도에 맞게 재랭킹한다.

    RAG 검색 + RRF 합산으로 가져온 후보를 LLM의 세계 지식으로 재평가하여:
    - DB 필터나 벡터 검색으로 잡을 수 없는 조건을 검증
    - 사용자 요청에 부적합한 후보를 제거/감점
    - 적합한 후보를 상위로 이동

    예: "아카데미 수상작" → DB에 수상 필드 없지만 LLM이 자체 지식으로 판단
    예: "실화 바탕" → overview에 없어도 LLM이 영화 지식으로 판단
    예: "OST 좋은 영화" → LLM이 영화별 OST 평가를 자체 지식으로 판단

    에러 시 원본 순서를 그대로 유지한다 (graceful degradation).

    Args:
        state: ChatAgentState (candidate_movies, current_input, emotion, preferences 필요)

    Returns:
        dict: candidate_movies(재랭킹된 list[CandidateMovie]) 업데이트
    """
    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        candidates = state.get("candidate_movies", [])
        current_input = state.get("current_input", "")
        emotion = state.get("emotion")
        preferences = state.get("preferences")

        if not candidates:
            return {"candidate_movies": []}

        # rerank_candidates 체인 호출 (Solar API)
        from monglepick.chains.rerank_chain import rerank_candidates

        reranked = await rerank_candidates(
            candidates=candidates,
            user_request=current_input,
            emotion=emotion,
            preferences=preferences,
        )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "llm_reranker_node_completed",
            original_count=len(candidates),
            reranked_count=len(reranked),
            top_reranked=[
                {"title": m.title, "rating": m.rating, "rrf": round(m.rrf_score, 4)}
                for m in reranked[:5]
            ],
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        return {"candidate_movies": reranked}

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "llm_reranker_error", error=str(e), error_type=type(e).__name__,
            stack_trace=traceback.format_exc(), elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id, user_id=user_id,
        )
        # 에러 시: 원본 순서 유지 (graceful degradation)
        return {"candidate_movies": state.get("candidate_movies", [])}


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

        # 사용자 요청 편수(requested_count) 가 있으면 존중, 아니면 기본 5편
        preferences = state.get("preferences")
        fallback_top_k = 5
        if preferences is not None and preferences.requested_count is not None:
            fallback_top_k = max(1, min(5, preferences.requested_count))

        sorted_candidates = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)
        ranked: list[RankedMovie] = []
        for i, c in enumerate(sorted_candidates[:fallback_top_k]):
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
                # ── 확장 메타데이터 필드 CandidateMovie → RankedMovie 복사 ──
                # SSE movie_card 이벤트로 프론트엔드에 전달되므로 반드시 복사해야 한다.
                runtime=c.runtime,
                popularity_score=c.popularity_score,
                vote_count=c.vote_count,
                backdrop_path=c.backdrop_path,
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
        # 사용자의 원래 요청 메시지 (추천 이유에 공감 문구 생성용)
        current_input = state.get("current_input", "")

        if not ranked:
            return {"ranked_movies": []}

        # ── 영화 정보 외부 검색 보강 (overview 부족 시 DuckDuckGo 검색) ──
        # overview가 250자 미만인 영화에 대해 Wikipedia/나무위키 등에서 줄거리를 수집하여
        # LLM 추천 이유 생성의 품질을 높인다. 검색 실패 시 원본 그대로 사용 (에러 전파 금지).
        ranked_dicts = [m.model_dump() for m in ranked]
        enriched_dicts = await enrich_movies_batch(ranked_dicts, max_concurrent=3)

        # 보강된 overview를 RankedMovie에 반영 (model_copy로 불변성 유지)
        enriched_ranked: list[RankedMovie] = []
        for original, enriched_dict in zip(ranked, enriched_dicts):
            if enriched_dict.get("_enriched"):
                # 외부 검색으로 보강된 영화 → overview 업데이트
                enriched_ranked.append(
                    original.model_copy(update={"overview": enriched_dict["overview"]})
                )
            else:
                enriched_ranked.append(original)

        # 시청 이력 제목 목록 (상위 5개)
        watch_titles = [
            wh.get("title", "") for wh in watch_history[:5] if wh.get("title")
        ]

        # 감정 문자열 + 무드태그 추출 (EmotionResult에서 분리)
        emotion_str = emotion.emotion if emotion else None
        user_mood_tags = emotion.mood_tags if emotion else None

        # 배치 생성 (보강된 영화 데이터 + 사용자 원문 메시지 + 무드태그 전달)
        explanations = await generate_explanations_batch(
            movies=enriched_ranked,
            emotion=emotion_str,
            user_mood_tags=user_mood_tags,
            user_message=current_input,
            preferences=prefs,
            watch_history_titles=watch_titles,
        )

        # 각 RankedMovie에 explanation 할당 (불변 모델이므로 새 인스턴스 생성)
        # 마크다운 후처리: LLM이 **bold** 등을 쓸 경우 순수 텍스트로 정리
        # 최종 반환에는 원본(ranked) 기반으로 생성하여 '[외부 정보]' 태그가
        # 사용자에게 노출되지 않도록 한다 (보강 overview는 LLM 입력에만 사용).
        updated_ranked: list[RankedMovie] = []
        for original_movie, explanation in zip(ranked, explanations):
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", explanation)
            clean = re.sub(r"\*(.+?)\*", r"\1", clean)
            clean = re.sub(r"^#{1,6}\s+", "", clean, flags=re.MULTILINE)
            clean = re.sub(r"^[-*]\s+", "", clean, flags=re.MULTILINE)
            updated = original_movie.model_copy(update={"explanation": clean})
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
    몽글이 LLM을 호출하여 최종 응답을 생성하고, messages에 assistant 메시지를 추가한다.

    Solar API가 처리한 데이터(추천 영화, 추천 이유, 감정, 선호 등)를
    몽글이에게 전달하여 자연스러운 대화체 응답을 생성한다.

    응답 유형:
    - 추천: ranked_movies → 몽글이가 대화체로 추천 답변 생성
    - 질문: follow_up_question → 몽글이가 대화체로 질문 전달
    - 에러: 에러 안내 메시지
    - 일반: 기존 response 그대로 사용 (general_responder에서 이미 몽글이가 생성)

    Args:
        state: ChatAgentState (ranked_movies, response, error, messages, emotion, preferences 필요)

    Returns:
        dict: response, messages 업데이트
    """
    from monglepick.chains.response_generation_chain import (
        generate_question_response,
        generate_recommendation_response,
    )

    # 노드 실행 타이밍 측정 시작
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        ranked = state.get("ranked_movies", [])
        existing_response = state.get("response", "")
        error = state.get("error")
        messages = list(state.get("messages", []))
        current_input = state.get("current_input", "")

        # 사용자 정보 추출 (몽글이에게 전달)
        emotion_obj = state.get("emotion")
        emotion_str = emotion_obj.emotion if emotion_obj else None
        preferences = state.get("preferences")
        clarification = state.get("clarification")

        # 에러 응답
        if error and not existing_response:
            response = "죄송해요, 지금은 추천이 어려워요. 다시 시도해주세요!"

        # 추천 응답: 몽글이가 Solar 데이터를 대화체로 변환
        elif ranked:
            response = await generate_recommendation_response(
                ranked_movies=ranked,
                emotion=emotion_str,
                preferences=preferences,
                user_message=current_input,
            )

            # ── 저품질 결과 안내 (Phase Q-2.2) ──
            # 모든 추천 영화가 포스터+줄거리 모두 부족하면 사용자에게 조건 완화 안내.
            # 프론트엔드에서 "No Poster" + 빈 설명 카드만 보이는 최악 UX 방지.
            high_quality_count = sum(
                1 for m in ranked
                if (m.poster_path and m.poster_path.strip())
                and (m.overview and len(m.overview.strip()) >= 20)
            )
            if high_quality_count == 0:
                response += "\n\n조건에 딱 맞는 영화를 찾기 어려웠어요. 조건을 조금 넓혀주시면 더 좋은 추천을 드릴 수 있을 것 같아요!"

        # 후속 질문 응답: clarification 텍스트를 그대로 유지 (vLLM 재작성 스킵)
        elif existing_response and clarification:
            # 2026-04-15 중복 답변 제거:
            # question_generator 가 이미 `clarification` SSE 이벤트로 question 텍스트를 송신했다.
            # 이 자리에서 vLLM (`generate_question_response`) 으로 같은 질문을 다시 rewrite 하면
            #   ① 동일 텍스트가 `clarification` + `token` 두 경로로 내려가 클라이언트가 중복 렌더
            #   ② 후속 질문 턴마다 vLLM 한 번이 추가로 돌아 ~30s 레이턴시 추가
            # 두 문제가 함께 발생했다. 질문 문장은 이미 Solar (`generate_clarification`) 가
            # 자연스러운 대화체로 만들어 둔 상태이므로 그대로 `response` 에 넣고, SSE 레이어가
            # clarification 존재 시 `token` 이벤트를 suppress 한다 (graph.py).
            response = existing_response

        # 기존 response 사용 (일반 대화 — general_responder에서 이미 몽글이가 생성)
        elif existing_response:
            response = existing_response
        else:
            response = "무엇을 도와드릴까요? 영화 추천이 필요하시면 말씀해주세요!"

        # ── 마크다운 후처리: LLM이 지시를 무시하고 마크다운을 쓸 경우 제거 ──
        # **굵게**, ##제목, - 목록 등을 순수 텍스트로 정리
        response = re.sub(r"\*\*(.+?)\*\*", r"\1", response)  # **bold** → bold
        response = re.sub(r"\*(.+?)\*", r"\1", response)      # *italic* → italic
        response = re.sub(r"^#{1,6}\s+", "", response, flags=re.MULTILINE)  # ## 제목 → 제목
        response = re.sub(r"^[-*]\s+", "", response, flags=re.MULTILINE)    # - 목록 → 목록

        # assistant 메시지 추가 (timestamp + movies + 외부 지도 결과 포함 — 이전 대화 복원 시 카드/지도 모두 복원)
        assistant_msg: dict = {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # 추천 영화 카드를 메시지에 포함: 이전 채팅 로드 시 영화 카드도 복원
        if ranked:
            try:
                assistant_msg["movies"] = [
                    m.model_dump() if hasattr(m, "model_dump") else m for m in ranked
                ]
            except Exception:
                pass  # 직렬화 실패 시 movies 제외 (안전 처리)

        # 외부 지도 연동 결과(영화관 + 박스오피스 + 사용자 위치)도 메시지에 포함:
        # 이전 채팅 복원 시 TheaterCard / NowShowingPanel / 미니맵 사용자 마커 모두 복원되도록.
        # tool_results 는 tool_executor_node 가 채워둔 dict (없으면 빈 dict).
        tool_results = state.get("tool_results") or {}
        if tool_results:
            try:
                theaters = tool_results.get("theater_search")
                if isinstance(theaters, list) and theaters:
                    assistant_msg["theaters"] = theaters
                now_showing = tool_results.get("kobis_now_showing")
                if isinstance(now_showing, list) and now_showing:
                    assistant_msg["nowShowing"] = now_showing
                # state.location 은 Pydantic Location 또는 dict — 둘 다 직렬화
                loc = state.get("location")
                if loc is not None:
                    if hasattr(loc, "model_dump"):
                        assistant_msg["userLocation"] = loc.model_dump()
                    elif isinstance(loc, dict):
                        assistant_msg["userLocation"] = loc
            except Exception:
                # 외부 지도 결과 직렬화 실패는 graceful — 다른 필드는 살린다.
                logger.warning(
                    "response_formatter_external_map_serialize_failed",
                    session_id=session_id,
                )

        messages.append(assistant_msg)

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "response_formatted_node",
            response_length=len(response),
            has_movies=bool(ranked),
            used_mongle_llm=bool(ranked) or bool(clarification),
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
        # 폴백: 몽글이 호출 실패 시 기계적 조합
        fallback = "죄송해요, 응답을 구성하는 중 문제가 생겼어요."
        messages = list(state.get("messages", []))
        messages.append({
            "role": "assistant",
            "content": fallback,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
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
# 13. tool_executor_node — 도구 실행 (Phase 6 외부 지도 연동)
# ============================================================

# 사용자 메시지에서 위치 후보 토큰을 뽑아낼 때 쓰는 휴리스틱.
# "강남역 근처 영화관" → "강남역" / "홍대 입구 영화관 알려줘" → "홍대 입구"
# 카카오 keyword fallback 이 충분히 강건하므로 정확한 NER 까진 필요 없다.
_LOCATION_HINT_PATTERNS: list[re.Pattern] = [
    # "○○역" 으로 끝나는 지하철역 토큰 (예: 강남역, 홍대입구역)
    re.compile(r"([가-힣A-Za-z0-9]+역)"),
    # 행정구역/지명 + "근처/주변/근방" 앞부분 (예: "강남 근처", "신촌 주변")
    re.compile(r"([가-힣A-Za-z0-9]{2,15})\s*(?:근처|근방|주변|인근)"),
    # "○○동/○○구" 행정구역 — "○○시" 는 "서울시" 같이 너무 광역이라 카카오 검색 정확도 떨어져 제외.
    # 시 단위 검색이 필요하면 사용자가 "강남" 처럼 더 좁은 토큰을 보내거나 좌표를 직접 보내면 된다.
    re.compile(r"([가-힣A-Za-z0-9]+(?:동|구))"),
]


def _extract_location_hint(text: str) -> str | None:
    """
    사용자 자연어에서 위치 후보 토큰을 추출한다 (geocoding 도구 입력용).

    완벽한 NER 가 아니라 카카오 키워드 검색이 매칭해줄 만한 "지명스러운 한 덩어리" 만
    뽑아내면 충분하다. 매칭 실패 시 None.

    Args:
        text: 사용자 입력 (예: "강남역 근처 영화관 알려줘")

    Returns:
        매칭된 첫 토큰 (예: "강남역") 또는 None
    """
    if not text:
        return None
    for pattern in _LOCATION_HINT_PATTERNS:
        match = pattern.search(text)
        if match:
            hint = match.group(1).strip()
            if hint:
                return hint
    return None


def _resolve_movie_id_from_state(state: ChatAgentState) -> str | None:
    """
    info/booking 의도에서 사용할 movie_id 를 state 에서 회수한다.

    우선순위:
    1) ranked_movies[0].id — 직전 추천 결과의 1순위
    2) candidate_movies[0].id — 검색 결과 1순위
    """
    ranked = state.get("ranked_movies") or []
    if ranked:
        first = ranked[0]
        rid = getattr(first, "id", None)
        if rid:
            return str(rid)
    candidates = state.get("candidate_movies") or []
    if candidates:
        first = candidates[0]
        cid = getattr(first, "id", None)
        if cid:
            return str(cid)
    return None


def _format_tool_response(
    intent: str,
    tool_results: dict[str, Any],
    location_address: str | None,
) -> str:
    """
    도구 실행 결과를 사용자 친화적 한국어 응답으로 정리한다 (LLM 미사용 — 템플릿).

    영화관 카드 / 박스오피스 카드는 SSE event 로 별도 전송될 예정이므로,
    여기서는 헤더 한 줄 + 항목 요약만 만든다. response_formatter 가 이 문자열을
    SSE token 이벤트로 흘려보낸다.

    Args:
        intent: 사용자 의도
        tool_results: execute_tool 반환값 (도구 이름 → 결과)
        location_address: 사용자 위치 (자연어 헤더에 포함)

    Returns:
        사용자에게 보여줄 정리된 텍스트
    """
    parts: list[str] = []

    # ── theater 의도: 영화관 N곳 + 현재 박스오피스 Top-N ──
    if intent in ("theater", "booking"):
        theaters = tool_results.get("theater_search")
        if isinstance(theaters, list) and theaters:
            head = f"{location_address} 근처" if location_address else "근처"
            parts.append(f"{head}에서 가까운 영화관 {len(theaters)}곳을 찾았어요.")
            # 상위 3곳만 텍스트 요약 — 나머지는 카드로 노출
            for t in theaters[:3]:
                distance = t.get("distance_m", 0)
                distance_text = f"{distance}m" if distance < 1000 else f"{distance / 1000:.1f}km"
                parts.append(f"• {t.get('name', '')} ({distance_text})")
        elif isinstance(theaters, str):
            # 도구가 안내 문자열을 직접 반환 (API 키 누락/타임아웃)
            parts.append(theaters)

        now_showing = tool_results.get("kobis_now_showing")
        if isinstance(now_showing, list) and now_showing:
            top_titles = [m.get("movie_nm", "") for m in now_showing[:5]]
            parts.append("")
            parts.append("지금 박스오피스 상위 영화: " + " / ".join(top_titles))

    # ── info 의도: 영화 상세 + OTT + 유사 영화 ──
    elif intent == "info":
        detail = tool_results.get("movie_detail")
        if isinstance(detail, dict) and detail.get("title"):
            title = detail.get("title", "")
            director = detail.get("director", "")
            runtime = detail.get("runtime", 0)
            head = f"'{title}'"
            if director:
                head += f" · 감독 {director}"
            if runtime:
                head += f" · {runtime}분"
            parts.append(head)
            overview = detail.get("overview", "")
            if overview:
                parts.append(overview[:200] + ("..." if len(overview) > 200 else ""))
        elif isinstance(detail, str):
            parts.append(detail)

        ott = tool_results.get("ott_availability")
        if isinstance(ott, list) and ott:
            parts.append("OTT: " + ", ".join(str(p) for p in ott[:5]))

        similar = tool_results.get("similar_movies")
        if isinstance(similar, list) and similar:
            sim_titles = [s.get("title", "") for s in similar[:3] if isinstance(s, dict)]
            if sim_titles:
                parts.append("비슷한 영화: " + " / ".join(sim_titles))

    if not parts:
        # 어떤 도구도 의미있는 결과를 못 줬을 때 — 솔직하게 안내
        return "관련 정보를 가져오지 못했어요. 잠시 후 다시 시도해주시거나, 영화 추천이 필요하면 말씀해주세요! 🎬"
    return "\n".join(parts)


@traceable(name="tool_executor_node", run_type="tool", metadata={"node": "13/13"})
async def tool_executor_node(state: ChatAgentState) -> dict:
    """
    외부 도구 실행 노드 (Phase 6 외부 지도 연동).

    처리 흐름:
    1) intent 확인 (info/theater/booking 만 처리)
    2) theater/booking 인데 location 미제공 → 메시지에서 지명 추출 → geocoding 으로 좌표 변환
       · 지명도 못 뽑으면 위치 재질의 안내 응답
    3) chains.execute_tool() 로 INTENT_TOOL_MAP 의 도구들을 병렬 실행
    4) 결과 dict 를 state.tool_results 에 저장 + _format_tool_response 로 자연어 응답 생성
    5) response_formatter 가 tool_results 를 SSE event 로 분기 송출 (별도 PR)

    Args:
        state: ChatAgentState (intent, location?, current_input, ranked/candidate_movies?)

    Returns:
        dict: tool_results, location?, response 업데이트
    """
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    try:
        intent_obj = state.get("intent")
        intent_str = intent_obj.intent if intent_obj else "unknown"
        current_input = state.get("current_input", "")

        # info/theater/booking 외 의도는 본 노드에서 처리하지 않는다 (라우팅 안전망)
        if intent_str not in ("info", "theater", "booking"):
            logger.info("tool_executor_node_unsupported_intent", intent=intent_str)
            return {"response": "영화 추천이 필요하시면 말씀해주세요! 🎬"}

        # ── 위치 해소 (theater/booking 만 해당) ──
        location: Location | None = state.get("location")
        location_dict: dict | None = None
        if intent_str in ("theater", "booking"):
            if location:
                # Client 가 이미 좌표를 보냈음
                location_dict = {
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "address": location.address,
                }
                external_map_location_source_total.labels(source="client_supplied").inc()
            else:
                # 메시지에서 지명 후보 추출 → geocoding 으로 좌표 변환
                hint = _extract_location_hint(current_input)
                if hint:
                    geo = await geocoding.ainvoke({"query": hint})
                    if geo and geo.get("latitude") and geo.get("longitude"):
                        location_dict = {
                            "latitude": geo["latitude"],
                            "longitude": geo["longitude"],
                            "address": geo.get("address") or hint,
                        }
                        location = Location(
                            latitude=geo["latitude"],
                            longitude=geo["longitude"],
                            address=geo.get("address") or hint,
                        )
                        external_map_location_source_total.labels(source="geocoded").inc()

            # 위치를 끝내 못 얻었으면 재질의 — execute_tool 호출조차 하지 않는다
            if not location_dict:
                msg = (
                    "어느 지역 근처에서 찾으실까요? 🗺️ "
                    "지하철역이나 동네 이름(예: '강남역', '홍대 입구', '잠실동')을 알려주세요."
                )
                elapsed_ms = (time.perf_counter() - node_start) * 1000
                logger.info(
                    "tool_executor_node_location_required",
                    intent=intent_str,
                    elapsed_ms=round(elapsed_ms, 1),
                    session_id=session_id,
                    user_id=user_id,
                )
                external_map_location_source_total.labels(source="missing").inc()
                return {"response": msg}

        # ── info 의도: state 에서 movie_id / movie_title 회수 ──
        movie_id = _resolve_movie_id_from_state(state) if intent_str == "info" else None
        movie_title = current_input if intent_str in ("info", "booking") else None

        # ── 도구 디스패치 (병렬) ──
        tool_results = await execute_tool(
            intent=intent_str,
            location=location_dict,
            movie_id=movie_id,
            movie_title=movie_title,
            user_id=user_id,
        )

        # ── 응답 생성 (LLM 미사용, 템플릿 기반) ──
        response = _format_tool_response(
            intent=intent_str,
            tool_results=tool_results,
            location_address=(location_dict or {}).get("address"),
        )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "tool_executor_node_done",
            intent=intent_str,
            executed_tools=list(tool_results.keys()),
            has_location=location is not None,
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )

        update: dict = {"tool_results": tool_results, "response": response}
        if location is not None:
            update["location"] = location
        return update

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "tool_executor_node_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        # 에러 전파 금지 — 친절한 fallback 메시지
        return {
            "tool_results": {},
            "response": "관련 정보를 가져오지 못했어요. 잠시 후 다시 시도해주세요! 🎬",
        }


# ============================================================
# 14. graph_traversal_node — relation Intent 전용 Neo4j 멀티홉 탐색
# ============================================================

@traceable(name="graph_traversal_node", run_type="chain", metadata={"node": "14/14"})
async def graph_traversal_node(state: ChatAgentState) -> dict:
    """
    relation Intent 전용 Neo4j 멀티홉 탐색 노드 (§관계_대사_검색_설계서.md §5.6).

    처리 흐름:
    1. LLM(extract_graph_query_plan)으로 GraphQueryPlan 추출
       - query_type: chain / intersection / person_filmography
       - start_entity, hop_genre, persons 등 탐색 파라미터 구조화
    2. graph_cypher_builder로 매개변수화된 Cypher 쿼리 생성
       - 모든 사용자 입력은 $param으로 전달 (인젝션 방지)
    3. search_neo4j_relation으로 Neo4j 멀티홉 탐색 실행
    4. 결과를 CandidateMovie 형태로 변환하여 candidate_movies에 저장
    5. recommendation_ranker → explanation_generator → response_formatter 흐름으로 직행
       (preference_refiner / rag_retriever 스킵)

    결과 없음:
    - candidate_movies=[] 일 때 final_answer를 설정하고 response_formatter로 이동.
    - response_formatter는 final_answer를 그대로 응답으로 사용한다.

    에러:
    - 모든 예외를 try/except로 처리, 에러 전파 금지.
    - 에러 시 final_answer에 안내 메시지를 설정한다.

    Args:
        state: ChatAgentState (current_input 필요)

    Returns:
        dict: graph_query_plan, traversal_results, candidate_movies,
              needs_clarification, [final_answer] 업데이트
    """
    node_start = time.perf_counter()
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    current_input: str = state.get("current_input", "")

    logger.info(
        "graph_traversal_node_start",
        input_preview=current_input[:80],
        session_id=session_id,
        user_id=user_id,
    )

    try:
        # 1. 그래프 탐색 계획 추출 (LLM)
        # LLM 장애 시 extract_graph_query_plan 내부에서 _DEFAULT_PLAN 반환 (에러 전파 없음)
        from monglepick.chains.graph_query_chain import extract_graph_query_plan
        from monglepick.rag.hybrid_search import search_neo4j_relation

        plan = await extract_graph_query_plan(current_input)
        logger.info(
            "graph_traversal_plan_ready",
            query_type=plan.get("query_type"),
            start_entity=plan.get("start_entity"),
            hop_genre=plan.get("hop_genre"),
            persons=plan.get("persons"),
        )

        # 2. Neo4j 멀티홉 탐색 실행
        # 오류 시 search_neo4j_relation 내부에서 [] 반환 (에러 전파 없음)
        raw_results = await search_neo4j_relation(
            graph_query_plan=plan,
            top_k=20,
        )

        # 3. SearchResult → CandidateMovie 변환
        # rrf_score 필드에 relation_score(0~1 정규화)를 저장하여
        # recommendation_ranker가 기존 흐름과 동일하게 처리할 수 있도록 한다.
        candidates: list[CandidateMovie] = []
        for r in raw_results:
            candidates.append(
                CandidateMovie(
                    # CandidateMovie의 PK는 'id' (models.py 기준)
                    id=r.movie_id,
                    title=r.title,
                    rrf_score=r.score,                  # 0~1 정규화된 relation_score
                    retrieval_source="neo4j_relation",
                    # metadata에서 추가 필드 복원
                    # 2026-04-15: hybrid_search Neo4j 결과 metadata key 를 `popularity_score`
                    # 로 통일했으므로 그대로 읽음. 하위 호환 fallback 으로 `popularity` 도 유지.
                    rating=float(r.metadata.get("rating") or 0.0),
                    popularity_score=float(
                        r.metadata.get("popularity_score")
                        or r.metadata.get("popularity")
                        or 0.0
                    ),
                )
            )

        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.info(
            "graph_traversal_node_done",
            candidate_count=len(candidates),
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
        )

        # 4. 결과 없음 처리: 찾지 못했을 때 안내 메시지를 설정하고 응답 포맷터로 직행
        if not candidates:
            entity_hint = (
                plan.get("start_entity")
                or (", ".join(plan.get("persons") or []))
                or "해당 인물"
            )
            no_result_msg = (
                f"'{entity_hint}' 관련 영화를 그래프에서 찾지 못했어요. "
                "인물명을 한국어로 정확히 입력하거나 검색어를 바꿔서 다시 시도해 주세요."
            )
            return {
                "graph_query_plan": plan,
                "traversal_results": [],
                "candidate_movies": [],
                "needs_clarification": False,
                "final_answer": no_result_msg,
            }

        # 5. 정상 결과: candidate_movies를 채우고 recommendation_ranker로 진행
        return {
            "graph_query_plan": plan,
            "traversal_results": [r.metadata for r in raw_results],
            "candidate_movies": candidates,
            "needs_clarification": False,
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - node_start) * 1000
        logger.error(
            "graph_traversal_node_error",
            error=str(e),
            error_type=type(e).__name__,
            stack_trace=traceback.format_exc(),
            elapsed_ms=round(elapsed_ms, 1),
            session_id=session_id,
            user_id=user_id,
        )
        # 에러 전파 금지: 안내 메시지로 응답 포맷터에서 처리
        return {
            "graph_query_plan": None,
            "traversal_results": [],
            "candidate_movies": [],
            "needs_clarification": False,
            "final_answer": "관계 기반 영화 검색 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요.",
        }
