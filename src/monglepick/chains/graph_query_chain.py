"""
그래프 탐색 계획 생성 체인.

relation Intent 처리 시 사용자 질문 → GraphQueryPlan 딕셔너리 변환.
구조화 추출에 특화된 intent_emotion LLM(Solar API hybrid / Qwen local)을 재사용한다.

처리 흐름:
1. GRAPH_QUERY_SYSTEM_PROMPT + 사용자 질문을 LLM에 전달
2. LLM 응답에서 JSON 블록 추출 (```json ... ``` 또는 중괄호 직접 파싱)
3. 파싱 성공 → GraphQueryPlan dict 반환
4. 파싱 실패 → _DEFAULT_PLAN에 raw_intent를 덮어씌워 안전하게 반환 (에러 전파 금지)
"""

from __future__ import annotations

import json
import time

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from monglepick.llm.factory import get_intent_emotion_llm, guarded_ainvoke
from monglepick.prompts.graph_query import GRAPH_QUERY_HUMAN_PROMPT, GRAPH_QUERY_SYSTEM_PROMPT

logger = structlog.get_logger()


# ============================================================
# 기본 플랜 (파싱 실패 시 폴백)
# ============================================================

# 에러/파싱 실패 시 반환할 안전한 기본 계획.
# query_type="chain", start_entity=None → build_cypher_from_plan에서 폴백 Cypher를 생성한다.
_DEFAULT_PLAN: dict = {
    "query_type": "chain",
    "start_entity": None,
    "start_relation": "DIRECTED",
    "hop_genre": None,
    "target_relation": "ACTED_IN",
    "depth": 2,
    "persons": [],
    "relation_type": None,
    "raw_intent": "",
}


# ============================================================
# JSON 추출 헬퍼
# ============================================================

def _extract_json_from_response(content: str) -> dict:
    """
    LLM 응답 문자열에서 JSON 딕셔너리를 추출한다.

    3단계 시도:
    1. ```json ... ``` 코드 블록 추출 후 파싱
    2. ``` ... ``` 코드 블록 추출 후 파싱
    3. 첫 번째 '{' ~ 마지막 '}' 직접 파싱

    Args:
        content: LLM 응답 원문

    Returns:
        파싱된 dict

    Raises:
        ValueError: 모든 시도 실패 시
    """
    # 시도 1: ```json ... ``` 블록
    if "```json" in content:
        inner = content.split("```json", 1)[1]
        inner = inner.split("```", 1)[0].strip()
        return json.loads(inner)

    # 시도 2: ``` ... ``` 블록 (언어 태그 없는 경우)
    if "```" in content:
        inner = content.split("```", 1)[1]
        inner = inner.split("```", 1)[0].strip()
        return json.loads(inner)

    # 시도 3: 중괄호로 직접 파싱
    start = content.find("{")
    end = content.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(content[start:end])

    raise ValueError("JSON 블록을 찾을 수 없습니다.")


# ============================================================
# 공개 함수: 그래프 탐색 계획 추출
# ============================================================

async def extract_graph_query_plan(user_query: str) -> dict:
    """
    사용자 쿼리에서 Neo4j 그래프 탐색 계획(GraphQueryPlan)을 추출한다.

    LLM에게 GRAPH_QUERY_SYSTEM_PROMPT와 사용자 질문을 전달하여
    query_type / start_entity / hop_genre 등 탐색 파라미터를 JSON으로 받는다.

    파싱 실패 또는 LLM 장애 시 _DEFAULT_PLAN을 반환한다 (에러 전파 금지).
    raw_intent에는 항상 원본 user_query를 기록한다.

    Args:
        user_query: 사용자 원문 질문 (예: "봉준호 감독이 찍은 스릴러에 나온 배우들의 영화")

    Returns:
        GraphQueryPlan dict. 필드:
        - query_type: "chain" | "intersection" | "person_filmography"
        - start_entity: 시작 인물명 (chain/person_filmography 전용)
        - start_relation: "DIRECTED" | "ACTED_IN"
        - hop_genre: 중간 장르 필터 (chain 전용, 없으면 None)
        - target_relation: "DIRECTED" | "ACTED_IN" (chain 전용)
        - depth: 탐색 깊이 (2 또는 3)
        - persons: 교집합 인물 목록 (intersection 전용)
        - relation_type: intersection 관계 유형 (intersection 전용)
        - raw_intent: 원본 사용자 질문
    """
    start_time = time.perf_counter()

    try:
        # intent_emotion LLM 재사용 — 구조화 추출에 특화된 모델 (Solar API hybrid / Qwen local)
        llm = get_intent_emotion_llm()

        messages = [
            SystemMessage(content=GRAPH_QUERY_SYSTEM_PROMPT),
            HumanMessage(
                content=GRAPH_QUERY_HUMAN_PROMPT.format(user_query=user_query)
            ),
        ]

        # guarded_ainvoke: 모델별 세마포어로 동시 호출 수 제한
        response = await guarded_ainvoke(llm, messages, model="graph_query", request_id="")

        # LangChain 응답에서 텍스트 추출
        content: str
        if hasattr(response, "content"):
            content = str(response.content)
        elif isinstance(response, str):
            content = response
        else:
            # 구조화 출력(Pydantic)으로 반환된 경우 — dict로 변환
            content = json.dumps(response.dict() if hasattr(response, "dict") else dict(response))

        # JSON 파싱
        plan = _extract_json_from_response(content)

        # raw_intent 보정: LLM이 빈 문자열을 반환하면 원본 쿼리로 채움
        if not plan.get("raw_intent"):
            plan["raw_intent"] = user_query

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "graph_query_plan_extracted",
            query_type=plan.get("query_type"),
            start_entity=plan.get("start_entity"),
            hop_genre=plan.get("hop_genre"),
            persons=plan.get("persons"),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return plan

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(
            "graph_query_plan_failed",
            error=str(e),
            error_type=type(e).__name__,
            user_query=user_query[:100],
            fallback=True,
            elapsed_ms=round(elapsed_ms, 1),
        )
        # 에러 전파 금지: 안전한 기본값 반환
        return {**_DEFAULT_PLAN, "raw_intent": user_query}
