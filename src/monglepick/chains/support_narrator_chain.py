"""
고객센터 AI 에이전트 v4 Narrator 체인 — 다중 hop 진단 답변.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §3.3 (narrator 확장)

역할:
- 단일 hop 의 단순 FAQ/정책 답변 대신, 다중 hop 에서 누적된 본인 데이터와
  정책 RAG 청크를 종합해 **진단 답변** 을 생성한다.
- "포인트 안 들어왔어요" → 포인트 이력 + 정책 RAG 를 같이 참고해
  "어제 리뷰 작성분은 24시간 뒤 오늘 23:50 에 적립 예정이에요" 같은 원인 진단을 내린다.

구조:
- `build_tool_results_table()`:  tool_call_history + tool_results_cache → 읽기 쉬운 표 문자열
- `build_rag_context_for_narrator()`: rag_chunks → narrator 프롬프트용 인용 블록
- `generate_narrator_response()`:  위 두 컨텍스트 + Solar Pro → 진단 답변 텍스트

참조 패턴:
- `agents/admin_assistant/nodes.py` narrator 노드 (다중 hop 결과 종합 모드)
- `agents/support_assistant/nodes.py` _generate_with_solar (기존 단일 hop narrator)
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from monglepick.llm.factory import get_solar_api_llm
from monglepick.prompts.support_assistant import (
    SUPPORT_NARRATOR_SYSTEM_PROMPT,
    SUPPORT_NARRATOR_HUMAN_PROMPT,
)

logger = structlog.get_logger()

# narrator 답변 실패 시 기본 fallback 메시지
_NO_RESULT_FALLBACK = (
    "죄송해요, 지금 당장 해당 내용을 찾지 못했어요. "
    "'문의하기' 탭에서 1:1 티켓으로 남겨주시면 담당자가 확인해 드릴게요."
)

# 게스트 로그인 권유 suffix (personal_data + is_guest 조합에서 붙임)
_LOGIN_REQUIRED_SUFFIX = (
    "\n\n로그인하시면 본인 계정 데이터를 직접 확인해 더 정확히 도와드릴 수 있어요."
)


# ============================================================
# 컨텍스트 빌더 — tool_results_cache → 표 문자열
# ============================================================

def build_tool_results_table(
    tool_call_history: list[dict],
    tool_results_cache: dict[str, Any],
) -> str:
    """
    tool_call_history 와 tool_results_cache 를 narrator 프롬프트용 표 문자열로 직렬화한다.

    형식 (각 tool 결과 블록):
    ```
    [Tool 1: lookup_my_ai_quota]
    상태: ok=True
    결과: {"dailyAiUsed": 2, "dailyAiLimit": 3, "remainingAiBonus": 0, ...}

    [Tool 2: lookup_my_point_history]
    상태: ok=True
    결과: [{"amount": 10, "type": "EARN", "source": "REVIEW", "createdAt": "2026-04-27T23:50:00"}, ...]
    ```

    - ok=False 인 경우: error 내용만 표시 (data 없음)
    - data 가 list 인 경우: 상위 5건만 포함 (토큰 절약)
    - data 가 dict 인 경우: 전체 포함 (일반적으로 작음)

    Args:
        tool_call_history: observation 노드가 append 한 이력.
            각 항목: {"hop": int, "tool_name": str, "ok": bool, "error": str|None}
        tool_results_cache: tool_executor 가 저장한 결과. ref_id = "{tool_name}_{hop-1}".

    Returns:
        표 문자열. 이력이 없으면 "(조회된 데이터 없음)".
    """
    if not tool_call_history:
        return "(조회된 데이터 없음)"

    blocks: list[str] = []
    for idx, entry in enumerate(tool_call_history, start=1):
        tool_name = entry.get("tool_name", "?")
        ok = entry.get("ok", False)
        hop = entry.get("hop", idx)
        error = entry.get("error")

        # ref_id: tool_name_{hop-1} — tool_executor 가 Phase 1 단일 hop 에서는 _0 으로 저장
        ref_id = f"{tool_name}_{hop - 1}"
        result = tool_results_cache.get(ref_id, {})

        header = f"[Tool {idx}: {tool_name}]"
        if not ok or not result.get("ok"):
            err_msg = error or result.get("error", "unknown_error")
            blocks.append(f"{header}\n상태: ok=False\n오류: {err_msg}")
            continue

        data = result.get("data", {})
        # list 는 상위 5건으로 절단
        if isinstance(data, list):
            data_to_show = data[:5]
            suffix = f"... (총 {len(data)}건)" if len(data) > 5 else ""
        else:
            data_to_show = data
            suffix = ""

        try:
            data_str = json.dumps(data_to_show, ensure_ascii=False, default=str)
        except Exception:
            data_str = str(data_to_show)

        blocks.append(
            f"{header}\n상태: ok=True\n결과: {data_str}{suffix}"
        )

    return "\n\n".join(blocks) if blocks else "(조회된 데이터 없음)"


# ============================================================
# 컨텍스트 빌더 — rag_chunks → 인용 블록
# ============================================================

def build_rag_context_for_narrator(rag_chunks: list[dict]) -> str:
    """
    Qdrant 정책 RAG 청크 리스트를 narrator 프롬프트용 인용 블록으로 변환한다.

    상위 3건 청크 본문 + 섹션 제목을 사용한다 (Solar Pro max_tokens=2048 여유 확보).

    Args:
        rag_chunks: tool_executor(lookup_policy) 가 state.rag_chunks 에 저장한 리스트.
            각 항목: {"doc_id", "section", "headings", "policy_topic", "text", "score"}

    Returns:
        인용 블록 문자열. 청크가 없으면 "(참고 정책 없음)".
    """
    if not rag_chunks:
        return "(참고 정책 없음)"

    lines: list[str] = []
    for i, chunk in enumerate(rag_chunks[:3], start=1):
        text = (chunk.get("text") or "").strip()
        section = (chunk.get("section") or "").strip()
        policy_topic = (chunk.get("policy_topic") or "").strip()
        if not text:
            continue
        # 섹션 + 주제 헤더 구성
        header_parts = []
        if section:
            header_parts.append(section)
        if policy_topic:
            header_parts.append(f"주제={policy_topic}")
        header = f"[정책 {i}]" + (f" ({', '.join(header_parts)})" if header_parts else "")
        lines.append(f"{header}\n{text}")

    return "\n\n".join(lines) if lines else "(참고 정책 없음)"


# ============================================================
# 핵심 함수 — Solar Pro 진단 답변 생성
# ============================================================

async def generate_narrator_response(
    user_message: str,
    intent_kind: str,
    intent_confidence: float,
    tool_call_history: list[dict],
    tool_results_cache: dict[str, Any],
    rag_chunks: list[dict],
    is_guest: bool,
    hop_count: int,
    fallback: str = _NO_RESULT_FALLBACK,
    history_context: str = "",
) -> str:
    """
    다중 hop 결과를 종합해 Solar Pro 로 진단 답변 텍스트를 생성한다.

    단일 hop 의 단순 FAQ/RAG 인용 답변과 달리, 본인 데이터(ai_quota, point_history 등)와
    정책 RAG 청크를 함께 참고해 원인 진단 + 행동 안내를 포함한 4~7문장 답변을 만든다.

    처리 흐름:
    1. build_tool_results_table() 로 누적 결과 표 생성
    2. build_rag_context_for_narrator() 로 RAG 인용 블록 생성
    3. SUPPORT_NARRATOR_SYSTEM_PROMPT + SUPPORT_NARRATOR_HUMAN_PROMPT 로 Solar Pro 호출
    4. 게스트 + personal_data 이면 _LOGIN_REQUIRED_SUFFIX 추가
    5. 실패 시 fallback 반환 (에러 전파 금지)

    Args:
        user_message:      현재 턴 사용자 발화
        intent_kind:       SupportIntent.kind
        intent_confidence: SupportIntent.confidence
        tool_call_history: 누적 tool 호출 이력 (observation 노드 append)
        tool_results_cache: 누적 tool 결과 (ref_id → result)
        rag_chunks:        policy RAG 청크 리스트
        is_guest:          비로그인 게스트 여부
        hop_count:         실행된 총 hop 수
        fallback:          LLM 실패 시 반환할 기본 텍스트

    Returns:
        narrator 가 생성한 진단 답변 텍스트. 실패 시 fallback.
    """
    # 컨텍스트 빌드
    tool_results_table = build_tool_results_table(tool_call_history, tool_results_cache)
    rag_context = build_rag_context_for_narrator(rag_chunks)

    try:
        llm = get_solar_api_llm(temperature=0.3)

        # 멀티턴 컨텍스트: 멀티라인 prefix 로 사용자 발화 앞에 [이전 대화] 블록 삽입.
        # SUPPORT_NARRATOR_HUMAN_PROMPT 의 placeholder 와 호환되도록 user_message
        # 자체에 prefix 를 합성한다 (템플릿 변수 추가 회피 → 호환성 유지).
        if history_context:
            user_message_with_history = (
                f"[이전 대화]\n{history_context}\n\n[현재 발화]\n{user_message}"
            )
        else:
            user_message_with_history = user_message

        human_content = SUPPORT_NARRATOR_HUMAN_PROMPT.format(
            user_message=user_message_with_history,
            intent_kind=intent_kind,
            intent_confidence=float(intent_confidence),
            hop_count=hop_count,
            tool_results_table=tool_results_table,
            rag_context=rag_context,
        )
        messages = [
            SystemMessage(content=SUPPORT_NARRATOR_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]
        response = await llm.ainvoke(messages)
        text = (getattr(response, "content", None) or "").strip()

        if not text:
            logger.warning(
                "support_narrator_chain_empty_response",
                intent_kind=intent_kind,
                hop_count=hop_count,
            )
            text = fallback

        # 서비스 이름 환각 후처리 — Solar Pro 도 가끔 '몽글' 단독 표기를 출력함.
        # nodes.py 와 동일한 치환 규칙으로 '몽글픽' 으로 강제 교정한다.
        # (순환 import 방지: 노드 모듈에서 import 하지 않고 동일 규칙을 인라인.)
        for src, dst in (
            ("몽블랑", "몽글픽"),
            ("몽블랭", "몽글픽"),
            ("몽글 ", "몽글픽 "),
            ("몽글의", "몽글픽의"),
            ("몽글에서", "몽글픽에서"),
            ("몽글이라는", "몽글픽이라는"),
            ("몽글 서비스", "몽글픽 서비스"),
            ("몽글 고객센터", "몽글픽 고객센터"),
            ("몽글 챗봇", "몽글픽 챗봇"),
        ):
            text = text.replace(src, dst)

        # 게스트 + personal_data: 로그인 권유 suffix 추가
        if is_guest and intent_kind == "personal_data":
            text = text + _LOGIN_REQUIRED_SUFFIX

        logger.info(
            "support_narrator_chain_done",
            intent_kind=intent_kind,
            hop_count=hop_count,
            is_guest=is_guest,
            rag_chunk_count=len(rag_chunks),
            text_length=len(text),
        )
        return text

    except Exception as exc:  # noqa: BLE001 — 에러 전파 금지
        logger.warning(
            "support_narrator_chain_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            intent_kind=intent_kind,
            hop_count=hop_count,
        )
        return fallback
