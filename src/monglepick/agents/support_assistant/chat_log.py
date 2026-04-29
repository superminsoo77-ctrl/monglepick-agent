"""
고객센터 챗봇 사용 통계·감사 로그.

매 턴 응답 직후 fire-and-forget 으로 `support_chat_log` 테이블에 INSERT 한다.

### 설계 의도 (2026-04-28)
- 사용자가 어떤 질문을 했고 봇이 어떤 의도로 분류했는지 / 1:1 유도가 됐는지 추적.
- 관리자 페이지에서 "어떤 질문이 자주 들어오는지", "1:1 유도 비율은 얼마인지" 분석.
- 응답 차단 금지 — INSERT 실패는 warning 으로만 남기고 사용자 응답에 영향 X.

### 테이블 DDL (Backend JPA `@Entity` + `ddl-auto=update` 가 자동 생성)
운영 환경에서는 Backend `SupportChatLog` 엔티티가 부팅 시 테이블을 자동 생성.
Agent 측은 INSERT 만 수행. 스키마는 Backend `SupportChatLog.java` 참조.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | BIGINT AUTO_INCREMENT PK | 서로게이트 |
| session_id | VARCHAR(64) | 세션 ID (LangGraph thread_id 와 동일) |
| user_id | VARCHAR(50) NULL | 로그인 사용자 ID. 게스트는 NULL |
| is_guest | TINYINT(1) | 게스트 여부 |
| user_message | TEXT | 원본 발화 |
| response_text | TEXT | 봇 응답 본문 |
| intent_kind | VARCHAR(32) | faq/personal_data/policy/redirect/smalltalk/complaint/unknown |
| intent_confidence | DECIMAL(3,2) | 0.00~1.00 |
| intent_reason | VARCHAR(255) | 분류 근거 한 줄 (잘림) |
| needs_human | TINYINT(1) | 1:1 유도 여부 |
| hop_count | INT | ReAct hop 수 |
| tool_calls_json | TEXT | tool_call_history JSON 직렬화 |
| created_at | DATETIME(3) | 자동 |

### 인덱스
- (created_at) — 시계열 통계
- (intent_kind, created_at) — 의도별 추이
- (needs_human, created_at) — 1:1 유도 비율 추이
- (session_id) — 세션 단위 트레이스
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import structlog

from monglepick.db.clients import get_mysql

logger = structlog.get_logger()

# fire-and-forget 단일 INSERT 의 타임아웃. 운영 DB 가 늦어도 사용자 응답은 즉시 끝낸다.
_INSERT_TIMEOUT_SEC = 3.0


async def insert_support_chat_log(
    session_id: str,
    user_id: str | None,
    is_guest: bool,
    user_message: str,
    response_text: str,
    intent_kind: str,
    intent_confidence: float,
    intent_reason: str,
    needs_human: bool,
    hop_count: int,
    tool_call_history: list[dict],
) -> None:
    """
    한 턴의 채팅 메타데이터를 `support_chat_log` 에 INSERT 한다.

    호출 측은 await 하지 말고 `asyncio.create_task(...)` 로 fire-and-forget.
    이 함수 내부에서 추가로 `asyncio.wait_for` 타임아웃을 적용해 운영 DB 지연이
    백그라운드 task 누적으로 이어지지 않도록 한다.

    예외는 모두 흡수 — 통계 INSERT 실패가 사용자 응답을 차단해서는 안 된다.
    """
    try:
        await asyncio.wait_for(
            _do_insert(
                session_id=session_id,
                user_id=user_id,
                is_guest=is_guest,
                user_message=user_message,
                response_text=response_text,
                intent_kind=intent_kind,
                intent_confidence=intent_confidence,
                intent_reason=intent_reason,
                needs_human=needs_human,
                hop_count=hop_count,
                tool_call_history=tool_call_history,
            ),
            timeout=_INSERT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "support_chat_log_insert_timeout",
            session_id=session_id,
            timeout_sec=_INSERT_TIMEOUT_SEC,
        )
    except Exception as exc:  # noqa: BLE001 — 응답 차단 금지
        logger.warning(
            "support_chat_log_insert_failed",
            session_id=session_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )


async def _do_insert(
    session_id: str,
    user_id: str | None,
    is_guest: bool,
    user_message: str,
    response_text: str,
    intent_kind: str,
    intent_confidence: float,
    intent_reason: str,
    needs_human: bool,
    hop_count: int,
    tool_call_history: list[dict],
) -> None:
    """
    실제 INSERT 본체. _do_insert 단계에서 발생한 예외는 호출자(insert_support_chat_log)
    가 흡수한다.
    """
    pool = await get_mysql()

    # tool_call_history JSON 직렬화 — TEXT 컬럼에 저장.
    # default=str 로 datetime 등도 안전하게 직렬화.
    try:
        tool_calls_json = json.dumps(tool_call_history, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        tool_calls_json = "[]"

    # intent_reason 길이 제한 (255자 컬럼)
    reason_truncated = (intent_reason or "")[:255]
    # confidence 는 DECIMAL(3,2) 범위 0.00~1.00 로 클램프
    conf_clamped = max(0.0, min(1.0, float(intent_confidence)))
    # 시간은 서버 NOW() 가 아닌 명시 UTC 로 — 멀티 인스턴스 시계 분산 회피
    now_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    sql = """
        INSERT INTO support_chat_log (
            session_id, user_id, is_guest,
            user_message, response_text,
            intent_kind, intent_confidence, intent_reason,
            needs_human, hop_count, tool_calls_json, created_at
        ) VALUES (
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s
        )
    """
    params = (
        session_id[:64],
        (user_id or None) if user_id else None,
        1 if is_guest else 0,
        user_message,
        response_text,
        intent_kind[:32],
        conf_clamped,
        reason_truncated,
        1 if needs_human else 0,
        int(hop_count),
        tool_calls_json,
        now_utc,
    )

    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
        await conn.commit()
