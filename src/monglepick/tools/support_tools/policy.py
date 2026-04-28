"""
고객센터 AI 에이전트 v4 — `lookup_policy` tool.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5.2

몽글픽 운영 정책 (등급 혜택, AI 쿼터, 구독 플랜, 리워드 적립, 환불 정책 등) 을
Qdrant `support_policy_v1` 컬렉션에서 벡터 유사도 검색한다.

특징:
- requires_login=False — 정책 정보는 공개 데이터이므로 게스트도 호출 가능.
- 개인화 없음 — 사용자 계정 상태와 무관한 일반 정책 질문에 사용.
  개인화가 필요한 경우(내 포인트 잔액, 내 구독 상태 등)는 별도 tool 사용.
- 에러 시 ok=False 반환 (예외 전파 금지).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from monglepick.chains.support_policy_rag_chain import search_policy

from . import ToolContext, SupportToolSpec, register_support_tool


# ============================================================
# 입력 스키마 (LLM 이 JSON 으로 채울 인터페이스)
# ============================================================

class LookupPolicyArgs(BaseModel):
    """
    `lookup_policy` tool 입력 스키마.

    LLM 은 사용자 발화에서 핵심 키워드만 추출하여 query 에 담아야 한다.
    topic 은 질문 맥락을 파악하여 해당 카테고리를 지정하면 검색 정밀도가 높아진다.
    """

    query: str = Field(
        description=(
            "정책 RAG 검색 쿼리. "
            "사용자 발화에서 핵심 키워드만 추출한다. "
            "예: '브론즈 등급 AI 하루 몇 번' → 'BRONZE 등급 AI 하루 사용 횟수'"
        )
    )
    topic: str | None = Field(
        default=None,
        description=(
            "정책 토픽 필터. 지정하면 해당 카테고리 청크만 검색한다. "
            "grade_benefit — 등급별 혜택 (6등급 팝콘 테마, AI 일일 한도, 리워드 배율) | "
            "ai_quota — AI 쿼터 정책 (3-소스 순서, daily/monthly/purchased 소비 규칙) | "
            "subscription — 구독 플랜 (monthly_basic/premium/yearly 요금·혜택·해지) | "
            "refund — 환불 정책 (결제 취소, 부분 환불 조건) | "
            "reward — 리워드 적립 (출석 체크, 활동 포인트, 이벤트) | "
            "payment — 결제 수단·정책 (포인트 1P=10원, Toss Payments) | "
            "general — 일반 서비스 이용 안내. "
            "확신이 없으면 None 으로 두어 전체 컬렉션에서 검색한다."
        ),
    )


# ============================================================
# Handler
# ============================================================

async def _handle_lookup_policy(
    ctx: ToolContext,
    query: str,
    topic: str | None = None,
) -> dict:
    """
    lookup_policy tool 실행 핸들러.

    search_policy() 로 Qdrant 에서 정책 청크를 검색하고,
    narrator 노드가 인용할 수 있는 구조화된 dict 를 반환한다.

    반환 스키마:
        ok=True 시:
        {
            "ok": True,
            "data": {
                "chunks": [
                    {
                        "doc_id":       str,   # 문서 식별자
                        "section":      str,   # 섹션 식별자
                        "headings":     list[str],  # 헤딩 경로
                        "policy_topic": str,   # grade_benefit / ai_quota / ...
                        "text":         str,   # 청크 본문
                        "score":        float  # 유사도 (소수점 3자리 반올림)
                    },
                    ...
                ]
            }
        }

        ok=False 시:
        {
            "ok": False,
            "error": str  # 오류 메시지
        }

    Args:
        ctx:   런타임 컨텍스트 (request_id 로그 상관관계 추적에 사용)
        query: 검색 쿼리
        topic: 정책 토픽 필터 (None 이면 전체 검색)
    """
    try:
        # search_policy 내부에서 이미 에러 → 빈 리스트 처리.
        # 여기서는 그 결과를 dict 로 포장하는 역할만 한다.
        chunks = await search_policy(
            query=query,
            top_k=5,
            topic_filter=topic,
            request_id=getattr(ctx, "request_id", ""),
        )

        return {
            "ok": True,
            "data": {
                "chunks": [
                    {
                        "doc_id": c.doc_id,
                        "section": c.section,
                        "headings": c.headings,
                        "policy_topic": c.policy_topic,
                        "text": c.text,
                        "score": round(c.score, 3),
                    }
                    for c in chunks
                ]
            },
        }

    except Exception as exc:
        # search_policy 가 에러를 삼켜야 하지만 만일의 경우 이중 방어
        return {
            "ok": False,
            "error": f"정책 검색 실패: {exc}",
        }


# ============================================================
# 레지스트리 등록 (import side-effect)
# ============================================================

register_support_tool(
    SupportToolSpec(
        name="lookup_policy",
        description=(
            "몽글픽 운영 정책 (등급 혜택, AI 쿼터, 구독 플랜, 리워드 적립, 환불 정책 등) 을 "
            "검색합니다. 개인화가 필요 없는 일반 정책 질문에 사용하세요. "
            "예: '브론즈 등급 AI 몇 번?', '구독 해지하면 어떻게 돼요?', "
            "'포인트 어떻게 쌓여요?', '환불 언제까지 가능해요?'"
        ),
        args_schema=LookupPolicyArgs,
        handler=_handle_lookup_policy,
        requires_login=False,  # 정책 정보는 공개 — 게스트도 호출 가능
    )
)
