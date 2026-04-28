"""
관리자 AI 에이전트 Tool 후보 필터 — 규칙 기반 (Qdrant 없음).

설계 배경 (2026-04-23):
- Tool 이 76개로 늘면서 매 턴 전부 `bind_tools` 에 싣기 부담스럽다
  (Solar-pro 프롬프트 토큰, tool 선택 정확도 저하).
- 하지만 76개뿐이고 이름 컨벤션이 명확해 **임베딩 검색까지는 불필요**하다.
  - `stats_*` / `dashboard_*` / `*_draft` / `goto_*` / 그 외 read
- intent_classifier 가 이미 `stats | query | action | smalltalk | report | sql` 로
  1차 분류해 주므로, kind 별로 카테고리를 매핑하면 20~30개로 쉽게 좁혀진다.

본 모듈은 **Qdrant · 임베딩 · 외부 의존 0** 으로 후보 tool 이름을 반환한다.
Fallback 은 role_allowed 전체 (완전 빈 결과 방지).

공개 함수: `shortlist_tools_by_category()`
"""

from __future__ import annotations

from typing import Any

import structlog

from monglepick.tools.admin_tools import (
    ADMIN_TOOL_REGISTRY,
    ToolSpec,
    list_tools_for_role,
)

logger = structlog.get_logger(__name__)


# ============================================================
# Tool kind 분류 — 이름 컨벤션 기반
# ============================================================

def classify_tool_kind(tool_name: str) -> str:
    """
    tool 이름만 보고 kind 를 판정한다. 규칙 우선순위:

    1. `*_draft` 로 끝남                        → "draft"   (폼 prefill 반환, Backend 쓰기 없음)
    2. `goto_*` 로 시작                         → "navigate" (화면 이동 링크 반환)
    3. `stats_*` 또는 `dashboard_*` 로 시작     → "stats"    (KPI/통계 조회)
    4. 그 외                                   → "read"     (일반 조회)

    이 분류는 설계서 §4.1~4.3 의 3종 분류와 정확히 일치한다.
    """
    if tool_name.endswith("_draft"):
        return "draft"
    if tool_name.startswith("goto_"):
        return "navigate"
    if tool_name.startswith("stats_") or tool_name.startswith("dashboard_"):
        return "stats"
    return "read"


# ============================================================
# Intent kind → 허용 tool kind 매핑
# ============================================================
# intent_classifier 가 돌려주는 kind 별로 "어떤 분류의 tool 을 보여줄지" 를 결정.
# 잘못 분류된 요청에 대비해 **read 는 대부분의 intent 에서 함께 허용** 한다
# (예: action 으로 분류됐지만 실제로는 조회만 해도 되는 경우).

_INTENT_TO_KINDS: dict[str, set[str]] = {
    # 통계/KPI 조회 → stats 로 시작하는 것들만. read 는 포함하지 않음(너무 넓어짐).
    "stats": {"stats"},
    # 일반 조회 — read 전체 + stats(수치 요청이 섞일 수 있음) + navigate(상세 화면 이동 겸용).
    # draft 는 제외.
    "query": {"read", "stats", "navigate"},
    # 쓰기 지향 — draft(폼 채우기) + navigate(위험 작업 화면 이동) + read(대상 검색용).
    # stats 는 불필요.
    "action": {"draft", "navigate", "read"},
    # 보고서(Phase 4, 2026-04-27) — stats + read + navigate 동시 허용.
    # ReAct 루프에서 통계 KPI(stats_*) 와 도메인 조회(reports_list/posts_list 등) 를 묶어
    # 종합 요약하고, 깊이 들여다볼 항목은 navigate(goto_*) 로 화면 이동을 권장한다.
    # draft 는 보고서 작성 행위가 아니므로 제외(별도 의도이면 action 으로 분류됨).
    "report": {"stats", "read", "navigate"},
    # SQL 은 미지원이라 사실상 smart_fallback 경로. 모든 것 허용해도 무해.
    "sql": {"read", "stats", "navigate", "draft"},
    # smalltalk 은 tool_selector 까지 오지 않지만 혹시 대비.
    "smalltalk": {"read", "stats", "navigate", "draft"},
}


# ============================================================
# 도메인 키워드 힌트 — 메시지 내 키워드로 후보를 더 좁힘
# ============================================================
# 키워드가 하나도 매칭되지 않으면 힌트 없이 카테고리 필터 결과를 그대로 반환.
# 여러 키워드가 매칭되면 합집합.
# 주의: 과도한 정규화는 오히려 false negative 유발하므로 포괄적인 한두 keyword 만 매핑.

_DOMAIN_HINTS: list[tuple[tuple[str, ...], set[str]]] = [
    # (매치 키워드 튜플, 선호 tool 이름/접두사 세트)
    # ── 공지/FAQ/도움말/배너/퀴즈 ──
    (("공지", "공지사항"), {"notice_", "notices_list", "notice_detail"}),
    (("faq", "자주묻는", "자주 묻는"), {"faq_", "faqs_list"}),
    (("도움말", "help"), {"help_article", "help_articles_list"}),
    (("배너",), {"banner_", "banners_list"}),
    (("퀴즈",), {"quiz_", "quizzes_list"}),
    (("약관",), {"term_", "terms_list"}),
    (("월드컵", "worldcup"), {"worldcup_candidate_"}),
    (("추천 칩", "추천칩", "chat_suggestion", "suggestion"), {"chat_suggestion_", "chat_suggestions_list"}),
    # ── 사용자·제재 ──
    (("유저", "사용자", "회원"), {"users_list", "user_detail", "user_activity", "user_rewards",
                               "user_points_history", "user_payments",
                               "user_suspension_history", "goto_user_"}),
    (("정지", "블락", "차단"), {"goto_user_suspend", "user_suspension_history"}),
    (("복구", "해제"), {"goto_user_activate", "user_suspension_history"}),
    (("권한", "역할"), {"goto_user_role_change"}),
    # ── 결제·포인트·구독 ──
    (("결제", "주문", "order"), {"orders_list", "order_detail", "goto_order_"}),
    (("환불", "refund"), {"goto_order_refund", "order_detail", "orders_list"}),
    (("구독", "subscription"), {"subscriptions_list", "goto_subscription_"}),
    (("포인트",), {"point_histories", "point_items", "goto_points_adjust",
                  "stats_point_economy", "point_pack_draft", "reward_policy_draft"}),
    (("이용권", "토큰", "token"), {"goto_token_grant"}),
    # ── 신고·리뷰·게시글 ──
    (("신고", "report"), {"reports_list", "goto_report_detail"}),
    (("혐오", "욕설", "toxicity"), {"toxicity_list"}),
    (("게시글", "post"), {"posts_list"}),
    (("리뷰", "review"), {"reviews_list", "review_verifications_list",
                         "review_verification_detail", "review_verifications_overview"}),
    # ── 고객센터 ──
    (("티켓", "ticket", "문의"), {"tickets_list", "ticket_detail", "goto_ticket_"}),
    # ── 감사로그·관리자 계정 ──
    (("감사", "audit"), {"audit_logs_list", "goto_audit_log"}),
    (("관리자 계정", "admin 계정"), {"admins_list"}),
    # ── AI 운영 ──
    (("챗봇", "chatbot"), {"chatbot_sessions_list", "chatbot_session_messages", "chatbot_stats"}),
    # ── 시스템 ──
    (("시스템", "상태", "헬스"), {"system_services_status", "system_config"}),
    # ── 대시보드 ──
    (("대시보드", "kpi", "지표"), {"dashboard_kpi", "dashboard_trends", "dashboard_recent"}),
    # ── 매출/가입/보존/퍼널 ──
    (("매출", "revenue"), {"stats_revenue"}),
    (("가입", "신규"), {"dashboard_kpi", "stats_retention"}),
    (("이탈", "churn"), {"stats_churn_risk"}),
    (("퍼널", "funnel", "전환"), {"stats_funnel"}),
    (("검색",), {"stats_search_popular"}),
    (("추천 성능", "추천 품질"), {"stats_recommendation"}),
]


def _extract_hinted_names(user_message: str) -> set[str]:
    """
    user_message 에서 도메인 키워드를 찾아 선호 tool 이름 집합을 반환.

    매칭된 각 항목은 접두사(`notice_` 등) 또는 완전 이름(`faqs_list` 등) 이 섞여 있으므로
    이름 매칭 단계는 호출측에서 `startswith` 또는 `==` 혼용.
    """
    msg = user_message.lower()
    hinted: set[str] = set()
    for keywords, names in _DOMAIN_HINTS:
        for kw in keywords:
            if kw.lower() in msg:
                hinted.update(names)
                break
    return hinted


def _name_matches_hint(tool_name: str, hint: str) -> bool:
    """hint 가 접두사(`*_`) 면 startswith, 완전 이름이면 == 매칭."""
    if hint.endswith("_"):
        return tool_name.startswith(hint)
    return tool_name == hint


# ============================================================
# 공개 API
# ============================================================

def shortlist_tools_by_category(
    *,
    user_message: str,
    admin_role: str,
    intent_kind: str,
    max_tools: int = 30,
) -> list[str]:
    """
    카테고리·키워드 기반으로 LLM 에 bind 할 tool 이름을 최대 `max_tools` 개 반환한다.

    알고리즘:
    1. `list_tools_for_role(admin_role)` 로 권한 필터 (SUPER_ADMIN 은 전체).
    2. intent_kind → allowed_kinds 매핑. 카테고리에 속하지 않는 tool 제거.
    3. user_message 에서 도메인 키워드 힌트를 추출 → 힌트에 매칭되는 tool 을 우선순위 상단.
    4. 결과가 비면 `list_tools_for_role(admin_role)` 전체로 fallback.
    5. `max_tools` 개로 절단.

    finish_task (가상 tool) 은 여기서 반환하지 않는다 — select_admin_tool() 이 항상 bind.

    Args:
        user_message: 현재 턴 관리자 발화 (원문).
        admin_role: 정규화된 AdminRoleEnum 값 (빈 문자열이면 빈 리스트 반환).
        intent_kind: intent_classifier 가 내려준 kind 문자열.
        max_tools: bind_tools 에 실을 최대 개수 상한 (기본 30).

    Returns:
        tool 이름 list. 중복 없음, 키워드 매칭 우선 정렬.
    """
    if not admin_role:
        logger.info("tool_filter_empty_role")
        return []

    # 1) Role 필터
    role_allowed_specs: list[ToolSpec] = list_tools_for_role(admin_role)
    role_allowed: set[str] = {s.name for s in role_allowed_specs}

    # 2) Category 필터
    allowed_kinds: set[str] = _INTENT_TO_KINDS.get(intent_kind, {"read", "stats", "navigate", "draft"})
    category_allowed: set[str] = {
        name for name in role_allowed if classify_tool_kind(name) in allowed_kinds
    }

    # 3) Keyword hints — 카테고리 안에서 우선순위
    hinted = _extract_hinted_names(user_message)
    hinted_matched: list[str] = []
    for name in sorted(category_allowed):
        if any(_name_matches_hint(name, h) for h in hinted):
            hinted_matched.append(name)

    non_hinted: list[str] = sorted(category_allowed - set(hinted_matched))

    # 4) Fallback — 카테고리 결과 빔 → role 전체
    if not category_allowed:
        logger.warning(
            "tool_filter_category_empty_fallback_role",
            intent_kind=intent_kind,
            role=admin_role,
        )
        merged = sorted(role_allowed)
    else:
        merged = hinted_matched + non_hinted

    # 5) max_tools 절단
    shortlisted = merged[: max_tools]

    logger.info(
        "tool_filter_shortlist",
        intent_kind=intent_kind,
        role=admin_role,
        role_total=len(role_allowed),
        category_total=len(category_allowed),
        hinted_hits=len(hinted_matched),
        returned=len(shortlisted),
    )
    return shortlisted
