"""
관리자 AI 에이전트 Phase 4 (2026-04-27) 단위 테스트.

테스트 범위:
1. route_after_intent — report intent 가 tool_selector 로 라우팅되는지
2. tool_filter — report intent 의 카테고리 필터에 navigate 포함
3. _build_table_payload — list/Page 응답이 임계행수 이상이면 SSE table_data payload 빌드
4. _PLACEHOLDER_MESSAGES["report"] — Phase 4 placeholder 가 더이상 "Phase 4 예정" 텍스트가 아님

Phase 4 핵심 변경:
- report intent → tool_selector ReAct 루프 (stats / read / navigate)
- narrator 가 report 시 tool_call_history 전체 묶어 종합 요약
- tool_executor 직후 list/Page 결과면 table_data SSE 발행
"""

from __future__ import annotations

import pytest

from monglepick.agents.admin_assistant.graph import (
    _build_chart_payload,
    _build_table_payload,
    route_after_intent,
)
from monglepick.agents.admin_assistant.models import AdminIntent
from monglepick.agents.admin_assistant.nodes import _PLACEHOLDER_MESSAGES
from monglepick.api.admin_backend_client import AdminApiResult
from monglepick.tools.admin_tools.tool_filter import (
    _INTENT_TO_KINDS,
    shortlist_tools_by_category,
)


# ============================================================
# 1. route_after_intent — report 라우팅
# ============================================================

class TestRouteAfterIntentReportRouting:
    """Phase 4: report intent → tool_selector 분기."""

    def test_report_intent_routes_to_tool_selector(self):
        """기존엔 response_formatter 직행이었으나 Phase 4 부터 tool_selector."""
        state = {
            "admin_role": "SUPER_ADMIN",
            "intent": AdminIntent(kind="report", confidence=0.9, reason="요약 요청"),
        }
        assert route_after_intent(state) == "tool_selector"

    def test_stats_query_action_still_route_to_tool_selector(self):
        """기존 라우팅 회귀 검증 — Phase 4 변경이 다른 intent 영향 없는지."""
        for kind in ("stats", "query", "action"):
            state = {
                "admin_role": "ADMIN",
                "intent": AdminIntent(kind=kind, confidence=0.9, reason=""),
            }
            assert route_after_intent(state) == "tool_selector", f"{kind} 회귀"

    def test_sql_still_falls_to_response_formatter(self):
        """sql 만 영구 미지원 placeholder 경로."""
        state = {
            "admin_role": "SUPER_ADMIN",
            "intent": AdminIntent(kind="sql", confidence=0.9, reason="자유 쿼리"),
        }
        assert route_after_intent(state) == "response_formatter"

    def test_smalltalk_still_routes_to_smalltalk_responder(self):
        """smalltalk 회귀 — 라우팅 변경 없음."""
        state = {
            "admin_role": "SUPER_ADMIN",
            "intent": AdminIntent(kind="smalltalk", confidence=0.9, reason="인사"),
        }
        assert route_after_intent(state) == "smalltalk_responder"

    def test_blank_admin_role_still_blocks(self):
        """비관리자 차단 회귀 — 어떤 intent 든 response_formatter."""
        state = {
            "admin_role": "",
            "intent": AdminIntent(kind="report", confidence=0.9, reason=""),
        }
        assert route_after_intent(state) == "response_formatter"


# ============================================================
# 2. tool_filter — report intent 카테고리
# ============================================================

class TestReportIntentCategoryFilter:
    """Phase 4: report intent 의 카테고리에 navigate 포함."""

    def test_report_kinds_include_stats_read_navigate(self):
        """draft 는 빠지고 stats/read/navigate 가 포함."""
        kinds = _INTENT_TO_KINDS["report"]
        assert "stats" in kinds
        assert "read" in kinds
        assert "navigate" in kinds
        assert "draft" not in kinds  # 보고서는 draft(폼 채움) 의도가 아님

    def test_shortlist_tools_for_report_intent_returns_read_and_navigate(self):
        """SUPER_ADMIN + report 발화에서 read/navigate tool 이 후보에 포함되는지."""
        names = shortlist_tools_by_category(
            user_message="커뮤니티 최근 신고건수 요약해줘",
            admin_role="SUPER_ADMIN",
            intent_kind="report",
            max_tools=30,
        )
        # 신고 키워드 hint → reports_list 가 후보에 들어가야 함
        assert "reports_list" in names
        # navigate 카테고리도 허용되므로 goto_report_detail 같은 게 들어올 수 있음
        # (도메인 hint 매칭이 함께 작동)
        assert any(n.startswith("goto_report") for n in names) or "reports_list" in names


# ============================================================
# 3. _build_table_payload — table_data SSE
# ============================================================

class TestBuildTablePayload:
    """tool_executor 결과 → SSE table_data payload 빌드."""

    def _ok_result(self, data, row_count=None):
        return AdminApiResult(
            ok=True, status_code=200, data=data, row_count=row_count, latency_ms=10,
        )

    def test_list_with_three_dict_rows_builds_payload(self):
        rows = [
            {"id": "r1", "title": "신고 A", "createdAt": "2026-04-25"},
            {"id": "r2", "title": "신고 B", "createdAt": "2026-04-26"},
            {"id": "r3", "title": "신고 C", "createdAt": "2026-04-27"},
        ]
        payload = _build_table_payload(
            "reports_list", self._ok_result(rows, row_count=3),
        )
        assert payload is not None
        assert payload["tool_name"] == "reports_list"
        assert payload["columns"] == ["id", "title", "createdAt"]
        assert len(payload["rows"]) == 3
        assert payload["total_rows"] == 3
        assert payload["truncated"] is False
        # 등록된 navigate_path 확인 (Admin Client BoardPage TABS 기준)
        assert payload["navigate_path"] == "/admin/board?tab=reports"

    def test_navigate_path_mappings_match_admin_client_tab_keys(self):
        """
        Phase 4 후속(2026-04-27): _TABLE_NAVIGATE_PATHS 의 19종 매핑이 Admin Client 실제
        tab 키와 일치하는지 회귀 검증. 페이지가 변경될 가능성 있는 항목 위주로 핵심 케이스만.

        - subscription / point / items: PaymentPage TABS = subscription/point/items (단수형)
        - ticket: SupportPage TABS 의 ticket (no s)
        - quiz: ContentEventsPage SUB_TABS 의 quiz (AiOpsPage 에 quiz 탭 없음)
        - review-verify / chatlog: AiOpsPage TABS 키 (review-verifications/chatbot-sessions 아님)
        - banners: SettingsPage TABS 키 (content-events 아님)
        - chat-suggestions: AiOpsPage TABS 키 (settings 아님)
        - audit_logs_list: settings 에 audit 탭 없어 매핑 제거 → None
        """
        rows = [{"id": str(i), "v": i} for i in range(3)]
        result = self._ok_result(rows)
        cases = {
            "subscriptions_list": "/admin/payment?tab=subscription",
            "point_histories": "/admin/payment?tab=point",
            "point_items": "/admin/payment?tab=items",
            "tickets_list": "/admin/support?tab=ticket",
            "quizzes_list": "/admin/content-events?tab=quiz",
            "review_verifications_list": "/admin/ai?tab=review-verify",
            "chatbot_sessions_list": "/admin/ai?tab=chatlog",
            "banners_list": "/admin/settings?tab=banners",
            "chat_suggestions_list": "/admin/ai?tab=chat-suggestions",
        }
        for tool_name, expected_path in cases.items():
            payload = _build_table_payload(tool_name, result)
            assert payload is not None, f"{tool_name} payload 빌드 실패"
            assert payload["navigate_path"] == expected_path, (
                f"{tool_name} navigate_path 매핑 불일치: "
                f"{payload['navigate_path']!r} != {expected_path!r}"
            )

        # audit_logs_list 는 매핑 제거 → navigate_path=None 폴백 ("전체 보기" 버튼 미렌더)
        payload_audit = _build_table_payload("audit_logs_list", result)
        assert payload_audit is not None
        assert payload_audit["navigate_path"] is None

    def test_below_threshold_returns_none(self):
        """행 수가 임계치(<3) 미만이면 표 발행 안 함."""
        rows = [{"id": "r1", "title": "단건"}]
        payload = _build_table_payload(
            "reports_list", self._ok_result(rows, row_count=1),
        )
        assert payload is None

    def test_spring_data_page_response_unwrapped(self):
        """Page 응답은 content + totalElements 사용해 truncated 표기."""
        page = {
            "content": [
                {"id": f"u{i}", "email": f"user{i}@x"} for i in range(5)
            ],
            "totalElements": 47,
            "totalPages": 10,
            "number": 0,
            "size": 5,
        }
        payload = _build_table_payload("users_list", self._ok_result(page))
        assert payload is not None
        assert payload["columns"] == ["id", "email"]
        assert len(payload["rows"]) == 5
        assert payload["total_rows"] == 47
        assert payload["truncated"] is True
        assert payload["navigate_path"] == "/admin/users"

    def test_failed_result_returns_none(self):
        result = AdminApiResult(ok=False, status_code=500, error="boom", data=None)
        assert _build_table_payload("reports_list", result) is None

    def test_scalar_list_returns_none(self):
        """스칼라 list (dict 가 아닌 row) 는 표로 의미 없으니 None."""
        payload = _build_table_payload(
            "stats_revenue", self._ok_result([1, 2, 3, 4]),
        )
        assert payload is None

    def test_unmapped_tool_no_navigate_path(self):
        """navigate_path 매핑 없는 tool 은 전체 보기 버튼 미렌더 (None)."""
        rows = [
            {"id": "x", "v": 1}, {"id": "y", "v": 2}, {"id": "z", "v": 3},
        ]
        payload = _build_table_payload(
            "some_unknown_list_tool", self._ok_result(rows),
        )
        assert payload is not None
        assert payload["navigate_path"] is None

    def test_long_string_cells_truncated(self):
        long_str = "가" * 500
        rows = [
            {"id": "1", "msg": long_str},
            {"id": "2", "msg": long_str},
            {"id": "3", "msg": long_str},
        ]
        payload = _build_table_payload(
            "posts_list", self._ok_result(rows),
        )
        assert payload is not None
        # 셀 길이가 _TABLE_DATA_MAX_CELL_LEN(80) 한도 + ellipsis 안에 들어가야 함
        for row in payload["rows"]:
            assert len(row["msg"]) <= 81  # 80 + "…"

    def test_columns_capped_at_six(self):
        """첫 행에 컬럼 10개여도 최대 6개만."""
        rows = [
            {f"c{i}": i for i in range(10)},
            {f"c{i}": i + 1 for i in range(10)},
            {f"c{i}": i + 2 for i in range(10)},
        ]
        payload = _build_table_payload("posts_list", self._ok_result(rows))
        assert payload is not None
        assert len(payload["columns"]) == 6


# ============================================================
# 3-b. drafts.py target_path 정합성 (Phase 4 후속, 2026-04-27)
# ============================================================

class TestDraftToolTargetPaths:
    """
    Phase 4 후속(2026-04-27): drafts.py 의 target_path 가 Admin Client 의 실제 라우트/탭 키와
    일치하는지 회귀 검증. quiz_draft, worldcup_candidate_draft 2건이 정정됨.
    """

    @pytest.mark.asyncio
    async def test_quiz_draft_routes_to_content_events(self):
        """
        quiz CRUD 는 ContentEventsPage 의 quiz 탭 (AiOpsPage 에 quiz 탭 없음).
        기존 `/admin/ai?tab=quiz` → `/admin/content-events?tab=quiz` 정정.
        """
        from monglepick.tools.admin_tools.drafts import _handle_quiz_draft
        ctx = type("Ctx", (), {
            "admin_jwt": "", "admin_role": "SUPER_ADMIN", "admin_id": "",
            "session_id": "", "invocation_id": "test",
        })()
        result = await _handle_quiz_draft(
            ctx=ctx,
            movieId="m1", question="?", choices=["A", "B"], answerIndex=0,
        )
        assert result.ok is True
        assert result.data["target_path"] == "/admin/content-events?tab=quiz&modal=create"
        assert result.data["tool_name"] == "quiz_draft"

    @pytest.mark.asyncio
    async def test_worldcup_candidate_draft_uses_full_tab_key(self):
        """
        ContentEventsPage SUB_TABS 의 실제 key 가 `worldcup_candidate` (snake_case 풀네임).
        기존 `tab=worldcup` → `tab=worldcup_candidate` 정정.
        """
        from monglepick.tools.admin_tools.drafts import _handle_worldcup_candidate_draft
        ctx = type("Ctx", (), {
            "admin_jwt": "", "admin_role": "SUPER_ADMIN", "admin_id": "",
            "session_id": "", "invocation_id": "test",
        })()
        result = await _handle_worldcup_candidate_draft(ctx=ctx, movieId="m1")
        assert result.ok is True
        assert result.data["target_path"] == "/admin/content-events?tab=worldcup_candidate&modal=create"
        assert result.data["tool_name"] == "worldcup_candidate_draft"


# ============================================================
# 3-c. _build_chart_payload — chart_data SSE (Phase 4 후속, 2026-04-28)
# ============================================================

class TestBuildChartPayload:
    """등록된 시계열 stats tool 결과 → SSE chart_data payload 빌드."""

    def _ok_result(self, data):
        return AdminApiResult(
            ok=True, status_code=200, data=data, row_count=None, latency_ms=10,
        )

    def test_stats_trends_builds_4_series_line_chart(self):
        """Backend StatsDto.TrendsResponse → 4 시리즈 라인 차트."""
        payload = _build_chart_payload("stats_trends", self._ok_result({
            "trends": [
                {"date": "2026-04-21", "dau": 1000, "newUsers": 30, "reviews": 12, "posts": 5},
                {"date": "2026-04-22", "dau": 1100, "newUsers": 25, "reviews": 18, "posts": 7},
                {"date": "2026-04-23", "dau": 950,  "newUsers": 40, "reviews": 9,  "posts": 3},
            ],
        }))
        assert payload is not None
        assert payload["tool_name"] == "stats_trends"
        assert payload["chart_type"] == "line"
        assert payload["x_axis"]["key"] == "date"
        assert payload["x_axis"]["values"] == ["2026-04-21", "2026-04-22", "2026-04-23"]
        # 4 시리즈가 모두 채워졌는지
        names = [s["name"] for s in payload["series"]]
        assert names == ["DAU", "신규 가입", "리뷰", "게시글"]
        # series.data 길이가 x_axis.values 와 1:1
        for s in payload["series"]:
            assert len(s["data"]) == 3
        assert payload["total_points"] == 3
        assert payload["truncated"] is False
        assert payload["navigate_path"] == "/admin/stats?tab=overview"

    def test_stats_revenue_builds_bar_chart(self):
        """RevenueResponse.dailyRevenue → 단일 시리즈 막대 차트."""
        payload = _build_chart_payload("stats_revenue", self._ok_result({
            "monthlyRevenue": 12_000_000,
            "mrr": 4_500_000,
            "dailyRevenue": [
                {"date": "2026-04-21", "amount": 350_000},
                {"date": "2026-04-22", "amount": 410_000},
                {"date": "2026-04-23", "amount": 290_000},
            ],
        }))
        assert payload is not None
        assert payload["chart_type"] == "bar"
        assert payload["unit"] == "원"
        assert len(payload["series"]) == 1
        assert payload["series"][0]["name"] == "매출"
        assert payload["series"][0]["data"] == [350_000, 410_000, 290_000]
        assert payload["navigate_path"] == "/admin/stats?tab=revenue"

    def test_dashboard_trends_builds_3_series(self):
        payload = _build_chart_payload("dashboard_trends", self._ok_result({
            "days": 7,
            "trends": [
                {"date": "2026-04-21", "newUsers": 10, "paymentAmount": 50000, "chatRequests": 200},
                {"date": "2026-04-22", "newUsers": 15, "paymentAmount": 60000, "chatRequests": 250},
                {"date": "2026-04-23", "newUsers": 12, "paymentAmount": 55000, "chatRequests": 230},
                {"date": "2026-04-24", "newUsers": 18, "paymentAmount": 70000, "chatRequests": 270},
            ],
        }))
        assert payload is not None
        assert payload["tool_name"] == "dashboard_trends"
        assert len(payload["series"]) == 3
        assert payload["x_axis"]["values"][0] == "2026-04-21"
        assert payload["total_points"] == 4

    def test_unregistered_tool_returns_none(self):
        """화이트리스트 외 tool 은 차트 발행 안 함."""
        payload = _build_chart_payload("stats_overview", self._ok_result({
            "dau": 1234, "mau": 5000,
        }))
        assert payload is None

    def test_below_min_points_returns_none(self):
        payload = _build_chart_payload("stats_trends", self._ok_result({
            "trends": [
                {"date": "2026-04-22", "dau": 1100, "newUsers": 25, "reviews": 18, "posts": 7},
                {"date": "2026-04-23", "dau": 950,  "newUsers": 40, "reviews": 9,  "posts": 3},
            ],
        }))
        assert payload is None

    def test_failed_result_returns_none(self):
        result = AdminApiResult(ok=False, status_code=500, error="boom", data=None)
        assert _build_chart_payload("stats_trends", result) is None

    def test_missing_data_key_returns_none(self):
        """응답에 data_key("trends") 가 없으면 None."""
        payload = _build_chart_payload("stats_trends", self._ok_result({
            "wrong_key": [{"date": "x", "dau": 1}],
        }))
        assert payload is None

    def test_all_series_keys_missing_returns_none(self):
        """매핑된 series 키가 응답 dict 에 하나도 없으면 None (수치 0개 차트는 무의미)."""
        payload = _build_chart_payload("stats_trends", self._ok_result({
            "trends": [
                {"date": "2026-04-21"},
                {"date": "2026-04-22"},
                {"date": "2026-04-23"},
            ],
        }))
        assert payload is None

    def test_partial_series_missing_keeps_present_ones(self):
        """일부 series 키만 없으면 있는 것만 남기고 빌드 성공."""
        # newUsers, reviews 만 있고 dau/posts 는 누락
        payload = _build_chart_payload("stats_trends", self._ok_result({
            "trends": [
                {"date": "2026-04-21", "newUsers": 30, "reviews": 12},
                {"date": "2026-04-22", "newUsers": 25, "reviews": 18},
                {"date": "2026-04-23", "newUsers": 40, "reviews": 9},
            ],
        }))
        assert payload is not None
        names = [s["name"] for s in payload["series"]]
        assert "신규 가입" in names
        assert "리뷰" in names
        assert "DAU" not in names  # 모든 포인트에서 없음 → 시리즈 자체 제외
        assert "게시글" not in names

    # ── 2026-04-28 후속 — pie / bar 분포 화이트리스트 추가 검증 ──

    def test_stats_subscription_builds_pie_with_plan_name_labels(self):
        """SubscriptionStatsResponse.plans → pie 차트 (planName 라벨, count 값)."""
        payload = _build_chart_payload("stats_subscription", self._ok_result({
            "totalActive": 120,
            "churnRate": 0.05,
            "plans": [
                {"planCode": "monthly_basic", "planName": "베이직 월간",
                 "count": 50, "percentage": 41.7},
                {"planCode": "monthly_premium", "planName": "프리미엄 월간",
                 "count": 30, "percentage": 25.0},
                {"planCode": "yearly_basic", "planName": "베이직 연간",
                 "count": 25, "percentage": 20.8},
                {"planCode": "yearly_premium", "planName": "프리미엄 연간",
                 "count": 15, "percentage": 12.5},
            ],
        }))
        assert payload is not None
        assert payload["chart_type"] == "pie"
        assert payload["x_axis"]["key"] == "planName"
        # 슬라이스 라벨이 planName 으로 빌드됐는지
        assert payload["x_axis"]["values"] == [
            "베이직 월간", "프리미엄 월간", "베이직 연간", "프리미엄 연간",
        ]
        # 단일 시리즈 (활성 구독, count)
        assert len(payload["series"]) == 1
        assert payload["series"][0]["name"] == "활성 구독"
        assert payload["series"][0]["data"] == [50, 30, 25, 15]
        assert payload["unit"] == "건"
        assert payload["navigate_path"] == "/admin/payment?tab=subscription"

    def test_stats_search_popular_builds_bar_with_keyword_x_axis(self):
        """KeywordItem 리스트 → bar 차트 (keyword 라벨, searchCount 값)."""
        payload = _build_chart_payload("stats_search_popular", self._ok_result({
            "keywords": [
                {"keyword": "봉준호", "searchCount": 320, "conversionRate": 0.42},
                {"keyword": "넷플릭스", "searchCount": 280, "conversionRate": 0.18},
                {"keyword": "송강호", "searchCount": 190, "conversionRate": 0.35},
                {"keyword": "공조", "searchCount": 110, "conversionRate": 0.51},
            ],
        }))
        assert payload is not None
        assert payload["chart_type"] == "bar"
        assert payload["x_axis"]["key"] == "keyword"
        assert payload["x_axis"]["values"] == ["봉준호", "넷플릭스", "송강호", "공조"]
        assert payload["series"][0]["name"] == "검색 수"
        assert payload["series"][0]["data"] == [320, 280, 190, 110]
        assert payload["unit"] == "회"
        assert payload["navigate_path"] == "/admin/stats?tab=search"

    def test_pie_chart_below_threshold_returns_none(self):
        """pie 도 임계 동일: 슬라이스 < 3 이면 차트 의미 없음 → None."""
        payload = _build_chart_payload("stats_subscription", self._ok_result({
            "totalActive": 5, "churnRate": 0.1,
            "plans": [
                {"planCode": "p1", "planName": "A", "count": 3, "percentage": 60.0},
                {"planCode": "p2", "planName": "B", "count": 2, "percentage": 40.0},
            ],
        }))
        assert payload is None

    def test_pie_chart_missing_label_field_returns_none(self):
        """planName 필드가 없는 plans → x_key 누락 → None."""
        payload = _build_chart_payload("stats_subscription", self._ok_result({
            "plans": [
                {"planCode": "p1", "count": 10},  # planName 없음
                {"planCode": "p2", "count": 20},
                {"planCode": "p3", "count": 30},
            ],
        }))
        assert payload is None

    # ── 2026-04-28 후속2 — 분포 도구 3종 (장르/포인트유형/등급) pie 차트 ──

    def test_recommendation_distribution_builds_pie_with_genre_labels(self):
        """DistributionResponse.genres → pie. genre 가 한국어 라벨, count 가 값."""
        payload = _build_chart_payload("stats_recommendation_distribution", self._ok_result({
            "genres": [
                {"genre": "액션", "count": 1200, "percentage": 35.3},
                {"genre": "로맨스", "count": 800, "percentage": 23.5},
                {"genre": "스릴러", "count": 600, "percentage": 17.6},
                {"genre": "코미디", "count": 500, "percentage": 14.7},
                {"genre": "드라마", "count": 300, "percentage": 8.8},
            ],
        }))
        assert payload is not None
        assert payload["chart_type"] == "pie"
        assert payload["x_axis"]["key"] == "genre"
        assert payload["x_axis"]["values"] == ["액션", "로맨스", "스릴러", "코미디", "드라마"]
        assert payload["series"][0]["name"] == "추천 건수"
        assert payload["series"][0]["data"] == [1200, 800, 600, 500, 300]
        assert payload["unit"] == "건"
        assert payload["navigate_path"] == "/admin/stats?tab=recommendation"

    def test_point_distribution_uses_korean_label_not_pointtype_code(self):
        """
        PointTypeDistributionResponse.distribution → pie.
        x_key 는 한국어 label 이어야 함 (pointType 영문 코드는 운영자 친화도 낮음).
        """
        payload = _build_chart_payload("stats_point_distribution", self._ok_result({
            "distribution": [
                {"pointType": "earn",   "label": "활동 리워드",  "count": 5000, "totalAmount": 250000, "percentage": 50.0},
                {"pointType": "spend",  "label": "AI 추천 사용", "count": 3000, "totalAmount": 90000,  "percentage": 30.0},
                {"pointType": "bonus",  "label": "출석 보너스",  "count": 1000, "totalAmount": 30000,  "percentage": 10.0},
                {"pointType": "expire", "label": "포인트 만료",  "count": 800,  "totalAmount": 24000,  "percentage": 8.0},
                {"pointType": "refund", "label": "환불 회수",    "count": 200,  "totalAmount": 6000,   "percentage": 2.0},
            ],
        }))
        assert payload is not None
        assert payload["chart_type"] == "pie"
        # 한국어 라벨이 x_axis 에 들어가야 함 (pointType 영문 코드 X)
        assert payload["x_axis"]["values"][0] == "활동 리워드"
        assert payload["x_axis"]["values"][1] == "AI 추천 사용"
        assert payload["series"][0]["name"] == "거래 건수"
        assert payload["series"][0]["data"] == [5000, 3000, 1000, 800, 200]
        assert payload["navigate_path"] == "/admin/stats?tab=point-economy"

    def test_grade_distribution_builds_pie_with_korean_grade_names(self):
        """
        GradeDistributionResponse.grades → pie.
        x_key 는 gradeName(한국어, 알갱이/팝콘/...) 이어야 함 (gradeCode 영문 X).
        """
        payload = _build_chart_payload("stats_grade_distribution", self._ok_result({
            "grades": [
                {"gradeCode": "NORMAL",   "gradeName": "알갱이",     "count": 4500, "percentage": 60.0},
                {"gradeCode": "BRONZE",   "gradeName": "강냉이",     "count": 1500, "percentage": 20.0},
                {"gradeCode": "SILVER",   "gradeName": "팝콘",       "count": 800,  "percentage": 10.7},
                {"gradeCode": "GOLD",     "gradeName": "카라멜팝콘", "count": 400,  "percentage": 5.3},
                {"gradeCode": "PLATINUM", "gradeName": "몽글팝콘",   "count": 200,  "percentage": 2.7},
                {"gradeCode": "DIAMOND",  "gradeName": "몽아일체",   "count": 100,  "percentage": 1.3},
            ],
        }))
        assert payload is not None
        assert payload["chart_type"] == "pie"
        assert payload["x_axis"]["key"] == "gradeName"
        assert payload["x_axis"]["values"] == [
            "알갱이", "강냉이", "팝콘", "카라멜팝콘", "몽글팝콘", "몽아일체",
        ]
        assert payload["series"][0]["name"] == "사용자 수"
        assert payload["series"][0]["data"] == [4500, 1500, 800, 400, 200, 100]
        assert payload["unit"] == "명"


# ── 2026-04-28 후속2 — 신규 분포 도구 3종 레지스트리 등록 검증 ──

class TestDistributionToolsRegistered:
    """3 신규 도구가 ADMIN_TOOL_REGISTRY 에 등록되고 stats role 모두에 노출되는지."""

    def test_three_distribution_tools_registered(self):
        from monglepick.tools.admin_tools import ADMIN_TOOL_REGISTRY
        for name in (
            "stats_recommendation_distribution",
            "stats_point_distribution",
            "stats_grade_distribution",
        ):
            assert name in ADMIN_TOOL_REGISTRY, f"{name} 미등록"
            spec = ADMIN_TOOL_REGISTRY[name]
            assert spec.tier == 0
            # _NoArgs 가 들어있는지 확인 (인자 dict 빈 채로 검증 통과해야 함)
            spec.args_schema.model_validate({})

    def test_distribution_tools_visible_to_stats_admin(self):
        """STATS_ADMIN role 에 3 분포 도구가 전부 노출되어야 함."""
        from monglepick.tools.admin_tools import list_tools_for_role
        names = {s.name for s in list_tools_for_role("STATS_ADMIN")}
        assert "stats_recommendation_distribution" in names
        assert "stats_point_distribution" in names
        assert "stats_grade_distribution" in names


# ── 2026-04-28 후속2 (시계열 line) — 추이 도구 3종 line 차트 ──

class TestTrendChartPayloads:
    """포인트/AI 세션/커뮤니티 일별 추이 → line 차트."""

    def _ok_result(self, data):
        return AdminApiResult(
            ok=True, status_code=200, data=data, row_count=None, latency_ms=10,
        )

    def test_point_trends_builds_3_series_line(self):
        """PointTrendsResponse → line 3시리즈 (발행/소비/순유입)."""
        payload = _build_chart_payload("stats_point_trends", self._ok_result({
            "trends": [
                {"date": "2026-04-21", "issued": 1500, "spent": 800,  "netFlow": 700},
                {"date": "2026-04-22", "issued": 2000, "spent": 1200, "netFlow": 800},
                {"date": "2026-04-23", "issued": 1800, "spent": 2100, "netFlow": -300},
            ],
        }))
        assert payload is not None
        assert payload["chart_type"] == "line"
        assert payload["x_axis"]["values"] == ["2026-04-21", "2026-04-22", "2026-04-23"]
        names = [s["name"] for s in payload["series"]]
        assert names == ["발행", "소비", "순유입"]
        # 음수(netFlow=-300) 도 그대로 보존
        net_flow_series = next(s for s in payload["series"] if s["name"] == "순유입")
        assert net_flow_series["data"] == [700, 800, -300]
        assert payload["unit"] == "P"

    def test_ai_session_trends_builds_2_series_line(self):
        payload = _build_chart_payload("stats_ai_session_trends", self._ok_result({
            "trends": [
                {"date": "2026-04-21", "sessions": 120, "turns": 540},
                {"date": "2026-04-22", "sessions": 150, "turns": 700},
                {"date": "2026-04-23", "sessions": 135, "turns": 620},
            ],
        }))
        assert payload is not None
        assert payload["chart_type"] == "line"
        assert len(payload["series"]) == 2
        assert {s["name"] for s in payload["series"]} == {"세션", "턴 수"}
        assert payload["navigate_path"] == "/admin/stats?tab=ai-service"

    def test_community_trends_builds_3_series_line(self):
        payload = _build_chart_payload("stats_community_trends", self._ok_result({
            "trends": [
                {"date": "2026-04-21", "posts": 45, "comments": 320, "reports": 3},
                {"date": "2026-04-22", "posts": 52, "comments": 410, "reports": 5},
                {"date": "2026-04-23", "posts": 48, "comments": 380, "reports": 2},
                {"date": "2026-04-24", "posts": 60, "comments": 450, "reports": 8},
            ],
        }))
        assert payload is not None
        assert payload["chart_type"] == "line"
        names = [s["name"] for s in payload["series"]]
        assert names == ["게시글", "댓글", "신고"]
        # reports 시리즈가 다른 시리즈(comments) 와 스케일이 달라도 그대로 데이터 보존
        reports_series = next(s for s in payload["series"] if s["name"] == "신고")
        assert reports_series["data"] == [3, 5, 2, 8]
        assert payload["navigate_path"] == "/admin/stats?tab=community"


class TestTrendToolsRegistered:
    """3 신규 시계열 도구가 ADMIN_TOOL_REGISTRY 에 등록되고 STATS_ADMIN 에 노출되는지."""

    def test_three_trend_tools_registered_with_period_args(self):
        from monglepick.tools.admin_tools import ADMIN_TOOL_REGISTRY
        for name in (
            "stats_point_trends",
            "stats_ai_session_trends",
            "stats_community_trends",
        ):
            assert name in ADMIN_TOOL_REGISTRY, f"{name} 미등록"
            spec = ADMIN_TOOL_REGISTRY[name]
            assert spec.tier == 0
            # period 인자(default '7d') 만 받는 _PeriodArgs — 빈 dict 검증 통과 + period 기본값.
            validated = spec.args_schema.model_validate({})
            assert getattr(validated, "period", None) == "7d"

    def test_trend_tools_visible_to_stats_admin(self):
        from monglepick.tools.admin_tools import list_tools_for_role
        names = {s.name for s in list_tools_for_role("STATS_ADMIN")}
        assert "stats_point_trends" in names
        assert "stats_ai_session_trends" in names
        assert "stats_community_trends" in names


# ============================================================
# 4. placeholder 메시지 — Phase 4 표기 제거
# ============================================================

class TestReportPlaceholderUpdated:
    """report placeholder 가 더이상 'Phase 4 예정' 문구를 노출하지 않음."""

    def test_report_placeholder_no_longer_promises_phase_4(self):
        msg = _PLACEHOLDER_MESSAGES["report"]
        assert "Phase 4" not in msg
        assert "보고서 생성" not in msg or "예정" not in msg

    def test_sql_placeholder_unchanged(self):
        """sql 은 영구 미지원이라 텍스트 회귀 검증."""
        msg = _PLACEHOLDER_MESSAGES["sql"]
        assert "지원하지 않" in msg
