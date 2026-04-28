"""
관리자 AI 에이전트 — Tier 0 Stats Extended Read-only Tool (16개).

설계서: docs/관리자_AI에이전트_v3_재설계.md §4.1, §5 (Role 매트릭스)

이 모듈은 stats.py 의 5개 기본 통계 EP 에 이어,
심층 분석 영역(추천/검색/행동/잔존율/구독/포인트/인게이지먼트/콘텐츠성과/퍼널/이탈위험)
+ 분포 시각화(추천 장르/포인트 유형/사용자 등급)
+ 시계열 추이(포인트 발행·소비/AI 세션·턴/커뮤니티 게시글·댓글·신고) 를
커버하는 16개 Read-only Tool 을 등록한다.

Backend 경로 (Prefix: `/api/v1/admin/stats`):
- stats_recommendation              — GET /recommendation?period={7d|30d|90d}
- stats_recommendation_distribution — GET /recommendation/distribution         (no params, pie)
- stats_search_popular              — GET /search/popular?period=...&limit=20  (bar)
- stats_behavior                    — GET /behavior?period=...
- stats_retention                   — GET /retention?period=...
- stats_subscription                — GET /subscription?period=...             (pie)
- stats_point_economy               — GET /point-economy/overview?period=...
- stats_point_distribution          — GET /point-economy/distribution          (no params, pie)
- stats_grade_distribution          — GET /point-economy/grades                (no params, pie)
- stats_point_trends                — GET /point-economy/trends?period=...     (line)
- stats_ai_session_trends           — GET /ai-service/trends?period=...        (line)
- stats_community_trends            — GET /community/trends?period=...         (line)
- stats_engagement                  — GET /engagement/overview?period=...
- stats_content_performance         — GET /content-performance/overview?period=...
- stats_funnel                      — GET /funnel/conversion?period=...
- stats_churn_risk                  — GET /churn-risk/overview?period=...

Backend 응답은 전부 `ApiResponse<T>` 래퍼 (`{success, data, error}`) 이므로
unwrap_api_response 로 data 만 언래핑해 AdminApiResult.data 에 재주입한다.

Role matrix (§5):
- 전부 Tier 0, required_roles = SUPER_ADMIN · ADMIN · DATA_ADMIN · STATS_ADMIN
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from monglepick.api.admin_backend_client import (
    AdminApiResult,
    get_admin_json,
    unwrap_api_response,
)
from monglepick.tools.admin_tools import (
    ToolContext,
    ToolSpec,
    register_tool,
)


# ============================================================
# Role matrix — stats_extended 허용 역할 (§5)
# ============================================================

_STATS_EXT_ROLES: set[str] = {
    "SUPER_ADMIN",
    "ADMIN",
    "DATA_ADMIN",
    "STATS_ADMIN",
}


# ============================================================
# Args Schemas (LLM bind 용 Pydantic 모델)
# ============================================================

class _PeriodArgs(BaseModel):
    """`period` 쿼리 하나만 받는 통계 EP 공통 args."""

    period: Literal["7d", "30d", "90d"] = Field(
        default="7d",
        description=(
            "조회 기간. '7d' = 최근 7일, '30d' = 최근 30일, '90d' = 최근 90일. "
            "사용자가 기간을 명시하지 않으면 '7d' 를 기본값으로 사용한다."
        ),
    )


class _SearchPopularArgs(BaseModel):
    """`stats_search_popular` 전용 args — period + limit 지원."""

    period: Literal["7d", "30d", "90d"] = Field(
        default="7d",
        description=(
            "조회 기간. '7d' = 최근 7일, '30d' = 최근 30일, '90d' = 최근 90일."
        ),
    )
    limit: Annotated[int, Field(ge=1, le=100)] = Field(
        default=20,
        description="반환할 인기 검색어 순위 개수 (1~100, 기본 20).",
    )


class _NoArgs(BaseModel):
    """파라미터 없는 EP 용 빈 스키마 (LLM 이 arguments={} 로 호출).

    2026-04-28 추가 — 분포 시각화 도구 3종(`stats_recommendation_distribution`,
    `stats_point_distribution`, `stats_grade_distribution`) 이 모두 무파라미터 GET 이라
    공용 사용. stats.py 의 `_NoArgs` 와 동일 정의를 모듈 내에 복제한 이유는 import
    체인을 단순하게 유지하기 위함 (stats_extended → stats 역의존 회피).
    """

    pass


# ============================================================
# Handlers
# ============================================================

async def _handle_stats_recommendation(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/recommendation?period=...` 호출 후 래퍼 언래핑.

    추천 엔진 사용 현황: 호출 수, 클릭율(CTR), 평균 추천 점수 등을 집계한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/recommendation",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_recommendation_distribution(
    ctx: ToolContext,
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/recommendation/distribution` (파라미터 없음).

    추천 장르 분포 — DistributionResponse{genres: [{genre, count, percentage}]}.
    Phase 4 후속 (2026-04-28) 에서 pie 차트로 시각화.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/recommendation/distribution",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_point_distribution(
    ctx: ToolContext,
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/point-economy/distribution` (파라미터 없음).

    포인트 유형(earn/spend/bonus/expire/refund/revoke + admin_grant/admin_revoke) 분포 —
    PointTypeDistributionResponse{distribution: [{pointType, label, count, totalAmount, percentage}]}.
    Phase 4 후속 (2026-04-28) 에서 pie 차트로 시각화 (한국어 label 을 슬라이스 라벨로).
    admin_grant/admin_revoke 는 운영 조정 전용 분류 — 백엔드가 "운영 지급"/"운영 회수"
    라벨로 응답하며 KPI(총발행/총소비) 합산에서는 자동 제외된다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/point-economy/distribution",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_grade_distribution(
    ctx: ToolContext,
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/point-economy/grades` (파라미터 없음).

    6등급(NORMAL/BRONZE/SILVER/GOLD/PLATINUM/DIAMOND, 한국어명: 알갱이~몽아일체)
    사용자 분포 — GradeDistributionResponse{grades: [{gradeCode, gradeName, count, percentage}]}.
    Phase 4 후속 (2026-04-28) 에서 pie 차트로 시각화.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/point-economy/grades",
        admin_jwt=ctx.admin_jwt,
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_point_trends(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/point-economy/trends?period=...` 호출 후 래퍼 언래핑.

    일별 포인트 발행·소비·순유입 추이 — PointTrendsResponse{trends: [{date, issued, spent, netFlow}]}.
    Phase 4 후속 (2026-04-28) 에서 line 차트 3시리즈로 시각화.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/point-economy/trends",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_ai_session_trends(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/ai-service/trends?period=...` 호출 후 래퍼 언래핑.

    일별 AI 세션·턴 추이 — AiSessionTrendsResponse{trends: [{date, sessions, turns}]}.
    Phase 4 후속 (2026-04-28) 에서 line 차트 2시리즈로 시각화.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/ai-service/trends",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_community_trends(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/community/trends?period=...` 호출 후 래퍼 언래핑.

    일별 커뮤니티 게시글·댓글·신고 추이 — CommunityTrendsResponse{trends: [{date, posts, comments, reports}]}.
    Phase 4 후속 (2026-04-28) 에서 line 차트 3시리즈로 시각화.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/community/trends",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_search_popular(
    ctx: ToolContext,
    period: str = "7d",
    limit: int = 20,
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/search/popular?period=...&limit=...` 호출 후 래퍼 언래핑.

    기간 내 인기 검색어 Top-N 과 검색 횟수를 반환한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/search/popular",
        admin_jwt=ctx.admin_jwt,
        params={"period": period, "limit": limit},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_behavior(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/behavior?period=...` 호출 후 래퍼 언래핑.

    사용자 행동 패턴(페이지뷰, 세션 시간, 기능 클릭 분포 등)을 집계한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/behavior",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_retention(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/retention?period=...` 호출 후 래퍼 언래핑.

    코호트 기반 D1/D7/D30 재방문율 및 잔존율 지표를 반환한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/retention",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_subscription(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/subscription?period=...` 호출 후 래퍼 언래핑.

    구독 신규 가입/해지/활성 구독자 수 및 플랜별 분포를 집계한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/subscription",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_point_economy(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/point-economy/overview?period=...` 호출 후 래퍼 언래핑.

    포인트 발행량/소비량/잔액 분포/등급별 포인트 현황 등 포인트 경제 지표를 반환한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/point-economy/overview",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_engagement(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/engagement/overview?period=...` 호출 후 래퍼 언래핑.

    좋아요/리뷰/플레이리스트 저장 등 사용자 인게이지먼트 종합 지표를 집계한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/engagement/overview",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_content_performance(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/content-performance/overview?period=...` 호출 후 래퍼 언래핑.

    영화별/장르별 조회수·리뷰 수·위시리스트 추가 수 등 콘텐츠 성과 지표를 반환한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/content-performance/overview",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_funnel(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/funnel/conversion?period=...` 호출 후 래퍼 언래핑.

    방문→회원가입→첫 추천→구독 각 단계별 전환율 퍼널 데이터를 반환한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/funnel/conversion",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


async def _handle_stats_churn_risk(
    ctx: ToolContext,
    period: str = "7d",
) -> AdminApiResult:
    """
    `GET /api/v1/admin/stats/churn-risk/overview?period=...` 호출 후 래퍼 언래핑.

    이탈 위험 사용자 비율, 위험도 구간별 분포, 최근 비활성 추이를 집계한다.
    """
    raw = await get_admin_json(
        "/api/v1/admin/stats/churn-risk/overview",
        admin_jwt=ctx.admin_jwt,
        params={"period": period},
        invocation_id=ctx.invocation_id,
    )
    return unwrap_api_response(raw)


# ============================================================
# Registration (모듈 import 시 즉시 실행 — 레지스트리 등록)
# ============================================================

register_tool(ToolSpec(
    name="stats_recommendation",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "AI 추천 엔진 사용 현황 조회. 기간 내 추천 호출 수, 클릭율(CTR), "
        "평균 추천 점수, 추천 소스별(CF/CBF/외부 검색) 비율을 집계해 반환한다. "
        "추천 시스템 성능·활용도 관련 질문에 사용한다."
    ),
    example_questions=[
        "AI 추천 클릭율(CTR) 지난 30일 기준으로 얼마야?",
        "추천 엔진 호출 수 추이 보여줘",
        "CF vs CBF 추천 비율 알려줘",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_recommendation,
))


register_tool(ToolSpec(
    name="stats_search_popular",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "인기 검색어 순위 조회. 기간 내 가장 많이 검색된 키워드 Top-N(기본 20)과 "
        "각 검색 횟수를 내림차순으로 반환한다. 콘텐츠 기획·트렌드 파악 질문에 사용한다."
    ),
    example_questions=[
        "이번 주 인기 검색어 Top 10 보여줘",
        "지난달 가장 많이 검색된 영화 제목 알려줘",
        "최근 30일 검색 트렌드 Top 20",
    ],
    args_schema=_SearchPopularArgs,
    handler=_handle_stats_search_popular,
))


register_tool(ToolSpec(
    name="stats_behavior",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "사용자 행동 패턴 분석. 기간 내 페이지뷰, 평균 세션 시간, "
        "기능별 클릭 분포(채팅/검색/플레이리스트 등)를 집계한다. "
        "UX 개선·기능 활용도 분석 질문에 사용한다."
    ),
    example_questions=[
        "사용자들이 어떤 기능을 가장 많이 쓰고 있어?",
        "평균 세션 시간 지난 7일 얼마나 돼?",
        "채팅 vs 검색 이용 비율 비교해줘",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_behavior,
))


register_tool(ToolSpec(
    name="stats_retention",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "코호트 기반 사용자 잔존율 조회. 가입 후 D1/D7/D30 재방문율 및 "
        "코호트별 잔존 곡선을 반환한다. 리텐션 개선·이탈 방지 전략 수립에 활용한다."
    ),
    example_questions=[
        "D7 재방문율 지난달 기준 얼마야?",
        "신규 가입자 D30 잔존율 보여줘",
        "최근 코호트 잔존율 추이 알려줘",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_retention,
))


register_tool(ToolSpec(
    name="stats_subscription",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "구독 현황 통계 조회. 기간 내 신규 구독/해지 수, 현재 활성 구독자 수, "
        "플랜별(basic/premium/연간) 분포를 반환한다. "
        "구독 성장·해지율 관련 질문에 사용한다."
    ),
    example_questions=[
        "현재 활성 구독자 몇 명이야?",
        "이번 달 구독 해지 수 얼마나 돼?",
        "플랜별 구독자 분포 보여줘",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_subscription,
))


register_tool(ToolSpec(
    name="stats_point_economy",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "포인트 경제 지표 조회. 기간 내 포인트 총 발행량·소비량·잔액 분포, "
        "등급별 포인트 보유 현황, 소비 카테고리별 비율을 반환한다. "
        "포인트 경제 건강도·이용 패턴 분석에 사용한다."
    ),
    example_questions=[
        "이번 달 포인트 발행량이랑 소비량 비교해줘",
        "등급별 평균 포인트 잔액 얼마야?",
        "포인트 주로 어디에 소비되고 있어?",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_point_economy,
))


register_tool(ToolSpec(
    name="stats_engagement",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "사용자 인게이지먼트 종합 지표 조회. 기간 내 좋아요/리뷰 작성/플레이리스트 저장/"
        "위시리스트 추가 건수 및 활성 사용자 비율을 집계한다. "
        "참여도·서비스 활성화 현황 파악에 사용한다."
    ),
    example_questions=[
        "이번 주 리뷰 작성 건수 얼마나 돼?",
        "플레이리스트 저장 수 지난 30일 추이",
        "사용자 인게이지먼트 지표 전반적으로 보여줘",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_engagement,
))


register_tool(ToolSpec(
    name="stats_content_performance",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "콘텐츠 성과 지표 조회. 기간 내 영화별/장르별 조회수·리뷰 수·위시리스트 추가 수·"
        "추천 클릭 수를 집계하고 상위 콘텐츠 순위를 반환한다. "
        "인기 영화·장르 트렌드 파악에 사용한다."
    ),
    example_questions=[
        "이번 달 가장 많이 본 영화 Top 10 알려줘",
        "장르별 리뷰 수 분포 보여줘",
        "위시리스트에 가장 많이 담긴 영화는?",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_content_performance,
))


register_tool(ToolSpec(
    name="stats_funnel",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "전환 퍼널 분석. 방문→회원가입→첫 AI 추천 사용→구독 결제 각 단계별 "
        "사용자 수와 전환율을 반환한다. 온보딩·구독 전환 개선 지점 파악에 사용한다."
    ),
    example_questions=[
        "회원가입에서 첫 AI 추천까지 전환율 얼마야?",
        "구독 결제 전환율 지난 30일 기준 보여줘",
        "퍼널 어느 단계에서 이탈이 가장 많아?",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_funnel,
))


register_tool(ToolSpec(
    name="stats_churn_risk",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "이탈 위험 사용자 현황 조회. 기간 내 이탈 위험 사용자 수·비율, "
        "위험도 구간별(고/중/저) 분포, 최근 비활성 기간 추이를 집계한다. "
        "이탈 방지 캠페인 타겟 설정·리텐션 전략 수립에 사용한다."
    ),
    example_questions=[
        "이탈 위험 사용자 지금 몇 명이야?",
        "고위험 이탈 구간 사용자 비율 알려줘",
        "최근 30일 비활성 사용자 증가 추이 보여줘",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_churn_risk,
))


# ============================================================
# 2026-04-28 — 분포 시각화 도구 3종 (pie 차트 화이트리스트와 짝)
# ============================================================
# 모두 무파라미터 GET. Backend 응답이 분포 형태(list of {label_key, value_key}) 라
# graph.py `_CHART_TOOL_SPECS` 에 등록되면 ChartDataCard 가 pie 로 렌더한다.

register_tool(ToolSpec(
    name="stats_recommendation_distribution",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "추천 장르 분포 조회. 장르별 추천/시청 비율을 반환한다. "
        "'장르별 분포', '어떤 장르가 많이 추천돼?' 같은 질문에 사용한다. 파라미터 없음."
    ),
    example_questions=[
        "추천 장르 분포 보여줘",
        "어떤 장르가 가장 많이 추천돼?",
        "장르별 비율 차트로 보고 싶어",
    ],
    args_schema=_NoArgs,
    handler=_handle_stats_recommendation_distribution,
))


register_tool(ToolSpec(
    name="stats_point_distribution",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "포인트 유형별 분포. earn/spend/bonus/expire/refund/revoke 유형별 거래 건수와 "
        "포인트 합계를 반환한다. '포인트 종류별 분포', '환불 vs 발행 비율' 같은 질문에 사용. "
        "파라미터 없음."
    ),
    example_questions=[
        "포인트 유형별 분포 보여줘",
        "발행 포인트 vs 소비 포인트 비율 어때?",
        "리워드 종류별 비중 차트",
    ],
    args_schema=_NoArgs,
    handler=_handle_stats_point_distribution,
))


register_tool(ToolSpec(
    name="stats_grade_distribution",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "6등급(알갱이/강냉이/팝콘/카라멜팝콘/몽글팝콘/몽아일체) 사용자 분포 조회. "
        "각 등급의 사용자 수와 비율을 반환한다. "
        "'등급별 사용자 분포', '몇 등급 유저가 가장 많아?' 같은 질문에 사용. 파라미터 없음."
    ),
    example_questions=[
        "등급별 사용자 분포 보여줘",
        "다이아몬드 등급 유저 비율 알려줘",
        "유저 등급 분포 차트",
    ],
    args_schema=_NoArgs,
    handler=_handle_stats_grade_distribution,
))


# ============================================================
# 2026-04-28 후속2 (시계열 추이) — 일별 line 차트 도구 3종
# ============================================================
# 모두 period 인자(7d|30d|90d) 받는 시계열 GET. graph.py `_CHART_TOOL_SPECS` 에 line 매핑.

register_tool(ToolSpec(
    name="stats_point_trends",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "일별 포인트 발행/소비/순유입(netFlow) 추이 조회. 기간 내 매일의 발행 합계, 소비 합계, "
        "차이를 반환한다. '포인트 흐름 추이', '소비 vs 발행 추세', '포인트 경제 변동' 같은 질문에 사용."
    ),
    example_questions=[
        "지난 30일 포인트 발행/소비 추이 보여줘",
        "포인트 순유입 추세 어떻게 돼?",
        "최근 일주일 포인트 흐름 차트",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_point_trends,
))


register_tool(ToolSpec(
    name="stats_ai_session_trends",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "AI 세션·턴 일별 추이 조회. 기간 내 매일의 채팅 세션 수와 총 턴 수 시계열을 반환한다. "
        "'AI 사용량 추세', '챗봇 세션 변화', '턴 수 변동' 같은 질문에 사용."
    ),
    example_questions=[
        "지난 30일 AI 세션 추이 보여줘",
        "챗봇 사용량 변화 추세",
        "AI 채팅 턴 수 일별 추이",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_ai_session_trends,
))


register_tool(ToolSpec(
    name="stats_community_trends",
    tier=0,
    required_roles=_STATS_EXT_ROLES,
    description=(
        "커뮤니티 일별 게시글/댓글/신고 추이 조회. 기간 내 매일의 게시글 수, 댓글 수, 신고 건수 "
        "시계열을 반환한다. '커뮤니티 활동 추세', '신고 건수 변동', '게시글 추이' 같은 질문에 사용."
    ),
    example_questions=[
        "지난 30일 커뮤니티 활동 추이 보여줘",
        "신고 건수 일별 추세",
        "최근 7일 게시글/댓글 변화 차트",
    ],
    args_schema=_PeriodArgs,
    handler=_handle_stats_community_trends,
))
