"""
고객센터 AI 에이전트 v4 Tool 레지스트리.

설계서: docs/고객센터_AI에이전트_v4_재설계.md §5 (Tool 목록) / §5.2 (lookup_policy)

구조:
- **ToolContext**: 런타임 컨텍스트. user_id / is_guest / session_id / request_id 를
  담고, handler 는 이걸로 사용자 개인화 여부를 판단한다. LLM 에는 노출되지 않는다.
- **SupportToolSpec**: 레지스트리 단위. name / description / args_schema / handler /
  requires_login. args_schema 는 Pydantic BaseModel — LLM 에 bind 되는 스키마.
- **SUPPORT_TOOL_REGISTRY**: name → SupportToolSpec dict.
  각 서브 모듈(policy.py 등)이 `register_support_tool()` 로 등록하며,
  import side-effect 로 자동 실행된다.

관리자 AI 에이전트의 `admin_tools/__init__.py` 와 동일한 레지스트리 패턴을 따른다.
단, 관리자 ToolContext(admin_jwt / admin_role) 와 별도로 분리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel


# ============================================================
# 런타임 컨텍스트
# ============================================================

@dataclass
class ToolContext:
    """
    고객센터 Tool 실행 시점의 런타임 컨텍스트 (LLM 에 비노출).

    JWT 검증 후 FastAPI 엔드포인트에서 주입된다.
    게스트(is_guest=True)는 user_id 가 빈 문자열이며,
    requires_login=True 인 tool 은 게스트 호출 시 거부해야 한다.

    Attributes:
        user_id:    JWT 에서 추출한 사용자 ID. 게스트는 빈 문자열.
        is_guest:   True 이면 비인증 게스트 접근.
        session_id: 현재 대화 세션 ID (Redis 키 prefix 등에 사용).
        request_id: 요청 추적 ID (로그 상관관계 추적, LangSmith run_id 등).
    """

    user_id: str = ""
    is_guest: bool = True
    session_id: str = ""
    request_id: str = ""


# ============================================================
# Tool 스펙
# ============================================================

# Handler 시그니처:
#   async def handler(ctx: ToolContext, **validated_args) -> dict[str, Any]
#
# 반환 dict 구조:
#   {"ok": True,  "data": {...}}   — 정상
#   {"ok": False, "error": "..."}  — 비즈니스 오류 (에러 전파 X)
ToolHandler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class SupportToolSpec:
    """
    고객센터 AI 에이전트 단일 tool 명세.

    name:           tool 식별자 (LLM bind_tools 에서 함수명으로 노출)
    description:    LLM 에게 제공되는 tool 설명 (한국어, 언제 사용할지 기술)
    args_schema:    Pydantic BaseModel — LLM 이 JSON 으로 채울 입력 스키마
    handler:        실제 실행 로직 (async)
    requires_login: True 이면 is_guest=True 컨텍스트에서 호출 불가
                    False 이면 게스트도 호출 가능 (정책 조회 등 공개 정보)
    """

    name: str
    description: str
    args_schema: type[BaseModel]
    handler: ToolHandler
    requires_login: bool = True


# ============================================================
# 레지스트리
# ============================================================

#: name → SupportToolSpec 매핑. 서브 모듈이 register_support_tool() 로 채운다.
SUPPORT_TOOL_REGISTRY: dict[str, SupportToolSpec] = {}


def register_support_tool(spec: SupportToolSpec) -> None:
    """
    SupportToolSpec 을 전역 레지스트리에 등록한다.

    서브 모듈 (policy.py 등) 최하단에서 호출되며,
    `from . import policy` import 시 side-effect 로 자동 실행된다.

    중복 등록 시 덮어쓴다 (개발 편의 + hot-reload 지원).

    Args:
        spec: 등록할 SupportToolSpec 인스턴스
    """
    SUPPORT_TOOL_REGISTRY[spec.name] = spec


# ============================================================
# 서브 모듈 import — side-effect 로 레지스트리에 tool 등록
# ============================================================

# noqa: E402 — 레지스트리 변수/함수 정의 이후에 import 해야 하므로 모듈 하단에 위치
from . import policy          # noqa: E402, F401
# Phase 1.1 (2026-04-28): 본인 데이터 조회 Read tool 8개 등록
from . import point_history   # noqa: E402, F401
from . import attendance      # noqa: E402, F401
from . import ai_quota        # noqa: E402, F401
from . import subscription    # noqa: E402, F401
from . import grade           # noqa: E402, F401
from . import orders          # noqa: E402, F401
from . import tickets         # noqa: E402, F401
from . import recent_activity  # noqa: E402, F401
