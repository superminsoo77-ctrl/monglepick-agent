"""
unit 테스트 공통 Fixture.

현재 등록된 fixture:
  _v3_compat_graph_swap  — v3 support_assistant 테스트 호환성 fixture
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import monglepick.agents.support_assistant.graph as _sa_graph_module


# =============================================================================
# v3 support_assistant 테스트 호환성 fixture
# =============================================================================


@pytest.fixture(autouse=True)
def _v3_compat_graph_swap(request):
    """
    v3 support_assistant 테스트 호환성 fixture.

    ## 배경
    Phase 1.8 에서 v4 그래프(9노드)를 도입했다. `run_support_assistant` /
    `run_support_assistant_sync` 는 모듈 수준 싱글턴 `support_assistant_graph`
    (v4) 를 직접 참조한다.

    v3 테스트(test_support_assistant_v3.py) 는 `_patch_reply` 로
    `generate_support_reply` 만 stub 한다.  v4 그래프는 `intent_classifier` 를
    먼저 실행한 뒤 `tool_selector → tool_executor(ES) → narrator(Solar)` 경로를
    타므로 `_patch_reply` 가 호출되는 `support_agent` 노드에 도달하지 않는다.
    결과적으로 ES/Solar 실제 호출이 일어나 시나리오가 어긋난다.

    ## 해결 방식
    이 fixture 는 `test_support_assistant_v3` 모듈에서만 자동으로 활성화되어
    `monglepick.agents.support_assistant.graph.support_assistant_graph` 를
    이미 컴파일되어 있는 `support_assistant_graph_v3` 로 교체한다.

    v3 그래프는 `context_loader → support_agent → response_formatter` 3노드로
    구성되어 있어 `_patch_reply` stub 이 정상 동작한다.

    ## 적용 범위
    `test_support_assistant_v3` 모듈에만 적용.
    v4 테스트(test_support_assistant_v4) 및 다른 모든 테스트에는 영향 없음
    (request.node.module.__name__ 체크).

    ## 교체 대상
    `monglepick.agents.support_assistant.graph.support_assistant_graph`
    → `support_assistant_graph_v3` (모듈 내 이미 컴파일된 싱글턴)

    patch() 의 컨텍스트 매니저가 테스트 종료 후 자동으로 원래 v4 그래프를 복원한다.
    """
    # v3 테스트 파일에서만 활성화
    if "test_support_assistant_v3" not in request.node.module.__name__:
        yield
        return

    # support_assistant_graph (v4 싱글턴) 을 v3 컴파일 그래프로 교체
    with patch.object(
        _sa_graph_module,
        "support_assistant_graph",
        _sa_graph_module.support_assistant_graph_v3,
    ):
        yield
