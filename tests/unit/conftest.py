"""
unit 테스트 공통 conftest.

Phase 2.5 cleanup (2026-04-28) 에서 v3 support_assistant 테스트와 v3 graph 가 모두
삭제되면서 `_v3_compat_graph_swap` autouse fixture 도 함께 제거되었다.
v4 테스트는 자체적으로 `_patch_intent` / `_patch_select_tool` 등의 헬퍼로 mock 을
관리하므로 conftest 차원의 자동 fixture 가 필요 없다.

향후 모든 테스트 모듈에 공통 적용해야 하는 fixture 가 생기면 이 파일에 추가하면 된다.
"""

from __future__ import annotations
