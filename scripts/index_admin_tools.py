"""
관리자 Tool RAG 1회 인덱싱 스크립트 (Step 7a, 2026-04-27).

ADMIN_TOOL_REGISTRY 76개 tool 의 description + example_questions 를 임베딩하여
Qdrant `admin_tool_registry` 컬렉션에 UPSERT 한다.

실행 방법 (monglepick-agent/ 디렉토리에서):
  PYTHONPATH=src uv run python scripts/index_admin_tools.py

사전 조건:
  - .env 에 QDRANT_URL, UPSTAGE_API_KEY 설정
  - Qdrant 서비스 실행 중 (보통 :6333)
  - ADMIN_TOOL_REGISTRY 가 import 시점에 자동 등록 (admin_tools/__init__.py 의 side-effect)

주의:
  - 서비스 기동 시 자동 실행되지 않는다 (기동 지연 + 운영 환경 예상치 못한 재인덱싱 방지).
  - tool 추가/수정 후 이 스크립트를 다시 실행해 재인덱싱한다.
  - `_tool_name_to_uuid` 가 deterministic 이므로 여러 번 실행해도 같은 ID 로 UPSERT — 멱등.
  - Upstage API 비용: 76개 tool × 평균 ~200 토큰 ≈ 15,200 토큰 (1회 실행 기준, 소액).
"""

from __future__ import annotations

import asyncio
import sys
import time


async def main() -> None:
    """인덱싱 메인 루틴."""
    print("=" * 60)
    print("몽글픽 관리자 Tool RAG 인덱싱 시작")
    print("=" * 60)

    # 1) 컬렉션 보장 (없으면 생성)
    print("\n[1/3] Qdrant 컬렉션 확인/생성 중...")
    from monglepick.tools.admin_tools.tool_rag import (
        ADMIN_TOOL_COLLECTION,
        ensure_admin_tool_collection,
        upsert_admin_tool_embeddings,
    )

    try:
        await ensure_admin_tool_collection()
        print(f"      컬렉션 '{ADMIN_TOOL_COLLECTION}' 준비 완료")
    except Exception as e:
        print(f"[오류] 컬렉션 생성 실패: {e}")
        print("      Qdrant 서비스 실행 여부와 QDRANT_URL 환경변수를 확인하세요.")
        sys.exit(1)

    # 2) 레지스트리 크기 확인 (admin_tools 패키지 import 시 76개 자동 등록)
    from monglepick.tools.admin_tools import ADMIN_TOOL_REGISTRY

    n = len(ADMIN_TOOL_REGISTRY)
    print(f"\n[2/3] ADMIN_TOOL_REGISTRY 로드 — {n}개 tool")
    if n == 0:
        print("[오류] 레지스트리가 비어 있음. admin_tools/__init__.py 의 import side-effect 를 확인하세요.")
        sys.exit(1)

    # 3) 임베딩 + UPSERT
    print(f"\n[3/3] 임베딩 + UPSERT 시작 (Upstage embedding-passage, dim=4096)")
    started = time.monotonic()
    try:
        count = await upsert_admin_tool_embeddings()
    except Exception as e:
        print(f"[오류] 인덱싱 실패: {e}")
        print("      UPSTAGE_API_KEY 환경변수와 Upstage 계정 quota 를 확인하세요.")
        sys.exit(1)

    elapsed_sec = time.monotonic() - started
    print(f"      UPSERT 완료 — {count}개 / {elapsed_sec:.1f}초")

    print("\n" + "=" * 60)
    print(f"✅ 완료: {ADMIN_TOOL_COLLECTION} 에 {count}개 tool 인덱싱")
    print("=" * 60)
    print("\n다음 단계:")
    print("  - tool_selector 노드에서 search_similar_tools() 호출로 활용")
    print("  - tool 추가/수정 후 본 스크립트를 다시 실행 (멱등 UPSERT)")


if __name__ == "__main__":
    asyncio.run(main())
