"""
기본 API 라우터.

헬스 체크용 ping 엔드포인트를 제공한다.
Phase 7에서 content_router, roadmap_router를 추가 등록한다.
main.py에서 /api/v1 접두사로 등록된다.
"""

from fastapi import APIRouter

from monglepick.api.content import content_router
from monglepick.api.roadmap import roadmap_router

api_router = APIRouter(tags=["system"])

# ── Phase 7: 콘텐츠 분석 + 로드맵 에이전트 라우터 등록 ──
api_router.include_router(content_router)
api_router.include_router(roadmap_router)


@api_router.get(
    "/ping",
    summary="서버 핑",
    responses={
        200: {
            "description": "서버 생존 확인",
            "content": {"application/json": {"example": {"message": "pong"}}},
        }
    },
)
async def ping():
    """서버 생존 확인용 핑 엔드포인트. 200 OK + {"message": "pong"} 반환."""
    return {"message": "pong"}
