"""
기본 API 라우터.

헬스 체크용 ping 엔드포인트를 제공한다.
main.py에서 /api/v1 접두사로 등록된다.
"""

from fastapi import APIRouter

api_router = APIRouter(tags=["system"])


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
