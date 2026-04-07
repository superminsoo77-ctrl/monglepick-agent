"""
사용자 리뷰 이력 MySQL 조회 도구 (Phase 6 Tool 6).

MySQL reviews 테이블에서 사용자의 영화 리뷰(= 시청 확인) 이력을 조회한다.
"내가 본 영화 알려줘" 또는 "내가 본 거 빼고 추천해줘" 등의
사용자 요청에 응답하거나, 추천 제외 목록 구성에 활용된다.

설계 결정:
    리뷰 작성 = 시청 완료 확인의 유일한 소스.
    watch_history는 Kaggle 26M 행 CF 학습용 시드 데이터 전용이며
    실제 서비스 사용자 행동을 기록하지 않는다.
    따라서 "본 영화" 판정은 reviews 테이블 기준으로 통일한다.

MySQL 스키마:
    reviews  (user_id, movie_id, rating, created_at, is_deleted, review_source)
    movies   (movie_id, title, genres, director, cast_members, mood_tags, ...)

aiomysql.DictCursor를 사용해 결과를 dict로 반환한다.
"""

from __future__ import annotations

import asyncio
import json

import aiomysql
import structlog
from langchain_core.tools import tool

from monglepick.db.clients import get_mysql

logger = structlog.get_logger()

# MySQL 쿼리 타임아웃 (초) — 커넥션 풀 획득 포함
_MYSQL_TIMEOUT_SEC = 5.0


@tool
async def user_history(
    user_id: str,
    limit: int = 20,
) -> list[dict]:
    """
    MySQL reviews 테이블에서 사용자가 리뷰를 남긴 영화 목록(= 시청 확인 이력)을 조회한다.

    사용자가 "내가 본 영화" 또는 "이미 본 거 빼고"라고 요청할 때 활용한다.
    익명 사용자(user_id 빈 문자열)는 빈 리스트를 반환한다.

    Args:
        user_id: 사용자 ID (예: "user_abc123"). 빈 문자열이면 익명으로 처리.
        limit: 조회할 최대 리뷰 이력 수 (기본 20, 최대 100). 최신순 정렬.

    Returns:
        리뷰(= 시청 확인) 이력 dict 목록 (최신 작성 순 내림차순):
        [
            {
                "movie_id": str,       # 영화 ID
                "title": str,          # 영화 제목
                "watched_at": str,     # 리뷰 작성 일시 (ISO 8601 문자열, ≈ 시청 일시)
                "rating": float,       # 사용자 평점 (0.0이면 평점 없음)
                "genres": list[str],   # 장르 목록
                "director": str,       # 감독명
                "review_source": str,  # 리뷰 작성 경로 (예: "detail", "chat_ses_001")
            }
        ]
        user_id 없음, DB 오류, 타임아웃 시: 빈 리스트 반환 (에러 전파 금지).
    """
    # 익명 사용자: DB 조회 없이 즉시 반환
    if not user_id:
        logger.info("user_history_tool_anonymous")
        return []

    # limit 범위 보정 (최소 1, 최대 100)
    safe_limit = max(1, min(int(limit), 100))

    try:
        pool = await get_mysql()

        async def _query() -> list[dict]:
            """
            MySQL 커넥션 풀에서 시청 이력을 조회하는 내부 코루틴.

            aiomysql.DictCursor를 사용하여 row를 dict로 반환한다.
            context_loader의 쿼리 패턴을 재사용하되 필요한 컬럼만 선택한다.
            """
            async with pool.acquire() as conn:
                # DictCursor: 결과 row를 컬럼명을 키로 하는 dict로 반환
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    # reviews 테이블 = 리뷰 작성 ≈ 시청 완료 확인의 단일 소스.
                    # watch_history는 Kaggle CF 시드 데이터 전용이므로 쿼리 대상에서 제외.
                    # is_deleted=false 필터로 삭제된 리뷰 제외.
                    # review_source: 어느 경로에서 리뷰를 작성했는지 (detail/chat_ses_xxx 등)
                    await cursor.execute(
                        """
                        SELECT
                            r.movie_id,
                            m.title,
                            r.created_at  AS watched_at,
                            r.rating,
                            m.genres,
                            m.director,
                            r.review_source
                        FROM reviews r
                        LEFT JOIN movies m ON r.movie_id = m.movie_id
                        WHERE r.user_id = %s
                          AND r.is_deleted = false
                        ORDER BY r.created_at DESC
                        LIMIT %s
                        """,
                        (user_id, safe_limit),
                    )
                    rows = await cursor.fetchall()
                    return list(rows)

        # 타임아웃 래핑 — 커넥션 풀 대기 + 쿼리 실행 포함
        raw_rows: list[dict] = await asyncio.wait_for(
            _query(),
            timeout=_MYSQL_TIMEOUT_SEC,
        )

        # 후처리: JSON 문자열 → list, datetime → ISO 8601 문자열
        results: list[dict] = []
        for row in raw_rows:
            # genres: DB에 JSON 문자열로 저장된 경우 파싱
            genres = row.get("genres", [])
            if isinstance(genres, str):
                try:
                    parsed = json.loads(genres)
                    genres = parsed if isinstance(parsed, list) else []
                except (json.JSONDecodeError, ValueError):
                    genres = []

            # watched_at: aiomysql이 datetime 객체로 반환 → ISO 8601 문자열 변환
            watched_at = row.get("watched_at")
            if watched_at is not None and hasattr(watched_at, "isoformat"):
                watched_at = watched_at.isoformat()
            elif watched_at is None:
                watched_at = ""

            results.append({
                "movie_id": str(row.get("movie_id", "")),
                "title": row.get("title", ""),
                "watched_at": str(watched_at),
                # rating: NULL이면 0.0으로 처리 (평점 미입력 리뷰)
                "rating": float(row.get("rating") or 0.0),
                "genres": genres,
                "director": row.get("director", ""),
                # review_source: 어느 경로에서 리뷰 작성 (detail, chat_ses_xxx 등)
                "review_source": row.get("review_source") or "",
            })

        logger.info(
            "user_history_tool_done",
            user_id=user_id,
            result_count=len(results),
            recent_titles=[r.get("title", "") for r in results[:3]],
        )
        return results

    except asyncio.TimeoutError:
        logger.error(
            "user_history_tool_timeout",
            user_id=user_id,
            timeout_sec=_MYSQL_TIMEOUT_SEC,
        )
        return []

    except Exception as e:
        # DB 연결 실패, 쿼리 오류 등 모든 예외 처리 (에러 전파 금지)
        logger.error(
            "user_history_tool_error",
            user_id=user_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return []
