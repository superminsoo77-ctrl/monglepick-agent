"""
Upstage Solar 임베딩 생성기.

Upstage Embedding API (OpenAI 호환)를 사용하여 텍스트를 벡터로 변환한다.

공식 문서: https://console.upstage.ai/docs/capabilities/embed

모델:
- embedding-passage: 문서/passage 임베딩 (긴 텍스트 단락용)
- embedding-query:   검색 쿼리 임베딩 (짧은 질문/검색어용)

API:
- Base URL: https://api.upstage.ai/v1
- 인증: Authorization: Bearer {UPSTAGE_API_KEY}
- Rate Limit: 100 RPM / 300,000 TPM
- 벡터 정규화: magnitude=1 (코사인 유사도 = 내적)

OpenAI 호환 API이므로 openai 패키지를 그대로 사용한다.
"""

from __future__ import annotations

import asyncio

import numpy as np
import structlog
from openai import OpenAI

from monglepick.config import settings

logger = structlog.get_logger()

# Upstage API 클라이언트 (싱글턴)
_client: OpenAI | None = None

# Upstage API Base URL (공식 문서 기준)
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"

# Rate Limit: 100 RPM → 안전하게 배치당 딜레이 추가
RATE_LIMIT_DELAY = 0.7  # 초 (100 RPM = 1.67 req/sec → 여유 확보)


def _get_client() -> OpenAI:
    """
    Upstage API 클라이언트를 반환한다 (싱글턴).

    모듈 레벨에서 한 번만 초기화되며, 이후 호출에서는 캐시된 인스턴스를 재사용한다.
    OpenAI 호환 API이므로 openai.OpenAI 클라이언트를 base_url만 변경하여 사용한다.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.UPSTAGE_API_KEY,
            base_url=UPSTAGE_BASE_URL,
        )
        logger.info("upstage_embedding_client_initialized")
    return _client


def embed_texts(texts: list[str], batch_size: int = 50) -> np.ndarray:
    """
    텍스트 리스트를 벡터로 변환한다 (문서/passage 용도).

    Upstage 'embedding-passage' 모델을 사용한다.
    긴 텍스트 단락에 최적화되어 있다.

    Rate Limit (100 RPM)을 준수하기 위해 배치 간 딜레이를 적용한다.

    Args:
        texts: 임베딩할 텍스트 리스트
        batch_size: API 배치 크기 (기본 50, TPM 고려)

    Returns:
        np.ndarray: shape (len(texts), embedding_dimension)
    """
    client = _get_client()
    all_embeddings: list[list[float]] = []

    # 배치 단위로 API 호출
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        response = client.embeddings.create(
            model="embedding-passage",
            input=batch,
        )

        # 응답의 data 배열 순서가 입력 순서와 다를 수 있으므로 index 기준 정렬
        batch_embeddings = [
            item.embedding
            for item in sorted(response.data, key=lambda x: x.index)
        ]
        all_embeddings.extend(batch_embeddings)

        completed = min(i + batch_size, len(texts))
        if completed % 500 == 0 or completed >= len(texts):
            logger.info("embedding_progress", completed=completed, total=len(texts))

        # 마지막 배치가 아닌 경우에만 Rate Limit 딜레이 적용 (100 RPM 준수)
        if i + batch_size < len(texts):
            import time
            time.sleep(RATE_LIMIT_DELAY)

    result = np.array(all_embeddings)
    logger.info("texts_embedded", count=len(texts), dimension=result.shape[1])
    return result


def embed_query(query: str) -> np.ndarray:
    """
    검색 쿼리를 벡터로 변환한다.

    Upstage 'embedding-query' 모델을 사용한다.
    짧은 검색 질문에 최적화되어 있으며, embed_texts(passage)와 동일 벡터 공간을 공유한다.

    Args:
        query: 검색 쿼리 텍스트 (예: "우울한데 볼 만한 영화")

    Returns:
        np.ndarray: shape (embedding_dimension,) — 4096차원 벡터
    """
    client = _get_client()

    response = client.embeddings.create(
        model="embedding-query",
        input=[query],
    )

    return np.array(response.data[0].embedding)


async def embed_query_async(query: str) -> np.ndarray:
    """
    embed_query의 비동기 래퍼. event loop 블로킹을 방지한다.

    embed_query()는 동기 HTTP 호출(Upstage API)을 수행하므로
    async 함수 내에서 직접 호출하면 event loop가 수백ms 동안 블로킹된다.
    asyncio.to_thread()로 별도 스레드에서 실행하여 블로킹을 방지한다.

    Args:
        query: 검색 쿼리 텍스트

    Returns:
        np.ndarray: shape (embedding_dimension,) — 4096차원 벡터
    """
    return await asyncio.to_thread(embed_query, query)
