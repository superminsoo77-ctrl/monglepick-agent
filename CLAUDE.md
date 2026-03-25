# monglepick-agent — AI Agent 서비스

> FastAPI + LangGraph + Ollama | 포트 8000 | Python 3.11+ | 패키지매니저 uv

## 빠른 시작

```bash
cd /Users/yoonhyungjoo/Documents/monglepick/monglepick-agent
PYTHONPATH=src uv run uvicorn monglepick.main:app --reload          # 서버
PYTHONPATH=src uv run --with pytest --with pytest-asyncio --with httpx -- python -m pytest tests/ -v  # 테스트
uv run ruff check src/ --fix && uv run ruff format src/             # 린트+포맷
```

## 프로젝트 구조

```
src/monglepick/
├── api/              # FastAPI 라우터 (chat.py: SSE/sync/upload, point_client.py)
├── agents/
│   ├── chat/         # 14노드 Chat Agent (graph.py, nodes.py, models.py)
│   └── recommendation/  # 7노드 추천 서브그래프
├── chains/           # LLM 체인 7개 (intent_emotion, preference, question, explanation, general, image_analysis, tool_executor)
├── prompts/          # 프롬프트 템플릿 11개
├── rag/              # 하이브리드 검색 (Qdrant+ES+Neo4j → RRF k=60)
├── data_pipeline/    # 데이터 수집/정제/임베딩/적재
├── llm/              # ChatOllama 팩토리 (싱글턴)
├── memory/           # Redis 세션 저장소 (TTL 30일)
├── db/               # DB 클라이언트 (Qdrant/Neo4j/Redis/ES/MySQL)
└── utils/
```

## 코딩 컨벤션

- **모든 함수**: `async def`, 반환 타입 `dict` (LangGraph 컨벤션)
- **에러 처리**: try/except → 유효한 fallback 반환 (에러 전파 금지)
- **State**: TypedDict (LangGraph 최상위) + Pydantic BaseModel (하위)
- **로깅**: structlog + `@traceable` 데코레이터
- **주석**: 한국어 상세 주석 필수
- **린트**: ruff (line-length=120, rules E/F/I/N/W)
- **임베딩**: Upstage Solar 4096차원 (100 RPM 제한)

## LLM 모델

| 역할 | 모델 | temperature |
|------|------|-------------|
| 대화/추천이유/선호추출 | `LGAI-EXAONE/exaone4.0:32b` | < 0.6 |
| 의도+감정/이미지분석 | `qwen3.5:35b-a3b` | 0.1 |

## 주요 설계 규칙

1. **선호 충분성**: genre(2.0)+mood(2.0)+context(1.0)+platform(1.0)+reference(1.5)+era(0.5)+exclude(0.5) ≥ 2.5 OR turn_count ≥ 2
2. **CF+CBF 가중치**: Cold(CBF 100%), Warm(CF 50%), 정상(CF 60%)
3. **MMR**: λ=0.7 (점수 0.7, 다양성 0.3)
4. **세션**: Redis TTL 30일, 영속 필드 6개
5. **RRF**: k=60 (Qdrant+ES+Neo4j 합산)

## 테스트

- 308개 테스트 (unit + integration)
- `tests/` 디렉토리 내 27개 파일
- `conftest.py`에서 mock DB 설정
