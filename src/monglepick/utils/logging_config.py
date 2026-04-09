"""
structlog 기반 통합 로깅 설정.

역할:
    - ELK 스택(Filebeat → Logstash → Elasticsearch) 에 공급할 구조화 로그를 출력한다.
    - 운영 환경에서는 JSON 포맷으로 stdout 출력 → docker json-file driver → Filebeat → Logstash
      파이프라인에서 ``fields.log_type == "agent_json"`` 분기로 파싱된다.
    - 개발 환경에서는 사람이 읽기 쉬운 컬러 콘솔 출력을 사용한다.

로그 필드 (JSON 모드):
    - timestamp     : ISO8601 UTC
    - level         : debug / info / warning / error / critical
    - logger        : 모듈 또는 logger 이름
    - event         : 메시지 (structlog 의 기본 메시지 필드)
    - trace_id      : (있는 경우) 요청 추적 ID
    - span_id       : (있는 경우) LangGraph 노드 단위 span
    - *             : 추가 keyword arguments 는 top-level 필드로 전개

환경변수:
    - LOG_FORMAT : "json"(기본 운영) / "console"(개발) — config.py 의 LOG_FORMAT 을 사용
    - LOG_LEVEL  : 기본 INFO
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


def configure_logging(
    log_level: str | None = None,
    log_format: str | None = None,
) -> None:
    """
    structlog + 표준 logging 을 초기화한다.

    stdlib ``logging`` 을 통해 출력되는 서드파티(FastAPI/uvicorn/sqlalchemy/langchain 등)
    로그도 동일한 structlog 프로세서 체인을 거쳐 JSON 으로 출력되도록 래핑한다.

    Args:
        log_level: "DEBUG"/"INFO"/"WARNING"/"ERROR" — 미지정 시 환경변수 LOG_LEVEL 또는 INFO
        log_format: "json" 또는 "console" — 미지정 시 환경변수 LOG_FORMAT 또는 json
    """
    # 1. 로그 레벨/포맷 결정
    level_str = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)
    fmt = (log_format or os.getenv("LOG_FORMAT", "json")).lower()

    # 2. 공통 프로세서 체인
    # structlog 프로세서: 이벤트 dict 에 필드를 추가하는 체인형 변환기
    shared_processors: list[Any] = [
        # stdlib logging 의 extra 인자를 이벤트 dict 에 merge
        structlog.contextvars.merge_contextvars,
        # logger 이름 자동 주입
        structlog.stdlib.add_logger_name,
        # level 추가
        structlog.stdlib.add_log_level,
        # logger.info("event %s", arg) 포맷 string % 치환 지원
        structlog.stdlib.PositionalArgumentsFormatter(),
        # ISO8601 UTC timestamp
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # 스택 정보 자동 캡처 (exc_info=True 인 경우)
        structlog.processors.StackInfoRenderer(),
        # exception 객체를 포맷팅
        structlog.processors.format_exc_info,
        # Unicode 안전 처리
        structlog.processors.UnicodeDecoder(),
    ]

    # 3. 최종 렌더러 선택 (JSON vs Console)
    if fmt == "json":
        # 운영: JSON 라인 출력 — Logstash ingest 와 호환
        final_processor = structlog.processors.JSONRenderer(sort_keys=False, ensure_ascii=False)
    else:
        # 개발: 컬러 콘솔 출력
        final_processor = structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback)

    # 4. structlog 자체 설정
    structlog.configure(
        processors=shared_processors
        + [
            # stdlib 로거로 넘어갈 때 필요한 마무리 단계
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    # 5. 표준 logging 의 Root 핸들러를 structlog 포맷터로 교체.
    # 이렇게 하면 logging.getLogger("uvicorn.access") 같은 서드파티 로거도
    # JSON 라인으로 통일된다.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            # structlog 내부 메타(_record, _from_structlog) 필드 제거
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            final_processor,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # 기존 핸들러 제거 (uvicorn/FastAPI 가 이미 추가했을 수 있음)
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # 6. 시끄러운 서드파티 로거 레벨 낮추기
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
