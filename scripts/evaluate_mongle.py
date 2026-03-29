"""
몽글이 품질 자동 평가 스크립트 (M-LLM-6).

Ollama에 등록된 몽글이 모델의 품질을 4가지 항목으로 자동 평가한다.
파인튜닝 완료 후, hybrid 모드 전환 전에 반드시 실행하여 PASS 여부를 확인한다.

평가 항목 (설계서 §6-3):
  1. 페르소나 일관성 (목표: ≥ 85%)
     - 몽글이 말투("~요", "~세요", "몽글" 키워드 등) 유지율
     - Solar API로 1~5점 자동 채점, 4점 이상이면 페르소나 유지로 판정
  2. 한국어 자연스러움 (목표: 오류율 ≤ 10%)
     - 반복 문자열, 깨진 문장, 빈 응답 등 규칙 기반 자동 검출
     - Solar API 추가 채점으로 문법 오류율 보완
  3. 후속 질문 적절성 (목표: 평균 ≥ 3.5/5)
     - 20개 고정 시나리오로 질문 생성 → Solar API가 1~5점 채점
  4. 응답 속도 (목표: ≥ 40 tok/s)
     - Ollama REST API /api/generate 엔드포인트로 10회 측정 후 평균

사용법:
    PYTHONPATH=src uv run python scripts/evaluate_mongle.py \\
      --model mongle \\
      --eval_data data/finetune/mongle_eval.jsonl \\
      --output data/finetune/eval_results.json

    # Solar API 채점 없이 규칙 기반만 실행 (빠른 검증)
    PYTHONPATH=src uv run python scripts/evaluate_mongle.py \\
      --model mongle \\
      --eval_data data/finetune/mongle_eval.jsonl \\
      --output data/finetune/eval_results.json \\
      --no-solar

전제 조건:
    - Ollama 서버가 실행 중이어야 한다 (기본: localhost:11434)
    - UPSTAGE_API_KEY 환경변수가 설정되어 있어야 한다 (Solar 채점 사용 시)
    - PYTHONPATH=src 로 실행해야 monglepick 패키지를 찾을 수 있다
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# ── 프로젝트 루트를 sys.path에 추가 ──
# PYTHONPATH=src 로 실행하지 않는 경우를 대비한 fallback
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monglepick.config import settings


# ============================================================
# 상수 정의
# ============================================================

# Ollama REST API 기본 URL
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL  # 기본: "http://localhost:11434"

# Solar API (Upstage) 채점 엔드포인트 — OpenAI 호환 ChatCompletion
SOLAR_API_URL = f"{settings.SOLAR_API_BASE_URL}/completions"

# 평가 PASS 기준 (설계서 §6-3)
THRESHOLD_PERSONA: float = 85.0     # 페르소나 일관성: 85% 이상
THRESHOLD_GRAMMAR_ERROR: float = 10.0  # 한국어 오류율: 10% 이하
THRESHOLD_QUESTION: float = 3.5     # 후속 질문 적절성: 5점 만점 중 3.5 이상
THRESHOLD_SPEED: float = 40.0       # 응답 속도: 40 tok/s 이상

# 속도 측정 반복 횟수
SPEED_BENCHMARK_ROUNDS = 10

# 후속 질문 평가 고정 시나리오 20개
# 각 항목은 {"missing_fields": [...], "user_message": "..."} 구조
QUESTION_SCENARIOS: list[dict[str, Any]] = [
    # 1~5: 장르 누락
    {"missing_fields": ["장르"], "user_message": "영화 추천해줘"},
    {"missing_fields": ["장르"], "user_message": "볼만한 거 없어?"},
    {"missing_fields": ["장르"], "user_message": "뭔가 재밌는 영화 보고 싶어"},
    {"missing_fields": ["장르"], "user_message": "오늘 심심한데 영화나 볼까"},
    {"missing_fields": ["장르", "분위기"], "user_message": "기분 전환용 영화 추천"},
    # 6~10: 분위기 누락
    {"missing_fields": ["분위기"], "user_message": "액션 영화 추천해줘"},
    {"missing_fields": ["분위기"], "user_message": "공포 영화 추천해줘"},
    {"missing_fields": ["분위기"], "user_message": "코미디 영화 볼래"},
    {"missing_fields": ["분위기", "상황"], "user_message": "스릴러 추천해줘"},
    {"missing_fields": ["분위기"], "user_message": "로맨스 영화 추천해줘"},
    # 11~15: 상황 누락
    {"missing_fields": ["상황"], "user_message": "친구랑 볼 영화 추천해줘"},
    {"missing_fields": ["상황"], "user_message": "혼자 볼 영화 추천해줘"},
    {"missing_fields": ["상황", "장르"], "user_message": "가족이랑 볼 영화 뭐 있어?"},
    {"missing_fields": ["상황"], "user_message": "데이트하면서 볼 영화 추천해줘"},
    {"missing_fields": ["상황"], "user_message": "야식 먹으면서 볼 영화 추천"},
    # 16~20: 복합 누락
    {"missing_fields": ["장르", "분위기", "상황"], "user_message": "추천해줘"},
    {"missing_fields": ["시대", "장르"], "user_message": "클래식 영화 추천해줘"},
    {"missing_fields": ["플랫폼"], "user_message": "넷플릭스에서 볼 거 추천해줘"},
    {"missing_fields": ["분위기", "플랫폼"], "user_message": "왓챠에서 볼 영화 추천"},
    {"missing_fields": ["장르", "시대"], "user_message": "80년대 영화 보고 싶어"},
]

# 페르소나 일관성 평가 프롬프트 (Solar API 채점용)
PERSONA_JUDGE_PROMPT = """\
다음은 영화 추천 AI '몽글이'의 응답입니다.
몽글이의 정체성: 친근하고 따뜻한 말투, 존댓말 기본, 영화에 대한 열정, 사용자 감정 공감.

응답: {response}

아래 기준으로 1~5점을 부여하세요. 점수만 숫자로 답하세요.
5점: 완벽하게 몽글이 페르소나를 유지 (존댓말, 친근함, 공감)
4점: 대체로 유지 (소소한 어색함)
3점: 절반 정도 유지
2점: 페르소나가 거의 느껴지지 않음
1점: 전혀 다른 톤/말투

점수 (1~5):"""

# 한국어 자연스러움 채점 프롬프트 (Solar API 채점용)
GRAMMAR_JUDGE_PROMPT = """\
다음 한국어 텍스트의 문법과 자연스러움을 평가하세요.

텍스트: {response}

아래 기준으로 1~5점을 부여하세요. 점수만 숫자로 답하세요.
5점: 완벽한 한국어, 자연스럽고 어색함 없음
4점: 약간의 어색함, 전달에 문제 없음
3점: 문법 오류 1~2개 또는 다소 어색한 표현
2점: 문법 오류 여러 개, 이해에 약간 어려움
1점: 문법 오류 다수, 의미 파악 어려움

점수 (1~5):"""

# 후속 질문 적절성 채점 프롬프트 (Solar API 채점용)
QUESTION_JUDGE_PROMPT = """\
다음은 영화 추천 AI가 사용자에게 부족한 정보를 얻기 위해 생성한 질문입니다.

사용자 메시지: {user_message}
부족한 정보 항목: {missing_fields}
AI 생성 질문: {question}

아래 기준으로 1~5점을 부여하세요. 점수만 숫자로 답하세요.
5점: 부족한 항목을 정확히 묻고, 친근하고 자연스러움
4점: 관련성 있고 대체로 자연스러움
3점: 부족한 항목과 어느 정도 관련 있음
2점: 질문이 부족한 항목과 거의 무관함
1점: 엉뚱한 질문 또는 응답이 질문 형태가 아님

점수 (1~5):"""


# ============================================================
# 유틸리티: Ollama REST API 호출
# ============================================================

async def ollama_generate(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    system: str | None = None,
) -> dict[str, Any]:
    """
    Ollama REST API /api/generate 를 호출하여 텍스트를 생성한다.

    Args:
        client: httpx 비동기 클라이언트
        model: Ollama 모델명 (예: "mongle")
        prompt: 사용자 입력 프롬프트
        system: 시스템 프롬프트 (없으면 Modelfile 기본 사용)

    Returns:
        Ollama API 응답 dict (response, eval_count, eval_duration 포함)

    Raises:
        httpx.HTTPStatusError: Ollama API 오류 시
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # 스트리밍 없이 전체 응답 수신
    }
    if system:
        payload["system"] = system

    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120.0,  # 첫 응답까지 최대 2분 허용
    )
    response.raise_for_status()
    return response.json()


async def solar_score(
    client: httpx.AsyncClient,
    prompt: str,
    api_key: str,
) -> int:
    """
    Solar API (Upstage) 로 1~5점 채점을 요청하고 정수를 반환한다.

    OpenAI 호환 ChatCompletion API를 사용한다.
    응답에서 숫자를 파싱하지 못하면 기본값 3(중간)을 반환한다.

    Args:
        client: httpx 비동기 클라이언트
        prompt: 채점 지시가 담긴 전체 프롬프트
        api_key: Upstage API 키

    Returns:
        1~5 정수 점수 (파싱 실패 시 3)
    """
    try:
        response = await client.post(
            SOLAR_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.SOLAR_API_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.0,  # 채점은 결정론적으로
            },
            timeout=30.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        # 응답에서 첫 번째 숫자 추출
        match = re.search(r"[1-5]", content)
        if match:
            return int(match.group())
    except Exception as exc:
        # Solar API 장애 시 중간값(3) 반환 — 평가 전체 중단 방지
        print(f"  [경고] Solar 채점 실패: {exc}")
    return 3  # 기본값: 중간 점수


# ============================================================
# 평가 항목 1: 페르소나 일관성
# ============================================================

def _check_persona_rules(response: str) -> bool:
    """
    규칙 기반 페르소나 일관성 1차 검사.

    존댓말 어미("~요", "~세요", "~드릴게요"), "몽글" 키워드,
    또는 친근한 표현("어떤", "~해볼까요") 중 하나라도 있으면 통과.

    Args:
        response: 몽글이 생성 응답 텍스트

    Returns:
        True이면 규칙 기반 페르소나 통과
    """
    # 존댓말 어미 패턴
    polite_patterns = [
        r"요\b",          # "~요"로 끝나는 문장
        r"세요\b",         # "~세요"
        r"드릴게요",       # "~드릴게요"
        r"해요",           # "~해요"
        r"었어요",         # "~었어요"
        r"겠습니다",       # "~겠습니다"
        r"합니다",         # "~합니다"
    ]
    # 몽글이 특징 키워드
    persona_keywords = ["몽글", "어떤", "추천", "영화"]

    has_polite = any(re.search(p, response) for p in polite_patterns)
    has_keyword = any(kw in response for kw in persona_keywords)

    return has_polite or has_keyword


async def evaluate_persona(
    client: httpx.AsyncClient,
    model: str,
    eval_data: list[dict[str, Any]],
    api_key: str,
    use_solar: bool,
) -> dict[str, Any]:
    """
    페르소나 일관성 평가 항목 실행.

    eval_data의 각 샘플에 대해 몽글이 응답을 생성하고,
    규칙 기반 + Solar API 채점으로 페르소나 유지율을 계산한다.

    Args:
        client: httpx 비동기 클라이언트
        model: 평가할 Ollama 모델명
        eval_data: 평가 데이터 리스트 (각 항목: {"input": str, "output": str})
        api_key: Upstage API 키
        use_solar: Solar API 채점 사용 여부

    Returns:
        {
            "score_pct": float,   # 페르소나 유지율 (%)
            "pass": bool,         # PASS 여부 (≥ 85%)
            "samples": [...],     # 샘플별 상세 결과
        }
    """
    print(f"\n[1/4] 페르소나 일관성 평가 ({len(eval_data)}개 샘플)...")

    samples: list[dict[str, Any]] = []
    pass_count = 0

    for i, item in enumerate(eval_data):
        user_input = item.get("input", "")
        if not user_input:
            continue

        print(f"  [{i+1}/{len(eval_data)}] 입력: {user_input[:40]}...")

        try:
            # 몽글이 응답 생성
            result = await ollama_generate(client, model, user_input)
            response_text = result.get("response", "").strip()
        except Exception as exc:
            print(f"  [오류] 응답 생성 실패: {exc}")
            samples.append({"input": user_input, "response": "", "rule_pass": False, "solar_score": 1, "passed": False})
            continue

        # 1차: 규칙 기반 검사
        rule_pass = _check_persona_rules(response_text)

        # 2차: Solar API 채점 (use_solar=True이고 API 키가 있을 때만)
        solar_sc = 3  # 기본값
        if use_solar and api_key:
            prompt = PERSONA_JUDGE_PROMPT.format(response=response_text)
            solar_sc = await solar_score(client, prompt, api_key)
            # Solar API 4점 이상 또는 규칙 통과이면 페르소나 유지로 판정
            passed = solar_sc >= 4 or rule_pass
        else:
            # Solar 없이: 규칙 기반만 사용
            passed = rule_pass

        if passed:
            pass_count += 1

        samples.append({
            "input": user_input,
            "response": response_text[:200],  # 너무 길면 잘라서 저장
            "rule_pass": rule_pass,
            "solar_score": solar_sc,
            "passed": passed,
        })
        print(f"    → 규칙={'OK' if rule_pass else 'FAIL'}, Solar={solar_sc}점, {'PASS' if passed else 'FAIL'}")

    total = len(samples)
    score_pct = (pass_count / total * 100) if total > 0 else 0.0

    return {
        "score_pct": round(score_pct, 1),
        "pass": score_pct >= THRESHOLD_PERSONA,
        "threshold": THRESHOLD_PERSONA,
        "pass_count": pass_count,
        "total": total,
        "samples": samples,
    }


# ============================================================
# 평가 항목 2: 한국어 자연스러움
# ============================================================

def _detect_grammar_errors(response: str) -> list[str]:
    """
    규칙 기반 한국어 오류 자동 검출.

    검사 항목:
    - 빈 응답
    - 과도한 반복 (같은 어절 3회 이상 연속)
    - 깨진 문장 (Unicode 오류 또는 비정상 바이트)
    - 영어+한국어 혼용 과다 (30% 이상 영문)

    Args:
        response: 검사할 응답 텍스트

    Returns:
        발견된 오류 설명 리스트 (빈 리스트이면 오류 없음)
    """
    errors: list[str] = []

    # 1. 빈 응답 검사
    if not response or len(response.strip()) < 5:
        errors.append("빈 응답 또는 매우 짧은 응답")
        return errors  # 이후 검사 의미 없으므로 바로 반환

    # 2. 반복 어절 검사 (3회 이상 연속)
    # 예: "추천해요 추천해요 추천해요" → 오류
    words = response.split()
    for j in range(len(words) - 2):
        if words[j] == words[j + 1] == words[j + 2]:
            errors.append(f"반복 어절 감지: '{words[j]}'")
            break

    # 3. 깨진 유니코드 검사 (replacement character)
    if "\ufffd" in response:
        errors.append("유니코드 깨짐 (replacement character 포함)")

    # 4. 영문 비율 과다 검사
    total_chars = len(response.replace(" ", ""))
    english_chars = len(re.findall(r"[a-zA-Z]", response))
    if total_chars > 0 and (english_chars / total_chars) > 0.3:
        errors.append(f"영문 비율 과다: {english_chars/total_chars*100:.0f}%")

    # 5. 문장 종결어미 없이 끝나는 경우 (마지막 문자가 한글 자모로 끝남)
    last_char = response.strip()[-1] if response.strip() else ""
    # 한글 자모만 있으면 불완전한 문장으로 판단
    if last_char and re.match(r"[ㄱ-ㅎㅏ-ㅣ]", last_char):
        errors.append("문장이 자모로 끝남 (불완전한 응답)")

    return errors


async def evaluate_grammar(
    client: httpx.AsyncClient,
    model: str,
    eval_data: list[dict[str, Any]],
    api_key: str,
    use_solar: bool,
) -> dict[str, Any]:
    """
    한국어 자연스러움 평가 항목 실행.

    규칙 기반 오류 검출 + Solar API 채점으로 오류율을 계산한다.

    Args:
        client: httpx 비동기 클라이언트
        model: 평가할 Ollama 모델명
        eval_data: 평가 데이터 리스트
        api_key: Upstage API 키
        use_solar: Solar API 채점 사용 여부

    Returns:
        {
            "error_rate_pct": float,   # 오류율 (%)
            "pass": bool,              # PASS 여부 (≤ 10%)
            "samples": [...],
        }
    """
    print(f"\n[2/4] 한국어 자연스러움 평가 ({len(eval_data)}개 샘플)...")

    samples: list[dict[str, Any]] = []
    error_count = 0

    for i, item in enumerate(eval_data):
        user_input = item.get("input", "")
        if not user_input:
            continue

        print(f"  [{i+1}/{len(eval_data)}] 입력: {user_input[:40]}...")

        try:
            result = await ollama_generate(client, model, user_input)
            response_text = result.get("response", "").strip()
        except Exception as exc:
            print(f"  [오류] 응답 생성 실패: {exc}")
            # 생성 자체 실패는 오류로 처리
            samples.append({"input": user_input, "response": "", "rule_errors": ["생성 실패"], "solar_score": 1, "has_error": True})
            error_count += 1
            continue

        # 1차: 규칙 기반 오류 검출
        rule_errors = _detect_grammar_errors(response_text)

        # 2차: Solar API 채점 (3점 이하이면 오류로 간주)
        solar_sc = 4  # 기본값: 오류 없음으로 가정
        if use_solar and api_key:
            prompt = GRAMMAR_JUDGE_PROMPT.format(response=response_text)
            solar_sc = await solar_score(client, prompt, api_key)

        # 오류 판정: 규칙 오류 있거나 Solar 3점 이하
        has_error = bool(rule_errors) or (use_solar and api_key and solar_sc <= 3)

        if has_error:
            error_count += 1

        samples.append({
            "input": user_input,
            "response": response_text[:200],
            "rule_errors": rule_errors,
            "solar_score": solar_sc,
            "has_error": has_error,
        })
        error_label = f"오류({', '.join(rule_errors)})" if rule_errors else "OK"
        print(f"    → 규칙={error_label}, Solar={solar_sc}점, {'오류' if has_error else 'OK'}")

    total = len(samples)
    error_rate_pct = (error_count / total * 100) if total > 0 else 0.0

    return {
        "error_rate_pct": round(error_rate_pct, 1),
        "pass": error_rate_pct <= THRESHOLD_GRAMMAR_ERROR,
        "threshold": THRESHOLD_GRAMMAR_ERROR,
        "error_count": error_count,
        "total": total,
        "samples": samples,
    }


# ============================================================
# 평가 항목 3: 후속 질문 적절성
# ============================================================

async def evaluate_question_quality(
    client: httpx.AsyncClient,
    model: str,
    api_key: str,
    use_solar: bool,
) -> dict[str, Any]:
    """
    후속 질문 적절성 평가 항목 실행.

    20개 고정 시나리오(QUESTION_SCENARIOS)로 질문을 생성하고
    Solar API 채점으로 적절성을 평가한다.

    프롬프트 형식:
      부족한 정보 항목: [장르, 분위기]
      사용자 메시지: "영화 추천해줘"
      위 정보를 자연스럽게 묻는 질문을 한 가지 만들어주세요.

    Args:
        client: httpx 비동기 클라이언트
        model: 평가할 Ollama 모델명
        api_key: Upstage API 키
        use_solar: Solar API 채점 사용 여부

    Returns:
        {
            "avg_score": float,    # 5점 만점 평균 점수
            "pass": bool,          # PASS 여부 (≥ 3.5)
            "samples": [...],
        }
    """
    print(f"\n[3/4] 후속 질문 적절성 평가 ({len(QUESTION_SCENARIOS)}개 시나리오)...")

    samples: list[dict[str, Any]] = []
    total_score = 0

    for i, scenario in enumerate(QUESTION_SCENARIOS):
        missing_fields = scenario["missing_fields"]
        user_message = scenario["user_message"]

        # 몽글이에게 후속 질문 생성 요청 프롬프트
        prompt = (
            f"부족한 정보 항목: {', '.join(missing_fields)}\n"
            f"사용자 메시지: \"{user_message}\"\n"
            f"위 정보를 자연스럽게 묻는 질문을 한 가지만 만들어주세요."
        )

        print(f"  [{i+1}/{len(QUESTION_SCENARIOS)}] 누락: {missing_fields} | 메시지: {user_message}")

        try:
            result = await ollama_generate(client, model, prompt)
            question_text = result.get("response", "").strip()
        except Exception as exc:
            print(f"  [오류] 질문 생성 실패: {exc}")
            samples.append({"scenario": scenario, "question": "", "solar_score": 1})
            total_score += 1
            continue

        # Solar API 채점
        sc = 3  # 기본값: 보통
        if use_solar and api_key:
            judge_prompt = QUESTION_JUDGE_PROMPT.format(
                user_message=user_message,
                missing_fields=", ".join(missing_fields),
                question=question_text,
            )
            sc = await solar_score(client, judge_prompt, api_key)
        else:
            # Solar 없이: 질문 형태인지만 간단히 체크 (물음표 존재)
            sc = 4 if "?" in question_text or "까요" in question_text or "나요" in question_text else 2

        total_score += sc
        samples.append({
            "scenario": scenario,
            "question": question_text[:300],
            "solar_score": sc,
        })
        print(f"    → 생성: {question_text[:60]}... | 점수: {sc}/5")

    total = len(samples)
    avg_score = (total_score / total) if total > 0 else 0.0

    return {
        "avg_score": round(avg_score, 2),
        "pass": avg_score >= THRESHOLD_QUESTION,
        "threshold": THRESHOLD_QUESTION,
        "total": total,
        "samples": samples,
    }


# ============================================================
# 평가 항목 4: 응답 속도
# ============================================================

async def evaluate_speed(
    client: httpx.AsyncClient,
    model: str,
) -> dict[str, Any]:
    """
    응답 속도 벤치마크 실행.

    Ollama /api/generate 응답의 eval_count(생성 토큰 수)와
    eval_duration(나노초 단위 생성 시간)을 활용하여 tok/s를 계산한다.

    SPEED_BENCHMARK_ROUNDS회 측정 후 평균값을 최종 속도로 사용한다.
    첫 호출(cold start)은 모델 로딩 시간이 포함될 수 있으므로 포함해서 측정한다.

    Args:
        client: httpx 비동기 클라이언트
        model: 평가할 Ollama 모델명

    Returns:
        {
            "avg_tps": float,       # 평균 토큰/초
            "min_tps": float,       # 최솟값
            "max_tps": float,       # 최댓값
            "pass": bool,           # PASS 여부 (≥ 40 tok/s)
            "rounds": [...],        # 각 라운드별 측정값
        }
    """
    print(f"\n[4/4] 응답 속도 벤치마크 ({SPEED_BENCHMARK_ROUNDS}회 측정)...")

    # 고정된 벤치마크 프롬프트 (일정한 길이의 응답이 나오도록 유도)
    benchmark_prompt = "영화 '인터스텔라'를 좋아하는 사람에게 어울리는 영화를 추천해주세요."

    rounds: list[dict[str, Any]] = []
    tps_values: list[float] = []

    for i in range(SPEED_BENCHMARK_ROUNDS):
        print(f"  [{i+1}/{SPEED_BENCHMARK_ROUNDS}] 측정 중...", end=" ", flush=True)
        try:
            start_time = time.perf_counter()
            result = await ollama_generate(client, model, benchmark_prompt)
            elapsed = time.perf_counter() - start_time

            # Ollama API가 반환하는 eval_count, eval_duration 활용
            eval_count = result.get("eval_count", 0)
            eval_duration_ns = result.get("eval_duration", 0)

            if eval_duration_ns > 0:
                # eval_duration은 나노초 → 초로 변환
                tps = eval_count / (eval_duration_ns / 1_000_000_000)
            elif elapsed > 0 and eval_count > 0:
                # eval_duration 없을 경우 실제 경과 시간으로 대체 추정
                tps = eval_count / elapsed
            else:
                tps = 0.0

            tps_values.append(tps)
            rounds.append({
                "round": i + 1,
                "eval_count": eval_count,
                "eval_duration_ms": round(eval_duration_ns / 1_000_000, 1) if eval_duration_ns else round(elapsed * 1000, 1),
                "tps": round(tps, 1),
            })
            print(f"{eval_count} tok / {tps:.1f} tok/s")

        except Exception as exc:
            print(f"실패: {exc}")
            rounds.append({"round": i + 1, "error": str(exc), "tps": 0.0})

    valid_tps = [r["tps"] for r in rounds if "error" not in r and r["tps"] > 0]
    avg_tps = sum(valid_tps) / len(valid_tps) if valid_tps else 0.0
    min_tps = min(valid_tps) if valid_tps else 0.0
    max_tps = max(valid_tps) if valid_tps else 0.0

    return {
        "avg_tps": round(avg_tps, 1),
        "min_tps": round(min_tps, 1),
        "max_tps": round(max_tps, 1),
        "pass": avg_tps >= THRESHOLD_SPEED,
        "threshold": THRESHOLD_SPEED,
        "rounds": rounds,
    }


# ============================================================
# 결과 출력 (콘솔 테이블)
# ============================================================

def print_results_table(
    persona: dict[str, Any],
    grammar: dict[str, Any],
    question: dict[str, Any],
    speed: dict[str, Any],
) -> bool:
    """
    평가 결과를 콘솔 테이블로 출력하고 최종 PASS/FAIL을 반환한다.

    Args:
        persona: 페르소나 일관성 결과
        grammar: 한국어 자연스러움 결과
        question: 후속 질문 적절성 결과
        speed: 응답 속도 결과

    Returns:
        True이면 전체 PASS, False이면 FAIL
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print("  몽글이 품질 평가 결과")
    print(sep)
    print(f"  {'항목':<22} {'결과':>12} {'기준':>12} {'판정':>6}")
    print("-" * 60)

    def _fmt_pass(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    rows = [
        (
            "1. 페르소나 일관성",
            f"{persona['score_pct']:.1f}%",
            f"≥ {THRESHOLD_PERSONA:.0f}%",
            persona["pass"],
        ),
        (
            "2. 한국어 오류율",
            f"{grammar['error_rate_pct']:.1f}%",
            f"≤ {THRESHOLD_GRAMMAR_ERROR:.0f}%",
            grammar["pass"],
        ),
        (
            "3. 후속 질문 적절성",
            f"{question['avg_score']:.2f}/5",
            f"≥ {THRESHOLD_QUESTION}/5",
            question["pass"],
        ),
        (
            "4. 응답 속도",
            f"{speed['avg_tps']:.1f} tok/s",
            f"≥ {THRESHOLD_SPEED:.0f} tok/s",
            speed["pass"],
        ),
    ]

    for name, result_val, threshold_val, passed in rows:
        mark = "PASS" if passed else "FAIL"
        print(f"  {name:<22} {result_val:>12} {threshold_val:>12} {mark:>6}")

    overall_pass = all(r[3] for r in rows)
    print(sep)
    print(f"  최종 판정: {'*** PASS ***' if overall_pass else '!!! FAIL !!!'}")
    print(sep)

    if not overall_pass:
        print("\n  FAIL 항목 대응 방안:")
        if not persona["pass"]:
            print("  - 페르소나: 몽글이 학습 데이터(페르소나 대화쌍) 보강 후 재파인튜닝")
        if not grammar["pass"]:
            print("  - 문법: 반복 패널티(repeat_penalty) 높이기, 학습 데이터 품질 검수")
        if not question["pass"]:
            print("  - 질문: 후속 질문 패턴 학습 데이터(200~300쌍) 보강")
        if not speed["pass"]:
            print("  - 속도: Q4_K_M 양자화 적용 여부 확인, num_ctx=2048로 축소 시도")

    return overall_pass


# ============================================================
# 메인 실행
# ============================================================

async def main(
    model: str,
    eval_data_path: str,
    output_path: str,
    use_solar: bool,
) -> int:
    """
    평가 전체 파이프라인 실행.

    Args:
        model: 평가할 Ollama 모델명
        eval_data_path: 평가 데이터 JSONL 경로
        output_path: 결과 JSON 저장 경로
        use_solar: Solar API 채점 사용 여부

    Returns:
        0: 전체 PASS, 1: FAIL 또는 오류
    """
    print("=" * 60)
    print(f"  몽글이 품질 평가 시작")
    print(f"  모델: {model}")
    print(f"  Ollama: {OLLAMA_BASE_URL}")
    print(f"  Solar 채점: {'사용' if use_solar else '미사용 (규칙 기반만)'}")
    print("=" * 60)

    # Solar API 키 확인
    api_key = settings.UPSTAGE_API_KEY
    if use_solar and not api_key:
        print("[경고] UPSTAGE_API_KEY 미설정 — Solar 채점 없이 규칙 기반으로만 평가합니다.")
        use_solar = False

    # 평가 데이터 로드
    eval_data: list[dict[str, Any]] = []
    eval_path = Path(eval_data_path)
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        eval_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"\n  평가 데이터: {len(eval_data)}건 로드 ({eval_path})")
    else:
        # 평가 데이터 파일이 없으면 내장 기본 샘플 10개 사용
        print(f"\n  [경고] 평가 데이터 파일 없음 ({eval_data_path})")
        print("  → 내장 기본 샘플 10개로 평가합니다.")
        eval_data = [
            {"input": "안녕하세요! 오늘 기분 좋은데 영화 추천해줘", "output": ""},
            {"input": "우울한데 힐링 영화 볼래요", "output": ""},
            {"input": "신나는 액션 영화 추천해줘", "output": ""},
            {"input": "가족이랑 볼 영화 뭐 있어?", "output": ""},
            {"input": "공포 영화 좋아하는데 추천해줘", "output": ""},
            {"input": "로맨스 영화 보고 싶어요", "output": ""},
            {"input": "스릴러 영화 추천 부탁해요", "output": ""},
            {"input": "SF 영화 중에 재밌는 거 알려줘", "output": ""},
            {"input": "다큐멘터리 영화 추천해줘", "output": ""},
            {"input": "애니메이션 영화 추천해줄 수 있어요?", "output": ""},
        ]

    # Ollama 연결 확인
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            resp.raise_for_status()
            available_models = [m["name"] for m in resp.json().get("models", [])]
            # 모델명은 태그 없이도 매칭 (예: "mongle:latest" → "mongle")
            model_found = any(model in m for m in available_models)
            if not model_found:
                print(f"\n[오류] Ollama에서 '{model}' 모델을 찾을 수 없습니다.")
                print(f"  등록된 모델: {available_models}")
                print(f"  ollama create {model} -f Modelfile.mongle 실행 후 재시도하세요.")
                return 1
            print(f"\n  Ollama 연결 확인 — '{model}' 모델 등록 확인")
        except Exception as exc:
            print(f"\n[오류] Ollama 서버 연결 실패: {exc}")
            print(f"  Ollama가 {OLLAMA_BASE_URL}에서 실행 중인지 확인하세요.")
            return 1

        # ── 4개 항목 순차 평가 ──
        # (동시 실행 시 Ollama GPU 경합 → 속도 측정 오염 방지)

        persona_result = await evaluate_persona(
            client, model, eval_data, api_key, use_solar
        )
        grammar_result = await evaluate_grammar(
            client, model, eval_data, api_key, use_solar
        )
        question_result = await evaluate_question_quality(
            client, model, api_key, use_solar
        )
        speed_result = await evaluate_speed(client, model)

    # 결과 테이블 출력
    overall_pass = print_results_table(
        persona_result, grammar_result, question_result, speed_result
    )

    # JSON 결과 저장
    output = {
        "model": model,
        "eval_data_path": str(eval_data_path),
        "use_solar": use_solar,
        "overall_pass": overall_pass,
        "results": {
            "persona_consistency": persona_result,
            "korean_grammar": grammar_result,
            "question_quality": question_result,
            "response_speed": speed_result,
        },
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {output_file}")

    return 0 if overall_pass else 1


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="몽글이 품질 자동 평가 스크립트 (M-LLM-6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 실행 (Solar 채점 포함)
  PYTHONPATH=src uv run python scripts/evaluate_mongle.py \\
    --model mongle \\
    --eval_data data/finetune/mongle_eval.jsonl \\
    --output data/finetune/eval_results.json

  # Solar API 없이 빠른 규칙 기반만 실행
  PYTHONPATH=src uv run python scripts/evaluate_mongle.py \\
    --model mongle \\
    --eval_data data/finetune/mongle_eval.jsonl \\
    --output data/finetune/eval_results.json \\
    --no-solar
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mongle",
        help="평가할 Ollama 모델명 (기본: mongle)",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="data/finetune/mongle_eval.jsonl",
        help="평가 데이터 JSONL 경로 (기본: data/finetune/mongle_eval.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/finetune/eval_results.json",
        help="결과 JSON 저장 경로 (기본: data/finetune/eval_results.json)",
    )
    parser.add_argument(
        "--no-solar",
        action="store_true",
        help="Solar API 채점 비활성화 (규칙 기반만 사용, API 키 불필요)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(
        main(
            model=args.model,
            eval_data_path=args.eval_data,
            output_path=args.output,
            use_solar=not args.no_solar,
        )
    )
    sys.exit(exit_code)
