"""
Upstage 크레딧 만료 / 401 모니터링 + Gmail SMTP 알림 워커.

Phase 2 / 4-B / 7 장시간 실행 중 Upstage API 크레딧 소진 또는 오류를
실시간 감지하고, 로그 기록 + Gmail SMTP 이메일 발송으로 즉시 알림.

감지 패턴:
    - mood_batch_all_attempts_failed   (5회 재시도 실패 → fallback 발동)
    - mood_api_error 401                (Unauthorized)
    - quota_exceeded / insufficient_quota
    - credit_exhausted / invalid_api_key

이메일 알림 조건:
    - 첫 감지 시 즉시 1회 (cooldown 시작)
    - cooldown (30분) 이후 재감지 시 추가 발송
    - 패턴별 개별 카운트 누적

환경 변수 (.env 추가 필요):
    EMAIL_SMTP_HOST=smtp.gmail.com
    EMAIL_SMTP_PORT=587
    EMAIL_SMTP_USER=<발송자_gmail>
    EMAIL_SMTP_APP_PASSWORD=<Gmail_앱_비밀번호>
        (https://myaccount.google.com/apppasswords 에서 발급)
    EMAIL_ALERT_TO=ujk6073@gmail.com
    EMAIL_ALERT_FROM=<발송자_gmail>   # 보통 USER 와 동일

SMTP 설정 없으면: 로그 기록만 (기존 동작 유지, 실패 경고 없음)

사용법:
    # 전체 로그 감시 (기본)
    PYTHONPATH=src uv run python scripts/monitor_upstage_quota.py \\
        --logs logs/full_reload.log,logs/phase2_*.log,logs/phase7_*.log &

    # 단일 파일
    PYTHONPATH=src uv run python scripts/monitor_upstage_quota.py \\
        --log logs/phase7_tagline.log

    # 첫 감지 후 종료
    PYTHONPATH=src uv run python scripts/monitor_upstage_quota.py --once

    # 테스트 이메일 발송
    PYTHONPATH=src uv run python scripts/monitor_upstage_quota.py --test-email
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import smtplib
import socket
import ssl
import sys
import threading
import time
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 감지 패턴 (대소문자 무시)
# ──────────────────────────────────────────────────────────────
PATTERNS = [
    re.compile(r"mood_batch_all_attempts_failed", re.IGNORECASE),
    re.compile(r"mood_api_error.*401", re.IGNORECASE),
    re.compile(r"\b401\b.*unauthorized", re.IGNORECASE),
    re.compile(r"unauthorized.*\b401\b", re.IGNORECASE),
    re.compile(r"quota_exceeded", re.IGNORECASE),
    re.compile(r"insufficient_quota", re.IGNORECASE),
    re.compile(r"credit_exhausted", re.IGNORECASE),
    re.compile(r"invalid_api_key", re.IGNORECASE),
    re.compile(r"incorrect api key", re.IGNORECASE),
    re.compile(r"upstage_api_error", re.IGNORECASE),
    re.compile(r"person_llm_error", re.IGNORECASE),  # Phase 4-B Person LLM 실패
    re.compile(r"movie_llm_error", re.IGNORECASE),   # Phase 7 Movie LLM 실패
]

DEFAULT_LOG = "logs/full_reload.log"
ALERT_LOG = Path("logs/upstage_quota_alert.log")
EMAIL_COOLDOWN_MINUTES = 30


# ──────────────────────────────────────────────────────────────
# 환경 변수 로드 (.env)
# ──────────────────────────────────────────────────────────────


def _load_env() -> dict[str, str]:
    """프로젝트 루트 .env 를 읽어 dict 로 반환."""
    env: dict[str, str] = {}
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


_ENV = _load_env()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key) or _ENV.get(key) or default


# ──────────────────────────────────────────────────────────────
# SMTP Gmail 이메일 전송
# ──────────────────────────────────────────────────────────────


class EmailAlertManager:
    """SMTP 기반 Gmail 알림 매니저 + 쿨다운 + 누적 카운터."""

    def __init__(self) -> None:
        self.smtp_host = _env("EMAIL_SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(_env("EMAIL_SMTP_PORT", "587"))
        self.smtp_user = _env("EMAIL_SMTP_USER", "")
        self.smtp_password = _env("EMAIL_SMTP_APP_PASSWORD", "")
        self.alert_to = _env("EMAIL_ALERT_TO", "ujk6073@gmail.com")
        self.alert_from = _env("EMAIL_ALERT_FROM", self.smtp_user)

        self.last_sent_at: datetime | None = None
        self.cooldown = timedelta(minutes=EMAIL_COOLDOWN_MINUTES)
        self.total_sent = 0
        self.pattern_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def enabled(self) -> bool:
        return bool(self.smtp_user and self.smtp_password and self.alert_to)

    def disabled_reason(self) -> str:
        missing = []
        if not self.smtp_user:
            missing.append("EMAIL_SMTP_USER")
        if not self.smtp_password:
            missing.append("EMAIL_SMTP_APP_PASSWORD")
        if not self.alert_to:
            missing.append("EMAIL_ALERT_TO")
        return f"누락 변수: {', '.join(missing)}" if missing else ""

    def should_send(self) -> bool:
        """쿨다운 기반 발송 여부 판단."""
        if self.last_sent_at is None:
            return True
        return datetime.now() - self.last_sent_at >= self.cooldown

    def count(self, pattern_name: str) -> None:
        with self._lock:
            self.pattern_counts[pattern_name] = (
                self.pattern_counts.get(pattern_name, 0) + 1
            )

    def send(
        self,
        subject: str,
        body: str,
        force: bool = False,
    ) -> bool:
        """이메일 발송. 쿨다운 중이면 skip (force=True 제외)."""
        if not self.enabled():
            return False

        with self._lock:
            if not force and not self.should_send():
                return False

            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = self.alert_from
            msg["To"] = self.alert_to
            msg.set_content(body)

            try:
                context = ssl.create_default_context()
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as smtp:
                    smtp.ehlo()
                    smtp.starttls(context=context)
                    smtp.ehlo()
                    smtp.login(self.smtp_user, self.smtp_password)
                    smtp.send_message(msg)

                self.last_sent_at = datetime.now()
                self.total_sent += 1
                return True

            except (smtplib.SMTPException, socket.error, OSError) as e:
                ts = datetime.now().isoformat()
                err_msg = f"[{ts}] [EMAIL_FAIL] {type(e).__name__}: {str(e)[:200]}"
                print(err_msg, flush=True)
                ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
                with ALERT_LOG.open("a", encoding="utf-8") as f:
                    f.write(err_msg + "\n")
                return False


# ──────────────────────────────────────────────────────────────
# Alert 로그 기록
# ──────────────────────────────────────────────────────────────


def _alert(line: str, log_path: Path, pattern_name: str) -> None:
    """감지한 라인을 stdout + alert 로그에 기록."""
    ts = datetime.now().isoformat()
    msg = f"[{ts}] [QUOTA_ALERT] [{log_path.name}] [{pattern_name}] {line.rstrip()}"
    print(msg, flush=True)

    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ALERT_LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _print_swap_instructions() -> None:
    """swap 절차 안내."""
    print()
    print("=" * 70)
    print("  ⚠️  Upstage API 크레딧 소진/만료 감지!")
    print("=" * 70)
    print()
    print("  계속 진행 시 LLM 보강 (mood/category/tagline 등) 이 fallback 으로")
    print("  떨어져 데이터 품질이 저하됩니다. 즉시 대응 필요.")
    print()
    print("  안전 swap 절차:")
    print()
    print("  1. 현재 장시간 작업 중인 프로세스 graceful 중단:")
    print("       ps aux | grep -E 'run_kobis_load|run_kmdb_load|run_kaggle_supplement|run_persons|run_movie_llm'")
    print("       kill -SIGINT <PID>")
    print()
    print("  2. .env 에 새 키가 있으면 swap:")
    print("       bash scripts/swap_upstage_keys.sh")
    print()
    print("  3. 해당 작업 --resume 재시작 (각 스크립트 로그 참조)")
    print()
    print("  알림 로그: logs/upstage_quota_alert.log")
    print("=" * 70)
    print()


# ──────────────────────────────────────────────────────────────
# 로그 파일 tail (다중 파일 + 회전 대응)
# ──────────────────────────────────────────────────────────────


def _expand_log_patterns(patterns_str: str) -> list[Path]:
    """콤마 구분 패턴을 Path 리스트로 확장 (glob 지원)."""
    out: list[Path] = []
    seen: set[str] = set()
    for raw in patterns_str.split(","):
        raw = raw.strip()
        if not raw:
            continue
        matched = sorted(glob.glob(raw))
        if matched:
            for m in matched:
                if m not in seen:
                    seen.add(m)
                    out.append(Path(m))
        else:
            p = Path(raw)
            if str(p) not in seen:
                seen.add(str(p))
                out.append(p)
    return out


def tail_follow_single(path: Path, stop_event: threading.Event, from_end: bool = True):
    """단일 파일 tail -f (파일 미존재 시 생성 대기)."""
    # 파일 생성 대기
    while not path.exists() and not stop_event.is_set():
        time.sleep(2)
    if stop_event.is_set():
        return

    with path.open("r", encoding="utf-8", errors="replace") as f:
        if from_end:
            f.seek(0, os.SEEK_END)

        while not stop_event.is_set():
            line = f.readline()
            if not line:
                # 파일 회전 감지
                try:
                    if path.stat().st_size < f.tell():
                        f.close()
                        time.sleep(0.5)
                        # 재귀 대신 동일 함수 재호출 (from_end=False 로 처음부터)
                        yield from tail_follow_single(path, stop_event, from_end=False)
                        return
                except FileNotFoundError:
                    time.sleep(2)
                    continue
                time.sleep(0.5)
                continue
            yield line


def _watch_file(
    path: Path,
    stop_event: threading.Event,
    email_mgr: EmailAlertManager,
    once: bool,
    from_start: bool,
    counters: dict,
    counters_lock: threading.Lock,
) -> None:
    """단일 로그 파일 감시 워커 (스레드)."""
    print(f"[WATCH] {path}", flush=True)
    try:
        for line in tail_follow_single(path, stop_event, from_end=not from_start):
            for pattern in PATTERNS:
                m = pattern.search(line)
                if m:
                    pattern_name = pattern.pattern[:40]
                    _alert(line, path, pattern_name)

                    with counters_lock:
                        counters["total"] = counters.get("total", 0) + 1
                        counters.setdefault("first_at", datetime.now())
                        counters.setdefault("patterns", {})
                        counters["patterns"][pattern_name] = (
                            counters["patterns"].get(pattern_name, 0) + 1
                        )

                    email_mgr.count(pattern_name)

                    # 첫 감지 시 + 쿨다운 이후 발송
                    if email_mgr.should_send():
                        subject = f"[몽글픽] Upstage API 알림 — {pattern_name}"
                        body = f"""Upstage API 오류가 감지되어 알림드립니다.

감지 시각: {datetime.now().isoformat()}
감지 패턴: {pattern_name}
로그 파일: {path}

감지된 라인:
  {line.rstrip()[:500]}

누적 감지 현황:
  총 감지: {counters.get('total', 0)} 건
  첫 감지: {counters.get('first_at', '')}
  패턴별 카운트: {counters.get('patterns', {})}

대응 절차:
  1. ps aux | grep -E 'run_kobis_load|run_kmdb_load|run_persons|run_movie_llm'
  2. kill -SIGINT <PID>  (graceful 종료)
  3. .env 에 새 UPSTAGE_API_KEY2 추가 후 bash scripts/swap_upstage_keys.sh
  4. 해당 작업 --resume 재시작

이메일 총 발송: {email_mgr.total_sent + 1}

— 몽글픽 AI Agent 모니터 워커
"""
                        if email_mgr.send(subject, body):
                            print(
                                f"[EMAIL] ✉️  알림 발송 완료 → {email_mgr.alert_to} (총 {email_mgr.total_sent}회)",
                                flush=True,
                            )

                    if once:
                        stop_event.set()
                        return

                    break
    except Exception as e:
        print(f"[ERROR] watch {path} — {type(e).__name__}: {e}", flush=True)


# ──────────────────────────────────────────────────────────────
# 테스트 이메일
# ──────────────────────────────────────────────────────────────


def _test_email(email_mgr: EmailAlertManager) -> int:
    """테스트 이메일 전송 + 결과 출력."""
    print("=" * 60)
    print("  SMTP 테스트 이메일 발송")
    print("=" * 60)
    print(f"  SMTP Host: {email_mgr.smtp_host}:{email_mgr.smtp_port}")
    print(f"  From: {email_mgr.alert_from}")
    print(f"  To: {email_mgr.alert_to}")
    print(f"  Enabled: {email_mgr.enabled()}")

    if not email_mgr.enabled():
        print(f"\n  ❌ SMTP 설정 미완료 — {email_mgr.disabled_reason()}")
        print()
        print("  .env 에 다음 필드를 추가하세요:")
        print("    EMAIL_SMTP_HOST=smtp.gmail.com")
        print("    EMAIL_SMTP_PORT=587")
        print("    EMAIL_SMTP_USER=<발송자_gmail>")
        print("    EMAIL_SMTP_APP_PASSWORD=<Gmail_앱_비밀번호>")
        print("    EMAIL_ALERT_TO=ujk6073@gmail.com")
        print("    EMAIL_ALERT_FROM=<발송자_gmail>")
        print()
        print("  Gmail 앱 비밀번호 발급:")
        print("    https://myaccount.google.com/apppasswords")
        print("    (2단계 인증 활성화 필수)")
        return 1

    subject = "[몽글픽 테스트] Upstage 모니터 워커 SMTP 점검"
    body = f"""테스트 이메일입니다.

발송 시각: {datetime.now().isoformat()}
호스트: {socket.gethostname()}
SMTP: {email_mgr.smtp_host}:{email_mgr.smtp_port}

이 이메일을 받으셨다면 Upstage API 알림 파이프라인이 정상 동작합니다.
실제 알림은 Phase 2 / 4-B / 7 실행 중 크레딧 소진 또는 오류 발생 시
자동으로 발송됩니다.

— 몽글픽 AI Agent 모니터 워커
"""
    ok = email_mgr.send(subject, body, force=True)
    if ok:
        print(f"\n  ✅ 테스트 이메일 발송 성공 → {email_mgr.alert_to}")
        return 0
    else:
        print("\n  ❌ 테스트 이메일 발송 실패 — 위 EMAIL_FAIL 로그 확인")
        return 1


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upstage 401/quota 만료 모니터링 + Gmail SMTP 알림",
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help=f"단일 감시 로그 파일 (기본: {DEFAULT_LOG})",
    )
    parser.add_argument(
        "--logs", type=str, default=None,
        help="다중 감시 파일 (콤마 구분, glob 지원). 예: logs/phase2_*.log,logs/phase7_*.log",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="첫 감지 후 종료",
    )
    parser.add_argument(
        "--from-start", action="store_true",
        help="로그 처음부터 스캔",
    )
    parser.add_argument(
        "--test-email", action="store_true",
        help="SMTP 테스트 이메일만 발송 후 종료",
    )
    args = parser.parse_args()

    # 이메일 매니저 초기화
    email_mgr = EmailAlertManager()

    # 테스트 이메일 모드
    if args.test_email:
        return _test_email(email_mgr)

    # 감시 파일 목록 결정
    if args.logs:
        log_paths = _expand_log_patterns(args.logs)
    elif args.log:
        log_paths = [Path(args.log)]
    else:
        log_paths = [Path(DEFAULT_LOG)]

    if not log_paths:
        print("[ERROR] 감시할 로그 파일이 없습니다.")
        return 1

    print("=" * 60)
    print("  Upstage Quota 모니터 시작")
    print("=" * 60)
    print(f"  감시 파일 ({len(log_paths)}):")
    for p in log_paths:
        print(f"    - {p}")
    print(f"  감지 패턴: {len(PATTERNS)}")
    print(f"  알림 로그: {ALERT_LOG}")
    print(f"  이메일 알림: {'✅ 활성화' if email_mgr.enabled() else '❌ 비활성 — ' + email_mgr.disabled_reason()}")
    if email_mgr.enabled():
        print(f"    To: {email_mgr.alert_to}")
        print(f"    쿨다운: {EMAIL_COOLDOWN_MINUTES}분")
    print(f"  모드: {'첫 감지 후 종료' if args.once else '계속 감시'}")
    print("=" * 60)
    print()

    # 다중 스레드 워커 시작
    stop_event = threading.Event()
    counters: dict = {}
    counters_lock = threading.Lock()
    threads: list[threading.Thread] = []

    for path in log_paths:
        t = threading.Thread(
            target=_watch_file,
            args=(path, stop_event, email_mgr, args.once, args.from_start, counters, counters_lock),
            daemon=True,
            name=f"watch-{path.name}",
        )
        t.start()
        threads.append(t)

    try:
        # 메인 스레드는 stop_event 대기
        while not stop_event.is_set():
            time.sleep(1)
            # 모든 워커가 종료됐는지 체크
            alive = any(t.is_alive() for t in threads)
            if not alive:
                break

    except KeyboardInterrupt:
        print()
        print("[INFO] 모니터링 종료 요청")
        stop_event.set()

    for t in threads:
        t.join(timeout=5)

    print()
    print("=" * 60)
    print("  모니터링 종료 요약")
    print("=" * 60)
    print(f"  총 감지: {counters.get('total', 0)} 건")
    if counters.get("first_at"):
        print(f"  첫 감지: {counters['first_at']}")
    if counters.get("patterns"):
        print(f"  패턴별:")
        for name, cnt in sorted(counters["patterns"].items(), key=lambda x: -x[1]):
            print(f"    {name}: {cnt}")
    print(f"  이메일 발송: {email_mgr.total_sent} 회")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
