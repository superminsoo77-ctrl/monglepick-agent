"""
Phase ML-1/2 데이터 품질 샘플 검증 스크립트.

Task #5 (run_full_reload.py) 가 적재 중인 Qdrant `movies` 컬렉션에서
무작위 샘플을 추출하여 다음 ML-1/2/4 적용률을 측정한다.

검증 항목 (Read-only, Task #5 무영향):

    [ML-1] 한영 이중 cast/director (Phase A-2/A-3)
        - director_original_name 존재율
        - cast_members 한글 + cast_original_names 영문 모두 존재율

    [ML-2] keywords 한국어 매핑 (Phase A-1/A-3)
        - keywords 배열 중 한국어 비율 (예: "우주", "복수")
        - 영문 누락률 (영문 키워드만 있는 비율)

    [ML-4] mood_tags 분포 (Solar Pro 3 정밀 무드)
        - mood_tags 배열 길이 분포 (0/1/2/3+)
        - 가장 자주 등장하는 태그 Top 10

    [기본] 다국어 필드 채움률
        - title_en, overview_en, alternative_titles 채움률

설계 진실 원본:
    docs/Phase_ML4_재적재_진행상황_세션인계.md §1.2 (Phase 1 검증)
    docs/다국어_영화검색_분석보고서.md (Phase ML 전략)

사용법:
    # 기본 — 1000건 샘플 (페이징 limit=1000)
    PYTHONPATH=src uv run python scripts/check_quality_ml1_ml2.py

    # 더 많이
    PYTHONPATH=src uv run python scripts/check_quality_ml1_ml2.py --sample 5000

    # 특정 ID 샘플 (인터스텔라/기생충 등)
    PYTHONPATH=src uv run python scripts/check_quality_ml1_ml2.py --ids 157336,496243
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

# .env 의 QDRANT_URL 직접 로드 (config import 는 무거우므로 회피)
_project_root = Path(__file__).resolve().parent.parent
_env_file = _project_root / ".env"
_env: dict[str, str] = {}
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            _env[k.strip()] = v.strip()

QDRANT_URL = _env.get("QDRANT_URL", "http://100.73.239.117:6333")
COLLECTION = "movies"

# 한글 1글자 정규식
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")


def _has_hangul(text: str) -> bool:
    return bool(_HANGUL_RE.search(text or ""))


def _http_post(path: str, body: dict) -> dict:
    """간단 HTTP POST → JSON."""
    url = f"{QDRANT_URL}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"[ERROR] {url} → {e.code} {e.reason}")
        raise
    except urllib.error.URLError as e:
        print(f"[ERROR] {url} → {e}")
        raise


def _scroll_sample(limit: int) -> list[dict]:
    """Qdrant scroll API 로 limit 건 페이로드 추출 (벡터 제외)."""
    points: list[dict] = []
    offset = None
    page_size = min(limit, 256)

    while len(points) < limit:
        body: dict = {
            "limit": page_size,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset

        data = _http_post(f"/collections/{COLLECTION}/points/scroll", body)
        result = data.get("result", {})
        chunk = result.get("points", [])
        if not chunk:
            break

        for p in chunk:
            payload = p.get("payload", {}) or {}
            payload["_id"] = p.get("id")
            points.append(payload)

        offset = result.get("next_page_offset")
        if offset is None:
            break

    return points[:limit]


def _fetch_by_ids(ids: list[str]) -> list[dict]:
    """특정 영화 ID 리스트 페이로드 조회."""
    body = {
        "ids": ids,
        "with_payload": True,
        "with_vector": False,
    }
    data = _http_post(f"/collections/{COLLECTION}/points", body)
    points = data.get("result", []) or []
    out: list[dict] = []
    for p in points:
        payload = p.get("payload", {}) or {}
        payload["_id"] = p.get("id")
        out.append(payload)
    return out


# ══════════════════════════════════════════════════════════════
# 품질 검증
# ══════════════════════════════════════════════════════════════


def analyze(samples: list[dict]) -> dict:
    n = len(samples)
    if n == 0:
        return {"error": "no samples"}

    # 카운터
    has_director_original = 0
    has_director_ko = 0
    has_cast = 0
    has_cast_original_names = 0
    cast_count_dist = Counter()  # 0~10+
    director_count_dist = Counter()

    has_title_en = 0
    has_overview_en = 0
    has_alternative_titles = 0

    keywords_total_arrays = 0
    keywords_total_terms = 0
    keywords_korean_terms = 0
    keywords_korean_movies = 0
    keywords_only_english = 0

    mood_count_dist = Counter()
    mood_tag_freq = Counter()

    has_release_date = 0
    has_genres = 0
    has_poster_path = 0

    for movie in samples:
        # 감독
        director_orig = movie.get("director_original_name") or ""
        director_ko = movie.get("director") or ""
        if isinstance(director_orig, list):
            director_orig = ", ".join(director_orig)
        if isinstance(director_ko, list):
            director_ko = ", ".join(director_ko)

        if director_orig and isinstance(director_orig, str) and director_orig.strip():
            has_director_original += 1
        if director_ko and _has_hangul(director_ko):
            has_director_ko += 1

        director_count_dist[
            min(len(director_ko.split(",")) if isinstance(director_ko, str) else 0, 5)
        ] += 1

        # 캐스트
        cast_members = movie.get("cast_members") or movie.get("cast") or []
        cast_original_names = movie.get("cast_original_names") or []
        if isinstance(cast_members, str):
            cast_members = [cast_members]
        if isinstance(cast_original_names, str):
            cast_original_names = [cast_original_names]

        if cast_members:
            has_cast += 1
        if cast_original_names:
            has_cast_original_names += 1

        cast_count_dist[min(len(cast_members), 10)] += 1

        # 다국어 필드
        if movie.get("title_en"):
            has_title_en += 1
        if movie.get("overview_en"):
            has_overview_en += 1
        if movie.get("alternative_titles"):
            has_alternative_titles += 1

        # 키워드 한국어 매핑
        keywords = movie.get("keywords") or []
        if isinstance(keywords, str):
            try:
                keywords = json.loads(keywords)
            except Exception:
                keywords = [keywords]
        if not isinstance(keywords, list):
            keywords = []

        if keywords:
            keywords_total_arrays += 1
            keywords_total_terms += len(keywords)

            korean_in_movie = 0
            for kw in keywords:
                if isinstance(kw, str) and _has_hangul(kw):
                    korean_in_movie += 1
                    keywords_korean_terms += 1

            if korean_in_movie > 0:
                keywords_korean_movies += 1
            else:
                keywords_only_english += 1

        # mood_tags
        mood_tags = movie.get("mood_tags") or []
        if isinstance(mood_tags, str):
            try:
                mood_tags = json.loads(mood_tags)
            except Exception:
                mood_tags = [mood_tags]
        if not isinstance(mood_tags, list):
            mood_tags = []

        mood_count_dist[min(len(mood_tags), 10)] += 1
        for tag in mood_tags:
            if isinstance(tag, str):
                mood_tag_freq[tag] += 1

        # 기본 필드 — Qdrant 페이로드는 release_year(int) 만 보유. release_date 필드 없음.
        if movie.get("release_year") or movie.get("release_date"):
            has_release_date += 1
        genres = movie.get("genres") or []
        if isinstance(genres, list) and genres:
            has_genres += 1
        if movie.get("poster_path"):
            has_poster_path += 1

    pct = lambda x: f"{x * 100 / n:5.1f}%"
    avg_keywords = keywords_total_terms / keywords_total_arrays if keywords_total_arrays else 0
    pct_korean = (
        keywords_korean_terms * 100 / keywords_total_terms if keywords_total_terms else 0
    )

    return {
        "total_samples": n,
        # ML-1
        "ML-1: director_original_name 존재율": pct(has_director_original),
        "ML-1: director(한글) 존재율": pct(has_director_ko),
        "ML-1: cast 존재율": pct(has_cast),
        "ML-1: cast_original_names 존재율": pct(has_cast_original_names),
        # ML-2
        "ML-2: keywords 배열 존재율": pct(keywords_total_arrays),
        "ML-2: 평균 keywords 개수": f"{avg_keywords:5.1f}",
        "ML-2: keywords 한국어 매핑 영화 비율": pct(keywords_korean_movies),
        "ML-2: keywords 영문만 (매핑 누락) 비율": pct(keywords_only_english),
        "ML-2: 전체 keywords term 한국어 비율": f"{pct_korean:5.1f}%",
        # ML-4
        "ML-4: mood_tags 분포": dict(sorted(mood_count_dist.items())),
        "ML-4: mood_tags Top 15": mood_tag_freq.most_common(15),
        # 다국어
        "다국어: title_en 존재율": pct(has_title_en),
        "다국어: overview_en 존재율": pct(has_overview_en),
        "다국어: alternative_titles 존재율": pct(has_alternative_titles),
        # 기본
        "기본: release_date 존재율": pct(has_release_date),
        "기본: genres 존재율": pct(has_genres),
        "기본: poster_path 존재율": pct(has_poster_path),
        # 분포
        "분포: cast_count": dict(sorted(cast_count_dist.items())),
        "분포: director_count": dict(sorted(director_count_dist.items())),
    }


def print_report(report: dict, samples: list[dict]) -> None:
    print()
    print("=" * 70)
    print(f"  Phase ML-1/2/4 데이터 품질 샘플 검증 — {report['total_samples']} 건")
    print("=" * 70)

    sections = [
        ("【 ML-1 한영 이중 cast/director (Phase A-2/A-3) 】", [
            "ML-1: director_original_name 존재율",
            "ML-1: director(한글) 존재율",
            "ML-1: cast 존재율",
            "ML-1: cast_original_names 존재율",
        ]),
        ("【 ML-2 keywords 한국어 매핑 (Phase A-1/A-3) 】", [
            "ML-2: keywords 배열 존재율",
            "ML-2: 평균 keywords 개수",
            "ML-2: keywords 한국어 매핑 영화 비율",
            "ML-2: keywords 영문만 (매핑 누락) 비율",
            "ML-2: 전체 keywords term 한국어 비율",
        ]),
        ("【 ML-4 mood_tags 분포 (Solar Pro 3 정밀) 】", [
            "ML-4: mood_tags 분포",
            "ML-4: mood_tags Top 15",
        ]),
        ("【 다국어 필드 채움률 】", [
            "다국어: title_en 존재율",
            "다국어: overview_en 존재율",
            "다국어: alternative_titles 존재율",
        ]),
        ("【 기본 필드 채움률 】", [
            "기본: release_date 존재율",
            "기본: genres 존재율",
            "기본: poster_path 존재율",
        ]),
        ("【 분포 】", [
            "분포: cast_count",
            "분포: director_count",
        ]),
    ]

    for title, keys in sections:
        print()
        print(title)
        for k in keys:
            v = report.get(k)
            if isinstance(v, list):
                print(f"  {k}:")
                for item in v:
                    print(f"      {item}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"      {kk}: {vv}")
            else:
                print(f"  {k:55s} {v}")

    # 합격 기준
    print()
    print("=" * 70)
    print("  품질 합격 기준 검증")
    print("=" * 70)

    def _ratio_from_pct(s: str) -> float:
        try:
            return float(s.replace("%", "").strip()) / 100
        except Exception:
            return 0.0

    checks = [
        ("ML-1 director_original_name", _ratio_from_pct(report["ML-1: director_original_name 존재율"]), 0.50),
        ("ML-1 director(한글)", _ratio_from_pct(report["ML-1: director(한글) 존재율"]), 0.40),
        ("ML-2 keywords 한국어 매핑", _ratio_from_pct(report["ML-2: keywords 한국어 매핑 영화 비율"]), 0.60),
        ("다국어 title_en", _ratio_from_pct(report["다국어: title_en 존재율"]), 0.70),
        ("다국어 overview_en", _ratio_from_pct(report["다국어: overview_en 존재율"]), 0.50),
        ("기본 release_date", _ratio_from_pct(report["기본: release_date 존재율"]), 0.85),
        ("기본 genres", _ratio_from_pct(report["기본: genres 존재율"]), 0.80),
    ]

    for name, ratio, threshold in checks:
        status = "✅ PASS" if ratio >= threshold else "⚠️  WARN"
        print(f"  {status}  {name:30s}  실측 {ratio*100:5.1f}%  기준 ≥{threshold*100:.0f}%")

    # ML-4: mood_tags 0건 비율
    mood_dist = report.get("ML-4: mood_tags 분포", {})
    zero_pct = mood_dist.get(0, 0) * 100 / report["total_samples"] if report["total_samples"] else 0
    status = "✅ PASS" if zero_pct < 30 else "⚠️  WARN"
    print(f"  {status}  ML-4 mood_tags 0건 영화         실측 {zero_pct:5.1f}%  기준 <30%")

    # 샘플 영화 5건 요약
    print()
    print("=" * 70)
    print("  샘플 영화 5건 미리보기")
    print("=" * 70)
    for i, m in enumerate(samples[:5], 1):
        title = m.get("title", "?")
        title_en = m.get("title_en", "")
        director = m.get("director", "")
        director_orig = m.get("director_original_name", "")
        keywords = m.get("keywords", [])[:5] if isinstance(m.get("keywords"), list) else []
        mood = m.get("mood_tags", [])[:5] if isinstance(m.get("mood_tags"), list) else []
        print(f"\n  [{i}] {title}  /  {title_en}")
        print(f"      director: {director}  ({director_orig})")
        print(f"      keywords: {keywords}")
        print(f"      mood:     {mood}")

    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1000, help="샘플 건수 (기본 1000)")
    parser.add_argument("--ids", type=str, default=None, help="특정 ID 콤마 구분")
    args = parser.parse_args()

    print(f"[INFO] Qdrant: {QDRANT_URL}/{COLLECTION}")
    print(f"[INFO] Sample: {args.sample}")

    if args.ids:
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
        print(f"[INFO] 특정 ID {len(ids)} 건 조회")
        samples = _fetch_by_ids(ids)
    else:
        print(f"[INFO] scroll {args.sample} 건 추출 중...")
        samples = _scroll_sample(args.sample)

    print(f"[INFO] 추출 완료: {len(samples)} 건\n")
    if not samples:
        print("[ERROR] 샘플 0 건 — Qdrant 연결 또는 컬렉션 확인 필요")
        return 1

    report = analyze(samples)
    print_report(report, samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
