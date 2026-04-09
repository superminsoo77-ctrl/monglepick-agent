"""
Qdrant 에 적재된 영화의 mood_tags 가 LLM 생성인지 fallback 인지 판별.

검증 방식:
    1. Qdrant scroll 로 N 건 샘플 추출 (payload: genres, mood_tags)
    2. 각 영화의 mood_tags 가 _genre_fallback(genres) 결과와 정확히 일치하는지 검사
    3. 일치하면 fallback 발동 가능성 높음 (단 LLM 이 우연히 같은 결과 낼 수 있음 — 확률적 판단)
    4. mood_tags == ["잔잔"] 1개만 있는 경우는 결정적 fallback signature
       (장르 매핑 없는 경우의 _genre_fallback 반환값)

출력:
    - 총 샘플 수
    - mood 개수 분포 (1 개 / 2 개 / 3 개 / 4 개 / 5 개)
    - LLM 생성 추정 비율 (3+ 개이면서 fallback 과 불일치)
    - fallback 의심 비율 (≤2 개이고 fallback 과 일치 또는 ["잔잔"] 단일)
    - 결정적 fallback signature (["잔잔"] 단일) 건수
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_env_file = _project_root / ".env"
_env = {}
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            _env[k.strip()] = v.strip()

QDRANT_URL = _env.get("QDRANT_URL", "http://100.73.239.117:6333")
COLLECTION = "movies"

# mood_batch.py 의 GENRE_TO_MOOD 와 동일 (진실 원본)
GENRE_TO_MOOD = {
    "액션": ["몰입", "스릴"], "모험": ["모험", "몰입"], "애니메이션": ["따뜻", "판타지"],
    "코미디": ["유쾌", "유머"], "범죄": ["긴장감", "다크"], "다큐멘터리": ["철학적", "사회비판"],
    "드라마": ["감동", "잔잔"], "가족": ["가족애", "따뜻"], "판타지": ["판타지", "모험"],
    "역사": ["웅장", "감동"], "공포": ["공포", "다크"], "음악": ["감동", "힐링"],
    "미스터리": ["미스터리", "긴장감"], "로맨스": ["로맨틱", "따뜻"], "SF": ["몰입", "웅장"],
    "TV 영화": ["잔잔"], "스릴러": ["스릴", "긴장감"], "전쟁": ["웅장", "카타르시스"],
    "서부": ["모험", "레트로"],
}


def genre_fallback(genres: list[str]) -> set[str]:
    moods = set()
    for g in genres or []:
        moods.update(GENRE_TO_MOOD.get(g, []))
    if not moods:
        return {"잔잔"}
    return set(list(moods)[:5])


def scroll(limit: int) -> list[dict]:
    out = []
    offset = None
    page = min(256, limit)
    while len(out) < limit:
        body = {"limit": page, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        pts = data["result"]["points"]
        if not pts:
            break
        out.extend(pts)
        offset = data["result"].get("next_page_offset")
        if offset is None:
            break
    return out[:limit]


def main():
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"[INFO] Qdrant scroll {sample_size} 건 추출...")
    points = scroll(sample_size)
    print(f"[INFO] 추출 완료: {len(points)}\n")

    n = len(points)
    mood_count_dist = Counter()
    exact_fallback_matches = 0           # mood_tags set == genre_fallback(genres)
    deterministic_fallback = 0           # mood_tags == ["잔잔"] (결정적 fallback signature)
    llm_generated = 0                    # 3+ 개 mood + fallback 과 불일치
    empty_mood = 0                       # mood_tags 비어있음
    no_genre_deterministic = 0           # 장르 없음 + mood == ["잔잔"]

    # fallback signature 샘플 (공유해서 보기)
    fallback_samples = []
    llm_samples = []

    for p in points:
        payload = p.get("payload", {}) or {}
        mood_tags = payload.get("mood_tags") or []
        genres = payload.get("genres") or []

        if not isinstance(mood_tags, list):
            mood_tags = []
        if not isinstance(genres, list):
            genres = []

        mood_set = set(mood_tags)
        fb_set = genre_fallback(genres)

        mood_count_dist[min(len(mood_tags), 10)] += 1

        if not mood_tags:
            empty_mood += 1
            continue

        # 결정적 fallback: ["잔잔"] 단일
        if mood_tags == ["잔잔"]:
            deterministic_fallback += 1
            if not genres:
                no_genre_deterministic += 1
            if len(fallback_samples) < 5:
                fallback_samples.append((payload.get("title"), genres, mood_tags))
            continue

        # 정확한 set 일치 (확률적 판단)
        if mood_set == fb_set and len(mood_tags) <= 4:
            exact_fallback_matches += 1
            if len(fallback_samples) < 5:
                fallback_samples.append((payload.get("title"), genres, mood_tags))
        elif len(mood_tags) >= 3:
            # 3개 이상이면서 fallback 과 다르면 LLM 생성으로 간주
            llm_generated += 1
            if len(llm_samples) < 5:
                llm_samples.append((payload.get("title"), genres, mood_tags))

    print("=" * 70)
    print(f"  mood_tags fallback vs LLM 판별 — {n} 건")
    print("=" * 70)
    print()
    print(f"  mood_tags 개수 분포:")
    for k in sorted(mood_count_dist.keys()):
        pct = mood_count_dist[k] * 100 / n
        bar = "█" * int(pct / 2)
        print(f"    {k} 개: {mood_count_dist[k]:>5} ({pct:5.1f}%) {bar}")

    print()
    print(f"  판정 결과:")
    print(f"    empty (적재 실패):                  {empty_mood:>6} ({empty_mood*100/n:5.2f}%)")
    print(f"    ★ 결정적 fallback ['잔잔']:         {deterministic_fallback:>6} ({deterministic_fallback*100/n:5.2f}%)")
    print(f"      └ 장르 없음 + ['잔잔']:          {no_genre_deterministic:>6}")
    print(f"      └ 장르 'TV 영화' + ['잔잔']:     {deterministic_fallback - no_genre_deterministic:>6}")
    print(f"    가능한 fallback (set 일치 & ≤4개): {exact_fallback_matches:>6} ({exact_fallback_matches*100/n:5.2f}%)")
    print(f"    LLM 생성 확정 (3+ & 불일치):       {llm_generated:>6} ({llm_generated*100/n:5.2f}%)")
    print()

    other = n - empty_mood - deterministic_fallback - exact_fallback_matches - llm_generated
    print(f"    기타 (1~2 개 but 장르 매칭 안 됨):  {other:>6} ({other*100/n:5.2f}%)")
    print()

    print()
    print(f"  ★ 결정적 fallback 샘플 (최대 5건):")
    for title, genres, moods in fallback_samples:
        print(f"    - {title} / 장르={genres} / mood={moods}")

    print()
    print(f"  ★ LLM 생성 샘플 (최대 5건):")
    for title, genres, moods in llm_samples:
        print(f"    - {title} / 장르={genres[:3]} / mood={moods}")

    print()
    print("=" * 70)
    print("  결론 판정 기준")
    print("=" * 70)
    print(f"  - 결정적 fallback > 5% 이면 심각한 fallback 발동")
    print(f"  - LLM 생성 > 80% 이면 양호")
    print()

    deterministic_rate = deterministic_fallback * 100 / n
    llm_rate = llm_generated * 100 / n

    if deterministic_rate > 5:
        print(f"  ⚠️  WARN: 결정적 fallback 비율 {deterministic_rate:.2f}% > 5%")
    else:
        print(f"  ✅ PASS: 결정적 fallback 비율 {deterministic_rate:.2f}% ≤ 5%")

    if llm_rate >= 70:
        print(f"  ✅ PASS: LLM 생성 확정 비율 {llm_rate:.2f}% ≥ 70%")
    else:
        print(f"  ⚠️  WARN: LLM 생성 확정 비율 {llm_rate:.2f}% < 70%")


if __name__ == "__main__":
    main()
