"""
Neo4j Person 노드 속성 보강 스크립트 (Phase ML §9.5 Phase 1 — C-7).

data/tmdb_persons/tmdb_persons.jsonl (TMDB Person + LLM 보강) 를 읽어서
Neo4j 의 기존 (:Person) 노드에 다음 속성을 추가/갱신한다:

    biography_ko, biography_en, style_tags, persona, top_movies,
    popularity, imdb_id, birthday, deathday, gender,
    known_for_department, original_name, place_of_birth, homepage

설계 진실 원본:
    docs/데이터_적재_프로세스_전체분석_및_개선계획.md §9.5 Phase 1 (C-7)

핵심 정책:
    - **person_id 매칭**: MATCH (p:Person {person_id: $pid}) — 기존 노드만 update
    - **신규 노드 만들지 않음**: Person 노드는 영화 적재 과정에서만 생성됨
    - **NULL safe**: 빈 값은 기존 속성 유지 (CASE WHEN 패턴)
    - **JSON 배열**: style_tags / top_movies 는 Neo4j list 로 저장
    - **멱등**: 같은 person_id 에 대해 여러 번 실행 안전 (마지막 값으로 덮어쓰기)
    - **배치**: UNWIND $batch 500건씩 처리

선결 조건:
    - run_tmdb_persons_collect.py 완료 (JSONL 존재)
    - (선택) run_persons_full_pipeline.py 완료 (LLM 보강 필드 포함)
    - Neo4j 가동 + 영화 적재 후 Person 노드 존재 (Task #5 또는 KOBIS/KMDb 적재 후)

Task #5 와의 충돌:
    - Task #5 는 (:Person) 노드를 MERGE 로 생성하지만 SET 은 person_id/profile_path 정도만
    - C-7 은 추가 속성 (biography_ko 등) 만 SET → 키 컬럼 충돌 없음
    - **단 C-7 실행은 Task #5 완료 후로 한정** (동시 쓰기 시 deadlock 가능성 회피)

성능 추정:
    - Neo4j Person 약 572K → JSONL 약 572K 라인
    - 배치 500 × 1144 → 약 5~10분

사용법:
    # 전체 실행
    PYTHONPATH=src uv run python scripts/run_persons_neo4j_enrich.py

    # 처음 N명만 (테스트)
    PYTHONPATH=src uv run python scripts/run_persons_neo4j_enrich.py --limit 100

    # 재개
    PYTHONPATH=src uv run python scripts/run_persons_neo4j_enrich.py --resume

    # 상태 확인
    PYTHONPATH=src uv run python scripts/run_persons_neo4j_enrich.py --status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# .env 로드
_env_file = _project_root / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

import structlog  # noqa: E402

from monglepick.db.clients import init_all_clients, close_all_clients, get_neo4j  # noqa: E402

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════

INPUT_JSONL = Path("data/tmdb_persons/tmdb_persons.jsonl")
CHECKPOINT_FILE = Path("data/tmdb_persons/neo4j_enrich_checkpoint.json")
DEFAULT_BATCH_SIZE = 500    # Neo4j UNWIND 배치


# ══════════════════════════════════════════════════════════════
# Cypher
# ══════════════════════════════════════════════════════════════

# 기존 (:Person {person_id: pid}) 노드만 update.
# CASE WHEN 으로 빈 값일 때 기존 속성 유지.
ENRICH_CYPHER = """
UNWIND $batch AS row
MATCH (p:Person {person_id: row.person_id})
SET p.biography_ko       = CASE WHEN row.biography_ko       <> '' THEN row.biography_ko       ELSE p.biography_ko       END,
    p.biography_en       = CASE WHEN row.biography_en       <> '' THEN row.biography_en       ELSE p.biography_en       END,
    p.original_name      = CASE WHEN row.original_name      <> '' THEN row.original_name      ELSE p.original_name      END,
    p.place_of_birth     = CASE WHEN row.place_of_birth     <> '' THEN row.place_of_birth     ELSE p.place_of_birth     END,
    p.birthday           = CASE WHEN row.birthday           <> '' THEN row.birthday           ELSE p.birthday           END,
    p.deathday           = CASE WHEN row.deathday           <> '' THEN row.deathday           ELSE p.deathday           END,
    p.gender             = CASE WHEN row.gender > 0          THEN row.gender                  ELSE p.gender             END,
    p.popularity         = CASE WHEN row.popularity > 0      THEN row.popularity              ELSE p.popularity         END,
    p.known_for_department = CASE WHEN row.known_for_department <> '' THEN row.known_for_department ELSE p.known_for_department END,
    p.imdb_id            = CASE WHEN row.imdb_id            <> '' THEN row.imdb_id            ELSE p.imdb_id            END,
    p.homepage           = CASE WHEN row.homepage           <> '' THEN row.homepage           ELSE p.homepage           END,
    p.style_tags         = CASE WHEN size(row.style_tags) > 0 THEN row.style_tags             ELSE p.style_tags         END,
    p.persona            = CASE WHEN row.persona            <> '' THEN row.persona            ELSE p.persona            END,
    p.top_movies         = CASE WHEN size(row.top_movies) > 0 THEN row.top_movies             ELSE p.top_movies         END,
    p.profile_path       = CASE WHEN row.profile_path       <> '' THEN row.profile_path       ELSE p.profile_path       END,
    p.updated_at         = timestamp()
RETURN count(p) AS updated
"""


# ══════════════════════════════════════════════════════════════
# 체크포인트
# ══════════════════════════════════════════════════════════════


def _new_checkpoint() -> dict:
    return {
        "phase": "",
        "last_jsonl_line": 0,
        "total_processed": 0,
        "total_updated": 0,        # Cypher RETURN count(p)
        "total_skipped": 0,         # JSON 파싱 실패 등
        "total_no_match": 0,        # JSONL 에는 있지만 Neo4j 에 person_id 없음 (배치 합계 - updated)
        "start_time": datetime.now().isoformat(),
        "last_updated": "",
    }


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("checkpoint_load_failed", error=str(e))
    return _new_checkpoint()


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════
# Person dict → Cypher row dict 변환
# ══════════════════════════════════════════════════════════════


def _person_to_cypher_row(person: dict) -> dict | None:
    """
    TMDB Person dict (LLM 보강 가능) → Cypher UNWIND row 매핑.

    빈 문자열로 정규화 (None → ""), int 로 정규화 (gender/popularity).
    list 필드는 그대로 (Neo4j 가 list 를 지원).
    """
    pid = person.get("id")
    if not pid:
        return None
    try:
        person_id = int(pid)
    except (ValueError, TypeError):
        return None

    aka = person.get("also_known_as") or []
    original_name = (aka[0] if aka else "") or ""

    return {
        "person_id":           person_id,
        "biography_ko":        (person.get("llm_biography_ko") or "").strip(),
        "biography_en":        (person.get("biography") or "").strip(),
        "original_name":       (original_name or "").strip()[:200],
        "place_of_birth":      (person.get("place_of_birth") or "").strip()[:200],
        "birthday":            (person.get("birthday") or "")[:10],
        "deathday":            (person.get("deathday") or "")[:10],
        "gender":              int(person.get("gender") or 0),
        "popularity":          float(person.get("popularity") or 0.0),
        "known_for_department": (person.get("known_for_department") or "")[:50],
        "imdb_id":             (person.get("imdb_id") or "")[:20],
        "homepage":            (person.get("homepage") or "")[:500],
        "style_tags":          [str(t) for t in (person.get("llm_style_tags") or []) if t][:10],
        "persona":             (person.get("llm_persona") or "").strip()[:150],
        "top_movies":          [str(m) for m in (person.get("llm_top_movies") or []) if m][:5],
        "profile_path":        (person.get("profile_path") or "")[:500],
    }


# ══════════════════════════════════════════════════════════════
# JSONL 배치 스트리밍
# ══════════════════════════════════════════════════════════════


def _jsonl_batches(path: Path, batch_size: int, skip_lines: int = 0):
    """JSONL → batch_size 단위로 (rows, last_line, invalid_count) yield."""
    rows: list[dict] = []
    line_no = 0
    invalid_count = 0

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line_no += 1
            if line_no <= skip_lines:
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid_count += 1
                continue

            row = _person_to_cypher_row(obj)
            if row is None:
                invalid_count += 1
                continue

            rows.append(row)

            if len(rows) >= batch_size:
                yield rows, line_no, invalid_count
                rows = []
                invalid_count = 0

    if rows:
        yield rows, line_no, invalid_count


# ══════════════════════════════════════════════════════════════
# Neo4j 배치 실행
# ══════════════════════════════════════════════════════════════


async def _enrich_batch(rows: list[dict]) -> int:
    """
    UNWIND 배치로 Person 노드 속성 update.

    Returns:
        실제 update 된 노드 수 (Neo4j 에 person_id 가 없으면 0)
    """
    if not rows:
        return 0
    driver = await get_neo4j()
    async with driver.session() as session:
        result = await session.run(ENRICH_CYPHER, batch=rows)
        record = await result.single()
        return int(record["updated"]) if record else 0


# ══════════════════════════════════════════════════════════════
# 메인 파이프라인
# ══════════════════════════════════════════════════════════════


async def run_persons_neo4j_enrich(
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
    resume: bool = False,
) -> None:
    pipeline_start = time.time()

    if not INPUT_JSONL.exists():
        print(f"[ERROR] JSONL 파일이 없습니다: {INPUT_JSONL}")
        print(f"        먼저 run_tmdb_persons_collect.py 를 실행하세요.")
        return

    checkpoint = _load_checkpoint() if resume else _new_checkpoint()
    skip_lines = checkpoint.get("last_jsonl_line", 0) if resume else 0

    print(f"[Step 0] Neo4j 클라이언트 초기화")
    await init_all_clients()

    try:
        # Neo4j Person 총 노드 수 확인
        driver = await get_neo4j()
        async with driver.session() as session:
            result = await session.run("MATCH (p:Person) RETURN count(p) AS cnt")
            record = await result.single()
            total_persons = int(record["cnt"]) if record else 0

        print(f"  Neo4j (:Person) 노드: {total_persons:,}")

        if total_persons == 0:
            print("[ERROR] Neo4j 에 Person 노드가 없습니다. 영화 적재 후 재실행하세요.")
            return

        print(f"\n[Step 1] JSONL → Person 노드 속성 보강")
        print(f"  JSONL: {INPUT_JSONL}")
        print(f"  batch_size: {batch_size}")
        print(f"  skip_lines: {skip_lines:,}")
        if limit:
            print(f"  limit: {limit:,}")
        print()

        checkpoint["phase"] = "enriching"

        for batch_rows, last_line, invalid_count in _jsonl_batches(
            INPUT_JSONL, batch_size, skip_lines=skip_lines
        ):
            chunk_start = time.time()

            try:
                updated = await _enrich_batch(batch_rows)
            except Exception as e:
                logger.error(
                    "neo4j_enrich_batch_failed",
                    last_line=last_line,
                    batch_size=len(batch_rows),
                    error=str(e)[:200],
                )
                # 실패해도 진행 (다음 배치로)
                updated = 0

            no_match = len(batch_rows) - updated
            checkpoint["total_processed"] += len(batch_rows) + invalid_count
            checkpoint["total_updated"] += updated
            checkpoint["total_no_match"] += no_match
            checkpoint["total_skipped"] += invalid_count
            checkpoint["last_jsonl_line"] = last_line
            _save_checkpoint(checkpoint)

            chunk_elapsed = time.time() - chunk_start
            total_elapsed = time.time() - pipeline_start
            rate = checkpoint["total_updated"] / total_elapsed if total_elapsed > 0 else 0

            print(
                f"  [Batch] +{updated:>4} updated, {no_match:>3} no_match, {invalid_count:>2} invalid | "
                f"누적 updated {checkpoint['total_updated']:>8,} | "
                f"속도 {rate:>5.1f}/s | "
                f"청크 {chunk_elapsed:>5.1f}s | "
                f"line {last_line:>9,}"
            )

            if limit and checkpoint["total_updated"] >= limit:
                print(f"  --limit {limit} 도달 → 중단")
                break

        checkpoint["phase"] = "done"
        _save_checkpoint(checkpoint)

        # 최종 검증: 보강된 Person 노드 수
        async with driver.session() as session:
            result = await session.run(
                "MATCH (p:Person) WHERE p.biography_ko IS NOT NULL OR p.persona IS NOT NULL "
                "RETURN count(p) AS enriched_count"
            )
            record = await result.single()
            enriched_in_neo4j = int(record["enriched_count"]) if record else 0

        total_elapsed = time.time() - pipeline_start
        print(f"\n{'=' * 60}")
        print(f"[Person Neo4j 보강 완료]")
        print(f"  Neo4j Person 전체:  {total_persons:>10,}")
        print(f"  처리 line:          {checkpoint['last_jsonl_line']:>10,}")
        print(f"  Cypher updated:     {checkpoint['total_updated']:>10,}")
        print(f"  no_match (JSONL에는 있지만 Neo4j 없음): {checkpoint['total_no_match']:>10,}")
        print(f"  invalid (파싱 실패):{checkpoint['total_skipped']:>10,}")
        print(f"  Neo4j 보강 완료 노드: {enriched_in_neo4j:>10,}")
        print(f"  소요:               {total_elapsed / 60:>10.1f} 분")
        print(f"{'=' * 60}")

    finally:
        await close_all_clients()


# ══════════════════════════════════════════════════════════════
# 상태 조회
# ══════════════════════════════════════════════════════════════


async def show_status() -> None:
    cp = _load_checkpoint()
    print("=" * 60)
    print(f"  Person Neo4j Enrich 체크포인트")
    print("=" * 60)
    print(f"  단계:                   {cp.get('phase', '미시작')}")
    print(f"  마지막 라인:            {cp.get('last_jsonl_line', 0):>10,}")
    print(f"  처리:                   {cp.get('total_processed', 0):>10,}")
    print(f"  Cypher updated:         {cp.get('total_updated', 0):>10,}")
    print(f"  no_match:               {cp.get('total_no_match', 0):>10,}")
    print(f"  invalid:                {cp.get('total_skipped', 0):>10,}")
    print(f"  마지막 갱신:            {cp.get('last_updated', '-')}")
    print()

    try:
        await init_all_clients()
        driver = await get_neo4j()
        async with driver.session() as session:
            r1 = await session.run("MATCH (p:Person) RETURN count(p) AS cnt")
            total = int((await r1.single())["cnt"])

            r2 = await session.run(
                "MATCH (p:Person) WHERE p.biography_ko IS NOT NULL OR p.persona IS NOT NULL "
                "RETURN count(p) AS cnt"
            )
            enriched = int((await r2.single())["cnt"])

            print(f"  Neo4j Person 전체:  {total:>10,}")
            print(f"  보강된 노드:        {enriched:>10,}")
            print(f"  보강 비율:          {enriched * 100 / max(total, 1):>9.1f}%")
    except Exception as e:
        print(f"  Neo4j 조회 실패: {e}")
    finally:
        try:
            await close_all_clients()
        except Exception:
            pass

    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Person JSONL → Neo4j (:Person) 노드 속성 보강",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"UNWIND 배치 크기 (기본 {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="최대 update 건수 (테스트)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="체크포인트 last_jsonl_line 부터 재개",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="현재 체크포인트 + Neo4j 통계만 출력",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.status:
        asyncio.run(show_status())
    else:
        asyncio.run(
            run_persons_neo4j_enrich(
                batch_size=args.batch_size,
                limit=args.limit,
                resume=args.resume,
            )
        )
