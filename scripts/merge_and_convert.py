"""
LoRA 어댑터 병합 + GGUF 변환 + 양자화 통합 스크립트 (M-LLM-5).

[개요]
  finetune_mongle.py로 학습된 LoRA 어댑터를 베이스 모델에 병합하고,
  llama.cpp 도구를 이용해 GGUF 포맷으로 변환 후 Q4_K_M 양자화까지
  하나의 스크립트에서 완료한다.

[실행 순서]
  Step 1: LoRA 병합 (peft merge_and_unload)
    - models/mongle-lora  +  LGAI-EXAONE/EXAONE-4.0-1.2B
    → models/mongle-merged  (HuggingFace 포맷, ~2.5GB)

  Step 2: GGUF 변환 (llama.cpp convert_hf_to_gguf.py)
    - models/mongle-merged
    → models/mongle-exaone4-f16.gguf  (FP16, ~2.3GB)

  Step 3: Q4_K_M 양자화 (llama.cpp llama-quantize)
    - models/mongle-exaone4-f16.gguf
    → models/mongle-exaone4-q4km.gguf  (~0.8GB, Ollama용)

[사용법]
  # 기본 (권장)
  python scripts/merge_and_convert.py \\
    --base_model "LGAI-EXAONE/EXAONE-4.0-1.2B" \\
    --lora_path "models/mongle-lora" \\
    --output_gguf "models/mongle-exaone4-q4km.gguf" \\
    --quantize Q4_K_M

  # 병합만 (GGUF 변환 스킵)
  python scripts/merge_and_convert.py \\
    --lora_path "models/mongle-lora" \\
    --merged_dir "models/mongle-merged" \\
    --skip_gguf

  # llama.cpp 경로 직접 지정
  python scripts/merge_and_convert.py \\
    --lora_path "models/mongle-lora" \\
    --llama_cpp_dir "/opt/llama.cpp" \\
    --quantize Q4_K_M

  # Ollama 자동 등록 포함
  python scripts/merge_and_convert.py \\
    --lora_path "models/mongle-lora" \\
    --output_gguf "models/mongle-exaone4-q4km.gguf" \\
    --register_ollama \\
    --modelfile "Modelfile.mongle"

[llama.cpp 설치]
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp && cmake -B build && cmake --build build --config Release -j
  pip install -r llama.cpp/requirements.txt  # convert_hf_to_gguf.py 의존성

[지원 양자화 방식]
  Q4_K_M  — 권장 (품질/크기 균형, ~0.8GB)
  Q5_K_M  — 고품질 (약 20% 크기 증가)
  Q8_0    — 거의 무손실 (~1.4GB)
  F16     — 무손실 FP16 (~2.3GB, 변환 단계 중간 산출물)
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("merge_and_convert")

# ─── 지원 양자화 방식 목록 ───────────────────────────────────────────────────
SUPPORTED_QUANT_TYPES = ["Q4_K_M", "Q5_K_M", "Q8_0", "F16", "Q4_0", "Q6_K"]


# ─── Step 1: LoRA 어댑터 병합 ────────────────────────────────────────────────

def merge_lora(
    base_model: str,
    lora_path: str,
    merged_dir: str,
) -> Path:
    """
    LoRA 어댑터를 베이스 모델에 병합하여 완전한 모델로 저장한다.

    peft의 merge_and_unload()를 사용하면 어댑터 가중치가 베이스 모델에
    수학적으로 통합되므로 추론 시 별도 어댑터 없이 단일 모델로 동작한다.

    Args:
        base_model: HuggingFace 모델 ID 또는 로컬 경로
        lora_path:  finetune_mongle.py 출력 경로 (LoRA 어댑터)
        merged_dir: 병합 모델 저장 경로

    Returns:
        병합된 모델이 저장된 Path 객체

    Raises:
        ImportError: transformers 또는 peft 미설치 시
        FileNotFoundError: lora_path 없을 때
    """
    # ─── 의존성 확인 ──────────────────────────────────────────────────────────
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise ImportError(
            f"필수 패키지 미설치: {e}\n"
            "  pip install transformers peft accelerate"
        )

    lora_path_obj = Path(lora_path)
    if not lora_path_obj.exists():
        raise FileNotFoundError(
            f"LoRA 어댑터 경로가 없습니다: {lora_path}\n"
            "  → finetune_mongle.py를 먼저 실행하여 어댑터를 학습하세요."
        )

    merged_path = Path(merged_dir)
    merged_path.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 1: LoRA 병합 시작 ===")
    logger.info("  베이스 모델: %s", base_model)
    logger.info("  LoRA 어댑터: %s", lora_path_obj.resolve())
    logger.info("  병합 출력:   %s", merged_path.resolve())

    # 베이스 모델 로드 (CPU로 로드 후 병합 — VRAM 절약)
    logger.info("베이스 모델 로드 중 (CPU)...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,  # FP16으로 병합 수행 (메모리 절약)
        device_map="cpu",           # 병합은 CPU에서 안전하게 수행
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    logger.info("베이스 모델 로드 완료")

    # LoRA 어댑터 부착
    logger.info("LoRA 어댑터 부착 중...")
    model = PeftModel.from_pretrained(
        base,
        str(lora_path_obj),
        torch_dtype=torch.float16,
    )
    logger.info("LoRA 어댑터 부착 완료")

    # 병합 + 언로드 — 어댑터 가중치를 베이스에 수학적으로 통합
    # 이 후 model은 일반 AutoModelForCausalLM과 동일하게 동작
    logger.info("merge_and_unload() 실행 중 (수 분 소요)...")
    model = model.merge_and_unload()
    logger.info("병합 완료")

    # 병합 모델 저장 (HuggingFace 포맷 — safetensors)
    logger.info("병합 모델 저장 중: %s", merged_path.resolve())
    model.save_pretrained(
        str(merged_path),
        safe_serialization=True,  # .safetensors 형식 (pytorch pickle 보다 안전)
    )
    tokenizer.save_pretrained(str(merged_path))
    logger.info("병합 모델 저장 완료 (~2.5GB 예상)")

    return merged_path


# ─── Step 2: GGUF 변환 ───────────────────────────────────────────────────────

def find_llama_cpp(llama_cpp_dir: Optional[str]) -> Tuple[Optional[Path], Optional[Path]]:
    """
    llama.cpp 도구 경로를 탐색한다.

    탐색 순서:
      1. --llama_cpp_dir 인자
      2. 환경변수 LLAMA_CPP_DIR
      3. 현재 디렉토리 하위 llama.cpp/
      4. 프로젝트 루트 기준 ../llama.cpp/
      5. PATH에서 llama-quantize 직접 탐색

    Args:
        llama_cpp_dir: 사용자 지정 llama.cpp 루트 경로 (없으면 None)

    Returns:
        (convert_script_path, quantize_bin_path) 튜플
        각각 없으면 None 반환
    """
    search_dirs = []

    # 1. 사용자 지정 경로
    if llama_cpp_dir:
        search_dirs.append(Path(llama_cpp_dir))

    # 2. 환경변수
    if env_dir := os.environ.get("LLAMA_CPP_DIR"):
        search_dirs.append(Path(env_dir))

    # 3. 프로젝트 상대 경로들
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    search_dirs.extend([
        project_root / "llama.cpp",
        project_root.parent / "llama.cpp",
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
    ])

    convert_script = None
    quantize_bin = None

    for d in search_dirs:
        if not d.is_dir():
            continue

        # convert_hf_to_gguf.py 탐색
        candidate_convert = d / "convert_hf_to_gguf.py"
        if candidate_convert.exists() and convert_script is None:
            convert_script = candidate_convert
            logger.debug("convert_hf_to_gguf.py 발견: %s", convert_script)

        # llama-quantize 바이너리 탐색 (빌드 결과물 위치는 버전마다 다름)
        for bin_path in [
            d / "build" / "bin" / "llama-quantize",
            d / "build" / "bin" / "llama-quantize.exe",
            d / "llama-quantize",
            d / "quantize",             # 구버전
        ]:
            if bin_path.exists() and quantize_bin is None:
                quantize_bin = bin_path
                logger.debug("llama-quantize 발견: %s", quantize_bin)

    # 4. PATH에서 llama-quantize 탐색
    if quantize_bin is None:
        path_bin = shutil.which("llama-quantize")
        if path_bin:
            quantize_bin = Path(path_bin)
            logger.debug("PATH에서 llama-quantize 발견: %s", quantize_bin)

    return convert_script, quantize_bin


def convert_to_gguf_f16(
    merged_dir: str,
    f16_gguf_path: str,
    convert_script: Path,
) -> Path:
    """
    병합된 HuggingFace 모델을 GGUF FP16 포맷으로 변환한다.

    llama.cpp의 convert_hf_to_gguf.py 스크립트를 subprocess로 호출한다.

    Args:
        merged_dir:      병합 모델 디렉토리 (Step 1 출력)
        f16_gguf_path:   FP16 GGUF 저장 경로 (중간 산출물)
        convert_script:  convert_hf_to_gguf.py 절대 경로

    Returns:
        생성된 FP16 GGUF 파일 Path

    Raises:
        subprocess.CalledProcessError: 변환 실패 시
    """
    f16_path = Path(f16_gguf_path)
    f16_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 2: GGUF 변환 (FP16) ===")
    logger.info("  입력: %s", merged_dir)
    logger.info("  출력: %s", f16_path)
    logger.info("  스크립트: %s", convert_script)

    cmd = [
        sys.executable,           # 현재 파이썬 인터프리터
        str(convert_script),
        merged_dir,                # 입력 디렉토리 (HuggingFace 포맷)
        "--outfile", str(f16_path),
        "--outtype", "f16",        # FP16 출력 (양자화 전 중간 단계)
    ]

    logger.info("실행: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # 진행 상황 실시간 출력
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"GGUF 변환 실패 (exit code {e.returncode})\n"
            f"  명령: {' '.join(cmd)}\n"
            "  → llama.cpp의 requirements.txt를 설치했는지 확인:\n"
            f"    pip install -r {convert_script.parent}/requirements.txt"
        )

    if not f16_path.exists():
        raise RuntimeError(f"GGUF 파일이 생성되지 않았습니다: {f16_path}")

    size_gb = f16_path.stat().st_size / (1024 ** 3)
    logger.info("FP16 GGUF 생성 완료: %.2fGB", size_gb)
    return f16_path


def quantize_gguf(
    f16_gguf_path: Path,
    output_gguf_path: str,
    quant_type: str,
    quantize_bin: Path,
) -> Path:
    """
    FP16 GGUF를 지정된 양자화 방식으로 압축한다.

    llama-quantize 바이너리를 subprocess로 호출한다.
    Q4_K_M 기준 ~0.8GB로 압축 (원본 FP16 ~2.3GB 대비 약 65% 감소).

    Args:
        f16_gguf_path:   FP16 GGUF 파일 경로 (Step 2 출력)
        output_gguf_path: 양자화 결과 GGUF 저장 경로
        quant_type:      양자화 방식 (Q4_K_M, Q5_K_M, Q8_0 등)
        quantize_bin:    llama-quantize 바이너리 경로

    Returns:
        생성된 양자화 GGUF 파일 Path

    Raises:
        subprocess.CalledProcessError: 양자화 실패 시
    """
    output_path = Path(output_gguf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 3: GGUF 양자화 (%s) ===", quant_type)
    logger.info("  입력:  %s", f16_gguf_path)
    logger.info("  출력:  %s", output_path)
    logger.info("  방식:  %s", quant_type)
    logger.info("  도구:  %s", quantize_bin)

    cmd = [
        str(quantize_bin),
        str(f16_gguf_path),  # 입력 FP16 GGUF
        str(output_path),    # 출력 양자화 GGUF
        quant_type,          # 양자화 방식 (Q4_K_M 등)
    ]

    logger.info("실행: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"양자화 실패 (exit code {e.returncode})\n"
            f"  명령: {' '.join(cmd)}\n"
            "  → llama.cpp를 최신 버전으로 빌드했는지 확인:\n"
            "    cd llama.cpp && cmake -B build && cmake --build build --config Release"
        )

    if not output_path.exists():
        raise RuntimeError(f"양자화 GGUF 파일이 생성되지 않았습니다: {output_path}")

    size_gb = output_path.stat().st_size / (1024 ** 3)
    logger.info("양자화 완료: %.2fGB (%s)", size_gb, quant_type)
    return output_path


# ─── Step 4 (선택): Ollama 등록 ──────────────────────────────────────────────

def register_ollama(gguf_path: Path, modelfile: str) -> None:
    """
    생성된 GGUF를 Ollama에 모델로 등록한다.

    ollama CLI를 subprocess로 호출한다.
    등록 후 `ollama run mongle` 으로 즉시 테스트 가능.

    Args:
        gguf_path:  양자화된 GGUF 파일 경로
        modelfile:  Modelfile.mongle 경로

    Raises:
        FileNotFoundError: Modelfile 없을 때
        RuntimeError: ollama CLI 없거나 등록 실패 시
    """
    modelfile_path = Path(modelfile)
    if not modelfile_path.exists():
        raise FileNotFoundError(
            f"Modelfile을 찾을 수 없습니다: {modelfile}\n"
            "  → 프로젝트 루트의 Modelfile.mongle을 확인하세요."
        )

    # ollama CLI 존재 확인
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        raise RuntimeError(
            "ollama CLI가 PATH에 없습니다.\n"
            "  → https://ollama.com/download 에서 설치하세요."
        )

    logger.info("=== Step 4: Ollama 모델 등록 ===")
    logger.info("  GGUF:      %s", gguf_path.resolve())
    logger.info("  Modelfile: %s", modelfile_path.resolve())

    # Modelfile에서 FROM 경로를 실제 GGUF 경로로 교체한 임시 파일 생성
    # (상대 경로 문제를 방지하기 위해 절대 경로 사용)
    modelfile_content = modelfile_path.read_text(encoding="utf-8")
    gguf_abs = str(gguf_path.resolve())

    # 'FROM ./' 또는 'FROM models/' 패턴을 실제 경로로 교체
    import re
    updated_content = re.sub(
        r"^FROM\s+.+$",
        f"FROM {gguf_abs}",
        modelfile_content,
        count=1,
        flags=re.MULTILINE,
    )

    # 임시 Modelfile 작성
    tmp_modelfile = Path("models") / "Modelfile.mongle.tmp"
    tmp_modelfile.parent.mkdir(parents=True, exist_ok=True)
    tmp_modelfile.write_text(updated_content, encoding="utf-8")
    logger.info("임시 Modelfile 생성 (절대 경로 적용): %s", tmp_modelfile)

    try:
        cmd = [ollama_bin, "create", "mongle", "-f", str(tmp_modelfile)]
        logger.info("실행: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, text=True)
        logger.info("Ollama 등록 완료: mongle")
        logger.info("")
        logger.info("테스트: ollama run mongle")
        logger.info("  또는 Ollama API: curl http://localhost:11434/api/chat ...")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Ollama 등록 실패 (exit code {e.returncode})\n"
            f"  명령: {' '.join(cmd)}"
        )
    finally:
        # 임시 파일 정리
        if tmp_modelfile.exists():
            tmp_modelfile.unlink()


# ─── 메인 실행 ────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    """
    전체 파이프라인 실행 (병합 → 변환 → 양자화 → [선택] Ollama 등록).

    Args:
        args: argparse.Namespace (CLI 파라미터 전체)
    """
    # ─── 경로 설정 ─────────────────────────────────────────────────────────────
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    merged_dir = args.merged_dir or str(models_dir / "mongle-merged")

    # FP16 GGUF는 최종 GGUF와 같은 디렉토리에 임시 저장
    output_gguf = Path(args.output_gguf)
    f16_gguf = output_gguf.parent / (output_gguf.stem + "-f16.gguf")

    logger.info("=" * 60)
    logger.info("몽글이 병합 + GGUF 변환 시작 (M-LLM-5)")
    logger.info("=" * 60)
    logger.info("LoRA 어댑터: %s", args.lora_path)
    logger.info("병합 출력:   %s", merged_dir)
    logger.info("최종 GGUF:  %s", output_gguf)
    logger.info("양자화:     %s", args.quantize)
    logger.info("")

    # ─── Step 1: LoRA 병합 ────────────────────────────────────────────────────
    merged_path = merge_lora(
        base_model=args.base_model,
        lora_path=args.lora_path,
        merged_dir=merged_dir,
    )
    logger.info("")

    # ─── Step 2 & 3: GGUF 변환 + 양자화 ─────────────────────────────────────
    if args.skip_gguf:
        logger.info("--skip_gguf 지정 — GGUF 변환/양자화 스킵")
        logger.info("병합 모델 위치: %s", merged_path.resolve())
        logger.info("")
        logger.info("수동 GGUF 변환 명령:")
        logger.info(
            "  python llama.cpp/convert_hf_to_gguf.py %s \\\n"
            "    --outfile models/mongle-exaone4-f16.gguf --outtype f16\n"
            "  llama.cpp/build/bin/llama-quantize \\\n"
            "    models/mongle-exaone4-f16.gguf \\\n"
            "    models/mongle-exaone4-q4km.gguf Q4_K_M",
            merged_path,
        )
        return

    # llama.cpp 도구 탐색
    convert_script, quantize_bin = find_llama_cpp(args.llama_cpp_dir)

    if convert_script is None or quantize_bin is None:
        # 도구가 없는 경우 — 안내 메시지 출력 후 종료
        logger.warning("=" * 60)
        logger.warning("llama.cpp 도구를 찾을 수 없습니다.")
        logger.warning("=" * 60)
        logger.warning("")
        if convert_script is None:
            logger.warning("  convert_hf_to_gguf.py  — 없음")
        else:
            logger.warning("  convert_hf_to_gguf.py  — 발견: %s", convert_script)

        if quantize_bin is None:
            logger.warning("  llama-quantize          — 없음")
        else:
            logger.warning("  llama-quantize          — 발견: %s", quantize_bin)

        logger.warning("")
        logger.warning("llama.cpp 설치 방법:")
        logger.warning("  git clone https://github.com/ggerganov/llama.cpp")
        logger.warning("  cd llama.cpp")
        logger.warning("  cmake -B build && cmake --build build --config Release -j4")
        logger.warning("  pip install -r requirements.txt")
        logger.warning("")
        logger.warning("설치 후 재실행:")
        logger.warning(
            "  python scripts/merge_and_convert.py \\\n"
            "    --base_model '%s' \\\n"
            "    --lora_path '%s' \\\n"
            "    --output_gguf '%s' \\\n"
            "    --llama_cpp_dir 'llama.cpp'",
            args.base_model,
            args.lora_path,
            args.output_gguf,
        )
        logger.warning("")
        logger.warning(
            "또는 HuggingFace GGUF 직접 사용 (파인튜닝 없는 경우):\n"
            "  https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF"
        )
        logger.warning("")
        logger.warning("병합 모델은 저장되었습니다: %s", merged_path.resolve())
        return

    # GGUF 변환 (FP16)
    f16_path = convert_to_gguf_f16(
        merged_dir=str(merged_path),
        f16_gguf_path=str(f16_gguf),
        convert_script=convert_script,
    )
    logger.info("")

    # 양자화
    final_gguf = quantize_gguf(
        f16_gguf_path=f16_path,
        output_gguf_path=args.output_gguf,
        quant_type=args.quantize,
        quantize_bin=quantize_bin,
    )
    logger.info("")

    # 중간 FP16 파일 정리 (선택적)
    if args.keep_f16:
        logger.info("FP16 GGUF 유지: %s", f16_path)
    else:
        logger.info("중간 FP16 파일 삭제: %s", f16_path)
        f16_path.unlink(missing_ok=True)

    # ─── Step 4 (선택): Ollama 등록 ──────────────────────────────────────────
    if args.register_ollama:
        logger.info("")
        register_ollama(
            gguf_path=final_gguf,
            modelfile=args.modelfile,
        )

    # ─── 완료 요약 ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ 전체 변환 완료!")
    logger.info("=" * 60)
    logger.info("병합 모델:   %s", merged_path.resolve())
    logger.info("GGUF (최종): %s (%.2fGB)",
                final_gguf.resolve(),
                final_gguf.stat().st_size / (1024 ** 3))
    logger.info("")

    if not args.register_ollama:
        logger.info("다음 단계 — Ollama 등록:")
        logger.info(
            "  python scripts/merge_and_convert.py \\\n"
            "    --lora_path '%s' \\\n"
            "    --output_gguf '%s' \\\n"
            "    --register_ollama --modelfile Modelfile.mongle\n"
            "\n"
            "  또는 직접 등록:\n"
            "  ollama create mongle -f Modelfile.mongle",
            args.lora_path,
            args.output_gguf,
        )


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description=(
            "LoRA 어댑터 병합 + GGUF 변환 + 양자화 통합 스크립트 (M-LLM-5)\n"
            "finetune_mongle.py 실행 후 이 스크립트를 실행한다."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 파이프라인 (병합 → FP16 변환 → Q4_K_M 양자화)
  python scripts/merge_and_convert.py \\
    --base_model "LGAI-EXAONE/EXAONE-4.0-1.2B" \\
    --lora_path "models/mongle-lora" \\
    --output_gguf "models/mongle-exaone4-q4km.gguf" \\
    --quantize Q4_K_M

  # Ollama 자동 등록 포함
  python scripts/merge_and_convert.py \\
    --lora_path "models/mongle-lora" \\
    --output_gguf "models/mongle-exaone4-q4km.gguf" \\
    --register_ollama

  # 병합만 (llama.cpp 없는 경우)
  python scripts/merge_and_convert.py \\
    --lora_path "models/mongle-lora" \\
    --skip_gguf
        """,
    )

    # ─── 모델 경로 ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--base_model",
        type=str,
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="베이스 모델 ID 또는 로컬 경로 (기본: LGAI-EXAONE/EXAONE-4.0-1.2B)",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="models/mongle-lora",
        help="LoRA 어댑터 경로 — finetune_mongle.py의 output_dir (기본: models/mongle-lora)",
    )
    parser.add_argument(
        "--merged_dir",
        type=str,
        default=None,
        help="병합 모델 저장 경로 (기본: models/mongle-merged)",
    )
    parser.add_argument(
        "--output_gguf",
        type=str,
        default="models/mongle-exaone4-q4km.gguf",
        help="최종 양자화 GGUF 저장 경로 (기본: models/mongle-exaone4-q4km.gguf)",
    )

    # ─── GGUF 변환 설정 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--quantize",
        type=str,
        default="Q4_K_M",
        choices=SUPPORTED_QUANT_TYPES,
        help="양자화 방식 (기본: Q4_K_M — 품질/크기 균형)",
    )
    parser.add_argument(
        "--llama_cpp_dir",
        type=str,
        default=None,
        help="llama.cpp 루트 디렉토리 경로 (미지정 시 자동 탐색)",
    )
    parser.add_argument(
        "--skip_gguf",
        action="store_true",
        default=False,
        help="GGUF 변환/양자화 스킵 — 병합만 수행 (llama.cpp 없는 경우)",
    )
    parser.add_argument(
        "--keep_f16",
        action="store_true",
        default=False,
        help="변환 중간 FP16 GGUF 파일 보존 (기본: 양자화 후 삭제)",
    )

    # ─── Ollama 등록 설정 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--register_ollama",
        action="store_true",
        default=False,
        help="GGUF 변환 후 Ollama에 mongle 모델로 자동 등록",
    )
    parser.add_argument(
        "--modelfile",
        type=str,
        default="Modelfile.mongle",
        help="Ollama Modelfile 경로 (기본: Modelfile.mongle, --register_ollama 시 사용)",
    )

    return parser.parse_args()


def main() -> None:
    """
    스크립트 진입점.

    선행 조건:
      1. finetune_mongle.py 실행 완료 (models/mongle-lora 존재)
      2. transformers, peft 설치
      3. GGUF 변환을 위한 llama.cpp 빌드 (없으면 --skip_gguf)
    """
    args = parse_args()

    try:
        run(args)
    except FileNotFoundError as e:
        logger.error("파일 없음: %s", e)
        sys.exit(1)
    except ImportError as e:
        logger.error("의존성 오류: %s", e)
        sys.exit(1)
    except RuntimeError as e:
        logger.error("실행 오류: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("예상치 못한 오류: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
