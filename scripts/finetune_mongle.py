"""
EXAONE 4.0 1.2B LoRA 파인튜닝 스크립트 — 몽글이 페르소나 학습 (M-LLM-4).

[개요]
  베이스 모델(LGAI-EXAONE/EXAONE-4.0-1.2B)에 LoRA 어댑터를 부착하여
  몽글픽 AI 어시스턴트 "몽글이" 페르소나를 학습한다.

  몽글이는 분석/추론이 아닌 텍스트 '생성(말하기)'만 담당하므로
  1.2B 소형 모델로 속도를 극대화하는 전략이다.

[학습 환경]
  - 권장: Google Colab T4 (무료 티어) — 1.2B는 T4에서 FP16으로 30분~1시간
  - 대안: A100 / 로컬 Apple Silicon (MPS)
  - VRAM: FP16 기준 ~3~4GB (T4 15GB의 25% 이하)

[사용법]
  # 기본 실행 (Google Colab 또는 로컬)
  python scripts/finetune_mongle.py \
    --base_model "LGAI-EXAONE/EXAONE-4.0-1.2B" \
    --data_path "data/finetune/mongle_train.jsonl" \
    --output_dir "models/mongle-lora" \
    --lora_r 8 --lora_alpha 16 \
    --num_epochs 3 --batch_size 8 --learning_rate 2e-4

  # 검증 데이터셋 포함 실행
  python scripts/finetune_mongle.py \
    --data_path "data/finetune/mongle_train.jsonl" \
    --eval_path "data/finetune/mongle_eval.jsonl" \
    --output_dir "models/mongle-lora"

  # CPU 전용 (MPS 미지원 환경)
  python scripts/finetune_mongle.py \
    --no_fp16 --batch_size 2 --gradient_accumulation_steps 8

[요구사항 설치]
  pip install transformers datasets peft trl accelerate bitsandbytes

[데이터 형식 — JSONL]
  {"instruction": "시스템 프롬프트", "input": "사용자 메시지", "output": "어시스턴트 응답"}

[EXAONE 4.0 채팅 템플릿]
  [|system|]{instruction}[|endofturn|]
  [|user|]{input}[|endofturn|]
  [|assistant|]{output}[|endofturn|]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ─── 루트에서 직접 실행 지원 ───────────────────────────────────────────────
# PYTHONPATH 없이도 실행 가능하도록 src 경로 추가 (선택적)
_project_root = Path(__file__).parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("finetune_mongle")


# ─── EXAONE 4.0 채팅 템플릿 ────────────────────────────────────────────────

def format_exaone_chat(instruction: str, user_input: str, assistant_output: str) -> str:
    """
    EXAONE 4.0 공식 채팅 템플릿을 적용하여 학습용 텍스트를 생성한다.

    EXAONE 4.0은 [|system|], [|user|], [|assistant|] 특수 토큰과
    [|endofturn|] 구분자를 사용한다.

    Args:
        instruction: 시스템 프롬프트 (몽글이 역할 정의)
        user_input:  사용자 메시지
        assistant_output: 어시스턴트 응답 (학습 타겟)

    Returns:
        EXAONE 채팅 형식의 완성된 텍스트
    """
    return (
        f"[|system|]{instruction}[|endofturn|]\n"
        f"[|user|]{user_input}[|endofturn|]\n"
        f"[|assistant|]{assistant_output}[|endofturn|]"
    )


def load_jsonl(file_path: str) -> list[dict]:
    """
    JSONL 파일을 읽어 딕셔너리 리스트로 반환한다.

    Args:
        file_path: JSONL 파일 경로

    Returns:
        각 줄을 파싱한 딕셔너리 리스트

    Raises:
        FileNotFoundError: 파일이 없을 때
        ValueError: JSONL 파싱 실패 또는 필수 키 누락 시
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {file_path}\n"
            "  → data/finetune/mongle_train.jsonl 경로를 확인하거나\n"
            "  → scripts/generate_training_data.py로 먼저 학습 데이터를 생성하세요."
        )

    records = []
    required_keys = {"instruction", "input", "output"}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # 빈 줄 건너뜀
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 파싱 실패 (line {line_num}): {e}")

            # 필수 키 검증
            missing = required_keys - record.keys()
            if missing:
                raise ValueError(
                    f"필수 키 누락 (line {line_num}): {missing}\n"
                    f"  필요한 키: {required_keys}"
                )
            records.append(record)

    logger.info("데이터 로드 완료: %d 건 (%s)", len(records), file_path)
    return records


def build_hf_dataset(records: list[dict]):
    """
    딕셔너리 리스트를 HuggingFace Dataset 객체로 변환하고
    EXAONE 채팅 템플릿을 적용한다.

    Args:
        records: instruction/input/output 딕셔너리 리스트

    Returns:
        'text' 컬럼이 추가된 HuggingFace Dataset
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "datasets 라이브러리가 없습니다.\n"
            "  pip install datasets"
        )

    def apply_template(example: dict) -> dict:
        """각 샘플에 EXAONE 4.0 채팅 템플릿을 적용한다."""
        return {
            "text": format_exaone_chat(
                instruction=example["instruction"],
                user_input=example["input"],
                assistant_output=example["output"],
            )
        }

    dataset = Dataset.from_list(records)
    dataset = dataset.map(apply_template, desc="채팅 템플릿 적용")
    return dataset


def check_gpu_environment() -> str:
    """
    학습에 사용할 디바이스를 결정하고 환경 정보를 로깅한다.

    Returns:
        디바이스 문자열 ("cuda", "mps", "cpu")
    """
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info("CUDA GPU 감지: %s (VRAM: %.1fGB)", gpu_name, vram_gb)
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS 감지 — FP16 미지원, BF16 또는 FP32 사용 권장")
            return "mps"
        else:
            logger.warning("GPU 없음 — CPU로 학습 (매우 느림, 권장하지 않음)")
            return "cpu"
    except ImportError:
        raise ImportError(
            "PyTorch가 없습니다.\n"
            "  pip install torch  또는\n"
            "  Colab 환경에서 런타임 → GPU 사용 설정 후 재실행하세요."
        )


def run_finetune(args: argparse.Namespace) -> None:
    """
    LoRA 파인튜닝 메인 로직.

    실행 순서:
      1. GPU 환경 확인
      2. 학습/검증 데이터 로드 + 템플릿 적용
      3. 베이스 모델 + 토크나이저 로드 (FP16)
      4. LoRA 어댑터 부착 (r=8, alpha=16)
      5. SFTTrainer로 학습
      6. LoRA 어댑터 저장 → output_dir

    Args:
        args: argparse.Namespace (CLI 파라미터 전체)
    """
    # ─── 의존성 체크 ─────────────────────────────────────────────────────────
    missing_packages = []
    for pkg, import_name in [
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("accelerate", "accelerate"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        raise ImportError(
            f"필수 패키지 누락: {missing_packages}\n"
            "  pip install transformers peft trl accelerate bitsandbytes"
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    # ─── 환경 확인 ────────────────────────────────────────────────────────────
    device = check_gpu_environment()

    # MPS는 FP16 지원 불완전 → 강제 비활성화
    use_fp16 = args.fp16 and (device == "cuda")
    if args.fp16 and device == "mps":
        logger.warning("MPS는 FP16 미지원 → FP16 비활성화 (BF16 또는 FP32로 학습)")

    # ─── 출력 디렉토리 생성 ──────────────────────────────────────────────────
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("LoRA 어댑터 저장 경로: %s", output_path.resolve())

    # ─── 데이터 로드 ─────────────────────────────────────────────────────────
    logger.info("=== 학습 데이터 로드 ===")
    train_records = load_jsonl(args.data_path)
    train_dataset = build_hf_dataset(train_records)
    logger.info("학습 샘플 수: %d", len(train_dataset))

    # 검증 데이터셋 (선택적)
    eval_dataset = None
    if args.eval_path:
        logger.info("검증 데이터 로드: %s", args.eval_path)
        eval_records = load_jsonl(args.eval_path)
        eval_dataset = build_hf_dataset(eval_records)
        logger.info("검증 샘플 수: %d", len(eval_dataset))
    else:
        logger.info("검증 데이터 없음 (--eval_path 미지정)")

    # ─── 베이스 모델 로드 ────────────────────────────────────────────────────
    logger.info("=== 베이스 모델 로드: %s ===", args.base_model)
    logger.info(
        "  주의: 최초 실행 시 HuggingFace에서 ~2.5GB 다운로드됩니다 "
        "(LGAI-EXAONE/EXAONE-4.0-1.2B)"
    )

    import torch

    # FP16 또는 auto 정밀도로 로드 (T4에서 FP16은 VRAM ~3GB)
    torch_dtype = torch.float16 if use_fp16 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",          # GPU/CPU 자동 배분
        trust_remote_code=True,     # EXAONE 커스텀 코드 허용
    )
    logger.info("베이스 모델 로드 완료")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # EXAONE 4.0 패딩 토큰 설정
    # — EOS와 동일하게 설정하는 것이 공식 권장 방식
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token을 eos_token으로 설정: %s", tokenizer.eos_token)

    # ─── LoRA 어댑터 설정 ────────────────────────────────────────────────────
    # 1.2B 소형 모델에 최적화된 r=8, alpha=16 설정
    # target_modules: Attention QKV + 출력 프로젝션 전체 커버
    logger.info(
        "=== LoRA 설정: r=%d, alpha=%d, dropout=%.2f ===",
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
    )
    lora_config = LoraConfig(
        r=args.lora_r,                   # LoRA 랭크 (파라미터 수 제어)
        lora_alpha=args.lora_alpha,      # 스케일링 팩터 (alpha/r = 2.0)
        target_modules=[                 # Attention 레이어 전체
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=args.lora_dropout,  # 과적합 방지
        bias="none",                     # 바이어스는 학습하지 않음
        task_type=TaskType.CAUSAL_LM,    # 인과적 언어 모델링
    )
    model = get_peft_model(model, lora_config)

    # 학습 파라미터 수 출력 (전체 대비 LoRA 파라미터 비율)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "학습 파라미터: %s / %s (%.2f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        100.0 * trainable_params / total_params,
    )

    # ─── 학습 인자 설정 ──────────────────────────────────────────────────────
    # gradient_accumulation_steps=2 → 실질적 배치 크기 = batch_size × 2
    # warmup_steps=30 → 초기 LR 워밍업으로 안정적 학습 시작
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=30,              # LR 워밍업 (30스텝)
        logging_steps=10,             # 10스텝마다 loss 출력
        save_strategy="epoch",        # 에폭마다 체크포인트 저장
        eval_strategy=(               # 검증 데이터 있을 때만 평가
            "epoch" if eval_dataset is not None else "no"
        ),
        load_best_model_at_end=(eval_dataset is not None),
        fp16=use_fp16,                # CUDA FP16 학습
        bf16=(device == "mps"),       # MPS는 BF16 사용
        report_to="none",             # LangSmith/WandB 비활성화 (선택적 활성화)
        dataloader_num_workers=0,     # Colab 환경 호환
        remove_unused_columns=True,
        # 학습 안정성: 그레이디언트 클리핑
        max_grad_norm=1.0,
    )

    # ─── SFTTrainer 구성 + 학습 ──────────────────────────────────────────────
    logger.info("=== SFTTrainer 초기화 ===")
    logger.info(
        "  에폭: %d | 배치: %d | 누적 스텝: %d | LR: %.2e",
        args.num_epochs,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,         # None이면 평가 스킵
        dataset_text_field="text",         # format_exaone_chat이 생성한 컬럼
        max_seq_length=args.max_seq_length,
        args=training_args,
        peft_config=lora_config,           # trl>=0.8 에서는 peft_config 직접 전달 가능
    )

    # ─── 학습 실행 ───────────────────────────────────────────────────────────
    logger.info("=== 학습 시작 (T4 기준 ~30분~1시간 예상) ===")
    train_result = trainer.train()
    logger.info("학습 완료: %s", train_result)

    # ─── LoRA 어댑터 저장 ────────────────────────────────────────────────────
    # merge_and_convert.py에서 이 경로를 --lora_path로 참조한다
    logger.info("=== LoRA 어댑터 저장: %s ===", output_path.resolve())
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("")
    logger.info("✓ 파인튜닝 완료!")
    logger.info("  LoRA 어댑터 경로: %s", output_path.resolve())
    logger.info("")
    logger.info("다음 단계 — LoRA 병합 + GGUF 변환:")
    logger.info(
        "  python scripts/merge_and_convert.py \\\n"
        "    --base_model '%s' \\\n"
        "    --lora_path '%s' \\\n"
        "    --output_gguf 'models/mongle-exaone4-q4km.gguf' \\\n"
        "    --quantize Q4_K_M",
        args.base_model,
        output_path,
    )


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="EXAONE 4.0 1.2B LoRA 파인튜닝 — 몽글이 페르소나 학습 (M-LLM-4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행
  python scripts/finetune_mongle.py \\
    --data_path data/finetune/mongle_train.jsonl \\
    --output_dir models/mongle-lora

  # 검증 데이터 포함
  python scripts/finetune_mongle.py \\
    --data_path data/finetune/mongle_train.jsonl \\
    --eval_path data/finetune/mongle_eval.jsonl \\
    --output_dir models/mongle-lora

  # CPU 환경 (느리지만 가능)
  python scripts/finetune_mongle.py \\
    --no_fp16 --batch_size 1 --gradient_accumulation_steps 16

데이터 형식 (JSONL):
  {"instruction": "시스템 프롬프트", "input": "사용자 메시지", "output": "어시스턴트 응답"}
        """,
    )

    # ─── 모델 설정 ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--base_model",
        type=str,
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="베이스 모델 HuggingFace ID (기본: LGAI-EXAONE/EXAONE-4.0-1.2B)",
    )

    # ─── 데이터 설정 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/finetune/mongle_train.jsonl",
        help="학습 데이터 JSONL 경로 (기본: data/finetune/mongle_train.jsonl)",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default=None,
        help="검증 데이터 JSONL 경로 (미지정 시 검증 스킵)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/mongle-lora",
        help="LoRA 어댑터 저장 디렉토리 (기본: models/mongle-lora)",
    )

    # ─── LoRA 하이퍼파라미터 ──────────────────────────────────────────────────
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA 랭크 r — 1.2B 모델에는 8이 최적 (기본: 8)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA 스케일 alpha (기본: 16, alpha/r=2.0 권장)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA 드롭아웃 (기본: 0.05)",
    )

    # ─── 학습 하이퍼파라미터 ──────────────────────────────────────────────────
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="학습 에폭 수 (기본: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="디바이스당 배치 크기 (기본: 8, T4에서 FP16 기준)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="그레이디언트 누적 스텝 (기본: 2, 실질 배치 = batch × steps)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="학습률 (기본: 2e-4, LoRA 학습 권장값)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="최대 시퀀스 길이 (기본: 2048, num_ctx와 맞춤)",
    )

    # ─── FP16/정밀도 설정 ────────────────────────────────────────────────────
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        default=False,
        dest="no_fp16",
        help="FP16 비활성화 (CPU 또는 MPS 환경에서 사용)",
    )

    args = parser.parse_args()
    # no_fp16 플래그를 fp16 플래그로 변환
    args.fp16 = not args.no_fp16
    return args


def main() -> None:
    """
    스크립트 진입점.

    실행 전 사전 조건:
      1. data/finetune/mongle_train.jsonl 존재
      2. transformers, peft, trl, accelerate 설치
      3. GPU 환경 권장 (CUDA T4 또는 A100)
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("몽글이 LoRA 파인튜닝 시작 (M-LLM-4)")
    logger.info("=" * 60)
    logger.info("베이스 모델: %s", args.base_model)
    logger.info("학습 데이터: %s", args.data_path)
    logger.info("출력 경로:   %s", args.output_dir)
    logger.info("에폭: %d | 배치: %d | LR: %.2e", args.num_epochs, args.batch_size, args.learning_rate)
    logger.info("LoRA r=%d, alpha=%d, dropout=%.2f", args.lora_r, args.lora_alpha, args.lora_dropout)
    logger.info("FP16: %s", args.fp16)
    logger.info("")

    try:
        run_finetune(args)
    except FileNotFoundError as e:
        logger.error("파일 없음: %s", e)
        sys.exit(1)
    except ImportError as e:
        logger.error("의존성 오류: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("파인튜닝 실패: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
