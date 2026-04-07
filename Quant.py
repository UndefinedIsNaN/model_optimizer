#!/usr/bin/env python3
"""
Квантизация модели → GGUF (без прунинга)

Поддерживает PTQ (быстро, без обучения) и QAT (с адаптацией)

Примеры:
  # PTQ - быстрая квантизация
  python quant_only.py --repo Qwen/Qwen2-0.5B --quant-method ptq --quant-type Q4_K_M
  
  # QAT - с адаптацией модели
  python quant_only.py --repo Qwen/Qwen2-0.5B --quant-method qat --quant-type Q4_K_M --qat-steps 100
  
  # CPU-версия
  python quant_only.py --repo Qwen/Qwen2-0.5B --quant-method ptq --quant-type Q4_K_M --device cpu
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from qat import inject_fake_quant, remove_fake_quant, qat_train
from gguf_backend import GGUFBackend

logger = logging.getLogger(__name__)


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_dataloader(
    tokenizer,
    dataset: str = "wikitext",
    config: str = "wikitext-2-raw-v1",
    split: str = "test",
    n_samples: int = 256,
    max_len: int = 512,
    batch_size: int = 2,
) -> DataLoader:
    """Создать DataLoader для QAT."""
    raw = load_dataset(dataset, config, split=split)

    text_col = next(
        (c for c in ("text", "content", "sentence") if c in raw.column_names),
        raw.column_names[0],
    )
    raw = raw.filter(lambda r: len(str(r[text_col]).strip()) > 50)
    raw = raw.select(range(min(len(raw), n_samples)))

    def tokenize(batch):
        out = tokenizer(
            batch[text_col],
            truncation=True, max_length=max_len, padding="max_length",
        )
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    ds = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    ds.set_format("torch")
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def pipeline_ptq(model, tokenizer, args, backend, work_dir, output_path):
    """PTQ: сохранение → GGUF fp16 → llama-quantize"""
    logger.info("━" * 50)
    logger.info("  ПАЙПЛАЙН: PTQ (Post-Training Quantization)")
    logger.info("━" * 50)
    
    model.to("cpu")
    
    hf_dir = work_dir / "hf_temp"
    if hf_dir.exists():
        shutil.rmtree(str(hf_dir))

    logger.info("Сохраняем HF checkpoint …")
    model.save_pretrained(str(hf_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(hf_dir))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tmp_fp16 = work_dir / "temp_fp16.gguf"

    try:
        logger.info("Конвертация в GGUF fp16 …")
        backend.convert_hf_to_gguf(hf_dir, tmp_fp16, dtype="f16")

        if args.quant_type in ("F16", "F32"):
            shutil.move(str(tmp_fp16), str(output_path))
            logger.info("Сохранён как F16/F32 (без квантизации)")
        else:
            logger.info("Квантизация %s через llama.cpp …", args.quant_type)
            backend.quantize(tmp_fp16, output_path, args.quant_type)
    finally:
        if tmp_fp16.exists():
            tmp_fp16.unlink(missing_ok=True)
        shutil.rmtree(str(hf_dir), ignore_errors=True)


def pipeline_qat(model, tokenizer, args, backend, work_dir, output_path):
    """QAT: адаптация → сохранение → GGUF fp16 → llama-quantize"""
    device = args.device if args.device else best_device()
    
    logger.info("━" * 50)
    logger.info("  ПАЙПЛАЙН: QAT (Quantization-Aware Training)")
    logger.info("━" * 50)
    
    model.to(device)
    model.eval()

    # Подготовка данных
    logger.info("Подготовка калибровочных данных …")
    dataloader = make_dataloader(
        tokenizer,
        dataset=args.cal_dataset,
        config=args.cal_config,
        n_samples=args.cal_samples,
        max_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    # QAT обучение
    logger.info("QAT: %d шагов, %d-bit …", args.qat_steps, args.qat_bits)
    inject_fake_quant(model, bits=args.qat_bits)
    qat_train(model, dataloader, steps=args.qat_steps, lr=args.lr, device=device)
    remove_fake_quant(model)

    # Конвертация в GGUF
    logger.info("Конвертация в GGUF …")
    pipeline_ptq(model, tokenizer, args, backend, work_dir, output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    QUANT_TYPES = [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
        "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
        "Q6_K", "Q8_0", "F16", "F32",
    ]

    p = argparse.ArgumentParser(
        description="Квантизация модели → GGUF (без прунинга)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # PTQ - быстро, без обучения
  python quant_only.py --repo Qwen/Qwen2-0.5B --quant-method ptq --quant-type Q4_K_M
  
  # QAT - с адаптацией (лучше качество)
  python quant_only.py --repo Qwen/Qwen2-0.5B --quant-method qat --quant-type Q4_K_M --qat-steps 100
  
  # F16 (без квантизации, только конвертация)
  python quant_only.py --repo Qwen/Qwen2-0.5B --quant-method ptq --quant-type F16
  
  # CPU-only
  python quant_only.py --repo Qwen/Qwen2-0.5B --quant-method ptq --quant-type Q4_K_M --device cpu

Типы квантизации:
  """ + ", ".join(QUANT_TYPES),
    )

    # Обязательные
    p.add_argument("--repo", required=True, help="HuggingFace repo ID или путь к локальной модели")
    p.add_argument("--quant-method", required=True, choices=["ptq", "qat"], help="ptq | qat")
    p.add_argument("--quant-type", required=True, choices=QUANT_TYPES, help="Тип GGUF-квантизации")

    # QAT параметры
    p.add_argument("--qat-steps", type=int, default=100, help="Шагов QAT (default: 100)")
    p.add_argument("--qat-bits", type=int, default=8, choices=[4, 8], help="Битность fake-quant (default: 8)")

    # Обучение
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=512)

    # Данные
    p.add_argument("--cal-dataset", default="wikitext")
    p.add_argument("--cal-config", default="wikitext-2-raw-v1")
    p.add_argument("--cal-samples", type=int, default=256)

    # Пути и устройство
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--work-dir", default="./work")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--device", default=None, help="cuda | cpu | mps (auto если не указано)")

    args = p.parse_args()

    # Определение устройства
    if not args.device:
        args.device = best_device()

    # ── Заголовок ──
    logger.info("Квантизация модели → GGUF")
    logger.info("  repo         : %s", args.repo)
    logger.info("  quant_method : %s", args.quant_method.upper())
    logger.info("  quant_type   : %s", args.quant_type)
    logger.info("  device       : %s", args.device)
    if args.quant_method == "qat":
        logger.info("  qat_steps    : %d", args.qat_steps)
        logger.info("  qat_bits     : %d", args.qat_bits)
    logger.info("")

    # ── Пути ──
    model_name = Path(args.repo).name if Path(args.repo).exists() else args.repo.rsplit("/", 1)[-1]
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_name}_{args.quant_method}_{args.quant_type}.gguf"

    # ── Загрузка модели ──
    if Path(args.repo).exists():
        # Локальная модель
        model_dir = Path(args.repo)
        logger.info("Загружаем локальную модель: %s", model_dir)
    else:
        # HuggingFace Hub
        model_dir = work_dir / "models" / model_name
        if not model_dir.exists() or not any(model_dir.iterdir()):
            logger.info("Скачиваем %s …", args.repo)
            snapshot_download(repo_id=args.repo, local_dir=str(model_dir), token=args.hf_token)
        else:
            logger.info("Модель уже скачана: %s", model_dir)

    logger.info("Загрузка весов …")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), token=args.hf_token, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        token=args.hf_token,
        trust_remote_code=True,
    )

    # ── Запуск пайплайна ──
    backend = GGUFBackend(work_dir)

    if args.quant_method == "ptq":
        pipeline_ptq(model, tokenizer, args, backend, work_dir, output_path)
    else:
        pipeline_qat(model, tokenizer, args, backend, work_dir, output_path)

    # ── Результат ──
    size_mb = output_path.stat().st_size / (1024 ** 2)
    logger.info("")
    logger.info(" Готово!")
    logger.info("    Файл   : %s", output_path)
    logger.info("    Размер : %.1f MB", size_mb)
    logger.info("    Метод  : %s (%s)", args.quant_method.upper(), args.quant_type)


if __name__ == "__main__":
    main()