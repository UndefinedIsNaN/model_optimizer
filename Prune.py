#!/usr/bin/env python3
"""
Pruning Only → GGUF — сохраняет обрезанную модель в GGUF формате (fp16 по умолчанию)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch

# Отключаем torchvision ДО импорта transformers
sys.modules['torchvision'] = None

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

try:
    from pruners import MagnitudePruner, StructuredPruner
    from qat import recovery_train
    from gguf_backend import GGUFBackend
except ImportError as e:
    logging.error(f"Не найдены модули: {e}")
    raise

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
    config: str | None = "wikitext-2-raw-v1",
    split: str = "test",
    n_samples: int = 256,
    max_len: int = 512,
    batch_size: int = 2,
):
    """Создать DataLoader для recovery."""
    if config:
        raw = load_dataset(dataset, config, split=split)
    else:
        raw = load_dataset(dataset, split=split)

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


def make_pruner(name: str):
    """Фабрика пруннеров."""
    if name == "magnitude_20":
        return MagnitudePruner(sparsity=0.20)
    elif name == "magnitude_30":
        return MagnitudePruner(sparsity=0.30)
    elif name == "structured":
        return StructuredPruner(ffn_sparsity=0.25, head_sparsity=0.25)
    else:
        raise ValueError(f"Неизвестный прунинг: {name}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Определяем режим квантизации
    model_name = args.repo.rsplit("/", 1)[-1]
    
    # Проверяем нужна ли квантизация
    if args.no_quant or not args.quant_type or args.quant_type.upper() == "F16":
        quant_suffix = "F16"
        do_quantize = False
    elif args.quant_type.upper() == "F32":
        quant_suffix = "F32"
        do_quantize = False
    else:
        quant_suffix = args.quant_type
        do_quantize = True

    logger.info("=" * 55)
    logger.info("  PRUNING → GGUF")
    logger.info("=" * 55)
    logger.info("  repo           : %s", args.repo)
    logger.info("  pruning        : %s", args.pruning)
    logger.info("  recovery_steps : %d", args.recovery_steps)
    logger.info("  квантизация    : %s", "ДА (%s)" % quant_suffix if do_quantize else "НЕТ (%s)" % quant_suffix)
    logger.info("  device         : %s", best_device())
    logger.info("")

    # Пути
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Имя файла
    out_name = f"{model_name}_pruned_{args.pruning}_{quant_suffix}.gguf"
    gguf_path = output_dir / out_name
    hf_temp_path = work_dir / "hf_temp" / f"{model_name}_pruned_{args.pruning}"

    # Скачать модель
    model_dir = work_dir / "models" / model_name
    if not model_dir.exists() or not any(model_dir.iterdir()):
        logger.info("Скачиваем %s …", args.repo)
        snapshot_download(
            repo_id=args.repo, 
            local_dir=str(model_dir), 
            token=args.hf_token,
        )
    else:
        logger.info("Модель уже скачана: %s", model_dir)

    # Загрузить модель
    logger.info("Загружаем модель …")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), 
        token=args.hf_token, 
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        token=args.hf_token,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    device = best_device()
    model.to(device)
    model.eval()

    # Прунинг
    pruner = make_pruner(args.pruning)
    
    logger.info("")
    logger.info("▶ Шаг 1: Прунинг (%s)", args.pruning)
    pruner.prune(model)

    # Recovery
    if args.recovery_steps > 0:
        logger.info("")
        logger.info("▶ Шаг 2: Recovery fine-tuning (%d шагов)", args.recovery_steps)
        dataloader = make_dataloader(
            tokenizer,
            dataset=args.cal_dataset, 
            config=args.cal_config,
            n_samples=args.cal_samples, 
            max_len=args.max_seq_len,
            batch_size=args.batch_size,
        )
        recovery_train(
            model, dataloader,
            steps=args.recovery_steps, 
            lr=args.lr, 
            device=device,
        )

    # Сохранение в HF формат (временно)
    logger.info("")
    logger.info("▶ Шаг 3: Сохранение в HF формате …")
    
    if hf_temp_path.exists():
        shutil.rmtree(str(hf_temp_path))
    
    model.to("cpu")
    model.save_pretrained(str(hf_temp_path), safe_serialization=True)
    tokenizer.save_pretrained(str(hf_temp_path))
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Конвертация в GGUF
    logger.info("")
    logger.info("▶ Шаг 4: Конвертация в GGUF …")
    
    try:
        backend = GGUFBackend(work_dir)
        
        if do_quantize:
            # HF → GGUF fp16 → Квантизация
            logger.info("   Конвертация HF → F16 …")
            temp_fp16 = work_dir / "temp_fp16.gguf"
            backend.convert_hf_to_gguf(hf_temp_path, temp_fp16, dtype="f16")
            
            logger.info("   Квантизация F16 → %s …", quant_suffix)
            backend.quantize(temp_fp16, gguf_path, quant_suffix)
            temp_fp16.unlink(missing_ok=True)
        else:
            # HF → GGUF F16/F32 (без квантизации)
            dtype = "f16" if quant_suffix == "F16" else "f32"
            logger.info("   Конвертация HF → %s (без квантизации) …", quant_suffix)
            backend.convert_hf_to_gguf(hf_temp_path, gguf_path, dtype=dtype)
        
        success = True
        
    except Exception as e:
        logger.error("Ошибка при конвертации в GGUF: %s", e)
        import traceback
        logger.debug(traceback.format_exc())
        success = False
    
    # Удаляем временные HF файлы
    if hf_temp_path.exists():
        shutil.rmtree(str(hf_temp_path))

    if success:
        size_mb = gguf_path.stat().st_size / (1024 ** 2)
        logger.info("")
        logger.info("=" * 55)
        logger.info("  ГОТОВО!")
        logger.info("  Файл:   %s", gguf_path)
        logger.info("  Размер: %.1f MB", size_mb)
        logger.info("=" * 55)
    else:
        logger.error("Не удалось создать GGUF файл")
        return 1

    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Pruning → GGUF")

    p.add_argument("--repo", required=True, help="HuggingFace repo ID")
    p.add_argument("--pruning", required=True, choices=["magnitude_20", "magnitude_30", "structured"])
    p.add_argument("--recovery-steps", type=int, default=0)
    
    # Квантизация: по умолчанию ВЫКЛЮЧЕНА (F16)
    g = p.add_argument_group("Квантизация (опционально)")
    g.add_argument("--quant-type", default=None,
                   choices=["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", 
                           "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
                           "Q6_K", "Q8_0", "F16", "F32"],
                   help="Тип квантизации (по умолчанию: F16 без квантизации)")
    g.add_argument("--no-quant", action="store_true",
                   help="Явно отключить квантизацию (F16)")
    
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--cal-dataset", default="wikitext")
    p.add_argument("--cal-config", default="wikitext-2-raw-v1")
    p.add_argument("--cal-samples", type=int, default=256)
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--work-dir", default="./work")
    p.add_argument("--hf-token", default=None)

    return p.parse_args()


if __name__ == "__main__":
    exit(main())