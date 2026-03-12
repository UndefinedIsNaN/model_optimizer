#!/usr/bin/env python3
"""

  Прунинг + Квантизация → GGUF                                  
                                                                
  Прунинг:                                                      
    magnitude_20  — обнуление 20% мелких весов                  
    magnitude_30  — обнуление 30% мелких весов                  
    structured    — физическое удаление нейронов и голов        
                                                                
  Квантизация:                                                  
    ptq  — Post-Training: llama-quantize (быстро, без обучения) 
    qat  — Quantization-Aware Training (обучение, лучше качество)
                                                                  
  Порядок:                                                       
    prune_first   — прунинг → квантизация                        
    quant_first   — квантизация → прунинг (только QAT)           


Примеры:

  # Прунинг → PTQ (самый быстрый):
  python main.py --repo Qwen/Qwen2-0.5B \\
      --pruning magnitude_20 --quant-type Q4_K_M \\
      --quant-method ptq --order prune_first

  # Прунинг → QAT (лучше качество):
  python main.py --repo Qwen/Qwen2-0.5B \\
      --pruning structured --quant-type Q4_K_M \\
      --quant-method qat --order prune_first --qat-steps 150

  # QAT → Прунинг (модель сначала адаптируется к квантизации):
  python main.py --repo Qwen/Qwen2-0.5B \\
      --pruning structured --quant-type Q4_K_M \\
      --quant-method qat --order quant_first --qat-steps 200

  # Все 10 комбинаций:
  for P in magnitude_20 magnitude_30 structured; do
    for Q in ptq qat; do
      for O in prune_first quant_first; do
        python main.py --repo Qwen/Qwen2-0.5B \\
            --pruning $P --quant-type Q4_K_M \\
            --quant-method $Q --order $O --qat-steps 100
      done
    done
  done
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from pruners import MagnitudePruner, StructuredPruner
from qat import inject_fake_quant, remove_fake_quant, qat_train, recovery_train
from gguf_backend import GGUFBackend

logger = logging.getLogger(__name__)

QUANT_TYPES = [
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0", "F16", "F32",
]


#  Утилиты

def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_dataloader(
    tokenizer,
    dataset: str = "wikitext",
    config: Optional[str] = "wikitext-2-raw-v1",
    split: str = "test",
    n_samples: int = 256,
    max_len: int = 512,
    batch_size: int = 2,
) -> DataLoader:
    """Создать DataLoader для QAT / recovery."""
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


def make_output_name(
    model_name: str,
    pruning: str,
    quant_method: str,
    quant_type: str,
    order: str,
) -> str:
    """
    Формирует имя выходного файла:
      Qwen2-0.5B_prune-first_magnitude_20_ptq_Q4_K_M.gguf
      Qwen2-0.5B_quant-first_structured_qat_Q4_K_M.gguf
    """
    return f"{model_name}_{order}_{pruning}_{quant_method}_{quant_type}.gguf"


def get_dataloader_if_needed(args, tokenizer) -> Optional[DataLoader]:
    """Создать DataLoader только если он нужен для выбранного пайплайна."""
    need_data = (
        args.quant_method == "qat"
        or args.recovery_steps > 0
    )
    if not need_data:
        return None

    logger.info("Готовим калибровочные данные …")
    return make_dataloader(
        tokenizer,
        dataset=args.cal_dataset,
        config=args.cal_config,
        n_samples=args.cal_samples,
        max_len=args.max_seq_len,
        batch_size=args.batch_size,
    )


#  Пайплайн 1: prune_first + ptq
#
#  Модель → Прунинг → [Recovery] → GGUF fp16 → llama-quantize
#
#  Самый простой и быстрый. Не требует GPU для квантизации.

def pipeline_prune_first_ptq(
    model, tokenizer, pruner, args, backend, work_dir, output_path,
):
    device = best_device()
    model.to(device)
    model.eval()

    logger.info("━" * 55)
    logger.info("  ПАЙПЛАЙН: prune_first + PTQ")
    logger.info("━" * 55)

    # ── Шаг 1: Прунинг ──
    logger.info("")
    logger.info("  ▶ Шаг 1: Прунинг (%s)", args.pruning)
    pruner.prune(model)

    # ── Шаг 2: Recovery (опционально) ──
    if args.recovery_steps > 0:
        logger.info("")
        logger.info("  ▶ Шаг 2: Recovery fine-tuning (%d шагов)", args.recovery_steps)
        dataloader = make_dataloader(
            tokenizer,
            dataset=args.cal_dataset, config=args.cal_config,
            n_samples=args.cal_samples, max_len=args.max_seq_len,
            batch_size=args.batch_size,
        )
        recovery_train(
            model, dataloader,
            steps=args.recovery_steps, lr=args.lr, device=device,
        )

    # ── Шаг 3: PTQ (сохранение + llama-quantize) ──
    logger.info("")
    logger.info("  ▶ Шаг 3: PTQ → GGUF (%s)", args.quant_type)
    _save_and_convert(
        model, tokenizer, args.quant_type, backend, work_dir, output_path,
    )


#  Пайплайн 2: prune_first + qat
#
#  Модель → Прунинг → [Recovery] → QAT → GGUF fp16 → llama-quantize
#
#  Прунинг упрощает модель, затем QAT адаптирует оставшиеся
#  веса к будущей квантизации. Лучше качество чем чистый PTQ.

def pipeline_prune_first_qat(
    model, tokenizer, pruner, args, backend, work_dir, output_path,
):
    device = best_device()
    model.to(device)
    model.eval()

    dataloader = make_dataloader(
        tokenizer,
        dataset=args.cal_dataset, config=args.cal_config,
        n_samples=args.cal_samples, max_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    logger.info("━" * 55)
    logger.info("  ПАЙПЛАЙН: prune_first + QAT")
    logger.info("━" * 55)

    # ── Шаг 1: Прунинг ──
    logger.info("")
    logger.info("  ▶ Шаг 1: Прунинг (%s)", args.pruning)
    pruner.prune(model)

    # ── Шаг 2: Recovery (опционально) ──
    if args.recovery_steps > 0:
        logger.info("")
        logger.info("  ▶ Шаг 2: Recovery fine-tuning (%d шагов)", args.recovery_steps)
        recovery_train(
            model, dataloader,
            steps=args.recovery_steps, lr=args.lr, device=device,
        )

    # ── Шаг 3: QAT ──
    logger.info("")
    logger.info("  ▶ Шаг 3: QAT (%d шагов, %d-bit)", args.qat_steps, args.qat_bits)
    inject_fake_quant(model, bits=args.qat_bits)
    qat_train(
        model, dataloader,
        steps=args.qat_steps, lr=args.lr, device=device,
    )
    remove_fake_quant(model)

    # ── Шаг 4: GGUF ──
    logger.info("")
    logger.info("  ▶ Шаг 4: Сохранение → GGUF (%s)", args.quant_type)
    _save_and_convert(
        model, tokenizer, args.quant_type, backend, work_dir, output_path,
    )


#  Пайплайн 3: quant_first + qat
#
#  Модель → QAT → Прунинг → [Recovery] → GGUF fp16 → llama-quantize
#
#  Модель сначала адаптируется к шуму квантизации (QAT),
#  затем прунинг удаляет наименее важные части из уже
#  адаптированной модели. Часто даёт лучший результат.

def pipeline_quant_first_qat(
    model, tokenizer, pruner, args, backend, work_dir, output_path,
):
    device = best_device()

    dataloader = make_dataloader(
        tokenizer,
        dataset=args.cal_dataset, config=args.cal_config,
        n_samples=args.cal_samples, max_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    logger.info("━" * 55)
    logger.info("  ПАЙПЛАЙН: quant_first + QAT")
    logger.info("━" * 55)

    # ── Шаг 1: QAT ──
    logger.info("")
    logger.info("  ▶ Шаг 1: QAT (%d шагов, %d-bit)", args.qat_steps, args.qat_bits)
    inject_fake_quant(model, bits=args.qat_bits)
    qat_train(
        model, dataloader,
        steps=args.qat_steps, lr=args.lr, device=device,
    )
    remove_fake_quant(model)

    # ── Шаг 2: Прунинг ──
    logger.info("")
    logger.info("  ▶ Шаг 2: Прунинг (%s)", args.pruning)
    model.eval()
    pruner.prune(model)

    # ── Шаг 3: Recovery (опционально) ──
    if args.recovery_steps > 0:
        logger.info("")
        logger.info("  ▶ Шаг 3: Recovery fine-tuning (%d шагов)", args.recovery_steps)
        recovery_train(
            model, dataloader,
            steps=args.recovery_steps, lr=args.lr, device=device,
        )

    # ── Шаг 4: GGUF ──
    logger.info("")
    logger.info("  ▶ Шаг 4: Сохранение → GGUF (%s)", args.quant_type)
    _save_and_convert(
        model, tokenizer, args.quant_type, backend, work_dir, output_path,
    )


#  Пайплайн 4: quant_first + ptq (fallback)
#
#  PTQ не модифицирует веса модели — это просто конвертация
#  в другой числовой формат. Поэтому «сначала PTQ, потом прунинг»
#  бессмысленно: PTQ применяется на уровне GGUF-файла, а прунинг
#  работает с PyTorch-тензорами.
#
#  Автоматически переключаемся на prune_first + ptq.

def pipeline_quant_first_ptq(
    model, tokenizer, pruner, args, backend, work_dir, output_path,
):
    logger.warning("━" * 55)
    logger.warning("  ⚠  quant_first + PTQ не имеет смысла!")
    logger.warning("")
    logger.warning("  PTQ (Post-Training Quantization) — это одноразовая")
    logger.warning("  конвертация весов, она не меняет модель в PyTorch.")
    logger.warning("")
    logger.warning("  → Переключение на prune_first + PTQ")
    logger.warning("━" * 55)

    pipeline_prune_first_ptq(
        model, tokenizer, pruner, args, backend, work_dir, output_path,
    )


#  Диспетчер пайплайнов

PIPELINES = {
    ("prune_first", "ptq"): pipeline_prune_first_ptq,
    ("prune_first", "qat"): pipeline_prune_first_qat,
    ("quant_first", "qat"): pipeline_quant_first_qat,
    ("quant_first", "ptq"): pipeline_quant_first_ptq,
}


# ═══════════════════════════════════════════════════════
#  Общий финал: сохранение → GGUF
# ═══════════════════════════════════════════════════════

def _save_and_convert(
    model, tokenizer, quant_type, backend, work_dir, output_path,
):
    """Сохранить HF → конвертировать GGUF fp16 → квантизировать."""
    model.to("cpu")

    hf_dir = work_dir / "hf_temp"
    if hf_dir.exists():
        shutil.rmtree(str(hf_dir))

    logger.info("Сохраняем HF checkpoint → %s", hf_dir)
    model.save_pretrained(str(hf_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(hf_dir))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tmp_fp16 = work_dir / "temp_fp16.gguf"

    try:
        backend.convert_hf_to_gguf(hf_dir, tmp_fp16, dtype="f16")

        if quant_type in ("F16", "F32"):
            shutil.move(str(tmp_fp16), str(output_path))
        else:
            backend.quantize(tmp_fp16, output_path, quant_type)
            tmp_fp16.unlink(missing_ok=True)
    finally:
        if tmp_fp16.exists():
            tmp_fp16.unlink(missing_ok=True)
        shutil.rmtree(str(hf_dir), ignore_errors=True)


#  Точка входа

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # ── Заголовок ──
    logger.info("Pruning + Quantization → GGUF")
    logger.info("  repo         : %s", args.repo)
    logger.info("  pruning      : %s", args.pruning)
    logger.info("  quant_method : %s", args.quant_method.upper())
    logger.info("  quant_type   : %s", args.quant_type)
    logger.info("  order        : %s", args.order)
    logger.info("  device       : %s", best_device())

    if args.quant_method == "qat":
        logger.info("  qat_steps    : %d", args.qat_steps)
        logger.info("  qat_bits     : %d", args.qat_bits)
    if args.recovery_steps > 0:
        logger.info("  recovery     : %d шагов", args.recovery_steps)
    logger.info("")

    # ── Пути ──
    model_name = args.repo.rsplit("/", 1)[-1]
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_name = make_output_name(
        model_name, args.pruning, args.quant_method,
        args.quant_type, args.order,
    )
    output_path = output_dir / out_name

    # ── Скачать модель ──
    model_dir = work_dir / "models" / model_name
    if not model_dir.exists() or not any(model_dir.iterdir()):
        logger.info("Скачиваем %s …", args.repo)
        snapshot_download(
            repo_id=args.repo, local_dir=str(model_dir), token=args.hf_token,
        )
    else:
        logger.info("Модель уже скачана: %s", model_dir)

    # ── Загрузить модель ──
    logger.info("Загружаем модель (fp32) …")
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

    # ── Создать пруннер и бэкенд ──
    pruner = make_pruner(args.pruning)
    backend = GGUFBackend(work_dir)

    # ── Выбрать и запустить пайплайн ──
    pipeline_fn = PIPELINES[(args.order, args.quant_method)]

    pipeline_fn(
        model, tokenizer, pruner, args,
        backend, work_dir, output_path,
    )

    # ── Результат ──
    size_mb = output_path.stat().st_size / (1024 ** 2)
    logger.info("")
    logger.info(" Готово!")
    logger.info("    Файл         : %s", output_path)
    logger.info("    Размер       : %.1f MB", size_mb)
    logger.info("    Порядок      : %s", args.order)
    logger.info("    Прунинг      : %s", args.pruning)
    logger.info("    Квантизация  : %s (%s)", args.quant_method.upper(), args.quant_type)


#  CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Прунинг + Квантизация → GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔═══════════════════════════════════════════════════════════════════════╗
║  Комбинации order × quant_method                                     ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  prune_first + ptq : Prune → [Recovery] → GGUF quantize              ║
║                      Быстро. Не нужен GPU для квантизации.           ║
║                                                                       ║
║  prune_first + qat : Prune → [Recovery] → QAT → GGUF quantize       ║
║                      Модель учится компенсировать шум после прунинга.║
║                                                                       ║
║  quant_first + qat : QAT → Prune → [Recovery] → GGUF quantize       ║
║                      Модель адаптируется к квантизации ДО прунинга.  ║
║                      Часто лучшее качество.                          ║
║                                                                       ║
║  quant_first + ptq : ⚠ Автоматически → prune_first + ptq            ║
║                      PTQ не меняет модель, порядок не важен.         ║
╚═══════════════════════════════════════════════════════════════════════╝

Примеры:

  # Самый быстрый (прунинг + PTQ):
  python main.py --repo Qwen/Qwen2-0.5B \\
      --pruning magnitude_20 --quant-method ptq \\
      --quant-type Q4_K_M --order prune_first

  # Лучшее качество (QAT → прунинг):
  python main.py --repo Qwen/Qwen2-0.5B \\
      --pruning structured --quant-method qat \\
      --quant-type Q4_K_M --order quant_first \\
      --qat-steps 200 --recovery-steps 100

  # Полное сравнение:
  for P in magnitude_20 magnitude_30 structured; do
    for Q in ptq qat; do
      for O in prune_first quant_first; do
        python main.py --repo Qwen/Qwen2-0.5B \\
            --pruning $P --quant-method $Q \\
            --quant-type Q4_K_M --order $O \\
            --qat-steps 100 --recovery-steps 50
      done
    done
  done

Типы квантизации:
  """ + ", ".join(QUANT_TYPES),
    )

    # ── Обязательные ──
    p.add_argument(
        "--repo", required=True,
        help="HuggingFace repo ID (напр. Qwen/Qwen2-0.5B)",
    )
    p.add_argument(
        "--pruning", required=True,
        choices=["magnitude_20", "magnitude_30", "structured"],
        help="Вид прунинга",
    )
    p.add_argument(
        "--quant-method", required=True,
        choices=["ptq", "qat"],
        help="Метод квантизации: ptq (быстро) | qat (обучение)",
    )
    p.add_argument(
        "--quant-type", required=True,
        choices=QUANT_TYPES,
        help="Тип GGUF-квантизации (Q4_K_M, Q5_K_M, ...)",
    )
    p.add_argument(
        "--order", required=True,
        choices=["prune_first", "quant_first"],
        help="Порядок: prune_first | quant_first",
    )

    # ── QAT ──
    g = p.add_argument_group("QAT (для quant-method=qat)")
    g.add_argument("--qat-steps", type=int, default=100,
                    help="Шагов QAT-обучения (по умолчанию: 100)")
    g.add_argument("--qat-bits", type=int, default=8, choices=[4, 8],
                    help="Битность fake-quant (по умолчанию: 8)")

    # ── Recovery ──
    g2 = p.add_argument_group("Recovery fine-tuning (после прунинга)")
    g2.add_argument("--recovery-steps", type=int, default=0,
                     help="Шагов recovery (0 = отключено)")

    # ── Обучение ──
    g3 = p.add_argument_group("Параметры обучения")
    g3.add_argument("--lr", type=float, default=2e-5)
    g3.add_argument("--batch-size", type=int, default=2)
    g3.add_argument("--max-seq-len", type=int, default=512)

    # ── Данные ──
    g4 = p.add_argument_group("Калибровочные данные")
    g4.add_argument("--cal-dataset", default="wikitext")
    g4.add_argument("--cal-config", default="wikitext-2-raw-v1")
    g4.add_argument("--cal-samples", type=int, default=256)

    # ── Пути ──
    g5 = p.add_argument_group("Пути")
    g5.add_argument("--output-dir", default="./output")
    g5.add_argument("--work-dir", default="./work")
    g5.add_argument("--hf-token", default=None,
                     help="HuggingFace токен (для gated-моделей)")

    return p.parse_args()


if __name__ == "__main__":
    main()