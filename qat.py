"""
Quantization-Aware Training (QAT).

Оборачивает Linear-слои в FakeQuantize: на каждом forward-проходе
веса проходят через round+clamp (имитация квантизации), но градиенты
текут через straight-through estimator.

Используется в пайплайне «quant_first»: модель сначала адаптируется
к шуму квантизации, а затем применяется прунинг.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Совместимость PyTorch ≥2.0 / <2.0
try:
    from torch.ao.quantization.fake_quantize import FakeQuantize
    from torch.ao.quantization.observer import MovingAverageMinMaxObserver
except ImportError:
    from torch.quantization import FakeQuantize
    from torch.quantization.observer import MovingAverageMinMaxObserver

SKIP = (
    "lm_head", "embed_tokens", "wte", "wpe",
    "word_embeddings", "rotary_emb",
    "layernorm", "layer_norm", "ln_", "norm",
)


class FakeQuantLinear(nn.Module):
    """Linear с fake-квантизацией весов."""

    def __init__(self, original: nn.Linear, bits: int = 8) -> None:
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.bias = original.bias

        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1

        self.fake_quant = FakeQuantize(
            observer=MovingAverageMinMaxObserver,
            quant_min=qmin, quant_max=qmax,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.fake_quant(self.weight), self.bias)


def _set_submodule(model: nn.Module, path: str, mod: nn.Module) -> None:
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], mod)


def inject_fake_quant(model: nn.Module, bits: int = 8) -> int:
    """Заменить все Linear → FakeQuantLinear. Вернуть кол-во заменённых."""
    count = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or not name:
            continue
        if any(p in name.lower() for p in SKIP):
            continue
        _set_submodule(model, name, FakeQuantLinear(mod, bits))
        count += 1
    logger.info("Fake-quant: обёрнуто %d слоёв (%d-bit)", count, bits)
    return count


def remove_fake_quant(model: nn.Module) -> int:
    """Заменить все FakeQuantLinear → обычные Linear."""
    count = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, FakeQuantLinear):
            continue
        lin = nn.Linear(
            mod.in_features, mod.out_features,
            bias=mod.bias is not None,
            device=mod.weight.device, dtype=mod.weight.dtype,
        )
        lin.weight = mod.weight
        if mod.bias is not None:
            lin.bias = mod.bias
        _set_submodule(model, name, lin)
        count += 1
    logger.info("Fake-quant: восстановлено %d слоёв", count)
    return count


def qat_train(
    model: nn.Module,
    dataloader: DataLoader,
    steps: int = 100,
    lr: float = 2e-5,
    device: str = "cpu",
) -> float:
    """
    QAT-обучение: модель учится компенсировать шум квантизации.
    Возвращает средний loss.
    """
    model.train()
    model.to(device)
    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    log_every = max(1, steps // 10)

    step = 0
    total_loss = 0.0
    logger.info("QAT обучение: %d шагов, lr=%.1e", steps, lr)

    while step < steps:
        for batch in dataloader:
            if step >= steps:
                break

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(
                input_ids=ids, attention_mask=mask, labels=labels,
            ).loss
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            step += 1

            if step % log_every == 0 or step == steps:
                logger.info(
                    "  QAT шаг %4d/%d │ loss=%.4f │ avg=%.4f",
                    step, steps, loss.item(), total_loss / step,
                )

    model.eval()
    return total_loss / max(step, 1)


def recovery_train(
    model: nn.Module,
    dataloader: DataLoader,
    steps: int = 100,
    lr: float = 2e-5,
    device: str = "cpu",
) -> float:
    """
    Recovery fine-tuning после прунинга.
    Идентичен QAT-обучению, но без fake-quant обёрток.
    """
    logger.info("Recovery fine-tuning: %d шагов", steps)
    return qat_train(model, dataloader, steps, lr, device)