"""
Два метода прунинга:

1. MagnitudePruner — обнуляет веса с наименьшим |w|.
   GGUF хранит плотные тензоры — размер файла не уменьшится.
   Полезно для экспериментов и сравнения.

2. StructuredPruner — ФИЗИЧЕСКИ удаляет нейроны FFN и attention heads.
   Матрицы становятся меньше → файл меньше → inference быстрее.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Слои, которые НИКОГДА не трогаем
SKIP_PATTERNS = (
    "lm_head", "embed_tokens", "embed_positions",
    "wte", "wpe", "word_embeddings",
    "layernorm", "layer_norm", "ln_", "norm",
    "rotary_emb",
)


@dataclass
class PruneReport:
    """Отчёт о прунинге."""
    method: str = ""
    params_before: int = 0
    params_after: int = 0
    details: Dict[str, str] = field(default_factory=dict)

    @property
    def reduction_pct(self) -> float:
        if self.params_before == 0:
            return 0.0
        return (1 - self.params_after / self.params_before) * 100

    def log(self) -> None:
        logger.info("─" * 55)
        logger.info("  Прунинг: %s", self.method)
        logger.info("  Параметры: %s → %s (−%.1f%%)",
                     f"{self.params_before:,}",
                     f"{self.params_after:,}",
                     self.reduction_pct)
        for k, v in self.details.items():
            logger.info("  %s: %s", k, v)
        logger.info("─" * 55)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _should_skip(name: str) -> bool:
    name_low = name.lower()
    return any(pat in name_low for pat in SKIP_PATTERNS)


#  Magnitude Pruner

class MagnitudePruner:
    """
    Обнуляет sparsity% весов с наименьшим |w| в каждом Linear-слое.

    Матрицы остаются того же размера — нули просто стоят внутри.
    На ARM это НЕ даёт ускорения, но позволяет исследовать
    влияние разреженности на качество модели.
    """

    def __init__(self, sparsity: float) -> None:
        assert 0 < sparsity < 1, f"sparsity должен быть в (0, 1), получено {sparsity}"
        self.sparsity = sparsity

    def prune(self, model: nn.Module) -> PruneReport:
        report = PruneReport(
            method=f"magnitude_{int(self.sparsity * 100)}%",
            params_before=_count_params(model),
        )

        n_layers = 0
        total_zeroed = 0
        total_weights = 0

        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear) or _should_skip(name):
                continue

            w = mod.weight.data
            magnitudes = w.abs()
            threshold = torch.quantile(
                magnitudes.float().flatten(), self.sparsity
            )
            mask = magnitudes >= threshold
            mod.weight.data = w * mask

            zeroed = (~mask).sum().item()
            total_zeroed += zeroed
            total_weights += w.numel()
            n_layers += 1

            logger.debug(
                "  %s: %d / %d обнулено (%.1f%%)",
                name, zeroed, w.numel(), 100.0 * zeroed / w.numel(),
            )

        report.params_after = _count_params(model)
        report.details["layers_processed"] = str(n_layers)
        report.details["weights_zeroed"] = f"{total_zeroed:,} / {total_weights:,}"
        report.details["actual_sparsity"] = (
            f"{100 * total_zeroed / max(total_weights, 1):.1f}%"
        )
        report.details["⚠ WARNING"] = (
            "Magnitude pruning НЕ уменьшает GGUF-файл и НЕ ускоряет inference"
        )
        report.log()
        return report


#  Structured Pruner

class StructuredPruner:
    """
    Физически удаляет нейроны FFN и attention heads.

    FFN (SwiGLU / стандартный):
      gate_proj: [intermediate, hidden]  ─┐
      up_proj:   [intermediate, hidden]   ├→ down_proj: [hidden, intermediate]
      Удаляем строки gate/up + столбцы down → intermediate уменьшается

    Attention:
      q_proj: [n_heads * head_dim, hidden]
      k_proj: [n_kv_heads * head_dim, hidden]
      v_proj: [n_kv_heads * head_dim, hidden]
      o_proj: [hidden, n_heads * head_dim]
      Удаляем головы → все проекции уменьшаются

    Результат: матрицы МЕНЬШЕ → GGUF файл меньше → быстрее на ARM.
    """

    def __init__(
        self,
        ffn_sparsity: float = 0.25,
        head_sparsity: float = 0.25,
    ) -> None:
        self.ffn_sparsity = ffn_sparsity
        self.head_sparsity = head_sparsity

    def prune(self, model: nn.Module) -> PruneReport:
        report = PruneReport(
            method="structured",
            params_before=_count_params(model),
        )

        # ── 1. FFN Neuron Pruning ──
        ffn_info = self._prune_ffn(model)
        report.details.update(ffn_info)

        # ── 2. Attention Head Pruning ──
        head_info = self._prune_heads(model)
        report.details.update(head_info)

        report.params_after = _count_params(model)
        report.log()
        return report

    #  FFN Pruning

    def _prune_ffn(self, model: nn.Module) -> Dict[str, str]:
        """Удалить нейроны FFN из всех слоёв."""
        triplets = self._find_ffn_triplets(model)
        if not triplets:
            logger.warning("FFN-слои не найдены!")
            return {"ffn": "не найдены"}

        original_size = triplets[0]["ups"][0].out_features
        n_remove = int(original_size * self.ffn_sparsity)
        n_keep = original_size - n_remove

        if n_remove == 0:
            return {"ffn": "нечего удалять"}

        logger.info(
            "FFN: %d → %d нейронов в каждом из %d блоков (−%d)",
            original_size, n_keep, len(triplets), n_remove,
        )

        for block in triplets:
            # Важность нейрона = произведение L2-норм входа и выхода
            up_weight = torch.cat([m.weight.data for m in block["ups"]], dim=1)
            score_in = up_weight.float().norm(dim=1)                 # [intermediate]
            score_out = block["down"].weight.data.float().norm(dim=0)  # [intermediate]
            importance = score_in * score_out

            _, keep_idx = torch.topk(importance, n_keep, largest=True)
            keep_idx = keep_idx.sort().values

            # Обрезаем up/gate (строки)
            for up_mod in block["ups"]:
                up_mod.weight = nn.Parameter(
                    up_mod.weight.data[keep_idx].contiguous()
                )
                if up_mod.bias is not None:
                    up_mod.bias = nn.Parameter(
                        up_mod.bias.data[keep_idx].contiguous()
                    )
                up_mod.out_features = n_keep

            # Обрезаем down (столбцы)
            down = block["down"]
            down.weight = nn.Parameter(
                down.weight.data[:, keep_idx].contiguous()
            )
            down.in_features = n_keep

        # Обновляем config
        if hasattr(model.config, "intermediate_size"):
            model.config.intermediate_size = n_keep
        if hasattr(model.config, "ffn_dim"):
            model.config.ffn_dim = n_keep

        return {
            "ffn_neurons": f"{original_size} → {n_keep} (−{n_remove})",
            "ffn_blocks": str(len(triplets)),
        }

    def _find_ffn_triplets(self, model: nn.Module) -> List[dict]:
        """
        Найти группы {ups: [gate_proj, up_proj], down: down_proj}
        для каждого трансформерного слоя.
        """
        modules = dict(model.named_modules())
        triplets = []
        seen = set()

        PATTERNS = [
            (["gate_proj", "up_proj"], "down_proj"),
            (["c_fc"], "c_proj"),
            (["fc1"], "fc2"),
            (["dense_h_to_4h"], "dense_4h_to_h"),
            (["wi", "wi_0", "wi_1"], "wo"),
        ]

        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear) or name in seen:
                continue

            for up_pats, down_pat in PATTERNS:
                matched_up = None
                for up_p in up_pats:
                    if up_p in name:
                        matched_up = up_p
                        break

                if matched_up is None:
                    continue

                prefix = name.rsplit(matched_up, 1)[0]
                down_name = prefix + down_pat
                down_mod = modules.get(down_name)

                if down_mod is None or not isinstance(down_mod, nn.Linear):
                    continue
                if down_name in seen:
                    continue

                # Собираем все up-слои этого блока
                ups = []
                for up_p in up_pats:
                    up_n = prefix + up_p
                    up_m = modules.get(up_n)
                    if up_m is not None and isinstance(up_m, nn.Linear):
                        ups.append(up_m)
                        seen.add(up_n)

                if ups:
                    seen.add(down_name)
                    triplets.append({"ups": ups, "down": down_mod})
                break

        return triplets

    #  Attention Head Pruning

    def _prune_heads(self, model: nn.Module) -> Dict[str, str]:
        """Удалить attention heads из всех слоёв."""
        cfg = model.config
        n_heads = getattr(cfg, "num_attention_heads", None)
        n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
        hidden = getattr(cfg, "hidden_size", None)

        if n_heads is None or hidden is None:
            logger.warning("Не удалось определить конфигурацию attention")
            return {"heads": "не определены"}

        head_dim = hidden // n_heads
        n_q_remove = max(1, int(n_heads * self.head_sparsity))
        n_q_keep = n_heads - n_q_remove

        # GQA: удаляем целые KV-группы
        is_gqa = n_kv_heads < n_heads
        q_per_kv = n_heads // n_kv_heads if is_gqa else 1

        if is_gqa:
            n_kv_remove = max(1, int(n_kv_heads * self.head_sparsity))
            n_kv_keep = n_kv_heads - n_kv_remove
            n_q_keep = n_kv_keep * q_per_kv
            n_q_remove = n_heads - n_q_keep
        else:
            n_kv_keep = n_q_keep
            n_kv_remove = n_q_remove

        logger.info(
            "Heads: Q %d→%d, KV %d→%d (head_dim=%d, GQA=%s)",
            n_heads, n_q_keep, n_kv_heads, n_kv_keep, head_dim, is_gqa,
        )

        # Вычисляем важность по KV-группам (агрегация по всем слоям)
        kv_importance = torch.zeros(n_kv_heads)
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear) or "q_proj" not in name:
                continue
            w = mod.weight.data.float()
            for kv_h in range(n_kv_heads):
                q_start = kv_h * q_per_kv * head_dim
                q_end = q_start + q_per_kv * head_dim
                if q_end <= w.shape[0]:
                    kv_importance[kv_h] += w[q_start:q_end].norm().item()

        # Выбираем KV-группы для сохранения
        _, keep_kv = torch.topk(kv_importance, n_kv_keep, largest=True)
        keep_kv = keep_kv.sort().values

        # Индексы строк для Q-проекции
        keep_q_rows = torch.cat([
            torch.arange(
                kv_h.item() * q_per_kv * head_dim,
                (kv_h.item() * q_per_kv + q_per_kv) * head_dim,
            )
            for kv_h in keep_kv
        ])

        # Индексы строк для K/V-проекций
        keep_kv_rows = torch.cat([
            torch.arange(
                kv_h.item() * head_dim,
                (kv_h.item() + 1) * head_dim,
            )
            for kv_h in keep_kv
        ])

        # Применяем ко всем слоям
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue

            if "q_proj" in name:
                _prune_rows(mod, keep_q_rows)
            elif "k_proj" in name or "v_proj" in name:
                _prune_rows(mod, keep_kv_rows)
            elif "o_proj" in name:
                _prune_cols(mod, keep_q_rows)

        # Обновляем config
        cfg.num_attention_heads = n_q_keep
        cfg.num_key_value_heads = n_kv_keep
        if hasattr(cfg, "num_heads"):
            cfg.num_heads = n_q_keep

        return {
            "Q_heads": f"{n_heads} → {n_q_keep} (−{n_q_remove})",
            "KV_heads": f"{n_kv_heads} → {n_kv_keep} (−{n_kv_remove})",
        }


#  Утилиты обрезки тензоров

def _prune_rows(linear: nn.Linear, keep: torch.Tensor) -> None:
    """Оставить указанные строки weight (и bias)."""
    linear.weight = nn.Parameter(linear.weight.data[keep].contiguous())
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias.data[keep].contiguous())
    linear.out_features = len(keep)


def _prune_cols(linear: nn.Linear, keep: torch.Tensor) -> None:
    """Оставить указанные столбцы weight."""
    linear.weight = nn.Parameter(linear.weight.data[:, keep].contiguous())
    linear.in_features = len(keep)