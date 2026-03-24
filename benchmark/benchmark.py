#!/usr/bin/env python3
"""
  GGUF Model Benchmark                                          
                                                                
  Измеряет время загрузки, TTFT, скорость генерации,            
  пиковую память и перплексию. Каждая модель запускается         
  в отдельном процессе с таймаутом.                              

Установка llama-cpp-python на ARM:
  pip install llama-cpp-python
  # Или с оптимизациями:
  CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \\
      pip install llama-cpp-python

Примеры:
  # Все GGUF в папке:
  python benchmark.py --models-dir ./output

  # Конкретные файлы:
  python benchmark.py --models model1.gguf model2.gguf

  # Для Raspberry Pi (экономим ресурсы):
  python benchmark.py --models-dir ./output \\
      --n-ctx 256 --gen-tokens 64 --ppl-samples 5 --timeout 300

  # Без перплексии (быстрее):
  python benchmark.py --models-dir ./output --skip-perplexity
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Опциональные зависимости ──
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil не установлен — память не будет отслеживаться")

# ═══════════════════════════════════════════════════════
#  Константы
# ═══════════════════════════════════════════════════════

DEFAULT_PROMPTS = [
    "Explain how computers work in simple terms.",
    "Write a short story about a cat exploring a garden.",
    "List five interesting facts about the ocean.",
]

EVAL_PAIRS = [
    (
        "What is the capital of France?",
        "The capital of France is Paris.",
    ),
    (
        "Summarize in one sentence: Water is essential for all known forms of life. "
        "It covers about 71 percent of the Earth's surface. "
        "Most of the Earth's water is found in its oceans.",
        "Water is essential for life and covers most of the Earth's surface.",
    ),
    (
        "Translate to French: The weather is nice today.",
        "Il fait beau aujourd'hui.",
    ),
    (
        "What is 2 + 2?",
        "2 + 2 equals 4.",
    ),
    (
        "Name three primary colors.",
        "The three primary colors are red, blue, and yellow.",
    ),
]

# ═══════════════════════════════════════════════════════
#  Монитор памяти
# ═══════════════════════════════════════════════════════

class MemoryMonitor:
    """
    Фоновый поток, отслеживающий пиковое RSS
    текущего процесса через psutil.
    """

    def __init__(self, interval: float = 0.3):
        self.peak_mb = 0.0
        self._interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not HAS_PSUTIL:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss / (1024 ** 2)
                self.peak_mb = max(self.peak_mb, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop.wait(self._interval)

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.peak_mb


# ═══════════════════════════════════════════════════════
#  Загрузка данных для перплексии
# ═══════════════════════════════════════════════════════

def load_perplexity_texts(
    source: str = "wikitext",
    n_samples: int = 10,
) -> List[str]:
    if source == "wikitext":
        try:
            from datasets import load_dataset
            ds = _load_wikitext_with_retry()
            if ds is None:
                return []
            texts = [
                r["text"] for r in ds
                if len(r["text"].strip()) > 100
            ]
            return texts[:n_samples]
        except ImportError:
            logger.warning("datasets не установлен")
            return []
        except Exception as e:
            logger.warning("Ошибка загрузки wikitext: %s", e)
            return []

    path = Path(source)
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = []
        step = 2000
        for i in range(0, len(text), step):
            chunk = text[i : i + step].strip()
            if len(chunk) > 100:
                chunks.append(chunk)
        return chunks[:n_samples]

    return []


def _load_wikitext_with_retry():
    """Загрузить wikitext, при битом кэше -- очистить и повторить."""
    from datasets import load_dataset
    import shutil

    try:
        return load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        error_msg = str(e)
        if "0 bytes" in error_msg or "Parquet" in error_msg or "ArrowInvalid" in error_msg:
            logger.warning("Битый кэш wikitext")
            _clear_wikitext_cache()
            try:
                return load_dataset(
                    "wikitext", "wikitext-2-raw-v1", split="test",
                    download_mode="force_redownload",
                )
            except Exception as e2:
                logger.warning("Повторная загрузка не удалась: %s", e2)
                return None
        else:
            raise


def _clear_wikitext_cache():
    import shutil
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub" / "datasets--wikitext",
        Path.home() / ".cache" / "huggingface" / "datasets" / "wikitext",
        Path("/root/.cache/huggingface/hub/datasets--wikitext"),
        Path("/root/.cache/huggingface/datasets/wikitext"),
    ]
    for d in cache_dirs:
        if d.exists():
            logger.info("Удаляем битый кэш: %s", d)
            shutil.rmtree(str(d), ignore_errors=True)

# ═══════════════════════════════════════════════════════
#  Worker — бенчмарк одной модели (child process)
# ═══════════════════════════════════════════════════════

def _benchmark_worker(
    model_path: str,
    config: Dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    """
    Запускается в дочернем процессе.
    Загружает модель, прогоняет тесты, кладёт результат в очередь.
    """

    result: Dict[str, Any] = {
        "model_name": Path(model_path).stem,
        "model_path": model_path,
        "model_size_mb": 0.0,
        "status": "error",
        "error_message": "",
        "load_time_sec": 0.0,
        "ttft_ms": 0.0,
        "gen_speed_tps": 0.0,
        "peak_memory_mb": 0.0,
        "perplexity": float("inf"),
        "n_tokens_generated": 0,
        "generated_text": "",
        "bleu": 0.0,
        "rouge_l": 0.0,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        result["model_size_mb"] = Path(model_path).stat().st_size / (1024 ** 2)
    except OSError:
        pass

    mem = MemoryMonitor()
    mem.start()

    try:
        # ── Импорт llama-cpp-python ──
        try:
            from llama_cpp import Llama
        except ImportError:
            result["error_message"] = (
                "llama-cpp-python не установлен. "
                "pip install llama-cpp-python"
            )
            result_queue.put(result)
            return

        # ── 1. Загрузка модели ──
        need_ppl = len(config.get("ppl_texts", [])) > 0

        t0 = time.perf_counter()
        llm = Llama(
            model_path=model_path,
            n_ctx=config["n_ctx"],
            n_threads=config.get("n_threads", os.cpu_count() or 4),
            n_gpu_layers=config.get("n_gpu_layers", 0),
            logits_all=need_ppl,
            verbose=False,
        )
        result["load_time_sec"] = time.perf_counter() - t0

        # ── 2. Генерация (TTFT + скорость) ──
        _measure_generation(llm, config, result)

        # ── 3. Перплексия ──
        ppl_texts = config.get("ppl_texts", [])
        if need_ppl:
            _measure_perplexity(llm, ppl_texts, config, result)

        # -- 4. BLEU / ROUGE-L --
        if not config.get("skip_bleu_rouge", False):
            _measure_bleu_rouge(llm, config, result)

        result["status"] = "ok"

    except MemoryError:
        result["status"] = "oom"
        result["error_message"] = "Out of memory"
    except Exception as e:
        result["status"] = "error"
        result["error_message"] = f"{type(e).__name__}: {e}"

    result["peak_memory_mb"] = mem.stop()
    result_queue.put(result)


def _measure_generation(
    llm, config: Dict[str, Any], result: Dict[str, Any],
) -> None:
    """
    Измерение TTFT и скорости генерации.

    Прогоняем каждый промпт gen_repeats раз,
    усредняем метрики.
    """
    prompts = config.get("prompts", DEFAULT_PROMPTS)
    max_tokens = config.get("gen_tokens", 128)
    n_repeats = config.get("gen_repeats", 1)

    all_ttft: List[float] = []
    all_speed: List[float] = []
    all_n_tokens: List[int] = []
    last_text = ""

    for prompt in prompts:
        for _ in range(n_repeats):
            t_start = time.perf_counter()
            first_token_time: Optional[float] = None
            chunk_count = 0
            gen_text = ""

            try:
                stream = llm.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    stream=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                for chunk in stream:
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    gen_text += chunk["choices"][0].get("text", "")
                    chunk_count += 1
            except Exception:
                continue

            t_end = time.perf_counter()

            if first_token_time is None or chunk_count == 0:
                continue

            # TTFT = время от начала до первого токена (включает prefill)
            ttft = (first_token_time - t_start) * 1000  # мс
            all_ttft.append(ttft)

            # Скорость decode (без prefill)
            if chunk_count > 1:
                decode_time = t_end - first_token_time
                speed = (chunk_count - 1) / decode_time if decode_time > 0 else 0
            else:
                total = t_end - t_start
                speed = 1.0 / total if total > 0 else 0

            all_speed.append(speed)
            all_n_tokens.append(chunk_count)
            last_text = gen_text

    if all_ttft:
        result["ttft_ms"] = sum(all_ttft) / len(all_ttft)
    if all_speed:
        result["gen_speed_tps"] = sum(all_speed) / len(all_speed)
    if all_n_tokens:
        result["n_tokens_generated"] = int(sum(all_n_tokens) / len(all_n_tokens))
    result["generated_text"] = last_text[:500]


def _measure_perplexity(
    llm, texts: List[str], config: Dict[str, Any], result: Dict[str, Any],
) -> None:
    total_nll = 0.0
    total_count = 0
    max_chars = config.get("n_ctx", 512) * 4

    # Метод 1: create_completion с echo=True (работает при logits_all=True)
    for text in texts:
        text = text.strip()
        if len(text) < 20:
            continue

        text = text[:max_chars]

        try:
            output = llm.create_completion(
                text,
                max_tokens=1,
                logprobs=1,
                echo=True,
                temperature=1.0,
            )

            choice = output["choices"][0]
            lp_data = choice.get("logprobs")
            if lp_data is None:
                continue

            token_lps = lp_data.get("token_logprobs", [])
            valid = [
                lp for lp in token_lps
                if lp is not None and math.isfinite(lp)
            ]

            if valid:
                total_nll -= sum(valid)
                total_count += len(valid)

        except Exception:
            continue

    # Метод 2 (fallback): tokenize + eval если метод 1 не дал результатов
    if total_count == 0:
        total_nll, total_count = _ppl_fallback_eval(llm, texts, config)

    if total_count > 0:
        avg_nll = min(total_nll / total_count, 100)
        result["perplexity"] = math.exp(avg_nll)
    else:
        result["perplexity"] = float("inf")


def _ppl_fallback_eval(
    llm, texts: List[str], config: Dict[str, Any],
) -> tuple:
    """Fallback: tokenize + eval + ручной подсчет logprobs."""
    total_nll = 0.0
    total_count = 0
    n_ctx = config.get("n_ctx", 512)

    for text in texts:
        text = text.strip()
        if len(text) < 20:
            continue

        try:
            tokens = llm.tokenize(text.encode("utf-8"))
            if len(tokens) < 2:
                continue
            tokens = tokens[:n_ctx]

            llm.reset()
            llm.eval(tokens)

            scores = llm.scores

            for i in range(len(tokens) - 1):
                logits = scores[i]
                target_id = tokens[i + 1]

                max_logit = max(logits)
                sum_exp = sum(math.exp(v - max_logit) for v in logits)
                log_sum_exp = max_logit + math.log(sum_exp)
                log_prob = logits[target_id] - log_sum_exp

                if math.isfinite(log_prob):
                    total_nll -= log_prob
                    total_count += 1

        except AttributeError:
            break
        except Exception:
            continue

    return total_nll, total_count

# -----------------------------------------------------------
#  BLEU и ROUGE-L
# -----------------------------------------------------------

def _tokenize_simple(text: str) -> List[str]:
    """Простая токенизация: lowercase + split по пробелам и пунктуации."""
    import re
    return re.findall(r'\w+', text.lower())


def _get_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    """Подсчитать n-граммы."""
    ngrams: Dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return ngrams


def _compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    Sentence-level BLEU.
    Среднее геометрическое n-gram precisions (n=1..max_n) + brevity penalty.
    """
    ref_tokens = _tokenize_simple(reference)
    hyp_tokens = _tokenize_simple(hypothesis)

    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens)))

    log_avg = 0.0
    n_valid = 0

    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        hyp_ngrams = _get_ngrams(hyp_tokens, n)

        if len(hyp_ngrams) == 0:
            continue

        clipped = 0
        total = 0
        for gram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(gram, 0))
            total += count

        if total == 0 or clipped == 0:
            return 0.0

        precision = clipped / total
        log_avg += math.log(precision)
        n_valid += 1

    if n_valid == 0:
        return 0.0

    return bp * math.exp(log_avg / n_valid)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Длина наибольшей общей подпоследовательности."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Оптимизация памяти: две строки вместо полной таблицы
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _compute_rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 на основе LCS."""
    ref_tokens = _tokenize_simple(reference)
    hyp_tokens = _tokenize_simple(hypothesis)

    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0

    lcs = _lcs_length(ref_tokens, hyp_tokens)

    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _measure_bleu_rouge(
    llm, config: Dict[str, Any], result: Dict[str, Any],
) -> None:
    """Генерация ответов на eval_pairs и подсчет BLEU / ROUGE-L."""
    eval_pairs = config.get("eval_pairs", EVAL_PAIRS)
    max_tokens = config.get("gen_tokens", 128)

    bleu_scores: List[float] = []
    rouge_scores: List[float] = []

    for prompt, reference in eval_pairs:
        try:
            output = llm.create_completion(
                prompt,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
            )
            hypothesis = output["choices"][0].get("text", "").strip()
        except Exception:
            continue

        if not hypothesis:
            continue

        bleu = _compute_bleu(reference, hypothesis)
        rouge = _compute_rouge_l(reference, hypothesis)

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)

    if bleu_scores:
        result["bleu"] = sum(bleu_scores) / len(bleu_scores)
    if rouge_scores:
        result["rouge_l"] = sum(rouge_scores) / len(rouge_scores)

#  Отчёты: CSV + TXT

class Reporter:
    """Записывает результаты в CSV и TXT."""

    CSV_FIELDS = [
        "model_name", "model_size_mb", "status", "error_message",
        "load_time_sec", "ttft_ms", "gen_speed_tps",
        "peak_memory_mb", "perplexity", "bleu", "rouge_l",
        "n_tokens_generated",
        "model_path", "timestamp",
    ]

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def write_all(
        self,
        ok_results: List[Dict],
        failed_results: List[Dict],
    ) -> tuple[Path, Path]:
        csv_path = self._write_csv(ok_results, failed_results)
        txt_path = self._write_txt(ok_results, failed_results)
        return csv_path, txt_path

    # ── CSV ──

    def _write_csv(self, ok: List[Dict], failed: List[Dict]) -> Path:
        path = self.output_dir / f"benchmark_{self.ts}.csv"
        all_rows = ok + failed

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=self.CSV_FIELDS, extrasaction="ignore",
            )
            w.writeheader()
            for r in all_rows:
                row = {}
                for k in self.CSV_FIELDS:
                    v = r.get(k, "")
                    if isinstance(v, float):
                        row[k] = "inf" if math.isinf(v) else f"{v:.4f}"
                    else:
                        row[k] = v
                w.writerow(row)

        return path

    # ── TXT ──

    def _write_txt(self, ok: List[Dict], failed: List[Dict]) -> Path:
        path = self.output_dir / f"benchmark_{self.ts}.txt"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "═" * 62,
            f"  BENCHMARK RESULTS — {now}",
            "═" * 62,
            "",
        ]

        # Успешные модели
        for r in ok:
            lines.extend(self._format_model(r))
            lines.append("")

        # Проваленные
        all_failed = [r for r in failed]
        # Добавляем не-ok из ok (на случай если status != ok попал туда)
        if all_failed:
            lines.append("═" * 62)
            lines.append("  FAILED / SKIPPED MODELS")
            lines.append("═" * 62)
            lines.append("")
            for r in all_failed:
                icon = {
                    "timeout": "⏰", "oom": "💾", "error": "❌",
                }.get(r.get("status", ""), "❌")
                lines.append(
                    f"  {icon}  {r.get('model_name', '?')}"
                    f"  —  {r.get('status', 'error')}: "
                    f"{r.get('error_message', 'unknown')}"
                )
            lines.append("")

        # Сводная таблица
        lines.extend(self._format_table(ok))

        # Итоги
        lines.extend(self._format_summary(ok, failed))

        text = "\n".join(lines)
        path.write_text(text, encoding="utf-8")

        # Дублируем в stdout
        print()
        print(text)

        return path

    @staticmethod
    def _fmt_ppl(v: float) -> str:
        if math.isinf(v) or v <= 0:
            return "—"
        return f"{v:.2f}"

    def _format_model(self, r: Dict) -> List[str]:
        ppl = self._fmt_ppl(r.get("perplexity", float("inf")))
        mem = r.get("peak_memory_mb", 0)
        mem_str = f"{mem:.1f} MB" if mem > 0 else "—"

        return [
            f"  ── {r.get('model_name', '?')} ──",
            f"      Размер файла     : {r.get('model_size_mb', 0):.1f} MB",
            f"      Статус           : ✅ OK",
            f"      Время загрузки   : {r.get('load_time_sec', 0):.2f} сек",
            f"      TTFT             : {r.get('ttft_ms', 0):.0f} мс",
            f"      Скорость генер.  : {r.get('gen_speed_tps', 0):.2f} tok/sec",
            f"      Пиковая память   : {mem_str}",
            f"      Perplexity    : {ppl}",
            f"      BLEU          : {r.get('bleu', 0):.4f}",
            f"      ROUGE-L       : {r.get('rouge_l', 0):.4f}",
            f"      Tokens gen    : {r.get('n_tokens_generated', 0)}",
            ]

    def _format_table(self, ok: List[Dict]) -> List[str]:
        if not ok:
            return []

        lines = [
            "COMPARISON TABLE",
            "",
            f"  {'Model':<32s} {'Size':>7s} {'Speed':>8s} "
            f"{'TTFT':>7s} {'PPL':>7s} {'BLEU':>6s} {'RGE-L':>6s} {'Mem':>7s}",
            "  " + "-" * 84,
        ]

        # Заголовок
        hdr = (
            f"  {'Model':<32s} {'Size':>7s} {'Speed':>8s} "
            f"{'TTFT':>7s} {'PPL':>7s} {'Mem':>7s}"
        )
        lines.append(hdr)
        lines.append("  " + "─" * 60)

        for r in sorted(ok, key=lambda x: x.get("gen_speed_tps", 0), reverse=True):
            name = r.get("model_name", "?")
            if len(name) > 30:
                name = name[:27] + "..."

            ppl = r.get("perplexity", float("inf"))
            ppl_s = f"{ppl:.1f}" if math.isfinite(ppl) else "—"
            mem = r.get("peak_memory_mb", 0)
            mem_s = f"{mem:.0f}MB" if mem > 0 else "—"

            bleu_v = r.get("bleu", 0)
            bleu_s = f"{bleu_v:.3f}" if bleu_v > 0 else "--"
            rouge_v = r.get("rouge_l", 0)
            rouge_s = f"{rouge_v:.3f}" if rouge_v > 0 else "--"

            row = (
                f"  {name:<32s} "
                f"{r.get('model_size_mb', 0):>5.0f}MB "
                f"{r.get('gen_speed_tps', 0):>6.2f}t/s "
                f"{r.get('ttft_ms', 0):>5.0f}ms "
                f"{ppl_s:>7s} "
                f"{bleu_s:>6s} "
                f"{rouge_s:>6s} "
                f"{mem_s:>7s}"
            )
            lines.append(row)

        lines.append("")
        return lines

    @staticmethod
    def _format_summary(ok: List[Dict], failed: List[Dict]) -> List[str]:
        n_total = len(ok) + len(failed)

        lines = [
            "═" * 62,
            "  SUMMARY",
            "═" * 62,
            "",
            f"  Всего моделей     : {n_total}",
            f"  Успешно           : {len(ok)}",
            f"  Не удалось        : {len(failed)}",
        ]

        if ok:
            best_speed = max(ok, key=lambda r: r.get("gen_speed_tps", 0))
            lines.append(
                f"  Лучшая скорость   : {best_speed['model_name']} "
                f"({best_speed.get('gen_speed_tps', 0):.2f} tok/sec)"
            )

            valid_ppl = [
                r for r in ok
                if math.isfinite(r.get("perplexity", float("inf")))
            ]
            if valid_ppl:
                best_ppl = min(
                    valid_ppl, key=lambda r: r.get("perplexity", float("inf"))
                )
                lines.append(
                    f"  Лучшая перплексия : {best_ppl['model_name']} "
                    f"({best_ppl.get('perplexity', 0):.2f})"
                )

            smallest = min(ok, key=lambda r: r.get("model_size_mb", float("inf")))
            lines.append(
                f"  Наименьший размер : {smallest['model_name']} "
                f"({smallest.get('model_size_mb', 0):.1f} MB)"
            )

            mem_models = [r for r in ok if r.get("peak_memory_mb", 0) > 0]
            if mem_models:
                least_mem = min(
                    mem_models, key=lambda r: r["peak_memory_mb"]
                )
                lines.append(
                    f"  Минимум памяти    : {least_mem['model_name']} "
                    f"({least_mem['peak_memory_mb']:.1f} MB)"
                )

        lines.extend(["", "═" * 62])
        return lines


#  Поиск моделей

def find_models(
    models_dir: Optional[str],
    model_files: Optional[List[str]],
) -> List[Path]:
    """Найти все GGUF-файлы для бенчмарка."""
    paths: List[Path] = []

    if model_files:
        for f in model_files:
            p = Path(f)
            if p.exists() and p.suffix == ".gguf":
                paths.append(p)
            else:
                logger.warning("Не найден / не GGUF: %s", f)

    if models_dir:
        d = Path(models_dir)
        if d.is_dir():
            found = sorted(d.glob("**/*.gguf"))
            paths.extend(found)
            logger.info("Найдено %d GGUF в %s", len(found), d)
        else:
            logger.warning("Директория не найдена: %s", d)

    # Дедупликация
    seen = set()
    unique = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return unique


#  Оркестрация

def run_benchmarks(args: argparse.Namespace) -> None:
    """Основной цикл: по модели → процесс → результат."""

    models = find_models(args.models_dir, args.models)
    if not models:
        logger.error("Нет моделей! Укажите --models-dir или --models")
        sys.exit(1)

    logger.info("")
    logger.info("GGUF Model Benchmark")
    logger.info("  Моделей      : %d", len(models))
    logger.info("  Контекст     : %d", args.n_ctx)
    logger.info("  Потоки       : %d", args.n_threads)
    logger.info("  Генерация    : %d токенов × %d повторов",
                args.gen_tokens, args.gen_repeats)
    logger.info("  Таймаут      : %d сек", args.timeout)
    logger.info("  Perplexity : %s",
                "off" if args.skip_perplexity else f"{args.ppl_samples} samples")
    logger.info("  BLEU/ROUGE : %s",
                "off" if args.skip_bleu_rouge else f"{len(EVAL_PAIRS)} pairs")
    logger.info("")

    for m in models:
        sz = m.stat().st_size / (1024 ** 2)
        logger.info("  • %s (%.1f MB)", m.name, sz)

    # ── Данные для перплексии ──
    ppl_texts: List[str] = []
    if not args.skip_perplexity:
        ppl_texts = load_perplexity_texts(args.ppl_source, args.ppl_samples)
        if not ppl_texts:
            logger.warning("Нет данных для перплексии — пропуск")

    # ── Конфиг для воркеров ──
    config: Dict[str, Any] = {
        "n_ctx": args.n_ctx,
        "n_threads": args.n_threads,
        "n_gpu_layers": args.n_gpu_layers,
        "gen_tokens": args.gen_tokens,
        "gen_repeats": args.gen_repeats,
        "prompts": args.prompts or DEFAULT_PROMPTS,
        "ppl_texts": ppl_texts,
        "skip_bleu_rouge": args.skip_bleu_rouge,
        "eval_pairs": EVAL_PAIRS,
    }

    ok_results: List[Dict] = []
    failed_results: List[Dict] = []

    # spawn — безопасный старт на всех ОС,
    # не наследует состояние родителя
    ctx = mp.get_context("spawn")

    for i, model_path in enumerate(models):
        sz = model_path.stat().st_size / (1024 ** 2)

        logger.info("")
        logger.info("━" * 55)
        logger.info(
            "  [%d/%d]  %s  (%.1f MB)",
            i + 1, len(models), model_path.name, sz,
        )
        logger.info("━" * 55)

        result_queue = ctx.Queue()
        proc = ctx.Process(
            target=_benchmark_worker,
            args=(str(model_path.resolve()), config, result_queue),
        )

        proc.start()
        proc.join(timeout=args.timeout)

        # ── Таймаут ──
        if proc.is_alive():
            logger.warning(
                "  ⏰  ТАЙМАУТ (%d сек) — убиваю процесс …", args.timeout
            )
            proc.kill()
            proc.join(timeout=10)

            failed_results.append({
                "model_name": model_path.stem,
                "model_path": str(model_path),
                "model_size_mb": sz,
                "status": "timeout",
                "error_message": f"Таймаут {args.timeout} сек",
                "timestamp": datetime.now().isoformat(),
            })
            continue

        # ── Получаем результат ──
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            result = {
                "model_name": model_path.stem,
                "model_path": str(model_path),
                "model_size_mb": sz,
                "status": "error",
                "error_message": "Процесс завершился без результата",
                "timestamp": datetime.now().isoformat(),
            }

        status = result.get("status", "error")

        if status == "ok":
            ok_results.append(result)

            ppl = result.get("perplexity", float("inf"))
            ppl_s = f"{ppl:.2f}" if math.isfinite(ppl) else "—"
            mem = result.get("peak_memory_mb", 0)
            mem_s = f"{mem:.0f}MB" if mem > 0 else "—"

            logger.info(
                "  ✅  OK  │  %.2f tok/sec  │  TTFT=%.0f ms  │  PPL=%s  │  %s",
                result.get("gen_speed_tps", 0),
                result.get("ttft_ms", 0),
                ppl_s, mem_s,
            )
        else:
            failed_results.append(result)
            logger.warning(
                "   %s  │  %s",
                status.upper(),
                result.get("error_message", ""),
            )

    # ── Отчёты ──
    reporter = Reporter(Path(args.output_dir))
    csv_path, txt_path = reporter.write_all(ok_results, failed_results)

    logger.info("")
    logger.info("═" * 55)
    logger.info("  Отчёты сохранены:")
    logger.info("    CSV : %s", csv_path)
    logger.info("    TXT : %s", txt_path)
    logger.info("═" * 55)


# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Бенчмарк GGUF-моделей: скорость, память, перплексия",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:

  # Все модели из папки:
  python benchmark.py --models-dir ./output

  # Конкретные файлы:
  python benchmark.py --models m1.gguf m2.gguf m3.gguf

  # Для Raspberry Pi 3B (экономим RAM):
  python benchmark.py --models-dir ./output \\
      --n-ctx 256 --gen-tokens 64 --ppl-samples 5 \\
      --timeout 300 --n-threads 4

  # Для Orange Pi 5+ (больше ресурсов):
  python benchmark.py --models-dir ./output \\
      --n-ctx 512 --gen-tokens 128 --ppl-samples 20 \\
      --n-threads 8

  # Быстрый тест (без перплексии):
  python benchmark.py --models-dir ./output --skip-perplexity

  # С GPU-ускорением:
  python benchmark.py --models-dir ./output --n-gpu-layers 99

  # Свои промпты:
  python benchmark.py --models-dir ./output \\
      --prompts "Расскажи про космос" "What is AI?"
""",
    )

    g1 = p.add_argument_group("Модели")
    g1.add_argument(
        "--models-dir",
        help="Папка с .gguf файлами (ищет рекурсивно)",
    )
    g1.add_argument(
        "--models", nargs="+",
        help="Пути к конкретным .gguf файлам",
    )

    g2 = p.add_argument_group("Inference")
    g2.add_argument(
        "--n-ctx", type=int, default=512,
        help="Размер контекста (по умолчанию: 512)",
    )
    g2.add_argument(
        "--n-threads", type=int, default=os.cpu_count() or 4,
        help=f"Потоки CPU (по умолчанию: {os.cpu_count() or 4})",
    )
    g2.add_argument(
        "--n-gpu-layers", type=int, default=0,
        help="Слоёв на GPU (0 = только CPU)",
    )
    g2.add_argument(
        "--gen-tokens", type=int, default=128,
        help="Токенов для генерации (по умолчанию: 128)",
    )
    g2.add_argument(
        "--gen-repeats", type=int, default=1,
        help="Повторов генерации на каждый промпт (по умолчанию: 1)",
    )
    g2.add_argument(
        "--prompts", nargs="+",
        help="Свои промпты (по умолчанию: 3 встроенных)",
    )

    g3 = p.add_argument_group("Перплексия")
    g3.add_argument(
        "--skip-perplexity", action="store_true",
        help="Пропустить оценку перплексии",
    )
    g3.add_argument(
        "--ppl-source", default="wikitext",
        help="Источник: 'wikitext' или путь к .txt (по умолчанию: wikitext)",
    )
    g3.add_argument(
        "--ppl-samples", type=int, default=10,
        help="Кол-во текстов для перплексии (по умолчанию: 10)",
    )
    g3.add_argument("--skip-bleu-rouge", action="store_true",
                     help="Skip BLEU/ROUGE evaluation")

    g4 = p.add_argument_group("Таймаут и изоляция")
    g4.add_argument(
        "--timeout", type=int, default=600,
        help="Таймаут на модель в секундах (по умолчанию: 600)",
    )

    g5 = p.add_argument_group("Вывод")
    g5.add_argument(
        "--output-dir", default="./bench_results",
        help="Папка для CSV/TXT отчётов",
    )

    return p.parse_args()


# ═══════════════════════════════════════════════════════
#  Точка входа

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_benchmarks(args)


if __name__ == "__main__":
    main()