#!/usr/bin/env bash
# Универсальный запуск бенчмарка на одноплатнике
#
# Использование:
#   ./run_benchmark.sh                    # все модели
#   ./run_benchmark.sh --skip-perplexity  # без перплексии
#   ./run_benchmark.sh --timeout 120      # свой таймаут

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "${BENCH_DIR}/venv" ]]; then
    source "${BENCH_DIR}/venv/bin/activate"
else
    echo "ОШИБКА: venv не найден в ${BENCH_DIR}/venv"
    echo "Запустите setup_*.sh сначала"
    exit 1
fi

cd "${BENCH_DIR}"

N_MODELS=$(find "${BENCH_DIR}/models" -name "*.gguf" 2>/dev/null | wc -l)

echo "Бенчмарк GGUF-моделей"
echo "  Устройство : $(hostname)"
echo "  RAM        : $(free -m | awk '/Mem:/ {print $2}') MB"
echo "  Swap       : $(free -m | awk '/Swap:/ {print $2}') MB"
echo "  CPU cores  : $(nproc)"
echo "  Моделей    : ${N_MODELS}"

if [[ -f /sys/class/thermal/thermal_zone0/temp ]]; then
    TEMP=$(($(cat /sys/class/thermal/thermal_zone0/temp) / 1000))
    echo "  Температура: ${TEMP} C"
fi
echo ""

if [[ ${N_MODELS} -eq 0 ]]; then
    echo "ОШИБКА: нет GGUF-моделей в ${BENCH_DIR}/models/"
    echo ""
    echo "Скопируйте с ПК:"
    echo "  scp your_pc:~/model_optimizer/output/*.gguf ${BENCH_DIR}/models/"
    exit 1
fi

echo "Модели:"
for f in "${BENCH_DIR}/models/"*.gguf; do
    [[ -f "$f" ]] || continue
    SZ=$(du -h "$f" | cut -f1)
    echo "  ${SZ}  $(basename "$f")"
done
echo ""

RAM_MB=$(free -m | awk '/Mem:/ {print $2}')
if (( RAM_MB < 2048 )); then
    DEFAULT_CTX=256
    DEFAULT_GEN=64
    DEFAULT_PPL=5
    DEFAULT_TIMEOUT=1200
    DEFAULT_THREADS=4
else
    DEFAULT_CTX=512
    DEFAULT_GEN=128
    DEFAULT_PPL=10
    DEFAULT_TIMEOUT=1200
    DEFAULT_THREADS=$(nproc)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="${BENCH_DIR}/logs/bench_${TIMESTAMP}.log"
mkdir -p "${BENCH_DIR}/logs"

echo "Параметры: ctx=${DEFAULT_CTX}, gen=${DEFAULT_GEN}, ppl=${DEFAULT_PPL} samples"
echo "Таймаут: ${DEFAULT_TIMEOUT} сек, потоков: ${DEFAULT_THREADS}"
echo "Лог: ${LOG}"
echo ""

python benchmark.py \
    --models-dir "${BENCH_DIR}/models" \
    --output-dir "${BENCH_DIR}/results" \
    --n-ctx "${DEFAULT_CTX}" \
    --n-threads "${DEFAULT_THREADS}" \
    --gen-tokens "${DEFAULT_GEN}" \
    --ppl-samples "${DEFAULT_PPL}" \
    --timeout "${DEFAULT_TIMEOUT}" \
    --eval-pairs-file "${BENCH_DIR}/eval_pairs.json" \
    "$@" \
    2>&1 | tee "${LOG}"

echo ""
echo "Результаты: ${BENCH_DIR}/results/"
echo "Лог:        ${LOG}"
echo ""
echo "Скопировать на ПК:"
echo "  scp $(whoami)@$(hostname):${BENCH_DIR}/results/benchmark_${TIMESTAMP}* ~/model_optimizer/bench_results/"