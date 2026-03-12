#!/usr/bin/env bash
# Все 9 комбинаций для одной модели
#
# Использование:
#   bash scripts/pc/run_all_combinations.sh <repo> [quant_type] [доп. аргументы]
#
# Примеры:
#   bash scripts/pc/run_all_combinations.sh Qwen/Qwen2-0.5B
#   bash scripts/pc/run_all_combinations.sh Qwen/Qwen2-0.5B Q5_K_M

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${PROJECT_DIR}/venv/bin/activate"
cd "${PROJECT_DIR}"

REPO="${1:?Использование: $0 <repo> [quant_type]}"
QUANT_TYPE="${2:-Q4_K_M}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="${PROJECT_DIR}/logs/all_combos_${TIMESTAMP}.log"
mkdir -p "${PROJECT_DIR}/logs"

echo "Все 9 комбинаций" | tee "${LOG}"
echo "  Repo:  ${REPO}" | tee -a "${LOG}"
echo "  Quant: ${QUANT_TYPE}" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

TOTAL=9
CURRENT=0
FAILED=0
SUCCEEDED=0
TIMES=()

run_one() {
    local pruning="$1" method="$2" order="$3"

    CURRENT=$((CURRENT + 1))
    echo "" | tee -a "${LOG}"
    echo "[${CURRENT}/${TOTAL}] ${pruning} + ${method} (${order})" | tee -a "${LOG}"

    local START_SEC=$SECONDS

    if python main.py \
        --repo "${REPO}" \
        --pruning "${pruning}" \
        --quant-method "${method}" \
        --quant-type "${QUANT_TYPE}" \
        --order "${order}" \
        --recovery-steps 50 \
        --qat-steps 100 \
        --output-dir "${PROJECT_DIR}/output" \
        --work-dir "${PROJECT_DIR}/work" \
        "$@" \
        2>&1 | tee -a "${LOG}"; then

        local ELAPSED=$(( SECONDS - START_SEC ))
        echo "  OK (${ELAPSED} сек)" | tee -a "${LOG}"
        SUCCEEDED=$((SUCCEEDED + 1))
        TIMES+=("${pruning}+${method}+${order}: ${ELAPSED}s")
    else
        echo "  FAIL" | tee -a "${LOG}"
        FAILED=$((FAILED + 1))
    fi
}

run_one magnitude_20 ptq prune_first
run_one magnitude_30 ptq prune_first
run_one structured   ptq prune_first

run_one magnitude_20 qat prune_first
run_one magnitude_30 qat prune_first
run_one structured   qat prune_first

run_one magnitude_20 qat quant_first
run_one magnitude_30 qat quant_first
run_one structured   qat quant_first

echo "" | tee -a "${LOG}"
echo "Итого: ${SUCCEEDED}/${TOTAL} успешно, ${FAILED} ошибок" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

for t in "${TIMES[@]}"; do
    echo "  ${t}" | tee -a "${LOG}"
done

echo "" | tee -a "${LOG}"
echo "Готовые модели:" | tee -a "${LOG}"
for f in "${PROJECT_DIR}"/output/*.gguf; do
    if [[ -f "$f" ]]; then
        SZ=$(du -h "$f" | cut -f1)
        echo "  ${SZ}  $(basename "$f")" | tee -a "${LOG}"
    fi
done

echo "" | tee -a "${LOG}"
echo "Отправьте на одноплатники: bash scripts/pc/deploy.sh" | tee -a "${LOG}"
echo "Лог: ${LOG}"