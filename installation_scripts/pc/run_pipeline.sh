#!/usr/bin/env bash
# Один пайплайн: прунинг + квантизация
#
# Использование:
#   bash scripts/pc/run_pipeline.sh <repo> <pruning> <quant_method> <quant_type> <order> [доп. аргументы]
#
# Примеры:
#   bash scripts/pc/run_pipeline.sh Qwen/Qwen2-0.5B structured ptq Q4_K_M prune_first
#   bash scripts/pc/run_pipeline.sh Qwen/Qwen2-0.5B magnitude_20 qat Q4_K_M quant_first --qat-steps 200

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${PROJECT_DIR}/venv/bin/activate"
cd "${PROJECT_DIR}"

REPO="${1:?Использование: $0 <repo> <pruning> <quant_method> <quant_type> <order>}"
PRUNING="${2:?Укажите pruning: magnitude_20 | magnitude_30 | structured}"
QUANT_METHOD="${3:?Укажите quant-method: ptq | qat}"
QUANT_TYPE="${4:?Укажите quant-type: Q4_K_M, Q5_K_M, ...}"
ORDER="${5:?Укажите order: prune_first | quant_first}"
shift 5

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="${PROJECT_DIR}/logs/pipeline_${PRUNING}_${QUANT_METHOD}_${ORDER}_${TIMESTAMP}.log"
mkdir -p "${PROJECT_DIR}/logs"

echo "Пайплайн обработки модели"
echo "  Repo:   ${REPO}"
echo "  Prune:  ${PRUNING}"
echo "  Quant:  ${QUANT_METHOD} (${QUANT_TYPE})"
echo "  Order:  ${ORDER}"
echo "  Log:    ${LOG}"
echo ""

python main.py \
    --repo "${REPO}" \
    --pruning "${PRUNING}" \
    --quant-method "${QUANT_METHOD}" \
    --quant-type "${QUANT_TYPE}" \
    --order "${ORDER}" \
    --output-dir "${PROJECT_DIR}/output" \
    --work-dir "${PROJECT_DIR}/work" \
    "$@" \
    2>&1 | tee "${LOG}"

echo ""
echo "Готово. Лог: ${LOG}"
echo "Модели в: ${PROJECT_DIR}/output/"