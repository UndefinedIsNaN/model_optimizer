#!/usr/bin/env bash
# Отправка GGUF-моделей и benchmark.py на одноплатники
#
# Использование:
#   bash scripts/pc/deploy.sh              # на все устройства
#   bash scripts/pc/deploy.sh rpi          # только Raspberry Pi
#   bash scripts/pc/deploy.sh opi          # только Orange Pi
#   bash scripts/pc/deploy.sh user@host    # произвольный хост

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEVICES=(
    "rpi|pi@raspberrypi.local" # Тут надо настраивать
    "opi|orangepi@orangepi.local"
)


GGUF_DIR="${PROJECT_DIR}/output"
FILTER="${1:-all}"

GGUF_COUNT=$(find "${GGUF_DIR}" -name "*.gguf" 2>/dev/null | wc -l)
if (( GGUF_COUNT == 0 )); then
    echo "ОШИБКА: нет GGUF-файлов в ${GGUF_DIR}"
    exit 1
fi

echo "Отправка моделей на одноплатники"
echo "  Моделей: ${GGUF_COUNT}"
echo ""

for f in "${GGUF_DIR}"/*.gguf; do
    [[ -f "$f" ]] || continue
    SZ=$(du -h "$f" | cut -f1)
    echo "  ${SZ}  $(basename "$f")"
done
echo ""

deploy_to() {
    local LABEL="$1"
    local TARGET="$2"
    local REMOTE_USER="${TARGET%%@*}"
    local REMOTE_PATH="/home/${REMOTE_USER}/benchmark"

    echo "--- ${LABEL} (${TARGET}) ---"

    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${TARGET}" "echo ok" >/dev/null 2>&1; then
        echo "  НЕДОСТУПЕН: ${TARGET}"
        echo "  Проверьте: ssh ${TARGET}"
        echo ""
        return 1
    fi

    ssh "${TARGET}" "mkdir -p ${REMOTE_PATH}/models ${REMOTE_PATH}/results ${REMOTE_PATH}/logs"

    echo "  benchmark.py ..."
    scp -q "${PROJECT_DIR}/benchmark.py" "${TARGET}:${REMOTE_PATH}/"

    echo "  run_benchmark.sh ..."
    scp -q "${PROJECT_DIR}/scripts/boards/run_benchmark.sh" "${TARGET}:${REMOTE_PATH}/" 2>/dev/null || true

    echo "  GGUF-модели (${GGUF_COUNT} шт.) ..."
    if command -v rsync >/dev/null 2>&1; then
        rsync -avh --progress "${GGUF_DIR}/"*.gguf "${TARGET}:${REMOTE_PATH}/models/"
    else
        scp "${GGUF_DIR}/"*.gguf "${TARGET}:${REMOTE_PATH}/models/"
    fi

    echo "  OK: ${LABEL}"
    echo ""
    echo "  Запуск бенчмарка:"
    echo "    ssh ${TARGET}"
    echo "    cd ${REMOTE_PATH}"
    echo "    bash run_benchmark.sh"
    echo ""
}

DEPLOYED=0

for entry in "${DEVICES[@]}"; do
    LABEL="${entry%%|*}"
    TARGET="${entry##*|}"

    if [[ "${FILTER}" == "all" ]] || [[ "${FILTER}" == "${LABEL}" ]]; then
        if deploy_to "${LABEL}" "${TARGET}"; then
            DEPLOYED=$((DEPLOYED + 1))
        fi
    fi
done

if [[ "${FILTER}" != "all" ]] && [[ "${FILTER}" != "rpi" ]] && [[ "${FILTER}" != "opi" ]]; then
    if [[ "${FILTER}" == *"@"* ]]; then
        deploy_to "custom" "${FILTER}"
        DEPLOYED=$((DEPLOYED + 1))
    fi
fi

echo "Отправлено на ${DEPLOYED} устройств(а)"