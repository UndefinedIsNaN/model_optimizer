#!/usr/bin/env bash
# Общая часть установки для одноплатников (только бенчмарк)
# Вызывается из setup_raspberry_pi.sh / setup_orange_pi.sh

set -euo pipefail

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "ОШИБКА: запустите с sudo: sudo bash $0"
        exit 1
    fi
}

ACTUAL_USER="${SUDO_USER:-$(whoami)}"
BENCH_DIR="${BENCH_DIR:-/home/${ACTUAL_USER}/benchmark}"
VENV_DIR="${BENCH_DIR}/venv"
SWAP_SIZE_MB="${SWAP_SIZE_MB:-2048}"

# 1. Системные пакеты

install_system_packages() {
    echo "[1/6] Системные пакеты"

    apt-get update -qq

    apt-get install -y -qq \
        build-essential \
        cmake \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        libopenblas-dev \
        liblapack-dev \
        gfortran \
        htop \
        tmux \
        wget

    echo "  OK"
}

# 2. Swap

setup_swap() {
    echo "[2/6] Swap (${SWAP_SIZE_MB} MB)"

    local SWAPFILE="/swapfile_bench"
    local current_swap
    current_swap=$(free -m | awk '/Swap:/ {print $2}')

    if (( current_swap >= SWAP_SIZE_MB )); then
        echo "  Swap уже достаточный: ${current_swap} MB"
        return
    fi

    if [[ -f "$SWAPFILE" ]]; then
        swapoff "$SWAPFILE" 2>/dev/null || true
        rm -f "$SWAPFILE"
    fi

    if systemctl is-active --quiet dphys-swapfile 2>/dev/null; then
        systemctl stop dphys-swapfile
        systemctl disable dphys-swapfile
    fi

    echo "  Создаем swap ${SWAP_SIZE_MB} MB ..."
    dd if=/dev/zero of="$SWAPFILE" bs=1M count="${SWAP_SIZE_MB}" status=progress
    chmod 600 "$SWAPFILE"
    mkswap "$SWAPFILE"
    swapon "$SWAPFILE"

    if ! grep -q "$SWAPFILE" /etc/fstab; then
        echo "${SWAPFILE} none swap sw 0 0" >> /etc/fstab
    fi

    sysctl -w vm.swappiness=10 >/dev/null
    grep -q "vm.swappiness" /etc/sysctl.conf || \
        echo "vm.swappiness=10" >> /etc/sysctl.conf

    echo "  Swap: $(free -m | awk '/Swap:/ {print $2}') MB"
}

# 3. Директории

create_directories() {
    echo "[3/6] Директории"

    mkdir -p "${BENCH_DIR}"/{models,results,logs}
    chown -R "${ACTUAL_USER}:${ACTUAL_USER}" "${BENCH_DIR}"

    echo "  ${BENCH_DIR}/"
    echo "    models/   -- сюда кладутся .gguf файлы"
    echo "    results/  -- CSV и TXT отчеты"
    echo "    logs/     -- логи"
    echo "    venv/     -- Python окружение"
}

# 4. Python venv + зависимости

setup_python() {
    echo "[4/6] Python окружение"

    if [[ ! -d "${VENV_DIR}" ]]; then
        sudo -u "${ACTUAL_USER}" python3 -m venv "${VENV_DIR}"
    fi

    local PIP="${VENV_DIR}/bin/pip"

    sudo -u "${ACTUAL_USER}" ${PIP} install --upgrade pip setuptools wheel -q

    echo "  Устанавливаем llama-cpp-python (с OpenBLAS) ..."
    echo "  Компиляция может занять 5-15 минут"

    sudo -u "${ACTUAL_USER}" \
        CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
        ${PIP} install --no-cache-dir llama-cpp-python \
        2>&1 || \
    sudo -u "${ACTUAL_USER}" ${PIP} install --no-cache-dir llama-cpp-python

    sudo -u "${ACTUAL_USER}" ${PIP} install --no-cache-dir -q \
        psutil \
        datasets

    echo "  OK"
}

# 5. Скрипт запуска

create_run_script() {
    echo "[5/6] Скрипт запуска"

    local RAM_MB
    RAM_MB=$(free -m | awk '/Mem:/ {print $2}')
    local N_CORES
    N_CORES=$(nproc)

    local CTX=512
    local GEN_TOKENS=128
    local PPL_SAMPLES=10
    local TIMEOUT=600
    local THREADS="${N_CORES}"

    if (( RAM_MB < 2048 )); then
        CTX=256
        GEN_TOKENS=64
        PPL_SAMPLES=5
        TIMEOUT=300
    fi

    cat > "${BENCH_DIR}/run_benchmark.sh" << RUNBENCH
#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="${BENCH_DIR}"
source "\${BENCH_DIR}/venv/bin/activate"
cd "\${BENCH_DIR}"

N_MODELS=\$(find "\${BENCH_DIR}/models" -name "*.gguf" 2>/dev/null | wc -l)

echo "Бенчмарк GGUF-моделей"
echo "  Устройство : \$(hostname)"
echo "  RAM        : \$(free -m | awk '/Mem:/ {print \$2}') MB"
echo "  CPU cores  : \$(nproc)"
echo "  Моделей    : \${N_MODELS}"
echo ""

if [[ \${N_MODELS} -eq 0 ]]; then
    echo "ОШИБКА: нет моделей в \${BENCH_DIR}/models/"
    echo "Скопируйте с ПК:"
    echo "  scp your_pc:~/model_optimizer/output/*.gguf \${BENCH_DIR}/models/"
    exit 1
fi

TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
LOG="\${BENCH_DIR}/logs/bench_\${TIMESTAMP}.log"
mkdir -p "\${BENCH_DIR}/logs"

${TASKSET_CMD:-}python benchmark.py \\
    --models-dir "\${BENCH_DIR}/models" \\
    --output-dir "\${BENCH_DIR}/results" \\
    --n-ctx ${CTX} \\
    --n-threads ${THREADS} \\
    --gen-tokens ${GEN_TOKENS} \\
    --ppl-samples ${PPL_SAMPLES} \\
    --timeout ${TIMEOUT} \\
    "\$@" \\
    2>&1 | tee "\${LOG}"

echo ""
echo "Результаты: \${BENCH_DIR}/results/"
echo "Лог:        \${LOG}"
echo ""
echo "Скопировать на ПК:"
echo "  scp \$(whoami)@\$(hostname):\${BENCH_DIR}/results/* ~/model_optimizer/bench_results/"
RUNBENCH

    chmod +x "${BENCH_DIR}/run_benchmark.sh"
    chown "${ACTUAL_USER}:${ACTUAL_USER}" "${BENCH_DIR}/run_benchmark.sh"

    echo "  OK: ${BENCH_DIR}/run_benchmark.sh"
}

# 6. Проверка

verify() {
    echo "[6/6] Проверка"

    local PY="${VENV_DIR}/bin/python"
    local ERRORS=0

    check() {
        local name="$1" cmd="$2"
        if eval "${cmd}" >/dev/null 2>&1; then
            echo "  OK  ${name}"
        else
            echo "  ERR ${name}"
            ERRORS=$((ERRORS + 1))
        fi
    }

    check "Python"           "${PY} --version"
    check "llama-cpp-python" "${PY} -c 'from llama_cpp import Llama; print(\"OK\")'"
    check "psutil"           "${PY} -c 'import psutil'"
    check "datasets"         "${PY} -c 'import datasets'"

    local RAM_MB SWAP_MB DISK_FREE
    RAM_MB=$(free -m | awk '/Mem:/ {print $2}')
    SWAP_MB=$(free -m | awk '/Swap:/ {print $2}')
    DISK_FREE=$(df -h "${BENCH_DIR}" | tail -1 | awk '{print $4}')

    echo ""
    echo "  RAM:  ${RAM_MB} MB"
    echo "  Swap: ${SWAP_MB} MB"
    echo "  Диск: ${DISK_FREE} свободно"
    echo "  CPU:  $(nproc) ядер"

    if [[ -f /sys/class/thermal/thermal_zone0/temp ]]; then
        local TEMP=$(($(cat /sys/class/thermal/thermal_zone0/temp) / 1000))
        echo "  Темп: ${TEMP} C"
    fi

    echo ""
    if (( ERRORS > 0 )); then
        echo "ОШИБКА: ${ERRORS} проблем"
        return 1
    fi
    echo "  Все проверки пройдены"
}

print_final_message() {
    local BOARD="$1"

    echo ""
    echo "${BOARD} готов к бенчмарку."
    echo ""
    echo "1. Скопируйте модели с ПК:"
    echo "   scp your_pc:~/model_optimizer/output/*.gguf ${BENCH_DIR}/models/"
    echo ""
    echo "   Или используйте deploy.sh на ПК:"
    echo "   bash scripts/pc/deploy.sh"
    echo ""
    echo "2. Запустите бенчмарк:"
    echo "   ${BENCH_DIR}/run_benchmark.sh"
    echo ""
    echo "3. Скопируйте результаты обратно:"
    echo "   scp ${ACTUAL_USER}@\$(hostname):${BENCH_DIR}/results/* ~/model_optimizer/bench_results/"
    echo ""
}