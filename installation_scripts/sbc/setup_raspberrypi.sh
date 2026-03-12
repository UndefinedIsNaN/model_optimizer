#!/usr/bin/env bash
# Установка бенчмарка для Raspberry Pi 3 Model B
# Только llama-cpp-python + psutil + datasets
#
# Запуск: sudo bash setup_raspberry_pi.sh
# Время: 15-25 минут

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_common.sh"

check_root

echo ""
echo "Raspberry Pi 3B -- установка бенчмарка"
echo "(только inference, без PyTorch)"
echo ""

read -p "Продолжить? (y/n) " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || exit 0

SWAP_SIZE_MB=4096
export SWAP_SIZE_MB

rpi_optimize() {
    echo "[RPi] Оптимизация"

    local CONFIG="/boot/config.txt"
    [[ -f "$CONFIG" ]] || CONFIG="/boot/firmware/config.txt"

    if [[ -f "$CONFIG" ]]; then
        if grep -q "^gpu_mem=" "$CONFIG"; then
            sed -i 's/^gpu_mem=.*/gpu_mem=16/' "$CONFIG"
        else
            echo "gpu_mem=16" >> "$CONFIG"
        fi
        echo "  GPU -> 16 MB (больше RAM для моделей)"
        echo "  ВНИМАНИЕ: нужна перезагрузка (sudo reboot)"
    fi

    for svc in bluetooth hciuart avahi-daemon triggerhappy; do
        systemctl disable --now "$svc" 2>/dev/null || true
    done

    sysctl -w vm.vfs_cache_pressure=200 >/dev/null 2>&1 || true
    sysctl -w vm.min_free_kbytes=16384  >/dev/null 2>&1 || true

    echo "  OK"
}

install_system_packages
rpi_optimize
setup_swap
create_directories
setup_python
create_run_script
verify
print_final_message "Raspberry Pi 3B"