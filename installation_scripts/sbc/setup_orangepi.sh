#!/usr/bin/env bash
# Установка бенчмарка для Orange Pi 5 Plus
# Только llama-cpp-python + psutil + datasets
#
# Запуск: sudo bash setup_orange_pi.sh
# Время: 10-20 минут

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_common.sh"

check_root

RAM_MB=$(free -m | awk '/Mem:/ {print $2}')

echo ""
echo "Orange Pi 5+ -- установка бенчмарка"
echo "(только inference, без PyTorch)"
echo "RAM: ${RAM_MB} MB"
echo ""

read -p "Продолжить? (y/n) " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || exit 0

if (( RAM_MB >= 16384 )); then
    SWAP_SIZE_MB=2048
else
    SWAP_SIZE_MB=4096
fi
export SWAP_SIZE_MB

TASKSET_CMD="taskset -c 4-7 "
export TASKSET_CMD

opi_optimize() {
    echo "[OPi] Оптимизация RK3588"

    for cpu in /sys/devices/system/cpu/cpu{4,5,6,7}/cpufreq/scaling_governor; do
        [[ -f "$cpu" ]] && echo "performance" > "$cpu" 2>/dev/null || true
    done

    for cpu in /sys/devices/system/cpu/cpu{0,1,2,3}/cpufreq/scaling_governor; do
        [[ -f "$cpu" ]] && echo "ondemand" > "$cpu" 2>/dev/null || true
    done

    echo "  CPU governors: A76=performance, A55=ondemand"

    sysctl -w vm.vfs_cache_pressure=150 >/dev/null 2>&1 || true
    sysctl -w vm.swappiness=10          >/dev/null 2>&1 || true

    echo "  OK"
}

install_system_packages
opi_optimize
setup_swap
create_directories
setup_python
create_run_script
verify
print_final_message "Orange Pi 5+"