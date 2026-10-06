#!/usr/bin/env bash
# One-time system fix (needs sudo) so a laptop suspend no longer breaks CUDA.
#
# With NVreg_PreserveVideoMemoryAllocations=0 (driver default here) the VRAM
# of live CUDA contexts is lost on suspend, and after resume every new process
# gets "CUDA unknown error" / "busy or unavailable" until nvidia_uvm is
# reloaded. Setting it to 1 makes nvidia-suspend.service save VRAM to
# NVreg_TemporaryFilePath and restore it on resume.
#
# Usage:  sudo scripts_gpu/fix_nvidia_suspend.sh && sudo reboot
# Revert: sudo rm /etc/modprobe.d/nvidia-power-management.conf && sudo update-initramfs -u && sudo reboot
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Hace falta root: sudo $0" >&2
  exit 1
fi

CONF=/etc/modprobe.d/nvidia-power-management.conf
cat > "$CONF" <<'EOF'
# Written by Rescue_ParaLingbot scripts_gpu/fix_nvidia_suspend.sh
# Keep VRAM of live CUDA contexts across suspend/resume.
options nvidia NVreg_PreserveVideoMemoryAllocations=1 NVreg_TemporaryFilePath=/var/tmp
EOF
echo "Escrito $CONF"

# Already enabled by the driver package on this machine; kept for idempotency.
systemctl enable nvidia-suspend.service nvidia-resume.service nvidia-hibernate.service

update-initramfs -u

# Also recover the currently broken CUDA context, if nothing holds nvidia_uvm.
if lsmod | grep -q '^nvidia_uvm '; then
  if rmmod nvidia_uvm 2>/dev/null && modprobe nvidia_uvm; then
    echo "nvidia_uvm recargado: CUDA usable ya en esta sesión."
  else
    echo "nvidia_uvm está en uso: el reinicio lo resuelve."
  fi
fi

echo
echo "Listo. Reiniciá para activar la protección: sudo reboot"
echo "Verificar después: grep PreserveVideoMemoryAllocations /proc/driver/nvidia/params   (debe decir 1)"
