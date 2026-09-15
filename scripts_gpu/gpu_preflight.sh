#!/usr/bin/env bash
# Pre-flight check before any GPU run of LingBot-Map on this machine.
# Detects the failure modes already seen here and prints the exact fix:
#   1) NVML driver/library mismatch after an unattended driver upgrade -> reboot
#   2) stale viewer/inference processes holding VRAM and the viser port
#   3) CUDA context broken after a laptop suspend/resume -> reload nvidia_uvm
# Usage: scripts_gpu/gpu_preflight.sh [--port 8080] [--kill-stale]
# Exit code 0 = ready for GPU, 1 = something must be fixed first.
set -u

PORT=""
KILL_STALE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --kill-stale) KILL_STALE=1; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "argumento desconocido: $1" >&2; exit 2 ;;
  esac
done

ok=1
say() { printf '%s\n' "$*"; }

# 1. Driver / NVML
if ! smi_out=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1); then
  say "[FALLA] nvidia-smi: $smi_out"
  if printf '%s' "$smi_out" | grep -qi mismatch; then
    say "        El driver se actualizó en segundo plano sin recargar el kernel. Solución: sudo reboot"
  fi
  exit 1
fi
say "[OK]    GPU: $smi_out"

# 2. Stale LingBot-Map processes (anchored on the python executable so the
#    run_gpu.sh wrapper, whose command line also contains "demo.py", never matches)
STALE_RE='^[^ ]*python[0-9.]*( -[^ ]+)* [^ ]*(demo|process_and_view|run_gpu_baseline|batch_demo)\.py'
stale=$(pgrep -af "$STALE_RE" || true)
if [ -n "$stale" ]; then
  say "[AVISO] Procesos de LingBot-Map vivos (retienen VRAM y el puerto del visor):"
  printf '%s\n' "$stale" | sed 's/^/        /'
  if [ "$KILL_STALE" = 1 ]; then
    pids=$(printf '%s\n' "$stale" | awk '{print $1}')
    kill $pids 2>/dev/null
    for _ in 1 2 3 4 5; do
      pgrep -f "$STALE_RE" >/dev/null || break
      sleep 1
    done
    pgrep -f "$STALE_RE" >/dev/null && kill -9 $(pgrep -f "$STALE_RE") 2>/dev/null
    say "        Detenidos."
  else
    say "        Detenerlos: scripts_gpu/gpu_preflight.sh --kill-stale   (o kill <PID>)"
    ok=0
  fi
fi

gpu_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$gpu_apps" ]; then
  say "[INFO]  Otros procesos usando VRAM (comparten los 8 GB):"
  printf '%s\n' "$gpu_apps" | sed 's/^/        /'
fi

# Port of the viser viewer
if [ -n "$PORT" ] && ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":$PORT\$"; then
  say "[FALLA] Puerto $PORT ocupado. Ver quién: ss -tlnp | grep ':$PORT '  (o usar --port <otro>)"
  ok=0
fi

# 3. CUDA usable from a fresh process
if cuda_out=$(python3 - 2>&1 <<'EOF'
import warnings
warnings.filterwarnings("ignore")
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() == False")
torch.zeros(1, device="cuda")
free, total = torch.cuda.mem_get_info()
print(f"torch {torch.__version__}, VRAM libre {free / 2**20:.0f}/{total / 2**20:.0f} MiB")
EOF
); then
  say "[OK]    CUDA: $(printf '%s\n' "$cuda_out" | tail -1)"
else
  say "[FALLA] CUDA: $(printf '%s\n' "$cuda_out" | tail -1)"
  suspends=$(cat /sys/power/suspend_stats/success 2>/dev/null || echo 0)
  if [ "${suspends:-0}" -gt 0 ]; then
    say "        Suspensiones desde el arranque: $suspends (causa probable: contexto CUDA roto al reanudar)."
  fi
  say "        Solución inmediata:  sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm"
  say "        Si rmmod dice 'in use': sudo reboot"
  ok=0
fi

# Advisory only: permanent protection and power source
preserve=$(awk -F': ' '/PreserveVideoMemoryAllocations/{print $2}' /proc/driver/nvidia/params 2>/dev/null)
if [ "$preserve" != "1" ]; then
  say "[AVISO] Protección contra suspensión NO instalada (PreserveVideoMemoryAllocations=$preserve)."
  say "        Instalar una sola vez: sudo scripts_gpu/fix_nvidia_suspend.sh && sudo reboot"
fi
for ac in /sys/class/power_supply/A*/online; do
  [ -r "$ac" ] && [ "$(cat "$ac")" = "0" ] && say "[AVISO] Equipo en batería: conectá el cargador para corridas largas."
done

if [ "$ok" = 1 ]; then
  say "LISTO para GPU."
  exit 0
fi
say "NO LISTO: corregí lo marcado arriba y volvé a correr este chequeo."
exit 1
