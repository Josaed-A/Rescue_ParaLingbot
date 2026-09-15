#!/usr/bin/env bash
# Run a GPU command with the pre-flight check first, and with suspend, idle
# sleep and lid-close suspend blocked for the command's whole lifetime
# (systemd-inhibit, no sudo needed). Suspending while a CUDA process is alive
# is what breaks nvidia_uvm on this laptop.
# Usage: scripts_gpu/run_gpu.sh [--kill-stale] [--port N] -- python3 demo.py ...
# The viewer port is taken from the command's --port (demo.py default 8080).
set -u

here="$(cd "$(dirname "$0")" && pwd)"

pre_args=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  pre_args+=("$1")
  shift
done
[ "${1:-}" = "--" ] && shift
if [ $# -eq 0 ]; then
  sed -n '2,7p' "$0"
  exit 2
fi

# Check the port the command will bind, unless given explicitly
if ! printf '%s\n' "${pre_args[@]}" | grep -qx -- --port; then
  port=""
  prev=""
  for a in "$@"; do
    [ "$prev" = "--port" ] && port="$a"
    prev="$a"
  done
  if [ -z "$port" ]; then
    case "$*" in
      *--no_serve*) ;;
      *process_and_view.py*) port=8082 ;;
      *demo.py*) port=8080 ;;
    esac
  fi
  [ -n "$port" ] && pre_args+=(--port "$port")
fi

if ! "$here/gpu_preflight.sh" "${pre_args[@]}"; then
  echo "run_gpu.sh: pre-flight falló, no se lanza el comando." >&2
  exit 1
fi

echo "run_gpu.sh: suspensión bloqueada mientras corra el comando (Ctrl+C para detenerlo y liberar la GPU)."
exec systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
  --who="LingBot-Map" --why="Proceso CUDA activo: suspender rompe nvidia_uvm" \
  "$@"
