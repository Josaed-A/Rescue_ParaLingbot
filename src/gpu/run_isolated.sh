#!/usr/bin/env bash
# Corre un comando en su propio cgroup (scope de systemd del usuario) con tope de RAM.
#
# Por qué existe (diagnóstico del 2026-10-03, ver la bitácora): todo lo que se lanza desde
# una terminal de VS Code -incluido nohup/setsid- queda dentro del cgroup de la ventana de
# VS Code. Cuando un trabajo pesado agota la RAM:
#   - systemd-oomd mata ese cgroup ENTERO (VS Code, sus terminales, Claude Code y los
#     trabajos en segundo plano: "killed 188 process(es)");
#   - el OOM killer del kernel elige primero procesos de VS Code y Firefox.
# Aislado en su scope y con MemoryMax, si el trabajo se pasa de memoria muere él solo.
#
# Uso: src/gpu/run_isolated.sh [--max 12G] [--name nombre] -- comando ...
# Por defecto MemoryMax = RAM disponible al lanzar - 4 GB (mínimo 3 GB, máximo RAM total - 8 GB):
# el escritorio, Firefox y VS Code ya ocupan ~14 GB en esta máquina, y un tope calculado sobre
# la RAM total (22 GB) dejó que la presión subiera tanto que systemd-oomd mató a GNOME Shell
# (2026-10-03). Sin MemoryHigh: frenar el proceso genera presión de memoria (PSI), que es justo
# lo que dispara a systemd-oomd; mejor que el kernel lo mate rápido dentro de su cgroup.
# Sin swap (MemorySwapMax=0): el swap de 2 GB lleno también dispara a oomd.
# GARDIAN_MEM_MAX sobrescribe el tope.
set -u
max="${GARDIAN_MEM_MAX:-}"; name="trabajo"
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  case "$1" in
    --max) max="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    *) echo "run_isolated.sh: opción desconocida $1" >&2; exit 2 ;;
  esac
done
[ "${1:-}" = "--" ] && shift
if [ $# -eq 0 ]; then sed -n '2,16p' "$0"; exit 2; fi

total_gb=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
avail_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
if [ -z "$max" ]; then
  m=$(( avail_gb - 4 ))
  cap=$(( total_gb - 8 ))
  [ "$m" -gt "$cap" ] && m=$cap
  if [ "$m" -lt 3 ]; then
    echo "run_isolated.sh: sólo hay ${avail_gb} GB de RAM disponibles; cerrá algo antes de lanzar." >&2
    exit 1
  fi
  max="${m}G"
fi

if command -v systemd-run >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  echo "run_isolated.sh: cgroup propio 'gardian-$name-$$' (MemoryMax=$max de ${avail_gb} GB disponibles, sin swap)" >&2
  # --scope: systemd-run registra el scope y ejecuta el comando con su mismo PID
  exec systemd-run --user --scope --quiet --collect --unit "gardian-$name-$$" \
    -p MemoryMax="$max" -p MemorySwapMax=0 "$@"
fi
echo "run_isolated.sh: systemd --user no disponible, se corre sin aislar" >&2
exec "$@"
