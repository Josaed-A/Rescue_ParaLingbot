#!/usr/bin/env bash
# Run several mapping variants back to back, each only if the laptop is on AC or
# has enough battery left (a run killed by an empty battery wastes the GPU time
# and can leave a half-written .npz).
#
# Usage: src/mapas/run_variants.sh <eval_dir> "<name>|<process_and_view args>" ...
# Each variant writes <eval_dir>/<name>.npz and <eval_dir>/<name>.log.
set -u
cd "$(dirname "$0")/../.."
EVAL="$1"; shift
MIN_BATTERY="${MIN_BATTERY:-30}"
mkdir -p "$EVAL"

power_ok() {
  local ac cap
  ac=$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)
  cap=$(cat /sys/class/power_supply/BAT*/capacity 2>/dev/null | head -1)
  [ "${ac:-1}" = "1" ] && return 0
  [ "${cap:-100}" -ge "$MIN_BATTERY" ] && return 0
  echo "batería en ${cap}% sin cargador (< ${MIN_BATTERY}%): se detiene la cadena"
  return 1
}

for spec in "$@"; do
  name="${spec%%|*}"
  vargs="${spec#*|}"
  power_ok || exit 3
  echo "== $name  ($(date +%H:%M:%S), batería $(cat /sys/class/power_supply/BAT*/capacity 2>/dev/null)%)"
  start=$(date +%s)
  # shellcheck disable=SC2086
  python3 src/captura/process_and_view.py $vargs --no_serve \
    --save_predictions "$EVAL/$name.npz" > "$EVAL/$name.log" 2>&1
  rc=$?
  echo "   exit=$rc en $(( $(date +%s) - start ))s; $(grep -E 'Inference done|Map built|OutOfMemory' "$EVAL/$name.log" | tr '\n' ' ')"
done
