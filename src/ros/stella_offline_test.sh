#!/usr/bin/env bash
# Prueba reproducible de Stella-VSLAM aislado sobre una fuente de ParaLingbot (etapa 2).
#
#   src/ros/stella_offline_test.sh <config.yaml> <out_dir> [args de camera_publisher.py...]
#
# Arranca, en este orden: el grabador de poses (record_stella.py), Stella (run_stella.sh) y el
# publicador de frames (camera_publisher.py) con los args que siguen. Cuando el publicador termina
# (fuente agotada o --max_frames), espera unos segundos y cierra todo. Deja en <out_dir>:
#   stella.jsonl     poses, estados y keyframes de Stella (última línea: resumen)
#   published.jsonl  un registro por frame publicado (frame_id, stamp)
#   stella.log, publisher.log, recorder.log
#   traj/            frame_trajectory.txt y keyframe_trajectory.txt (TUM) que escribe run_slam al salir
#
# Ejemplo (video del fablab, 30 fps, reescalado a 540x960, 30 s):
#   src/ros/stella_offline_test.sh src/ros/stella/unisabana_portrait_540x960.yaml /tmp/stella_fablab \
#       --source video --path captures/pruebas_reales/unisabana/prueba_3/source/muestra_2_fablab.mp4 \
#       --resize 540x960 --max_frames 900
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$1"; OUT="$2"; shift 2
mkdir -p "$OUT/traj"
cd "$HERE/../.."

if pgrep -x run_slam > /dev/null; then echo "ya hay un run_slam corriendo (pgrep -af run_slam); detenerlo antes"; exit 1; fi
python3 "$HERE/record_stella.py" --out "$OUT/stella.jsonl" > "$OUT/recorder.log" 2>&1 &
REC=$!
"$HERE/run_stella.sh" "$CONFIG" --eval-log-dir "$OUT/traj" > "$OUT/stella.log" 2>&1 &
STELLA=$!
sleep 6                                   # carga del vocabulario (45 MB) y descubrimiento DDS
if ! kill -0 $STELLA 2>/dev/null; then echo "Stella no arrancó:"; tail -20 "$OUT/stella.log"; kill $REC 2>/dev/null; exit 1; fi

t0=$(date +%s.%N)
python3 "$HERE/camera_publisher.py" --log "$OUT/published.jsonl" --camera_info "$CONFIG" "$@" > "$OUT/publisher.log" 2>&1
PUB_RC=$?
t1=$(date +%s.%N)
sleep 4                                   # que Stella procese lo que le queda en cola
# run_stella.sh hace exec de "ros2 run", que a su vez lanza run_slam: la señal tiene que llegar al
# run_slam real (si no, queda huérfano publicando en los mismos topics y contamina la siguiente corrida).
# Con SIGINT run_slam cierra ordenado: espera el BA de bucle y escribe las trayectorias en traj/.
pkill -INT -x run_slam 2>/dev/null
for _ in $(seq 1 60); do pgrep -x run_slam > /dev/null || break; sleep 0.5; done
if pgrep -x run_slam > /dev/null; then echo "run_slam no cerró en 30 s: matándolo"; pkill -9 -x run_slam; fi
kill $STELLA 2>/dev/null; wait $STELLA 2>/dev/null
kill -INT $REC 2>/dev/null; wait $REC 2>/dev/null

echo "publicador rc=$PUB_RC, $(python3 -c "print(round($t1-$t0,1))") s"
tail -2 "$OUT/publisher.log"
echo "--- resumen de Stella ---"
tail -1 "$OUT/stella.jsonl" | python3 -c "import json,sys; d=json.load(sys.stdin); d.pop('type',None); print(json.dumps(d, indent=1, ensure_ascii=False))"
echo "--- últimas líneas del log de Stella ---"
grep -v "^\[.*\] \[I\]" "$OUT/stella.log" | tail -5
ls "$OUT/traj"
