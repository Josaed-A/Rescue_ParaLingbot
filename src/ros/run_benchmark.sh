#!/usr/bin/env bash
# Benchmark y matriz de pruebas de la integración Stella (etapas 17 y 18).
#
#   src/ros/run_benchmark.sh <carpeta_salida> [corridas...]
#
# Cada corrida es una sesión de replay_live.py (la misma ruta de código que el servidor en vivo) con
# monitoreo de recursos (resource_monitor.py). Videos reproducidos en tiempo real como una cámara.
# Configuraciones del plan (etapa 17):
#   A  baseline actual: sin ROS2 ni Stella
#   D  BASIC + STELLA: puente ROS2 + Stella en núcleos 4-11 (modelo en 0-3,12-19); se calculan los
#      tres modos (B = BASIC, C = STELLA, D = HYBRID) y la geometría se registra con BASIC
#   E  D + geometría registrada con la referencia HYBRID
#   S  STELLA + geometría registrada con la referencia STELLA (correcciones por keyframes en vivo)
# Pruebas extra (etapa 18): T14 Stella se cae a mitad; T15 el modelo se detiene 8 s; T1 webcam
# estática; T13 cámara remota (celular por adb), estática.
set -uo pipefail
OUT="$1"; shift
cd "$(dirname "$0")/../.."
U=captures/pruebas_reales/unisabana
declare -A VID=( [fablab]=$U/prueba_3/source/muestra_2_fablab.mp4 [escaleras]=$U/prueba_4/source/Prueba_4_Desnivel.mp4 [pasillos]=$U/prueba_2/source/muestra_unisabana.mp4 )
declare -A CFGY=( [fablab]=unisabana_portrait_540x960 [escaleras]=unisabana_prueba4_540x960 [pasillos]=unisabana_portrait_540x960 )
STELLA="--ros2 --ros2_resize 540x960 --ros2_pub_every 2 --cpus 0-3,12-19 --stella_cpus 4-11"
mkdir -p "$OUT"

run() {   # nombre, args de replay_live...
  local name=$1; shift
  local d="$OUT/$name"
  rm -rf "$d"; mkdir -p "$d"
  if pgrep -x run_slam > /dev/null; then echo "run_slam previo: lo detengo"; pkill -INT -x run_slam; sleep 5; pkill -9 -x run_slam; fi
  echo "### $name $(date +%T)"
  python3 src/ros/resource_monitor.py --out "$d/recursos.csv" --watch modelo=replay_live.py --watch stella=run_slam \
      --interval 0.5 --stop_when_gone 20 > "$d/monitor.log" 2>&1 &
  local mon=$!
  local t0=$(date +%s.%N)
  src/gpu/run_gpu.sh -- python3 src/vivo/replay_live.py --captures_dir "$d/cap" "$@" > "$d/replay.log" 2>&1
  local rc=$?
  echo "{\"segundos_reloj\": $(python3 -c "print(round($(date +%s.%N)-$t0,1))"), \"rc\": $rc}" > "$d/corrida.json"
  sleep 3; kill $mon 2>/dev/null; wait $mon 2>/dev/null
  local s=$(ls -d "$d"/cap/streaming/sin_guardar/*/ 2>/dev/null | head -1)
  [ -n "$s" ] && mv "$s" "$d/sesion"
  mv "$d"/cap/stella_*.log "$d"/ 2>/dev/null
  rm -rf "$d/cap"
  grep -v "^\[" "$d/replay.log" | tail -1 | cut -c1-260
  echo "run_slam vivos: $(pgrep -x run_slam | wc -l)"
}

for r in "$@"; do
  case "$r" in
    A_*) s=${r#A_}; run "$r" --source video --path "${VID[$s]}" --rotation 90 --realtime --context ;;
    D2_*) s=${r#D2_}; run "$r" --source video --path "${VID[$s]}" --rotation 90 --realtime --context $STELLA \
            --stella_config src/ros/stella/${CFGY[$s]}.yaml --tracking_mode hybrid ;;
    D_*) s=${r#D_}; run "$r" --source video --path "${VID[$s]}" --rotation 90 --realtime --context $STELLA \
            --stella_config src/ros/stella/${CFGY[$s]}.yaml --tracking_mode hybrid ;;
    E_*) s=${r#E_}; run "$r" --source video --path "${VID[$s]}" --rotation 90 --realtime --context $STELLA \
            --stella_config src/ros/stella/${CFGY[$s]}.yaml --tracking_mode hybrid --register_with_reference ;;
    S_*) s=${r#S_}; run "$r" --source video --path "${VID[$s]}" --rotation 90 --realtime --context $STELLA \
            --stella_config src/ros/stella/${CFGY[$s]}.yaml --tracking_mode stella --register_with_reference ;;
    T14_fablab) run "$r" --source video --path "${VID[fablab]}" --rotation 90 --realtime --context $STELLA \
            --stella_config src/ros/stella/${CFGY[fablab]}.yaml --tracking_mode hybrid --debug_kill_stella_at_frame 30 ;;
    T15_fablab) run "$r" --source video --path "${VID[fablab]}" --rotation 90 --realtime --context $STELLA \
            --stella_config src/ros/stella/${CFGY[fablab]}.yaml --tracking_mode hybrid --debug_stall_at_frame 30 --debug_stall_s 8 ;;
    T1_webcam) run "$r" --source webcam --device 0 --max_frames 450 --context --ros2 --cpus 0-3,12-19 --stella_cpus 4-11 \
            --stella_config src/ros/stella/webcam_640x480.yaml --tracking_mode hybrid ;;
    T13_android) run "$r" --source android --serial "${ANDROID_SERIAL:-}" --camera_id 0 --cam_size 960x540 --max_frames 450 --context \
            --ros2 --cpus 0-3,12-19 --stella_cpus 4-11 --stella_config src/ros/stella/android_960x540.yaml --tracking_mode hybrid ;;
    *) echo "corrida desconocida: $r" ;;
  esac
done
echo "### FIN $(date +%T)"
