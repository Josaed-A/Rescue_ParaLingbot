#!/usr/bin/env bash
# Corre Stella-VSLAM (stella_vslam_ros, run_slam) aislado, escuchando los frames de ParaLingbot.
#
#   src/ros/run_stella.sh <config.yaml> [--image TOPIC] [--eval-log-dir DIR] [--map-db-out F] [-- más args de run_slam]
#
# Topics resultantes (nodo "stella", sin namespace):
#   entrada   TOPIC (por defecto /paralingbot/camera/image_raw)
#   salida    /stella/camera_pose  /stella/keyframes  /stella/keyframes_2d  /stella/tracking_state
#
# Entorno: ROS2 Jazzy de ~/ros2_jazzy, el workspace ~/Rescue/stella_ws (cv_bridge + stella_vslam_ros) y
# las librerías compiladas en ~/Rescue/stella_ws/prefix (g2o, FBoW, stella_vslam). Sin sudo, nada del sistema.
set -eo pipefail   # sin -u: los setup.bash de ROS2 usan variables sin definir
STELLA_WS="${STELLA_WS:-$HOME/Rescue/stella_ws}"
VOCAB="${STELLA_VOCAB:-$STELLA_WS/vocab/orb_vocab.fbow}"
[ $# -ge 1 ] || { sed -n 2,12p "$0"; exit 1; }
CONFIG="$1"; shift
IMAGE_TOPIC="/paralingbot/camera/image_raw"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE_TOPIC="$2"; shift 2 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
[ -f "$CONFIG" ] || { echo "no existe el config $CONFIG"; exit 1; }
[ -f "$VOCAB" ] || { echo "no existe el vocabulario $VOCAB"; exit 1; }

# shellcheck disable=SC1091
source "$HOME/ros2_jazzy/install/setup.bash"
# shellcheck disable=SC1091
source "$STELLA_WS/install/setup.bash"
# g2o, FBoW y stella_vslam del prefix, y el yaml-cpp 0.8 de ROS2 (yaml_cpp_vendor no exporta su lib al entorno)
export LD_LIBRARY_PATH="$STELLA_WS/prefix/lib:$HOME/ros2_jazzy/install/yaml_cpp_vendor/opt/yaml_cpp_vendor/lib:${LD_LIBRARY_PATH:-}"

echo "Stella-VSLAM: config=$CONFIG  imagen=$IMAGE_TOPIC  vocab=$VOCAB" >&2
exec ros2 run stella_vslam_ros run_slam -v "$VOCAB" -c "$CONFIG" --viewer none "${EXTRA[@]}" \
  --ros-args -r __node:=stella -r camera/image_raw:="$IMAGE_TOPIC" -p publish_tf:=false -p publish_keyframes:="${STELLA_PUBLISH_KF:-true}" \
  -p map_frame:=stella_map -p camera_frame:=stella_camera_link     # etapa 8: no colisionar con map del robot
