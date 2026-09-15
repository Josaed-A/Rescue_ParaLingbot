#!/usr/bin/env bash
# Instala (o repara) el entorno de este repo en cualquier equipo Linux, sin sudo.
# Versiones: env/constraints.txt (validadas juntas). Archivos de modelo: env/assets.json.
#
# Modular: elegir solo lo necesario con --profiles (separados por coma):
#   core        lingbot_map + demo.py: torch, numpy<2, scipy, opencv...     [por defecto]
#   vis         visor viser, export GLB, --mask_sky                          [por defecto]
#   gpu         monitoreo RAM/VRAM de scripts_gpu/ y scripts_seq/            [por defecto]
#   flashinfer  KV cache paginado (solo útil en GPUs de 16 GB o más)
#   render      demo_render/ (MP4 offline): open3d, extensiones CUDA, Kaolin
#               compilado desde fuente (necesita CUDA toolkit), ffmpeg
#   bench       benchmark/: evo, OpenEXR, plyfile, open3d
#   all         todos los anteriores
#
# Uso:
#   ./setup_env.sh                                  # .venv con core,vis,gpu + archivos de modelo
#   ./setup_env.sh --profiles all
#   ./setup_env.sh --source-dir /media/usb/Rescue_ParaLingbot   # copiar modelos en vez de descargar
#   ./setup_env.sh --mode user                      # instalar en ~/.local (sin venv)
#   ./setup_env.sh --torch cpu                      # forzar torch de CPU (auto: según el driver)
#   ./setup_env.sh --profiles render --install-cuda-toolkit      # baja el toolkit a ~/cuda-X.Y
#   ./setup_env.sh --dry-run                        # mostrar el plan sin ejecutar nada
# Después: source .venv/bin/activate && python tools/doctor.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TORCH_VERSION="2.12.0"
TORCHVISION_VERSION="0.27.0"
KAOLIN_REF="d52da9f86d460e8abcd99037e21ff0bec57997ed"   # compilado y validado 2026-09-10 (reporta 0.18.0)

PROFILES="core,vis,gpu"
MODE="venv"
VENV="$ROOT/.venv"
TORCH="auto"
SOURCE_DIRS=()
SKIP_ASSETS=0
INSTALL_CUDA=0
DRY=0
PY="${PYTHON:-python3}"

usage() { sed -n '2,/^set -euo/p' "$0" | sed -e '/^set -euo/d' -e 's/^# \{0,1\}//'; }
step() { printf '\n==> %s\n' "$*"; }
run() { printf '    $'; printf ' %q' "$@"; printf '\n'; [ "$DRY" = 1 ] || "$@"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --profiles) PROFILES="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --venv) VENV="$(realpath -m "$2")"; MODE="venv"; shift 2 ;;
    --torch) TORCH="$2"; shift 2 ;;
    --source-dir) SOURCE_DIRS+=(--source-dir "$(realpath -m "$2")"); shift 2 ;;
    --skip-assets) SKIP_ASSETS=1; shift ;;
    --install-cuda-toolkit) INSTALL_CUDA=1; shift ;;
    --python) PY="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "opción desconocida: $1 (ver --help)" ;;
  esac
done

[ "$PROFILES" = "all" ] && PROFILES="core,vis,gpu,flashinfer,render,bench"
case ",$PROFILES," in *,core,*) ;; *) PROFILES="core,$PROFILES" ;; esac
for p in ${PROFILES//,/ }; do
  case "$p" in core|vis|gpu|flashinfer|render|bench) ;; *) die "perfil desconocido: $p" ;; esac
done
has() { case ",$PROFILES," in *,"$1",*) return 0 ;; *) return 1 ;; esac; }

# --- Comprobaciones previas -------------------------------------------------
[ "$(uname -s)" = "Linux" ] || die "este instalador es para Linux; en Windows seguir SETUP.md"
command -v "$PY" >/dev/null || die "no se encontró $PY (usar --python python3.X)"
PYVER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$PYVER" in 3.10|3.11|3.12|3.13) ;;
  *) die "Python $PYVER: torch $TORCH_VERSION solo publica ruedas para 3.10-3.13 (usar --python python3.X)" ;;
esac
case "$MODE" in venv|user) ;; *) die "--mode debe ser venv o user" ;; esac

DRIVER=""; GPU_CAP=""
if command -v nvidia-smi >/dev/null; then
  if q=$(nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader 2>/dev/null); then
    DRIVER=$(echo "$q" | head -1 | cut -d, -f1 | tr -d ' ')
    GPU_CAP=$(echo "$q" | head -1 | cut -d, -f2 | tr -d ' ')
  else
    echo "AVISO: nvidia-smi falla (¿driver actualizado sin reiniciar?). Reiniciar, o pasar --torch cu130|cu126."
  fi
fi
if [ "$TORCH" = "auto" ]; then
  if [ -z "$DRIVER" ]; then
    TORCH="cpu"
  elif [ "${DRIVER%%.*}" -ge 580 ]; then
    TORCH="cu130"
  elif [ "${DRIVER%%.*}" -ge 560 ]; then
    TORCH="cu126"
  else
    TORCH="cpu"
    echo "AVISO: driver $DRIVER < 560: torch $TORCH_VERSION con CUDA no corre; se instala la build de CPU."
  fi
fi
case "$TORCH" in cu130|cu126|cpu) ;; *) die "--torch debe ser auto, cu130, cu126 o cpu (torch $TORCH_VERSION no tiene cu128)" ;; esac
if has render && [ "$TORCH" = "cpu" ]; then die "el perfil render necesita GPU NVIDIA y torch con CUDA"; fi

step "Plan"
echo "    perfiles: $PROFILES"
echo "    modo:     $MODE$([ "$MODE" = venv ] && echo " ($VENV)")"
echo "    python:   $PY ($PYVER)"
echo "    torch:    $TORCH_VERSION+$TORCH   (driver: ${DRIVER:-sin GPU NVIDIA})"
# ROS2 (setup.bash) exporta PYTHONPATH con sus site-packages: se cuelan en cualquier venv.
# Se ignora para instalar; tools/doctor.py avisa si al ejecutar tapa algún paquete.
[ -n "${PYTHONPATH:-}" ] && echo "    PYTHONPATH definido (¿ROS2?): se ignora durante la instalación"
[ "$DRY" = 1 ] && echo "    (dry-run: solo se muestran los comandos)"

# --- Entorno ----------------------------------------------------------------
C=(-c "$ROOT/env/constraints.txt")
if [ "$MODE" = "venv" ]; then
  step "Entorno virtual"
  PYBIN="$VENV/bin/python"
  BINDIR="$VENV/bin"
  if [ ! -x "$PYBIN" ]; then
    run env -u PYTHONPATH "$PY" -m venv "$VENV" || die "no se pudo crear el venv (Ubuntu: sudo apt install python$PYVER-venv), o usar --mode user"
  fi
  PIP=(env -u PYTHONPATH "$PYBIN" -m pip install)
  # setuptools < 80: Kaolin todavía importa pkg_resources al compilar
  run "${PIP[@]}" --upgrade "pip>=24" "setuptools>=70,<80" wheel
else
  PYBIN="$PY"
  BINDIR="$HOME/.local/bin"
  PIP=(env -u PYTHONPATH "$PY" -m pip install --user)
fi

step "PyTorch $TORCH_VERSION ($TORCH)"
REINSTALL=()
# torch.__version__ lleva la variante (+cu130); los metadatos del paquete no siempre
CUR_TORCH=$([ -x "$(command -v "$PYBIN")" ] && CUDA_VISIBLE_DEVICES="" env -u PYTHONPATH "$PYBIN" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)
if [ -n "$CUR_TORCH" ] && [ "$CUR_TORCH" != "$TORCH_VERSION+$TORCH" ]; then
  echo "    instalado: $CUR_TORCH -> se reemplaza por $TORCH_VERSION+$TORCH"
  REINSTALL=(--force-reinstall)
fi
# con -c desde el primer paso: sin él, torch arrastra numpy 2.x (el índice de PyTorch también tiene 1.26.4)
run "${PIP[@]}" "${C[@]}" "${REINSTALL[@]}" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "https://download.pytorch.org/whl/$TORCH"

step "Dependencias de los perfiles"
REQS=()
for p in ${PROFILES//,/ }; do REQS+=(-r "env/requirements/$p.txt"); done
run "${PIP[@]}" "${C[@]}" "${REQS[@]}"
if has render; then
  # onnxruntime (vis) y onnxruntime-gpu (render) comparten archivos: que gane la de GPU
  run "${PIP[@]}" "${C[@]}" --force-reinstall --no-deps onnxruntime-gpu
fi

step "lingbot_map importable desde cualquier carpeta"
if [ "$MODE" = "venv" ]; then
  run "${PIP[@]}" --no-deps -e "$ROOT"
else
  # pip/setuptools del sistema en Ubuntu 22.04 no soportan editable PEP 660: se expone
  # solo el paquete con un .pth que apunta a una carpeta con un symlink a ESTA copia.
  USER_SITE=$("$PY" -m site --user-site)
  LINK_DIR="$HOME/.local/share/lingbot_map_dev"
  run mkdir -p "$USER_SITE" "$LINK_DIR"
  run ln -sfn "$ROOT/lingbot_map" "$LINK_DIR/lingbot_map"
  printf '    $ echo %q > %q\n' "$LINK_DIR" "$USER_SITE/lingbot_map_dev.pth"
  [ "$DRY" = 1 ] || echo "$LINK_DIR" > "$USER_SITE/lingbot_map_dev.pth"
fi

# --- Render offline -----------------------------------------------------------
if has render; then
  step "Render offline: ffmpeg"
  if command -v ffmpeg >/dev/null || [ -x "$BINDIR/ffmpeg" ]; then
    echo "    ffmpeg ya disponible"
  else
    [ "$(uname -m)" = "x86_64" ] || die "ffmpeg: instalarlo con el gestor de paquetes (no hay build estática para $(uname -m))"
    TMP=$(mktemp -d)
    run curl -fL --retry 5 -o "$TMP/ffmpeg.tar.xz" https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    run tar -xJf "$TMP/ffmpeg.tar.xz" -C "$TMP"
    run mkdir -p "$BINDIR"
    [ "$DRY" = 1 ] || install -m 755 "$TMP"/ffmpeg-*-static/ffmpeg "$TMP"/ffmpeg-*-static/ffprobe "$BINDIR/"
    rm -rf "$TMP"
    case ":$PATH:" in *":$BINDIR:"*) ;; *) echo "    AVISO: $BINDIR no está en el PATH" ;; esac
  fi

  step "Render offline: CUDA toolkit (nvcc)"
  CUDA_MAJOR=${TORCH#cu}; CUDA_MAJOR=${CUDA_MAJOR:0:2}
  find_cuda_home() {
    local c nvcc
    nvcc=$(command -v nvcc 2>/dev/null || true)
    for c in "${CUDA_HOME:-}" "${nvcc:+$(dirname "$(dirname "$(readlink -f "$nvcc")")")}" /usr/local/cuda \
             $(ls -d "$HOME"/cuda-* 2>/dev/null | sort -rV); do
      [ -n "$c" ] && [ -x "$c/bin/nvcc" ] || continue
      "$c/bin/nvcc" --version | grep -q "release $CUDA_MAJOR\." && { echo "$c"; return 0; }
    done
    return 1
  }
  if ! CUDA_HOME_FOUND=$(find_cuda_home); then
    case "$TORCH" in
      cu130) TK_VER="13.0.0"; TK_RUN="cuda_13.0.0_580.65.06_linux.run" ;;
      cu126) TK_VER="12.6.3"; TK_RUN="cuda_12.6.3_560.35.05_linux.run" ;;
    esac
    TK_DIR="$HOME/cuda-${TK_VER%.*}"
    TK_URL="https://developer.download.nvidia.com/compute/cuda/$TK_VER/local_installers/$TK_RUN"
    if [ "$INSTALL_CUDA" = 1 ]; then
      run curl -fL -C - --retry 10 --speed-limit 1024 --speed-time 60 -o "$HOME/$TK_RUN" "$TK_URL"
      # solo el toolkit: nunca el driver del instalador (degradaría el del sistema)
      run bash "$HOME/$TK_RUN" --silent --toolkit --toolkitpath="$TK_DIR" --no-opengl-libs --no-drm --override
      CUDA_HOME_FOUND="$TK_DIR"
    else
      die "falta un CUDA toolkit $CUDA_MAJOR.x (nvcc) para compilar. Opciones:
  ./setup_env.sh --profiles $PROFILES --install-cuda-toolkit   (~4 GB, sin sudo, queda en $TK_DIR)
  o exportar CUDA_HOME apuntando a un toolkit $CUDA_MAJOR.x existente"
    fi
  fi
  export CUDA_HOME="$CUDA_HOME_FOUND" PATH="$CUDA_HOME_FOUND/bin:$PATH"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$GPU_CAP}"
  [ -n "$TORCH_CUDA_ARCH_LIST" ] || die "no se pudo leer la compute capability; exportar TORCH_CUDA_ARCH_LIST (ej. 8.9)"
  echo "    CUDA_HOME=$CUDA_HOME  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

  step "Render offline: extensiones CUDA y Kaolin (compilación, varios minutos)"
  run bash -c "cd '$ROOT/demo_render/render_cuda_ext' && env -u PYTHONPATH '$PYBIN' setup.py build_ext --inplace"
  run "${PIP[@]}" "${C[@]}" --no-build-isolation "git+https://github.com/NVIDIAGameWorks/kaolin.git@$KAOLIN_REF"
fi

# --- Archivos de modelo -----------------------------------------------------------
if [ "$SKIP_ASSETS" = 0 ]; then
  step "Archivos de modelo (verificados con SHA256)"
  run "$PY" tools/fetch_assets.py --profiles "$PROFILES" "${SOURCE_DIRS[@]}"
fi

# --- Verificación ---------------------------------------------------------------
step "Verificación (tools/doctor.py)"
if [ "$DRY" = 1 ]; then
  echo "    (dry-run: se omite)"
  exit 0
fi
set +e
"$PYBIN" tools/doctor.py --profiles "$PROFILES"
STATUS=$?
set -e

step "Siguiente paso"
[ "$MODE" = "venv" ] && echo "    En cada terminal nueva:  source ${VENV#"$ROOT"/}/bin/activate"
if [ -r /proc/driver/nvidia/params ] && ls /sys/class/power_supply/BAT* >/dev/null 2>&1 \
   && ! grep -q 'PreserveVideoMemoryAllocations: 1' /proc/driver/nvidia/params; then
  echo "    Laptop, una sola vez por equipo:  sudo scripts_gpu/fix_nvidia_suspend.sh && sudo reboot"
fi
exit $STATUS
