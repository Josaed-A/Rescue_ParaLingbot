# Instalar este repositorio en otro equipo

Guía para clonar y dejar funcionando `Rescue_ParaLingbot` en una máquina nueva, o para reparar el entorno en la actual. Todo sale de tres archivos versionados: [env/constraints.txt](env/constraints.txt) (versiones validadas juntas), [env/requirements/](env/requirements/) (qué instala cada perfil) y [env/assets.json](env/assets.json) (archivos de modelo con su SHA256).

## Qué viaja con git y qué no

| Qué | ¿En git? | Cómo llega al equipo nuevo |
|---|---|---|
| Código, scripts, `env/`, `tools/` | Sí | `git clone` |
| `checkpoints/lingbot-map.pt`, `skyseg.onnx`, `skyseg_batch.onnx` | No (4.8 GB) | `tools/fetch_assets.py`: los descarga o los copia de otra carpeta, verificando SHA256 |
| Entorno Python (`.venv/`) | No | `setup_env.sh` lo crea |
| Extensiones compiladas (`demo_render/render_cuda_ext/*.so`), Kaolin | No | `setup_env.sh --profiles render` las compila para ese equipo (dependen de su Python, torch y CUDA) |
| Resultados (`captures/`) | No | Copiarlos a mano si hacen falta: `rsync -a equipo_viejo:Rescue_ParaLingbot/captures/ captures/` |
| Driver NVIDIA, corrección de suspensión | No (es del sistema) | Una vez por equipo, ver abajo |

## Requisitos del equipo

- Linux x86_64 con Python 3.10 a 3.13, `git` y `curl`. En Ubuntu, si falta el módulo venv: `sudo apt install python3.10-venv`.
- GPU NVIDIA opcional. Con driver ≥ 580 se instala torch cu130; con ≥ 560, cu126; con uno más viejo o sin GPU, la build de CPU.
- No hace falta `sudo` para nada más.

## Instalación

```bash
git clone https://github.com/Josaed-A/Rescue_ParaLingbot.git
cd Rescue_ParaLingbot
./setup_env.sh                        # crea .venv con core,vis,gpu y baja los modelos
source .venv/bin/activate
python tools/doctor.py                # debe terminar en "0 FALLA"
```

Para no descargar 4.8 GB, copiar los modelos desde el equipo viejo (un disco USB o la carpeta del repo viejo montada):

```bash
./setup_env.sh --source-dir /media/usb/Rescue_ParaLingbot
```

Si el equipo es una **laptop con GPU NVIDIA**, correr una sola vez (no hay que repetirlo al encender):

```bash
sudo scripts_gpu/fix_nvidia_suspend.sh && sudo reboot
```

Sin esto, suspender la laptop con un proceso de GPU abierto deja CUDA inutilizable hasta reiniciar.

`./setup_env.sh --dry-run` muestra todos los comandos sin ejecutar nada.

## Perfiles

Se combinan con `--profiles core,vis,gpu,render` o `--profiles all`. `core` siempre se incluye.

| Perfil | Para qué | Qué agrega |
|---|---|---|
| `core` | `lingbot_map` y `demo.py` | torch, numpy < 2, scipy, opencv, checkpoint |
| `vis` | Visor web, export GLB, `--mask_sky` | viser, trimesh, matplotlib, onnxruntime, `skyseg.onnx` |
| `gpu` | Monitoreo de `scripts_gpu/` y `scripts_seq/` | psutil, nvidia-ml-py |
| `flashinfer` | Atención con KV cache paginado (sin `--use_sdpa`) | flashinfer-python. No cabe en GPUs de 8 GB |
| `render` | Video MP4 offline de `demo_render/` | open3d, extensiones CUDA, Kaolin, ffmpeg, `skyseg_batch.onnx`. Necesita un CUDA toolkit: `--install-cuda-toolkit` lo baja a `~/cuda-X.Y` sin sudo (~4 GB) |
| `bench` | `benchmark/` | evo, OpenEXR, plyfile, open3d |

## Uso diario

```bash
source .venv/bin/activate
scripts_gpu/run_gpu.sh -- python demo.py --model_path checkpoints/lingbot-map.pt \
    --image_folder example/courthouse --mask_sky \
    --use_sdpa --num_scale_frames 2 --kv_cache_sliding_window 16 --offload_to_cpu
```

Esos flags son los validados para 8 GB de VRAM. `tools/doctor.py` indica cuáles usar según la GPU del equipo. `run_gpu.sh` revisa antes que no haya procesos viejos ocupando la GPU y bloquea la suspensión mientras el comando corre.

## Cuando algo falla

Correr `python tools/doctor.py` (o `--profiles all`). Cada `[FALLA]` trae debajo el comando que la corrige. Casos frecuentes:

- **Se movió o se renombró la carpeta del repo:** `import lingbot_map` carga otra copia o falla. Volver a correr `./setup_env.sh`; no reinstala lo que ya está.
- **Un `pip install` suelto subió numpy a 2.x:** instalar siempre con `-c env/constraints.txt`, que lo impide.
- **CUDA dejó de funcionar:** `scripts_gpu/gpu_preflight.sh` distingue entre un proceso viejo, una suspensión y un driver actualizado sin reiniciar.
- **Archivo de modelo incompleto o corrupto:** `python tools/fetch_assets.py` lo detecta por SHA256, lo aparta como `.invalid` y lo vuelve a obtener.

## Actualizar una versión

1. Probar el cambio en un entorno aparte: `./setup_env.sh --venv /tmp/venv-prueba --profiles all`.
2. Correr ahí `tools/doctor.py --profiles all` y una corrida real de `demo.py`.
3. Recién entonces editar la línea en `env/constraints.txt` y hacer commit.

## Instalación en modo usuario (sin venv)

La máquina de referencia está instalada así, en `~/.local`: `./setup_env.sh --mode user`. Funciona igual, pero comparte paquetes con el resto del sistema (en esa máquina, workspaces de ROS2). Para un equipo nuevo se recomienda el venv.

## Windows (no probado)

`setup_env.sh` es solo para Linux. `tools/doctor.py` y `tools/fetch_assets.py` sí funcionan en Windows. Instalación manual de `core` y `vis` en PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -c env\constraints.txt -r env\requirements\core.txt -r env\requirements\vis.txt
.venv\Scripts\python -m pip install --no-deps -e .
.venv\Scripts\python tools\fetch_assets.py
.venv\Scripts\python tools\doctor.py --profiles core,vis
```

En la línea de torch, elegir la variante según el equipo: `cu130` con driver NVIDIA ≥ 580, `cu126` con ≥ 560 y `cpu` sin GPU NVIDIA. Hay ruedas de torch 2.12.0 para Windows en las tres, con Python 3.10 a 3.13. Los perfiles `render` y `flashinfer` no están soportados en Windows.
