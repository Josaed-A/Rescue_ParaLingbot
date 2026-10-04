#!/usr/bin/env python3
"""Construye el mismo juego de mapas para una prueba, a partir de sus predicciones (.npz).

Una sola receta para todas las pruebas, así cada una queda con los mismos tipos de mapa
en las mismas carpetas (y el visor los encuentra igual):

  nube     exports/<name>_denso.ply                      nube fusionada por vóxel
  alta     exports/densidad_alta/<name>_denso_alta.ply   nube fusionada, más fina
  cruda    exports/webgl/<name>_raw.ply + _cameras.json  nube por frame + trayectoria
  malla    exports/malla/<name>_malla.{ply,glb}          superficie por fusión TSDF
  splat    exports/splat/<name>_splat.ply                Gaussian Splatting (gsplat)
  video    exports/render_ruta_completa/                 video del recorrido (batch_demo)

y, con el filtro geométrico (geo_filter.py; no cambia la nube ni las predicciones):

  filtro   exports/estructura/<name>_estructura.glb      paredes y piso por planos + esquinas,
           exports/estructura/<name>_filtro.npz          máscaras y ajuste a planos
  malla_f  exports/malla/<name>_filtrado_malla.{ply,glb} malla TSDF con el filtro
  splat_f  exports/splat/<name>_filtrado_splat.ply       Gaussian Splatting con el filtro

y una tarea previa opcional:

  windowed  reprocesa frames/ en modo windowed (ventana --window_size) y usa ESE npz para el
            resto. Es lo que conviene después de una sesión en vivo: en vivo sólo existe el
            modo streaming, que deriva (ver la bitácora del 2026-09-19).

Cada tarea corre como proceso aparte (si una falla, las demás siguen) e imprime una línea
"### <tarea> (i/n)" al empezar y "### ok|falla <tarea> <s>s" al terminar, que el servidor
del visor usa para mostrar el progreso.

    python scripts_context/build_maps.py captures/pruebas_reales/unisabana/prueba_3 \
        --npz eval/final_m2.npz --tasks nube,cruda,malla,splat
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TASKS = ["windowed", "filtro", "nube", "alta", "cruda", "malla", "malla_f", "splat", "splat_f", "video"]


def gpu_env():
    env = dict(os.environ)
    cuda = os.path.expanduser("~/cuda-13.0")
    if os.path.isdir(cuda):                      # gsplat compila sus kernels con nvcc
        env.setdefault("CUDA_HOME", cuda)
        env["PATH"] = os.path.join(cuda, "bin") + os.pathsep + env.get("PATH", "")
        env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    return env


PEAK = {"mb": 0}
FILTER_ARGS = []        # variante del filtro elegida por la medición del 2026-10-04 (ver la bitácora)


def run(cmd, env=None):
    """Corre una tarea y anota su RAM pico (ru_maxrss del hijo, vía os.wait4)."""
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, cwd=REPO)
    _, status, ru = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    PEAK["mb"] = max(PEAK["mb"], ru.ru_maxrss // 1024)
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prueba", help="carpeta de la prueba (captures/.../prueba_N)")
    ap.add_argument("--npz", default=None, help="predicciones, relativo a la prueba (por defecto el único eval/*.npz)")
    ap.add_argument("--name", default=None, help="prefijo de los archivos (por defecto el nombre del npz)")
    ap.add_argument("--tasks", default="nube,cruda,malla,splat")
    ap.add_argument("--voxel_nube", type=float, default=0.0004)
    ap.add_argument("--voxel_alta", type=float, default=0.00025)
    ap.add_argument("--window_size", type=int, default=24, help="para la tarea windowed")
    ap.add_argument("--splat_iters", type=int, default=7000)
    ap.add_argument("--force", action="store_true", help="rehacer aunque el archivo ya exista")
    a = ap.parse_args()

    P = os.path.abspath(a.prueba)
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        sys.exit(f"tareas desconocidas: {bad} (válidas: {TASKS})")
    tasks = [t for t in TASKS if t in tasks]       # orden fijo: windowed primero

    if a.npz:
        npz = os.path.join(P, a.npz)
    else:
        cands = sorted(f for f in os.listdir(os.path.join(P, "eval")) if f.endswith(".npz")) \
            if os.path.isdir(os.path.join(P, "eval")) else []
        if len(cands) != 1 and "windowed" not in tasks:
            sys.exit(f"no se puede elegir el npz solo ({cands}); pasá --npz")
        npz = os.path.join(P, "eval", cands[0]) if cands else None
    name = a.name or (os.path.splitext(os.path.basename(npz))[0] if npz else os.path.basename(P))
    py = sys.executable
    ex = os.path.join(P, "exports")
    results = []
    t_all = time.time()
    for n, t in enumerate(tasks, 1):
        print(f"### {t} ({n}/{len(tasks)})", flush=True)
        t0 = time.time()
        PEAK["mb"] = 0
        rc = 0
        if t == "windowed":
            frames = os.path.join(P, "frames")
            if not os.path.isdir(frames):
                print("no hay frames/ en la prueba"); rc = 1
            else:
                name = name + "_windowed" if not name.endswith("_windowed") else name
                out = os.path.join(P, "eval", name + ".npz")
                os.makedirs(os.path.dirname(out), exist_ok=True)   # pruebas viejas no traen eval/
                if os.path.isfile(out) and not a.force:
                    print("ya existe", out)
                else:
                    rc = run([py, "scripts_webcam/process_and_view.py", "--image_folder", frames,
                              "--model_path", "checkpoints/lingbot-map.pt", "--mode", "windowed",
                              "--window_size", str(a.window_size), "--use_sdpa", "--num_scale_frames", "2",
                              "--kv_cache_sliding_window", "16", "--camera_num_iterations", "4",
                              "--offload_to_cpu", "--keep_images_on_cpu", "--save_predictions", out,
                              "--save_ds", "1", "--no_serve"])
                if rc == 0:
                    npz = out
        elif npz is None or not os.path.isfile(npz):
            print("no hay predicciones (.npz) para esta tarea"); rc = 1
        elif t == "filtro":
            out = os.path.join(ex, "estructura", name)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.isfile(out + "_filtro.npz") and not a.force:
                print("ya existe", out + "_filtro.npz")
            else:
                rc = run([py, "scripts_context/geo_filter.py", npz, "--out", out], env=gpu_env())
        elif t in ("malla_f", "splat_f"):
            flt = os.path.join(ex, "estructura", name + "_filtro.npz")
            kind = "malla" if t == "malla_f" else "splat"
            out = os.path.join(ex, kind, name + "_filtrado_" + kind)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            done = out + (".glb" if kind == "malla" else ".ply")
            if not os.path.isfile(flt):
                print("falta el filtro: correr antes la tarea 'filtro'"); rc = 1
            elif os.path.isfile(done) and not a.force:
                print("ya existe", done)
            elif kind == "malla":
                rc = run([py, "scripts_context/tsdf_mesh.py", npz, "--out", out, "--filter", flt] + FILTER_ARGS)
            else:
                rc = run([py, "scripts_context/gsplat_train.py", npz, "--out", out, "--iters", str(a.splat_iters),
                          "--filter", flt] + FILTER_ARGS, env=gpu_env())
        elif t == "nube":
            out = os.path.join(ex, name + "_denso.ply")
            if os.path.isfile(out) and not a.force:
                print("ya existe", out)
            else:
                rc = run([py, "scripts_context/export_dense_cloud.py", npz, "--out_ply", out, "--float32",
                          "--voxel_rel", str(a.voxel_nube), "--conf_percentile", "35", "--chunk_frames", "20"])
        elif t == "alta":
            out = os.path.join(ex, "densidad_alta", name + "_denso_alta.ply")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.isfile(out) and not a.force:
                print("ya existe", out)
            else:
                rc = run([py, "scripts_context/export_dense_cloud.py", npz, "--out_ply", out, "--float32",
                          "--voxel_rel", str(a.voxel_alta), "--conf_percentile", "20", "--chunk_frames", "20"])
        elif t == "cruda":
            out = os.path.join(ex, "webgl", name + "_raw.ply")
            if os.path.isfile(out) and not a.force:
                print("ya existe", out)
            else:
                rc = run([py, "scripts_context/npz_to_webgl.py", npz, "--out_dir", os.path.join(ex, "webgl"),
                          "--name", name, "--mode", "raw", "--max_points_per_frame", "8000"])
        elif t == "malla":
            out = os.path.join(ex, "malla", name + "_malla")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.isfile(out + ".glb") and not a.force:
                print("ya existe", out + ".glb")
            else:
                rc = run([py, "scripts_context/tsdf_mesh.py", npz, "--out", out])
        elif t == "splat":
            out = os.path.join(ex, "splat", name + "_splat")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            if os.path.isfile(out + ".ply") and not a.force:
                print("ya existe", out + ".ply")
            else:
                rc = run([py, "scripts_context/gsplat_train.py", npz, "--out", out,
                          "--iters", str(a.splat_iters)], env=gpu_env())
        elif t == "video":
            od = os.path.join(ex, "render_ruta_completa")
            os.makedirs(od, exist_ok=True)
            tmp = os.path.join(od, name + "_render_input.npz")
            rc = run([py, "scripts_context/npz_for_render.py", npz, "--out", tmp])
            if rc == 0:
                rc = run([py, "scripts_context/render_route.py", "--load_predictions", tmp,
                          "--output_folder", od, "--downsample_factor", "5"], env=gpu_env())
            if os.path.isfile(tmp):
                os.remove(tmp)                   # intermedio grande y reproducible
        dt = time.time() - t0
        results.append((t, rc))
        print(f"### {'ok' if rc == 0 else 'falla'} {t} {dt:.0f}s RAM pico {PEAK['mb']} MB"
              + (" (código -9: lo mató el OOM killer)" if rc == -9 else ""), flush=True)
    print(f"### fin {time.time() - t_all:.0f}s " + " ".join(f"{t}={'ok' if rc == 0 else 'falla'}" for t, rc in results),
          flush=True)
    sys.exit(0 if all(rc == 0 for _, rc in results) else 1)


if __name__ == "__main__":
    main()
