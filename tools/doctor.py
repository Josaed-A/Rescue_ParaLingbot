#!/usr/bin/env python3
"""Check that this checkout can run on the current machine and say how to fix what can't.

Run it with the same interpreter you use for demo.py (the venv's python, or python3).
Standard library only at module level, so it works even when the environment is broken;
heavy packages are imported in child processes so a crashing extension can't take the
doctor down. Linux and Windows; GPU and laptop-suspend checks are skipped where they
don't apply.

Usage:
  python3 tools/doctor.py                          # profiles core,vis,gpu
  python3 tools/doctor.py --profiles all
  python3 tools/doctor.py --verify-assets          # also SHA256 the model files (~30 s)
Exit code 0 = no FALLA, 1 = something must be fixed.
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(ROOT, "env")
ALL_PROFILES = ["core", "vis", "gpu", "flashinfer", "render", "bench"]

# Import names probed per profile (distribution names come from env/requirements/<profile>.txt).
IMPORTS = {
    "core": ["numpy", "scipy", "cv2", "PIL", "einops", "safetensors", "huggingface_hub", "tqdm",
             "torch", "torchvision"],
    "vis": ["matplotlib", "trimesh", "onnxruntime", "requests", "viser"],
    "gpu": ["psutil", "pynvml"],
    "flashinfer": ["flashinfer"],
    "render": ["open3d", "yaml", "aiohttp", "kaolin"],
    "bench": ["open3d", "plyfile", "evo", "OpenEXR", "Imath", "yaml"],
}
# Oldest Linux driver for each CUDA runtime a torch wheel can be built against.
MIN_DRIVER = {"11.8": 520, "12.1": 530, "12.4": 550, "12.6": 560, "12.8": 570, "13.0": 580}

counts = {"OK": 0, "AVISO": 0, "FALLA": 0}


def report(level, msg, fix=None):
    if level in counts:
        counts[level] += 1
    print(f"{'[' + level + ']':8s}{msg}")
    for line in (fix or "").splitlines():
        print(f"        -> {line}")


def section(title):
    print(f"\n== {title}")


def run(cmd, env=None, cwd=None, timeout=600):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, "", str(e)


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def in_venv():
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def pip_cmd(spec):
    user = "" if in_venv() else "--user "
    return f'{sys.executable} -m pip install {user}-c env/constraints.txt "{spec}"'


def setup_cmd(profile):
    if os.name == "nt":
        return f"ver SETUP.md (Windows), perfil {profile}"
    mode = "" if in_venv() else " --mode user"
    return f"./setup_env.sh --profiles {profile}{mode}"


def req_names(profile):
    path = os.path.join(ENV_DIR, "requirements", f"{profile}.txt")
    names = []
    if os.path.exists(path):
        for line in open(path):
            line = line.split("#")[0].strip()
            if line:
                names.append(re.split(r"[<>=!~\[; ]", line)[0])
    return names


def constraints():
    pins = {}
    for line in open(os.path.join(ENV_DIR, "constraints.txt")):
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)", line.split("#")[0].strip())
        if m:
            pins[norm(m.group(1))] = m.group(2)
    return pins


def child_json(code, args=(), env=None, cwd=None, timeout=600):
    rc, out, err = run([sys.executable, "-c", code, *args], env=env, cwd=cwd, timeout=timeout)
    for line in out.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:]), rc, err
    return None, rc, err


# ---------------------------------------------------------------------------

def check_python():
    section("Python")
    where = f"venv {sys.prefix}" if in_venv() else f"sin venv ({sys.executable})"
    if (3, 10) <= sys.version_info[:2] <= (3, 13):
        report("OK", f"Python {platform.python_version()}, {where}")
    else:
        report("FALLA", f"Python {platform.python_version()}: torch 2.12.0 solo publica ruedas para 3.10-3.13",
               "Instalar Python 3.10-3.13 y repetir la instalación")


def check_packages(profiles):
    section("Paquetes (contra las versiones validadas de env/constraints.txt)")
    pins = constraints()
    seen = set()
    for prof in profiles:
        names = (["torch", "torchvision"] if prof == "core" else []) + req_names(prof)
        names += ["kaolin"] if prof == "render" else []
        for name in names:
            key = norm(name)
            if key in seen:
                continue
            seen.add(key)
            special = name in ("torch", "torchvision", "kaolin")
            try:
                ver = metadata.version(name)
            except metadata.PackageNotFoundError:
                report("FALLA", f"{name}: no instalado (perfil {prof})",
                       setup_cmd(prof) if special else pip_cmd(f"{name}=={pins[key]}" if key in pins else name))
                continue
            pin = pins.get(key)
            if pin is None or ver.split("+")[0] == pin:
                report("OK", f"{name} {ver}")
            else:
                report("AVISO", f"{name} {ver} (validada: {pin})",
                       setup_cmd(prof) if special else pip_cmd(f"{name}=={pin}"))
    try:
        if int(metadata.version("numpy").split(".")[0]) >= 2:
            report("FALLA", "numpy >= 2: rompe scipy 1.14 y kaolin (compilados contra numpy 1.x)",
                   pip_cmd("numpy==1.26.4"))
    except metadata.PackageNotFoundError:
        pass


def check_pip_conflicts(profiles):
    section("Conflictos declarados (pip check)")
    ours = {norm(n) for p in profiles for n in req_names(p)} | {"torch", "torchvision", "kaolin", "numpy"}
    rc, out, err = run([sys.executable, "-m", "pip", "check"], timeout=180)
    if rc == -1 or "No module named pip" in err:
        report("AVISO", "no se pudo correr pip check", err.strip()[:200])
        return
    pins = constraints()
    hits = [line for line in out.splitlines() if line.strip() and norm(line.split()[0]) in ours]
    for line in hits:
        pkg = line.split()[0]
        fix = pip_cmd(f"{pkg}=={pins[norm(pkg)]}") if norm(pkg) in pins else pip_cmd(pkg)
        report("FALLA", line.strip(), fix + "\n(un pip install futuro podría 'arreglarlo' subiendo numpy a 2.x)")
    if not hits:
        report("OK", "sin conflictos entre los paquetes de este repo")


IMPORT_PROBE = r'''
import importlib, json, sys, warnings
out = {}
for name in sys.argv[1:]:
    mod = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            mod = importlib.import_module(name)
            err = None
        except BaseException as e:
            text = str(e).strip()
            err = "%s: %s" % (type(e).__name__, text.splitlines()[-1] if text else "")
    bad = [str(w.message).splitlines()[0] for w in caught
           if "numpy" in str(w.message).lower() and any(k in str(w.message).lower() for k in ("version", "compiled", "dtype size"))]
    out[name] = {"err": err if err else ("WARN " + bad[0] if bad else None),
                 "file": getattr(mod, "__file__", None)}
print("@@" + json.dumps(out))
'''


def probe_imports(mods, cwd=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # importing must not touch the GPU
    res, rc, err = child_json(IMPORT_PROBE, mods, env=env, cwd=cwd)
    if res is not None:
        return res
    if len(mods) == 1:
        last = (err.strip().splitlines() or [""])[-1]
        return {mods[0]: {"err": f"el proceso terminó con código {rc}: {last}", "file": None}}
    merged = {}
    for m in mods:  # something crashed hard: isolate it
        merged.update(probe_imports([m], cwd))
    return merged


def check_imports(profiles):
    section("Imports (proceso aparte, sin tocar la GPU)")
    owner = {}
    for p in profiles:
        for m in IMPORTS.get(p, []):
            owner.setdefault(m, p)
    # ROS2's setup.bash exports PYTHONPATH with its own site-packages; they leak into any venv
    shadow = [os.path.realpath(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if shadow:
        report("INFO", f"PYTHONPATH tiene {len(shadow)} carpetas (¿ROS2 cargado?): se revisa que no tapen paquetes")
    results = probe_imports(list(owner))
    for m, prof in owner.items():
        res = results.get(m) or {"err": "sin resultado", "file": None}
        r = res["err"]
        src = os.path.realpath(res["file"]) if res["file"] else ""
        hit = next((s for s in shadow if src.startswith(s + os.sep)), None)
        if r is None and hit:
            report("AVISO", f"import {m} se carga desde PYTHONPATH, no desde este entorno: {src}",
                   "correr en una terminal sin ROS2 cargado, o: env -u PYTHONPATH python ...")
        elif r is None:
            report("OK", f"import {m}")
        elif "numpy" in r.lower() and ("dtype size" in r or "_ARRAY_API" in r or r.startswith("WARN")):
            report("FALLA", f"import {m}: {r.removeprefix('WARN ')}",
                   "numpy y un paquete compilado no coinciden: " + pip_cmd("numpy==1.26.4")
                   + "\ny revisar la versión de " + m + " en la sección Paquetes")
        else:
            report("FALLA", f"import {m}: {r}", setup_cmd(prof))


def check_repo_package(profiles):
    section("lingbot_map de ESTE repo")
    expected = os.path.realpath(os.path.join(ROOT, "lingbot_map"))
    code = "import json, os, lingbot_map; print('@@' + json.dumps(os.path.realpath(os.path.dirname(lingbot_map.__file__))))"
    got, rc, err = child_json(code, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), cwd=tempfile.gettempdir())
    fix = (f"{sys.executable} -m pip install --no-deps -e {ROOT}" if in_venv()
           else setup_cmd("core") + "   (rehace el link hacia esta carpeta)")
    if got is None:
        report("AVISO", "lingbot_map solo se importa desde la raíz del repo", fix)
    elif got != expected:
        report("FALLA", f"'import lingbot_map' carga OTRA copia: {got}", fix + "\n(pasa al mover o clonar el repo en otra carpeta)")
    else:
        report("OK", f"import lingbot_map desde cualquier carpeta -> {got}")
    # lingbot_map.vis needs matplotlib/viser: only probe it when the vis profile was asked for
    mods = ["lingbot_map.models.gct_stream"] + (["lingbot_map.vis"] if "vis" in profiles else [])
    res = probe_imports(mods, cwd=ROOT)
    for m, r in res.items():
        if r["err"] is None:
            report("OK", f"import {m}")
        else:
            report("FALLA", f"import {m}: {r['err']}", setup_cmd("core,vis"))


GPU_PROBE = r'''
import json, torch
d = {"torch": torch.__version__, "cuda": torch.version.cuda, "ok": False}
try:
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
        free, total = torch.cuda.mem_get_info()
        d.update(ok=True, free_mib=free // 2**20, total_mib=total // 2**20)
    else:
        d["error"] = "torch.cuda.is_available() == False"
except Exception as e:
    d["error"] = (str(e).strip().splitlines() or [type(e).__name__])[0]
print("@@" + json.dumps(d))
'''


def nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    rc, out, err = run([exe, "--query-gpu=driver_version,name,memory.total,compute_cap",
                        "--format=csv,noheader,nounits"], timeout=60)
    text = (out + err).strip()
    if rc != 0:
        return {"error": text.splitlines()[-1] if text else "nvidia-smi falló"}
    drv, name, mem, cap = [x.strip() for x in out.splitlines()[0].split(",")]
    return {"driver": drv, "name": name, "mem_mib": int(float(mem)), "cap": cap}


def check_gpu(profiles):
    section("GPU / CUDA")
    smi = nvidia_smi()
    if smi is None:
        report("INFO", "sin GPU NVIDIA (no hay nvidia-smi): correr en CPU",
               'CUDA_VISIBLE_DEVICES="" python3 demo.py ... --use_sdpa --camera_num_iterations 1')
        return None
    if "error" in smi:
        fix = "reiniciar el equipo (driver actualizado sin recargar el módulo)" if "mismatch" in smi["error"].lower() \
            else "revisar la instalación del driver NVIDIA"
        report("FALLA", f"nvidia-smi: {smi['error']}", fix)
        return None
    report("OK", f"{smi['name']}, {smi['mem_mib']} MiB, driver {smi['driver']}, capability {smi['cap']}")
    t, rc, err = child_json(GPU_PROBE, timeout=300)
    if t is None:
        report("FALLA", "no se pudo importar torch para probar CUDA", setup_cmd("core"))
        return None
    if not t["cuda"]:
        report("AVISO", f"torch {t['torch']} es la build de CPU aunque hay GPU NVIDIA", setup_cmd("core") + " --torch auto")
        return t
    drv_major = int(smi["driver"].split(".")[0])
    need = MIN_DRIVER.get(t["cuda"])
    if need and drv_major < need:
        variant = "cu130" if drv_major >= 580 else "cu126" if drv_major >= 560 else "cpu"
        report("FALLA", f"torch {t['torch']} (CUDA {t['cuda']}) necesita driver >= {need}; este equipo tiene {smi['driver']}",
               f"{setup_cmd('core')} --torch {variant}   (o actualizar el driver)")
    if t["ok"]:
        report("OK", f"CUDA usable desde torch {t['torch']}: {t['free_mib']}/{t['total_mib']} MiB libres")
    else:
        fix = ("si el equipo se suspendió: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm (o reiniciar)\n"
               "scripts_gpu/gpu_preflight.sh muestra procesos que retienen la GPU") if sys.platform.startswith("linux") \
            else "reiniciar el equipo"
        report("FALLA", f"torch no puede usar CUDA: {t.get('error')}", fix)
    mem = smi["mem_mib"]
    if mem <= 8700:
        report("INFO", f"{mem} MiB de VRAM: usar los flags validados para 8 GB",
               "--use_sdpa --num_scale_frames 2 --kv_cache_sliding_window 16 --offload_to_cpu"
               "\n(+ --keep_images_on_cpu en scripts_webcam/process_and_view.py)")
    else:
        report("INFO", f"{mem} MiB de VRAM: configuración no medida en este repo",
               "empezar con los flags de 8 GB y subir --kv_cache_sliding_window / --num_scale_frames mientras no haya OOM")
    if "flashinfer" in profiles and mem < 16000:
        report("AVISO", "FlashInfer preasigna ~12.9 GiB de KV cache a 518x518 (medido): no cabe en esta GPU",
               "correr con --use_sdpa")
    return t


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_assets(profiles, verify):
    section("Archivos de modelo (gitignored, ver env/assets.json)")
    with open(os.path.join(ENV_DIR, "assets.json")) as f:
        manifest = json.load(f)
    for name, a in manifest.items():
        if not set(a["profiles"]) & set(profiles):
            continue
        path = os.path.join(ROOT, a["dest"])
        fix = f"{os.path.basename(sys.executable)} tools/fetch_assets.py --profiles {','.join(profiles)}" \
              "   (--source-dir <otra copia> para no descargar)"
        if not os.path.isfile(path):
            report("FALLA", f"{a['dest']}: falta ({a['used_by']})", fix)
        elif os.path.getsize(path) != a["size"]:
            report("FALLA", f"{a['dest']}: {os.path.getsize(path)} bytes, esperado {a['size']} (copia incompleta)", fix)
        elif verify and sha256_of(path) != a["sha256"]:
            report("FALLA", f"{a['dest']}: SHA256 no coincide (archivo distinto o corrupto)", fix)
        else:
            report("OK", f"{a['dest']}: " + ("SHA256 verificado" if verify else "tamaño correcto (SHA256 con --verify-assets)"))


def find_cuda_home(major):
    exe = ".exe" if os.name == "nt" else ""
    cands = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    nvcc = shutil.which("nvcc")
    if nvcc:
        cands.append(os.path.dirname(os.path.dirname(os.path.realpath(nvcc))))
    cands += ["/usr/local/cuda"] + sorted(glob.glob(os.path.expanduser("~/cuda-*")), reverse=True)
    found = []
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "bin", "nvcc" + exe)):
            rc, out, err = run([os.path.join(c, "bin", "nvcc" + exe), "--version"], timeout=60)
            m = re.search(r"release (\d+)\.(\d+)", out)
            if m:
                found.append((c, f"{m.group(1)}.{m.group(2)}"))
                if major and m.group(1) == major:
                    return c, found
    return None, found


def check_render(torch_info):
    section("Render offline (demo_render/)")
    ff = shutil.which("ffmpeg")
    report("OK", f"ffmpeg: {ff}") if ff else report("FALLA", "ffmpeg no está en el PATH", setup_cmd("render"))
    major = (torch_info or {}).get("cuda", "") or ""
    major = major.split(".")[0]
    home, found = find_cuda_home(major)
    if home:
        report("OK", f"CUDA toolkit para compilar: {home}")
    elif found:
        report("FALLA", f"toolkits encontrados {found}, pero torch usa CUDA {major}.x",
               setup_cmd("render") + " --install-cuda-toolkit")
    else:
        report("FALLA", "no hay CUDA toolkit (nvcc): solo hace falta para compilar las extensiones y Kaolin",
               setup_cmd("render") + " --install-cuda-toolkit   (~4 GB, sin sudo)")
    ext_dir = os.path.join(ROOT, "demo_render", "render_cuda_ext")
    tag = f"cpython-{sys.version_info[0]}{sys.version_info[1]}"
    built = [os.path.basename(p) for p in glob.glob(os.path.join(ext_dir, "*_ext*.so")) + glob.glob(os.path.join(ext_dir, "*_ext*.pyd"))]
    for mod in ("voxel_morton_ext", "frustum_cull_ext"):
        files = [b for b in built if b.startswith(mod)]
        if not files:
            report("FALLA", f"{mod}: no compilada", setup_cmd("render"))
        elif not any(tag in b or b.endswith(".pyd") for b in files):
            report("FALLA", f"{mod}: compilada para otro Python ({files}), este es {tag}", setup_cmd("render"))
    if len(built) >= 2:
        code = ("import json, sys, torch; sys.path.insert(0, %r)\n"
                "import voxel_morton_ext, frustum_cull_ext\nprint('@@' + json.dumps('ok'))") % ext_dir
        got, rc, err = child_json(code, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        if got == "ok":
            report("OK", "extensiones CUDA importan con este torch")
        else:
            report("FALLA", "las extensiones no importan: " + ((err.strip().splitlines() or ["?"])[-1]),
                   "compiladas contra otro torch/CUDA: " + setup_cmd("render"))


def check_suspend():
    params = "/proc/driver/nvidia/params"
    if not os.path.exists(params) or not glob.glob("/sys/class/power_supply/BAT*"):
        return
    section("Suspensión (laptop con GPU NVIDIA)")
    m = re.search(r"PreserveVideoMemoryAllocations: (\d+)", open(params).read())
    if m and m.group(1) == "1":
        report("OK", "PreserveVideoMemoryAllocations=1: suspender no rompe CUDA")
    else:
        report("AVISO", "suspender con un proceso CUDA vivo deja CUDA inutilizable hasta reiniciar",
               "una sola vez por equipo: sudo scripts_gpu/fix_nvidia_suspend.sh && sudo reboot")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profiles", default="core,vis,gpu", help="separados por coma, o 'all': " + ",".join(ALL_PROFILES))
    p.add_argument("--verify-assets", action="store_true", help="calcular SHA256 de los archivos de modelo")
    args = p.parse_args()
    profiles = ALL_PROFILES if args.profiles == "all" else args.profiles.split(",")
    unknown = [x for x in profiles if x not in ALL_PROFILES]
    if unknown:
        p.error(f"perfiles desconocidos: {unknown}")
    if "core" not in profiles:
        profiles = ["core"] + profiles

    print(f"Repo: {ROOT}\nPerfiles: {','.join(profiles)}  |  {platform.system()} {platform.machine()}")
    check_python()
    check_packages(profiles)
    check_pip_conflicts(profiles)
    check_imports(profiles)
    check_repo_package(profiles)
    torch_info = check_gpu(profiles)
    check_assets(profiles, args.verify_assets)
    if "render" in profiles:
        check_render(torch_info)
    check_suspend()

    print(f"\nResumen: {counts['OK']} OK, {counts['AVISO']} AVISO, {counts['FALLA']} FALLA")
    return 1 if counts["FALLA"] else 0


if __name__ == "__main__":
    sys.exit(main())
