"""Cámaras de este equipo y configuración de cámara de Stella para cualquier fuente.

1. `list_local_cameras()`: TODAS las cámaras V4L2 del equipo, por su nombre (el de /sys/class/video4linux),
   sin los nodos de metadatos que cada cámara UVC agrega (index 1). Cada una se prueba en paralelo y con
   reintentos (~2.5 s): algunas, como la Logitech B910, tardan ~1 s y entregan un primer JPEG corrupto, y la
   prueba anterior (una sola lectura) las daba por inservibles. Las que no dan imagen se listan igual, con
   el motivo. La cámara que está usando la sesión en vivo no se vuelve a abrir.

2. `stella_config_for(cfg)`: con el puente ROS2 activo, Stella necesita un yaml con los intrínsecos de la
   imagen que se le publica. Si la sesión no trae uno: se calcula el tamaño de la imagen publicada (fuente,
   rotación y reescalado), se busca en src/ros/stella/ un yaml con ese tamaño que corresponda a la fuente, y
   si no hay se genera uno en ~/.cache/paralingbot/stella/ con intrínsecos ESTIMADOS (campo de visión
   horizontal de 70°, centro óptico en el medio, sin distorsión). Fuentes grandes se publican reescaladas a
   lo sumo a 960 px de lado (Stella no gana precisión y pierde ritmo).
"""
import glob
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STELLA_DIR = os.path.join(REPO, "src", "ros", "stella")
AUTO_DIR = os.path.expanduser("~/.cache/paralingbot/stella")
MAX_SIDE_ROS = 960
HFOV_DEG = 70.0


def _num(p):
    return int(re.sub(r"\D", "", os.path.basename(p)) or 0)


def _by_id():
    out = {}
    for l in glob.glob("/dev/v4l/by-id/*"):
        try:
            out[os.path.realpath(l)] = l
        except OSError:
            pass
    return out


def _probe(dev, timeout=2.5):
    import cv2
    for fourcc in (None, "MJPG"):
        cap = cv2.VideoCapture(_num(dev), cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                return None, "no se puede abrir (¿ocupada por otro programa o sin permiso?)"
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            t0 = time.time()
            while time.time() - t0 < timeout:
                ok, fr = cap.read()
                if ok and fr is not None and fr.size:
                    return (int(fr.shape[1]), int(fr.shape[0])), None
                time.sleep(0.05)
        finally:
            cap.release()
    return None, "abre pero no entrega imagen"


def list_local_cameras(in_use=None):
    """[{kind, index, path, by_id, name, usable, w, h, label}] de todas las cámaras de este equipo."""
    byid = _by_id()
    nodes = []
    for s in sorted(glob.glob("/sys/class/video4linux/video*"), key=_num):
        dev = "/dev/" + os.path.basename(s)
        idx = open(os.path.join(s, "index")).read().strip() if os.path.isfile(os.path.join(s, "index")) else "0"
        if idx not in ("", "0"):
            continue                                   # nodo de metadatos de una cámara UVC
        name = open(os.path.join(s, "name")).read().strip() if os.path.isfile(os.path.join(s, "name")) else dev
        name = name.split(":")[0].strip() if name.count(":") == 1 and name.split(":")[0] == name.split(":")[1].strip() else name
        if name.lower().startswith("uvc camera") or not name.strip():
            # nombre genérico del driver: usar el del dispositivo USB (p. ej. "HD Webcam B910")
            usb = os.path.dirname(os.path.realpath(os.path.join(s, "device")))
            prod = open(os.path.join(usb, "product")).read().strip() if os.path.isfile(os.path.join(usb, "product")) else ""
            man = open(os.path.join(usb, "manufacturer")).read().strip() if os.path.isfile(os.path.join(usb, "manufacturer")) else ""
            if prod:
                name = f"{man} {prod}".strip() if man and man.lower() not in prod.lower() else prod
            else:
                m = re.search(r"([0-9a-f]{4}):([0-9a-f]{4})", name, re.I)
                if m:                                   # sin texto en el descriptor: base de datos de lsusb
                    try:
                        import subprocess
                        o = subprocess.run(["lsusb", "-d", f"{m.group(1)}:{m.group(2)}"], capture_output=True,
                                           text=True, timeout=3).stdout.strip().splitlines()
                        if o:
                            name = o[0].split(f"{m.group(1)}:{m.group(2)}", 1)[1].strip() or name
                    except Exception:
                        pass
        nodes.append({"kind": "webcam", "index": _num(dev), "path": dev, "by_id": byid.get(dev), "name": name})

    def run(n):
        if in_use is not None and str(in_use) in (n["path"], str(n["index"])):
            n.update(usable=False, label=f"{n['name']} ({n['path']}) — en uso por la sesión en vivo")
            return n
        wh, why = _probe(n["path"])
        if wh:
            n.update(usable=True, w=wh[0], h=wh[1], label=f"{n['name']} ({n['path']}, {wh[0]}x{wh[1]})")
        else:
            n.update(usable=False, label=f"{n['name']} ({n['path']}) — {why}")
        return n
    if not nodes:
        return []
    with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        return list(ex.map(run, nodes))


# ---------------------------------------------------------------------------
# Stella: yaml de cámara para la imagen publicada
# ---------------------------------------------------------------------------
def _source_size(cfg):
    """(w, h) de la imagen que entrega la fuente, ya rotada; None si no se puede saber sin abrirla."""
    rot = int(cfg.get("rotation") or 0) % 360
    src = cfg.get("source")
    wh = None
    if src in ("android", "ssh"):
        wh = tuple(int(v) for v in str(cfg.get("cam_size", "1280x720")).lower().split("x"))
        if src == "android":
            rot = rot                                   # AndroidCamera ya rota: el tamaño final gira igual
    elif src == "webcam":
        wh = cfg.get("_wh")
        if not wh:
            r, _ = _probe(f"/dev/video{int(cfg.get('device') or 0)}")
            wh = r
    elif src in ("video", "url"):
        import cv2
        cap = cv2.VideoCapture(cfg.get("path"))
        ok, fr = cap.read()
        cap.release()
        wh = (int(fr.shape[1]), int(fr.shape[0])) if ok else None
    elif src == "folder":
        import cv2
        fs = sorted(f for f in os.listdir(cfg["path"]) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        im = cv2.imread(os.path.join(cfg["path"], fs[0])) if fs else None
        wh = (int(im.shape[1]), int(im.shape[0])) if im is not None else None
    if wh and rot in (90, 270):
        wh = (wh[1], wh[0])
    return wh


def _yaml_size(path):
    try:
        t = open(path).read()
    except OSError:
        return None, ""
    c = re.search(r"^\s*cols:\s*(\d+)", t, re.M)
    r = re.search(r"^\s*rows:\s*(\d+)", t, re.M)
    n = re.search(r'^\s*name:\s*"([^"]*)"', t, re.M)
    return ((int(c.group(1)), int(r.group(1))) if c and r else None), (n.group(1) if n else "")


def _hint(cfg):
    src = cfg.get("source")
    if src == "android":
        return "celular"
    if src == "webcam":
        return "webcam"
    p = (cfg.get("path") or "").lower()
    if "prueba_4" in p:
        return "prueba4"
    if "unisabana" in p:
        return "unisabana"
    return ""


def stella_config_for(cfg):
    """(ruta del yaml, 'WxH' para ros2_resize o None, nota). Respeta un stella_config ya dado."""
    if cfg.get("stella_config"):
        return cfg["stella_config"], cfg.get("ros2_resize"), "configuración indicada"
    wh = _source_size(cfg)
    if not wh:
        return None, None, "no se pudo saber el tamaño de la imagen de la fuente"
    resize = cfg.get("ros2_resize")
    if resize:
        wh = tuple(int(v) for v in str(resize).lower().split("x"))
    elif max(wh) > MAX_SIDE_ROS:
        k = MAX_SIDE_ROS / max(wh)
        wh = (int(round(wh[0] * k / 2) * 2), int(round(wh[1] * k / 2) * 2))
        resize = f"{wh[0]}x{wh[1]}"
    hint = _hint(cfg)
    best = None
    for y in sorted(glob.glob(os.path.join(STELLA_DIR, "*.yaml"))):
        size, name = _yaml_size(y)
        if size == tuple(wh):
            score = (hint and (hint in os.path.basename(y).lower() or hint in name.lower())) and 2 or 1
            if best is None or score > best[0]:
                best = (score, y)
    if best and best[0] == 2:
        return os.path.relpath(best[1], REPO), resize, f"configuración del repo para {wh[0]}x{wh[1]}"
    # generar una con intrínsecos estimados (a partir de la de la webcam, que tiene los parámetros ORB/mapeo)
    os.makedirs(AUTO_DIR, exist_ok=True)
    out = os.path.join(AUTO_DIR, f"auto_{wh[0]}x{wh[1]}.yaml")
    tmpl = open(os.path.join(STELLA_DIR, "webcam_640x480.yaml")).read()
    f = wh[0] / (2 * math.tan(math.radians(HFOV_DEG) / 2))
    vals = {"fx": f, "fy": f, "cx": wh[0] / 2, "cy": wh[1] / 2, "k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0}
    for k, v in vals.items():
        tmpl = re.sub(rf"(^\s*{k}:\s*)[-\d.e]+", rf"\g<1>{v:.1f}", tmpl, count=1, flags=re.M)
    tmpl = re.sub(r"(^\s*cols:\s*)\d+", rf"\g<1>{wh[0]}", tmpl, count=1, flags=re.M)
    tmpl = re.sub(r"(^\s*rows:\s*)\d+", rf"\g<1>{wh[1]}", tmpl, count=1, flags=re.M)
    tmpl = re.sub(r'(^\s*name:\s*)"[^"]*"', rf'\1"auto {wh[0]}x{wh[1]} (intrínsecos ESTIMADOS, FoV {HFOV_DEG:g}°)"',
                  tmpl, count=1, flags=re.M)
    open(out, "w").write(tmpl)
    return out, resize, f"configuración generada para {wh[0]}x{wh[1]} (intrínsecos estimados, sin calibrar)"


if __name__ == "__main__":
    for c in list_local_cameras():
        print(("OK   " if c["usable"] else "--   ") + c["label"])
