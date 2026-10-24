"""Cámaras de otros equipos por SSH (la Pi del robot GARDIAN, vigia-1, cualquier Linux con una cámara)
como fuente del mapeo en vivo.

Es el mismo método con que GARDIAN mira la cámara de vigia-1 (`~/Rescue/vigia_cam/vigia_cam_view.py`):
en el equipo remoto, ffmpeg entrega MJPEG por la salida estándar; llega por SSH y aquí se decodifica.

  - cámara UVC/V4L2 (webcam, Logitech de la Pi, RGB de la Astra): si la cámara da MJPEG, ffmpeg lo copia
    sin recomprimir (`-c:v copy`, casi sin CPU en la Pi); si no, comprime YUYV a MJPEG.
  - cámara MIPI de vigia-1 (unicam, IMX296 mono): `~/bin/y10cap` (se sube y compila si falta, fuente en
    ~/Rescue/vigia_cam/y10cap.c) | ffmpeg MJPEG, igual que vigia_cam_view.py.

Descubrimiento: los hosts de ~/.ssh/config más los equipos recordados en
~/.config/paralingbot/camaras_ssh.json. Cada equipo en que se encontró una cámara se guarda ahí
("dispositivo recurrente"): la próxima vez aparece en la lista aunque esté apagado, marcado sin conexión.
Se usa `ssh -o BatchMode=yes` (sólo llaves, nunca pide contraseña) con un tiempo de conexión corto.

Ojo: si en el robot la cámara la tiene abierta un nodo de ROS2 (la estación de GARDIAN), V4L2 la da por
ocupada; ahí el error lo dice.
"""
import json
import os
import re
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

CONF = os.path.expanduser("~/.config/paralingbot/camaras_ssh.json")
SSH_CONFIG = os.path.expanduser("~/.ssh/config")
Y10CAP_SRC = os.path.expanduser("~/Rescue/vigia_cam/y10cap.c")
# equipos que conviene probar aunque no estén en ~/.ssh/config (se pueden agregar más en CONF)
DEFAULT_HOSTS = [{"host": "vigia-1@192.168.80.33", "nombre": "vigia-1 (Pi Zero 2 W, IMX296)"}]
# nodos V4L2 que no son cámaras (codificadores, ISP y metadatos de la Raspberry Pi)
NOT_CAMERA = re.compile(r"bcm2835-codec|bcm2835-isp|rpivid|pispbe|rp1-cfe-(?!.*image)|-meta|hevc", re.I)
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2"]

PROBE = r"""
echo "H|$(hostname)"
for s in /sys/class/video4linux/video*; do
  [ -e "$s" ] || continue
  d=/dev/$(basename "$s"); echo "V|$d|$(cat "$s/index" 2>/dev/null)|$(cat "$s/name" 2>/dev/null)"
  if command -v v4l2-ctl >/dev/null; then
    f=$(v4l2-ctl -d "$d" --list-formats 2>/dev/null | grep -o "'[A-Z0-9 ]*'" | tr -d "' " | tr '\n' ',')
    echo "F|$d|$f"
  fi
done
command -v ffmpeg >/dev/null && echo "X|ffmpeg" || true
[ -x "$HOME/bin/y10cap" ] && echo "X|y10cap" || true
"""


# ---------------------------------------------------------------------------
# equipos recordados
# ---------------------------------------------------------------------------
def _load_conf():
    try:
        with open(CONF) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_conf(d):
    os.makedirs(os.path.dirname(CONF), exist_ok=True)
    tmp = CONF + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    os.replace(tmp, CONF)


def ssh_config_hosts(path=SSH_CONFIG):
    """Alias de ~/.ssh/config (sin comodines). Un bloque con varios alias cuenta una vez."""
    out = []
    try:
        for line in open(path):
            m = re.match(r"\s*Host\s+(.+)", line, re.I)
            if m:
                names = [h for h in m.group(1).split() if not any(c in h for c in "*?!")]
                if names:
                    out.append(names[0])
    except OSError:
        pass
    return out


def candidate_hosts():
    conf = _load_conf()
    hosts = {}
    for h in ssh_config_hosts():
        hosts[h] = {"host": h}
    for h in DEFAULT_HOSTS + conf.get("equipos_extra", []):
        hosts.setdefault(h["host"], dict(h))
    for h, info in conf.get("recordados", {}).items():
        hosts.setdefault(h, {"host": h})
        hosts[h]["recordado"] = info
    for h in conf.get("ignorar", []):
        hosts.pop(h, None)
    return list(hosts.values())


# ---------------------------------------------------------------------------
# descubrimiento
# ---------------------------------------------------------------------------
def _ssh(host, cmd, timeout=12, ssh_prefix=None):
    pre = ssh_prefix if ssh_prefix is not None else ["ssh", *SSH_OPTS, host]
    return subprocess.run(pre + [cmd], capture_output=True, text=True, timeout=timeout)


def probe_host(host, ssh_prefix=None, timeout=8):
    """{'ok', 'hostname', 'ffmpeg', 'y10cap', 'cameras': [{device, name, formats, tipo}], 'error'}"""
    try:
        r = _ssh(host, PROBE, timeout=timeout, ssh_prefix=ssh_prefix)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "sin respuesta"}
    if r.returncode != 0 and not r.stdout:
        err = (r.stderr.strip().splitlines() or ["error de ssh"])[-1]
        return {"ok": False, "error": err[:120]}
    info = {"ok": True, "hostname": host, "ffmpeg": False, "y10cap": False, "cameras": []}
    vids, fmts = {}, {}
    for line in r.stdout.splitlines():
        p = line.split("|")
        if p[0] == "H" and len(p) > 1:
            info["hostname"] = p[1]
        elif p[0] == "V" and len(p) >= 4:
            vids[p[1]] = (p[2], p[3])
        elif p[0] == "F" and len(p) >= 3:
            fmts[p[1]] = [f for f in p[2].split(",") if f]
        elif p[0] == "X" and len(p) > 1:
            info[p[1]] = True
    for dev, (idx, name) in sorted(vids.items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 0)):
        if idx not in ("0", "") or NOT_CAMERA.search(name):
            continue
        f = fmts.get(dev, [])
        if "unicam" in name.lower():
            tipo = "y10cap"                     # MIPI en la Pi (vigia-1): sólo con y10cap
        elif f and not ({"MJPG", "YUYV", "YUY2"} & set(f)):
            continue                            # nodo sin formato de imagen utilizable
        else:
            tipo = "mjpeg" if "MJPG" in f or not f else "yuyv"
        info["cameras"].append({"device": dev, "name": name.strip() or dev, "formats": f, "tipo": tipo})
    return info


def remote_devices(ssh_prefix=None, remember=True):
    """Entradas para la interfaz: una por cámara de cada equipo alcanzable; los recordados que no
    responden aparecen sin conexión."""
    hosts = candidate_hosts()
    with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as ex:
        res = list(ex.map(lambda h: probe_host(h["host"], ssh_prefix), hosts))
    conf = _load_conf()
    rec = conf.setdefault("recordados", {})
    out, changed = [], False
    for h, r in zip(hosts, res):
        label_host = h.get("nombre") or h["host"]
        if r.get("ok") and r["cameras"]:
            if remember:
                rec[h["host"]] = {"hostname": r["hostname"], "visto": time.strftime("%Y-%m-%d %H:%M"),
                                  "camaras": [{"device": c["device"], "name": c["name"], "tipo": c["tipo"]}
                                              for c in r["cameras"]]}
                changed = True
            for c in r["cameras"]:
                usable = r["ffmpeg"] and (c["tipo"] != "y10cap" or r["y10cap"] or os.path.isfile(Y10CAP_SRC))
                why = "" if usable else (" — falta ffmpeg en el equipo" if not r["ffmpeg"] else " — falta y10cap")
                out.append({"kind": "ssh", "host": h["host"], "device": c["device"], "tipo": c["tipo"],
                            "usable": bool(usable), "model": f"{r['hostname']} {c['name']}",
                            "label": f"{label_host} · {c['name']} ({c['device']}){why}"})
        elif "recordado" in h:
            for c in h["recordado"].get("camaras", []):
                out.append({"kind": "ssh", "host": h["host"], "device": c["device"], "tipo": c.get("tipo", "mjpeg"),
                            "usable": False, "label": f"{label_host} · {c['name']} — sin conexión "
                                                       f"(visto {h['recordado'].get('visto', '?')})"})
    if changed:
        _save_conf(conf)
    return out


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------
def remote_command(device, tipo="mjpeg", size="1280x720", fps=15, quality=5, exposure=400, gain=0):
    """Comando para el equipo remoto. Imprime su PID por stderr (para poder cerrarlo) y deja MJPEG en
    stdout."""
    w, h = (int(v) for v in str(size).lower().split("x"))
    dev = shlex.quote(device)
    if tipo == "y10cap":
        scale = 1 if w >= 1400 else 2
        ow, oh = 1456 // scale, 1088 // scale
        body = (f"v4l2-ctl -d /dev/v4l-subdev0 -c exposure={int(exposure)} -c analogue_gain={int(gain)} >/dev/null 2>&1; "
                f"$HOME/bin/y10cap {scale} {float(fps):g} | ffmpeg -nostdin -loglevel error -f rawvideo -pix_fmt gray "
                f"-s {ow}x{oh} -i - -c:v mjpeg -q:v {int(quality)} -f mjpeg -")
        return f"echo PID:$$ >&2; exec sh -c {shlex.quote(body)}"
    if tipo == "mjpeg":
        inp = f"-f v4l2 -input_format mjpeg -video_size {w}x{h} -framerate {int(fps)} -i {dev} -c:v copy"
    else:
        inp = (f"-f v4l2 -input_format yuyv422 -video_size {w}x{h} -framerate {int(fps)} -i {dev} "
               f"-c:v mjpeg -q:v {int(quality)}")
    return f"echo PID:$$ >&2; exec ffmpeg -nostdin -loglevel error {inp} -f mjpeg -"


def ensure_y10cap(host, ssh_prefix=None):
    """Como vigia_cam_view.py: sube y compila y10cap en la Pi si falta."""
    if _ssh(host, "test -x $HOME/bin/y10cap", ssh_prefix=ssh_prefix).returncode == 0:
        return
    if not os.path.isfile(Y10CAP_SRC):
        raise RuntimeError(f"falta y10cap en {host} y no está {Y10CAP_SRC} para compilarlo")
    subprocess.run(["scp", *SSH_OPTS, "-q", Y10CAP_SRC, f"{host}:y10cap.c"], check=True, timeout=30)
    r = _ssh(host, "mkdir -p $HOME/bin && gcc -O3 -o $HOME/bin/y10cap $HOME/y10cap.c", timeout=120,
             ssh_prefix=ssh_prefix)
    if r.returncode != 0:
        raise RuntimeError(f"no se pudo compilar y10cap en {host}: {r.stderr.strip()[-200:]}")


class SshCamera:
    """Misma interfaz que AndroidCamera: read_meta() / read() / latest() / close() y on_frame."""

    def __init__(self, host, device="/dev/video0", tipo="mjpeg", size="1280x720", fps=15, rotation=0,
                 ssh_prefix=None, start_timeout=15.0):
        import cv2
        self.cv2 = cv2
        self.host, self.device, self.tipo = host, device, tipo
        self.rotation = int(rotation) % 360
        self.ssh_prefix = ssh_prefix
        self.frame, self.frame_id, self.last_id, self.stamp = None, 0, 0, 0.0
        self.on_frame = None
        self.cv = threading.Condition()
        self.closed = False
        self.error = None
        self.remote_pid = None
        self.log = []
        if tipo == "y10cap":
            ensure_y10cap(host, ssh_prefix)
        cmd = remote_command(device, tipo, size, fps)
        pre = ssh_prefix if ssh_prefix is not None else ["ssh", *SSH_OPTS, host]
        self.proc = subprocess.Popen(pre + [cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     stdin=subprocess.DEVNULL, bufsize=0)
        threading.Thread(target=self._read_err, daemon=True).start()
        threading.Thread(target=self._read_mjpeg, daemon=True).start()
        with self.cv:
            self.cv.wait_for(lambda: self.frame_id > 0 or self.closed, start_timeout)
        if self.frame_id == 0:
            msg = self.error or " | ".join(self.log[-3:]) or "sin imagen"
            self.close()
            raise RuntimeError(f"cámara {device} en {host}: {msg}")

    def _read_err(self):
        for raw in self.proc.stderr:
            line = raw.decode(errors="replace").strip()
            if line.startswith("PID:"):
                self.remote_pid = line[4:].strip()
            elif line:
                self.log = (self.log + [line])[-20:]

    def _read_mjpeg(self):
        """Corta el flujo MJPEG en JPEG (SOI FFD8 ... EOI FFD9) y decodifica cada uno."""
        buf = b""
        out = self.proc.stdout
        try:
            while not self.closed:
                chunk = out.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                while True:
                    a = buf.find(b"\xff\xd8")
                    if a < 0:
                        buf = buf[-1:]
                        break
                    b = buf.find(b"\xff\xd9", a + 2)
                    if b < 0:
                        buf = buf[a:]
                        break
                    jpg, buf = buf[a:b + 2], buf[b + 2:]
                    img = self.cv2.imdecode(np.frombuffer(jpg, np.uint8), self.cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    if self.rotation:
                        img = self.cv2.rotate(img, {90: self.cv2.ROTATE_90_CLOCKWISE, 180: self.cv2.ROTATE_180,
                                                    270: self.cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotation])
                    t = time.time()
                    with self.cv:
                        self.frame, self.frame_id, self.stamp = img, self.frame_id + 1, t
                        fid = self.frame_id
                        self.cv.notify_all()
                    if self.on_frame is not None:
                        try:
                            self.on_frame(img, t, fid)
                        except Exception as e:
                            self.error = self.error or f"on_frame: {e}"
        except Exception as e:
            self.error = self.error or f"lectura: {e}"
        finally:
            if not self.closed and self.proc.poll() is not None and not self.error:
                self.error = " | ".join(self.log[-3:]) or f"ssh terminó (código {self.proc.returncode})"
            with self.cv:
                self.closed = True
                self.cv.notify_all()

    def read_meta(self, timeout=8.0):
        with self.cv:
            ok = self.cv.wait_for(lambda: self.frame_id > self.last_id or self.closed, timeout)
            if not ok or self.frame_id <= self.last_id:
                return None
            self.last_id = self.frame_id
            return self.frame.copy(), self.stamp, self.frame_id

    def read(self, timeout=8.0):
        r = self.read_meta(timeout)
        return None if r is None else r[0]

    def latest(self):
        with self.cv:
            return self.frame_id, self.frame

    def close(self):
        self.closed = True
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(3)
            except Exception:
                self.proc.kill()
        # que la captura remota no quede abierta (y la cámara ocupada) si ffmpeg no notó el cierre
        if self.remote_pid and self.ssh_prefix is None:
            try:
                subprocess.run(["ssh", *SSH_OPTS, self.host,
                                f"pkill -P {int(self.remote_pid)} 2>/dev/null; kill {int(self.remote_pid)} 2>/dev/null; true"],
                               capture_output=True, timeout=8)
            except Exception:
                pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listar", action="store_true", help="buscar cámaras en los equipos SSH")
    ap.add_argument("--host")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--tipo", default="mjpeg", choices=["mjpeg", "yuyv", "y10cap"])
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--local", action="store_true", help="prueba sin SSH: corre el comando en este equipo")
    a = ap.parse_args()
    pre = ["bash", "-c"] if a.local else None
    if a.listar:
        for d in remote_devices(ssh_prefix=pre, remember=not a.local):
            print(("OK   " if d["usable"] else "--   ") + d["label"])
    else:
        cam = SshCamera(a.host or "local", a.device, a.tipo, a.size, a.fps, ssh_prefix=pre)
        t0, n = time.time(), 0
        while time.time() - t0 < 5:
            if cam.read(3) is not None:
                n += 1
        fid, fr = cam.latest()
        print(f"{n} frames en 5 s, último {None if fr is None else fr.shape}, pid remoto {cam.remote_pid}")
        cam.close()
