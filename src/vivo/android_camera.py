"""Cámaras remotas para el mapeo en vivo: celulares Android por adb y cámaras IP por URL.

Android (sin apps en el teléfono, Android 12+): se usa el `scrcpy-server` del repo hermano
`cel-en-rescue` (o el que indique `SCRCPY_SERVER`). Se sube con un nombre propio a
/data/local/tmp, se arranca con `app_process` en modo cámara y `raw_stream=true`, y el video
H.264 llega por un túnel `adb forward`. ffmpeg lo decodifica a frames BGR. Un hilo se queda
siempre con el último frame: el modelo va a ~2 frames/s y la cámara a 15-30, así que no
tiene sentido encolar.

adb: el de `ADB`, el del sistema o el portable de `cel-en-rescue/datos/platform-tools`. Usar
el mismo binario que ya tiene el servidor adb corriendo evita que se reinicie (y que se
corten otras sesiones, por ejemplo un scrcpy abierto en otra terminal).

Cámara IP (IP Webcam, RTSP, MJPEG): cualquier URL que abra `cv2.VideoCapture`.

Uso suelto, para probar:  python3 src/vivo/android_camera.py [serial] [camera_id]
"""
import os
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CEL_REPO = os.environ.get("CEL_RESCUE_DIR", os.path.join(os.path.dirname(REPO), "cel-en-rescue"))
SCRCPY_VERSION = os.environ.get("SCRCPY_VERSION", "4.1")      # debe coincidir con el jar
REMOTE_JAR = "/data/local/tmp/lingbot-scrcpy-server.jar"     # nombre propio: no pisa el de scrcpy


def adb_bin():
    for c in (os.environ.get("ADB"), shutil.which("adb"),
              os.path.join(CEL_REPO, "datos", "platform-tools", "adb")):
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def server_jar():
    for c in (os.environ.get("SCRCPY_SERVER"), os.path.join(CEL_REPO, "datos", "scrcpy", "scrcpy-server"),
              "/usr/share/scrcpy/scrcpy-server", "/usr/local/share/scrcpy/scrcpy-server"):
        if c and os.path.isfile(c):
            return c
    return None


def ffmpeg_bin():
    for c in (os.environ.get("FFMPEG"), shutil.which("ffmpeg"), os.path.expanduser("~/.local/bin/ffmpeg")):
        if c and os.path.isfile(c):
            return c
    return None


def _adb(args, serial=None, timeout=15):
    a = adb_bin()
    if not a:
        raise RuntimeError("no hay adb (ADB=..., el del sistema o cel-en-rescue/datos/platform-tools)")
    cmd = [a] + (["-s", serial] if serial else []) + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def list_devices():
    """Dispositivos adb en estado 'device': [{'serial', 'model'}]."""
    if not adb_bin():
        return []
    try:
        out = _adb(["devices", "-l"], timeout=8).stdout
    except Exception:
        return []
    devs = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            model = next((p.split(":", 1)[1] for p in parts if p.startswith("model:")), parts[0])
            devs.append({"serial": parts[0], "model": model.replace("_", " ")})
    return devs


_CAM_CACHE = {}
_CAM_RE = re.compile(r"--camera-id=(\S+)\s+\((\w+),\s*(\d+)x(\d+)(?:,\s*fps=\{([^}]*)\})?")


def _push_jar(serial):
    jar = server_jar()
    if not jar:
        raise RuntimeError("no se encontró scrcpy-server (SCRCPY_SERVER o cel-en-rescue/datos/scrcpy)")
    r = _adb(["push", jar, REMOTE_JAR], serial, timeout=30)
    if r.returncode != 0:
        raise RuntimeError("adb push del servidor falló: " + (r.stderr or r.stdout).strip())


def list_cameras(serial, refresh=False):
    """Cámaras del teléfono (solo lee sus características: no abre ninguna).
    [{'id', 'facing', 'w', 'h', 'fps': [..]}]"""
    if serial in _CAM_CACHE and not refresh:
        return _CAM_CACHE[serial]
    _push_jar(serial)
    scid = "%08x" % secrets.randbits(31)
    r = _adb(["shell", f"CLASSPATH={REMOTE_JAR}", "app_process", "/", "com.genymobile.scrcpy.Server",
              SCRCPY_VERSION, f"scid={scid}", "log_level=info", "list_cameras=true",
              "video=false", "audio=false", "control=false", "tunnel_forward=true", "cleanup=false"],
             serial, timeout=30)
    cams = []
    for m in _CAM_RE.finditer(r.stdout + r.stderr):
        fps = [int(x) for x in (m.group(5) or "").replace(" ", "").split(",") if x.isdigit()]
        cams.append({"id": m.group(1), "facing": m.group(2), "w": int(m.group(3)), "h": int(m.group(4)),
                     "fps": fps})
    _CAM_CACHE[serial] = cams
    return cams


def remote_devices():
    """Lo que ofrece la interfaz: una entrada por cámara de cada teléfono conectado."""
    out = []
    for d in list_devices():
        try:
            cams = list_cameras(d["serial"])
        except Exception as e:
            out.append({"kind": "android", "serial": d["serial"], "camera_id": None, "usable": False,
                        "label": f"{d['model']} — {e}"})
            continue
        names = {"back": "trasera", "front": "frontal", "external": "externa"}
        for c in cams:
            out.append({"kind": "android", "serial": d["serial"], "camera_id": c["id"], "usable": True,
                        "label": f"{d['model']} · {names.get(c['facing'], c['facing'])} (cámara {c['id']})",
                        "fps": c["fps"]})
    return out


class AndroidCamera:
    """Stream de la cámara de un Android por adb + scrcpy-server. read() -> BGR o None."""

    def __init__(self, serial, camera_id="0", size="1280x720", fps=15, rotation=0, bit_rate=4_000_000,
                 start_timeout=12.0):
        self.serial, self.camera_id = serial, str(camera_id)
        self.w, self.h = (int(v) for v in size.lower().split("x"))
        self.rotation = int(rotation) % 360
        self.fps = int(fps)
        self.scid = "%08x" % secrets.randbits(31)
        self.port = None
        self.server = self.ffmpeg = self.sock = None
        self.frame, self.frame_id, self.last_id = None, 0, 0
        self.stamp = 0.0                     # hora (time.time()) en que se decodificó self.frame
        self.on_frame = None                 # callable(frame, stamp, frame_id) por cada frame decodificado
        self.cv = threading.Condition()
        self.closed = False
        self.error = None
        _push_jar(serial)
        a = adb_bin()
        self.server = subprocess.Popen(
            [a, "-s", serial, "shell", f"CLASSPATH={REMOTE_JAR}", "app_process", "/",
             "com.genymobile.scrcpy.Server", SCRCPY_VERSION, f"scid={self.scid}", "log_level=info",
             "video_source=camera", f"camera_id={self.camera_id}", f"camera_size={self.w}x{self.h}",
             f"camera_fps={self.fps}", f"video_bit_rate={int(bit_rate)}", "audio=false", "control=false",
             "tunnel_forward=true", "raw_stream=true", "send_dummy_byte=false", "cleanup=false"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.log = []
        threading.Thread(target=self._read_log, daemon=True).start()
        r = _adb(["forward", "tcp:0", f"localabstract:scrcpy_{self.scid}"], serial)
        if r.returncode != 0:
            self.close()
            raise RuntimeError("adb forward falló: " + (r.stderr or r.stdout).strip())
        self.port = int(r.stdout.strip())
        # el servidor tarda en abrir el socket: adb acepta la conexión igual y la cierra
        # enseguida si nadie escucha, así que se reintenta hasta recibir datos
        t0 = time.time()
        first = b""
        while time.time() - t0 < start_timeout and not first:
            if self.server.poll() is not None:
                break
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=3)
                s.settimeout(5)
                first = s.recv(65536)
                if first:
                    self.sock = s
                    break
                s.close()
            except OSError:
                pass
            time.sleep(0.3)
        if not first:
            msg = " ".join(self.log[-6:]) or "el servidor no entregó video"
            self.close()
            raise RuntimeError(f"la cámara no arrancó ({msg})")
        self.sock.settimeout(None)
        vf = [f"scale={self.w}:{self.h}"]
        if self.rotation == 90:
            vf.append("transpose=1")
        elif self.rotation == 180:
            vf.append("transpose=1,transpose=1")
        elif self.rotation == 270:
            vf.append("transpose=2")
        self.ow, self.oh = (self.h, self.w) if self.rotation in (90, 270) else (self.w, self.h)
        ff = ffmpeg_bin()
        if not ff:
            self.close()
            raise RuntimeError("no hay ffmpeg")
        # sin "-fflags nobuffer": con H.264 crudo (sin marcas de tiempo) hace que ffmpeg no
        # entregue ningún frame. "-threads 1": el decodificado con hilos por frame retiene
        # varios frames antes de soltarlos; "-fps_mode passthrough": no duplica ni descarta
        # frames para ajustarse a un fps inventado (el H.264 crudo no trae marcas de tiempo)
        self.ffmpeg = subprocess.Popen(
            [ff, "-loglevel", "error", "-threads", "1", "-flags", "low_delay", "-probesize", "32",
             "-analyzeduration", "0", "-f", "h264", "-i", "pipe:0", "-vf", ",".join(vf),
             "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.ffmpeg.stdin.write(first)
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._decode, daemon=True).start()

    def _read_log(self):
        for line in self.server.stdout:
            line = line.strip()
            if line:
                self.log = (self.log + [line])[-30:]

    def _pump(self):
        try:
            while not self.closed:
                b = self.sock.recv(1 << 16)
                if not b:
                    break
                self.ffmpeg.stdin.write(b)
        except Exception as e:
            self.error = self.error or f"conexión: {e}"
        finally:
            try:
                self.ffmpeg.stdin.close()
            except Exception:
                pass

    def _decode(self):
        n = self.ow * self.oh * 3
        out = self.ffmpeg.stdout
        while not self.closed:
            buf = out.read(n)
            if not buf or len(buf) < n:
                break
            fr = np.frombuffer(buf, np.uint8).reshape(self.oh, self.ow, 3)
            t = time.time()                  # hora de captura (al salir del decodificador)
            with self.cv:
                self.frame, self.frame_id, self.stamp = fr, self.frame_id + 1, t
                fid = self.frame_id
                self.cv.notify_all()
            if self.on_frame is not None:
                try:
                    self.on_frame(fr, t, fid)
                except Exception as e:
                    self.error = self.error or f"on_frame: {e}"
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def read_meta(self, timeout=8.0):
        """(frame, stamp, frame_id) del más reciente que no se devolvió; None si el stream terminó."""
        with self.cv:
            ok = self.cv.wait_for(lambda: self.frame_id > self.last_id or self.closed, timeout)
            if not ok or self.frame_id <= self.last_id:
                return None
            self.last_id = self.frame_id
            return self.frame.copy(), self.stamp, self.frame_id

    def read(self, timeout=8.0):
        """Siguiente frame (el más reciente que no se devolvió); None si el stream terminó."""
        r = self.read_meta(timeout)
        return None if r is None else r[0]

    def latest(self):
        """(id, frame) del último frame decodificado, sin consumirlo (para la vista previa)."""
        with self.cv:
            return self.frame_id, self.frame

    def close(self):
        self.closed = True
        for p in (self.ffmpeg, self.server):
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(3)
                except Exception:
                    p.kill()
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        if self.port:
            try:
                _adb(["forward", "--remove", f"tcp:{self.port}"], self.serial, timeout=5)
            except Exception:
                pass


if __name__ == "__main__":
    import sys
    import cv2
    devs = list_devices()
    print("dispositivos:", devs)
    serial = sys.argv[1] if len(sys.argv) > 1 else (devs[0]["serial"] if devs else None)
    print("cámaras:", list_cameras(serial))
    cam = AndroidCamera(serial, sys.argv[2] if len(sys.argv) > 2 else "0")
    t0, n = time.time(), 0
    while time.time() - t0 < 5:
        fr = cam.read()
        if fr is None:
            break
        n += 1
    print(f"{n} frames en 5 s, tamaño {None if fr is None else fr.shape}")
    if fr is not None:
        cv2.imwrite("cel_prueba.jpg", fr)
    cam.close()
