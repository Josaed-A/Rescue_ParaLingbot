"""Servidor único del visor: archivos estáticos, nubes ya exportadas y mapeo EN VIVO.

Un solo proceso y un solo puerto para las dos cosas, para no duplicar servidores:

  GET  /                     el visor (src/mapas/webgl_viewer/)
  GET  /api/captures         qué pruebas hay exportadas (selector)
  GET  /data/<archivo>       sirve .ply/.glb/.json de captures/ (con HTTP Range)
  WS   /ws                   canal en vivo: cada frame reconstruido se empuja al navegador
  POST /api/live/start       arranca una sesión de mapeo en vivo (fuente: webcam/carpeta/video)
  POST /api/live/stop        la detiene y libera la GPU
  GET  /api/live/status      estado actual
  GET  /api/explorer         catálogo de pruebas (explorador): mapas por tipo, categorías, zona
  GET  /api/files            contenido de una carpeta de una prueba (perezoso)
  POST /api/prueba/meta      editar título / zona / categorías / notas (info.json de la prueba)
  POST /api/prueba/save      guardar una sesión en vivo (sale de streaming/sin_guardar/)
  POST /api/prueba/discard   descartar una sesión en vivo sin guardar
  POST /api/jobs/start       construir mapas de una prueba (src/mapas/build_maps.py)
  POST /api/jobs/cancel      cancelar el trabajo en curso
  GET  /api/jobs             estado del trabajo

Cada sesión en vivo graba lo que predijo el modelo (profundidad, confianza, poses,
intrínsecos e imagen de cada frame) y al terminar lo deja en
captures/streaming/sin_guardar/sesion_<fecha>/ con el mismo formato .npz que
--save_predictions, más los frames en frames/. Así una sesión en vivo es una prueba
más: se puede guardar con nombre y categorías, construirle los mapas (nube, malla,
splat, video) o reprocesarla en windowed, que es lo que corrige la deriva.

**El modelo se carga sólo al arrancar una sesión en vivo** y se descarga al detenerla:
mientras nadie pida streaming, este proceso no toca la GPU ni ocupa VRAM. Mirar nubes
ya exportadas no carga nada.

Protocolo del canal en vivo:
  - Mensajes de texto: JSON de estado ({"type": "status"|"started"|"stopped"|"error"|"done"}).
  - Mensajes binarios: un frame del mapa, con esta cabecera de 72 bytes (little endian):
        uint32  frame_idx
        uint32  n_points
        float32[16]  c2w  (4x4 row-major, cámara->mundo)
    seguida de n_points*3 float32 (xyz mundo) y n_points*3 uint8 (rgb).

Uso:
  python3 src/vivo/live_server.py [--port 8090]
  python3 src/vivo/live_server.py --port 8090 --captures_dir captures
"""
import argparse
import asyncio
import json
import os
import struct
import sys
import threading
import time
import traceback

import numpy as np
from aiohttp import web, WSMsgType

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
VIEWER_DIR = os.path.join(REPO_ROOT, "src/mapas", "webgl_viewer")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

from src.mapas.webgl_viewer.captures_index import scan_captures  # noqa: E402
from src.mapas.webgl_viewer import catalog  # noqa: E402
from tracking import FrameMeta, BasicTrackingProvider  # noqa: E402
from pose_buffer import PoseBuffer, QueryKind  # noqa: E402
from hybrid_tracking import HybridParams, ReferenceTracker, TrackingMode  # noqa: E402

HEADER = struct.Struct("<II16f")


# ---------------------------------------------------------------------------
# Fuentes de frames
# ---------------------------------------------------------------------------
class FrameSource:
    """Entrega frames BGR (numpy) uno por uno, cada uno con su FrameMeta (hora de captura e
    identificador). Cierra con close().

    Hora de captura: en cámaras en vivo (webcam, URL, celular) es time.time() en el momento en
    que el hilo lector recibe el frame, el mismo reloj que usará ROS2. En carpeta y video es
    reproducible: en carpetas, índice / fps nominal (`source_fps`) desde 0 (`synthetic`); en videos,
    el timestamp de presentación del frame en el contenedor (`video_pts`). Los videos de celular
    suelen ser de tasa variable (el del fablab tiene 13 huecos de hasta 168 ms): índice / fps se
    desviaba hasta 0.6 s del tiempo real (medido el 2026-10-04, etapa 4). `stamp_kind` dice cuál es.
    """

    def __init__(self, kind, path=None, device=0, fps=None, max_frames=0, start=0,
                 serial=None, camera_id="0", rotation=0, cam_size="1280x720", cam_fps=15,
                 source_fps=10.0, realtime=False, on_frame=None, host=None, cam_tipo="mjpeg"):
        import cv2
        self.cv2 = cv2
        self.kind = kind
        self.fps = fps
        self.max_frames = max_frames
        self.n = 0
        self._cap = None
        self._files = None
        self._i = start
        self._remote = None
        self._rot = int(rotation or 0) % 360
        self._nsrc = 0                      # frames leídos de la fuente (carpeta/video)
        self._src_fps = float(source_fps or 10.0)
        self.stamp_kind = "synthetic"
        # on_frame(img_bgr, FrameMeta[, prep]): se llama por CADA frame capturado, en el hilo de la
        # cámara (el "tee" hacia ROS2 de la etapa 3). Si se pasa `prep`, la imagen todavía no está
        # rotada y el consumidor la aplica sólo a los frames que use. None = nada.
        self.on_frame = on_frame
        # realtime: carpeta/video reproducidos a su fps nominal por un hilo propio, "último gana",
        # como una cámara en vivo. Así una grabación reproduce lo que vería el sistema en vivo (el
        # modelo toma ~2 de cada 15 frames y Stella recibe todos). Por defecto (False) la carpeta se
        # consume frame a frame al ritmo del modelo, como siempre (baseline reproducible).
        self._realtime = bool(realtime) and kind in ("folder", "video")
        # en modo realtime la reproducción espera a resume(): así el video no corre mientras el
        # modelo se carga (~8 s) y la sesión empieza en el primer frame, como una grabación real
        self._go = threading.Event()
        if kind == "android":
            # cámara de un teléfono por adb (scrcpy-server en el teléfono, H.264 → ffmpeg)
            from android_camera import AndroidCamera
            self._remote = AndroidCamera(serial, camera_id=str(camera_id), size=cam_size,
                                         fps=int(cam_fps), rotation=self._rot)
            self._rot = 0                       # ya lo rota ffmpeg
        elif kind == "ssh":
            # cámara de otro equipo por SSH (Pi del robot, vigia-1...): ffmpeg MJPEG por la salida estándar
            from ssh_camera import SshCamera
            self._remote = SshCamera(host, device=str(device), tipo=cam_tipo, size=cam_size,
                                     fps=int(cam_fps), rotation=self._rot)
            self._rot = 0                       # ya la rota SshCamera
        elif kind == "url":
            # cámara IP / stream (RTSP, HTTP MJPEG, etc.)
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise RuntimeError(f"no se pudo abrir el stream {path}")
        elif kind == "webcam":
            self._cap = cv2.VideoCapture(int(device), cv2.CAP_V4L2)
            if not self._cap.isOpened():
                raise RuntimeError(f"no se pudo abrir la cámara {device}")
        elif kind == "video":
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise RuntimeError(f"no se pudo abrir el video {path}")
            self._src_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or self._src_fps)
            self.stamp_kind = "video_pts"
        elif kind == "folder":
            exts = (".png", ".jpg", ".jpeg")
            self._files = sorted(
                os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(exts)
            )
            if not self._files:
                raise RuntimeError(f"la carpeta no tiene imágenes: {path}")
        else:
            raise ValueError(f"fuente desconocida: {kind}")
        # Cámaras en vivo (webcam / URL): un hilo lee sin parar y se queda con el último
        # frame. Sin esto, leyendo al ritmo del modelo (~2 frames/s) la webcam entrega frames
        # viejos acumulados en su buffer, y la vista previa no puede ir más rápido que el modelo.
        self._live = kind in ("webcam", "url")
        if self._live or self._remote is not None:
            self.stamp_kind = "live"
        if self._remote is not None and on_frame is not None:
            self._remote.on_frame = lambda img, stamp, fid: on_frame(img, FrameMeta(stamp=stamp, frame_id=fid))
        if self._live or self._realtime:
            self._cv = threading.Condition()
            self._frame, self._fid, self._last, self._stamp = None, 0, 0, 0.0
            self._ended = False
            self._grabber = threading.Thread(target=self._grab_realtime if self._realtime else self._grab,
                                             daemon=True)
            self._grabber.start()

    @property
    def is_live(self):
        return self._live or self._remote is not None or self._realtime

    def resume(self):
        """Arranca la reproducción en modo realtime (no hace nada en las demás fuentes)."""
        self._go.set()

    def _grab(self):
        while not self._ended:
            ok, img = self._cap.read()
            t = time.time()                     # hora de captura: al recibir el frame, no al usarlo
            with self._cv:
                if not ok:
                    self._ended = True
                else:
                    self._frame, self._fid, self._stamp = img, self._fid + 1, t
                self._cv.notify_all()
            if ok and self.on_frame is not None:
                self.on_frame(img, FrameMeta(stamp=t, frame_id=self._fid), self._rotate)

    def _video_pts(self):
        """Timestamp de presentación (s) del frame que se acaba de leer del video."""
        return self._cap.get(self.cv2.CAP_PROP_POS_MSEC) / 1000.0

    def _grab_realtime(self):
        """Carpeta/video a su ritmo, como si fuera una cámara: la carpeta a su fps nominal (stamp
        índice / fps), el video según el timestamp real de cada frame; el modelo toma el último."""
        period = 1.0 / self._src_fps
        self._go.wait()
        t0 = time.time()
        k = 0
        while not self._ended:
            if self._files is not None:
                if self._i >= len(self._files):
                    img = None
                else:
                    img = self.cv2.imread(self._files[self._i])
                    self._i += 1
            else:
                ok, img = self._cap.read()
                img = img if ok else None
            stamp = k * period if self._files is not None else (self._video_pts() if img is not None else 0.0)
            with self._cv:
                if img is None:
                    self._ended = True
                else:
                    self._frame, self._fid, self._stamp = img, self._fid + 1, stamp
                    self._wall = time.time()
                self._cv.notify_all()
            if img is None:
                break
            if self.on_frame is not None:
                # la rotación la hace el consumidor sólo si va a usar el frame (etapa 19: el puente
                # publica 1 de cada N y girar 1080x1920 a 30 fps era la mitad del hilo lector)
                self.on_frame(img, FrameMeta(stamp=stamp, frame_id=self._fid, wall=self._wall), self._rotate)
            k += 1
            nxt = k * period if self._files is not None else stamp + period
            sleep = t0 + nxt - time.time()
            if sleep > 0:
                time.sleep(sleep)

    def _rotate(self, img):
        if self._rot:
            img = self.cv2.rotate(img, {90: self.cv2.ROTATE_90_CLOCKWISE, 180: self.cv2.ROTATE_180,
                                        270: self.cv2.ROTATE_90_COUNTERCLOCKWISE}[self._rot])
        return img

    def latest(self):
        """(id, frame BGR ya rotado) del último frame de una cámara en vivo, sin consumirlo."""
        if self._remote is not None:
            return self._remote.latest()
        if not (self._live or self._realtime):
            return 0, None
        with self._cv:
            fid, img = self._fid, self._frame
        return fid, (None if img is None else self._rotate(img))

    def total(self):
        if self._files is not None:
            n = len(self._files) - self._i
        elif self.kind == "video":
            n = int(self._cap.get(self.cv2.CAP_PROP_FRAME_COUNT)) or 0
        else:
            n = 0            # la webcam no tiene total
        return min(n, self.max_frames) if self.max_frames else n

    def read(self):
        """(frame BGR ya rotado, FrameMeta) del siguiente frame; None cuando la fuente termina."""
        if self.max_frames and self.n >= self.max_frames:
            return None
        if self._realtime:
            with self._cv:      # igual que una cámara en vivo: el último frame no devuelto
                self._cv.wait_for(lambda: self._fid > self._last or self._ended, 8.0)
                if self._fid <= self._last:
                    return None
                self._last, img, stamp, wall = self._fid, self._frame, self._stamp, getattr(self, "_wall", None)
            meta = FrameMeta(stamp=stamp, frame_id=self._last, wall=wall)
        elif self._files is not None:
            if self._i >= len(self._files):
                return None
            img = self.cv2.imread(self._files[self._i])
            meta = FrameMeta(stamp=self._i / self._src_fps, frame_id=self._i, wall=time.time())
            self._i += 1
            if self.on_frame is not None:
                self.on_frame(self._rotate(img), meta)
        elif self._remote is not None:
            r = self._remote.read_meta()
            if r is None:
                return None
            img, stamp, fid = r
            meta = FrameMeta(stamp=stamp, frame_id=fid, wall=stamp)
        elif self._live:
            with self._cv:      # el frame más reciente que todavía no se devolvió
                self._cv.wait_for(lambda: self._fid > self._last or self._ended, 8.0)
                if self._fid <= self._last:
                    return None
                self._last, img, stamp = self._fid, self._frame, self._stamp
            meta = FrameMeta(stamp=stamp, frame_id=self._last, wall=stamp)
        else:
            ok, img = self._cap.read()
            if not ok:
                return None
            meta = FrameMeta(stamp=self._video_pts(), frame_id=self._nsrc, wall=time.time())
            self._nsrc += 1
            if self.on_frame is not None:
                self.on_frame(self._rotate(img), meta)
        img = self._rotate(img)
        self.n += 1
        return img, meta

    def close(self):
        if self._live or self._realtime:
            self._ended = True
            self._go.set()                   # por si nunca se llamó a resume()
            self._grabber.join(timeout=2)    # no soltar la cámara con un read() en curso
        if self._cap is not None:
            self._cap.release()
        if self._remote is not None:
            self._remote.close()


# ---------------------------------------------------------------------------
# Sesión de mapeo en vivo
# ---------------------------------------------------------------------------
class LiveSession:
    """Corre la inferencia en un hilo y empuja cada frame al loop de asyncio.

    Usa la primitiva de streaming del modelo (`forward(..., causal_inference=True)`),
    la misma que `inference_streaming` internamente: fase de escala con los primeros
    `num_scale_frames`, y después un frame a la vez con el KV cache persistente.
    """

    def __init__(self, loop, broadcast, cfg):
        self.loop = loop
        self.broadcast = broadcast          # callable(msg) -> corutina programada
        self.cfg = cfg
        self.thread = None
        self._model = None
        self.stop_flag = threading.Event()
        self.state = {"running": False, "frames": 0, "total": 0, "fps": 0.0,
                      "vram_mb": 0, "msg": "detenido", "source": cfg.get("source"),
                      "saved": None}
        self.rec = {"depth": [], "conf": [], "images": [], "extrinsic": [], "intrinsic": [],
                    # tracking (etapa 1 de la integración Stella): tiempo e identidad de cada
                    # frame registrado y la estimación BASIC con sus señales observables
                    "stamps": [], "frame_ids": [], "pose_basic": [], "pose_model": [], "estatico": [],
                    "track_conf": [], "track_motion": [], "track_sharp": [],
                    # etapa 3: la pose de Stella para el MISMO frame (NaN si no la hubo) y su estado
                    "pose_stella": [], "stella_status": [],
                    # etapa 5: cómo se obtuvo la pose de Stella EN VIVO (lo que habría usado un
                    # selector en ese momento): QueryKind y distancia a la pose de Stella más cercana
                    "stella_assoc_live": [], "stella_dt_live": [],
                    # etapa 6: pose de referencia de los modos STELLA e HYBRID (la de BASIC es pose_basic),
                    # qué fuente usó HYBRID en cada frame y, si fue BASIC, por qué
                    "pose_ref_stella": [], "pose_ref_hybrid": [], "hybrid_usado": [], "hybrid_motivo": [],
                    "stella_rate": [],
                    # etapa 10: pose con que se registró la geometría (BASIC o la referencia) y latencia
                    # captura -> emisión del frame
                    "extrinsic_reg": [], "pose_source_reg": [], "latency": []}
        # etapa 12: ancla de cada frame registrado a un keyframe de Stella, para corregir la geometría
        # histórica cuando Stella mueve sus keyframes (BA / loop closure)
        from keyframe_correction import FrameAnchors
        self.kf_anchors = FrameAnchors()
        self.frame_seg = {}
        self.emitted_pose = {}             # frame_idx -> c2w con que se emitieron sus puntos
        self.kf_version_seen = 0
        self.corrections = []              # eventos de corrección aplicados en vivo
        self.t_begin = time.time()
        # El tracking actual, expuesto con la interfaz común (src/vivo/tracking.py).
        # estimate_sinks: callables(TrackingEstimate) para otros consumidores (p. ej. el puente
        # ROS2 de la etapa 3); vacío por defecto, no cambia nada.
        self.tracker = BasicTrackingProvider()
        self.estimate_sinks = []
        # etapa 5: buffer de las poses BASIC (una por frame del modelo, ~2 Hz). Para consultar la
        # pose BASIC en instantes de otra fuente; interpolar BASIC es de baja calidad (medido: 3-7% del
        # desplazamiento en 1 s ya a 0.2 s, por el ruido propio por frame), de ahí el max_gap amplio
        # pero marcado: quien consulte debe mirar `kind` y `gap`.
        self.basic_buffer = PoseBuffer(max_gap=0.75, edge_tol=0.05)
        # etapa 6: el selector corre en los tres modos a la vez (sólo con el puente ROS2: sin Stella los
        # tres son BASIC y nada cambia); `tracking_mode` elige cuál se anuncia como referencia. La
        # geometría se sigue registrando con BASIC: usar la referencia para registrar es la etapa 10.
        self.tracking_mode = TrackingMode(cfg.get("tracking_mode", "basic"))
        self.ref_trackers = None
        self._published_anchors = {}
        # Puente ROS2 (etapa 3): solo si cfg["ros2"]; si no, None y nada cambia respecto al baseline.
        self.bridge = None
        self.stella_proc = None

    # -- utilidades ---------------------------------------------------------
    def _emit(self, obj):
        self.loop.call_soon_threadsafe(self.broadcast, obj)

    def _preview_loop(self, src):
        import cv2
        period = 1.0 / max(1.0, float(self.cfg.get("preview_fps", 15)))
        width = int(self.cfg.get("preview_width", 640))
        last = 0
        while not self.stop_flag.is_set() and self.state.get("running"):
            t = time.time()
            fid, img = src.latest()
            if img is not None and fid != last:
                last = fid
                h, w = img.shape[:2]
                if w > width:
                    img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
                ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                if ok:
                    self._emit({"__bin__": struct.pack("<I", 0xFFFFFFFF) + jpg.tobytes(), "__drop__": "preview"})
            time.sleep(max(0.0, period - (time.time() - t)))

    def _status(self, msg=None, **kw):
        if msg:
            self.state["msg"] = msg
        self.state.update(kw)
        self._emit({"type": "status", **self.state})

    # -- ciclo de vida ------------------------------------------------------
    def start(self):
        if self.state["running"]:
            return False
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.state["running"] = True
        self.thread.start()
        return True

    def stop(self):
        self.stop_flag.set()

    # -- trabajo pesado (hilo aparte) ---------------------------------------
    def _run(self):
        try:
            self._infer()
        except Exception as e:
            traceback.print_exc()
            self._emit({"type": "error", "msg": f"{type(e).__name__}: {e}"})
        finally:
            # cerrar Stella y el puente ROS2 pase lo que pase: si la inferencia falla a mitad, sin
            # esto run_slam queda huérfano publicando en los mismos topics y el hilo de rclpy aborta
            # el proceso al salir (encontrado en la etapa 5)
            self._stop_ros2()
            self._release()
            try:
                self._save_session()
            except Exception as e:
                traceback.print_exc()
                self._emit({"type": "error", "msg": f"no se pudo guardar la sesión: {e}"})
            self.state["running"] = False
            self._status("detenido")
            self._emit({"type": "stopped", "saved": self.state.get("saved")})

    def _stop_ros2(self):
        """Idempotente: cierra Stella (si la arrancó esta sesión) y el puente ROS2."""
        if self.stella_proc is not None:
            self._status("cerrando Stella-VSLAM...")
            try:
                self.stella_proc.stop()
            except Exception:
                traceback.print_exc()
            self.stella_proc = None
        if self.bridge is not None and self.bridge.ok:
            self.state["ros2"] = self.bridge.summary()
            try:
                self.bridge.stop()
            except Exception:
                traceback.print_exc()

    def _save_session(self):
        """Deja la sesión como prueba "sin guardar", en el formato de --save_predictions."""
        n = len(self.rec["depth"])
        if not self.cfg.get("record", True) or n < 3:
            return
        import cv2
        self._status(f"guardando la sesión ({n} frames)...")
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.t_begin))
        P = os.path.join(self.cfg["captures_dir"], "streaming", "sin_guardar", f"sesion_{ts}")
        os.makedirs(os.path.join(P, "eval"), exist_ok=True)
        os.makedirs(os.path.join(P, "frames"), exist_ok=True)
        extrinsic_basic = np.stack(self.rec["extrinsic"])
        # etapa 10: `extrinsic` = pose con la que se registró la geometría (la que leen build_maps y el
        # visor); `extrinsic_basic` = BASIC siempre. Sin register_with_reference son iguales.
        extrinsic = np.stack(self.rec["extrinsic_reg"]) if self.rec["extrinsic_reg"] else extrinsic_basic
        stella_traj = self.bridge.stella.arrays() if self.bridge else (np.zeros(0), np.zeros((0, 4, 4), np.float32),
                                                                       np.zeros(0, np.int32))
        stella_ev = self.bridge.stella.event_arrays() if self.bridge else (np.zeros(0), np.zeros(0, np.uint8))
        kf_corr = {}
        if self.bridge is not None and self.ref_trackers is not None:
            hist = self.bridge.stella.kf_history
            corr = self.kf_anchors.corrections(hist)
            anchors = self.ref_trackers[TrackingMode.STELLA].anchor
            ps_corr = np.full((n, 4, 4), np.nan, np.float32)
            ref_corr = (np.stack(self.rec["pose_ref_stella"]).copy() if self.rec["pose_ref_stella"]
                        else np.full((n, 4, 4), np.nan, np.float32))
            disp = np.full(n, np.nan, np.float32)
            for fid, (_, Tc, dsp) in corr.items():
                if fid >= n:
                    continue
                ps_corr[fid], disp[fid] = Tc, dsp
                sg = self.frame_seg.get(fid)
                if sg in anchors:
                    sa, R, t = anchors[sg]
                    T = np.eye(4)
                    T[:3, :3] = R @ Tc[:3, :3]
                    T[:3, 3] = sa * R @ Tc[:3, 3] + t
                    ref_corr[fid] = T
            kf_corr = {"pose_stella_kfcorr": ps_corr, "kfcorr_desplazamiento": disp,
                       "pose_ref_stella_corr": ref_corr,
                       "kf_mensajes": np.array(json.dumps(self.bridge.stella.kf_messages[-200:]))}
        # asociación "a posteriori": consultar el buffer de Stella con TODAS sus poses ya llegadas
        # (en vivo, la pose de Stella posterior a un frame puede no haber llegado todavía)
        st_arr = np.array(self.rec["stamps"], dtype=np.float64)
        if self.bridge is not None:
            ps_post, kind_post, dt_post, seg_post = self.bridge.stella.buffer.query_many(st_arr)
        else:
            ps_post = np.full((n, 4, 4), np.nan)
            kind_post, dt_post, seg_post = np.zeros(n, np.uint8), np.full(n, np.inf), np.full(n, -1, np.int32)
        np.savez(os.path.join(P, "eval", "sesion.npz"),
                 depth=np.stack(self.rec["depth"]), depth_conf=np.stack(self.rec["conf"]),
                 images=np.stack(self.rec["images"]), extrinsic=extrinsic,
                 intrinsic=np.stack(self.rec["intrinsic"]), ds=1,
                 is_real=np.ones(n, dtype=bool), source_index=np.arange(n),
                 # --- tracking (claves nuevas, aditivas: las herramientas leen por nombre) ---
                 # extrinsic = pose de referencia con la que se registra la geometría (hoy BASIC);
                 # extrinsic_basic la conserva cuando la referencia pase a ser otra (etapa 6).
                 extrinsic_basic=extrinsic_basic,
                 pose_basic=np.stack(self.rec["pose_basic"]).astype(np.float32),     # c2w 4x4
                 # momentos estáticos: pose del modelo sin retener y qué frames fueron estáticos
                 pose_model=np.stack(self.rec["pose_model"] or self.rec["pose_basic"]).astype(np.float32),
                 estatico=np.array(self.rec["estatico"] or [False] * n, dtype=bool),
                 pose_source=np.array(self.rec["pose_source_reg"] or [0] * n, dtype=np.uint8),   # 0 BASIC, 1 STELLA, 2 HYBRID
                 latency_s=np.array(self.rec["latency"] or [np.nan] * n, dtype=np.float32),
                 stamps=np.array(self.rec["stamps"], dtype=np.float64),
                 frame_ids=np.array(self.rec["frame_ids"], dtype=np.int64),
                 stamp_kind=str(self.state.get("stamp_kind", "synthetic")),
                 track_conf_basic=np.array(self.rec["track_conf"], dtype=np.float32),
                 track_motion_px=np.array(self.rec["track_motion"], dtype=np.float32),
                 track_sharpness=np.array(self.rec["track_sharp"], dtype=np.float32),
                 # --- etapa 3: Stella, si el puente ROS2 estuvo activo (si no, NaN / vacío) ---
                 # pose_stella[i]: c2w de Stella (convención CV) para el frame i, NaN si no la hubo;
                 # stella_status[i]: TrackingStatus de Stella al registrar el frame (0 = UNKNOWN);
                 # stella_traj_*: TODAS las poses de Stella a su propio ritmo (para la etapa 4).
                 pose_stella=(np.stack(self.rec["pose_stella"]) if self.rec["pose_stella"]
                              else np.full((n, 4, 4), np.nan, np.float32)),
                 stella_status=np.array(self.rec["stella_status"] or [0] * n, dtype=np.uint8),
                 stella_traj_stamps=stella_traj[0], stella_traj_c2w=stella_traj[1],
                 # etapa 4: mapa (segmento) de cada pose de Stella y cambios de estado con hora local
                 stella_traj_segment=stella_traj[2],
                 stella_events_t=stella_ev[0], stella_events_status=stella_ev[1],
                 # etapa 5 (QueryKind: 0 ninguna, 1 exacta, 2 interpolada, 3 la más cercana):
                 # *_live = lo disponible al procesar el frame; pose_stella_post = consulta final.
                 stella_assoc_live=np.array(self.rec["stella_assoc_live"] or [0] * n, dtype=np.uint8),
                 stella_dt_live=np.array(self.rec["stella_dt_live"] or [np.inf] * n, dtype=np.float32),
                 pose_stella_post=ps_post.astype(np.float32), stella_assoc_post=kind_post,
                 stella_dt_post=dt_post.astype(np.float32), stella_segment_post=seg_post,
                 # etapa 6: selector (vacío sin puente ROS2)
                 tracking_mode=str(self.tracking_mode.value),
                 pose_ref_stella=(np.stack(self.rec["pose_ref_stella"]) if self.rec["pose_ref_stella"]
                                  else np.full((n, 4, 4), np.nan, np.float32)),
                 pose_ref_hybrid=(np.stack(self.rec["pose_ref_hybrid"]) if self.rec["pose_ref_hybrid"]
                                  else np.full((n, 4, 4), np.nan, np.float32)),
                 hybrid_usado=np.array(self.rec["hybrid_usado"] or [0] * n, dtype=np.uint8),
                 hybrid_motivo=np.array(self.rec["hybrid_motivo"] or [""] * n),
                 stella_rate_hz=np.array(self.rec["stella_rate"] or [np.nan] * n, dtype=np.float32),
                 # etapa 12: pose de Stella de cada frame corregida con la ÚLTIMA versión de los keyframes
                 # (NaN si no se pudo anclar) y la referencia STELLA recalculada con ella
                 **kf_corr)
        for k, im in enumerate(self.rec["images"]):
            cv2.imwrite(os.path.join(P, "frames", f"{k:06d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        src = self.cfg.get("source")
        cats = ["streaming", "sin guardar"] + {"webcam": ["webcam"], "android": ["celular"],
                                               "url": ["cámara IP"], "ssh": ["cámara por SSH"]}.get(src, [])
        info = {
            "titulo": f"Sesión en vivo {time.strftime('%Y-%m-%d %H:%M', time.localtime(self.t_begin))}",
            "zona": "", "categorias": cats, "notas": "",
            "fecha": time.strftime("%Y-%m-%d", time.localtime(self.t_begin)),
            "modo": "streaming (en vivo): el mapa deriva; reprocesar con la tarea 'windowed'",
            # sin serial ni IP del teléfono (no se guardan datos personales en las pruebas)
            "fuente": ({"tipo": src, "camara": self.cfg.get("camera_id"), "modelo": self.cfg.get("device_label"),
                        "rotacion": self.cfg.get("rotation", 0), "resolucion": self.cfg.get("cam_size")}
                       if src == "android" else
                       {"tipo": src, "equipo": self.cfg.get("host"), "device": self.cfg.get("device"),
                        "modelo": self.cfg.get("device_label"), "rotacion": self.cfg.get("rotation", 0),
                        "resolucion": self.cfg.get("cam_size")}
                       if src == "ssh" else
                       {"tipo": src, "path": None if src == "url" else self.cfg.get("path"),
                        "device": self.cfg.get("device"), "rotacion": self.cfg.get("rotation", 0)}),
            "frames": n, "fps_objetivo": self.cfg.get("fps"), "fps_real": self.state.get("fps"),
            "parametros": {k: self.cfg[k] for k in ("num_scale_frames", "kv_cache_sliding_window",
                                                    "camera_num_iterations", "conf_percentile")},
            "tracking": {"source": "BASIC", "stamp_kind": self.state.get("stamp_kind", "synthetic"),
                         "source_fps": self.cfg.get("source_fps"), "source_realtime": bool(self.cfg.get("source_realtime")),
                         "frame_ids": [int(self.rec["frame_ids"][0]), int(self.rec["frame_ids"][-1])],
                         "stella": ({**(self.state.get("ros2") if isinstance(self.state.get("ros2"), dict) else self.bridge.summary()),
                                     "config": self.cfg.get("stella_config"),
                                     "frames_con_pose_stella": int(sum(1 for m in self.rec["pose_stella"] if np.isfinite(m[0, 0]))),
                                     "asociacion_en_vivo": {QueryKind(int(k)).name: int(v) for k, v in
                                                            zip(*np.unique(self.rec["stella_assoc_live"] or [0], return_counts=True))}}
                                    if self.bridge else None),
                         "registro_geometria": ("referencia" if self.cfg.get("register_with_reference") else "basic"),
                         "latencia_s_mediana": (round(float(np.nanmedian(self.rec["latency"])), 3)
                                                if self.rec["latency"] and np.isfinite(self.rec["latency"]).any() else None),
                         "correcciones_keyframes_en_vivo": self.corrections[-50:],
                         # momentos estáticos (StaticHold): frames en que la pose no avanzó
                         "momentos_estaticos": ({"frames": int(sum(self.rec["estatico"])), "de": len(self.rec["estatico"]),
                                                 "umbral_px": round(self.static_hold.thr, 2)}
                                                if getattr(self, "static_hold", None) is not None else None),
                         "selector": ({"modo_referencia": self.tracking_mode.value,
                                       **{m.value: tr.summary() for m, tr in self.ref_trackers.items()}}
                                      if self.ref_trackers else None)},
            "analizador_contexto": ({"sintesis": self.cfg.get("context_synth"),
                                     "intensidad_sintesis": self.cfg.get("context_synth_strength"),
                                     "paso_px": self.cfg.get("context_step_px"), **(self.state.get("context") or {})}
                                    if self.cfg.get("context") else None),
        }
        with open(os.path.join(P, "info.json"), "w") as fh:
            json.dump(info, fh, indent=1, ensure_ascii=False)
        rid = os.path.relpath(P, self.cfg["captures_dir"]).replace(os.sep, "/")
        self.state["saved"] = rid
        self.rec = {k: [] for k in self.rec}             # liberar RAM
        self._emit({"type": "saved", "prueba": rid, "frames": n})

    def _release(self):
        """Suelta modelo y VRAM pase lo que pase: si una excepción corta la
        inferencia a mitad, sin esto el checkpoint queda ocupando la GPU."""
        m = getattr(self, "_model", None)
        if m is not None:
            try:
                m.clean_kv_cache()
            except Exception:
                pass
            self._model = None
            del m
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # Devolver al sistema la RAM que glibc retiene tras liberar el modelo (medido el 2026-10-04:
        # el servidor quedaba con 12 GB de RSS anónima tras dos sesiones, con la VRAM ya liberada;
        # eso deja sin memoria a los trabajos aislados de src/gpu/run_isolated.sh y a Stella).
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def _infer(self):
        import torch
        import cv2
        import demo
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
        from lingbot_map.utils.geometry import closed_form_inverse_se3_general

        cfg = self.cfg
        # -- ROS2 / Stella (etapa 3): opcional; sin ros2 el camino es exactamente el de siempre --
        if cfg.get("ros2"):
            from ros2_bridge import Ros2Bridge, StellaProcess
            self._status("arrancando el puente ROS2...")
            self.bridge = Ros2Bridge(cfg)
            if not self.bridge.start():
                self._emit({"type": "error", "msg": self.bridge.error})
                self.bridge = None
            else:
                self.estimate_sinks.append(self.bridge.publish_basic)
                if cfg.get("stella_config"):
                    self.stella_proc = StellaProcess(cfg["stella_config"], image_topic=self.bridge.topic,
                                                     ns=self.bridge.ns, log_dir=cfg.get("stella_log_dir"),
                                                     cpus=cfg.get("stella_cpus"))
                    self._status("arrancando Stella-VSLAM...")
                    self.stella_proc.start()
        self._status("abriendo la fuente de frames...")
        src = FrameSource(cfg["source"], path=cfg.get("path"), device=cfg.get("device", 0),
                          fps=cfg.get("fps"), max_frames=cfg.get("max_frames", 0),
                          serial=cfg.get("serial"), camera_id=cfg.get("camera_id", "0"),
                          rotation=cfg.get("rotation", 0), cam_size=cfg.get("cam_size", "1280x720"),
                          cam_fps=cfg.get("cam_fps", 15), source_fps=cfg.get("source_fps", 10.0),
                          realtime=cfg.get("source_realtime", False),
                          on_frame=self.bridge.publish_frame if self.bridge else None,
                          host=cfg.get("host"), cam_tipo=cfg.get("cam_tipo", "mjpeg"))
        self.state["total"] = src.total()
        self.state["stamp_kind"] = src.stamp_kind
        self.state["ros2"] = bool(self.bridge)
        self.tracker.reset()
        if self.bridge is not None:
            hp = HybridParams(min_pose_rate_hz=float(cfg.get("hybrid_min_rate_hz", HybridParams.min_pose_rate_hz)),
                              stella_scale=str(cfg.get("stella_scale", HybridParams.stella_scale)))   # etapa 9
            self.ref_trackers = {m: ReferenceTracker(m, hp) for m in TrackingMode}
        # Vista del video a su propio ritmo (no al del modelo): con una cámara en vivo,
        # un hilo manda el último frame de la cámara a ~preview_fps; si el navegador no
        # alcanza, el servidor descarta frames en vez de encolarlos (ver broadcast).
        live_preview = src.is_live and cfg.get("preview", True)
        if live_preview:
            threading.Thread(target=self._preview_loop, args=(src,), daemon=True).start()

        self._status("cargando el modelo (sólo ahora se toca la GPU)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = argparse.Namespace(
            model_path=cfg["model_path"], image_size=518, patch_size=14,
            enable_3d_rope=True, max_frame_num=cfg.get("max_frame_num", 16384),
            num_scale_frames=cfg["num_scale_frames"],
            kv_cache_sliding_window=cfg["kv_cache_sliding_window"],
            use_sdpa=True, camera_num_iterations=cfg["camera_num_iterations"],
        )
        t0 = time.time()
        model = demo.load_model(args, device)
        self._model = model
        dtype = torch.float32
        if device.type == "cuda":
            cap = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
            model.aggregator.to(dtype=dtype)
        model.eval()
        self._status(f"modelo listo en {time.time()-t0:.1f}s, esperando frames...")

        scale_n = cfg["num_scale_frames"]
        max_pts = cfg["points_per_frame"]
        conf_pct = cfg["conf_percentile"]
        period = 1.0 / cfg["fps"] if cfg.get("fps") else 0.0

        model.clean_kv_cache()
        src.resume()                          # realtime: la grabación empieza recién ahora
        buf, idx, t_start, last = [], 0, time.time(), 0.0
        rng = np.random.default_rng(0)

        def preprocess(bgr):
            """Mismo crop/resize que load_and_preprocess_images en modo 'crop',
            pero sobre el array en memoria (sin pasar por disco)."""
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            nw = 518
            nh = round(h * nw / w / 14) * 14
            r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            if nh > 518:                       # recorte centrado, igual que el pipeline
                top = (nh - 518) // 2
                r = r[top:top + 518]
            t = torch.from_numpy(r).permute(2, 0, 1).float() / 255.0
            return t, r

        gate = None
        if cfg.get("context"):
            from context_gate import ContextGate
            gate = ContextGate(step_px=cfg.get("context_step_px", 36.0), max_skip=cfg.get("context_max_skip", 6),
                               synth=bool(cfg.get("context_synth")),
                               synth_strength=float(cfg.get("context_synth_strength", 1.0)))
        self.state["context"] = None
        static_hold = None
        if cfg.get("static_hold", True):
            from context_gate import StaticHold
            static_hold = StaticHold(step_px=cfg.get("context_step_px", 36.0),
                                     static_frac=float(cfg.get("static_frac", 0.25)))
        self.static_hold = static_hold
        buf_synth = []
        buf_meta = []
        n_real = [0]

        special_keep = int(cfg.get("special_keep", 64))
        camera_keep = int(cfg.get("camera_keep", 1024))

        def bound_cache():
            """Memoria acotada en sesiones largas. Dos cachés del modelo crecen sin límite:
            - cuando un frame sale de la ventana, el agregador conserva sus tokens especiales
              (cámara, registro, escala) para siempre: ~1.1 MB por frame;
            - la cabeza de cámara guarda un token de pose por frame y nunca desaloja (el
              desalojo del modelo solo actúa con más de un token por frame): ~0.25 MB por frame.
            Se dejan los especiales de los últimos special_keep frames desalojados y, en la cabeza
            de cámara, los frames de escala más los últimos camera_keep. Formato de las cachés:
            [B, cabezas, frames, tokens, dim]."""
            agg = getattr(model.aggregator, "kv_cache", None)
            cam = getattr(model.camera_head, "kv_cache", None)
            cam = [c for c in cam if isinstance(c, dict)] if isinstance(cam, list) else []
            for c in ([agg] if isinstance(agg, dict) else []) + cam:
                for key, val in list(c.items()):
                    if not (torch.is_tensor(val) and val.dim() >= 3):
                        continue
                    if key.endswith("_special"):
                        if 0 < special_keep < val.shape[2]:
                            c[key] = val[:, :, -special_keep:].contiguous()
                    elif c is not agg and key.startswith(("k_", "v_")) and 0 < camera_keep \
                            and val.shape[2] > scale_n + camera_keep:
                        c[key] = torch.cat([val[:, :, :scale_n], val[:, :, -camera_keep:]], dim=2).contiguous()

        kf_int = max(1, int(cfg.get("keyframe_interval", 1)))
        n_stream = [0]

        def run_model(t_img, rgb_small, synth, meta):
            """Un frame al modelo. Los sintéticos solo alimentan el contexto (KV cache).
            `meta` (FrameMeta) es la hora de captura e identidad del frame: viaja con la imagen
            porque el analizador puede elegir un frame anterior al último leído."""
            nonlocal idx, buf
            amp = torch.amp.autocast("cuda", dtype=dtype) if device.type == "cuda" \
                else torch.autocast("cpu", enabled=False)
            with torch.no_grad(), amp:
                if idx < scale_n:
                    buf.append(t_img)
                    buf_synth.append(synth)
                    buf_meta.append(meta)
                    if len(buf) < scale_n:
                        idx += 1
                        return
                    batch = torch.stack(buf).unsqueeze(0).to(device)
                    out = model.forward(batch, num_frame_for_scale=scale_n,
                                        num_frame_per_block=scale_n, causal_inference=True)
                    imgs_for_pose = batch
                    emit_range = range(scale_n)
                else:
                    batch = t_img.unsqueeze(0).unsqueeze(0).to(device)
                    # keyframes: su KV queda en la caché; el resto la consulta sin quedarse
                    is_kf = kf_int <= 1 or n_stream[0] % kf_int == 0
                    n_stream[0] += 1
                    if not is_kf:
                        model._set_skip_append(True)
                    out = model.forward(batch, num_frame_for_scale=scale_n,
                                        num_frame_per_block=1, causal_inference=True)
                    if not is_kf:
                        model._set_skip_append(False)
                    imgs_for_pose = batch
                    emit_range = range(1)

                extr, intr = pose_encoding_to_extri_intri(out["pose_enc"], imgs_for_pose.shape[-2:])
                e4 = torch.zeros((*extr.shape[:-2], 4, 4), device=extr.device, dtype=extr.dtype)
                e4[..., :3, :4] = extr
                e4[..., 3, 3] = 1.0
                # misma convención que el resto del repo: lo guardado es w2c y el
                # mundo sale de invertirlo (ver export_dense_cloud.py)
                w2c = closed_form_inverse_se3_general(e4)[..., :3, :4]
                depth = out["depth"].float().cpu().numpy()
                conf = out["depth_conf"].float().cpu().numpy()
                K = intr.float().cpu().numpy()
                W2C = w2c.float().cpu().numpy()
            del out
            bound_cache()

            # Vista de referencia: el frame real que entró al modelo, en JPEG.
            # Se marca con frame_idx = 0xFFFFFFFF, que nunca usa un frame real,
            # así el cliente distingue los dos tipos sin romper el protocolo.
            if cfg.get("preview", True) and not synth and not live_preview:
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR),
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    self._emit({"__bin__": struct.pack("<I", 0xFFFFFFFF) + jpg.tobytes()})

            for j in emit_range:
                is_synth = buf_synth[j] if len(emit_range) > 1 else synth
                if is_synth:
                    continue
                dj = depth[0, j, ..., 0] if depth.ndim == 5 else depth[0, j]
                cj = conf[0, j]
                Kj = K[0, j].astype(np.float64)
                E = np.eye(4); E[:3, :4] = W2C[0, j].astype(np.float64)
                c2w = np.linalg.inv(E)
                meta_j = buf_meta[j] if len(emit_range) > 1 else meta
                # el frame emitido puede venir del buffer de escala: usar su propia imagen
                img_src = rgb_small if len(emit_range) == 1 else None
                if img_src is None:
                    img_src = (buf[j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                # momentos estáticos (misma regla de movimiento que el analizador): si la cámara no se
                # movió, la pose no avanza; el frame igual aporta sus puntos al mapa
                c2w_model = c2w
                quieto, mov = False, None
                if static_hold is not None:
                    c2w, quieto, mov = static_hold.update(
                        c2w_model, img_src, meta_j.motion_px if (gate is not None and meta_j is not None) else None)
                # tracking BASIC: la pose de este frame con su tiempo, identidad y señales
                est = self.tracker.estimate(meta_j, c2w, cj)
                self.basic_buffer.add(est.stamp, est.c2w)
                # tracking STELLA para el mismo instante (etapas 3 y 5): desde el buffer de Stella
                # (exacta, interpolada o la más cercana); None si no hay. Solo se registra: la
                # geometría se sigue registrando con BASIC. Al guardar la sesión se vuelve a consultar
                # con todas las poses ya llegadas (asociación "a posteriori").
                se = self.bridge.stella_estimate(est.stamp) if self.bridge else None
                s_status = self.bridge.stella.status() if self.bridge else None
                if cfg.get("record", True):
                    self.rec["pose_stella"].append(se.c2w.astype(np.float32) if se is not None
                                                   else np.full((4, 4), np.nan, np.float32))
                    self.rec["stella_status"].append(int(s_status) if s_status is not None else 0)
                    self.rec["stella_assoc_live"].append(int(QueryKind[se.confidence["assoc"]]) if se is not None else 0)
                    self.rec["stella_dt_live"].append(float(se.confidence["dt_nearest"]) if se is not None else np.inf)
                ref_est = None
                if self.ref_trackers is not None:
                    win = self.ref_trackers[TrackingMode.HYBRID].p.rate_window_s
                    rate = self.bridge.stella.buffer.count_between(est.stamp - win, est.stamp) / win
                    seg = se.confidence["segment"] if se is not None else -1
                    outs = {m: tr.update(est.stamp, est.c2w, None if se is None else se.c2w, seg, s_status,
                                         stella_rate_hz=rate) for m, tr in self.ref_trackers.items()}
                    ref_est = outs[self.tracking_mode]
                    if cfg.get("record", True):
                        self.rec["pose_ref_stella"].append(outs[TrackingMode.STELLA].c2w.astype(np.float32))
                        self.rec["pose_ref_hybrid"].append(outs[TrackingMode.HYBRID].c2w.astype(np.float32))
                        hc = outs[TrackingMode.HYBRID].confidence
                        self.rec["hybrid_usado"].append(1 if hc["usado"] == "STELLA" else 0)
                        self.rec["hybrid_motivo"].append(hc["motivo"] or "")
                        self.rec["stella_rate"].append(rate)
                # etapa 10: con register_with_reference, la geometría se registra con la pose de
                # referencia del selector (mundo y escala BASIC) en vez de BASIC
                c2w_reg = c2w
                src_reg = 0
                if cfg.get("register_with_reference") and ref_est is not None:
                    c2w_reg = ref_est.c2w
                    src_reg = {TrackingMode.BASIC: 0, TrackingMode.STELLA: 1, TrackingMode.HYBRID: 2}[self.tracking_mode]
                lat = (time.time() - meta_j.wall) if meta_j.wall is not None else np.nan
                if self.bridge is not None and se is not None:
                    self.kf_anchors.register(n_real[0], est.stamp, se.c2w, self.bridge.stella.kf_history)
                    self.frame_seg[n_real[0]] = se.confidence["segment"]
                if cfg.get("record", True):
                    self.rec["extrinsic_reg"].append(np.linalg.inv(c2w_reg)[:3, :4].astype(np.float32))
                    self.rec["pose_source_reg"].append(src_reg)
                    self.rec["latency"].append(lat)
                if cfg.get("record", True):
                    self.rec["depth"].append(dj.astype(np.float16))
                    self.rec["conf"].append(cj.astype(np.float16))
                    self.rec["images"].append(np.ascontiguousarray(img_src))
                    self.rec["extrinsic"].append(W2C[0, j].astype(np.float32))
                    self.rec["intrinsic"].append(K[0, j].astype(np.float32))
                    self.rec["stamps"].append(est.stamp)
                    self.rec["frame_ids"].append(est.frame_id)
                    self.rec["pose_basic"].append(est.c2w)
                    self.rec["pose_model"].append(c2w_model)
                    self.rec["estatico"].append(quieto)
                    self.rec["track_conf"].append(np.nan if est.confidence["conf_mean"] is None else est.confidence["conf_mean"])
                    self.rec["track_motion"].append(np.nan if meta_j.motion_px is None else meta_j.motion_px)
                    self.rec["track_sharp"].append(np.nan if meta_j.sharpness is None else meta_j.sharpness)
                fidx = n_real[0]
                n_real[0] += 1
                # mensaje de texto aparte del binario (que no cambia): el visor ignora los tipos
                # que no conoce, así que es compatible con el cliente actual
                tmsg = {"type": "tracking", "frame_idx": fidx, **est.to_json()}
                if self.bridge is not None:
                    tmsg["stella"] = None if se is None else se.to_json()
                    tmsg["stella_status"] = s_status.name
                tmsg["latencia_s"] = None if not np.isfinite(lat) else round(float(lat), 3)
                tmsg["estatico"] = quieto
                tmsg["movimiento_px"] = None if mov is None else round(float(mov), 2)
                tmsg["stamp_kind"] = self.state.get("stamp_kind")
                if ref_est is not None:
                    tmsg["referencia"] = {"modo": self.tracking_mode.value, "usado": ref_est.confidence["usado"],
                                          "motivo": ref_est.confidence["motivo"],
                                          "registra_geometria": bool(src_reg) or self.tracking_mode == TrackingMode.BASIC
                                          if cfg.get("register_with_reference") else False,
                                          "position": [round(float(v), 5) for v in ref_est.position],
                                          "ritmo_stella_hz": round(float(rate), 1)}
                    # las tres trayectorias en el mismo mundo que la nube (etapa 16)
                    tmsg["trayectorias"] = {m.value: [round(float(v), 5) for v in outs[m].position] for m in TrackingMode}
                    self.bridge.publish_reference(ref_est)
                    self.bridge.publish_reference_tf(ref_est)
                    # alineación Stella -> mapa del modo STELLA (Sim3 anclada al empezar cada mapa)
                    anchors = self.ref_trackers[TrackingMode.STELLA].anchor
                    for sg, (sa, Ra, ta) in anchors.items():
                        # con stella_scale="window" (etapa 9) la escala y la traslación cambian: se
                        # vuelve a publicar cuando cambian, no sólo al anclar el mapa
                        if self._published_anchors.get(sg) != (sa, tuple(ta)):
                            self.bridge.publish_stella_alignment(sg, sa, Ra, ta)
                            self._published_anchors[sg] = (sa, tuple(ta))
                self._emit(tmsg)
                for sink in self.estimate_sinks:
                    try:
                        sink(est)
                    except Exception:
                        traceback.print_exc()

                thr = np.percentile(cj, conf_pct) if conf_pct > 0 else -np.inf
                m = np.isfinite(dj) & (dj > 0) & (cj >= thr)
                v, u = np.nonzero(m)
                if len(v) == 0:
                    continue
                if max_pts and len(v) > max_pts:
                    sel = rng.choice(len(v), max_pts, replace=False)
                    u, v = u[sel], v[sel]
                z = dj[v, u].astype(np.float64)
                x = (u - Kj[0, 2]) / Kj[0, 0] * z
                y = (v - Kj[1, 2]) / Kj[1, 1] * z
                pts = (np.stack([x, y, z], 1) @ c2w_reg[:3, :3].T + c2w_reg[:3, 3]).astype(np.float32)
                cols = img_src[v, u].astype(np.uint8)

                blob = HEADER.pack(fidx, len(pts), *c2w_reg.astype(np.float32).ravel().tolist())
                self._emit({"__bin__": blob + pts.tobytes() + cols.tobytes()})
                self.emitted_pose[fidx] = np.asarray(c2w_reg, np.float64)
            idx += 1
            apply_kf_corrections()
            if self.bridge is not None and n_real[0] % 10 == 0:
                emit_keyframes()
            # gancho de prueba (etapa 18, prueba 15): el modelo se detiene un rato; la cámara y Stella siguen
            if cfg.get("debug_stall_at_frame") is not None and n_real[0] >= int(cfg["debug_stall_at_frame"]) \
                    and not self.state.get("_stalled"):
                self.state["_stalled"] = True
                self._status(f"PRUEBA: modelo detenido {cfg.get('debug_stall_s', 10)} s")
                time.sleep(float(cfg.get("debug_stall_s", 10)))
            # gancho de prueba (etapa 18, prueba 14): Stella se cae (SIGKILL) a mitad de sesión
            if cfg.get("debug_kill_stella_at_frame") is not None and self.stella_proc is not None \
                    and n_real[0] >= int(cfg["debug_kill_stella_at_frame"]) and not self.state.get("_killed"):
                import signal
                self.state["_killed"] = True
                try:
                    os.killpg(self.stella_proc.proc.pid, signal.SIGKILL)
                except Exception:
                    traceback.print_exc()
                self._emit({"type": "status", "msg": "PRUEBA: Stella detenida con SIGKILL"})

        def emit_keyframes():
            """Etapa 16: posiciones de los keyframes vigentes de Stella llevadas al mundo de la nube con
            la Sim(3) del mapa actual (modo STELLA). Sólo si ese mapa ya está anclado."""
            anchors = self.ref_trackers[TrackingMode.STELLA].anchor if self.ref_trackers else {}
            if not self.frame_seg:
                return
            sg = self.frame_seg[max(self.frame_seg)]
            if sg not in anchors:
                return
            sa, R, t = anchors[sg]
            cur = self.bridge.stella.kf_history.current
            if not cur:
                return
            P = np.array([T[:3, 3] for _, T in cur.values()])
            W = (sa * (P @ R.T)) + t
            self._emit({"type": "keyframes", "segmento": int(sg), "positions": np.round(W, 5).tolist()})

        def apply_kf_corrections():
            """Etapa 12: si Stella movió keyframes, recalcular la pose de los frames anclados y, si la
            geometría se registra con el modo STELLA, re-registrarla (mensaje `repose` al visor)."""
            if self.bridge is None or self.ref_trackers is None:
                return
            hist = self.bridge.stella.kf_history
            if hist.version == self.kf_version_seen or not hist.events:
                return
            self.kf_version_seen = hist.version
            anchors = self.ref_trackers[TrackingMode.STELLA].anchor
            corr = self.kf_anchors.corrections(hist)
            moved = 0
            mx = 0.0
            for fid, (_, Tc, disp) in corr.items():
                sg = self.frame_seg.get(fid)
                if sg not in anchors or disp <= 0:
                    continue
                sa, R, t = anchors[sg]
                T = np.eye(4)
                T[:3, :3] = R @ Tc[:3, :3]
                T[:3, 3] = sa * R @ Tc[:3, 3] + t
                if self.tracking_mode == TrackingMode.STELLA and cfg.get("register_with_reference") \
                        and fid in self.emitted_pose:
                    d = float(np.linalg.norm(T[:3, 3] - self.emitted_pose[fid][:3, 3]))
                    if d > 1e-3:
                        delta = T @ np.linalg.inv(self.emitted_pose[fid])
                        self._emit({"type": "repose", "frame_idx": int(fid),
                                    "delta": [round(float(v), 7) for v in delta.ravel()]})
                        self.emitted_pose[fid] = T
                        moved += 1
                        mx = max(mx, d)
            ev = hist.events[-1]
            info = {"type": "correccion", "keyframes_movidos": len(ev[3]), "desplazamiento_kf_max": round(ev[2], 5),
                    "frames_re_registrados": moved, "desplazamiento_frame_max": round(mx, 5)}
            self.corrections.append(info)
            self._emit(info)

        stride = max(1, int(cfg.get("stride", 1)))
        n_read = 0
        while not self.stop_flag.is_set():
            tick = time.time()
            r = src.read()
            if r is None:
                if gate is not None:
                    for rgb_g, syn, meta_g in gate.flush():
                        run_model(torch.from_numpy(rgb_g).permute(2, 0, 1).float() / 255.0, rgb_g, syn, meta_g)
                self._emit({"type": "done", "frames": n_real[0]})
                break
            bgr, meta = r
            n_read += 1
            if (n_read - 1) % stride:
                continue
            t_img, rgb_small = preprocess(bgr)
            if gate is None:
                run_model(t_img, rgb_small, False, meta)
            else:
                for rgb_g, syn, meta_g in gate.feed(rgb_small, meta):
                    run_model(torch.from_numpy(rgb_g).permute(2, 0, 1).float() / 255.0, rgb_g, syn, meta_g)
                self.state["context"] = dict(gate.stats)

            now = time.time()
            self.state["frames"] = n_real[0]
            if now - last > 0.4:
                last = now
                vram = int(torch.cuda.memory_allocated() / 2**20) if device.type == "cuda" else 0
                extra = {"ros2": self.bridge.summary()} if self.bridge else {}
                self._status(f"mapeando en vivo ({n_real[0]} frames)", frames=n_real[0],
                             fps=round(n_real[0] / max(now - t_start, 1e-3), 2), vram_mb=vram, **extra)
            if period:
                sleep = period - (time.time() - tick)
                if sleep > 0:
                    time.sleep(sleep)

        src.close()
        self._stop_ros2()
        self._status("liberando el modelo...")


# ---------------------------------------------------------------------------
# Trabajos en segundo plano: construir mapas de una prueba
# ---------------------------------------------------------------------------
class JobRunner:
    """Corre src/mapas/build_maps.py como proceso aparte (uno a la vez) y va
    publicando su progreso por el WebSocket. Las líneas "### ..." que imprime
    build_maps.py marcan el inicio y el fin de cada tarea."""

    def __init__(self, broadcast):
        self.broadcast = broadcast
        self.proc = None
        self.state = {"running": False, "prueba": None, "tasks": [], "current": None,
                      "step": 0, "total": 0, "done": {}, "log": [], "rc": None, "msg": "sin trabajos"}

    def running(self):
        return self.proc is not None and self.proc.returncode is None

    async def start(self, prueba_dir, rid, npz, tasks):
        cmd = [sys.executable, "-u", os.path.join(REPO_ROOT, "src/mapas", "build_maps.py"),
               prueba_dir, "--tasks", ",".join(tasks)]
        if npz:
            cmd += ["--npz", npz]
        iso = os.path.join(REPO_ROOT, "src/gpu", "run_isolated.sh")
        if os.path.isfile(iso):
            # cgroup propio con tope de RAM: si un mapa agota la memoria muere el trabajo,
            # no VS Code ni este servidor (ver src/gpu/run_isolated.sh)
            cmd = [iso, "--name", "mapas", "--"] + cmd
        self.state = {"running": True, "prueba": rid, "tasks": tasks, "current": None, "step": 0,
                      "total": len(tasks), "done": {}, "log": [], "rc": None, "msg": "arrancando"}
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=REPO_ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True)   # grupo propio: cancelar también mata a los hijos
        asyncio.create_task(self._pump())
        self._emit()

    async def _pump(self):
        last = 0.0
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            txt = line.decode(errors="replace").rstrip()
            # tqdm reescribe la misma línea con \r: quedarse con lo último
            txt = txt.split("\r")[-1]
            if not txt:
                continue
            self.state["log"] = (self.state["log"] + [txt])[-40:]
            if txt.startswith("### "):
                parts = txt[4:].split()
                if parts[0] in ("ok", "falla") and len(parts) > 1:
                    self.state["done"][parts[1]] = parts[0]
                elif parts[0] != "fin":
                    self.state["current"] = parts[0]
                    self.state["step"] = len(self.state["done"]) + 1
                self.state["msg"] = txt[4:]
                self._emit()
            elif time.time() - last > 1.0:
                last = time.time()
                self._emit()
        rc = await self.proc.wait()
        self.state.update(running=False, rc=rc, current=None,
                          msg="terminado" if rc == 0 else f"terminado con fallas (código {rc})")
        self._emit()

    def cancel(self):
        if self.running():
            import signal
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def _emit(self):
        self.broadcast({"type": "job", **self.state})


# ---------------------------------------------------------------------------
# Servidor
# ---------------------------------------------------------------------------
def build_app(captures_dir, model_path):
    @web.middleware
    async def isolation(request, handler):
        # COOP/COEP: habilita SharedArrayBuffer, que el visor de splats usa para
        # ordenar las gaussianas en un worker sin copiar memoria. Todo es mismo origen.
        resp = await handler(request)
        resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        return resp

    app = web.Application(client_max_size=1 << 20, middlewares=[isolation])
    app["clients"] = set()
    app["state"] = {"session": None}
    app["captures_dir"] = captures_dir
    app["model_path"] = model_path

    def broadcast(obj):
        """Se llama desde el loop; despacha a todos los WS conectados."""
        dead = []
        drop = obj.get("__drop__")
        busy = app["busy"]
        for ws in app["clients"]:
            try:
                if drop:
                    # mensajes descartables (vista previa): si el envío anterior a este
                    # cliente no terminó, se salta este frame en vez de encolarlo
                    key = (id(ws), drop)
                    if key in busy:
                        continue
                    busy.add(key)
                    t = asyncio.create_task(ws.send_bytes(obj["__bin__"]))
                    t.add_done_callback(lambda _t, k=key: busy.discard(k))
                elif "__bin__" in obj:
                    asyncio.create_task(ws.send_bytes(obj["__bin__"]))
                else:
                    asyncio.create_task(ws.send_json(obj))
            except Exception:
                dead.append(ws)
        for ws in dead:
            app["clients"].discard(ws)

    app["busy"] = set()
    app["broadcast"] = broadcast
    app["jobs"] = JobRunner(broadcast)

    async def index(_req):
        return web.FileResponse(os.path.join(VIEWER_DIR, "index.html"))

    async def api_captures(req):
        return web.json_response(scan_captures(req.app["captures_dir"]))

    async def data_file(req):
        rel = req.match_info["path"]
        root = os.path.realpath(req.app["captures_dir"])
        full = os.path.realpath(os.path.join(root, rel))
        if not (full == root or full.startswith(root + os.sep)) or not os.path.isfile(full):
            raise web.HTTPNotFound()
        return web.FileResponse(full)      # aiohttp ya maneja Range

    async def live_devices(req):
        """Todas las cámaras de este equipo (por nombre, sin nodos de metadatos, probadas en paralelo y con
        reintentos: video_devices.py), los teléfonos por adb y las cámaras de otros equipos por SSH."""
        from video_devices import list_local_cameras
        cur = req.app["state"].get("session")
        in_use = None
        if cur is not None and cur.state.get("running") and cur.cfg.get("source") == "webcam":
            in_use = cur.cfg.get("device")
        out = await asyncio.get_running_loop().run_in_executor(None, list_local_cameras, in_use)
        # cámaras remotas: teléfonos conectados por adb (USB o Wi-Fi)
        remote = []
        try:
            from android_camera import remote_devices
            remote = await asyncio.get_running_loop().run_in_executor(None, remote_devices)
        except Exception as e:
            remote = [{"kind": "android", "usable": False, "label": f"adb no disponible: {e}"}]
        # cámaras de otros equipos por SSH (hosts de ~/.ssh/config y equipos recordados)
        ssh_list = []
        try:
            from ssh_camera import remote_devices as ssh_devices
            ssh_list = await asyncio.get_running_loop().run_in_executor(None, ssh_devices)
        except Exception as e:
            ssh_list = [{"kind": "ssh", "usable": False, "label": f"búsqueda por SSH falló: {e}"}]
        return web.json_response({"devices": out, "remote": remote, "ssh": ssh_list})

    async def live_status(req):
        s = req.app["state"]["session"]
        st = dict(s.state) if s else {"running": False, "msg": "detenido"}
        st["ros2_default"] = bool(req.app.get("ros2_default"))
        if s is not None:
            st["stella_config"] = s.cfg.get("stella_config")
            st["stella_config_nota"] = s.cfg.get("stella_config_nota")
        return web.json_response(st)

    async def live_start(req):
        cur = req.app["state"]["session"]
        if cur and cur.state["running"]:
            return web.json_response({"ok": False, "msg": "ya hay una sesión en vivo"}, status=409)
        if req.app["jobs"].running():
            return web.json_response({"ok": False, "msg": "hay un trabajo construyendo mapas (usa la GPU); "
                                      "esperá a que termine o cancelalo"}, status=409)
        body = await req.json()
        cfg = {
            "source": body.get("source", "folder"),
            "path": body.get("path"),
            "device": body.get("device", 0),
            "serial": body.get("serial"),
            "host": body.get("host"),                       # fuente "ssh": equipo (alias de ~/.ssh/config)
            "cam_tipo": body.get("cam_tipo", "mjpeg"),       # fuente "ssh": mjpeg | yuyv | y10cap
            "camera_id": str(body.get("camera_id", "0")),
            "device_label": body.get("device_label"),
            "rotation": int(body.get("rotation", 0)),
            "cam_size": body.get("cam_size", "1280x720"),
            "cam_fps": int(body.get("cam_fps", 15)),
            "fps": float(body.get("fps", 4)) or None,
            "max_frames": int(body.get("max_frames", 0)),
            "points_per_frame": int(body.get("points_per_frame", 6000)),
            "conf_percentile": float(body.get("conf_percentile", 30)),
            "preview": bool(body.get("preview", True)),
            "preview_fps": float(body.get("preview_fps", 15)),
            "preview_width": int(body.get("preview_width", 640)),
            "num_scale_frames": int(body.get("num_scale_frames", 2)),
            "kv_cache_sliding_window": int(body.get("kv_cache_sliding_window", 16)),
            "camera_num_iterations": int(body.get("camera_num_iterations", 4)),
            "model_path": req.app["model_path"],
            "record": bool(body.get("record", True)),
            "captures_dir": req.app["captures_dir"],
            "context": bool(body.get("context", False)),
            "context_synth": bool(body.get("context_synth", False)),
            "context_step_px": float(body.get("context_step_px", 36)),
            "context_synth_strength": float(body.get("context_synth_strength", 1)),
            "stride": int(body.get("stride", 1)),
            "source_fps": float(body.get("source_fps", 10)),   # fps nominal de carpetas (stamps sintéticos)
            "source_realtime": bool(body.get("source_realtime", False)),  # carpeta/video como cámara en vivo
            # etapa 3: puente ROS2 (tee de frames + poses de Stella) y arranque opcional de Stella
            "ros2": bool(body.get("ros2", False)),
            "stella_config": body.get("stella_config"),
            "stella_log_dir": body.get("stella_log_dir"),
            "ros2_image_topic": body.get("ros2_image_topic", "/paralingbot/camera/image_raw"),
            "ros2_stella_ns": body.get("ros2_stella_ns", "/stella"),
            "ros2_camera_info": body.get("ros2_camera_info") or body.get("stella_config"),
            "ros2_resize": body.get("ros2_resize"),
            "ros2_pub_every": int(body.get("ros2_pub_every", 1)),
            "stella_cpus": body.get("stella_cpus"),
            "tracking_mode": body.get("tracking_mode", "basic"),        # etapa 6: basic | stella | hybrid
            "register_with_reference": bool(body.get("register_with_reference", False)),   # etapa 10
            "stella_scale": str(body.get("stella_scale", "anchor")),                          # etapa 9
            "hybrid_min_rate_hz": float(body.get("hybrid_min_rate_hz", 10.0)),
            "special_keep": int(body.get("special_keep", 64)),
            "camera_keep": int(body.get("camera_keep", 1024)),
            "keyframe_interval": int(body.get("keyframe_interval", 1)),
            "static_hold": bool(body.get("static_hold", True)),     # momentos estáticos: la pose no avanza
            "static_frac": float(body.get("static_frac", 0.25)),    # fracción del paso del analizador
        }
        if cfg["source"] in ("folder", "video", "url") and not cfg["path"]:
            return web.json_response({"ok": False, "msg": "falta 'path'"}, status=400)
        # etapa 3 + launch general: con el puente activo por defecto, la sesión publica por ROS2 y arranca
        # Stella con una configuración de cámara para la imagen publicada (video_devices.stella_config_for)
        if "ros2" not in body:
            cfg["ros2"] = bool(req.app.get("ros2_default"))
        if cfg["ros2"] and not cfg.get("stella_config") and body.get("stella", True):
            from video_devices import stella_config_for
            try:
                yml, rs, nota = await asyncio.get_running_loop().run_in_executor(None, stella_config_for, cfg)
            except Exception as e:
                yml, rs, nota = None, None, f"no se pudo elegir la configuración de Stella: {e}"
            if yml:
                cfg["stella_config"], cfg["ros2_camera_info"], cfg["ros2_resize"] = yml, yml, rs
            cfg["stella_config_nota"] = nota
        if cfg["ros2"] and cfg.get("stella_config") and not cfg.get("stella_cpus") and (os.cpu_count() or 0) >= 16:
            cfg["stella_cpus"] = "4-11"          # medido en la etapa 6: Stella con núcleos propios rinde más
        if cfg["source"] == "ssh" and not cfg["host"]:
            return web.json_response({"ok": False, "msg": "falta el equipo ('host')"}, status=400)
        if cfg["source"] == "android" and not cfg["serial"]:
            return web.json_response({"ok": False, "msg": "falta el teléfono ('serial')"}, status=400)
        sess = LiveSession(asyncio.get_running_loop(), req.app["broadcast"], cfg)
        req.app["state"]["session"] = sess
        sess.start()
        return web.json_response({"ok": True, "cfg": {k: v for k, v in cfg.items()
                                                       if k not in ("model_path", "captures_dir")}})

    async def live_stop(req):
        s = req.app["state"]["session"]
        if s:
            s.stop()
        return web.json_response({"ok": True})

    # -- explorador de pruebas ----------------------------------------------
    def bad(e, status=400):
        return web.json_response({"ok": False, "msg": str(e)}, status=status)

    async def api_explorer(req):
        return web.json_response(catalog.scan_pruebas(req.app["captures_dir"]))

    async def api_files(req):
        try:
            return web.json_response(catalog.list_dir(req.app["captures_dir"], req.query.get("prueba", ""),
                                                      req.query.get("dir", "")))
        except ValueError as e:
            return bad(e)

    async def api_meta(req):
        b = await req.json()
        try:
            info = catalog.update_meta(req.app["captures_dir"], b.get("prueba", ""), b)
        except ValueError as e:
            return bad(e)
        return web.json_response({"ok": True, "info": info})

    async def api_save(req):
        b = await req.json()
        try:
            nid = catalog.save_session(req.app["captures_dir"], b.get("prueba", ""), b.get("destino", ""), b)
        except (ValueError, OSError) as e:
            return bad(e)
        return web.json_response({"ok": True, "prueba": nid})

    async def api_discard(req):
        b = await req.json()
        try:
            catalog.discard_session(req.app["captures_dir"], b.get("prueba", ""))
        except (ValueError, OSError) as e:
            return bad(e)
        return web.json_response({"ok": True})

    async def jobs_start(req):
        jobs = req.app["jobs"]
        if jobs.running():
            return bad("ya hay un trabajo en curso", 409)
        s = req.app["state"]["session"]
        if s and s.state["running"]:
            return bad("hay una sesión en vivo usando la GPU; detenela primero", 409)
        b = await req.json()
        try:
            pdir = catalog.resolve_prueba(req.app["captures_dir"], b.get("prueba", ""))
        except ValueError as e:
            return bad(e)
        tasks = [t for t in b.get("tasks", []) if t in ("windowed", "filtro", "nube", "alta", "cruda", "malla", "malla_f",
                                                         "splat", "splat_f", "video")]
        if not tasks:
            return bad("no hay tareas válidas")
        await jobs.start(pdir, b["prueba"], b.get("npz"), tasks)
        return web.json_response({"ok": True})

    async def jobs_cancel(req):
        req.app["jobs"].cancel()
        return web.json_response({"ok": True})

    async def jobs_status(req):
        return web.json_response(req.app["jobs"].state)

    async def ws_handler(req):
        ws = web.WebSocketResponse(max_msg_size=0, heartbeat=30)
        await ws.prepare(req)
        req.app["clients"].add(ws)
        s = req.app["state"]["session"]
        await ws.send_json({"type": "status", **(s.state if s else {"running": False, "msg": "detenido"})})
        await ws.send_json({"type": "job", **req.app["jobs"].state})
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT and msg.data == "ping":
                    await ws.send_json({"type": "pong"})
        finally:
            req.app["clients"].discard(ws)
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/api/captures", api_captures)
    app.router.add_get("/api/live/devices", live_devices)
    app.router.add_get("/api/live/status", live_status)
    app.router.add_post("/api/live/start", live_start)
    app.router.add_post("/api/live/stop", live_stop)
    app.router.add_get("/api/explorer", api_explorer)
    app.router.add_get("/api/files", api_files)
    app.router.add_post("/api/prueba/meta", api_meta)
    app.router.add_post("/api/prueba/save", api_save)
    app.router.add_post("/api/prueba/discard", api_discard)
    app.router.add_post("/api/jobs/start", jobs_start)
    app.router.add_post("/api/jobs/cancel", jobs_cancel)
    app.router.add_get("/api/jobs", jobs_status)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/data/{path:.*}", data_file)
    app.router.add_static("/vendor/", os.path.join(VIEWER_DIR, "vendor"))
    for f in ("main.js", "live.js", "explorer.js"):
        p = os.path.join(VIEWER_DIR, f)
        if os.path.isfile(p):
            app.router.add_get("/" + f, lambda r, _p=p: web.FileResponse(_p))
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--captures_dir", default=os.path.join(REPO_ROOT, "captures"))
    p.add_argument("--model_path", default=os.path.join(REPO_ROOT, "checkpoints", "lingbot-map.pt"))
    p.add_argument("--cpus", default=None,
                   help="fijar el servidor (y el modelo) a estas CPUs, p. ej. 0-3,12-19, para dejar las demás a Stella")
    p.add_argument("--ros2", action="store_true",
                   help="sesiones en vivo con el puente ROS2 + Stella por defecto (lo usa launch.py)")
    args = p.parse_args()
    if args.cpus:
        from cpu_affinity import pin_process
        print(f"CPUs del servidor: {sorted(pin_process(args.cpus))}", flush=True)
    app = build_app(os.path.abspath(args.captures_dir), args.model_path)
    app["ros2_default"] = bool(args.ros2)
    print(f"Visor + streaming en http://localhost:{args.port}", flush=True)
    print(f"Capturas: {os.path.abspath(args.captures_dir)}", flush=True)
    print("El modelo se carga sólo al iniciar una sesión en vivo.", flush=True)
    # Apagado ordenado con SIGTERM: aiohttp espera hasta shutdown_timeout (60 s por defecto) a que se
    # cierren las conexiones, y el visor del navegador mantiene un WebSocket abierto: el proceso
    # soltaba el puerto pero no terminaba (visto dos veces el 2026-10-04). Se cierran los WebSockets
    # al apagar y se detiene la sesión en vivo si la hay (su finally cierra Stella y el puente ROS2).
    async def on_shutdown(app_):
        s = app_["state"].get("session")
        if s is not None and s.state.get("running"):
            s.stop()
        for ws in list(app_["clients"]):
            try:
                await ws.close(code=1001, message=b"servidor detenido")
            except Exception:
                pass

    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host="0.0.0.0", port=args.port, print=None, shutdown_timeout=5.0)


if __name__ == "__main__":
    main()
