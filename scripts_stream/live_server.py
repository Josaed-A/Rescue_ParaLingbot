"""Servidor único del visor: archivos estáticos, nubes ya exportadas y mapeo EN VIVO.

Un solo proceso y un solo puerto para las dos cosas, para no duplicar servidores:

  GET  /                     el visor (scripts_context/webgl_viewer/)
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
  POST /api/jobs/start       construir mapas de una prueba (scripts_context/build_maps.py)
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
  python3 scripts_stream/live_server.py [--port 8090]
  python3 scripts_stream/live_server.py --port 8090 --captures_dir captures
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
REPO_ROOT = os.path.dirname(HERE)
VIEWER_DIR = os.path.join(REPO_ROOT, "scripts_context", "webgl_viewer")
sys.path.insert(0, REPO_ROOT)

from scripts_context.webgl_viewer.captures_index import scan_captures  # noqa: E402
from scripts_context.webgl_viewer import catalog  # noqa: E402

HEADER = struct.Struct("<II16f")


# ---------------------------------------------------------------------------
# Fuentes de frames
# ---------------------------------------------------------------------------
class FrameSource:
    """Entrega frames BGR (numpy) uno por uno. Cierra con close()."""

    def __init__(self, kind, path=None, device=0, fps=None, max_frames=0, start=0):
        import cv2
        self.cv2 = cv2
        self.kind = kind
        self.fps = fps
        self.max_frames = max_frames
        self.n = 0
        self._cap = None
        self._files = None
        self._i = start
        if kind == "webcam":
            self._cap = cv2.VideoCapture(int(device), cv2.CAP_V4L2)
            if not self._cap.isOpened():
                raise RuntimeError(f"no se pudo abrir la cámara {device}")
        elif kind == "video":
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise RuntimeError(f"no se pudo abrir el video {path}")
        elif kind == "folder":
            exts = (".png", ".jpg", ".jpeg")
            self._files = sorted(
                os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(exts)
            )
            if not self._files:
                raise RuntimeError(f"la carpeta no tiene imágenes: {path}")
        else:
            raise ValueError(f"fuente desconocida: {kind}")

    def total(self):
        if self._files is not None:
            n = len(self._files) - self._i
        elif self.kind == "video":
            n = int(self._cap.get(self.cv2.CAP_PROP_FRAME_COUNT)) or 0
        else:
            n = 0            # la webcam no tiene total
        return min(n, self.max_frames) if self.max_frames else n

    def read(self):
        if self.max_frames and self.n >= self.max_frames:
            return None
        if self._files is not None:
            if self._i >= len(self._files):
                return None
            img = self.cv2.imread(self._files[self._i])
            self._i += 1
        else:
            ok, img = self._cap.read()
            if not ok:
                return None
        self.n += 1
        return img

    def close(self):
        if self._cap is not None:
            self._cap.release()


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
        self.rec = {"depth": [], "conf": [], "images": [], "extrinsic": [], "intrinsic": []}
        self.t_begin = time.time()

    # -- utilidades ---------------------------------------------------------
    def _emit(self, obj):
        self.loop.call_soon_threadsafe(self.broadcast, obj)

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
            self._release()
            try:
                self._save_session()
            except Exception as e:
                traceback.print_exc()
                self._emit({"type": "error", "msg": f"no se pudo guardar la sesión: {e}"})
            self.state["running"] = False
            self._status("detenido")
            self._emit({"type": "stopped", "saved": self.state.get("saved")})

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
        np.savez(os.path.join(P, "eval", "sesion.npz"),
                 depth=np.stack(self.rec["depth"]), depth_conf=np.stack(self.rec["conf"]),
                 images=np.stack(self.rec["images"]), extrinsic=np.stack(self.rec["extrinsic"]),
                 intrinsic=np.stack(self.rec["intrinsic"]), ds=1,
                 is_real=np.ones(n, dtype=bool), source_index=np.arange(n))
        for k, im in enumerate(self.rec["images"]):
            cv2.imwrite(os.path.join(P, "frames", f"{k:06d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        src = self.cfg.get("source")
        cats = ["streaming", "sin guardar"] + (["webcam"] if src == "webcam" else [])
        info = {
            "titulo": f"Sesión en vivo {time.strftime('%Y-%m-%d %H:%M', time.localtime(self.t_begin))}",
            "zona": "", "categorias": cats, "notas": "",
            "fecha": time.strftime("%Y-%m-%d", time.localtime(self.t_begin)),
            "modo": "streaming (en vivo): el mapa deriva; reprocesar con la tarea 'windowed'",
            "fuente": {"tipo": src, "path": self.cfg.get("path"), "device": self.cfg.get("device")},
            "frames": n, "fps_objetivo": self.cfg.get("fps"), "fps_real": self.state.get("fps"),
            "parametros": {k: self.cfg[k] for k in ("num_scale_frames", "kv_cache_sliding_window",
                                                    "camera_num_iterations", "conf_percentile")},
            "analizador_contexto": ({"sintesis": self.cfg.get("context_synth"),
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

    def _infer(self):
        import torch
        import cv2
        import demo
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
        from lingbot_map.utils.geometry import closed_form_inverse_se3_general

        cfg = self.cfg
        self._status("abriendo la fuente de frames...")
        src = FrameSource(cfg["source"], path=cfg.get("path"), device=cfg.get("device", 0),
                          fps=cfg.get("fps"), max_frames=cfg.get("max_frames", 0))
        self.state["total"] = src.total()

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
                               synth=bool(cfg.get("context_synth")))
        self.state["context"] = None
        buf_synth = []
        n_real = [0]

        special_keep = int(cfg.get("special_keep", 64))

        def bound_cache():
            """Memoria acotada en sesiones largas. Cuando un frame sale de la ventana, el modelo
            conserva sus tokens especiales (cámara, registro, escala) y los concatena para
            siempre: ~3 MB de VRAM por frame, y una sesión de unos 550 frames llenaba los 8 GB.
            Se dejan los de los últimos special_keep frames desalojados."""
            if special_keep <= 0:
                return
            caches = []
            kv = getattr(model.aggregator, "kv_cache", None)
            if isinstance(kv, dict):
                caches.append(kv)
            ch = getattr(model.camera_head, "kv_cache", None)
            if isinstance(ch, list):
                caches += [c for c in ch if isinstance(c, dict)]
            for c in caches:
                for key, val in c.items():
                    if key.endswith("_special") and torch.is_tensor(val) and val.dim() >= 3 \
                            and val.shape[2] > special_keep:
                        c[key] = val[:, :, -special_keep:].contiguous()

        kf_int = max(1, int(cfg.get("keyframe_interval", 1)))
        n_stream = [0]

        def run_model(t_img, rgb_small, synth):
            """Un frame al modelo. Los sintéticos solo alimentan el contexto (KV cache)."""
            nonlocal idx, buf
            amp = torch.amp.autocast("cuda", dtype=dtype) if device.type == "cuda" \
                else torch.autocast("cpu", enabled=False)
            with torch.no_grad(), amp:
                if idx < scale_n:
                    buf.append(t_img)
                    buf_synth.append(synth)
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
            if cfg.get("preview", True) and not synth:
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
                # el frame emitido puede venir del buffer de escala: usar su propia imagen
                img_src = rgb_small if len(emit_range) == 1 else None
                if img_src is None:
                    img_src = (buf[j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                if cfg.get("record", True):
                    self.rec["depth"].append(dj.astype(np.float16))
                    self.rec["conf"].append(cj.astype(np.float16))
                    self.rec["images"].append(np.ascontiguousarray(img_src))
                    self.rec["extrinsic"].append(W2C[0, j].astype(np.float32))
                    self.rec["intrinsic"].append(K[0, j].astype(np.float32))
                fidx = n_real[0]
                n_real[0] += 1

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
                pts = (np.stack([x, y, z], 1) @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)
                cols = img_src[v, u].astype(np.uint8)

                blob = HEADER.pack(fidx, len(pts), *c2w.astype(np.float32).ravel().tolist())
                self._emit({"__bin__": blob + pts.tobytes() + cols.tobytes()})
            idx += 1

        stride = max(1, int(cfg.get("stride", 1)))
        n_read = 0
        while not self.stop_flag.is_set():
            tick = time.time()
            bgr = src.read()
            if bgr is None:
                if gate is not None:
                    for rgb_g, syn in gate.flush():
                        run_model(torch.from_numpy(rgb_g).permute(2, 0, 1).float() / 255.0, rgb_g, syn)
                self._emit({"type": "done", "frames": n_real[0]})
                break
            n_read += 1
            if (n_read - 1) % stride:
                continue
            t_img, rgb_small = preprocess(bgr)
            if gate is None:
                run_model(t_img, rgb_small, False)
            else:
                for rgb_g, syn in gate.feed(rgb_small):
                    run_model(torch.from_numpy(rgb_g).permute(2, 0, 1).float() / 255.0, rgb_g, syn)
                self.state["context"] = dict(gate.stats)

            now = time.time()
            self.state["frames"] = n_real[0]
            if now - last > 0.4:
                last = now
                vram = int(torch.cuda.memory_allocated() / 2**20) if device.type == "cuda" else 0
                self._status(f"mapeando en vivo ({n_real[0]} frames)", frames=n_real[0],
                             fps=round(n_real[0] / max(now - t_start, 1e-3), 2), vram_mb=vram)
            if period:
                sleep = period - (time.time() - tick)
                if sleep > 0:
                    time.sleep(sleep)

        src.close()
        self._status("liberando el modelo...")


# ---------------------------------------------------------------------------
# Trabajos en segundo plano: construir mapas de una prueba
# ---------------------------------------------------------------------------
class JobRunner:
    """Corre scripts_context/build_maps.py como proceso aparte (uno a la vez) y va
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
        cmd = [sys.executable, "-u", os.path.join(REPO_ROOT, "scripts_context", "build_maps.py"),
               prueba_dir, "--tasks", ",".join(tasks)]
        if npz:
            cmd += ["--npz", npz]
        iso = os.path.join(REPO_ROOT, "scripts_gpu", "run_isolated.sh")
        if os.path.isfile(iso):
            # cgroup propio con tope de RAM: si un mapa agota la memoria muere el trabajo,
            # no VS Code ni este servidor (ver scripts_gpu/run_isolated.sh)
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
        for ws in app["clients"]:
            try:
                if "__bin__" in obj:
                    asyncio.create_task(ws.send_bytes(obj["__bin__"]))
                else:
                    asyncio.create_task(ws.send_json(obj))
            except Exception:
                dead.append(ws)
        for ws in dead:
            app["clients"].discard(ws)

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

    async def live_devices(_req):
        """Qué /dev/videoN hay y cuáles abren de verdad (varios son nodos de
        metadatos que existen pero no entregan imagen)."""
        import glob
        import cv2
        out = []
        for path in sorted(glob.glob("/dev/video*"),
                           key=lambda p: int("".join(c for c in p if c.isdigit()) or 0)):
            idx = int("".join(c for c in path if c.isdigit()) or 0)
            info = {"index": idx, "path": path, "usable": False, "label": path}
            cap = None
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    ok, fr = cap.read()
                    if ok and fr is not None:
                        info["usable"] = True
                        info["w"], info["h"] = int(fr.shape[1]), int(fr.shape[0])
                        info["label"] = f"{path} ({fr.shape[1]}x{fr.shape[0]})"
            except Exception:
                pass
            finally:
                if cap is not None:
                    cap.release()
            out.append(info)
        return web.json_response({"devices": out})

    async def live_status(req):
        s = req.app["state"]["session"]
        return web.json_response(s.state if s else {"running": False, "msg": "detenido"})

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
            "fps": float(body.get("fps", 4)) or None,
            "max_frames": int(body.get("max_frames", 0)),
            "points_per_frame": int(body.get("points_per_frame", 6000)),
            "conf_percentile": float(body.get("conf_percentile", 30)),
            "preview": bool(body.get("preview", True)),
            "num_scale_frames": int(body.get("num_scale_frames", 2)),
            "kv_cache_sliding_window": int(body.get("kv_cache_sliding_window", 16)),
            "camera_num_iterations": int(body.get("camera_num_iterations", 4)),
            "model_path": req.app["model_path"],
            "record": bool(body.get("record", True)),
            "captures_dir": req.app["captures_dir"],
            "context": bool(body.get("context", False)),
            "context_synth": bool(body.get("context_synth", False)),
            "context_step_px": float(body.get("context_step_px", 36)),
            "stride": int(body.get("stride", 1)),
            "special_keep": int(body.get("special_keep", 64)),
            "keyframe_interval": int(body.get("keyframe_interval", 1)),
        }
        if cfg["source"] in ("folder", "video") and not cfg["path"]:
            return web.json_response({"ok": False, "msg": "falta 'path'"}, status=400)
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
    args = p.parse_args()
    app = build_app(os.path.abspath(args.captures_dir), args.model_path)
    print(f"Visor + streaming en http://localhost:{args.port}", flush=True)
    print(f"Capturas: {os.path.abspath(args.captures_dir)}", flush=True)
    print("El modelo se carga sólo al iniciar una sesión en vivo.", flush=True)
    web.run_app(app, host="0.0.0.0", port=args.port, print=None)


if __name__ == "__main__":
    main()
