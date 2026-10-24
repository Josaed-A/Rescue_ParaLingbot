"""Analizador de contexto para el mapeo en vivo (la idea de context-to-image, en streaming).

Decide, frame a frame y sin mirar el futuro, qué le llega al modelo:

  - movimiento: flujo óptico DIS (OpenCV, CPU, ~2 ms a 256 px) entre frames consecutivos;
    se acumula la mediana del desplazamiento desde el último frame enviado;
  - nitidez: varianza del Laplaciano, comparada con el percentil 75 de los últimos frames;
  - se envía un frame cuando el movimiento acumulado llega a step_px (los frames casi
    iguales al anterior son redundantes y se saltan), eligiendo el más nítido del tramo, o
    cuando ya se saltaron max_skip frames seguidos;
  - con synth=True, si entre el último frame enviado y el nuevo hay un salto grande (más de
    synth_factor x step_px / synth_strength, y menos de flow_max_px, donde la interpolación se
    desarma), se generan intermedios cada step_px / synth_strength (hasta max_synth por salto),
    por flujo bidireccional (aproximación de Super-SloMo). Esos
    frames entran al modelo solo como contexto temporal: no se dibujan ni se graban.

El mismo objeto sirve fuera de línea (replay_live.py) para medir su efecto.
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracking import FrameMeta  # noqa: E402


class ContextGate:
    def __init__(self, step_px=36.0, blur_rel=0.5, max_skip=6, synth=False, synth_factor=1.5,
                 flow_max_px=240.0, work=256, synth_strength=1.0, max_synth=8):
        self.step = step_px
        self.blur_rel = blur_rel
        self.max_skip = max_skip
        self.synth = synth
        # intensidad del amortiguador: x1 sintetiza en saltos > 1.5 pasos, un intermedio por paso;
        # xS baja el umbral y el espaciado de los intermedios S veces
        self.synth_strength = max(1.0, float(synth_strength))
        self.synth_factor = synth_factor / self.synth_strength
        self.synth_step = step_px / self.synth_strength
        self.max_synth = max_synth
        self.flow_max = flow_max_px
        self.work = work
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.prev_gray = None
        self.cum = 0.0                       # movimiento acumulado desde el último enviado
        self.cand = []                       # [(rgb, nitidez, cum, meta)] desde el último enviado
        self.last_sent = None
        self.last_meta = None                # FrameMeta del último real enviado (para los sintéticos)
        self.sharp_hist = []
        self.stats = {"read": 0, "sent": 0, "synth": 0, "skipped": 0, "blurry_avoided": 0,
                      "forced": 0, "motion_px_median": 0.0}
        self._motions = []

    def _gray(self, rgb):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        s = self.work / g.shape[1]
        return cv2.resize(g, (self.work, round(g.shape[0] * s)), interpolation=cv2.INTER_AREA), s

    def feed(self, rgb, meta=None):
        """rgb [H,W,3] uint8 (ya recortado para el modelo); meta: FrameMeta del frame (hora de
        captura e identificador), que viaja pegado a la imagen porque el frame elegido puede ser
        uno anterior al último leído. Devuelve [(rgb, sintético, meta)]; en los sintéticos el
        meta tiene el stamp interpolado y frame_id -1 (None si no se pasaron metas)."""
        self.stats["read"] += 1
        g, s = self._gray(rgb)
        sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
        self.sharp_hist = (self.sharp_hist + [sharp])[-60:]
        if self.prev_gray is None:                               # primer frame: siempre
            self.prev_gray = g
            return self._send(rgb, 0.0, meta, sharp)
        flow = self.dis.calc(self.prev_gray, g, None)
        self.prev_gray = g
        mot = float(np.median(np.linalg.norm(flow, axis=-1))) / s
        self._motions.append(mot)
        self.cum += mot
        self.cand.append((rgb, sharp, self.cum, meta))
        if self.cum < self.step and len(self.cand) < self.max_skip:
            self.stats["skipped"] = self.stats["read"] - self.stats["sent"]
            return []
        if self.cum < self.step:
            self.stats["forced"] += 1
        # el más nítido de la segunda mitad del tramo (para no quedar pegado al anterior)
        half = [k for k, c in enumerate(self.cand) if c[2] >= 0.5 * self.cum] or [len(self.cand) - 1]
        k = max(half, key=lambda k: self.cand[k][1])
        best = self.cand[k]
        p75 = float(np.percentile(self.sharp_hist, 75))
        if self.cand[-1][1] < self.blur_rel * p75 and k != len(self.cand) - 1:
            self.stats["blurry_avoided"] += 1
        out = self._send(best[0], best[2], best[3], best[1])
        rest = self.cand[k + 1:]
        self.cand = [(r, sh, c - best[2], m) for r, sh, c, m in rest]
        self.cum -= best[2]
        return out

    def flush(self):
        """Al terminar la fuente: el último frame pendiente, si lo hay."""
        if not self.cand:
            return []
        rgb, sharp, c, meta = self.cand[-1]
        self.cand = []
        return self._send(rgb, c, meta, sharp)

    def _send(self, rgb, motion, meta=None, sharp=None):
        out = []
        if meta is not None:
            meta.motion_px = float(motion)
            meta.sharpness = None if sharp is None else float(sharp)
        if self.synth and self.last_sent is not None and self.synth_factor * self.step < motion <= self.flow_max:
            n = min(self.max_synth, int(math.ceil(motion / self.synth_step)) - 1)
            ims = interpolate(self.last_sent, rgb, n, self.work)
            for k, im in enumerate(ims):
                m = (FrameMeta.between(self.last_meta, meta, (k + 1) / (n + 1))
                     if meta is not None and self.last_meta is not None else None)
                out.append((im, True, m))
            self.stats["synth"] += len(out)
        out.append((rgb, False, meta))
        self.last_sent = rgb
        self.last_meta = meta
        self.stats["sent"] += 1
        self.stats["skipped"] = self.stats["read"] - self.stats["sent"]
        if self._motions:
            self.stats["motion_px_median"] = round(float(np.median(self._motions)), 2)
        return out


class StaticHold:
    """Momentos estáticos: la misma regla del analizador, usada al revés.

    El analizador salta los frames que se movieron poco respecto del anterior enviado (amortigua el ruido
    de frames casi iguales). Aquí, con la misma medida -- mediana del flujo óptico DIS, acumulada desde el
    frame anterior del modelo (meta.motion_px cuando el analizador está activo; si no, se calcula igual
    entre frames consecutivos del modelo) --, un frame con movimiento menor que static_frac x step_px es un
    momento ESTÁTICO: la cámara no avanzó.

    En un momento estático la pose que se muestra, se registra y se publica NO avanza (se repite la
    anterior), aunque el modelo, que en streaming deriva aun con la cámara quieta, la mueva. El modelo y el
    mapeo siguen: los puntos del frame se agregan a la nube con la pose retenida. La deriva acumulada
    durante la pausa se guarda en una corrección `fix` (pose mostrada = fix · pose del modelo) que se
    mantiene al reanudar, así no hay salto al volver a moverse.
    """

    def __init__(self, step_px=36.0, static_frac=0.25, work=256):
        self.thr = float(step_px) * float(static_frac)
        self.work = work
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.prev = None
        self.fix = np.eye(4)
        self.last = None
        self.n_static = 0
        self.n = 0

    def _flow(self, rgb):
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        s = self.work / g.shape[1]
        g = cv2.resize(g, (self.work, round(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
        prev, self.prev = self.prev, g
        if prev is None or prev.shape != g.shape:
            return None
        flow = self.dis.calc(prev, g, None)
        return float(np.median(np.linalg.norm(flow, axis=-1))) / s

    def update(self, c2w_model, rgb=None, motion_px=None):
        """-> (pose a mostrar/registrar, estático?, movimiento en px). motion_px: el del analizador; si es
        None se mide aquí con rgb (el frame que entró al modelo)."""
        own = self._flow(rgb) if rgb is not None else None
        m = motion_px if motion_px is not None else own
        c2w_model = np.asarray(c2w_model, np.float64)
        static = self.last is not None and m is not None and m < self.thr
        if static:
            out = self.last.copy()
            self.fix = out @ np.linalg.inv(c2w_model)
            self.n_static += 1
        else:
            out = self.fix @ c2w_model
        self.last = out
        self.n += 1
        return out, bool(static), m


def interpolate(i0, i1, n, work=256):
    """n frames intermedios entre i0 e i1 (uint8 RGB) con flujo DIS bidireccional:
    F_t0 = -(1-t)t F01 + t^2 F10,  F_t1 = (1-t)^2 F01 - t(1-t) F10,
    I_t = (1-t) I0(x + F_t0) + t I1(x + F_t1)  (Jiang et al. 2018, sin red de refinamiento)."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    H, W = i0.shape[:2]
    s = work / W
    g0 = cv2.resize(cv2.cvtColor(i0, cv2.COLOR_RGB2GRAY), (work, round(H * s)))
    g1 = cv2.resize(cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY), (work, round(H * s)))
    f01 = cv2.resize(dis.calc(g0, g1, None), (W, H)) / s
    f10 = cv2.resize(dis.calc(g1, g0, None), (W, H)) / s
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = []
    for k in range(1, n + 1):
        t = k / (n + 1)
        ft0 = -(1 - t) * t * f01 + t * t * f10
        ft1 = (1 - t) ** 2 * f01 - t * (1 - t) * f10
        w0 = cv2.remap(i0, gx + ft0[..., 0], gy + ft0[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        w1 = cv2.remap(i1, gx + ft1[..., 0], gy + ft1[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out.append(((1 - t) * w0.astype(np.float32) + t * w1.astype(np.float32)).clip(0, 255).astype(np.uint8))
    return out


def crop518(rgb):
    """Mismo recorte que el modelo (modo 'crop'): 518 de ancho, centro de 518 de alto."""
    h, w = rgb.shape[:2]
    nh = round(h * 518 / w / 14) * 14
    r = cv2.resize(rgb, (518, nh), interpolation=cv2.INTER_AREA)
    if nh > 518:
        top = (nh - 518) // 2
        r = r[top:top + 518]
    return r


def main():
    """Fuera de línea: pasa todos los frames de un video por el analizador y escribe los
    elegidos (y los sintéticos) en out_dir/frames con un manifest.json compatible con
    process_and_view.py --manifest (los sintéticos quedan marcados y no entran al mapa)."""
    import argparse
    import json
    import os
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--frames_dir", required=True, help="todos los frames del video (p. ej. candidates_full)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--step_px", type=float, default=36.0)
    ap.add_argument("--max_skip", type=int, default=6)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--synth_strength", type=float, default=1.0,
                    help="intensidad del amortiguador (1 = umbral 1.5 pasos y un intermedio por paso; 2, 3 = más)")
    ap.add_argument("--source_fps", type=float, default=30.0, help="fps nominal de los frames (para los stamps del manifest)")
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    od = os.path.join(a.out_dir, "frames")
    os.makedirs(od, exist_ok=True)
    g = ContextGate(step_px=a.step_px, max_skip=a.max_skip, synth=a.synth, synth_strength=a.synth_strength)
    entries, k = [], 0
    for si, f in enumerate(files):
        full = cv2.cvtColor(cv2.imread(os.path.join(a.frames_dir, f)), cv2.COLOR_BGR2RGB)
        small = crop518(full)
        # el analizador puede elegir un frame anterior: el índice de origen viaja en el meta
        out = g.feed(small, FrameMeta(stamp=si / a.source_fps, frame_id=si))
        if si == len(files) - 1:
            out += g.flush()
        for img, syn, meta in out:
            name = f"{k:06d}.png"
            # se guarda el recorte de 518 que ve el modelo (el modelo lo recorta igual)
            cv2.imwrite(os.path.join(od, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            src = -1 if syn else meta.frame_id
            entries.append({"file": name, "kind": "synthetic" if syn else "real", "source_index": src,
                            "stamp": None if meta is None else round(meta.stamp, 4)})
            k += 1
    json.dump({"summary": g.stats, "frames": entries}, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=0)
    print(json.dumps(g.stats))


if __name__ == "__main__":
    main()
