#!/usr/bin/env python3
"""Reproduce el selector BASIC / STELLA / HYBRID sobre datos grabados (etapa 6).

Recorre los frames de una sesión de LingBot en orden y, para cada uno, hace exactamente lo que haría
el servidor en vivo: consulta la pose de Stella de ese instante en un PoseBuffer (etapa 5) y el
estado de Stella en ese instante, y actualiza los tres ReferenceTracker. Guarda las tres
trayectorias para compararlas contra la referencia con compare_tracking.py (fuente `poses:`).

Stella puede venir de la misma sesión (`session_stella:<npz>`: lo que Stella dio en vivo, con su
estado grabado por frame) o de una corrida aislada sobre el mismo video (`stella_run:<dir>`: Stella
con CPU propia; el estado sale de sus eventos). Las dos usan el reloj PTS del video, así que
combinar la pose BASIC de una sesión con una Stella aislada es una simulación válida de "Stella con
CPU suficiente".

  python3 src/mapas/simulate_hybrid.py --video V.mp4 --basic S/eval/sesion.npz \\
      --stella stella_run:DIR --out OUT.npz
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vivo"))
sys.path.insert(0, HERE)

import compare_tracking as ct  # noqa: E402
from hybrid_tracking import HybridParams, ReferenceTracker, TrackingMode  # noqa: E402
from pose_buffer import PoseBuffer  # noqa: E402
from ros2_bridge import STELLA_EDGE_TOL, STELLA_MAX_GAP  # noqa: E402
from tracking import TrackingStatus  # noqa: E402

STATES = {"Initializing": TrackingStatus.INITIALIZING, "Tracking": TrackingStatus.TRACKING,
          "Lost": TrackingStatus.LOST}


def status_fn(stella, basic, basic_npz, same_session):
    """Estado de Stella en el instante t (lo que el nodo había publicado hasta entonces)."""
    if same_session:
        # estado grabado por frame en la sesión; se indexa con los tiempos YA convertidos a PTS
        # (basic["t"]), que conservan el orden de los frames del npz
        st = np.load(basic_npz)["stella_status"].astype(int)
        lut = {round(float(t), 6): TrackingStatus(int(s)) for t, s in zip(basic["t"], st)}
        return lambda t: lut.get(round(float(t), 6), TrackingStatus.UNKNOWN)
    ev = sorted(stella["events"])
    if not ev:
        return lambda t: TrackingStatus.UNKNOWN
    te = np.array([e[0] for e in ev])

    def f(t):
        k = int(np.searchsorted(te, t, side="right")) - 1
        return STATES.get(ev[k][1], TrackingStatus.UNKNOWN) if k >= 0 else TrackingStatus.UNKNOWN
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--basic", required=True, help="sesion.npz (pose_basic + stamps)")
    ap.add_argument("--stella", required=True, help="session_stella:<npz> o stella_run:<dir>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stella_scale", default="anchor", choices=["anchor", "window"],
                    help="escala del modo STELLA (etapa 9)")
    ap.add_argument("--no_rate_gate", dest="rate_gate", action="store_false",
                    help="no usar el ritmo de poses de Stella como criterio (para medir su efecto)")
    a = ap.parse_args()

    clock = ct.VideoClock(a.video)
    basic = ct.load("basic", "session:" + a.basic, clock, None, None)
    stella = ct.load("stella", a.stella, clock, None, None)
    same = a.stella.startswith("session_stella:") and os.path.abspath(a.stella.split(":", 1)[1]) == os.path.abspath(a.basic)
    status_at = status_fn(stella, basic, a.basic, same)
    buf = PoseBuffer.from_arrays(stella["t"], stella["T"], stella["segments"],
                                 max_gap=STELLA_MAX_GAP, edge_tol=STELLA_EDGE_TOL)
    trackers = {m: ReferenceTracker(m, HybridParams(stella_scale=a.stella_scale)) for m in TrackingMode}
    out = {m: [] for m in TrackingMode}
    used = {m: [] for m in TrackingMode}
    win = HybridParams().rate_window_s
    rates = []
    for t, Tb in zip(basic["t"], basic["T"]):
        q = buf.query(t)
        st = status_at(t)
        # ritmo de poses de Stella en la ventana que termina en t: lo que ya había llegado en vivo
        rate = buf.count_between(t - win, t) / win if a.rate_gate else None
        rates.append(np.nan if rate is None else rate)
        for m, tr in trackers.items():
            e = tr.update(t, Tb, q.c2w if q.ok else None, q.segment, st, stella_rate_hz=rate)
            out[m].append(e.c2w)
            used[m].append(1 if e.confidence["usado"] == "STELLA" else 0)
    np.savez(a.out, stamps=basic["t"], pose_basic=np.array(out[TrackingMode.BASIC]),
             pose_stella=np.array(out[TrackingMode.STELLA]), pose_hybrid=np.array(out[TrackingMode.HYBRID]),
             usado_stella_stella=np.array(used[TrackingMode.STELLA], np.uint8),
             usado_stella_hybrid=np.array(used[TrackingMode.HYBRID], np.uint8),
             ritmo_stella_hz=np.array(rates, np.float32))
    summ = {m.value: trackers[m].summary() for m in TrackingMode}
    summ["stella_fuente"] = a.stella
    summ["estado_de"] = "misma sesión" if same else "eventos de la corrida aislada"
    json.dump(summ, open(a.out.replace(".npz", "_resumen.json"), "w"), indent=1, ensure_ascii=False)
    for m in TrackingMode:
        s = summ[m.value]
        print(f"{m.value:7s} frames {s['frames']}  con Stella {s['fraccion_stella']:.0%}  motivos BASIC {s['motivos_basic']}")


if __name__ == "__main__":
    main()
