#!/usr/bin/env python3
"""Compara trayectorias de cámara de distintas fuentes sobre la misma secuencia (etapa 4).

Fuentes (cada una `nombre=tipo:ruta`):

  windowed:<npz>        predicciones fuera de línea de LingBot (process_and_view / windowed_lean):
                        c2w = inv(extrinsic); el instante de cada frame sale de emparejar por
                        contenido la imagen que vio el modelo (clave `images` del npz) con el frame
                        real del --video (las carpetas extraídas por ffmpeg están corridas 0.2-0.27 s
                        y re-muestreadas a tasa fija, y source_index no siempre indexa la misma).
                        Se guarda en caché (--cache).
  session:<npz>         pose BASIC de una sesión en vivo (pose_basic + stamps; etapas 1 y 3).
  session_stella:<npz>  trayectoria de Stella grabada en esa misma sesión (stella_traj_*).
  stella_run:<dir>      corrida de Stella aislado (src/ros/stella_offline_test.sh + stella_report.py).

Relojes: todo se lleva al timestamp de presentación (PTS) del --video. Las corridas y sesiones
grabadas antes de la etapa 4 usaban "índice / fps" (stamp_kind=synthetic), que en el video del
fablab (tasa variable) se desvía hasta 0.6 s: se convierten con el índice de frame -> PTS. Las
sesiones nuevas ya traen stamp_kind=video_pts y no se tocan.

Por cada par (a = referencia, b), siguiendo el orden del plan (tiempo -> ejes -> escala -> errores):
  - asociación por stamp (vecino más cercano a <= --tol s); cobertura de b sobre el tramo de a;
  - desfase temporal residual por correlación de la velocidad angular (debe dar ~0);
  - rotación de cámara C entre convenciones (mano-ojo; debe dar ~0°) y su condicionamiento;
  - Sim(3) de Umeyama b->a: escala, ATE (en % del largo del tramo de a); SE(3) para comparar;
  - error de orientación tras alinear el mundo; RPE a 1 s (traslación relativa y rotación);
  - escala por ventanas de 5 s (¿es estable? si no, una Sim(3) global no alcanza);
  - por segmento de b (Stella se reinicia y cambia de mundo tras un reset).
Escribe <out>/<a>__<b>.json y .png, y <out>/resumen.json.

  python3 src/mapas/compare_tracking.py --video V.mp4 --out OUT \\
      ref=windowed:P/eval/final_m2.npz basic=session:S/eval/sesion.npz stella=session_stella:S/eval/sesion.npz \\
      --pairs ref:basic,ref:stella,basic:stella
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src/vivo"))

from traj_align import (apply_sim3, apply_sim3_to_poses, associate, ate, estimate_time_offset,  # noqa: E402
                        path_length, rot_angle_deg, rotation_offsets, rpe, stats, umeyama, windowed_scale)
from pose_buffer import PoseBuffer, QueryKind  # noqa: E402

R_ROS_TO_CV = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# reloj del video
# ---------------------------------------------------------------------------
class VideoClock:
    def __init__(self, path):
        import cv2
        c = cv2.VideoCapture(path)
        self.fps = float(c.get(cv2.CAP_PROP_FPS))
        ts = []
        while c.grab():
            ts.append(c.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        self.pts = np.array(ts)
        self.path = path

    def from_index_fps(self, s, fps=None):
        """Stamp viejo (índice / fps) -> PTS real del mismo frame."""
        fps = fps or self.fps
        n = np.clip(np.round(np.asarray(s) * fps).astype(int), 0, len(self.pts) - 1)
        return self.pts[n]


def windowed_times(npz_path, images, clock, cache, rotate=90):
    """PTS de cada frame del npz, emparejando por contenido la imagen que vio el modelo (la clave
    `images` del npz: el recorte de 518 de ancho, centrado en alto) con los frames reales del
    video, recortados igual. Búsqueda monótona (los frames del npz están en orden). No depende de
    qué carpeta de frames usó la corrida: `source_index` indexa a veces candidates_full y a veces
    frames/ (cada 3), y ffmpeg extrajo esas carpetas corridas 0.2-0.27 s y a tasa fija."""
    import cv2
    key = os.path.join(cache, os.path.basename(npz_path).replace(".npz", "") + "_tiempos.npz")
    if os.path.isfile(key):
        c = np.load(key)
        if len(c["t"]) == len(images):
            return c["t"], c["err"]
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotate)
    S = 64

    def crop_like_model(f):
        h, w = f.shape[:2]
        nh = int(round(h * S / w))
        g = cv2.resize(f, (S, nh), interpolation=cv2.INTER_AREA)
        top = max(0, (nh - S) // 2)
        return g[top:top + S].astype(np.float32)

    cap = cv2.VideoCapture(clock.path)
    thumbs = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if rot is not None:
            f = cv2.rotate(f, rot)
        thumbs.append(crop_like_model(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    thumbs = np.stack(thumbs)
    t = np.zeros(len(images))
    err = np.zeros(len(images))
    n_prev = 0
    for i in range(len(images)):
        ref = crop_like_model(np.asarray(images[i]))
        lo, hi = (0, min(len(thumbs), 90)) if i == 0 else (n_prev, min(len(thumbs), n_prev + 45))
        d = np.abs(thumbs[lo:hi] - ref).mean(axis=(1, 2, 3))
        n = lo + int(np.argmin(d))
        t[i], err[i], n_prev = clock.pts[n], float(d.min()), n
    os.makedirs(cache, exist_ok=True)
    np.savez(key, t=t, err=err)
    return t, err


# ---------------------------------------------------------------------------
# cargadores
# ---------------------------------------------------------------------------
def _info_for(npz_path):
    for cand in (os.path.join(os.path.dirname(os.path.dirname(npz_path)), "info.json"),
                 os.path.join(os.path.dirname(npz_path), "info.json")):
        if os.path.isfile(cand):
            return json.load(open(cand))
    return {}


def _session_clock(d, npz_path, clock, stamps):
    kind = str(d["stamp_kind"]) if "stamp_kind" in d.files else "synthetic"
    if kind == "video_pts" or clock is None:
        return stamps, kind
    fps = (_info_for(npz_path).get("tracking") or {}).get("source_fps") or clock.fps
    return clock.from_index_fps(stamps, fps), f"{kind} -> PTS (índice/{fps})"


def load(name, spec, clock, frames_dir, cache):
    kind, path = spec.split(":", 1)
    out = {"name": name, "kind": kind, "path": path, "segments": None, "events": [], "extra": {}}
    if kind == "windowed":
        d = np.load(path)
        real = d["is_real"].astype(bool)
        E = np.tile(np.eye(4), (int(real.sum()), 1, 1))
        E[:, :3, :4] = d["extrinsic"][real]
        out["T"] = np.linalg.inv(E)
        if clock is None:
            raise SystemExit("windowed necesita --video")
        out["t"], err = windowed_times(path, d["images"][real], clock, cache)
        out["clock"] = f"emparejado por imagen (error medio {err.mean():.2f}, máx {err.max():.2f} de 255)"
    elif kind in ("session", "session_stella"):
        d = np.load(path)
        if kind == "session":
            T, st = d["pose_basic"].astype(np.float64), d["stamps"]
            for k in ("track_conf_basic", "track_motion_px", "track_sharpness", "stella_status"):
                if k in d.files:
                    out["extra"][k] = d[k]
            ps = d["pose_stella"] if "pose_stella" in d.files else None
            if ps is not None:
                out["extra"]["has_stella_exact"] = np.isfinite(ps[:, 0, 0])
        else:
            T, st = d["stella_traj_c2w"].astype(np.float64), d["stella_traj_stamps"]
            if "stella_traj_segment" in d.files and len(d["stella_traj_segment"]) == len(st):
                out["segments"] = d["stella_traj_segment"]
        out["t"], out["clock"] = _session_clock(d, path, clock, st)
        out["T"] = T
    elif kind == "poses":
        # npz con `stamps` (PTS) y una clave de poses c2w: `poses:<npz>:<clave>` (p. ej. pose_hybrid de
        # simulate_hybrid.py)
        npz, key = path.rsplit(":", 1)
        d = np.load(npz)
        out["t"], out["T"], out["clock"] = d["stamps"], d[key].astype(np.float64), "stamps del npz"
        out["path"] = npz
    elif kind == "stella_run":
        tr = np.load(os.path.join(path, "stella_traj.npz"))
        rel = tr["stamp_rel"]
        ok = np.isfinite(rel)
        out["T"] = tr["c2w_cv"][ok].astype(np.float64)
        rows = [json.loads(line) for line in open(os.path.join(path, "stella.jsonl"))]
        pub = [json.loads(line) for line in open(os.path.join(path, "published.jsonl"))]
        tp = np.array([p["t_pub"] for p in pub])
        sr = np.array([p["stamp_rel"] for p in pub])
        fid = np.array([p["frame_id"] for p in pub])
        # cada pose -> su frame (por el registro del publicador) -> PTS real de ese frame. Sirve con el
        # reloj viejo (índice / fps, etapas 2-3) y con el nuevo (PTS, desde la etapa 4) sin adivinar.
        by_rel = {round(float(a), 6): int(b) for a, b in zip(sr, fid)}
        if clock is not None:
            n_pose = np.array([by_rel.get(round(float(x), 6), -1) for x in rel[ok]])
            good = n_pose >= 0
            out["T"] = out["T"][good]
            out["t"] = clock.pts[np.clip(n_pose[good], 0, len(clock.pts) - 1)]
            out["clock"] = "frame_id del publicador -> PTS"
            ok_idx = np.flatnonzero(ok)[good]
            ok = np.zeros_like(ok)
            ok[ok_idx] = True
        else:
            out["t"] = rel[ok]
            out["clock"] = "stamp_rel"

        def video_t(t_recv):
            k = int(np.argmin(np.abs(tp - t_recv)))
            return float(clock.pts[min(int(fid[k]), len(clock.pts) - 1)]) if clock is not None else float(sr[k])

        seg, cur, seen_pose, segs = 0, 0, False, []
        for r in rows:
            if r["type"] == "state":
                out["events"].append((video_t(r["t_recv"]), r["state"]))
                if r["state"] == "Initializing" and seen_pose:
                    cur += 1
                    seen_pose = False
            elif r["type"] == "pose":
                seen_pose = True
                segs.append(cur)
            seg = cur
        segs = np.array(segs)
        out["segments"] = segs[ok] if len(segs) == len(ok) else None
    else:
        raise SystemExit(f"tipo desconocido: {kind}")
    out["T"] = np.asarray(out["T"], np.float64).reshape(-1, 4, 4)
    o = np.argsort(out["t"], kind="stable")
    out["t"], out["T"] = np.asarray(out["t"])[o], out["T"][o]
    if out["segments"] is not None:
        out["segments"] = out["segments"][o]
    for k in list(out["extra"]):
        out["extra"][k] = np.asarray(out["extra"][k])[o]
    return out


# ---------------------------------------------------------------------------
# comparación
# ---------------------------------------------------------------------------
def compare_segment(a, b, ia, Tb, tol):
    ta, Ta = a["t"][ia], a["T"][ia]
    Pa, Pb = Ta[:, :3, 3], Tb[:, :3, 3]
    # largo del recorrido de a en todo el tramo (no sólo entre muestras asociadas: con un hueco de
    # tracking de 60 s eso contaría una línea recta en vez del camino)
    m_full = (a["t"] >= ta.min()) & (a["t"] <= ta.max())
    L = path_length(a["T"][m_full, :3, 3])
    res = {"pares": int(len(ia)), "tramo_s": [round(float(ta.min()), 2), round(float(ta.max()), 2)],
           "largo_a": round(L, 4)}
    # 1. tiempo: desfase residual (sobre las trayectorias completas en el tramo)
    m_a = (a["t"] >= ta.min() - 1) & (a["t"] <= ta.max() + 1)
    m_b = (b["t"] >= ta.min() - 1) & (b["t"] <= ta.max() + 1)
    d, c, c0 = estimate_time_offset(a["t"][m_a], a["T"][m_a, :3, :3], b["t"][m_b], b["T"][m_b, :3, :3])
    # con muestras escasas (BASIC en vivo va a ~2 Hz) la velocidad angular no se puede estimar bien:
    # la correlación cae y el desfase no es fiable (se informa igual, marcado)
    res["desfase_temporal_s"] = None if d is None else {"d": round(d, 3), "corr": round(c, 3), "corr_en_0": round(c0, 3),
                                                        "fiable": bool(c >= 0.5)}
    # 2. ejes: rotación de cámara entre convenciones
    W, C, info = rotation_offsets(Ta[:, :3, :3], Tb[:, :3, :3])
    res["rot_camara_C_deg"] = round(float(rot_angle_deg(C)), 2)
    res["rot_camara_cond"] = None if info["cond"] is None else [round(v, 3) for v in info["cond"]]
    # 3. escala + mundo: Sim(3) y SE(3)
    s, R, t = umeyama(Pb, Pa, with_scale=True)
    e_sim = ate(Pa, apply_sim3(s, R, t, Pb))
    _, R1, t1 = umeyama(Pb, Pa, with_scale=False)
    e_se3 = ate(Pa, apply_sim3(1.0, R1, t1, Pb))
    res["sim3"] = {"escala_b_a": round(s, 5), "rot_mundo_deg": round(float(rot_angle_deg(R)), 2),
                   "ate": {k: round(v, 5) if isinstance(v, float) else v for k, v in stats(e_sim).items()},
                   "ate_rmse_pct_largo": round(100 * float(np.sqrt((e_sim ** 2).mean())) / max(L, 1e-9), 2)}
    res["se3_ate_rmse_pct_largo"] = round(100 * float(np.sqrt((e_se3 ** 2).mean())) / max(L, 1e-9), 2)
    # orientación tras alinear el mundo con la R de la Sim(3) (sin corregir C: si C≈I, igual)
    Tb_al = apply_sim3_to_poses(s, R, t, Tb)
    rot_err = rot_angle_deg(np.einsum("nji,njk->nik", Ta[:, :3, :3], Tb_al[:, :3, :3]))
    res["error_orientacion_deg"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in stats(rot_err).items()}
    # 4. RPE a 1 s (local, no depende de la deriva)
    r = rpe(ta, Ta, Tb, scale=s, delta=1.0, tol=0.2)
    if len(r["t"]):
        mv = r["disp_a"] > 0.05 * np.median(r["disp_a"])            # ignorar pares casi quietos
        res["rpe_1s"] = {"pares": int(mv.sum()),
                         "traslacion_rel_mediana": round(float(np.nanmedian(r["trans_rel"][mv])), 3),
                         "traslacion_rel_p90": round(float(np.nanpercentile(r["trans_rel"][mv], 90)), 3),
                         "rotacion_deg_mediana": round(float(np.median(r["rot_deg"])), 3),
                         "rotacion_deg_p90": round(float(np.percentile(r["rot_deg"], 90)), 3)}
    # 5. escala por ventanas
    cs, ss = windowed_scale(ta, Pa, Pb, win=5.0, step=1.0)
    if len(ss):
        res["escala_ventanas_5s"] = {"n": int(len(ss)), "mediana": round(float(np.median(ss)), 4),
                                     "cv": round(float(np.std(ss) / np.mean(ss)), 3),
                                     "min": round(float(ss.min()), 4), "max": round(float(ss.max()), 4)}
    plot = {"ta": ta, "Pa": Pa, "L": L, "Pb_al": apply_sim3(s, R, t, Pb), "err": e_sim, "cs": cs, "ss": ss,
            "Pb_all_al": apply_sim3(s, R, t, b["T"][:, :3, 3]), "tb_all": b["t"]}
    return res, plot


def compare(a, b, tol, min_pairs=20, assoc="nearest", max_gap=0.25):
    if len(b["t"]) == 0 or len(a["t"]) == 0:
        return {"a": a["name"], "b": b["name"], "muestras_a": int(len(a["t"])), "muestras_b": int(len(b["t"])),
                "asociados": 0, "cobertura_b_sobre_a": 0.0, "segmentos": [],
                "nota": "una de las trayectorias está vacía (p. ej. Stella nunca inicializó)"}, []
    if assoc == "interp":
        # etapa 5: consultar el buffer de b en los instantes de a (exacta, interpolada lineal+SLERP
        # entre dos muestras del mismo segmento a <= max_gap, o la más cercana a <= tol)
        buf = PoseBuffer.from_arrays(b["t"], b["T"], b["segments"], max_gap=max_gap, edge_tol=tol)
        Tq, kind, dtq, segq = buf.query_many(a["t"])
        ia = np.flatnonzero(kind > 0)
        Tb_all, segs_all, dt = Tq[ia], segq[ia], dtq[ia]
        kinds = {QueryKind(int(k)).name: int(v) for k, v in zip(*np.unique(kind[ia], return_counts=True))}
    else:
        ia, ib, dt = associate(a["t"], b["t"], tol)
        Tb_all = b["T"][ib]
        segs_all = b["segments"][ib] if b["segments"] is not None and len(ib) else np.zeros(len(ib), int)
        kinds = None
    span = (a["t"] >= b["t"].min()) & (a["t"] <= b["t"].max()) if len(b["t"]) else np.zeros(len(a["t"]), bool)
    res = {"a": a["name"], "b": b["name"], "reloj_a": a.get("clock"), "reloj_b": b.get("clock"),
           "muestras_a": int(len(a["t"])), "muestras_b": int(len(b["t"])), "tol_s": tol,
           "tramo_a_s": [round(float(a["t"].min()), 2), round(float(a["t"].max()), 2)],
           "tramo_b_s": None if not len(b["t"]) else [round(float(b["t"].min()), 2), round(float(b["t"].max()), 2)],
           "asociacion": assoc, "max_gap_s": max_gap if assoc == "interp" else None, "tipos": kinds,
           "asociados": int(len(ia)), "dt_asociacion_ms_mediana": round(1000 * float(np.median(np.abs(dt))), 1) if len(dt) else None,
           "cobertura_b_sobre_a": round(len(ia) / max(len(a["t"]), 1), 3),
           "cobertura_b_en_su_tramo": round(len(ia) / max(int(span.sum()), 1), 3)}
    if b["events"]:
        ev = b["events"]
        lost = []
        for k, (t0, st) in enumerate(ev):
            if st == "Lost":
                nxt = next(((t1, s1) for t1, s1 in ev[k + 1:] if s1 != "Lost"), None)
                lost.append({"desde_s": round(t0, 2), "hasta_s": None if nxt is None else round(nxt[0], 2),
                             "recupera_con": None if nxt is None else nxt[1]})
        res["perdidas_b"] = lost
        res["reinicios_b"] = sum(1 for k, (_, st) in enumerate(ev) if st == "Initializing" and k > 0)
    segs = segs_all
    res["segmentos"] = []
    plots = []
    for sg in np.unique(segs):
        m = segs == sg
        if m.sum() < min_pairs:
            res["segmentos"].append({"segmento": int(sg), "pares": int(m.sum()), "nota": "muy pocos pares"})
            continue
        r, p = compare_segment(a, b, ia[m], Tb_all[m], tol)
        r["segmento"] = int(sg)
        res["segmentos"].append(r)
        plots.append((sg, p))
    # señales BASIC frente al estado de Stella (si la sesión trae las dos)
    ex = a["extra"] if "stella_status" in a["extra"] else b["extra"]
    if "stella_status" in ex and "track_sharpness" in ex:
        st, sh, mo = ex["stella_status"], ex["track_sharpness"], ex["track_motion_px"]
        sig = {}
        for code, nm in ((1, "TRACKING"), (2, "LOST"), (3, "INITIALIZING")):
            m = st == code
            if m.sum():
                sig[nm] = {"frames": int(m.sum()), "nitidez_mediana": round(float(np.nanmedian(sh[m])), 1),
                           "movimiento_px_mediana": round(float(np.nanmedian(mo[m])), 2)}
        res["senales_basic_vs_estado_stella"] = sig
    return res, plots


def draw(res, plots, a, b, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    ax[0].plot(a["T"][:, 0, 3], a["T"][:, 2, 3], "-", color="0.6", lw=1.5, label=f"{a['name']} (completa)")
    colors = plt.cm.tab10.colors
    for k, (sg, p) in enumerate(plots):
        c = colors[(k + 1) % 10]
        ax[0].plot(p["Pb_al"][:, 0], p["Pb_al"][:, 2], ".", ms=3, color=c, label=f"{b['name']} seg {sg} (Sim3)")
        ax[1].plot(p["ta"], 100 * p["err"] / max(p["L"], 1e-9), ".", ms=3, color=c)
        if len(p["ss"]):
            ax[2].plot(p["cs"], p["ss"], "o-", ms=3, color=c)
    ax[0].plot(a["T"][0, 0, 3], a["T"][0, 2, 3], "go")
    ax[0].set_aspect("equal")
    ax[0].legend(fontsize=7)
    ax[0].set_title(f"planta x-z del mundo de {a['name']}")
    for (t0, st) in b["events"]:
        if st == "Lost":
            ax[1].axvline(t0, color="r", alpha=0.4)
    ax[1].set_xlabel("s de video (PTS)")
    ax[1].set_ylabel("ATE tras Sim(3), % del largo del tramo")
    ax[1].set_title("error de posición (rojo: Stella pasa a Lost)")
    ax[2].set_xlabel("s de video")
    ax[2].set_ylabel(f"escala {b['name']} -> {a['name']}")
    ax[2].set_title("escala Sim(3) en ventanas de 5 s")
    fig.suptitle(f"{a['name']} vs {b['name']}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="nombre=tipo:ruta")
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames_dir", default=None, help="(obsoleto: windowed se empareja con las imágenes del npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=None, help="a:b,a:c,... (por defecto, la primera contra cada una)")
    ap.add_argument("--tol", type=float, default=0.05, help="tolerancia de asociación (s)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--assoc", default="nearest", choices=["nearest", "interp"],
                    help="nearest: vecino más cercano a <= tol; interp: PoseBuffer de b consultado en los instantes de a")
    ap.add_argument("--max_gap", type=float, default=0.25, help="interp: hueco máximo entre muestras de b (s)")
    ap.add_argument("--span", default=None, help="t0,t1: comparar sólo ese tramo (s de video)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = a.cache or os.path.join(a.out, "cache")
    clock = VideoClock(a.video) if a.video else None
    src = {}
    for spec in a.sources:
        name, rest = spec.split("=", 1)
        src[name] = load(name, rest, clock, a.frames_dir, cache)
        tt = src[name]["t"]
        rng = f"{tt.min():.2f}-{tt.max():.2f} s" if len(tt) else "vacía"
        print(f"{name}: {len(tt)} poses, {rng}, reloj {src[name].get('clock')}", flush=True)
    if a.span:
        t0, t1 = (float(x) for x in a.span.split(","))
        for v in src.values():
            m = (v["t"] >= t0) & (v["t"] <= t1)
            v["t"], v["T"] = v["t"][m], v["T"][m]
            if v["segments"] is not None:
                v["segments"] = v["segments"][m]
            for k in list(v["extra"]):
                v["extra"][k] = v["extra"][k][m]
    names = list(src)
    pairs = [p.split(":") for p in a.pairs.split(",")] if a.pairs else [(names[0], n) for n in names[1:]]
    summary = []
    for pa, pb in pairs:
        res, plots = compare(src[pa], src[pb], a.tol, assoc=a.assoc, max_gap=a.max_gap)
        base = os.path.join(a.out, f"{pa}__{pb}")
        json.dump(res, open(base + ".json", "w"), indent=1, ensure_ascii=False)
        if plots:
            draw(res, plots, src[pa], src[pb], base + ".png")
        best = max((s for s in res["segmentos"] if "sim3" in s), key=lambda s: s["pares"], default=None)
        row = {"par": f"{pa} vs {pb}", "asociados": res["asociados"], "cobertura": res["cobertura_b_sobre_a"]}
        if best:
            row.update({"tramo_s": best["tramo_s"], "escala": best["sim3"]["escala_b_a"],
                        "ate_pct": best["sim3"]["ate_rmse_pct_largo"], "se3_ate_pct": best["se3_ate_rmse_pct_largo"],
                        "orient_med_deg": best["error_orientacion_deg"].get("mediana"),
                        "rpe1s_tras": (best.get("rpe_1s") or {}).get("traslacion_rel_mediana"),
                        "rpe1s_rot_deg": (best.get("rpe_1s") or {}).get("rotacion_deg_mediana"),
                        "C_deg": best["rot_camara_C_deg"], "desfase_s": (best["desfase_temporal_s"] or {}).get("d"),
                        "desfase_fiable": (best["desfase_temporal_s"] or {}).get("fiable"),
                        "escala_cv": (best.get("escala_ventanas_5s") or {}).get("cv")})
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    json.dump(summary, open(os.path.join(a.out, "resumen.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
