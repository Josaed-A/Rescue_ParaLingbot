#!/usr/bin/env python3
"""Corrección histórica por keyframes de Stella sobre corridas grabadas (etapa 12).

Para una corrida de Stella aislado (stella_offline_test.sh, con ~/keyframes_full) o una sesión en
vivo con el puente: ancla cada pose de Stella al keyframe vigente en su momento y la corrige con la
última versión de los keyframes (keyframe_correction.correct_offline). Informa cuánto se movieron
las poses históricas y, si se da --ref, si la trayectoria corregida coincide mejor con la referencia
(LingBot windowed) y cierra mejor (|fin - inicio|).

  python3 src/mapas/kf_correction_report.py --run DIR --video V.mp4 --out OUT [--ref NPZ]
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
from keyframe_correction import correct_offline  # noqa: E402
from traj_align import apply_sim3, associate, path_length, umeyama  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="carpeta de stella_offline_test.sh")
    ap.add_argument("--video", required=True)
    ap.add_argument("--ref", default=None, help="npz windowed de LingBot (referencia de acuerdo)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clock = ct.VideoClock(a.video)
    st = ct.load("stella", "stella_run:" + a.run, clock, None, None)
    pub = [json.loads(x) for x in open(os.path.join(a.run, "published.jsonl"))]
    by_abs = {round(p["stamp"], 4): p["frame_id"] for p in pub}
    msgs = []
    for line in open(os.path.join(a.run, "stella.jsonl")):
        r = json.loads(line)
        if r.get("type") != "keyframes_full":
            continue
        rows = np.array(r["kf"], np.float64).reshape(-1, 9)
        # timestamp de imagen del keyframe (reloj ROS del publicador) -> PTS del video
        ts = []
        for x in rows[:, 1]:
            fid = by_abs.get(round(float(x), 4))
            ts.append(clock.pts[fid] if fid is not None else np.nan)
        rows[:, 1] = ts
        stamp_fid = by_abs.get(round(float(r["stamp"]), 4))
        msgs.append((clock.pts[stamp_fid] if stamp_fid is not None else np.nanmax(ts), rows[np.isfinite(rows[:, 1])]))
    corr, disp, summ = correct_offline(st["t"], st["T"], msgs)
    summ["mensajes_keyframes"] = len(msgs)
    ok = np.isfinite(corr[:, 0, 0])
    res = {"run": a.run, **summ}
    if ok.sum() > 2:
        P0, P1 = st["T"][ok, :3, 3], corr[ok, :3, 3]
        L = path_length(P0)
        res["desplazamiento_rel_largo_pct"] = {"mediana": round(100 * float(np.nanmedian(disp[ok])) / L, 3),
                                              "max": round(100 * float(np.nanmax(disp[ok])) / L, 3)}
        res["cierre_pct"] = {"original": round(100 * float(np.linalg.norm(P0[-1] - P0[0])) / L, 3),
                             "corregida": round(100 * float(np.linalg.norm(P1[-1] - P1[0])) / path_length(P1), 3)}
        if a.ref:
            ref = ct.load("ref", "windowed:" + a.ref, clock, None, a.cache or os.path.join(a.out, "cache"))
            for name, T in (("original", st["T"][ok]), ("corregida", corr[ok])):
                ia, ib, _ = associate(ref["t"], st["t"][ok], 0.05)
                if len(ia) > 10:
                    s, R, t = umeyama(T[ib, :3, 3], ref["T"][ia, :3, 3])
                    e = np.linalg.norm(apply_sim3(s, R, t, T[ib, :3, 3]) - ref["T"][ia, :3, 3], axis=1)
                    Lr = path_length(ref["T"][(ref["t"] >= ref["t"][ia].min()) & (ref["t"] <= ref["t"][ia].max()), :3, 3])
                    res.setdefault("ate_vs_windowed_pct", {})[name] = round(100 * float(np.sqrt((e ** 2).mean())) / Lr, 3)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(P0[:, 0], P0[:, 2], ".", ms=2, color="#ff8c1a", label="Stella (como la publicó)")
        ax[0].plot(P1[:, 0], P1[:, 2], ".", ms=2, color="#cc2222", label="corregida con los keyframes finales")
        ax[0].set_aspect("equal"); ax[0].legend(fontsize=8); ax[0].set_title("planta x-z (CV, mundo Stella)")
        ax[1].plot(st["t"][ok], 100 * disp[ok] / L, ".", ms=3)
        ax[1].set_xlabel("s de video"); ax[1].set_ylabel("desplazamiento de la corrección, % del largo")
        ax[1].set_title("cuánto movió la corrección cada pose histórica")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "correccion_keyframes.png"), dpi=95); plt.close(fig)
        np.savez(os.path.join(a.out, "stella_corregida.npz"), stamps=st["t"], pose_original=st["T"], pose_corregida=corr,
                 desplazamiento=disp)
    json.dump(res, open(os.path.join(a.out, "correccion_keyframes.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
