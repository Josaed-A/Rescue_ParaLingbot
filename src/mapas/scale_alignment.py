#!/usr/bin/env python3
"""¿SE(3) o Sim(3) entre Stella y LingBot? Experimento de la etapa 9 (integración Stella-VSLAM).

Stella es monocular: su mapa tiene una escala arbitraria, fijada al inicializar y que deriva. LingBot
también es monocular (escala aprendida, en "unidades del modelo"). La pregunta del plan es cuál
transformación hay que estimar entre los dos mundos y con qué frecuencia. Para no esconder la escala
dentro de otra transformación se comparan, sobre los mismos pares asociados por tiempo, cinco
alineaciones de Stella (b) hacia la referencia (a):

  se3_global      SE(3) de Umeyama sin escala, con todos los pares (escala supuesta = 1)
  sim3_global     Sim(3) con todos los pares (una sola escala)
  sim3_inicio     Sim(3) estimada SÓLO con los primeros --init_s s y aplicada al resto
                  (lo que se puede hacer en vivo al arrancar)
  sim3_ventana    Sim(3) causal: en cada instante, la estimada con los --win s ANTERIORES (lo que
                  hace el selector en vivo si se re-estima por ventana); se evalúa sobre el
                  siguiente tramo, no sobre los datos con que se estimó
  sim3_oraculo    Sim(3) por ventana evaluada sobre sus propios datos (cota inferior, no causal)

Error: ATE (RMSE de posición) en % del largo del recorrido de la referencia en el tramo. Además, la
escala por ventana en el tiempo (gráfica) y su coeficiente de variación.

Pares usados (sin ground truth métrico; dos referencias distintas para no depender de una):
  ref = LingBot windowed (fuera de línea)     b = Stella aislada (corridas con keyframes, etapa 12)
  ref = BASIC en vivo (sesión del benchmark)   b = Stella en vivo de esa misma sesión

  python3 src/mapas/scale_alignment.py --out captures/stella/etapa9_2026-10-05
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src/vivo"))
sys.path.insert(0, HERE)

import compare_tracking as ct  # noqa: E402
from pose_buffer import PoseBuffer  # noqa: E402
from traj_align import apply_sim3, path_length, umeyama  # noqa: E402

U = os.path.join(REPO, "captures", "pruebas_reales", "unisabana")
SEQ = {"fablab": (f"{U}/prueba_3/source/muestra_2_fablab.mp4", f"{U}/prueba_3/eval/final_m2.npz"),
       "escaleras": (f"{U}/prueba_4/source/Prueba_4_Desnivel.mp4", f"{U}/prueba_4/eval/final_m4.npz"),
       "pasillos": (f"{U}/prueba_2/source/muestra_unisabana.mp4", f"{U}/prueba_2/eval/final_m1.npz")}
SOLO = os.path.join(REPO, "captures", "stella", "etapa12_2026-10-05", "stella_solo_kf")
BENCH = os.path.join(REPO, "captures", "stella", "benchmark_2026-10-05")
METHODS = ["se3_global", "sim3_global", "sim3_inicio", "sim3_ventana", "sim3_oraculo"]


def associate_interp(a, b, max_gap=0.25):
    """Posiciones de b interpoladas (PoseBuffer, sin extrapolar) en los instantes de a, por segmento
    de b. Devuelve t, Pa, Pb, seg (sólo los pares válidos)."""
    segs = b["segments"] if b["segments"] is not None else np.zeros(len(b["t"]), int)
    out = []
    for sg in np.unique(segs):
        m = segs == sg
        if m.sum() < 2:
            continue
        buf = PoseBuffer(max_gap=max_gap)
        for t, T in zip(b["t"][m], b["T"][m]):
            buf.add(float(t), T)
        for i, t in enumerate(a["t"]):
            q = buf.query(float(t))
            if q.ok:
                out.append((t, a["T"][i][:3, 3], q.c2w[:3, 3], sg))
    if not out:
        return None
    t, Pa, Pb, sg = (np.array(x) for x in zip(*out))
    o = np.argsort(t)
    return t[o], Pa[o], Pb[o], sg[o]


def rmse(x):
    return float(np.sqrt(np.mean(np.sum(x ** 2, axis=1)))) if len(x) else float("nan")


def evaluate(t, Pa, Pb, win, init_s, min_pts=8):
    L = path_length(Pa) + 1e-12
    res = {}
    s, R, tt = umeyama(Pb, Pa, with_scale=False)
    res["se3_global"] = rmse(Pa - apply_sim3(s, R, tt, Pb)) / L * 100
    s, R, tt = umeyama(Pb, Pa)
    res["sim3_global"] = rmse(Pa - apply_sim3(s, R, tt, Pb)) / L * 100
    res["escala_global"] = s
    m0 = t <= t[0] + init_s
    if m0.sum() >= min_pts and np.ptp(Pa[m0], axis=0).max() > 1e-6:
        s0, R0, t0 = umeyama(Pb[m0], Pa[m0])
        res["sim3_inicio"] = rmse(Pa[~m0] - apply_sim3(s0, R0, t0, Pb[~m0])) / L * 100 if (~m0).any() else None
    else:
        res["sim3_inicio"] = None
    # causal: la Sim(3) estimada con [t-win, t) se aplica a [t, t+step); oráculo: la misma ventana sobre sí
    err_c, err_o, scales = [], [], []
    step = win / 2
    for c in np.arange(t[0] + win, t[-1], step):
        past = (t >= c - win) & (t < c)
        nxt = (t >= c) & (t < c + step)
        if past.sum() < min_pts or not nxt.any() or np.ptp(Pa[past], axis=0).max() < 0.02 * np.ptp(Pa, axis=0).max():
            continue
        sp, Rp, tp = umeyama(Pb[past], Pa[past])
        err_c.append(Pa[nxt] - apply_sim3(sp, Rp, tp, Pb[nxt]))
        err_o.append(Pa[past] - apply_sim3(sp, Rp, tp, Pb[past]))
        scales.append((c - win / 2, sp))
    res["sim3_ventana"] = rmse(np.concatenate(err_c)) / L * 100 if err_c else None
    res["sim3_oraculo"] = rmse(np.concatenate(err_o)) / L * 100 if err_o else None
    sc = np.array([x[1] for x in scales])
    res["escala_cv_ventanas"] = float(sc.std() / sc.mean()) if len(sc) > 2 else None
    res["escala_rango"] = [float(sc.min()), float(sc.max())] if len(sc) else None
    res["pares"] = int(len(t))
    res["tramo_s"] = [round(float(t[0]), 2), round(float(t[-1]), 2)]
    return res, scales


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=float, default=5.0)
    ap.add_argument("--init_s", type=float, default=5.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = []
    for seq, (video, ref) in SEQ.items():
        clock = ct.VideoClock(video)
        cache = os.path.join(REPO, "captures", "stella", "etapa4_2026-10-04", seq, "cache")
        run = os.path.join(SOLO, seq)
        if os.path.isfile(os.path.join(run, "stella_traj.npz")):
            cases.append((seq, "windowed vs Stella aislada",
                          lambda r=ref, c=clock, k=cache: ct.load("ref", "windowed:" + r, c, None, k),
                          lambda r=run, c=clock: ct.load("stella", "stella_run:" + r, c, None, None)))
        for sess in sorted(glob.glob(os.path.join(BENCH, f"*_{seq}", "sesion", "eval", "sesion.npz"))):
            name = sess.split(os.sep)[-4]
            if name.startswith("A_"):
                continue
            cases.append((seq, f"BASIC vs Stella en vivo ({name})",
                          lambda s=sess, c=clock: ct.load("basic", "session:" + s, c, None, None),
                          lambda s=sess, c=clock: ct.load("stella", "session_stella:" + s, c, None, None)))

    rows, curves = [], []
    for seq, label, la, lb in cases:
        A, B = la(), lb()
        r = {"secuencia": seq, "par": label}
        asc = associate_interp(A, B) if len(B["t"]) >= 2 else None
        if asc is None or len(asc[0]) < 15:
            r["nota"] = f"sin pares suficientes ({0 if asc is None else len(asc[0])})"
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            continue
        t, Pa, Pb, sg = asc
        # el segmento más largo de Stella (tras un reinicio Stella cambia de mundo y de escala)
        main_seg = np.bincount(sg.astype(int) - sg.min()).argmax() + sg.min()
        m = sg == main_seg
        r["segmentos_stella"] = int(len(np.unique(sg)))
        res, scales = evaluate(t[m], Pa[m], Pb[m], a.win, a.init_s)
        r.update(res)
        rows.append(r)
        curves.append((seq, label, scales))
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()},
                         ensure_ascii=False), flush=True)

    json.dump(rows, open(os.path.join(a.out, "escala.json"), "w"), indent=1, ensure_ascii=False)
    cols = ["secuencia", "par", "pares", "tramo_s", "segmentos_stella", "escala_global", "escala_rango",
            "escala_cv_ventanas"] + METHODS + ["nota"]

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.3g}"
        if isinstance(v, list):
            return "–".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in v)
        return str(v)
    md = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for r in rows:
        md += "| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n"
    open(os.path.join(a.out, "escala.md"), "w").write(
        "ATE en % del largo de la referencia en el tramo (segmento más largo de Stella).\n\n" + md)
    print(md)

    # gráficas: escala por ventana en el tiempo y ATE por método
    seqs = list(SEQ)
    fig, ax = plt.subplots(1, len(seqs), figsize=(6 * len(seqs), 4), squeeze=False)
    for i, seq in enumerate(seqs):
        for s, label, sc in curves:
            if s == seq and sc:
                x, y = zip(*sc)
                y = np.array(y) / np.median(y)
                ax[0, i].plot(x, y, "o-", ms=3, label=label.replace("BASIC vs Stella en vivo ", "vivo "))
        ax[0, i].axhline(1, color="0.6", lw=0.8)
        ax[0, i].set_title(f"{seq}: escala Sim(3) por ventana de {a.win:g} s / mediana")
        ax[0, i].set_xlabel("s de video")
        ax[0, i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "escala_en_el_tiempo.png"), dpi=95)
    plt.close(fig)

    ok = [r for r in rows if "se3_global" in r]
    if ok:
        fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(ok)), 4.5))
        w = 0.16
        for j, mth in enumerate(METHODS):
            vals = [r.get(mth) if r.get(mth) is not None else np.nan for r in ok]
            ax.bar(np.arange(len(ok)) + (j - 2) * w, vals, w, label=mth)
        ax.set_xticks(range(len(ok)))
        ax.set_xticklabels([f"{r['secuencia']}\n{r['par'].replace('BASIC vs Stella en vivo ', 'vivo ')}" for r in ok],
                           fontsize=7, rotation=20)
        ax.set_ylabel("ATE, % del largo")
        ax.set_yscale("log")
        ax.set_title("Alineación Stella -> referencia: SE(3) vs Sim(3) (global, inicial, por ventana)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "ate_por_metodo.png"), dpi=95)
        plt.close(fig)


if __name__ == "__main__":
    main()
