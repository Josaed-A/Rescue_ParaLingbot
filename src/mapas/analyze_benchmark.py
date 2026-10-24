#!/usr/bin/env python3
"""Tablas y gráficas del benchmark de la integración Stella (etapas 17 y 18).

Lee las corridas de src/ros/run_benchmark.sh (cada una: sesion/, recursos.csv, replay.log,
stella_*.log) y produce en --out:

  tablas/   tracking.csv/.md, geometria.csv/.md, rendimiento.csv/.md, resumen.json
  graficas/ trayectorias_<seq>.png       planta de LingBot windowed (ref), BASIC, STELLA y HYBRID
            recursos_<corrida>.png       CPU modelo / Stella, RAM, VRAM y GPU en el tiempo
            barras_<metrica>.png         la métrica por secuencia y configuración
            escala_<seq>.png             escala Stella->BASIC por ventana (etapa 9)
  registro/<corrida>/<ref>_*           salida de register_map.py (mapa fusionado, npz registrado, métricas)

Métricas (ver docs/TRACKING_BENCHMARK.md): sin ground truth métrico. Referencias independientes de
LingBot: el croquis a mano (fablab, pasillos) y el regreso a la entrada (escaleras). LingBot windowed
como referencia de acuerdo (sesgada hacia BASIC, etapa 6).

  python3 src/mapas/analyze_benchmark.py --bench DIR --out DIR
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src/vivo"))
sys.path.insert(0, HERE)

U = os.path.join(REPO, "captures", "pruebas_reales", "unisabana")
SEQ = {
    "fablab": {"video": f"{U}/prueba_3/source/muestra_2_fablab.mp4", "ref": f"{U}/prueba_3/eval/final_m2.npz",
               "sketch": f"{U}/rutas_reales/ruta_real_muestra_2_fablab.jpeg"},
    "escaleras": {"video": f"{U}/prueba_4/source/Prueba_4_Desnivel.mp4", "ref": f"{U}/prueba_4/eval/final_m4.npz",
                  "sketch": None},
    "pasillos": {"video": f"{U}/prueba_2/source/muestra_unisabana.mp4", "ref": f"{U}/prueba_2/eval/final_m1.npz",
                 "sketch": f"{U}/rutas_reales/ruta_real_muestra_1_unisabana.jpeg"},
}
MODES = {"basic": "pose_basic", "stella": "pose_ref_stella", "hybrid": "pose_ref_hybrid"}
COLORS = {"ref": "0.55", "basic": "#555555", "stella": "#ff8c1a", "hybrid": "#22aa55", "stella_corr": "#cc2222"}


def seq_of(run):
    for s in SEQ:
        if run.endswith(s):
            return s
    return None


def md_table(rows, cols):
    out = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for r in rows:
        out += "| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |\n"
    return out


def write_table(path, rows, cols):
    with open(path + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    open(path + ".md", "w").write(md_table(rows, cols))


def resources(csvp):
    if not os.path.isfile(csvp):
        return {}
    import pandas as pd
    d = pd.read_csv(csvp)
    d = d[d["modelo_n"] > 0] if "modelo_n" in d else d
    if not len(d):
        return {}
    r = {"cpu_modelo_pct_med": round(float(d["modelo_cpu_pct"].median()), 1),
         "ram_modelo_mb_max": round(float(d["modelo_rss_mb"].max()), 0),
         "vram_mb_max": round(float(d["vram_mb"].max()), 0),
         "gpu_pct_med": round(float(d["gpu_pct"].median()), 1),
         "gpu_w_med": round(float(d["gpu_w"].median()), 1)}
    if "stella_cpu_pct" in d and (d["stella_n"] > 0).any():
        s = d[d["stella_n"] > 0]
        r["cpu_stella_pct_med"] = round(float(s["stella_cpu_pct"].median()), 1)
        r["ram_stella_mb_max"] = round(float(s["stella_rss_mb"].max()), 0)
    return r


def plot_resources(csvp, png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    if not os.path.isfile(csvp):
        return
    d = pd.read_csv(csvp)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(d["t"], d["modelo_cpu_pct"], label="modelo (CPU %)")
    if "stella_cpu_pct" in d:
        ax[0].plot(d["t"], d["stella_cpu_pct"], label="Stella (CPU %)")
    ax[0].set_ylabel("% de un núcleo"); ax[0].legend(fontsize=8)
    ax[1].plot(d["t"], d["modelo_rss_mb"] / 1024, label="modelo RAM")
    if "stella_rss_mb" in d:
        ax[1].plot(d["t"], d["stella_rss_mb"] / 1024, label="Stella RAM")
    ax[1].plot(d["t"], d["vram_mb"] / 1024, label="VRAM")
    ax[1].set_ylabel("GB"); ax[1].legend(fontsize=8)
    ax[2].plot(d["t"], d["gpu_pct"], label="GPU %")
    ax[2].plot(d["t"], d["gpu_w"], label="GPU W")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("s")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    plt.close(fig)


def stella_log_info(run_dir):
    logs = glob.glob(os.path.join(run_dir, "stella_*.log"))
    if not logs:
        return {}
    txt = open(logs[0], errors="replace").read()
    return {"inicializaciones": len(re.findall(r"initialization succeeded", txt)),
            "reinicios": len(re.findall(r"resetting system", txt)),
            "relocalizaciones": len(re.findall(r"relocalization succeeded", txt)),
            "perdidas_log": len(re.findall(r"tracking lost:", txt)),
            "loop_closures": len(re.findall(r"(?i)loop.*(detected|closed|correct)", txt))}


def register(sess_npz, ref, out, name, sketch, extra=()):
    js = os.path.join(out, f"{name}_metricas.json")
    if not os.path.isfile(js):
        cmd = [sys.executable, os.path.join(HERE, "register_map.py"), sess_npz, "--reference", ref, "--out", out,
               "--name", name, "--points_per_frame", "8000"] + (["--sketch", sketch] if sketch else []) + list(extra)
        subprocess.run(cmd, capture_output=True)
    return json.load(open(js)) if os.path.isfile(js) else None


def compare_ref(seq, sess_npz, out):
    """ATE / RPE contra LingBot windowed de los tres modos (compare_tracking.py, buffer)."""
    js = os.path.join(out, "resumen.json")
    if not os.path.isfile(js):
        srcs = [f"ref=windowed:{SEQ[seq]['ref']}"]
        d = np.load(sess_npz)
        for m, k in MODES.items():
            if k in d.files and np.isfinite(d[k][:, 0, 0]).any():
                srcs.append(f"{m}=poses:{sess_npz}:{k}")
        cache = os.path.join(REPO, "captures", "stella", "etapa4_2026-10-04", seq, "cache")
        subprocess.run([sys.executable, os.path.join(HERE, "compare_tracking.py"), "--video", SEQ[seq]["video"],
                        "--out", out, "--cache", cache, "--assoc", "interp"] + srcs, capture_output=True)
    return {r["par"].split(" vs ")[1]: r for r in json.load(open(js))} if os.path.isfile(js) else {}


def plot_traj(seq, runs, ref_npz, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from traj_align import umeyama
    import frames as F
    d = np.load(ref_npz)
    E = np.tile(np.eye(4), (len(d["extrinsic"]), 1, 1))
    E[:, :3, :4] = d["extrinsic"]
    Pr = np.linalg.inv(E)[:, :3, 3]
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5.5), squeeze=False)
    for ax, (run, npz) in zip(axes[0], runs):
        s = np.load(npz)
        Pb = s["pose_basic"][:, :3, 3].astype(np.float64)
        # todo al mundo y escala de BASIC; la ref (windowed) se alinea con Sim(3) a BASIC para dibujar
        n = min(len(Pr), len(Pb))
        idx = np.linspace(0, len(Pr) - 1, len(Pb)).astype(int)
        sc, R, t = umeyama(Pr[idx], Pb)
        Pr_al = (sc * (Pr @ R.T)) + t
        down = np.mean(s["pose_basic"][:, :3, 1], axis=0)
        down /= np.linalg.norm(down)
        e1 = np.array([1.0, 0, 0]) - down * down[0]
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(down, e1)
        top = lambda P: (P @ e1, P @ e2)
        ax.plot(*top(Pr_al), color=COLORS["ref"], lw=4, alpha=0.5, label="LingBot windowed (ref)")
        for m, k in MODES.items():
            if k in s.files and np.isfinite(s[k][:, 0, 0]).all() and not (m != "basic" and np.allclose(s[k], s["pose_basic"])):
                ax.plot(*top(s[k][:, :3, 3].astype(np.float64)), color=COLORS[m], lw=1.6, label=m.upper())
            elif m == "basic":
                ax.plot(*top(Pb), color=COLORS[m], lw=1.6, label="BASIC")
        if "pose_ref_stella_corr" in s.files and np.isfinite(s["pose_ref_stella_corr"][:, 0, 0]).all() \
                and not np.allclose(s["pose_ref_stella_corr"], s["pose_ref_stella"]):
            ax.plot(*top(s["pose_ref_stella_corr"][:, :3, 3].astype(np.float64)), "--", color=COLORS["stella_corr"],
                    lw=1.2, label="STELLA corregida (keyframes)")
        ax.plot(*top(Pb[:1]), "go", ms=9)
        ax.set_aspect("equal", adjustable="datalim")   # amplía los límites en vez de achicar el cuadro
        ax.set_title(run)
        ax.legend(fontsize=7)
    fig.suptitle(f"{seq}: planta (vertical media de la cámara), mundo y escala BASIC")
    fig.tight_layout()
    fig.savefig(png, dpi=95)
    plt.close(fig)


def plot_bars(rows, metric, png, title, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in rows if r.get(metric) is not None]
    if not rows:
        return
    labels = [f"{r['secuencia']}\n{r['config']}" for r in rows]
    vals = [float(r[metric]) for r in rows]
    cols = [COLORS.get(r.get("modo", "basic"), "#4477aa") for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(rows)), 4))
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=70, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(png, dpi=95)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for sub in ("tablas", "graficas", "registro", "comparacion"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    runs = sorted(d for d in os.listdir(a.bench) if os.path.isdir(os.path.join(a.bench, d, "sesion")))
    trk, geo, perf = [], [], []
    traj_runs = {}
    for run in runs:
        rd = os.path.join(a.bench, run)
        sess = os.path.join(rd, "sesion", "eval", "sesion.npz")
        info = json.load(open(os.path.join(rd, "sesion", "info.json")))
        tr = info.get("tracking") or {}
        st = (tr.get("stella") or {})
        sel = tr.get("selector") or {}
        seq = seq_of(run) or run.split("_", 1)[-1]
        cfg = run.split("_")[0]
        d = np.load(sess)
        # -- rendimiento
        res = resources(os.path.join(rd, "recursos.csv"))
        plot_resources(os.path.join(rd, "recursos.csv"), os.path.join(a.out, "graficas", f"recursos_{run}.png"), run)
        perf.append({"corrida": run, "secuencia": seq, "config": cfg, "frames": info.get("frames"),
                     "fps_modelo": info.get("fps_real"), "latencia_s_med": tr.get("latencia_s_mediana"),
                     **res})
        # -- tracking
        stl = stella_log_info(rd)
        sstat = d["stella_status"] if "stella_status" in d.files else np.zeros(len(d["stamps"]))
        row = {"corrida": run, "secuencia": seq, "config": cfg, "frames": int(len(d["stamps"])),
               "stella_poses": (st.get("stella") or {}).get("poses"),
               "stella_frames_procesados": None, "frac_frames_stella_tracking": round(float((sstat == 1).mean()), 3),
               "hybrid_usa_stella": (sel.get("hybrid") or {}).get("fraccion_stella"),
               "stella_usa_stella": (sel.get("stella") or {}).get("fraccion_stella"),
               **stl, "correcciones_kf_en_vivo": len(tr.get("correcciones_keyframes_en_vivo") or [])}
        states = (st.get("stella") or {}).get("states") or {}
        if states:
            row["stella_frames_procesados"] = sum(states.values())
        trk.append(row)
        # -- geometría y deriva por referencia
        if seq in SEQ:
            cmp = compare_ref(seq, sess, os.path.join(a.out, "comparacion", run))
            for m, k in MODES.items():
                if k not in d.files or not np.isfinite(d[k][:, 0, 0]).any():
                    continue
                if m != "basic" and np.allclose(d[k], d["pose_basic"], atol=1e-9) and cfg == "A":
                    continue
                met = register(sess, m, os.path.join(a.out, "registro", run), m, SEQ[seq]["sketch"])
                if not met:
                    continue
                c = cmp.get(m, {})
                mc = met["consistencia_multivista"]
                geo.append({"corrida": run, "secuencia": seq, "config": f"{cfg}:{m}", "modo": m,
                            "inlier_consecutivos": mc["consecutivos"]["inlier"], "inlier_1s": mc["a_1s"]["inlier"],
                            "relerr_1s": mc["a_1s"]["relerr"],
                            "voxeles": met["mapa"]["voxeles"], "puntos_crudos": met["mapa"]["puntos_crudos"],
                            "duplicacion": met["mapa"]["duplicacion_crudos_por_voxel"],
                            "voxeles_2+_frames": met["mapa"]["voxeles_vistos_por_2_o_mas_frames"],
                            "cierre_pct": met["deriva"]["cierre_fin_inicio_pct"],
                            "croquis_pct": (met.get("croquis") or {}).get("error_pct"),
                            "escala_vs_basic": (met.get("escala_vs_basic") or {}).get("sim3_global"),
                            "escala_cv_5s": (met.get("escala_vs_basic") or {}).get("ventanas_5s_cv"),
                            "ate_vs_windowed_pct": c.get("ate_pct"), "rpe1s_rot_deg": c.get("rpe1s_rot_deg"),
                            "jerk_p95": met["continuidad"]["jerk_rel_p95"]})
            if cfg in ("D", "S", "E"):
                traj_runs.setdefault(seq, []).append((run, sess))
    for seq, rs in traj_runs.items():
        plot_traj(seq, rs, SEQ[seq]["ref"], os.path.join(a.out, "graficas", f"trayectorias_{seq}.png"))
    T = os.path.join(a.out, "tablas")
    write_table(os.path.join(T, "tracking"), trk, ["corrida", "secuencia", "config", "frames", "stella_poses",
                "stella_frames_procesados", "frac_frames_stella_tracking", "hybrid_usa_stella", "stella_usa_stella",
                "inicializaciones", "reinicios", "perdidas_log", "relocalizaciones", "loop_closures", "correcciones_kf_en_vivo"])
    write_table(os.path.join(T, "geometria"), geo, ["corrida", "secuencia", "config", "inlier_consecutivos", "inlier_1s",
                "relerr_1s", "voxeles", "puntos_crudos", "duplicacion", "voxeles_2+_frames", "cierre_pct", "croquis_pct",
                "escala_vs_basic", "escala_cv_5s", "ate_vs_windowed_pct", "rpe1s_rot_deg", "jerk_p95"])
    write_table(os.path.join(T, "rendimiento"), perf, ["corrida", "secuencia", "config", "frames", "fps_modelo",
                "latencia_s_med", "cpu_modelo_pct_med", "cpu_stella_pct_med", "ram_modelo_mb_max", "ram_stella_mb_max",
                "vram_mb_max", "gpu_pct_med", "gpu_w_med"])
    json.dump({"tracking": trk, "geometria": geo, "rendimiento": perf}, open(os.path.join(T, "resumen.json"), "w"),
              indent=1, ensure_ascii=False)
    G = os.path.join(a.out, "graficas")
    plot_bars(geo, "inlier_1s", os.path.join(G, "barras_consistencia_1s.png"), "Coherencia multivista a 1 s (más es mejor)", "fracción a <5%")
    plot_bars(geo, "cierre_pct", os.path.join(G, "barras_cierre.png"), "|fin - inicio| / largo (escaleras: debería ser ~0)", "%")
    plot_bars([g for g in geo if g.get("croquis_pct") is not None], "croquis_pct", os.path.join(G, "barras_croquis.png"),
              "Error de forma contra el croquis (menos es mejor)", "% del largo")
    plot_bars(geo, "duplicacion", os.path.join(G, "barras_duplicacion.png"), "Puntos crudos por vóxel", "")
    plot_bars(geo, "ate_vs_windowed_pct", os.path.join(G, "barras_ate_windowed.png"), "ATE contra LingBot windowed (sesgada a BASIC)", "% del largo")
    plot_bars(perf, "fps_modelo", os.path.join(G, "barras_fps.png"), "frames/s del modelo", "frames/s")
    plot_bars(perf, "latencia_s_med", os.path.join(G, "barras_latencia.png"), "Latencia captura -> mapa (mediana)", "s")
    plot_bars(perf, "vram_mb_max", os.path.join(G, "barras_vram.png"), "VRAM máxima", "MB")
    plot_bars(perf, "cpu_stella_pct_med", os.path.join(G, "barras_cpu_stella.png"), "CPU de Stella (mediana)", "% de un núcleo")
    print(md_table(perf, ["corrida", "fps_modelo", "latencia_s_med", "cpu_modelo_pct_med", "cpu_stella_pct_med", "vram_mb_max"]))
    print(md_table(trk, ["corrida", "stella_poses", "frac_frames_stella_tracking", "hybrid_usa_stella", "relocalizaciones", "correcciones_kf_en_vivo"]))
    print(md_table(geo, ["corrida", "config", "inlier_1s", "duplicacion", "cierre_pct", "croquis_pct", "ate_vs_windowed_pct"]))


if __name__ == "__main__":
    main()
