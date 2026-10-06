#!/usr/bin/env python3
"""Informe de una corrida de Stella aislado (salida de stella_offline_test.sh) y trayectoria
en la convención de ParaLingbot (etapa 2; insumo de la comparación de la etapa 4).

Lee <dir>/stella.jsonl y <dir>/published.jsonl y calcula:
  - frames publicados vs procesados por Stella (los que no procesó los descartó la cola best-effort);
  - tiempo hasta inicializar, fracción de frames en Tracking, episodios Lost (inicio, duración,
    segundo de video en que ocurrieron, si se recuperó);
  - fps de poses, keyframes finales, correcciones de keyframes (candidatas a loop closure).

Convierte las poses del frame `map` de ROS (x adelante, y izquierda, z arriba) a la convención
CV/OpenCV con que Stella trabaja internamente y que comparte LingBot (x derecha, y abajo, z adelante):
el nodo publica T_ros = R · T_cv · R⁻¹ con R = [[0,0,1],[-1,0,0],[0,-1,0]], así que T_cv = R⁻¹ · T_ros · R
(docs/STELLA_INTEGRATION_AUDIT.md § 3.2). Guarda <dir>/stella_traj.npz con `stamps` (los del frame
publicado), `c2w_cv` (N,4,4), `c2w_ros` (N,4,4) y `stamp_rel` (segundo de video cuando la fuente es
carpeta/video), y <dir>/stella_report.json con las métricas. Con --png dibuja la planta (x-z de CV).

  python3 src/ros/stella_report.py /tmp/stella_e2/fablab_30fps --png
"""
import argparse
import json
import os

import numpy as np

R_ROS_TO_CV = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)   # rot_ros_to_cv_map_frame


def quat_to_mat(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.dir, "stella.jsonl"))]
    pub = [json.loads(l) for l in open(os.path.join(a.dir, "published.jsonl"))]
    summ = next((r for r in rows if r["type"] == "summary"), {})
    poses = [r for r in rows if r["type"] == "pose"]
    states = [r for r in rows if r["type"] == "state"]
    kfs = [r for r in rows if r["type"] == "keyframes"]

    # asociación stamp -> frame publicado (misma hora exacta: Stella reemite el header.stamp)
    by_stamp = {round(p["stamp"], 6): p for p in pub}
    n_proc = summ.get("estados", {}) and sum(summ["estados"].values())
    t_pub0 = pub[0]["t_pub"] if pub else None

    def video_s(t_recv):
        """segundo de video más cercano a una hora de recepción (por t_pub del publicador)."""
        if not pub:
            return None
        p = min(pub, key=lambda q: abs(q["t_pub"] - t_recv))
        return round(p["stamp_rel"], 2)

    # episodios Lost: desde el cambio a Lost hasta el siguiente Tracking
    episodes = []
    for i, s in enumerate(states):
        if s["state"] == "Lost":
            nxt = next((q for q in states[i + 1:] if q["state"] == "Tracking"), None)
            t_last = max(r["t_recv"] for r in rows if "t_recv" in r)
            end = nxt["t_recv"] if nxt else t_last
            episodes.append({"video_s": video_s(s["t_recv"]), "dur_s": round(end - s["t_recv"], 2),
                             "recuperado": nxt is not None})
    init = next((s for s in states if s["state"] == "Tracking"), None)
    corrections = [{"video_s": video_s(k["t_recv"]), "n": k["n"], "max_shift": k["max_shift"]}
                   for k in kfs if "poses" in k and k["max_shift"] > 0]

    # trayectoria
    stamps, rel, c2w_ros, c2w_cv = [], [], [], []
    Rinv = R_ROS_TO_CV.T
    for p in poses:
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(*p["q"])
        T[:3, 3] = p["p"]
        Tcv = np.eye(4)
        Tcv[:3, :3] = Rinv @ T[:3, :3] @ R_ROS_TO_CV
        Tcv[:3, 3] = Rinv @ T[:3, 3]
        stamps.append(p["stamp"])
        q = by_stamp.get(round(p["stamp"], 6))
        rel.append(q["stamp_rel"] if q else np.nan)
        c2w_ros.append(T)
        c2w_cv.append(Tcv)
    stamps, rel = np.array(stamps), np.array(rel)
    c2w_ros, c2w_cv = np.array(c2w_ros).reshape(-1, 4, 4), np.array(c2w_cv).reshape(-1, 4, 4)
    np.savez(os.path.join(a.dir, "stella_traj.npz"), stamps=stamps, stamp_rel=rel, c2w_cv=c2w_cv, c2w_ros=c2w_ros)

    path_len = float(np.linalg.norm(np.diff(c2w_cv[:, :3, 3], axis=0), axis=1).sum()) if len(c2w_cv) > 1 else 0.0
    rep = {
        "frames_publicados": len(pub),
        "frames_procesados": int(n_proc or 0),
        "fraccion_procesada": round((n_proc or 0) / max(len(pub), 1), 3),
        "video_s_publicado": round(pub[-1]["stamp_rel"] - pub[0]["stamp_rel"], 1) if pub else None,
        "t_inicializacion_s": None if init is None or t_pub0 is None else round(init["t_recv"] - t_pub0, 2),
        "poses": len(poses), "fps_poses": summ.get("fps_poses"),
        "fraccion_tracking": round(summ.get("estados", {}).get("Tracking", 0) / max(n_proc or 1, 1), 3),
        "estados": summ.get("estados"),
        "episodios_lost": episodes,
        "keyframes_finales": summ.get("keyframes_finales"),
        "correcciones_keyframes": corrections[:20], "n_correcciones": len(corrections),
        "largo_trayectoria_unidades_stella": round(path_len, 3),
        "pose_asociada_a_frame_publicado": int(np.isfinite(rel).sum()),
        "stamp_dt_median_s": summ.get("stamp_dt_median_s"), "stamp_gap_max_s": summ.get("stamp_gap_max_s"),
    }
    json.dump(rep, open(os.path.join(a.dir, "stella_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))

    if a.png and len(c2w_cv):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c = c2w_cv[:, :3, 3]
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        sc = ax[0].scatter(c[:, 0], c[:, 2], c=rel, s=6, cmap="viridis")
        ax[0].plot(c[0, 0], c[0, 2], "go"); ax[0].plot(c[-1, 0], c[-1, 2], "ro")
        ax[0].set_aspect("equal"); ax[0].set_title("planta (x-z de CV), color = segundo de video")
        plt.colorbar(sc, ax=ax[0])
        ax[1].plot(rel, -c[:, 1], ".", ms=2)
        ax[1].set_xlabel("segundo de video"); ax[1].set_ylabel("altura (-y de CV)"); ax[1].set_title("altura")
        for e in episodes:
            if e["video_s"] is not None:
                ax[1].axvspan(e["video_s"], e["video_s"] + e["dur_s"], color="red", alpha=0.15)
        fig.tight_layout(); fig.savefig(os.path.join(a.dir, "stella_traj.png"), dpi=110)
        print("figura:", os.path.join(a.dir, "stella_traj.png"))


if __name__ == "__main__":
    main()
