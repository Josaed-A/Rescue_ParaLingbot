#!/usr/bin/env python3
"""Perfil de altura de un recorrido: ¿el mapa ve los cambios de nivel (escaleras, rampas)?

Toma un .npz de --save_predictions y mide cuánto sube o baja la cámara a lo largo del video.

La vertical se estima con el eje X de las cámaras: con el teléfono sin girar sobre su eje
(roll ~ 0), el eje "derecha" de la cámara siempre es horizontal, así que la vertical es la
dirección más perpendicular a todos ellos (autovector menor de sum x x^T). A diferencia de
promediar el eje Y (lo que hace compare_route.py), no la sesga la inclinación del teléfono,
que en una escalera es grande y sistemática.

La escala es arbitraria (profundidad monocular) y no se convierte a metros: anclarla a la
altura de la cámara sobre el piso varió 4x entre corridas del mismo video. Las medidas son
relativas: "end_over_max" = altura final / desnivel máximo (≈0 si el recorrido vuelve al
nivel de partida, ≈1 si se queda arriba) y "horizontal_over_max" = largo en planta / desnivel.

    python scripts_context/height_profile.py eval/mapa.npz --out_json h.json --out_png h.png
"""
import argparse
import json

import cv2
import numpy as np


def load(npz):
    d = np.load(npz)
    real = d["is_real"]
    E = np.tile(np.eye(4), (len(real), 1, 1))
    E[:, :3, :4] = d["extrinsic"].astype(np.float64)
    c2w = np.linalg.inv(E)                     # el visor también la invierte
    return d, real, c2w


def vertical(c2w):
    x = c2w[:, :3, 0]
    w, v = np.linalg.eigh(x.T @ x)
    up = v[:, 0]
    if up @ c2w[:, :3, 1].mean(0) > 0:         # cámara +Y apunta al piso
        up = -up
    yd = -c2w[:, :3, 1].mean(0)
    yd /= np.linalg.norm(yd)
    return up, float(np.degrees(np.arccos(np.clip(up @ yd, -1, 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz")
    ap.add_argument("--fps", type=float, default=10.0, help="cadencia de los frames del npz")
    ap.add_argument("--flat_s", type=float, default=6.0, help="segundos planos al inicio y al final")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_png", required=True)
    a = ap.parse_args()

    d, real, c2w = load(a.npz)
    c2w = c2w[real]
    up, ang = vertical(c2w)
    C = c2w[:, :3, 3]
    h = C @ up
    h = h - h[: max(1, int(a.flat_s * a.fps))].mean()
    t = np.arange(len(h)) / a.fps
    hs = cv2.GaussianBlur(h.reshape(-1, 1), (1, 0), 3).ravel() if len(h) > 9 else h

    horiz = C - np.outer(C @ up, up)
    step = np.linalg.norm(np.diff(horiz, axis=0), axis=1)
    horiz_len = float(step.sum())

    n = int(a.flat_s * a.fps)

    i_max, i_min = int(np.argmax(hs)), int(np.argmin(hs))
    res = {
        "frames": int(len(h)),
        "vertical_vs_mean_camY_deg": round(ang, 2),
        "height_start_end": round(float(hs[-n:].mean() - hs[:n].mean()), 4),
        "height_max": round(float(hs[i_max]), 4), "t_max_s": round(float(t[i_max]), 1),
        "height_min": round(float(hs[i_min]), 4), "t_min_s": round(float(t[i_min]), 1),
        "horizontal_length": round(horiz_len, 3),
        "end_over_max": round(float(hs[-n:].mean() / max(abs(hs.max()), abs(hs.min()))), 3),
        "horizontal_over_max": round(horiz_len / max(abs(hs.max()), abs(hs.min())), 2),
    }
    json.dump(res, open(a.out_json, "w"), indent=1)
    print(json.dumps(res, indent=1))

    # figura: perfil de altura + vista lateral coloreada por altura
    Wd, Ht = 1400, 520
    img = np.full((Ht, Wd, 3), 255, np.uint8)
    pad = 60
    def px(tt, hh, lo, hi, x0, x1):
        return (int(x0 + (tt - t[0]) / (t[-1] - t[0] + 1e-9) * (x1 - x0)),
                int(Ht - pad - (hh - lo) / (hi - lo + 1e-9) * (Ht - 2 * pad)))
    lo, hi = hs.min(), hs.max()
    for k in range(len(hs) - 1):
        cv2.line(img, px(t[k], hs[k], lo, hi, pad, Wd - pad), px(t[k + 1], hs[k + 1], lo, hi, pad, Wd - pad), (40, 90, 200), 2)
    for k in range(len(h)):
        cv2.circle(img, px(t[k], h[k], lo, hi, pad, Wd - pad), 1, (170, 170, 170), -1)
    cv2.line(img, (pad, Ht - pad), (Wd - pad, Ht - pad), (0, 0, 0), 1)
    cv2.line(img, (pad, pad), (pad, Ht - pad), (0, 0, 0), 1)
    cv2.putText(img, "altura de la camara (unidades del modelo) vs tiempo (s)", (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    for tt in range(0, int(t[-1]) + 1, 10):
        x, _ = px(tt, lo, lo, hi, pad, Wd - pad)
        cv2.putText(img, str(tt), (x - 8, Ht - pad + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for v in np.linspace(lo, hi, 5):
        _, y = px(t[0], v, lo, hi, pad, Wd - pad)
        cv2.line(img, (pad - 5, y), (Wd - pad, y), (225, 225, 225), 1)
        cv2.putText(img, f"{v:+.2f}", (2, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(a.out_png, img)


if __name__ == "__main__":
    main()
