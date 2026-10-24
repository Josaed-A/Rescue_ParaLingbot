"""Ruta de la etapa 1 y su estabilización, sesión vieja (analizador + sintéticos) contra la nueva. Correr desde la raíz de
Rescue_ParaLingbot: python3 <este archivo> <carpeta de salida>."""
import json, sys
import numpy as np, cv2
R = "captures/pruebas_reales"
SES = {"vieja (analizador + sinteticos)": f"{R}/indoor/prueba_piso1_capas/cap/streaming/sin_guardar/sesion_20261017_155600",
       "nueva (cadencia uniforme + ICP robusto)": f"{R}/capas/piso1/streaming/sin_guardar/sesion_20261017_203401"}
def centers(ext):
    E = np.tile(np.eye(4), (len(ext), 1, 1)); E[:, :3, :4] = ext[:, :3, :4]
    return np.linalg.inv(E)[:, :3, 3]
out, plots = {}, []
for name, s in SES.items():
    d = np.load(f"{s}/eval/sesion.npz")
    c = centers(d["extrinsic"])
    st = np.linalg.norm(np.diff(c, axis=0), axis=1)
    med = np.median(st)
    k = int(np.argmax(np.linalg.norm(c - c[0], axis=1)))
    e = json.load(open(f"{s}/exports/estable/sesion_estable.json"))["grafo_poses"]
    out[name] = {"frames": len(c), "paso_mediano": round(float(med), 4), "saltos_5x": int((st > 5 * med).sum()),
                 "salto_max_sobre_mediana": round(float(st.max() / med), 1),
                 "recorrido": round(float(st.sum()), 2), "inicio_fin": round(float(np.linalg.norm(c[-1] - c[0])), 2),
                 # la vuelta sobre la ida (el giro: el punto más lejano del inicio): distancia mediana de cada punto de
                 # la vuelta al tramo de ida, relativa a la distancia del giro
                 "separacion_ida_vuelta_rel": round(float(np.median(np.min(np.linalg.norm(
                     c[k:, None] - c[None, :k], axis=2), axis=1)) / np.linalg.norm(c[k] - c[0])), 3),
                 "grafo": e["decision"].split(":")[0], "aristas": e.get("conteo_aristas")}
    # planta: el eje de menor varianza es la vertical
    w, v = np.linalg.eigh(np.cov((c - c.mean(0)).T)); P = (c - c.mean(0)) @ v[:, 1:]
    P = (P - P.min(0)) / max(np.ptp(P, 0).max(), 1e-9) * 440 + 30
    img = np.full((500, 500, 3), 255, np.uint8)
    for i in range(len(P) - 1):
        col = (0, 0, 220) if st[i] > 5 * med else (int(200 * i / len(P)), 120, int(200 * (1 - i / len(P))))
        cv2.line(img, tuple(P[i].astype(int)), tuple(P[i + 1].astype(int)), col, 3 if st[i] > 5 * med else 1, cv2.LINE_AA)
    cv2.putText(img, name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"saltos >5x el paso: {out[name]['saltos_5x']} (rojo)", (8, 492), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    plots.append(img)
cv2.imwrite(sys.argv[1] + "/rutas_vieja_nueva.png", np.hstack(plots))
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(sys.argv[1] + "/rutas_vieja_nueva.json", "w"), indent=1, ensure_ascii=False)
