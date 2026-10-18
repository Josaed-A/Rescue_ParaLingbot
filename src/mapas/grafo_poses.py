"""Grafo geométrico de poses (nivel A): fragmentos de frames, restricciones medidas y optimización con Open3D.

Nodos: fragmentos de frames consecutivos que se solapan (frag_len frames cada frag_step). Cada nodo guarda la
corrección rígida que el grafo le asigna (la pose de su nube en el mundo común); la pose de cada frame no se
reemplaza: se le aplica la corrección interpolada entre fragmentos y las originales quedan guardadas.

Aristas:
  odometria  entre fragmentos que comparten frames (consecutivos). LingBot es consistente entre frames
             vecinos (±2.5 %, bitácora 2026-10-04), así que son restricciones confiables.
  cierre     entre fragmentos lejanos en el tiempo que vuelven a ver la misma zona (revisitas). Candidatos por
             cercanía de cámaras y solape; se exige verificación visual (ORB + RANSAC de la matriz
             fundamental entre keyframes) y geométrica (ICP punto-a-plano: ajuste, residuo, plausibilidad y
             correspondencias que no sean de un solo plano, que dejaría deslizar el registro, como en un
             pasillo de paredes repetidas). Entran como restricciones inciertas (line process de Open3D).

La optimización es la global de Open3D (Levenberg-Marquardt con poda de aristas); después se mide el residuo de
cada cierre contra la solución y los que no la respaldan quedan "podados" y se reoptimiza sin ellos. La
corrección se acepta solo si mejora la consistencia entre fragmentos y ningún fragmento se mueve más de un
límite: una transformación rígida no debe tapar un error de escala ni de forma de la profundidad.

Persistencia: <out>_grafo.json (nodos, aristas con su procedencia y estado, versión de poses de entrada y de
salida). Las nubes no se copian en el grafo: cada nodo referencia sus frames del .npz.

Se eligió el grafo de poses de Open3D y no GTSAM/iSAM2: la necesidad medida es reconstrucción fuera de línea
(cientos de fragmentos, se optimiza en segundos) y Open3D ya está instalado. El JSON tiene todo lo que haría
falta para un grafo de factores incremental (Pose3 + BetweenFactor con su información) si la optimización
pasara al servidor en vivo.
"""
import hashlib
import json

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

GRAPH_VERSION = "grafo-poses-1"


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def pose_hash(c2w):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(c2w, np.float32)).tobytes()).hexdigest()[:16]


def prior_information(scale, up=None, stiff=None):
    """Información (6x6, giro y después traslación, como Open3D) de una arista prior de LingBot. Con `up`, 4 grados
    de libertad: rígida (`stiff`) en la inclinación (giro sobre los ejes horizontales) y en la altura, `scale` en el
    rumbo y el desplazamiento horizontal. Las correcciones de los nodos son casi la identidad, así que el marco de
    la arista es el mundo de LingBot."""
    info = np.eye(6) * scale
    if up is not None:
        u = up / np.linalg.norm(up)
        h1 = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
        h1 /= np.linalg.norm(h1)
        B = np.stack([h1, np.cross(u, h1), u], 1)
        info[:3, :3] = B @ np.diag([stiff, stiff, scale]) @ B.T
        info[3:, 3:] = B @ np.diag([scale, scale, stiff]) @ B.T
    return info


def pcd(P):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, np.float64)))


def umeyama_scale(A, B):
    a, b = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(b.T @ a / len(A))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return float(np.trace(np.diag(S) @ D) / ((a ** 2).sum() / len(A)))


# --- identidad --------------------------------------------------------------------------------------------

def session_id(npz_path, extrinsic):
    """Identificador estable de una reconstrucción: carpeta de la prueba, nombre del .npz y huella de sus poses."""
    import os
    prueba = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(npz_path))))
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    return f"{prueba}/{stem}/{pose_hash(extrinsic)[:8]}"


def frame_ids_of(d, real):
    """Ids de frame que no dependen del orden de los archivos: source_index (el frame de frames/ o del video del
    que salió), que comparten la reconstrucción en vivo y su reproceso windowed; si no está, los del vivo."""
    for k in ("source_index", "frame_ids"):
        if k in d:
            return np.asarray(d[k])[real].astype(np.int64)
    return np.asarray(real, np.int64)


# --- keyframes --------------------------------------------------------------------------------------------

def sharpness(img):
    g = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def select_keyframes(c2w, images, med, trans_rel=0.15, rot_deg_=15.0, max_gap=10, blur_rel=0.3, lost=None):
    """Keyframe cuando la cámara se movió trans_rel x prof. mediana o giró rot_deg_ desde el último, cada max_gap
    frames como mínimo, o donde se perdió el seguimiento. Si el elegido está borroso (nitidez < blur_rel de la
    mediana) se toma el más nítido de los dos siguientes. Solo decide qué frames buscan cierres: ningún frame
    se descarta del mapa."""
    n = len(c2w)
    sharp = np.array([sharpness(images[i][::2, ::2]) for i in range(n)])
    med_s = float(np.median(sharp)) or 1.0
    kf = np.zeros(n, bool)
    last = 0
    kf[0] = True
    i = 1
    while i < n:
        moved = np.linalg.norm(c2w[i, :3, 3] - c2w[last, :3, 3]) > trans_rel * med
        turned = rot_deg(c2w[last, :3, :3].T @ c2w[i, :3, :3]) > rot_deg_
        lost_now = lost is not None and lost[i] and not lost[i - 1]
        if moved or turned or lost_now or i - last >= max_gap:
            j = i
            if sharp[i] < blur_rel * med_s:
                cand = list(range(i, min(n, i + 3)))
                j = max(cand, key=lambda c: sharp[c])
            kf[j] = True
            last = j
            i = j + 1
        else:
            i += 1
    return kf, sharp


# --- verificación visual ----------------------------------------------------------------------------------

_ORB = None


def orb_features(img):
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=1500)
    g = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
    return _ORB.detectAndCompute(g, None)


def visual_matches(fa, fb, ratio=0.75):
    """Correspondencias ORB que pasan el test de razón y el RANSAC de la matriz fundamental: (n, pa, pb)."""
    (ka, da), (kb, db) = fa, fb
    empty = (0, np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return empty
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < ratio * x[1].distance]
    if len(good) < 15:
        return empty
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5, 0.99)
    if mask is None:
        return empty
    mask = mask.ravel().astype(bool)
    return int(mask.sum()), pa[mask], pb[mask]


def kabsch(A, B):
    """Rotación y traslación que llevan A sobre B (mínimos cuadrados, sin escala)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - cb).T @ (A - ca))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cb - R @ ca
    return T


def ransac_rigid(A, B, thr, iters=500, seed=0):
    """Transformación rígida A -> B robusta (RANSAC de 3 puntos + Kabsch sobre los inliers)."""
    if len(A) < 6:
        return None, 0
    rng = np.random.default_rng(seed)
    best_in = np.zeros(len(A), bool)
    for _ in range(iters):
        idx = rng.choice(len(A), 3, replace=False)
        T = kabsch(A[idx], B[idx])
        inl = np.linalg.norm(A @ T[:3, :3].T + T[:3, 3] - B, axis=1) < thr
        if inl.sum() > best_in.sum():
            best_in = inl
    if best_in.sum() < 6:
        return None, int(best_in.sum())
    return kabsch(A[best_in], B[best_in]), int(best_in.sum())


# --- fragmentos y mediciones ------------------------------------------------------------------------------

def make_fragments(ids, frag_len, frag_step):
    starts = list(range(0, max(1, len(ids) - frag_len + 1), frag_step))
    if starts[-1] + frag_len < len(ids):
        starts.append(len(ids) - frag_len)
    return [ids[a:a + frag_len] for a in starts]


def fragment_cloud(P, sl, members, cam, voxel):
    pts = np.concatenate([P[sl[i][0]:sl[i][1]] for i in members if i in sl] or [np.zeros((0, 3), np.float32)])
    pc = pcd(pts).voxel_down_sample(voxel)
    if len(pc.points) >= 30:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
        pc.orient_normals_towards_camera_location(cam)
    return pc


def normal_spread(normals):
    """Qué tan restringido queda un registro por las normales de sus correspondencias: segundo autovalor
    sobre el primero de sum(n n^T). ~0 = un solo plano (el registro puede deslizar sobre él)."""
    if len(normals) < 10:
        return 0.0
    ev = np.linalg.eigvalsh(normals.T @ normals / len(normals))[::-1]
    return float(ev[1] / max(ev[0], 1e-12))


def point_to_plane(dist, tukey_rel):
    """ICP punto a plano; con tukey_rel > 0, robusto: núcleo de Tukey de ancho tukey_rel x la distancia de
    correspondencia, así los atípicos (un piso que resbala, un mueble que se movió) no tiran de la medición."""
    reg = o3d.pipelines.registration
    if tukey_rel > 0:
        return reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=tukey_rel * dist))
    return reg.TransformationEstimationPointToPlane()


def measure(src, tgt, voxel, dists, eval_mult, T0=None, tukey_rel=0.0):
    reg = o3d.pipelines.registration
    d_eval = eval_mult * voxel
    before = reg.evaluate_registration(src, tgt, d_eval, np.eye(4))
    T = np.eye(4) if T0 is None else T0
    for mult in dists:
        T = reg.registration_icp(src, tgt, mult * voxel, T, point_to_plane(mult * voxel, tukey_rel),
                                 reg.ICPConvergenceCriteria(max_iteration=60)).transformation
    after = reg.evaluate_registration(src, tgt, d_eval, T)
    corr = np.asarray(after.correspondence_set)
    pts = np.asarray(src.points)
    c = pts.mean(0)
    out = {"T": T, "fitness_antes": before.fitness, "rmse_antes": before.inlier_rmse, "fitness": after.fitness,
           "rmse": after.inlier_rmse, "correspondencias": int(len(corr)),
           "desplazamiento": float(np.linalg.norm(T[:3, :3] @ c + T[:3, 3] - c)), "giro_deg": rot_deg(T[:3, :3]),
           "escala": None, "dispersion_normales": 0.0}
    if len(corr) >= 50:
        A = pts[corr[:, 0]] @ T[:3, :3].T + T[:3, 3]
        out["escala"] = umeyama_scale(A, np.asarray(tgt.points)[corr[:, 1]])
        out["dispersion_normales"] = normal_spread(np.asarray(tgt.normals)[corr[:, 1]])
        out["informacion"] = reg.get_information_matrix_from_point_clouds(src, tgt, d_eval, T)
    return out


class PoseGraph:
    """Grafo de fragmentos persistible. Las poses de los frames viven fuera: el grafo solo corrige."""

    def __init__(self, session, med, voxel, params):
        self.session, self.med, self.voxel, self.p = session, med, voxel, params
        self.nodes, self.edges = [], []
        self.decision = None
        self.stats = {}

    # nodos ---------------------------------------------------------------------------------------------
    def add_fragments(self, frags, fids, pos, c2w_of, P, sl, keyframes):
        ordered = sorted({i for m in frags for i in m})
        path = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.array([c2w_of[i][:3, 3] for i in ordered]), axis=0),
                                                 axis=1))]
        along = dict(zip(ordered, path))
        for members in frags:
            cam = np.mean([c2w_of[i][:3, 3] for i in members], 0)
            fid = [int(fids[pos[i]]) for i in members]
            pc = fragment_cloud(P, sl, members, cam, self.voxel)
            kfs = [i for i in members if keyframes[pos[i]]] or [members[len(members) // 2]]
            self.nodes.append({
                "id": f"{self.session}/frag/{fid[0]}-{fid[-1]}", "frames": [int(i) for i in members],
                "frame_ids": fid, "keyframes": [int(i) for i in kfs], "camara_media": cam,
                "centro": float(np.mean([pos[i] for i in members])), "pose_inicial": np.eye(4),
                "recorrido": float(np.mean([along[i] for i in members])),
                "pose_optimizada": np.eye(4), "estado": "valido" if len(pc.points) >= 30 else "sin_puntos",
                "_pcd": pc})

    # aristas -------------------------------------------------------------------------------------------
    def _edge(self, s, t, kind, m, visual=None):
        p = self.p
        reasons = []
        if m["fitness"] < p.min_fitness:
            reasons.append(f"ajuste {m['fitness']:.2f} < {p.min_fitness}")
        if m["rmse"] > m["rmse_antes"] * 1.02 + 1e-12 and m["fitness_antes"] > 0:
            reasons.append("el residuo empeora")
        if kind == "cierre":
            # Un cierre puede pedir una corrección grande, pero proporcional a lo recorrido entre los dos
            # fragmentos: la deriva se acumula con el camino.
            dist = abs(self.nodes[t]["recorrido"] - self.nodes[s]["recorrido"])
            lim_t = max(p.max_loop_trans_rel * self.med, p.drift_rate * dist)
            lim_r = p.max_loop_rot_deg
        else:
            lim_t, lim_r = p.max_trans_rel * self.med, p.max_rot_deg
        if m["desplazamiento"] > lim_t or m["giro_deg"] > lim_r:
            reasons.append(f"corrección implausible ({m['desplazamiento']:.3f} > {lim_t:.3f} o "
                           f"{m['giro_deg']:.1f}° > {lim_r}°)")
        if m["dispersion_normales"] < p.min_normal_spread:
            reasons.append(f"correspondencias casi de un solo plano ({m['dispersion_normales']:.3f}): puede deslizar")
        if kind == "cierre" and (visual is None or visual < p.min_visual_inliers):
            reasons.append(f"sin verificación visual ({visual} inliers ORB < {p.min_visual_inliers})")
        if kind == "cierre" and m.get("dif_con_visual") is not None and m["dif_con_visual"] > lim_t / 2:
            reasons.append(f"ICP se aleja de la transformación de las correspondencias visuales "
                           f"({m['dif_con_visual']:.3f})")
        e = {"id": f"{self.nodes[s]['id']}->{self.nodes[t]['id']}", "tipo": kind, "s": s, "t": t,
             "metodo": "ICP punto-a-plano multiescala (Open3D) desde las poses actuales",
             "fitness": m["fitness"], "rmse": m["rmse"], "fitness_antes": m["fitness_antes"],
             "rmse_antes": m["rmse_antes"], "correspondencias": m["correspondencias"],
             "desplazamiento": m["desplazamiento"], "giro_deg": m["giro_deg"], "escala": m["escala"],
             "dispersion_normales": m["dispersion_normales"], "inliers_visuales": visual,
             "inliers_3d": m.get("inliers_3d"), "dif_con_visual": m.get("dif_con_visual"),
             "incierta": kind == "cierre", "estado": "aceptada" if not reasons else "rechazada",
             "motivo": "; ".join(reasons) or None, "T": m["T"], "informacion": m.get("informacion")}
        self.edges.append(e)
        return e

    def measure_odometry(self):
        near = max(1, int(np.ceil(self.p.frag_len / self.p.frag_step)))
        for s in range(len(self.nodes) - 1):
            a, b = self.nodes[s]["_pcd"], self.nodes[s + 1]["_pcd"]
            if len(a.points) < 30 or len(b.points) < 30 or not b.has_normals():
                continue
            self._edge(s, s + 1, "odometria", measure(a, b, self.voxel, self.p.icp_dist_mult, self.p.eval_dist_mult, tukey_rel=getattr(self.p, "icp_tukey_rel", 0.0)))
        return near

    def search_loops(self, images, near, lift_px=None):
        """Cierres: fragmentos no vecinos con cámaras cerca o nubes que se solapan. Primero la verificación visual
        (ORB + RANSAC de la fundamental entre keyframes); con las correspondencias visuales y la profundidad de
        cada keyframe (lift_px) sale una transformación 3D-3D robusta que inicializa el ICP: la deriva acumulada
        puede dejar las dos nubes fuera del alcance de un ICP que arranca de las poses actuales."""
        p = self.p
        cams = np.array([n["camara_media"] for n in self.nodes])
        pts = [np.asarray(n["_pcd"].points) for n in self.nodes]
        trees = [cKDTree(x) if len(x) else None for x in pts]
        feats = {}
        tested = visual_ok = 0
        for s in range(len(self.nodes)):
            cands = []
            if self.nodes[s]["estado"] != "valido":
                continue
            for t in range(s + near + 1, len(self.nodes)):
                # Solo fragmentos con nube y normales (ICP punto-a-plano las necesita en el destino).
                if trees[s] is None or trees[t] is None or self.nodes[t]["estado"] != "valido" \
                        or not self.nodes[t]["_pcd"].has_normals():
                    continue
                close = np.linalg.norm(cams[s] - cams[t]) < p.loop_radius_rel * self.med
                dist, _ = trees[t].query(pts[s][:: max(1, len(pts[s]) // 2000)], k=1,
                                         distance_upper_bound=p.loop_search_vox * self.voxel)
                ov = float(np.isfinite(dist).mean())
                if close or ov >= p.loop_overlap:
                    cands.append((ov, t))
            for ov, t in sorted(cands, reverse=True)[:p.loops_per_fragment]:
                tested += 1
                best = (0, None, None, None, None)
                for ka in self.nodes[s]["keyframes"]:
                    for kb in self.nodes[t]["keyframes"]:
                        for k in (ka, kb):
                            if k not in feats:
                                feats[k] = orb_features(images[k])
                        n_in, pa, pb = visual_matches(feats[ka], feats[kb])
                        if n_in > best[0]:
                            best = (n_in, ka, kb, pa, pb)
                n_vis, ka, kb, pa, pb = best
                if n_vis < p.min_visual_inliers:
                    self.edges.append({"id": f"{self.nodes[s]['id']}->{self.nodes[t]['id']}", "tipo": "cierre",
                                       "s": s, "t": t, "metodo": "verificación visual ORB", "estado": "rechazada",
                                       "motivo": f"sin verificación visual ({n_vis} inliers ORB < "
                                                 f"{p.min_visual_inliers})", "inliers_visuales": n_vis,
                                       "solape_inicial": ov, "incierta": True, "T": None, "informacion": None})
                    continue
                visual_ok += 1
                T0, n3d = None, 0
                if lift_px is not None:
                    A, okA, za = lift_px(ka, pa)
                    B, okB, zb = lift_px(kb, pb)
                    ok = okA & okB
                    # Umbral proporcional a la profundidad de las correspondencias (LingBot: ±2.5 % entre vistas).
                    thr = max(3 * self.voxel, 0.04 * float(np.median(np.r_[za[ok], zb[ok]]))) if ok.any() else 0
                    T0, n3d = ransac_rigid(A[ok], B[ok], thr, seed=p.seed)
                m = measure(self.nodes[s]["_pcd"], self.nodes[t]["_pcd"], self.voxel,
                            p.loop_icp_mult if T0 is None else p.icp_dist_mult, p.eval_dist_mult, T0,
                            tukey_rel=getattr(p, "icp_tukey_rel", 0.0))
                m["inliers_3d"] = n3d
                if T0 is not None:
                    c = np.asarray(self.nodes[s]["_pcd"].points).mean(0)
                    m["dif_con_visual"] = float(np.linalg.norm((m["T"][:3, :3] - T0[:3, :3]) @ c + m["T"][:3, 3] - T0[:3, 3]))
                e = self._edge(s, t, "cierre", m, n_vis)
                if T0 is not None:
                    # Lo que las correspondencias visuales dicen de la deriva entre las dos visitas, se acepte o no
                    # el cierre (un diagnóstico aunque falte la verificación geométrica para corregirla).
                    e["deriva_visual"] = {"desplazamiento": float(np.linalg.norm(T0[:3, :3] @ c + T0[:3, 3] - c)),
                                          "giro_deg": rot_deg(T0[:3, :3]), "inliers_3d": int(n3d),
                                          "recorrido_entre": abs(self.nodes[t]["recorrido"] - self.nodes[s]["recorrido"])}
                e["solape_inicial"] = ov
                e["keyframes"] = [int(ka), int(kb)]
        self.stats["candidatos_cierre"] = tested
        self.stats["cierres_con_verificacion_visual"] = visual_ok
        dv = [e["deriva_visual"] for e in self.edges if e.get("deriva_visual") and e["deriva_visual"]["inliers_3d"] >= 15]
        if dv:
            worst = max(dv, key=lambda x: x["giro_deg"] + x["desplazamiento"] / max(self.med, 1e-9))
            self.stats["deriva_estimada_por_cierres_visuales"] = {k: round(v, 4) if isinstance(v, float) else v
                                                                   for k, v in worst.items()}

    # optimización --------------------------------------------------------------------------------------
    def _optimize(self, edges):
        """Optimización global de Open3D. La distancia de correspondencia del line process se fija con los
        cierres verificados: con la distancia de una arista de odometría, un cierre que pide corregir una deriva
        acumulada (varias veces esa distancia) quedaría como atípico desde el arranque."""
        reg = o3d.pipelines.registration
        loops = [e["desplazamiento"] for e in edges if e["tipo"] == "cierre"]
        dist = max(self.p.eval_dist_mult * self.voxel, 1.5 * float(np.percentile(loops, 90)) if loops else 0.0)
        self.stats["distancia_line_process"] = dist
        pg = reg.PoseGraph()
        for _ in self.nodes:
            pg.nodes.append(reg.PoseGraphNode(np.eye(4)))
        for e in edges:
            pg.edges.append(reg.PoseGraphEdge(e["s"], e["t"], e["T"], e["informacion"], uncertain=e["incierta"]))
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(),
                                reg.GlobalOptimizationConvergenceCriteria(),
                                reg.GlobalOptimizationOption(
                                    max_correspondence_distance=dist,
                                    edge_prune_threshold=0.25, preference_loop_closure=self.p.loop_preference,
                                    reference_node=0))
        return [np.asarray(nd.pose) for nd in pg.nodes]

    def add_lingbot_priors(self):
        """Pose relativa original de LingBot entre fragmentos consecutivos como restricción débil (cierta).
        Mantiene el grafo conectado donde ICP no pudo medir la odometría (Open3D no optimiza un grafo no
        conectado: lo rechaza y deja las poses como estaban). Peso: una fracción de la odometría medida."""
        odo = [np.trace(e["informacion"]) for e in self.edges if e["tipo"] == "odometria" and e["estado"] == "aceptada"
               and e.get("informacion") is not None]
        scale = self.p.prior_weight * (float(np.median(odo)) / 6.0 if odo else 1e3)
        # dof4: la inclinación y la altura las fijan la gravedad y el piso, los cierres solo corrigen rumbo y
        # desplazamiento horizontal (en piso1 la corrección de 6 GDL giraba la inclinación hasta 6° y empeoraba el
        # piso entre visitas un 40 %)
        up = getattr(self, "up", None)
        info = prior_information(scale)
        if up is not None:
            info = prior_information(scale, up, self.p.dof4_rigidez * (float(np.median(odo)) / 6.0 if odo else 1e3))
            self.stats["grados_de_libertad"] = 4
        for s in range(len(self.nodes) - 1):
            # También entre nodos sin puntos: sin mediciones, solo los sostiene la pose de LingBot.
            self.edges.append({"id": f"{self.nodes[s]['id']}~{self.nodes[s + 1]['id']}", "tipo": "prior_lingbot",
                               "s": s, "t": s + 1, "metodo": "pose relativa original de LingBot (sin ICP)",
                               "estado": "aceptada", "motivo": None, "incierta": False, "T": np.eye(4),
                               "informacion": info, "desplazamiento": 0.0, "giro_deg": 0.0})

    def connected(self, edges):
        parent = list(range(len(self.nodes)))

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in edges:
            parent[root(e["s"])] = root(e["t"])
        return len({root(k) for k in range(len(self.nodes))})       # Open3D exige todos los nodos conectados

    def residual(self, e, G):
        """Cuánto se aparta la solución de la medición de una arista (desplazamiento del centro de la nube)."""
        R = np.linalg.inv(G[e["t"]]) @ G[e["s"]] @ np.linalg.inv(e["T"])
        c = np.asarray(self.nodes[e["s"]]["_pcd"].points).mean(0)
        return float(np.linalg.norm(R[:3, :3] @ c + R[:3, 3] - c)), rot_deg(R[:3, :3])

    def consistency(self, G):
        reg = o3d.pipelines.registration
        fit, rmse = [], []
        for e in self.edges:
            if e["estado"] == "rechazada" or e["tipo"] == "prior_lingbot":
                continue
            ev = reg.evaluate_registration(self.nodes[e["s"]]["_pcd"], self.nodes[e["t"]]["_pcd"],
                                           self.p.eval_dist_mult * self.voxel, np.linalg.inv(G[e["t"]]) @ G[e["s"]])
            fit.append(ev.fitness)
            rmse.append(ev.inlier_rmse)
        return {"fitness_mediana": float(np.median(fit)) if fit else None,
                "rmse_mediana": float(np.median(rmse)) if rmse else None, "aristas": len(fit)}

    def solve(self):
        p = self.p
        identity = [np.eye(4) for _ in self.nodes]
        acc = [e for e in self.edges if e["estado"] == "aceptada"]
        self.stats["antes"] = self.consistency(identity)
        if not acc:
            self.decision = "sin restricciones aceptadas: se conservan las poses"
            return identity
        disp = np.array([e["desplazamiento"] for e in acc])
        rots = np.array([e["giro_deg"] for e in acc])
        if np.percentile(disp, 90) < p.tol_trans_vox * self.voxel and rots.max() < p.tol_rot_deg:
            self.decision = "restricciones consistentes con las poses actuales: se conservan"
            return identity
        G = identity
        self.add_lingbot_priors()
        for it in range(3):                                  # optimizar, podar cierres sin respaldo, repetir
            acc = [e for e in self.edges if e["estado"] == "aceptada"]
            comps = self.connected(acc)
            self.stats["componentes_conexas"] = comps
            if comps > 1:
                self.decision = (f"grafo no conectado ({comps} componentes): no se optimiza, se conservan las poses")
                return identity
            G = self._optimize(acc)
            pruned = 0
            for e in acc:
                if e["tipo"] != "cierre":
                    continue
                r, rr = self.residual(e, G)
                e["residuo_solucion"] = r
                if r > p.loop_residual_vox * self.voxel or rr > 2 * p.tol_rot_deg:
                    e["estado"] = "podada"
                    e["motivo"] = f"no la respalda la solución (residuo {r:.4f}, {rr:.2f}°)"
                    pruned += 1
            self.stats[f"podadas_iter{it}"] = pruned
            if not pruned:
                break
        moves = []
        for k, n in enumerate(self.nodes):
            c = np.asarray(n["_pcd"].points).mean(0) if len(n["_pcd"].points) else np.zeros(3)
            moves.append((float(np.linalg.norm(G[k][:3, :3] @ c + G[k][:3, 3] - c)), rot_deg(G[k][:3, :3])))
        self.stats["correccion_max"] = max(x for x, _ in moves)
        self.stats["giro_max_deg"] = max(y for _, y in moves)
        self.stats["despues"] = self.consistency(G)
        b, a = self.stats["antes"], self.stats["despues"]
        # La corrección absoluta puede ser grande (deriva acumulada), pero no debe deformar la geometría local:
        # las aristas de odometría (vecinos que LingBot ve consistentes) tienen que seguir cumpliéndose.
        odo = [self.residual(e, G) for e in self.edges if e["tipo"] == "odometria" and e["estado"] == "aceptada"]
        self.stats["odometria_residuo_max"] = max((r for r, _ in odo), default=0.0)
        self.stats["odometria_giro_residuo_max_deg"] = max((rr for _, rr in odo), default=0.0)
        deformed = [r for r, rr in odo if r > p.odo_residual_vox * self.voxel or rr > p.odo_residual_deg]
        over = [self.nodes[k]["id"] for k, (x, y) in enumerate(moves)
                if x > max(p.max_trans_rel * self.med, p.drift_rate * self.nodes[k]["recorrido"])
                or y > p.max_loop_rot_deg]
        better = (a["rmse_mediana"] is not None and b["rmse_mediana"] is not None and
                  (a["rmse_mediana"] < b["rmse_mediana"] * 0.98 or a["fitness_mediana"] > b["fitness_mediana"] + 0.02))
        if deformed:
            self.decision = (f"corrección rechazada: deformaría la geometría local ({len(deformed)} aristas de "
                             "odometría dejarían de cumplirse); posible error de escala o de forma de la "
                             "profundidad, que una transformación rígida no debe tapar")
            return identity
        if over:
            self.decision = (f"corrección rechazada: {len(over)} fragmentos se moverían más que la deriva plausible "
                             f"({p.drift_rate:.0%} de lo recorrido)")
            self.stats["fragmentos_sobre_el_limite"] = over
            return identity
        if not better:
            self.decision = "corrección rechazada: no mejora la consistencia entre fragmentos"
            return identity
        for k, n in enumerate(self.nodes):
            n["pose_optimizada"] = G[k]
        self.decision = "poses corregidas por el grafo (originales conservadas)"
        return G

    def solve_multiway(self, min_overlap=0.3, prune=0.25, log=None):
        """Registro multivía como el tutorial de Open3D (pipelines/multiway_registration):
        - registro por pares con ICP punto-a-plano grueso (15 vóxeles) y fino (1.5 vóxeles), con su matriz de
          información;
        - odometría entre fragmentos consecutivos (cierta) y cierres entre los demás (inciertos);
        - optimización global de Levenberg-Marquardt con poda (edge_prune_threshold), con el fragmento 0 de
          referencia.
        Diferencias con el tutorial, porque tal cual rompe los mapas reales (en la sesión del desnivel del 15/10
        el ICP resbaló en piso y escaleras, la cadena de odometría acumuló 54° y las cámaras se movieron ~3
        unidades):
        - solo se registran los pares que comparten espacio a la escala del ICP grueso (todos contra todos, con
          cientos de fragmentos, son decenas de miles de ICP sin solapamiento);
        - un registro por pares que no ajusta, es implausible o puede deslizar sobre un plano no entra: la
          odometría usa entonces la pose relativa de LingBot con poco peso, y el cierre se descarta;
        - la solución se rechaza si se aparta de LingBot entre fragmentos vecinos más que odo_residual_*.
        Sin verificación visual de los cierres (la de solve()). Devuelve la corrección de cada fragmento."""
        reg = o3d.pipelines.registration
        n = len(self.nodes)
        pcds = [nd["_pcd"] for nd in self.nodes]
        valid = [len(pc.points) >= 30 and pc.has_normals() for pc in pcds]

        p = self.p

        def check(s, t, kind, T, r):
            """Por qué no se puede usar un registro por pares (vacío: se usa). El tutorial confía en todos; aquí
            se descartan los que no ajustan, los implausibles y los que pueden deslizar sobre un solo plano."""
            if r is None:
                return ["fragmento sin puntos o sin normales"]
            reasons = []
            c = np.asarray(pcds[s].points).mean(0)
            disp, rot = float(np.linalg.norm(T[:3, :3] @ c + T[:3, 3] - c)), rot_deg(T[:3, :3])
            if kind == "cierre":
                dist = abs(self.nodes[t]["recorrido"] - self.nodes[s]["recorrido"])
                lim_t, lim_r = max(p.max_loop_trans_rel * self.med, p.drift_rate * dist), p.max_loop_rot_deg
            else:
                lim_t, lim_r = p.max_trans_rel * self.med, p.max_rot_deg
            if r.fitness < p.min_fitness:
                reasons.append(f"ajuste {r.fitness:.2f} < {p.min_fitness}")
            if disp > lim_t or rot > lim_r:
                reasons.append(f"corrección implausible ({disp:.3f} > {lim_t:.3f} o {rot:.1f}° > {lim_r}°)")
            corr = np.asarray(r.correspondence_set)
            spread = normal_spread(np.asarray(pcds[t].normals)[corr[:, 1]]) if len(corr) >= 10 else 0.0
            if spread < p.min_normal_spread:
                reasons.append(f"correspondencias casi de un solo plano ({spread:.3f}): puede deslizar")
            return reasons

        def record(s, t, kind, T, info, r, reasons, overlap=None):
            c = np.asarray(pcds[s].points).mean(0) if len(pcds[s].points) else np.zeros(3)
            self.edges.append({
                "id": f"{self.nodes[s]['id']}->{self.nodes[t]['id']}", "tipo": kind, "s": s, "t": t,
                "metodo": "multiway (tutorial de Open3D): ICP punto-a-plano 15 -> 1.5 vóxeles",
                "fitness": None if r is None else r.fitness, "rmse": None if r is None else r.inlier_rmse,
                "solapamiento": overlap, "desplazamiento": float(np.linalg.norm(T[:3, :3] @ c + T[:3, 3] - c)),
                "giro_deg": rot_deg(T[:3, :3]), "incierta": kind == "cierre",
                "estado": "rechazada" if reasons else "aceptada",
                "motivo": "; ".join(reasons) + (" -> pose relativa de LingBot" if kind == "odometria" else "")
                if reasons else None, "T": T, "informacion": info})
            return len(self.edges) - 1

        reg_ = reg
        pg = reg_.PoseGraph()
        in_graph = []                                   # índice en self.edges de cada arista del grafo de Open3D
        odometry = np.identity(4)
        pg.nodes.append(reg_.PoseGraphNode(odometry))
        for s in range(n - 1):
            T, info, r = (pairwise_registration(pcds[s], pcds[s + 1], self.voxel, getattr(self.p, "icp_tukey_rel", 0.0))
                          if valid[s] and valid[s + 1]
                          else (np.identity(4), np.identity(6), None))
            why = check(s, s + 1, "odometria", T, r)
            k = record(s, s + 1, "odometria", T, info, r, why)
            if why:                  # la pose relativa de LingBot (identidad: las nubes ya están en su mundo)
                T = np.identity(4)
                info = (p.prior_weight * reg_.get_information_matrix_from_point_clouds(
                    pcds[s], pcds[s + 1], 1.5 * self.voxel, T) if r is not None else np.identity(6))
            odometry = T @ odometry
            pg.nodes.append(reg_.PoseGraphNode(np.linalg.inv(odometry)))
            pg.edges.append(reg_.PoseGraphEdge(s, s + 1, T, info, uncertain=False))
            in_graph.append(k)
        pairs = overlapping_pairs(pcds, 15 * self.voxel, min_overlap)
        for s, t, ov in pairs:
            if valid[s] and valid[t]:
                T, info, r = pairwise_registration(pcds[s], pcds[t], self.voxel, getattr(self.p, "icp_tukey_rel", 0.0))
                why = check(s, t, "cierre", T, r)
                k = record(s, t, "cierre", T, info, r, why, ov)
                if not why:
                    pg.edges.append(reg_.PoseGraphEdge(s, t, T, info, uncertain=True))
                    in_graph.append(k)
        loops = [e for e in self.edges if e["tipo"] == "cierre"]
        if log:
            log(f"multiway: {n} fragmentos, {len(pairs)} pares que se solapan, cierres plausibles "
                f"{sum(e['estado'] == 'aceptada' for e in loops)}, odometrías ICP descartadas "
                f"{sum(e['estado'] == 'rechazada' for e in self.edges if e['tipo'] == 'odometria')}", flush=True)
        identity = [np.eye(4) for _ in self.nodes]
        self.stats["antes"] = self.consistency(identity)
        option = reg_.GlobalOptimizationOption(max_correspondence_distance=1.5 * self.voxel,
                                               edge_prune_threshold=prune, reference_node=0)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            reg_.global_optimization(pg, reg_.GlobalOptimizationLevenbergMarquardt(),
                                     reg_.GlobalOptimizationConvergenceCriteria(), option)
        G = [np.asarray(nd.pose).copy() for nd in pg.nodes]
        for k, pe in zip(in_graph, pg.edges):
            e = self.edges[k]
            e["confianza"] = float(pe.confidence)
            if e["incierta"] and pe.confidence < prune:
                e["estado"] = "podada"
        # Cuánto se aparta la solución de LingBot entre fragmentos vecinos (comparten frames: LingBot los registra
        # bien entre sí). Un ICP que resbala en piso o escaleras encadena giros: el tutorial no lo detecta.
        dev = []
        for s in range(n - 1):
            R = np.linalg.inv(G[s + 1]) @ G[s]
            c = np.asarray(pcds[s].points).mean(0) if len(pcds[s].points) else np.zeros(3)
            dev.append((float(np.linalg.norm(R[:3, :3] @ c + R[:3, 3] - c)), rot_deg(R[:3, :3])))
        dev_t = max((a for a, _ in dev), default=0.0) / self.voxel
        dev_r = max((b for _, b in dev), default=0.0)
        self.stats["multiway"] = {
            "pares_solapados": len(pairs), "cierres": len(loops),
            "cierres_plausibles": sum(e["estado"] != "rechazada" for e in loops),
            "cierres_conservados": sum(e["estado"] == "aceptada" for e in loops),
            "odometrias_icp_descartadas": sum(e["estado"] == "rechazada" for e in self.edges if e["tipo"] == "odometria"),
            "desvio_de_lingbot_entre_vecinos": {"max_vox": dev_t, "max_deg": dev_r},
            "correccion_max_vox": float(max(np.linalg.norm(g[:3, 3]) for g in G)) / self.voxel,
            "giro_max_deg": float(max(rot_deg(g[:3, :3]) for g in G))}
        if dev_t > p.odo_residual_vox or dev_r > p.odo_residual_deg:
            self.stats["despues"] = self.stats["antes"]
            self.decision = (f"multiway rechazado: se aparta de LingBot entre fragmentos vecinos ({dev_t:.1f} vóxeles, "
                             f"{dev_r:.1f}°; límite {p.odo_residual_vox}, {p.odo_residual_deg}°): se conservan las poses")
            return identity
        for nd, g in zip(self.nodes, G):
            nd["pose_optimizada"] = g
        self.stats["despues"] = self.consistency(G)
        self.decision = (f"multiway (tutorial de Open3D, con guardas): {self.stats['multiway']['cierres_conservados']} "
                         f"de {len(loops)} cierres conservados; poses corregidas")
        return G

    # persistencia --------------------------------------------------------------------------------------
    def to_json(self, extra=None):
        def mat(x):
            return None if x is None else np.round(np.asarray(x, float), 9).tolist()

        nodes = [{"id": n["id"], "sesion": self.session, "frames": n["frames"], "frame_ids": n["frame_ids"],
                  "intervalo": [n["frame_ids"][0], n["frame_ids"][-1]], "keyframes": n["keyframes"],
                  "pose_inicial": mat(n["pose_inicial"]), "pose_optimizada": mat(n["pose_optimizada"]),
                  "marco": "mundo de LingBot-Map (c2w = inv(extrinsic)); la pose del nodo corrige su nube",
                  "unidades": "unidades del modelo (sin escala métrica)",
                  "nube": "frames del .npz de la sesión (no se copia)", "estado": n["estado"],
                  "incertidumbre": None} for n in self.nodes]
        edges = [{k: (mat(v) if k in ("T", "informacion") else (round(v, 6) if isinstance(v, float) else v))
                  for k, v in e.items()} for e in self.edges]
        for e in edges:
            e["origen"], e["destino"] = self.nodes[e.pop("s")]["id"], self.nodes[e.pop("t")]["id"]
        counts = {}
        for e in self.edges:
            counts[f"{e['tipo']}_{e['estado']}"] = counts.get(f"{e['tipo']}_{e['estado']}", 0) + 1
        out = {"version": GRAPH_VERSION, "sesion": self.session, "decision": self.decision,
               "parametros": {k: v for k, v in vars(self.p).items()} if not hasattr(self.p, "__dataclass_fields__")
               else __import__("dataclasses").asdict(self.p),
               "nodos": nodes, "aristas": edges, "conteo_aristas": counts, "estadisticas": self.stats}
        out.update(extra or {})
        return out

    def save(self, path, extra=None):
        with open(path, "w") as fh:
            json.dump(self.to_json(extra), fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def pairwise_registration(source, target, voxel, tukey_rel=0.0):
    """El registro por pares del tutorial multiway de Open3D: ICP punto-a-plano grueso y fino, y la matriz de
    información a la distancia fina."""
    reg = o3d.pipelines.registration
    coarse, fine = 15 * voxel, 1.5 * voxel
    icp_coarse = reg.registration_icp(source, target, coarse, np.identity(4), point_to_plane(coarse, tukey_rel))
    icp_fine = reg.registration_icp(source, target, fine, icp_coarse.transformation, point_to_plane(fine, tukey_rel))
    T = np.asarray(icp_fine.transformation).copy()
    return T, np.asarray(reg.get_information_matrix_from_point_clouds(source, target, fine, T)), icp_fine


def overlapping_pairs(pcds, cell, min_overlap):
    """Pares (s, t) no consecutivos cuyas nubes ocupan las mismas celdas de `cell` en al menos min_overlap de la
    más chica."""
    occ = []
    for pc in pcds:
        P = np.asarray(pc.points)
        ijk = np.floor(P / cell).astype(np.int64) + (1 << 20)
        occ.append(np.unique((ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]) if len(P) else np.zeros(0, np.int64))
    pairs = []
    for s in range(len(pcds)):
        for t in range(s + 2, len(pcds)):
            a, b = occ[s], occ[t]
            if len(a) and len(b):
                ov = len(np.intersect1d(a, b, assume_unique=True)) / min(len(a), len(b))
                if ov >= min_overlap:
                    pairs.append((s, t, float(ov)))
    return pairs


def per_frame_correction(nodes, G, ids):
    """Corrección de cada frame: interpolada entre los centros de los fragmentos (sin zigzag)."""
    centers = np.array([n["centro"] for n in nodes])
    if len(G) == 1:
        return {i: G[0] for i in ids}
    q = np.clip(np.arange(len(ids), dtype=np.float64), centers[0], centers[-1])
    rots = Slerp(centers, Rotation.from_matrix(np.array([g[:3, :3] for g in G])))(q).as_matrix()
    tr = np.stack([np.interp(q, centers, [g[k, 3] for g in G]) for k in range(3)], 1)
    out = {}
    for j, i in enumerate(ids):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = rots[j], tr[j]
        out[i] = T
    return out


def load_graph(path):
    with open(path) as fh:
        return json.load(fh)
