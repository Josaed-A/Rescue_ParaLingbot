"""Grafo estructural (nivel B): planos dominantes de la nube estabilizada como entidades y sus relaciones.

No modifica la geometría: describe. Las entidades salen de un RANSAC iterativo sobre los vóxeles (solo
diagnóstico, no reemplaza a las observaciones) y se clasifican por orientación y por el grupo semántico
mayoritario del filtro geométrico, si lo hay. Las relaciones entre superficies paralelas usan distancia,
orientación de las caras (hacia las cámaras que las vieron), solape en planta, evidencia temporal (de qué
momento del recorrido vienen sus puntos) y semántica; ninguna se fusiona:

  POSIBLE_DUPLICADO_DE  misma cara, solapadas, vistas en momentos distintos: la misma superficie dos veces
  PARALELA_A            caras opuestas que se miran: dos superficies físicas (un pasillo)
  DOS_CARAS_DE          caras opuestas de espaldas: las dos caras de una pared
  AMBIGUA_CON           misma cara vistas a la vez (escalón real, ruido o un objeto delante): sin evidencia
  CERCA_DE              grupos semánticos distintos y confiables (objetos distintos próximos)
  ADYACENTE_A           planos no paralelos que se tocan (esquinas, pared con piso)
  SOPORTA               un piso bajo una pared que llega hasta él
  OBSERVADO_DESDE       fragmentos del grafo de poses que aportan sus puntos

Identificadores: sesión, versión de poses y una huella de la geometría cuantizada (no el orden del RANSAC).
"""
import hashlib

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

GROUP_NAMES = ["otro", "pared", "puerta", "ventana", "piso", "escalera", "techo", "columna"]
KIND_BY_GROUP = {"pared": "pared", "puerta": "puerta", "ventana": "pared", "piso": "piso",
                 "escalera": "escalera", "techo": "techo", "columna": "columna"}
RELATION_OF_CLASS = {"duplicada": "POSIBLE_DUPLICADO_DE", "paralelas_fisicas": "PARALELA_A",
                     "dos_caras": "DOS_CARAS_DE", "ambigua": "AMBIGUA_CON", "objetos_distintos": "CERCA_DE"}


def pcd(P):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, np.float64)))


def largest_component(X, n, cell):
    """Índices de la componente conexa más grande (8-vecinos) de los puntos X proyectados al plano de normal n."""
    from scipy import ndimage

    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    g = np.floor(np.c_[X @ u, X @ w] / cell).astype(np.int64)
    g -= g.min(0)
    if g.max() > 20000:                       # plano gigante y ralo: no armar una grilla enorme
        return np.arange(len(X))
    grid = np.zeros(g.max(0) + 1, bool)
    grid[g[:, 0], g[:, 1]] = True
    lab, k = ndimage.label(grid, structure=np.ones((3, 3)))
    if k <= 1:
        return np.arange(len(X))
    pl = lab[g[:, 0], g[:, 1]]
    return np.flatnonzero(pl == np.bincount(pl)[1:].argmax() + 1)


def dominant_planes(V, cams, up, p, sub=None):
    """Planos dominantes por RANSAC iterativo con la normal orientada hacia las cámaras que vieron cada plano."""
    o3d.utility.random.seed(p.seed)
    idx_all = np.arange(len(V["centers"])) if sub is None else np.sort(sub)
    rest = idx_all.copy()
    planes = []
    dist = p.plane_dist_vox * V["voxel"]
    min_pts = max(50, int(p.plane_min_frac * len(idx_all)))
    tries = 0
    while len(planes) < p.planes_max and len(rest) > min_pts and tries < 4 * p.planes_max:
        tries += 1
        model, inl = pcd(V["centers"][rest]).segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=1000)
        inl = np.asarray(inl, np.int64)
        if len(inl) < min_pts:
            break
        n = np.array(model[:3], float)
        n /= np.linalg.norm(n)
        # Una entidad es una superficie continua: de los inliers del plano (que pueden juntar trozos de
        # superficies distintas que cruzan esa altura) queda la componente conexa más grande en el plano.
        comp = largest_component(V["centers"][rest[inl]], n, 3 * V["voxel"])
        if len(comp) < min_pts:
            rest = np.delete(rest, inl[comp])
            continue
        inl = inl[comp]
        members = rest[inl]
        c = V["centers"][members].mean(0)
        side = np.einsum("j,ij->i", n, cams[V["fmin"][members]] - V["centers"][members])
        if np.median(side) < 0:
            n = -n
        tilt = float(np.degrees(np.arccos(min(1.0, abs(n @ up)))))
        g = gf = None
        if "group" in V:
            bc = np.bincount(V["group"][members], minlength=len(GROUP_NAMES))
            g, gf = int(bc.argmax()), float(bc.max() / len(members))
        res = np.abs((V["centers"][members] - c) @ n)
        planes.append({"n": n, "c": c, "members": members, "voxeles": int(len(members)),
                       "orientacion": "horizontal" if tilt < 20 else ("vertical" if tilt > 70 else "oblicua"),
                       "group": g, "group_frac": gf, "rms": float(np.sqrt((res ** 2).mean()))})
        rest = np.delete(rest, inl)
    return planes


def plane_frames(V, F, plane, n_frames_total, bins=20):
    """Histograma normalizado de los frames que aportan puntos a los vóxeles del plano."""
    inplane = np.zeros(len(V["centers"]), bool)
    inplane[plane["members"]] = True
    fr = F[inplane[V["inv"]]]
    h = np.bincount(np.minimum(fr * bins // max(1, n_frames_total), bins - 1), minlength=bins).astype(float)
    return h / max(1.0, h.sum()), float(np.median(fr)) if len(fr) else float("nan"), fr


def classify_pairs(V, F, planes, med, p, n_frames_total, frames=None):
    """Pares de superficies paralelas cercanas, clasificados (duplicada, paralelas_fisicas, dos_caras,
    objetos_distintos, ambigua). Ver el docstring del módulo."""
    out = []
    cell = 4 * V["voxel"]
    frames = frames or [plane_frames(V, F, pl, n_frames_total) for pl in planes]
    for a in range(len(planes)):
        for b in range(a + 1, len(planes)):
            A, B = planes[a], planes[b]
            cosang = float(A["n"] @ B["n"])
            if abs(cosang) < np.cos(np.radians(p.dup_angle_deg)):
                continue
            sep = float(abs((B["c"] - A["c"]) @ A["n"]))
            if sep <= 2 * p.plane_dist_vox * V["voxel"] or sep > p.dup_max_rel * med:
                continue
            u = np.cross(A["n"], [1.0, 0, 0] if abs(A["n"][0]) < 0.9 else [0, 1.0, 0])
            u /= np.linalg.norm(u)
            w = np.cross(A["n"], u)
            occ = []
            for pl in (A, B):
                q = V["centers"][pl["members"]]
                g = np.floor(np.c_[q @ u, q @ w] / cell).astype(np.int64)
                occ.append(np.unique((g[:, 0] << 32) + g[:, 1]))
            overlap = len(np.intersect1d(occ[0], occ[1], assume_unique=True)) / max(1, min(len(occ[0]), len(occ[1])))
            if overlap < p.dup_min_overlap:
                continue
            (ha, ma, _), (hb, mb, _) = frames[a], frames[b]
            inter = float(np.minimum(ha, hb).sum())
            same_side = cosang > 0
            toward = float(A["n"] @ (B["c"] - A["c"]))
            labels = (A["group"], B["group"])
            confident = A["group_frac"] is not None and A["group_frac"] > 0.6 and B["group_frac"] > 0.6
            if confident and labels[0] != labels[1] and 0 not in labels:
                kind = "objetos_distintos"
            elif same_side and inter < 0.5:
                kind = "duplicada"
            elif not same_side and toward > 0 and float(B["n"] @ (A["c"] - B["c"])) > 0:
                kind = "paralelas_fisicas"
            elif not same_side and toward < 0:
                kind = "dos_caras"
            else:
                kind = "ambigua"
            out.append({"a": a, "b": b, "clase": kind, "separacion": sep, "separacion_rel": sep / med,
                        "angulo_deg": float(np.degrees(np.arccos(min(1.0, abs(cosang))))), "solape": overlap,
                        "interseccion_temporal": inter, "separacion_temporal": abs(ma - mb) / max(1, n_frames_total),
                        "mismo_lado": same_side,
                        "grupos": [GROUP_NAMES[x] if x is not None else None for x in labels],
                        "orientacion": A["orientacion"], "voxeles": [A["voxeles"], B["voxeles"]]})
    return out


def stacks_of(pairs, n_planes):
    """Pilas: planos unidos por pares (una superficie partida en capas). Clase de la pila: la más fuerte."""
    parent = list(range(n_planes))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for q in pairs:
        parent[root(q["a"])] = root(q["b"])
    rank = ["ambigua", "objetos_distintos", "paralelas_fisicas", "dos_caras", "duplicada"]
    stacks = {}
    for q in pairs:
        st = stacks.setdefault(root(q["a"]), {"planos": set(), "clase": "ambigua", "separacion_max_rel": 0.0})
        st["planos"] |= {q["a"], q["b"]}
        if rank.index(q["clase"]) > rank.index(st["clase"]):
            st["clase"] = q["clase"]
        st["separacion_max_rel"] = max(st["separacion_max_rel"], q["separacion_rel"])
    return [{"planos": sorted(v["planos"]), "clase": v["clase"], "separacion_max_rel": v["separacion_max_rel"]}
            for v in stacks.values()]


def entity_id(session, pose_version, kind, n, c, med):
    q = np.r_[np.round(n, 2), np.round(c / (0.05 * med))]
    return f"{session}/estructura/{pose_version[:8]}/{kind}-{hashlib.sha1(q.tobytes()).hexdigest()[:10]}"


def build(V, F, planes, pairs, med, up, session, pose_version, fragments=None, n_frames_total=1):
    """Entidades y relaciones. `fragments`: nodos del grafo de poses ([{"id", "frames"}]) para OBSERVADO_DESDE."""
    frames = [plane_frames(V, F, pl, n_frames_total) for pl in planes]
    ents = []
    for k, pl in enumerate(planes):
        gname = GROUP_NAMES[pl["group"]] if pl["group"] is not None else None
        if pl["orientacion"] == "horizontal":
            kind = "escalera" if gname == "escalera" else ("techo" if gname == "techo" or pl["n"] @ up < 0 else "piso")
        else:
            kind = KIND_BY_GROUP.get(gname, "plano") if pl["orientacion"] == "vertical" else "plano_oblicuo"
        q = V["centers"][pl["members"]]
        u = np.cross(pl["n"], up if abs(pl["n"] @ up) < 0.9 else [1.0, 0, 0])
        u /= np.linalg.norm(u)
        w = np.cross(pl["n"], u)
        ext = [float(np.ptp(np.percentile(q @ u, [5, 95]))), float(np.ptp(np.percentile(q @ w, [5, 95])))]
        fr = frames[k][2]
        e = {"id": entity_id(session, pose_version, kind, pl["n"], pl["c"], med), "tipo": kind,
             "orientacion": pl["orientacion"], "normal": pl["n"].tolist(), "centro": pl["c"].tolist(),
             "extension": ext, "area": float(len(pl["members"]) * V["voxel"] ** 2), "voxeles": pl["voxeles"],
             "incertidumbre_rms": pl["rms"], "grupo_semantico": gname, "frac_grupo": pl["group_frac"],
             "frames": [int(fr.min()), int(fr.max())] if len(fr) else None,
             "confianza": float(min(1.0, pl["voxeles"] / 2000) * min(1.0, len(np.unique(fr)) / 5)),
             "estado": "valida", "procedencia": "RANSAC sobre la nube estabilizada (diagnóstico, no reemplaza "
                                                 "a las observaciones)", "version_poses": pose_version}
        if fragments:
            obs = []
            for nd in fragments:
                share = np.isin(fr, nd["frames"]).mean() if len(fr) else 0.0
                if share >= 0.05:
                    obs.append(nd["id"])
            e["observado_desde"] = obs
        ents.append(e)
    rels = []
    for q in pairs:
        rel = RELATION_OF_CLASS[q["clase"]]
        rels.append({"tipo": rel, "origen": ents[q["a"]]["id"], "destino": ents[q["b"]]["id"],
                     "separacion": q["separacion"], "separacion_rel": q["separacion_rel"], "solape": q["solape"],
                     "interseccion_temporal": q["interseccion_temporal"], "angulo_deg": q["angulo_deg"]})
        if rel == "POSIBLE_DUPLICADO_DE":
            for x in (q["a"], q["b"]):
                ents[x]["estado"] = "posible_duplicado"
        elif rel == "AMBIGUA_CON":
            for x in (q["a"], q["b"]):
                if ents[x]["estado"] == "valida":
                    ents[x]["estado"] = "ambigua"
    # Adyacencia (planos no paralelos que se tocan) y soporte (piso bajo una pared que llega hasta él).
    touch = 2.5 * V["voxel"]
    rng = np.random.default_rng(0)
    sub = [V["centers"][pl["members"][rng.choice(len(pl["members"]), min(3000, len(pl["members"])), replace=False)]]
           for pl in planes]
    trees = [cKDTree(s) for s in sub]
    for a in range(len(planes)):
        for b in range(a + 1, len(planes)):
            if abs(planes[a]["n"] @ planes[b]["n"]) > np.cos(np.radians(45)):
                continue
            dmin = trees[b].query(sub[a], k=1, distance_upper_bound=touch)[0]
            if not np.isfinite(dmin).any():
                continue
            rels.append({"tipo": "ADYACENTE_A", "origen": ents[a]["id"], "destino": ents[b]["id"],
                         "contacto_puntos": int(np.isfinite(dmin).sum())})
            ea, eb = ents[a], ents[b]
            for fl, wl, pf, pw in ((ea, eb, planes[a], planes[b]), (eb, ea, planes[b], planes[a])):
                if fl["tipo"] == "piso" and wl["orientacion"] == "vertical":
                    low = np.percentile(V["centers"][pw["members"]] @ up, 2)
                    if abs(low - pf["c"] @ up) < 3 * V["voxel"]:
                        rels.append({"tipo": "SOPORTA", "origen": fl["id"], "destino": wl["id"]})
    for e in ents:
        for nid in e.pop("observado_desde", []):
            rels.append({"tipo": "OBSERVADO_DESDE", "origen": e["id"], "destino": nid})
    counts = {}
    for r in rels:
        counts[r["tipo"]] = counts.get(r["tipo"], 0) + 1
    return {"entidades": ents, "relaciones": rels, "conteo_relaciones": counts,
            "estados": {s: sum(e["estado"] == s for e in ents) for s in ("valida", "posible_duplicado", "ambigua")}}
