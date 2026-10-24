"""Estabilización Open3D antes del TSDF (o3d_estable.py, grafo_poses.py, grafo_estructura.py, tsdf_mesh.py --stable)
con escenas sintéticas de geometría conocida: la profundidad se renderiza con el raycasting de Open3D desde poses
exactas, así se sabe qué es error inyectado y qué es geometría real.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_o3d_estable.py -q -p no:cacheprovider
"""
import json
import os
import subprocess
import sys

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src", "mapas"))

import grafo_poses as GP  # noqa: E402
import o3d_estable as OE  # noqa: E402

W = H = 160
FX = 110.0
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1.0]])


# --- escenas ----------------------------------------------------------------------------------------------

def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = o3d.geometry.TriangleMesh.create_box(*(hi - lo))
    m.translate(lo)
    return m.subdivide_midpoint(3)                   # triángulos chicos: textura con esquinas para ORB


def room():
    """Habitación de 8 x 6 con: tabique interior (y = 3, x de 0 a 5, 0.15 de espesor) que se ve de los dos lados,
    puerta en la pared y = 0 (x de 3.5 a 4.5), obstáculo, escalón (plataforma de 0.25) y techo."""
    parts = [box([-1, -1, -0.1], [9, 7, 0]),                                   # piso
             box([-0.1, 0, 0], [0, 6, 2.5]), box([8, 0, 0], [8.1, 6, 2.5]),    # paredes x
             box([0, -0.1, 0], [3.5, 0, 2.5]), box([4.5, -0.1, 0], [8, 0, 2.5]),   # pared y=0 con puerta
             box([0, 6, 0], [8, 6.1, 2.5]),
             box([0, 2.925, 0], [5, 3.075, 2.5]),                              # tabique
             box([6.6, 0.9, 0], [7.3, 1.5, 0.8]),                              # obstáculo
             box([0, 4.6, 0], [1.6, 6, 0.25]),                                  # escalón
             box([-1, -1, 2.5], [9, 7, 2.6])]                                  # techo
    return parts


def corridor(period=4.0, reps=4):
    """Pasillo recto con un tramo repetido (misma geometría y misma textura cada `period`)."""
    seg = [box([0, -1, -0.1], [period, 1, 0]), box([0, -1.1, 0], [period, -1, 2.4]), box([0, 1, 0], [period, 1.1, 2.4]),
           box([1.2, -1, 0], [1.6, -0.6, 1.0])]
    return [m.translate([k * period, 0, 0], relative=True) for k in range(reps) for m in [o3d.geometry.TriangleMesh(x)
                                                                                          for x in seg]]


def look(pos, fwd, pitch_deg=-12.0):
    f = np.asarray(fwd, float)
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    f = Rotation.from_rotvec(np.radians(pitch_deg) * np.cross(f, up) / np.linalg.norm(np.cross(f, up))).apply(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, down, f], 1)
    T[:3, 3] = pos
    return T


def path(points, n, height=1.4):
    """Recorrido por los puntos dados, mirando hacia adelante, con n cámaras."""
    P = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = np.linspace(0, s[-1], n)
    xy = np.stack([np.interp(t, s, P[:, k]) for k in range(2)], 1)
    poses = []
    for i in range(n):
        j = min(i + 3, n - 1) if i < n - 1 else i
        f = xy[j] - xy[max(0, j - 3)] if j > 0 else xy[1] - xy[0]
        f = np.r_[f, 0.0]
        poses.append(look(np.r_[xy[i], height], f))
    return np.array(poses)


def render(parts, c2w, seed=0, repeat_every=None):
    """Profundidad (z de la cámara) e imagen con un color al azar por triángulo. Con repeat_every, la parte j
    lleva los mismos colores que la parte j % repeat_every (textura repetida, como un pasillo de tramos iguales)."""
    mesh = o3d.geometry.TriangleMesh()
    cols = []
    for j, p_ in enumerate(parts):
        mesh += p_
        k = j if repeat_every is None else j % repeat_every
        cols.append(np.random.default_rng(seed * 1000 + k).integers(30, 230, (len(p_.triangles), 3), dtype=np.uint8))
    tri_col = np.concatenate(cols)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    # Open3D tira el rayo del píxel u por u + 0.5 y con dirección de z = 1 (t_hit ya es la profundidad z):
    # con el punto principal corrido medio píxel, el píxel u corresponde a (u - cx) / fx, como al levantar.
    Kr = K.copy()
    Kr[0, 2] -= 0.5
    Kr[1, 2] -= 0.5
    depth, images = [], []
    for T in c2w:
        rays = scene.create_rays_pinhole(o3d.core.Tensor(Kr), o3d.core.Tensor(np.linalg.inv(T)), W, H)
        r = scene.cast_rays(rays)
        t = r["t_hit"].numpy()
        z = np.where(np.isfinite(t), t, 0).astype(np.float32)
        pid = r["primitive_ids"].numpy()
        img = np.zeros((H, W, 3), np.uint8)
        hit = np.isfinite(t)
        img[hit] = tri_col[pid[hit]]
        depth.append(z)
        images.append(img)
    return np.array(depth), np.array(images)


def session(parts, c2w_true, c2w_used=None, path_="/tmp/sint/prueba_x/eval/sesion.npz", repeat_every=None):
    depth, images = render(parts, c2w_true, repeat_every=repeat_every)
    used = c2w_true if c2w_used is None else c2w_used
    n = len(c2w_true)
    return {"depth": depth, "depth_conf": np.full_like(depth, 2.0), "images": images,
            "intrinsic": np.tile(K, (n, 1, 1)).astype(np.float32),
            "extrinsic": np.linalg.inv(used)[:, :3, :4].astype(np.float32), "is_real": np.ones(n, bool),
            "path": path_}


def drift(c2w, start, trans, yaw_deg):
    """Deriva que crece desde el frame `start` hasta el final: una transformación del mundo en rampa."""
    out = c2w.copy()
    n = len(c2w)
    for i in range(start, n):
        a = (i - start) / max(1, n - 1 - start)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("z", a * yaw_deg, degrees=True).as_matrix()
        T[:3, 3] = a * np.asarray(trans)
        out[i] = T @ c2w[i]
    return out


LOOP = [(1.5, 1.5), (6.0, 2.2), (6.6, 4.0), (3.0, 4.5), (6.6, 4.0), (6.0, 2.2), (1.5, 1.5)]
# Circuito cerrado alrededor de un tabique corto: el final vuelve a recorrer el comienzo mirando hacia el mismo
# lado (una revisita que se puede reconocer por imagen).
CIRCUIT = [(1.0, 1.5), (6.8, 1.5), (6.8, 4.5), (1.0, 4.5), (1.0, 1.5), (4.5, 1.5)]


def room_short_partition():
    parts = room()
    parts[6] = box([2.0, 2.925, 0], [5, 3.075, 2.5])
    return parts


def params(**kw):
    p = OE.Params(stride=2, planes_max=25)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    c2w = path(LOOP, 120)
    d = session(room(), c2w)
    out = str(tmp_path_factory.mktemp("limpia") / "sint")
    return d, c2w, OE.run(d, out, params(), log=lambda *a, **k: None), out


# --- pruebas ----------------------------------------------------------------------------------------------

def test_clean_session_keeps_poses_and_real_geometry(clean):
    d, c2w, rep, out = clean
    assert "corregidas" not in rep["grafo_poses"]["decision"]           # se conservan (o se rechaza corregir)
    assert rep["correccion_por_frame"]["camara_desplazamiento_max"] < 1e-6
    assert rep["filtrado"]["frac_quitada"] < 0.03                          # casi nada que quitar
    z = np.load(out + "_estable.npz")
    assert np.allclose(z["c2w"], z["c2w_original"], atol=1e-6) and np.allclose(z["c2w_original"], c2w, atol=1e-5)
    pc = np.asarray(o3d.io.read_point_cloud(out + "_nube_estable.ply").points)
    # La puerta sigue abierta, el obstáculo y el escalón siguen ahí.
    door = (pc[:, 0] > 3.6) & (pc[:, 0] < 4.4) & (np.abs(pc[:, 1]) < 0.12) & (pc[:, 2] > 0.1) & (pc[:, 2] < 2.3)
    obstacle = (pc[:, 0] > 6.55) & (pc[:, 0] < 7.35) & (pc[:, 1] > 0.85) & (pc[:, 1] < 1.55) & (np.abs(pc[:, 2] - 0.8) < 0.03)
    step = (pc[:, 0] < 1.55) & (pc[:, 1] > 4.65) & (np.abs(pc[:, 2] - 0.25) < 0.03)
    assert door.sum() == 0 and obstacle.sum() > 50 and step.sum() > 50


def test_the_partition_seen_from_both_sides_is_two_faces_not_a_duplicate(clean):
    rep = clean[2]
    pairs = rep["diagnostico"]["estable"]["pares"]
    two = [q for q in pairs if q["clase"] == "dos_caras" and abs(q["separacion"] - 0.15) < 0.04]
    assert two, [(q["clase"], round(q["separacion"], 3)) for q in pairs]
    assert not [q for q in pairs if q["clase"] == "duplicada" and abs(q["separacion"] - 0.15) < 0.04]


def test_outliers_are_masked_and_true_surface_kept(tmp_path):
    c2w = path(LOOP[:4], 60)
    d = session(room(), c2w)
    rng = np.random.default_rng(1)
    spikes = rng.random(d["depth"].shape) < 0.01
    d["depth"] = np.where(spikes, d["depth"] * rng.uniform(0.3, 0.7, d["depth"].shape), d["depth"]).astype(np.float32)
    out = str(tmp_path / "ruido")
    OE.run(d, out, params(register=False), log=lambda *a, **k: None)
    z = np.load(out + "_estable.npz")
    s = int(z["stride"])
    sp = spikes[:, s // 2::s, s // 2::s] & (d["depth"][:, s // 2::s, s // 2::s] > 0)
    keep = z["keep"]
    assert (~keep[sp]).mean() > 0.8                     # la mayoría de los atípicos, fuera
    assert keep[~sp].mean() > 0.97                      # la superficie real, casi entera


def test_drift_on_the_return_is_closed_and_reduced(tmp_path):
    truth = path(CIRCUIT, 160)
    drifted = drift(truth, 40, trans=(0.3, -0.25, 0.08), yaw_deg=3.0)
    d = session(room_short_partition(), truth, drifted)
    out = str(tmp_path / "deriva")
    rep = OE.run(d, out, params(), log=lambda *a, **k: None)
    g = json.load(open(out + "_grafo.json"))
    assert g["conteo_aristas"].get("cierre_aceptada", 0) >= 1, g["conteo_aristas"]
    assert "corregidas" in rep["grafo_poses"]["decision"], rep["grafo_poses"]["decision"]
    z = np.load(out + "_estable.npz")
    err0 = np.linalg.norm(z["c2w_original"][:, :3, 3] - truth[:, :3, 3], axis=1)
    err1 = np.linalg.norm(z["c2w"][:, :3, 3] - truth[:, :3, 3], axis=1)
    assert err1[120:].mean() < 0.5 * err0[120:].mean()
    # La vuelta deja la misma pared dos veces: el diagnóstico la ve en la original.
    assert rep["diagnostico"]["original"]["resumen"]["por_clase"].get("duplicada", 0) >= 1


def test_repetitive_corridor_does_not_create_a_false_loop(tmp_path):
    c2w = path([(0.3, 0.0), (15.0, 0.0)], 90)
    d = session(corridor(), c2w, repeat_every=4)          # misma geometría y misma textura cada tramo
    out = str(tmp_path / "pasillo")
    # Todos los fragmentos son candidatos a cierre: tienen que caer por plausibilidad, no por no buscarlos.
    rep = OE.run(d, out, params(loop_radius_rel=100.0, loop_overlap=0.0), log=lambda *a, **k: None)
    g = json.load(open(out + "_grafo.json"))
    loops = [e for e in g["aristas"] if e["tipo"] == "cierre"]
    # Cierres entre fragmentos que ven el mismo tramo de pasillo son legítimos (corrección ~0). Lo que no puede
    # pasar es aceptar la coincidencia con el tramo repetido: correrse un período (4).
    assert loops and not [e for e in loops if e["estado"] == "aceptada" and e["desplazamiento"] > 2.0]
    # Hubo coincidencias con el tramo repetido que pasaron la verificación visual, y las frenó la plausibilidad.
    shifted = [e for e in loops if (e.get("desplazamiento") or 0) > 2.0]
    assert shifted and all(e["estado"] == "rechazada" and "implausible" in e["motivo"] for e in shifted)
    assert rep["correccion_por_frame"]["camara_desplazamiento_max"] < 0.05


def test_graph_ids_do_not_depend_on_traversal_order():
    assert GP.session_id("/a/prueba_3/eval/m.npz", np.ones((2, 3, 4))) == \
        GP.session_id("/b/prueba_3/eval/m.npz", np.ones((2, 3, 4)))
    frags = GP.make_fragments(list(range(20)), 8, 4)
    assert frags[0] == list(range(8)) and frags[1][0] == 4 and frags[-1][-1] == 19


def write_npz(d, path_):
    os.makedirs(os.path.dirname(path_), exist_ok=True)
    np.savez(path_, **{k: v for k, v in d.items() if k != "path"})


def test_tsdf_uses_stable_inputs_skips_when_unchanged_and_keeps_the_door(tmp_path):
    c2w = path(LOOP[:4], 50)
    npz = str(tmp_path / "prueba_s" / "eval" / "sesion.npz")
    write_npz(session(room(), c2w), npz)
    py = sys.executable
    st = str(tmp_path / "prueba_s" / "exports" / "estable" / "sesion")
    r = subprocess.run([py, "src/mapas/o3d_estable.py", npz, "--out", st, "--stride", "2"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    mesh = str(tmp_path / "malla")
    cmd = [py, "src/mapas/tsdf_mesh.py", npz, "--out", mesh, "--stable", st + "_estable.npz", "--skip_if_same"]
    r1 = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr[-2000:]
    info = json.load(open(mesh + "_info.json"))
    assert info["stable"] and info["version_poses"] and "Open3D" in info["tipo_geometria"]
    r2 = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert "sin cambios en las entradas" in r2.stdout
    # Otra versión de poses (sin registro: misma geometría, otra huella de parámetros) -> se reintegra.
    subprocess.run([py, "src/mapas/o3d_estable.py", npz, "--out", st, "--stride", "2", "--no_sor"], cwd=REPO,
                   capture_output=True, text=True, check=True)
    r3 = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert "sin cambios" not in r3.stdout and r3.returncode == 0
    # La puerta sigue abierta en la malla: un rayo por su centro no choca con nada en la pared.
    m = o3d.io.read_triangle_mesh(mesh + ".ply")
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    ray = o3d.core.Tensor([[4.0, 1.0, 1.2, 0.0, -1.0, 0.0]], dtype=o3d.core.Dtype.Float32)
    t = float(sc.cast_rays(ray)["t_hit"].numpy()[0])
    assert not np.isfinite(t) or t > 1.3
    # Una estabilización de otra reconstrucción se rechaza.
    other = str(tmp_path / "otra" / "eval" / "sesion.npz")
    write_npz(session(room(), path(LOOP[:4], 50) + np.r_[0, 0, 0, 0.01].reshape(1, 4) * 0), other)
    d2 = np.load(other)
    np.savez(other, **{k: (d2[k] * 1.01 if k == "extrinsic" else d2[k]) for k in d2.files})
    r4 = subprocess.run([py, "src/mapas/tsdf_mesh.py", other, "--out", str(tmp_path / "m2"), "--stable",
                         st + "_estable.npz"], cwd=REPO, capture_output=True, text=True)
    assert r4.returncode != 0 and "otra reconstrucción" in (r4.stderr + r4.stdout)


def test_genpc_interface_flags_inferred_geometry_that_closes_a_door():
    import genpc_interfaz as GI

    parts = room() + [box([-1, -1.6, 0], [9, -1.5, 2.5])]        # pared exterior: se ve a través de la puerta
    c2w = np.array([look(np.array([x, 2.0, 1.4]), np.array([0, -1.0, 0]), pitch_deg=-5) for x in (3.6, 4.0, 4.4)])
    depth, _ = render(parts, c2w)
    ys, xs = np.mgrid[0:H, 0:W]
    obs = []
    for k in range(len(c2w)):
        z = depth[k]
        ok = z > 0
        pc = np.stack([(xs[ok] - K[0, 2]) / K[0, 0] * z[ok], (ys[ok] - K[1, 2]) / K[1, 1] * z[ok], z[ok]], 1)
        obs.append(pc @ c2w[k, :3, :3].T + c2w[k, :3, 3])
    obs = np.concatenate(obs)
    g = np.arange(0.0, 1.0, 0.05)
    door_fill = np.array([[3.55 + a, -0.05, 0.3 + b] for a in g[:18] for b in np.arange(0, 1.8, 0.05)])
    hidden_fill = np.array([[6.75 + a, 1.0 + b, 0.2 + c] for a in g[:10] for b in g[:8] for c in g[:10]])  # dentro del obstáculo
    res = GI.validate(obs, np.concatenate([obs, door_fill, hidden_fill]), depth, np.tile(K, (len(c2w), 1, 1)), c2w,
                      range(len(c2w)), voxel=0.02)
    inf, bad = res["mascara_inferidos"], res["mascara_contradicen"]
    n_obs = len(obs)
    door_bad = bad[:len(door_fill)] if inf[n_obs:n_obs + len(door_fill)].all() else None
    assert door_bad is not None and door_bad.mean() > 0.8          # rellenar la puerta contradice lo observado
    assert bad[len(door_fill):].mean() < 0.05                      # dentro del obstáculo nadie vio espacio libre
    assert res["observaciones_perdidas"] == 0 and not res["aceptable"]


def test_floor_permanence_sees_a_floor_split_in_layers_between_visits():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 2, (40000, 2))
    pos = np.repeat(np.arange(4), 10000)                       # 4 visitas de 16 frames
    z = rng.normal(0, 0.002, len(xy))
    G = np.full(len(xy), OE.GROUP_NAMES.index("piso"))
    up = np.array([0, 0, 1.0])
    one = OE.floor_permanence(np.c_[xy, z], pos * 16, G, up, 0.1, 16)
    split = OE.floor_permanence(np.c_[xy, z + 0.1 * (pos >= 2)], pos * 16, G, up, 0.1, 16)
    assert one["frac_celdas_separadas"] == 0 and one["diferencia_entre_visitas"]["p90"] < 0.01
    assert split["frac_celdas_separadas"] > 0.9 and abs(split["diferencia_entre_visitas"]["mediana"] - 0.1) < 0.01
    assert OE.floor_permanence(np.c_[xy, z], pos * 16, G * 0, up, 0.1, 16)["celdas"] == 0   # sin piso


def test_multiway_registration_closes_the_drift_and_keeps_the_floor_in_one_layer(tmp_path):
    """El registro multivía del tutorial de Open3D sobre la deriva al volver: cámaras y piso."""
    truth = path(CIRCUIT, 160)
    drifted = drift(truth, 40, trans=(0.3, -0.25, 0.08), yaw_deg=3.0)
    d = session(room_short_partition(), truth, drifted)
    out = str(tmp_path / "multiway")
    rep = OE.run(dict(d), out, params(registro="multiway"), log=lambda *a, **k: None)
    assert rep["grafo_poses"]["decision"].startswith("multiway")
    assert os.path.isfile(out + "_combinada.ply") and len(o3d.io.read_point_cloud(out + "_combinada.ply").points)
    z = np.load(out + "_estable.npz")
    err0 = np.linalg.norm(z["c2w_original"][:, :3, 3] - truth[:, :3, 3], axis=1)
    err1 = np.linalg.norm(z["c2w"][:, :3, 3] - truth[:, :3, 3], axis=1)
    assert err1[120:].mean() < 0.2 * err0[120:].mean()
    # El piso (z = 0 con las poses verdaderas) visto en distintas visitas vuelve a una sola capa.
    p = params()
    real = np.flatnonzero(d["is_real"])
    ids = [int(i) for i in real]
    med = float(np.median(d["depth"][real][d["depth"][real] > 0]))
    cthr = float(np.percentile(d["depth_conf"][real[::10]], p.conf_percentile))
    dmax = float(np.percentile(d["depth"][real[::10]][d["depth"][real[::10]] > 0], p.depth_max_percentile))
    Pt, F, *_ = OE.lift(d, ids, {i: truth[i] for i in ids}, p, cthr, dmax)
    G = np.where(np.abs(Pt[:, 2]) < 0.03, OE.GROUP_NAMES.index("piso"), 0)
    pos = np.searchsorted(real, F)

    def floor(c2w):
        P, *_ = OE.lift(d, ids, {i: c2w[k] for k, i in enumerate(ids)}, p, cthr, dmax)
        return OE.floor_permanence(P, pos, G, np.array([0, 0, 1.0]), p.floor_cell_rel * med, p.floor_visit_frames)

    before, after = floor(z["c2w_original"]), floor(z["c2w"])
    assert before["frac_celdas_separadas"] > 0.3
    assert after["frac_celdas_separadas"] < 0.02
    assert after["diferencia_entre_visitas"]["p90"] < 0.2 * before["diferencia_entre_visitas"]["p90"]


def test_floor_alignment_brings_each_frame_to_the_combined_floor():
    """Cada frame ve el mismo piso (z = 0) con su propio desfase e inclinación, como los errores de profundidad
    de un modelo monocular: la alineación lo lleva a una sola capa y no toca lo que el piso no fija."""
    rng = np.random.default_rng(1)
    n, per = 30, 3000
    P, F, Z = [], [], []
    truth_off = []
    for i in range(n):
        xy = np.c_[rng.uniform(0, 3, per), rng.uniform(0, 2, per)] + [0.05 * i, 0]
        dz, a, b = rng.normal(0, 0.03), rng.normal(0, 0.02), rng.normal(0, 0.02)
        z = dz + a * (xy[:, 0] - xy[:, 0].mean()) + b * (xy[:, 1] - xy[:, 1].mean()) + rng.normal(0, 0.002, per)
        P.append(np.c_[xy, z])
        F.append(np.full(per, i))
        Z.append(np.full(per, 1.0))
        truth_off.append(dz)
    P, F, Z = np.concatenate(P).astype(np.float32), np.concatenate(F), np.concatenate(Z)
    G = np.full(len(P), OE.GROUP_NAMES.index("piso"))
    real = np.arange(n)
    p = params(floor_min_pts=200, floor_cell_rel=0.05, floor_max_dz_rel=0.5, floor_max_tilt_deg=5.0, floor_smooth=1)
    up = np.array([0, 0, 1.0])
    corr, st = OE.align_floor(P, F, Z, G, up, real, 1.0, p)
    assert corr is not None and st["frames_con_piso"] == n
    Q = np.concatenate([P[F == i] @ corr[i][:3, :3].T + corr[i][:3, 3] for i in range(n)])
    before = OE.floor_permanence(P, F, G, up, 0.2, 1)                  # celdas con >= 10 puntos por frame
    after = OE.floor_permanence(Q, np.concatenate([F[F == i] for i in range(n)]), G, up, 0.2, 1)
    assert after["espesor_p10_p90"]["mediana"] < 0.3 * before["espesor_p10_p90"]["mediana"]
    assert after["diferencia_entre_visitas"]["p90"] < 0.3 * before["diferencia_entre_visitas"]["p90"]
    # solo altura e inclinación: en planta los puntos no se mueven más que lo que gira una inclinación chica
    shift = np.concatenate([(P[F == i] @ corr[i][:3, :3].T + corr[i][:3, 3] - P[F == i])[:, :2] for i in range(n)])
    assert np.abs(shift).max() < 0.01



def test_4dof_priors_let_the_loop_fix_heading_but_not_tilt():
    """Un cierre entre el primer y el último nodo pide 8° de rumbo y 6° de inclinación: con las aristas de LingBot
    rígidas en inclinación y altura (dof4), Open3D corrige el rumbo y deja la inclinación."""
    reg = o3d.pipelines.registration
    up = np.array([0.0, 0.0, 1.0])
    loop = np.eye(4)
    loop[:3, :3] = Rotation.from_euler("zx", [8, 6], degrees=True).as_matrix()
    for dof4, tilt_max in ((False, None), (True, 0.3)):
        pg = reg.PoseGraph()
        for _ in range(4):
            pg.nodes.append(reg.PoseGraphNode(np.eye(4)))
        for s in range(3):
            pg.edges.append(reg.PoseGraphEdge(s, s + 1, np.eye(4), GP.prior_information(1.0, up if dof4 else None, 1e4)))
        pg.edges.append(reg.PoseGraphEdge(0, 3, loop, np.eye(6) * 10.0, uncertain=False))
        reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(),
                                reg.GlobalOptimizationOption(max_correspondence_distance=1.0, reference_node=0))
        R = np.asarray(pg.nodes[3].pose)[:3, :3]
        tilt = np.degrees(np.arccos(np.clip(R @ up @ up, -1, 1)))
        yaw = abs(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        assert yaw > 2                                        # el rumbo se corrige en los dos casos
        if dof4:
            assert tilt < tilt_max                            # la inclinación no
        else:
            assert tilt > 1                                   # sin dof4, el cierre también inclina


def test_4dof_projection_keeps_heading_and_horizontal_motion_only():
    """Con dof4 las mediciones se restringen a los mismos grados de libertad que los priors (auditoría 2026-10-18)."""
    up, c = np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 0.5])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("zx", [8, 6], degrees=True).as_matrix()
    T[:3, 3] = [0.3, -0.2, 0.15]
    T.setflags(write=False)                                                   # como las aristas de Open3D
    P = GP.project_4dof(T, up, c)
    assert np.allclose(P[:3, :3] @ up, up)                                    # sin inclinación
    yaw = np.degrees(np.arctan2(P[1, 0], P[0, 0]))
    assert abs(yaw - 8) < 0.5                                                 # el rumbo se conserva
    d, dp = T[:3, :3] @ c + T[:3, 3] - c, P[:3, :3] @ c + P[:3, 3] - c
    assert abs(dp[2]) < 1e-9 and np.allclose(dp[:2], d[:2])                  # el centro se mueve solo en horizontal


def test_pairwise_scale_sees_scale_that_a_truncated_icp_cannot():
    rng = np.random.default_rng(0)
    A = rng.uniform(-2, 2, (300, 3))
    R = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    B = 1.15 * A @ R.T + [0.4, 0.1, 0.0]
    B[:30] += rng.normal(0, 0.3, (30, 3))                                    # 10 % de correspondencias malas
    assert abs(GP.pairwise_scale(A, B) - 1.15) < 0.02
