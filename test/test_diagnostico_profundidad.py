"""El diagnóstico de profundidad contra pose atribuye bien la causa en escenas sintéticas con un solo error conocido.

Cada escena tiene la profundidad exacta del render y UN tipo de error inyectado; el modelo que más baja el residuo
en los píxeles no usados para ajustar tiene que ser el de ese error."""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_o3d_estable as T  # noqa: E402
import diagnostico_profundidad as DP  # noqa: E402  (src/mapas ya está en el path por test_o3d_estable)


def diagnose(tmp_path, d, name):
    path = str(tmp_path / f"{name}.npz")
    T.write_npz(d, path)
    p = DP.Params(blocks=4, margin_frames=8, stride=2, min_corr=200)
    r = DP.run(path, str(tmp_path / f"{name}_diag"), p, log=lambda *a, **k: None)
    return {m: r["resumen"][m]["ajuste_todo"]["todos"][0] for m in DP.MODELS}, r


def test_depth_scale_error_is_explained_by_depth_not_by_pose(tmp_path):
    truth = T.path(T.CIRCUIT, 120)
    d = T.session(T.room_short_partition(), truth)
    t = np.arange(len(truth))
    s = 1 + 0.05 * np.sin(2 * np.pi * t / 45 + 1.0)
    d["depth"] = (d["depth"].astype(np.float64) * s[:, None, None]).astype(np.float32)
    res, r = diagnose(tmp_path, d, "escala")
    assert min(res["C"], res["D"]) < 0.6 * res["A"]
    assert min(res["C"], res["D"]) < res["P"] and min(res["C"], res["D"]) < res["F"]
    est = {o["frame"]: o["modelos"]["C"]["ajuste_todo"]["parametros"][0] for o in r["observaciones"]
           if o["modelos"]["C"]["ajuste_todo"]["n_corr"]}
    f = np.array(sorted(est))
    # Corrige en contra del error inyectado. La correlación no es perfecta: la referencia está hecha con otros
    # frames, que tienen sus propios errores de escala (se estima una escala relativa).
    assert np.corrcoef([est[i] for i in f], s[f])[0, 1] < -0.3


def test_pose_error_is_explained_by_pose_not_by_depth(tmp_path):
    truth = T.path(T.CIRCUIT, 120)
    rng = np.random.default_rng(0)
    pert = truth.copy()
    for i in range(len(truth)):
        pert[i, :3, :3] = Rot.from_euler("xyz", rng.normal(0, 2.0, 3), degrees=True).as_matrix() @ truth[i, :3, :3]
        pert[i, :3, 3] = truth[i, :3, 3] + rng.normal(0, 0.05, 3)
    d = T.session(T.room_short_partition(), truth, pert)
    res, _ = diagnose(tmp_path, d, "pose")
    assert res["P"] < 0.85 * res["A"]
    assert res["P"] < min(res["C"], res["D"], res["F"])


def test_corrected_npz_only_corrects_off_floor_with_a_plausible_model_and_keeps_the_original(tmp_path):
    """La corrección escribe un .npz nuevo: el original no cambia, el piso no se corrige, se usa el primer modelo
    aceptado (E, si no C), un ajuste implausible se rechaza, y la procedencia queda en el archivo."""
    import json
    n, H, W = 3, 8, 8
    depth = np.ones((n, H, W), np.float32)
    src = str(tmp_path / "s.npz")
    np.savez(src, depth=depth, depth_conf=np.ones_like(depth), images=np.zeros((n, H, W, 3), np.uint8),
             intrinsic=np.tile([[8.0, 0, 4], [0, 8.0, 4], [0, 0, 1]], (n, 1, 1)),
             extrinsic=np.tile(np.eye(4)[:3], (n, 1, 1)), is_real=np.ones(n, bool))
    labels = np.zeros((n, H, W), np.uint8)
    labels[:, 4:] = 3                                                   # mitad de abajo: piso
    np.savez(str(tmp_path / "f.npz"), labels=labels)
    p = DP.Params(min_corr=10)

    def fit(par):
        return {"parametros": par, "n_corr": 100, "estable": True, "se": [0.001] * len(par), "colinealidad": 0.5,
                "prueba": {"no_piso": {"n": 100, "mediana": 0.5}}}

    def obs(i, e=None, c=None):
        mods = {"A": {"ajuste_todo": {"prueba": {"no_piso": {"n": 100, "mediana": 1.0}}}}}
        if e is not None:
            mods["E"] = {"ajuste_sin_piso": fit(e)}
        if c is not None:
            mods["C"] = {"ajuste_sin_piso": fit(c)}
        return {"frame": i, "modelos": mods}
    report = {"npz": src, "version_poses_medidas": "v", "profundidad_mediana": 1.0,
              "observaciones": [obs(0, e=[0.1, 0.0]), obs(1, e=[0.5, 0.0], c=[-0.1]), obs(2)]}
    out = str(tmp_path / "s_prof.npz")
    meta = DP.write_corrected(report, str(tmp_path / "f.npz"), out, p)
    z, z0 = np.load(out), np.load(src)
    assert np.array_equal(z0["depth"], depth)                           # el original no cambia
    assert np.allclose(z["depth"][0, :4], 1.1) and np.allclose(z["depth"][0, 4:], 1.0)   # E; el piso no
    assert np.allclose(z["depth"][1, :4], 0.9) and np.allclose(z["depth"][1, 4:], 1.0)   # E implausible -> C
    assert np.allclose(z["depth"][2], 1.0)                                                # sin referencia
    assert meta["frames_corregidos"] == {"E": 1, "C": 1}
    prov = json.loads(str(z["correccion_profundidad"]))
    assert prov["meta"]["npz_original"] == src and "implausible" in prov["por_frame"]["1"]["motivo"] or \
        prov["por_frame"]["1"]["modelo"] == "C"
