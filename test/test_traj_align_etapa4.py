"""Etapa 4: alineación y métricas de trayectorias con casos de solución conocida.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_traj_align_etapa4.py -q -p no:cacheprovider
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "vivo"))

from traj_align import (apply_sim3, apply_sim3_to_poses, associate, estimate_time_offset, rot_angle_deg,  # noqa: E402
                        rotation_offsets, rpe, umeyama, windowed_scale)


def _rot(v):
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(v, np.float64))
    return R


def _trajectory(n=300, seed=0):
    """Camino con giros y algo de altura; poses c2w en convención OpenCV."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 30, n)
    P = np.stack([np.sin(t / 4) * 3, 0.1 * np.sin(t), t * 0.4], 1)
    T = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        T[i, :3, :3] = _rot([0.05 * np.sin(t[i]), 0.6 * np.sin(t[i] / 3), 0.02 * np.cos(t[i] / 2)])
    T[:, :3, 3] = P
    return t, T + 0 * rng.normal()


def _transform(T, s, R, tr, C=np.eye(3)):
    out = apply_sim3_to_poses(s, R, tr, T)
    out[:, :3, :3] = out[:, :3, :3] @ C
    return out


def test_umeyama_recovers_known_sim3():
    t, Ta = _trajectory()
    R = _rot([0.3, -1.1, 0.4])
    s, tr = 2.7, np.array([1.0, -2.0, 0.5])
    Pb = (Ta[:, :3, 3] - tr) @ R / s                        # Pa = s R Pb + t
    s2, R2, t2 = umeyama(Pb, Ta[:, :3, 3])
    assert abs(s2 - s) < 1e-9 and np.allclose(R2, R, atol=1e-9) and np.allclose(t2, tr, atol=1e-9)
    assert np.allclose(apply_sim3(s2, R2, t2, Pb), Ta[:, :3, 3], atol=1e-9)


def test_rotation_offsets_detects_camera_convention_mismatch():
    t, Ta = _trajectory()
    W = _rot([0.2, 0.9, -0.3])
    Tb = _transform(Ta, 1.0, W.T, np.zeros(3))               # mismo trayecto en otro mundo
    W2, C2, _ = rotation_offsets(Ta[:, :3, :3], Tb[:, :3, :3])
    assert rot_angle_deg(C2) < 1e-4 and np.allclose(W2, W, atol=1e-6)   # C2 = I a 1e-15; arccos da ~1e-6°
    Cbad = _rot([0.0, 0.0, np.pi / 2])                        # p. ej. ejes de cámara ROS vs CV
    Tc = Tb.copy()
    Tc[:, :3, :3] = Tb[:, :3, :3] @ Cbad
    _, C3, info = rotation_offsets(Ta[:, :3, :3], Tc[:, :3, :3])
    assert abs(rot_angle_deg(C3) - 90.0) < 1e-4 and info["pares"] > 100


def test_associate_nearest_within_tolerance():
    ta = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
    tb = np.array([0.02, 0.19, 0.45, 2.0])
    ia, ib, dt = associate(ta, tb, tol=0.06)
    assert ia.tolist() == [0, 2, 3] and ib.tolist() == [0, 1, 2]
    assert np.allclose(dt, [0.02, -0.01, -0.05])


def test_time_offset_found():
    t, Ta = _trajectory(n=900)
    d_true = 0.3
    tb = t + d_true                                           # b ve lo mismo d_true s después
    d, c, c0 = estimate_time_offset(t, Ta[:, :3, :3], tb, Ta[:, :3, :3], max_lag=1.0)
    assert abs(d - d_true) < 0.03 and c > c0


def test_rpe_zero_for_scaled_copy_and_detects_scale_error():
    t, Ta = _trajectory()
    R = _rot([0.1, 0.4, 0.0])
    Tb = _transform(Ta, 0.25, R, np.array([3.0, 0, 0]))       # b = a en otro mundo, escala 1/4
    r = rpe(t, Ta, Tb, scale=4.0, delta=1.0)
    assert len(r["t"]) > 100 and np.nanmax(r["trans_rel"]) < 1e-9 and r["rot_deg"].max() < 1e-4   # arccos cerca de 1: ruido numérico ~1e-6°
    r2 = rpe(t, Ta, Tb, scale=2.0, delta=1.0)
    assert abs(np.nanmedian(r2["trans_rel"]) - 0.5) < 1e-9


def test_windowed_scale_tracks_a_scale_change():
    t, Ta = _trajectory(n=600)
    Pa = Ta[:, :3, 3]
    s_t = np.where(t < 15, 1.0, 2.0)                          # la escala de b salta a mitad de camino
    Pb = np.zeros_like(Pa)
    Pb[0] = Pa[0]
    for i in range(1, len(t)):
        Pb[i] = Pb[i - 1] + (Pa[i] - Pa[i - 1]) / s_t[i]
    c, s = windowed_scale(t, Pa, Pb, win=4.0, step=1.0)
    assert np.allclose(s[c < 13], 1.0, atol=1e-6) and np.allclose(s[c > 17], 2.0, atol=1e-6)
