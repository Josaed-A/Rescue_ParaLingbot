"""Etapas 12 y 13: mapa con geometría re-registrable y corrección por keyframes.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_map_kf_etapas12_13.py -q -p no:cacheprovider
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "vivo"))

import frames as F  # noqa: E402
from keyframe_correction import FrameAnchors, KeyframeHistory, correct_offline  # noqa: E402
from map_accumulator import MapAccumulator, backproject  # noqa: E402


def _pose(rotvec, p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = p
    return T


def _kf_row(kid, ts, T_cv):
    T = F.c2w_cv_to_map_link(T_cv)
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return [kid, ts, *T[:3, 3], *q]


def test_backproject_and_world_points():
    depth = np.full((4, 6), 2.0)
    K = np.array([[2.0, 0, 3], [0, 2.0, 2], [0, 0, 1]])
    P, (v, u) = backproject(depth, K)
    assert len(P) == 24 and np.allclose(P[:, 2], 2.0)
    m = MapAccumulator(voxel=0.1)
    m.add_frame(0, P, np.zeros((24, 3), np.uint8), _pose([0, 0, 0], [1, 0, 0]))
    W, _, _ = m.world_points()
    assert np.allclose(W, P + [1, 0, 0])


def test_repose_moves_history_without_double_surfaces():
    """El mismo plano visto desde dos frames: con poses consistentes, un vóxel por punto y visto
    por los dos frames; si un frame está corrido (pared doble), duplica; al corregir su pose, vuelve."""
    rng = np.random.default_rng(0)
    plane = np.c_[rng.uniform(-1, 1, (3000, 2)), np.full(3000, 3.0)]
    m = MapAccumulator(voxel=0.05)
    m.add_frame(0, plane, np.zeros((3000, 3), np.uint8), np.eye(4))
    T1 = _pose([0, 0, 0], [0, 0, 0.5])                      # el segundo frame avanzó 0.5
    m.add_frame(1, plane - [0, 0, 0.5], np.zeros((3000, 3), np.uint8), T1)
    good = m.stats()
    m.set_pose(1, _pose([0, 0, 0], [0, 0, 0.8]))            # pose equivocada: pared doble
    bad = m.stats()
    assert bad["voxeles"] > 1.5 * good["voxeles"]
    assert bad["voxeles_vistos_por_2_o_mas_frames"] < 0.2 * good["voxeles_vistos_por_2_o_mas_frames"]
    m.set_pose(1, T1)                                       # corrección (loop closure): vuelve
    assert m.stats()["voxeles"] == good["voxeles"] and m.n_reposes == 2


def test_keyframe_anchor_propagates_correction():
    hist = KeyframeHistory()
    Tk = _pose([0, 0.3, 0], [1, 0, 2])
    hist.update(10.0, [_kf_row(7, 10.0, Tk)])
    Tf = _pose([0, 0.35, 0], [1.2, 0, 2.4])                 # frame registrado cerca del keyframe 7
    anchors = FrameAnchors()
    assert anchors.register(0, 10.1, Tf, hist)
    corr = _pose([0, 0.1, 0], [0.5, 0, -0.3])               # loop closure: el keyframe se mueve
    mx, moved = hist.update(20.0, [_kf_row(7, 10.0, corr @ Tk)])
    assert moved == [7] and mx > 0.1
    assert np.allclose(anchors.corrected(0, hist), corr @ Tf, atol=1e-9)
    hist.update(30.0, [_kf_row(8, 25.0, Tk)])               # el keyframe 7 se borró (culling)
    assert anchors.corrected(0, hist) is None


def test_correct_offline_uses_keyframes_valid_at_frame_time():
    Tk_old = _pose([0, 0, 0], [0, 0, 1])
    Tk_new = _pose([0, 0, 0], [0.4, 0, 1])                  # BA posterior movió el keyframe 0.4
    msgs = [(1.0, [_kf_row(1, 1.0, Tk_old)]), (5.0, [_kf_row(1, 1.0, Tk_new)])]
    stamps = np.array([1.5, 2.0])
    poses = np.stack([_pose([0, 0, 0], [0, 0, 1.2]), _pose([0, 0, 0], [0, 0, 1.4])])
    out, disp, s = correct_offline(stamps, poses, msgs)
    assert np.allclose(disp, 0.4) and s["corregidos"] == 2
    assert np.allclose(out[:, :3, 3], poses[:, :3, 3] + [0.4, 0, 0])
