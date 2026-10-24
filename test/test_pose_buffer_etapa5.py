"""Etapa 5: PoseBuffer (interpolación lineal + SLERP, segmentos, cortes, tolerancias).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_pose_buffer_etapa5.py -q -p no:cacheprovider
"""
import os
import sys
import threading

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "vivo"))

from pose_buffer import PoseBuffer, QueryKind, interpolate_pose, mat_to_quat, quat_to_mat, slerp  # noqa: E402


def _pose(rotvec, p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = p
    return T


def test_quaternion_roundtrip_including_180_degrees():
    rng = np.random.default_rng(0)
    mats = list(Rotation.random(200, random_state=1).as_matrix())
    mats += [Rotation.from_rotvec([np.pi, 0, 0]).as_matrix(), Rotation.from_rotvec([0, np.pi, 0]).as_matrix(),
             Rotation.from_rotvec([0, 0, np.pi]).as_matrix(), np.eye(3)]
    for R in mats:
        assert np.allclose(quat_to_mat(mat_to_quat(R)), R, atol=1e-9)
    del rng


def test_slerp_matches_scipy_and_takes_short_arc():
    R = Rotation.random(2, random_state=3)
    q0, q1 = R.as_quat()
    ref = Slerp([0, 1], R)
    for a in (0.0, 0.25, 0.5, 0.9, 1.0):
        mine = quat_to_mat(slerp(q0, q1, a))
        assert np.allclose(mine, ref([a]).as_matrix()[0], atol=1e-9)
    # -q1 es la misma rotación: el resultado no debe cambiar
    assert np.allclose(quat_to_mat(slerp(q0, -q1, 0.3)), quat_to_mat(slerp(q0, q1, 0.3)), atol=1e-9)


def test_interpolation_is_not_elementwise():
    """Entre una rotación de 0° y una de 170° el promedio de matrices no es una rotación; SLERP sí,
    y gira justo la mitad del ángulo."""
    T0, T1 = _pose([0, 0, 0], [0, 0, 0]), _pose([0, np.radians(170), 0], [2, 0, 0])
    T = interpolate_pose(T0, T1, 0.5)
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
    ang = np.degrees(Rotation.from_matrix(R).magnitude())
    assert abs(ang - 85.0) < 1e-6 and np.allclose(T[:3, 3], [1, 0, 0])
    avg = 0.5 * (T0[:3, :3] + T1[:3, :3])
    assert abs(np.linalg.det(avg) - 1) > 0.1           # lo que NO hay que hacer


def test_query_kinds_and_tolerances():
    b = PoseBuffer(max_gap=0.25, edge_tol=0.05)
    b.add(1.0, _pose([0, 0, 0], [0, 0, 0]))
    b.add(1.2, _pose([0, 0.2, 0], [1, 0, 0]))
    b.add(2.0, _pose([0, 0.4, 0], [5, 0, 0]))           # hueco de 0.8 s antes de esta muestra
    q = b.query(1.0)
    assert q.kind is QueryKind.EXACT
    q = b.query(1.05)
    assert q.kind is QueryKind.INTERP and np.allclose(q.c2w[:3, 3], [0.25, 0, 0]) and abs(q.gap - 0.2) < 1e-12
    assert q.meta["alpha"] == 0.25 / 1.0 or abs(q.meta["alpha"] - 0.25) < 1e-9
    q = b.query(1.6)                                     # dentro del hueco grande: no se inventa
    assert q.kind is QueryKind.NONE and abs(q.dt_nearest - 0.4) < 1e-9
    q = b.query(1.23)                                    # hueco grande pero cerca de una muestra
    assert q.kind is QueryKind.NEAREST and np.allclose(q.c2w[:3, 3], [1, 0, 0])
    assert b.query(0.97).kind is QueryKind.NEAREST       # antes del inicio, dentro de edge_tol
    assert b.query(0.9).kind is QueryKind.NONE           # antes del inicio: no se extrapola
    assert b.query(2.2).kind is QueryKind.NONE


def test_no_interpolation_across_segments_or_breaks():
    b = PoseBuffer(max_gap=1.0, edge_tol=0.0)
    b.add(0.0, _pose([0, 0, 0], [0, 0, 0]), segment=0)
    b.add(0.5, _pose([0, 0, 0], [1, 0, 0]), segment=1)   # Stella se reinició: otro mundo
    assert b.query(0.25).kind is QueryKind.NONE
    c = PoseBuffer(max_gap=1.0, edge_tol=0.0)
    c.add(0.0, _pose([0, 0, 0], [0, 0, 0]))
    c.add(0.5, _pose([0, 0, 0], [1, 0, 0]))
    assert c.query(0.25).kind is QueryKind.INTERP
    c.add_break(0.3)                                     # p. ej. corrección de loop closure
    assert c.query(0.25).kind is QueryKind.NONE


def test_out_of_order_insert_duplicates_and_capacity():
    b = PoseBuffer(max_samples=5)
    for t in (3.0, 1.0, 2.0, 5.0, 4.0):
        b.add(t, _pose([0, 0, 0], [t, 0, 0]))
    b.add(2.0, _pose([0, 0, 0], [20, 0, 0]))             # mismo stamp: reemplaza, no duplica
    t, T, _ = b.arrays()
    assert t.tolist() == [1, 2, 3, 4, 5] and T[1, 0, 3] == 20
    b.add(6.0, _pose([0, 0, 0], [6, 0, 0]))
    assert b.arrays()[0].tolist() == [2, 3, 4, 5, 6]


def test_query_many_and_thread_safety():
    b = PoseBuffer(max_gap=0.1)

    def writer(off):
        for k in range(500):
            b.add(off + k * 0.05, _pose([0, 0.001 * k, 0], [k, 0, 0]))

    th = [threading.Thread(target=writer, args=(o,)) for o in (0.0, 0.01, 0.02)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    t, _, _ = b.arrays()
    assert len(t) == 1500 and np.all(np.diff(t) > 0)
    T, kind, dt, seg = b.query_many([0.005, 100.0])
    assert kind.tolist() == [int(QueryKind.INTERP), int(QueryKind.NONE)] and np.isnan(T[1]).all()
