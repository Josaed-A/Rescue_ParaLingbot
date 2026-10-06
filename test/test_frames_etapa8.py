"""Etapa 8: marcos de coordenadas y conversiones CV <-> ROS.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_frames_etapa8.py -q -p no:cacheprovider
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "vivo"))

import frames as F  # noqa: E402
from ros2_bridge import R_ROS_TO_CV, ros_pose_to_c2w_cv  # noqa: E402


def _pose(rotvec, p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = p
    return T


def test_axes_are_right_handed_and_mapping_is_rep103():
    A = F.A_CV_TO_ROS
    assert abs(np.linalg.det(A) - 1) < 1e-12 and np.allclose(A @ A.T, np.eye(3))
    assert np.allclose(A @ [0, 0, 1], [1, 0, 0])       # adelante CV (z) = adelante ROS (x)
    assert np.allclose(A @ [1, 0, 0], [0, -1, 0])      # derecha CV (x) = -izquierda ROS (-y)
    assert np.allclose(A @ [0, 1, 0], [0, 0, -1])      # abajo CV (y) = -arriba ROS (-z)
    assert np.allclose(A, R_ROS_TO_CV)                  # la misma que usa stella_vslam_ros


def test_first_frame_is_identity_in_map():
    assert np.allclose(F.c2w_cv_to_map_link(np.eye(4)), np.eye(4))
    # y el marco óptico del primer frame cuelga de camera_link con la rotación fija estándar
    assert np.allclose(F.map_optical(np.eye(4)), F.T_LINK_OPTICAL)


def test_walking_forward_is_plus_x_and_turning_left_is_plus_yaw():
    fwd = F.c2w_cv_to_map_link(_pose([0, 0, 0], [0, 0, 2.0]))     # avanzar 2 por el eje óptico
    assert np.allclose(fwd[:3, 3], [2, 0, 0])
    # girar a la izquierda = la cámara gira -y en CV (y apunta abajo) -> +yaw (z arriba) en ROS
    left = F.c2w_cv_to_map_link(_pose([0, -np.radians(30), 0], [0, 0, 0]))
    yaw = Rotation.from_matrix(left[:3, :3]).as_euler("zyx", degrees=True)[0]
    assert abs(yaw - 30) < 1e-9


def test_roundtrip_and_chain_consistency():
    rng = np.random.default_rng(0)
    for _ in range(20):
        c2w = _pose(rng.normal(size=3), rng.normal(size=3))
        T = F.c2w_cv_to_map_link(c2w)
        assert np.allclose(F.map_link_to_c2w_cv(T), c2w)
        # cadena TF: map -> camera_link -> optical debe dar el óptico en el mapa
        assert np.allclose(T @ F.T_LINK_OPTICAL, F.map_optical(c2w))


def test_stella_ros_pose_matches_our_ros_convention():
    """Stella publica camera_link en stella_map con la misma conversión: si su mundo coincidiera
    con el nuestro, la pose ROS que publica y la nuestra serían iguales."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        c2w = _pose(rng.normal(size=3), rng.normal(size=3))
        T_ros = F.c2w_cv_to_map_link(c2w)                     # lo que publicaría el nodo de Stella
        q = Rotation.from_matrix(T_ros[:3, :3]).as_quat()
        assert np.allclose(ros_pose_to_c2w_cv(T_ros[:3, 3], q), c2w, atol=1e-9)


def test_sim3_conversion_preserves_the_relation():
    rng = np.random.default_rng(2)
    s, R, t = 2.5, Rotation.from_rotvec(rng.normal(size=3)).as_matrix(), rng.normal(size=3)
    p_b = rng.normal(size=(5, 3))
    p_a = (s * (R @ p_b.T)).T + t
    s2, R2, t2 = F.sim3_cv_to_ros(s, R, t)
    pa_ros, pb_ros = (F.A_CV_TO_ROS @ p_a.T).T, (F.A_CV_TO_ROS @ p_b.T).T
    assert np.allclose((s2 * (R2 @ pb_ros.T)).T + t2, pa_ros)


def test_gravity_tilt():
    level = [np.eye(4)] * 3
    assert F.gravity_tilt_deg(level) < 1e-9
    tilted = [_pose([np.radians(9), 0, 0], [0, 0, 0])] * 3    # cámara mirando 9° hacia abajo
    assert abs(F.gravity_tilt_deg(tilted) - 9) < 1e-9
