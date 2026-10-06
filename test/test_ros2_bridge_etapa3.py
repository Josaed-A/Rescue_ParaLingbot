"""Etapa 3: adaptador de Stella a TrackingEstimate y mensajes del puente ROS2 (sin red, sin nodo).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_ros2_bridge_etapa3.py -q -p no:cacheprovider
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "vivo"))

from ros2_bridge import (R_ROS_TO_CV, StellaTrackingProvider, image_msg, quat_to_mat, ros_pose_to_c2w_cv,  # noqa: E402
                         stamp_key, to_time_msg)
from tracking import TrackingSource, TrackingStatus  # noqa: E402


def _rand_pose(rng):
    import cv2
    R, _ = cv2.Rodrigues(rng.normal(size=3))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.normal(size=3)
    return T


def _mat_to_quat_xyzw(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()


def test_ros_to_cv_undoes_the_node_conversion():
    """El nodo publica T_ros = R·T_cv·R⁻¹ (publish_pose en stella_vslam_ros.cc); el adaptador debe devolver T_cv."""
    rng = np.random.default_rng(3)
    for _ in range(5):
        T_cv = _rand_pose(rng)
        T_ros = np.eye(4)
        T_ros[:3, :3] = R_ROS_TO_CV @ T_cv[:3, :3] @ R_ROS_TO_CV.T
        T_ros[:3, 3] = R_ROS_TO_CV @ T_cv[:3, 3]
        q = _mat_to_quat_xyzw(T_ros[:3, :3])
        back = ros_pose_to_c2w_cv(T_ros[:3, 3], q)
        assert np.allclose(back, T_cv, atol=1e-9)


def test_ros_axes_meaning():
    """Avanzar +x en ROS (adelante) es +z en CV (eje óptico); +z ROS (arriba) es -y CV (y mira abajo)."""
    fwd = ros_pose_to_c2w_cv((1.0, 0.0, 0.0), (0, 0, 0, 1))[:3, 3]
    up = ros_pose_to_c2w_cv((0.0, 0.0, 1.0), (0, 0, 0, 1))[:3, 3]
    assert np.allclose(fwd, [0, 0, 1]) and np.allclose(up, [0, -1, 0])


def test_quat_to_mat_matches_scipy():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(1)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    assert np.allclose(quat_to_mat(*q), Rotation.from_quat(q).as_matrix(), atol=1e-12)


def test_stamp_roundtrip_through_ros_time():
    """FrameMeta.stamp -> builtin_interfaces/Time -> float debe caer en la misma clave de asociación."""
    for st in (0.0, 0.1, 1791168058.972106, 123.4567891234):
        m = to_time_msg(st)
        back = m.sec + m.nanosec * 1e-9
        assert stamp_key(back) == stamp_key(st)


def test_image_msg_layout():
    img = np.zeros((20, 30, 3), np.uint8)
    img[3, 4] = (1, 2, 3)
    m = image_msg(img, 12.5, "cam")
    assert (m.height, m.width, m.encoding, m.step) == (20, 30, "bgr8", 90)
    assert len(m.data) == 20 * 30 * 3 and m.header.frame_id == "cam"
    assert bytes(m.data[(3 * 30 + 4) * 3:(3 * 30 + 4) * 3 + 3]) == b"\x01\x02\x03"
    assert m.header.stamp.sec == 12 and m.header.stamp.nanosec == 500_000_000


def test_stella_provider_states_and_exact_association():
    p = StellaTrackingProvider(stale_s=1.0)
    assert p.status(now=0.0) is TrackingStatus.UNKNOWN
    p.on_state("Initializing", t=10.0)
    assert p.status(now=10.5) is TrackingStatus.INITIALIZING
    p.on_state("Tracking", t=11.0)
    e = p.on_pose(11.0, (0.0, 0.0, 0.0), (0, 0, 0, 1), t=11.0)
    assert e.source is TrackingSource.STELLA and e.status is TrackingStatus.TRACKING
    assert p.estimate_at(11.0) is e
    assert p.estimate_at(11.0 + 4e-7) is e          # mismo microsegundo (ida y vuelta por ROS)
    assert p.estimate_at(11.1) is None              # frame que Stella no procesó: sin inventar
    p.on_state("Lost", t=12.0)
    assert p.status(now=12.5) is TrackingStatus.LOST
    assert p.status(now=14.0) is TrackingStatus.UNKNOWN      # sin noticias de Stella: no se sabe
    p.on_keyframes(3)
    p.on_keyframes(3)
    p.on_keyframes(5)
    s = p.summary()
    assert s["poses"] == 1 and s["keyframes"] == 5 and s["keyframe_updates"] == 2
    st, c2w, seg = p.arrays()
    assert st.shape == (1,) and c2w.shape == (1, 4, 4) and seg.tolist() == [0]


def test_stella_provider_new_segment_after_reset():
    """Si Stella vuelve a Initializing después de dar poses, se reinició: mundo nuevo."""
    p = StellaTrackingProvider()
    p.on_state("Initializing", t=0.0)
    p.on_state("Tracking", t=1.0)
    p.on_pose(1.0, (0, 0, 0), (0, 0, 0, 1))
    p.on_state("Lost", t=2.0)
    p.on_state("Initializing", t=3.0)        # reinicio
    p.on_state("Initializing", t=3.1)        # repetido: no es otro reinicio
    p.on_state("Tracking", t=4.0)
    p.on_pose(4.0, (1, 0, 0), (0, 0, 0, 1))
    _, _, seg = p.arrays()
    assert seg.tolist() == [0, 1] and p.summary()["segmentos"] == 2
    et, es = p.event_arrays()
    assert len(et) == 5 and es.tolist()[:2] == [int(TrackingStatus.INITIALIZING), int(TrackingStatus.TRACKING)]


def test_stella_estimate_near_interpolates_within_segment_only():
    """Etapa 5: entre dos poses del mismo mapa a <= 0.25 s se interpola; tras un reinicio, no."""
    p = StellaTrackingProvider()
    p.on_state("Tracking", t=0.0)
    p.on_pose(10.0, (0.0, 0.0, 0.0), (0, 0, 0, 1))
    p.on_pose(10.2, (1.0, 0.0, 0.0), (0, 0, 0, 1))          # +x ROS = +z CV
    e = p.estimate_near(10.05)
    assert e.confidence["assoc"] == "INTERP" and np.allclose(e.c2w[:3, 3], [0, 0, 0.25])
    assert p.estimate_near(10.0).confidence["assoc"] == "EXACT"
    assert p.estimate_near(10.23).confidence["assoc"] == "NEAREST"
    assert p.estimate_near(11.0) is None
    p.on_state("Lost", t=1.0)
    p.on_state("Initializing", t=2.0)                         # reinicio: mundo nuevo
    p.on_state("Tracking", t=3.0)
    p.on_pose(10.4, (50.0, 0.0, 0.0), (0, 0, 0, 1))
    assert p.estimate_near(10.3) is None                      # no se interpola entre mapas
