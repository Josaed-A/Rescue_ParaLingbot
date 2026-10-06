"""Momentos estáticos (StaticHold, context_gate.py): con la cámara quieta la pose no avanza aunque el
modelo derive; al reanudar no hay salto y el movimiento relativo del modelo se conserva."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "vivo"))

from context_gate import StaticHold  # noqa: E402


def T(x):
    m = np.eye(4)
    m[0, 3] = x
    return m


def test_static_frames_hold_pose_and_resume_without_jump():
    h = StaticHold(step_px=36, static_frac=0.25)          # umbral 9 px
    xs = [0.0, 0.1, 0.2] + [0.2 + 0.01 * k for k in range(1, 11)] + [0.30 + 0.1 * k for k in range(1, 4)]
    mov = [0, 40, 40] + [1.0] * 10 + [40, 40, 40]           # quieto 10 frames, pero el modelo deriva 0.1
    out = [h.update(T(x), motion_px=m) for x, m in zip(xs, mov)]
    P = np.array([o[0][0, 3] for o in out])
    quieto = [o[1] for o in out]
    assert quieto == [False, False, False] + [True] * 10 + [False] * 3
    assert np.allclose(P[3:13], 0.2)                       # no avanza durante la pausa
    assert np.allclose(np.diff(P[12:]), 0.1)               # al reanudar: mismos pasos que el modelo, sin salto
    assert h.n_static == 10


def test_without_motion_info_measures_flow_itself():
    h = StaticHold()
    rng = np.random.default_rng(0)
    import cv2
    img = cv2.GaussianBlur((rng.random((240, 320, 3)) * 255).astype(np.uint8), (0, 0), 3)   # textura seguible
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    p0, q0, _ = h.update(T(0.0), rgb=img)
    p1, q1, m1 = h.update(T(0.05), rgb=img.copy())         # misma imagen: flujo ~0 => estático
    assert not q0 and q1 and m1 < 1.0 and np.allclose(p1, p0)
    moved = np.roll(img, 20, axis=1)
    _, q2, m2 = h.update(T(0.10), rgb=moved)              # imagen corrida 20 px => movimiento
    assert not q2 and m2 > 9
