"""Etapa 6: selector BASIC / STELLA / HYBRID con casos de solución conocida.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_hybrid_etapa6.py -q -p no:cacheprovider
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "vivo"))

from hybrid_tracking import HybridParams, ReferenceTracker, TrackingMode  # noqa: E402
from tracking import TrackingStatus  # noqa: E402

TRK, LOST = TrackingStatus.TRACKING, TrackingStatus.LOST


def _truth(n=80, dt=0.5):
    """Trayectoria verdadera: avanza en z girando en yaw (cámara OpenCV, c2w)."""
    t = np.arange(n) * dt
    T = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        T[i, :3, :3] = Rotation.from_rotvec([0, 0.03 * i, 0]).as_matrix()
        T[i, :3, 3] = [np.sin(0.03 * i) * 2, 0.0, 0.3 * i]
    return t, T


def _drift_rotation(T, deg_per_step):
    """BASIC con deriva de rotación acumulada: cada paso relativo gira deg_per_step de más."""
    out = T.copy()
    for i in range(1, len(T)):
        d = np.linalg.inv(T[i - 1]) @ T[i]
        d[:3, :3] = Rotation.from_rotvec([0, np.radians(deg_per_step), 0]).as_matrix() @ d[:3, :3]
        out[i] = out[i - 1] @ d
    return out


def _stella_world(T, s, rotvec, tr):
    """La misma trayectoria vista por Stella: otro mundo y otra escala."""
    R = Rotation.from_rotvec(rotvec).as_matrix()
    out = T.copy()
    out[:, :3, :3] = R @ T[:, :3, :3]
    out[:, :3, 3] = (s * (R @ T[:, :3, 3].T)).T + tr
    return out


def _run(mode, t, Tb, Ts, status, seg=None, params=None):
    tr = ReferenceTracker(mode, params)
    seg = np.zeros(len(t), int) if seg is None else seg
    out = np.array([tr.update(t[i], Tb[i], None if Ts[i] is None else Ts[i], seg[i], status[i]).c2w
                    for i in range(len(t))])
    return out, tr.summary()


def _ate(A, B):
    return float(np.linalg.norm(A[:, :3, 3] - B[:, :3, 3], axis=1).mean())


def test_basic_mode_is_exactly_basic():
    t, T = _truth()
    Tb = _drift_rotation(T, 1.0)
    Ts = list(_stella_world(T, 0.3, [0.1, 0.5, 0], [1, 2, 3]))
    out, sm = _run(TrackingMode.BASIC, t, Tb, Ts, [TRK] * len(t))
    assert np.array_equal(out, Tb) and sm["usado"] == {"BASIC": len(t), "STELLA": 0}


def test_hybrid_follows_good_stella_and_beats_drifting_basic():
    t, T = _truth()
    Tb = _drift_rotation(T, 1.5)                               # BASIC deriva en rotación
    Ts = list(_stella_world(T, 0.3, [0.1, 0.5, 0], [1, 2, 3])) # Stella sin deriva, otro mundo y escala
    out, sm = _run(TrackingMode.HYBRID, t, Tb, Ts, [TRK] * len(t))
    assert sm["fraccion_stella"] > 0.9
    assert _ate(out, T) < 0.2 * _ate(Tb, T)
    # sin saltos: ningún paso de la referencia es más largo que 2 veces el verdadero
    steps = np.linalg.norm(np.diff(out[:, :3, 3], axis=0), axis=1)
    true_steps = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)
    assert np.all(steps < 2 * true_steps.max())


def test_hybrid_falls_back_to_basic_when_stella_lost_or_missing():
    t, T = _truth()
    Tb = T.copy()
    Ts = list(_stella_world(T, 0.3, [0, 0, 0], [0, 0, 0]))
    status = [TRK] * len(t)
    for i in range(30, 45):
        status[i] = LOST
        Ts[i] = None
    out, sm = _run(TrackingMode.HYBRID, t, Tb, Ts, status)
    assert sm["usado"]["BASIC"] >= 15 and "stella_lost" in sm["motivos_basic"]
    assert _ate(out, T) < 1e-6                                 # con BASIC perfecto, nada se rompe


def test_hybrid_rejects_scale_jump_and_does_not_propagate_stella_scale_drift():
    t, T = _truth()
    Ts = _stella_world(T, 0.3, [0, 0, 0], [0, 0, 0])
    Ts_bad = Ts.copy()
    Ts_bad[50:, :3, 3] += np.array([0, 0, 5.0])                # salto grande de Stella en el frame 50
    out, sm = _run(TrackingMode.HYBRID, t, T, list(Ts_bad), [TRK] * len(t))
    assert sm["motivos_basic"].get("salto_de_escala", 0) >= 1
    assert _ate(out, T) < 0.05
    # deriva lenta de escala de Stella (como la etapa 4 sin CPU): la escala la fija BASIC
    Ts_drift = Ts.copy()
    for i in range(1, len(t)):
        k = 1.0 + 0.04 * i                                     # Stella "encoge" el mundo de a poco
        Ts_drift[i, :3, 3] = Ts_drift[i - 1, :3, 3] + (Ts[i, :3, 3] - Ts[i - 1, :3, 3]) / k
    out2, _ = _run(TrackingMode.HYBRID, t, T, list(Ts_drift), [TRK] * len(t))
    assert _ate(out2, T) < 0.1 * np.linalg.norm(T[-1, :3, 3] - T[0, :3, 3])


def test_no_steps_across_stella_maps():
    t, T = _truth()
    Ts = _stella_world(T, 0.3, [0, 0, 0], [0, 0, 0])
    Ts2 = Ts.copy()
    Ts2[40:] = _stella_world(T[40:], 0.9, [0, 1.0, 0], [9, 9, 9])   # reinicio: otro mapa
    seg = np.r_[np.zeros(40, int), np.ones(len(t) - 40, int)]
    out, sm = _run(TrackingMode.HYBRID, t, T, list(Ts2), [TRK] * len(t), seg=seg)
    assert sm["motivos_basic"].get("sin_paso_previo_mismo_mapa", 0) >= 1
    assert _ate(out, T) < 0.05


def test_stella_mode_uses_absolute_stella_and_relocalization_corrects():
    t, T = _truth()
    Tb = _drift_rotation(T, 1.5)
    Ts = list(_stella_world(T, 0.3, [0.1, 0.5, 0], [1, 2, 3]))
    status = [TRK] * len(t)
    for i in range(30, 50):                                    # Stella perdida: avanza con BASIC (deriva)
        status[i] = LOST
        Ts[i] = None
    out, sm = _run(TrackingMode.STELLA, t, Tb, Ts, status)
    e = np.linalg.norm(out[:, :3, 3] - T[:, :3, 3], axis=1)
    eb = np.linalg.norm(Tb[:, :3, 3] - T[:, :3, 3], axis=1)
    # el ancla hereda el error que BASIC ya tenía al fijarse (aquí ~0.3), pero después no acumula la
    # deriva de BASIC: en el frame 25 queda muy por debajo de BASIC
    assert e[25] < 0.3 * eb[25]
    assert e[49] > e[25]                                       # durante la pérdida avanza con BASIC (deriva)
    assert e[60] < 0.5 * e[49]                                 # al volver al mismo mapa, Stella corrige


def test_starved_stella_is_not_used():
    """Stella con pocas poses por segundo (sin CPU) no se usa aunque diga TRACKING."""
    t, T = _truth()
    Ts = list(_stella_world(T, 0.3, [0, 0, 0], [0, 0, 0]))
    tr = ReferenceTracker(TrackingMode.HYBRID, HybridParams(min_pose_rate_hz=10.0))
    for i in range(len(t)):
        tr.update(t[i], T[i], Ts[i], 0, TRK, stella_rate_hz=4.0)
    sm = tr.summary()
    assert sm["usado"]["STELLA"] == 0 and sm["motivos_basic"].get("stella_famelica", 0) > 70
    tr2 = ReferenceTracker(TrackingMode.HYBRID, HybridParams(min_pose_rate_hz=10.0))
    for i in range(len(t)):
        tr2.update(t[i], T[i], Ts[i], 0, TRK, stella_rate_hz=18.0)
    assert tr2.summary()["fraccion_stella"] > 0.9


def test_stella_window_scale_follows_scale_drift_without_jumps():
    """Etapa 9: con stella_scale='window', el modo STELLA sigue una deriva lenta de escala de Stella
    (la escala la fija BASIC por ventana) y no salta al re-escalar; con 'anchor' la arrastra."""
    import numpy as np
    from hybrid_tracking import HybridParams, ReferenceTracker, TrackingMode
    from tracking import TrackingStatus

    def run(scale_mode):
        tr = ReferenceTracker(TrackingMode.STELLA, HybridParams(stella_scale=scale_mode, min_pose_rate_hz=0))
        out, ps = [], np.zeros(3)
        for i in range(200):
            t = i * 0.1
            Tb = np.eye(4); Tb[0, 3] = 0.1 * i                   # BASIC: recta a 0.1 por frame
            k = 1.0 + 0.004 * i                                  # Stella: su escala deriva (1 -> 1.8)
            ps = ps + np.array([0.1 / k, 0, 0]) if i else ps
            Ts = np.eye(4); Ts[:3, 3] = ps
            out.append(tr.update(t, Tb, Ts, 0, TrackingStatus.TRACKING).c2w[:3, 3])
        return np.array(out)
    a, w = run("anchor"), run("window")
    target = 0.1 * 199
    assert abs(w[-1, 0] - target) < 0.4 * abs(a[-1, 0] - target)   # la mediana de 6 s retrasa algo
    assert np.max(np.linalg.norm(np.diff(w, axis=0), axis=1)) < 0.2   # sin saltos
