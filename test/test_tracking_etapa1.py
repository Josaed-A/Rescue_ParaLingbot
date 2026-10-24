"""Etapa 1 de la integración Stella: tiempo e identidad de cada frame, y el tracking actual
expuesto con la interfaz común. Todo en CPU, sin modelo.

    python3 -m pytest test/test_tracking_etapa1.py -q
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "vivo"))

from tracking import FrameMeta, BasicTrackingProvider, TrackingSource, TrackingStatus, c2w_from_w2c  # noqa: E402
from context_gate import ContextGate  # noqa: E402


def _frames(n, w=518, h=294, shift=12, seed=0):
    """Secuencia sintética con movimiento horizontal constante (ruido con textura, desplazado)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h, w + shift * n, 3), dtype=np.uint8)
    base = np.ascontiguousarray(base)
    import cv2
    base = cv2.GaussianBlur(base, (0, 0), 2)
    return [np.ascontiguousarray(base[:, shift * i:shift * i + w]) for i in range(n)]


def test_frame_meta_between_interpolates_stamp_and_marks_synthetic():
    a, b = FrameMeta(stamp=10.0, frame_id=4), FrameMeta(stamp=12.0, frame_id=7)
    m = FrameMeta.between(a, b, 0.25)
    assert abs(m.stamp - 10.5) < 1e-9
    assert m.frame_id == -1 and m.synthetic


def test_gate_carries_meta_and_stamps_are_monotonic():
    frames = _frames(30)
    g = ContextGate(step_px=36.0, max_skip=6)
    out = []
    for i, f in enumerate(frames):
        out += g.feed(f, FrameMeta(stamp=i / 15.0, frame_id=i))
    out += g.flush()
    assert out, "el analizador no dejó pasar ningún frame"
    for rgb, syn, meta in out:
        assert not syn
        assert meta is not None and meta.frame_id >= 0
        assert meta.motion_px is not None and meta.sharpness is not None
        # el frame elegido es realmente el que dice su meta
        assert np.array_equal(rgb, frames[meta.frame_id])
        assert abs(meta.stamp - meta.frame_id / 15.0) < 1e-9
    stamps = [m.stamp for _, _, m in out]
    assert all(b > a for a, b in zip(stamps, stamps[1:]))
    assert g.stats["sent"] == len(out)


def test_gate_synthetic_frames_get_interpolated_meta():
    # desplazamiento de ~30 px por frame con paso de 10 px: cada salto supera 1.5 pasos y el
    # analizador genera intermedios (el flujo DIS a 256 px no mide saltos mucho mayores)
    frames = _frames(6, shift=30)
    g = ContextGate(step_px=10.0, max_skip=6, synth=True)
    out = []
    for i, f in enumerate(frames):
        out += g.feed(f, FrameMeta(stamp=float(i), frame_id=i))
    out += g.flush()
    synth = [(m, k) for k, (_, syn, m) in enumerate(out) if syn]
    assert synth, "esperaba frames sintéticos"
    for m, k in synth:
        assert m.frame_id == -1 and m.synthetic
        prev_real = next(mm for _, s, mm in reversed(out[:k]) if not s)
        next_real = next(mm for _, s, mm in out[k:] if not s)
        assert prev_real.stamp < m.stamp < next_real.stamp


def test_gate_without_meta_still_works():
    g = ContextGate()
    out = g.feed(_frames(1)[0])
    assert out and out[0][2] is None


def test_basic_provider_estimate():
    p = BasicTrackingProvider()
    w2c = np.eye(4)[:3]
    conf = np.full((4, 4), 2.0, np.float32)
    e0 = p.estimate(FrameMeta(stamp=1.0, frame_id=0, motion_px=3.0, sharpness=9.0), c2w_from_w2c(w2c), conf)
    assert e0.source is TrackingSource.BASIC and e0.status is TrackingStatus.TRACKING
    assert e0.c2w.shape == (4, 4) and e0.confidence["conf_mean"] == 2.0
    assert e0.confidence["step"] is None and e0.confidence["motion_px"] == 3.0
    c2w1 = np.eye(4)
    c2w1[:3, 3] = [0.3, 0.0, 0.4]
    e1 = p.estimate(FrameMeta(stamp=1.5, frame_id=7), c2w1, conf)
    assert abs(e1.confidence["step"] - 0.5) < 1e-9
    j = e1.to_json()
    assert j["source"] == "BASIC" and j["frame_id"] == 7 and j["stamp"] == 1.5
    assert p.count == 2


def test_c2w_from_w2c_is_inverse():
    rng = np.random.default_rng(1)
    q = rng.normal(size=3)
    import cv2
    R, _ = cv2.Rodrigues(q)
    w2c = np.hstack([R, rng.normal(size=(3, 1))])
    E = np.eye(4)
    E[:3, :4] = w2c
    assert np.allclose(c2w_from_w2c(w2c) @ E, np.eye(4), atol=1e-9)


def test_frame_source_folder_stamps(tmp_path):
    import cv2
    from live_server import FrameSource
    for i in range(5):
        cv2.imwrite(str(tmp_path / f"{i:06d}.png"), np.zeros((20, 30, 3), np.uint8))
    src = FrameSource("folder", path=str(tmp_path), source_fps=10.0)
    assert src.stamp_kind == "synthetic"
    got = []
    while True:
        r = src.read()
        if r is None:
            break
        img, meta = r
        got.append(meta)
    assert [m.frame_id for m in got] == [0, 1, 2, 3, 4]
    assert np.allclose([m.stamp for m in got], [0.0, 0.1, 0.2, 0.3, 0.4])
    src.close()


def test_to_json_is_strict_json():
    """Etapa 5: el mensaje `tracking` debe ser JSON estricto (sin NaN) y aceptar textos."""
    import json
    from tracking import TrackingEstimate
    e = TrackingEstimate(stamp=1.0, frame_id=3, c2w=np.eye(4), source=TrackingSource.STELLA,
                         status=TrackingStatus.TRACKING,
                         confidence={"assoc": "NEAREST", "gap": float("nan"), "dt": float("inf"), "n": 3, "x": None})
    j = e.to_json()
    s = json.dumps(j, allow_nan=False)
    assert j["assoc"] == "NEAREST" and j["gap"] is None and j["dt"] is None and j["n"] == 3 and "NaN" not in s
