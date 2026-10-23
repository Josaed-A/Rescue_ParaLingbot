"""build_maps.refiltrar: el filtro se recalcula con las poses estabilizadas solo si el grafo de poses las corrigió.

    PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_refiltrar.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mapas"))
import build_maps as BM  # noqa: E402


def _stable(path, moved):
    c2w = np.tile(np.eye(4), (3, 1, 1))
    fixed = c2w.copy()
    fixed[2, 0, 3] += 0.2 if moved else 0.0
    np.savez(path, real=np.arange(3), c2w_original=c2w, c2w=fixed)
    return fixed


def test_refiltrar_runs_the_filter_only_when_the_poses_were_corrected(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(BM, "run", lambda cmd, env=None: calls.append(cmd) or 0)
    st = str(tmp_path / "s_estable.npz")
    _stable(st, moved=False)
    assert BM.refiltrar("s.npz", st, "py") == 0 and calls == []          # rechazada: sirve el filtro de la etapa 1
    fixed = _stable(st, moved=True)
    assert BM.refiltrar("s.npz", st, "py") == 0
    assert calls[-1][-4:] == ["--out", st[:-4], "--poses", st]
    np.savez(st[:-4] + "_filtro.npz", c2w_fix=fixed)                     # ya calculado con esas poses
    calls.clear()
    assert BM.refiltrar("s.npz", st, "py") == 0 and calls == []
