"""Run src/upstream/demo_render/batch_demo.py's renderer on this machine's Kaolin build.

Two problems stop the upstream renderer here, neither of them in the repo:

1. Kaolin 0.18.0 compiled from source for CUDA 13 (see the bitácora, 2026-09-10)
   exposes `kaolin.ops.spc.points_to_morton` / `morton_to_points` in Python, but
   its compiled `_C.ops.spc` has no `points_to_morton_cuda` / `morton_to_points_cuda`,
   so building the octree raises AttributeError. Both are pure bit interleaving,
   reimplemented here in torch and patched in before the renderer starts. The
   bit order matches Kaolin's own docstring example ([0,0,1]->1, [0,0,2]->8,
   [0,1,0]->2): z is the least significant axis, then y, then x.
2. src/upstream/demo_render/demo.py::load_model loads the checkpoint straight onto the GPU and
   then moves the model there too (~9 GB, does not fit in 8 GB). So this wrapper
   is meant for --load_predictions, rendering predictions computed by
   src/captura/process_and_view.py (see src/mapas/npz_for_render.py).

Usage (same flags as batch_demo.py):
  python3 src/mapas/render_route.py --load_predictions in.npz --output_folder out/
"""
import os
import runpy
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _spread_bits(v):
    """Insert two zero bits after each of the low 21 bits of v (int64)."""
    v = v.to(torch.int64) & 0x1FFFFF
    v = (v | (v << 32)) & 0x1F00000000FFFF
    v = (v | (v << 16)) & 0x1F0000FF0000FF
    v = (v | (v << 8)) & 0x100F00F00F00F00F
    v = (v | (v << 4)) & 0x10C30C30C30C30C3
    v = (v | (v << 2)) & 0x1249249249249249
    return v


def _compact_bits(v):
    """Inverse of _spread_bits."""
    v = v.to(torch.int64) & 0x1249249249249249
    v = (v | (v >> 2)) & 0x10C30C30C30C30C3
    v = (v | (v >> 4)) & 0x100F00F00F00F00F
    v = (v | (v >> 8)) & 0x1F0000FF0000FF
    v = (v | (v >> 16)) & 0x1F00000000FFFF
    v = (v | (v >> 32)) & 0x1FFFFF
    return v


def points_to_morton(points):
    shape = list(points.shape)[:-1]
    p = points.reshape(-1, 3).to(torch.int64)
    m = (_spread_bits(p[:, 0]) << 2) | (_spread_bits(p[:, 1]) << 1) | _spread_bits(p[:, 2])
    return m.reshape(shape)


def morton_to_points(morton):
    m = morton.reshape(-1).to(torch.int64)
    p = torch.stack([_compact_bits(m >> 2), _compact_bits(m >> 1), _compact_bits(m)], dim=-1)
    return p.to(torch.int16).reshape(list(morton.shape) + [3])


def patch_kaolin():
    import kaolin.ops.spc as spc
    ok = True
    try:  # only patch what the build is actually missing
        import kaolin._C as _C
        ok = hasattr(_C.ops.spc, "points_to_morton_cuda")
    except Exception:
        ok = False
    if ok:
        return False
    spc.points_to_morton = points_to_morton
    spc.morton_to_points = morton_to_points
    for mod in ("kaolin.ops.conversions.pointcloud", "kaolin.rep.spc"):
        try:
            __import__(mod)
            m = sys.modules[mod]
            if hasattr(m, "points_to_morton"):
                m.points_to_morton = points_to_morton
            if hasattr(m, "morton_to_points"):
                m.morton_to_points = morton_to_points
        except Exception:
            pass
    return True


def self_test():
    pts = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 0]], dtype=torch.int16)
    m = points_to_morton(pts)
    assert m.tolist() == [0, 1, 8, 9, 2], m.tolist()
    assert morton_to_points(m).tolist() == pts.tolist()


if __name__ == "__main__":
    self_test()
    print("parche morton:", "aplicado" if patch_kaolin() else "no hacía falta", flush=True)
    # batch_demo.py does `from demo import ...`, meaning src/upstream/demo_render/demo.py (not the
    # repo-root demo.py): running it by path skips the implicit sys.path[0], so set it.
    sys.path.insert(0, os.path.join(ROOT, "src/upstream/demo_render"))
    sys.argv = ["batch_demo.py"] + sys.argv[1:]
    runpy.run_path(os.path.join(ROOT, "src/upstream/demo_render", "batch_demo.py"), run_name="__main__")
