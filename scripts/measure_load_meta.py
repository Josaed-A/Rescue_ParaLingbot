"""Test `device="meta"` + `load_state_dict(assign=True)` as a peak-reducing
loading strategy for demo.py::load_model() — see CLAUDE.md, "Fix aplicado y
verificado" section, for why the del+gc.collect() fix does not touch the
peak, and this experiment's design rationale.

Static analysis first found ONE blocking incompatibility under the current
config (enable_3d_rope=True always, use_sdpa=True, pretrained_path=''):
`AggregatorStream.rope3d.freqs` (lingbot_map/layers/rope.py:334) is a plain
attribute (NOT nn.Parameter, NOT a registered buffer) computed with real
math in __init__. Under device="meta" it becomes a meta tensor that
`load_state_dict()` never touches (it isn't in any state_dict) and whose
later `.to(device)` in forward() (rope.py:365) produces uninitialized
memory, not the real frequencies — a silent correctness bug, not a crash.
Same pattern, smaller impact: `_resnet_mean`/`_resnet_std`
(aggregator/base.py:162), non-persistent buffers.

This script runs BOTH variants so the comparison is empirical, not just
theoretical:
  A) "naive"   — meta + assign, no repair. Confirms (via .is_meta checks)
                 that rope3d.freqs / resnet mean+std are left broken.
  B) "repaired" — same, then manually rebuild rope3d (re-running the
                 model's own _init_3d_rope() on the real device) and
                 re-register the resnet mean/std buffers. Verifies the
                 repaired values are bit-identical to a fresh, independently
                 constructed WanRotaryPosEmbed with the same args (cheap,
                 doesn't require loading the full model twice) — i.e. not
                 just "doesn't crash" but "numerically correct".

Does not modify demo.py or any lingbot_map code — everything here is
test-script-only, per instruction.
"""
import argparse
import ctypes
from ctypes import wintypes
import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root


# ---------------------------------------------------------------------------
# Windows memory instrumentation (same as measure_load_only.py / verify_load_fix.py)
# ---------------------------------------------------------------------------
class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("sullAvailExtendedVirtual", ctypes.c_uint64),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.GetCurrentProcess.argtypes = []
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCountersEx), wintypes.DWORD]
_kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
_kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]


def _proc_mem():
    c = _ProcessMemoryCountersEx()
    c.cb = ctypes.sizeof(_ProcessMemoryCountersEx)
    if not _psapi.GetProcessMemoryInfo(_kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "ws_mb": c.WorkingSetSize / 1e6,
        "peak_ws_mb": c.PeakWorkingSetSize / 1e6,
        "private_mb": c.PrivateUsage / 1e6,
        "peak_pagefile_mb": c.PeakPagefileUsage / 1e6,
    }


def _sys_mem():
    m = _MemoryStatusEx()
    m.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not _kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"sys_avail_mb": m.ullAvailPhys / 1e6, "sys_load_pct": m.dwMemoryLoad}


_T0 = time.time()


def stage(label):
    p, s = _proc_mem(), _sys_mem()
    print(
        f"[STAGE] t={time.time() - _T0:7.1f}s {label:44s} "
        f"ws={p['ws_mb']:9.1f}MB peak_ws={p['peak_ws_mb']:9.1f}MB "
        f"private={p['private_mb']:9.1f}MB peak_pagefile={p['peak_pagefile_mb']:9.1f}MB "
        f"| sys_avail={s['sys_avail_mb']:9.1f}MB sys_load={s['sys_load_pct']}%",
        flush=True,
    )


def _build_args():
    return SimpleNamespace(
        image_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        num_scale_frames=8,
        kv_cache_sliding_window=64,
        use_sdpa=True,
        camera_num_iterations=1,
    )


def _count_meta(model):
    """Count meta params/buffers PyTorch's own module system tracks, plus the
    known plain-attribute case (rope3d.freqs) it can't see."""
    n_meta_params = sum(1 for _, p in model.named_parameters() if p.is_meta)
    n_meta_bufs = sum(1 for _, b in model.named_buffers() if b.is_meta)
    rope_meta = getattr(getattr(model.aggregator, "rope3d", None), "freqs", None)
    rope_is_meta = torch.is_tensor(rope_meta) and rope_meta.is_meta
    return n_meta_params, n_meta_bufs, rope_is_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--repair", action="store_true",
                     help="After load_state_dict, manually rebuild rope3d and the "
                          "resnet mean/std buffers on the real device (variant B). "
                          "Without this flag, runs variant A (naive, left broken).")
    cli = ap.parse_args()

    assert hasattr(torch, "device"), "torch.device context manager required"
    print(f"torch version: {torch.__version__}", flush=True)

    stage("00 start (torch imported)")

    from lingbot_map.models.gct_stream import GCTStream
    args = _build_args()
    device = torch.device("cpu")

    t_total0 = time.time()

    with torch.device("meta"):
        model = GCTStream(
            img_size=args.image_size,
            patch_size=args.patch_size,
            enable_3d_rope=args.enable_3d_rope,
            max_frame_num=args.max_frame_num,
            kv_cache_sliding_window=args.kv_cache_sliding_window,
            kv_cache_scale_frames=args.num_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=args.use_sdpa,
            camera_num_iterations=args.camera_num_iterations,
        )
    stage("01 model instantiated under device='meta'")
    n_p, n_b, rope_meta = _count_meta(model)
    print(f"  meta params={n_p} meta buffers={n_b} rope3d.freqs.is_meta={rope_meta}", flush=True)

    ckpt = torch.load(cli.model_path, map_location=device, weights_only=False)
    stage("02 torch.load done (ckpt dict in RAM)")

    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    stage("03 load_state_dict(assign=True) done (ckpt still referenced)")

    n_p, n_b, rope_meta = _count_meta(model)
    print(
        f"  AFTER load_state_dict: meta params={n_p} meta buffers={n_b} "
        f"rope3d.freqs.is_meta={rope_meta}",
        flush=True,
    )
    if n_p == 0 and n_b == 0 and not rope_meta:
        print("  -> all nn.Parameter/buffer tensors materialized; rope3d.freqs OK", flush=True)
    else:
        print(
            "  -> CONFIRMED BROKEN if not repaired: some tensors are still meta "
            "(uninitialized) and load_state_dict never touched them.",
            flush=True,
        )

    del ckpt, state_dict
    gc.collect()
    stage("04 after del ckpt+state_dict / gc.collect()")

    if cli.repair:
        # Rebuild rope3d on the real device by re-running the model's own
        # constructor logic (reuses the exact same math/params, no duplication).
        model.aggregator._init_3d_rope()
        from lingbot_map.aggregator.base import _RESNET_MEAN, _RESNET_STD
        model.aggregator.register_buffer(
            "_resnet_mean", torch.FloatTensor(_RESNET_MEAN).view(1, 1, 3, 1, 1), persistent=False
        )
        model.aggregator.register_buffer(
            "_resnet_std", torch.FloatTensor(_RESNET_STD).view(1, 1, 3, 1, 1), persistent=False
        )
        stage("05 REPAIRED: rope3d + resnet mean/std rebuilt on real device")

        n_p, n_b, rope_meta = _count_meta(model)
        print(f"  AFTER repair: meta params={n_p} meta buffers={n_b} rope3d.freqs.is_meta={rope_meta}", flush=True)

        # Correctness check: freqs must be bit-identical to an independently
        # constructed WanRotaryPosEmbed with the same args (cheap — no need
        # to load the full model a second time).
        from lingbot_map.layers.rope import WanRotaryPosEmbed
        num_heads = 16
        head_dim = model.aggregator.embed_dim // num_heads
        ref = WanRotaryPosEmbed(
            attention_head_dim=head_dim,
            patch_size=(1, args.patch_size, args.patch_size),
            max_seq_len=args.max_frame_num,
        )
        identical = torch.equal(model.aggregator.rope3d.freqs, ref.freqs)
        print(f"  rope3d.freqs bit-identical to fresh reference: {identical}", flush=True)

    model = model.to(device).eval()
    load_time = time.time() - t_total0
    stage("06 model.to(device).eval() done")
    print(f"LOAD_TIME_S={load_time:.2f}", flush=True)

    print("DONE - measure_load_meta complete (no image loading, no inference, no viser).", flush=True)


if __name__ == "__main__":
    main()
