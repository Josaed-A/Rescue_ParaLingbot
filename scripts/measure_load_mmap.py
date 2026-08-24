"""Test torch.load(..., mmap=True) as a peak-reducing loading strategy for
demo.py::load_model(). See CLAUDE.md, "Experimento: torch.load(mmap=True)",
for compatibility checks (torch 2.13.0 has the mmap param; the checkpoint is
zip-based, a lightweight mmap=True probe completed in 0.6s vs ~70-90s
without it) and this experiment's rationale.

This is an EXACT clone of scripts/measure_load_only.py (the script used for
the original 16.1GB-peak baseline) with the ONE change requested: the
torch.load call gets `mmap=True`. Everything else — model construction,
args, stage sequence, del+gc.collect step at the end — is identical, so the
comparison against the documented baseline is apples-to-apples and isolates
mmap's effect only. Does NOT touch demo.py, meta-device, safetensors, or
quantization — mmap alone.

No image loading, no inference, no viser.
"""
import argparse
import ctypes
from ctypes import wintypes
import gc
import time

import torch


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
        f"[STAGE] t={time.time() - _T0:7.1f}s {label:34s} "
        f"ws={p['ws_mb']:9.1f}MB peak_ws={p['peak_ws_mb']:9.1f}MB "
        f"private={p['private_mb']:9.1f}MB peak_pagefile={p['peak_pagefile_mb']:9.1f}MB "
        f"| sys_avail={s['sys_avail_mb']:9.1f}MB sys_load={s['sys_load_pct']}%",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=1024)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--camera_num_iterations", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cpu")
    stage("00 start (torch imported)")

    from lingbot_map.models.gct_stream import GCTStream
    stage("01 GCTStream class imported")

    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=args.camera_num_iterations,
    )
    stage("02 model instantiated (random init)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  param count: {n_params:,} ({n_params * 4 / 1e6:.1f}MB @ float32)", flush=True)

    # ONLY CHANGE vs. the original baseline (measure_load_only.py): mmap=True.
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False, mmap=True)
    stage("03 torch.load(mmap=True) done")

    state_dict = ckpt.get("model", ckpt)
    stage("04 state_dict extracted from ckpt")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    stage("05 load_state_dict done (ckpt still referenced)")

    # Functional sanity check: sample tensors must be real, finite, non-zero
    # (not meta, not uninitialized garbage) — cheap fingerprint, no need to
    # reload without mmap to compare bit-for-bit.
    sample_names = ["aggregator.camera_token", "aggregator.patch_embed.pos_embed"]
    all_named = dict(model.named_parameters())
    all_named.update(dict(model.named_buffers()))
    for name in sample_names:
        t = all_named.get(name)
        if t is None:
            print(f"  [functional check] {name}: NOT FOUND", flush=True)
            continue
        print(
            f"  [functional check] {name}: shape={tuple(t.shape)} is_meta={t.is_meta} "
            f"norm={t.float().norm().item():.4f} has_nan={torch.isnan(t).any().item()}",
            flush=True,
        )

    del ckpt, state_dict
    gc.collect()
    stage("06 after del ckpt+state_dict / gc.collect()")

    model = model.to(device).eval()
    stage("07 model.to(device).eval() done")

    print("DONE - load-only mmap probe complete (no image loading, no inference, no viser).", flush=True)


if __name__ == "__main__":
    main()
