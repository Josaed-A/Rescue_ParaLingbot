"""Test safetensors as a peak-reducing loading strategy for
demo.py::load_model() — isolated experiment, mirrors the exact methodology
of scripts/measure_load_mmap.py (the mmap=True baseline: 13055.6MB peak,
174.5MB min free RAM, ~110.6s total). See CLAUDE.md, "Investigación:
conversión a safetensors", for the checkpoint-level verification that
preceded this (bit-exact conversion, 0 shared-storage tensors).

Unlike a naive `load_file()` + `model.load_state_dict(dict)` (which would
still materialize the full ~4.63GB state_dict as one Python object,
structurally identical to what torch.load(mmap=True) already does), this
script uses `safe_open()` and copies tensors ONE AT A TIME directly into
the model's own parameters (`param.data.copy_(f.get_tensor(name))`,
deleting each source tensor immediately after the copy) — the "avoid
unnecessary materializations" strategy requested. At any instant only ONE
transient tensor is alive beyond the model itself, not the whole ~4.63GB
dict. This is a manual, faithful reimplementation of what
`load_state_dict(strict=False)` does per-tensor under the hood
(`param.data.copy_(input_param)`), just fed from `safe_open()` instead of
a pre-built dict.

Does NOT touch demo.py, the original .pt checkpoint, or any lingbot_map
code. Does NOT combine with meta/quantization/dtype changes. No image
loading, no inference, no viser — matches the mmap experiment exactly for
a clean comparison.
"""
import argparse
import ctypes
from ctypes import wintypes
import gc
import time

import torch
from safetensors import safe_open


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="path to the .safetensors checkpoint")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=1024)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--camera_num_iterations", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cpu")
    t_total0 = time.time()
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

    # --- Streaming, per-tensor load via safe_open() — avoids ever holding the
    #     full ~4.63GB state_dict as a separate object at once. ---
    targets = dict(model.named_parameters())
    targets.update(dict(model.named_buffers()))

    f = safe_open(args.model_path, framework="pt", device="cpu")
    stage("03 safe_open() done (file header parsed, no tensors materialized yet)")

    ckpt_keys = set(f.keys())
    model_keys = set(targets.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    matched = ckpt_keys & model_keys

    copied = 0
    for name in matched:
        t = f.get_tensor(name)          # materializes ONE tensor from the mmap-backed file
        targets[name].data.copy_(t)     # same op load_state_dict does internally per-tensor
        del t                           # immediately release — never accumulates
        copied += 1
    del f  # closes the safe_open file handle

    print(f"  copied={copied} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    stage("04 streaming per-tensor load complete (all matched tensors copied)")

    # Functional sanity check — same tensors checked in the mmap experiment.
    sample_names = ["aggregator.camera_token", "aggregator.patch_embed.pos_embed"]
    for name in sample_names:
        t = targets.get(name)
        if t is None:
            print(f"  [functional check] {name}: NOT FOUND", flush=True)
            continue
        print(
            f"  [functional check] {name}: shape={tuple(t.shape)} is_meta={t.is_meta} "
            f"norm={t.float().norm().item():.4f} has_nan={torch.isnan(t).any().item()}",
            flush=True,
        )

    del targets
    gc.collect()
    stage("05 after del targets / gc.collect()")

    model = model.to(device).eval()
    load_time = time.time() - t_total0
    stage("06 model.to(device).eval() done")
    print(f"LOAD_TIME_S={load_time:.2f}", flush=True)

    print("DONE - load-only safetensors probe complete (no image loading, no inference, no viser).", flush=True)


if __name__ == "__main__":
    main()
