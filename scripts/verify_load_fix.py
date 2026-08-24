"""Verify the del+gc.collect() fix in demo.py::load_model() against the
2026-08-23 baseline (see CLAUDE.md, "Reporte: prueba aislada de load_model()").

Unlike scripts/measure_load_only.py (a standalone reimplementation of the
load sequence, used to find WHERE the peak was), this script imports and
calls the REAL demo.load_model(args, device) — so it measures the actual
pipeline code, including the fix, not a parallel simulation. No image
loading, no inference, no viser.

Reuses the same ctypes-based Windows memory instrumentation as
measure_load_only.py (GetProcessMemoryInfo / GlobalMemoryStatusEx — OS-
tracked peaks, not polling estimates). Run measure_ram.ps1 in parallel as
before for the fine-grained minimum-free-RAM reading the discrete stage
snapshots can't catch on their own.
"""
import argparse
import ctypes
from ctypes import wintypes
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import demo`


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
_psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(_ProcessMemoryCountersEx),
    wintypes.DWORD,
]

_kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
_kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]


def _proc_mem():
    c = _ProcessMemoryCountersEx()
    c.cb = ctypes.sizeof(_ProcessMemoryCountersEx)
    ok = _psapi.GetProcessMemoryInfo(_kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    if not ok:
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
    ok = _kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "sys_avail_mb": m.ullAvailPhys / 1e6,
        "sys_total_mb": m.ullTotalPhys / 1e6,
        "sys_load_pct": m.dwMemoryLoad,
    }


_T0 = time.time()


def stage(label):
    p, s = _proc_mem(), _sys_mem()
    print(
        f"[STAGE] t={time.time() - _T0:7.1f}s {label:40s} "
        f"ws={p['ws_mb']:9.1f}MB peak_ws={p['peak_ws_mb']:9.1f}MB "
        f"private={p['private_mb']:9.1f}MB peak_pagefile={p['peak_pagefile_mb']:9.1f}MB "
        f"| sys_avail={s['sys_avail_mb']:9.1f}MB sys_load={s['sys_load_pct']}%",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    cli = ap.parse_args()

    stage("00 start (torch imported)")

    import demo  # real pipeline module at repo root (does not run main() on import)
    stage("01 demo module imported")

    args = SimpleNamespace(
        model_path=cli.model_path,
        image_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        num_scale_frames=8,
        kv_cache_sliding_window=64,
        use_sdpa=True,
        camera_num_iterations=1,
        mode="streaming",
    )
    device = torch.device("cpu")

    t0 = time.time()
    model = demo.load_model(args, device)
    load_time = time.time() - t0
    stage(f"02 demo.load_model() returned")
    print(f"LOAD_TIME_S={load_time:.2f}", flush=True)

    print("DONE - verify_load_fix complete (real demo.load_model(), no image loading, no inference, no viser).", flush=True)


if __name__ == "__main__":
    main()
