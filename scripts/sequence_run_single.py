"""Single run of the long-sequence memory-stability campaign — one fresh
process, one frame count, matching EXACTLY the current baseline
(FP32 + mmap=True + del/gc.collect(), demo.py unmodified, no new
optimizations). See CLAUDE.md, "Campaña de secuencia larga", for the
campaign design and results.

Calls demo.load_model() and demo.load_images() UNCHANGED (real production
code, same as every prior verification script) and
model.inference_streaming(...) with the exact same arguments demo.py's
main() would use for a sequence of this length (num_scale_frames=8,
keyframe_interval=1 — matches demo.py's own auto-selection logic for
sequences <=320 frames, which covers every tier in this campaign; not a
parameter change, just replicating what demo.py already does by default
for this length range).

Per-frame memory snapshots are NOT taken inside inference_streaming() —
doing so would require wrapping/reimplementing its internal per-frame loop,
risking an unintended behavior change to the exact code path being
measured. Instead, per the "o intervalo fijo" allowance, temporal
resolution comes from the external high-frequency monitor
(scripts/measure_ram_safety.ps1, 1s samples) run in parallel by the
orchestrator — this script only emits stage snapshots at fixed boundaries
(start, after load, after inference) using the OS-tracked peak counters
(peak_ws/peak_pagefile), which are valid over the whole process lifetime
regardless of when the true peak occurred.

Writes one JSON result file with all requested metrics. On ANY failure
(exception, OOM, etc.) still writes a result record with success=false and
the error — failed runs must be recorded, not silently dropped.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
    snap = {"t_s": time.time() - _T0, "label": label, **p, **s}
    print(f"[STAGE] t={snap['t_s']:7.1f}s {label:30s} "
          f"ws={p['ws_mb']:9.1f}MB peak_ws={p['peak_ws_mb']:9.1f}MB "
          f"private={p['private_mb']:9.1f}MB peak_pagefile={p['peak_pagefile_mb']:9.1f}MB "
          f"| sys_avail={s['sys_avail_mb']:9.1f}MB", flush=True)
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--sequence_dir", required=True)
    ap.add_argument("--n_frames", type=int, required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=1024)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--camera_num_iterations", type=int, default=1)
    args = ap.parse_args()

    result = {
        "run_id": args.run_id,
        "n_frames": args.n_frames,
        "success": False,
        "error": None,
        "stages": [],
    }
    device = torch.device("cpu")

    try:
        result["stages"].append(stage("00 start"))

        import demo  # real production module, unchanged

        SimpleArgs = type("A", (), {})()
        SimpleArgs.model_path = args.model_path
        SimpleArgs.image_size = args.image_size
        SimpleArgs.patch_size = args.patch_size
        SimpleArgs.enable_3d_rope = True
        SimpleArgs.max_frame_num = args.max_frame_num
        SimpleArgs.num_scale_frames = args.num_scale_frames
        SimpleArgs.kv_cache_sliding_window = args.kv_cache_sliding_window
        SimpleArgs.use_sdpa = True
        SimpleArgs.camera_num_iterations = args.camera_num_iterations
        SimpleArgs.mode = "streaming"

        t0 = time.time()
        model = demo.load_model(SimpleArgs, device)  # UNCHANGED real load_model()
        load_time = time.time() - t0
        result["load_time_s"] = load_time
        after_load = stage("01 model loaded")
        result["stages"].append(after_load)
        result["private_after_load_mb"] = after_load["private_mb"]
        result["ws_after_load_mb"] = after_load["ws_mb"]
        result["sys_avail_after_load_mb"] = after_load["sys_avail_mb"]

        images, paths, _ = demo.load_images(
            image_folder=args.sequence_dir, first_k=args.n_frames,
            image_size=args.image_size, patch_size=args.patch_size,
        )
        images = images.to(device)
        result["n_images_loaded"] = images.shape[0]
        result["stages"].append(stage("02 images loaded"))

        num_scale = min(images.shape[0], args.num_scale_frames)
        t0 = time.time()
        with torch.no_grad():
            predictions = model.inference_streaming(
                images, num_scale_frames=num_scale, keyframe_interval=1, output_device=None,
            )
        infer_time = time.time() - t0
        result["infer_time_s"] = infer_time
        result["time_per_frame_s"] = infer_time / max(1, args.n_frames)
        final = stage("03 inference done")
        result["stages"].append(final)
        result["private_final_mb"] = final["private_mb"]
        result["ws_final_mb"] = final["ws_mb"]
        result["sys_avail_final_mb"] = final["sys_avail_mb"]
        result["peak_ws_mb"] = final["peak_ws_mb"]
        result["peak_pagefile_mb"] = final["peak_pagefile_mb"]
        result["predictions_keys"] = list(predictions.keys())

        n_nan = sum(torch.isnan(v).sum().item() for v in predictions.values() if torch.is_tensor(v))
        n_inf = sum(torch.isinf(v).sum().item() for v in predictions.values() if torch.is_tensor(v))
        result["output_nan_count"] = n_nan
        result["output_inf_count"] = n_inf

        result["success"] = True
        print("DONE - sequence_run_single complete.", flush=True)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        print(f"RUN FAILED: {result['error']}", flush=True)
        print(result["traceback"], flush=True)

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
