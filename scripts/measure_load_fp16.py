"""Test FP16 precision reduction for GCTStream.aggregator as a memory-saving
strategy — isolated experiment, does NOT touch demo.py or the production
pipeline. See CLAUDE.md, "Análisis: reducción de precisión a FP16", for the
code-level analysis behind the safe/unsafe boundary used here.

Boundary used (NOT a guess — matches the boundary already implemented by
the model's authors for GPU inference, demo.py lines 467-471):
  - `model.aggregator` (DINOv2 backbone + GCT frame_blocks/global_blocks,
    79% of params) -> cast to float16.
  - `camera_head`, `depth_head` (21% of params) -> stay float32, matching
    the explicit `.float()` + `autocast(enabled=False)` already present in
    gct_base.py's _predict_camera/_predict_depth for ALL dtypes.

Known extra risk versus the authors' GPU validation (documented in
CLAUDE.md): production always wraps this cast in `torch.amp.autocast('cuda',
dtype=dtype)`, which auto-upcasts sensitive ops (LayerNorm, softmax) even
with fp16/bf16 weights. `torch.amp.autocast` on CPU only supports
bfloat16, not float16 — so this experiment runs fp16 math WITHOUT that
safety net. This script exists to measure the real consequence, not assume
it away.

Runs TWO models sequentially (fp32 baseline, then fp16-aggregator),
freeing the first before building the second to avoid ~13GB simultaneous
peak, and compares their outputs on the SAME 3 real images for numerical
stability — not just "did it crash".
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
    print(
        f"[STAGE] t={time.time() - _T0:7.1f}s {label:44s} "
        f"ws={p['ws_mb']:9.1f}MB peak_ws={p['peak_ws_mb']:9.1f}MB "
        f"private={p['private_mb']:9.1f}MB peak_pagefile={p['peak_pagefile_mb']:9.1f}MB "
        f"| sys_avail={s['sys_avail_mb']:9.1f}MB sys_load={s['sys_load_pct']}%",
        flush=True,
    )


def param_memory_by_dtype(model):
    from collections import defaultdict
    mb_by_dtype = defaultdict(float)
    for p in model.parameters():
        mb_by_dtype[str(p.dtype)] += p.numel() * p.element_size() / 1e6
    return dict(mb_by_dtype)


def build_and_load(args, device, cast_aggregator_fp16: bool):
    from lingbot_map.models.gct_stream import GCTStream

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
    stage(f"  model instantiated (fp32 random init){' [about to cast]' if cast_aggregator_fp16 else ''}")

    if cast_aggregator_fp16:
        before = param_memory_by_dtype(model)
        model.aggregator = model.aggregator.to(dtype=torch.float16)
        after = param_memory_by_dtype(model)
        print(f"    aggregator param memory: before={before} after={after}", flush=True)
        stage("  aggregator cast to float16 (heads stay float32)")

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False, mmap=True)
    state_dict = ckpt.get("model", ckpt)
    # Standard (non-assign) load_state_dict: per-tensor `.copy_()`, which
    # auto-casts dtype (fp32 source -> fp16 destination for aggregator
    # params, fp32 -> fp32 for heads) — no special handling needed.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"    missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    del ckpt, state_dict
    gc.collect()
    stage("  checkpoint loaded + del/gc.collect()")

    model = model.to(device).eval()
    return model


def run_inference(model, images, args):
    num_scale = min(images.shape[0], args.num_scale_frames)
    with torch.no_grad():
        predictions = model.inference_streaming(
            images, num_scale_frames=num_scale, keyframe_interval=1, output_device=None,
        )
    return predictions


def compare_outputs(pred_a, pred_b, label_a, label_b):
    print(f"\n=== NUMERICAL STABILITY: {label_a} vs {label_b} ===", flush=True)
    for key in ("pose_enc", "depth"):
        a = pred_a[key].float()
        b = pred_b[key].float()
        if a.shape != b.shape:
            print(f"  {key}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}", flush=True)
            continue
        diff = (a - b).abs()
        rel = diff / (a.abs().clamp_min(1e-6))
        n_nan = torch.isnan(b).sum().item()
        n_inf = torch.isinf(b).sum().item()
        print(
            f"  {key}: max_abs_diff={diff.max().item():.6g} mean_abs_diff={diff.mean().item():.6g} "
            f"max_rel_diff={rel.max().item():.6g} mean_rel_diff={rel.mean().item():.6g} "
            f"| {label_b}: NaN={n_nan} Inf={n_inf}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image_folder", default="test_images")
    ap.add_argument("--n_images", type=int, default=3)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--max_frame_num", type=int, default=1024)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=64)
    ap.add_argument("--camera_num_iterations", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cpu")
    stage("00 start (torch imported)")

    import demo  # for load_images() only — production image preprocessing, unchanged
    stage("01 demo module imported")

    # ============================== PHASE A: FP32 baseline ==============================
    print("\n########## PHASE A: FP32 baseline (matches production dtype) ##########", flush=True)
    t0 = time.time()
    model_fp32 = build_and_load(args, device, cast_aggregator_fp16=False)
    load_time_fp32 = time.time() - t0
    stage("02 model_fp32 ready")
    print(f"LOAD_TIME_FP32_S={load_time_fp32:.2f}", flush=True)
    print(f"  param memory by dtype: {param_memory_by_dtype(model_fp32)}", flush=True)

    images, paths, _ = demo.load_images(
        image_folder=args.image_folder, first_k=args.n_images,
        image_size=args.image_size, patch_size=args.patch_size,
    )
    images = images.to(device)
    stage(f"03 loaded {images.shape[0]} real images")

    t0 = time.time()
    predictions_fp32 = run_inference(model_fp32, images, args)
    infer_time_fp32 = time.time() - t0
    stage("04 fp32 inference done")
    print(f"INFER_TIME_FP32_S={infer_time_fp32:.2f}", flush=True)
    # Detach + clone the two outputs we'll compare, then free everything else.
    kept_fp32 = {"pose_enc": predictions_fp32["pose_enc"].detach().clone(),
                 "depth": predictions_fp32["depth"].detach().clone()}

    del model_fp32, predictions_fp32, images
    gc.collect()
    stage("05 phase A freed (del model_fp32 + gc.collect())")

    # ============================== PHASE B: FP16 aggregator ==============================
    print("\n########## PHASE B: FP16 aggregator, FP32 heads ##########", flush=True)
    t0 = time.time()
    model_fp16 = build_and_load(args, device, cast_aggregator_fp16=True)
    load_time_fp16 = time.time() - t0
    stage("06 model_fp16 ready")
    print(f"LOAD_TIME_FP16_S={load_time_fp16:.2f}", flush=True)
    print(f"  param memory by dtype: {param_memory_by_dtype(model_fp16)}", flush=True)

    # Functional / weight-sanity check before running anything.
    n_nan_weights = sum(torch.isnan(p).sum().item() for p in model_fp16.parameters())
    n_inf_weights = sum(torch.isinf(p).sum().item() for p in model_fp16.parameters())
    print(f"  [weight check] NaN={n_nan_weights} Inf={n_inf_weights} across all parameters", flush=True)
    for name, p in model_fp16.named_parameters():
        if name in ("aggregator.camera_token",):
            print(f"  [dtype check] {name}: dtype={p.dtype}", flush=True)
    for name, p in model_fp16.named_parameters():
        if name.startswith("camera_head.") or name.startswith("depth_head."):
            print(f"  [dtype check] {name}: dtype={p.dtype} (should stay float32)", flush=True)
            break

    images, paths, _ = demo.load_images(
        image_folder=args.image_folder, first_k=args.n_images,
        image_size=args.image_size, patch_size=args.patch_size,
    )
    images = images.to(device)
    # NOTE: no autocast on CPU for float16 (only bfloat16 is supported there),
    # so there is no automatic dtype bridge between fp32 input and fp16
    # aggregator weights the way production gets on GPU. Cast the input
    # explicitly here (test-script only, not demo.py) to isolate whether the
    # REST of the pipeline is numerically fine once dtypes match, separately
    # from documenting that production's autocast-free assumption doesn't
    # hold as-is on CPU.
    images = images.half()
    stage(f"07 loaded {images.shape[0]} real images (again, cast to float16 for the fp16 model)")

    t0 = time.time()
    try:
        predictions_fp16 = run_inference(model_fp16, images, args)
        infer_time_fp16 = time.time() - t0
        stage("08 fp16 inference done")
        print(f"INFER_TIME_FP16_S={infer_time_fp16:.2f}", flush=True)

        compare_outputs(kept_fp32, predictions_fp16, "fp32", "fp16-aggregator")
    except Exception as e:
        print(f"\nFP16 INFERENCE FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()

    print("\nDONE - measure_load_fp16 complete.", flush=True)


if __name__ == "__main__":
    main()
