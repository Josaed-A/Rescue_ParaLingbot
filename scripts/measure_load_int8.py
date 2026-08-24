"""Test weight-only INT8 dynamic quantization for GCTStream.aggregator —
isolated experiment, does NOT touch demo.py or the production pipeline.
See CLAUDE.md, "Análisis: viabilidad de cuantización INT8 weight-only", for
the code-level analysis behind this design.

Strategy (confirmed viable by static analysis before writing this script):
`torch.ao.quantization.quantize_dynamic(model.aggregator, {nn.Linear},
dtype=torch.qint8)` — replaces every `nn.Linear` submodule (qkv, proj,
gate_proj, fc1/fc2, w12/w3 — verified as real nn.Linear instances, not
manual matmul) with a dynamically-quantized version: INT8 weights
(computed once, from the real loaded values — NOT `model.to(torch.int8)`,
which would just truncate floats into the int8 range with no scale/
zero-point and destroy the model), FP32 activations quantized on-the-fly
per forward call and dequantized back to FP32 after the matmul. No
calibration dataset needed. `patch_embed.proj` (nn.Conv2d) and everything
outside `nn.Linear` (LayerNorm, attention softmax, RoPE) stays FP32
automatically — the API doesn't touch them.

Same aggregator-only boundary as the FP16 experiment (camera_head/
depth_head stay untouched — gct_base.py forces `.float()` +
`autocast(enabled=False)` on all 4 `_predict_*` heads).

Known, already-documented constraint (not discovered here, confirmed):
quantize_dynamic needs the REAL loaded weight values to compute
quantization scale — the model must be loaded in FP32 first, THEN
converted. The loading peak cannot improve with this strategy (same
conclusion as FP16, for a different underlying reason). Measured anyway
for a direct, honest comparison against the mmap+gc baseline.

Hypothesis under test (unlike FP16): CPU int8 dynamic quantization via the
`onednn` backend is the intended use case for this exact API — inference
time is expected to be equal-or-better than FP32, not worse.
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
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic

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


def build_and_load(args, device):
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
    stage("  model instantiated (fp32 random init)")

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False, mmap=True)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"    missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    del ckpt, state_dict
    gc.collect()
    stage("  checkpoint loaded + del/gc.collect() (fp32, real weight values)")

    return model.to(device).eval()


def count_linear_layers(module):
    return sum(1 for m in module.modules() if isinstance(m, nn.Linear))


def count_quantized_linear_layers(module):
    from torch.ao.nn.quantized.dynamic import Linear as QLinear
    return sum(1 for m in module.modules() if isinstance(m, QLinear))


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
    print(f"quantized engine: {torch.backends.quantized.engine} "
          f"(supported: {torch.backends.quantized.supported_engines})", flush=True)

    import demo
    stage("01 demo module imported")

    # ============================== PHASE A: FP32 baseline ==============================
    print("\n########## PHASE A: FP32 baseline (matches production dtype) ##########", flush=True)
    t0 = time.time()
    model_fp32 = build_and_load(args, device)
    load_time_fp32 = time.time() - t0
    stage("02 model_fp32 ready")
    print(f"LOAD_TIME_FP32_S={load_time_fp32:.2f}", flush=True)
    param_mb_fp32 = sum(p.numel() * p.element_size() for p in model_fp32.parameters()) / 1e6
    print(f"  param memory (fp32): {param_mb_fp32:.1f}MB", flush=True)

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
    kept_fp32 = {"pose_enc": predictions_fp32["pose_enc"].detach().clone(),
                 "depth": predictions_fp32["depth"].detach().clone()}

    del model_fp32, predictions_fp32, images
    gc.collect()
    stage("05 phase A freed (del model_fp32 + gc.collect())")

    # ============================== PHASE B: INT8 dynamic quantization ==============================
    print("\n########## PHASE B: INT8 dynamic quantization (aggregator Linear layers) ##########", flush=True)
    t0 = time.time()
    model_int8 = build_and_load(args, device)  # same load as phase A — cannot avoid the fp32 peak here
    load_time_int8_fp32_stage = time.time() - t0
    stage("06 model loaded (still fp32, about to quantize)")
    print(f"LOAD_TIME_FP32_STAGE_S={load_time_int8_fp32_stage:.2f}", flush=True)

    n_linear = count_linear_layers(model_int8.aggregator)
    print(f"  nn.Linear layers found in aggregator: {n_linear}", flush=True)

    t0 = time.time()
    model_int8.aggregator = quantize_dynamic(model_int8.aggregator, {nn.Linear}, dtype=torch.qint8)
    quant_time = time.time() - t0
    stage("  quantize_dynamic(aggregator) done")
    print(f"QUANT_TIME_S={quant_time:.2f}", flush=True)

    n_quantized = count_quantized_linear_layers(model_int8.aggregator)
    print(f"  quantized (torch.ao.nn.quantized.dynamic.Linear) layers now in aggregator: {n_quantized}", flush=True)

    model_int8 = model_int8.eval()
    total_load_time_int8 = load_time_int8_fp32_stage + quant_time
    print(f"LOAD_TIME_INT8_TOTAL_S={total_load_time_int8:.2f}", flush=True)

    images, paths, _ = demo.load_images(
        image_folder=args.image_folder, first_k=args.n_images,
        image_size=args.image_size, patch_size=args.patch_size,
    )
    images = images.to(device)
    # NOTE: unlike FP16, weight-only dynamic quantization keeps the module's
    # public input/output dtype as float32 (activations are quantized
    # internally, on the fly, and dequantized back) — no input cast needed.
    stage(f"07 loaded {images.shape[0]} real images (no dtype cast needed for weight-only quant)")

    t0 = time.time()
    try:
        predictions_int8 = run_inference(model_int8, images, args)
        infer_time_int8 = time.time() - t0
        stage("08 int8 inference done")
        print(f"INFER_TIME_INT8_S={infer_time_int8:.2f}", flush=True)

        n_nan = torch.isnan(predictions_int8["pose_enc"]).sum().item() + torch.isnan(predictions_int8["depth"]).sum().item()
        n_inf = torch.isinf(predictions_int8["pose_enc"]).sum().item() + torch.isinf(predictions_int8["depth"]).sum().item()
        print(f"  [output check] total NaN={n_nan} Inf={n_inf} across pose_enc+depth", flush=True)

        compare_outputs(kept_fp32, predictions_int8, "fp32", "int8-aggregator")
    except Exception as e:
        print(f"\nINT8 INFERENCE FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()

    print("\nDONE - measure_load_int8 complete.", flush=True)


if __name__ == "__main__":
    main()
