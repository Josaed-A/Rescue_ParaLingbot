"""Diagnostic-only memory audit of the loaded GCTStream model.

Reuses demo.py's REAL, current (mmap=True + del/gc.collect already applied)
load_model()/load_images() functions unchanged — this script does not alter
loading behavior, dtype, quantization, or architecture in any way. It only
measures and reports.

Produces, per top-level component (aggregator, camera_head, depth_head,
point_head, local_point_head) and per aggregator sub-component (patch_embed
i.e. the DINOv2 backbone, frame_blocks, global_blocks, special tokens,
rope3d, resnet mean/std buffers):
  - parameter count, dtype(s), parameter memory (bytes, from numel*elemsize)
  - buffer memory (persistent + non-persistent, via module.buffers())
  - "other tensors" not caught by the nn.Module system (rope3d.freqs, a
    plain attribute — see CLAUDE.md "Experimento: device=meta" for why this
    one is special)

Cross-checks the sum of all live tensors reachable via gc.get_objects()
(deduplicated by storage identity) against both the theoretical parameter
count and the actual measured process memory (ctypes/Windows), to quantify
how much of the post-load footprint is genuinely PyTorch tensors vs.
process/allocator overhead not attributable to any specific component.

Finally loads a handful of real images (scripts/../test_images) and runs ONE
small streaming inference call (not the full 10-frame demo) to measure the
memory delta attributable to inference (activations, KV cache) separately
from the permanent, resident model weights.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `import demo`


# ---------------------------------------------------------------------------
# Windows memory instrumentation (same as the other scripts/*.py probes)
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
        f"[STAGE] t={time.time() - _T0:7.1f}s {label:40s} "
        f"ws={p['ws_mb']:9.1f}MB peak_ws={p['peak_ws_mb']:9.1f}MB "
        f"private={p['private_mb']:9.1f}MB peak_pagefile={p['peak_pagefile_mb']:9.1f}MB "
        f"| sys_avail={s['sys_avail_mb']:9.1f}MB sys_load={s['sys_load_pct']}%",
        flush=True,
    )
    return p, s


# ---------------------------------------------------------------------------
# Component memory summarization
# ---------------------------------------------------------------------------
def summarize_module(module):
    """Sum parameter/buffer memory for a module (recursive). Returns a dict."""
    param_bytes = 0
    n_params = 0
    dtypes = set()
    for p in module.parameters(recurse=True):
        param_bytes += p.numel() * p.element_size()
        n_params += p.numel()
        dtypes.add(str(p.dtype))

    buffer_bytes = 0
    n_buffers = 0
    for b in module.buffers(recurse=True):
        buffer_bytes += b.numel() * b.element_size()
        n_buffers += b.numel()

    return {
        "n_params": n_params,
        "param_mb": param_bytes / 1e6,
        "dtypes": sorted(dtypes),
        "n_buffer_elems": n_buffers,
        "buffer_mb": buffer_bytes / 1e6,
    }


def print_component_table(model):
    print("\n=== COMPONENT MEMORY BREAKDOWN ===", flush=True)
    header = f"{'component':40s} {'params':>14s} {'dtype':>10s} {'param_MB':>10s} {'buffer_MB':>10s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    grand_param_mb = 0.0
    grand_buffer_mb = 0.0

    for top_name, top_mod in model.named_children():
        if top_mod is None:
            print(f"{top_name:40s} {'(None / not built)':>14s}", flush=True)
            continue
        s = summarize_module(top_mod)
        print(
            f"{top_name:40s} {s['n_params']:14,d} {','.join(s['dtypes']) or '-':>10s} "
            f"{s['param_mb']:10.1f} {s['buffer_mb']:10.1f}",
            flush=True,
        )
        grand_param_mb += s["param_mb"]
        grand_buffer_mb += s["buffer_mb"]

        # One level deeper for aggregator (patch_embed=DINOv2, frame_blocks, global_blocks, ...)
        if top_name == "aggregator":
            for sub_name, sub_mod in top_mod.named_children():
                ss = summarize_module(sub_mod)
                print(
                    f"    aggregator.{sub_name:28s} {ss['n_params']:14,d} "
                    f"{','.join(ss['dtypes']) or '-':>10s} {ss['param_mb']:10.1f} {ss['buffer_mb']:10.1f}",
                    flush=True,
                )
            # Direct (non-child) parameters/buffers of the aggregator itself
            # (special tokens: camera_token, register_token, scale_token; resnet mean/std)
            direct_params = list(top_mod.parameters(recurse=False))
            direct_bufs = list(top_mod.buffers(recurse=False))
            dp_mb = sum(p.numel() * p.element_size() for p in direct_params) / 1e6
            db_mb = sum(b.numel() * b.element_size() for b in direct_bufs) / 1e6
            print(
                f"    aggregator.<direct params/bufs>{'':10s} "
                f"{sum(p.numel() for p in direct_params) + sum(b.numel() for b in direct_bufs):14,d} "
                f"{'-':>10s} {dp_mb:10.1f} {db_mb:10.1f}",
                flush=True,
            )
            # The known non-buffer plain attribute: rope3d.freqs (see CLAUDE.md)
            rope3d = getattr(top_mod, "rope3d", None)
            freqs = getattr(rope3d, "freqs", None) if rope3d is not None else None
            if torch.is_tensor(freqs):
                freqs_mb = freqs.numel() * freqs.element_size() / 1e6
                print(
                    f"    aggregator.rope3d.freqs (plain attr, NOT in state_dict) "
                    f"{freqs.numel():14,d} {str(freqs.dtype):>10s} {freqs_mb:10.4f} {'0.0':>10s}",
                    flush=True,
                )
                grand_param_mb += freqs_mb  # count it as "other tensor", added to grand total below

    print("-" * len(header), flush=True)
    print(f"{'TOTAL (params+buffers, enumerated)':40s} {'':>14s} {'':>10s} {grand_param_mb:10.1f} {grand_buffer_mb:10.1f}", flush=True)
    return grand_param_mb, grand_buffer_mb


def sum_live_tensor_bytes():
    """Enumerate ALL live torch tensors via gc, dedup by storage identity.

    Cross-check against the module-tree enumeration above: if this total is
    much higher, something is alive outside model.parameters()/buffers()
    (e.g. a residual reference to the checkpoint dict, or PyTorch-internal
    caches). If it's close to the module-tree total, the model IS the
    memory — no hidden duplication.
    """
    seen_storage_ids = set()
    total_bytes = 0
    n_tensors = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            try:
                st = obj.untyped_storage()
            except Exception:
                continue
            sid = st.data_ptr()
            if sid in seen_storage_ids or sid == 0:
                continue
            seen_storage_ids.add(sid)
            total_bytes += st.nbytes()
            n_tensors += 1
    return n_tensors, total_bytes / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image_folder", default="test_images")
    ap.add_argument("--n_infer_images", type=int, default=3)
    args_cli = ap.parse_args()

    stage("00 start (torch imported)")

    import demo  # real production module: load_model(), load_images() unchanged
    stage("01 demo module imported")

    n_before, mb_before = sum_live_tensor_bytes()
    print(f"  [gc cross-check] live tensors before model load: n={n_before} total={mb_before:.1f}MB", flush=True)

    args = SimpleNamespace(
        model_path=args_cli.model_path,
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

    model = demo.load_model(args, device)  # REAL load_model(): mmap=True + del/gc.collect already inside
    proc_after_load, _ = stage("02 demo.load_model() returned (mmap+gc active)")

    n_after, mb_after = sum_live_tensor_bytes()
    print(f"  [gc cross-check] live tensors after model load: n={n_after} total={mb_after:.1f}MB", flush=True)

    grand_param_mb, grand_buffer_mb = print_component_table(model)

    n_params_total = sum(p.numel() for p in model.parameters())
    theoretical_mb = n_params_total * 4 / 1e6  # all params are float32 on CPU

    print("\n=== RECONCILIATION ===", flush=True)
    print(f"  Total params (numel):                {n_params_total:,}", flush=True)
    print(f"  Theoretical param memory (float32):   {theoretical_mb:9.1f} MB", flush=True)
    print(f"  Enumerated param+buffer memory:       {grand_param_mb + grand_buffer_mb:9.1f} MB", flush=True)
    print(f"  gc-based live-tensor total (dedup):   {mb_after:9.1f} MB", flush=True)
    print(f"  Process private memory (measured):    {proc_after_load['private_mb']:9.1f} MB", flush=True)
    print(
        f"  UNACCOUNTED (process private - gc live-tensor total): "
        f"{proc_after_load['private_mb'] - mb_after:9.1f} MB  "
        f"(process/allocator overhead not attributable to any live PyTorch tensor)",
        flush=True,
    )

    model = model.eval()
    stage("03 model.eval() done")

    # --- Real images, minimal streaming inference, to isolate inference-time delta ---
    images, paths, _ = demo.load_images(
        image_folder=args_cli.image_folder,
        first_k=args_cli.n_infer_images,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    images = images.to(device)
    stage(f"04 loaded {images.shape[0]} real images for inference probe")

    n_before_inf, mb_before_inf = sum_live_tensor_bytes()
    proc_before_inf = _proc_mem()
    print(
        f"  [before inference] live tensors n={n_before_inf} total={mb_before_inf:.1f}MB "
        f"| process private={proc_before_inf['private_mb']:.1f}MB",
        flush=True,
    )

    num_scale = min(args_cli.n_infer_images, args.num_scale_frames)
    with torch.no_grad():
        predictions = model.inference_streaming(
            images, num_scale_frames=num_scale, keyframe_interval=1, output_device=None,
        )
    stage("05 first inference_streaming() call done")

    n_after_inf, mb_after_inf = sum_live_tensor_bytes()
    proc_after_inf = _proc_mem()
    print(
        f"  [after inference] live tensors n={n_after_inf} total={mb_after_inf:.1f}MB "
        f"| process private={proc_after_inf['private_mb']:.1f}MB",
        flush=True,
    )
    print(
        f"  INFERENCE DELTA: gc-live-tensor +{mb_after_inf - mb_before_inf:.1f}MB, "
        f"process private +{proc_after_inf['private_mb'] - proc_before_inf['private_mb']:.1f}MB "
        f"(activations, KV cache, and other transient tensors from {images.shape[0]} frames)",
        flush=True,
    )
    print(f"  predictions keys: {list(predictions.keys())}", flush=True)

    print("\nDONE - audit_model_memory complete.", flush=True)


if __name__ == "__main__":
    main()
