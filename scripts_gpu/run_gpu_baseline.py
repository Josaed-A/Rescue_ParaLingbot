"""Detailed single-run GPU/NVIDIA baseline for LingBot-Map.

Unlike scripts_seq/run_single.py (which forced CPU to preserve the FP32 baseline
for the sequential-characterization campaign, see CLAUDE.md), this script does
NOT touch CUDA_VISIBLE_DEVICES -- it reuses demo.load_model()/demo.load_images()
unmodified and replicates demo.py main()'s own device/dtype/aggregator-cast logic
verbatim, so "the baseline" here is exactly what `python demo.py` would do on
this Linux/NVIDIA machine with nothing overridden: CUDA device, bf16 aggregator
cast (compute capability 8.9 >= 8), FP32 heads, use_sdpa=True,
camera_num_iterations=1, kv_cache_sliding_window=64, keyframe_interval=1 (since
200 < 320), image_size=518 -- no architecture/dtype/quantization/keyframe/
resolution changes of any kind.

Monitors RAM+VRAM+GPU utilization/temp/power continuously via
scripts_gpu/monitor_gpu.py, and timestamps every individual streaming-inference
frame (via a tqdm.update() patch, restored afterward) so memory can be
correlated to frame index, not just wall-clock time.
"""
import argparse
import json
import os
import sys
import time
import traceback
import warnings

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import demo  # noqa: E402  (repo's real production entrypoint, unmodified)
from scripts_gpu.monitor_gpu import GPUMemoryMonitor  # noqa: E402


def build_args(num_frames, sequence_dir, model_path):
    ns = argparse.Namespace()
    ns.image_folder = sequence_dir
    ns.video_path = None
    ns.fps = 10
    ns.first_k = num_frames
    ns.stride = 1
    ns.rotate_clockwise_90 = False
    ns.model_path = model_path
    ns.image_size = 518
    ns.patch_size = 14
    ns.mode = "streaming"
    ns.enable_3d_rope = True
    ns.max_frame_num = 1024
    ns.num_scale_frames = 8
    ns.keyframe_interval = None  # let demo.py's own auto-selection logic decide
    ns.kv_cache_sliding_window = 64
    ns.camera_num_iterations = 1
    ns.use_sdpa = True
    ns.compile = False
    ns.offload_to_cpu = False
    return ns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_frames", type=int, required=True)
    p.add_argument("--sequence_dir", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--run_id", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--csv_out", type=str, required=True)
    p.add_argument("--sample_interval", type=float, default=0.3)
    p.add_argument("--safety_free_ram_mb", type=float, default=2048)
    p.add_argument("--safety_free_vram_mb", type=float, default=400)
    args = p.parse_args()

    assert torch.cuda.is_available(), (
        "This is the GPU baseline script -- CUDA must be visible. "
        "Use scripts_seq/run_single.py for the CPU-forced baseline."
    )

    warnings_captured = []
    _orig_showwarning = warnings.showwarning

    def _capture_warning(message, category, filename, lineno, file=None, line=None):
        warnings_captured.append(f"{category.__name__}: {message}")

    warnings.showwarning = _capture_warning

    monitor = GPUMemoryMonitor(
        args.csv_out, interval_s=args.sample_interval,
        safety_free_ram_mb=args.safety_free_ram_mb,
        safety_free_vram_mb=args.safety_free_vram_mb,
    )
    monitor.start()

    result = {"run_id": args.run_id, "num_frames": args.num_frames, "success": False}
    partial = {}  # populated incrementally so a mid-run crash doesn't lose progress

    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_capability = torch.cuda.get_device_capability(0)
        gpu_total_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
        result["gpu_name"] = gpu_name
        result["gpu_capability"] = list(gpu_capability)
        result["gpu_total_vram_mb"] = round(gpu_total_vram_mb, 1)

        snap_start = monitor.snapshot("start")
        result["snapshot_start"] = snap_start

        model_args = build_args(args.num_frames, args.sequence_dir, args.model_path)
        device = torch.device("cuda")

        t0 = time.time()
        images, paths, resolved_folder = demo.load_images(
            image_folder=model_args.image_folder, first_k=model_args.first_k,
            image_size=model_args.image_size, patch_size=model_args.patch_size,
        )
        t_images = time.time() - t0
        snap_after_images = monitor.snapshot("after_images")
        result["images_load_time_s"] = round(t_images, 3)
        result["snapshot_after_images"] = snap_after_images

        t0 = time.time()
        model = demo.load_model(model_args, device)
        t_load = time.time() - t0
        snap_after_load = monitor.snapshot("after_load")
        result["model_load_time_s"] = round(t_load, 3)
        result["snapshot_after_load"] = snap_after_load

        # Verbatim copy of demo.py main()'s dtype selection + aggregator cast.
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        aggregator_cast = False
        if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
            model.aggregator = model.aggregator.to(dtype=dtype)
            aggregator_cast = True
        result["dtype"] = str(dtype)
        result["aggregator_cast_to_dtype"] = aggregator_cast

        images = images.to(device)
        num_frames_loaded = images.shape[0]
        snap_after_images_to_device = monitor.snapshot("after_images_to_device")
        result["snapshot_after_images_to_device"] = snap_after_images_to_device

        if model_args.keyframe_interval is None:
            if num_frames_loaded > 320:
                model_args.keyframe_interval = (num_frames_loaded + 319) // 320
            else:
                model_args.keyframe_interval = 1
        result["keyframe_interval_used"] = model_args.keyframe_interval

        # Timestamp each individual streaming-inference frame against the
        # monitor's clock, without touching lingbot_map/demo.py source files.
        import tqdm.auto as tqdm_auto
        _orig_update = tqdm_auto.tqdm.update

        def _patched_update(self_bar, n=1):
            ret = _orig_update(self_bar, n)
            if getattr(self_bar, "desc", None) == "Streaming inference":
                monitor.note_frame(self_bar.n)
            return ret

        tqdm_auto.tqdm.update = _patched_update

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            predictions = model.inference_streaming(
                images,
                num_scale_frames=model_args.num_scale_frames,
                keyframe_interval=model_args.keyframe_interval,
                output_device=None,
            )
        torch.cuda.synchronize()
        t_inference = time.time() - t0

        tqdm_auto.tqdm.update = _orig_update  # restore

        snap_final = monitor.snapshot("final")
        monitor_summary = monitor.stop()

        result.update({
            "success": True,
            "model_load_time_s": round(t_load, 3),
            "images_load_time_s": round(t_images, 3),
            "inference_time_s": round(t_inference, 3),
            "per_frame_time_s": round(t_inference / num_frames_loaded, 4),
            "fps_effective": round(num_frames_loaded / t_inference, 4),
            "wall_time_s": round(t_load + t_inference, 3),
            "snapshot_start": snap_start,
            "snapshot_after_images": snap_after_images,
            "snapshot_after_load": snap_after_load,
            "snapshot_after_images_to_device": snap_after_images_to_device,
            "snapshot_final": snap_final,
            "monitor_summary": monitor_summary,
            "torch_cuda_max_memory_allocated_mb":
                round(torch.cuda.max_memory_allocated() / 1e6, 1),
            "torch_cuda_max_memory_reserved_mb":
                round(torch.cuda.max_memory_reserved() / 1e6, 1),
            "frame_events": monitor.frame_events,
            "warnings": warnings_captured,
            "prediction_keys": list(predictions.keys()),
        })

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["frame_events"] = monitor.frame_events
        try:
            result["monitor_summary"] = monitor.stop()
        except Exception:
            pass
        result["warnings"] = warnings_captured
    finally:
        warnings.showwarning = _orig_showwarning

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"DONE run_id={args.run_id} success={result['success']}", flush=True)
    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
