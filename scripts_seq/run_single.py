"""Run ONE sequential-campaign experiment as a fresh process.

Reuses the real `demo.load_model` / `demo.load_images` and the real
`model.inference_streaming` — this script does not modify demo.py or any file
under lingbot_map/. It only wires them together with the same arguments the
documented CPU baseline in CLAUDE.md used (--use_sdpa, --camera_num_iterations 1),
plus the memory/timing instrumentation from monitor.py.

CPU is forced by the CALLER via CUDA_VISIBLE_DEVICES="" in the subprocess
environment (must be set before the interpreter starts / before `import torch`),
so that dtype stays float32 end-to-end, matching "FP32 + mmap=True +
del ckpt/state_dict + gc.collect()" exactly, with zero new optimizations.

Writes one JSON result file. Never silently swallows a failure: any exception is
caught, recorded with success=false, and re-raised (non-zero exit code) so the
campaign driver can tell a real crash apart from "no file produced" (SIGKILL/OOM).
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts_seq.monitor import MemoryMonitor  # noqa: E402


def build_args(model_path):
    """Mirrors demo.py's argparse defaults, pinned to the values the documented
    CPU baseline commands in CLAUDE.md always passed explicitly."""
    ns = argparse.Namespace()
    ns.model_path = model_path
    ns.image_size = 518
    ns.patch_size = 14
    ns.mode = "streaming"
    ns.enable_3d_rope = True
    ns.max_frame_num = 1024
    ns.num_scale_frames = 8
    ns.kv_cache_sliding_window = 64
    ns.camera_num_iterations = 1     # matches CLAUDE.md baseline commands
    ns.use_sdpa = True               # matches CLAUDE.md baseline commands
    ns.compile = False
    ns.keyframe_interval = None      # auto: 1 for num_frames <= 320 (all our N are)
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
    p.add_argument("--sample_interval", type=float, default=0.5)
    p.add_argument("--safety_free_mb", type=float, default=2048)
    cli = p.parse_args()

    import torch  # deferred: CUDA_VISIBLE_DEVICES must already be set by the caller

    assert not torch.cuda.is_available(), (
        "This script must run with CUDA hidden (CUDA_VISIBLE_DEVICES='') to preserve "
        "the FP32 CPU baseline documented in CLAUDE.md — refusing to run on GPU."
    )
    device = torch.device("cpu")

    result = {
        "run_id": cli.run_id,
        "num_frames": cli.num_frames,
        "pid": os.getpid(),
        "start_ts": time.time(),
        "success": False,
    }

    mon = MemoryMonitor(cli.csv_out, interval_s=cli.sample_interval,
                         safety_free_mb=cli.safety_free_mb)
    mon.start()

    try:
        import demo  # local module, repo root on sys.path via cwd

        result["snapshot_start"] = mon.snapshot("start")

        t0 = time.time()
        images, paths, _ = demo.load_images(
            image_folder=cli.sequence_dir,
            first_k=cli.num_frames,
            image_size=cli.image_size if hasattr(cli, "image_size") else 518,
            patch_size=14,
        )
        result["n_images_loaded"] = int(images.shape[0])
        result["image_load_time_s"] = round(time.time() - t0, 2)
        result["snapshot_after_images"] = mon.snapshot("after_images")

        args_ns = build_args(cli.model_path)
        t0 = time.time()
        model = demo.load_model(args_ns, device)
        result["model_load_time_s"] = round(time.time() - t0, 2)
        result["snapshot_after_load"] = mon.snapshot("after_load")

        images = images.to(device)
        dtype = torch.float32  # CPU path in demo.py always uses float32

        num_frames = images.shape[0]
        # Verbatim copy of demo.py main()'s own auto-selection logic (was
        # hardcoded to 1 here before -- silently wrong for num_frames > 320,
        # where demo.py's real default is > 1). All N in Phases 1-5 were <= 200,
        # so this bug never actually changed behavior until now.
        if num_frames > 320:
            keyframe_interval = (num_frames + 319) // 320
        else:
            keyframe_interval = 1
        result["keyframe_interval_used"] = keyframe_interval

        # Timestamp each individual streaming-inference frame against the
        # monitor's clock, without touching lingbot_map/demo.py source files.
        import tqdm.auto as tqdm_auto
        _orig_update = tqdm_auto.tqdm.update

        def _patched_update(self_bar, n=1):
            ret = _orig_update(self_bar, n)
            if getattr(self_bar, "desc", None) == "Streaming inference":
                mon.note_frame(self_bar.n)
            return ret

        tqdm_auto.tqdm.update = _patched_update

        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args_ns.num_scale_frames,
                keyframe_interval=keyframe_interval,
                output_device=None,
            )
        result["inference_time_s"] = round(time.time() - t0, 2)

        tqdm_auto.tqdm.update = _orig_update  # restore
        result["frame_events"] = mon.frame_events
        result["per_frame_time_s"] = round(result["inference_time_s"] / num_frames, 4)
        result["snapshot_after_inference"] = mon.snapshot("after_inference")

        # Touch predictions so the dict isn't dead-code-eliminated and so its
        # tensors are genuinely materialized before we measure "final" memory.
        result["output_keys"] = sorted(predictions.keys())
        del predictions
        gc.collect()
        result["snapshot_final"] = mon.snapshot("final")

        result["success"] = True
        result["exit_reason"] = "ok"

    except Exception as e:
        result["success"] = False
        result["exit_reason"] = "exception"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["frame_events"] = mon.frame_events

    finally:
        mon_summary = mon.stop()
        result["monitor_summary"] = mon_summary
        result["end_ts"] = time.time()
        result["wall_time_s"] = round(result["end_ts"] - result["start_ts"], 2)
        if mon_summary.get("safety_aborted"):
            # In practice os._exit(75) inside the monitor thread will have already
            # killed the process before we get here; this branch only fires if the
            # abort raced with normal completion.
            result["exit_reason"] = "safety_abort"
        with open(cli.out_json, "w") as f:
            json.dump(result, f, indent=2)

    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
