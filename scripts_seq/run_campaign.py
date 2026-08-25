"""Drive the sequential-characterization campaign across frame-count tiers.

Launches scripts_seq/run_single.py as a completely fresh subprocess for every
repetition (never reuses a process), with CUDA hidden so the FP32 CPU baseline is
preserved. Never silently drops a failed run: if run_single.py exits non-zero, times
out, or gets killed (OOM/SIGKILL) without writing a JSON, a synthetic failure record
is appended anyway. Results are appended to results/campaign_results.jsonl as they
land, so a partial campaign is never lost.
"""
import json
import os
import signal
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQUENCE_DIR = os.path.join(REPO_ROOT, "example", "courthouse")
MODEL_PATH = os.path.join(REPO_ROOT, "checkpoints", "lingbot-map.pt")
RESULTS_JSONL = os.path.join(REPO_ROOT, "results", "campaign_results.jsonl")
JSON_DIR = os.path.join(REPO_ROOT, "results", "json")
CSV_DIR = os.path.join(REPO_ROOT, "results", "csv")

# (num_frames, num_repetitions) — matches the campaign design in CLAUDE.md.
TIERS = [(25, 5), (50, 5), (100, 3), (200, 3)]

# Per-run timeout. Phases 1-4 (N=10/25/50/100) showed inference time is NOT linear
# in num_frames -- it accelerates (per-frame cost grows with sequence length,
# consistent with attention over a linearly-growing KV-cache under
# keyframe_interval=1). A flat per-frame budget (the original approach) is what
# caused n200_rep1 to time out at 3120s. Model total inference time as quadratic,
# fit on the observed Phase 3/4 means (N=50: 502.9s, N=100: 1449.2s):
#   a*50^2 + b*50 = 502.9 ;  a*100^2 + b*100 = 1449.2  =>  a=0.0887, b=5.62
# then apply a 2x safety margin on top of the model's own prediction.
QUADRATIC_TIME_A = 0.0887
QUADRATIC_TIME_B = 5.62
TIMEOUT_SAFETY_FACTOR = 2.0
BASE_OVERHEAD_S = 120


def run_one(num_frames, rep_idx, sequence_dir=SEQUENCE_DIR, timeout_s=None,
            run_id_prefix="n"):
    run_id = f"{run_id_prefix}{num_frames}_rep{rep_idx}"
    out_json = os.path.join(JSON_DIR, f"{run_id}.json")
    csv_out = os.path.join(CSV_DIR, f"{run_id}.csv")
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "scripts_seq", "run_single.py"),
        "--num_frames", str(num_frames),
        "--sequence_dir", sequence_dir,
        "--model_path", MODEL_PATH,
        "--run_id", run_id,
        "--out_json", out_json,
        "--csv_out", csv_out,
        "--sample_interval", "0.5",
        "--safety_free_mb", "2048",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    if timeout_s is None:
        # NOTE: this quadratic fit was calibrated on N=50/100 and already
        # over-predicted N=200 by ~27% (Phase 5 showed the growth decelerating,
        # not staying quadratic) -- it's a safe (generous) upper bound for
        # extrapolating further, not an accurate estimate beyond N=200. For runs
        # past the fitted range, pass an explicit timeout_s instead of trusting it.
        predicted_inference_s = (QUADRATIC_TIME_A * num_frames ** 2
                                  + QUADRATIC_TIME_B * num_frames)
        timeout_s = int(BASE_OVERHEAD_S + predicted_inference_s * TIMEOUT_SAFETY_FACTOR)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] START {run_id} (timeout={timeout_s}s)", flush=True)

    record = None
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, timeout=timeout_s,
            start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        wall = time.time() - t0
        if os.path.exists(out_json):
            with open(out_json) as f:
                record = json.load(f)
            record["driver_returncode"] = proc.returncode
        else:
            record = {
                "run_id": run_id, "num_frames": num_frames, "success": False,
                "exit_reason": "no_output_file", "driver_returncode": proc.returncode,
                "driver_wall_time_s": round(wall, 2),
                "tail_output": proc.stdout.decode(errors="replace")[-2000:] if proc.stdout else "",
            }
    except subprocess.TimeoutExpired as e:
        wall = time.time() - t0
        record = {
            "run_id": run_id, "num_frames": num_frames, "success": False,
            "exit_reason": "timeout", "timeout_s": timeout_s,
            "driver_wall_time_s": round(wall, 2),
            "tail_output": (e.stdout or b"").decode(errors="replace")[-2000:] if e.stdout else "",
        }
        # subprocess.run with a timeout does not guarantee the child (and any of its
        # own children) are dead; belt-and-suspenders cleanup by process group.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    except Exception as e:
        wall = time.time() - t0
        record = {
            "run_id": run_id, "num_frames": num_frames, "success": False,
            "exit_reason": "driver_exception", "error": str(e),
            "driver_wall_time_s": round(wall, 2),
        }

    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")

    status = "OK" if record.get("success") else f"FAIL({record.get('exit_reason')})"
    print(f"[{time.strftime('%H:%M:%S')}] END   {run_id} {status} "
          f"wall={time.time()-t0:.1f}s", flush=True)
    return record


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated num_frames:reps to run instead of the full TIERS "
             "list, e.g. '200:3' -- for resuming a single tier after fixing a bug "
             "without re-running tiers that already completed cleanly.",
    )
    p.add_argument(
        "--sequence_dir", type=str, default=SEQUENCE_DIR,
        help="Override the frame sequence folder (default: example/courthouse, "
             "the one used by Phases 1-5). Pass a different real ordered "
             "trajectory folder (e.g. example/university) when the default "
             "sequence doesn't have enough frames for the requested N.",
    )
    p.add_argument(
        "--timeout_s", type=int, default=None,
        help="Manual per-run timeout override, bypassing the quadratic estimate "
             "-- required when extrapolating past N=200 (the model's fitted "
             "range), since it's known to over-predict there (see run_one()).",
    )
    p.add_argument(
        "--run_id_prefix", type=str, default="n",
        help="Prefix for run_id/output filenames, e.g. 'univ_n' to avoid "
             "colliding with existing n<N>_rep<r> files from a different sequence.",
    )
    args = p.parse_args()

    tiers = TIERS
    if args.only:
        tiers = []
        for chunk in args.only.split(","):
            n_str, reps_str = chunk.split(":")
            tiers.append((int(n_str), int(reps_str)))

    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    for num_frames, reps in tiers:
        for rep_idx in range(1, reps + 1):
            run_one(num_frames, rep_idx, sequence_dir=args.sequence_dir,
                    timeout_s=args.timeout_s, run_id_prefix=args.run_id_prefix)
            time.sleep(5)  # let the system settle between fresh processes
    print("CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
