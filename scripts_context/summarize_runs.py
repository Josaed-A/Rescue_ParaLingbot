"""Combined scoreboard for every run of a sample: route shape + self-consistency.

Two metrics that disagree on purpose:
  * route (compare_route.py, needs a sketch): global shape against the real walk.
    Catches drift — a reconstruction that curls scores badly even if every pair
    of neighbouring frames agrees.
  * consistency (evaluate_consistency.py): local/medium-range agreement between
    real frames. Catches noisy depth and pose jitter, but is blind to slow drift.

Usage:
  python3 scripts_context/summarize_runs.py --eval_dir <dir> --sketch <ruta.jpeg> \
      [--out combined.json] [--skip_consistency]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--sketch", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--skip_consistency", action="store_true")
    args = p.parse_args()

    runs = sorted(glob.glob(os.path.join(args.eval_dir, "*.npz")))
    rutas = os.path.join(args.eval_dir, "rutas")
    os.makedirs(rutas, exist_ok=True)
    rows = {}
    for npz in runs:
        name = os.path.splitext(os.path.basename(npz))[0]
        rj = os.path.join(rutas, f"{name}_ruta.json")
        if not os.path.exists(rj):
            subprocess.run([sys.executable, os.path.join(HERE, "compare_route.py"),
                            "--npz", npz, "--sketch", args.sketch, "--out_json", rj,
                            "--label", name], capture_output=True)
        if os.path.exists(rj):
            rows[name] = json.load(open(rj))

    if not args.skip_consistency:
        cj = os.path.join(args.eval_dir, "report.json")
        specs = [f"{n}={os.path.join(args.eval_dir, n)}.npz" for n in rows]
        subprocess.run([sys.executable, os.path.join(HERE, "evaluate_consistency.py"),
                        *specs, "--out", cj], capture_output=True)
        if os.path.exists(cj):
            cons = json.load(open(cj))
            for n, r in rows.items():
                if n in cons:
                    r["inlier_1s"] = cons[n]["lag30"].get("inlier")
                    r["inlier_05s"] = cons[n]["lag15"].get("inlier")
                    r["frames_real"] = cons[n]["frames_real"]

    order = sorted(rows, key=lambda n: rows[n]["error_pct"])
    head = f"{'corrida':<24}{'err ruta%':>10}{'rectitud':>10}{'largo':>7}{'giro°':>8}{'coh 1s':>8}{'frames':>8}"
    print(head)
    print("-" * len(head))
    for n in order:
        r = rows[n]
        print(f"{n:<24}{r['error_pct']:>10}{r['straightness_est']:>10}{r['length_ratio']:>7}"
              f"{r['turn_total_deg_est']:>8}{r.get('inlier_1s', '-'):>8}{r.get('frames_real', '-'):>8}")
    ref = rows[order[0]]
    print(f"{'CROQUIS (referencia)':<24}{'':>10}{ref['straightness_sketch']:>10}{1.0:>7}{ref['turn_total_deg_sketch']:>8}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
        print("escrito", args.out)


if __name__ == "__main__":
    main()
