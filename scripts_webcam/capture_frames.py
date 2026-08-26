"""Capture real frames from a live webcam into a numbered-PNG folder that
demo.py / run_single.py / run_gpu_baseline.py can consume directly via
--image_folder, exactly like example/courthouse or example/university.

This is deliberately the simplest possible bridge from "live camera" to "the
pipeline we already trust": rather than reimplementing inference_streaming's
two phases against a live cv2.VideoCapture loop (analyzed but not built yet,
see CLAUDE.md "Analisis de interfaz para webcam RGB en vivo"), it captures a
real trajectory first (record), then hands the resulting folder to the exact
same demo.py / run_single.py entrypoints already used and validated for
example/courthouse and example/university. Does not touch demo.py or
lingbot_map.

Frame count is decoupled from model speed on purpose: capture runs at the
camera's own pace (real motion, real timing), inference runs later/separately
at whatever pace the chosen device supports.
"""
import argparse
import os
import time

import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_frames", type=int, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--fps", type=float, default=10.0,
                    help="Target capture rate (frames/sec). The camera itself "
                         "reports up to 30fps; this throttles how often we save "
                         "a frame, not how fast the sensor runs.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    native_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    native_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {native_w:.0f}x{native_h:.0f} @ {native_fps:.1f}fps "
          f"(native), capturing at target {args.fps}fps", flush=True)

    interval_s = 1.0 / args.fps
    saved = 0
    t_start = time.time()
    next_capture_t = t_start

    try:
        while saved < args.num_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"WARNING: camera.read() failed at frame {saved}, retrying...",
                      flush=True)
                continue
            now = time.time()
            if now < next_capture_t:
                continue  # drain the buffer without saving, keep pace with target fps
            path = os.path.join(args.out_dir, f"{saved:06d}.png")
            cv2.imwrite(path, frame)
            saved += 1
            next_capture_t += interval_s
            if saved % 20 == 0 or saved == args.num_frames:
                elapsed = now - t_start
                print(f"  captured {saved}/{args.num_frames} "
                      f"(elapsed {elapsed:.1f}s)", flush=True)
    finally:
        cap.release()

    total_s = time.time() - t_start
    print(f"DONE: {saved} frames saved to {args.out_dir} in {total_s:.1f}s "
          f"({saved/total_s:.2f} fps effective)", flush=True)


if __name__ == "__main__":
    main()
