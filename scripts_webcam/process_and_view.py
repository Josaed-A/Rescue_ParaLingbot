"""Run the real demo.py pipeline (load_model/load_images/inference_streaming/
postprocess/prepare_for_visualization -- all reused unmodified) on a folder of
captured webcam frames, then export the reconstructed point cloud as a
portable .glb file via PointCloudViewer's existing GLB-export code path
(lingbot_map/vis/point_cloud_viewer.py::_export_glb, normally triggered by a
GUI button click -- called directly here instead, since there's no browser in
this environment to click it from).

Does not modify demo.py or any file under lingbot_map/. CPU is forced by the
CALLER via CUDA_VISIBLE_DEVICES="" (must be set before the interpreter starts),
matching the sequential-campaign baseline that's already validated up to
N=320 without failures -- the GPU baseline OOMs around frame 35 with the
unmodified config, so CPU is the only proven-safe path to a real 400-frame
run (see CLAUDE.md).

After exporting the GLB, also renders a quick static PNG preview (matplotlib
3D scatter, no GPU/EGL context needed) so there's something viewable without
opening the .glb in a separate tool, and leaves the viser server running
(non-blocking) so it can also be viewed live/interactively afterward.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_folder", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--first_k", type=int, default=None)
    p.add_argument("--glb_out", type=str, required=True)
    p.add_argument("--preview_png", type=str, required=True)
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--conf_threshold", type=float, default=1.5)
    p.add_argument("--downsample_factor", type=int, default=10)
    p.add_argument("--keep_images_on_cpu", action="store_true",
                    help="Leave the full image tensor on CPU instead of moving it "
                         "to GPU up front (demo.py always moves it). "
                         "inference_streaming is explicitly written for this -- it "
                         "slices and moves one frame at a time, so GPU input memory "
                         "becomes O(1) frames instead of O(sequence length).")
    p.add_argument("--num_scale_frames", type=int, default=8,
                    help="Bidirectional scale frames processed together in phase 1 "
                         "(demo.py default 8). The README's 'Running on Limited GPU "
                         "Memory' section recommends 2 to shrink that phase's "
                         "activation peak -- needed for 518x518 (portrait-sourced) "
                         "frames on an 8GB card, which OOM at the default 8.")
    p.add_argument("--offload_to_cpu", action="store_true",
                    help="Move per-frame predictions to CPU as they're produced "
                         "(demo.py's own flag; its --help claims it's on by default "
                         "but the actual argparse default is False). Cuts GPU peak "
                         "memory growth over a long sequence.")
    p.add_argument("--kv_cache_sliding_window", type=int, default=64,
                    help="demo.py's own flag (default 64): max frames kept in the "
                         "streaming KV cache. On an 8GB GPU at 518x518 the SDPA "
                         "path OOMs once ~26 frames are cached, so a window below "
                         "that (e.g. 16) is what makes long sequences fit -- at "
                         "the cost of less temporal context for pose estimation.")
    p.add_argument("--use_sdpa", action="store_true",
                    help="Force PyTorch SDPA attention instead of FlashInfer "
                         "(demo.py's own flag, default off). The SDPA streaming "
                         "path clones the KV cache on every attention call; "
                         "FlashInfer's paged KV cache is the memory-efficient path "
                         "the README recommends for streaming on GPU.")
    p.add_argument("--no_serve", action="store_true",
                    help="Skip viewer.run() at the end (which never returns by "
                         "design, keeping the process alive to serve the viser "
                         "UI) -- use this for quick validation runs where only "
                         "the GLB/PNG matter, not a live interactive session.")
    args = p.parse_args()

    import torch
    import demo  # real production entrypoint, unmodified

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    t0 = time.time()
    images, paths, resolved_image_folder = demo.load_images(
        image_folder=args.image_folder, first_k=args.first_k,
        image_size=518, patch_size=14,
    )
    print(f"Loaded {images.shape[0]} images in {time.time()-t0:.1f}s", flush=True)

    model_args = argparse.Namespace(
        model_path=args.model_path, image_size=518, patch_size=14,
        mode="streaming", enable_3d_rope=True, max_frame_num=1024,
        num_scale_frames=args.num_scale_frames, kv_cache_sliding_window=args.kv_cache_sliding_window,
        camera_num_iterations=1, use_sdpa=args.use_sdpa, compile=False,
    )
    t0 = time.time()
    model = demo.load_model(model_args, device)
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
            print(f"Casting aggregator to {dtype}", flush=True)
            model.aggregator = model.aggregator.to(dtype=dtype)

    if not args.keep_images_on_cpu:
        images = images.to(device)
    num_frames = images.shape[0]
    keyframe_interval = 1 if num_frames <= 320 else (num_frames + 319) // 320
    print(f"num_frames={num_frames} keyframe_interval={keyframe_interval} "
          f"dtype={dtype}", flush=True)

    output_device = torch.device("cpu") if args.offload_to_cpu else None
    print(f"num_scale_frames={args.num_scale_frames} "
          f"offload_to_cpu={args.offload_to_cpu} use_sdpa={args.use_sdpa} "
          f"kv_cache_sliding_window={args.kv_cache_sliding_window}", flush=True)

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(
            images, num_scale_frames=args.num_scale_frames,
            keyframe_interval=keyframe_interval,
            output_device=output_device,
        )
    print(f"Inference done in {time.time()-t0:.1f}s "
          f"({(time.time()-t0)/num_frames:.2f}s/frame)", flush=True)

    # Mirrors demo.py main()'s own post-inference handling for offload_to_cpu:
    # the GPU copy of the images is freed and the CPU copy that
    # inference_streaming already produced is reused instead.
    if args.offload_to_cpu:
        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        images_for_post = predictions["images"]
    else:
        images_for_post = images

    predictions, images_cpu = demo.postprocess(predictions, images_for_post)
    vis_dict = demo.prepare_for_visualization(predictions, images_cpu)

    from lingbot_map.vis import PointCloudViewer
    viewer = PointCloudViewer(
        pred_dict=vis_dict,
        port=args.port,
        vis_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        point_size=0.00001,
        image_folder=resolved_image_folder,
    )

    # --- Export GLB (same code path as the "Export GLB" GUI button) ---
    os.makedirs(os.path.dirname(os.path.abspath(args.glb_out)) or ".", exist_ok=True)
    viewer.glb_output_path.value = args.glb_out
    viewer._export_glb()
    print(f"GLB export status: {viewer.glb_status.value}", flush=True)

    # --- Quick static preview (no GL/EGL context needed) ---
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_points, all_colors = [], []
        for step in viewer.all_steps:
            pc = viewer.pcs[step]["pc"]
            color = viewer.pcs[step]["color"]
            conf = viewer.pcs[step]["conf"]
            pts, cols = viewer.parse_pc_data(
                pc, color, conf, None, set_border_color=False,
                downsample_factor=args.downsample_factor,
            )
            if len(pts) > 0:
                all_points.append(pts)
                all_colors.append(cols if cols.dtype == np.uint8
                                   else (np.clip(cols, 0, 1) * 255).astype(np.uint8))
        pts = np.concatenate(all_points, axis=0)
        cols = np.concatenate(all_colors, axis=0).astype(np.float32) / 255.0

        # Subsample further for a snappy preview render.
        if len(pts) > 200_000:
            idx = np.random.choice(len(pts), 200_000, replace=False)
            pts, cols = pts[idx], cols[idx]

        # Plain 2D projections instead of mpl_toolkits.mplot3d -- this machine
        # has two conflicting matplotlib installs (system apt package 3.5.1 vs
        # pip 3.10.9) that break the mplot3d/Axes3D import specifically; the
        # GLB export already gives a real interactive 3D view, so a 2D
        # projection is enough for a quick static preview here.
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        views = [
            (0, 1, "top-down (X vs Y)"),
            (0, 2, "front (X vs Z)"),
            (1, 2, "side (Y vs Z)"),
        ]
        for ax, (xi, yi, title) in zip(axes, views):
            ax.scatter(pts[:, xi], pts[:, yi], c=cols, s=0.4, marker=".")
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_axis_off()
        fig.suptitle(f"LingBot-Map reconstruction -- {num_frames} real webcam frames")
        fig.tight_layout()
        fig.savefig(args.preview_png, dpi=150)
        plt.close(fig)
        print(f"Preview PNG saved to {args.preview_png}", flush=True)
    except Exception as e:
        print(f"Preview render failed (non-fatal): {e}", flush=True)

    if args.no_serve:
        print("DONE (--no_serve: not starting the live viewer)", flush=True)
        return

    print(f"Viewer live at http://localhost:{args.port} "
          f"(and http://<lan-ip>:{args.port})", flush=True)
    print("DONE (viewer left running in background -- this process will now "
          "stay alive indefinitely to keep serving it)", flush=True)
    viewer.run(background_mode=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
