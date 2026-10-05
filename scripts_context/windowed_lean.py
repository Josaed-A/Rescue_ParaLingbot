"""Modo windowed con poca RAM, para secuencias con muchos frames sintéticos.

Hace exactamente lo mismo que `inference_windowed` en modo de intervalo fijo (el que usa
`process_and_view.py --mode windowed`): ventanas solapadas, frames de escala, keyframes cada
`keyframe_interval`, caché limpia por ventana y la misma alineación y costura entre ventanas
(`_align_and_stitch_windows` del propio modelo). Cambian solo tres cosas, todas de memoria:

  - las imágenes se leen de disco ventana por ventana, no todas juntas;
  - profundidad y confianza se guardan a media resolución (cada 2 píxeles, lo mismo que deja
    `--save_ds 2`) y en float16, en vez de float32 a resolución completa;
  - al npz solo van los frames reales (los sintéticos fueron solo contexto).

Con ~2000-2900 frames, `process_and_view.py` acumula >16 GB de salidas en CPU y muere; esto
queda en ~6-7 GB. La escala entre ventanas se estima con la mediana del cociente de
profundidades sobre la grilla de media resolución en vez de la completa (diferencia mínima).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--manifest", default=None, help="manifest.json (real/synthetic por frame)")
    ap.add_argument("--model_path", default=os.path.join(REPO, "checkpoints", "lingbot-map.pt"))
    ap.add_argument("--window_size", type=int, default=16)
    ap.add_argument("--num_scale_frames", type=int, default=2)
    ap.add_argument("--camera_num_iterations", type=int, default=4)
    ap.add_argument("--keyframe_interval", type=int, default=None)
    ap.add_argument("--ds", type=int, default=2)
    ap.add_argument("--save_predictions", required=True)
    a = ap.parse_args()

    import demo
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general

    exts = (".png", ".jpg", ".jpeg")
    paths = sorted(os.path.join(a.image_folder, f) for f in os.listdir(a.image_folder) if f.lower().endswith(exts))
    S = len(paths)
    is_real = np.ones(S, bool)
    source_index = np.arange(S)
    if a.manifest:
        entries = json.load(open(a.manifest))["frames"]
        assert [os.path.basename(p) for p in paths] == [e["file"] for e in entries], "manifest distinto"
        is_real = np.array([e["kind"] == "real" for e in entries])
        source_index = np.array([e.get("source_index", -1) for e in entries])
    kf_int = a.keyframe_interval or (1 if S <= 320 else (S + 319) // 320)
    print(f"{S} frames ({int(is_real.sum())} reales) | keyframe_interval={kf_int}", flush=True)

    device = torch.device("cuda")
    margs = argparse.Namespace(model_path=a.model_path, image_size=518, patch_size=14, mode="windowed",
                               enable_3d_rope=True, max_frame_num=1024, num_scale_frames=a.num_scale_frames,
                               kv_cache_sliding_window=16, camera_num_iterations=a.camera_num_iterations,
                               use_sdpa=True, compile=False)
    model = demo.load_model(margs, device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model.aggregator = model.aggregator.to(dtype=dtype)
    model.eval()

    ws = min(a.num_scale_frames, S)
    eff_overlap = min(ws, S - 1) if S > 1 else 0
    actual_window = ws + max(a.window_size - ws, 0) * max(kf_int, 1)
    eff_window = min(actual_window, S)
    step = max(eff_window - eff_overlap, 1)
    if eff_window >= S:
        windows = [(0, S)]
    else:
        windows = []
        for st in range(0, S, step):
            en = min(st + eff_window, S)
            if en - st >= eff_overlap or en == S:
                windows.append((st, en))
            if en == S:
                break
    print(f"{len(windows)} ventanas de {eff_window} frames (solape {eff_overlap})", flush=True)

    d = a.ds
    real_imgs = {}
    all_preds = []
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        for wi, (start, end) in enumerate(windows):
            imgs = load_and_preprocess_images(paths[start:end], mode="crop", image_size=518, patch_size=14)
            H, W = imgs.shape[-2:]
            for k in range(end - start):                     # imágenes de los reales para el npz
                g = start + k
                if is_real[g] and g not in real_imgs:
                    real_imgs[g] = (imgs[k, :, ::d, ::d].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            wimg = imgs.unsqueeze(0).to(device)
            model.clean_kv_cache()
            wscale = min(ws, end - start)
            lists = {"pose_enc": [], "depth": [], "depth_conf": []}
            ftype = []

            def collect(out):
                lists["pose_enc"].append(out["pose_enc"].float().cpu())
                lists["depth"].append(out["depth"][:, :, ::d, ::d].half().cpu())
                lists["depth_conf"].append(out["depth_conf"][:, :, ::d, ::d].half().cpu())

            collect(model.forward(wimg[:, :wscale], num_frame_for_scale=wscale, num_frame_per_block=wscale,
                                  causal_inference=True))
            ftype += [0] * wscale
            for i in range(wscale, end - start):
                is_kf = kf_int <= 1 or ((i - wscale) % kf_int == 0)
                if not is_kf:
                    model._set_skip_append(True)
                out = model.forward(wimg[:, i:i + 1], num_frame_for_scale=wscale, num_frame_per_block=1,
                                    causal_inference=True)
                if not is_kf:
                    model._set_skip_append(False)
                collect(out)
                ftype.append(1 if is_kf else 2)
                del out
            ft = torch.tensor(ftype, dtype=torch.uint8).unsqueeze(0)
            all_preds.append({"pose_enc": torch.cat(lists["pose_enc"], 1), "depth": torch.cat(lists["depth"], 1),
                              "depth_conf": torch.cat(lists["depth_conf"], 1), "frame_type": ft,
                              "is_keyframe": ft != 2})
            del wimg, imgs
            if wi % 10 == 0:
                print(f"ventana {wi + 1}/{len(windows)}  {time.time() - t0:.0f}s", flush=True)
    model._last_window_size = eff_overlap
    model._last_overlap_size = eff_overlap
    pred = model._align_and_stitch_windows(all_preds, scale_mode="median")
    print(f"inferencia {time.time() - t0:.0f}s ({(time.time() - t0) / S:.2f} s/frame)", flush=True)

    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], (H, W))
    e4 = torch.zeros((*extr.shape[:-2], 4, 4), dtype=extr.dtype)
    e4[..., :3, :4] = extr
    e4[..., 3, 3] = 1.0
    extr = closed_form_inverse_se3_general(e4)[..., :3, :4]          # igual que demo.postprocess
    keep = np.flatnonzero(is_real)
    depth = pred["depth"][0, :, :, :, 0].float().numpy()[keep]
    conf = pred["depth_conf"][0].float().numpy()[keep]
    os.makedirs(os.path.dirname(os.path.abspath(a.save_predictions)), exist_ok=True)
    np.savez(a.save_predictions, depth=depth.astype(np.float16), depth_conf=conf.astype(np.float16),
             images=np.stack([real_imgs[g] for g in keep]), extrinsic=extr[0].numpy()[keep].astype(np.float32),
             intrinsic=intr[0].numpy()[keep].astype(np.float32), ds=d, is_real=np.ones(len(keep), bool),
             source_index=source_index[keep], keyframe_interval=kf_int, frames_total=S)
    print(f"guardado {a.save_predictions} ({len(keep)} frames reales)", flush=True)


if __name__ == "__main__":
    main()
