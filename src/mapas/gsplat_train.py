#!/usr/bin/env python3
"""Gaussian Splatting (gsplat) entrenado con las imágenes y poses de LingBot-Map.

Inicializa las gaussianas desde la profundidad por píxel del modelo (sin COLMAP) y las
optimiza contra las imágenes de los frames, con la profundidad del modelo como guía
(peso que decae a lo largo del entrenamiento).

Configuración por defecto elegida midiendo en el fablab sobre frames no vistos
(bitácora del 2026-10-03): guía de profundidad absoluta (error relativo L1) en vez de sólo
su forma (Pearson), sin refinar poses y 7000 iteraciones. Fue la mejor en imagen y en
geometría a la vez; Pearson dejaba que la geometría se alejara (coincidencia 0.14) y
entrenar 15000 iteraciones sobreajustaba.

Evaluación honesta: 1 de cada --test_every frames reales queda fuera del entrenamiento y
se mide sobre esos frames (PSNR, SSIM y coincidencia de profundidad <5%), con la misma
partición que usa tsdf_mesh.py --holdout_every, para comparar los dos métodos.

    python scripts_context/gsplat_train.py eval/mapa.npz --out exports/splat/mapa_splat
      -> mapa_splat.ply (formato 3DGS estándar, lo carga el visor), mapa_splat_info.json,
         mapa_splat_vistas.png (frames de prueba: real | splatting)

Requiere CUDA_HOME apuntando al toolkit (gsplat compila sus kernels la primera vez).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


def ssim(a, b, win=11, sigma=1.5, mask=None):
    """SSIM medio de imágenes [B,3,H,W] en [0,1] (ventana gaussiana, Wang et al. 2004).
    Con mask [H,W] se promedia solo sobre los píxeles de la máscara."""
    x = torch.arange(win, device=a.device, dtype=a.dtype) - win // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    k = (g[:, None] * g[None, :]).expand(3, 1, win, win).contiguous()
    pad = win // 2
    mu_a = F.conv2d(a, k, padding=pad, groups=3)
    mu_b = F.conv2d(b, k, padding=pad, groups=3)
    s_aa = F.conv2d(a * a, k, padding=pad, groups=3) - mu_a ** 2
    s_bb = F.conv2d(b * b, k, padding=pad, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, k, padding=pad, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    if mask is not None:
        w = mask.to(m.dtype).expand_as(m)
        return (m * w).sum() / w.sum().clamp(min=1)
    return m.mean()


def se3_exp(xi):
    """xi [N,6] = (rot axis-angle, traslación) -> [N,4,4], diferenciable."""
    n = xi.shape[0]
    M = torch.zeros(n, 4, 4, device=xi.device, dtype=xi.dtype)
    w, v = xi[:, :3], xi[:, 3:]
    M[:, 0, 1], M[:, 0, 2], M[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
    M[:, 1, 0], M[:, 2, 0], M[:, 2, 1] = w[:, 2], -w[:, 1], w[:, 0]
    M[:, :3, 3] = v
    return torch.linalg.matrix_exp(M)


def split(n_real, every):
    idx = np.arange(n_real)
    test = idx[::every] if every > 0 else idx[:0]
    train = np.setdiff1d(idx, test)
    return train, test


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz")
    ap.add_argument("--out", required=True, help="prefijo de salida (sin extensión)")
    ap.add_argument("--iters", type=int, default=7000,
                    help="más no ayuda: con 15000 los frames no vistos empeoran (sobreajuste)")
    ap.add_argument("--test_every", type=int, default=8, help="0 = entrenar con todos los frames")
    ap.add_argument("--sh_degree", type=int, default=1)
    ap.add_argument("--init_stride", type=int, default=4, help="paso en píxeles al inicializar desde la profundidad")
    ap.add_argument("--init_voxel_rel", type=float, default=0.0015, help="vóxel de la nube inicial (rel. a la diagonal)")
    ap.add_argument("--conf_percentile", type=float, default=35.0)
    ap.add_argument("--depth_weight", type=float, default=0.1, help="peso de la guía de profundidad (Pearson)")
    ap.add_argument("--depth_loss", choices=["pearson", "l1"], default="l1",
                    help="pearson = sólo la forma de la profundidad (invariante a escala); "
                         "l1 = error relativo absoluto contra la profundidad del modelo")
    ap.add_argument("--pose_lr", type=float, default=0.0,
                    help="refinar poses (p. ej. 2e-5); en el fablab empeoró los frames no vistos")
    ap.add_argument("--ssim_lambda", type=float, default=0.2)
    ap.add_argument("--max_gaussians", type=int, default=3_000_000, help="tope de densificación (VRAM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filter", default=None,
                    help="<out>_filtro.npz de geo_filter.py: máscaras, ajuste a planos y selección de frames")
    ap.add_argument("--no_snap", action="store_true", help="con --filter: guía de profundidad sin ajuste a planos")
    ap.add_argument("--no_align", action="store_true",
                    help="con --filter: no usar las poses realineadas por planos (implica --no_snap)")
    ap.add_argument("--select_frames", action="store_true", help="con --filter: entrenar solo con los frames elegidos")
    ap.add_argument("--filter_eval_only", action="store_true",
                    help="con --filter: entrenar como sin filtro y usarlo solo para medir sin objetos dinámicos")
    a = ap.parse_args()

    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda")
    t0 = time.time()
    d = np.load(a.npz)
    real = np.flatnonzero(d["is_real"])
    imgs = d["images"][real]                                   # [N,H,W,3] uint8
    depth = d["depth"][real].astype(np.float32)
    conf = d["depth_conf"][real].astype(np.float32)
    K = d["intrinsic"][real].astype(np.float32)
    E = np.tile(np.eye(4, dtype=np.float32), (len(real), 1, 1))
    E[:, :3, :4] = d["extrinsic"][real]                        # mundo -> cámara (w2c)
    c2w = np.linalg.inv(E)
    N, H, W = depth.shape
    cthr = float(np.percentile(conf[::5], a.conf_percentile))
    train, test = split(N, a.test_every)
    flt = None
    if a.filter:
        fz = np.load(a.filter)
        assert np.array_equal(fz["real"], real), "el filtro no corresponde a estas predicciones"
        flt = {"keep": fz["keep"], "snap": fz["depth_snap"], "photo": fz["photo_ok"],
               "use": fz["use_frame"][real]}
        if a.select_frames and not a.filter_eval_only:
            train = train[flt["use"][train]]
        if a.no_align:
            a.no_snap = True                     # el ajuste a planos se calculó con las poses realineadas
        flt["c2w_fix"] = (fz["c2w_fix"] if "c2w_fix" in fz.files and bool(fz["aligned"])
                          and not a.no_align and not a.filter_eval_only else None)
        print(f"filtro geométrico: {a.filter} | ajuste a planos {'no' if a.no_snap or a.filter_eval_only else 'sí'}"
              f" | selección {'sí' if a.select_frames and not a.filter_eval_only else 'no'}"
              f" | solo medición {'sí' if a.filter_eval_only else 'no'}")
    use_flt = flt is not None and not a.filter_eval_only
    c2w_orig = c2w
    if use_flt and flt["c2w_fix"] is not None:   # poses realineadas por planos (todas, también las de prueba)
        c2w = flt["c2w_fix"].astype(np.float32)
        print("poses realineadas por planos")
    depth_tr = depth
    if use_flt:                                     # profundidad con la que se construye
        depth_tr = depth.copy()
        if not a.no_snap:
            sn = flt["snap"].astype(np.float32)
            depth_tr = np.where((sn > 0) & (depth_tr > 0), sn, depth_tr)
        depth_tr[~flt["keep"]] = 0
    print(f"{N} frames ({len(train)} entrenamiento, {len(test)} prueba), {W}x{H}, confianza > {cthr:.3f}")

    # --- nube inicial desde la profundidad de los frames de entrenamiento -------------
    pts, cols = [], []
    s = a.init_stride
    ys, xs = np.mgrid[0:H:s, 0:W:s]
    for i in train:
        z = depth_tr[i, ys, xs]
        m = (z > 0) & (conf[i, ys, xs] >= cthr)
        Pc = np.stack([(xs[m] - K[i, 0, 2]) * z[m] / K[i, 0, 0], (ys[m] - K[i, 1, 2]) * z[m] / K[i, 1, 1], z[m]], 1)
        pts.append(Pc @ c2w[i, :3, :3].T + c2w[i, :3, 3])
        cols.append(imgs[i, ys, xs][m])
    pts = np.concatenate(pts).astype(np.float64)
    cols = np.concatenate(cols).astype(np.float64) / 255.0
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.colors = o3d.utility.Vector3dVector(cols)
    diag = float(np.linalg.norm(np.percentile(pts, 99, 0) - np.percentile(pts, 1, 0)))
    pc = pc.voxel_down_sample(a.init_voxel_rel * diag)
    pts = np.asarray(pc.points, dtype=np.float32)
    cols = np.asarray(pc.colors, dtype=np.float32)
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(pts).query(pts, k=4)
    dist = np.clip(dist[:, 1:].mean(1), 1e-7, None).astype(np.float32)
    centers = c2w[train, :3, 3]
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    print(f"gaussianas iniciales: {len(pts):,} | escala de escena {scene_scale:.3f}")

    # --- parámetros ----------------------------------------------------------------
    n0 = len(pts)
    C0 = 0.28209479177387814                                     # armónico esférico de grado 0
    sh0 = (torch.from_numpy(cols) - 0.5) / C0
    kdeg = (a.sh_degree + 1) ** 2
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(pts)),
        "scales": torch.nn.Parameter(torch.log(torch.from_numpy(dist)).unsqueeze(1).repeat(1, 3)),
        "quats": torch.nn.Parameter(F.normalize(torch.randn(n0, 4), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n0,), 0.1))),
        "sh0": torch.nn.Parameter(sh0.unsqueeze(1)),
        "shN": torch.nn.Parameter(torch.zeros(n0, kdeg - 1, 3)),
    }).to(dev)
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / a.iters))
    strategy = DefaultStrategy(refine_stop_iter=int(a.iters * 0.6), reset_every=3000, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    pose_delta = torch.nn.Parameter(torch.zeros(N, 6, device=dev))
    pose_opt = torch.optim.Adam([pose_delta], lr=a.pose_lr) if a.pose_lr > 0 else None

    T_imgs = torch.from_numpy(imgs).to(dev).float() / 255.0      # [N,H,W,3]
    T_depth = torch.from_numpy(depth).to(dev)
    T_mask = torch.from_numpy(conf >= cthr).to(dev) & (T_depth > 0)
    if use_flt:                                       # guía de profundidad de entrenamiento
        T_depth_tr = torch.from_numpy(depth_tr).to(dev).half()
        T_mask_tr = T_mask & (T_depth_tr > 0)
        T_photo = torch.from_numpy(flt["photo"]).to(dev)
    else:
        T_depth_tr, T_mask_tr, T_photo = T_depth, T_mask, None
    T_static = torch.from_numpy(flt["photo"]).to(dev) if flt is not None else None
    del depth_tr
    T_K = torch.from_numpy(K).to(dev)
    T_c2w = torch.from_numpy(c2w).to(dev)

    def render(i, sh_deg, refine=True):
        cw = T_c2w[i:i + 1]
        if refine and pose_opt is not None:
            cw = cw @ se3_exp(pose_delta[i:i + 1])
        vm = torch.linalg.inv(cw)
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        out, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors, vm, T_K[i:i + 1], W, H,
            sh_degree=sh_deg, render_mode="RGB+ED", packed=False)
        return out[0, ..., :3], out[0, ..., 3], alpha[0, ..., 0], info

    def pearson(x, y):
        x = x - x.mean(); y = y - y.mean()
        return (x * y).sum() / (x.norm() * y.norm() + 1e-8)

    # --- entrenamiento ------------------------------------------------------------
    log = []
    order = np.random.permutation(np.repeat(train, math.ceil(a.iters / len(train))))[: a.iters]
    for step in range(a.iters):
        i = int(order[step])
        sh_deg = min(step // 1000, a.sh_degree)
        rgb, dep, alpha, info = render(i, sh_deg)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        gt = T_imgs[i]
        if T_photo is not None:                       # sin personas ni animales en la pérdida
            pm = T_photo[i]
            l1 = ((rgb - gt).abs().mean(-1) * pm).sum() / pm.sum().clamp(min=1)
            ls = 1 - ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], mask=pm)
        else:
            l1 = (rgb - gt).abs().mean()
            ls = 1 - ssim(rgb.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = (1 - a.ssim_lambda) * l1 + a.ssim_lambda * ls
        m = T_mask_tr[i] & (alpha > 0.5)
        if a.depth_weight > 0 and m.sum() > 100:
            w = a.depth_weight * max(0.0, 1 - step / a.iters)     # la guía pesa menos al final
            tdep = T_depth_tr[i][m].float()
            if a.depth_loss == "pearson":
                loss = loss + w * (1 - pearson(dep[m], tdep))
            else:
                loss = loss + w * ((dep[m] - tdep).abs() / tdep).mean()
        loss.backward()
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        if pose_opt is not None:
            pose_opt.step(); pose_opt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if len(params["means"]) > a.max_gaussians and step < strategy.refine_stop_iter:
            strategy.refine_stop_iter = step               # tope de memoria: dejar de densificar
        if step % 500 == 0 or step == a.iters - 1:
            msg = (f"it {step:5d}  loss {loss.item():.4f}  l1 {l1.item():.4f}  "
                   f"gaussianas {len(params['means']):,}  {time.time() - t0:.0f}s")
            print(msg, flush=True)
            log.append({"step": step, "loss": round(loss.item(), 5), "n": len(params["means"])})

    # --- evaluación sobre frames no vistos ----------------------------------------
    @torch.no_grad()
    def evaluate(ids, refine):
        ps, ss, inl, pst = [], [], [], []
        for i in ids:
            rgb, dep, alpha, _ = render(int(i), a.sh_degree, refine=refine)
            gt = T_imgs[i]
            mse = ((rgb.clamp(0, 1) - gt) ** 2).mean().item()
            ps.append(-10 * math.log10(max(mse, 1e-10)))
            if T_static is not None:                  # PSNR sin personas (no se pueden reconstruir)
                sm = T_static[i]
                mse_s = (((rgb.clamp(0, 1) - gt) ** 2).mean(-1) * sm).sum().item() / max(1, sm.sum().item())
                pst.append(-10 * math.log10(max(mse_s, 1e-10)))
            ss.append(ssim(rgb.clamp(0, 1).permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).item())
            m = T_mask[i] & (alpha > 0.5)
            if m.sum() > 0:
                r = ((dep[m] - T_depth[i][m]).abs() / T_depth[i][m])
                inl.append((r < 0.05).float().mean().item())
            else:
                inl.append(0.0)
        out = {"psnr": round(float(np.mean(ps)), 3), "ssim": round(float(np.mean(ss)), 4),
               "inlier5": round(float(np.mean(inl)), 4), "frames": len(ids)}
        if pst:
            out["psnr_static"] = round(float(np.mean(pst)), 3)
        return out

    res_test = evaluate(test, refine=False) if len(test) else None
    res_test_orig = None
    if len(test) and c2w_orig is not c2w:        # misma evaluación con las poses originales
        T_c2w_fix = T_c2w.clone()
        T_c2w.copy_(torch.from_numpy(c2w_orig).to(dev))
        res_test_orig = evaluate(test, refine=False)
        T_c2w.copy_(T_c2w_fix)
        print("prueba con poses originales:", res_test_orig)
    res_train = evaluate(train[:: max(1, len(train) // 60)], refine=True)
    print("prueba (frames no vistos):", res_test)
    print("entrenamiento (muestra):", res_train)

    # --- vistas de ejemplo: real | splatting ---------------------------------------
    import cv2
    rows = []
    ids = test if len(test) else train
    for i in ids[np.linspace(0, len(ids) - 1, min(6, len(ids))).astype(int)]:
        with torch.no_grad():
            rgb, _, _, _ = render(int(i), a.sh_degree, refine=not len(test))
        pair = np.hstack([imgs[i], (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)])
        rows.append(cv2.resize(pair, (pair.shape[1] // 2, pair.shape[0] // 2)))
    grid = np.vstack([np.hstack(rows[k:k + 2]) if k + 1 < len(rows) else np.hstack([rows[k], np.zeros_like(rows[k])])
                      for k in range(0, len(rows), 2)])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out + "_vistas.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # --- exportar en formato 3DGS estándar -----------------------------------------
    with torch.no_grad():
        means = params["means"].cpu().numpy()
        f_dc = params["sh0"][:, 0, :].cpu().numpy()
        f_rest = params["shN"].transpose(1, 2).reshape(len(means), -1).cpu().numpy()
        opac = params["opacities"].cpu().numpy()[:, None]
        scl = params["scales"].cpu().numpy()
        rot = F.normalize(params["quats"], dim=-1).cpu().numpy()
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{k}" for k in range(3)]
             + [f"f_rest_{k}" for k in range(f_rest.shape[1])] + ["opacity"]
             + [f"scale_{k}" for k in range(3)] + [f"rot_{k}" for k in range(4)])
    data = np.concatenate([means, np.zeros_like(means), f_dc, f_rest, opac, scl, rot], 1).astype(np.float32)
    with open(a.out + ".ply", "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(data)}\n"
                 + "".join(f"property float {n}\n" for n in names) + "end_header\n").encode())
        f.write(data.tobytes())

    info = {"npz": a.npz, "frames": int(N), "train_frames": int(len(train)), "test_frames": int(len(test)),
            "test_every": a.test_every, "iters": a.iters, "sh_degree": a.sh_degree,
            "gaussians_init": int(n0), "gaussians": int(len(means)),
            "depth_weight": a.depth_weight, "depth_loss": a.depth_loss, "pose_lr": a.pose_lr,
            "filter": a.filter, "filter_eval_only": a.filter_eval_only, "snap": bool(use_flt and not a.no_snap),
            "select_frames": bool(use_flt and a.select_frames),
            "test": res_test, "test_original_poses": res_test_orig, "train_sample": res_train,
            "aligned_poses": bool(c2w_orig is not c2w),
            "pose_delta_rot_deg_median": round(float(np.degrees(pose_delta[:, :3].norm(dim=1).median().item())), 4),
            "seconds": round(time.time() - t0, 1),
            "vram_peak_mb": int(torch.cuda.max_memory_allocated() / 2 ** 20),
            "ply_mb": round(os.path.getsize(a.out + ".ply") / 2 ** 20, 1), "log": log}
    json.dump(info, open(a.out + "_info.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "log"}, indent=1))


if __name__ == "__main__":
    main()
