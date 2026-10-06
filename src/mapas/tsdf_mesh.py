#!/usr/bin/env python3
"""Malla por fusión TSDF a partir de las predicciones del modelo (profundidad + pose por frame).

Trata a LingBot-Map como un sensor RGB-D virtual: cada frame real aporta su imagen, su
profundidad por píxel, sus intrínsecos y su pose, y se integran en una grilla de vóxeles con
distancia con signo truncada (Open3D ScalableTSDFVolume). De ahí sale una malla con color.

Con --refine hace un segundo paso: arma la malla sólo con los frames que coinciden con ella,
realinea los demás contra esa malla (ICP punto a punto con escala, porque el error de esos
frames incluye escala de profundidad) y vuelve a integrar los que quedan coincidiendo.

Además de la malla mide cuánto la respalda cada frame: proyecta la malla desde algunas
cámaras (raycasting) y compara esa profundidad con la que predijo el modelo. Si las poses o la
escala no son consistentes entre frames, la fusión promedia superficies desalineadas y esta
coincidencia baja (paredes "dobles" o engrosadas).

    python src/mapas/tsdf_mesh.py eval/mapa.npz --out exports/mapa_malla
      -> mapa_malla.ply (malla completa) + mapa_malla.glb (para el visor) + mapa_malla_info.json
"""
import argparse
import json
import time

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz")
    ap.add_argument("--out", required=True, help="prefijo de salida (sin extensión)")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="tamaño de vóxel en unidades del modelo (0 = mediana de profundidad / 150)")
    ap.add_argument("--trunc_voxels", type=float, default=4.0, help="truncamiento del TSDF, en vóxeles")
    ap.add_argument("--conf_percentile", type=float, default=35.0)
    ap.add_argument("--depth_max_percentile", type=float, default=97.0,
                    help="ignora la profundidad más lejana (menos confiable)")
    ap.add_argument("--min_cluster_frac", type=float, default=0.002,
                    help="descarta pedazos sueltos con menos de esta fracción de triángulos")
    ap.add_argument("--refine", action="store_true",
                    help="malla de consenso + realinear (ICP con escala) los frames que no coinciden")
    ap.add_argument("--accept", type=float, default=0.3,
                    help="coincidencia mínima (fracción de píxeles a <5%%) para entrar al consenso")
    ap.add_argument("--holdout_every", type=int, default=0,
                    help="deja fuera 1 de cada N frames reales y mide sobre ellos (misma partición "
                         "que gsplat_train.py --test_every)")
    ap.add_argument("--filter", default=None,
                    help="<out>_filtro.npz de geo_filter.py: máscaras y ajuste a planos para los frames que se integran")
    ap.add_argument("--no_snap", action="store_true", help="con --filter: no usar la profundidad ajustada a planos")
    ap.add_argument("--no_align", action="store_true",
                    help="con --filter: no usar las poses realineadas por planos (implica --no_snap)")
    ap.add_argument("--select_frames", action="store_true", help="con --filter: integrar solo los frames elegidos")
    ap.add_argument("--glb_max_tris", type=int, default=3_000_000,
                    help="la .glb para el navegador se simplifica a este máximo")
    a = ap.parse_args()

    t0 = time.time()
    d = np.load(a.npz)
    real = np.flatnonzero(d["is_real"])
    depth_all = d["depth"]
    conf_all = d["depth_conf"]
    sample = depth_all[real[::10]].astype(np.float32)
    med = float(np.median(sample[sample > 0]))
    dmax = float(np.percentile(sample[sample > 0], a.depth_max_percentile))
    cthr = float(np.percentile(conf_all[real[::10]].astype(np.float32), a.conf_percentile))
    voxel = a.voxel or med / 150.0
    print(f"{len(real)} frames | profundidad mediana {med:.3f}, máx usada {dmax:.3f} | "
          f"vóxel {voxel:.5f} | confianza > {cthr:.3f}")

    poses = {}                                   # i -> (c2w 4x4, escala de profundidad)
    for i in real:
        E = np.eye(4)
        E[:3, :4] = d["extrinsic"][i]
        poses[int(i)] = (np.linalg.inv(E), 1.0)

    def frame(i):
        dep = depth_all[i].astype(np.float32)
        dep[conf_all[i].astype(np.float32) < cthr] = 0
        return dep

    flt = None
    aligned = False
    if a.filter:
        fz = np.load(a.filter)
        aligned = "c2w_fix" in fz.files and bool(fz["aligned"]) and not a.no_align
        if a.no_align:
            a.no_snap = True                     # el ajuste a planos se calculó con las poses realineadas
        flt = {"pos": {int(r): k for k, r in enumerate(fz["real"])}, "keep": fz["keep"],
               "snap": None if a.no_snap else fz["depth_snap"], "use": fz["use_frame"]}
        poses_orig = dict(poses)
        if aligned:
            for r_, k_ in flt["pos"].items():
                poses[r_] = (fz["c2w_fix"][k_].astype(np.float64), 1.0)
        print(f"filtro geométrico: {a.filter} (poses realineadas: {'sí' if aligned else 'no'}, "
              f"ajuste a planos: {'no' if a.no_snap else 'sí'}, "
              f"selección de frames: {'sí' if a.select_frames else 'no'})")

    def frame_build(i):
        """Profundidad con la que se construye: la original filtrada por confianza y, con
        --filter, sin los píxeles descartados y llevada a los planos donde corresponde."""
        dep = frame(i)
        if flt is not None:
            k = flt["pos"][i]
            if flt["snap"] is not None:
                sn = flt["snap"][k].astype(np.float32)
                dep = np.where((sn > 0) & (dep > 0), sn, dep)
            dep = np.where(flt["keep"][k], dep, 0).astype(np.float32)
        return dep

    def integrate(ids):
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=a.trunc_voxels * voxel,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        for i in ids:
            c2w, sc = poses[i]
            dep = frame_build(i) * sc
            K = d["intrinsic"][i]
            H, W = dep.shape
            intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(d["images"][i])), o3d.geometry.Image(dep),
                depth_scale=1.0, depth_trunc=dmax * sc, convert_rgb_to_intensity=False)
            vol.integrate(rgbd, intr, np.linalg.inv(c2w))
        mesh = vol.extract_triangle_mesh()
        n_raw = len(mesh.triangles)
        cl, cnt, _ = mesh.cluster_connected_triangles()
        cl, cnt = np.asarray(cl), np.asarray(cnt)
        mesh.remove_triangles_by_mask(~(cnt[cl] >= a.min_cluster_frac * n_raw))
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh, n_raw

    def agreement(mesh, ids):
        """Por frame: fracción de píxeles confiables a <5% de la malla, y cobertura."""
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        res = {}
        for i in ids:
            c2w, sc = poses[i]
            K = d["intrinsic"][i]
            H, W = depth_all[i].shape
            rays = scene.create_rays_pinhole(o3d.core.Tensor(K.astype(np.float64)),
                                             o3d.core.Tensor(np.linalg.inv(c2w)), W, H)
            t_hit = scene.cast_rays(rays)["t_hit"].numpy()
            ys, xs = np.mgrid[0:H, 0:W]
            z_mesh = t_hit / np.sqrt(((xs - K[0, 2]) / K[0, 0]) ** 2 + ((ys - K[1, 2]) / K[1, 1]) ** 2 + 1)
            dp = frame(i) * sc
            ok = (dp > 0) & (dp < dmax * sc)
            hit = ok & np.isfinite(z_mesh)
            if ok.sum() == 0 or hit.sum() == 0:
                res[i] = (0.0, 0.0, float("nan"))
                continue
            r = np.abs(z_mesh[hit] - dp[hit]) / dp[hit]
            res[i] = (float((r < 0.05).mean()), float(hit.sum() / ok.sum()), float(np.median(r)))
        return res

    def summary(res):
        v = np.array([x[0] for x in res.values()])
        return {"inlier5_mean": round(float(v.mean()), 4), "inlier5_p10": round(float(np.percentile(v, 10)), 4),
                "frames_below_0.2": int((v < 0.2).sum()),
                "coverage_mean": round(float(np.mean([x[1] for x in res.values()])), 4),
                "rel_err_median": round(float(np.nanmedian([x[2] for x in res.values()])), 4)}

    def novel_views(mesh, ids_):
        """PSNR/SSIM del color de la malla visto desde cámaras que no se integraron, y
        coincidencia de profundidad: lo mismo que mide gsplat_train.py en esos frames."""
        import sys, os, torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gsplat_train import ssim as ssim_t
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        tris = np.asarray(mesh.triangles)
        vcol = np.asarray(mesh.vertex_colors)
        ps, ss, inl = [], [], []
        for i in ids_:
            c2w, sc = poses[i]
            K = d["intrinsic"][i]
            H, W = depth_all[i].shape
            rays = scene.create_rays_pinhole(o3d.core.Tensor(K.astype(np.float64)),
                                             o3d.core.Tensor(np.linalg.inv(c2w)), W, H)
            r = scene.cast_rays(rays)
            t_hit = r["t_hit"].numpy()
            hitm = np.isfinite(t_hit)
            pid = r["primitive_ids"].numpy()
            uv = r["primitive_uvs"].numpy()
            img = np.zeros((H, W, 3))
            tv = tris[pid[hitm]]
            u, v = uv[hitm, 0:1], uv[hitm, 1:2]
            img[hitm] = (1 - u - v) * vcol[tv[:, 0]] + u * vcol[tv[:, 1]] + v * vcol[tv[:, 2]]
            gt = d["images"][i].astype(np.float64) / 255.0
            mse = float(((img - gt) ** 2).mean())
            ps.append(-10 * np.log10(max(mse, 1e-10)))
            ss.append(float(ssim_t(torch.from_numpy(img).permute(2, 0, 1)[None].float(),
                                   torch.from_numpy(gt).permute(2, 0, 1)[None].float())))
            ys, xs = np.mgrid[0:H, 0:W]
            z_mesh = t_hit / np.sqrt(((xs - K[0, 2]) / K[0, 0]) ** 2 + ((ys - K[1, 2]) / K[1, 1]) ** 2 + 1)
            dp = frame(i)
            ok = (dp > 0) & hitm
            inl.append(float((np.abs(z_mesh[ok] - dp[ok]) / dp[ok] < 0.05).mean()) if ok.any() else 0.0)
        return {"psnr": round(float(np.mean(ps)), 3), "ssim": round(float(np.mean(ss)), 4),
                "inlier5": round(float(np.mean(inl)), 4), "frames": len(ids_)}

    ids = [int(i) for i in real]
    test_ids = []
    if a.holdout_every > 0:
        test_ids = ids[:: a.holdout_every]
        ids = [i for i in ids if i not in set(test_ids)]
        print(f"{len(ids)} frames para la malla, {len(test_ids)} apartados para medir")
    if flt is not None and a.select_frames:
        ids = [i for i in ids if flt["use"][i]]
        print(f"selección de frames: {len(ids)} frames para la malla")
    mesh, n_raw = integrate(ids)
    res = agreement(mesh, ids)
    info = {"npz": a.npz, "filter": a.filter, "snap": bool(a.filter and not a.no_snap),
            "select_frames": bool(a.filter and a.select_frames), "frames": len(ids), "voxel": voxel, "sdf_trunc": a.trunc_voxels * voxel,
            "depth_median": med, "depth_max_used": dmax, "conf_threshold": cthr,
            "pass1": {"triangles": int(len(mesh.triangles)), **summary(res)}}
    print("paso 1 (todos los frames):", info["pass1"])

    if a.refine:
        good = [i for i in ids if res[i][0] >= a.accept]
        bad = [i for i in ids if res[i][0] < a.accept]
        cons, _ = integrate(good)
        print(f"consenso: {len(good)} frames, {len(bad)} a realinear")
        target = o3d.geometry.PointCloud(cons.vertices)
        target.normals = cons.vertex_normals
        before = agreement(cons, bad)
        fixed, report = [], []
        for i in bad:
            c2w, sc = poses[i]
            dep = frame(i)[::4, ::4]
            K = d["intrinsic"][i]
            ys, xs = np.mgrid[0:dep.shape[0], 0:dep.shape[1]] * 4
            m = (dep > 0) & (dep < dmax)
            Z = dep[m]
            Pc = np.stack([(xs[m] - K[0, 2]) * Z / K[0, 0], (ys[m] - K[1, 2]) * Z / K[1, 1], Z], 1)
            Pw = Pc @ c2w[:3, :3].T + c2w[:3, 3]
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pw))
            T = np.eye(4)
            for dist in (0.15 * med, 0.07 * med, 0.03 * med):
                r = o3d.pipelines.registration.registration_icp(
                    src, target, dist, T,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
                T = r.transformation
            s_ = float(np.cbrt(np.linalg.det(T[:3, :3])))
            R = T[:3, :3] / s_
            new_c2w = np.eye(4)
            new_c2w[:3, :3] = R @ c2w[:3, :3]
            new_c2w[:3, 3] = s_ * (R @ c2w[:3, 3]) + T[:3, 3]
            old = poses[i]
            poses[i] = (new_c2w, sc * s_)
            after = agreement(cons, [i])[i]
            if after[0] >= a.accept:
                fixed.append(i)
            else:
                poses[i] = old
            report.append({"frame": i, "before": round(before[i][0], 3), "after": round(after[0], 3),
                           "scale": round(s_, 3), "fitness": round(float(r.fitness), 3)})
        print(f"realineados y aceptados: {len(fixed)} de {len(bad)}")
        final_ids = sorted(good + fixed)
        mesh, n_raw = integrate(final_ids)
        res = agreement(mesh, ids)
        info["refine"] = {"accept": a.accept, "consensus_frames": len(good), "realigned_ok": len(fixed),
                          "dropped": len(bad) - len(fixed), "used_frames": len(final_ids),
                          "per_frame": report}
        info["pass2"] = {"triangles": int(len(mesh.triangles)), **summary(res)}
        print("paso 2 (consenso + realineados):", info["pass2"])
    if test_ids:
        info["test"] = novel_views(mesh, test_ids)
        print("frames apartados:", info["test"])
        if aligned:
            cur = dict(poses)
            poses.update(poses_orig)
            info["test_original_poses"] = novel_views(mesh, test_ids)
            poses.update(cur)
            print("frames apartados (poses originales):", info["test_original_poses"])
    info["triangles"] = int(len(mesh.triangles))
    info["vertices"] = int(len(mesh.vertices))
    info["per_frame_inlier5"] = {str(i): round(res[i][0], 3) for i in ids}
    o3d.io.write_triangle_mesh(a.out + ".ply", mesh)

    # versión para el navegador
    web = mesh
    if len(web.triangles) > a.glb_max_tris:
        web = mesh.simplify_quadric_decimation(a.glb_max_tris)
        web.compute_vertex_normals()
    import trimesh
    col = (np.asarray(web.vertex_colors) * 255).astype(np.uint8)
    tm = trimesh.Trimesh(np.asarray(web.vertices), np.asarray(web.triangles),
                         vertex_colors=np.c_[col, np.full(len(col), 255, np.uint8)], process=False)
    tm.export(a.out + ".glb")
    info["glb_triangles"] = int(len(web.triangles))
    info["seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(a.out + "_info.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k not in ("per_frame_inlier5", "refine")}, indent=1))


if __name__ == "__main__":
    main()
