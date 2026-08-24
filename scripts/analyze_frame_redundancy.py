"""Estimate temporal redundancy between consecutive frames in test_images/.

Pure image analysis — does NOT load LingBot-Map / GCTStream, does NOT run
any model inference, does NOT touch demo.py. Conceptual groundwork for the
"Lightweight Context Analyzer" idea in CLAUDE.md (frame selection based on
motion/change, Paragraphica-inspired context-aware computation) — this
script only measures how much consecutive frames actually differ, as an
experimental estimate, not a change-detector implementation.

Metrics per consecutive pair (grayscale, at native 1600x1200 resolution —
independent of the model's own 518x392 crop/resize):
  - MAD: mean absolute pixel difference (0-255 scale)
  - SSIM: structural similarity, classic windowed Gaussian formula
    (Wang et al. 2004; skimage is not installed, so implemented directly
    with cv2 — same formula, comparable numbers)
  - % significantly-different pixels, at three thresholds (15/25/40 on the
    0-255 abs-diff map) — a coarse "how much of the frame actually moved"
    proxy
  - histogram correlation (cv2.compareHist, grayscale intensity histogram)
  - Canny edge-diff ratio (structural/edge change, distinct from raw pixel
    change — a scene can shift pixel values via lighting without much edge
    change, or vice versa)
  - spatial distribution: 4x4 grid of per-cell "% changed" (threshold=25),
    plus the change centroid (normalized 0-1 x/y) and its spread (std) —
    reveals whether change is localized (e.g. camera pan into one region)
    or spread across the whole frame (e.g. forward motion / whole-scene
    change)

No LingBot-Map inference is run — this is intentionally decoupled.
"""
import os
import numpy as np
import cv2


IMG_DIR = "test_images"
GRID_N = 4  # 4x4 spatial grid
THRESHOLDS = [15, 25, 40]  # abs-diff thresholds (0-255) for "significantly different" pixel %


def compute_ssim(img1_gray, img2_gray):
    """Classic windowed-Gaussian SSIM (Wang et al. 2004), manual implementation
    since skimage is not installed in this environment. Returns (mean_ssim, ssim_map)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1_gray.astype(np.float64)
    img2 = img2_gray.astype(np.float64)
    kernel_1d = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel_1d, kernel_1d.transpose())

    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 * img1, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    # Trim border (Gaussian filter edge effects), matches skimage's default crop behavior roughly.
    b = 5
    ssim_map = ssim_map[b:-b, b:-b]
    return float(ssim_map.mean()), ssim_map


def spatial_distribution(diff_map, threshold, grid_n=GRID_N):
    """Grid of per-cell % changed pixels, plus change centroid (normalized 0-1) and spread."""
    h, w = diff_map.shape
    changed = (diff_map > threshold).astype(np.float64)

    grid_pct = np.zeros((grid_n, grid_n))
    for gy in range(grid_n):
        for gx in range(grid_n):
            y0, y1 = int(gy * h / grid_n), int((gy + 1) * h / grid_n)
            x0, x1 = int(gx * w / grid_n), int((gx + 1) * w / grid_n)
            cell = changed[y0:y1, x0:x1]
            grid_pct[gy, gx] = 100.0 * cell.mean()

    ys, xs = np.indices((h, w))
    weight = diff_map.astype(np.float64)
    total_w = weight.sum()
    if total_w > 1e-9:
        cy = float((ys * weight).sum() / total_w) / h
        cx = float((xs * weight).sum() / total_w) / w
        spread_y = float(np.sqrt(((ys / h - cy) ** 2 * weight).sum() / total_w))
        spread_x = float(np.sqrt(((xs / w - cx) ** 2 * weight).sum() / total_w))
    else:
        cy = cx = spread_y = spread_x = float("nan")

    return grid_pct, (cx, cy), (spread_x, spread_y)


def main():
    files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"Found {len(files)} images: {files}")

    frames_gray = []
    frames_color = []
    for f in files:
        img = cv2.imread(os.path.join(IMG_DIR, f))
        frames_color.append(img)
        frames_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    h, w = frames_gray[0].shape
    print(f"Resolution: {w}x{h} (native, NOT the model's 518x392 crop/resize)\n")

    results = []
    for i in range(len(frames_gray) - 1):
        g1, g2 = frames_gray[i], frames_gray[i + 1]
        diff = np.abs(g1.astype(np.int16) - g2.astype(np.int16)).astype(np.float64)

        mad = float(diff.mean())
        ssim_val, _ = compute_ssim(g1, g2)

        pct_by_thresh = {t: 100.0 * float((diff > t).mean()) for t in THRESHOLDS}

        hist1 = cv2.calcHist([g1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([g2], [0], None, [256], [0, 256])
        hist_corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))

        edges1 = cv2.Canny(g1, 100, 200)
        edges2 = cv2.Canny(g2, 100, 200)
        edge_diff = np.abs(edges1.astype(np.int16) - edges2.astype(np.int16))
        edge_union = np.maximum(edges1, edges2).astype(np.float64).sum() + 1e-9
        edge_change_ratio = float(edge_diff.astype(np.float64).sum() / edge_union)

        grid_pct, centroid, spread = spatial_distribution(diff, threshold=25)

        results.append({
            "pair": f"{files[i]} -> {files[i+1]}",
            "mad": mad,
            "ssim": ssim_val,
            "pct_by_thresh": pct_by_thresh,
            "hist_corr": hist_corr,
            "edge_change_ratio": edge_change_ratio,
            "grid_pct": grid_pct,
            "centroid": centroid,
            "spread": spread,
        })

        print(f"[{i:02d}] {files[i]} -> {files[i+1]}")
        print(f"  MAD={mad:.3f}  SSIM={ssim_val:.4f}  hist_corr={hist_corr:.4f}  edge_change_ratio={edge_change_ratio:.4f}")
        print(f"  %changed_px: " + ", ".join(f"t={t}:{pct_by_thresh[t]:.2f}%" for t in THRESHOLDS))
        print(f"  change centroid (x,y, normalized 0-1)=({centroid[0]:.3f},{centroid[1]:.3f})  spread=({spread[0]:.3f},{spread[1]:.3f})")
        print(f"  grid % changed (threshold=25):")
        for row in grid_pct:
            print("    " + " ".join(f"{v:5.1f}" for v in row))
        print()

    # ---- Aggregate summary ----
    print("=" * 70)
    print("AGGREGATE SUMMARY (9 consecutive-pair transitions)")
    print("=" * 70)
    mads = [r["mad"] for r in results]
    ssims = [r["ssim"] for r in results]
    hist_corrs = [r["hist_corr"] for r in results]
    edge_ratios = [r["edge_change_ratio"] for r in results]
    print(f"MAD:  mean={np.mean(mads):.3f}  min={np.min(mads):.3f}  max={np.max(mads):.3f}  std={np.std(mads):.3f}")
    print(f"SSIM: mean={np.mean(ssims):.4f}  min={np.min(ssims):.4f}  max={np.max(ssims):.4f}  std={np.std(ssims):.4f}")
    print(f"hist_corr: mean={np.mean(hist_corrs):.4f}  min={np.min(hist_corrs):.4f}  max={np.max(hist_corrs):.4f}")
    print(f"edge_change_ratio: mean={np.mean(edge_ratios):.4f}  min={np.min(edge_ratios):.4f}  max={np.max(edge_ratios):.4f}")
    for t in THRESHOLDS:
        pcts = [r["pct_by_thresh"][t] for r in results]
        print(f"%changed_px (t={t}): mean={np.mean(pcts):.2f}%  min={np.min(pcts):.2f}%  max={np.max(pcts):.2f}%")

    print("\n" + "=" * 70)
    print("SKIPPABLE-FRAME ESTIMATE under different %changed_px thresholds")
    print("(a transition is 'potentially skippable' if %changed_px, at t=25, is")
    print(" BELOW the skip-threshold below — i.e. the next frame looks similar")
    print(" enough to the previous one that it might not need a full new")
    print(" inference. This is a rough estimate from pixel-level redundancy")
    print(" only — NOT a validation of map quality, NOT accounting for what")
    print(" LingBot-Map's KV cache / streaming mode already does internally.)")
    print("=" * 70)
    pct25 = np.array([r["pct_by_thresh"][25] for r in results])
    for skip_thresh in [1, 2, 5, 10, 20, 30]:
        n_skippable = int((pct25 < skip_thresh).sum())
        print(f"  skip-threshold=%changed_px<{skip_thresh}%: {n_skippable}/{len(results)} transitions "
              f"({100.0*n_skippable/len(results):.1f}%) potentially skippable")

    print("\nDONE - analyze_frame_redundancy complete.")


if __name__ == "__main__":
    main()
