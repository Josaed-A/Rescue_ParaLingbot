"""Comparación y alineación de trayectorias de cámara (integración Stella-VSLAM, etapa 4).

Funciones puras con numpy, sin ROS ni modelo. Las usa src/mapas/compare_tracking.py y
las usarán el modo híbrido (etapa 6) y la alineación de escala (etapa 9).

Convención: poses c2w 4x4 (cámara -> mundo), cámara OpenCV (x derecha, y abajo, z adelante).
Las dos trayectorias que se comparan tienen cada una su propio mundo (la cámara de su primer
frame) y su propia escala (monocular): nada se compara sin alinear antes, y la alineación es una
etapa explícita cuyo resultado (escala, rotación, traslación) se informa, no se esconde.

Orden de corrección que exige el plan (comparar sólo después de corregir):
  1. tiempo     associate() por stamp; estimate_time_offset() comprueba que no quede desfase;
  2. ejes       rotation_offsets() estima la rotación de mundo W y la de cámara C con
                R_a ≈ W·R_b·C; si las convenciones de cámara coinciden, C ≈ I;
  3. escala     umeyama(..., with_scale=True) = Sim(3); windowed_scale() mide si la escala es
                estable en el tiempo (si no, una Sim(3) global no alcanza).
"""
import numpy as np


# ---------------------------------------------------------------------------
# rotaciones
# ---------------------------------------------------------------------------
def rot_angle_deg(R):
    """Ángulo (grados) de una rotación 3x3 o de un lote (...,3,3)."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def project_so3(M):
    """La rotación más cercana (Frobenius) a una matriz 3x3."""
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


# ---------------------------------------------------------------------------
# alineación
# ---------------------------------------------------------------------------
def umeyama(src, dst, with_scale=True):
    """(s, R, t) que minimiza sum ||dst - (s R src + t)||² (Umeyama 1991). src, dst: (N,3)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    S = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(S)
    E = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ E @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ E) / var_s) if (with_scale and var_s > 0) else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(s, R, t, P):
    return (s * (np.asarray(P) @ R.T)) + t


def apply_sim3_to_poses(s, R, t, T):
    """Lleva poses c2w (N,4,4) de un mundo al otro: la rotación de cámara gira con R, el centro
    se escala y traslada. (La escala no afecta la orientación.)"""
    out = np.array(T, dtype=np.float64, copy=True)
    out[:, :3, :3] = R @ out[:, :3, :3]
    out[:, :3, 3] = apply_sim3(s, R, t, out[:, :3, 3])
    return out


def rotation_offsets(Ra, Rb, step=1, min_angle_deg=0.5):
    """W, C, info con Ra_i ≈ W·Rb_i·C. W = cambio de mundo; C = diferencia de convención de
    cámara (si las dos trayectorias usan la misma cámara OpenCV, C ≈ I).

    Solución cerrada tipo mano-ojo (Park y Martin 1994): las rotaciones relativas en el marco de la
    cámara, A = Ra_i^T Ra_j y B = Rb_i^T Rb_j, cumplen A = C^T B C (W se cancela), así que sus
    vectores de rotación cumplen rotvec(B) = C · rotvec(A): C sale de alinear esos ejes (Kabsch).
    Luego W = proj(sum Ra_i C^T Rb_i^T).

    Limitación física, no numérica: si casi todo el giro es alrededor de un eje (caminar girando
    sólo en yaw), C queda indeterminado alrededor de ese eje. `info["cond"]` = valores singulares
    normalizados de la nube de ejes; si el segundo es chico, C no está bien determinado."""
    from scipy.spatial.transform import Rotation
    Ra = np.asarray(Ra, np.float64)
    Rb = np.asarray(Rb, np.float64)
    A = np.einsum("nji,njk->nik", Ra[:-step], Ra[step:])
    B = np.einsum("nji,njk->nik", Rb[:-step], Rb[step:])
    a = Rotation.from_matrix(A).as_rotvec()
    b = Rotation.from_matrix(B).as_rotvec()
    keep = (np.degrees(np.linalg.norm(a, axis=1)) >= min_angle_deg) & \
           (np.degrees(np.linalg.norm(b, axis=1)) >= min_angle_deg)
    info = {"pares": int(keep.sum()), "cond": None}
    if keep.sum() >= 3:
        a, b = a[keep], b[keep]
        C = project_so3(b.T @ a)                                       # sum b a^T
        sv = np.linalg.svd(a.T @ a, compute_uv=False)
        info["cond"] = [float(v) for v in sv / (sv[0] + 1e-12)]
    else:
        C = np.eye(3)
    W = project_so3(np.einsum("nij,kj,nlk->il", Ra, C, Rb))            # sum Ra C^T Rb^T
    return W, C, info


# ---------------------------------------------------------------------------
# tiempo
# ---------------------------------------------------------------------------
def associate(ta, tb, tol):
    """Para cada stamp de `ta`, el índice del stamp más cercano de `tb` si está a <= tol s.
    Devuelve (ia, ib, dt) con dt = tb[ib] - ta[ia]. `tb` debe estar ordenado."""
    ta = np.asarray(ta, np.float64)
    tb = np.asarray(tb, np.float64)
    if len(ta) == 0 or len(tb) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    j = np.clip(np.searchsorted(tb, ta), 1, len(tb) - 1) if len(tb) > 1 else np.zeros(len(ta), int)
    if len(tb) > 1:
        left = j - 1
        j = np.where(np.abs(tb[left] - ta) <= np.abs(tb[j] - ta), left, j)
    dt = tb[j] - ta
    ok = np.abs(dt) <= tol
    return np.flatnonzero(ok), j[ok], dt[ok]


def angular_speed(t, R, max_gap=0.5):
    """Velocidad angular (grados/s) entre muestras consecutivas, en el instante medio.
    Se descartan pares separados por más de max_gap s (huecos de tracking)."""
    t = np.asarray(t, np.float64)
    if len(t) < 2:
        return np.zeros(0), np.zeros(0)
    dt = np.diff(t)
    rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
    w = rot_angle_deg(rel) / np.maximum(dt, 1e-6)
    ok = (dt > 0) & (dt <= max_gap)
    return 0.5 * (t[:-1] + t[1:])[ok], w[ok]


def estimate_time_offset(ta, Ra, tb, Rb, max_lag=1.0, step=0.01, smooth=0.1):
    """Desfase d (s) tal que la trayectoria b en el instante t+d se parece a la a en t, por
    correlación de la velocidad angular (no depende del mundo ni de la escala). Devuelve
    (d, correlación en d, correlación en 0). Si las dos usan el mismo reloj, d ≈ 0."""
    xa, wa = angular_speed(ta, Ra)
    xb, wb = angular_speed(tb, Rb)
    if len(xa) < 10 or len(xb) < 10:
        return None, None, None
    lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
    if hi - lo < 4 * max_lag:
        return None, None, None
    g = np.arange(lo + max_lag, hi - max_lag, step)
    k = max(1, int(round(smooth / step)))
    ker = np.ones(k) / k

    def sig(x, w, shift):
        v = np.interp(g + shift, x, w)
        return np.convolve(v, ker, mode="same")

    a = sig(xa, wa, 0.0)
    a = (a - a.mean()) / (a.std() + 1e-9)
    best, best_c, c0 = 0.0, -np.inf, None
    for d in np.arange(-max_lag, max_lag + step / 2, step):
        b = sig(xb, wb, d)
        b = (b - b.mean()) / (b.std() + 1e-9)
        c = float((a * b).mean())
        if abs(d) < step / 2:
            c0 = c
        if c > best_c:
            best, best_c = float(d), c
    return best, best_c, c0


# ---------------------------------------------------------------------------
# métricas
# ---------------------------------------------------------------------------
def path_length(P):
    P = np.asarray(P)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def ate(Pa, Pb_aligned):
    """Error absoluto de trayectoria por muestra (misma unidad que Pa)."""
    return np.linalg.norm(np.asarray(Pa) - np.asarray(Pb_aligned), axis=1)


def rpe(t, Ta, Tb, scale, delta=1.0, tol=0.15):
    """Error relativo entre pares de muestras separadas ~delta s, en el marco de la cámara del
    primer elemento del par (no depende de la deriva acumulada ni del cambio de mundo).
    Tb ya en la escala de a (se multiplica la traslación relativa por `scale`).
    Devuelve dict con arrays: trans_rel (|err|/|desplazamiento a|), rot_deg, disp_a."""
    t = np.asarray(t, np.float64)
    out = {"trans_rel": [], "rot_deg": [], "disp_a": [], "t": []}
    for i in range(len(t)):
        j = int(np.searchsorted(t, t[i] + delta))
        cands = [k for k in (j - 1, j) if 0 <= k < len(t) and k != i]
        if not cands:
            continue
        j = min(cands, key=lambda k: abs(t[k] - t[i] - delta))
        if abs(t[j] - t[i] - delta) > tol:
            continue
        da = np.linalg.inv(Ta[i]) @ Ta[j]
        db = np.linalg.inv(Tb[i]) @ Tb[j]
        ta_, tb_ = da[:3, 3], scale * db[:3, 3]
        na = np.linalg.norm(ta_)
        out["disp_a"].append(na)
        out["trans_rel"].append(np.linalg.norm(ta_ - tb_) / na if na > 0 else np.nan)
        out["rot_deg"].append(float(rot_angle_deg(da[:3, :3].T @ db[:3, :3])))
        out["t"].append(t[i])
    return {k: np.array(v) for k, v in out.items()}


def windowed_scale(t, Pa, Pb, win=5.0, step=1.0, min_pts=10, min_extent_rel=0.05):
    """Escala Sim(3) b->a estimada en ventanas deslizantes de `win` s. Se descartan ventanas
    con pocas muestras o casi sin movimiento (extensión < min_extent_rel del total), donde la
    escala no está determinada. Devuelve (centros, escalas)."""
    t = np.asarray(t, np.float64)
    if len(t) < min_pts:
        return np.zeros(0), np.zeros(0)
    total = np.ptp(Pa, axis=0).max() + 1e-12
    cs, ss = [], []
    for c0 in np.arange(t.min(), t.max() - win + 1e-9, step):
        m = (t >= c0) & (t < c0 + win)
        if m.sum() < min_pts or np.ptp(Pa[m], axis=0).max() < min_extent_rel * total:
            continue
        s, _, _ = umeyama(Pb[m], Pa[m], with_scale=True)
        cs.append(c0 + win / 2)
        ss.append(s)
    return np.array(cs), np.array(ss)


def stats(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "rmse": float(np.sqrt((x ** 2).mean())), "mediana": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max())}
