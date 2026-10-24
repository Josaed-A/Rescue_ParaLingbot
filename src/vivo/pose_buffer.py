"""Buffer temporal de poses (integración Stella-VSLAM, etapa 5).

Responde "¿cuál era la pose de esta fuente en el instante t?" a partir de las muestras
vecinas, usando SIEMPRE los timestamps originales de adquisición (nunca la hora actual).
Asocia un frame de LingBot con la pose BASIC y la de Stella del mismo instante aunque las
tres fuentes vayan a ritmos distintos: la cámara a 15-30 fps, Stella con los frames que alcanza a
procesar y LingBot a ~2 por segundo.

Interpolación (regla 8 del plan): posición LINEAL, orientación por SLERP de cuaterniones. Nunca
se interpolan los elementos de la matriz 4x4: el promedio de dos rotaciones no es una rotación.

Qué devuelve `query(t)`:

  EXACT    hay una muestra con ese stamp (a <= exact_tol)
  INTERP   t queda entre dos muestras del MISMO segmento, separadas por <= max_gap, sin un
           corte explícito (add_break) entre ellas
  NEAREST  t queda fuera del rango (o el hueco es muy grande) pero la muestra más cercana está a
           <= edge_tol: se devuelve esa muestra tal cual, sin extrapolar
  NONE     nada de lo anterior: no se inventa una pose

Segmentos: cuando Stella se reinicia su mundo cambia; dos muestras de segmentos distintos no se
interpolan entre sí. Cortes: `add_break(t)` marca una discontinuidad dentro de un segmento (p. ej.
una corrección de loop closure, etapa 12): tampoco se interpola a través de ella.

La tolerancia por defecto (max_gap = 0.25 s) sale de medir el error de interpolación en
trayectorias reales (docs/STELLA_INTEGRATION.md, etapa 5).
"""
import bisect
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class QueryKind(IntEnum):
    NONE = 0
    EXACT = 1
    INTERP = 2
    NEAREST = 3


# ---------------------------------------------------------------------------
# cuaterniones (x, y, z, w), sin dependencias
# ---------------------------------------------------------------------------
def mat_to_quat(R):
    """Matriz de rotación 3x3 -> cuaternión unitario (x, y, z, w) con w >= 0."""
    R = np.asarray(R, np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-300)) * 2
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def quat_to_mat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, a):
    """Interpolación esférica entre dos cuaterniones unitarios, por el camino corto."""
    q0 = np.asarray(q0, np.float64)
    q1 = np.asarray(q1, np.float64)
    d = float(np.dot(q0, q1))
    if d < 0:                       # q y -q son la misma rotación: tomar el arco corto
        q1, d = -q1, -d
    if d > 0.9995:                  # casi iguales: lineal y normalizar (estable)
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def interpolate_pose(T0, T1, a):
    """Pose c2w intermedia: centro lineal, rotación por SLERP. a en [0, 1]."""
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(slerp(mat_to_quat(T0[:3, :3]), mat_to_quat(T1[:3, :3]), a))
    T[:3, 3] = (1 - a) * T0[:3, 3] + a * T1[:3, 3]
    return T


# ---------------------------------------------------------------------------
@dataclass
class PoseQuery:
    kind: QueryKind
    c2w: Optional[np.ndarray] = None
    dt_nearest: float = float("inf")     # distancia (s) a la muestra más cercana
    gap: float = float("nan")            # separación de las dos muestras usadas (INTERP)
    segment: int = -1
    meta: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.kind != QueryKind.NONE


class PoseBuffer:
    """Muestras (stamp, c2w, segmento) ordenadas por stamp; admite llegadas fuera de orden.
    Seguro entre hilos (el puente ROS2 escribe desde su hilo y el modelo consulta desde el suyo).

    max_samples acota la memoria: se descartan las más viejas. Por defecto alcanza para ~3 h a 20 Hz,
    porque el loop closure (etapa 12) necesita la historia."""

    def __init__(self, max_gap=0.25, edge_tol=0.05, exact_tol=1e-6, max_samples=200_000):
        self.max_gap = float(max_gap)
        self.edge_tol = float(edge_tol)
        self.exact_tol = float(exact_tol)
        self.max_samples = int(max_samples)
        self._t = []
        self._T = []
        self._seg = []
        self._meta = []
        self._breaks = []
        self._lock = threading.Lock()

    def __len__(self):
        with self._lock:
            return len(self._t)

    def clear(self):
        with self._lock:
            self._t, self._T, self._seg, self._meta, self._breaks = [], [], [], [], []

    def add(self, stamp, c2w, segment=0, meta=None):
        c2w = np.asarray(c2w, np.float64)
        if c2w.shape == (3, 4):
            m = np.eye(4)
            m[:3, :4] = c2w
            c2w = m
        with self._lock:
            i = bisect.bisect_right(self._t, float(stamp))
            if i > 0 and abs(self._t[i - 1] - stamp) <= self.exact_tol:   # mismo stamp: reemplaza
                self._T[i - 1], self._seg[i - 1], self._meta[i - 1] = c2w, int(segment), meta or {}
                return
            self._t.insert(i, float(stamp))
            self._T.insert(i, c2w)
            self._seg.insert(i, int(segment))
            self._meta.insert(i, meta or {})
            if len(self._t) > self.max_samples:
                k = len(self._t) - self.max_samples
                del self._t[:k], self._T[:k], self._seg[:k], self._meta[:k]

    def add_break(self, stamp):
        """Discontinuidad: no se interpola a través de este instante."""
        with self._lock:
            bisect.insort(self._breaks, float(stamp))

    def _break_between(self, t0, t1):
        i = bisect.bisect_right(self._breaks, t0)
        return i < len(self._breaks) and self._breaks[i] < t1

    def query(self, t) -> PoseQuery:
        t = float(t)
        with self._lock:
            n = len(self._t)
            if n == 0:
                return PoseQuery(QueryKind.NONE)
            i = bisect.bisect_left(self._t, t)        # self._t[i-1] < t <= self._t[i]
            cands = [k for k in (i - 1, i) if 0 <= k < n]
            k_near = min(cands, key=lambda k: abs(self._t[k] - t))
            dt_near = abs(self._t[k_near] - t)
            if dt_near <= self.exact_tol:
                return PoseQuery(QueryKind.EXACT, self._T[k_near].copy(), dt_near, 0.0, self._seg[k_near],
                                 dict(self._meta[k_near]))
            if 0 < i < n:
                t0, t1 = self._t[i - 1], self._t[i]
                gap = t1 - t0
                if (gap <= self.max_gap and self._seg[i - 1] == self._seg[i]
                        and not self._break_between(t0, t1)):
                    a = (t - t0) / gap
                    T = interpolate_pose(self._T[i - 1], self._T[i], a)
                    return PoseQuery(QueryKind.INTERP, T, dt_near, gap, self._seg[i - 1],
                                     {"alpha": a})
            if dt_near <= self.edge_tol:
                return PoseQuery(QueryKind.NEAREST, self._T[k_near].copy(), dt_near, float("nan"),
                                 self._seg[k_near], dict(self._meta[k_near]))
            return PoseQuery(QueryKind.NONE, None, dt_near)

    def query_many(self, ts):
        """Consulta un arreglo de instantes. Devuelve (c2w (N,4,4) con NaN donde NONE, kind, dt, segment)."""
        ts = np.asarray(ts, np.float64)
        T = np.full((len(ts), 4, 4), np.nan)
        kind = np.zeros(len(ts), np.uint8)
        dt = np.full(len(ts), np.inf)
        seg = np.full(len(ts), -1, np.int32)
        for j, t in enumerate(ts):
            q = self.query(t)
            kind[j], dt[j], seg[j] = int(q.kind), q.dt_nearest, q.segment
            if q.ok:
                T[j] = q.c2w
        return T, kind, dt, seg

    def count_between(self, t0, t1):
        """Cuántas muestras hay con t0 < stamp <= t1 (p. ej. ritmo de poses de Stella)."""
        with self._lock:
            return bisect.bisect_right(self._t, t1) - bisect.bisect_right(self._t, t0)

    def arrays(self):
        with self._lock:
            if not self._t:
                return np.zeros(0), np.zeros((0, 4, 4)), np.zeros(0, np.int32)
            return np.array(self._t), np.stack(self._T), np.array(self._seg, np.int32)

    @classmethod
    def from_arrays(cls, stamps, c2w, segments=None, **kw):
        b = cls(**kw)
        segs = np.zeros(len(stamps), int) if segments is None else segments
        for t, T, s in zip(stamps, c2w, segs):
            b.add(t, T, int(s))
        return b
