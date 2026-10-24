"""Selector de la pose de referencia: BASIC, STELLA o HYBRID (integración Stella-VSLAM, etapa 6).

Primera versión del plan: comparación + selección, sin fusión matemática ni optimización. Se
actualiza una vez por frame del modelo (LingBot), que es donde se registra la geometría, con la pose
BASIC de ese frame y la pose de Stella del mismo instante (PoseBuffer, etapa 5). Todo el criterio usa
estados observables reales; ninguna "confianza" inventada (regla 8).

Modos (la pose de referencia queda siempre en el mundo y la escala de BASIC: `paralingbot_map`):

  BASIC   la pose de LingBot tal cual. Es el baseline y el comportamiento sin Stella.

  STELLA  Stella como fuente: su pose absoluta llevada al mundo BASIC con una Sim(3) que se fija al
          empezar cada mapa (segmento) de Stella, en el primer frame en que ya hay escala. Conserva lo
          que Stella tiene de global (relocalización, loop closure). Mientras Stella no da pose
          válida, la referencia avanza con los pasos de BASIC (degradación controlada, regla 15); al
          volver Stella al mismo mapa, la referencia vuelve a su pose absoluta (puede saltar: es la
          corrección de Stella).

  HYBRID  la referencia se encadena paso a paso: en cada frame se toma el movimiento relativo de
          Stella entre el frame anterior y este si es válido, y si no el de BASIC. La escala del
          paso de Stella se lleva a la de BASIC con el cociente de largos de paso en una ventana
          reciente (mediana). Así la escala la sigue fijando BASIC (la deriva de escala de Stella sin
          CPU, medida en la etapa 4, no se propaga) y Stella aporta la forma y la orientación del
          movimiento, que en la etapa 4 derivaban menos que en el streaming. No salta al cambiar de
          fuente.

Un paso de Stella es válido (si no, se usa el de BASIC y se anota el motivo):
  - Stella en TRACKING (estado real del nodo, etapa 2);
  - pose de Stella en este frame y en el anterior, del mismo mapa;
  - escala conocida: al menos `min_scale_pairs` pasos con movimiento en la ventana;
  - el largo del paso de Stella, ya en escala BASIC, no difiere del de BASIC más de `step_ratio_max`
    veces (salto o reinicio de escala), cuando BASIC se movió lo suficiente para medirlo;
  - el giro relativo de Stella no difiere del de BASIC más de `rot_disagree_max_deg` (fallas
    groseras: relocalización a otro lugar, mapa equivocado). Es generoso a propósito: BASIC deriva
    en rotación y no se quiere rechazar a Stella por eso;
  - Stella no está famélica: su ritmo de poses en los últimos `rate_window_s` es de al menos
    `min_pose_rate_hz` (si se pasa `stella_rate_hz`). Medido en la etapa 6: Stella aislada da ~20
    poses/s y no empeora el recorrido; en vivo sin CPU da 3-8 poses/s y lo empeora (error de forma
    contra el croquis del fablab 2.04% -> 2.77%). El umbral por defecto (10/s) es PROVISORIO: hay que
    recalibrarlo con Stella en núcleos reservados.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from tracking import TrackingEstimate, TrackingSource, TrackingStatus


class TrackingMode(str, Enum):
    BASIC = "basic"
    STELLA = "stella"
    HYBRID = "hybrid"


@dataclass
class HybridParams:
    scale_window_s: float = 6.0       # ventana para el cociente de largos de paso BASIC / Stella
    min_scale_pairs: int = 4          # pasos con movimiento necesarios para conocer la escala
    min_motion_rel: float = 0.15      # un paso cuenta como movimiento si supera esta fracción del paso BASIC mediano
    step_ratio_max: float = 2.5       # salto de escala tolerado en un paso
    rot_disagree_max_deg: float = 20.0
    max_dt_s: float = 1.5             # pasos más largos que esto (huecos del modelo) no estiman escala
    min_pose_rate_hz: float = 10.0    # PROVISORIO, ver docstring
    rate_window_s: float = 2.0
    # etapa 9: escala del modo STELLA. "anchor" = la del momento de anclar cada mapa (fija);
    # "window" = la escala por ventana (la misma de HYBRID) re-estimada en cada frame, re-anclando
    # la traslación sobre la pose de Stella del frame ANTERIOR para no borrar sus saltos de
    # relocalización. Medido en src/mapas/scale_alignment.py: una Sim(3) fijada al inicio
    # extrapola 2-4 veces peor que una re-estimada por ventana.
    stella_scale: str = "anchor"


def _rel(T0, T1):
    return np.linalg.inv(T0) @ T1


def _rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


class ReferenceTracker:
    """Una instancia por sesión. `update()` una vez por frame del modelo, en orden de stamp."""

    def __init__(self, mode=TrackingMode.HYBRID, params: Optional[HybridParams] = None):
        self.mode = TrackingMode(mode)
        self.p = params or HybridParams()
        self.reset()

    def reset(self):
        self.prev_t = None
        self.prev_basic = None
        self.prev_stella = None            # (c2w, segment) o None
        self.ref = None                    # pose de referencia actual (c2w, mundo BASIC)
        self.pairs = deque()               # (t, |paso basic|, |paso stella|, mapa) con movimiento
        self.basic_steps = deque(maxlen=200)
        self.anchor = {}                   # segmento de Stella -> (s, R, t) para el modo STELLA
        self.counts = {"BASIC": 0, "STELLA": 0}
        self.reasons = {}

    # -- escala ---------------------------------------------------------------
    def scale(self, t, segment):
        """Escala Stella -> BASIC del mapa `segment`: cada mapa de Stella tiene su propia escala, así
        que sólo cuentan los pasos de ese mapa (tras un reinicio se vuelve a estimar desde cero)."""
        while self.pairs and self.pairs[0][0] < t - self.p.scale_window_s:
            self.pairs.popleft()
        r = [b / s for _, b, s, sg in self.pairs if sg == segment and s > 0]
        if len(r) < self.p.min_scale_pairs:
            return None
        return float(np.median(r))

    # -- actualización -------------------------------------------------------
    def update(self, t, basic_c2w, stella_c2w=None, stella_segment=-1,
               stella_status=TrackingStatus.UNKNOWN, stella_rate_hz=None) -> TrackingEstimate:
        Tb = np.asarray(basic_c2w, np.float64)
        Ts = None if stella_c2w is None else np.asarray(stella_c2w, np.float64)
        info = {"usado": "BASIC", "motivo": None, "escala": None, "ratio_paso": None, "rot_desacuerdo_deg": None,
                "ritmo_stella_hz": stella_rate_hz}

        if self.prev_t is None:                       # primer frame: la referencia arranca en BASIC
            self.ref = Tb.copy()
            info["motivo"] = "primer_frame"
            return self._finish(t, Tb, Ts, stella_segment, info)

        dt = t - self.prev_t
        db = _rel(self.prev_basic, Tb)
        nb = float(np.linalg.norm(db[:3, 3]))
        self.basic_steps.append(nb)
        med_b = float(np.median(self.basic_steps)) if self.basic_steps else 0.0

        ds = None
        same_map = (Ts is not None and self.prev_stella is not None
                    and self.prev_stella[1] == stella_segment)
        if same_map:
            ds = _rel(self.prev_stella[0], Ts)
            ns = float(np.linalg.norm(ds[:3, 3]))
            moving = med_b > 0 and nb > self.p.min_motion_rel * med_b and ns > 0
            if moving and dt <= self.p.max_dt_s and stella_status == TrackingStatus.TRACKING:
                self.pairs.append((t, nb, ns, stella_segment))
        s = self.scale(t, stella_segment)
        info["escala"] = s

        # ¿el paso de Stella es usable?
        why = None
        if stella_status != TrackingStatus.TRACKING:
            why = "stella_" + TrackingStatus(stella_status).name.lower()
        elif Ts is None:
            why = "stella_sin_pose"
        elif stella_rate_hz is not None and stella_rate_hz < self.p.min_pose_rate_hz:
            why = "stella_famelica"
        elif not same_map:
            why = "sin_paso_previo_mismo_mapa"
        elif s is None:
            why = "sin_escala"
        else:
            ns_b = s * float(np.linalg.norm(ds[:3, 3]))
            if med_b > 0 and nb > self.p.min_motion_rel * med_b:
                ratio = ns_b / nb if nb > 0 else np.inf
                info["ratio_paso"] = ratio
                if not (1.0 / self.p.step_ratio_max <= ratio <= self.p.step_ratio_max):
                    why = "salto_de_escala"
            rd = _rot_deg(db[:3, :3].T @ ds[:3, :3])
            info["rot_desacuerdo_deg"] = rd
            if why is None and rd > self.p.rot_disagree_max_deg:
                why = "desacuerdo_de_rotacion"
        stella_ok = why is None

        if self.mode == TrackingMode.BASIC:
            self.ref = Tb.copy()
            info["usado"] = "BASIC"
            info["motivo"] = "modo_basic" if stella_ok else why
        elif self.mode == TrackingMode.HYBRID:
            step = db
            if stella_ok:
                step = ds.copy()
                step[:3, 3] *= s
                info["usado"] = "STELLA"
            else:
                info["motivo"] = why
            self.ref = self.ref @ step
        else:                                          # STELLA
            anchored = stella_segment in self.anchor
            if stella_ok and not anchored:
                # Sim(3) del mapa de Stella al mundo BASIC, fijada en este frame: continuidad exacta
                R = self.ref[:3, :3] @ Ts[:3, :3].T
                tr = self.ref[:3, 3] - s * R @ Ts[:3, 3]
                self.anchor[stella_segment] = (s, R, tr)
                anchored = True
            if Ts is not None and anchored and stella_status == TrackingStatus.TRACKING and why in (None, "sin_escala"):
                sa, R, tr = self.anchor[stella_segment]
                if (self.p.stella_scale == "window" and s is not None and same_map
                        and abs(s - sa) > 1e-9 * max(abs(sa), 1.0)):
                    # re-escalar alrededor de la pose de Stella del frame anterior (continuidad con
                    # la referencia anterior, sin tocar el salto entre el frame anterior y este)
                    Tp = self.prev_stella[0]
                    p_prev = sa * R @ Tp[:3, 3] + tr
                    tr = p_prev - s * R @ Tp[:3, 3]
                    sa = s
                    self.anchor[stella_segment] = (sa, R, tr)
                info["escala_stella"] = sa
                T = np.eye(4)
                T[:3, :3] = R @ Ts[:3, :3]
                T[:3, 3] = sa * R @ Ts[:3, 3] + tr
                self.ref = T
                info["usado"] = "STELLA"
            else:
                self.ref = self.ref @ db
                info["motivo"] = why or "mapa_sin_anclar"
        return self._finish(t, Tb, Ts, stella_segment, info)

    def _finish(self, t, Tb, Ts, seg, info):
        self.prev_t, self.prev_basic = t, Tb
        self.prev_stella = None if Ts is None else (Ts, seg)
        self.counts[info["usado"]] = self.counts.get(info["usado"], 0) + 1
        if info["motivo"]:
            self.reasons[info["motivo"]] = self.reasons.get(info["motivo"], 0) + 1
        src = {TrackingMode.BASIC: TrackingSource.BASIC, TrackingMode.STELLA: TrackingSource.STELLA,
               TrackingMode.HYBRID: TrackingSource.HYBRID}[self.mode]
        return TrackingEstimate(stamp=float(t), frame_id=-1, c2w=self.ref.copy(), source=src,
                                status=TrackingStatus.TRACKING, confidence=info)

    def summary(self):
        n = sum(self.counts.values())
        return {"modo": self.mode.value, "frames": n, "usado": dict(self.counts),
                "fraccion_stella": round(self.counts.get("STELLA", 0) / max(n, 1), 3),
                "motivos_basic": dict(sorted(self.reasons.items(), key=lambda kv: -kv[1]))}
