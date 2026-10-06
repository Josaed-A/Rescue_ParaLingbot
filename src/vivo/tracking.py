"""Representación común del seguimiento de cámara (integración Stella-VSLAM, etapas 1 y 3).

Qué hay acá y por qué:

  FrameMeta          hora de captura e identificador de cada frame. Viaja junto a la imagen
                     desde la fuente (FrameSource / AndroidCamera) hasta el modelo, pasando por
                     el analizador de contexto (que puede elegir un frame anterior al último
                     leído: por eso el stamp tiene que ir pegado a la imagen y no leerse "ahora").
  TrackingEstimate   una pose con tiempo, fuente, estado y confianza. Es lo que comparan y
                     eligen los modos BASIC / STELLA / HYBRID.
  BasicTrackingProvider
                     el tracking actual de ParaLingbot: la pose que la cabeza de cámara de
                     LingBot-Map emite para cada frame (ver docs/STELLA_INTEGRATION_AUDIT.md § 2).
                     No calcula nada nuevo: envuelve lo que LiveSession.run_model ya tiene
                     (c2w, confianza de profundidad, movimiento y nitidez del analizador) para
                     que exista una interfaz estable sin tocar demo.py ni lingbot_map/.

Convenciones (docs/STELLA_INTEGRATION_AUDIT.md § 3, docs/MATEMATICA.md § 1): la pose es c2w
(cámara -> mundo) 4x4, cámara OpenCV (x derecha, y abajo, z adelante), mundo = cámara del
primer frame de la sesión, unidades del modelo (no metros).

Sobre el estado del tracking BASIC: el modelo siempre devuelve una pose, no tiene noción de
"perdido". Por eso el estado BASIC es TRACKING siempre que hubo pose, y la confianza son las
señales observables reales (media de depth_conf del frame, movimiento en píxeles, nitidez),
no una métrica inventada. Cuándo esas señales significan "pose dudosa" se decide en la etapa
de comparación con Stella, con datos.
"""
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Optional

import numpy as np


class TrackingSource(IntEnum):
    BASIC = 0       # cabeza de cámara de LingBot-Map (el tracking actual)
    STELLA = 1      # Stella-VSLAM (etapa 3)
    HYBRID = 2      # selección / fusión (etapa 6)


class TrackingStatus(IntEnum):
    UNKNOWN = 0
    TRACKING = 1
    LOST = 2
    INITIALIZING = 3
    RELOCALIZING = 4


@dataclass
class FrameMeta:
    """Identidad temporal de un frame de la fuente.

    stamp      segundos. En cámaras en vivo es time.time() al recibir el frame (el mismo reloj
               que usará ROS2); en carpeta/video es sintético: índice / fps nominal, desde 0.
    frame_id   contador de la fuente (crece con cada frame capturado, no con cada frame que
               llega al modelo). -1 en frames sintéticos del analizador de contexto.
    synthetic  True si la imagen la generó el analizador (solo contexto; nunca se registra).
    motion_px  movimiento acumulado (mediana del flujo óptico, px) desde el frame anterior que
               entró al modelo; lo rellena ContextGate. None si el analizador está apagado.
    sharpness  varianza del Laplaciano a 256 px; lo rellena ContextGate.
    """
    stamp: float
    frame_id: int
    synthetic: bool = False
    motion_px: Optional[float] = None
    sharpness: Optional[float] = None
    wall: Optional[float] = None     # hora del reloj (time.time()) en que la fuente entregó el frame: latencia

    @staticmethod
    def between(a: "FrameMeta", b: "FrameMeta", t: float) -> "FrameMeta":
        """Meta de un frame sintético situado en la fracción t (0..1) entre a y b."""
        return FrameMeta(stamp=a.stamp + (b.stamp - a.stamp) * t, frame_id=-1, synthetic=True)


@dataclass
class TrackingEstimate:
    stamp: float
    frame_id: int
    c2w: np.ndarray                       # 4x4 float64, cámara -> mundo
    source: TrackingSource
    status: TrackingStatus
    confidence: dict = field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        return self.c2w[:3, 3]

    def to_json(self) -> dict:
        """Para el canal WebSocket / logs (sin la matriz completa, que ya viaja en binario).
        JSON estricto: NaN e infinito pasan a null (el JSON.parse del navegador rechaza NaN), los
        textos (p. ej. el tipo de asociación) quedan como texto."""
        def val(v):
            if v is None or isinstance(v, (str, bool)):
                return v
            f = float(v)
            return round(f, 4) if np.isfinite(f) else None
        return {"stamp": float(self.stamp), "frame_id": int(self.frame_id),
                "source": self.source.name, "status": self.status.name,
                "position": [round(float(v), 5) for v in self.position],
                **{k: val(v) for k, v in self.confidence.items()}}


class BasicTrackingProvider:
    """El tracking actual (pose de LingBot-Map) con la interfaz común.

    Uso, desde LiveSession.run_model, por cada frame real emitido:
        est = provider.estimate(meta, c2w, depth_conf)
    """
    source = TrackingSource.BASIC

    def __init__(self):
        self.count = 0
        self.last: Optional[TrackingEstimate] = None

    def reset(self):
        self.count = 0
        self.last = None

    def estimate(self, meta: FrameMeta, c2w: np.ndarray, depth_conf: Optional[np.ndarray]) -> TrackingEstimate:
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape == (3, 4):
            m = np.eye(4)
            m[:3, :4] = c2w
            c2w = m
        conf = {"conf_mean": None, "conf_p50": None, "motion_px": meta.motion_px, "sharpness": meta.sharpness,
                "step": None}
        if depth_conf is not None and depth_conf.size:
            dc = np.asarray(depth_conf, dtype=np.float32)
            conf["conf_mean"] = float(dc.mean())
            conf["conf_p50"] = float(np.median(dc))
        if self.last is not None:
            conf["step"] = float(np.linalg.norm(c2w[:3, 3] - self.last.c2w[:3, 3]))
        est = TrackingEstimate(stamp=float(meta.stamp), frame_id=int(meta.frame_id), c2w=c2w,
                               source=self.source, status=TrackingStatus.TRACKING, confidence=conf)
        self.count += 1
        self.last = est
        return est


def c2w_from_w2c(w2c_3x4: np.ndarray) -> np.ndarray:
    """Convención del repo: lo guardado en `extrinsic` es w2c; el mundo sale de invertirlo."""
    e = np.eye(4)
    e[:3, :4] = np.asarray(w2c_3x4, dtype=np.float64)
    return np.linalg.inv(e)
