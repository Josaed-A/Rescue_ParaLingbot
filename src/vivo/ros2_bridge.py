"""Puente ROS2 del mapeo en vivo (integración Stella-VSLAM, etapa 3).

Dos cosas, en un nodo rclpy que gira en su propio hilo, sin tocar el modelo:

  1. "Tee" de la cámara: cada frame CAPTURADO (no solo los que llegan al modelo, que son ~2 por
     segundo) se publica como sensor_msgs/Image bgr8 + CameraInfo con header.stamp = FrameMeta.stamp.
     La cámara se abre una sola vez (webcam y celular no admiten dos lectores): LingBot la consume en
     proceso como siempre y Stella la recibe por ROS2. Los frames sintéticos del analizador de contexto
     nunca pasan por aquí (regla 13).
  2. Adaptador de Stella: suscribe /stella/camera_pose, /stella/keyframes y /stella/tracking_state y
     produce TrackingEstimate(source=STELLA) en la convención de ParaLingbot (cámara OpenCV, c2w).
     Las poses llevan el header.stamp exacto de la imagen que las produjo, así que la asociación con
     el frame que procesó LingBot es exacta cuando Stella procesó ese frame (sin interpolar; la
     interpolación es de la etapa 5).

Conversión de frames (docs/STELLA_INTEGRATION_AUDIT.md § 3.2): el nodo de Stella publica
T_ros = R · T_cv · R⁻¹ con R = [[0,0,1],[-1,0,0],[0,-1,0]]; aquí se deshace: T_cv = R⁻¹ · T_ros · R.

Sin cv_bridge (riesgo R5 de la auditoría): los mensajes se arman con numpy.
"""
import os
import threading
import time
from typing import Optional

import numpy as np

from tracking import FrameMeta, TrackingEstimate, TrackingSource, TrackingStatus
from pose_buffer import PoseBuffer, QueryKind
import frames
from keyframe_correction import KeyframeHistory

# Tolerancias del buffer de Stella, medidas en la etapa 5 (src/mapas/eval_pose_interp.py):
# interpolar entre poses de Stella separadas hasta 0.25 s cuesta 1.9-2.4% del desplazamiento en 1 s y
# 0.3-0.7° (mediana), unas 10 veces menos que el desacuerdo entre trackers (RPE a 1 s, etapa 4); a 0.5 s
# la rotación ya se acerca a ese desacuerdo. edge_tol: ~1.5 frames a 30 fps.
STELLA_MAX_GAP = 0.25
STELLA_EDGE_TOL = 0.05

R_ROS_TO_CV = frames.A_CV_TO_ROS        # = rot_ros_to_cv_map_frame de stella_vslam_ros (etapa 8: frames.py)

STELLA_STATES = {"Initializing": TrackingStatus.INITIALIZING, "Tracking": TrackingStatus.TRACKING,
                 "Lost": TrackingStatus.LOST}


# ---------------------------------------------------------------------------
# utilidades puras (sin ROS): también las usa src/ros/camera_publisher.py y los tests
# ---------------------------------------------------------------------------
def quat_to_mat(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def ros_pose_to_c2w_cv(p, q):
    """(x,y,z), (qx,qy,qz,qw) del frame `map` de ROS -> c2w 4x4 en la convención CV/OpenCV."""
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(*q)
    T[:3, 3] = p
    Rinv = R_ROS_TO_CV.T
    Tcv = np.eye(4)
    Tcv[:3, :3] = Rinv @ T[:3, :3] @ R_ROS_TO_CV
    Tcv[:3, 3] = Rinv @ T[:3, 3]
    return Tcv


def stamp_key(stamp: float) -> int:
    """Clave entera (microsegundos) para asociar stamps que viajaron por ROS2 (sec + nanosec)."""
    return int(round(stamp * 1e6))


def to_time_msg(stamp: float):
    from builtin_interfaces.msg import Time as TimeMsg
    sec = int(np.floor(stamp))
    return TimeMsg(sec=sec, nanosec=int(round((stamp - sec) * 1e9)) % 1_000_000_000)


def image_msg(img_bgr: np.ndarray, stamp: float, frame_id: str):
    from sensor_msgs.msg import Image
    msg = Image()
    msg.header.stamp = to_time_msg(stamp)
    msg.header.frame_id = frame_id
    msg.height, msg.width = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(img_bgr).tobytes()
    return msg


def camera_info_msg(yaml_path: str, frame_id: str):
    """CameraInfo desde el yaml de Stella (Camera: fx fy cx cy k1 k2 p1 p2 k3 cols rows)."""
    import yaml
    from sensor_msgs.msg import CameraInfo
    cam = yaml.safe_load(open(yaml_path))["Camera"]
    ci = CameraInfo()
    ci.header.frame_id = frame_id
    ci.width, ci.height = int(cam["cols"]), int(cam["rows"])
    ci.distortion_model = "plumb_bob"
    ci.d = [float(cam.get(k, 0.0)) for k in ("k1", "k2", "p1", "p2", "k3")]
    fx, fy, cx, cy = (float(cam[k]) for k in ("fx", "fy", "cx", "cy"))
    ci.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ci.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return ci


# ---------------------------------------------------------------------------
# adaptador de Stella (sin ROS: recibe los datos ya extraídos de los mensajes)
# ---------------------------------------------------------------------------
class StellaTrackingProvider:
    """Convierte lo que publica stella_vslam_ros en TrackingEstimate(source=STELLA) y los guarda.

    Estado: el último `tracking_state` recibido (Initializing / Tracking / Lost). Una pose siempre
    llega con estado TRACKING (el nodo solo la publica cuando el tracking tuvo éxito). Si no llega
    ningún estado en `stale_s`, el estado se considera UNKNOWN (Stella no corre o no recibe frames).
    """
    source = TrackingSource.STELLA

    def __init__(self, stale_s: float = 2.0):
        self.lock = threading.Lock()
        self.by_stamp = {}                 # stamp_key -> TrackingEstimate
        self.estimates = []                # en orden de llegada
        self.state = TrackingStatus.UNKNOWN
        self.state_t = 0.0
        self.state_counts = {}
        self.n_keyframes = 0
        self.keyframe_updates = 0          # mensajes de keyframes cuyo número cambió
        self.stale_s = stale_s
        self.last_pose_t = 0.0
        # segmento = mapa de Stella: cuando vuelve a "Initializing" después de haber dado poses, se
        # reinició y su mundo (origen, escala) es otro; las poses de segmentos distintos no se
        # pueden alinear con la misma Sim(3) (etapa 4)
        self.segment = 0
        self._poses_in_segment = 0
        self.events = []                   # (hora local, estado) en cada cambio de estado
        self._last_state_str = None
        self.pose_segments = []
        # etapa 5: las mismas poses en un buffer temporal para consultar pose(t) en cualquier instante
        self.buffer = PoseBuffer(max_gap=STELLA_MAX_GAP, edge_tol=STELLA_EDGE_TOL)
        # etapa 12: keyframes con id y timestamp de imagen; correcciones de BA / loop closure
        self.kf_history = KeyframeHistory(shift_thr=1e-3)
        self.kf_messages = []            # (stamp, filas) de cada versión que movió algo o cambió el conjunto
        self._kf_ids = None

    def on_state(self, state_str: str, t: Optional[float] = None):
        st = STELLA_STATES.get(state_str, TrackingStatus.UNKNOWN)
        with self.lock:
            self.state, self.state_t = st, (t if t is not None else time.time())
            self.state_counts[state_str] = self.state_counts.get(state_str, 0) + 1
            if state_str != self._last_state_str:
                self.events.append((self.state_t, state_str))
                self._last_state_str = state_str
                if state_str == "Initializing" and self._poses_in_segment > 0:
                    self.segment += 1
                    self._poses_in_segment = 0

    def on_pose(self, stamp: float, p, q, t: Optional[float] = None) -> TrackingEstimate:
        c2w = ros_pose_to_c2w_cv(p, q)
        with self.lock:
            seg = self.segment
            self._poses_in_segment += 1
        est = TrackingEstimate(stamp=float(stamp), frame_id=-1, c2w=c2w, source=self.source,
                               status=TrackingStatus.TRACKING,
                               confidence={"n_keyframes": self.n_keyframes, "segment": seg})
        with self.lock:
            self.by_stamp[stamp_key(stamp)] = est
            self.estimates.append(est)
            self.pose_segments.append(seg)
            self.last_pose_t = t if t is not None else time.time()
        self.buffer.add(stamp, c2w, segment=seg)
        return est

    def on_keyframes_full(self, stamp: float, rows):
        """Versión nueva de los keyframes (ts ya en el reloj interno). Devuelve (máx. desplazamiento, ids movidos)."""
        with self.lock:
            ids = frozenset(int(r[0]) for r in rows)
            mx, moved = self.kf_history.update(stamp, rows)
            if moved or ids != self._kf_ids:
                self.kf_messages.append((float(stamp), [list(map(float, r)) for r in rows]))
            self._kf_ids = ids
            return mx, moved

    def on_keyframes(self, n: int):
        with self.lock:
            if n != self.n_keyframes:
                self.keyframe_updates += 1
            self.n_keyframes = n

    def status(self, now: Optional[float] = None) -> TrackingStatus:
        now = time.time() if now is None else now
        with self.lock:
            if self.state_t and now - self.state_t > self.stale_s:
                return TrackingStatus.UNKNOWN
            return self.state

    def estimate_at(self, stamp: float) -> Optional[TrackingEstimate]:
        """La estimación de Stella para el frame con ese stamp exacto (None si Stella no lo procesó
        o lo procesó sin pose: perdido / inicializando)."""
        with self.lock:
            return self.by_stamp.get(stamp_key(stamp))

    def estimate_near(self, stamp: float) -> Optional[TrackingEstimate]:
        """Pose de Stella en el instante `stamp` desde el buffer (etapa 5): exacta, interpolada
        (centro lineal + SLERP entre dos poses del mismo mapa separadas <= 0.25 s) o la más cercana a
        <= 0.05 s. None si no hay ninguna de esas: no se inventa. confidence trae cómo se obtuvo."""
        q = self.buffer.query(stamp)
        if not q.ok:
            return None
        return TrackingEstimate(stamp=float(stamp), frame_id=-1, c2w=q.c2w, source=self.source,
                                status=TrackingStatus.TRACKING,
                                confidence={"assoc": q.kind.name, "dt_nearest": q.dt_nearest,
                                            "gap": q.gap, "segment": q.segment})

    def summary(self):
        with self.lock:
            return {"poses": len(self.estimates), "state": self.state.name, "states": dict(self.state_counts),
                    "keyframes": self.n_keyframes, "keyframe_updates": self.keyframe_updates,
                    "segmentos": self.segment + 1 if self.estimates else 0}

    def arrays(self):
        """Trayectoria completa de Stella (a su propio ritmo) para el .npz de la sesión:
        (stamps, c2w, segmento de cada pose)."""
        with self.lock:
            if not self.estimates:
                return np.zeros(0), np.zeros((0, 4, 4), np.float32), np.zeros(0, np.int32)
            return (np.array([e.stamp for e in self.estimates]),
                    np.stack([e.c2w for e in self.estimates]).astype(np.float32),
                    np.array(self.pose_segments, np.int32))

    def event_arrays(self):
        """Cambios de estado de Stella: (hora local en s, estado como TrackingStatus)."""
        with self.lock:
            return (np.array([t for t, _ in self.events], np.float64),
                    np.array([int(STELLA_STATES.get(s, TrackingStatus.UNKNOWN)) for _, s in self.events], np.uint8))


# ---------------------------------------------------------------------------
# el nodo
# ---------------------------------------------------------------------------
class Ros2Bridge:
    """Nodo rclpy en un hilo propio: publica los frames capturados y recibe las poses de Stella.

    cfg (todas opcionales):
      ros2_image_topic   /paralingbot/camera/image_raw
      ros2_frame_id      paralingbot_camera_optical
      ros2_camera_info   yaml de Stella del que sacar K (opcional; si no hay, no se publica CameraInfo)
      ros2_stella_ns     /stella
      ros2_resize        "WxH" para publicar reescalado (p. ej. 960x540 si el celular da 1280x720)
      ros2_pub_basic     publicar la estimación BASIC como PoseStamped en /paralingbot/tracking/basic_pose
      ros2_pub_every     publicar 1 de cada N frames capturados (1 = todos; 2 con un video a 30 fps imita
                         los 15 fps del celular y deja CPU a Stella cuando comparte máquina con el modelo)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.topic = cfg.get("ros2_image_topic", "/paralingbot/camera/image_raw")
        self.frame_id = cfg.get("ros2_frame_id", "paralingbot_camera_optical")
        self.ns = cfg.get("ros2_stella_ns", "/stella")
        rs = cfg.get("ros2_resize")
        self.resize = tuple(int(v) for v in str(rs).lower().split("x")) if rs else None
        self.stella = StellaTrackingProvider()
        self.pub_every = max(1, int(cfg.get("ros2_pub_every", 1) or 1))
        self.n_seen = 0
        self.n_pub = 0
        self.t_last_pub = 0.0
        self._lock = threading.Lock()
        self._node = None
        self._exec = None
        self._thread = None
        # Origen de tiempo hacia ROS2 (etapa 8). Carpetas y videos tienen stamps que empiezan en 0, y en
        # tf2 el tiempo 0 significa "la transformación más reciente": el primer frame se buscaba mal. Para
        # esas fuentes, todo lo que sale a ROS2 (imágenes, poses, TF) se corre con la hora de arranque y
        # a lo que vuelve de Stella se le resta, así los stamps internos del repo no cambian. Las cámaras
        # en vivo ya usan la hora del reloj (offset 0).
        self.stamp_offset = time.time() if cfg.get("source") in ("folder", "video") else 0.0
        self._ci = None
        self._pub_basic = self._pub_ref = None
        self._tf = self._static_tf = self._pub_align = None
        self.ok = False
        self.error = None

    # -- ciclo de vida ------------------------------------------------------
    def start(self):
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            from sensor_msgs.msg import Image, CameraInfo
            from nav_msgs.msg import Odometry
            from geometry_msgs.msg import PoseArray, PoseStamped
            from std_msgs.msg import String
        except Exception as e:                                    # ROS2 no disponible: el servidor sigue
            self.error = f"ROS2 no disponible: {e}"
            return False
        if not rclpy.ok():
            # sin los manejadores de señales de rclpy: si no, reemplazan los de Python y un SIGINT/SIGTERM
            # sólo apaga ROS2 mientras la sesión sigue corriendo (etapa 17, robustez)
            try:
                from rclpy.signals import SignalHandlerOptions
                rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
            except ImportError:
                rclpy.init(args=None)
        self._node = Node("paralingbot_live")
        img_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self._pub_img = self._node.create_publisher(Image, self.topic, img_qos)
        self._pub_ci = self._node.create_publisher(CameraInfo, self.topic.rsplit("/", 1)[0] + "/camera_info", img_qos)
        self._pub_basic = (self._node.create_publisher(PoseStamped, "/paralingbot/tracking/basic_pose", 10)
                           if self.cfg.get("ros2_pub_basic", True) else None)
        self._pub_ref = self._node.create_publisher(PoseStamped, "/paralingbot/tracking/reference_pose", 10)
        # etapa 8: TF2 (REP-105) y alineación Stella -> ParaLingbot (Sim3, latcheada)
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
        from rclpy.qos import DurabilityPolicy
        self._tf = TransformBroadcaster(self._node)
        self._static_tf = StaticTransformBroadcaster(self._node)
        self._publish_static_tf()
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub_align = self._node.create_publisher(String, "/paralingbot/alignment/stella", latched)
        ci_yaml = self.cfg.get("ros2_camera_info")
        if ci_yaml and os.path.isfile(ci_yaml):
            self._ci = camera_info_msg(ci_yaml, self.frame_id)
        self._node.create_subscription(Odometry, f"{self.ns}/camera_pose", self._on_pose, 20)
        self._node.create_subscription(PoseArray, f"{self.ns}/keyframes", self._on_kf, 5)
        self._node.create_subscription(String, f"{self.ns}/tracking_state", self._on_state, 20)
        from std_msgs.msg import Float64MultiArray
        from rclpy.qos import qos_profile_sensor_data
        self._node.create_subscription(Float64MultiArray, f"{self.ns}/keyframes_full", self._on_kf_full, qos_profile_sensor_data)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True, name="ros2_bridge")
        self._thread.start()
        self.ok = True
        return True

    def _spin(self):
        while not self._stop.is_set():
            try:
                self._exec.spin_once(timeout_sec=0.1)
            except Exception as e:                                # no tumbar el mapeo por el puente
                self.error = f"{type(e).__name__}: {e}"
                time.sleep(0.1)

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._exec.remove_node(self._node)
            self._node.destroy_node()
        except Exception:
            pass
        self.ok = False

    # -- callbacks ROS -------------------------------------------------------
    def _on_pose(self, m):
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 - self.stamp_offset
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.stella.on_pose(st, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w))

    def _on_kf(self, m):
        self.stella.on_keyframes(len(m.poses))

    def _on_state(self, m):
        self.stella.on_state(m.data)

    def _on_kf_full(self, m):
        d = np.asarray(m.data, np.float64)
        if len(d) < 2:
            return
        n = int(d[1])
        rows = d[2:2 + 9 * n].reshape(n, 9).copy() if n else np.zeros((0, 9))
        rows[:, 1] -= self.stamp_offset            # timestamp de imagen del keyframe -> reloj interno
        self.stella.on_keyframes_full(float(d[0]) - self.stamp_offset, rows)

    # -- hacia afuera --------------------------------------------------------
    def publish_frame(self, img_bgr: np.ndarray, meta: FrameMeta, prep=None):
        """Llamado por la fuente por cada frame capturado (en el hilo de la cámara). `prep` (la
        rotación de la fuente) se aplica sólo a los frames que se publican."""
        if not self.ok:
            return
        self.n_seen += 1
        if (self.n_seen - 1) % self.pub_every:
            return
        try:
            if prep is not None:
                img_bgr = prep(img_bgr)
            if self.resize is not None and (img_bgr.shape[1], img_bgr.shape[0]) != self.resize:
                import cv2
                img_bgr = cv2.resize(img_bgr, self.resize, interpolation=cv2.INTER_AREA)
            msg = image_msg(img_bgr, meta.stamp + self.stamp_offset, self.frame_id)
            self._pub_img.publish(msg)
            if self._ci is not None:
                self._ci.header.stamp = msg.header.stamp
                self._pub_ci.publish(self._ci)
            with self._lock:
                self.n_pub += 1
                self.t_last_pub = time.time()
        except Exception as e:
            self.error = f"publicando frame: {e}"

    def publish_basic(self, est: TrackingEstimate):
        self._publish_pose(self._pub_basic, est, "basic_pose")

    def publish_reference(self, est: TrackingEstimate):
        """Pose de referencia del selector (etapa 6) en /paralingbot/tracking/reference_pose."""
        self._publish_pose(self._pub_ref, est, "reference_pose")

    def _publish_pose(self, pub, est: TrackingEstimate, what: str):
        """PoseStamped de camera_link en paralingbot_map, con ejes ROS (REP-103; etapa 8). Antes de la
        etapa 8 se publicaba la c2w interna (ejes OpenCV) con frame_id paralingbot_map, que no es lo
        que una herramienta ROS espera de un frame `map`."""
        if not self.ok or pub is None:
            return
        try:
            from geometry_msgs.msg import PoseStamped
            msg = PoseStamped()
            msg.header.stamp = to_time_msg(est.stamp + self.stamp_offset)
            msg.header.frame_id = frames.MAP
            T = frames.c2w_cv_to_map_link(est.c2w)
            q = frames.mat_to_quat_xyzw(T[:3, :3])
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in T[:3, 3])
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = \
                (float(v) for v in q)
            pub.publish(msg)
        except Exception as e:
            self.error = f"publicando {what}: {e}"

    # -- TF2 (etapa 8) --------------------------------------------------------
    def _tf_msg(self, T, parent, child, stamp=None):
        from geometry_msgs.msg import TransformStamped
        m = TransformStamped()
        if stamp is None:
            m.header.stamp = self._node.get_clock().now().to_msg()
        else:
            m.header.stamp = to_time_msg(stamp)
        m.header.frame_id, m.child_frame_id = parent, child
        q = frames.mat_to_quat_xyzw(T[:3, :3])
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in T[:3, 3])
        m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = \
            (float(v) for v in q)
        return m

    def _publish_static_tf(self):
        """Transformaciones fijas: map -> odom (identidad: no hay odometría propia), map -> map_cv (el
        mundo interno con ejes OpenCV) y camera_link -> camera_optical (rotación estándar REP-103)."""
        self._static_tf.sendTransform([
            self._tf_msg(np.eye(4), frames.MAP, frames.ODOM),
            self._tf_msg(frames.T_MAP_MAPCV, frames.MAP, frames.MAP_CV),
            self._tf_msg(frames.T_LINK_OPTICAL, frames.CAMERA_LINK, frames.CAMERA_OPTICAL),
        ])

    def publish_reference_tf(self, est: TrackingEstimate):
        """odom -> camera_link con la pose de referencia, sellado con el stamp de adquisición del
        frame (no con la hora actual: un consumidor TF que busque el instante de una imagen la encuentra)."""
        if not self.ok or self._tf is None:
            return
        try:
            self._tf.sendTransform(self._tf_msg(frames.c2w_cv_to_map_link(est.c2w), frames.ODOM,
                                                frames.CAMERA_LINK, est.stamp + self.stamp_offset))
        except Exception as e:
            self.error = f"publicando TF: {e}"

    def publish_stella_alignment(self, segment, s, R, t):
        """Sim(3) que lleva el mundo de Stella (mapa `segment`) al de ParaLingbot, en ejes ROS:
        p_map = escala · R · p_stella_map + t. Va por un topic propio y NO por TF, porque TF es rígido:
        publicarla como TF escondería la escala (regla del plan). JSON en std_msgs/String, latcheado."""
        if not self.ok or self._pub_align is None:
            return
        try:
            import json
            from std_msgs.msg import String
            s2, R2, t2 = frames.sim3_cv_to_ros(s, R, t)
            self._pub_align.publish(String(data=json.dumps({
                "parent": frames.MAP, "child": frames.STELLA_MAP, "segmento": int(segment),
                "escala": float(s2), "R": np.asarray(R2).round(9).tolist(), "t": np.asarray(t2).round(9).tolist(),
                "formula": "p_parent = escala * R @ p_child + t"})))
        except Exception as e:
            self.error = f"publicando alineación: {e}"

    def stella_estimate(self, stamp: float) -> Optional[TrackingEstimate]:
        """Pose de Stella para el frame con ese stamp (buffer de la etapa 5)."""
        return self.stella.estimate_near(stamp)

    def summary(self):
        with self._lock:
            s = {"ok": self.ok, "error": self.error, "frames_capturados": self.n_seen, "frames_publicados": self.n_pub,
                 "stamp_offset_ros": self.stamp_offset,
                 "pub_every": self.pub_every, "topic": self.topic}
        s["stella"] = self.stella.summary()
        s["stella"]["status"] = self.stella.status().name
        return s


# ---------------------------------------------------------------------------
# Stella como subproceso (opcional): src/ros/run_stella.sh <config>
# ---------------------------------------------------------------------------
class StellaProcess:
    """Arranca `src/ros/run_stella.sh` en su propio grupo de procesos y lo cierra con SIGINT al
    grupo entero (run_stella.sh hace exec de `ros2 run`, que lanza `run_slam` como hijo: la señal
    tiene que llegar al hijo, si no queda huérfano publicando en los mismos topics)."""

    def __init__(self, config: str, image_topic: str, ns: str = "/stella", log_dir: Optional[str] = None,
                 cpus: Optional[str] = None):
        import subprocess
        self.subprocess = subprocess
        self.cpus = cpus                  # p. ej. "4-11": taskset (etapa 6, ver cpu_affinity.py)
        self.config = config
        self.image_topic = image_topic
        self.ns = ns
        self.log_dir = log_dir
        self.proc = None
        self.log_path = None

    def start(self):
        here = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(os.path.dirname(here), "ros", "run_stella.sh")
        if not os.path.isfile(self.config):
            raise FileNotFoundError(f"config de Stella no encontrado: {self.config}")
        log_dir = self.log_dir or os.path.join(os.path.dirname(here), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"stella_{time.strftime('%Y%m%d_%H%M%S')}.log")
        cmd = [script, self.config, "--image", self.image_topic]
        if self.cpus:
            cmd = ["taskset", "-c", str(self.cpus)] + cmd
        if self.ns != "/stella":
            cmd += ["--", "--ros-args", "-r", f"__node:={self.ns.strip('/')}"]
        self.proc = self.subprocess.Popen(cmd, stdout=open(self.log_path, "w"), stderr=self.subprocess.STDOUT,
                                          stdin=self.subprocess.DEVNULL, start_new_session=True)
        return self

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float = 30.0):
        import signal
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGINT)          # a todo el grupo: ros2 run + run_slam
        except ProcessLookupError:
            return
        t0 = time.time()
        while time.time() - t0 < timeout:
            # run_slam puede seguir vivo aunque el envoltorio haya salido: mirar el grupo entero
            try:
                os.killpg(self.proc.pid, 0)
            except ProcessLookupError:
                break
            if self.proc.poll() is not None and not self._group_alive():
                break
            time.sleep(0.25)
        if self._group_alive():
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=2)
        except Exception:
            pass

    def _group_alive(self):
        """¿Queda algún proceso en el grupo (pgid = pid del envoltorio)?"""
        pgid = self.proc.pid
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/stat") as fh:
                    fields = fh.read().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid:            # campo 5 del stat = pgrp
                    return True
            except Exception:
                continue
        return False
