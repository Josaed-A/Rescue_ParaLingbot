#!/usr/bin/env python3
"""Publica en ROS2 los frames de cualquier fuente de ParaLingbot, con su hora de captura.

Reutiliza `FrameSource` de src/vivo/live_server.py (carpeta, video, webcam, URL, celular
por adb), así Stella-VSLAM ve exactamente las mismas imágenes, con la misma rotación y el mismo
`FrameMeta` (stamp + frame_id) que LingBot (docs/STELLA_INTEGRATION.md, etapa 2). Es el nodo de
cámara para correr Stella aislado; el "tee" dentro del servidor en vivo llega en la etapa 3.

Mensajes: sensor_msgs/Image (bgr8) en --topic y sensor_msgs/CameraInfo en <topic sin
/image_raw>/camera_info, con header.stamp = FrameMeta.stamp y header.frame_id = --frame_id.
Sin cv_bridge (ver riesgo R5 de la auditoría): el mensaje se arma con numpy.

Stamps: en cámaras en vivo son time.time() al recibir el frame. En carpeta/video son sintéticos
(índice / --source_fps desde 0); con --stamp_origin now se les suma la hora de arranque, para que
ROS2 (TF, rosbag) vea tiempos actuales; los stamps relativos se conservan en el registro.

Registro: --log escribe un JSONL con una línea por frame publicado
{"i", "frame_id", "stamp", "stamp_rel", "t_pub"} para asociar después con las poses de Stella.

Ejemplos:
  python3 src/ros/camera_publisher.py --source folder --path captures/.../frames --source_fps 10 \
      --camera_info src/ros/stella/android_960x540.yaml --log /tmp/pub.jsonl
  python3 src/ros/camera_publisher.py --source webcam --device 0 --camera_info src/ros/stella/webcam_640x480.yaml
  python3 src/ros/camera_publisher.py --source android --serial <serial> --camera_id 0 --rotation 90 --cam_size 960x540
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src/vivo"))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # noqa: E402
from sensor_msgs.msg import Image, CameraInfo  # noqa: E402

from live_server import FrameSource  # noqa: E402


# Los mensajes se arman con las mismas funciones que usa el puente del servidor en vivo
# (src/vivo/ros2_bridge.py): una sola implementación.
from ros2_bridge import to_time_msg, image_msg, camera_info_msg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="folder", choices=["folder", "video", "webcam", "url", "android"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--camera_id", default="0")
    ap.add_argument("--rotation", type=int, default=0, help="90 con el celular en vertical (igual que el panel)")
    ap.add_argument("--cam_size", default="960x540")
    ap.add_argument("--cam_fps", type=int, default=15)
    ap.add_argument("--source_fps", type=float, default=10.0, help="fps nominal de una carpeta (stamps sintéticos)")
    ap.add_argument("--rate", type=float, default=None,
                    help="ritmo de publicación para carpeta/video (por defecto = fps nominal; 0 = sin pausa)")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--resize", default=None, help="WxH: reescalar antes de publicar (p. ej. 540x960 para un video 1080x1920)")
    ap.add_argument("--skip_frames", type=int, default=0, help="descartar los primeros N frames de la fuente (carpeta/video)")
    ap.add_argument("--topic", default="/paralingbot/camera/image_raw")
    ap.add_argument("--frame_id", default="paralingbot_camera_optical")
    ap.add_argument("--camera_info", default=None, help="yaml de Stella del que sacar K (opcional)")
    ap.add_argument("--stamp_origin", default="now", choices=["now", "zero"],
                    help="carpeta/video: sumar la hora de arranque a los stamps sintéticos (now) o dejarlos desde 0")
    ap.add_argument("--qos", default="reliable", choices=["reliable", "sensor"],
                    help="reliable: no se pierde ningún frame al reproducir; sensor: best effort, como una cámara")
    ap.add_argument("--log", default=None, help="JSONL con un registro por frame publicado")
    a = ap.parse_args()

    rclpy.init()
    node = Node("paralingbot_camera_publisher")
    if a.qos == "sensor":
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    else:
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
    pub = node.create_publisher(Image, a.topic, qos)
    info_topic = a.topic.rsplit("/", 1)[0] + "/camera_info"
    pub_info = node.create_publisher(CameraInfo, info_topic, qos)
    ci = camera_info_msg(a.camera_info, a.frame_id) if a.camera_info else None

    src = FrameSource(a.source, path=a.path, device=a.device, max_frames=a.max_frames, serial=a.serial,
                      camera_id=a.camera_id, rotation=a.rotation, cam_size=a.cam_size, cam_fps=a.cam_fps,
                      source_fps=a.source_fps)
    import cv2
    resize = tuple(int(v) for v in a.resize.lower().split("x")) if a.resize else None
    # carpetas (índice / fps) y videos (PTS del contenedor, desde la etapa 4) son reproducciones: se
    # publican a su ritmo. Antes sólo se reconocía "synthetic" y, al pasar los videos a PTS, se
    # publicaban a toda velocidad (encontrado en la etapa 9: Stella procesaba unos pocos frames).
    synthetic = src.stamp_kind in ("synthetic", "video_pts")
    origin = time.time() if (synthetic and a.stamp_origin == "now") else 0.0
    rate = a.rate if a.rate is not None else (src._src_fps if synthetic else 0.0)
    period = 1.0 / rate if rate and synthetic else 0.0
    # con un video y sin --rate explícito, el ritmo lo marca el PTS de cada frame (tasa variable)
    pace_by_stamp = src.stamp_kind == "video_pts" and a.rate is None
    t_wall0, t_stamp0 = None, None
    log = open(a.log, "w") if a.log else None
    node.get_logger().info(f"fuente {a.source} ({src.stamp_kind}) -> {a.topic} [{a.qos}]"
                           + (f", {rate:.1f} fps" if period else "") + (f", K de {a.camera_info}" if ci else ""))

    n_pub, n_read, t0 = 0, 0, time.time()
    try:
        for _ in range(a.skip_frames):
            if src.read() is None:
                break
        while rclpy.ok():
            tick = time.time()
            r = src.read()
            if r is None:
                break
            n_read += 1
            if (n_read - 1) % a.stride:
                continue
            img, meta = r
            if resize is not None and (img.shape[1], img.shape[0]) != resize:
                img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
            stamp = meta.stamp + origin
            msg = image_msg(img, stamp, a.frame_id)
            pub.publish(msg)
            if ci is not None:
                if (ci.width, ci.height) != (msg.width, msg.height):
                    node.get_logger().warn(f"camera_info {ci.width}x{ci.height} != imagen {msg.width}x{msg.height}",
                                           throttle_duration_sec=5.0)
                ci.header.stamp = msg.header.stamp
                pub_info.publish(ci)
            if log:
                log.write(json.dumps({"i": n_pub, "frame_id": meta.frame_id, "stamp": stamp,
                                      "stamp_rel": meta.stamp, "t_pub": time.time()}) + "\n")
            n_pub += 1
            if n_pub % 50 == 0:
                node.get_logger().info(f"{n_pub} frames publicados ({n_pub / (time.time() - t0):.1f}/s)")
            if pace_by_stamp:
                if t_wall0 is None:
                    t_wall0, t_stamp0 = tick, meta.stamp
                # el próximo frame sale cuando corresponde según su propio PTS (aprox.: el actual + 1/fps)
                s = (t_wall0 + (meta.stamp - t_stamp0) + (a.stride / src._src_fps)) - time.time()
                if s > 0:
                    time.sleep(s)
            elif period:
                s = period - (time.time() - tick)
                if s > 0:
                    time.sleep(s)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if log:
            log.close()
        node.get_logger().info(f"fin: {n_pub} frames publicados en {time.time() - t0:.1f} s")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
