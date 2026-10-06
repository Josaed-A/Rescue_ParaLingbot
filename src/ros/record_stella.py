#!/usr/bin/env python3
"""Graba lo que publica Stella-VSLAM (stella_vslam_ros) y resume cómo le fue (etapa 2).

Suscribe:
  /stella/camera_pose     nav_msgs/Odometry      pose de la cámara (solo cuando el tracking tiene éxito)
  /stella/keyframes       geometry_msgs/PoseArray poses actuales de TODOS los keyframes, en cada frame
  /stella/tracking_state  std_msgs/String        "Initializing" | "Tracking" | "Lost" (parche nuestro al nodo)

Escribe un JSONL (--out) con una línea por mensaje, tal cual llega (frame `map` de ROS, sin
convertir; la conversión a la convención de LingBot es de la etapa 3):
  {"type": "pose", "stamp", "t_recv", "frame_id", "child_frame_id", "p": [x,y,z], "q": [x,y,z,w]}
  {"type": "state", "t_recv", "state"}
  {"type": "keyframes", "stamp", "t_recv", "n", "max_shift", "poses": [...] }   (poses completas solo
       cuando cambia el número de keyframes o alguno se movió más de --kf_shift: así quedan
       registradas las correcciones de loop closure sin guardar lo mismo en cada frame)

Al terminar (Ctrl+C o --duration) imprime el resumen: poses recibidas, fps, tiempo hasta la
primera pose, histograma de estados, episodios de pérdida, keyframes finales, correcciones.

  python3 src/ros/record_stella.py --out /tmp/stella_run.jsonl --duration 60
"""
import argparse
import json
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, String


class Recorder(Node):
    def __init__(self, out, ns, kf_shift):
        super().__init__("paralingbot_stella_recorder")
        self.f = open(out, "w")
        self.kf_shift = kf_shift
        self.t0 = time.time()
        self.n_pose = 0
        self.first_pose_t = None
        self.last_pose_t = None
        self.states = {}
        self.state_seq = []           # (t, state) solo en cambios
        self.last_state = None
        self.kf_prev = None
        self.kf_n = 0
        self.corrections = 0
        self.stamps = []
        self.create_subscription(Odometry, f"{ns}/camera_pose", self.on_pose, 10)
        self.create_subscription(PoseArray, f"{ns}/keyframes", self.on_kf, 10)
        self.create_subscription(String, f"{ns}/tracking_state", self.on_state, 10)
        # etapa 12: keyframes con id y timestamp de imagen (parche al nodo); se guarda cada mensaje que
        # cambia el conjunto o mueve algún keyframe (correcciones de BA / loop closure)
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(Float64MultiArray, f"{ns}/keyframes_full", self.on_kf_full, qos_profile_sensor_data)
        self.kf_full_prev = None
        self.kf_full_n = 0
        self.get_logger().info(f"grabando {ns}/camera_pose, {ns}/keyframes, {ns}/tracking_state -> {out}")

    def _w(self, obj):
        self.f.write(json.dumps(obj) + "\n")

    def on_pose(self, m):
        t = time.time()
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self._w({"type": "pose", "stamp": stamp, "t_recv": t, "frame_id": m.header.frame_id,
                 "child_frame_id": m.child_frame_id, "p": [p.x, p.y, p.z], "q": [q.x, q.y, q.z, q.w]})
        self.n_pose += 1
        self.stamps.append(stamp)
        if self.first_pose_t is None:
            self.first_pose_t = t
        self.last_pose_t = t

    def on_state(self, m):
        t = time.time()
        self.states[m.data] = self.states.get(m.data, 0) + 1
        if m.data != self.last_state:
            self.state_seq.append((round(t - self.t0, 3), m.data))
            self.last_state = m.data
            self._w({"type": "state", "t_recv": t, "state": m.data})

    def on_kf(self, m):
        t = time.time()
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        P = np.array([[p.position.x, p.position.y, p.position.z] for p in m.poses], dtype=np.float64).reshape(-1, 3)
        n = len(P)
        max_shift = 0.0
        changed = self.kf_prev is None or n != len(self.kf_prev)
        if not changed and n:
            max_shift = float(np.linalg.norm(P - self.kf_prev, axis=1).max())
            changed = max_shift > self.kf_shift
            if changed:
                self.corrections += 1
        self.kf_n = n
        row = {"type": "keyframes", "stamp": stamp, "t_recv": t, "n": n, "max_shift": round(max_shift, 6)}
        if changed:
            row["poses"] = [[p.position.x, p.position.y, p.position.z,
                             p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w] for p in m.poses]
            self.kf_prev = P
        self._w(row)

    def on_kf_full(self, m):
        d = np.array(m.data, dtype=np.float64)
        if len(d) < 2:
            return
        stamp, n = d[0], int(d[1])
        rows = d[2:2 + 9 * n].reshape(n, 9) if n else np.zeros((0, 9))
        key = {int(r[0]): r[2:5] for r in rows}
        changed = self.kf_full_prev is None or set(key) != set(self.kf_full_prev)
        shift = 0.0
        if not changed and key:
            shift = max(float(np.linalg.norm(key[k] - self.kf_full_prev[k])) for k in key)
            changed = shift > self.kf_shift
        if changed:
            self._w({"type": "keyframes_full", "stamp": float(stamp), "t_recv": time.time(), "n": n,
                     "max_shift": round(shift, 6), "kf": rows.tolist()})
            self.kf_full_prev = key
            self.kf_full_n += 1

    def summary(self):
        dur = time.time() - self.t0
        s = {"duracion_s": round(dur, 1), "poses": self.n_pose,
             "fps_poses": round(self.n_pose / max(self.last_pose_t - self.first_pose_t, 1e-6), 2) if self.n_pose > 1 else 0.0,
             "t_primera_pose_s": None if self.first_pose_t is None else round(self.first_pose_t - self.t0, 2),
             "estados": self.states,
             "cambios_de_estado": self.state_seq[:40],
             "episodios_lost": sum(1 for _, st in self.state_seq if st == "Lost"),
             "keyframes_finales": self.kf_n, "correcciones_keyframes": self.corrections,
             "mensajes_keyframes_full": self.kf_full_n}
        if len(self.stamps) > 2:
            d = np.diff(np.array(self.stamps))
            s["stamp_dt_median_s"] = round(float(np.median(d)), 4)
            s["stamp_gap_max_s"] = round(float(d.max()), 3)
        return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="/stella", help="prefijo de los topics del nodo de Stella")
    ap.add_argument("--duration", type=float, default=0.0, help="segundos (0 = hasta Ctrl+C)")
    ap.add_argument("--kf_shift", type=float, default=0.01,
                    help="desplazamiento de un keyframe (unidades de Stella) que cuenta como corrección")
    a = ap.parse_args()
    rclpy.init()
    rec = Recorder(a.out, a.ns, a.kf_shift)
    try:
        if a.duration > 0:
            end = time.time() + a.duration
            while rclpy.ok() and time.time() < end:
                rclpy.spin_once(rec, timeout_sec=0.2)
        else:
            rclpy.spin(rec)
    except KeyboardInterrupt:
        pass
    finally:
        s = rec.summary()
        rec.f.write(json.dumps({"type": "summary", **s}) + "\n")
        rec.f.close()
        print(json.dumps(s, indent=1, ensure_ascii=False))
        rec.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
