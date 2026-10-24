#!/usr/bin/env python3
"""Verifica de punta a punta los marcos TF2 que publica ParaLingbot (etapa 8).

Mientras corre una sesión en vivo con el puente ROS2, escucha /tf, /tf_static,
/paralingbot/tracking/reference_pose y /paralingbot/alignment/stella. Para cada pose de referencia
recibida pide a tf2 la transformación paralingbot_map <- paralingbot_camera_optical EN EL INSTANTE
de ese frame (el stamp de adquisición) y la guarda. Al terminar escribe un .npz con esas
transformaciones; con --session compara contra la sesión grabada:

  - la cadena TF (map -> odom -> camera_link -> camera_optical) reproduce la pose del .npz:
    T_map<-optical == A · c2w (frames.map_optical);
  - el topic reference_pose y la TF dicen lo mismo;
  - los estáticos existen y son los de frames.py.

  python3 src/ros/check_frames.py --out /tmp/tf.npz --duration 60 [--session S/eval/sesion.npz --key pose_basic]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vivo"))

import frames as F  # noqa: E402


def tf_to_mat(tr):
    from pose_buffer import quat_to_mat
    q = tr.rotation
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(np.array([q.x, q.y, q.z, q.w]))
    T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
    return T


def record(a):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from rclpy.time import Time
    from rclpy.duration import Duration
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = Node("paralingbot_check_frames")
    buf = Buffer(cache_time=Duration(seconds=3600))
    TransformListener(buf, node)
    refs, aligns = [], []

    def on_ref(m):
        refs.append((m.header.stamp.sec, m.header.stamp.nanosec, m.header.frame_id,
                     [m.pose.position.x, m.pose.position.y, m.pose.position.z],
                     [m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w]))

    node.create_subscription(PoseStamped, "/paralingbot/tracking/reference_pose", on_ref, 50)
    latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_subscription(String, "/paralingbot/alignment/stella", lambda m: aligns.append(json.loads(m.data)), latched)
    t_end = time.time() + a.duration
    idle_since = None
    while rclpy.ok() and time.time() < t_end:
        n0 = len(refs)
        rclpy.spin_once(node, timeout_sec=0.2)
        if refs and len(refs) == n0:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > a.idle_stop:
                break
        else:
            idle_since = None
    # transformaciones por instante de frame
    out = {"stamps": [], "T_map_optical": [], "T_map_link_tf": [], "T_map_link_topic": [], "ok": []}
    for sec, nsec, frame, p, q in refs:
        t = Time(seconds=sec, nanoseconds=nsec)
        try:
            Tmo = tf_to_mat(buf.lookup_transform(F.MAP, F.CAMERA_OPTICAL, t).transform)
            Tml = tf_to_mat(buf.lookup_transform(F.MAP, F.CAMERA_LINK, t).transform)
            ok = True
        except Exception as e:
            Tmo = Tml = np.full((4, 4), np.nan)
            ok = False
            print("lookup falló:", e, flush=True)
        from pose_buffer import quat_to_mat
        Tt = np.eye(4)
        Tt[:3, :3] = quat_to_mat(np.array(q))
        Tt[:3, 3] = p
        out["stamps"].append(sec + nsec * 1e-9)
        out["T_map_optical"].append(Tmo)
        out["T_map_link_tf"].append(Tml)
        out["T_map_link_topic"].append(Tt)
        out["ok"].append(ok)
    statics = {}
    for parent, child in ((F.MAP, F.ODOM), (F.MAP, F.MAP_CV), (F.CAMERA_LINK, F.CAMERA_OPTICAL)):
        try:
            statics[f"{parent}->{child}"] = tf_to_mat(buf.lookup_transform(parent, child, Time()).transform).tolist()
        except Exception as e:
            statics[f"{parent}->{child}"] = str(e)
    node.destroy_node()
    rclpy.shutdown()
    np.savez(a.out, **{k: np.array(v) for k, v in out.items()}, alineaciones=json.dumps(aligns),
             estaticos=json.dumps(statics))
    print(f"{len(refs)} poses de referencia, {sum(out['ok'])} con TF; {len(aligns)} alineaciones de Stella", flush=True)
    return a.out


def compare(tf_npz, session, key):
    d = np.load(tf_npz)
    s = np.load(session)
    # el puente corre los stamps de carpetas y videos hacia ROS2 (stamp_offset_ros en info.json)
    info = json.load(open(os.path.join(os.path.dirname(os.path.dirname(session)), "info.json")))
    off = ((info.get("tracking") or {}).get("stella") or {}).get("stamp_offset_ros") or 0.0
    st = {round(float(t) + off, 4): i for i, t in enumerate(s["stamps"])}
    res = {"poses_tf": int(len(d["stamps"])), "stamp_offset_ros": off}
    e_chain, e_topic, n = [], [], 0
    for t, Tmo, Tml, Tt, ok in zip(d["stamps"], d["T_map_optical"], d["T_map_link_tf"], d["T_map_link_topic"], d["ok"]):
        i = st.get(round(float(t), 4))
        if i is None or not ok:
            continue
        n += 1
        c2w = s[key][i].astype(np.float64)
        e_chain.append(np.abs(Tmo - F.map_optical(c2w)).max())
        e_topic.append(np.abs(Tml - Tt).max())
    res.update({"emparejadas_con_sesion": n, "clave": key,
                "max_err_cadena_tf_vs_npz": float(max(e_chain)) if e_chain else None,
                "max_err_tf_vs_topic": float(max(e_topic)) if e_topic else None})
    est = json.loads(str(d["estaticos"]))
    res["estaticos_ok"] = {
        k: (isinstance(v, list) and np.allclose(np.array(v), ref, atol=1e-6))
        for k, v, ref in ((f"{F.MAP}->{F.ODOM}", est.get(f"{F.MAP}->{F.ODOM}"), np.eye(4)),
                          (f"{F.MAP}->{F.MAP_CV}", est.get(f"{F.MAP}->{F.MAP_CV}"), F.T_MAP_MAPCV),
                          (f"{F.CAMERA_LINK}->{F.CAMERA_OPTICAL}", est.get(f"{F.CAMERA_LINK}->{F.CAMERA_OPTICAL}"), F.T_LINK_OPTICAL))}
    res["alineaciones_stella"] = json.loads(str(d["alineaciones"]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--idle_stop", type=float, default=8.0, help="terminar si no llegan poses durante N s")
    ap.add_argument("--session", default=None)
    ap.add_argument("--key", default="pose_basic")
    ap.add_argument("--only_compare", action="store_true")
    a = ap.parse_args()
    if not a.only_compare:
        record(a)
    if a.session:
        print(json.dumps(compare(a.out, a.session, a.key), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
