"""Corre una sesión de mapeo en vivo sin navegador, para medir el analizador de contexto.

Usa exactamente el mismo código que el servidor (LiveSession de live_server.py): mismo
preprocesado, mismo modelo en streaming y misma grabación. La sesión queda guardada en
<captures_dir>/streaming/sin_guardar/sesion_<fecha>/eval/sesion.npz, el mismo formato
que --save_predictions, así se puede comparar con compare_route.py.

Ejemplo, reproduciendo todos los frames de un video (30 fps) con el analizador:
  src/gpu/run_gpu.sh -- python3 src/vivo/replay_live.py \\
      --path captures/.../candidates_full --captures_dir /tmp/rep --context --synth
y la referencia sin analizador, a 10 fps fijos:
  ... --path captures/.../candidates_full --captures_dir /tmp/rep --stride 3
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_server import LiveSession, REPO_ROOT  # noqa: E402


class _Loop:
    def call_soon_threadsafe(self, fn, *a):
        fn(*a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None, help="carpeta de frames o video")
    ap.add_argument("--source", default="folder", choices=["folder", "video", "webcam", "android"])
    ap.add_argument("--device", type=int, default=0, help="webcam: índice /dev/videoN")
    ap.add_argument("--serial", default=None, help="android: serial adb")
    ap.add_argument("--camera_id", default="0")
    ap.add_argument("--cam_size", default="960x540")
    ap.add_argument("--captures_dir", required=True)
    ap.add_argument("--model_path", default=os.path.join(REPO_ROOT, "checkpoints", "lingbot-map.pt"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--context", action="store_true")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--synth_strength", type=float, default=1.0)
    ap.add_argument("--step_px", type=float, default=36.0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--special_keep", type=int, default=64, help="0 = comportamiento original (crece sin límite)")
    ap.add_argument("--keyframe_interval", type=int, default=1)
    ap.add_argument("--camera_keep", type=int, default=1024, help="0 = la caché de la cabeza de cámara crece sin límite")
    ap.add_argument("--source_fps", type=float, default=10.0,
                    help="fps nominal de la carpeta de frames: define los stamps sintéticos (índice / fps)")
    ap.add_argument("--rotation", type=int, default=0, help="rotar la fuente (90 para los videos del celular en vertical)")
    ap.add_argument("--realtime", action="store_true",
                    help="reproducir la carpeta/video a su fps nominal como una cámara en vivo (el modelo toma el último frame)")
    ap.add_argument("--ros2", action="store_true", help="puente ROS2: publicar los frames y recibir las poses de Stella (etapa 3)")
    ap.add_argument("--stella_config", default=None, help="con --ros2: arrancar Stella con este yaml (src/ros/stella/*.yaml)")
    ap.add_argument("--ros2_resize", default=None, help="publicar los frames reescalados, WxH")
    ap.add_argument("--ros2_pub_every", type=int, default=1, help="publicar 1 de cada N frames capturados")
    ap.add_argument("--cpus", default=None, help="fijar este proceso (modelo) a estas CPUs, p. ej. 0-3,12-19")
    ap.add_argument("--stella_cpus", default=None, help="fijar Stella a estas CPUs, p. ej. 4-11")
    ap.add_argument("--tracking_mode", default="basic", choices=["basic", "stella", "hybrid"],
                    help="qué pose se anuncia como referencia (etapa 6)")
    ap.add_argument("--stella_scale", default="anchor", choices=["anchor", "window"],
                    help="escala del modo STELLA: fija al anclar cada mapa o re-estimada por ventana (etapa 9)")
    ap.add_argument("--no_static_hold", dest="static_hold", action="store_false",
                    help="sin retener la pose en los momentos estáticos (comportamiento anterior al 2026-10-05)")
    ap.add_argument("--static_frac", type=float, default=0.25,
                    help="movimiento < static_frac x step_px del analizador = momento estático")
    ap.add_argument("--register_with_reference", action="store_true",
                    help="registrar la geometría con la pose de referencia en vez de BASIC (etapa 10)")
    ap.add_argument("--debug_stall_at_frame", type=int, default=None, help="prueba 15: detener el modelo en este frame")
    ap.add_argument("--debug_stall_s", type=float, default=10.0)
    ap.add_argument("--debug_kill_stella_at_frame", type=int, default=None, help="prueba 14: matar Stella en este frame")
    a = ap.parse_args()
    if a.cpus:
        from cpu_affinity import pin_process
        print(f"CPUs del modelo: {sorted(pin_process(a.cpus))}", flush=True)
    cfg = {"source": a.source, "path": a.path, "device": a.device, "serial": a.serial, "camera_id": a.camera_id,
           "cam_size": a.cam_size, "fps": None, "max_frames": a.max_frames,
           "points_per_frame": 2000, "conf_percentile": 30.0, "preview": False,
           "num_scale_frames": 2, "kv_cache_sliding_window": 16, "camera_num_iterations": 4,
           "model_path": a.model_path, "record": True, "captures_dir": a.captures_dir,
           "context": a.context, "context_synth": a.synth, "context_step_px": a.step_px, "context_synth_strength": a.synth_strength, "stride": a.stride,
           "special_keep": a.special_keep, "camera_keep": a.camera_keep, "keyframe_interval": a.keyframe_interval,
           "source_fps": a.source_fps, "rotation": a.rotation, "source_realtime": a.realtime,
           "ros2": a.ros2, "stella_config": a.stella_config, "ros2_camera_info": a.stella_config,
           "ros2_resize": a.ros2_resize, "ros2_pub_every": a.ros2_pub_every, "stella_log_dir": a.captures_dir,
           "stella_cpus": a.stella_cpus, "tracking_mode": a.tracking_mode,
           "register_with_reference": a.register_with_reference, "stella_scale": a.stella_scale,
           "static_hold": a.static_hold, "static_frac": a.static_frac,
           "debug_stall_at_frame": a.debug_stall_at_frame,
           "debug_stall_s": a.debug_stall_s, "debug_kill_stella_at_frame": a.debug_kill_stella_at_frame}
    msgs = {"n": 0, "last": -100, "vram_max": 0}

    def broadcast(obj):
        if isinstance(obj, dict) and obj.get("type") in ("error", "saved", "done"):
            print(json.dumps(obj), flush=True)
        if isinstance(obj, dict) and obj.get("type") == "status":
            msgs["vram_max"] = max(msgs["vram_max"], obj.get("vram_mb") or 0)
            if obj.get("frames", 0) - msgs["last"] >= 100:
                msgs["last"] = obj.get("frames", 0)
                print(f"frames {obj.get('frames')}  VRAM {obj.get('vram_mb')} MB  fps {obj.get('fps')}", flush=True)
        msgs["n"] += 1

    s = LiveSession(_Loop(), broadcast, cfg)
    # Ctrl+C / SIGTERM: detener la sesión como el botón "Detener" (guarda lo grabado y cierra Stella)
    import signal

    def _stop(signum, frame):
        print(f"señal {signum}: deteniendo la sesión", flush=True)
        s.stop()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    t0 = time.time()
    s.state["running"] = True
    s._run()
    print(json.dumps({"seconds": round(time.time() - t0, 1), "frames": s.state.get("frames"),
                      "fps": s.state.get("fps"), "saved": s.state.get("saved"),
                      "context": s.state.get("context"), "vram_max_mb": msgs["vram_max"],
                      "ros2": s.state.get("ros2")}), flush=True)


if __name__ == "__main__":
    main()
