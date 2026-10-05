"""Corre una sesión de mapeo en vivo sin navegador, para medir el analizador de contexto.

Usa exactamente el mismo código que el servidor (LiveSession de live_server.py): mismo
preprocesado, mismo modelo en streaming y misma grabación. La sesión queda guardada en
<captures_dir>/streaming/sin_guardar/sesion_<fecha>/eval/sesion.npz, el mismo formato
que --save_predictions, así se puede comparar con compare_route.py.

Ejemplo, reproduciendo todos los frames de un video (30 fps) con el analizador:
  scripts_gpu/run_gpu.sh -- python3 scripts_stream/replay_live.py \\
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
    ap.add_argument("--path", required=True, help="carpeta de frames o video")
    ap.add_argument("--source", default="folder", choices=["folder", "video"])
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
    a = ap.parse_args()
    cfg = {"source": a.source, "path": a.path, "device": 0, "fps": None, "max_frames": a.max_frames,
           "points_per_frame": 2000, "conf_percentile": 30.0, "preview": False,
           "num_scale_frames": 2, "kv_cache_sliding_window": 16, "camera_num_iterations": 4,
           "model_path": a.model_path, "record": True, "captures_dir": a.captures_dir,
           "context": a.context, "context_synth": a.synth, "context_step_px": a.step_px, "context_synth_strength": a.synth_strength, "stride": a.stride,
           "special_keep": a.special_keep, "camera_keep": a.camera_keep, "keyframe_interval": a.keyframe_interval}
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
    t0 = time.time()
    s.state["running"] = True
    s._run()
    print(json.dumps({"seconds": round(time.time() - t0, 1), "frames": s.state.get("frames"),
                      "fps": s.state.get("fps"), "saved": s.state.get("saved"),
                      "context": s.state.get("context"), "vram_max_mb": msgs["vram_max"]}), flush=True)


if __name__ == "__main__":
    main()
