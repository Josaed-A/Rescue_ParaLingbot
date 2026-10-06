#!/usr/bin/env python3
"""Lanza el visor (nubes ya exportadas + mapeo en vivo) y abre el navegador.

Es un solo servidor y un solo puerto para las dos cosas. El modelo se carga
únicamente si se inicia una sesión en vivo desde la página: mirar nubes ya
exportadas no toca la GPU.

    src/mapas/webgl_viewer/launch.py [--port 8090] [--no-open] [--no-ros2]
"""
import argparse
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SERVER = REPO / "src/vivo" / "live_server.py"


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--captures_dir")
    ap.add_argument("--model_path")
    ap.add_argument("--no-open", action="store_true", help="no abrir el navegador")
    ap.add_argument("--no-ros2", action="store_true",
                    help="sin el puente ROS2 + Stella por defecto (el panel igual puede activarlo por sesión)")
    args = ap.parse_args()

    url = f"http://localhost:{args.port}"
    if port_in_use(args.port):
        print(f"El puerto {args.port} ya está en uso. Abrí {url} o probá --port otro.")
        return 1

    cmd = [sys.executable, str(SERVER), "--port", str(args.port)]
    iso = REPO / "src/gpu" / "run_isolated.sh"
    if iso.is_file():
        # cgroup propio con tope de RAM: una sesión en vivo que agote la memoria muere ella
        # sola en vez de arrastrar a VS Code (ver src/gpu/run_isolated.sh)
        cmd = [str(iso), "--name", "visor", "--"] + cmd
    if args.captures_dir:
        cmd += ["--captures_dir", args.captures_dir]
    if args.model_path:
        cmd += ["--model_path", args.model_path]
    if not args.no_ros2:
        cmd += ["--ros2"]       # cada sesión en vivo publica por ROS2 y arranca Stella (src/vivo/ros2_bridge.py)
    env = None
    if not args.no_ros2:
        # el puente necesita rclpy: si este Python no lo ve (lanzado desde un acceso directo, sin el
        # entorno de ROS2 cargado), se arranca el servidor con el setup.bash de ROS2 Jazzy
        setup = Path.home() / "ros2_jazzy" / "install" / "setup.bash"
        try:
            import rclpy  # noqa: F401
        except ImportError:
            if setup.is_file():
                cmd = ["bash", "-c", f'source "{setup}" >/dev/null 2>&1; exec "$@"', "visor"] + cmd
            else:
                print("Aviso: sin rclpy ni ~/ros2_jazzy: las sesiones correrán sin el puente ROS2.")
    proc = subprocess.Popen(cmd, env=env)

    def stop(*_):
        if proc.poll() is None:
            proc.terminate()
    signal.signal(signal.SIGTERM, stop)

    for _ in range(40):
        if proc.poll() is not None:
            return proc.returncode or 1
        try:
            urllib.request.urlopen(f"{url}/api/captures", timeout=1).close()
            break
        except Exception:
            time.sleep(0.3)
    print(f"Visor en {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        stop()
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 130


if __name__ == "__main__":
    sys.exit(main())
