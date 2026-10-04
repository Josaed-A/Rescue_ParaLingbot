#!/usr/bin/env python3
"""Lanza el visor (nubes ya exportadas + mapeo en vivo) y abre el navegador.

Es un solo servidor y un solo puerto para las dos cosas. El modelo se carga
únicamente si se inicia una sesión en vivo desde la página: mirar nubes ya
exportadas no toca la GPU.

    scripts_context/webgl_viewer/launch.py [--port 8090] [--no-open]
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

REPO = Path(__file__).resolve().parents[2]
SERVER = REPO / "scripts_stream" / "live_server.py"


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
    args = ap.parse_args()

    url = f"http://localhost:{args.port}"
    if port_in_use(args.port):
        print(f"El puerto {args.port} ya está en uso. Abrí {url} o probá --port otro.")
        return 1

    cmd = [sys.executable, str(SERVER), "--port", str(args.port)]
    iso = REPO / "scripts_gpu" / "run_isolated.sh"
    if iso.is_file():
        # cgroup propio con tope de RAM: una sesión en vivo que agote la memoria muere ella
        # sola en vez de arrastrar a VS Code (ver scripts_gpu/run_isolated.sh)
        cmd = [str(iso), "--name", "visor", "--"] + cmd
    if args.captures_dir:
        cmd += ["--captures_dir", args.captures_dir]
    if args.model_path:
        cmd += ["--model_path", args.model_path]
    proc = subprocess.Popen(cmd)

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
