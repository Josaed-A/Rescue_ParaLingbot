#!/usr/bin/env python3
"""Muestrea CPU, RAM, VRAM y uso de GPU de los procesos de una prueba (etapas 17-19).

Cada --interval s escribe una fila CSV con, para cada grupo de procesos (por patrón de línea de
comandos o nombre exacto): % de CPU (100 = un núcleo), RSS en MB; y de la GPU: VRAM usada (MB),
uso (%) y potencia (W). Termina con --duration, con Ctrl+C o cuando desaparecen todos los procesos
vigilados (--stop_when_gone).

  python3 src/ros/resource_monitor.py --out res.csv --watch modelo=replay_live.py --watch stella=run_slam
"""
import argparse
import csv
import os
import time

import psutil


def find(pattern):
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == os.getpid():
                continue
            name = p.info["name"] or ""
            cmd = " ".join(p.info["cmdline"] or [])
            if name == pattern or (pattern in cmd and "resource_monitor" not in cmd):
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--watch", action="append", default=[], help="grupo=patrón (nombre exacto o texto de la línea de comandos)")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=3600)
    ap.add_argument("--stop_when_gone", type=float, default=15.0,
                    help="terminar si ningún proceso vigilado existe durante N s (después de haber visto alguno)")
    a = ap.parse_args()
    groups = [w.split("=", 1) for w in a.watch]
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        h = None
    procs = {g: {} for g, _ in groups}
    cols = ["t"] + [f"{g}_{k}" for g, _ in groups for k in ("cpu_pct", "rss_mb", "n")] + ["vram_mb", "gpu_pct", "gpu_w"]
    t0 = time.time()
    seen_any, gone_since = False, None
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        while time.time() - t0 < a.duration:
            row = [round(time.time() - t0, 2)]
            alive = 0
            for g, pat in groups:
                cur = {p.pid: p for p in find(pat)}
                for pid, p in cur.items():
                    if pid not in procs[g]:
                        try:
                            p.cpu_percent(None)          # primera llamada: arranca la medición
                        except psutil.Error:
                            pass
                        procs[g][pid] = p
                for pid in list(procs[g]):
                    if pid not in cur:
                        procs[g].pop(pid)
                cpu = rss = 0.0
                for p in list(procs[g].values()):
                    try:
                        cpu += p.cpu_percent(None)
                        rss += p.memory_info().rss / 2 ** 20
                        # hijos (ros2 run -> run_slam, run_gpu.sh -> python)
                        for c in p.children(recursive=True):
                            if c.pid not in procs[g]:
                                c.cpu_percent(None)
                                procs[g][c.pid] = c
                    except psutil.Error:
                        pass
                row += [round(cpu, 1), round(rss, 1), len(procs[g])]
                alive += len(procs[g])
            if h is not None:
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2 ** 20
                    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    pw = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except Exception:
                    mem = util = pw = float("nan")
            else:
                mem = util = pw = float("nan")
            row += [round(mem, 0), util, round(pw, 1)]
            w.writerow(row)
            f.flush()
            if alive:
                seen_any, gone_since = True, None
            elif seen_any:
                gone_since = gone_since or time.time()
                if time.time() - gone_since > a.stop_when_gone:
                    break
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
