"""Afinidad de CPU para compartir la máquina entre LingBot y Stella-VSLAM (etapa 6, adelanto de la 19).

i7-13700H de esta máquina: CPUs 0-11 = 6 núcleos de rendimiento (P, con hyperthreading), CPUs 12-19 = 8
núcleos de eficiencia (E). Por defecto torch abre 14 hilos y OpenCV 20: el proceso del modelo puede
ocupar todos los núcleos en ráfagas (preprocesado, flujo óptico del analizador de contexto, armado de
mensajes) y Stella, que necesita tiempo real, queda sin CPU (etapa 4: rinde una fracción en vivo).
"""
import os


def parse_cpus(spec):
    """'0-3,12-19' -> {0,1,2,3,12,...,19}."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def pin_process(spec, limit_threads=True):
    """Fija la afinidad de TODOS los hilos del proceso actual (los que se creen después la heredan) y,
    si limit_threads, ajusta los hilos de torch y OpenCV a ese número de CPUs para no sobresuscribir.
    Devuelve el conjunto aplicado."""
    cpus = parse_cpus(spec)
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except OSError:
            pass
    if limit_threads:
        n = max(1, len(cpus))
        try:
            import torch
            torch.set_num_threads(n)
        except Exception:
            pass
        try:
            import cv2
            cv2.setNumThreads(n)
        except Exception:
            pass
    return cpus
