"""Índice de nubes exportadas bajo captures/, para el selector del visor.

Se separó del servidor para que haya una sola implementación: la usa
scripts_stream/live_server.py (el único servidor) y sirve para cualquier script
que quiera listar qué pruebas hay exportadas.
"""
import os
import re


# Which files count as a "loadable cloud" for the picker, and how to label them.
CLOUD_PATTERNS = [
    (re.compile(r"_denso_alta\.ply$"), "fusionada ALTA densidad"),
    (re.compile(r"_denso\.ply$"), "fusionada (densa)"),
    (re.compile(r"_raw\.ply$"), "cruda (por-frame) + trayectoria"),
    (re.compile(r"_dense\.ply$"), "fusionada (densa)"),
    (re.compile(r"_malla\.glb$"), "malla TSDF (superficie)"),
    (re.compile(r"(?<!_denso)(?<!_dense)(?<!_raw)(?<!_malla)\.glb$"), "mapa (glb, submuestreado)"),
]


def scan_captures(captures_dir):
    """Walk captures/ and group every recognized cloud file by site/prueba."""
    sites = {}
    if not os.path.isdir(captures_dir):
        return {"root": captures_dir, "found": False, "sites": {}}
    for dirpath, _dirnames, filenames in os.walk(captures_dir):
        rel_dir = os.path.relpath(dirpath, captures_dir)
        for fn in filenames:
            label = None
            for pat, lbl in CLOUD_PATTERNS:
                if pat.search(fn):
                    label = lbl
                    break
            if label is None:
                continue
            full = os.path.join(dirpath, fn)
            rel_file = os.path.join(rel_dir, fn) if rel_dir != "." else fn
            # group key: captures/<top>/<...>/<prueba_N>/... -> "<top>/.../<prueba_N>"
            parts = rel_dir.split(os.sep)
            group = os.sep.join(parts[:3]) if len(parts) >= 3 else rel_dir
            group = group.replace(os.sep, "/")
            entry = sites.setdefault(group, {"group": group, "clouds": []})

            base = fn
            for suffix in ("_denso_alta.ply", "_denso.ply", "_raw.ply", "_dense.ply", ".ply", ".glb"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            cameras_rel = None
            cam_candidate = os.path.join(dirpath, base + "_cameras.json")
            if os.path.isfile(cam_candidate):
                cameras_rel = os.path.relpath(cam_candidate, captures_dir).replace(os.sep, "/")
            info_rel = None
            for suffix in ("_info.json",):
                cand = os.path.join(dirpath, base.split("_raw")[0].split("_dense")[0] + suffix)
                if os.path.isfile(cand):
                    info_rel = os.path.relpath(cand, captures_dir).replace(os.sep, "/")
                    break

            entry["clouds"].append({
                "label": label,
                "name": base,
                "file": rel_file.replace(os.sep, "/"),
                "ext": os.path.splitext(fn)[1].lstrip("."),
                "size_mb": round(os.path.getsize(full) / 2**20, 1),
                "cameras": cameras_rel,
                "info": info_rel,
            })
    for entry in sites.values():
        entry["clouds"].sort(key=lambda c: c["file"])
    return {"root": os.path.abspath(captures_dir), "found": True,
            "sites": dict(sorted(sites.items()))}
