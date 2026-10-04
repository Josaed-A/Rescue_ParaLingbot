"""Catálogo de pruebas para el explorador del visor.

Una "prueba" es cualquier carpeta bajo captures/ llamada prueba_N o sesion_* (las sesiones
en vivo guardadas). Para cada una se informa:
  - sus mapas, por tipo (nube, alta, cruda, malla, splat, glb), con la trayectoria de
    cámaras asociada si existe (exports/webgl/<name>_cameras.json), así todos los tipos
    de mapa de una misma prueba se orientan y normalizan igual en el visor;
  - su metadato editable, guardado en el info.json de la prueba: titulo, zona,
    categorias, notas. Se lee y se reescribe preservando el resto de las claves;
  - si tiene predicciones (eval/*.npz) y frames/, para ofrecer "construir mapas".

El listado de archivos es perezoso (una carpeta por pedido): frames/ o candidates_full/
tienen miles de archivos y no hace falta recorrerlos para armar el árbol.
"""
import json
import os
import re
import shutil
import time

PRUEBA_RE = re.compile(r"^(prueba_\d+|sesion_[\w\-]+)$")

MAP_TYPES = [   # (patrón, tipo, etiqueta) -- el orden es el de presentación
    (re.compile(r"_denso\.ply$"), "nube", "nube densa"),
    (re.compile(r"_denso_alta\.ply$"), "alta", "nube alta densidad"),
    (re.compile(r"_raw\.ply$"), "cruda", "cruda + trayectoria"),
    (re.compile(r"_filtrado_malla\.glb$"), "malla_f", "malla TSDF filtrada"),
    (re.compile(r"_malla\.glb$"), "malla", "malla TSDF"),
    (re.compile(r"_filtrado_splat\.ply$"), "splat_f", "gaussian splat filtrado"),
    (re.compile(r"_splat\.ply$"), "splat", "gaussian splat"),
    (re.compile(r"_estructura\.glb$"), "estructura", "estructura (paredes y piso)"),
    (re.compile(r"_dense\.ply$"), "nube", "nube densa"),
    (re.compile(r"(?<!_malla)(?<!_estructura)\.glb$"), "glb", "glb (viejo)"),
]
SUFFIXES = ["_filtrado_malla.glb", "_filtrado_splat.ply", "_estructura.glb", "_denso_alta.ply", "_denso.ply", "_raw.ply", "_malla.glb", "_splat.ply", "_dense.ply", ".glb"]
DEFAULT_CATEGORIES = ["video grabado", "streaming", "webcam", "desnivel", "escaleras", "plano",
                      "interior", "exterior", "prueba local", "sin guardar"]
SKIP_DIRS = {"candidates", "candidates_full", "frames", "variantes", "source", "__pycache__"}


def _rel(path, root):
    return os.path.relpath(path, root).replace(os.sep, "/")


def _read_info(p):
    f = os.path.join(p, "info.json")
    if os.path.isfile(f):
        try:
            with open(f) as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {"_contenido": d}
        except Exception:
            return {}
    return {}


def _map_base(fn):
    for s in SUFFIXES:
        if fn.endswith(s):
            return fn[: -len(s)]
    return os.path.splitext(fn)[0]


def _maps(p, root):
    out = []
    ex = os.path.join(p, "exports")
    if not os.path.isdir(ex):
        return out
    for dirpath, dirnames, filenames in os.walk(ex):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            for pat, typ, label in MAP_TYPES:
                if pat.search(fn):
                    base = _map_base(fn)
                    cams = None
                    for cand in (os.path.join(dirpath, base + "_cameras.json"),
                                 os.path.join(ex, "webgl", base + "_cameras.json")):
                        if os.path.isfile(cand):
                            cams = _rel(cand, root)
                            break
                    info = None
                    for cand in (os.path.join(dirpath, base + "_splat_info.json"),
                                 os.path.join(dirpath, base + "_malla_info.json"),
                                 os.path.join(dirpath, fn[:-4] + "_info.json"),
                                 os.path.join(dirpath, base + "_filtro_info.json"),
                                 os.path.join(dirpath, fn[:-4] + "_info.json")):
                        if os.path.isfile(cand):
                            info = _rel(cand, root)
                            break
                    full = os.path.join(dirpath, fn)
                    out.append({"type": typ, "label": label, "name": base, "file": _rel(full, root),
                                "ext": fn.rsplit(".", 1)[-1], "size_mb": round(os.path.getsize(full) / 2**20, 1),
                                "cameras": cams, "info": info})
                    break
    order = {}
    for k, (_, t, _) in enumerate(MAP_TYPES):
        order.setdefault(t, k)
    out.sort(key=lambda m: (order.get(m["type"], 99), m["file"]))
    return out


def scan_pruebas(root):
    pruebas = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for d in list(dirnames):
            if not PRUEBA_RE.match(d):
                continue
            p = os.path.join(dirpath, d)
            info = _read_info(p)
            ev = os.path.join(p, "eval")
            npzs = sorted(f for f in os.listdir(ev) if f.endswith(".npz")) if os.path.isdir(ev) else []
            rid = _rel(p, root)
            pruebas.append({
                "id": rid,
                "sitio": _rel(dirpath, root),
                "nombre": d,
                "titulo": info.get("titulo") or "",
                "zona": info.get("zona") or "",
                "categorias": info.get("categorias") or [],
                "notas": info.get("notas") or "",
                "sin_guardar": "/sin_guardar/" in "/" + rid + "/",
                "npz": ["eval/" + f for f in npzs],
                "tiene_frames": os.path.isdir(os.path.join(p, "frames")),
                "maps": _maps(p, root),
                "fecha": str(info.get("fecha") or info.get("date") or
                             time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(p))))[:10],
            })
        # no seguir bajando dentro de una prueba
        dirnames[:] = [d for d in dirnames if not PRUEBA_RE.match(d)]
    cats = sorted(set(DEFAULT_CATEGORIES) | {c for p in pruebas for c in p["categorias"]})
    zonas = sorted({p["zona"] for p in pruebas if p["zona"]})
    return {"root": os.path.abspath(root), "pruebas": pruebas, "categorias": cats, "zonas": zonas}


def resolve_prueba(root, rid):
    root = os.path.realpath(root)
    p = os.path.realpath(os.path.join(root, rid))
    if not p.startswith(root + os.sep) or not os.path.isdir(p) or not PRUEBA_RE.match(os.path.basename(p)):
        raise ValueError(f"prueba inválida: {rid}")
    return p


def list_dir(root, rid, sub=""):
    """Contenido de una carpeta de la prueba (archivos y subcarpetas, con tamaño)."""
    p = resolve_prueba(root, rid)
    d = os.path.realpath(os.path.join(p, sub))
    if not (d == p or d.startswith(p + os.sep)) or not os.path.isdir(d):
        raise ValueError(f"carpeta inválida: {sub}")
    entries = []
    with os.scandir(d) as it:
        items = sorted(it, key=lambda e: (not e.is_dir(), e.name))
    for e in items[:400]:
        if e.is_dir():
            try:
                n = sum(1 for _ in os.scandir(e.path))
            except OSError:
                n = 0
            entries.append({"name": e.name, "dir": True, "count": n, "path": _rel(e.path, p)})
        else:
            entries.append({"name": e.name, "dir": False, "size": e.stat().st_size,
                            "path": _rel(e.path, p), "data": _rel(e.path, root)})
    return {"prueba": rid, "dir": _rel(d, p) if d != p else "", "entries": entries,
            "truncated": len(items) > 400, "total": len(items)}


def update_meta(root, rid, fields):
    p = resolve_prueba(root, rid)
    f = os.path.join(p, "info.json")
    info = _read_info(p)
    for k in ("titulo", "zona", "notas"):
        if k in fields:
            info[k] = str(fields[k]).strip()
    if "categorias" in fields:
        cats = [str(c).strip() for c in fields["categorias"] if str(c).strip()]
        info["categorias"] = list(dict.fromkeys(cats))
    tmp = f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(info, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, f)
    return info


def _next_prueba(dest):
    n = 1
    while os.path.exists(os.path.join(dest, f"prueba_{n}")):
        n += 1
    return os.path.join(dest, f"prueba_{n}")


def save_session(root, rid, destino, fields):
    """Pasa una sesión en vivo de captures/streaming/sin_guardar/ a <destino>/prueba_N."""
    p = resolve_prueba(root, rid)
    if "/sin_guardar/" not in "/" + rid + "/":
        raise ValueError("sólo se pueden mover sesiones sin guardar")
    destino = destino.strip().strip("/") or "streaming"
    droot = os.path.realpath(os.path.join(root, destino))
    if not droot.startswith(os.path.realpath(root) + os.sep) or PRUEBA_RE.match(os.path.basename(droot)):
        raise ValueError(f"destino inválido: {destino}")
    os.makedirs(droot, exist_ok=True)
    new = _next_prueba(droot)
    shutil.move(p, new)
    nid = _rel(new, root)
    cats = [c for c in fields.get("categorias", []) if c != "sin guardar"]
    update_meta(root, nid, {**fields, "categorias": cats})
    return nid


def discard_session(root, rid):
    p = resolve_prueba(root, rid)
    if "/sin_guardar/" not in "/" + rid + "/":
        raise ValueError("sólo se pueden descartar sesiones sin guardar")
    shutil.rmtree(p)
