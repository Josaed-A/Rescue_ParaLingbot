#!/usr/bin/env python3
"""Download, import or verify the model files this repo needs (they are gitignored).

Every file is checked against the size and SHA256 in env/assets.json, so a copy from
another machine or a resumed download is only accepted when it is bit-identical.

Sources, tried in order:
  1. --source-dir DIR   another checkout, a USB drive... (looked up at DIR/<dest> and DIR/<file>)
  2. --endpoint URL, $HF_ENDPOINT, https://huggingface.co, https://hf-mirror.com
     (huggingface.co has throttled robbyant/lingbot-map to a few KB/s from some networks;
     a server slower than 200 KB/s for 45 s is abandoned for the next one)
Downloads resume from <dest>.part. Standard library only: runs before any pip install.

Usage:
  python3 tools/fetch_assets.py                          # files for profiles core,vis
  python3 tools/fetch_assets.py --profiles core,vis,render
  python3 tools/fetch_assets.py --source-dir /media/usb/Rescue_ParaLingbot
  python3 tools/fetch_assets.py --verify                 # check only, download nothing
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "env", "assets.json")
CHUNK = 1 << 20
MIN_SPEED = 200 * 1024
SLOW_WINDOW_S = 45


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def matches(path, asset):
    if not os.path.isfile(path) or os.path.getsize(path) != asset["size"]:
        return False
    return sha256_of(path) == asset["sha256"]


def import_from(source_dirs, asset, dest):
    for src_root in source_dirs:
        for cand in (os.path.join(src_root, asset["dest"]), os.path.join(src_root, asset["file"])):
            if not os.path.exists(cand) or os.path.realpath(cand) == os.path.realpath(dest):
                continue
            if matches(cand, asset):
                print(f"    copiando desde {cand}")
                shutil.copyfile(cand, dest + ".part")
                os.replace(dest + ".part", dest)
                return True
            print(f"    {cand} existe pero no coincide (tamaño/SHA256): se ignora")
    return False


def endpoints(extra):
    eps = list(extra)
    if os.environ.get("HF_ENDPOINT"):
        eps.append(os.environ["HF_ENDPOINT"])
    eps += ["https://huggingface.co", "https://hf-mirror.com"]
    return list(dict.fromkeys(e.rstrip("/") for e in eps))


def fetch_into(url, part, total, may_abandon):
    """Append the missing bytes of `url` to `part`. Returns 'done', 'slow' or 'short'."""
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have >= total:
        return "done"
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={have}-", "User-Agent": "Rescue_ParaLingbot/fetch_assets"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        mode = "ab"
        if have and resp.status != 206:  # server ignored the Range header: start over
            have, mode = 0, "wb"
        start = last = time.time()
        got = 0
        with open(part, mode) as out:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                out.write(block)
                got += len(block)
                now = time.time()
                if now - last >= 2:
                    rate = got / (now - start)
                    sys.stdout.write(f"\r    {(have + got) / 2**20:7.0f}/{total / 2**20:.0f} MiB"
                                     f"  {rate / 2**20:6.2f} MiB/s ")
                    sys.stdout.flush()
                    last = now
                    if may_abandon and now - start > SLOW_WINDOW_S and rate < MIN_SPEED:
                        print("\n    demasiado lento: se prueba el siguiente servidor")
                        return "slow"
    print()
    return "done" if os.path.getsize(part) >= total else "short"


def download(asset, dest, extra_endpoints):
    part = dest + ".part"
    eps = endpoints(extra_endpoints)
    for i, ep in enumerate(eps):
        url = f"{ep}/{asset['repo']}/resolve/main/{asset['file']}"
        print(f"    {url}")
        for attempt in range(3):
            try:
                state = fetch_into(url, part, asset["size"], may_abandon=i < len(eps) - 1)
            except (urllib.error.URLError, OSError) as e:
                print(f"\n    error: {e} (intento {attempt + 1}/3)")
                time.sleep(3)
                continue
            if state == "slow":
                break
            if state == "short":
                continue
            if os.path.getsize(part) == asset["size"] and sha256_of(part) == asset["sha256"]:
                os.replace(part, dest)
                return True
            print("    SHA256 no coincide: se descarta la descarga")
            os.remove(part)
            break
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profiles", default="core,vis", help="separados por coma; 'all' = todos los archivos")
    p.add_argument("--source-dir", action="append", default=[],
                   help="copiar desde aquí antes de descargar (repetible)")
    p.add_argument("--endpoint", action="append", default=[],
                   help="servidor compatible con Hugging Face, se prueba primero")
    p.add_argument("--verify", action="store_true", help="solo verificar, no descargar nada")
    p.add_argument("--dest-root", default=ROOT, help=argparse.SUPPRESS)
    args = p.parse_args()

    with open(MANIFEST) as f:
        manifest = json.load(f)
    wanted = None if args.profiles == "all" else set(args.profiles.split(","))
    failures = []
    for name, asset in manifest.items():
        if wanted is not None and not wanted & set(asset["profiles"]):
            continue
        dest = os.path.join(args.dest_root, asset["dest"])
        print(f"[{name}] {asset['dest']} ({asset['size'] / 2**20:.0f} MiB) — {asset['used_by']}")
        if matches(dest, asset):
            print("    OK: presente, SHA256 verificado")
            continue
        if args.verify:
            print("    FALLA: " + ("falta" if not os.path.exists(dest) else "tamaño o SHA256 incorrecto"))
            failures.append(name)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            print(f"    el archivo actual no coincide: se renombra a {os.path.basename(dest)}.invalid")
            os.replace(dest, dest + ".invalid")
        if import_from(args.source_dir, asset, dest) or download(asset, dest, args.endpoint):
            print("    OK: listo y verificado")
        else:
            print("    FALLA: no se pudo obtener")
            failures.append(name)
    if failures:
        print(f"\nFaltan o no coinciden: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
