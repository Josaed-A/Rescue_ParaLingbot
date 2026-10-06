#!/usr/bin/env python3
"""Reorganiza el repo con la estructura de GARDIAN (src/, scripts/, docs/, test/), 2026-10-05.

Uso:  python3 reorganizar_repo.py <raíz del repo> [--dry]

Mueve (git mv si está versionado, mv si no) y corrige las rutas:

1. Raíz del repo. Para cada archivo movido se conoce su profundidad original. Una expresión que
   sube directorios (cadenas de os.path.dirname sobre __file__ o sobre HERE, Path.parent/.parents[i],
   `$(dirname "$0")/..` en bash) se corrige sólo si antes llegaba a la raíz: se le suman los niveles
   que el archivo bajó. Las que llegaban a un directorio dentro del árbol movido no cambian (los
   hermanos siguen siendo hermanos: scripts_stream <-> scripts_context pasan a src/vivo <-> src/mapas).
2. Rutas a un hermano calculadas desde el padre del directorio propio (`os.path.join(os.path.dirname
   (HERE), "scripts_stream")`) pasan al nombre nuevo del hermano ("vivo").
3. Rutas relativas a la raíz escritas como texto ("scripts_context/geo_filter.py", "scripts_stream",
   "demo_render", tools/doctor.py, ...) pasan a las nuevas, en el código y en la documentación vigente.

NO reescribe README.md (la bitácora es append-only: se edita a mano solo la parte de arriba).
demo.py y lingbot_map/ no se mueven (núcleo de LingBot; `import demo` y el enlace .pth de
lingbot_map dependen de que estén en la raíz). El código upstream (src/upstream) sólo recibe la
corrección de profundidad, no reescritura de texto.
"""
import os
import re
import shutil
import subprocess
import sys

MOVES = [  # (origen, destino); el orden importa (tests antes que su carpeta)
    ("scripts_stream/tests", "test"),
    ("scripts_stream", "src/vivo"),
    ("scripts_context", "src/mapas"),
    ("scripts_ros", "src/ros"),
    ("scripts_webcam", "src/captura"),
    ("scripts_gpu", "src/gpu"),
    ("scripts_seq", "src/secuencias"),
    ("scripts", "src/diagnostico"),
    ("gct_profile.py", "src/diagnostico/gct_profile.py"),
    ("demo_render", "src/upstream/demo_render"),
    ("benchmark", "src/upstream/benchmark"),
    ("preprocess", "src/upstream/preprocess"),
    ("tools", "scripts"),
    ("setup_env.sh", "scripts/setup_env.sh"),
    ("MATEMATICA.md", "docs/MATEMATICA.md"),
    ("SETUP.md", "docs/SETUP.md"),
    ("lingbot-map_paper.pdf", "docs/referencias/lingbot-map_paper.pdf"),
    ("example", "datos/example"),
    ("test_images", "datos/test_images"),
    ("assets", "datos/assets"),
    ("calibration_logs", "registros/calibration_logs"),
    ("sequence_results", "registros/sequence_results"),
    ("results", "registros/results"),
    ("results_gpu", "registros/results_gpu"),
]
ROOT_LOG = re.compile(r"^(.*\.(log|err\.log)|ram_report.*\.csv)$")
SIBLING = {"scripts_stream": "vivo", "scripts_context": "mapas", "scripts_ros": "ros", "scripts_webcam": "captura",
           "scripts_gpu": "gpu", "scripts_seq": "secuencias"}

OLD_DIAG = ["measure_load_fp16.py", "measure_load_int8.py", "measure_load_int8_v2.py", "measure_load_meta.py",
            "measure_load_mmap.py", "measure_load_only.py", "measure_load_safetensors.py", "measure_ram.ps1",
            "measure_ram_safety.ps1", "analyze_frame_redundancy.py", "audit_model_memory.py",
            "audit_safetensors_conversion.py", "benchmark_gct_memory.py", "sequence_run_single.py", "verify_load_fix.py"]
# texto relativo a la raíz (lo más específico primero)
SUBS = [
    (r"scripts_stream/tests", "test"),
    (r"scripts_stream", "src/vivo"),
    (r"scripts_context", "src/mapas"),
    (r"scripts_ros", "src/ros"),
    (r"scripts_webcam", "src/captura"),
    (r"scripts_gpu", "src/gpu"),
    (r"scripts_seq", "src/secuencias"),
    (r"(?<![\w/])tools/doctor\.py", "scripts/doctor.py"),
    (r"(?<![\w/])tools/fetch_assets\.py", "scripts/fetch_assets.py"),
    (r"(?<![\w/])tools/", "scripts/"),
    (r"\./setup_env\.sh", "scripts/setup_env.sh"),
    (r"(?<![\w/.])setup_env\.sh", "scripts/setup_env.sh"),
    (r"(?<![\w/])MATEMATICA\.md", "docs/MATEMATICA.md"),
    (r"(?<![\w/])SETUP\.md", "docs/SETUP.md"),
    (r"(?<![\w/])demo_render(?=/|\b)", "src/upstream/demo_render"),
    (r"(?<![\w/])example/", "datos/example/"),
    (r"(?<![\w/])test_images\b", "datos/test_images"),
    (r"(?<![\w/])gct_profile\.py", "src/diagnostico/gct_profile.py"),
    (r"(?<![\w/])sequence_results/", "registros/sequence_results/"),
    (r"(?<![\w/])results_gpu/", "registros/results_gpu/"),
    (r"(?<![\w/])calibration_logs/", "registros/calibration_logs/"),
] + [(r"(?<![\w/])scripts/" + re.escape(n), "src/diagnostico/" + n) for n in OLD_DIAG]
ROOT_DATA = {"example": "datos/example", "test_images": "datos/test_images", "assets": "datos/assets",
             "results": "registros/results", "results_gpu": "registros/results_gpu",
             "sequence_results": "registros/sequence_results", "calibration_logs": "registros/calibration_logs"}
# en documentos que ya están dentro de docs/, los enlaces relativos a MATEMATICA/SETUP no llevan docs/
DOC_LOCAL = [(r"\(docs/(MATEMATICA|SETUP)\.md", r"(\1.md")]
TEXT_EXT = (".py", ".sh", ".md", ".js", ".html", ".json", ".yaml", ".yml", ".txt", ".toml", ".cfg", ".gitignore")
TEST_PATH_OLD = "sys.path.insert(0, os.path.dirname(HERE))"
TEST_PATH_NEW = 'sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "vivo"))'


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def tracked(root, path):
    return bool(run(["git", "ls-files", path], root).stdout.strip())


def move(root, src, dst, dry):
    s, d = os.path.join(root, src), os.path.join(root, dst)
    if not os.path.exists(s):
        return None
    if dry:
        return f"{src} -> {dst}"
    os.makedirs(os.path.dirname(d), exist_ok=True)
    if tracked(root, src):
        r = run(["git", "mv", src, dst], root)
        if r.returncode != 0:
            raise SystemExit(f"git mv {src} {dst}: {r.stderr}")
        if os.path.exists(s):                  # quedan archivos sin versionar (p. ej. __pycache__, nuevos)
            for dp, _, fns in os.walk(s, topdown=False):
                for fn in fns:
                    a = os.path.join(dp, fn)
                    b = os.path.join(d, os.path.relpath(a, s))
                    os.makedirs(os.path.dirname(b), exist_ok=True)
                    shutil.move(a, b)
                os.rmdir(dp)
    else:
        shutil.move(s, d)
    return f"{src} -> {dst}"


def ncomp(rel):
    return len([c for c in rel.split("/") if c])


def fix_depth(text, c_old, d, is_sh):
    """Suma `d` niveles a las expresiones que, con el archivo a `c_old` componentes de la raíz,
    llegaban a la raíz (o más arriba)."""
    if d == 0:
        return text
    reaches = lambda remaining: remaining <= 0

    def dn_file(m):                      # dirname^n(abspath(__file__)); los ")" sobrantes son de afuera
        n = m.group(1).count("os.path.dirname(")
        closing = len(m.group(2))
        if n >= 2 and closing >= n and reaches(c_old - n):
            n2 = n + d
            return "os.path.dirname(" * n2 + "os.path.abspath(__file__)" + ")" * n2 + ")" * (closing - n)
        return m.group(0)
    text = re.sub(r"((?:os\.path\.dirname\()+)os\.path\.abspath\(__file__\)(\)+)", dn_file, text)

    def dn_here(m):                      # dirname^k(HERE), HERE = directorio propio
        k = m.group(1).count("os.path.dirname(")
        closing = len(m.group(2))
        if closing >= k and reaches(c_old - 1 - k):
            k2 = k + d
            return "os.path.dirname(" * k2 + "HERE" + ")" * k2 + ")" * (closing - k)
        return m.group(0)
    text = re.sub(r"((?:os\.path\.dirname\()+)HERE(\)+)", dn_here, text)

    def pparent(m):                      # Path(__file__).resolve().parent.parent...
        n = m.group(2).count(".parent")
        if reaches(c_old - n):
            return m.group(1) + ".parent" * (n + d)
        return m.group(0)
    text = re.sub(r"(Path\(__file__\)(?:\.resolve\(\))?)((?:\.parent)+)(?![s\w])", pparent, text)

    def pparents(m):                     # Path(__file__).resolve().parents[i]
        i = int(m.group(2))
        if reaches(c_old - (i + 1)):
            return f"{m.group(1)}.parents[{i + d}]"
        return m.group(0)
    text = re.sub(r"(Path\(__file__\)(?:\.resolve\(\))?)\.parents\[(\d+)\]", pparents, text)

    if is_sh:
        def sh_up(m):                    # $(dirname "$0")/.. | $(dirname "${BASH_SOURCE[0]}")[/..]* | "$HERE/.."
            k = m.group(2).count("/..")
            if reaches(c_old - 1 - k):
                return m.group(1) + "/.." * (k + d)
            return m.group(0)
        text = re.sub(r'(\$\(dirname "\$(?:0|\{BASH_SOURCE\[0\]\})"\))((?:/\.\.)*)', sh_up, text)
        text = re.sub(r'(\$HERE)((?:/\.\.)+)', sh_up, text)
    return text


def map_old(path):
    """Ruta vieja (relativa a la raíz) -> nueva, según MOVES."""
    path = os.path.normpath(path)
    for src, dst in MOVES:
        if path == src or path.startswith(src + "/"):
            return dst + path[len(src):]
    if ROOT_LOG.match(path) and "/" not in path:
        return "registros/raiz/" + path
    return path


def fix_md_links(text, old_rel, new_rel):
    """Enlaces relativos de un .md movido: se resuelven contra su ubicación vieja y se reescriben
    relativos a la nueva (los destinos también pueden haberse movido)."""
    def f(m):
        u, frag = m.group(1), m.group(2) or ""
        if re.match(r"^(https?|mailto|data):", u) or u.startswith("/"):
            return m.group(0)
        tgt = map_old(os.path.normpath(os.path.join(os.path.dirname(old_rel), u)))
        nu = os.path.relpath(tgt, os.path.dirname(new_rel) or ".")
        return f"]({nu}{frag})"
    return re.sub(r"\]\(([^)#\s]+)(#[^)]*)?\)", f, text)


def main():
    root = os.path.abspath(sys.argv[1])
    dry = "--dry" in sys.argv
    # profundidad original de cada archivo que se va a mover: {ruta nueva: (componentes viejos, delta)}
    origin = {}
    for src, dst in MOVES:
        s = os.path.join(root, src)
        if os.path.isfile(s):
            origin[dst] = (ncomp(src), ncomp(dst) - ncomp(src), src)
        elif os.path.isdir(s):
            for dp, dns, fns in os.walk(s):
                dns[:] = [x for x in dns if x != "__pycache__"]
                for fn in fns:
                    rel = os.path.relpath(os.path.join(dp, fn), s)
                    if rel.startswith("tests/") and src == "scripts_stream":
                        continue                 # ya cubierto por scripts_stream/tests -> test
                    new = os.path.join(dst, rel)
                    if new not in origin:
                        origin[new] = (ncomp(os.path.join(src, rel)), ncomp(dst) - ncomp(src), os.path.join(src, rel))
    done = []
    for src, dst in MOVES:
        r = move(root, src, dst, dry)
        if r:
            done.append(r)
    for f in sorted(os.listdir(root)):
        if os.path.isfile(os.path.join(root, f)) and ROOT_LOG.match(f):
            r = move(root, f, f"registros/raiz/{f}", dry)
            if r:
                done.append(r)
    if dry:
        print("\n".join(done))
        return
    changed = []
    skip_dirs = {".git", "captures", "checkpoints", "logs", "lingbot_map", "datos", "registros", ".venv",
                 "lingbot_map.egg-info", "__pycache__", "node_modules", "vendor", "build"}
    for dp, dns, fns in os.walk(root):
        dns[:] = [x for x in dns if x not in skip_dirs and not x.startswith(".")]
        for fn in fns:
            if not (fn.endswith(TEXT_EXT) or fn == ".gitignore") or fn == "reorganizar_repo.py":
                continue
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            if rel in ("README.md", "demo.py"):     # bitácora append-only; demo.py es núcleo de LingBot
                continue
            try:
                text = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            new = text
            upstream = rel.startswith("src/upstream/")
            if rel in origin:
                c_old, d, old_rel = origin[rel]
                if fn.endswith(".py"):
                    # hermanos calculados desde el padre del directorio propio (antes de tocar dirname(HERE))
                    # (con un marcador, para que la corrección de profundidad no lo toque: el padre del
                    # directorio propio sigue siendo el padre de los hermanos)
                    for old, leaf in SIBLING.items():
                        new = re.sub(r'os\.path\.join\(os\.path\.dirname\((HERE|here)\),(\s*)"%s"' % old,
                                     r'os.path.join(__PADRE_DE_\1__,\2"%s"' % leaf, new)
                    # imports con puntos (scripts_gpu.monitor_gpu -> src.gpu.monitor_gpu), antes de SUBS
                    for old, leaf in SIBLING.items():
                        new = re.sub(r"\b%s\.(?=[A-Za-z_])" % old, "src.%s." % leaf, new)
                    if rel.startswith("test/"):
                        new = new.replace(TEST_PATH_OLD, TEST_PATH_NEW)
                    else:
                        new = fix_depth(new, c_old, d, False)
                    new = new.replace("__PADRE_DE_HERE__", "os.path.dirname(HERE)").replace(
                        "__PADRE_DE_here__", "os.path.dirname(here)")
                elif fn.endswith(".sh"):
                    new = fix_depth(new, c_old, d, True)
            if fn.endswith(".md") and not upstream:
                new = fix_md_links(new, origin[rel][2] if rel in origin else rel, rel)
            if not upstream:
                if fn.endswith(".py"):
                    for old, leaf in SIBLING.items():
                        new = re.sub(r"\b%s\.(?=[A-Za-z_])" % old, "src.%s." % leaf, new)
                if fn.endswith(".py"):
                    # carpetas de datos unidas a la raíz como componente suelto: os.path.join(REPO, "results", ...)
                    for old, nw in ROOT_DATA.items():
                        new = re.sub(r'(os\.path\.join\(\s*(?:REPO|REPO_ROOT|ROOT)\s*,\s*)"%s"' % old, r'\1"%s"' % nw, new)
                for a, b in SUBS:
                    new = re.sub(a, b, new)
            if new != text:
                open(p, "w", encoding="utf-8").write(new)
                changed.append(rel)
    # .gitignore: carpetas que cambiaron de lugar
    gi = os.path.join(root, ".gitignore")
    if os.path.isfile(gi):
        t = open(gi).read()
        t2 = re.sub(r"(?m)^src/upstream/demo_render/$", "src/upstream/demo_render/", t)
        t2 = re.sub(r"(?m)^demo_render/$", "src/upstream/demo_render/", t2)
        t2 = re.sub(r"(?m)^results/$", "registros/results/", t2)
        t2 = re.sub(r"(?m)^results_gpu/$", "registros/results_gpu/", t2)
        if t2 != t:
            open(gi, "w").write(t2)
    print(f"{len(done)} movimientos, {len(changed)} archivos con rutas actualizadas")
    print("\n".join(changed))


if __name__ == "__main__":
    main()
