# Estructura del repositorio y equivalencias con la anterior

El 2026-10-05 el repo se ordenó con la estructura de GARDIAN (`src/`, `scripts/`, `docs/`, `test/`). La raíz pasó de ~75 entradas a 19. Lo hizo `scripts/reorganizar_repo.py`, que movió con `git mv` lo versionado y corrigió las rutas en el código y en la documentación vigente.

Las secciones de la bitácora del README anteriores a esa fecha **no se reescribieron** (la bitácora es append-only): usan las rutas viejas de esta tabla.

| Antes | Ahora |
|---|---|
| `scripts_stream/` | `src/vivo/` |
| `scripts_stream/tests/` | `test/` |
| `scripts_context/` (incluye `webgl_viewer/`) | `src/mapas/` |
| `scripts_ros/` | `src/ros/` |
| `scripts_webcam/` | `src/captura/` |
| `scripts_gpu/` | `src/gpu/` |
| `scripts_seq/` | `src/secuencias/` |
| `scripts/` (diagnósticos de memoria, mayormente Windows) y `gct_profile.py` | `src/diagnostico/` |
| `demo_render/`, `benchmark/`, `preprocess/` (upstream) | `src/upstream/` |
| `tools/doctor.py`, `tools/fetch_assets.py`, `setup_env.sh` | `scripts/` |
| `SETUP.md`, `MATEMATICA.md` | `docs/` |
| `lingbot-map_paper.pdf` | `docs/referencias/` |
| `example/`, `test_images/`, `assets/` | `datos/` |
| `results/`, `results_gpu/`, `sequence_results/`, `calibration_logs/` | `registros/` |
| `*.log`, `*.err.log`, `ram_report*.csv` de la raíz | `registros/raiz/` |

Sin cambios: `demo.py` y `lingbot_map/` (núcleo de LingBot; `import demo` y el enlace `.pth` del paquete dependen de que estén en la raíz), `env/`, `captures/`, `checkpoints/`, `logs/`, `skyseg.onnx`, `skyseg_batch.onnx`, `README.md`, `CLAUDE.md`, `LICENSE.txt`, `pyproject.toml`.

## Cómo se corrigieron las rutas

- **Raíz del repo en el código.** Para cada archivo movido se conoce su profundidad original. Una expresión que sube directorios (cadenas de `os.path.dirname`, `Path.parent`/`parents[i]`, `$(dirname "$0")/..`) se corrigió solo si antes llegaba a la raíz. Las que apuntaban dentro del árbol movido no cambian: los hermanos siguen siendo hermanos (`src/vivo` ↔ `src/mapas`).
- **Rutas escritas como texto** (`scripts_context/geo_filter.py`, `tools/doctor.py`, imports `scripts_gpu.monitor_gpu` → `src.gpu.monitor_gpu`, ...), en código y documentación vigente.
- **Enlaces de Markdown** de los documentos movidos: resueltos contra la ubicación vieja y reescritos relativos a la nueva.

## Cómo se verificó

Primero en una copia del repo y después en el real: `compileall` sin errores, las 48 pruebas unitarias, `--help` de los 54 scripts con argumentos (fallan solo los 10 diagnósticos de Windows, que ya fallaban en Linux), sin rutas viejas en el código ni enlaces rotos en `docs/`, `setup_env.sh --dry-run`, `doctor.py`, y corridas reales: una sesión en vivo con Stella por `src/ros/run_benchmark.sh` (266 poses, recursos registrados), su análisis, `build_maps.py`, y el visor con un trabajo de "construir mapas" lanzado desde su API. `git status` muestra todo lo versionado como renombres (`R`), sin borrados.
