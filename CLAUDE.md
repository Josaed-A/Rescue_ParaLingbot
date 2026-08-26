# GARDIAN — LingBot-Map Lightweight/Adaptativo

## Qué es esto
Subproyecto de percepción 3D dentro de **GARDIAN** (Ground-Aerial Response for Disaster Intelligence and Assistance Network), inicialmente para un robot de búsqueda y rescate. Objetivo: mapeo 3D con **cámara RGB convencional** (sin depender de RGB-D o LiDAR como sensor principal), usando LingBot-Map como motor de percepción geométrica.

RGB camera → LingBot-Map → depth + pose estimados → point cloud, tratando la cámara RGB como un "sensor RGB-D virtual" (proyección pinhole estándar: Z=D(u,v), X=(u-cx)Z/fx, Y=(v-cy)Z/fy, luego transformación al frame global con la pose estimada).

**No se va a abandonar LingBot-Map por falta de GPU.** La estrategia es una segunda etapa de investigación: reducir su coste computacional (no reescribir la arquitectura) antes de considerar distillation o cambios profundos al modelo.

## Estado actual (resumen ejecutivo — actualizado 2026-08-24, leer esto primero)

**Baseline oficial vigente:** `demo.py::load_model()` con `torch.load(..., mmap=True)` + `del ckpt, state_dict; gc.collect()` — ambos cambios YA aplicados y activos en el código real (no en un script de prueba). Pico de carga ≈13.1GB, memoria de reposo tras cargar ≈8.5GB, carga en ~90-110s (vs 218-254s del original sin optimizar). Pipeline completo verificado end-to-end (10 imágenes + visor `viser`) sin errores. El baseline *original* sin optimizar (pico 16.1GB) queda documentado más abajo como referencia histórica, sin editar.

**Líneas de investigación cerradas temporalmente** (cada una con análisis + prueba aislada + decisión documentada abajo, ninguna aplicada a producción salvo mmap+gc):
- `device="meta"` + `load_state_dict(assign=True)` — descartada: la construcción del modelo falla dentro de `lingbot_map` (`vision_transformer.py:126`), requeriría parchear la librería, no solo el script de carga.
- `safetensors` — descartada: no mejora el pico (lo empeora, 17.7GB) respecto a `mmap=True`.
- `FP16` (aggregator en float16, cabezas en float32 — frontera ya definida por los propios autores) — descartada *para esta CPU*: recupera memoria real pero la inferencia es 4× más lenta (sin AVX512, sin aceleración de hardware). Conservada como candidata para GPU.
- `INT8` dinámico weight-only (`quantize_dynamic`) — descartada: pico de conversión ~17GB y RAM libre mínima de un solo dígito de MB, confirmado **estructural** al mecanismo `inplace=False` (deep-copy interno), no un bug de limpieza — se verificó con una repetición corregida. Estabilidad numérica también mucho peor que FP16.
- **Ninguna de estas se combinó entre sí ni con quantization INT4** — quedan como líneas individuales evaluadas, no descartadas para siempre, sino pausadas con evidencia documentada.

**Auditoría de memoria del modelo cargado:** confirmó que no hay duplicación de tensores ni fugas — los ~4.63GB de parámetros reales coinciden exactamente con lo enumerado; la diferencia contra la memoria de proceso observada es overhead del *allocator* (no devuelve memoria liberada al SO de inmediato), no memoria "necesaria". Desglose por componente: backbone DINOv2 (`patch_embed`) 1217.5MB (26%), GCT (`frame_blocks`+`global_blocks`) 2419.0MB (52%, la parte más grande del modelo), cabezas (`camera_head`+`depth_head`) 995.3MB (22%).

**Línea de investigación actual, distinta a todo lo anterior:** redundancia temporal entre frames (base conceptual para el "Lightweight Context Analyzer" / ideas tipo Paragraphica de la filosofía original, ver abajo). Resultado preliminar (10 imágenes, análisis de imagen puro, sin tocar el modelo): **baja redundancia** en esta muestra específica — solo ~11% de las transiciones entre frames consecutivos calificarían como "potencialmente saltables" bajo un umbral estricto. Muestra pequeña (9 transiciones, una sola secuencia) — no es una conclusión definitiva, ver limitaciones documentadas en su sección.

**Todos los scripts de experimentos viven en `scripts/`** (cada uno documentado y enlazado en su sección correspondiente más abajo) — son herramientas de diagnóstico reutilizables, no parte del pipeline de producción.

**Actualización 2026-08-24 — DOS líneas paralelas e independientes investigando la misma pregunta, en máquinas distintas, fusionadas en este documento (no se descarta ninguna de las dos):**

1. **Máquina Linux nueva** (esta sesión): "Campaña de caracterización secuencial" — GPU disponible pero **CPU forzada deliberadamente** para preservar el baseline FP32 exacto. **COMPLETA: 5 fases (N=10/25/50/100/200), 21 corridas, 0 fallos de pipeline.** Conclusión: memoria NO acumula (satura, ΔRAM/frame decreciente monótono 329.5→66.8 MB/frame), tiempo/frame crece pero se frena marcadamente pasado N≈100 (coherente con `kv_cache_sliding_window=64`). Ver sección "Reporte final de la campaña de caracterización secuencial" más abajo para el detalle completo, incluida la tabla de las 5 fases y las limitaciones explícitas.
2. **Máquina Windows original** (sesión paralela, documentada arriba en todo el resto del archivo): "Campaña de secuencia larga" — primera corrida real de reconocimiento completada (20 frames, éxito), tras recalibrar el umbral de seguridad del monitor externo (bajado a 15MB — el margen normal de este baseline en esa máquina ya es de pocos cientos de MB). Dato único: 148.89s/frame promedio, ΔRAM/frame=250.7MB. Ver sección "Campaña de secuencia larga: estabilidad de memoria en `inference_streaming`" más abajo.

**Ambas atacan la misma pregunta (¿memoria estable o acumulativa al crecer la secuencia?) con el mismo dataset (`example/courthouse`) y el mismo baseline, pero en hardware muy distinto — tratarlas como dos fuentes de evidencia independientes, no fusionar sus números.** La máquina Windows es ~20× más lenta por frame en esta comparación preliminar (148.89s/frame vs ~6-8s/frame en Linux) y opera con márgenes de RAM mucho más ajustados — buen caso para eventualmente comparar si la tendencia (constante/creciente/etc.) es la misma en ambas o si es un artefacto de una máquina específica. **No editar ninguna de las dos secciones de campaña como si fuera la otra — son corridas y datos reales de máquinas distintas.**

**Actualización 2026-08-25 — Baseline GPU real (Linux/NVIDIA) medido, distinto de la campaña CPU-forzada de arriba:** con GPU real (NVIDIA RTX 2000 Ada, 8GB VRAM, capability 8.9, `demo.py` castea el aggregator a bf16 automáticamente sin tocar nada), el baseline de 200 frames sin modificar **NO es alcanzable — OOM real y reproducible en el frame 35/200**, muy por debajo del punto donde `kv_cache_sliding_window=64` llegaría a saturar la memoria (ese mecanismo nunca llega a activarse; el techo de VRAM de esta GPU llega primero). Inferencia en sí ~25× más rápida que CPU (~0.25s/frame vs ~6.3s/frame). Ver sección "Baseline GPU real (Linux/NVIDIA)" más abajo para el detalle completo (progresión de VRAM por frame, dos regímenes de crecimiento, utilización/temperatura/potencia de GPU). Además, análisis (sin implementar) de qué haría falta para alimentar `demo.py` con una webcam en vivo — ver sección "Análisis de interfaz para webcam RGB en vivo" inmediatamente después: el código actual de `--video_path` no sirve para un dispositivo en vivo (tres problemas concretos identificados), pero el modelo ya expone la primitiva de streaming por-frame necesaria (`GCTStream.forward(..., causal_inference=True)`) sin necesidad de tocar `lingbot_map`.

**Actualización 2026-08-25 (más reciente) — Extensión de la campaña secuencial CPU a N=320:** mismo baseline CPU-forzado de la campaña de 5 fases, extendido más allá de N=200. `courthouse` no llega a 400 frames (286 máximo) ni `university` (324) — se usó `university` con N=320 (el máximo que se mantiene en `keyframe_interval=1`, evitando cruzar el umbral de auto-selección de `demo.py` en 320). **Resultado: la saturación de RAM y la estabilización de tiempo/frame se confirman con más fuerza todavía** — el análisis de deltas *marginales* dentro de la misma corrida (no solo acumulados) muestra ΔRAM/frame prácticamente cero (-0.34 a +9.16 MB/frame) y tiempo/frame marginal convergiendo a una constante (~21.3-21.6s/frame) desde el frame 100 en adelante. Ningún régimen nuevo aparece después de 200 frames — el mecanismo (`kv_cache_sliding_window=64`) sigue siendo la explicación consistente. Ver sección "Extensión de la campaña secuencial: N=320" más abajo (antes de la sección de GPU).

**Pendiente de decisión del usuario:** implementar el driver de captura de webcam en vivo (análisis ya hecho, nada implementado todavía), decidir si mitigar el techo de VRAM de la GPU antes de ese experimento (dado que a ~4 FPS el mismo OOM se alcanzaría en ~9 segundos de captura continua), continuar la campaña secuencial en la máquina Windows dado el costo de tiempo mucho mayor ahí, continuar la línea de redundancia de frames, o recién ahí empezar la integración/optimización con Paragraphica.

## Estado del diagnóstico (ya hecho, no repetir)
- Máquina: Windows, AMD Ryzen 5 3500U (4 cores/8 threads), **sin GPU NVIDIA/CUDA**, gráficos integrados Vega.
- **RAM: ~10GB totales, con frecuencia solo ~800MB libres.** Restricción más apretada que "sin GPU" — el checkpoint solo (4.6GB) ya se come casi toda la RAM libre disponible en uso normal. Tenerlo en cuenta antes de correr baselines: cerrar apps pesadas o esperar RAM libre antes de lanzar `demo.py`.
- `demo.py` tiene fallback a CPU y flag `--use_sdpa` (evita depender de FlashInfer, que requiere GPU).
- Kaolin / onnxruntime-gpu / `batch_demo.py` solo son necesarios para renderizado offline y máscara de cielo — no obligatorios para un primer baseline.
- Los pesos del modelo (`lingbot-map.pt`, 4.6GB) **ya están descargados** en `C:\Users\josae\Rescu\lingbot-map\lingbot-map.pt`. `test_images\` tiene 10 imágenes de prueba.
- Código del repo ya copiado (sin `example/` de HF) en esta misma carpeta: `C:\Users\josae\Rescu\lingbot-map\`.
- **No se usó conda.** `conda.exe` existe (`C:\Users\josae\miniconda3\Scripts\conda.exe`) pero nunca se inicializó ni se creó el env `lingbot-map`. En su lugar se instaló todo en el **Python 3.14 del sistema** (`python` en PATH → `C:\Python314\python.exe`), que ya traía torch/opencv/pillow/numpy/tqdm de un intento previo. Se completó con `pip install -e .` dentro de `lingbot-map\` (añade einops, huggingface_hub, safetensors, scipy). **Seguir usando este entorno**, no crear el conda env del plan original (evita reinstalar deps pesadas).
- Extras de visualización (`viser`, `trimesh`, `matplotlib`, `onnxruntime`, target `pip install -e ".[vis]"`) **todavía no instalados** — no bloquean el baseline (demo.py cae a imprimir las keys de `predictions` si `viser` falta).

### Baseline mínimo reproducible — CORRIDO CON ÉXITO (2026-08-23)
```powershell
cd C:\Users\josae\Rescu\lingbot-map
python demo.py --model_path lingbot-map.pt --image_folder test_images --use_sdpa --camera_num_iterations 1 --first_k 3
```
Resultado (exit code 0, `demo_retry_run.log`), 3 imágenes, CPU, `dtype=float32`:
- Carga del checkpoint (`load_model` completo, incluye construir el modelo + `torch.load` de 4.6GB): **254.4 s**. Dominado por I/O/RAM ajustada, no por cómputo — candidato a medir aparte si se repite mucho durante experimentación.
- Inferencia streaming de 3 frames: **177.3 s** → ~59 s/frame en CPU. Extremadamente lento, pero confirma el pipeline funciona end-to-end.
- Predicciones generadas correctamente: `pose_enc, depth, depth_conf, frame_type, is_keyframe, extrinsic, intrinsic`.
- El warning `Failed to load pretrained weights: [Errno 2] No such file or directory: ''` es **benigno**: viene de `lingbot_map/aggregator/base.py:220-230`, es un intento de precargar un backbone DINOv2 antes de que el checkpoint completo de LingBot-Map se cargue encima (`load_model` en `demo.py`) — no afecta el resultado. El `demo_run.log` viejo en esta carpeta que solo mostraba esa línea y nada más era de un intento que murió después (probablemente por RAM), no por ese warning.

### Reestructuración de repos (2026-08-23) — leer antes de seguir
- `C:\Users\josae\Rescu\lingbot-map\` es ahora la **copia de referencia intacta** (sin pesos versionados, sin git) — no más experimentos ni entornos ahí.
- **Este repo** (`C:\Users\josae\Rescu\lingbot-map-test\`) es un `git clone` real de `github.com/robbyant/lingbot-map` (rama `main`, historial completo) — aquí es donde se hacen pruebas y, más adelante, la rama lightweight. Trae de fábrica `example/courthouse/` (203MB, dataset oficial de muestra) además de `test_images/` (10 imágenes propias, copiadas, gitignored).
- Entorno: `.venv/` propio en este repo (Python 3.14 del sistema como base, aislado vía `python -m venv`), **no** el Python de sistema ni conda. `pip install -e ".[vis]"` ya corrido aquí (incluye `viser`). Esto es lo portable/reproducible: cualquier clone nuevo solo necesita `python -m venv .venv && .venv\Scripts\pip install -e ".[vis]"` + bajar el checkpoint de HF.
- El Python de sistema quedó limpio (revertido a como estaba antes de instalar nada de este proyecto ahí).
- `.gitignore` de este repo tiene añadido `.venv/`, `*.pt`, `*.log`, `test_images/` (no committear checkpoint ni logs de prueba).

### Bug de Windows corregido en este repo (no upstream todavía)
`load_images()` en `demo.py` (y no exponía `--image_ext` por CLI) construía la lista de paths iterando `.jpg,.png,.JPG` con `glob.glob` — en Windows (filesystem case-insensitive) esto duplica cada imagen `.jpg` (matchea tanto el patrón `*.jpg` como `*.JPG`), doblando frames y tiempo de cómputo sin avisar. Arreglado con deduplicación por `os.path.normcase(os.path.abspath(p))` antes de ordenar — funciona igual en Windows/Linux/Mac. Si en algún momento se sincroniza con upstream, este fix no está ahí todavía.

### Bug de compatibilidad matplotlib corregido en este repo
`lingbot_map/vis/point_cloud_viewer.py` y `lingbot_map/vis/utils.py` usaban `matplotlib.cm.get_cmap(...)`, removido en matplotlib recientes (≥3.11, la versión que instala `pip install -e ".[vis]"` hoy). Reemplazado por `matplotlib.colormaps.get_cmap(...)` (API vigente), con `import matplotlib` agregado donde faltaba. Sin este fix, el visor crashea justo al terminar la inferencia con `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'`.

### Demo con las 10 imágenes + visor — CORRIDO CON ÉXITO (2026-08-23)
```powershell
cd C:\Users\josae\Rescu\lingbot-map-test
.venv\Scripts\python.exe demo.py --model_path lingbot-map.pt --image_folder test_images --use_sdpa --camera_num_iterations 1
```
Lanzado como proceso desacoplado (`Start-Process` en PowerShell) para que el servidor `viser` sobreviva más allá de cualquier timeout de la sesión/herramienta que lo lanzó — importante porque el visor debe quedar sirviendo indefinidamente en `http://localhost:8080` mientras alguien lo mira.

Resultado (dtype=float32, CPU):
- Carga del checkpoint: 218.6s (consistente con el run de 3 imágenes: ~210-255s, dominado por I/O/RAM, no escala con nº de frames).
- Inferencia streaming de 10 frames: 652.1s → ~65s/frame promedio. El desglose por el progreso de tqdm mostró que los primeros 8 frames ("scale frames" agrupados) se procesan casi de golpe, y los frames 9-10 (streaming real, uno a la vez) tardaron ~135s y ~91s respectivamente — el costo por frame individual en modo streaming es bastante mayor que el de los frames de "escala" iniciales, consistente con el overhead de KV-cache en CPU.
- Visor `viser` levantado correctamente en `http://localhost:8080`, sirviendo el point cloud reconstruido.

**No repetir el diagnóstico de por qué el visor fallaba** — ya está resuelto (ver bug de matplotlib arriba).

### Reporte: medición de pico de RAM — Intento 1, CRASH (2026-08-23)

**Objetivo:** medir RAM pico (proceso + sistema) durante carga del checkpoint + inferencia con las 10 imágenes de test, siguiendo el punto 4 de "Próximo paso inmediato".

**Método:** `demo.py` lanzado como proceso desacoplado (`Start-Process`, mismo comando de siempre: `--use_sdpa --camera_num_iterations 1`, sin `--first_k`, 10 imágenes). Monitoreado con [scripts/measure_ram.ps1](scripts/measure_ram.ps1) (nuevo, reusable): muestrea cada 3s `WorkingSet64` y `PrivateMemorySize64` del proceso `python.exe` real (PID hijo del launcher — `Start-Process` en este setup lanza un proceso padre casi vacío y el trabajo real corre en un hijo; hay que ubicar el PID hijo vía `Get-CimInstance Win32_Process -Filter "ParentProcessId=..."`, no asumir que el PID devuelto por `Start-Process` es el que hay que medir) + `FreePhysicalMemory` del sistema vía `Get-CimInstance Win32_OperatingSystem`. Datos crudos en `ram_report.csv` (gitignored).

**Condición confirmada por el usuario: ninguna otra aplicación pesada corría en paralelo.** Esto descarta contención externa como causa — el resultado de abajo es el pipeline solo, en condiciones limpias.

**Resultado: CRASH a los ~72-90s, durante la carga del checkpoint (nunca llegó a imprimir "Checkpoint loaded").** Confirmado por el Visor de Eventos de Windows (`Get-WinEvent -LogName Application`): **APPCRASH**, excepción `0xc0000005` (access violation) en `c10.dll` (núcleo C++ de PyTorch), proceso `python.exe`. No es un `MemoryError` de Python controlado — es un fallo nativo, por eso no aparece traceback en stderr.

Progresión de memoria hasta el crash (`ram_report.csv`, 21 muestras cada 3s):
| t (s) | proc WorkingSet (MB) | proc Private (MB) | sys Free (MB) |
|---|---|---|---|
| 0.0 | 293.1 | 719.2 | 3081.4 |
| 19.3 | 3126.0 | 6005.4 | 380.3 |
| 22.6 | 2102.9 | **6774.9** | **249.8 (mínimo)** |
| 42.4 | 2811.3 | **7928.6 (máximo)** | 676.1 |
| 71.6 (última muestra) | 2914.4 | 7832.6 | 1094.6 |

RAM total del sistema: 10177 MB. Es decir, solo *cargar* el checkpoint (aún sin terminar) ya comprometía ~7.8-7.9GB de memoria privada — dejando menos de 2GB para el resto del SO, en una máquina de ~10GB. El mínimo de RAM libre del sistema (249.8MB) ocurrió *antes* del pico de memoria del proceso, sugiriendo que hubo un momento de presión aún mayor que no se capturó entre muestras (intervalo de 3s puede perderse picos más cortos).

**Interpretación:**
- El pipeline original, tal cual, está **al límite absoluto de lo sostenible en esta máquina** — no es un margen ajustado pero funcional, es un límite que ya se cruzó y causó un crash real.
- El crash ocurrió en la fase de *carga*, no de inferencia — más barato de reproducir/depurar que si hubiera sido a mitad de un run largo.
- Corridas previas (218.6s-254.4s de carga) con el mismo comando **completaron sin crashear**, dos veces. Este intento falló la tercera vez. Esto es evidencia de que el pipeline está en el borde: unas veces pasa, otras no — típico de un sistema con menos de ~200MB de margen real. No asumir que "ya está confirmado que funciona" solo porque corrió bien antes.
- Sospecha para investigar (no confirmada): `torch.load(args.model_path, map_location=device, weights_only=False)` en `demo.py::load_model` (línea ~154) carga el `.pt` completo a RAM sin `mmap=True`; combinado con que el modelo (`GCTStream`) ya se instanció en memoria *antes* de cargar el checkpoint (`load_model` primero construye el modelo, línea ~139, y recién después hace `torch.load`), el pico transitorio podría ser: modelo instanciado (pesos random) + checkpoint deserializado en RAM + copia al hacer `load_state_dict` — varias copias simultáneas del mismo ~4.6GB antes de que el garbage collector libere las intermedias.

### Próximo paso pedido por el usuario (2026-08-23): campaña controlada de ≥10 repeticiones
El usuario pidió explícitamente que la próxima medición sea **mucho más controlada y con al menos 10 corridas**, para que picos de RAM/tiempos sean mínimamente confiables (una sola corrida no distingue una medición representativa de un outlier — y ya vimos que el comportamiento no es determinístico: 2 éxitos, 1 crash con el mismo comando).

Diseño propuesto (pendiente de ejecutar, no iniciar sin confirmar con el usuario dado el costo de tiempo: ~10 × 4-15 min ≈ 1-2.5 horas):
1. Script que corra N=10 repeticiones secuenciales de `demo.py` (mismo comando), cada una como proceso nuevo (no reusar), con pausa entre corridas para que el sistema libere memoria por completo antes de la siguiente.
2. Por corrida, registrar: tiempo de carga, tiempo de inferencia, RAM pico (workingset y private), RAM libre mínima del sistema, y éxito/fallo (crash sí/no, vía Event Viewer + código de salida).
3. Detener el visor (`viser`) apenas se confirme que arrancó en cada corrida en vez de dejarlo sirviendo — no es necesario mantenerlo vivo para medir RAM y así se libera memoria antes de la siguiente repetición.
4. Reportar al final: media, desviación estándar, mín, máx de cada métrica + tasa de fallos sobre las 10 corridas.

### Reporte: prueba aislada de `load_model()` — sin imágenes, sin inferencia, sin visor (2026-08-23)

**Objetivo:** antes de lanzar la campaña de 10 corridas completas, aislar `load_model()` (instanciación de `GCTStream` + `torch.load` + `load_state_dict`) para localizar exactamente en qué operación ocurre el pico de RAM, sin la variable adicional de imágenes/inferencia/visor. Pedido explícito del usuario, no ejecutar la campaña grande sin este baseline primero.

**Script nuevo:** [scripts/measure_load_only.py](scripts/measure_load_only.py) — reproduce la secuencia exacta de `demo.py::load_model()` (mismos argumentos: `image_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024, num_scale_frames=8, kv_cache_sliding_window=64, use_sdpa=True, camera_num_iterations=1`), con un snapshot de memoria después de cada paso. Mide con la API de Windows directamente (`GetProcessMemoryInfo` / `GlobalMemoryStatusEx` vía `ctypes`) en vez de polling externo — da picos exactos rastreados por el propio SO (`PeakWorkingSetSize`, `PeakPagefileUsage`), no estimaciones por muestreo. **No modifica `demo.py` ni ningún código de `lingbot_map`.**

Nota de depuración: la primera versión del script devolvía 0.0MB en todos los campos — bug de binding de ctypes (faltaba declarar `restype`/`argtypes` en `GetCurrentProcess`/`GetProcessMemoryInfo`/`GlobalMemoryStatusEx`, causaba truncamiento del handle en 64-bit). Corregido antes de tomar ninguna medición real.

**Metodología de doble medición:** además de los snapshots internos (8 puntos discretos, uno por operación), se corrió en paralelo el monitor externo [scripts/measure_ram.ps1](scripts/measure_ram.ps1) a intervalo de 1s (`ram_report_loadonly.csv`, gitignored) como red de seguridad forense. **Resultó ser necesario, no redundante:** el muestreo externo capturó dos ventanas de RAM casi agotada que los 8 snapshots discretos no vieron, porque cayeron *dentro* de operaciones de larga duración (instanciación ~20s, `load_state_dict` ~154s) en vez de en sus bordes. Lección metodológica: instrumentación de pocos puntos discretos subestima el riesgo real en operaciones largas — hace falta muestreo continuo fino para las ventanas de peligro reales.

**Resultado por etapa (snapshots internos, `python.exe`, proceso aislado):**
| t (s) | Etapa | proc private (MB) | sys avail (MB) |
|---|---|---|---|
| 0.0 | torch importado | 426.5 | 6604.9 |
| 2.0 | clase `GCTStream` importada | 707.6 | 6544.2 |
| 20.2 | **modelo instanciado** (pesos aleatorios) | **8411.4** | 1790.2 |
| 90.4 | **`torch.load` del checkpoint listo** | **16114.1 (pico)** | 1078.7 |
| 90.5 | `state_dict` extraído del ckpt | 16114.1 | 1081.1 |
| 244.5 | `load_state_dict` completo | 16114.0 | 382.8 |
| 246.1 | tras `del ckpt, state_dict; gc.collect()` | **12901.7** | 1060.2 |
| 246.2 | `model.to(device).eval()` | 12901.7 | 1059.2 |

Conteo de parámetros del modelo: **1,157,943,540** (~4.63GB solo en pesos float32) — más grande de lo asumido hasta ahora; explica por qué instanciar el modelo *solo*, sin tocar el checkpoint, ya cuesta 8.4GB.

**Lo que reveló el monitor externo (fino, 1s) que los snapshots discretos no vieron — dos ventanas de RAM casi agotada:**
1. **~elapsed 19-29s (durante/justo después de instanciar el modelo):** RAM libre del sistema entre **24MB y 104MB**, sostenido varios segundos. El monitor mismo sufrió un salto de ~47s sin poder tomar muestra (`Get-CimInstance` bloqueado), consistente con el sistema en estado de contención severa, no solo "poca RAM" — llegando casi a paralizarse.
2. **~elapsed 178-198s (durante `load_state_dict`):** RAM libre del sistema entre **63MB y 280MB**, sostenido ~20s.

Es decir, el mínimo real de RAM libre no fue el ~383MB que sugerían los snapshots discretos — fue **~24MB**, dos veces, en dos operaciones distintas. Esta corrida sobrevivió ambas veces por margen mínimo; la corrida anterior del pipeline completo (ver reporte de crash arriba) no tuvo esa suerte y crasheó en el mismo tramo (`torch.load`). Confirma: el comportamiento es al borde y probabilístico, no determinístico, y **el problema es 100% atribuible a `load_model()`** — no a las imágenes, la inferencia ni el visor, que ni siquiera se ejecutaron en esta prueba.

**Desglose de dónde se va la memoria:**
- **Instanciar el modelo** (pesos aleatorios, antes de tocar el checkpoint): +7.7GB (707MB → 8.4GB). Casi el doble de lo que explican los 4.63GB de parámetros — causa exacta no confirmada (sospecha: buffers temporales de inicialización de pesos, overhead de construcción de módulos del backbone DINOv2-giant/large con `block_chunks`, no investigado a fondo todavía).
- **`torch.load` del checkpoint**: +7.7GB adicionales (8.4GB → 16.1GB, el pico de memoria comprometida de toda la prueba). El archivo pesa 4.6GB en disco pero el costo en RAM fue ~7.7GB — consistente con tener temporalmente tanto el buffer serializado como los tensores deserializados en memoria a la vez durante el unpickling.
- **`load_state_dict`**: no sube la memoria *comprometida* (se mantiene en 16.1GB), pero tarda **154 segundos** — el costo aquí es traer de vuelta a RAM física páginas que ya habían sido paginadas a disco (el working set del proceso sube de 521MB a 6.3GB durante este paso), no nueva asignación. Es el tramo de mayor duración y donde el monitor externo vio la segunda ventana crítica.
- **`del ckpt, state_dict; gc.collect()`** (que `demo.py` **no hace hoy**): libera **3.2GB** (16.1GB → 12.9GB). Confirma la sospecha original — el dict del checkpoint queda referenciado sin necesidad después de copiarse a los parámetros del modelo.
- **Estado final "listo para inferencia"** (tras `.to(device).eval()`, sin el `del`+gc que demo.py no hace): **12.9GB de memoria comprometida / 5.6GB de working set** — ya por sí solo supera el total físico del sistema (10.2GB), sostenido únicamente por el pagefile.

**Conclusión para decidir próximo paso:** el candidato de optimización más simple, de bajo riesgo y ya cuantificado es agregar `del ckpt, state_dict; gc.collect()` en `load_model()` justo después de `load_state_dict()` — recupera 3.2GB verificados, sin tocar arquitectura ni resultados. No se aplicó todavía (pedido explícito del usuario: solo diagnóstico por ahora). Pendiente de decisión del usuario: aplicar ese fix antes de la campaña de 10 corridas, o correr la campaña primero contra el código sin modificar para tener un baseline "as-is" documentado.

### Fix aplicado y verificado: `del ckpt, state_dict; gc.collect()` en `demo.py::load_model()` (2026-08-23)

**Cambio real, mínimo, único** (nada más del pipeline tocado): en `demo.py::load_model()`, justo después de `model.load_state_dict(state_dict, strict=False)` y antes de `print("  Checkpoint loaded.")`, se agregó:
```python
del ckpt, state_dict
gc.collect()
```
(+ `import gc` al inicio del archivo). El baseline documentado arriba **se conserva sin editar** — esta sección es la comparación, no un reemplazo.

**Script de verificación:** [scripts/verify_load_fix.py](scripts/verify_load_fix.py) — a diferencia de `measure_load_only.py` (reimplementación standalone, usada para localizar el pico por etapas), este script **importa y llama la función real `demo.load_model(args, device)`**, para medir el código de producción tal cual, con el fix ya adentro. Mismos argumentos, mismo checkpoint, misma metodología de doble medición (snapshots internos vía `ctypes` + monitor externo `measure_ram.ps1` a 1s en paralelo, `ram_report_verifyfix.csv`).

**Comparación baseline (sin fix) vs. con fix:**

| Métrica | Baseline (sin fix) | Con fix | Cambio |
|---|---|---|---|
| Pico memoria comprometida (peak_pagefile, snapshot interno) | 16114.1 MB | 16209.9 MB | **≈ igual** (+96MB, dentro del ruido) |
| Pico memoria privada (monitor externo, más fino) | 15367.6 MB | 15459.0 MB | **≈ igual** |
| RAM libre mínima del sistema (monitor externo, 1s) | **24 MB** | **20.8 MB** | **≈ igual o levemente peor** |
| Memoria privada al terminar la carga | 12901.7 MB | 11552.0 MB | **-1350 MB** (mejora real, pero menor a los -3200MB medidos en el script aislado) |
| Working set al terminar la carga | 5624.4 MB | 4774.1 MB | -850 MB (mejora real) |
| Tiempo total de carga | ≈244.2s (instanciación→eval, baseline) | **301.35s** (medido directo alrededor de `load_model()`) | **+57s (~23% más lento)** — no atribuible con confianza al fix; la variabilidad entre corridas ya observada en el pipeline completo (218.6s vs 254.4s de carga, ±14%) es del mismo orden. Necesitaría repeticiones para separar señal de ruido. |

**Limitación confirmada explícitamente (condición anticipada por el usuario, se cumplió):** el fix **reduce la memoria posterior a la carga** (~1-1.35GB menos, working set y privada) pero **NO reduce el pico durante `torch.load()`/`load_state_dict()`**, ni mejora el mínimo de RAM libre del sistema durante esa ventana crítica — porque el `del`+`gc.collect()` ocurre *después* de que el pico ya se alcanzó y ya se cruzó por la ventana de mayor riesgo. El momento en que el sistema estuvo más cerca de un crash (20.8MB libres) sigue intacto, sin mitigar. Esto significa que el fix **no reduce el riesgo de que vuelva a crashear como en el "Intento 1"** — solo deja el modelo en un estado de reposo más liviano una vez que ya sobrevivió la carga.

**Detenido aquí, según lo pedido — no se ejecuta la campaña de 10 corridas ni se toca más código.** Antes de continuar, hace falta decidir una estrategia que ataque el *pico* en sí, no solo el estado posterior. Opciones a evaluar (ninguna implementada todavía, solo enumeradas para decisión):
1. **`torch.load(..., mmap=True)`** (soportado desde PyTorch 2.1+): en vez de deserializar el `.pt` completo a RAM de una sola vez, mapea el archivo a memoria virtual y el SO trae páginas bajo demanda — podría evitar que el checkpoint completo y el modelo instanciado coexistan enteros en RAM física a la vez. Riesgo/incógnita: compatibilidad con `weights_only=False` y con el formato exacto de este checkpoint (dict con clave `"model"`), no probado todavía.
2. **Construir el modelo en `torch.device("meta")`** (sin asignar memoria real para los parámetros) y cargar los pesos del checkpoint directamente en su lugar (`load_state_dict(..., assign=True)`) — evita el costo de "instanciar con pesos aleatorios" (que ya de por sí cuesta 8.4GB, la mitad del problema) seguido de sobreescribirlos.
3. **Convertir el checkpoint a `safetensors`** (ya es dependencia declarada en `pyproject.toml`) — permite mmap nativo y lectura perezosa tensor por tensor, en vez de deserializar un pickle monolítico de una vez.
4. Cargar el `state_dict` por partes/streaming (más complejo, depende de que el formato lo permita).
5. Mitigación de infraestructura, no de código: aumentar el pagefile de Windows para dar más margen antes de un crash — no reduce el pico real, solo el riesgo de que ese pico tumbe el proceso.

Ninguna de estas se implementó — quedan como opciones para que el usuario decida el rumbo antes de tocar más código o de correr la campaña de 10 repeticiones.

### Experimento: `device="meta"` + `load_state_dict(assign=True)` — análisis + prueba, DETENIDO por bloqueante de construcción (2026-08-23)

**Análisis estático previo (antes de correr nada):** por lectura de código se identificó un riesgo — no bloqueante para la construcción en sí, pero de corrección silenciosa — en `AggregatorStream.rope3d` (`lingbot_map/layers/rope.py:334`, siempre construido porque `enable_3d_rope=True` es el default fijo de `demo.py`):
```python
self.freqs = torch.cat(freqs, dim=1)   # atributo plano, NO nn.Parameter, NO register_buffer
```
Calculado con matemática real (`torch.arange`, `torch.outer`, `torch.polar`) en `__init__`. Bajo `device="meta"` heredaría `meta`; al no ser `nn.Parameter` ni buffer registrado, **nunca aparece en `state_dict()`**, así que `load_state_dict(assign=True)` nunca lo repara — y su uso posterior en `forward()` (`.to(device)`, rope.py:365) no recupera los valores reales, produce memoria sin inicializar sin lanzar error. Mismo patrón, menor impacto: `_resnet_mean`/`_resnet_std` (`aggregator/base.py:162`, buffers `persistent=False`).

Compatible sin cambios (confirmado por lectura): inicializaciones `nn.init.trunc_normal_`/`nn.init.normal_` sobre `nn.Parameter` reales (son no-op en tensores meta desde PyTorch ≥1.13, y sí se reparan vía `load_state_dict(assign=True)` por ser parámetros de verdad); `RotaryPositionEmbedding2D.frequency_cache` / `PositionGetter.position_cache` (rope 2D del backbone, cachés vacíos poblados perezosamente en el primer `forward()` real, no en `__init__`).

**Prueba diseñada** ([scripts/measure_load_meta.py](scripts/measure_load_meta.py)): dos variantes — `--naive` (meta+assign sin reparar, para confirmar empíricamente el tensor roto) y `--repair` (reconstruye `rope3d` y los buffers resnet en el dispositivo real después de `load_state_dict`, y verifica que `rope3d.freqs` quede bit-idéntico a una instancia de referencia construida de forma independiente — no solo "no crashea", sino numéricamente correcto).

**Resultado de la variante `--naive`: CRASH inmediato, antes de llegar a ninguna medición útil.**
```
RuntimeError: Tensor.item() cannot be called on meta tensors
```
en `lingbot_map/layers/vision_transformer.py:126`:
```python
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
```
`torch.linspace(...)` se construye sin `device=` explícito → hereda `meta` bajo el context manager → `.item()` sobre cada elemento intenta leer un valor escalar real de un tensor sin almacenamiento, y PyTorch lo rechaza explícitamente (esto sí es un error duro, no silencioso — mejor que el riesgo de `rope3d.freqs`, pero bloquea la construcción por completo). Esto ocurre en la **primera línea del backbone DINOv2** (`vit_large()`, llamada muy temprano en `GCTStream.__init__`) — no se llegó a ejecutar el resto de la construcción, así que **no se puede descartar que existan más puntos con el mismo patrón** (`.item()`/`.tolist()`/indexado escalar sobre un tensor construido sin `device=` explícito) más adelante en el código, todavía no explorados porque el primero ya detuvo la ejecución.

**Conclusión — cambia la evaluación de viabilidad de esta estrategia:** el análisis estático inicial sugería que "meta + assign" solo necesitaría reparar 2 casos conocidos *después* de cargar el checkpoint (post-proceso, sin tocar la librería). La prueba real muestra que **la construcción bajo `meta` en sí ya falla dentro de `lingbot_map` antes de llegar siquiera a `torch.load`** — para que esta estrategia funcione haría falta **modificar código dentro de `lingbot_map` (al menos `layers/vision_transformer.py:126`, posiblemente más puntos no descubiertos)**, no solo un script de carga externo. Es un nivel de invasividad mayor al que sugería el análisis por lectura sola.

**Detenido aquí — no se ejecutó la variante `--repair` ni se midió el pico de memoria real de esta estrategia**, porque hacerlo requeriría empezar a parchear código de la librería (aunque sea vía monkeypatch acotado al script de prueba) de forma reactiva, punto por punto, sin saber cuántos puntos faltan por descubrir — no es la comparación "limpia" contra el baseline de 16.1GB que se buscaba. Pendiente de decisión del usuario: (a) seguir parcheando reactivamente para llegar a una medición completa, aceptando que puede haber varias rondas más de crashes por el mismo patrón, o (b) descartar esta estrategia por su invasividad y pasar a evaluar `mmap`/`safetensors`, que no requieren tocar la arquitectura del modelo.

**Decisión del usuario (2026-08-23): descartada temporalmente por invasividad — pasar a `torch.load(mmap=True)` a continuación**, sin tocar `demo.py` ni ningún otro componente todavía.

### Experimento: `torch.load(..., mmap=True)` — mejora real confirmada (2026-08-23)

**Verificación de compatibilidad (antes de correr nada):**
- `torch.load` en `torch 2.13.0+cpu` (venv de este repo) tiene el parámetro `mmap` — soportado desde PyTorch 2.1.
- `lingbot-map.pt` es un archivo zip (formato moderno de `torch.save`), condición necesaria para `mmap=True`.
- Prueba liviana (sin construir el modelo, solo `torch.load(..., mmap=True)` + inspección de un tensor): completó en **0.6s**, tensor de muestra con `is_meta=False`, forma/dtype correctos. Compatible confirmado.

**Cambio aplicado — únicamente en un script de prueba, `demo.py` sigue intacto:** [scripts/measure_load_mmap.py](scripts/measure_load_mmap.py) es una copia exacta de `measure_load_only.py` (el script del baseline original de 16.1GB) con el único cambio pedido:
```python
ckpt = torch.load(args.model_path, map_location=device, weights_only=False, mmap=True)
```
Se optó deliberadamente por **no** aplicarlo todavía sobre `demo.py::load_model()` (que ya tiene el fix `del+gc.collect()` de la sección anterior) para que la comparación aísle exclusivamente el efecto de `mmap`, sin mezclarlo con esa otra intervención — tal como se pidió.

**Resultado — comparación completa contra el baseline original (sin mmap, sin del+gc.collect):**

| Métrica | Baseline (sin mmap) | Con `mmap=True` | Cambio |
|---|---|---|---|
| Memoria al instanciar el modelo | 8411.4 MB | 8410.9 MB | igual (esperado — mmap no toca esta etapa) |
| Tiempo de `torch.load` | ~70.2s | **0.7s** | **-99%** |
| **Pico de memoria comprometida** (peak_pagefile) | **16114.1 MB** | **13055.6 MB** | **-3058.5 MB (-19%)** |
| Pico de memoria privada (monitor externo, más fino) | 15367.6 MB | 12450.8 MB | -2916.8 MB (-19%) |
| **RAM libre mínima del sistema** (monitor externo, 1s) | **24 MB** | **174.5 MB** | **~7× más margen** |
| Memoria tras `load_state_dict` (antes de cleanup) | 16114.0 MB | 13055.5 MB | -3058.5 MB (-19%) |
| Memoria tras `del ckpt, state_dict; gc.collect()` | 12901.7 MB | **8414.1 MB** | **-4487.6 MB (-35%)** |
| Tiempo total (instanciación → `eval()`) | ≈244.2s | **≈110.6s** | **-133.6s (-55%)** |
| ¿Modelo funcional? | (no verificado con este método) | **Sí** — tensores de muestra con valores reales, sin NaN, `is_meta=False` | Confirmado |

**Por qué baja el pico:** sin `mmap`, `torch.load` deserializa el `.pt` completo a un buffer en RAM y *luego* construye los tensores del `state_dict` — durante una ventana breve coexisten el buffer serializado y los tensores ya deserializados, más el modelo ya instanciado (8.4GB), de ahí el pico de 16.1GB. Con `mmap=True`, los datos se leen directamente desde el archivo mapeado bajo demanda hacia los tensores del `state_dict`, sin ese buffer intermedio — de ahí la reducción de ~3GB en el pico observado, además de que `torch.load` en sí pasa de ~70s a <1s (no lee el archivo completo de inmediato, solo mapea).

**Matiz importante:** `load_state_dict` (que sí copia todos los datos a los parámetros del modelo, tocando cada página del mapeo) sigue tardando un tiempo comparable al baseline (~88-138s vs ~154s) — el ahorro de tiempo está concentrado en `torch.load`, no en `load_state_dict`. Y el contador `private`/`peak_pagefile` de Windows sube a ~13GB *inmediatamente* al hacer `torch.load(mmap=True)` (antes de tocar ningún dato) — es una reserva de compromiso virtual por el mapeo en sí, no RAM física tocada (el *working set* apenas sube en ese momento: 4905→4909MB), así que el número "pico" que baja es real pero no elimina por completo la sensación de "casi 13GB comprometidos" que muestra el Administrador de tareas.

**Conclusión: mejora significativa y real, en una sola línea, sin tocar la arquitectura del modelo.** A diferencia del fix `del+gc.collect()` (que no tocaba el pico), `mmap=True` sí reduce el pico real (~19%) y además acelera la carga total ~55%. Combinado con el fix de `del+gc.collect()` ya aplicado en `demo.py`, el estado final de reposo pasaría de 12.9GB a un estimado ~8.4GB (no medido combinado todavía, ya que se mantuvieron aislados intencionalmente).

**No se combinó con `meta`, `safetensors` ni quantization**, tal como se pidió. Pendiente de decisión del usuario: aplicar `mmap=True` a `demo.py::load_model()` (junto al fix ya aplicado), correr la campaña de 10 repeticiones con ambos cambios, o seguir evaluando `safetensors` antes de tocar el pipeline real.

## PRIMERA OPTIMIZACIÓN REAL DEL PIPELINE APLICADA: `mmap=True` en `demo.py::load_model()` (2026-08-23)

**Decisión del usuario:** aplicar `mmap=True` a la carga real del checkpoint en `demo.py`, manteniendo el `del ckpt, state_dict; gc.collect()` ya validado. Sin ningún otro cambio. El **baseline original (16.1GB de pico, sección "Reporte: prueba aislada de load_model()" arriba) se conserva intacto y sin editar** — esta sección documenta el estado optimizado, no lo reemplaza.

**Cambio real, único, en `demo.py::load_model()`:**
```python
ckpt = torch.load(args.model_path, map_location=device, weights_only=False, mmap=True)
```
(una palabra agregada a la llamada ya existente; el `del ckpt, state_dict; gc.collect()` de la optimización anterior permanece sin cambios inmediatamente después).

Con esto, `demo.py::load_model()` — el código de producción real, no una reimplementación de prueba — queda con **ambas optimizaciones validadas activas**: `mmap=True` (ataca el pico) + `del`/`gc.collect()` (ataca la memoria posterior a la carga).

**Verificación 1 — prueba aislada de `load_model()` real (`scripts/verify_load_fix.py`, sin imágenes/inferencia/visor):**

| Métrica | Experimento `mmap` aislado (script standalone) | `demo.load_model()` real (con `mmap`+`gc` juntos) | ¿Coincide? |
|---|---|---|---|
| Pico memoria comprometida (peak_pagefile) | 13055.6 MB | 13152.3 MB | Sí (+0.7%, ruido normal) |
| Pico memoria privada (monitor externo) | 12450.8 MB | 12543.0 MB | Sí (+0.7%) |
| RAM libre mínima del sistema (monitor externo) | 174.5 MB | 181.6 MB | Sí (+4%) |
| Memoria tras la carga (post `gc.collect()`) | 8414.1 MB | 8510.7 MB | Sí (+1.1%) |
| Tiempo de carga | ≈110.6s (aislado, sin `gc` combinado en la misma medición) | **97.67s** (`LOAD_TIME_S`, con `mmap`+`gc` juntos, código real) | Sí, del mismo orden |

Confirmado: el cambio real en `demo.py` reproduce fielmente los números del experimento aislado — no hay sorpresas al integrarlo al código de producción.

**Verificación 2 — demo completo con las 10 imágenes de `test_images/` (pipeline end-to-end, con visor):**
```powershell
python demo.py --model_path lingbot-map.pt --image_folder test_images --use_sdpa --camera_num_iterations 1
```
Resultado (`demo_10img_mmap_verify.log`, proceso desacoplado, exit limpio):
- **Carga del checkpoint: 90.0s** (vs 218.6s-254.4s en las corridas del pipeline completo antes de estas optimizaciones — mejora de ~60%, consistente con lo medido en aislamiento).
- Inferencia streaming de 10 frames: 823.1s (dentro del rango de variabilidad ya observado en esta máquina — 652.1s-726.9s en corridas previas; la fase de inferencia no la toca ninguna de las dos optimizaciones aplicadas, solo afectan la carga).
- Visor `viser` levantado correctamente en `http://localhost:8080`, sin errores ni crashes.

**Este es ahora el nuevo baseline optimizado del pipeline**, con dos cambios validados y activos en `demo.py::load_model()`: `mmap=True` + `del ckpt, state_dict; gc.collect()`. El baseline original (sin optimizar, 16.1GB de pico) permanece documentado arriba sin editar, como referencia histórica. **No se avanzó a `safetensors`, quantization, `device="meta"`, reducción de frames, ni optimizaciones inspiradas en Paragraphica** — quedan pendientes de decisión del usuario como próximos pasos.

## Auditoría de memoria del modelo ya cargado — diagnóstico puro, sin cambios (2026-08-23)

**Objetivo:** explicar la diferencia entre los ~4.63GB teóricos de parámetros FP32 y la memoria real del proceso tras la carga (con `mmap=True`+`gc.collect()` ya activos: ~8.5GB de memoria privada en esta corrida). Puramente diagnóstico — no se tocó dtype, arquitectura, quantization ni el pipeline de carga/inferencia.

**Script:** [scripts/audit_model_memory.py](scripts/audit_model_memory.py) — reutiliza `demo.load_model()` y `demo.load_images()` **reales, sin modificarlos**. Camina `model.named_children()` recursivamente sumando `numel()×element_size()` de parámetros y buffers por componente, y hace un **chequeo cruzado independiente**: enumera *todos* los tensores PyTorch vivos vía `gc.get_objects()`, deduplicados por identidad de `storage()` (para no contar dos veces tensores que comparten memoria), y compara ese total contra la enumeración por módulos y contra la memoria de proceso medida.

### Desglose por componente (params, dtype, memoria de parámetros)

| Componente | Parámetros | dtype | Memoria params (MB) | Memoria buffers (MB) |
|---|---|---|---|---|
| **aggregator** (total) | 909,114,368 | float32 | 3636.5 | 0.0 |
| ↳ `aggregator.patch_embed` (**backbone DINOv2** ViT-L/14 + register tokens) | 304,372,736 | float32 | **1217.5** | 0.0 |
| ↳ `aggregator.frame_blocks` (**GCT**, atención por-frame) | 302,364,672 | float32 | **1209.5** | 0.0 |
| ↳ `aggregator.global_blocks` (**GCT**, atención global) | 302,364,672 | float32 | **1209.5** | 0.0 |
| ↳ tokens especiales directos (`camera_token`, `register_token`, `scale_token`) | 12,294 | float32 | ~0.05 | 0.0 |
| ↳ `aggregator.rope3d.freqs` (**otro tensor relevante** — atributo plano, no `state_dict`, ver sección "device=meta" arriba) | 32,768 | **complex128** | 0.52 | — |
| **camera_head** | 216,174,610 | float32 | 864.7 | 0.0 |
| **depth_head** (DPTHead) | 32,654,562 | float32 | 130.6 | 0.0 |
| `point_head` / `local_point_head` | — | — | no construidos (`enable_point=False`, `enable_local_point=False` en esta config) | — |
| **TOTAL** | **1,157,943,540** | | **4632.3** | **0.0** |

Suma exacta: `909,114,368 + 216,174,610 + 32,654,562 = 1,157,943,540` — coincide con el conteo global ya reportado. **Memoria "estimada de DINOv2"** (el backbone visual, `patch_embed`) = 1217.5MB / 304.4M params (26% del modelo). **Memoria "del GCT propiamente dicho"** (`frame_blocks`+`global_blocks`, los bloques de atención temporal/espacial añadidos sobre el backbone) = 2419.0MB / 604.7M params (52% del modelo) — es la parte más grande. `camera_head`+`depth_head` = 995.3MB / 248.8M params (22%). **Memoria de buffers real: prácticamente 0** en todos los componentes — no hay buffers grandes escondidos; el único "tensor fuera del sistema de módulos" es `rope3d.freqs` (0.52MB, irrelevante en magnitud).

### Reconciliación — la respuesta a "¿de dónde sale la diferencia?"

| Medición | Valor |
|---|---|
| Memoria teórica de parámetros (params × 4 bytes, float32) | 4631.8 MB |
| Memoria enumerada por módulos (parámetros + buffers + `rope3d.freqs`) | 4632.3 MB |
| **Memoria de TODOS los tensores PyTorch vivos** (gc, deduplicado por storage) | **4632.3 MB** |
| Memoria de proceso medida (private) tras la carga | 8511.8 MB |
| **NO CONTABILIZADO** (proceso − tensores vivos) | **3879.5 MB** |

**Hallazgo central: no hay duplicación de tensores ni referencias residuales del checkpoint.** El chequeo cruzado por `gc` (que enumera *cualquier* tensor vivo en el proceso, no solo los que cuelgan de `model.parameters()`) da **exactamente el mismo número** que la suma por módulos (4632.3MB ambos) — si hubiera una copia adicional del `state_dict`, un `ckpt` sin liberar, o buffers duplicados en otro lugar, el conteo de `gc` sería mayor que el de módulos, y no lo es. El `del ckpt, state_dict; gc.collect()` ya validado limpia completamente lo que promete: cero residuos.

Los ~3.88GB "no contabilizados" **no son memoria de ningún tensor vivo** — son overhead del *allocator* del proceso (C10/PyTorch + el runtime de Windows) que no se ha devuelto al sistema operativo tras el pico transitorio de carga (`peak_pagefile` llegó a 13153.3MB durante `torch.load(mmap=True)`+`load_state_dict`, según el log de esta misma corrida). Es un patrón típico de *allocators* con caché: liberan la memoria a nivel de Python/PyTorch, pero retienen el bloque de memoria del proceso para reutilización futura en vez de devolverlo inmediatamente al SO — de ahí que el proceso siga "pesando" 8.5GB aunque solo 4.63GB estén realmente en uso por tensores vivos. No es una fuga ni un bug del modelo: es comportamiento normal del allocator, y no hay evidencia de que sea corregible sin tocar configuración del allocator (fuera del alcance de este diagnóstico).

### Memoria antes/después de la primera inferencia — separando modelo permanente de memoria temporal

Con 3 imágenes reales de `test_images/` (todas como "scale frames", sin frames streaming individuales — la vía rápida del pipeline):

| Momento | Tensores vivos (gc) | Memoria de proceso (private) |
|---|---|---|
| Antes de la primera inferencia | 4639.6 MB | 8530.1 MB |
| Después de la primera inferencia | 5875.9 MB | 10693.5 MB |
| **Delta atribuible a la inferencia** | **+1236.2 MB** | **+2163.4 MB** |

La inferencia en sí (activaciones, KV cache inicial, tensores de predicción — `pose_enc, depth, depth_conf, images, frame_type, is_keyframe`) añade **~1.24GB de tensores realmente vivos**, verificado por el mismo chequeo cruzado de `gc`. La memoria de proceso sube más (~2.16GB) que los tensores vivos (~1.24GB) — mismo patrón que en la carga: el delta de proceso es mayor que el delta de tensores vivos, indicando que también durante la inferencia el allocator retiene algo de margen adicional sin devolverlo al SO, aunque en proporción bastante menor que durante la carga.

**Conclusión práctica:** de los ~10.7GB de memoria de proceso tras cargar + una inferencia mínima, ~5.9GB son tensores PyTorch genuinamente vivos (4.63GB modelo permanente + 1.24GB de la inferencia), y ~4.8GB son overhead de allocator no liberado — la mayor parte de ese overhead (~3.9GB) ya estaba presente desde la fase de carga, no lo genera la inferencia. Confirma que el modelo permanente es "solo" ~4.63GB, tal como sugiere la teoría — el resto de lo que se observa en el Administrador de tareas es un artefacto del ciclo de vida del allocator durante la carga, no memoria "necesaria" en un sentido estricto.

**No se hicieron cambios de código en `demo.py` ni en `lingbot_map` — auditoría puramente diagnóstica**, tal como se pidió. Pendiente de decisión del usuario: evaluar `safetensors`, iniciar la campaña de 10 repeticiones, o investigar si el comportamiento del allocator es ajustable.

## Investigación: conversión a `safetensors` — nivel checkpoint únicamente (2026-08-23)

**Alcance deliberadamente acotado** (pedido explícito del usuario): solo analizar/convertir/verificar el checkpoint en un script independiente. **No se tocó `demo.py`, no se tocó `lingbot-map.pt` original, no se instanció `GCTStream`, no se corrió la campaña de 10 repeticiones.**

**Congelado como referencia (baseline actual, punto de partida de esta investigación):** checkpoint `.pt` + `torch.load(mmap=True)` + `del ckpt, state_dict; gc.collect()` → pico ≈13.1GB, modelo funcional, 1,157,943,540 parámetros, 4.63GB de parámetros FP32.

### Paso 1 — Análisis de compatibilidad (antes de convertir nada)

```
1342 tensores, dtypes presentes: ['torch.float32'] (uniforme, sin sorpresas)
tensores no contiguos: 0
grupos con storage compartido (weight-tying / aliasing): 0
```

Sin bloqueantes conocidos de `safetensors` (que requiere tensores contiguos y sin memoria compartida entre distintas claves, salvo manejo especial). Checkpoint "ideal" para conversión directa.

### Paso 2 — Conversión (copia, no toca el original)

Script: [scripts/audit_safetensors_conversion.py](scripts/audit_safetensors_conversion.py). `lingbot-map.pt` → `lingbot-map_converted.safetensors` (gitignored, archivo local de prueba).

```
Conversión: 138.21s
Tamaños: lingbot-map.pt = 4.632GB | lingbot-map_converted.safetensors = 4.632GB (diff: -0.38MB, safetensors ligeramente más chico — header más compacto que el contenedor zip+pickle de PyTorch)
```

### Paso 3 — Verificación (conteo, nombres, shapes, dtypes, valores)

```
load_file() (carga eager) recargó 1342 tensores en 0.42s
key sets identical: True
shape mismatches: 0 | dtype mismatches: 0 | value mismatches (torch.equal, bit-exacto): 0
VERIFICACIÓN: PASSED
```

**Conversión exacta y sin pérdida** — los 1342 tensores son bit-idénticos al original, mismos nombres, mismas formas, mismos dtypes.

### Paso 4 — Viabilidad de estrategia de carga/mapeo con `safetensors`

- `load_file()` (carga eager, todo el dict a RAM de una vez): **0.42s** para 4.63GB — más rápido incluso que `torch.load(mmap=True)` (0.6-0.7s en pruebas anteriores), y muchísimo más rápido que `torch.load` sin mmap (~70-90s).
- `safe_open()` (acceso perezoso por tensor, **mmap nativo siempre activo en `safetensors`**, no opcional como en `torch.load`): abrir el archivo (solo parsear el header) tomó 1.934s; materializar un único tensor pequeño vía `get_tensor()` tomó 1.253s — más lento de lo esperado para un tensor trivial (8KB), probablemente overhead de primer acceso a página de un archivo de 4.6GB recién mapeado (caché de disco fría), no representativo del costo de accesos subsiguientes. No se investigó más a fondo — es una observación, no un hallazgo concluyente.
- **Compatibilidad de claves con `GCTStream` ya confirmada indirectamente**: los 1342 nombres de tensores en el checkpoint son exactamente los mismos que ya reportaron `missing=0 unexpected=0` en cada carga anterior (baseline, `mmap`, `verify_load_fix`) — no hizo falta re-instanciar el modelo en este script para confirmar compatibilidad de nombres, ya está establecido.
- `safetensors` permite tanto una estrategia "eager" (equivalente a lo que ya hacemos con `mmap=True`, pero más rápida) como una estrategia "lazy por tensor" (potencialmente más eficiente en memoria que cargar todo el `state_dict` de una vez — no medida todavía a nivel de pico de RAM real dentro del pipeline).

### Conclusión — detenido aquí, tal como se pidió

**La conversión y verificación son correctas.** El checkpoint es 100% compatible con `safetensors` sin necesidad de ningún tratamiento especial (sin tensores no-contiguos, sin storage compartido), la conversión es exacta (bit-idéntica), y `load_file()` es más rápido que la estrategia `mmap=True` actual solo para la etapa de lectura del archivo. **No se ha medido todavía** el efecto de `safetensors` sobre el pico de memoria real dentro de `load_model()` (instanciación del modelo + `load_state_dict`) — eso requeriría el mismo tipo de experimento aislado que se hizo para `mmap` (`scripts/measure_load_mmap.py`), pendiente de que el usuario lo pida como siguiente paso.

**No se modificó `demo.py` ni el checkpoint original. No se ejecutó la campaña de 10 repeticiones.**

## Experimento aislado de pico de memoria con `safetensors` — resultado MIXTO, pipeline real NO modificado (2026-08-23)

**Metodología:** réplica exacta de `scripts/measure_load_mmap.py` (mismos stages, misma instrumentación `ctypes`, mismo monitor externo a 1s), cambiando solo la estrategia de carga del checkpoint. Script: [scripts/measure_load_safetensors.py](scripts/measure_load_safetensors.py).

**Estrategia probada — carga "streaming" tensor-por-tensor, no `load_file()`:** en vez de `load_file()` (que materializaría el `state_dict` completo como un solo dict, estructuralmente igual al riesgo que ya vimos con `torch.load`), se usó `safe_open()` + un bucle que copia cada tensor **uno a la vez** directamente al parámetro correspondiente del modelo (`param.data.copy_(f.get_tensor(name)); del t`), sin mantener nunca el `state_dict` completo como objeto separado — la estrategia "evitar materializaciones innecesarias" pedida. Es una reimplementación manual fiel de lo que `load_state_dict(strict=False)` hace por tensor internamente, alimentada desde `safe_open()` en vez de un dict pre-armado.

### Resultado — comparación de las tres estrategias

| Métrica | Baseline (sin optimizar) | `mmap=True` (actual, en producción) | `safetensors` (streaming, experimental) |
|---|---|---|---|
| Memoria al instanciar el modelo | 8411.4 MB | 8410.9 MB | 8411.5 MB (igual, esperado) |
| **Pico de memoria comprometida** (peak_pagefile, contador interno de Windows) | 16114.1 MB | 13055.6 MB | **17693.8 MB — el peor de los tres** |
| Pico de memoria privada (monitor externo, muestreo a 1s) | 15367.6 MB | 12450.8 MB | 12448.4 MB (≈ igual a `mmap`) |
| **RAM libre mínima del sistema** (monitor externo) | 24 MB | **174.5 MB (mejor)** | 126.4 MB |
| Memoria tras la carga (post `gc.collect()`) | 12901.7 MB | 8414.1 MB | **8411.9 MB (≈ igual a `mmap`, incluso 2.2MB mejor)** |
| Tiempo total de carga | ≈244.2s | **≈110.6s (mejor)** | 139.4s |
| ¿Funcional? | — | Sí | Sí, verificado (tensores reales, sin NaN) |
| `missing`/`unexpected` al comparar claves | 0/0 | 0/0 | 2/0 — **ver nota abajo, no es un problema real** |

**Nota sobre `missing=2`:** no es una incompatibilidad de `safetensors`. Mi script comparó las claves del checkpoint contra `model.named_parameters()` + `model.named_buffers()` — y `named_buffers()` incluye los buffers con `persistent=False` (`_resnet_mean`, `_resnet_std`, ver sección "device=meta" arriba), que **nunca estuvieron en ningún checkpoint** (ni en el `.pt` original) porque por diseño no se guardan. El `load_state_dict()` estándar de PyTorch compara contra `model.state_dict()`, que excluye buffers no-persistentes — por eso el baseline siempre reportó `missing=0`. Es una diferencia metodológica de mi script de prueba, no un hallazgo sobre `safetensors`.

### El hallazgo central: el pico transitorio, no el patrón general

El **contador interno de Windows** (que trackea el máximo histórico real del proceso, sin importar la frecuencia de muestreo) detectó un pico de **17693.8MB durante `safe_open()` mismo** (antes de copiar ningún tensor) — más alto que el pico del baseline sin optimizar. El **monitor externo** (muestreo cada 1s) nunca vio ese valor — su pico máximo fue 12448.4MB, casi idéntico al de `mmap`. Esto significa que el pico de 17.7GB fue un evento **muy breve y agudo** (sub-segundo), probablemente durante cómo `safe_open()` internamente mapea/reserva el archivo al abrirlo — no sostenido en el tiempo, pero real y detectado por el contador que no depende de sampling.

También hay una brecha grande entre memoria comprometida (`peak_pagefile`=17693.8MB) y memoria físicamente residente (`peak_ws`=5984.4MB) en este experimento — mayor que en `mmap`. Sugiere que buena parte de ese pico nunca fue RAM física real simultánea, pero el compromiso de memoria virtual (que sí puede causar fallos de asignación / crashes, como vimos en el "Intento 1" del baseline original) fue más agresivo que con `mmap=True`.

**Hipótesis no investigadas para explicar el pico** (quedan abiertas, no confirmadas): el backend en Rust de `safetensors` podría reservar un mapeo con sobre-compromiso al abrir el archivo (más agresivo que el mapeo de `torch.load(mmap=True)`), o el patrón de 1342 asignaciones/liberaciones pequeñas en el bucle Python podría fragmentar el allocator. No se profundizó más — fuera del alcance de esta ronda.

### Conclusión — NO mejora el pico, pipeline real sin modificar

**`safetensors` (con esta estrategia de carga) NO reduce el pico de memoria — lo empeora respecto al `mmap=True` ya en producción.** Sí iguala (o mejora levemente) el estado de reposo tras la carga, pero eso ya lo teníamos con `mmap`+`gc.collect()`. El tiempo total también fue peor que `mmap` (139.4s vs 110.6s), aunque mejor que el baseline sin optimizar.

**Siguiendo la instrucción del usuario: como `safetensors` no mejora significativamente el pico (de hecho lo empeora), se documenta este resultado y NO se modifica `demo.py` ni el pipeline real.** El baseline optimizado sigue siendo `.pt` + `mmap=True` + `del/gc.collect()`, sin cambios.

**No investigado todavía (posible siguiente paso si se quiere seguir esta línea):** probar la estrategia "eager" simple de `safetensors` (`load_file()` + `model.load_state_dict(dict)`) en vez del streaming tensor-por-tensor — podría tener un patrón de pico distinto al observado aquí, ya que no pasaría por el bucle de 1342 asignaciones/liberaciones pequeñas. No se ejecutó en esta ronda.

**Línea de investigación de `safetensors` cerrada temporalmente (decisión del usuario, 2026-08-23).** `.pt` + `mmap=True` sigue siendo el mejor baseline conocido.

## Análisis: reducción de precisión a FP16 — antes de cualquier código (2026-08-23)

**Frontera de precisión ya definida por los propios autores, no una hipótesis nueva.** Evidencia directa en el código:

- **`demo.py` (comentario original, líneas 467-471):** "Cast the aggregator (DINOv2-style trunk) to the inference dtype to remove the redundant fp32 master weight copy + autocast bf16 weight cache (~2-3 GB saved, **no measurable quality change**). `gct_base._predict_*` upcasts inputs to fp32 and runs each head under `autocast(enabled=False)`, so camera/depth/point heads keep fp32 weights automatically." — este casting de producción (GPU) **ya existe en el código**, solo está inactivo en CPU porque `dtype = torch.float32` cuando `not torch.cuda.is_available()`.
- **Confirmado en `lingbot_map/models/gct_base.py`:** `_predict_camera`, `_predict_depth`, `_predict_point`, `_predict_local_point` — las 4 cabezas — hacen `.float()` explícito sobre sus entradas y envuelven la llamada en `torch.amp.autocast('cuda', enabled=False)`, forzando FP32 sin importar el dtype del resto del modelo.

**Conclusión de la frontera segura (según diseño de los autores, ya validado en GPU):**
- **Seguro para FP16:** `model.aggregator` — backbone DINOv2 (`patch_embed`) + bloques GCT (`frame_blocks`, `global_blocks`). Es el 79% del modelo (3636.5MB de los 4632.3MB totales, ver auditoría arriba).
- **Debe quedarse en FP32:** `camera_head`, `depth_head` (y `point_head`/`local_point_head` si estuvieran activos) — 21% del modelo (995.3MB). Forzado por diseño explícito (`.float()` + `autocast(enabled=False)`), no por precaución nuestra.

**Riesgo real específico de este experimento (CPU, no GPU) — no presente en la validación original de los autores:**
1. En producción, el cast a FP16/BF16 siempre corre bajo `torch.amp.autocast('cuda', dtype=dtype)`, que en GPU sube automáticamente a FP32 operaciones sensibles (LayerNorm, softmax) aunque los pesos estén en baja precisión — es una red de seguridad activa. **`torch.amp.autocast` en CPU solo soporta `bfloat16`, no `float16`** — así que un cast estático a FP16 en CPU, sin autocast, ejecuta esas operaciones sensibles en FP16 puro, sin la protección que los autores validaron en GPU.
2. Esta CPU (AMD Ryzen 5 3500U, Zen+) no tiene AVX512 — ni FP16 ni BF16 tienen aceleración de hardware real aquí; es emulado, y algunos kernels de PyTorch CPU directamente no implementan `Half` (pueden lanzar `RuntimeError: "X" not implemented for 'Half'`).

**Por lo tanto: el resultado de "no measurable quality change" de los autores es válido para GPU+autocast, no se puede asumir válido para CPU sin autocast — se prueba empíricamente, no se asume.**

### Experimento aislado: FP16 en el `aggregator`, FP32 en las cabezas (2026-08-23)

**Script:** [scripts/measure_load_fp16.py](scripts/measure_load_fp16.py) — corre **dos modelos completos, secuencialmente** (libera el primero antes de construir el segundo, para no duplicar picos), sobre las mismas 3 imágenes reales de `test_images/`: Fase A = FP32 baseline (idéntico al `mmap=True` actual), Fase B = `model.aggregator.to(dtype=torch.float16)` aplicado **antes** de `load_state_dict` (así el `.copy_()` interno hace el downcast fp32→fp16 automáticamente por tensor, sin `assign=True` ni tocar `meta`). No modifica `demo.py`.

**Primer intento: falló exactamente como anticipaba el análisis.** `RuntimeError: Input type (float) and bias type (struct c10::Half) should be the same` en el primer `Conv2d` del `patch_embed` — las imágenes de entrada seguían en float32 (nunca se tocan en el pipeline normal) y no había `autocast` para convertirlas al vuelo como en GPU. Se aplicó un fix mínimo **solo en el script de prueba** (`images = images.half()` antes de la inferencia en la Fase B) — un único punto de fallo conocido y acotado, no un patrón sistemático como en el experimento de `meta`, así que se completó el experimento en vez de detenerse.

### Resultado — comparación FP32 vs FP16(aggregator)+FP32(heads)

| Métrica | FP32 (baseline, `mmap=True`) | FP16 aggregator + FP32 heads |
|---|---|---|
| Memoria de parámetros | 4631.8 MB (float32) | **2813.5 MB** — 1818.2MB float16 (aggregator) + 995.3MB float32 (heads) |
| Pico de memoria comprometida durante la carga | 13152.7 MB | **13152.7 MB — idéntico** |
| Memoria tras la carga (`gc.collect()`) | 8511.2 MB | **5109.1 MB (-40%)** |
| Tiempo de carga | 86.10s | **65.65s (-24%)** |
| **Tiempo de inferencia** (3 imágenes) | 208.58s | **828.42s — 4.0× MÁS LENTO** |
| Pesos: NaN/Inf | — | 0 / 0 |
| RAM libre mínima (todo el experimento, monitor externo) | 196.9 MB | (misma corrida, ambas fases) |

**El pico de memoria durante la carga NO baja con FP16.** El pico ocurre en `torch.load(mmap=True)`, que lee el checkpoint **tal cual está guardado (float32)** desde disco — el cast a FP16 solo afecta el tensor de *destino* (el parámetro del modelo), no el *origen* (el `state_dict` recién deserializado). Mismo patrón ya visto con el fix `del+gc.collect()`: reduce el estado de reposo, no el momento de mayor riesgo de crash.

**La inferencia es 4× más lenta, no más rápida — el hallazgo más importante.** Confirma el riesgo anticipado en el análisis: esta CPU (Ryzen 5 3500U, sin AVX512) no tiene aceleración de hardware para FP16, así que cada operación se emula, con overhead de conversión en vez de ahorro. FP16 aquí es una estrategia de **memoria pura**, y tiene un costo de **rendimiento severo** — no es un "más liviano y más rápido" como sería en GPU con tensor cores.

**Estabilidad numérica — no catastrófica, pero con desviaciones no triviales, no "sin cambio medible":**

| Salida | max abs diff | mean abs diff | max rel diff | mean rel diff | NaN/Inf |
|---|---|---|---|---|---|
| `pose_enc` | 0.000184 | 0.000045 | **30.8%** | 4.25% | 0 / 0 |
| `depth` | **0.691** | 0.00154 | **25.7%** | 0.12% | 0 / 0 |

Sin `NaN`/`Inf` — el modelo no "explota" numéricamente. Pero hay desviaciones relativas máximas del 26-31% en algunos elementos (aunque el promedio es mucho más bajo, 0.1-4.3%) y una diferencia absoluta máxima de profundidad de 0.69 unidades — no despreciable para un pipeline de mapeo 3D. Esto contradice, para esta configuración específica (CPU, sin autocast), la nota de "no measurable quality change" que los autores documentaron para GPU+autocast — confirma que esa validación no se traslada automáticamente a este entorno.

### Conclusión

**FP16 (aggregator) en esta máquina recupera memoria real (-40% tras la carga, -39% en parámetros) pero al costo de una inferencia 4× más lenta y desviaciones numéricas no triviales — un mal trade-off para esta CPU sin aceleración de hardware.** No reduce el pico durante la carga (el riesgo de crash que motivó toda esta línea de investigación sigue intacto). **No se modificó `demo.py` ni el pipeline real.** No se combinó con quantization INT8/INT4, cambios arquitectónicos, reducción de frames ni Paragraphica, tal como se pidió.

**Nota para contexto de hardware futuro:** este resultado es específico de una CPU sin AVX512. En una GPU (el "Hardware objetivo (progresivo)" de la sección de arriba incluye "2. PC con GPU modesta") o en una CPU con soporte real de FP16/BF16, el resultado de tiempo de inferencia probablemente sea muy distinto (los propios autores ya validan "no measurable quality change" + ahorro real en GPU) — no descartar FP16 en general, descartarlo para *esta* máquina CPU-only tal como está.

**Línea FP16 en CPU cerrada temporalmente (decisión del usuario, 2026-08-23).** Resultado conservado como posible estrategia futura para GPU, no para esta máquina.

## Análisis: viabilidad de cuantización INT8 weight-only (2026-08-23)

**`model.to(torch.int8)` NO es una estrategia de cuantización válida — se descarta explícitamente antes de cualquier prueba.** INT8 no es solo "menos bits que float32": para representar valores reales en un rango de 8 bits con signo (-128 a 127) hace falta una **escala y un zero-point** (por tensor o por canal) que mapeen el rango real de los pesos al rango entero. Un cast de dtype directo (`.to(torch.int8)`) trunca/envuelve los valores flotantes en ese rango sin ningún mapeo — destruye el modelo, no lo comprime. La cuantización real requiere una API dedicada que calcule y almacene esa escala.

### Verificación de compatibilidad con `GCTStream` (código, no supuesto)

- `attention.py`: `self.qkv`, `self.proj`, `self.gate_proj` son **instancias reales de `nn.Linear`** (no matmul manual con `nn.Parameter`).
- `mlp.py`: `self.fc1`, `self.fc2` — `nn.Linear`.
- `swiglu_ffn.py`: `self.w12`, `self.w3` — `nn.Linear`.
- `patch_embed.py`: `self.proj` es **`nn.Conv2d`**, no `nn.Linear` — la proyección inicial de parches del backbone DINOv2, un componente trivial en parámetros (~600K de los 304M del backbone).
- Entorno: `torch.ao.quantization.quantize_dynamic` disponible en `torch 2.13.0` **sin instalar ninguna librería adicional**. Backend cuantizado disponible: `onednn` (sucesor de fbgemm/MKL-DNN para x86 en builds recientes de PyTorch) — funcional de fábrica.

### Estrategia viable: cuantización dinámica (`torch.ao.quantization.quantize_dynamic`), no estática

- **Dinámica** (la que se evalúa aquí): cuantiza los **pesos** de `nn.Linear` de forma estática una vez (a `qint8`, con escala calculada del propio tensor de pesos), pero las **activaciones** se cuantizan dinámicamente en cada forward (calculadas al vuelo, sin necesidad de un dataset de calibración) y se des-cuantizan de vuelta a FP32 tras el matmul. Es la definición operativa de "weight-only" en el ecosistema de PyTorch — no requiere calibración, no requiere insertar `QuantStub`/`DeQuantStub`, no requiere fusionar módulos. Coincide con lo pedido ("comenzando por weight-only INT8").
- **Estática** (descartada para esta ronda): cuantiza también las activaciones de forma fija, requiere un pase de calibración con datos representativos y modificar la arquitectura (stubs, fusión de módulos) — más invasiva, no es "weight-only". No se evalúa en este experimento.
- `quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` reemplaza automáticamente **cualquier instancia de `nn.Linear`** en el árbol de módulos, sin importar qué clase custom la contenga (`Attention`, `Mlp`, `SwiGLUFFN`) — no hace falta modificar `lingbot_map` para que la detecte.

### Qué queda cuantizado y qué no (por diseño de la API, no por elección nuestra)

- **Se cuantiza:** los pesos de `qkv`, `proj`, `gate_proj`, `fc1`/`fc2`, `w12`/`w3` — es decir, la inmensa mayoría de los parámetros de `frame_blocks` y `global_blocks` (el GCT propiamente dicho, 604.7M de los 1157.9M parámetros totales) y del backbone DINOv2 dentro de `patch_embed` (menos el conv2d inicial).
- **No se cuantiza (la API no lo soporta de forma estándar):** `patch_embed.proj` (`nn.Conv2d`, trivial en tamaño), `LayerNorm`, la matemática de atención/softmax/SDPA, RoPE (`rope`, `rope3d.freqs`) — todo permanece en FP32 automáticamente, sin intervención nuestra.
- **Frontera elegida para este experimento (misma que en FP16, por consistencia y porque ya está justificada arquitectónicamente):** cuantizar solo `model.aggregator` (`patch_embed` + `frame_blocks` + `global_blocks`); mantener `camera_head`/`depth_head` sin cuantizar (FP32) — mismo argumento que en FP16: `gct_base.py` fuerza `.float()` + `autocast(enabled=False)` en las 4 funciones `_predict_*`, señal explícita de los autores de que las cabezas necesitan precisión completa.

### Restricción de orden ya identificada — el pico durante la carga NO va a mejorar (documentado antes de correr nada)

A diferencia de FP16 (donde el cast se pudo aplicar *antes* de `load_state_dict`), la cuantización dinámica **requiere que el modelo ya tenga los valores reales de los pesos** — necesita leer cada tensor de peso para calcular su escala de cuantización antes de convertirlo. El orden obligatorio es: **cargar el modelo completo en FP32 (mismo pico que el baseline, ~13.1GB) → después convertir a cuantización dinámica.** El pico durante la carga no puede bajar con esta estrategia — se anticipa el mismo resultado que en FP16 en ese aspecto específico, y se mide para confirmarlo, no para descubrirlo por sorpresa.

### Hipótesis de rendimiento — a diferencia de FP16, aquí SÍ se espera aceleración real

FP16 en esta CPU fue 4× más lento porque no hay aceleración de hardware para FP16 (sin AVX512). La cuantización INT8 dinámica en CPU x86 vía `onednn` **es exactamente el caso de uso para el que esta API fue diseñada** — acelerar matmuls de transformers en inferencia CPU con instrucciones enteras SIMD. La hipótesis a probar (no asumida): tiempo de inferencia mejor o igual a FP32, no peor — contrario al resultado de FP16.

**Viabilidad confirmada, sin bloqueantes de librerías ni de arquitectura. Se procede a diseñar la prueba aislada.**

**Hallazgo de viabilidad a largo plazo (no bloqueante hoy):** `torch.ao.quantization.quantize_dynamic` está **deprecado** en `torch 2.13.0` (`DeprecationWarning: torch.ao.quantization is deprecated and will be removed in 2.10`). Funciona sin problema para este experimento, pero para una eventual adopción en producción, PyTorch recomienda migrar a `torchao` (`torchao.quantization.quantize_`) — una librería **adicional**, no instalada actualmente en este entorno. Documentado aquí, no evaluado en esta ronda.

### Verificación de compatibilidad — smoke test con pesos aleatorios (antes del experimento completo)

Antes de comprometerse a la corrida completa (~15-20 min), se corrió `quantize_dynamic` sobre el `aggregator` con pesos aleatorios (sin cargar el checkpoint): **288 capas `nn.Linear` encontradas, las 288 reemplazadas por su versión cuantizada (cobertura 100%)**, forward pass con datos sintéticos exitoso sin errores de dtype — a diferencia de FP16, **no hizo falta convertir el input**: la cuantización dinámica "weight-only" mantiene la interfaz pública del módulo en float32 (cuantiza/decuantiza activaciones internamente, de forma transparente). Viabilidad técnica confirmada antes de gastar tiempo en el experimento completo.

### Experimento aislado con el checkpoint real (2026-08-23)

**Script:** [scripts/measure_load_int8.py](scripts/measure_load_int8.py) — misma metodología de dos fases que FP16 (FP32 baseline, luego INT8, sobre las mismas 3 imágenes reales, liberando la fase A antes de construir la B). Restricción ya documentada arriba confirmada en la práctica: hubo que cargar el modelo completo en FP32 (mismo pico ~13.2GB) **antes** de poder cuantizar — `quantize_dynamic` necesita los valores reales de los pesos para calcular la escala de cuantización, no puede aplicarse antes de cargar como sí se pudo con FP16.

| Métrica | FP32 (baseline, `mmap=True`) | INT8 dinámico (aggregator) |
|---|---|---|
| Capas `nn.Linear` cuantizadas | — | 288 / 288 (100%) |
| Pico durante la carga (antes de cuantizar) | 13152.9 MB | 13269.2 MB (≈ igual, esperado) |
| **Pico durante la conversión a INT8** | — | **17008.2 MB — el segundo peor pico de toda la investigación** |
| **RAM libre mínima (todo el experimento, monitor externo)** | — | **8.7 MB — el peor resultado de toda la investigación, peor que el baseline sin optimizar (24MB)** |
| Tiempo de carga (FP32) | 111.05s | 92.54s (parte FP32) |
| Tiempo de conversión a INT8 | — | **169.76s** |
| Tiempo total de carga | 111.05s | **262.30s — 2.4× más lento que el baseline** |
| Memoria tras cargar/convertir (antes de inferencia) | 8511.4 MB | **12456.4 MB — peor que FP32, ver nota abajo** |
| **Tiempo de inferencia** (3 imágenes) | 185.23s | **168.98s — 8.8% más rápido, la única mejora de velocidad observada en toda la investigación** |
| Salidas: NaN/Inf | — | 0 / 0 |
| Compatibilidad con CPU | — | Confirmada, backend `onednn`, sin librería adicional |

**Nota metodológica importante — memoria post-conversión probablemente inflada por el script, no necesariamente por la técnica:** a diferencia de los demás experimentos, este script **no llamó `gc.collect()` inmediatamente después de `quantize_dynamic()`** (solo antes, tras `load_state_dict`). Es plausible que los pesos FP32 originales del `aggregator` no se hayan liberado del todo antes de medir "memoria tras cargar/convertir", inflando tanto ese número como el pico de conversión. **No se repitió el experimento con ese ajuste** — se documenta como limitación identificada y candidato a verificar antes de sacar conclusiones definitivas sobre el footprint final, en vez de asumir que 12456.4MB es el verdadero mínimo alcanzable con esta técnica.

**Estabilidad numérica — significativamente peor que FP16, la señal más preocupante de este experimento:**

| Salida | max abs diff | mean abs diff | max rel diff | mean rel diff | NaN/Inf |
|---|---|---|---|---|---|
| `pose_enc` | 0.00597 | 0.00175 | **489.7%** | **78.3%** | 0 / 0 |
| `depth` | **4.540** | **0.104** | **285.1%** | **7.9%** | 0 / 0 |

Comparar contra FP16 (misma tabla, sección anterior): `pose_enc` max rel diff 30.8% vs **489.7%** aquí; `depth` max abs diff 0.69 vs **4.54** aquí. Sin `NaN`/`Inf` — no hay colapso catastrófico — pero las diferencias relativas promedio (7.9%-78.3%) y máximas (285%-490%) son de un orden de magnitud mayor que FP16. Esto sugiere que la cuantización dinámica automática (calibración min/max por tensor, sin dataset representativo, sin ajuste fino posterior) es agresiva para esta arquitectura tal como se aplicó — cuantizando las 288 capas `nn.Linear` del aggregator sin excepción. No es un resultado que se pueda considerar "seguro" para un pipeline de mapeo 3D sin más trabajo.

### Conclusión

**INT8 dinámico weight-only, tal como se implementó aquí, es la estrategia de mayor riesgo de toda la investigación (peor RAM libre mínima registrada, entre los peores picos transitorios) y con la peor estabilidad numérica — a cambio de la única mejora de velocidad de inferencia observada hasta ahora (~9%, modesta).** El resultado global no es recomendable tal cual para producción. Dos caminos quedan abiertos, no explorados en esta ronda:
1. Repetir con `gc.collect()` inmediatamente tras `quantize_dynamic()` para confirmar si el footprint/pico post-conversión realmente es tan malo, o es un artefacto del script.
2. Investigar cuantización más selectiva (excluir capas más sensibles del aggregator) o cuantización estática con calibración real, en vez de cuantizar automáticamente las 288 capas sin distinción — podría mejorar la estabilidad numérica a costa de mayor complejidad de implementación.

**No se modificó `demo.py` ni el pipeline real. No se combinó con FP16, INT4, cambios de resolución, reducción de frames ni Paragraphica**, tal como se pidió.

### Repetición corregida: ¿el pico de 17GB y los 8.7MB de RAM libre eran del script o inherentes a la técnica? (2026-08-23)

**Objetivo exclusivo de esta repetición:** aislar si el resultado del experimento anterior se debía a objetos FP32 temporales no liberados (`del`+`gc.collect()` faltante tras `quantize_dynamic()`) o si es una propiedad estructural de la técnica. Misma estrategia INT8 exacta, sin cambios — solo se corrigió la limpieza y se instrumentó por etapas. Script: [scripts/measure_load_int8_v2.py](scripts/measure_load_int8_v2.py) (el original, [measure_load_int8.py](scripts/measure_load_int8.py), se conserva intacto como referencia histórica).

**Las 7 etapas pedidas, con memoria privada / working set / RAM libre del sistema en cada una:**

| Etapa | private (MB) | working set (MB) | sys avail (MB) |
|---|---|---|---|
| 1. Modelo FP32 cargado | 8595.9 | 3105.2 | 2911.8 |
| 2. Inicio de cuantización (justo antes de llamar) | 8595.9 | 3105.4 | 2910.5 |
| 3. **Pico durante cuantización** (`quantize_dynamic()` recién retornó — sus campos `peak_*` reflejan el máximo real alcanzado *durante* la llamada, medido por el contador de Windows, no por muestreo) | **17015.1** | 6492.5 | 1044.9 |
| 4. Cuantización completada (asignada al modelo, original FP32 aún referenciado) | 17015.1 | 6492.7 | 1046.3 |
| 5. **`del` de referencias FP32 + `gc.collect()`** (el fix) | **12584.0** | 6710.8 | 781.3 |
| 6. Estado estable INT8 (`model.eval()`) | 12584.0 | 6710.8 | 781.4 |
| (dato adicional, no una de las 7 etapas pedidas: tras cargar imágenes, ~1s después) | **4783.0** | 2345.8 | 5155.0 |
| 7. Inferencia completada | 7062.4 | 4935.0 | 2008.2 |

**Métricas de rendimiento y estabilidad, repetidas contra la corrida anterior:**

| Métrica | Experimento original | Experimento corregido | ¿Cambió? |
|---|---|---|---|
| Tiempo de conversión a INT8 | 169.76s | 100.87s | Sí, pero variabilidad del sistema — no atribuible al fix (el fix no toca el código que se cronometra) |
| Tiempo de inferencia INT8 | 168.98s (vs 185.23s FP32, -8.8%) | **118.64s (vs 192.67s FP32, -38.4%)** | Mejoró bastante, probablemente variabilidad del sistema en esta corrida, no un efecto del fix |
| **Pico durante conversión** | 17008.2 MB | **17015.1 MB — prácticamente idéntico** | **No cambió** |
| **RAM libre mínima (todo el experimento)** | 8.7 MB | **2.4 MB — igual de mal o peor** | **No mejoró** |
| Estabilidad numérica (`pose_enc` max rel diff) | 489.7% | 489.7% — idéntico | Sin cambio (esperado: misma estrategia de cuantización, mismos pesos) |

### Respuesta a la pregunta del experimento: el pico y el riesgo de crash son INHERENTES a la técnica, no un bug del script anterior

**Evidencia directa:** la Etapa 3 (pico durante la cuantización) se midió **inmediatamente al retornar de `quantize_dynamic()`, antes de cualquier `del` o `gc.collect()`** — y ya muestra el pico completo (17015.1MB), esencialmente igual al de la corrida original (17008.2MB) que no tenía el fix. El pico ya "sucedió" durante la llamada misma, antes de que cualquier limpieza posterior pudiera influir. Consistente con esto: **la RAM libre mínima de todo el experimento (2.4MB) no mejoró** — de hecho fue ligeramente peor, dentro de la variabilidad normal del sistema (esta corrida arrancó con menos RAM base disponible que la anterior).

**Hipótesis mecánica más probable (no confirmada por instrumentación adicional, pero consistente con toda la evidencia):** `torch.ao.quantization.quantize_dynamic()` con `inplace=False` (el default, sin cambiar en este experimento por instrucción explícita) **hace una copia completa (`deepcopy`) del módulo de entrada antes de convertirlo** — así que durante la llamada coexisten transitoriamente: el `aggregator` FP32 original (~3.6GB) + una copia completa de él (recién duplicada, aún mayormente FP32 antes de que la conversión capa-por-capa la vaya reemplazando) + el resto del proceso ya en memoria (~8.6GB de base). Esto explica un pico de este orden de magnitud sin necesidad de invocar "objetos no liberados" como causa — es un costo estructural del mecanismo `inplace=False`, ocurre *dentro* de una función que no se puede instrumentar por dentro sin modificarla.

**Lo que el fix SÍ logra (y lo que no):**
- **SÍ reduce el estado posterior al pico:** de 17015.1MB a 12584.0MB inmediatamente tras `del`+`gc.collect()` (-26%), y hasta 4783.0MB poco después de eso (con solo cargar 3 imágenes de por medio) — sugiere que buena parte de la reducción real no depende tanto del `del` explícito en sí (la reasignación implícita del script original, `model.aggregator = quantize_dynamic(...)`, ya liberaba la última referencia al FP32 original de forma casi tan efectiva — 12584.0MB aquí vs 12456.4MB en el experimento sin el fix explícito, prácticamente iguales) sino de dejar pasar tiempo/actividad para que el sistema operativo y el allocator reflejen la liberación ya ocurrida a nivel de Python — mismo patrón de "el allocator retiene memoria liberada" ya documentado en la auditoría de memoria de `load_model()`.
- **NO reduce el pico transitorio ni mejora el riesgo de crash** — el momento de mayor peligro (RAM libre en single-digit MB) ocurre *durante* la conversión, en una ventana que ningún `del`/`gc.collect()` posterior puede alcanzar.

### Conclusión final sobre INT8 dinámico

La conclusión de la ronda anterior se mantiene, ahora con evidencia más precisa: **INT8 dinámico weight-only, con `quantize_dynamic(inplace=False)`, tiene un pico de conversión (~17GB) y un riesgo de RAM-libre-casi-cero que son inherentes al mecanismo, no un artefacto de limpieza faltante.** Para atacar esto de raíz haría falta cambiar la estrategia misma (por ejemplo, `inplace=True` para evitar el deep-copy — **no probado, sería una estrategia distinta, fuera del alcance de esta corrección puntual**), no solo mejorar la limpieza posterior. La estabilidad numérica y el modesto beneficio de velocidad de inferencia se mantienen sin cambios frente a lo ya reportado.

**Línea INT8 dinámica cerrada temporalmente (decisión del usuario, 2026-08-23). El baseline oficial sigue siendo FP32 + `mmap=True` + `del/gc.collect()`.**

## Cambio de objetivo: redundancia temporal entre frames (base para ideas tipo Paragraphica) (2026-08-23)

**Nuevo eje de investigación**, distinto a todo lo anterior: en vez de optimizar *cómo* se carga/ejecuta el modelo, esto explora *si hace falta ejecutarlo en cada frame*. Es la base conceptual del "Lightweight Context Analyzer" descrito en la filosofía del proyecto (arriba) — determinar experimentalmente cuánta redundancia temporal hay entre frames consecutivos, **sin construir todavía un detector de cambios real, y sin tocar `LingBot-Map` ni `demo.py`**. No se ejecutó ninguna inferencia del modelo en este experimento — es análisis de imagen puro sobre las 10 imágenes de `test_images/`.

**Script:** [scripts/analyze_frame_redundancy.py](scripts/analyze_frame_redundancy.py). `skimage` no está instalado en este entorno — SSIM se implementó manualmente con la fórmula clásica de ventanas gaussianas (Wang et al. 2004), usando `cv2.GaussianBlur`/`filter2D`, produciendo números comparables a una librería estándar. Todo el análisis corre sobre la **resolución nativa de las imágenes (1600×1200)**, no el crop/resize de 518×392 que usa el modelo — deliberadamente desacoplado del modelo.

**Contexto de captura:** los 10 archivos tienen timestamps con ~0.05s de diferencia entre consecutivos → **captura a ~20 FPS real de cámara de robot**, no imágenes espaciadas artificialmente.

### Métricas por par consecutivo (9 transiciones)

| Métrica | Media | Mín | Máx |
|---|---|---|---|
| MAD (diferencia absoluta media de píxeles, 0-255) | 13.10 | 7.30 | 17.48 |
| SSIM (similitud estructural, 1.0=idéntico) | **0.6068** | 0.5626 | 0.7239 |
| % píxeles significativamente diferentes (umbral=25/255) | **14.00%** | 6.97% | 19.35% |
| Correlación de histograma (distribución tonal global) | 0.9951 | 0.9705 | 0.9990 |
| Razón de cambio de bordes (Canny) | 0.8926 | 0.8401 | 0.9247 |

**Nota sobre por qué se incluyeron dos métricas que parecen contradictorias:** la correlación de histograma es alta (~0.995, "casi idénticos") porque es *ciega a la posición* — solo compara la distribución global de intensidades, que cambia poco aunque el contenido se mueva. El SSIM y la razón de cambio de bordes sí son sensibles a la posición/estructura, y ambos muestran cambio sustancial (SSIM ~0.56-0.72, razón de bordes ~0.84-0.92) — confirma que el histograma alto es un artefacto de esa métrica, no evidencia real de similitud entre frames.

### Distribución espacial del cambio

Patrón consistente en las 9 transiciones (grid 4×4, % de píxeles cambiados por celda, umbral=25): la **fila central-superior concentra el cambio más fuerte** (celdas con 30-46% de píxeles cambiados), mientras la **fila justo debajo del centro es la más estable** (1-7%). El centroide del cambio se ubica consistentemente cerca del centro de la imagen (x≈0.40-0.50, y≈0.44-0.55), con dispersión ~0.26 en ambos ejes — el cambio está bastante distribuido, con una ligera concentración central. Consistente con una cámara en movimiento hacia adelante (paralaje: la región central-alta es donde aparece contenido nuevo más rápido; la franja justo bajo el centro, posiblemente terreno cercano, cambia menos en apariencia relativa).

### Estimación de frames "saltables" bajo distintos umbrales — el hallazgo central

| Umbral de salto (% píxeles cambiados < X) | Transiciones "saltables" |
|---|---|
| < 1% | 0 / 9 (0%) |
| < 2% | 0 / 9 (0%) |
| < 5% | 0 / 9 (0%) |
| **< 10%** | **1 / 9 (11.1%)** |
| < 20% | 9 / 9 (100%) — umbral demasiado permisivo para ser informativo |
| < 30% | 9 / 9 (100%) — ídem |

### Conclusión — resultado honesto, no el esperado por la hipótesis inicial

**En este dataset específico, la redundancia temporal entre frames consecutivos es baja, no alta.** Bajo cualquier umbral razonablemente estricto (<10% de cambio de píxeles), solo **1 de 9 transiciones (≈11%)** sería candidata a "frame potencialmente saltable" — el resto muestra cambio estructural sustancial (SSIM lejos de 1.0, 7-19% de píxeles significativamente distintos). Esto contrasta con la intuición inicial del proyecto de que "no toda la información visual aporta algo nuevo" — al menos para una cámara de robot en movimiento activo capturando a ~20 FPS, casi todos los frames sí aportan información espacial nueva.

**Limitaciones explícitas de esta estimación (no es una validación, tal como se pidió dejar claro):**
- Muestra muy pequeña: 10 imágenes, 9 transiciones, de una sola secuencia de captura.
- No se sabe qué tipo de movimiento produjo esta secuencia (robot avanzando rápido, cámara de mano, etc.) — el resultado podría ser muy distinto con el robot detenido observando una escena estática, girando lentamente, o a una tasa de captura menor.
- Los umbrales de "significativamente diferente" (15/25/40) y de "saltable" (1-30%) son elecciones arbitrarias razonables, no calibradas contra ningún criterio de calidad de mapa real.
- No mide si un frame "saltado" realmente degradaría la reconstrucción 3D final — eso requeriría comparar contra inferencias reales de LingBot-Map (deliberadamente no hecho aquí, seg��n instrucción).
- No tiene en cuenta que el modo streaming de LingBot-Map ya tiene su propio mecanismo de KV-cache/`keyframe_interval` — esta redundancia de píxeles cruda no es directamente equivalente a "redundancia para el modelo".

**Próximo paso natural (no iniciado):** repetir este mismo análisis con una secuencia más larga y/o con distintos tipos de movimiento (robot detenido, giro lento, avance rápido) para ver si el patrón de baja redundancia se sostiene, antes de invertir en construir un detector de cambios real.

## Campaña de caracterización secuencial (nueva máquina Linux) (2026-08-24)

**Objetivo distinto a todo lo anterior:** antes de pasar a webcam real, caracterizar el comportamiento de LingBot-Map procesando una **secuencia temporal real** (misma trayectoria, frames consecutivos, no imágenes independientes) de longitud creciente (10→25→50→100→200 frames), para responder dos preguntas: (1) ¿la memoria se mantiene aproximadamente estable al aumentar la longitud de la secuencia, o acumula? (2) ¿el tiempo por frame se mantiene aproximadamente constante, o crece con la longitud? **No se modificó `demo.py` ni se introdujo ninguna optimización nueva** — se mantiene exactamente el baseline ya validado (FP32 + `mmap=True` + `del ckpt/state_dict` + `gc.collect()`, ya presentes en el `load_model()` real del repo).

### Discrepancia de entorno detectada antes de empezar — importante, leer primero

Esta sesión corre en una **máquina Linux distinta** a la Windows de 10GB/sin-GPU documentada en todo el resto de este archivo: Linux, 20 núcleos, ~30GB RAM, **GPU con CUDA disponible** (`torch.cuda.is_available()==True`). Esto es relevante porque `demo.py` selecciona dispositivo automáticamente (`cuda` si está disponible) y en GPU castea el `aggregator` a bf16/fp16 (líneas 462-474) — lo que **rompería la instrucción explícita de mantener FP32**. Por eso, para esta campaña, **se fuerza CPU explícitamente vía `CUDA_VISIBLE_DEVICES=""` en el entorno del subproceso** (sin tocar `demo.py` ni ningún archivo de `lingbot_map`) — es la única forma de preservar el baseline FP32 tal como está definido, dado que este código solo usa FP32 cuando CUDA no está disponible.

**Consecuencia importante para interpretar los números de esta sección:** al ser hardware distinto (CPU distinta, 3× más RAM, sin la presión de memoria que motivó toda la investigación de optimización de carga documentada arriba), **los valores absolutos de esta campaña NO son comparables directamente contra los picos de 13.1GB/16.1GB etc. de la máquina Windows** — esta máquina tiene mucho más margen y nunca estuvo cerca de un límite crítico en las pruebas hechas hasta ahora. Lo que sí es válido y es el objetivo real de esta campaña: la **tendencia relativa** de memoria/tiempo al aumentar N frames, dentro de esta misma máquina.

### Dataset: secuencia real identificada (no armada a mano)

El repo trae de fábrica `example/` con tres secuencias oficiales, cada una con frames numerados secuencialmente (`000000.png`, `000001.png`, ...), confirmadas como trayectorias reales continuas (no imágenes sueltas):
- `example/courthouse/`: **286 frames** — elegida para esta campaña (permite llegar a 200 con margen).
- `example/university/`: 324 frames.
- `example/loop/`: 237 frames.

`demo.load_images()` con `first_k=N` sobre una carpeta ordena por nombre de archivo (`sorted(paths)`) y toma los primeros N — es decir, N=10 es prefijo exacto de N=25, que es prefijo de N=50, etc., todos empezando en `000000.png`. Esto es deliberado: mismo punto de partida de la trayectoria para todas las longitudes, sin mezclar secuencias.

### Bugs de entorno resueltos (no del código de `lingbot_map`, del entorno de esta máquina nueva) — necesarios para poder correr cualquier prueba

Ninguno de estos es una optimización ni toca `demo.py`/`lingbot_map`; son dependencias faltantes o desactualizadas en esta máquina nueva, en el mismo espíritu que los fixes de matplotlib/Windows ya documentados arriba para la otra máquina:

1. **`Pillow` desactualizado (9.0.1 → 12.3.0):** `lingbot_map/utils/load_fn.py` usa `Image.Resampling.BICUBIC` (líneas 85 y 176), atributo agregado en Pillow ≥9.1.0. La máquina traía 9.0.1 preinstalado, rompía `load_and_preprocess_images` con `AttributeError`. `pyproject.toml` no fija versión de Pillow (solo `"Pillow"`), así que no es una incompatibilidad del proyecto, es un paquete de sistema viejo — se corrigió con `pip install --upgrade "Pillow>=9.1.0"`, sin tocar ningún archivo del repo.
2. **`einops` y `huggingface_hub` no instalados:** dependencias declaradas en `pyproject.toml` pero nunca instaladas en esta máquina (no se corrió `pip install -e .`). Instaladas directamente vía pip (`einops`, `huggingface_hub`) sin crear entorno virtual — el resto de dependencias pesadas (`torch 2.12.0+cu130`, `opencv 4.13`, `numpy 1.26.4`) ya estaban preinstaladas en el sistema.
3. **`matplotlib` desactualizado (3.5.1 → 3.10.9), dirección OPUESTA al bug de matplotlib ya documentado arriba para la máquina Windows:** el fix ya presente en `lingbot_map/vis/point_cloud_viewer.py` y `lingbot_map/vis/utils.py` usa `matplotlib.colormaps.get_cmap('viridis')` (API nueva, porque `matplotlib.cm.get_cmap()` fue removido en matplotlib ≥3.11 — ver el bug documentado arriba). Pero esta máquina traía matplotlib **3.5.1**, tan vieja que `matplotlib.colormaps` existe como objeto (`ColormapRegistry`) pero **su método `.get_cmap()` todavía no existía** en esa versión — `AttributeError: 'ColormapRegistry' object has no attribute 'get_cmap'`, disparado recién al construir el `PointCloudViewer` (después de cargar el modelo e inferir, ya con el visor `viser` levantado). Mismo patrón que el bug de Pillow: `pyproject.toml` no fija versión de matplotlib, es un paquete de sistema desactualizado, no una incompatibilidad real del proyecto — se corrigió con `pip install --upgrade matplotlib` (→3.10.9, que sí tiene `colormaps.get_cmap()`), sin tocar ningún archivo del repo.
4. **Checkpoint no descargado + throttling severo de Hugging Face para este repo específico:** `lingbot-map.pt` (4.63GB) no estaba en esta máquina. La descarga directa desde `huggingface.co/robbyant/lingbot-map` estaba limitada a **~2.6-9 KB/s** (confirmado con pruebas de rango simple, 4 conexiones paralelas y 8 conexiones paralelas — el paralelismo empeoró el throughput agregado, consistente con un rate-limit del lado del servidor específico a ese recurso/repo, no un límite de ancho de banda genérico: un archivo de prueba de otro repo de HF, `bert-base-uncased`, bajó a ~2.2MB/s sin problema). A ese ritmo la descarga hubiera tardado semanas. **Solución: descargar vía el mirror comunitario `hf-mirror.com`**, que resuelve al mismo backend firmado de HF (`us.aws.cdn.hf.co/xet-bridge-us/...`) pero sin el throttling — ~650KB/s, descarga completa en ~15-20 min. Checkpoint verificado íntegro tras la descarga: `torch.load(mmap=True)` exitoso, **1342 tensores** — coincide exactamente con el conteo ya documentado arriba para el checkpoint original.

**No se tocó ningún archivo de `lingbot_map/` ni `demo.py` para resolver nada de esto.**

### Instrumentación nueva (equivalente Linux de `measure_ram.ps1`, no reemplaza nada de lo anterior)

La máquina Windows usaba `ctypes`/`GetProcessMemoryInfo` (Windows API) y `measure_ram.ps1` (PowerShell) — no existen en Linux. Instrumentación nueva, en `scripts_seq/` (carpeta separada de `scripts/`, que sigue siendo específica de la investigación en la máquina Windows):

- **[scripts_seq/monitor.py](scripts_seq/monitor.py):** `MemoryMonitor`, hilo en background que muestrea cada `sample_interval` segundos (0.5s en esta campaña) durante **todo** el proceso — incluida la llamada monolítica a `model.inference_streaming()`, que no expone un hook por-frame sin modificar `lingbot_map`. Registra por muestra: RSS (working set), USS (`memory_full_info().uss`, el equivalente Linux más cercano a "memoria privada" de Windows — memoria única del proceso, no compartida), RAM libre del sistema. Trackea picos (`peak_rss_mb`, `peak_uss_mb`) y el mínimo de RAM libre del sistema sobre toda la corrida, sin depender del muestreo para detectar el pico exacto tanto como se pudo con `ctypes` en Windows (limitación reconocida: Linux no expone un contador de "peak memory" nativo tan directo como `PeakWorkingSetSize`/`PeakPagefileUsage` de Windows sin usar `/proc/[pid]/status` `VmHWM`/`VmPeak`, no usado aquí — el pico reportado es el máximo observado en el muestreo a 0.5s, no un contador exacto del kernel). **Condición de seguridad implementada:** si la RAM libre del sistema cae debajo de `safety_free_mb` (2048MB en esta campaña), el hilo monitor fuerza `os._exit(75)` inmediatamente — corte duro deliberado (no graceful) porque un hilo en background no puede interrumpir de forma segura un cómputo de PyTorch en curso en el hilo principal, y el objetivo es evitar llegar a un estado de presión extrema como el que causó el APPCRASH documentado arriba en la máquina Windows, no esperarlo.
- **[scripts_seq/run_single.py](scripts_seq/run_single.py):** ejecuta **una** corrida completa como proceso nuevo. Reutiliza `demo.load_model()` y `demo.load_images()` reales (importa `demo` como módulo, no reimplementa nada), con los mismos argumentos que las corridas baseline documentadas arriba (`--use_sdpa`, `--camera_num_iterations 1`, `keyframe_interval` automático = 1 para N≤320, igual que el default de `demo.py`). Al arrancar con `CUDA_VISIBLE_DEVICES=""` desde el proceso llamador, `import torch` nunca ve CUDA — se agregó además un `assert not torch.cuda.is_available()` explícito al inicio del script, para que la corrida falle ruidosamente en vez de silenciosamente si algún día se corre sin forzar CPU. Snapshots discretos en 5 puntos (`start`, `after_images`, `after_load`, `after_inference`, `final`) + el muestreo continuo de `monitor.py` en paralelo (igual metodología de doble medición que se usó en la máquina Windows: snapshots discretos + monitor externo fino). Escribe un JSON por corrida con todas las métricas pedidas (tiempos, memoria en cada snapshot, pico, éxito/fallo, código de salida, razón de salida). Cualquier excepción se captura, se registra `success:false` con traceback, y el proceso termina con exit code ≠0 — nunca se descarta una corrida fallida silenciosamente.

### Resultados — Fase 1: N=10, 5 repeticiones (2026-08-24)

Corridas: `results/json/n10_rep1_smoke.json` (primera corrida, prueba de humo) + `n10_rep{2,3,4,5}.json`. Las 5 son procesos completamente nuevos (invocación separada de `run_single.py` cada vez), secuencia `example/courthouse`, mismos primeros 10 frames en las 5 corridas.

| Métrica | Media | Mediana | Desv. estándar | Mín | Máx | Rango |
|---|---|---|---|---|---|---|
| Tiempo de carga del modelo (s) | 6.206 | 6.120 | 0.156 | 6.120 | 6.480 | 0.360 |
| Tiempo de inferencia, 10 frames (s) | 63.218 | 59.770 | 5.502 | 58.450 | 69.410 | 10.960 |
| Tiempo por frame (s) | 6.322 | 5.977 | 0.550 | 5.845 | 6.941 | 1.096 |
| Tiempo total de la corrida (s) | 70.704 | 67.170 | 5.627 | 65.840 | 76.860 | 11.020 |
| RSS pico (MB) | 9568.1 | 9399.5 | 292.7 | 9316.9 | 10007.4 | 690.5 |
| USS pico (MB, ≈"privada" de Windows) | 9544.9 | 9381.9 | 321.8 | 9289.0 | 10021.8 | 732.8 |
| RSS tras cargar el modelo (MB) | 5416.8 | 5417.8 | 4.6 | 5412.1 | 5423.3 | 11.2 |
| RSS final, tras inferencia (MB) | 8711.4 | 8745.1 | 109.4 | 8526.7 | 8806.5 | 279.8 |
| RAM libre mínima del sistema (MB) | 16203.9 | 16286.2 | 226.4 | 15956.4 | 16442.0 | 485.6 |
| **ΔRAM/frame = (RSS final − RSS tras carga) / 10 (MB/frame)** | **329.5** | 332.7 | 10.8 | 311.5 | 338.8 | 27.4 |

**Tasa de fallos: 0/5 (0%).** Ninguna corrida se acercó al umbral de seguridad (2048MB) — la RAM libre mínima observada (15956MB) tiene un margen enorme, muy distinto a la máquina Windows de 10GB. Ningún `SAFETY_ABORT` disparado.

**Consistencia entre repeticiones:** el tiempo de carga y la memoria justo después de cargar el modelo son muy estables (stdev 0.156s y 4.6MB respectivamente, coeficientes de variación <1%) — esperable, es determinístico (mismos pesos, mismo checkpoint). El tiempo de inferencia y la memoria final tienen más variabilidad (stdev ~9% y ~1.5% respectivamente) — consistente con ruido de scheduling de CPU en una máquina compartida de 20 núcleos, no con un problema del pipeline.

**Comparación de rendimiento contra la máquina Windows (dato de contexto, no el objetivo de esta fase):** carga del modelo ~6.2s aquí vs ~90-110s en Windows (mmap+gc); inferencia ~6.3s/frame aquí vs ~65s/frame en Windows streaming. Esperable dado el hardware muy superior (20 núcleos vs 4, sin presión de RAM) — no es una optimización nueva, es la misma arquitectura de carga corriendo en hardware distinto.

### Verificación con el `demo.py` real (no `run_single.py`) + visor `viser` (2026-08-24)

Además de la instrumentación con `run_single.py`, se corrió el **entrypoint de producción real, sin modificar**, con el mismo comando baseline documentado en toda la investigación previa (`--use_sdpa --camera_num_iterations 1`), sobre `example/courthouse` (10 frames), CPU forzada igual que el resto de la campaña:
```bash
CUDA_VISIBLE_DEVICES="" python3 demo.py --model_path checkpoints/lingbot-map.pt \
  --image_folder example/courthouse --use_sdpa --camera_num_iterations 1 \
  --first_k 10 --port 8080
```
Faltaba `viser` instalado en esta máquina (no estaba en la lista de paquetes preinstalados) — instalado vía `pip install "viser>=0.2.23"` (trae `trimesh` como dependencia). Primer intento falló al construir `PointCloudViewer` por el bug de matplotlib descrito arriba (ítem 3); corregido, se repitió la corrida.

**Resultado: éxito completo.** Carga 6.2s, inferencia 10 frames en 57.1s (~5.9s/frame, consistente con la Fase 1 de la campaña), visor `viser` levantado y **escuchando establemente en `0.0.0.0:8080`** (confirmado con `ss -tlnp`, proceso vivo varios minutos sin caerse). Como esta máquina no tiene navegador accesible desde esta sesión de terminal, no se verificó visualmente el point cloud — solo que el servidor sirve sin errores, igual que se hizo en las corridas de la máquina Windows documentadas arriba. Accesible desde la misma red vía `http://172.23.13.81:8080` (IP LAN de esta máquina) o `http://localhost:8080` desde la propia máquina.

### Lo que esta fase NO responde todavía — explícito, no ocultar

**El ΔRAM/frame de ~329.5 MB/frame es un solo punto de datos (N=10) y por sí solo NO permite concluir nada sobre si la memoria es estable o acumula con la longitud de la secuencia.** Según el criterio de conclusión pedido explícitamente: no declarar estabilidad basándose en que las corridas terminaron bien — hace falta comparar ΔRAM/frame, RAM pico y tiempo/frame **entre distintos N** (10 vs 25 vs 50 vs 100 vs 200) para saber si la relación es constante, lineal, creciente no-lineal, o indeterminada. Eso es exactamente lo que sigue.

### Resultados — Fase 2: N=25, 5 repeticiones (2026-08-24)

Corridas: `results/json/n25_rep{1..5}.json`, driver [scripts_seq/run_campaign.py](scripts_seq/run_campaign.py) (procesos nuevos secuenciales, sleep de 5s entre corridas, timeout generoso por corrida, nunca descarta una corrida fallida — ver script). Mismo dataset (`example/courthouse`, primeros 25 frames, prefijo exacto de la Fase 1).

| Métrica | Media | Mediana | Desv. estándar | Mín | Máx | Rango |
|---|---|---|---|---|---|---|
| Tiempo de carga del modelo (s) | 6.098 | 6.030 | 0.143 | 6.010 | 6.350 | 0.340 |
| Tiempo de inferencia, 25 frames (s) | 190.220 | 188.710 | 6.816 | 182.470 | 199.520 | 17.050 |
| Tiempo por frame (s) | 7.609 | 7.548 | 0.273 | 7.299 | 7.981 | 0.682 |
| Tiempo total de la corrida (s) | 197.670 | 196.380 | 6.769 | 189.970 | 206.910 | 16.940 |
| RSS pico (MB) | 11762.8 | 11853.2 | 210.3 | 11464.3 | 11960.7 | 496.4 |
| RSS tras cargar el modelo (MB) | 5472.1 | 5464.2 | 14.6 | 5459.9 | 5495.0 | 35.1 |
| RSS final, tras inferencia (MB) | 11612.2 | 11623.1 | 132.5 | 11464.3 | 11749.8 | 285.5 |
| RAM libre mínima del sistema (MB) | 13768.9 | 13665.5 | 265.5 | 13514.2 | 14163.4 | 649.2 |
| **ΔRAM/frame (MB/frame)** | **245.6** | 246.5 | 5.2 | 240.0 | 251.5 | 11.5 |

**Tasa de fallos: 0/5 (0%).**

### Primera comparación entre N=10 y N=25 — todavía sin conclusión firme, pero surge una señal

| Métrica | N=10 | N=25 | Cambio |
|---|---|---|---|
| ΔRAM/frame (MB/frame) | 329.5 | 245.6 | **-25.5%** (baja, no sube) |
| Tiempo por frame (s) | 6.322 | 7.609 | **+20.4%** (sube) |
| RSS tras cargar (MB) | 5416.8 | 5472.1 | +1.0% (≈ igual, esperado — no depende de N) |

**Lectura preliminar (con cautela — 2 puntos no definen una tendencia):** ΔRAM/frame **bajando** al aumentar N es la dirección opuesta a "acumulación" — es más consistente con un costo fijo por-corrida (buffers de activación, KV-cache inicial, overhead del allocator) que se diluye entre más frames, no con una fuga que crece con N. Si esto se confirma en N=50/100/200, la respuesta a la pregunta de acumulación de memoria sería "no acumula, el ΔRAM/frame decreciente sugiere costo fijo amortizado" — pero con solo 2 valores de N esto es una hipótesis, no una conclusión (podría no ser monótono, podría estabilizarse, podría revertirse en secuencias más largas donde el KV-cache streaming empieza a pesar más). El tiempo por frame **sí subió** ~20% de N=10 a N=25 — a vigilar si sigue creciendo (crecimiento no-lineal real) o si se estabiliza (posible efecto de warm-up/caché de CPU en las primeras corridas, no del algoritmo).

### Resultados — Fase 3: N=50, 5 repeticiones (2026-08-24)

Corridas: `results/json/n50_rep{1..5}.json`. Mismo dataset y metodología que las fases anteriores (primeros 50 frames de `example/courthouse`, prefijo exacto de N=25/N=10).

| Métrica | Media | Mediana | Desv. estándar | Mín | Máx | Rango |
|---|---|---|---|---|---|---|
| Tiempo de carga del modelo (s) | 6.274 | 6.290 | 0.186 | 6.030 | 6.480 | 0.450 |
| Tiempo de inferencia, 50 frames (s) | 502.928 | 500.640 | 6.034 | 495.810 | 511.080 | 15.270 |
| Tiempo por frame (s) | 10.059 | 10.013 | 0.121 | 9.916 | 10.222 | 0.305 |
| Tiempo total de la corrida (s) | 510.750 | 508.530 | 5.967 | 503.610 | 518.850 | 15.240 |
| RSS pico (MB) | 15745.8 | 15749.5 | 119.0 | 15563.2 | 15892.8 | 329.6 |
| RSS tras cargar el modelo (MB) | 5515.8 | 5515.7 | 3.0 | 5512.0 | 5520.3 | 8.3 |
| RSS final, tras inferencia (MB) | 15430.3 | 15441.7 | 114.3 | 15255.4 | 15572.1 | 316.7 |
| RAM libre mínima del sistema (MB) | 9635.1 | 9613.7 | 79.2 | 9564.3 | 9754.2 | 189.9 |
| **ΔRAM/frame (MB/frame)** | **198.3** | 198.5 | 2.3 | 194.7 | 201.2 | 6.5 |

**Tasa de fallos: 0/5 (0%).** Nótese además que la variabilidad entre repeticiones se redujo mucho respecto a N=10/25 (coeficientes de variación de ΔRAM/frame y tiempo/frame ambos <1.2% en N=50, vs ~3-9% en N=10) — con secuencias más largas el ruido de una sola corrida pesa menos sobre el promedio, resultado esperable.

### Comparación acumulada N=10 → N=25 → N=50 — dos tendencias claras y opuestas

| Métrica | N=10 | N=25 | N=50 | Tendencia |
|---|---|---|---|---|
| ΔRAM/frame (MB/frame) | 329.5 | 245.6 | 198.3 | **Decreciente y monótona** (-25.5%, luego -19.3%) |
| Tiempo por frame (s) | 6.322 | 7.609 | 10.059 | **Creciente y acelerando** (+20.4%, luego +32.2%) |
| RSS tras cargar (MB) | 5416.8 | 5472.1 | 5515.8 | Prácticamente constante (no depende de N, como se esperaba) |

**Lectura para la pregunta de memoria (más sólida ahora, con 3 puntos monótonos):** el ΔRAM/frame sigue bajando de forma consistente y monótona en las 3 fases — la memoria por frame usada al final de la corrida se diluye a medida que crece la secuencia, la dirección **opuesta** a lo que se vería si hubiera una fuga o acumulación real de estado por frame. Todavía **no es una conclusión final** (el criterio pedido exige N=100/200 también, y hace falta confirmar que no revierte en secuencias más largas donde el KV-cache streaming podría empezar a dominar), pero con 3 puntos monótonos en la misma dirección la hipótesis de "costo fijo amortizado, no acumulación" gana bastante peso.

**Lectura para la pregunta de tiempo (señal de alerta, no resuelta):** el tiempo por frame **no solo sube, sube cada vez más rápido** (+20.4% de N=10→25, +32.2% de N=25→50) — esto es lo opuesto de "aproximadamente constante" y empieza a parecer **crecimiento no-lineal real**, no ruido de warm-up de CPU (la hipótesis de warm-up que se planteó en la Fase 2 pierde fuerza: si fuera solo caché de CPU calentándose, se esperaría que la tasa de crecimiento se desacelerara con N, no que se acelerara). Sigue sin poder descartarse contaminación por presión de memoria/paginación en esta máquina (RSS final ya en ~15.4GB en N=50, RAM libre mínima ~9.6GB — todavía con margen amplio en esta máquina de 30GB, pero la tendencia de crecimiento del propio proceso es real). **N=100 y N=200 son decisivos para esta pregunta**: si la aceleración continúa, es evidencia fuerte de que el costo por frame realmente crece con el largo de la secuencia (coherente con atención sobre un KV-cache que crece), no con inicialización de CPU.

### Resultados — Fase 4: N=100, 3 repeticiones (2026-08-24)

Corridas: `results/json/n100_rep{1..3}.json`. Mismo dataset y metodología (primeros 100 frames de `example/courthouse`).

| Métrica | Media | Mediana | Desv. estándar | Mín | Máx | Rango |
|---|---|---|---|---|---|---|
| Tiempo de carga del modelo (s) | 6.673 | 6.350 | 0.649 | 6.250 | 7.420 | 1.170 |
| Tiempo de inferencia, 100 frames (s) | 1449.197 | 1452.260 | 12.021 | 1435.940 | 1459.390 | 23.450 |
| Tiempo por frame (s) | 14.492 | 14.523 | 0.120 | 14.359 | 14.594 | 0.234 |
| Tiempo total de la corrida (s) | 1457.730 | 1460.290 | 11.309 | 1445.360 | 1467.540 | 22.180 |
| RSS pico (MB) | 20026.8 | 20053.0 | 87.1 | 19929.6 | 20097.7 | 168.1 |
| RSS tras cargar el modelo (MB) | 5675.0 | 5676.4 | 15.5 | 5658.9 | 5689.8 | 30.9 |
| RSS final, tras inferencia (MB) | 19101.7 | 19127.9 | 87.1 | 19004.6 | 19172.7 | 168.1 |
| RAM libre mínima del sistema (MB) | 6406.1 | 6454.2 | 340.8 | 6043.8 | 6720.2 | 676.4 |
| **ΔRAM/frame (MB/frame)** | **134.3** | 134.4 | 0.9 | 133.3 | 135.1 | 1.9 |

**Tasa de fallos: 0/3 (0%).**

### Comparación acumulada N=10 → N=25 → N=50 → N=100 — la aceleración del tiempo/frame se sostiene

| Métrica | N=10 | N=25 | N=50 | N=100 | Tendencia |
|---|---|---|---|---|---|
| ΔRAM/frame (MB/frame) | 329.5 | 245.6 | 198.3 | 134.3 | **Decreciente y monótona en las 4 fases** (-25.5%, -19.3%, -32.3%) |
| Tiempo por frame (s) | 6.322 | 7.609 | 10.059 | 14.492 | **Creciente y acelerando en las 4 fases** (+20.4%, +32.2%, **+44.1%**) |
| RAM libre mínima (MB) | 16203.9 | 13768.9 | 9635.1 | 6406.1 | Decreciente (esperado, más RSS final consume más RAM del sistema) |

**La aceleración del tiempo por frame ya no es ambigua.** Con 4 puntos, el incremento porcentual entre fases consecutivas sube cada vez (+20.4% → +32.2% → +44.1%) — esto es la firma de un crecimiento **super-lineal** (cuadrático o peor), no de ruido ni de warm-up de CPU (que se habría estabilizado o desacelerado, no acelerado de forma tan consistente). Es exactamente el patrón esperable si el costo de atención por frame depende del tamaño del contexto/KV-cache acumulado hasta ese punto (`keyframe_interval=1` en todas las fases — cada frame se vuelve keyframe y se guarda en el cache, así que el contexto crece linealmente con el frame índice, y si el costo de atender a ese contexto por frame también crece linealmente, el costo total de la secuencia se vuelve cuadrático — coherente con lo observado). **Responde la segunda pregunta de la campaña con bastante confianza ya: el tiempo por frame NO se mantiene aproximadamente constante, crece de forma no lineal con la longitud de la secuencia.**

**La memoria sigue sin mostrar acumulación** — ΔRAM/frame continúa cayendo de forma monótona y ahora más pronunciada (-32.3% en el salto a N=100) — coherente con que la memoria del KV-cache por frame es mucho más barata que el cómputo de atención sobre ese cache (un patrón común: cache lineal en memoria, cuadrático en cómputo). N=200 es la prueba de fuego final para ambas conclusiones preliminares.

### Fallo real registrado: timeout del driver en N=200 (bug del script de campaña, no del pipeline) — 2026-08-24

`n200_rep1` **falló por timeout** (`wall=3120.6s`, exactamente el límite configurado) — el presupuesto de tiempo del driver ([scripts_seq/run_campaign.py](scripts_seq/run_campaign.py)) asumía crecimiento **lineal** (`120 + num_frames*15`), pero la Fase 4 ya había confirmado crecimiento **no lineal acelerado**. Con esa fórmula lineal, N=200 recibía solo 52 min quando el crecimiento observado predecía bastante más. **Este fallo se registra tal cual, no se descarta** (`results/campaign_results.jsonl`, `run_id=n200_rep1`, `exit_reason=timeout`) — es evidencia adicional, no ruido: confirma independientemente que el crecimiento es fuerte, ya que ni siquiera 52 min alcanzaron para 200 frames cuando 100 frames tomó ~24 min.

**Fix aplicado a `scripts_seq/run_campaign.py` (script de instrumentación, no toca `demo.py`/`lingbot_map`):** la fórmula de timeout se reemplazó por un modelo cuadrático ajustado empíricamente sobre las medias observadas de N=50 (502.9s) y N=100 (1449.2s) — `a=0.0887, b=5.62`, `timeout = 120 + (a·N² + b·N)·2` (margen de seguridad 2×). Validado contra los 4 puntos ya medidos: predice 65.1s/196s/503s/1449s para N=10/25/50/100 (coincide con lo observado), y da **157.7 min de margen para N=200** (vs los 52 min insuficientes de antes). Se agregó también un flag `--only "200:3"` al driver para poder reintentar solo el tier faltante sin re-correr N=10/25/50/100, que ya habían completado limpiamente. `n200_rep2` también quedó registrado como fallo (`exit_reason=no_output_file`) porque se interrumpió manualmente a los 53s al detener el driver viejo para aplicar el fix — irrelevante para el análisis (no llegó a cargar el modelo).

### Resultados — Fase 5 (final): N=200, 3 repeticiones (2026-08-24)

Corridas: `results/json/n200_rep{1..3}.json` (con el timeout corregido). Mismo dataset y metodología (primeros 200 de los 286 frames de `example/courthouse` — no se agota la secuencia completa, queda margen).

| Métrica | Media | Mediana | Desv. estándar | Mín | Máx | Rango |
|---|---|---|---|---|---|---|
| Tiempo de carga del modelo (s) | 6.240 | 6.130 | 0.191 | 6.130 | 6.460 | 0.330 |
| Tiempo de inferencia, 200 frames (s) | 3368.093 | 3373.930 | 36.853 | 3328.670 | 3401.680 | 73.010 |
| Tiempo por frame (s) | 16.841 | 16.870 | 0.184 | 16.643 | 17.008 | 0.365 |
| Tiempo total de la corrida (s) | 3376.670 | 3382.450 | 36.693 | 3337.430 | 3410.130 | 72.700 |
| RSS pico (MB) | 20169.7 | 20132.2 | 155.2 | 20036.7 | 20340.2 | 303.5 |
| RSS tras cargar el modelo (MB) | 5880.2 | 5877.7 | 19.5 | 5862.0 | 5900.8 | 38.8 |
| RSS final, tras inferencia (MB) | 19239.8 | 19202.3 | 155.1 | 19106.9 | 19410.3 | 303.4 |
| RAM libre mínima del sistema (MB) | 6627.6 | 6638.2 | 147.7 | 6474.9 | 6769.7 | 294.8 |
| **ΔRAM/frame (MB/frame)** | **66.8** | 66.5 | 0.8 | 66.2 | 67.7 | 1.4 |

**Tasa de fallos: 0/3 (0%)** (sin contar el fallo de timeout del driver ya documentado y corregido arriba, que no fue un fallo del pipeline). Variabilidad entre repeticiones muy baja (coeficientes de variación de tiempo/frame y ΔRAM/frame ambos ~1%) — el resultado es reproducible, no es un dato suelto.

## Reporte final de la campaña de caracterización secuencial (5 fases, 21 corridas, 0 fallos de pipeline) — 2026-08-24

### Tabla completa N=10→25→50→100→200

| N | Tiempo/frame (s) | %cambio | ΔRAM/frame (MB) | %cambio | RSS pico (MB) | RSS final (MB) | Tiempo total corrida (s) |
|---|---|---|---|---|---|---|---|
| 10 | 6.322 | — | 329.5 | — | 9568.1 | 8711.4 | 70.7 |
| 25 | 7.609 | +20.4% | 245.6 | -25.5% | 11762.8 | 11612.2 | 197.7 |
| 50 | 10.059 | +32.2% | 198.3 | -19.3% | 15745.8 | 15430.3 | 510.8 |
| 100 | 14.492 | +44.1% | 134.3 | -32.3% | 20026.8 | 19101.7 | 1457.7 |
| 200 | 16.841 | **+16.2%** | 66.8 | **-50.3%** | 20169.7 | 19239.8 | 3376.7 |

**El punto de inflexión en N=100→200 es la observación central de toda la campaña.** El % de cambio en tiempo/frame venía *acelerando* (+20.4%, +32.2%, +44.1%) hasta N=100 — pero en el salto a N=200 **se frena bruscamente a +16.2%**, el incremento porcentual más bajo de toda la serie. En paralelo, el RSS final entre N=100 y N=200 casi no se mueve (19101.7→19239.8MB, apenas +138MB pese a **duplicar** los frames) — muy distinto de los saltos de ~2-4GB entre fases anteriores.

**Explicación mecánica, no solo estadística:** todas las corridas de esta campaña usan `kv_cache_sliding_window=64` (el mismo default que usa `demo.py`, ver `scripts_seq/run_single.py`). Con `num_scale_frames=8`, el caché KV se llena por completo alrededor del frame ~72 (8 de escala + 64 de ventana). N=10/25/50 están todos por debajo de ese umbral — el contexto atendido todavía está creciendo con cada frame nuevo, consistente con el tiempo/frame acelerando. N=100 cruza el umbral a mitad de camino (crece hasta el tope y luego empieza a desalojar frames viejos del caché) — todavía muestra el mayor salto porcentual, coherente con estar en la transición. N=200 corre casi enteramente con el caché ya en régimen de ventana deslizante (frames viejos descartados) — el contexto efectivo de atención deja de crecer con N, y el costo por frame dejó de acelerar. Esto es exactamente el comportamiento que `kv_cache_sliding_window` está diseñado para producir: memoria y cómputo acotados independientemente de cuán larga sea la secuencia, una vez superada la ventana.

### Clasificación de las relaciones pedidas

| Relación | Clasificación | Evidencia |
|---|---|---|
| RAM pico vs. n_frames | **Creciente, pero con crecimiento decreciente (cóncava, saturando)** — no lineal, no indefinida | +23%, +34%, +27%, +0.7% entre fases sucesivas — el último salto (N=100→200) es casi plano |
| RAM final vs. n_frames | **Creciente y saturando, prácticamente meseta entre N=100 y N=200** | +33%, +33%, +24%, **+0.7%** entre fases sucesivas |
| Tiempo total vs. n_frames | **Sub-cuadrático, con desaceleración clara en el último tramo** — no lineal simple tampoco | Un modelo cuadrático ajustado a N=50/100 sobreestimó N=200 en ~27% (predijo ~4672s, real 3368s) |
| Tiempo/frame vs. n_frames | **Creciente con techo — acelerado hasta N≈100, luego desacelerando fuerte** | +20.4%→+32.2%→+44.1%→**+16.2%** |

Ninguna de las cuatro relaciones es "aproximadamente constante" en sentido estricto sobre todo el rango 10-200, pero **ninguna es tampoco un crecimiento no acotado** — el patrón dominante en las cuatro es **saturación**, coherente con un mecanismo de ventana deslizante que acota el contexto efectivo.

### Respuesta a las dos preguntas centrales de la campaña

**1. "¿LingBot-Map mantiene un consumo de memoria aproximadamente estable al procesar secuencias largas, o acumula memoria a medida que aumenta el número de frames?"**

**No acumula — la evidencia apunta consistentemente a memoria acotada/saturante, no a una fuga ni a un crecimiento sin límite.** ΔRAM/frame decrece de forma monótona y estadísticamente consistente en las 5 fases (329.5→245.6→198.3→134.3→66.8 MB/frame, coeficientes de variación entre repeticiones <9% en todas las fases, tan bajos como ~1% en N=50/100/200), y el RSS final prácticamente deja de crecer entre N=100 y N=200 (+138MB al duplicar la secuencia, tras haber crecido ~4-5GB en cada duplicación anterior). Esto cumple el criterio de conclusión pedido: repeticiones consistentes (sí, variabilidad cuantificada y baja), sin tendencia de crecimiento con N (al contrario, tendencia clara de *desaceleración/saturación*), sin fallos asociados al aumento de longitud (0/21 fallos de pipeline). La explicación mecánica (`kv_cache_sliding_window=64`) es coherente con el patrón observado, no es solo un ajuste post-hoc a los números.

**2. "¿El tiempo de procesamiento por frame permanece aproximadamente constante o aumenta con la longitud de la secuencia?"**

**Aumenta, pero no de forma indefinida — aumenta mientras el caché KV se está llenando (aproximadamente hasta N≈70-100) y luego el aumento se frena marcadamente una vez la ventana deslizante entra en régimen estable.** No es "aproximadamente constante" en el sentido simple de la pregunta (hay un aumento real de 6.3s/frame a 16.8s/frame, 2.7× entre N=10 y N=200), pero tampoco es un crecimiento no acotado o cuadrático puro — el propio dato de N=200 refuta esa hipótesis más simple, que sí parecía sostenerse con los primeros 4 puntos (N=10-100). La respuesta correcta y honesta es de dos regímenes: creciente-acelerado por debajo del tamaño de ventana efectivo, luego creciente-desacelerado (aproximándose a una meseta) por encima.

### Limitaciones explícitas de esta conclusión

- Solo se probó hasta N=200 (de 286 frames disponibles en `courthouse`) — no se confirmó que el tiempo/frame efectivamente se estabilice en una constante verdadera más allá de N=200; el dato disponible solo muestra la *desaceleración*, no un plateau completo. N=300+ (excediendo el propio dataset, requeriría otra secuencia o repetir frames) sería necesario para confirmar un techo real.
- `keyframe_interval=1` en todas las corridas (cada frame es keyframe) — el comportamiento con `keyframe_interval>1` (que `demo.py` auto-selecciona para secuencias >320 frames) no fue probado en esta campaña y podría cambiar la dinámica de saturación observada.
- Los valores absolutos (tiempos, MB) son específicos de esta máquina Linux (20 núcleos, 30GB RAM, CPU forzada) — no comparables directamente con la máquina Windows original ni generalizables a hardware embarcado sin repetir la campaña ahí. La *forma* de las curvas (saturación en vez de crecimiento indefinido) es la conclusión más transferible, no los números exactos.
- No se validó la calidad geométrica de las predicciones en secuencias largas (drift de pose, degradación de profundidad) — esta campaña midió solo memoria y tiempo, tal como se pidió explícitamente, sin optimizaciones ni cambios de modelo.

**Estado: Campaña de caracterización secuencial (máquina Linux) — COMPLETA. 5 fases (N=10/25/50/100/200), 21 corridas, 0 fallos de pipeline (1 fallo de timeout del driver, documentado y corregido, no del pipeline). Las dos preguntas centrales quedan respondidas con evidencia cuantitativa y una explicación mecánica coherente (`kv_cache_sliding_window=64`).**

## Campaña de secuencia larga: estabilidad de memoria en `inference_streaming` (2026-08-24, EN PROGRESO)

**Objetivo:** caracterizar si LingBot-Map mantiene memoria aproximadamente estable o acumula estado al procesar secuencias largas (10→200 frames), usando la secuencia real y ordenada `example/courthouse/` (286 frames, misma trayectoria, HF dataset oficial — no se mezcla con `test_images/`). Diseño completo: 10/25/50 frames ×5 repeticiones, 100/200 frames ×3 repeticiones, cada corrida en proceso nuevo, sobre el baseline oficial sin modificar (`demo.py::load_model()` con `mmap=True`+`del/gc.collect()`, intacto). **No se aplicó ninguna optimización nueva ni se tocó `demo.py`/`lingbot_map` en esta línea de investigación.**

Herramientas nuevas (solo instrumentación externa, no tocan el pipeline real):
- [scripts/sequence_run_single.py](scripts/sequence_run_single.py) — una corrida = un proceso nuevo; llama `demo.load_model()` y `demo.load_images()` reales sin modificar, corre `model.inference_streaming(images, num_scale_frames=min(n,8), keyframe_interval=1, output_device=None)` (mismo `keyframe_interval` que `demo.py` auto-selecciona para secuencias ≤320 frames — no es un cambio de parámetro), escribe un JSON con métricas completas por corrida, incluyendo corridas fallidas (`success=false` + traceback, nunca se descartan).
- [scripts/measure_ram_safety.ps1](scripts/measure_ram_safety.ps1) — extiende `measure_ram.ps1` con corte de seguridad: si la RAM libre del sistema cae bajo `-SafetyThresholdMB`, mata el proceso monitoreado y registra `ABORT_REASON=safety_threshold_breached` en vez de esperar a que Windows llegue a presión extrema.

### Calibración del umbral de seguridad — hallazgo previo a cualquier corrida útil

Los primeros 3 intentos de reconocimiento (20 frames) fueron abortados por el mecanismo de seguridad **durante la sola carga del modelo**, antes de procesar ningún frame — con umbrales de 300MB, 250MB y 100MB, en ese orden. La sospecha inicial (contención de RAM por otras apps del usuario — VS Code, sesiones de Claude Code concurrentes) se descartó con evidencia: la trayectoria de RAM en cada intento abortado muestra la memoria privada del propio proceso subiendo de forma continua y consumiendo el RAM libre en paralelo — es la fase ya documentada de instanciación del modelo + `torch.load(mmap=True)` (que reserva ~13GB de memoria comprometida casi instantáneamente), no un pico externo.

La causa raíz real: esta máquina tiene 10.177GB de RAM física total, y el baseline optimizado ya tiene un pico de carga documentado de ~13.1-13.2GB (ver sección de `mmap=True` arriba) — una brecha estructural de ~3GB que **ningún cierre de aplicaciones puede compensar** (el uso combinado de apps no esenciales en esta máquina es de ~2-3GB, no ~3GB+ recuperables). Además, corridas históricas *exitosas* de este mismo baseline ya habían tocado 20.8-196.9MB de RAM libre como comportamiento normal (ver experimentos INT8 y `verify_load_fix` arriba) — no hay margen cómodo entre "corrida exitosa" y "corrida en riesgo" en este hardware. Se verificó también que el pagefile (23.3GB, gestionado automáticamente) no es el cuello de botella — el límite es la RAM física en sí.

**Decisión (consultada con el usuario):** recalibrar el umbral a 15MB — protege solo contra agotamiento literal, aceptando que corridas exitosas legítimas pasarán muy cerca del límite, tal como ya lo hacían antes de que existiera este mecanismo. Los fallos se registran, no se descartan, tal como pide la metodología de esta campaña.

### Primera corrida real completada: 20 frames, 1 repetición (reconocimiento, no forma parte todavía del conteo de repeticiones de la campaña)

Con el umbral recalibrado, la corrida de reconocimiento (20 frames) **completó con éxito** (`sequence_results/recon_d.json`, `run_id=recon_d`):

| Etapa | t (s) | private (MB) | working set (MB) | sys avail (MB) |
|---|---|---|---|---|
| 00 inicio | 0.0 | 427.2 | 209.7 | 4808.4 |
| 01 modelo cargado | 108.6 | 8512.2 | 1649.9 | 1583.5 |
| 02 imágenes cargadas (20) | 109.7 | 8631.4 | 1742.8 | 1539.1 |
| 03 inferencia completa | 3087.6 | **13525.9** | 5350.4 | 1000.1 |

| Métrica | Valor |
|---|---|
| Tiempo de carga | 103.69 s (consistente con el baseline ya documentado, ~90-110s) |
| Tiempo total de inferencia streaming (20 frames) | **2977.80 s (≈49.6 min)** |
| Tiempo medio por frame (sobre los 20) | 148.89 s/frame |
| Pico de working set (contador interno) | 6488.9 MB |
| **Pico de memoria comprometida** (peak_pagefile) | **14219.4 MB — más alto que el pico de solo-carga ya documentado (~13.1-13.2GB)**, confirma que la inferencia en sí añade al pico, no solo la carga |
| RAM libre mínima del sistema (nunca cruzó el umbral de 15MB) | 1000.1 MB al final; mínimos puntuales más bajos vistos por el monitor externo durante la inferencia (ver `calibration_logs/ram_recon_20d_part2.csv`) |
| **ΔRAM/frame** = (final − tras carga) / n_frames = (13525.9 − 8512.2) / 20 | **250.7 MB/frame** (un solo dato — no permite todavía distinguir tendencia creciente de meseta; requiere los demás tiers) |
| Salidas: NaN/Inf | 0 / 0 |
| Claves de predicción | `pose_enc, depth, depth_conf, images, frame_type, is_keyframe` |

**Hallazgo preliminar sobre tiempo por frame — no confirmado, solo 1 corrida:** el desglose por frame individual (extraído del progreso interno, frames 9-20 tras el lote inicial de 8 "scale frames" que se procesan casi de golpe) muestra una tendencia **creciente, pero ruidosa**, no una constante limpia:

| Frame (streaming individual) | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tiempo (s) | 176 | 120 | 122 | 137 | 143 | 214 | 200 | 393 | 328 | 211 | 214 | 245 |

Los primeros frames individuales (9-13) rondan 120-176s; los últimos (14-20) rondan 200-393s — sugiere una tendencia al alza, pero con picos irregulares (frame 16 a 393s, casi 2× el frame anterior) que podrían deberse tanto a crecimiento real de costo computacional (KV-cache/contexto acumulado) como a paginación/presión de memoria en esta máquina tan ajustada (memoria privada subiendo hacia 13.5GB durante estos mismos frames) — **no se puede distinguir la causa con una sola corrida**. Coherente con la pregunta central de la campaña, pero no es evidencia suficiente todavía (criterio de conclusión de la campaña exige repeticiones consistentes, no una corrida).

### Estado de la campaña — no completada, esto es 1 de 21 corridas mínimas

**Esta única corrida de 20 frames NO forma parte del diseño formal (10/25/50/100/200)** — fue reconocimiento para calibrar el umbral de seguridad y obtener un dato real de tiempo antes de comprometerse al diseño completo. Con ~50 min para una sola corrida de 20 frames, y una tendencia de tiempo-por-frame que no es plana, el presupuesto de tiempo real de las 21 corridas propuestas (10×5, 25×5, 50×5, 100×3, 200×3) es sustancialmente mayor a lo estimado originalmente — plausgalmente muchas horas a días de ejecución continua, sobre todo en los tiers de 50/100/200 frames si el crecimiento por frame se sostiene.

**Pendiente, sin iniciar:** las 21 corridas formales del diseño (o el subconjunto reducido que el usuario decida usando su propia cláusula de contingencia ya expresada: priorizar 10/25/50 con repeticiones completas, usar 100/200 solo para confirmar tendencia con menos repeticiones, documentando explícitamente cualquier reducción). Análisis estadístico completo (media/mediana/desviación/mín/máx/tasa de fallos por tier, ΔRAM/frame vs n_frames, tiempo/frame vs n_frames, clasificación constante/lineal/no-lineal) pendiente de tener múltiples corridas por tier — no se puede hacer con n=1.

## Extensión de la campaña secuencial: N=320 (2026-08-25)

**Objetivo:** comprobar si la saturación de RAM y la estabilización de tiempo/frame observadas hasta N=200 (ver "Reporte final de la campaña de caracterización secuencial" más abajo) se mantienen al duplicar nuevamente la longitud de la secuencia. **Mismo baseline exacto que esa campaña** (CPU forzada vía `CUDA_VISIBLE_DEVICES=""`, FP32 + `mmap=True` + `del/gc.collect()`, sin tocar arquitectura/dtype/cuantización/`keyframe_interval`/resolución) — esta es una extensión de la campaña CPU, no relacionada con el baseline GPU documentado en la sección siguiente (que sí tuvo un fallo de VRAM y no llegó a saturar).

### Limitación de dataset — 400 frames no es alcanzable, resuelto con el usuario antes de correr nada

Ninguna de las tres secuencias reales del repo llega a 400 frames: `courthouse` (usada en la campaña original) tiene 286, `university` 324, `loop` 237. Pedirle al usuario cómo resolverlo (en vez de mezclar secuencias o repetir frames por mi cuenta) — decisión: usar `university` (324 frames), la trayectoria real más larga disponible, sin concatenar ni loopear nada.

**Ajuste técnico adicional, no pedido explícitamente pero necesario para mantener "no modificar `keyframe_interval`":** `demo.py` auto-selecciona `keyframe_interval=(num_frames+319)//320` cuando `num_frames>320` — es decir, a partir de 321 frames el propio pipeline sin modificar ya deja de usar `keyframe_interval=1`. Usar los 324 frames completos de `university` hubiera cruzado ese umbral, introduciendo una variable de confusión (cambio de régimen de cacheo) mezclada con el efecto de longitud de secuencia que se quiere medir. Se usó **N=320** (el máximo posible con `university` que se mantiene en `keyframe_interval=1`, igual que las 5 fases anteriores) — así la comparación against N≤200 sigue siendo una extensión limpia del mismo régimen, no una mezcla de dos efectos distintos. `university` tiene margen (324≥320).

**Bug real encontrado y corregido antes de correr esto:** `scripts_seq/run_single.py` tenía `keyframe_interval` **hardcodeado a 1** en vez de implementar la lógica de auto-selección real de `demo.py` — nunca importó en las Fases 1-5 (N≤200, todas por debajo del umbral de 320), pero para esta extensión sí hubiera importado si se llegaba a probar N>320 en el futuro. Corregido para replicar textualmente la lógica de `demo.py::main()`. No cambia ningún resultado ya documentado (todas las fases anteriores tenían N≤200, donde el valor correcto y el hardcodeado coinciden).

### Instrumentación extendida (no reemplaza nada, agrega lo pedido que faltaba)

- **`scripts_seq/monitor.py`:** se agregó utilización de CPU (`psutil.cpu_percent`, sistema y proceso) — no estaba trackeada en la campaña original. Se agregó también un timeline de frames (`note_frame()`) para poder correlacionar memoria con el número de frame exacto, no solo con tiempo transcurrido.
- **`scripts_seq/run_single.py`:** se agregó el mismo parche de `tqdm.update()` ya usado en el baseline GPU (`scripts_gpu/run_gpu_baseline.py`) para timestampear cada frame individual del streaming contra el reloj del monitor.
- **`scripts_seq/run_campaign.py`:** se agregaron `--sequence_dir` (la constante estaba hardcodeada a `courthouse`) y `--timeout_s` manual — el modelo cuadrático de timeout ya documentado arriba (calibrado con N=50/100) sobreestimó N=200 en ~27%, así que confiar en él para extrapolar a N=320 hubiera sido aún menos confiable; se usó un timeout manual generoso (3h) en vez de forzar el modelo fuera de su rango calibrado.

### Resultados — N=320, 3 repeticiones, secuencia `university` (2026-08-25)

| Métrica | Media | Mediana | Desv. estándar | Mín | Máx | Rango |
|---|---|---|---|---|---|---|
| Tiempo de carga del modelo (s) | 6.383 | 6.430 | 0.108 | 6.260 | 6.460 | 0.200 |
| Tiempo de inferencia, 320 frames (s) | 6169.660 | 6181.230 | 40.436 | 6124.700 | 6203.050 | 78.350 |
| Tiempo por frame (s) | 19.280 | 19.316 | 0.126 | 19.140 | 19.384 | 0.245 |
| Tiempo total de la corrida (s) | 6178.867 | 6190.350 | 40.534 | 6133.830 | 6212.420 | 78.590 |
| RSS pico (MB) | 20381.6 | 20391.8 | 187.1 | 20189.6 | 20563.3 | 373.7 |
| RSS tras cargar el modelo (MB) | 6149.0 | 6135.9 | 48.3 | 6108.7 | 6202.5 | 93.8 |
| RSS final, tras inferencia (MB) | 19437.4 | 19430.9 | 186.9 | 19253.8 | 19627.5 | 373.7 |
| RAM libre mínima del sistema (MB) | 6456.9 | 6362.0 | 258.9 | 6258.8 | 6749.9 | 491.1 |
| **ΔRAM/frame (MB/frame)** | **41.5** | 41.5 | 0.4 | 41.1 | 42.0 | 0.9 |
| Utilización CPU pico (sistema) | 81.5% | 81.5% | 2.2 | 79.3% | 83.6% | 4.3 |
| Utilización CPU pico (proceso) | 1436.7% | 1436.8% | 2.1 | 1434.6% | 1438.7% | 4.1 |
| `keyframe_interval` usado | 1 (las 3 repeticiones) | — | — | — | — | — |

**Tasa de fallos: 0/3 (0%).** Variabilidad entre repeticiones extremadamente baja (CV de tiempo/frame ~0.65%, de ΔRAM/frame ~1.1%) — resultado reproducible. **VRAM: no aplica** (CUDA oculta deliberadamente para preservar el baseline FP32 CPU, igual que en toda esta campaña — no confundir con la sección de baseline GPU más abajo, que sí midió VRAM real en una configuración distinta).

### Evolución frame a frame dentro de la corrida (promedio de las 3 repeticiones, misma secuencia `university`)

Checkpoints pedidos en 10/25/50/100/200/300/320 frames, correlacionando el timeline de frames (`frame_events`) con las muestras continuas de RAM:

| Frames | Tiempo acumulado (s) | RSS (MB) | Tiempo/frame acumulado (s) |
|---|---|---|---|
| 10 | 70.75 | 9659.1 | 7.075 |
| 25 | 204.88 | 12148.6 | 8.195 |
| 50 | 513.52 | 16181.2 | 10.270 |
| 100 | 1474.66 | 19467.7 | 14.747 |
| 200 | 3603.18 | 19433.9 | 18.016 |
| 300 | 5746.41 | 19472.1 | 19.155 |
| 320 | 6178.30 | 19655.3 | 19.307 |

**El hallazgo más limpio de toda esta extensión: los deltas *marginales* entre checkpoints, no solo los acumulados.**

| Intervalo | Δframes | ΔRAM (MB) | ΔRAM/frame marginal (MB) | Δtiempo (s) | Δtiempo/frame marginal (s) |
|---|---|---|---|---|---|
| 10→25 | 15 | +2489.5 | 165.97 | 134.1 | 8.942 |
| 25→50 | 25 | +4032.6 | 161.30 | 308.6 | 12.345 |
| 50→100 | 50 | +3286.5 | 65.73 | 961.1 | 19.223 |
| 100→200 | 100 | **-33.8** | **-0.34** | 2128.5 | 21.285 |
| 200→300 | 100 | **+38.2** | **0.38** | 2143.2 | 21.432 |
| 300→320 | 20 | +183.2 | 9.16 | 431.9 | 21.595 |

**RAM: el delta marginal es esencialmente cero desde el frame 100 en adelante** (-0.34, 0.38, 9.16 MB/frame — comparar contra 165.97/161.30/65.73 MB/frame de los tramos anteriores). No es solo "una tasa decreciente" como se veía hasta N=200 — es una **meseta genuina**: agregar 220 frames más (de 100 a 320) sumó un total neto de apenas ~187MB, dentro del ruido de medición entre repeticiones.

**Tiempo: el costo marginal por frame converge a una constante ≈21.3-21.6s/frame desde el frame 100.** Esto es más informativo que el tiempo/frame *acumulado* (que sigue subiendo lentamente, 14.7→18.0→19.2→19.3s, porque promedia los frames baratos del principio con los frames en régimen estable) — el costo real y actual de procesar CADA frame nuevo, una vez pasada la fase de llenado del caché KV, es prácticamente constante, no creciente. Esto confirma con más fuerza todavía la hipótesis de `kv_cache_sliding_window=64`: una vez el caché está lleno y en régimen de ventana deslizante, el costo de atención por frame deja de crecer — se estabiliza en un valor fijo, no solo desacelera.

### Comparación contra los tiers históricos (courthouse, N=10-200) — con una salvedad importante

| N | Secuencia | Tiempo/frame (s) | ΔRAM/frame (MB) |
|---|---|---|---|
| 10 | courthouse | 6.322 | 329.5 |
| 25 | courthouse | 7.609 | 245.6 |
| 50 | courthouse | 10.059 | 198.3 |
| 100 | courthouse | 14.492 | 134.3 |
| 200 | courthouse | 16.841 | 66.8 |
| **320** | **university** | **19.280** | **41.5** |

Cambio 200→320: tiempo/frame **+14.5%** (continúa desacelerando: +20.4%→+32.2%→+44.1%→+16.2%→**+14.5%**), ΔRAM/frame **-37.8%** (continúa cayendo: -25.5%→-19.3%→-32.3%→-50.3%→**-37.8%**).

**Salvedad explícita:** esta tabla compara `courthouse` (N≤200) contra `university` (N=320) — **secuencias distintas, contenido de escena distinto.** La prueba de humo N=10 en `university` (antes de la corrida completa) ya mostró una diferencia de ~8% en tiempo/frame respecto al mismo N en `courthouse` (5.824s vs 6.322s) — atribuible al contenido de la escena, no a la longitud de secuencia. Por eso **el análisis de deltas marginales dentro de la misma corrida (tabla anterior, todo en `university`) es la evidencia más confiable** para responder si el comportamiento cambia con N — no esta comparación cruzada, que mezcla dos efectos. Se incluye igual porque fue pedida explícitamente y la dirección de la tendencia (ambas desacelerando/cayendo) es consistente con el análisis marginal, no lo contradice.

### Respuestas a las cuatro preguntas de esta extensión

**1. ¿La memoria sigue saturándose entre 200 y 320(400) frames?** **Sí, con más fuerza que antes.** El ΔRAM/frame marginal es prácticamente cero (-0.34 a +9.16 MB/frame) en todo el tramo 100→320 — no es una tendencia decreciente todavía en marcha, es una meseta ya alcanzada.

**2. ¿El tiempo por frame sigue estabilizándose?** **Sí, y de forma más contundente que con el análisis acumulado únicamente.** El costo marginal por frame converge a una constante (~21.3-21.6s/frame) desde el frame 100 — el tiempo/frame acumulado sigue subiendo despacio solo porque todavía está promediando con los frames baratos del arranque, no porque el costo real siga creciendo.

**3. ¿Existe evidencia de un nuevo régimen de consumo después de 200 frames?** **No.** Todo lo contrario: el régimen que empezaba a insinuarse en la Fase 5 (N=200, ver "Reporte final" abajo) se confirma y se profundiza — mismo mecanismo (`kv_cache_sliding_window=64`), ahora con evidencia marginal mucho más limpia que la disponible con los datos hasta N=200 solamente.

**4. ¿Podemos considerar caracterizado el comportamiento temporal de LingBot-Map hasta 400 frames?** **Hasta 320, sí, con buena confianza — hasta 400 exactos, no se probó (limitación de dataset, no de comportamiento).** La evidencia marginal (RAM prácticamente plana, tiempo/frame marginal constante) es lo bastante clara y consistente entre repeticiones (variabilidad <1.2% en ambas métricas) como para esperar que el mismo régimen se sostenga si se consiguiera una secuencia real de 400+ frames — pero esa extrapolación de 320 a 400 no está medida, es una expectativa razonable, no un hecho verificado.

### Limitaciones explícitas

- No se alcanzaron 400 frames reales (limitación de dataset, resuelta con el usuario, documentada arriba) — el máximo probado es 320.
- La comparación contra los tiers históricos mezcla dos secuencias distintas (`courthouse` vs `university`) — mitigado usando el análisis de deltas marginales dentro de la misma corrida como evidencia principal, no la comparación cruzada.
- Solo se probó `keyframe_interval=1` (igual que toda la campaña anterior) — el comportamiento con `keyframe_interval>1` (el régimen que se activaría pasando de 320 frames) sigue sin probarse.
- No se validó calidad geométrica (drift de pose, degradación de profundidad) — esta extensión, igual que la campaña original, midió solo memoria y tiempo.

**Estado: extensión a N=320 completa. 3/3 corridas exitosas, 0 fallos. Las cuatro preguntas de esta extensión quedan respondidas con evidencia marginal más limpia que la disponible en el reporte original — refuerza, no contradice, las conclusiones de la campaña de 5 fases (10→200).**

## Baseline GPU real (Linux/NVIDIA) — antes de cualquier optimización/integración con Paragraphica (2026-08-25)

**Distinto a todo lo anterior: primera medición con GPU real (no CPU forzada).** La campaña de caracterización secuencial (sección arriba) forzó CPU deliberadamente para preservar el baseline FP32 y comparar contra la máquina Windows. Esta sección es la contraparte: **el baseline de rendimiento tal cual corre `demo.py` sin ningún override**, en esta misma máquina Linux, dejando que use la GPU disponible y todo lo que eso implica (incluido el cast automático a bf16 del aggregator que hace el propio `demo.py`, sin tocar ni un parámetro). Motivación: establecer este baseline antes de empezar cualquier modificación para Paragraphica, según lo pedido explícitamente. **No se repitió la campaña 10→200** (ya确ada arriba, memoria satura, sin fallos) — esto es una medición nueva y distinta: un baseline GPU detallado.

### Hardware confirmado (tras el reinicio de la máquina)

`nvidia-smi` había fallado al inicio de esta sesión (mismatch de versión de driver/NVML); tras el reinicio necesario para el apagado solicitado anteriormente, quedó funcional:
- **GPU:** NVIDIA RTX 2000 Ada Generation Laptop GPU, **8188 MiB VRAM totales** (~8GB), compute capability **8.9** (Ada Lovelace).
- Con capability≥8, `demo.py` selecciona automáticamente `dtype = torch.bfloat16` para el aggregator (líneas 462-474, código sin tocar) — cabezas (`camera_head`, `depth_head`) se mantienen en FP32 por diseño de los autores (ya documentado en la sección de FP16 más arriba, para CPU).

### Instrumentación nueva: `scripts_gpu/` (RAM+VRAM+GPU, no reemplaza `scripts_seq/`)

- **[scripts_gpu/monitor_gpu.py](scripts_gpu/monitor_gpu.py):** extiende el monitor de la campaña secuencial (RSS/USS/RAM libre vía `psutil`) agregando VRAM y métricas de GPU vía `pynvml` (NVML directo, sin overhead de invocar `nvidia-smi` por subproceso en cada muestra): `torch.cuda.memory_allocated/reserved` (preciso por-proceso) + `nvidia-smi`-equivalente vía NVML (`memory.used/free`, `utilization.gpu`, `temperature.gpu`, `power.draw`) — todo en un hilo de fondo a intervalos fijos (0.2-0.3s), igual que en la campaña CPU. **Umbral de seguridad nuevo: VRAM libre**, además del ya existente de RAM libre — necesario porque, a diferencia de la RAM (30GB en esta máquina, nunca en riesgo), la VRAM de 8GB sí es un límite real y cercano, como confirmó el primer intento (ver abajo).
- **[scripts_gpu/run_gpu_baseline.py](scripts_gpu/run_gpu_baseline.py):** reutiliza `demo.load_model()`/`demo.load_images()` **sin modificar**, y replica textualmente la lógica de selección de dispositivo/dtype/cast del aggregator de `demo.py::main()` (mismas líneas, no reinterpretadas) — así "el baseline" medido es exactamente lo que correría `python demo.py` en esta máquina, no una variante. Mismos flags documentados en toda la investigación (`use_sdpa=True`, `camera_num_iterations=1`, `kv_cache_sliding_window=64`, `num_scale_frames=8`, `image_size=518`, `keyframe_interval` auto=1 para N≤320). Se agregó un parche de `tqdm.update()` (solo en este script, no toca `lingbot_map`) para timestampear cada frame individual del loop de streaming contra el reloj del monitor — permite correlacionar memoria con número de frame exacto, no solo con tiempo transcurrido.

### Pruebas de humo — encontraron el problema central antes de comprometerse a 200 frames

- **N=10: éxito limpio.** `dtype=torch.bfloat16` confirmado, carga 9.1s, inferencia 10 frames en 2.495s (**0.25s/frame, ~4 FPS efectivo — ≈25× más rápido que CPU** en la campaña anterior, ~6.3s/frame). Pero VRAM pico ya en 6922MB de 8187MB totales (84.6%), con solo 1.66GB libres — señal de alerta.
- **N=50: el propio umbral de seguridad de VRAM se disparó** (`vram_safety_threshold_breached`, RAM libre de VRAM cayó a 225MB) antes de completar — el monitor abortó limpiamente vía `os._exit()`, sin crash real, pero **confirmó que la VRAM no se comporta como la RAM en CPU: no muestra señales de saturar**, sigue subiendo agresivamente.

**Decisión consultada con el usuario:** en vez de aceptar un N menor como "el baseline detallado" o mantener el margen de seguridad conservador, se bajó el umbral de VRAM casi a cero (30MB) para **dejar que ocurra el OOM real** — es una excepción de PyTorch normal y capturable (`torch.OutOfMemoryError`), no un crash de sistema como el `APPCRASH` de `c10.dll` documentado en la máquina Windows — así que dejarlo fallar de verdad es seguro y da el dato más preciso (el techo real, no una aproximación conservadora).

### Resultado: OOM real, reproducible, en el frame 35 de 200 (2026-08-25)

Corrida real apuntando a N=200 (`results_gpu/json/gpu_baseline_n200.json`), **reproducido dos veces en el mismo punto exacto** (primera corrida antes de un fix menor al script para no perder los snapshots parciales en caso de excepción; segunda corrida, limpia, usada para los números de abajo):

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 38.00 MiB.
GPU 0 has a total capacity of 7.62 GiB of which 14.62 MiB is free.
this process has 7.59 GiB memory in use (7.16 GiB allocated by PyTorch, 309.35 MiB reserved but unallocated)
```

Traceback: la excepción ocurre en `lingbot_map/heads/dpt_head.py::_apply_pos_embed` → `lingbot_map/heads/utils.py::make_sincos_pos_embed` (una asignación pequeña, 38MB, de la cabeza de profundidad) — **no es un pico puntual de una operación grande, es la consecuencia de haber agotado la VRAM gradualmente en los frames anteriores.** Excepción de Python normal, capturada limpiamente por el script (no crasheó el proceso ni el driver — confirmado con `nvidia-smi` inmediatamente después, GPU vuelta a estado idle, 14MiB en uso).

**Progresión de carga (snapshots discretos, corrida limpia):**

| Etapa | t (s) | RSS proceso (MB) | VRAM alloc (MB) | VRAM reserved (MB) | VRAM NVML usada (MB) |
|---|---|---|---|---|---|
| Inicio | 0.01 | 731.4 | 0.0 | 0.0 | 415.5 |
| Imágenes cargadas (200) | 0.93 | 1247.3 | 0.0 | 0.0 | 415.5 |
| Checkpoint cargado (`load_model`) | 8.56 | 1630.8 | 4631.8 | 4643.1 | 5162.5 |
| Imágenes movidas a GPU | 8.65 | 1277.3 | 3179.1 | 4643.1 | 5175.1 |

Tiempo de carga del modelo: **7.618s** (más rápido que en CPU con `mmap`, ~90-110s en la máquina Windows / ~6.2s en CPU de esta misma máquina Linux — comparable, la carga en sí no es GPU-bound).

**Progresión de VRAM por frame durante el streaming (vía el parche de `tqdm`, 27 frames individuales capturados, frames 9-35):**

| Rango de frames | ΔVRAM/frame | Patrón |
|---|---|---|
| 9 → 25 (17 frames) | **~2.1 MB/frame** | Crecimiento lento y suave, casi plano |
| 26 → 35 (9 frames, hasta el crash) | **~65-212 MB/frame** (irregular, picos de hasta +211.8MB en un solo frame) | Aceleración brusca, muy por encima del régimen anterior |

**Esto es la observación central de esta sección.** El `kv_cache_sliding_window=64` que explicó la saturación de RAM en la campaña CPU (memoria acotada una vez el caché se llena, ver reporte final arriba) **nunca llega a activarse aquí** — la ventana tiene 64 frames de capacidad, pero la GPU se queda sin VRAM en el frame ~35, **antes de que el mecanismo de acotamiento del propio modelo tenga oportunidad de actuar.** Es un hallazgo distinto y complementario al de la campaña CPU, no una repetición: en CPU (30GB de margen) el techo de memoria nunca fue el factor limitante y se pudo observar la saturación completa; en esta GPU de 8GB, el techo de hardware llega primero.

**Hipótesis no confirmada sobre el cambio de régimen en el frame ~26** (no investigada a fondo — está fuera de alcance según lo pedido, "no comiences ninguna optimización todavía"): el mensaje de error de PyTorch menciona explícitamente `"309.35 MiB is reserved by PyTorch but unallocated"` y sugiere `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — consistente con fragmentación del *allocator* cacheado de PyTorch, posiblemente por el patrón de concatenación incremental del KV-cache (cada frame nuevo requiere un tensor más grande que el anterior, y mientras el cache crece hacia el tamaño de la ventana de 64, cada `torch.cat` transitorio necesita tanto la versión vieja como la nueva en memoria a la vez). No confirmado con instrumentación adicional — queda como hipótesis para una eventual fase de optimización, no como diagnóstico cerrado.

**Utilización de GPU, temperatura y potencia (picos de toda la corrida, vía NVML):**

| Métrica | Valor |
|---|---|
| Utilización GPU pico | **100%** (sostenido durante toda la fase de streaming) |
| Temperatura GPU pico | **54°C** |
| Potencia GPU pico | **40.4W** (GPU de laptop, limitada por diseño — el `nvidia-smi` inicial mostró un cap de 40W) |
| RAM (proceso) pico | 9621.5 MB — similar al pico de RAM en CPU para N pequeños de la campaña anterior, coherente (mismo `load_model()`, mismo mmap del checkpoint en RAM antes de mover a GPU) |
| RAM libre mínima del sistema | 20220.6 MB — sin riesgo, esta máquina tiene 30GB |
| Errores/warnings capturados | Solo el `torch.OutOfMemoryError` final; ningún warning adicional durante la ejecución (el warning benigno de `scipy`/`Failed to load pretrained weights` ya documentados arriba siguen apareciendo, no son nuevos) |

### Conclusión de esta sección

**El baseline de 200 frames tal cual, sin modificar nada, NO es alcanzable en esta GPU de 8GB — el techo real medido es 35 de 200 frames (17.5% de la secuencia objetivo).** Esto no es un fallo del pipeline ni de la metodología: es el resultado correcto y esperado de medir el baseline honestamente, tal como se pidió. Confirma que, para esta GPU específica, la memoria de VRAM (no la de RAM del sistema, que sigue con margen enorme) es el recurso limitante real, y que el mecanismo de `kv_cache_sliding_window` que protege la RAM en CPU no alcanza a proteger la VRAM en este hardware porque el límite físico llega primero. **Ninguna optimización se aplicó ni se investigó a fondo** (tal como se pidió) — este resultado queda como el punto de partida cuantitativo para cualquier trabajo futuro de reducción de memoria en GPU, no como un problema ya resuelto.

**Limitaciones explícitas:** una sola corrida detallada (no una campaña con repeticiones — apropiado para esta medición exploratoria de "dónde está el techo", distinto del objetivo estadístico de la campaña CPU; el punto de crash sí se reprodujo idéntico en dos corridas, dando algo de confianza en que no es ruido). La hipótesis sobre fragmentación del allocator como causa del cambio de régimen en el frame ~26 no está confirmada. No se probó con `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (ya lo activa `demo.py` en la línea 38 salvo que se pase `--compile`, así que en rigor **ya estaba activo** en esta corrida — el mensaje de error de PyTorch lo sugiere de todos modos porque es un mensaje genérico, no específico a si ya está activado o no; no se profundizó en si ayudaría más allá de lo que ya hace).

## Análisis de interfaz para webcam RGB en vivo — solo análisis, nada implementado (2026-08-25)

**Pedido explícito:** antes de tocar Paragraphica, preparar el análisis (sin ejecutar el experimento todavía) de qué necesitaría `demo.py` para recibir un stream RGB de una webcam convencional en vez de una carpeta de imágenes o un archivo de video. Investigación de código + una prueba mínima de apertura de cámara (read-only, no se integró nada al pipeline).

### Hardware disponible en esta máquina (confirmado, no asumido)

`/dev/video0` y `/dev/video1` existen. Prueba directa con OpenCV:
```python
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)   # opened: True
# 640x480 @ 30fps, cap.read() devuelve un frame real (480, 640, 3)
```
Cámara real y funcional en esta máquina — no hace falta hardware adicional para el próximo experimento.

### Por qué el `--video_path` actual de `demo.py` NO sirve tal cual para una cámara en vivo

`load_images()` (`demo.py:57-130`) tiene una rama para `video_path` que:
```python
cap = cv2.VideoCapture(video_path)                       # (1)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))     # (2)
while True:
    ret, frame = cap.read()
    if not ret: break                                     # (3)
    ...guarda cada frame como .jpg en disco...
```
Tres problemas concretos, confirmados empíricamente (no solo por lectura de código):

1. **`cv2.VideoCapture(video_path)` con `video_path` como `str`** (así está tipado el argumento `--video_path` en el CLI) funciona para archivos de video, pero no es la forma confiable de abrir un dispositivo en vivo — hace falta un **índice entero** (`cv2.VideoCapture(0, cv2.CAP_V4L2)`), no un string. Pasar `"/dev/video0"` como string es dependiente del backend y no confiable.
2. **`cap.get(cv2.CAP_PROP_FRAME_COUNT)` devuelve `-1` para una cámara en vivo** (confirmado arriba, no supuesto) — no hay un "total de frames" para un stream sin fin, rompe la barra de progreso `tqdm(total=total_frames, ...)` (que recibiría `total=-1`).
3. **El bucle `while True: ...; if not ret: break` no tiene condición de salida natural para una cámara** — un archivo de video eventualmente devuelve `ret=False` al llegar al final; una cámara en vivo nunca deja de mandar frames en operación normal, así que **este código quedaría colgado indefinidamente**, nunca llegaría a la fase de inferencia.

Más allá de estos tres bugs puntuales, el diseño completo de esta rama es **"extraer todos los frames a disco primero, inferir después"** (pensado para video offline) — incompatible de raíz con un flujo en vivo real (capturar → inferir → mostrar, frame por frame, sin un total conocido de antemano).

### Lo que SÍ existe y se puede reutilizar sin tocar `lingbot_map` ni `demo.py`

- **`GCTStream.forward(frame_tensor, num_frame_for_scale=scale_frames, num_frame_per_block=1, causal_inference=True)`** (`lingbot_map/models/gct_stream.py:480-485`) — el modelo YA tiene una primitiva de streaming por-frame-individual con KV-cache persistente entre llamadas (es literalmente lo que usa `inference_streaming()` internamente en su Fase 2, frame por frame). **No hace falta modificar la arquitectura ni el modelo** para alimentar un frame a la vez — el hook ya existe.
- **`load_and_preprocess_images()`** (`lingbot_map/utils/load_fn.py:104`) reutiliza el mismo crop/resize/alineado-a-`patch_size` que usa todo el pipeline — pero **requiere rutas de archivo** (`Image.open(image_path)`), no acepta un frame en memoria (numpy array de OpenCV) directamente. Reutilizable tal cual solo si cada frame capturado se escribe primero a un archivo temporal (ida y vuelta a disco por frame — funciona, pero no es ideal para tiempo real).

### Lo que falta construir (en un script nuevo, no en `demo.py` ni `lingbot_map`, mismo patrón que toda la investigación hasta ahora)

1. **Un driver de captura en vivo** — análogo a `demo.py` pero abre `cv2.VideoCapture(index, cv2.CAP_V4L2)` en vez de leer una carpeta/archivo.
2. **Preprocesamiento por-frame** — o reutilizar `load_and_preprocess_images` vía archivo temporal (cero código nuevo de preprocesamiento, con costo de I/O de disco por frame), o escribir una versión mínima que opere directamente sobre el array en memoria replicando la misma lógica de crop/resize (más código, mejor para tiempo real de verdad). Ninguna de las dos implica tocar `lingbot_map`.
3. **Reimplementar las dos fases de `inference_streaming()` pero alimentadas por la cámara**: Fase 1 — acumular `num_scale_frames=8` frames iniciales de la cámara y correr el forward de escala una vez; Fase 2 — por cada frame nuevo capturado, `model.forward(..., causal_inference=True)` como arriba.
4. **Visualización en vivo** — el `PointCloudViewer` actual (`viser`) construye la escena de una vez a partir de un diccionario de predicciones ya completo (`read_data()`); no se investigó todavía si soporta actualización incremental frame-a-frame o si haría falta re-invocarlo/extenderlo. Sin resolver, marcado como pendiente explícito.

### Implicación directa del hallazgo de VRAM de esta misma sesión (conectar, no ignorar)

El baseline GPU de arriba mostró OOM real en el frame 35 (de 200) con esta misma configuración sin modificar. **Una cámara en vivo alimentando continuamente al modelo llegaría al mismo techo de VRAM en cuestión de segundos** (a ~4 FPS efectivos observados, frame 35 se alcanza en ~9 segundos de captura continua) — no es un problema exclusivo de "procesar 200 frames de un dataset", es un problema que el experimento de webcam va a encontrar de inmediato si se conecta tal cual al baseline sin modificar. Esto no bloquea preparar el análisis (ya hecho arriba), pero sí es información necesaria para cuando se decida *ejecutar* el experimento de webcam — probablemente necesite alguna mitigación de memoria (aunque sea mínima, como `output_device=cpu` para descargar predicciones ya procesadas) antes de poder correr de forma sostenida, no solo unos pocos segundos.

**Nada de esto se implementó — es análisis puro, tal como se pidió.** No se tocó `demo.py`, no se tocó `lingbot_map`, no se empezó ningún trabajo de integración con Paragraphica.

## Primera prueba con cámara real (webcam) — 400 frames reales, pipeline completo (2026-08-26)

**Pasos 2 y 3 de la prioridad experimental del usuario:** implementar la interfaz de webcam (analizada, no implementada, en la sección anterior) y verificar el funcionamiento con cámara real. **Todavía no se tocó Paragraphica**, tal como se pidió explícitamente.

### Diseño: capturar primero, procesar después (evita el problema de acoplar velocidad de cámara con velocidad de inferencia)

En vez de implementar el loop de streaming en vivo (frame de cámara → inferencia → frame de cámara..., que con CPU a ~20-25s/frame haría la captura absurdamente lenta e irreal), se separó en dos scripts nuevos, ninguno toca `demo.py` ni `lingbot_map`:

- **[scripts_webcam/capture_frames.py](scripts_webcam/capture_frames.py):** abre `/dev/video0` vía `cv2.VideoCapture(0, cv2.CAP_V4L2)` (la forma confiable identificada en el análisis anterior, no el `--video_path` de `demo.py` que no sirve para dispositivos en vivo) y graba frames reales a un ritmo fijo (cámara nativa) a una carpeta de PNGs numerados — misma estructura que `example/courthouse`. Captura 400 frames reales en **26.6s** (15fps efectivo).
- **[scripts_webcam/process_and_view.py](scripts_webcam/process_and_view.py):** reutiliza `demo.load_model`/`load_images`/`postprocess`/`prepare_for_visualization` sin modificar, corre la inferencia, y — hallazgo útil no buscado — descubrió que `PointCloudViewer` ya tiene un botón "Export GLB" en su GUI (`_export_glb()`) que exporta el point cloud reconstruido a un archivo `.glb` portable vía `trimesh`. Como no hay navegador en este entorno para hacer clic en el botón, se llama `_export_glb()` directamente desde Python — mismo código, sin modificar `lingbot_map/vis/point_cloud_viewer.py`.

### Dos bugs de entorno nuevos, encontrados y resueltos en la prueba de humo (15 frames) antes de la corrida completa

1. **`mpl_toolkits.mplot3d`/`Axes3D` roto** — dos instalaciones de matplotlib conflictivas en esta máquina (system apt `python3-matplotlib` 3.5.1 vs pip 3.10.9) rompen específicamente ese import (`ImportError: cannot import name 'docstring' from 'matplotlib'`, el paquete del sistema intenta usar una API de matplotlib vieja). Resuelto evitando `mpl_toolkits` por completo para el preview estático: proyecciones 2D simples (top-down, frontal, lateral) en vez de un eje 3D — el `.glb` ya da una vista 3D interactiva real, el preview solo necesita ser rápido y confiable.
2. **Falso "colgado" de 2+ horas, en realidad ya había terminado en ~15s:** al probar el script piped a través de `tail -40` (`comando | tail -40`, sin `-f`), `tail` no imprime nada hasta ver EOF del stream — y `viewer.run(background_mode=True)` está diseñado para no retornar nunca (mantiene el proceso vivo sirviendo el visor). El proceso ya había exportado el GLB y guardado el preview en los primeros ~15 segundos; lo que parecía un cuelgue de 2+ horas era simplemente `tail` esperando un EOF que nunca iba a llegar por diseño. Se agregó un flag `--no_serve` para corridas de validación rápida, y se corrigió la metodología de lanzamiento (redirección a archivo + `grep`/`tail -F` para monitorear, nunca `| tail` sin `-f` sobre un proceso que se deja corriendo a propósito).

### Captura real: gente moviéndose en el taller, no una escena estática

La webcam de esta máquina apunta al techo/taller. La prueba de humo (15 frames) mostró una escena mayormente estática (techo, cableado, equipos cubiertos). La captura de 400 frames coincidió con personas moviéndose frente a la cámara (dos personas visibles interactuando en frames intermedios) — escena bastante más dinámica que las secuencias `courthouse`/`university`/`loop` del dataset oficial, y a diferencia de esas, generada en vivo por primera vez en esta investigación.

### Resultado: pipeline completo, real, de punta a punta

```bash
CUDA_VISIBLE_DEVICES="" python3 scripts_webcam/process_and_view.py \
  --image_folder captures/webcam_400 --model_path checkpoints/lingbot-map.pt \
  --glb_out captures/exports/webcam_400.glb \
  --preview_png captures/exports/webcam_400_preview.png --port 8084
```

**CPU forzada deliberadamente** (no GPU) — el baseline GPU documentado arriba hace OOM real en el frame ~35 con la config sin modificar; CPU es el único camino ya validado (hasta N=320 sin fallos) para llegar a 400 frames reales sin que el pipeline falle a mitad de camino.

| Métrica | Valor |
|---|---|
| Frames capturados | 400 (reales, cámara en vivo) |
| Tiempo de captura | 26.6s (15fps efectivo) |
| Tiempo de carga del modelo | 6.0s |
| `keyframe_interval` auto-seleccionado | **2** (no 1 — `demo.py` cruza su propio umbral de auto-selección en N>320; comportamiento real sin modificar, distinto del régimen kf=1 usado en toda la campaña secuencial hasta N=320, ver nota abajo) |
| Tiempo de inferencia | 10101.9s (≈168.4 min ≈ 2.8h) |
| Tiempo por frame (promedio) | 25.25s/frame |
| Puntos en el point cloud exportado | **6,036,204** |
| Tamaño del `.glb` exportado | ~97MB |
| Errores/warnings | Ninguno fatal (el warning benigno de `Axes3D` ya resuelto arriba, no afecta el resultado) |

**Nota sobre `keyframe_interval=2`:** no es un bug ni una modificación — es la lógica de auto-selección real de `demo.py` (`(num_frames+319)//320`) actuando exactamente como en el código sin tocar, simplemente nunca se había cruzado ese umbral en las campañas anteriores (todas ≤320 a propósito). El tiempo por frame (25.25s) no es directamente comparable al de la extensión N=320 (19.28s, régimen kf=1) por este motivo, además del contenido de escena distinto (interior dinámico vs las secuencias del dataset oficial) — dos variables de confusión, no una.

**Entregables:** `.glb` exportado (visualización 3D real, portable) + preview PNG (proyecciones 2D rápidas) + visor `viser` interactivo dejado corriendo en `http://localhost:8084` / `http://172.23.13.81:8084`. No se tocó `demo.py` ni `lingbot_map` para nada de esto — solo los dos scripts nuevos en `scripts_webcam/`.

**Nada de Paragraphica todavía** — este resultado es el cierre del paso 3 de la prioridad experimental del usuario (verificar funcionamiento con cámara real), no el comienzo del paso 4.

## Filosofía de la investigación (orden estricto — no saltarse pasos)
1. Revisar estado actual del repo / lo ya instalado.
2. Confirmar CPU/RAM/GPU/SO disponibles (ya hecho: sin GPU).
3. Ejecutar un **baseline mínimo reproducible** de LingBot-Map original (pocas imágenes, CPU) y medir.
4. Identificar qué partes del pipeline consumen más recursos.
5. Reducir información **antes** de la inferencia (no tocar el modelo todavía).
6. Medir la pérdida de precisión de cada reducción contra el baseline.
7. Combinar solo las optimizaciones que mantengan calidad aceptable.
8. Recién ahí: estudiar cuantización / distillation.

Mantener el LingBot-Map original intacto como referencia; el trabajo lightweight va en una rama o carpeta separada. **No asumir que una optimización funciona — medir siempre contra el baseline.**

## Hipótesis principal
Un pipeline de reconstrucción 3D basado en LingBot-Map puede reducir mucho su coste computacional para robótica si adapta dinámicamente cuánta información visual procesa (según movimiento, complejidad y contexto de la escena), manteniendo geometría suficiente para navegación. Si funciona, el resultado es un **pipeline adaptativo para hardware restringido**, no solo "LingBot-Map más rápido".

## Inspiración: Paragraphica (Bjørn Karmann)
https://github.com/bjoernkarmann/Paragraphica — cámara que generaba imágenes a partir de contexto (GPS, hora, clima, lugares cercanos → párrafo → modelo generativo), NO una cámara óptica real.

**Lo que se reutiliza es solo la filosofía**, no la técnica: no procesar toda la información disponible indiscriminadamente — construir una representación contextual compacta y procesar solo lo relevante ("context-aware computation"). Traducido a este proyecto: un "Lightweight Context Analyzer" decide cuánto presupuesto de cómputo darle a LingBot-Map según la escena (radio contextual / cantidad de contexto ≈ resolución / frames / ventana temporal usados).

## Arquitectura conceptual objetivo
```
RGB CAMERA
   │
LIGHTWEIGHT LAYER (motion / complexity / context)
   │
COMPUTE BUDGET → LOW / MEDIUM / HIGH → sparse / normal / dense mapping
   │
LingBot-Map → Depth + Pose + Point Cloud
   │
MAP REPRESENTATION → ROS2
```

## Estrategias de optimización a investigar (en este orden)
1. **Reducción de frames** — no tomar 1 frame cada N segundos a ciegas; frame selection basado en eventos (optical flow, diferencia entre imágenes, movimiento aparente, rotación/traslación, detección de keyframes).
2. **Reducción de resolución** — comparar 1920×1080 → 1280×720 → 640×480 → otras; la pregunta es la resolución mínima que conserva precisión geométrica suficiente para navegación, medida cuantitativamente, no visualmente.
3. **Procesamiento por ventanas** — LingBot-Map ya tiene anchor context / pose-reference window / trajectory memory / keyframe_interval; investigar el efecto de reducir tamaño de ventana, frames de contexto, memoria temporal.
4. **Reconstrucción local** — mapa local + fusión/actualización a mapa global, en vez de reconstrucción global densa permanente.
5. **Point cloud downsampling** — voxelization / downsampling → representación de ocupación local para navegación (resolución del orden de centímetros puede bastar).
6. **Cuantización** — FP32 → FP16 → INT8 → INT4, siempre midiendo pérdida geométrica (depth, pose, reconstrucción, estabilidad temporal). No asumir que es gratis.
7. **Distillation** (etapa posterior, solo si lo anterior no basta) — teacher (LingBot-Map original) → student más pequeño.

## Plan experimental
Baseline (original) → Exp1 resolución → Exp2 nº de frames → Exp3 frame selection inteligente → Exp4 ventana temporal → Exp5 reconstrucción local/sparse → Exp6 point-cloud downsampling → Exp7 quantization → Exp8 combinación → Exp9 distillation (condicional a que 1-8 no basten).

## Métricas a registrar (no solo FPS)
FPS, latencia por frame, RAM, VRAM (si existe), uso CPU/GPU, tiempo total de reconstrucción, frames procesados vs. descartados, densidad de point cloud, error de profundidad, error de trayectoria, drift, estabilidad temporal, calidad geométrica, consumo energético (si se prueba en hardware embarcado). Métrica conceptual central: **calidad geométrica / coste computacional**.

## Hardware objetivo (progresivo, no todo desde el día 1)
1. Este PC sin GPU (debe funcionar primero aquí).
2. PC con GPU modesta.
3. Raspberry Pi 5.
4. Mini-PC.
5. Hardware embarcado en el robot.

## Integración ROS2 (objetivo, no inmediato)
Tópicos esperados: `/camera/image_raw`, `/camera/depth`, `/camera/pose`, `/camera/points`, fusionados con `/imu`, `/odom`, `/scan`. LingBot-Map es un **sensor visual/geométrico adicional**, no un reemplazo de LiDAR/IMU/odometría.

## Cuidado: escala métrica
La profundidad monocular tiene ambigüedad de escala — no asumir que es una medición métrica perfecta. Evaluar corrección/estabilización de escala con IMU, odometría, LiDAR o referencias conocidas, especialmente antes de usar esto en robot real.

## Próximo paso inmediato (actualizado 2026-08-24 — ver "Estado actual" al inicio del archivo para el resumen completo)

**Hecho, no repetir:**
1. ~~Confirmar estado del repo / lo instalado.~~
2. ~~Baseline mínimo con pocas imágenes.~~
3. ~~Baseline con las 10 imágenes + visor interactivo, bugs de Windows/matplotlib corregidos.~~
4. ~~Medir RAM pico durante carga e inferencia~~ — hecho exhaustivamente: crash inicial diagnosticado (APPCRASH en `c10.dll`), causa raíz localizada en `load_model()`, pico reducido de 16.1GB a 13.1GB con `mmap=True`+`del/gc.collect()` (ambos ya aplicados a `demo.py`).
5. ~~Evaluar reducción de precisión~~ — FP16 y INT8 dinámico probados y descartados para esta CPU (ver "Estado actual"); ninguno se aplicó a producción.
6. ~~Primer análisis de redundancia temporal entre frames~~ — hecho con las 10 imágenes existentes, análisis de imagen puro; resultado: baja redundancia en esta muestra (ver sección correspondiente).
7. ~~Campaña de caracterización secuencial, las 5 fases (N=10/25/50/100/200)~~ — **COMPLETA** en la máquina Linux nueva (ver "Reporte final de la campaña de caracterización secuencial" arriba), 21 corridas, 0 fallos de pipeline. Las dos preguntas centrales (¿acumula memoria? ¿tiempo/frame constante?) quedaron respondidas: memoria satura (no acumula), tiempo/frame acelera hasta N≈100 y luego se frena, coherente con `kv_cache_sliding_window=64`.

**Pendiente, sin iniciar (requiere decisión del usuario antes de arrancar):**
- Confirmar el plateau de tiempo/frame más allá de N=200 (requeriría otra secuencia más larga que las 286 imágenes de `courthouse`, o `keyframe_interval>1`) — limitación explícita documentada en el reporte final, no crítica.
- Campaña controlada de ≥10 repeticiones sobre el baseline actual optimizado (diseño completo ya documentado arriba, en la sección del reporte de crash original — adaptar al baseline con mmap+gc antes de correr, ya que fue diseñada contra el baseline sin optimizar). Esta es la campaña de **la máquina Windows**, distinta de la campaña secuencial de arriba.
- Repetir el análisis de redundancia de frames con secuencias más largas/variadas (robot detenido, giro lento, avance rápido) antes de construir un detector de cambios real.
- Crear rama git separada en este repo para una eventual versión "lightweight" (`git checkout -b lightweight`), dejando `main` como espejo del upstream intacto — no creada todavía, todo el trabajo hasta ahora vive en `main` como scripts de diagnóstico, sin tocar `lingbot_map/` ni la arquitectura.
- Reducción de resolución (Exp1 del plan experimental original) — no evaluada todavía.
- `torchao` como sucesor no-deprecado de `torch.ao.quantization` si se retoma la línea INT8 — no evaluado.

No tienes acceso a herramientas de Cowork (device_bash, etc.) aquí — esta es la terminal real de Windows.
