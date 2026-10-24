# Integración Stella-VSLAM + ROS2 Jazzy en ParaLingbot — estado por etapas

Documento vivo de la implementación. La auditoría previa (arquitectura, convenciones,
riesgos, plan) está en [STELLA_INTEGRATION_AUDIT.md](STELLA_INTEGRATION_AUDIT.md). Cada
etapa deja aquí: qué cambió, cómo se probó, criterio de éxito y cómo se revierte.

Reglas que se respetan en todas las etapas: no tocar `demo.py` ni `lingbot_map/`; el
baseline (ParaLingbot sin Stella) siempre debe poder ejecutarse con el comportamiento
actual; comparar antes de fusionar; no interpolar imágenes como solución; no inventar
métricas de confianza.

| Etapa | Estado | Fecha |
|---|---|---|
| 0 Auditoría | hecha | 2026-10-04 |
| 1 Preservar y exponer el tracking actual (tiempo + `BasicTrackingProvider`) | hecha, verificada | 2026-10-04 |
| 2 Stella aislado (compilar, publicar frames por ROS2, medir) | compilada y medida en video; falta calibración y prueba en vivo | 2026-10-04 |
| 3 Adaptador de tracking: puente ROS2 en el servidor en vivo (tee de frames + poses de Stella) | hecha, verificada en video | 2026-10-04 |
| 4 Comparación BASIC vs Stella (tiempo, ejes, escala, ATE/RPE) | hecha; resultados en TRACKING_BENCHMARK.md | 2026-10-04 |
| 5 `PoseBuffer` (pose(t) con interpolación lineal + SLERP) | hecha, verificada | 2026-10-04 |
| 6 Selector BASIC / STELLA / HYBRID | hecha; BASIC sigue por defecto (la mejora no está demostrada) | 2026-10-04 |
| 7 Fusión avanzada | estudiada y **no implementada**: sin evidencia que la justifique | 2026-10-05 |
| 8 Marcos de coordenadas y TF2 | hecha, verificada (pruebas físicas → etapa 18) | 2026-10-05 |
| 9 Escala Stella ↔ LingBot (SE(3) vs Sim(3)) | hecha: Sim(3) re-estimada por ventana; opción `stella_scale=window`, `anchor` sigue por defecto | 2026-10-05 |
| 10 Geometría registrada con la pose de referencia (externa a LingBot) | hecha (`register_with_reference`, `register_map.py`) | 2026-10-05 |
| 11 Registro espacial: métricas del mapa | hecha; circuito y lazo pendientes de grabar | 2026-10-05 |
| 12 Correcciones de Stella (keyframes) sobre la geometría histórica | hecha; sin un loop closure real en los datos | 2026-10-05 |
| 13 Acumulación del mapa re-registrable (`MapAccumulator`) | hecha | 2026-10-05 |
| 14 TSDF y Gaussian Splatting sobre el mapa registrado | hecha y medida | 2026-10-05 |
| 15 Segmentación de cielo (filtro auxiliar) | hecha; apagada por defecto (falsos positivos en interiores) | 2026-10-05 |
| 16 Visualizador (tres trayectorias, fuente usada, keyframes, correcciones) | hecha, verificada en navegador | 2026-10-05 |
| 17 Benchmark A / B / C / D / E | hecho (3 videos + repetición del fablab) | 2026-10-05 |
| 18 Matriz de pruebas | 11 de 15 cubiertas con grabaciones, 2 parciales y 2 pendientes de las pruebas físicas (rotación pura, loop closure) | 2026-10-05 |
| 19 Optimización | hecha sobre lo medido: CPU del puente 490% → 308% | 2026-10-05 |

---

## Etapa 1 — tiempo e identidad de cada frame, y el tracking actual como interfaz

### Qué faltaba

La auditoría encontró que no había timestamps en ningún punto del pipeline en vivo y que el
analizador de contexto puede elegir un frame **anterior** al último leído, así que la hora
de captura tiene que viajar pegada a la imagen. Sin eso no hay forma de asociar una pose de
Stella con el frame que LingBot procesó.

### Qué cambió (todo fuera del núcleo de LingBot)

**Nuevo `src/vivo/tracking.py`**

- `FrameMeta(stamp, frame_id, synthetic, motion_px, sharpness)`: identidad temporal de un
  frame. `stamp` en segundos: `time.time()` al recibir el frame en cámaras en vivo (mismo
  reloj que usará ROS2); sintético y reproducible en carpeta/video (`índice / fps nominal`,
  desde 0). `frame_id` es el contador de la fuente (crece con cada frame capturado, no con
  cada frame que llega al modelo); `-1` en frames sintéticos.
- `TrackingEstimate(stamp, frame_id, c2w 4x4, source, status, confidence)` y los enums
  `TrackingSource {BASIC, STELLA, HYBRID}` y `TrackingStatus {UNKNOWN, TRACKING, LOST,
  INITIALIZING, RELOCALIZING}`.
- `BasicTrackingProvider.estimate(meta, c2w, depth_conf)`: el tracking actual con la
  interfaz común. No calcula nada nuevo: envuelve la pose que la cabeza de cámara ya emite.
  Estado: `TRACKING` siempre que hubo pose (el modelo no tiene noción de "perdido").
  Confianza: solo señales observables que ya existían: `conf_mean` y `conf_p50` de
  `depth_conf` del frame, `motion_px` y `sharpness` del analizador (None si está apagado) y
  `step` (desplazamiento respecto a la estimación anterior).

**`src/vivo/live_server.py`**

- `FrameSource.read()` devuelve `(frame, FrameMeta)`. El hilo lector de webcam/URL toma la
  hora al recibir el frame; carpeta y video generan stamps sintéticos (`source_fps`, 10 por
  defecto, o el fps del archivo). `FrameSource.stamp_kind` ∈ {`live`, `synthetic`}.
- `LiveSession` tiene `tracker = BasicTrackingProvider()` y `estimate_sinks` (lista de
  callables que reciben cada `TrackingEstimate`; vacía por defecto; es el enganche del
  puente ROS2 de la etapa 3).
- `run_model(..., meta)`: por cada frame real registrado calcula la estimación BASIC, la
  graba y la emite por el WebSocket como mensaje de texto
  `{"type": "tracking", "frame_idx", "stamp", "frame_id", "source", "status", "position",
  "conf_mean", "conf_p50", "motion_px", "sharpness", "step"}`. El protocolo binario no cambia
  y el visor actual ignora los tipos que no conoce.
- `_save_session` añade al `.npz` (claves aditivas, todos los consumidores leen por nombre):
  `stamps` (float64), `frame_ids` (int64), `stamp_kind`, `pose_basic` (S,4,4 c2w),
  `extrinsic_basic` (copia de `extrinsic`, para cuando la referencia pase a ser otra),
  `pose_source` (uint8, 0 = BASIC), `track_conf_basic`, `track_motion_px`,
  `track_sharpness` (NaN si el analizador estaba apagado). `info.json` gana `tracking`.
- `POST /api/live/start` acepta `source_fps`.

**`src/vivo/context_gate.py`**: `feed(rgb, meta=None)` y `flush()` devuelven
`(rgb, sintético, meta)`; el meta del frame elegido es el suyo (no el del último leído);
los sintéticos llevan stamp interpolado y `frame_id = -1`; el gate rellena `motion_px` y
`sharpness`. El modo fuera de línea (`main`) escribe `stamp` en el `manifest.json`
(`--source_fps`, 30 por defecto).

**`src/vivo/android_camera.py`**: `read_meta()` → `(frame, stamp, frame_id)`; `read()`
sigue igual.

**`src/vivo/replay_live.py`**: `--source_fps`.

### Cómo se probó

1. **Unitarias (CPU, sin modelo):** `test/test_tracking_etapa1.py`, 7 pruebas:
   interpolación de `FrameMeta`; el gate conserva el meta del frame elegido (la imagen que
   sale es la del `frame_id` que dice), stamps monótonos, señales rellenadas; sintéticos con
   stamp entre los reales vecinos y `frame_id = -1`; gate sin meta sigue funcionando;
   `BasicTrackingProvider`; inversa w2c→c2w; stamps de carpeta.

   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_tracking_etapa1.py -q -p no:cacheprovider
   ```
   (el `pytest` 6.2.5 del sistema choca con el plugin de `anyio` instalado en `~/.local`;
   la variable evita cargarlo).

2. **Baseline intacto (GPU):** misma secuencia (`unisabana/prueba_3/frames`, 60 frames) con
   `replay_live.py`, antes y después del cambio, sin y con analizador de contexto.
   Resultado: `extrinsic`, `intrinsic`, `depth`, `depth_conf` e `images` **bit a bit
   idénticos** en los dos casos (60 y 37 frames); misma velocidad (1.98 → 2.00 frames/s) y
   misma VRAM pico (6231 MB). `pose_basic == inv(extrinsic)`, `extrinsic_basic == extrinsic`,
   stamps monótonos (`0.0, 0.1, …`; con analizador `0.0, 0.2, 0.4, 0.6, 0.7, …` con
   `frame_ids 0, 2, 4, 6, 7, …`: se ve qué frames eligió).

3. **Consumidores existentes** sobre el `.npz` nuevo, sin cambios: `npz_to_webgl.py`
   (nube + `_cameras.json`), `evaluate_consistency.py` (inlier 0.98 consecutivo), el catálogo
   del explorador (`catalog.scan_pruebas`).

4. **Mensajes `tracking` y `estimate_sinks`:** sesión de 8 frames sin grabar: un mensaje por
   frame real y el sink recibe cada estimación (ver bitácora del README, 2026-10-04).

### Criterio de éxito (cumplido)

- ParaLingbot sin Stella produce exactamente el mismo mapa que antes.
- Cada frame registrado tiene hora de captura e identidad, y la pose BASIC queda disponible
  como `TrackingEstimate` y en el `.npz`.
- Ninguna herramienta existente requirió cambios.

### Rollback

Las claves del `.npz` y el mensaje `tracking` son aditivos: ignorarlos equivale al estado
anterior. Para volver atrás del todo: revertir los cuatro archivos de `src/vivo/`
(`git checkout -- src/vivo/`) y borrar `src/vivo/tracking.py` y
`test/`.

### Limitaciones y UNKNOWN que deja

- Los stamps de cámaras en vivo se toman al **recibir** el frame en el PC (después del
  USB / Wi-Fi / decodificado H.264), no en el sensor. La latencia cámara→PC es un sesgo
  común a LingBot y Stella si ambos usan el mismo stamp, así que no afecta la asociación;
  sí afectaría una fusión con IMU u odometría del robot (fuera de alcance).
- `stamp_kind = synthetic` empieza en 0: cuando una carpeta se publique por ROS2 (etapa 2)
  habrá que decidir si conviene un origen absoluto (TF y rosbag toleran tiempos cercanos a
  0, pero algunas herramientas no).
- El visor todavía no muestra la estimación (etapa 16); el mensaje ya llega.

---

## Etapa 2 — Stella-VSLAM aislado

### 2a. Instalación (sin sudo, sin tocar `~/ros2_jazzy`)

Todo vive en `~/Rescue/stella_ws` (fuera del repo: son 150 MB de fuentes y binarios de terceros):

```
~/Rescue/stella_ws/
  deps/        fuentes de terceros (COLCON_IGNORE): stella_vslam 0.7.0 (e445b54), FBoW stella-cv (c6e3c29), g2o 20230223_git
  prefix/      instalación de g2o, FBoW y stella_vslam (CMAKE_INSTALL_PREFIX)
  src/         paquetes ROS2: vision_opencv 4.1.0 (solo cv_bridge, sin Python) y un symlink a ~/Rescue/stella_vslam_ros
  install/     colcon: cv_bridge y stella_vslam_ros
  vocab/       orb_vocab.fbow (45 MB, github.com/stella-cv/FBoW_orb_vocab)
  cmake/       yaml_cpp_ros_shim.cmake (ver abajo)
  logs/
```

Dependencias que ya estaban en el sistema y se usaron tal cual: Eigen 3.4, OpenCV 4.5.4 (apt, con
contrib/aruco), yaml-cpp 0.7 (no se usa, ver abajo), spdlog 1.9, SuiteSparse, sqlite3, TBB. No
se compiló ningún visor (Pangolin/Iridescence/socket): `--viewer none`; la visualización es la de
ParaLingbot.

Comandos (orden): g2o → FBoW → stella_vslam → `colcon build --packages-select cv_bridge
--cmake-args -DCV_BRIDGE_DISABLE_PYTHON=ON` → `colcon build --packages-select stella_vslam_ros
--cmake-args -DCMAKE_PREFIX_PATH=$W/prefix -DCMAKE_PROJECT_stella_vslam_ros_INCLUDE=$W/cmake/yaml_cpp_ros_shim.cmake`.
Los flags exactos están en los logs de `~/Rescue/stella_ws/logs/` (`*_cmake.log`).

**Problemas encontrados y cómo se resolvieron** (todos reproducibles en otra máquina Jazzy/22.04):

| Problema | Causa | Solución |
|---|---|---|
| `cv_bridge` no existe en `~/ros2_jazzy` | no se compiló `vision_opencv` con ROS2 | `vision_opencv` tag 4.1.0 (Jazzy), solo `cv_bridge`, `-DCV_BRIDGE_DISABLE_PYTHON=ON` (no hay `libboost-python`; en Python no se usa, riesgo R5). Ojo: `~/Rescue/Pedros-Rescue/install` ya tiene otro `cv_bridge` (GARDIAN); el de `stella_ws` lo sobrepone (aviso de colcon). |
| colcon quería compilar `stella_vslam` como paquete | `deps/stella_vslam` tiene `package.xml` | `COLCON_IGNORE` en `deps/`, `prefix/`, `vocab/`, `logs/` |
| `Target "run_slam" links to target "yaml-cpp::yaml-cpp" but the target was not found` | ROS2 Jazzy trae su propio **yaml-cpp 0.8** (`yaml_cpp_vendor`, target `yaml-cpp::yaml-cpp`), que rosbag2/rcl enlazan; stella_vslam y su nodo enlazan el target plano `yaml-cpp`, que resolvía al 0.7 del sistema: **dos yaml-cpp en el mismo proceso** | `cmake/yaml_cpp_ros_shim.cmake`, inyectado con `-DCMAKE_PROJECT_<proyecto>_INCLUDE` en stella_vslam y en stella_vslam_ros: fuerza `yaml-cpp_DIR` al de ROS2 y define `yaml-cpp` como alias de `yaml-cpp::yaml-cpp`. Verificado con `ldd`: un solo `libyaml-cpp.so.0.8`. |
| `library_path.dsv` "generated by multiple different commands" | en Jazzy `ament_export_targets(... HAS_LIBRARY_TARGET)` ya registra el hook de LIBRARY_PATH y el CMake del nodo lo registraba otra vez | guardado con `ament_cmake_VERSION VERSION_LESS 2.0.0` (parche al fork) |
| `cv_bridge/cv_bridge.h: No such file` | vision_opencv 4.x renombró el encabezado a `.hpp` | `#if __has_include(<cv_bridge/cv_bridge.hpp>)` (parche al fork) |
| `libyaml-cpp.so.0.8 => not found` al ejecutar | `yaml_cpp_vendor` no exporta su `lib/` al entorno | `run_stella.sh` lo añade a `LD_LIBRARY_PATH` junto con `prefix/lib` |
| `COLCON_TRACE: unbound variable` | `set -u` con los `setup.bash` de ROS2 | sin `-u` en `run_stella.sh` |
| `run_slam` huérfano tras cerrar la prueba | `run_stella.sh` hace `exec ros2 run`, que lanza `run_slam` como hijo; la señal al envoltorio no llegaba al hijo, y el huérfano seguía publicando en los mismos topics (contaminó una corrida, descartada) | `stella_offline_test.sh` manda SIGINT al `run_slam` real, espera el cierre ordenado (escribe las trayectorias) y se niega a arrancar si ya hay uno |

**Parche al fork `~/Rescue/stella_vslam_ros`** (28 líneas, `git diff` en ese repo): publica el estado del
tracker (`~/tracking_state`, `std_msgs/String`: `Initializing` | `Tracking` | `Lost`) en cada frame,
porque `camera_pose` solo se emite cuando el tracking tiene éxito y el silencio no es un estado
(auditoría § 3.2); más los dos arreglos de compilación de la tabla.

### 2b. Herramientas nuevas en el repo (`src/ros/`)

| Archivo | Qué hace |
|---|---|
| `stella/*.yaml` | configs monoculares de Stella: webcam 640x480, celular 960x540 y 1280x720, videos unisabana en vertical 540x960 y 1080x1920 (y prueba_4 reescalado). **Intrínsecos estimados desde LingBot** (mediana del FoV predicho, reescalada a la imagen completa, principal al centro, sin distorsión); hay que reemplazarlos por una calibración con patrón. |
| `camera_publisher.py` | nodo de cámara: cualquier `FrameSource` de ParaLingbot (carpeta, video, webcam, URL, celular) → `sensor_msgs/Image` bgr8 + `CameraInfo`, con `header.stamp = FrameMeta.stamp` y `frame_id = paralingbot_camera_optical`; `--rotation`, `--resize`, `--stride`, `--rate`; sin cv_bridge; registra cada frame publicado en un JSONL. |
| `run_stella.sh` | lanza `run_slam` (nodo `stella`, sin visor) con el entorno correcto; topics `/stella/camera_pose`, `/stella/keyframes`, `/stella/tracking_state`; entrada `/paralingbot/camera/image_raw`. |
| `record_stella.py` | graba poses, estados y keyframes (las poses completas de los keyframes solo cuando cambian: ahí quedan las correcciones de loop closure) y resume. |
| `stella_offline_test.sh` | prueba reproducible: grabador + Stella + publicador sobre una fuente; deja `stella.jsonl`, `published.jsonl`, logs y `traj/` (TUM). |
| `stella_report.py` | métricas de una corrida (frames publicados vs procesados, inicialización, fracción en tracking, episodios Lost con segundo de video, keyframes, correcciones) y trayectoria convertida a la convención CV de ParaLingbot (`stella_traj.npz`, `--png`). |

Conversión de frames usada por el informe: el nodo publica `T_ros = R · T_cv · R⁻¹` con
`R = [[0,0,1],[-1,0,0],[0,-1,0]]`; se deshace con `T_cv = R⁻¹ · T_ros · R`. Las poses de Stella llevan
el **mismo `header.stamp`** que la imagen que las produjo, así que la asociación con el frame
publicado (y con el que ve LingBot) es exacta, sin interpolar, en los frames que Stella procesó.

### 2c. Medición: Stella aislado sobre los videos de las pruebas reales

Fuente: los `.mp4` de `captures/pruebas_reales/unisabana/` publicados con `camera_publisher.py`
(rotación 90° horario, verificada contra los frames de ffmpeg; reescalados a 540x960; a su fps
nativo, 30; QoS reliable). Stella con intrínsecos estimados desde LingBot, parámetros por defecto,
sin visor, en CPU (20 núcleos). El nodo se suscribe con QoS `sensor_data` (profundidad 1): cuando no
alcanza, descarta frames en vez de encolarlos. Artefactos en `captures/stella/etapa2_2026-10-04/`.

| Corrida | Video | Frames publicados / procesados | Inicializa | En Tracking | Pérdidas | Keyframes | Trayectoria |
|---|---|---|---|---|---|---|---|
| fablab_30s | fablab (lenta), 30 s | 900 / 644 (72%) | 0.46 s | 81% de los procesados | 1 a los 24.6 s, no recupera | 69 | 1.77 u., continua |
| fablab_30fps | fablab completo, 47 s | 1397 / 890 (64%) | 0.9 s | 47% | 1 a los 22.3 s, no recupera (25 s perdido) | 50 | 1.50 u., continua hasta la pérdida |
| fablab_15fps | fablab, stride 2 a 15 fps | 699 / 412 (59%) | 0.6 s | 36% | 1 a los 21.2 s, no recupera | 40 | 1.54 u. |
| pasillos_30fps | pasillos (rápida), 66 s | 1962 / 1349 (69%) | **nunca** | 0% | — | 0 | — |
| pasillos_desde20s | pasillos desde el s 20 | 1362 / 869 | a los 30 s, y el sistema se resetea enseguida (dos veces) | 9% | — | 0 | — |
| escaleras_30fps | escaleras ida y vuelta, 74 s | 2197 / 1354 (62%) | 0.36 s | 17% | 1 a los 11.5 s; **relocaliza a los 72 s** al volver al punto de partida (frame 1317 ↔ keyframe 22) | 49 | 3.34 u. |

Tiempo de tracking por frame (`track_times.txt` de `run_slam`): mediana 13-20 ms, p90 19-30 ms,
máximo 61 ms. O sea, Stella podría ir a 50+ fps en esta CPU; lo que limita el 60-70% de frames
procesados es la cola de profundidad 1 del nodo frente a un publicador a 30 fps (la extracción ORB y
el tracking compiten por el hilo de la cola), no el costo. A 15 fps (la cadencia real del celular)
procesa el 59% y el tracking es **peor**, no mejor: más movimiento entre frames.

**Qué se aprendió (hechos, no suposiciones):**

1. **Stella inicializa en menos de un segundo** con la caminata lenta y en las escaleras, con K
   estimada y sin distorsión. En el fablab la trayectoria es continua y suave (ver `stella_traj.png`);
   la "altura" crece linealmente con el avance porque el eje y del primer frame está inclinado ~9°
   (el teléfono apuntaba un poco hacia abajo), no es deriva: el mismo efecto que `compare_route.py`
   corrige promediando el eje "abajo".
2. **Pierde el tracking con desenfoque de movimiento**, y no vuelve hasta ver un lugar conocido. En
   el fablab la pérdida coincide con un giro hacia un objeto envuelto en plástico y una pared blanca
   borrosa (nitidez 813 → 209, ORB 1769 → 681 a 540x960); en las escaleras, con un frame borroso al
   subir (nitidez 114). Después la caminata sigue hacia zonas nuevas y la relocalización monocular
   **no puede** funcionar ahí por diseño: no hay mapa. En las escaleras sí relocalizó, 61 s después,
   al regresar a la entrada. Esto es exactamente la "recuperación" y "relocalización" que la
   arquitectura quería medir.
3. **La caminata rápida de los pasillos no se puede trackear con esta configuración**: los primeros
   15 s están muy desenfocados (nitidez 37-65, ORB 250-600) y aun saltándolos la inicialización
   tarda 30 s y el sistema se resetea al perder el tracking enseguida. LingBot sí produjo un
   recorrido de esa muestra (4.75% de error de forma en windowed). **Confirma la regla 3 del plan:
   Stella no es mejor por defecto.**
4. **Las correcciones de keyframes se observan** (1-2 por corrida, desplazamientos de 0.01 u.):
   son del BA local, no cierres de bucle; ningún video cierra un bucle salvo el regreso de las
   escaleras, que llegó por relocalización.
5. **Asociación temporal exacta:** las poses de Stella llevan el `header.stamp` de la imagen; el
   100% de las poses se asociaron a un frame publicado sin interpolar.

**Implicaciones para las etapas siguientes:**

- El modo HYBRID tendrá que convivir con **Stella perdido durante tramos largos** (25-60 s) y con
  Stella **sin inicializar**. BASIC es imprescindible como portador; la etapa 6 necesita además una
  política de **reinicio de Stella** (`request_reset`) tras N segundos perdido para abrir un submapa
  nuevo y volver a alinearlo con BASIC por Sim(3), en vez de esperar una relocalización que puede no
  llegar nunca.
- El `tracking_state` que añadimos al nodo es necesario: sin él, "perdido" y "sin frames" se confunden.
- Para que Stella procese todos los frames de una reproducción hay que publicar más despacio o con
  QoS reliable en el nodo (parámetro a añadir en la etapa 3); en vivo, descartar es lo correcto.
- La nitidez y el movimiento que ya mide `ContextGate` **predicen** las pérdidas de Stella: son
  señales válidas para la lógica híbrida (regla 8), no hace falta inventar otras.

**Pendiente de la etapa 2 (no bloquea la 3):**

- Calibración con patrón de la webcam y del celular (los K actuales son estimados; UNKNOWN 6).
- Prueba en vivo con una persona caminando con el celular por adb (`camera_publisher.py --source
  android`) para medir fps y latencia reales de extremo a extremo (UNKNOWN 1 en vivo); la cámara
  estática no inicializa un SLAM monocular, así que no se puede automatizar sin alguien que camine.
- Stella sobre la imagen recortada de 518 que ve LingBot (UNKNOWN nuevo): no se probó.
- Repetir pasillos con `Initializer.verbose: true` y probar `Preprocessing.min_size` o imagen
  1080x1920 para ver si inicializa con la caminata rápida.

### Criterio de éxito de la etapa 2

Cumplido: compila, ejecuta, consume la misma cámara que ParaLingbot por ROS2, publica pose, keyframes y
estado con timestamps asociables, y se midieron FPS, latencia, pérdida, relocalización y keyframes
en tres recorridos distintos. Falta la calibración y la prueba en vivo, documentadas arriba.

### Rollback

Nada de la etapa 2 toca ParaLingbot: `src/ros/` es nuevo y `~/Rescue/stella_ws` es externo
(borrarlo revierte la instalación). El fork `~/Rescue/stella_vslam_ros` tiene 4 archivos cambiados
(`git diff`); `git checkout -- .` los revierte.

---

## Etapa 3 — adaptador de Stella dentro del servidor en vivo (puente ROS2)

### Qué cambió

**Nuevo `src/vivo/ros2_bridge.py`**

- `Ros2Bridge`: nodo `rclpy` (`paralingbot_live`) en un hilo propio. **Tee de la cámara:** `FrameSource`
  llama `on_frame(img, meta)` por cada frame **capturado** (no solo los ~2/s que llegan al modelo),
  en el hilo de la cámara; el puente lo publica como `sensor_msgs/Image` bgr8 + `CameraInfo` con
  `header.stamp = FrameMeta.stamp` en `/paralingbot/camera/image_raw` (QoS best effort, profundidad
  2; `ros2_resize` y `ros2_pub_every` opcionales). Los sintéticos del analizador nunca pasan por aquí.
  **Adaptador de Stella:** suscribe `/stella/camera_pose`, `/stella/keyframes`, `/stella/tracking_state`
  y llena un `StellaTrackingProvider`. Publica además la estimación BASIC como `PoseStamped` en
  `/paralingbot/tracking/basic_pose` (para rosbag / comparación externa).
- `StellaTrackingProvider`: convierte cada pose (`T_cv = R⁻¹·T_ros·R`) en
  `TrackingEstimate(source=STELLA, status=TRACKING)`, indexada por **stamp exacto** (clave en
  microsegundos: el stamp va y vuelve por ROS como `sec + nanosec`); `status()` = último
  `tracking_state` (`INITIALIZING` / `TRACKING` / `LOST`), o `UNKNOWN` si Stella no dijo nada en 2 s
  (no corre, o no recibe frames). `estimate_at(stamp)` devuelve `None` si Stella no procesó ese frame
  o lo procesó sin pose: no se inventa nada (la interpolación es de la etapa 5).
- `StellaProcess`: arranca `src/ros/run_stella.sh` en su propio grupo de procesos y lo cierra con
  SIGINT **al grupo** (así la señal llega al `run_slam` hijo de `ros2 run`), esperando el cierre
  ordenado y matando el grupo si no termina. Verificado: 0 procesos huérfanos tras las corridas.

**`src/vivo/live_server.py`**

- `FrameSource(..., on_frame=, realtime=)`. `realtime`: carpeta/video reproducidos **a su fps nominal
  por un hilo propio, "último gana"**, como una cámara (el modelo toma ~2 de cada 15-30 frames y
  Stella recibe todos). Es la forma de que una grabación reproduzca lo que vería el sistema en vivo,
  y por tanto la base de la comparación de la etapa 4. La reproducción espera a `resume()`, que la
  sesión llama cuando el modelo ya está cargado: la sesión empieza en el frame 0 del video. Sin
  `realtime` (por defecto) la carpeta se consume frame a frame al ritmo del modelo, como siempre.
- `LiveSession`: con `ros2: true` crea el puente, lo engancha al `FrameSource` y a `estimate_sinks`, y
  con `stella_config` arranca Stella. En `run_model`, por cada frame real: `se = bridge.stella_estimate(stamp)`
  y el estado de Stella; **solo se registran** (`rec["pose_stella"]`, `rec["stella_status"]`): la
  geometría se sigue registrando con la pose BASIC. El mensaje `tracking` del WebSocket gana
  `stella` (estimación o `null`) y `stella_status`; el `status` gana `ros2` (frames capturados /
  publicados, poses y estado de Stella). Al terminar cierra Stella y el puente.
- `.npz`: `pose_stella` (S,4,4, NaN sin pose), `stella_status` (S, uint8 `TrackingStatus`),
  `stella_traj_stamps` y `stella_traj_c2w` (todas las poses de Stella a su ritmo, para la etapa 4).
  `info.json` → `tracking.stella`.
- `POST /api/live/start` acepta `ros2`, `stella_config`, `source_realtime`, `ros2_resize`,
  `ros2_pub_every`, `ros2_image_topic`, `ros2_stella_ns`, `ros2_camera_info`, `stella_log_dir`.
  `replay_live.py` tiene las mismas opciones (`--ros2 --stella_config ... --realtime --rotation 90`).
- `_release()` llama `malloc_trim(0)`: el servidor quedaba con **12 GB de RSS anónima** tras dos
  sesiones con la VRAM ya liberada (glibc no devolvía la memoria del modelo); eso dejaba sin RAM
  a los trabajos aislados de `run_isolated.sh` (una de nuestras corridas murió por eso) y a
  Stella. Medido tras el cambio: 51 MB → 1.98 GB después de una sesión (contexto CUDA + torch).

`src/ros/camera_publisher.py` ahora usa las mismas funciones de mensajes que el puente
(`image_msg`, `camera_info_msg`, `to_time_msg`): una sola implementación.

### Cómo se probó

1. **Unitarias (13 en total, CPU):** `tests/test_ros2_bridge_etapa3.py` comprueba que el adaptador
   deshace exactamente la conversión del nodo (`T_ros = R·T_cv·R⁻¹` → `T_cv`) con poses aleatorias,
   el significado de los ejes (+x ROS = +z CV, +z ROS = −y CV), el cuaternión contra scipy, que el
   stamp sobrevive el viaje `float → builtin_interfaces/Time → float` en la misma clave, la
   disposición del `Image` bgr8, y los estados / asociación exacta / `UNKNOWN` por silencio del
   proveedor de Stella.
2. **Baseline intacto (GPU):** `replay_live.py` de 60 frames **sin** `--ros2`: `extrinsic`, `depth`,
   `depth_conf`, `intrinsic`, `images` idénticos bit a bit a la etapa 1; `pose_stella` todo NaN y
   `stella_traj_*` vacíos.
3. **Integración (GPU + ROS2 + Stella), video fablab en tiempo real** (`--realtime --rotation 90
   --context --ros2 --stella_config unisabana_portrait_540x960.yaml --ros2_resize 540x960`),
   artefactos en `captures/stella/etapa3_2026-10-04/`:

   | | publicando todos los frames (29 fps) | publicando 1 de cada 2 (15 fps, como el celular) |
   |---|---|---|
   | frames capturados / publicados | 1397 / 1397 | 1397 / 699 |
   | frames que Stella procesó | 340 (24%) | 377 (54%) |
   | Stella: poses / keyframes | 103 / 20 | 176 / 54 |
   | frames del modelo (LingBot) | 70 (empezó en el s 10: el video corría mientras cargaba el modelo; corregido con `resume()`) | 87 (desde el s 0) |
   | frames del modelo dentro del tramo trackeado por Stella | 25 | 41 |
   | de ellos con pose de Stella **exacta** (mismo stamp) | 5 | 9 |
   | a ≤ 70 ms de una pose de Stella / a ≤ 200 ms | — | 37 / 41 |
   | `run_slam` huérfanos al terminar | 0 | 0 |

   Lecturas: (a) cuando LingBot y Stella comparten la CPU, Stella procesa muchos menos frames que
   sola (54% a 15 fps frente al 64% sola a 30 fps; 24% si se le publican 30 fps): el publicador en
   Python compite con el preprocesado del modelo, y el nodo descarta lo que no alcanza. Publicar a 15
   fps es lo correcto para el celular y deja CPU a Stella. (b) **La asociación exacta por stamp es
   insuficiente** cuando los dos consumen subconjuntos distintos de los frames: solo 9 de los 41
   frames del modelo tienen pose de Stella del mismo frame, pero 37 tienen una a menos de 70 ms y
   los 41 a menos de 200 ms. Esto fija el requisito de la etapa 5: `PoseBuffer` con vecino más
   cercano dentro de una tolerancia e interpolación (lineal + SLERP) entre vecinos.

### Criterio de éxito (cumplido)

- La misma cámara alimenta a LingBot (en proceso) y a Stella (por ROS2) con el mismo stamp por frame.
- Las poses de Stella llegan como `TrackingEstimate` en la convención de ParaLingbot y quedan
  grabadas junto a las BASIC, con el estado real de Stella (incluido "no sé nada de Stella").
- Sin `ros2` el sistema es bit a bit el de antes. Con `ros2` y sin Stella corriendo, el mapeo sigue
  (el estado queda `UNKNOWN`, `pose_stella` NaN).

### Rollback

Bandera `ros2` (por defecto apagada). Para volver atrás del todo: revertir `live_server.py`,
`android_camera.py`, `replay_live.py` y borrar `ros2_bridge.py` y su test.

### Hallazgos operativos

- El servidor viejo no terminó con SIGTERM al reiniciarlo (soltó el puerto y quedó colgado con 12 GB;
  hubo que matarlo con SIGKILL). Pendiente de mirar: probablemente un hilo no-daemon o el cierre de
  aiohttp sin TTY.
- Dos corridas murieron por el entorno, no por el código: una por VRAM ocupada por una sesión en
  vivo del celular que estaba en curso, otra por el tope de RAM de `run_isolated.sh` (5 GB) con solo
  9 GB disponibles por los 12 GB retenidos del servidor. Las dos se repitieron limpias.

---

## Etapa 4 — comparación BASIC frente a Stella

Resultados completos y lectura en [TRACKING_BENCHMARK.md](TRACKING_BENCHMARK.md). Aquí, qué cambió en el código y cómo se verificó.

### Qué cambió

- **`FrameSource` (video) usa el timestamp real del contenedor (PTS)**, en modo normal y realtime (`stamp_kind=video_pts`). Antes usaba índice / fps, que en el video del fablab (tasa variable, 13 huecos de hasta 168 ms) se desviaba hasta 0.6 s. Carpetas y cámaras no cambian.
- **Nuevo `src/vivo/traj_align.py`.** Funciones puras, reutilizables en las etapas 6 y 9:
  - `umeyama` para Sim(3) y SE(3);
  - `rotation_offsets`, solución mano-ojo cerrada con su condicionamiento;
  - `associate`, vecino más cercano con tolerancia;
  - `estimate_time_offset`, por correlación de la velocidad angular;
  - `rpe`, `windowed_scale`, `ate` y `stats`.
- **Nuevo `src/mapas/compare_tracking.py`.** Carga cuatro tipos de trayectoria: windowed, sesión BASIC, sesión Stella y corrida de Stella aislada. Las lleva al reloj PTS: el windowed se empareja por contenido de imagen con caché, y los stamps viejos de índice / fps se convierten. Compara por pares y por segmento de Stella. Escribe JSON y PNG por par y un `resumen.json`.
- **El puente ROS2 graba los reinicios de Stella.**
  - `StellaTrackingProvider` asigna un **segmento** (mapa) a cada pose: abre uno nuevo cuando Stella vuelve a `Initializing` después de haber dado poses.
  - Guarda también los cambios de estado con su hora.
  - El `.npz` de la sesión gana `stella_traj_segment`, `stella_events_t` y `stella_events_status`.
  - Sin esto, una sesión con reinicios mezclaba dos mundos en la misma Sim(3).

### Cómo se probó

1. **20 pruebas unitarias en CPU.** Seis nuevas de `traj_align` con solución conocida:
   - Umeyama recupera una Sim(3) exacta;
   - mano-ojo detecta un desfase de 90° de convención de cámara y devuelve I cuando no lo hay;
   - asociación con tolerancia;
   - desfase temporal de 0.3 s recuperado a menos de 0.03 s;
   - RPE nulo para una copia escalada, y 0.5 con la escala equivocada a la mitad;
   - la escala por ventanas detecta un salto de escala.

   Una nueva del puente: el segmento cambia tras un reinicio y no con estados repetidos.
2. **Baseline.** La repetición de 60 frames de carpeta sigue idéntica bit a bit al baseline anterior a la etapa 1.
3. **Reloj de video.** Una sesión corta desde video graba `stamps == PTS[frame_ids]` exacto.
4. **Comparación** sobre los tres videos con tres sesiones nuevas en vivo, más las corridas de las etapas 2 y 3. Ver el benchmark.

### Dos errores propios encontrados y corregidos durante la etapa

- **Mapeo de tiempos de windowed.** Al principio usaba `source_index` como índice de `candidates_full`. Para escaleras es índice de `frames/`, que toma uno de cada 3, y la referencia quedaba comprimida en 0-24 s de un video de 74 s. Ahora se empareja con las imágenes del propio `.npz`. Con eso las tres secuencias cubren el video completo, y el fablab, recalculado, dio los mismos números.
- **Normalización del ATE.** Usaba el largo entre muestras asociadas. Con un hueco de 60 s contaba una recta en vez del camino. Ahora usa el largo de `ref` en todo el tramo.

### Criterio de éxito

Cumplido. Ambos trackers corrieron sobre las mismas secuencias. Se registraron stamp, pose, estado y confianza de cada uno. Tiempo, ejes y escala se corrigieron y verificaron antes de interpretar. Se midieron ATE, error angular, RPE, pérdida de tracking y relocalización. No se fusionó nada.

### Rollback

- `compare_tracking.py` y `traj_align.py` son nuevos y solo leen.
- El cambio de stamps de video solo afecta el tiempo, no la geometría. Se revierte en `FrameSource` volviendo a índice / fps.
- Las claves nuevas del `.npz` son aditivas.

### Implicaciones para las etapas siguientes

- **Etapa 5.** El `PoseBuffer` con interpolación es imprescindible: la asociación exacta en vivo da 4-28 pares.
- **Etapa 6.**
  - Stella sirve cuando está en TRACKING y bien alimentada. Además de su estado, el selector necesita un criterio observable de **deriva de escala**: comparar el largo de los pasos de Stella con los de BASIC en una ventana.
  - En la caminata rápida, BASIC es el único tracker.
- **Etapa 9.** Sim(3) por ventana, no global.
- **Etapa 19, adelantable.** Medir Stella con núcleos reservados. En vivo rinde una fracción de lo que rinde aislada, y eso decide si HYBRID sirve en vivo.

---

## Etapa 5 — buffer temporal de poses (`PoseBuffer`)

### Qué cambió

- **Nuevo `src/vivo/pose_buffer.py`.** `PoseBuffer.query(t)` responde la pose de una fuente en el instante `t` a partir de sus muestras. Usa siempre los timestamps de adquisición, nunca la hora actual. Devuelve uno de cuatro tipos:

  | Tipo | Cuándo |
  |---|---|
  | `EXACT` | hay una muestra con ese stamp |
  | `INTERP` | entre dos muestras del **mismo segmento**, separadas ≤ `max_gap` y sin un corte (`add_break`) entre ellas |
  | `NEAREST` | la muestra más cercana está a ≤ `edge_tol`; se devuelve tal cual, **sin extrapolar** |
  | `NONE` | ninguno de los anteriores; no se inventa |

  La posición se interpola de forma lineal y la orientación por SLERP de cuaterniones, con implementación propia verificada contra scipy. **Nunca se interpolan los elementos de la matriz.** Admite llegadas fuera de orden y stamps repetidos, que reemplazan la muestra. Tiene memoria acotada y es seguro entre hilos.
- **Tolerancias medidas, no supuestas** (`src/mapas/eval_pose_interp.py`). Sobre trayectorias reales densas, se interpola cada muestra a partir de vecinas separadas `g` segundos y se compara con la medida:

  | Hueco | Stella aislada, fablab / escaleras | LingBot windowed, fablab / escaleras |
  |---|---|---|
  | 0.1 s | 1.3 / 1.8% · 0.15 / 0.38° | — |
  | 0.25 s | **1.9 / 2.4% · 0.29 / 0.68°** | 3.1 / 7.1% · 0.40 / 1.09° |
  | 0.5 s | 3.2 / 3.4% · 0.68 / 1.36° | 4.9 / 8.9% · 0.86 / 2.30° |
  | 1.0 s | 6.6 / 5.2% · 1.24 / 2.42° | 8.8 / 11.7% · 1.44 / 3.28° |

  Medianas: error de posición en % del desplazamiento en 1 s, y error de rotación. Como comparación, el desacuerdo entre trackers (RPE a 1 s, etapa 4) es de ~20% y 0.8-1.9°. Decisiones que salen de la tabla:
  - **Buffer de Stella: `max_gap = 0.25 s`, `edge_tol = 0.05 s`.** A 0.25 s la interpolación es unas diez veces menor que el desacuerdo entre trackers; a 0.5 s la rotación ya se le acerca.
  - **Las poses de LingBot tienen ruido propio por frame:** ya a 0.2 s dan 3-7%. Interpolar BASIC, que va a ~2 Hz, es de baja calidad. El diseño correcto es **consultar a Stella, que es densa, en los instantes de los frames de LingBot**, que es donde se registra la geometría.
  - Igual existe un `basic_buffer` (`max_gap = 0.75 s`) para la etapa 6. Quien lo consulte debe mirar `kind` y `gap`.
- **Puente ROS2.** `StellaTrackingProvider` llena un `PoseBuffer` con cada pose y su segmento. `estimate_near(stamp)` devuelve una `TrackingEstimate` con `assoc`, `dt_nearest`, `gap` y `segment`. `Ros2Bridge.stella_estimate` usa el buffer.
- **Sesión.** El `.npz` gana dos asociaciones:
  - **en vivo** (`stella_assoc_live`, `stella_dt_live`): lo que habría usado un selector en ese momento;
  - **a posteriori** (`pose_stella_post`, `stella_assoc_post`, `stella_dt_post`, `stella_segment_post`): la consulta repetida al guardar, con todas las poses ya llegadas.

  `info.json` gana `asociacion_en_vivo`.
- **Comparador.** `compare_tracking.py --assoc interp [--max_gap]` consulta el buffer de `b` en los instantes de `a`.
- **Dos defectos encontrados al probar en vivo, corregidos:**
  1. **`TrackingEstimate.to_json`.** Fallaba con el campo de texto `assoc` y habría emitido `NaN`, que el `JSON.parse` del navegador rechaza. Ahora es JSON estricto: NaN e infinito pasan a `null`.
  2. **Cierre ante errores.** Si la inferencia fallaba, el cierre de Stella y del puente solo ocurría en el camino normal: quedaba un `run_slam` huérfano y el proceso abortaba por el hilo de rclpy. Ahora `_stop_ros2()`, idempotente, va en el `finally` de la sesión. Verificado forzando un checkpoint inexistente: salida normal, cero huérfanos.
- **El servidor no terminaba con SIGTERM** (pendiente de la etapa 3). aiohttp esperaba hasta 60 s a que se cerraran las conexiones, y el visor mantiene un WebSocket. Ahora cierra los WebSockets al apagar, detiene la sesión en vivo si la hay, y usa `shutdown_timeout=5`. Verificado con un cliente WebSocket conectado: termina en 0.3 s.

### Cómo se probó

1. **29 pruebas unitarias en CPU.** Siete nuevas del buffer:
   - ida y vuelta de cuaterniones, incluido 180°;
   - SLERP igual a scipy y por el arco corto;
   - el caso 0°→170°, que el promedio de matrices rompe y SLERP no;
   - los cuatro tipos y sus tolerancias;
   - sin interpolación entre segmentos ni a través de cortes;
   - fuera de orden, duplicados y capacidad;
   - escritura concurrente desde tres hilos.

   Más una del adaptador de Stella (interpola dentro de un mapa y no entre mapas) y una de `to_json` estricto.
2. **Baseline.** La repetición de 60 frames de carpeta sigue idéntica bit a bit.
3. **Sesión en vivo del fablab con ROS2 y Stella.**
   - La asociación en vivo coincide con la a posteriori: la pose de Stella siempre llegó antes de que el modelo terminara su frame, así que la latencia no es un problema.
   - De los 20 frames del modelo dentro del tramo trackeado, 13 tienen pose con el buffer (4 exactas, 3 interpoladas, 6 cercanas). Con asociación exacta eran 4.
   - Los 7 restantes caen en huecos de Stella mayores que 0.25 s: no se inventan.
4. **Comparador, vecino más cercano frente a buffer.** Más pares donde Stella es densa: 176 → 185 en el fablab y 110 → 121 en escaleras. El ATE queda igual al centésimo (1.34 / 1.34%, 1.55 / 1.54%): la interpolación no distorsiona. BASIC frente a Stella dentro de la misma sesión sigue con 11-34 pares. **El límite ya no es la asociación, sino que Stella en vivo da pocas poses** (etapa 4).

### Criterio de éxito

Cumplido. `pose(t)` a partir de muestras cercanas, con timestamps de adquisición. Interpolación lineal más SLERP, nunca sobre la matriz. El frame de LingBot, la pose BASIC y la de Stella quedan asociados por instante. Las tolerancias salen de mediciones.

### Rollback

`pose_buffer.py` y `eval_pose_interp.py` son nuevos. Para volver a la asociación exacta de la etapa 3, cambiar `Ros2Bridge.stella_estimate` a `estimate_at`. Las claves nuevas del `.npz` son aditivas. Sin `ros2` nada cambia.

---

## Etapa 6 — selector de la pose de referencia: BASIC, STELLA, HYBRID

### Qué cambió

- **Nuevo `src/vivo/hybrid_tracking.py`** (`ReferenceTracker`). Se actualiza una vez por frame de LingBot con la pose BASIC, la pose de Stella de ese instante (buffer de la etapa 5), su mapa, su estado y su ritmo de poses. La referencia queda siempre en el mundo y la escala de BASIC. Hay tres modos:
  - **BASIC.** La pose de LingBot tal cual. Es el baseline.
  - **STELLA.** La pose absoluta de Stella, llevada al mundo BASIC con una Sim(3) que se fija al empezar cada mapa de Stella. Conserva lo global de Stella, como la relocalización. Sin Stella válida, avanza con los pasos de BASIC; al volver Stella al mismo mapa, salta a su pose, que es su corrección.
  - **HYBRID.** Se encadena paso a paso: el movimiento relativo de Stella entre dos frames si es válido, y si no el de BASIC. La escala del paso de Stella se lleva a la de BASIC con la mediana del cociente de largos de paso en 6 s, **por mapa de Stella**. Así la escala la fija siempre BASIC y la deriva de escala de Stella no se propaga. No salta al cambiar de fuente.
- **Criterios observables para usar un paso de Stella.** Si alguno falla, se usa el de BASIC y se graba el motivo.
  - Stella en TRACKING.
  - Pose de Stella en este frame y en el anterior, del mismo mapa.
  - Escala conocida: 4 pasos con movimiento.
  - Sin salto de escala: cociente de paso dentro de ×2.5.
  - Sin desacuerdo grosero de rotación: menos de 20°.
  - **Stella no famélica: al menos 10 poses/s en los últimos 2 s.** Es el criterio que más pesa en vivo; ver abajo por qué existe. El umbral es **provisorio**.
- **Afinidad de CPU** (adelanto de la etapa 19). `src/vivo/cpu_affinity.py`, con `--cpus` en el servidor y en `replay_live.py`, y `stella_cpus` en la sesión, que usa `taskset`.
- **Servidor y sesión.**
  - Con el puente ROS2, el selector corre en los tres modos a la vez.
  - El `.npz` gana `pose_ref_stella`, `pose_ref_hybrid`, `hybrid_usado`, `hybrid_motivo`, `stella_rate_hz` y `tracking_mode`.
  - `tracking_mode` (`basic` por defecto) elige cuál se anuncia: en el mensaje `tracking` del WebSocket (`referencia`) y en `/paralingbot/tracking/reference_pose`.
  - **La geometría se sigue registrando con BASIC.** Registrar con la referencia es la etapa 10.
- **Herramientas.**
  - `src/mapas/simulate_hybrid.py` reproduce el selector frame a frame sobre sesiones grabadas. Puede combinar la pose BASIC de una sesión con Stella aislada del mismo video para simular "Stella con CPU suficiente".
  - `compare_tracking.py` gana la fuente `poses:<npz>:<clave>` y la opción `--span`.

### Cómo se probó

1. **36 pruebas unitarias.** Siete del selector, con soluciones conocidas:
   - BASIC es exactamente BASIC;
   - HYBRID sigue a una Stella buena y reduce a menos del 20% el error de un BASIC que deriva en rotación, sin saltos;
   - cae a BASIC con Stella perdida o sin pose;
   - rechaza un salto de escala y no propaga una deriva lenta de escala de Stella;
   - no encadena pasos entre mapas;
   - STELLA corrige al relocalizar;
   - no usa una Stella famélica.
2. **El selector en vivo coincide con el simulador.** En una sesión en vivo en modo hybrid, el simulador corrido sobre la misma sesión da las mismas poses de referencia, con diferencias de 1e-15, y el mismo uso de Stella frame a frame.
3. **Baseline.** La repetición de 60 frames sigue idéntica bit a bit.

### Resultados

**Corrección de la etapa 4.** La afirmación "Stella coincide con windowed mejor que el streaming" y la de "3.4 veces menos deriva de rotación en escaleras" estaban **sesgadas por el tramo**. Stella se comparó en 0-23 s y BASIC en 0-47 s, o en 0-74 s. **Comparadas en el mismo tramo, BASIC coincide más con windowed:**

| Tramo | ATE BASIC | ATE Stella aislada | RPE rot. 1 s BASIC | RPE rot. 1 s Stella |
|---|---|---|---|---|
| fablab, 0-23 s | 0.83-0.94% | 1.34-1.5% | 0.58-0.62° | 0.81° |
| escaleras, 0-12 s | 1.23% | ~1.5% | 0.93° | 1.5-1.9° |

El error grande de rotación del streaming en escaleras está en la parte donde Stella ya estaba perdida, de 30 a 62 s. Y windowed también es LingBot: **favorece al streaming por errores compartidos.** Por eso la etapa 6 se evaluó contra referencias **independientes de LingBot**.

**Referencias independientes:**

| Caso | Indicador | BASIC | HYBRID | STELLA |
|---|---|---|---|---|
| escaleras, Stella aislada (simulada) | regreso a la puerta de entrada: \|fin − inicio\| / largo | 4.16% | 1.57% | **0.99%** |
| fablab, Stella aislada (simulada, dos sesiones BASIC) | error de forma contra el croquis | 2.04-2.11% | 1.98-2.03% | 1.95-1.97% |
| fablab, Stella **en vivo**, CPU repartida, Stella sana (291 poses, 12.5/s) | croquis | 2.07% | 2.09% | 2.11% |
| fablab, Stella en vivo famélica, **sin** criterio de ritmo | croquis | 2.04% | 2.76% | 2.78% |
| fablab, Stella en vivo famélica, **con** criterio de ritmo | croquis | 2.04% | **2.13%** | 3.61% |
| pasillos (Stella casi no inicializa) | croquis | 15.85% | 15.91% | 16.18% |

Windowed da 2.47% de forma en el fablab y 2.36% de cierre en escaleras.

**Lectura:**
1. **La hipótesis "tracking actual + Stella > tracking actual" no queda demostrada.** Contra el croquis, las diferencias son de ±0.1 puntos, dentro del ruido del método. El único indicio a favor es el cierre de escaleras: la relocalización de Stella al volver a la entrada deja el modo STELLA a 0.99% y HYBRID a 1.57%, frente a 4.16% de BASIC. Es **una sola secuencia** y es la clase de caso, con revisitas, donde Stella debería ayudar.
2. **Una Stella famélica empeora el recorrido.** Pasa de 2.04% a 2.77% contra el croquis. **El criterio de ritmo lo evita en HYBRID**, que queda en 2.13%. El modo STELLA no se salva, porque usa la pose absoluta de un mapa malo: no debe ser el modo por defecto.
3. **La degradación es controlada.** Sin Stella, con Stella perdida o famélica, HYBRID es BASIC.
4. **CPU repartida.** Con Stella en los núcleos 4-11 y el modelo en el resto, Stella dio 291 y 129 poses, frente a 176, 32 y 41 sin repartir. Procesó el 67% y el 40% de los frames, frente a 26-54%. El modelo va un 3-5% más lento. **Ayuda, pero no elimina la variabilidad.** Stella no es determinista: una tercera corrida repartida, la de modo hybrid, dio 111 poses. **UNKNOWN:** de dónde viene la variabilidad que queda; candidatos: la cola del nodo, el azar del RANSAC y la inicialización.

**Decisión:** `tracking_mode = basic` por defecto. HYBRID queda disponible y es seguro: no empeora con Stella famélica. STELLA es experimental. Para decidir con datos hacen falta secuencias con revisitas y lazos (pruebas 9 y 10 de la matriz de la etapa 18) y repeticiones.

### Rollback

`tracking_mode=basic`, que es el valor por defecto. Sin `ros2` el selector ni se crea. Los archivos nuevos son aditivos.

---

## Etapa 7 — fusión avanzada (estudiada, no implementada)

El plan dice: "Sólo después de validar el selector básico estudiar pose fusion, filtrado,
weighting, confidence-based fusion, trajectory optimization. No implementar complejidad
matemática si no existe evidencia de que sea necesaria."

**Decisión: no se implementa ahora.** La evidencia de la etapa 6 no la justifica.

- El selector simple, en sus dos variantes (paso a paso en HYBRID, absoluto en STELLA), no mostró mejora frente a BASIC contra referencias independientes: ±0.1 puntos contra el croquis. Una fusión ponderada combina las mismas dos fuentes y no puede sacar información que no está en los datos.
- Para pesar fuentes hace falta su incertidumbre. Ni LingBot ni Stella dan una covarianza de pose; solo hay señales observables. Calibrar pesos a partir de ellas requiere ground truth, que es justo lo que falta.
- La variabilidad de Stella entre corridas (111 a 291 poses con la misma configuración) es mayor que cualquier ganancia medida. Una fusión ajustada a una corrida no generalizaría.

**Cuándo retomarla.** Cuando la matriz de pruebas (etapas 17-18) con referencia independiente muestre que HYBRID o STELLA mejoran a BASIC en algún tipo de escena, y que la mejora supera la variación entre repeticiones. Las candidatas, en ese orden:

1. Optimización de grafo de poses con los cierres de bucle y relocalizaciones de Stella como restricciones sobre la trayectoria BASIC. Es lo que sugiere el único indicio a favor: el cierre de escaleras.
2. Un filtro de Kalman sobre el movimiento relativo, con covarianzas estimadas de las señales observables.

`traj_align.py`, `pose_buffer.py` y `hybrid_tracking.py` ya dan las piezas necesarias.

---

## Etapa 8 — marcos de coordenadas y TF2

Referencia completa en [COORDINATE_FRAMES.md](COORDINATE_FRAMES.md).

### Qué cambió

- **Nuevo `src/vivo/frames.py`.** Es la fuente única de convenciones: OpenCV dentro del repo, REP-103 hacia ROS2. Contiene los nombres de marcos, la matriz `A` y las conversiones `c2w_cv_to_map_link`, `map_optical`, `sim3_cv_to_ros` y `gravity_tilt_deg`. `ros2_bridge.R_ROS_TO_CV` ahora sale de ahí.
- **El puente ROS2 publica TF2 (REP-105).**
  - Estáticos: `paralingbot_map → paralingbot_odom` (identidad), `paralingbot_map → paralingbot_map_cv` y `paralingbot_camera_link → paralingbot_camera_optical`.
  - Dinámico: `paralingbot_odom → paralingbot_camera_link` con la pose de referencia, sellado con el stamp de adquisición del frame.
- **Topics de pose en convención ROS.** `basic_pose` y `reference_pose` publican `camera_link` en `paralingbot_map`. Antes publicaban la c2w interna con ejes OpenCV bajo un `frame_id` de mapa.
- **Alineación Stella → ParaLingbot** en `/paralingbot/alignment/stella`. Es una Sim(3) en JSON, latcheada, una por mapa de Stella anclado. **No va por TF**, porque TF es rígido y escondería la escala.
- **Stella con nombres propios.** `map_frame:=stella_map` y `camera_frame:=stella_camera_link`, en `run_stella.sh`, para no colisionar con el `map` de GARDIAN.
- **Origen de tiempo hacia ROS2.** Carpetas y videos empiezan en stamp 0, y en tf2 el tiempo 0 significa "lo más reciente": el primer frame se buscaba mal (error 0.77). Ahora, para esas fuentes, todo lo que sale a ROS2 se corre con la hora de arranque (`stamp_offset_ros`), y a lo que vuelve de Stella se le resta.
- **Nuevo `src/ros/check_frames.py`.** Graba TF, pose y alineación durante una sesión, busca `map ← optical` con tf2 en el instante de cada frame y lo compara con el `.npz`.

### Cómo se probó

- **43 pruebas unitarias.** Siete nuevas de marcos: quiralidad, adelante = +x, izquierda = +yaw, primer frame = identidad, cadena TF, ida y vuelta, conversión de Stella, Sim(3) CV ↔ ROS e inclinación.
- **De punta a punta, en vivo:**
  - Modo BASIC: la TF reproduce la pose grabada en 60/60 frames (8.7e-8); la TF y el topic coinciden (3e-16); los estáticos son correctos.
  - Modo STELLA, con Stella en núcleos reservados: 86/86 frames (2.9e-7).
  - La Sim(3) publicada reproduce la referencia en los 9 frames en que el modo STELLA usó a Stella (5e-8).
- Sin `ros2` nada de esto se ejecuta: el camino del baseline no cambió en esta etapa.

### Limitaciones

- El mapa no está nivelado con la gravedad: 4.5°, 23° y 33° en tres sesiones. Necesita una IMU o una estimación de la vertical (`paralingbot_map_level`, en el robot).
- Las pruebas físicas de la etapa 8 (estática, lineal, rotación, circuito, regreso) van con la matriz de la etapa 18.
- Las unidades no son metros (UNKNOWN hasta tener una referencia métrica).

### Rollback

Sin `ros2` no cambia nada. Para volver atrás: revertir `ros2_bridge.py` y `run_stella.sh`, y borrar `frames.py` y `check_frames.py`.

---

## Etapa 9 — escala: SE(3) o Sim(3) entre Stella y LingBot

### Qué cambió

- **Nuevo `src/mapas/scale_alignment.py`.** Sobre los mismos pares asociados por tiempo (PoseBuffer, sin extrapolar, segmento más largo de Stella), compara cinco alineaciones de Stella hacia la referencia:
  - `se3_global`: Umeyama sin escala (escala supuesta 1);
  - `sim3_global`: una sola Sim(3) con todos los pares;
  - `sim3_inicio`: Sim(3) estimada con los primeros 5 s y aplicada al resto (lo único que se puede hacer en vivo si se fija al arrancar);
  - `sim3_ventana`: **causal**, la Sim(3) de los 5 s anteriores evaluada sobre los 2.5 s siguientes (no sobre los datos con que se estimó);
  - `sim3_oraculo`: la misma ventana evaluada sobre sí misma (cota inferior, no causal).
- Dos referencias, para no depender de una: LingBot windowed contra Stella aislada (corridas de la etapa 12, con keyframes) y BASIC en vivo contra la Stella de esa misma sesión (benchmark de la etapa 17).
- **Selector (`hybrid_tracking.py`), opción `stella_scale`.** `anchor` (por defecto, sin cambios): la escala del modo STELLA se fija al anclar cada mapa. `window`: se re-estima en cada frame con la misma escala por ventana que usa HYBRID, re-anclando la traslación sobre la pose de Stella del frame **anterior**, así no se borran los saltos de relocalización. Expuesta en el POST (`stella_scale`), en `replay_live.py --stella_scale` y en `simulate_hybrid.py --stella_scale`. La Sim(3) de `/paralingbot/alignment/stella` se vuelve a publicar cuando cambia (antes, una vez por mapa).
- El problema de escala no queda escondido: el modo STELLA publica su Sim(3) por topic propio (etapa 8) y el selector anota la escala de cada frame (`confidence["escala"]`, `["escala_stella"]`).

### Resultados

ATE en % del largo de la referencia en el tramo (`captures/stella/etapa9_2026-10-05/escala.md`, gráficas `ate_por_metodo.png` y `escala_en_el_tiempo.png`):

| Secuencia, par | Pares | SE(3) | Sim(3) global | Sim(3) inicio | **Sim(3) ventana (causal)** | Oráculo | Escala (rango por ventana) |
|---|---|---|---|---|---|---|---|
| fablab, windowed vs Stella aislada | 239 | 2.66 | 1.17 | 3.91 | **1.15** | 0.50 | 1.11 (1.03-1.27) |
| fablab, BASIC vs Stella en vivo (4 sesiones) | 31-34 | 1.34-2.21 | 0.70-0.96 | 2.30-4.83 | 1.18-1.54 | 0.30-0.47 | 1.04-1.08 (0.97-1.14) |
| escaleras, windowed vs Stella aislada | 110 | 10.1 | 5.57 | 9.02 | **3.02** | 1.29 | 2.40 (1.86-2.38) |
| escaleras, BASIC vs Stella en vivo (2 sesiones) | 11-15 | 5.5-20.3 | 0.96-1.72 | 1.77-3.68 | 1.77-3.68 | 0.76-1.78 | 1.2-2.7 |
| pasillos | ≤50 | — | — | — | — | — | Stella casi no trackea: no concluyente |

Selector en modo STELLA, misma simulación que la etapa 6 (`captures/stella/etapa9_2026-10-05/selector/`):

| Caso | BASIC | HYBRID | STELLA `anchor` | STELLA `window` |
|---|---|---|---|---|
| escaleras, regreso a la entrada (\|fin − inicio\| / largo) | 4.16% | 1.57% | 0.99% | **0.72%** |
| fablab, forma contra el croquis (3 casos) | 2.03-2.29% | 2.03-2.20% | 2.03-2.11% | 2.03-2.10% |

### Lectura

1. **Hace falta Sim(3).** SE(3) es entre 1.4 y 12 veces peor que Sim(3) en todos los pares: la escala de Stella nunca es la de LingBot (1.04 a 2.7 según la secuencia y la corrida).
2. **Una Sim(3) fijada al principio no alcanza.** Estimada con los primeros 5 s, extrapola entre 2 y 5 veces peor que la global; la escala relativa varía 4-8% en el fablab y hasta 30% en escaleras.
3. **Re-estimarla por ventana causal generaliza mejor.** En escaleras deja el error en 3.02% contra 5.57% de la global y 9.02% de la inicial; en el fablab iguala a la global (1.15 contra 1.17). La distancia al oráculo (0.5-1.3%) es lo que queda de ruido de pose dentro de la ventana.
4. **En el selector, la escala por ventana mejora el único caso con revisita** (escaleras: 0.99% → 0.72%) y es neutra en el fablab. Es una secuencia: **`anchor` sigue por defecto**; `window` queda disponible para las pruebas finales.

### Método documentado

Asociación por tiempo (PoseBuffer, `max_gap` 0.25 s) → por mapa de Stella (no se mezclan segmentos) → Umeyama con escala en ventanas de 5 s → aplicación causal. En el selector, la escala por ventana es la mediana del cociente de largos de paso BASIC/Stella en 6 s (`HybridParams.scale_window_s`), con un mínimo de 4 pasos con movimiento.

### Rollback

`stella_scale` por defecto es `anchor`: sin pasarlo, el selector es el de la etapa 6. Borrar `scale_alignment.py` no afecta a nada más.

---

## Etapa 10 — integración con LingBot: geometría registrada con la pose de referencia

### Qué cambió

- **La pose entra después del modelo.** LingBot no se toca: predice profundidad, confianza e intrínsecos por frame igual que siempre. La geometría se desproyecta con la pose BASIC (baseline) o, con `register_with_reference`, con la pose de referencia del selector (`live_server.py`). El binario del WebSocket y el `.npz` llevan la pose con la que se registró (`extrinsic_reg`, `pose_source_reg`); `extrinsic_basic` y `pose_basic` se conservan siempre.
- **Fuera de línea, `src/mapas/register_map.py`.** Registra una sesión grabada con `--reference basic | hybrid | stella | NPZ:CLAVE` y escribe la nube fusionada, un `.npz` con `extrinsic` = referencia (entrada directa de `build_maps.py`, TSDF y splat) y las métricas de la etapa 11.
- Bandera en el POST (`register_with_reference`), en `replay_live.py` y en el benchmark (configuraciones E y S).

### Cómo se probó

- Sin la bandera, el `.npz` y el WebSocket son los de antes (`extrinsic_reg` = `extrinsic`).
- Benchmark de la etapa 17: configuraciones D (geometría con BASIC), E (con HYBRID) y S (con STELLA) sobre los tres videos, más una repetición del fablab.

### Resultado

Los resultados están con la etapa 11 y la 17. En resumen: registrar con la referencia de Stella no cambia la forma del recorrido más allá de la variación entre corridas, y baja la coherencia multivista (la profundidad de LingBot se predijo junto con la pose BASIC).

### Rollback

`register_with_reference` apagado (por defecto).

---

## Etapa 11 — registro espacial: métricas del mapa registrado

### Qué cambió

`register_map.py` mide, sin ground truth salvo el croquis:

| Métrica | Qué mide |
|---|---|
| coherencia multivista | la profundidad de un frame proyectada en otro con las poses de referencia, contra la que LingBot predijo en el otro (fracción a <5%), en pares consecutivos y a ~1 s |
| duplicación | puntos crudos por vóxel y fracción de vóxeles vistos por ≥2 frames |
| continuidad | jerk relativo (p95) y saltos de rotación entre frames |
| deriva | \|fin − inicio\| / largo (útil si el recorrido vuelve) y error de forma contra el croquis |
| escala | Sim(3) de la referencia frente a BASIC y su variación en ventanas de 5 s |

### Validación progresiva (plan: movimiento pequeño → lineal → pared → habitación → circuito → regreso → loop closure)

Con lo grabado que hay (las pruebas físicas nuevas quedaron para el final por decisión del usuario):

| Caso del plan | Secuencia usada | Qué se vio |
|---|---|---|
| 1. movimiento pequeño / cámara estática | T1 webcam, T13 celular quietos | LingBot mapea; Stella no inicializa nunca (monocular sin paralaje): todo queda en BASIC |
| 2. movimiento lineal, 7. pasillo | pasillos (caminata rápida) | Stella casi no trackea; las tres referencias son BASIC; error de forma 8.1-9.9% según la corrida |
| 3-4. pared, habitación | fablab | ver tabla |
| 6. regreso al punto inicial | escaleras | cierre 1.1-2.8% con las tres referencias; Stella trackea solo el 13-16% de los frames en vivo |
| 5. circuito, 7. loop closure | no hay secuencia | **pendiente para las pruebas físicas** |

Fablab, 7 sesiones en vivo (rango entre corridas; tablas completas en `captures/stella/benchmark_2026-10-05/resultados/tablas/geometria.md` y en la repetición `benchmark_recursos_2026-10-05/`):

| Referencia | Croquis | Coherencia a 1 s | Puntos por vóxel | Vóxeles vistos ≥2 frames | ATE contra windowed | Jerk p95 |
|---|---|---|---|---|---|---|
| BASIC | 1.94-2.32% | 0.90-0.94 | 3.06-3.32 | 0.34-0.39 | 1.57-1.68% | 1.16-1.41 |
| HYBRID | 1.80-2.32% | 0.84-0.94 | 2.98-3.32 | 0.32-0.39 | 1.48-1.73% | 1.19-1.41 |
| STELLA | 1.89-2.32% | 0.84-0.94 | 2.82-3.32 | 0.29-0.39 | 1.30-1.72% | 1.25-1.55 |

### Lectura

1. **La variación entre corridas de la misma configuración es del mismo tamaño que la diferencia entre referencias.** BASIC solo va de 1.94% a 2.32% contra el croquis en el fablab y de 8.1% a 9.9% en pasillos, porque en tiempo real el modelo toma frames distintos en cada corrida. Ninguna referencia gana de forma consistente.
2. **Con la referencia de Stella baja la coherencia multivista** (0.84-0.89 contra 0.90-0.94) y la fracción de vóxeles vistos por varios frames: la profundidad de LingBot se predijo junto con su propia pose, y con otra pose encaja peor entre frames. Menos "duplicación" aquí no es una mejora: es menos solape.
3. Contra LingBot windowed, la referencia STELLA queda más cerca en 3 de 4 sesiones (1.30-1.31% contra 1.57-1.60%). Windowed favorece a BASIC (etapa 6), así que es un indicio a favor de Stella, no una prueba.

### Limitaciones

Sin ground truth métrico, sin circuito ni lazo grabado. La coherencia multivista usa la profundidad del propio modelo.

---

## Etapa 12 — loop closure: las correcciones de Stella llegan a la geometría histórica

### Qué cambió

- **Nodo de Stella (fork `~/Rescue/stella_vslam_ros`).** Publica `~/keyframes_full` (`Float64MultiArray`: stamp, n y por keyframe id, timestamp de su imagen y pose). Al implementarlo, el nodo se colgaba tras ~150 poses: un backtrace con gdb mostró que `Eigen::Transform::rotation()` hace una SVD (JacobiSVD) que no termina con valores no finitos. Se usa `linear()` (la pose ya es una rotación pura) y se saltan los keyframes no finitos. `run_stella.sh` lo activa (`STELLA_PUBLISH_KF=false` lo apaga).
- **`src/vivo/keyframe_correction.py`.** `KeyframeHistory` guarda las versiones de los keyframes y detecta cuáles se movieron; `FrameAnchors` ancla cada frame registrado al keyframe de Stella más cercano en el tiempo que ya existía y guarda la pose relativa `T_rel = T_k⁻¹ · T_f`. Cuando llega una versión nueva, `T_f' = T_k(nueva) · T_rel`. Los frames anclados a keyframes que Stella borró quedan como estaban y se cuentan.
- **En vivo (`live_server.py`).** Con el puente, cada frame se ancla y, si Stella movió keyframes, se recalculan las poses de los frames afectados. Si la geometría se registra con el modo STELLA, el servidor manda al visor `repose` (la corrección de ese frame) y el visor mueve sus puntos sin recibirlos de nuevo. Cada evento queda en `info.json` (`correcciones_keyframes_en_vivo`, las últimas 50) y las poses corregidas en el `.npz` (`pose_stella_kfcorr`, `pose_ref_stella_corr`).
- **Fuera de línea, `src/mapas/kf_correction_report.py`** aplica lo mismo a una corrida grabada y mide cuánto se movieron las poses históricas.

### Resultados

Stella aislada con keyframes (`captures/stella/etapa12_2026-10-05/`):

| Secuencia | Frames anclados / corregidos | Desplazamiento (% del largo), mediana / máx. | Cierre fin-inicio, original → corregida | ATE contra windowed, original → corregida |
|---|---|---|---|---|
| fablab | 538 / 493 (45 con keyframe borrado) | 0.12 / 1.49 | 73.0% → 71.1% | 1.18% → **1.05%** |
| escaleras | 227 / 227 | 0.30 / 8.88 | 48.9% → **41.4%** | 3.33% → **3.12%** |
| pasillos | 161 / 0 | — | — | sin keyframes vigentes (Stella no retuvo mapa) |

En vivo (benchmark): solo el modo S (STELLA + geometría con la referencia) re-registra; 15-18 frames movidos por sesión, desplazamiento máximo 0.037-0.047 (≈1.2-1.5% del largo del recorrido). En los demás modos las correcciones se calculan y graban, pero la geometría queda con BASIC.

### Lectura

- Las correcciones de Stella (BA local y relocalización) sí llegan a la historia: las poses corregidas coinciden mejor con windowed y cierran mejor que las publicadas en su momento.
- **No hubo ningún loop closure de verdad** en estos videos (los logs de Stella no tienen ninguno): todo lo medido es BA local y la relocalización de escaleras. El camino para un cierre de lazo es el mismo, pero queda sin probar con uno.

### Rollback

Sin el puente ROS2 no se ejecuta nada. El nodo sigue funcionando sin el topic nuevo (`publish_keyframes:=false`).

---

## Etapa 13 — acumulación del mapa

### Qué cambió

`src/vivo/map_accumulator.py` (`MapAccumulator`): cada frame guarda sus puntos **en coordenadas de su cámara** (la profundidad de LingBot no cambia) y, aparte, la pose con que se registra. El mapa en el mundo se arma al pedirlo, `P_mundo = T(frame) · P_cámara`, y se fusiona por vóxel (promedio de posición y color, un punto por vóxel). Una corrección de pose (etapa 12) o un cambio de referencia es cambiar una matriz por frame: no quedan superficies dobles de la pose vieja y la nueva, y no hay que correr el modelo otra vez. Memoria acotada con `max_points_per_frame`; el vóxel se fija relativo a la profundidad mediana. Lo usa `register_map.py`; los filtros que ya existían (confianza, vóxel, `geo_filter.py`) se aplican igual.

### Cómo se probó

Pruebas unitarias (`test/test_map_kf_etapas12_13.py`): re-registrar un frame mueve solo sus puntos; dos frames que ven lo mismo con poses coherentes fusionan en los mismos vóxeles; la corrección por keyframes reproduce la pose esperada. En el benchmark: 0.65-1.0 M de puntos crudos por sesión → 0.2-0.47 M de vóxeles, 9-15 MB de memoria, 4-7 s por sesión.

### Rollback

Módulo nuevo, no lo usa ningún camino del baseline.

---

## Etapa 14 — TSDF y Gaussian Splatting sobre el mapa registrado

### Qué cambió

Nada nuevo en el pipeline de mapas: `register_map.py` deja un `.npz` con `extrinsic` = pose de referencia, que es la entrada que ya aceptan `export_dense_cloud.py`, `tsdf_mesh.py`, `gsplat_train.py` y `build_maps.py`. Las cuatro representaciones se mantienen y se comparan (`captures/stella/etapa14_2026-10-05/run.sh` y `resumen.py`).

### Resultado

Sesión S_fablab del benchmark (Stella en núcleos propios, correcciones por keyframes), misma geometría de LingBot registrada con dos referencias. TSDF y splat medidos en 11 frames apartados (1 de cada 8), con los parámetros de siempre:

| | Nube cruda | Nube filtrada (confianza p35 + vóxel) | TSDF: PSNR / SSIM / coincidencia prof. | Splat: PSNR / SSIM / coincidencia prof. |
|---|---|---|---|---|
| geometría con BASIC | 14.1 M puntos | 4.84 M | **12.06** / **0.436** / **0.480** | **16.69** / **0.576** / **0.499** (0.91 M gaussianas) |
| geometría con STELLA | 14.1 M | 4.62 M | 11.32 / 0.403 / 0.444 | 15.67 / 0.538 / 0.482 (0.80 M) |

Gráfica: `comparacion_representaciones.png`.

### Lectura

- Con la pose de Stella las dos representaciones empeoran (−0.7 dB la malla, −1.0 dB el splat) y la coincidencia de profundidad baja. Es lo mismo que mostró la etapa 11: la profundidad de LingBot es coherente con su propia pose; montarla sobre otra trayectoria deja frames que no encajan entre sí.
- TSDF y splat siguen siendo complementarios: el splat gana en imagen (+4.4 a +4.6 dB) con las dos referencias; ninguno es "la geometría verdadera".

### Limitaciones

Una sesión, una corrida de splat por referencia (el splat es estocástico en la densificación: ±0.1-0.2 dB).

---

## Etapa 15 — segmentación de cielo (filtro auxiliar)

### Qué cambió

`register_map.py --sky` quita del mapa los píxeles que `skyseg.onnx` marca como cielo, con la sesión de onnxruntime de `lingbot_map/vis/sky_segmentation.py` (sin cambiarla). LingBot sigue siendo la reconstrucción; `skyseg` solo descarta puntos.

### Resultado (`captures/stella/etapa15_2026-10-05/`)

Escaleras (prueba 4: empieza y termina en exterior), sesión A del benchmark:

| | Puntos crudos | Vóxeles | Tiempo |
|---|---|---|---|
| sin filtro | 1.01 M | 468 k | 4.8 s |
| con `skyseg` | 0.92 M (−8.7%) | 415 k | 61.6 s |

`skyseg` marcó como cielo el 23% de los píxeles y más del 5% en 88 de 135 frames. El mosaico (`mosaico_escaleras.png`, cielo en rojo) muestra que **casi todo es falso positivo**: paredes blancas, el vidrio de las puertas, el exterior visto a través de una puerta y hasta el piso movido (31% en un frame borroso). El cielo real solo aparece en el último frame, y ahí también marca la fachada.

### Decisión

Apagado por defecto. Sirve para recorridos en exterior con cielo abierto; en interiores quita geometría válida. No se usa para nada más que filtrar (no reemplaza a LingBot ni corre LingBot por ONNX).

---

## Etapa 16 — visualizador

### Qué cambió

`src/mapas/webgl_viewer/` (`index.html`, `main.js`), sobre el visor existente:

- Panel "seguimiento" (aparece solo con el puente ROS2): modo de referencia, **fuente usada en cada frame** (BASIC o STELLA, con el motivo cuando cae a BASIC), estado de Stella, ritmo de poses de Stella, stamp del frame, latencia captura → mapa y número de correcciones por keyframes.
- Trayectorias BASIC (gris), STELLA (naranja) y HYBRID (verde) en el mismo mundo que la nube, cada una con su casilla; keyframes de Stella (azul) llevados al mundo con la Sim(3) del mapa; marcas en los frames re-registrados.
- Mensajes nuevos del servidor: `tracking` con `referencia` y `trayectorias`, `keyframes`, `correccion` y `repose` (mueve los puntos ya dibujados de un frame sin volver a mandarlos). El visor ignora los tipos que no conoce, así que un cliente viejo sigue funcionando.
- Ya existían y se mantienen: cámara, pose actual (marcador), nube, nube filtrada, TSDF y splat en el explorador, fps y VRAM.

### Cómo se probó

`captures/stella/etapa16_2026-10-05/prueba_visor.py`: Chromium headless abre el visor, arranca por la API una sesión en vivo (video del fablab en tiempo real, Stella en núcleos propios, modo STELLA con la geometría registrada con la referencia) y toma capturas cada 15 s. Resultado: el panel pasa de **STELLA / TRACKING / 12 poses/s** a **BASIC (stella_lost) / LOST / 0 poses/s** cuando Stella se pierde (a los ~22 s, como en la etapa 2); llegan 47 mensajes `tracking`, 46 `correccion`, 43 `repose` y 4 `keyframes`; 282 000 puntos; **ningún error de JS**. Capturas `visor_0..3.png` y la sesión en `sesion_visor/`.

### Limitaciones

Probado en un navegador sin GPU (dibuja a ~10 cuadros/s por software). La fluidez con GPU real no se midió.

---

## Etapa 17 — benchmark

### Qué cambió

- `src/ros/run_benchmark.sh`: corre cada configuración con `replay_live.py` (el mismo código que el servidor en vivo, video reproducido en tiempo real como una cámara) y un monitor de recursos.
- `src/ros/resource_monitor.py`: CPU y RAM por grupo de procesos (modelo y Stella), VRAM, uso y potencia de la GPU cada 0.5 s.
- `src/mapas/analyze_benchmark.py`: tablas (tracking, geometría, rendimiento), gráficas (trayectorias en planta, recursos en el tiempo, barras por métrica) y el registro de la geometría con cada referencia (etapas 10-11).
- Configuraciones: **A** baseline sin ROS2; **D** BASIC + STELLA (puente y Stella en núcleos 4-11; se calculan B = BASIC, C = STELLA y D = HYBRID y se registra con cada uno); **E** D con la geometría registrada con HYBRID en vivo; **S** STELLA con la geometría registrada con su referencia y correcciones por keyframes en vivo.

### Resultados

Detalle, tablas y gráficas en [TRACKING_BENCHMARK.md](TRACKING_BENCHMARK.md) (sección "Benchmark de las etapas 17-18") y en `captures/stella/benchmark_2026-10-05/resultados/` y `captures/stella/benchmark_recursos_2026-10-05/resultados/`.

### Dos defectos encontrados al correrlo

- El monitor de recursos moría en la primera muestra (modificaba el diccionario de procesos mientras lo recorría) y, como su salida iba a `/dev/null`, el primer lote quedó sin recursos. Corregido; el lote se repitió para el fablab (`benchmark_recursos_2026-10-05`) y el monitor ahora deja `monitor.log`.
- `replay_live.py` no terminaba con SIGINT ni SIGTERM: `rclpy.init()` instala sus propios manejadores, que solo apagan ROS2. Ahora el puente inicia rclpy sin manejadores y `replay_live.py` detiene la sesión como el botón "Detener": termina en 3 s, guarda lo grabado y cierra Stella.

---

## Etapa 18 — escenarios de prueba

La matriz completa, con la evidencia de cada caso, está en [TRACKING_BENCHMARK.md](TRACKING_BENCHMARK.md). Por decisión del usuario (2026-10-05), las pruebas físicas nuevas (circuito, lazo, rotación pura, alguien caminando con el celular) se hacen al final; la matriz marca cuáles quedaron cubiertas con las grabaciones existentes y cuáles no.

Dos pruebas de falla se agregaron como ganchos de prueba en el servidor (`debug_kill_stella_at_frame`, `debug_stall_at_frame` / `debug_stall_s`) y en `replay_live.py`.

---

## Etapa 19 — optimización (solo sobre lo medido)

### Qué se midió

Con el puente ROS2, el proceso del modelo usaba ~490% de CPU (mediana) contra ~300% sin él, y el modelo iba 5-8% más lento. Un perfil con `py-spy` (instalado solo en el scratchpad) mostró dos puntos calientes:

1. El hilo del puente pasaba casi todo su tiempo activo en `numpy.linalg.inv`: `frames.map_link_to_c2w_cv` invertía una matriz constante por cada keyframe de cada mensaje de Stella (~50 keyframes por mensaje).
2. El lector de video en tiempo real giraba **todos** los frames a 1080×1920 para el puente, que publica 1 de cada 2.

Publicar en sí cuesta poco: 3.7 ms el reescalado, 2.1 ms armar el mensaje, 0.3 ms publicar (≈9% de un núcleo a 15 fps).

### Qué cambió

- `frames.py`: las inversas de las transformaciones fijas se precalculan (son rotaciones: inversa = transpuesta).
- `FrameSource` pasa la rotación al consumidor (`on_frame(img, meta, prep)`) y el puente la aplica solo a los frames que publica.

### Resultado (fablab, misma configuración D, `captures/stella/etapa19_2026-10-05/`)

| | CPU del proceso del modelo (mediana) | fps del modelo | Poses de Stella |
|---|---|---|---|
| D antes (3 corridas) | 481-492% | 1.64-1.66 | 213-234 |
| **D después** | **308%** | 1.68 | 209 |
| A, sin ROS2 (referencia) | 297-311% | 1.77-1.78 | — |

El puente ya no cuesta CPU medible. Lo que queda entre A y D en fps (~5%) es Stella compartiendo la máquina.

### Lo que no se optimizó (y por qué)

- **QoS:** imagen `BEST_EFFORT` profundidad 2 hacia Stella; la cola de Stella es de profundidad 1 por diseño del nodo. No hay evidencia de pérdida por QoS: Stella procesa ~50% de lo que se le publica porque su tracking por frame no alcanza a 15 fps en estas condiciones (etapas 2-3).
- **VRAM:** 7.8-7.9 GB usados en total en la GPU de 8 GB durante las corridas (el modelo 6.2-6.3 GB según torch, el resto contexto CUDA y el visor). No hay margen para más contexto; no se cambió nada.
- **RAM:** 8.1-9.6 GB pico del proceso del modelo, Stella 215 MB. Sin cambios.
- **Timestamps, buffers, registro:** sin cuellos de botella medidos.
