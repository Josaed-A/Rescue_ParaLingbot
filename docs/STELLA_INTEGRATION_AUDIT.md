# Auditoría previa a la integración Stella-VSLAM + ROS2 Jazzy (Etapa 0)

Fecha: 2026-10-04. Rama auditada: `Vizuallizador` (commit `7b32866`, "Streaming remoto con android").
Máquina: Ubuntu 22.04, Python 3.10, RTX 2000 Ada 8 GB, 20 núcleos, 30 GB RAM, ROS 2 Jazzy
compilado desde fuente en `~/ros2_jazzy`.

Esta auditoría **no modificó código**. Su objetivo es fijar, con citas al código, dónde ocurre
hoy la captura, el seguimiento, la pose, la inferencia, el registro, la acumulación y la
visualización, y a partir de eso proponer los puntos de integración, los planes de
coordenadas, escala, tiempo y loop closure, los riesgos y las fases. Lo que no pudo
determinarse desde el repositorio queda marcado como **UNKNOWN**.

---

## 0. Resumen ejecutivo

1. **No existe un módulo de tracking separado.** El "seguimiento actual" de ParaLingbot es la
   **cabeza de cámara de LingBot-Map**: por cada frame el modelo emite una pose de 9 números
   (`absT_quaR_FoV`) junto con la profundidad. Pose y geometría salen del mismo forward. Esto
   define el baseline (MODO BASIC): la pose de LingBot.
2. **No existe estado de tracking** (válido / perdido / relocalizando) ni confianza de pose.
   Las señales observables que sí existen son: confianza de profundidad por píxel
   (`depth_conf`, usada por percentil), movimiento en píxeles por flujo óptico y nitidez
   (`ContextGate`), y suavidad de trayectoria (jerk / salto de rotación, en
   `evaluate_consistency.py`). La lógica híbrida debe construirse sobre estas, no inventar otras.
3. **No existen timestamps en todo el pipeline.** Las fuentes entregan frames sin hora de
   captura, el modelo recibe un subconjunto (descarte "último gana" y selección del
   `ContextGate`, que puede elegir un frame *anterior* al más reciente), y el `.npz` guardado
   solo tiene índices. Es la primera deuda a saldar: sin hora de captura no hay asociación
   LingBot ↔ Stella.
4. **Convención de pose del repositorio:** `extrinsic` guardado = **w2c** (mundo → cámara),
   OpenCV (x derecha, y abajo, z adelante); el mundo es la cámara del **primer frame de la
   sesión** (identidad, verificado numéricamente). Las unidades son "unidades del modelo",
   no metros. Stella monocular también tiene escala propia ⇒ la alineación Stella ↔ LingBot
   es **Sim(3)** como hipótesis de partida, posiblemente con escala variable en el tiempo.
5. **Registro y acumulación hoy:** cada frame se desproyecta con su propia pose al mundo del
   frame 0 y se concatena; no hay ICP ni optimización de poses en vivo. Fuera de línea hay
   fusión por vóxel, TSDF (Open3D) y Gaussian Splatting, siempre a partir del `.npz` por
   frame (profundidad + confianza + imagen + K + pose). **Eso hace viable el loop closure**:
   corregir poses y re-registrar es re-desproyectar, sin tocar la geometría predicha.
6. **En vivo el visor no conserva la asociación punto → frame.** El servidor sí la manda
   (cabecera con `frame_idx` y `c2w`), pero `main.js` acumula los puntos en un buffer plano.
   Para corregir geometría histórica en vivo hará falta guardar rangos por frame y un
   mensaje de "re-pose".
7. **Estado de Stella en esta máquina:** `stella_vslam_ros` está clonado (`~/Rescue/stella_vslam_ros`,
   v0.2.1) pero **sin submódulos inicializados** y **sin la librería `stella_vslam`** ni sus
   dependencias nativas (g2o, FBoW) instaladas. Tampoco existe `cv_bridge` en el ROS 2 compilado
   (`~/ros2_jazzy/install`), que `stella_vslam_ros` requiere. Hay `image_transport`, `tf2_*`,
   `message_filters`, `rosbag2` y `rclpy`. El mismo `python3` del sistema importa `torch 2.12+cu130`
   y `rclpy`, así que un puente ROS2 dentro del proceso de ParaLingbot es técnicamente posible.
8. **Punto de integración recomendado:** `src/vivo/live_server.py` (`FrameSource` y
   `LiveSession.run_model`) como "tee": la cámara se captura una sola vez, cada frame se
   publica en ROS2 con su timestamp, LingBot sigue consumiendo en proceso (baseline intacto),
   Stella consume por ROS2 y devuelve poses que un `PoseBuffer` asocia por tiempo. Sin ROS2
   activo, el comportamiento es exactamente el actual.

---

## 1. CURRENT ARCHITECTURE — arquitectura actual

### 1.1 Mapa del repositorio (lo relevante)

| Zona | Qué hace | Regla |
|---|---|---|
| `demo.py`, `lingbot_map/` | núcleo LingBot-Map (modelo, cabezas, utilidades de pose/geometría, visor viser) | **no tocar** (CLAUDE.md) |
| `src/vivo/live_server.py` | servidor único (aiohttp): visor + mapeo en vivo por WebSocket + trabajos de mapas | punto de integración principal |
| `src/vivo/context_gate.py` | analizador de contexto en vivo: flujo óptico DIS + nitidez; decide qué frames entran al modelo; síntesis opcional | fuente de señales de movimiento |
| `src/vivo/android_camera.py` | cámara de un teléfono Android por adb + scrcpy-server + ffmpeg | fuente de frames |
| `src/vivo/replay_live.py` | corre `LiveSession` sin navegador (misma ruta de código) | herramienta de benchmark |
| `src/captura/process_and_view.py` | pipeline fuera de línea (streaming o windowed) → `.npz` | productor del formato `.npz` |
| `src/mapas/windowed_lean.py` | windowed con poca RAM, misma alineación entre ventanas que el modelo | ídem |
| `src/mapas/export_dense_cloud.py`, `npz_to_webgl.py` | desproyección + fusión por vóxel; nube cruda + trayectoria | registro / acumulación |
| `src/mapas/tsdf_mesh.py` | malla TSDF (Open3D `ScalableTSDFVolume`), ICP con escala opcional | TSDF |
| `src/mapas/gsplat_train.py` | Gaussian Splatting (gsplat) inicializado desde profundidad + poses | GSplat |
| `src/mapas/geo_filter.py` | filtro geométrico (SegFormer, bordes, consistencia multivista, planos) | filtros |
| `src/mapas/compare_route.py`, `evaluate_consistency.py`, `height_profile.py`, `wall_thickness.py` | métricas existentes | benchmark |
| `src/mapas/build_maps.py` | receta única de mapas por prueba (subprocesos) | orquestador |
| `src/mapas/webgl_viewer/` | visor three.js (`main.js`), explorador (`explorer.js`), catálogo (`catalog.py`) | visualización |
| `lingbot_map/vis/sky_segmentation.py`, `skyseg*.onnx` | segmentación de cielo (filtro auxiliar) | no es reconstrucción |
| `src/upstream/demo_render/` | renderizador RGB-D del upstream (video de recorrido) | consumidor del `.npz` |

No hay carpeta `docs/` previa (esta es la primera), no hay integración ROS2 en el repo
(solo la intención en README § "Integración ROS2 (objetivo, no inmediato)"), y no hay
nada de Stella en el repo.

### 1.2 Pipeline en vivo (lo que hay que preservar)

```
FrameSource (webcam | android | url | video | folder)        live_server.py:71-205
   │  hilo lector "último gana" (webcam/url)                   :131-138
   │  AndroidCamera: ffmpeg → último frame                     android_camera.py
   ▼
LiveSession._infer                                             live_server.py:355-610
   preprocess → 518 de ancho, recorte central 518 alto         :393-404   (= crop518 del gate)
   ContextGate.feed (opcional, por defecto ON en el panel)     :406-412, context_gate.py:55-88
   run_model: model.forward(..., causal_inference=True)        :442-470
      - fase de escala: num_scale_frames (2) juntos
      - después 1 frame por llamada, KV cache persistente
      - keyframe_interval: no-keyframes no quedan en caché
   pose_enc → extrinsic/intrinsic → inversión → "W2C"          :472-481
   depth, depth_conf, K, W2C → rec[]  (grabación)              :503-508
   puntos: percentil de conf, muestreo, desproyección c2w      :512-526
   mensaje binario: frame_idx, n, c2w[16], xyz, rgb            :528-529, HEADER :64
   vista previa JPEG (tag 0xFFFFFFFF) a 15 fps aparte          :230-246
   bound_cache(): special_keep / camera_keep                   :417-437
   _save_session → captures/streaming/sin_guardar/sesion_*/    :282-330
         eval/sesion.npz + frames/ + info.json
   ▼
main.js liveOnFrame                                            main.js:773-833
   buffer plano de 6 M puntos (addUpdateRange), línea de       :786-795
   trayectoria, marcador magenta, "seguir la cámara"           :797-817, 840-864
```

Frecuencias medidas por el propio proyecto: cámara 15 fps; modelo 1.7-2.4 frames/s;
vista previa 13.4 imágenes/s; VRAM 5.4-6.2 GB en marcha, 170 MB al detener; carga ~8 s.

### 1.3 Pipeline fuera de línea

`process_and_view.py` / `windowed_lean.py` → `eval/<name>.npz` → `build_maps.py` →
`exports/` (nube, alta, cruda+trayectoria, malla, splat, video; filtro, malla_f, splat_f).
El modo `windowed` es el que corrige la deriva y **necesita toda la secuencia**: no existe en
vivo. Una sesión en vivo se guarda como prueba y se reprocesa en windowed después.

### 1.4 Formato `.npz` (contrato entre todas las herramientas)

Verificado en `captures/streaming/sin_guardar/sesion_20261004_194615/eval/sesion.npz`
(287 frames, celular 960x540 → recorte 294x518):

| clave | forma / tipo | significado |
|---|---|---|
| `depth` | (S, H, W) float16 | profundidad por píxel, unidades del modelo |
| `depth_conf` | (S, H, W) float16 | confianza ≥ 1 (no es probabilidad) |
| `images` | (S, H, W, 3) uint8 | el recorte que vio el modelo |
| `extrinsic` | (S, 3, 4) float32 | **w2c** (ver § 3) |
| `intrinsic` | (S, 3, 3) float32 | K predicha por frame (FoV del modelo), `cx=W/2, cy=H/2` |
| `ds` | int | submuestreo espacial guardado (1 en vivo) |
| `is_real` | (S,) bool | frames reales (los sintéticos no se guardan en vivo) |
| `source_index` | (S,) int | índice de origen |

**No hay timestamps ni identificador de frame de cámara.** Añadir claves nuevas es seguro:
todos los consumidores leen por nombre (`d["depth"]`, etc.).

---

## 2. CURRENT TRACKING — el seguimiento actual

### 2.1 Qué es

La pose la produce `CameraCausalHead` (`lingbot_map/heads/camera_head.py:157-294`) a partir
de los tokens agregados, con `camera_num_iterations` (4 en vivo) pasos de refinamiento
iterativo (`trunk_fn`, :296). Salida: `pose_enc [B,S,9]` = traslación (3) + cuaternión
XYZW (4) + FoV (2) (`lingbot_map/utils/pose_enc.py:72-139`). La cabeza es causal y mantiene
su propia caché KV (`kv_cache`, :247-253), que el servidor acota con `camera_keep`.

**Entrada/salida por frame** (`gct_base.py:287-355`): `pose_enc`, `depth [B,S,H,W,1]`,
`depth_conf [B,S,H,W]`, `world_points`, `world_points_conf` (+ puntos locales). El servidor
usa solo `pose_enc`, `depth`, `depth_conf`.

### 2.2 Qué NO hay

- No hay estado `TRACKING / LOST / RELOCALIZING`: el modelo siempre devuelve una pose.
- No hay confianza de pose ni covarianza.
- No hay detección de keyframes geométricos (el `keyframe_interval` es un ahorro de memoria
  fijo, no una decisión por paralaje).
- No hay relocalización ni loop closure: el streaming es causal con ventana deslizante
  (`kv_cache_sliding_window=16`) y deriva (README: 2.6× el largo real en la muestra 1).
- No hay optimización de trayectoria en vivo.

### 2.3 Señales observables existentes (para la lógica híbrida, REGLA 8)

| Señal | Dónde | Uso actual | Uso posible como "estado" BASIC |
|---|---|---|---|
| `depth_conf` por píxel | modelo | descarte por percentil (`conf_percentile`, 30-60) | media / percentil por frame como confianza de frame (en la sesión real: media por frame entre 1.007 y 5.385, mediana 2.735) |
| movimiento en px (mediana del flujo DIS a 256 px) | `ContextGate.feed` (`context_gate.py:66-68`), `stats["motion_px_median"]` | decidir si el frame entra al modelo | detectar cámara estática / movimiento rápido; comparable con el paralaje que Stella necesita |
| nitidez (var. Laplaciano vs p75 reciente) | `ContextGate` :60-61 | elegir el más nítido del tramo | marcar frames borrosos (Stella pierde features ahí) |
| salto de posición y de rotación entre frames | `evaluate_consistency.py:trajectory` (jerk, `rot_step_deg`) | métrica fuera de línea | detector de saltos de pose en vivo (hoy no se calcula en vivo) |
| frames "forzados" por `max_skip` | `stats["forced"]` | estadística | frames enviados sin movimiento suficiente |
| `frame_type` / `is_keyframe` | `inference_streaming` | memoria | ninguno |

**UNKNOWN:** si la magnitud del último delta de refinamiento de la cabeza de cámara
(`pred_pose_enc_delta`, `camera_head.py:133`) es informativa como convergencia. No se expone
hoy; medirlo requeriría un hook, no un cambio del núcleo.

### 2.4 Origen, escala y continuidad

- Verificado numéricamente: el frame 0 de la sesión tiene `R = I`, `t ≈ 0`. **El mundo es la
  cámara del primer frame de escala.** Cada sesión en vivo tiene su propio origen.
- La escala la fija implícitamente la fase de escala (`num_scale_frames=2` en vivo). En la
  sesión real: profundidad mediana 0.657, paso mediano entre frames 0.030, largo total 11.4
  (unidades del modelo). No hay metros.
- En windowed la escala además se re-estima por ventana con la mediana del cociente de
  profundidades (`_pairwise_alignment`, `gct_stream_window.py:757`; docs/MATEMATICA.md § 2.7): o
  sea, **dentro del propio LingBot la escala puede cambiar a lo largo de la secuencia**.
- Sesiones largas: `special_keep=64`, `camera_keep=1024`, `max_frame_num=16384` acotan la
  VRAM; pasado ~1024 frames el modelo opera fuera de lo que vio en entrenamiento
  (src/vivo/README.md § "Sesiones largas").

### 2.5 Baseline reproducible (REGLA 5)

`src/vivo/replay_live.py --path <frames> --captures_dir <out> [--context]` ejecuta la
misma clase `LiveSession` sin navegador y deja `eval/sesion.npz`. Las métricas existentes
sobre ese `.npz` son `compare_route.py` (forma del recorrido contra croquis),
`evaluate_consistency.py` (autoconsistencia, jerk, saltos de rotación), `height_profile.py`,
`tsdf_mesh.py --holdout_every`, `gsplat_train.py --test_every`. Resultados de referencia ya
registrados en la bitácora del README (2026-09-19, 2026-10-04).

---

## 3. Convenciones de coordenadas y pose (hallazgos)

### 3.1 LingBot / ParaLingbot

- Cámara **OpenCV**: x derecha, y abajo, z adelante. El `+Y` de la cámara llevado al mundo es
  "abajo"; lo usan `compare_route.py:131`, `main.js:358, 799` y `geo_filter.py` (§ 8.4).
- `pose_encoding_to_extri_intri` devuelve una 3x4 que el **docstring del upstream** llama
  "camera from world" (`pose_enc.py:98-100`). Pero `demo.postprocess` la **invierte** y llama al
  resultado `extrinsic` (`demo.py:303-316`), y todos los consumidores del repo vuelven a
  invertir `extrinsic` para obtener c2w (`export_dense_cloud.py:41`, `npz_to_webgl.py:40`,
  `tsdf_mesh.py:72`, `gsplat_train.py:112`, `point_cloud_viewer.py:224`, `main.js`). Como los
  mapas y las rutas salen bien (4.75% de error de forma contra croquis), la conclusión
  empírica, ya escrita en docs/MATEMATICA.md § 1, es:

  > `extrinsic` guardado = **w2c**; `c2w = inv(extrinsic)`. Usarlo al revés da trayectorias
  > sin sentido (bitácora 2026-09-17).

  Equivalentemente: lo que decodifica `pose_enc` **es c2w** en la práctica, y el comentario
  "Convert w2c to c2w" de `demo.py:305` y la variable `w2c` de `live_server.py:479` están
  nombrados al revés del significado efectivo. No es un bug (todo es consistente), pero es
  una trampa para quien conecte Stella: **no fiarse de los nombres, fiarse de docs/MATEMATICA.md § 1
  y verificar con movimiento conocido** (Etapa 8).
- Intrínsecos: predichos por frame (FoV), principal en el centro del recorte. El recorte
  (518 de ancho, centro vertical) **conserva el eje óptico**, así que el `camera_optical_frame`
  de LingBot y el de la imagen completa coinciden; solo cambia K.
- Rotación de 90° del celular: se aplica **antes** del recorte (`FrameSource._rotate`). Stella
  debe recibir la imagen ya rotada (o calibrarse para la rotada) para compartir el frame óptico.

### 3.2 Stella-VSLAM / stella_vslam_ros

- Internamente Stella trabaja en coordenadas **CV** (misma convención OpenCV, mundo = cámara
  inicial). `feed_monocular_frame` devuelve `cam_pose_wc` (cámara → mundo) **solo cuando el
  tracking tiene éxito** (`stella_vslam_ros.cc:237-250`): la ausencia de mensaje es la única
  señal de pérdida que publica el nodo. El estado interno `tracker_state_t`
  (Initializing / Tracking / Lost) **no se publica** en ningún topic. ⇒ hará falta un
  publicador de estado (cambio pequeño en `stella_vslam_ros`, o lectura del `frame_publisher`).
- Publica `~/camera_pose` (`nav_msgs/Odometry`, `frame_id = map`, `child_frame_id =
  camera_frame`, QoS `sensor_data`, profundidad 1) con la pose **convertida al mundo ROS**
  (x adelante, y izquierda, z arriba) mediante `rot_ros_to_cv_map_frame = [[0,0,1],[-1,0,0],[0,-1,0]]`
  aplicada a la izquierda y su inversa a la derecha (`:49-61`). Es decir: el mapa se rota a
  ROS **y** la pose de la cámara se expresa como `camera_frame` (no óptico).
- Publica `~/keyframes` y `~/keyframes_2d` (`PoseArray`, frame `map`) **en cada frame**, con
  las poses actuales de **todos** los keyframes (`publish_keyframes`, `:95-116`). Tras un loop
  closure esas poses vienen ya corregidas: **este es el canal natural de corrección histórica**.
- TF `map → odom` solo si `publish_tf` (el `config/param.yaml` lo deja en `false`) y requiere
  que exista `camera_optical_frame → odom` en TF (`:66-69`); si no, error. Para ParaLingbot sin
  robot, dejarlo apagado y publicar nosotros `map → camera_optical_frame` si hace falta.
- Suscribe `camera/image_raw` (relativo al namespace del nodo); `header.frame_id` del primer
  mensaje pasa a ser `camera_optical_frame_`; `header.stamp` es el timestamp que recibe el SLAM
  (`:231-235`). **La asociación temporal LingBot ↔ Stella será exacta si ambos consumen el mismo
  mensaje con el mismo stamp.**
- Necesita calibración fija (yaml con fx, fy, cx, cy, distorsión, fps, cols/rows) y un
  vocabulario ORB (`orb_vocab.fbow`). **UNKNOWN:** no existe calibración de la webcam ni del
  teléfono en el repo (`registros/calibration_logs/` son CSV de RAM, no cámaras). Punto de partida
  aceptable: K mediana que predice LingBot reescalada a la resolución completa, sin distorsión,
  y después una calibración real con patrón.
- Stella monocular: escala arbitraria fijada en la inicialización; el cierre de bucle
  monocular corrige en **Sim(3)** (puede cambiar la escala del mapa). ⇒ la escala relativa
  Stella/LingBot puede saltar en un loop closure.

### 3.3 GARDIAN (Pedros-Rescue), para no colisionar

El robot ya usa `/robot/camera/front/image_raw/compressed`, `/robot/camera/astra/color/image_raw/compressed`,
`/camera/color/image_raw`, `/camera/depth/points`, frames `base_link`, `camera_link`,
`camera_optical_link`, `map`/`odom` de `slam_toolbox` / RTAB-Map. ParaLingbot debe usar un
**namespace propio** (p. ej. `/paralingbot/...`, `/stella/...`) y nombres de frame propios
cuando corra en la misma red DDS; `map` de Stella **no** es el `map` del robot.

---

## 4. Geometría, registro y acumulación (lo que consume la pose)

Todos los lugares donde una pose registra geometría:

| Lugar | Cómo | Pose usada |
|---|---|---|
| `live_server.py:519-526` | desproyección por píxel con K y `c2w`, puntos al mundo del frame 0, enviados al navegador | `c2w = inv(W2C[j])` del mismo frame |
| `export_dense_cloud.py:unproject` / `merge` | desproyección por frame + fusión por vóxel (promedio por celda, vóxel relativo a la diagonal) | `c2w = inv(extrinsic)` |
| `npz_to_webgl.py` | ídem, modo raw o denso; escribe `*_cameras.json` con `c2w` e `intrinsic` por frame (el visor orienta **todos** los mapas con este archivo) | ídem |
| `tsdf_mesh.py:integrate` | `ScalableTSDFVolume.integrate(rgbd, K, inv(c2w))`; vóxel = mediana de profundidad / 150; `--refine`: ICP punto a punto **con escala** por frame contra la malla de consenso | `poses[i] = (c2w, escala)` |
| `gsplat_train.py` | inicialización de gaussianas desde profundidad + c2w; optimización con guía de profundidad L1; `--pose_lr` opcional (desactivado: empeoró) | `c2w` |
| `geo_filter.py` | consistencia multivista (proyecta entre frames vecinos y revisitas), planos, `--align` ICP punto-plano suavizado (opcional, apagado) | `c2w` / `w2c` |
| `src/upstream/demo_render/` (vía `npz_for_render.py`) | render RGB-D del recorrido | `extrinsic` |

Consecuencias para la integración:

- **La geometría nunca se "pega" a la pose en el `.npz`**: profundidad y K viven por frame y la
  pose es un array aparte. Cambiar poses y re-registrar es barato y no toca al modelo. Esto
  satisface REGLA 14 fuera de línea sin rediseñar nada.
- **TSDF y fusión por vóxel no se pueden des-integrar**; una corrección de poses obliga a
  re-integrar desde cero (segundos a minutos; aceptable fuera de línea, no por frame en vivo).
- **En vivo**, el cliente conserva `frame_idx` y `c2w` solo mientras procesa el mensaje; los
  puntos quedan en un buffer plano sin rangos por frame (`main.js:786-795`). Para corregir
  historia en vivo hay dos opciones: (a) guardar `[inicio, fin]` por frame en el cliente y
  aplicar `T_new · inv(T_old)` a ese rango al recibir un mensaje "repose"; (b) re-enviar.
  (a) es O(puntos del frame) y mantiene el protocolo (añadir un tipo de mensaje).
- Filtros existentes aprovechables (REGLA de Etapa 13): percentil de `depth_conf`, fusión por
  vóxel (`merge`), `depth_max_percentile`, clúster mínimo de triángulos, `geo_filter`
  (semántica, bordes, consistencia multivista, planos). `skyseg.onnx` solo como filtro de
  cielo (Etapa 15); no se usa en interiores hoy.

Métricas existentes: forma de ruta contra croquis (`error_pct`, `length_ratio`, rectitud,
giro total), autoconsistencia (`inlier`, `relerr`, `photo`, `overlap`), suavidad (`jerk`,
`rot_step_deg`), perfil de altura, coincidencia malla/frame (`inlier5`), PSNR/SSIM en frames
apartados, espesor de pared. Falta todo lo que sea **comparación entre dos trayectorias
con tiempo** (ATE/RPE tras Sim(3) de Umeyama): `compare_route.umeyama` ya implementa Umeyama
2D; hay que generalizarlo a 3D (o usar `evo`, perfil `bench`).

---

## 5. Timestamps, frecuencias y memoria (estado actual)

- **Timestamps:** no existen. `FrameSource.read()` devuelve solo la imagen; el hilo lector lleva
  un `fid` creciente (`live_server.py:137`) que no se propaga; `AndroidCamera.read()/latest()`
  tampoco devuelven hora. `ContextGate.feed` recibe solo `rgb` y puede devolver un frame
  **anterior** (el más nítido del tramo, `:76-79`); la hora de captura debe viajar con la imagen.
  La grabación usa índices (`source_index = arange`).
- **Frecuencias:** cámara 15 fps (webcam/celular); modelo ~2 fps; `ContextGate` descarta ~60%
  de los leídos en la sesión real (700 leídos, 286 enviados, 72 sintéticos). Stella monocular
  en CPU a 640x480 suele ir a 15-30 fps (**UNKNOWN en esta máquina: medir en Etapa 2**).
- **Hilos y GIL:** `LiveSession` corre en un hilo Python (`threading.Thread`), el servidor en
  asyncio, la vista previa en otro hilo. Añadir un `rclpy` spin en el mismo proceso compite
  por el GIL con el preprocesado (la inferencia libera el GIL dentro de CUDA). Es viable para
  publicar imágenes y recibir poses (cargas pequeñas), pero debe medirse (riesgo R6).
- **GPU:** 5.4-6.2 GB de los 8 GB durante el mapeo. Stella no usa GPU. `build_maps.py` y una
  sesión en vivo no pueden coexistir (el servidor ya lo impide: `live_start`/`jobs_start`).
- **RAM en vivo:** `rec[]` guarda por frame depth f16 + conf f16 + imagen uint8 a 294x518
  ≈ 1.1 MB/frame (≈1.1 GB por 1000 frames) hasta `_save_session`. Añadir poses/stamps es
  despreciable; añadir frames a resolución completa para Stella **no** (hacerlo por rosbag2,
  no en RAM).
- **Entorno Python:** no hay `.venv` en el repo en esta máquina; se usa `/usr/bin/python3`
  (3.10) con `torch 2.12.0+cu130`, `cv2 4.11` (pip), `open3d 0.19`, `aiohttp 3.14` y `rclpy`
  (Jazzy desde fuente, compilado para Python 3.10). El README (2026-09-15) advierte que el
  `PYTHONPATH` de ROS2 se cuela en cualquier venv; si se recrea el `.venv`, hay que mantener
  `rclpy` visible a propósito.

---

## 6. Estado de Stella-VSLAM y ROS2 en la máquina

| Componente | Estado | Acción |
|---|---|---|
| ROS 2 Jazzy | `~/ros2_jazzy/install` (fuente), auto-sourced en `~/.bashrc`; `ros2`, `rclpy`, `rclcpp_components`, `image_transport`, `camera_info_manager`, `tf2_ros`, `tf2_eigen`, `tf2_geometry_msgs`, `message_filters`, `nav_msgs`, `sensor_msgs`, `rosbag2` (sqlite3 + mcap) presentes; `rmw_cyclonedds_cpp` y `rmw_fastrtps_cpp` presentes | usar tal cual |
| `cv_bridge` / `vision_opencv` | **no está** en `~/ros2_jazzy/install` | compilar `vision_opencv` (rama jazzy) en el workspace de Stella |
| Driver de cámara ROS2 (`v4l2_camera`, `usb_cam`) | no están; solo `image_tools cam2image` (mínimo) | para webcam sirve `cam2image` en Etapa 2; para celular/URL/video hace falta nuestro publicador (es la razón del "tee" en `FrameSource`) |
| `stella_vslam` (librería) | **no instalada**, no hay fuentes clonadas | clonar `stella-cv/stella_vslam` (con submódulos), compilar e instalar en un prefix de usuario |
| g2o, FBoW | **no instalados** (ni apt ni `/usr/local`) | compilar (versiones indicadas por la doc de stella-cv) |
| Eigen 3.4, yaml-cpp 0.7, spdlog 1.9, sqlite3, SuiteSparse, OpenCV 4.5.4 (apt) | presentes | ok; **no mezclar** con el OpenCV 4.11 de pip (ver R5) |
| `stella_vslam_ros` | `~/Rescue/stella_vslam_ros` v0.2.1, **submódulo `3rd/filesystem` vacío** (`git submodule status` muestra `-`); requiere `cv_bridge`, `stella_vslam`, `rosbag2_cpp`; viewers opcionales (Pangolin / Iridescence / socket) no instalados → `--viewer none` | `git submodule update --init --recursive`; compilar en un workspace propio (p. ej. `~/Rescue/stella_ws`) sobre `~/ros2_jazzy` |
| Vocabulario ORB (`orb_vocab.fbow`) | no está | descargar (doc stella-cv) |
| Calibración de cámara | no existe | Etapa 2 (ver § 3.2) |

**No instalar nada durante la auditoría.** Todo lo anterior es Etapa 2.

---

## 7. STELLA INTEGRATION POINT

Stella se ejecuta **como proceso aparte** (`run_slam` de `stella_vslam_ros`, C++), fuera del
proceso de ParaLingbot, y se comunica solo por ROS2. Razones: no toca `lingbot_map`, no compite
por VRAM, su caída no detiene el mapeo (REGLA 15), y se puede apagar sin cambiar nada.

Entradas: `/paralingbot/camera/image_raw` (+ `camera_info`), **imagen completa rotada**, no
el recorte de 518. Salidas: `/stella/camera_pose` (Odometry), `/stella/keyframes` (PoseArray),
y un topic de estado a añadir (`/stella/tracking_state`: INITIALIZING / TRACKING / LOST +
número de keyframes + bandera de loop closure reciente). **UNKNOWN hasta compilar:** la forma
exacta de exponer el estado (API `system::get_frame_publisher()->get_tracking_state()` o
equivalente en la versión que se compile).

---

## 8. ROS2 INTEGRATION POINT

**Dónde:** `src/vivo/live_server.py`, en dos puntos, detrás de una bandera (`ros2: true`
en el POST de arranque / `--ros2` en `replay_live.py`), sin cambiar la ruta actual cuando está
apagada:

1. **Publicación de frames ("tee") en `FrameSource`**: en el hilo lector (`_grab`, :131-138) y en
   `AndroidCamera._decode` el frame recién capturado se marca con hora de captura y `fid`, y se
   publica como `sensor_msgs/Image` (`bgr8`) + `CameraInfo` con `header.stamp` = esa hora y
   `header.frame_id = paralingbot_camera_optical`. La cámara se abre **una sola vez** (webcam y
   celular no admiten dos lectores), así que publicar desde aquí es la única forma de que
   LingBot y Stella vean los mismos frames. Para `folder`/`video` se publica al leer, con
   stamps sintéticos a la cadencia nominal (reproducible).
2. **Suscripción de poses en `LiveSession`**: un nodo `rclpy` (hilo propio con su executor)
   recibe `/stella/camera_pose`, `/stella/keyframes` y el estado, y llena un `PoseBuffer`
   (Etapa 5) indexado por stamp. `run_model` consulta `pose_stella(t_frame)` para el stamp del
   frame que acaba de procesar y lo graba junto a la pose BASIC. En esta etapa **no cambia qué
   pose registra la geometría**: solo se graba y se compara (Etapa 4).

Alternativa descartada por ahora: convertir `live_server` en suscriptor puro de ROS2 (la
cámara publicada por otro nodo). Rompe el baseline (`FrameSource` dejaría de ser la fuente) y
duplica copias; se puede reconsiderar cuando el sistema corra en el robot, donde la cámara ya
es un nodo ROS2.

Topics propuestos (adaptables): `/paralingbot/camera/image_raw`, `/paralingbot/camera/camera_info`,
`/paralingbot/tracking/basic_pose` (PoseStamped, frame `paralingbot_map`),
`/stella/camera_pose`, `/stella/keyframes`, `/stella/tracking_state`,
`/paralingbot/tracking/hybrid_pose` (Etapa 6), `/paralingbot/map` (PointCloud2, Etapa 11+, opcional).
QoS: `sensor_data` (best effort, profundidad corta) para imágenes y poses en vivo; `reliable`
para keyframes y estado. Grabación reproducible: `rosbag2` (mcap) de imagen + poses.

---

## 9. LINGBOT INTEGRATION POINT

LingBot **no cambia**. La "pose de referencia" híbrida entra **después** del modelo, en
`run_model` (`live_server.py:510-529`) y en los scripts fuera de línea, como la `c2w` con la que
se desproyecta cada frame. Concretamente:

- En vivo: `c2w = pose_reference(frame)` en lugar de `inv(W2C)` cuando el modo sea STELLA o
  HYBRID, manteniendo `depth`, `K` y `depth_conf` del modelo. Se graban **ambas** poses.
- Fuera de línea: el `.npz` gana claves `stamps`, `frame_ids`, `pose_basic`, `pose_stella`,
  `pose_source`, `extrinsic` pasa a ser la pose de referencia elegida (para que
  `build_maps.py` y el visor funcionen sin cambios), y una copia `extrinsic_basic` conserva el
  baseline. Toda herramienta actual sigue funcionando.

No se intenta inyectar la pose de Stella **dentro** del modelo (REGLA 1): la cabeza de cámara y
la profundidad están acopladas por los tokens; forzar la pose no cambiaría la profundidad y
exigiría tocar `lingbot_map`.

---

## 10. COORDINATE FRAME PLAN

Frames (TF2) a definir y documentar en `docs/COORDINATE_FRAMES.md` (Etapa 8; **hecho el 2026-10-05**, ver ese documento: `map` está en ejes ROS y no CV, y la Sim(3) con Stella va por topic y no por TF):

| Frame | Convención | Origen | Quién lo publica |
|---|---|---|---|
| `paralingbot_camera_optical` | OpenCV (x der, y abajo, z adelante) | centro óptico de la cámara, imagen ya rotada | implícito (frame_id de las imágenes) |
| `paralingbot_camera_link` | ROS (x adelante, y izq, z arriba) = óptico · R_optical→link | mismo centro | estático, `tf2_ros static_transform_publisher` |
| `paralingbot_map` | **mundo BASIC**: cámara óptica del frame 0 de la sesión (OpenCV, +Y abajo) | frame 0 | ParaLingbot (`basic_pose` se expresa aquí) |
| `stella_map` | mundo de Stella **en convención ROS** (ya rotado por `rot_ros_to_cv_map_frame`) | cámara de inicialización de Stella | `stella_vslam_ros` (`map_frame` renombrado) |
| `odom` | no se usa sin robot; en el robot, el `odom` del robot | — | robot |

Transformaciones explícitas a estimar y publicar:

- `T(paralingbot_map ← stella_map)`: **Sim(3)** (Etapa 9), estimada por Umeyama 3D sobre
  pares `(pose_basic(t), pose_stella(t))` válidos, re-estimada por ventana. Hasta que se mida,
  no asumir SE(3).
- Conversión interna de la pose de Stella a óptico/OpenCV: deshacer la rotación del nodo
  (`rot_ros_to_cv_map_frame`) **o** leer la pose cruda `cam_pose_wc` añadiendo un publicador CV
  al nodo. Decidir en Etapa 2 según lo que resulte más verificable; documentar la elección.

Validación (Etapa 8): cámara estática (ambas poses constantes), traslación pura hacia
adelante (+z óptico en ambas tras conversión), rotación pura (centro fijo), circuito,
regreso al inicio.

---

## 11. SCALE PLAN

- Hipótesis de partida: **Sim(3)** entre `stella_map` y `paralingbot_map`, con escala `s(t)`
  potencialmente **no constante**: LingBot cambia de escala entre ventanas (windowed) y deriva
  en streaming; Stella puede re-escalar en loop closure.
- Estimación: Umeyama 3D con escala sobre una ventana deslizante de N pares válidos
  (ambos trackers en estado válido, movimiento suficiente según `motion_px_median`), con
  rechazo robusto (mediana de cocientes de distancias recorridas como inicialización, igual
  que `_depth_ratio_scale` usa mediana).
- Registrar siempre `s`, su varianza y su deriva temporal en `docs/TRACKING_BENCHMARK.md`.
  Si `s` resulta estable (< 2-3% de variación en secuencias sin loop closure), se puede fijar
  por sesión y tratar el resto como SE(3); si no, se mantiene Sim(3) por ventana.
- No ocultar la escala dentro de la pose de referencia: la geometría de LingBot se registra
  con la pose de referencia **y** una escala explícita `s_frame` (igual que `tsdf_mesh.py`
  ya hace con `(c2w, escala)` en `--refine`).
- Métrica: para los recorridos con croquis, `length_ratio` de `compare_route.py` separa el
  problema de forma del de escala.

---

## 12. TIMESTAMP PLAN

1. Hora de captura en la fuente (`time.time()` en el hilo lector / en `_decode`), convertida a
   `builtin_interfaces/Time` al publicar; mismo stamp en la imagen ROS2 y en el frame que
   sigue hacia LingBot. Para carpeta/video: stamps sintéticos `t0 + i/fps_nominal`.
2. `FrameSource.read()` devuelve `(img, stamp, fid)`; `ContextGate.feed` recibe y devuelve
   `(rgb, stamp, fid, sintético)`; los sintéticos llevan stamp interpolado y `fid = -1` y **no
   se registran** (como hoy).
3. `rec` y el `.npz` ganan `stamps` (float64, s) y `frame_ids`; el mensaje binario del
   WebSocket gana el stamp (ampliar la cabecera o un campo nuevo; mantener compatibilidad).
4. `PoseBuffer` (Etapa 5): muestras `(stamp, pose, estado, confianza, fuente)`; consulta
   `pose(t)` con búsqueda binaria; si no hay muestra en `t ± ε`, interpolación **lineal en
   traslación y SLERP en rotación** entre vecinos (REGLA de § 8 del prompt); si la brecha
   supera un umbral (p. ej. 0.3 s) o el estado es LOST, devuelve `None` (no inventar).
5. Nunca usar `now()` como stamp de un dato histórico. Las poses de Stella llevan el stamp
   de la imagen que las produjo (`stella_vslam_ros.cc:235, 246`), así que la asociación es
   exacta cuando Stella procesó ese frame; solo se interpola cuando lo descartó (QoS
   profundidad 1).
6. Reloj: todos los procesos en la misma máquina y reloj del sistema; en el robot, revisar
   sincronización (fuera de alcance ahora).

---

## 13. LOOP CLOSURE PLAN

1. **Asociación** por frame registrado: `frame_id`, `stamp`, keyframe de Stella más cercano
   en el tiempo (`kf_id`, con `T_kf→cam` relativa calculada en el momento del registro) y el
   rango de puntos emitido (en vivo) / el índice en el `.npz` (fuera de línea).
2. **Detección**: comparar `/stella/keyframes` entre mensajes consecutivos; si la pose de
   algún keyframe cambia más que un umbral (o el estado anuncia loop closure), hay corrección.
3. **Propagación**: para cada frame asociado a un keyframe corregido,
   `pose_new = T_map←stella · T_kf_new · T_kf→cam`; la geometría del frame se vuelve a
   registrar con `pose_new` (en vivo: aplicar `pose_new · inv(pose_old)` a su rango de
   puntos mediante un mensaje "repose"; fuera de línea: reescribir `extrinsic` y re-correr
   `build_maps.py`, que ya re-integra TSDF y re-entrena el splat desde cero).
4. **Escala**: si el loop closure cambió la escala de Stella, re-estimar `T_map←stella`
   (Sim(3)) antes de propagar.
5. **Verificación**: Etapa 12 con el TEST 10 (loop closure) y TEST 9 (revisita): medir
   duplicación de superficie (`wall_thickness.py`, `inlier5` de la malla) antes y después.

---

## 14. RISKS — riesgos

| # | Riesgo | Evidencia | Mitigación |
|---|---|---|---|
| R1 | Sin timestamps no hay asociación; añadirlos atraviesa `FrameSource`, `ContextGate`, `run_model`, `rec`, protocolo WS y `.npz` | § 5 | hacerlo primero (Etapa 1), de forma aditiva; probar con `replay_live.py` que el `.npz` sigue siendo consumible por `build_maps.py` |
| R2 | Confusión de convención de pose (nombres invertidos respecto al significado efectivo) | § 3.1 | única fuente de verdad: docs/MATEMATICA.md § 1; test de movimiento conocido en Etapa 8 antes de cualquier fusión |
| R3 | Escala no constante en ambos lados | § 2.4, § 3.2 | Sim(3) por ventana, escala explícita, medir antes de fijar |
| R4 | Stella sin estado publicado → "perdido" solo se infiere por silencio | § 3.2 | añadir publicador de estado en `stella_vslam_ros`; mientras tanto, timeout en `PoseBuffer` |
| R5 | Dos OpenCV: apt 4.5.4 (stella, cv_bridge C++) y pip 4.11 (Python). `cv_bridge` Python en el mismo proceso que `cv2` de pip puede fallar por ABI | § 5 | no usar `cv_bridge` en Python: construir `sensor_msgs/Image` a mano con numpy (trivial para `bgr8`) |
| R6 | GIL / latencia: `rclpy` dentro de `live_server` compite con preprocesado y vista previa | § 5 | executor en hilo propio, QoS best effort, medir fps del modelo con y sin ROS2; si degrada, mover la publicación a un subproceso que reciba frames por memoria compartida |
| R7 | Compilar Stella y dependencias (g2o, FBoW, cv_bridge) en una máquina compartida con GARDIAN | § 6 | prefix de usuario (`~/.local` o `~/Rescue/stella_ws/install`), nunca `sudo` ni tocar `~/ros2_jazzy`; documentar en `docs/STELLA_INTEGRATION.md` |
| R8 | Sin calibración de cámara, Stella puede no inicializar o derivar | § 3.2 | calibración con patrón en Etapa 2; mientras tanto K de LingBot reescalada, sin distorsión, y registrar la diferencia |
| R9 | Las 13 sesiones en vivo existentes guardan solo el recorte 518 sin stamps: no sirven para Stella | § 1.4 | grabar nuevas secuencias con rosbag2 (imagen completa) + `.npz` baseline; las antiguas quedan como baseline histórico |
| R10 | El visor en vivo no puede corregir historia | § 4 | rangos por frame + mensaje "repose" (Etapa 12/16) |
| R11 | VRAM: 5.4-6.2 GB de 8 GB ya ocupados; Stella no usa GPU, pero `build_maps` sí | § 5 | mantener la exclusión mutua existente; Stella en CPU |
| R12 | Colisión de topics/frames con GARDIAN en la misma red DDS | § 3.3 | namespaces `/paralingbot`, `/stella`; `ROS_DOMAIN_ID` propio en pruebas de escritorio |
| R13 | `ContextGate` descarta ~60% de frames y puede elegir uno anterior: la pose de Stella debe buscarse por el stamp del frame **elegido**, no del último leído | § 5 | stamps viajan con la imagen (R1) |
| R14 | Síntesis de frames (`context_synth`) alimenta al modelo con imágenes inventadas; no deben llegar a Stella | REGLA 13 | el "tee" publica solo frames capturados; los sintéticos nunca pasan por ROS2 |
| R15 | Sesiones > 1024 frames fuera de distribución del modelo | § 2.4 | fuera de alcance de Stella; registrar en el benchmark |

---

## 15. IMPLEMENTATION PHASES — fases, archivos y criterios

Las etapas siguen el prompt maestro; aquí se concretan en archivos del repo. Todo nuevo vive
en `src/vivo/`, `src/mapas/`, `src/ros/` (nueva) y `docs/`; nada en `demo.py`
ni `lingbot_map/`.

| Etapa | Qué | Archivos | Criterio de éxito | Rollback |
|---|---|---|---|---|
| 0 | esta auditoría | `docs/STELLA_INTEGRATION_AUDIT.md` | documento revisado | — |
| 1 | stamps + `frame_id` de punta a punta; `BasicTrackingProvider` = envoltorio de lo que `run_model` ya calcula (stamp, c2w, conf media de frame, motion_px, nitidez) → `TrackingEstimate(source=BASIC)`; grabar `stamps`, `frame_ids`, `pose_basic` en el `.npz`; stamp en el mensaje WS | `src/vivo/live_server.py`, `context_gate.py`, `android_camera.py`, nuevo `src/vivo/tracking.py` | `replay_live.py` da el mismo `error_pct` que antes en `unisabana/prueba_3` (baseline intacto); `build_maps.py` consume el `.npz` nuevo sin cambios | bandera; claves nuevas son aditivas |
| 2 | Stella aislado: clonar/compilar `stella_vslam` + deps + `vision_opencv` + `stella_vslam_ros` en `~/Rescue/stella_ws`; vocabulario; calibración; `cam2image` o publicador propio; medir fps, latencia, pérdida, relocalización, keyframes, loop closure en un recorrido conocido | `docs/STELLA_INTEGRATION.md`, `src/ros/stella/*.yaml`, `src/ros/launch/` | Stella trackea la webcam y el celular a ≥ 10 fps y publica `camera_pose`/`keyframes`; estado publicado | no toca ParaLingbot |
| 3 | `TrackingEstimate` común + adaptador Stella (suscriptor rclpy → estimate en frame óptico/OpenCV) | `src/vivo/tracking.py`, `src/ros/stella_bridge.py` | dos listas de estimates con stamps sobre la misma secuencia | bandera |
| 4 | comparación: grabar ambas poses por frame, Umeyama 3D Sim(3), ATE/RPE, pérdida/recuperación, tabla por secuencia | `src/mapas/compare_tracking.py`, `docs/TRACKING_BENCHMARK.md` | tabla A/B/C sobre ≥ 3 secuencias (estática, lineal, circuito) | — |
| 5 | `PoseBuffer` (lineal + SLERP, timeout, estados) | `src/vivo/pose_buffer.py` + tests | `pose(t)` exacto cuando hay muestra; `None` en brechas; test unitario | — |
| 6 | modos BASIC / STELLA / HYBRID (selección por estados reales) y registro con la pose de referencia en `run_model` | `live_server.py`, `tracking.py` | el mapa en vivo se construye con cualquiera de los tres modos; sin Stella → BASIC automáticamente | modo BASIC |
| 7 | fusión avanzada solo si Etapa 4 lo justifica | — | evidencia escrita | — |
| 8-9 | frames TF2 + validación de ejes; escala Sim(3) vs SE(3) medida | `docs/COORDINATE_FRAMES.md`, `src/ros/static_tf.launch.py` | pruebas de movimiento conocido pasan; informe de escala | — |
| 10-11 | integración con LingBot (pose de referencia + escala explícita) y registro progresivo (lineal → habitación → circuito) | `live_server.py`, `build_maps.py` (modo `--poses hybrid`) | menor `error_pct` / menor duplicación que BASIC en al menos una secuencia, medido | `extrinsic_basic` |
| 12 | loop closure: asociación frame↔keyframe, "repose" en vivo, re-registro fuera de línea | `live_server.py`, `main.js`, `build_maps.py` | TEST 10: duplicación de pared baja tras el cierre | desactivar corrección |
| 13-15 | acumulación con filtros existentes; TSDF/GSplat desde el mapa registrado; skyseg solo como filtro | reutilizar `export_dense_cloud.py`, `tsdf_mesh.py`, `gsplat_train.py`, `geo_filter.py` | mismas métricas de malla/splat que hoy, comparadas | — |
| 16 | visor: trayectorias BASIC/STELLA/HYBRID, estado, keyframes, loop closures, fuente actual, fps/latencia | `main.js`, `index.html`, `live_server.py` | se ve el cambio de fuente | ocultable |
| 17-18 | benchmark A-E y matriz de 15 pruebas (rosbag2 + `.npz`) | `docs/TRACKING_BENCHMARK.md`, `src/mapas/run_tracking_bench.py` | tabla completa | — |
| 19 | optimización (QoS, copias, buffers) | — | solo con mediciones | — |

Datos disponibles para empezar: `captures/pruebas_reales/unisabana/prueba_{2,3,4}` (videos
con croquis; sirven para A/B/C fuera de línea publicando el video por ROS2 a su cadencia) y
13 sesiones de celular en `captures/streaming/sin_guardar/` (solo baseline histórico, R9).

---

## 16. Lista de UNKNOWN (a resolver por prueba, no por suposición)

1. FPS y latencia reales de Stella monocular en esta CPU con 1280x720 y 640x480. **Resuelto parcialmente (etapa 2):** a 540x960, 13-20 ms por frame (mediana), p90 ≤ 30 ms; falta en vivo.
2. API exacta para publicar el estado de tracking desde `stella_vslam_ros` en la versión que
   se compile (nombre de `tracker_state_t` / `frame_publisher`). **Resuelto (etapa 2):** `slam_->get_frame_publisher()->get_tracking_state()` devuelve `Initializing` | `Tracking` | `Lost`; publicado en `~/tracking_state`.
3. Si la escala relativa Stella/LingBot es estable en una sesión sin loop closure. **Resuelto (etapa 4): no.** Varía un 12-21% en ventanas de 5 s con Stella aislada y llega a 4.7× con Stella sin CPU suficiente. Hace falta Sim(3) por ventana.
4. Si el delta de refinamiento de la cabeza de cámara sirve como confianza de pose.
5. Coste en fps del modelo al añadir `rclpy` al proceso del servidor (R6). **Medido (etapas 3-6):** el modelo va a 1.79-1.80 frames/s con el puente frente a 1.98-2.0 sin él, y a 1.71-1.75 fijándolo a 12 CPUs para dejar 8 a Stella.
6. Calibración de la webcam `/dev/video0` y del HONOR WDY-LX3 (y si la distorsión importa).
7. Si Stella inicializa con la imagen rotada 90° del celular a 15 fps caminando (paralaje).
8. Comportamiento de `cv_bridge` compilado contra OpenCV 4.5.4 junto a `torch` en el mismo
   proceso (se evita no usándolo en Python, pero el nodo C++ lo necesita).
