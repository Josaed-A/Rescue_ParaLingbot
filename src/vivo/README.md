# Mapeo en vivo (streaming) + visor, en un solo servidor

`live_server.py` es **el único servidor** del visor. Sirve las dos cosas en el mismo
puerto, y el modelo se carga **solo** cuando se inicia una sesión en vivo:

| Ruta | Qué hace | ¿Toca la GPU? |
|---|---|---|
| `GET /` | el visor (`src/mapas/webgl_viewer/`) | no |
| `GET /api/captures` | qué pruebas hay exportadas (selector) | no |
| `GET /data/<archivo>` | sirve `.ply`/`.glb`/`.json` de `captures/` (con HTTP Range) | no |
| `WS /ws` | canal en vivo: un mensaje binario por frame reconstruido | — |
| `POST /api/live/start` | **carga el modelo** y arranca el mapeo | sí |
| `POST /api/live/stop` | lo detiene y libera la VRAM | libera |
| `GET /api/explorer` | catálogo del explorador: pruebas, mapas por tipo, zona, categorías | no |
| `GET /api/files?prueba=&dir=` | contenido de una carpeta de una prueba (perezoso) | no |
| `POST /api/prueba/meta` | editar título, zona, categorías y notas (`info.json` de la prueba) | no |
| `POST /api/prueba/save` \| `/discard` | guardar \| descartar una sesión en vivo sin guardar | no |
| `POST /api/jobs/start` \| `/cancel`, `GET /api/jobs` | construir mapas de una prueba (`build_maps.py`), uno a la vez | sí, según la tarea |

```bash
src/mapas/webgl_viewer/launch.py            # http://localhost:8090
# o directo:
python3 src/vivo/live_server.py --port 8090
```

Después, en la página, panel **"Mapeo en vivo"**: elegir fuente, `Iniciar`. El mapa
crece frame a frame mientras se navega (órbita o primera persona, con mouse o control).

## Las fuentes

| Fuente | Para qué | Campo `path` |
|---|---|---|
| **Carpeta de frames** | repetir una prueba ya grabada como si llegara en vivo — es la forma reproducible de probar el streaming | carpeta con `000000.png`... |
| **Cámara en vivo (webcam, celular o equipo por SSH)** | cámara real: una webcam de este equipo, la cámara de un teléfono Android conectado por adb o la de otro equipo por SSH (robot, vigia-1) | se elige en un desplegable: `/api/live/devices` lista los `/dev/videoN` (y **prueba cuál entrega imagen**: varios nodos existen pero no sirven) y, en `remote`, una entrada por cámara de cada teléfono que aparece en `adb devices` |
| **Cámara IP (URL)** | apps tipo IP Webcam, cámaras RTSP, streams MJPEG | `rtsp://...` o `http://...` (lo que abra `cv2.VideoCapture`) |
| **Archivo de video** | un `.mp4` reproducido frame a frame | ruta al video |

### Cámara de un celular (`android_camera.py`)

No hace falta instalar nada en el teléfono (Android 12 o más nuevo, con depuración por adb,
por USB o por Wi-Fi). Usa el `scrcpy-server` del repo hermano `cel-en-rescue`
(`datos/scrcpy/scrcpy-server`, o el que indique `SCRCPY_SERVER`):

1. lo sube al teléfono con un nombre propio (`/data/local/tmp/lingbot-scrcpy-server.jar`, así no pisa el de scrcpy);
2. lo arranca en modo cámara con `raw_stream=true` y abre un túnel `adb forward`;
3. ffmpeg decodifica el H.264 a frames BGR, y un hilo se queda siempre con el último (el modelo va a ~2 frames/s y la cámara a 15).

Al terminar la sesión cierra ffmpeg, el servidor del teléfono y el túnel; no quedan procesos.

- **adb:** usa el de `ADB`, el del sistema o el portable de `cel-en-rescue/datos/platform-tools`. Conviene que sea el mismo binario que ya tiene el servidor adb corriendo: uno de otra versión lo reinicia y corta otras sesiones (por ejemplo un scrcpy abierto en otra terminal). Nunca hace `adb kill-server`.
- **La cámara la usa una sola app a la vez.** Si hay un scrcpy de cámara abierto, hay que cerrarlo antes.
- **Rotación:** el teléfono entrega la imagen en la orientación del sensor, que es horizontal, sin importar cómo se lo sostenga. El modelo necesita la imagen derecha (gravedad hacia abajo): con el **celular vertical, 90°**; horizontal, 0°. El panel pone 90° al elegir un teléfono. No se puede leer la orientación del teléfono por adb de forma confiable (con la pantalla apagada Android reporta siempre rotación 0), así que se elige a mano.
- **Resolución:** 1280x720 a 15 fps y 4 Mbit/s por defecto (el modelo recorta a 518 de ancho igual). 1920x1080 pide más Wi-Fi sin ganar detalle en el mapa.
- `info.json` de la sesión guarda modelo del teléfono, cámara, rotación y resolución, con la categoría "celular", pero **no el serial ni la IP**.

Probar la cámara suelta, sin el modelo: `python3 src/vivo/android_camera.py [serial] [cámara]`
(escribe `cel_prueba.jpg`).

Parámetros del panel: `fps` (0 = tan rápido como pueda el modelo), `máx. frames`
(0 = sin límite; cuenta frames leídos de la fuente, antes del analizador de contexto), `puntos/frame` (cuántos puntos se mandan al navegador por frame) y
`conf. (percentil)` (descarta el X% de píxeles menos confiables).

### Todas las cámaras de este equipo (`video_devices.py`, 2026-10-05)

La lista sale de `/sys/class/video4linux`. Cada cámara aparece por su nombre (el del driver, o el de `lsusb` si el driver da uno genérico como "UVC Camera (046d:0823)") y se omiten los nodos de metadatos que agrega cada cámara UVC. Se prueban en paralelo y con reintentos durante ~2.5 s. La prueba anterior hacía una sola lectura y daba por inservible a la Logitech B910, que tarda ~1 s y entrega un primer JPEG corrupto. Las cámaras que no dan imagen se listan igual, con el motivo. La que está usando la sesión en vivo no se vuelve a abrir.

### Puente ROS2 + Stella desde el lanzador general

`launch.py` arranca el servidor con `--ros2`: cada sesión en vivo publica por ROS2 (imagen, poses, TF) y arranca Stella sin pasos extra. Se apaga con `--no-ros2` o, por sesión, con la casilla "puente ROS2 + Stella" del panel, que trae al lado el modo de referencia (HYBRID por defecto; el mapa se dibuja siempre con BASIC). Si el Python del lanzador no ve `rclpy`, arranca el servidor con `~/ros2_jazzy/install/setup.bash`.

La configuración de cámara de Stella se elige sola (`stella_config_for`). Primero se calcula el tamaño de la imagen publicada, según la fuente, la rotación y el reescalado; las fuentes de más de 960 px de lado se publican reescaladas. Si en `src/ros/stella/` hay un yaml de ese tamaño para esa fuente, se usa ese. Si no, se genera uno en `~/.cache/paralingbot/stella/` con intrínsecos **estimados** (campo de visión horizontal de 70°, sin calibrar). La nota sale en el estado de la sesión (`stella_config_nota`). Stella va en los núcleos 4-11 si el equipo tiene 16 o más.

### Cámara de otro equipo por SSH (`ssh_camera.py`, 2026-10-05)

El mismo método con que GARDIAN mira la cámara de vigia-1 (`~/Rescue/vigia_cam/vigia_cam_view.py`): en el equipo remoto ffmpeg entrega MJPEG por la salida estándar, llega por SSH y aquí se decodifica. Sirve para la Pi del robot (`gardian`), vigia-1 o cualquier Linux con una cámara.

- **Descubrimiento:** al buscar cámaras, se prueban en paralelo los hosts de `~/.ssh/config` y los equipos recordados (`ssh -o BatchMode=yes`, sólo llaves, nunca pide contraseña; 8 s como máximo). En cada equipo se listan los nodos de `/sys/class/video4linux` y se descartan los que no son cámaras (códecs e ISP de la Raspberry Pi, metadatos).
- **Equipos recordados:** cada equipo donde se encontró una cámara queda en `~/.config/paralingbot/camaras_ssh.json` y aparece la próxima vez aunque esté apagado ("sin conexión, visto ..."). En ese archivo también se pueden agregar equipos que no están en `~/.ssh/config` (`equipos_extra`) o esconder alguno (`ignorar`).
- **Tipos de cámara:** UVC con MJPEG → ffmpeg copia el MJPEG sin recomprimir (casi sin CPU en la Pi); sólo YUYV → lo comprime a MJPEG; la MIPI de vigia-1 (unicam, IMX296 mono) → `y10cap` (se sube y compila si falta, fuente en `~/Rescue/vigia_cam/y10cap.c`), como en `vigia_cam_view.py`.
- **Requisitos en el equipo remoto:** acceso por llave, `ffmpeg`, y que la cámara no la tenga abierta otro programa: si la estación de GARDIAN está corriendo con los nodos de cámara del robot, V4L2 la da por ocupada y el error lo dice.
- Al cerrar se termina la captura remota por su PID (el comando remoto lo informa al empezar).

Probar sin el modelo: `python3 src/vivo/ssh_camera.py --listar` y `python3 src/vivo/ssh_camera.py --host gardian --device /dev/video0`. Con `--local` el mismo comando corre en este equipo, sin SSH (así se probó: webcam a 15 fps en MJPEG 1280x720 y en YUYV 640x480).

## Rendimiento medido en esta máquina (RTX 2000 Ada, 8 GB)

| | |
|---|---|
| Velocidad | **2.1-2.4 frames/s** repitiendo frames de 1080x1920; **3.9 frames/s** con la webcam (640x480); **1.7-2.2 frames/s** con la cámara de un celular por Wi-Fi (1280x720, analizador de contexto activado) |
| VRAM en marcha | 5.4-6.2 GB (dentro del presupuesto de 8 GB) |
| VRAM al detener | vuelve a ~170 MB (solo el contexto CUDA) |
| Carga del modelo | ~8 s, una sola vez por sesión |
| Datos por frame | 60 KB con 4000 puntos/frame |

## Vista del video real mientras se mapea

Además de la nube, el servidor manda el frame que entró al modelo como JPEG, para
poder comparar lo que ve la cámara contra lo que se está reconstruyendo. Aparece como
una ventana en una esquina y el botón **"pantalla completa"** intercambia cuál de las
dos ocupa la pantalla: video grande con el render en la esquina, o al revés.

Va en el mismo canal binario, marcado con `frame_idx = 0xFFFFFFFF` (un índice que
ningún frame real usa), así no hizo falta cambiar el protocolo de los puntos. Se puede
apagar con `"preview": false` en el POST de arranque.

### Fluidez: el video va a su propio ritmo

Con una cámara en vivo (webcam, celular o URL), la vista del video **no** espera al modelo:
un hilo manda el último frame de la cámara a `preview_fps` (15) y 640 px de ancho. Si el
navegador no alcanza, el servidor descarta imágenes en vez de encolarlas. En el mapa, una
pirámide magenta marca la cámara actual y se desliza hacia cada pose nueva; con **"seguir la
cámara"** (activado por defecto) la vista la acompaña desde atrás y arriba. Arrastrar la vista
lo desactiva. El mapa en sí crece al ritmo del modelo (~2 frames/s).

## Guardar y reconstruir una sesión en vivo

Cada sesión graba lo que predijo el modelo para cada frame (profundidad, confianza, pose,
intrínsecos y la imagen que entró) y, al terminar, la deja en
`captures/streaming/sin_guardar/sesion_<fecha>/` con el mismo formato que
`process_and_view.py --save_predictions` (`eval/sesion.npz`), más los frames en `frames/`.
O sea que una sesión en vivo es una prueba más y le sirven todas las herramientas del repo.

En el explorador aparece con la categoría "sin guardar":

1. **💾 guardar sesión:** título, zona, categorías, notas y carpeta de destino (por
   defecto `streaming/...`); se mueve a `<destino>/prueba_N`. También se puede descartar.
2. **⚙ construir mapas:** elegir tareas de `src/mapas/build_maps.py`. Para una sesión
   en vivo conviene marcar **windowed**: reprocesa los frames guardados en modo windowed
   (ventana 24), que corrige la deriva del streaming, y el resto de los mapas sale de ahí.

El trabajo corre en segundo plano, **en su propio cgroup con tope de RAM**
(`src/gpu/run_isolated.sh`): si un mapa agota la memoria muere ese trabajo, no VS Code
ni el escritorio. El servidor mismo, lanzado con `launch.py`, también corre aislado. Su progreso llega por el mismo WebSocket (barra al pie
del explorador). Mientras corre no se puede iniciar el mapeo en vivo, y al revés: comparten
la GPU. Verificado con una sesión de 30 frames: guardado automático, guardado con nombre y
las cinco tareas (windowed, nube, cruda, malla, splat) en 12.5 min.

`"record": false` en el POST de arranque desactiva la grabación. La grabación vive en RAM
hasta que termina la sesión (~1.9 MB por frame a 518x518).

## Lo que hay que saber antes de la prueba

**En vivo sólo existe el modo streaming, y el streaming deriva.** El modo `windowed`
—el que arregla la deriva y da los mejores mapas (ver la bitácora del 2026-09-19)—
necesita tener toda la secuencia por adelantado: procesa ventanas solapadas con
atención bidireccional. En vivo no hay futuro que mirar, así que el mapa en vivo tiene
la misma deriva del baseline en streaming (recorre ~2.6x el largo real en la muestra 1).
**Para el mapa bueno hay que reprocesar la grabación después, en `windowed`.** El vivo
sirve para ver la cobertura mientras se camina, no para el mapa final.

**A ~2.2 frames/s**, una caminata normal deja huecos si se avanza rápido: conviene
caminar despacio, o capturar con `capture_frames.py` y reproducir la carpeta después.

## Analizador de contexto (context-to-image en vivo)

`context_gate.py`, activado con la casilla **analizador de contexto** del panel
(`"context": true` en el POST de arranque). Decide frame a frame qué le llega al modelo:

- mide el movimiento entre frames con flujo óptico DIS (CPU, ~12 ms por frame contando el
  recorte) y la nitidez con la varianza del laplaciano;
- se salta los frames casi iguales al anterior y, cuando el movimiento acumulado llega a
  `context_step_px` (36 px por defecto), envía el más nítido del tramo;
- con **sintetizar frames intermedios** (`"context_synth": true`), en saltos grandes
  genera frames intermedios por flujo bidireccional. Entran al modelo **solo como
  contexto** (caché KV): no se dibujan ni se graban. El selector x1/x2/x3
  (`"context_synth_strength"`) baja el umbral y agrega más intermedios por salto. En vivo,
  x2 y x3 empeoran mucho el recorrido porque la caché se llena de frames inventados
  (bitácora del 2026-10-04, tarde). Por defecto, x1.

Con el analizador el ritmo lo pone el movimiento de la cámara, no el campo `fps`. Las
estadísticas (enviados / leídos / sintéticos) aparecen en el panel y en el `info.json` de
la sesión guardada. Qué tanto ayuda está medido en la bitácora del 2026-10-04.

El mismo analizador sirve fuera de línea: `python3 src/vivo/context_gate.py
--frames_dir <todos los frames> --out_dir <salida> [--synth]` deja los frames elegidos y un
`manifest.json` para `process_and_view.py --manifest`. Para medir una sesión en vivo sin
navegador está `replay_live.py`, que usa exactamente el mismo código que el servidor.

## Momentos estáticos: la cámara no avanza, el mapeo sigue (2026-10-05)

En streaming el modelo deriva aunque la cámara esté quieta. En una sesión con el celular tapado, la pose "avanzó" 3.8 unidades en 202 frames sin movimiento, y la cámara virtual seguía moviéndose. `StaticHold` (`context_gate.py`) usa la **misma regla de movimiento que el analizador de contexto**: la mediana del flujo óptico DIS, acumulada desde el frame anterior del modelo. Si el analizador está activo, el valor es `meta.motion_px`; si no, se mide igual entre frames consecutivos del modelo.

- **Momento estático:** un frame con movimiento menor que `static_frac × context_step_px` (0.25 × 36 = 9 px). Con la cámara quieta se miden ~0.2-1 px; caminando, el mínimo medido fue de ~34 px.
- **Qué se retiene:** en un momento estático se repite la pose anterior. Eso alcanza a la cámara virtual del visor (marcador y "seguir la cámara"), a la pose BASIC que va al selector y a ROS2, y a la pose con que se registran los puntos.
- **Qué sigue igual:** el modelo sigue procesando y **los puntos del frame se agregan a la nube** con la pose retenida.
- **Al reanudar:** la deriva acumulada durante la pausa queda en una corrección (`pose mostrada = fix · pose del modelo`) que se mantiene, así que no hay salto.
- **Qué se graba:** el `.npz` guarda `pose_model` (sin retener) y `estatico` por frame; `info.json` lleva `tracking.momentos_estaticos`. El panel muestra "cámara: quieta (x px) — la pose no avanza" o "en movimiento".
- **Cómo se apaga:** `static_hold: false` en el POST, o `replay_live.py --no_static_hold`, que devuelve el comportamiento anterior.

Medido: con la Logitech quieta 70 s, el modelo derivó 0.25 unidades y la pose mostrada se movió 0.0, con los 135 frames mapeados. En el fablab caminando no hubo ningún falso estático, y el recorrido sale idéntico con y sin retención (2.23 % contra el croquis).

## Sesiones largas: memoria y límite de frames

Dos límites del modelo que no se veían con sesiones cortas (encontrados el 2026-10-04):

- **Tokens especiales que crecen sin límite.** Cuando un frame sale de la ventana de la
  caché, el modelo conserva sus tokens de cámara, registro y escala y los concatena para
  siempre: la VRAM crecía 1.1 MB por frame. El servidor los recorta a los de los últimos
  64 frames desalojados (`special_keep`) y el crecimiento baja a 0.25 MB por frame.
- **Caché de la cabeza de cámara que crece sin límite.** Guarda un token de pose por frame
  y el desalojo del modelo solo actúa con más de un token por frame: 0.25 MB por frame, que
  cortaba las sesiones de ~1900 pasadas. El servidor deja los frames de escala más los
  últimos 1024 (`camera_keep`); las sesiones más cortas no cambian. Con los dos recortes la
  VRAM queda plana (medido, 400 frames).
- **Tabla de posiciones de 1024 frames.** La posición temporal del RoPE 3D está
  precalculada para `max_frame_num` frames (1024 en `demo.py`). Pasado ese número, la parte
  temporal de la codificación queda vacía. El servidor construye el modelo con 16384.
  Ojo: el modelo nunca vio distancias temporales tan largas entre los frames de escala y el
  frame actual, así que una sesión muy larga puede degradarse.

`"keyframe_interval": N` hace que solo uno de cada N frames quede en la caché (los otros
la consultan sin quedarse), como en `inference_streaming`.

## Protocolo del canal (por si hace falta otro cliente)

Mensajes de texto: JSON `{"type": "status"|"started"|"stopped"|"error"|"done"|"tracking", ...}`.
Mensajes binarios: un frame, cabecera de 72 bytes little-endian y después los datos.

```
uint32       frame_idx
uint32       n_points
float32[16]  c2w  (4x4 row-major, cámara -> mundo)
float32[n*3] xyz en coordenadas de mundo
uint8 [n*3]  rgb
```

Por cada frame real sale además un mensaje de texto `{"type": "tracking", "frame_idx", "stamp",
"frame_id", "source": "BASIC", "status": "TRACKING", "position", "conf_mean", "conf_p50",
"motion_px", "sharpness", "step"}` con la estimación del tracking actual (la pose de la cabeza
de cámara del modelo) y su hora de captura: `stamp` es `time.time()` al recibir el frame en
cámaras en vivo y `índice / fps nominal` (desde 0) en carpetas y videos. Es la base de la
integración con Stella-VSLAM (`docs/STELLA_INTEGRATION.md`); el visor ignora los tipos que no
conoce. El `.npz` de la sesión guarda lo mismo en `stamps`, `frame_ids`, `pose_basic` (c2w),
`track_conf_basic`, `track_motion_px`, `track_sharpness`, más `extrinsic_basic` y `pose_source`.

`src/vivo/` sólo tiene este archivo: el cliente vive en
`src/mapas/webgl_viewer/main.js` (misma escena three.js que el visor normal, así
no hay dos visores que mantener).
