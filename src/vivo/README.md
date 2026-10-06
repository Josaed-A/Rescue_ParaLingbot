# Mapeo en vivo (streaming) + visor, en un solo servidor

`live_server.py` es **el único servidor** del visor. Sirve las dos cosas en el mismo
puerto, y el modelo se carga **solo** cuando se inicia una sesión en vivo:

| Ruta | Qué hace | ¿Toca la GPU? |
|---|---|---|
| `GET /` | el visor (`scripts_context/webgl_viewer/`) | no |
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
scripts_context/webgl_viewer/launch.py            # http://localhost:8090
# o directo:
python3 scripts_stream/live_server.py --port 8090
```

Después, en la página, panel **"Mapeo en vivo"**: elegir fuente, `Iniciar`. El mapa
crece frame a frame mientras se navega (órbita o primera persona, con mouse o control).

## Las fuentes

| Fuente | Para qué | Campo `path` |
|---|---|---|
| **Carpeta de frames** | repetir una prueba ya grabada como si llegara en vivo — es la forma reproducible de probar el streaming | carpeta con `000000.png`... |
| **Cámara en vivo (webcam o celular)** | cámara real: una webcam de este equipo o la cámara de un teléfono Android conectado por adb | se elige en un desplegable: `/api/live/devices` lista los `/dev/videoN` (y **prueba cuál entrega imagen**: varios nodos existen pero no sirven) y, en `remote`, una entrada por cámara de cada teléfono que aparece en `adb devices` |
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

Probar la cámara suelta, sin el modelo: `python3 scripts_stream/android_camera.py [serial] [cámara]`
(escribe `cel_prueba.jpg`).

Parámetros del panel: `fps` (0 = tan rápido como pueda el modelo), `máx. frames`
(0 = sin límite; cuenta frames leídos de la fuente, antes del analizador de contexto), `puntos/frame` (cuántos puntos se mandan al navegador por frame) y
`conf. (percentil)` (descarta el X% de píxeles menos confiables).

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
2. **⚙ construir mapas:** elegir tareas de `scripts_context/build_maps.py`. Para una sesión
   en vivo conviene marcar **windowed**: reprocesa los frames guardados en modo windowed
   (ventana 24), que corrige la deriva del streaming, y el resto de los mapas sale de ahí.

El trabajo corre en segundo plano, **en su propio cgroup con tope de RAM**
(`scripts_gpu/run_isolated.sh`): si un mapa agota la memoria muere ese trabajo, no VS Code
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

El mismo analizador sirve fuera de línea: `python3 scripts_stream/context_gate.py
--frames_dir <todos los frames> --out_dir <salida> [--synth]` deja los frames elegidos y un
`manifest.json` para `process_and_view.py --manifest`. Para medir una sesión en vivo sin
navegador está `replay_live.py`, que usa exactamente el mismo código que el servidor.

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

Mensajes de texto: JSON `{"type": "status"|"started"|"stopped"|"error"|"done", ...}`.
Mensajes binarios: un frame, cabecera de 72 bytes little-endian y después los datos.

```
uint32       frame_idx
uint32       n_points
float32[16]  c2w  (4x4 row-major, cámara -> mundo)
float32[n*3] xyz en coordenadas de mundo
uint8 [n*3]  rgb
```

`scripts_stream/` sólo tiene este archivo: el cliente vive en
`scripts_context/webgl_viewer/main.js` (misma escena three.js que el visor normal, así
no hay dos visores que mantener).
