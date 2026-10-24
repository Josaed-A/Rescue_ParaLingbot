# Benchmark de tracking: BASIC (LingBot) frente a Stella-VSLAM

Resultados medidos de la comparación entre trackers. La primera tanda es la de la etapa 4
(2026-10-04). Las etapas 17 y 18 ampliarán este documento con la matriz completa de pruebas.
Contexto y decisiones de diseño en [STELLA_INTEGRATION.md](STELLA_INTEGRATION.md).

## Qué se compara

| Nombre | Qué es | Cadencia |
|---|---|---|
| `ref` | LingBot **windowed** fuera de línea (`eval/final_m*.npz`), la mejor trayectoria disponible: 4.75% / 2.47% de error de forma contra los croquis de pasillos y fablab | ~10 Hz |
| `basic_vivo` | LingBot **streaming en vivo** (la pose BASIC), con `replay_live.py --realtime` reproduciendo el video como una cámara y el analizador de contexto encendido | ~1.8 Hz |
| `stella_solo` | Stella-VSLAM **aislado**, con la CPU para sí (corridas de la etapa 2, a 30 fps) | 15-20 Hz |
| `stella_vivo` | Stella-VSLAM **en vivo**, en la misma máquina que LingBot, recibiendo 15 fps por el puente ROS2 (etapa 3) | variable |

**No hay ground truth métrico.** `ref` es LingBot también: no es la verdad. Lo que sí vale como
evidencia es el **acuerdo entre estimadores independientes**. Si Stella, con features ORB y
bundle adjustment, coincide con LingBot windowed mejor que LingBot streaming, eso indica que
Stella corrige al streaming en ese tramo.

## Método (`src/mapas/compare_tracking.py`, `src/vivo/traj_align.py`)

El plan exige corregir tiempo, ejes y escala antes de interpretar. Se hace en ese orden.

1. **Tiempo.** Todo se lleva al timestamp de presentación (PTS) del video.
   - **Tasa variable.** Los videos del celular lo son: el del fablab tiene 13 huecos de hasta 168 ms, e "índice / fps" se desvía hasta **0.6 s**. Se corrigió `FrameSource` para que los videos usen el PTS (`stamp_kind=video_pts`). Las corridas anteriores se convierten índice → PTS.
   - **Frames de windowed.** Los frames que extrajo ffmpeg están corridos 0.2-0.27 s y re-muestreados a tasa fija. Además, `source_index` indexa carpetas distintas según la corrida. Por eso el instante de cada frame windowed se obtiene **emparejando por contenido** la imagen que vio el modelo (clave `images` del `.npz`) con el frame real del video. El error medio del emparejamiento fue de 0.9 a 1.9 sobre 255.
   - **Verificación.** Un desfase temporal residual estimado por correlación de la velocidad angular da **|d| ≤ 0.05 s** en todos los pares con correlación fiable.
2. **Ejes.** Se estima la rotación de cámara C entre convenciones con una solución mano-ojo cerrada. Al caminar casi todo el giro es de guiñada, así que C queda indeterminada alrededor de ese eje: el condicionamiento da valores singulares de 1 / 0.03-0.06. Por eso la prueba que vale es otra: tras alinear solo el mundo, el error de orientación de Stella aislada es de 3.4° de mediana en el fablab. Con convenciones distintas sería de decenas de grados. **Las convenciones coinciden.**
3. **Escala.** Se usa una Sim(3) de Umeyama, y también SE(3) para mostrar la diferencia. Además se estima la escala en ventanas de 5 s, para ver si una sola Sim(3) alcanza.
4. **Errores.**
   - ATE tras la Sim(3), en % del largo del recorrido de `ref` en el tramo.
   - Error de orientación.
   - RPE a 1 s: traslación relativa y rotación. Es local, no depende de la deriva acumulada y es la métrica más honesta cuando una trayectoria deriva.
5. **Segmentos.** Si Stella se reinicia, su mundo cambia: cada segmento se alinea por separado.

## Resultados

Cada fila compara una trayectoria contra `ref` en el tramo que ambas cubren.

| Secuencia | Trayectoria | Pares | Tramo (s) | Escala →ref | ATE Sim(3) | ATE SE(3) | RPE 1 s tras. | RPE 1 s rot. | CV escala 5 s |
|---|---|---|---|---|---|---|---|---|---|
| fablab | stella_solo | 176 | 1-23 | 1.15 | **1.34%** | 3.27% | 0.19 | 0.81° | 0.12 |
| fablab | basic_vivo | 87 | 0-47 | 1.05 | 1.85% | 2.10% | 0.20 | 0.83° | 0.30 |
| fablab | stella_vivo (corrida A) | 29 | 1-11 | 1.32 | 3.27% | 7.27% | 0.27 | 0.77° | 0.09 |
| fablab | stella_vivo (corrida B, etapa 3) | 139 | 1-23 | 1.41 | 6.00% | 9.10% | 0.57 | 1.99° | **0.63** |
| escaleras | stella_solo | 110 | 0-74 | 1.78 | **1.55%** | 2.25% | 0.22 | **1.89°** | 0.21 |
| escaleras | basic_vivo | 134 | 0-74 | 1.13 | 2.75% | 2.85% | 0.80 | **6.42°** | 0.25 |
| escaleras | stella_vivo | 13 | — | — | — | — | — | — | — |
| pasillos | stella_solo | 42 | 55-58 | 8.49 | 15.7% | 25.9% | 0.59 | 2.54° | — |
| pasillos | basic_vivo | 119 | 0-66 | 3.20 | 12.2% | 16.6% | 1.39 | 3.12° | 0.55 |
| pasillos | stella_vivo | 0 | — | — | — | — | — | — | — |

Cobertura de Stella sobre el recorrido:

| Secuencia | stella_solo | stella_vivo |
|---|---|---|
| fablab | 0-22.7 s de 47.5 s; pierde a los 22.7 s y no recupera | corrida A: 1-11 s; corrida B: 1-23 s |
| escaleras | 0-11.5 s y 72-74 s (relocaliza al volver a la entrada) | 13 poses en 74 s: se reinicia dos veces por perder el tracking a menos de 5 s de inicializar |
| pasillos | 3 s útiles en 2 segmentos tras 2 reinicios | nunca inicializa |

Por par: `captures/stella/etapa4_2026-10-04/<secuencia>/<a>__<b>.json` y `.png`.

> **Corrección (etapa 6, 2026-10-04).** El punto 1 de "Qué dicen los datos" está **sesgado por el tramo**: compara a Stella en el tramo que cubre (0-23 s en el fablab, 0-12 s en escaleras) contra BASIC en todo el recorrido. En el **mismo** tramo, BASIC coincide más con windowed: fablab 0-23 s, ATE 0.83-0.94% frente a 1.34-1.5%; escaleras 0-12 s, 1.23% frente a ~1.5% y RPE de rotación 0.93°/s frente a 1.5-1.9°/s. El error de rotación de 6.4°/s del streaming en escaleras está en la parte donde Stella ya estaba perdida. Además, windowed es LingBot y favorece al streaming. Con referencias independientes (croquis, regreso al punto de partida), ver la sección de la etapa 6 al final.

## Qué dicen los datos

1. **Con imagen nítida y CPU propia, Stella coincide con LingBot windowed mejor que LingBot streaming.** El ATE es de 1.3-1.6% del largo, frente a 1.9-2.8% del streaming.
   - En el fablab, el error local es el mismo para los dos trackers: RPE de traslación 0.19 frente a 0.20, de rotación 0.81° frente a 0.83°.
   - **En escaleras, el streaming acumula error de rotación 3.4 veces más rápido** que Stella: 6.4°/s frente a 1.9°/s. Su orientación absoluta se aleja de windowed de 34° al principio a 90-115° en la bajada, aunque las dos trayectorias giran al mismo ritmo (20-22°/s). Es deriva, no ruido. Es exactamente el caso en que HYBRID debería ayudar.
2. **La escala nunca es la misma, y no es constante.**
   - Sim(3) es necesaria: el ATE con SE(3) es entre 1.2 y 2.4 veces mayor.
   - La escala relativa varía un 12-21% en ventanas de 5 s aun con Stella aislada. En escaleras se acomoda en los primeros 8 s, de 0.67 a 1.8.
   - Una Sim(3) global no alcanza: la etapa 9 tendrá que re-estimarla por ventana.
3. **Stella en vivo, compartiendo la máquina con LingBot, es mucho peor que aislada.** Este es el hallazgo principal de la etapa.
   - Cobertura: 6% frente a 37% en el fablab, 2% frente a 15% en escaleras, y ninguna pose en pasillos.
   - En la corrida B del fablab la escala deriva de 1 a **4.7** en 10 s. La deriva es de esa Stella, no de la referencia: contra Stella aislada deriva 3.4 veces.
   - Además, Stella no es determinista. Con el mismo video y la misma configuración, la corrida B dio 176 poses y la A 32.
   - La causa probable es la falta de CPU y de frames: procesa el 24-54% de lo que recibe (etapa 3). **UNKNOWN:** no se aisló si es CPU, cola o azar del RANSAC. Hace falta repetir con Stella en núcleos reservados (etapa 19, adelantable).
4. **La relocalización conserva el mapa.**
   - En escaleras, con la Sim(3) estimada **solo** en los primeros 12 s, las poses de Stella al volver a la entrada (72-74 s) caen a 4.3% del largo del recorrido de donde las pone windowed, sin reajustar nada.
   - Windowed, a su vez, pone el final a 2.4% del inicio. Es la base para la etapa 12 (loop closure).
5. **Pasillos (caminata rápida) no es comparable.** Stella aislada solo da 3 s en dos mapas tras dos reinicios. La Sim(3) sobre 3 s de avance casi recto da escala 8.5 y 54° de error: no concluyente. Para la caminata rápida, BASIC es el único tracker.
6. **Las señales BASIC predicen el estado de Stella.** En la sesión en vivo del fablab, la nitidez mediana de los frames con Stella en TRACKING fue 1154, frente a 793 con Stella LOST. El movimiento no cambió: 37 frente a 40 px. Es una muestra chica (20 frente a 65 frames). Refuerza lo visto en la etapa 2.
7. **La asociación exacta no alcanza en vivo.** BASIC va a ~1.8 Hz y Stella procesa subconjuntos distintos de la cámara. Entre las dos sesiones en vivo solo 4-28 pares caen a menos de 50 ms. Para comparar y fusionar en vivo hace falta el `PoseBuffer` con interpolación de la etapa 5.

## Asociación con el buffer de poses (etapa 5)

Repitiendo la comparación con `--assoc interp` (buffer de `b` consultado en los instantes de `a`, `max_gap` 0.25 s):

| Par | Pares, vecino → buffer | ATE Sim(3), vecino → buffer |
|---|---|---|
| fablab: ref vs stella_solo | 176 → 185 | 1.34% → 1.34% |
| escaleras: ref vs stella_solo | 110 → 121 | 1.55% → 1.54% |
| fablab: basic vs stella, sesión de la etapa 3 | 28 → 34 | 7.06% → 6.68% |
| fablab: basic vs stella, sesiones de las etapas 4 y 5 | 10 → 11 y 14 → 14 | — (muy pocos) |

La interpolación no cambia los resultados. Dentro de una sesión en vivo, lo que limita los pares es la escasez de poses de Stella, no la asociación.

## Limitaciones

- `ref` no es ground truth. Los porcentajes miden acuerdo, no exactitud.
- Una corrida por configuración en vivo, salvo el fablab, que tiene dos y muestra la variabilidad. El benchmark de la etapa 17 necesita repeticiones.
- RPE y ATE con `basic_vivo` usan pocos pares, de 87 a 134, por la cadencia del modelo.
- Intrínsecos de Stella estimados desde LingBot, sin calibración con patrón.

## Reproducir

```bash
# sesiones en vivo (GPU + ROS2 + Stella), una por video: ver captures/stella/etapa4_2026-10-04/sesiones/sessions.sh
src/gpu/run_gpu.sh -- python3 src/vivo/replay_live.py --source video --path <video.mp4> --rotation 90 \
    --realtime --context --ros2 --stella_config src/ros/stella/unisabana_portrait_540x960.yaml \
    --ros2_resize 540x960 --ros2_pub_every 2 --captures_dir <out>

# comparación (CPU)
python3 src/mapas/compare_tracking.py --video <video.mp4> --out <dir> \
    ref=windowed:<prueba>/eval/final_mN.npz stella_solo=stella_run:<corrida etapa 2> \
    basic_vivo=session:<sesion.npz> stella_vivo=session_stella:<sesion.npz> \
    --pairs ref:stella_solo,ref:basic_vivo,ref:stella_vivo,basic_vivo:stella_vivo
```

## Etapa 6: modos de referencia contra referencias independientes

Detalle y método en [STELLA_INTEGRATION.md](STELLA_INTEGRATION.md), etapa 6.

| Caso | Indicador | BASIC | HYBRID | STELLA |
|---|---|---|---|---|
| escaleras, Stella aislada (simulada) | regreso a la entrada, \|fin − inicio\| / largo | 4.16% | 1.57% | **0.99%** |
| fablab, Stella aislada (simulada) | error de forma contra croquis | 2.04-2.11% | 1.98-2.03% | 1.95-1.97% |
| fablab, Stella en vivo sana, CPU repartida | croquis | 2.07% | 2.09% | 2.11% |
| fablab, Stella en vivo famélica, con criterio de ritmo | croquis | 2.04% | 2.13% | 3.61% |
| pasillos | croquis | 15.85% | 15.91% | 16.18% |

La mejora de HYBRID sobre BASIC no está demostrada. El único indicio es el cierre de escaleras por relocalización, en una sola secuencia. HYBRID con el criterio de ritmo no empeora.

## Benchmark de las etapas 17-18 (2026-10-05)

Todo con `replay_live.py`, el mismo código que el servidor en vivo, reproduciendo los videos del celular en tiempo real como una cámara (el modelo toma ~1.7 frames/s y Stella recibe 15 fps). Stella en los núcleos 4-11 y el modelo en el resto. Analizador de contexto activado. Una corrida por configuración y video; el fablab tiene además una repetición completa con recursos y una tercera tras la optimización.

| Config | Qué es |
|---|---|
| A | baseline actual: sin ROS2 ni Stella |
| B / C / D | BASIC, STELLA y HYBRID calculados en la misma sesión con el puente y Stella (la geometría en vivo, con BASIC) |
| E | como D, con la geometría registrada en vivo con HYBRID |
| S | modo STELLA con la geometría registrada con su referencia y las correcciones por keyframes aplicadas en vivo |

Datos y gráficas: `captures/stella/benchmark_2026-10-05/resultados/` (tablas `tracking`, `geometria`, `rendimiento` en CSV y Markdown; `graficas/trayectorias_<seq>.png`, `barras_*.png`; `registro/<corrida>/` con el mapa registrado por referencia; `comparacion/` con ATE/RPE contra windowed) y `captures/stella/benchmark_recursos_2026-10-05/resultados/` (repetición del fablab con recursos, `graficas/recursos_*.png`).

### Tracking

| Secuencia | Poses de Stella por sesión | Frames del modelo con Stella en TRACKING | HYBRID usa Stella | Reinicios / relocalizaciones / loop closures |
|---|---|---|---|---|
| fablab (7 sesiones) | 182-234 | 44-49% | 8-26% | 0 / 0-1 / 0 |
| escaleras (3) | 80-99 | 13-16% | 0-3% | 1-2 / 0-1 / 0 |
| pasillos (3) | 0-8 | 0-3% | 0% | 0-1 / 0 / 0 |
| cámara quieta, webcam y celular (T1, T13) | 0 | 0% (Stella no inicializa sin paralaje) | 0% | — |

Stella se pierde en el fablab a los ~22 s (giro a una pared blanca con desenfoque, igual que aislada) y no se recupera, porque el recorrido sigue por lugares nuevos. Por eso HYBRID usa a Stella en una fracción chica de los frames aun donde trackea: el criterio de ritmo (≥10 poses/s) la descarta cuando está famélica.

### Geometría y deriva (rango entre sesiones; la geometría de LingBot registrada con cada referencia)

| Secuencia | Referencia | Croquis (forma) | Cierre fin-inicio | Coherencia a 1 s | Puntos por vóxel | ATE contra windowed |
|---|---|---|---|---|---|---|
| fablab | BASIC | 1.94-2.32% | (no vuelve) | 0.90-0.94 | 3.06-3.32 | 1.57-1.68% |
| fablab | HYBRID | 1.80-2.32% | | 0.84-0.94 | 2.98-3.32 | 1.48-1.73% |
| fablab | STELLA | 1.89-2.32% | | 0.84-0.94 | 2.82-3.32 | 1.30-1.72% |
| escaleras | BASIC | — | 1.09-2.82% | 0.14-0.31 | 2.16-2.22 | 1.81-2.38% |
| escaleras | HYBRID | — | 1.11-2.67% | 0.14-0.38 | 2.16-2.20 | 1.90-2.39% |
| escaleras | STELLA | — | 0.97-2.66% | 0.15-0.31 | 2.16-2.20 | 1.89-2.39% |
| pasillos | las tres (Stella casi no trackea) | 8.12-9.88% | 21.7-37.1% | 0.80-0.94 | 2.10-2.23 | 3.76-7.56% |

### Rendimiento (fablab; repetición con monitor y corrida tras la optimización de la etapa 19)

| Config | fps del modelo | Latencia captura → mapa (mediana) | CPU modelo (mediana, 100 = 1 núcleo) | CPU Stella | RAM modelo / Stella (pico) | VRAM total en la GPU (pico) | GPU (mediana) |
|---|---|---|---|---|---|---|---|
| A | 1.77-1.83 | 0.56 s | 297-311% | — | 8.6-9.6 GB / — | 7.8-7.9 GB | 65-77%, 37 W |
| D, E, S (antes de la etapa 19) | 1.64-1.79 | 0.58-0.60 s | 481-492% | 33-36% | 8.1-8.8 GB / 215 MB | 7.8 GB | 57-71%, 35 W |
| D (después de la etapa 19) | 1.68 | — | **308%** | 36% | 8.7 GB / 215 MB | 7.9 GB | 73%, 35 W |
| T13 celular / T1 webcam | 1.39 / 1.63 | 0.48 s | — | — | — | 4.8 / 5.4 GB (torch) | — |

La VRAM total incluye el contexto CUDA y el visor (170 MB); el modelo solo, según torch, usa 6.2-6.3 GB. Tamaño del mapa por sesión (registro, 8000 puntos por frame): 0.63-1.0 M puntos crudos → 0.20-0.47 M vóxeles, 9-15 MB.

### Matriz de pruebas (etapa 18)

| # | Prueba | Evidencia | Resultado | Estado |
|---|---|---|---|---|
| 1 | Cámara estática | T1 (webcam), T13 (celular quieto) | LingBot mapea; Stella se queda en Initializing (sin paralaje) y todo sigue en BASIC | cubierta |
| 2 | Movimiento lineal | pasillos (recto con giros) | ver 7 | parcial: no hay un tramo recto aislado |
| 3 | Rotación | — | sin grabación de rotación pura | **pendiente (prueba física)** |
| 4 | Movimiento lento | fablab | Stella trackea ~22 s; las tres referencias quedan a ±0.2 puntos contra el croquis | cubierta |
| 5 | Movimiento rápido | pasillos | Stella no inicializa (desenfoque); BASIC es el único tracker | cubierta |
| 6 | Habitación | fablab | ver 4 | cubierta |
| 7 | Pasillo | pasillos | ver 5; error de forma 8.1-9.9% en vivo (streaming) | cubierta |
| 8 | Escaleras / desnivel | escaleras | Stella trackea 13-16% de los frames; cierre 1.0-2.8% con todas las referencias | cubierta |
| 9 | Regreso a una superficie ya vista | escaleras (vuelta a la puerta) | Stella aislada relocaliza al volver; en vivo, 1 relocalización en 3 sesiones; simulación: STELLA 0.72-0.99% de cierre contra 4.16% de BASIC | cubierta (una secuencia) |
| 10 | Loop closure | — | ningún loop closure en los datos | **pendiente (prueba física)** |
| 11 | Pérdida temporal de tracking | todas las sesiones con Stella | al perderse Stella, HYBRID y STELLA caen a BASIC con el motivo anotado; sin saltos | cubierta |
| 12 | Recuperación | escaleras, pasillos | relocalización (escaleras) y reinicio con mapa nuevo (pasillos): el selector re-ancla el mapa nuevo | cubierta |
| 13 | Cámara remota | T13 (celular por adb, Wi-Fi) | funciona a 1.39 frames/s; un hueco de 4.8 s entre frames del modelo (la fuente); sin Stella por cámara quieta | parcial: falta caminando |
| 14 | Stella desconectado | T14: SIGKILL a Stella en el frame 30 | la sesión sigue (84 frames, 1.75 fps); estado de Stella UNKNOWN y todo en BASIC desde ahí | cubierta |
| 15 | LingBot detenido | T15: el modelo se detiene 8 s en el frame 30 | la cámara y Stella siguen; hueco de 8.6 s entre frames del modelo y la sesión continúa (70 frames) | cubierta |

### Qué dicen estos datos

1. **La variación entre sesiones iguales es tan grande como la diferencia entre referencias.** BASIC solo va de 1.94% a 2.32% contra el croquis en el fablab y de 8.1% a 9.9% en pasillos: en tiempo real el modelo toma frames distintos en cada corrida. Con una corrida por configuración no se puede ordenar BASIC, HYBRID y STELLA.
2. **Stella en vivo trackea poco en estos videos** (44-49% de los frames en el fablab, 13-16% en escaleras, casi nada en pasillos), sobre todo por desenfoque, y la relocalización solo ayuda cuando el recorrido vuelve.
3. **La referencia de Stella no mejora la geometría de LingBot:** baja la coherencia multivista y empeora TSDF y splat (etapas 11 y 14).
4. **El sistema degrada de forma controlada** en todos los casos de falla probados (11, 14, 15): BASIC sigue y el mapa se sigue construyendo.
5. **Costo:** con la etapa 19, el puente ROS2 no agrega CPU medible al proceso del modelo; Stella usa ~35% de un núcleo y 215 MB, y el modelo va ~5% más lento compartiendo la máquina.

**Decisión: BASIC sigue por defecto.** HYBRID es seguro (no empeora) y STELLA queda experimental. Para decidir hace falta lo que esta matriz no tiene: circuito con lazo, rotación pura y repeticiones de cada configuración con alguien caminando.

### Reproducir

```bash
src/ros/run_benchmark.sh captures/stella/<carpeta> A_fablab D_fablab E_fablab S_fablab A_escaleras ... T14_fablab T15_fablab T1_webcam T13_android
python3 src/mapas/analyze_benchmark.py --bench captures/stella/<carpeta> --out captures/stella/<carpeta>/resultados
python3 src/mapas/scale_alignment.py --out captures/stella/etapa9_2026-10-05          # etapa 9
python3 src/mapas/kf_correction_report.py --run <corrida> --video V.mp4 --ref windowed.npz --out OUT   # etapa 12
```
