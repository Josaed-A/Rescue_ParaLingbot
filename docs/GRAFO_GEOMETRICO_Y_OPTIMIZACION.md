# Grafo geométrico de poses y optimización (nivel A)

**Código:**
- [`src/mapas/grafo_poses.py`](../src/mapas/grafo_poses.py): el grafo.
- [`src/mapas/o3d_estable.py`](../src/mapas/o3d_estable.py): la etapa del pipeline que lo usa.
- Tarea `estable` de `build_maps.py`.

**Pruebas:** [`test/test_o3d_estable.py`](../test/test_o3d_estable.py).

## Qué resuelve y qué no

El grafo detecta y corrige deriva **rígida** entre partes de una reconstrucción de LingBot-Map, usando lo que la propia reconstrucción ve dos veces. No corrige:

- **escala:** la mide y la informa;
- **forma de la profundidad dentro de un frame:** la bitácora del 2026-10-01 encontró que el ICP con escala por frame casi no ayudaba;
- **errores que no se pueden verificar con la geometría disponible.**

Cuando no hay evidencia suficiente conserva las poses, y lo dice.

## Nodos

**Fragmentos** de `frag_len = 8` frames consecutivos cada `frag_step = 4`; se solapan para que la odometría entre vecinos se pueda medir aun con pocos frames por segundo. No se crea un nodo por frame.

Cada nodo guarda:
- su identificador persistente (`<sesión>/frag/<frame inicial>-<frame final>`, con `source_index`);
- sesión, frames e intervalo, keyframes y estado;
- la pose inicial (identidad: los fragmentos ya están en el mundo de LingBot) y la optimizada (la corrección del grafo);
- marco y unidades: del modelo, sin escala métrica.

La nube del fragmento (submuestreada al vóxel de registro, la profundidad mediana / 60) se recalcula desde el `.npz`; no se copia en el grafo.

**Keyframes:** un frame cuando la cámara se movió 0.15 × la profundidad mediana o giró 15°, cada 10 frames como mínimo, o donde Stella pierde el seguimiento. Si sale borroso se toma el más nítido de los dos siguientes. Los keyframes solo deciden qué frames buscan cierres; **ningún frame se descarta del mapa**.

## Aristas

| Tipo | Medición | Información | Certeza |
|---|---|---|---|
| `odometria` | ICP punto-a-plano multiescala (4, 2, 1 vóxeles) entre fragmentos consecutivos, desde las poses actuales | `get_information_matrix_from_point_clouds` | cierta |
| `prior_lingbot` | la pose relativa original de LingBot (T = I) entre fragmentos consecutivos | 5 % de la mediana de la odometría | cierta, débil |
| `cierre` | ORB + RANSAC de la fundamental entre keyframes → correspondencias levantadas con la profundidad → RANSAC rígido 3D-3D → ICP desde esa transformación | como la odometría | incierta (line process) |

**Por qué hace falta el prior.** Open3D rechaza un grafo no conectado («Invalid PoseGraph»), y con la verbosidad en Error lo hacía en silencio. Donde ICP no puede medir la odometría (giros rápidos, frames ralos), el grafo quedaba partido y no se optimizaba. El prior lo mantiene conectado, y el número de componentes queda registrado.

**Validación de cada arista.** Que ICP converja no basta. Cada arista guarda su estado (`aceptada`, `rechazada`, `podada`) y el motivo:

| Control | Rechaza cuando |
|---|---|
| Ajuste | fitness < 0.3 |
| Residuo | el RMSE empeora más de 2 % respecto de la pose actual |
| Plausibilidad | odometría: la corrección pasa de 0.15 × la profundidad mediana o de 5°. Cierre: pasa de `max(0.3 × prof. mediana, 8 % de lo recorrido entre los dos fragmentos)` o de 15°. |
| Correspondencias | casi todas de un solo plano (segundo autovalor de Σ n nᵀ < 0.05 del primero): el registro puede deslizar sobre él, como en un pasillo |
| Visual (cierres) | menos de 30 inliers ORB |
| Coherencia (cierres) | el ICP termina lejos de la transformación de las correspondencias visuales |

Cada arista guarda la transformación medida, la información, el método, fitness y RMSE antes y después, las correspondencias, la escala relativa (Umeyama sobre las correspondencias, medida y **no** corregida) y la dispersión de normales.

## Optimización

1. **Si todas las restricciones aceptadas ya concuerdan con las poses** (p90 de las correcciones < 0.5 vóxel y < 1°), se conservan las poses sin optimizar.
2. **Si no:**
   - Se comprueba la conectividad.
   - Se ejecuta la optimización global de Open3D: Levenberg-Marquardt con line process, nodo de referencia 0 y `preference_loop_closure = 2`. Su distancia de correspondencia se toma de los cierres verificados; con la de una arista de odometría, un cierre que corrige una deriva acumulada quedaría como atípico desde el arranque.
   - Se mide el residuo de cada cierre contra la solución. Los que no la respaldan quedan `podada` y se reoptimiza sin ellos, hasta 3 veces.
3. **La corrección se acepta solo si:**
   - no deforma la geometría local: las aristas de odometría siguen cumpliéndose con un residuo menor a 3 vóxeles y 1.5°;
   - ningún fragmento se mueve más que la deriva plausible (8 % de lo recorrido desde el inicio);
   - mejora la consistencia entre fragmentos (RMSE o fitness de las aristas aceptadas).

   Si no se cumple, se conservan las poses y se registra por qué.
4. **La corrección de cada frame** se interpola entre los centros de los fragmentos (SLERP y lineal), sin zigzag de frame a frame. Las poses originales quedan en `c2w_original` y las nuevas tienen versión.

**Robustez ante coincidencias falsas.** En una zona repetitiva (pasillo de tramos iguales), un cierre falso puede pasar la verificación visual y la geométrica: en la prueba, 186 inliers ORB y fitness 0.96. Lo que lo frena es que pide correr la escena un tramo entero (4 unidades) entre fragmentos separados por esa misma distancia, muy por encima del 8 % plausible. Si un cierre falso se acepta igual, el line process y el residuo contra la solución lo podan.

## Por qué Open3D y no GTSAM

| | Open3D (elegido) | GTSAM / iSAM2 |
|---|---|---|
| Instalado | sí (0.19) | no (hay rueda 4.2 para Python 3.10) |
| Lo que hace falta hoy | registro de fragmentos y optimización fuera de línea, después de cada sesión | incremental, durante la captura |
| Registro geométrico (ICP, información) | incluido | habría que hacerlo con Open3D igual |
| Costo medido | grafo de 33 nodos: 12–30 s con la búsqueda de cierres; fablab, 118 nodos: 100 s | — |

No conviene tener dos optimizadores moviendo las mismas poses. Si la optimización pasa al servidor en vivo, el grafo persistido (`_grafo.json`: nodos, aristas con T e información, estado de cada arista) se traduce directo a un grafo de factores (`Pose3` y `BetweenFactorPose3` con su información; los cierres con pérdida robusta). Open3D seguiría calculando las restricciones y GTSAM sería el único dueño de las poses.

## Persistencia

`<name>_grafo.json` contiene:
- nodos y aristas con su procedencia y estado;
- conteo por tipo y estado;
- estadísticas: consistencia antes y después, residuos, componentes, deriva estimada por los cierres visuales;
- parámetros;
- versión de poses de entrada y de salida.

`<name>_estable.npz` contiene las poses originales y las estabilizadas, la máscara y el metadato. Con esos dos archivos y el `.npz` de la sesión, todo se puede rehacer sin estado oculto.

## Pruebas sintéticas

`test/test_o3d_estable.py`: escenas renderizadas con el raycasting de Open3D desde poses exactas.

| Escena | Resultado |
|---|---|
| Habitación limpia (puerta, tabique, obstáculo, escalón) | poses conservadas; 1.9 % de puntos quitados; la puerta sigue abierta, y el obstáculo y el escalón siguen ahí |
| Deriva en la vuelta de un circuito (0.4 de traslación y 3° en rampa) | 33 cierres verificados aceptados y ninguno podado. Error de las últimas 40 cámaras: 0.221 → 0.012 (−95 %); de todas: 0.083 → 0.031. Superficies duplicadas: 5 → 0. Residuo máximo de la odometría: 0.008 (no deforma). |
| Pasillo de tramos iguales (geometría y textura) | 20 cierres falsos pasaron lo visual y lo geométrico y pedían correrse un tramo: los 20 rechazados por plausibilidad. Cámaras sin mover. |
| Ruido (1 % de picos de profundidad) | 99.3 % de los picos enmascarados y 100 % de la superficie real conservada |
| Identificadores | no dependen de la ruta ni del orden |

Resultados sobre sesiones reales: [PIPELINE_OPEN3D_TSDF.md](PIPELINE_OPEN3D_TSDF.md#resultados-sobre-sesiones-reales).

## ICP robusto y corrección a 4 grados de libertad (2026-10-17)

**ICP punto a plano robusto.** Las mediciones del grafo (odometría, cierres y el registro multivía) usan ICP punto a plano con un núcleo de Tukey (`TransformationEstimationPointToPlane(TukeyLoss(k))`). El núcleo tiene un ancho de `icp_tukey_rel` × la distancia de correspondencia de cada pasada (0.5 por defecto; 0 = sin núcleo, como antes). Así los atípicos no tiran de la medición: un piso que resbala o un mueble que se movió entre visitas. En piso1 (sesión a 10 frames/s) las odometrías aceptadas subieron de 81-84 a 101 (rechazadas: de 23-28 a 10).

**4 grados de libertad (`--dof4`, apagado por defecto).** Las aristas *prior* de LingBot llevan una matriz de información anisótropa (`prior_information`): rígidas (`dof4_rigidez` × una odometría típica) en la inclinación y en la altura, y con el peso habitual en el rumbo y en el desplazamiento horizontal. Así la optimización Levenberg-Marquardt de Open3D solo corrige rumbo y plano horizontal; la inclinación y la altura quedan como las fija la gravedad o el piso.
- **Por qué:** en piso1 la corrección de 6 grados de libertad mejoraba algo las paredes, pero inclinaba fragmentos y empeoraba el piso entre visitas un 40 %.
- **Intento previo descartado:** proyectar la solución de 6 grados a 4 después de optimizar desparejaba a los vecinos.
- **Test:** `test_4dof_priors_let_the_loop_fix_heading_but_not_tilt`.

**Desde `build_maps`:** `O3D_ESTABLE_ARGS="--dof4" build_maps.py <prueba> --tasks estable` pasa opciones extra a `o3d_estable.py`. Es lo que usa el proceso por capas de Pruebas_Macondo en su etapa 2.

**Resultado en piso1:** con la ruta de LingBot a cadencia uniforme, el grafo aceptó 14 cierres verificados y podó 41. La corrección se rechazó porque dejaba 1 odometría sin cumplir, así que se conserva la ruta de LingBot. Es coherente con el diagnóstico del 2026-10-15: el error entre visitas de estos videos es de profundidad, no de pose.
