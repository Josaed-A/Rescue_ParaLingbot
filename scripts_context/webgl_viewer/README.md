# Visor WebGL de pruebas (navegación libre + control Xbox)

Interfaz local para mirar las nubes de puntos ya exportadas, con un selector de
prueba y dos modos de navegación, más un panel de **mapeo en vivo**. Mientras no se
inicie una sesión en vivo no se carga el modelo ni se toca la GPU: sólo sirve archivos
que ya están en `captures/`.

```bash
scripts_context/webgl_viewer/launch.py            # http://localhost:8090
scripts_context/webgl_viewer/launch.py --port 8095 --no-open
```

El servidor es **uno solo** (`scripts_stream/live_server.py`) y hace dos cosas: sirve las
nubes ya exportadas y ofrece el **mapeo en vivo** por WebSocket. El modelo se carga
únicamente si se inicia una sesión en vivo; mirar nubes no toca la GPU.
Ver [../../scripts_stream/README.md](../../scripts_stream/README.md).

## Qué agrega respecto del visor `viser`

| | `view_cloud.py` / `view_npz.py` (viser) | este visor |
|---|---|---|
| Elegir qué prueba ver | un proceso por archivo, con la ruta en el comando | selector en la página, se cambia sin reiniciar nada |
| Órbita (subir/bajar, acercar, desplazar) | sí | sí, igual |
| Primera persona (vuelo libre tipo gameplay) | no | **sí** (WASD + mouse) |
| Mirar con el mouse | no (arrastrar orbita) | **sí** (pointer lock, como un shooter) |
| Control de Xbox | no | **sí** (Gamepad API del navegador) |
| Trayectoria y cámaras del video | sólo `view_npz.py` | sí, con salto a cualquier cámara |
| Ver el mapa construirse en vivo | no | **sí** (panel "Mapeo en vivo") |

## Explorador de pruebas

El panel de la izquierda reemplaza al selector de dos listas. Muestra cada prueba como una
tarjeta con su título, su zona, sus categorías y un botón por cada mapa que tenga:

| Botón | Archivo | Qué es |
|---|---|---|
| nube | `exports/<n>_denso.ply` | nube fusionada por vóxel |
| alta dens. | `exports/densidad_alta/<n>_denso_alta.ply` | nube fusionada más fina |
| cruda + tray. | `exports/webgl/<n>_raw.ply` + `_cameras.json` | nube por frame y trayectoria |
| malla | `exports/malla/<n>_malla.glb` | superficie por fusión TSDF |
| splat | `exports/splat/<n>_splat.ply` | Gaussian Splatting |
| malla filtr. | `exports/malla/<n>_filtrado_malla.glb` | malla TSDF con el filtro geométrico (`geo_filter.py`) |
| splat filtr. | `exports/splat/<n>_filtrado_splat.ply` | splat con el filtro: sin personas, profundidad depurada y ajustada a planos |
| estructura | `exports/estructura/<n>_estructura.glb` | malla simple de paredes y piso hecha con las esquinas detectadas |

El panel "Mapeo en vivo" tiene además las casillas **analizador de contexto** (activada por
defecto: salta frames redundantes y elige el más nítido) y **sintetizar frames intermedios**
(ver [scripts_stream/README.md](../../scripts_stream/README.md)).

- **Agrupar** por zona, por categoría (una prueba aparece en cada una de sus categorías),
  por carpeta o por fecha; **buscar** por texto; **filtrar** con los chips de categoría.
- **✎ datos:** título, zona de la universidad, categorías (sugeridas o nuevas) y notas. Se
  guardan en el `info.json` de la prueba sin tocar el resto de sus claves.
- **⚙ construir mapas:** genera los mapas que falten con `scripts_context/build_maps.py`.
- **📁 archivos:** árbol de la carpeta de la prueba. Los mapas se cargan con un clic; el resto
  (imágenes, videos, json) se abre en otra pestaña.
- Las sesiones en vivo quedan como "sin guardar" hasta que se guardan con nombre (ver
  `scripts_stream/README.md`).

Todos los mapas de una prueba usan la trayectoria de `exports/webgl/<n>_cameras.json`,
así que la nube, la malla y el splat se orientan igual (no sólo la nube cruda).

**Splats:** se dibujan con [gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D)
0.4.7 (vendorizado en `vendor/`), metido en la misma escena: órbita, primera persona,
control y salto a cámaras funcionan igual. El servidor manda las cabeceras COOP/COEP para
que la librería pueda ordenar las gaussianas en un worker con memoria compartida. El
revelado gradual de la librería está desactivado: con muchas gaussianas parecía que faltaba
media escena.

## Controles

**Órbita** (el modo por defecto, igual que antes): arrastrar para rotar, rueda
para acercar, clic derecho para desplazar.

**Primera persona:** clic para capturar el mouse, `WASD` para moverse,
`Espacio`/`Ctrl` para subir y bajar, `Shift` para correr, `Esc` para soltar el
mouse. La velocidad se ajusta con el control del panel.

**Control Xbox** (en primera persona, con o sin el mouse capturado): stick
izquierdo mueve, stick derecho mira, `RT`/`LT` suben y bajan, `A` corre,
`Start` alterna órbita ↔ primera persona. El navegador sólo ve el control
después de la primera pulsación de un botón.

## Qué archivos detecta

`/api/captures` recorre `captures/` y arma el selector con lo que encuentre:

| Patrón | Qué es |
|---|---|
| `*_denso_alta.ply` | nube fusionada de alta densidad (`export_dense_cloud.py` con vóxel chico) |
| `*_denso.ply` | nube fusionada, densidad base |
| `*_raw.ply` + `*_cameras.json` | nube por-frame (con el solape que produce el modelo) más la trayectoria |
| `*.glb` | mapas exportados por `process_and_view.py` / el visor viser |

Para generar los dos primeros ver `scripts_context/export_dense_cloud.py`; para
el tercero, `scripts_context/npz_to_webgl.py` (convierte un `.npz` de
`--save_predictions` en nube + trayectoria).

## Todos los mapas se navegan igual

La escala de cada captura es arbitraria (profundidad monocular), así que cada mapa
llegaba con un tamaño distinto y se navegaba distinto: en uno el vuelo quedaba pegado,
en otro disparado, y la vista inicial no era comparable. Ahora **cada mapa se normaliza
al cargarlo**: se centra en el origen y se escala a una diagonal canónica, de modo que
la vista de entrada, la velocidad de vuelo, el tamaño de punto y los planos de recorte
son los mismos para cualquier nube, sea de 5 M o de 46 M de puntos. Las poses de cámara
se transforman con el mismo grupo, así el salto a una cámara sigue cayendo donde debe.

## Cómo se dibujan las cámaras

Cada cámara es una **flecha** que apunta a donde miraba (antes era una pirámide de
frustum, que con 150 superpuestas tapaba el mapa). El color sigue el **degradado viridis
por avance del recorrido**, el mismo que usa el visor del repositorio original
(`lingbot_map/vis/point_cloud_viewer.py`): violeta al empezar, amarillo al terminar. La
línea de trayectoria usa ese mismo degradado.

## Estructura

```
webgl_viewer/
  launch.py           arranca el servidor único y abre el navegador
  catalog.py          catálogo del explorador: pruebas, mapas, metadatos, archivos
  captures_index.py   índice viejo de nubes (/api/captures, se mantiene por compatibilidad)
  index.html          interfaz (explorador, navegación, controles, panel en vivo)
  explorer.js         explorador de pruebas, edición de datos, construir mapas
  main.js             escena three.js: órbita, primera persona, gamepad,
                      trayectoria, y el cliente WebSocket del mapeo en vivo
  vendor/             three.js 0.160, sus addons y gaussian-splats-3d 0.4.7 (funciona offline)
```
El servidor vive en `scripts_stream/live_server.py`.

## Límites conocidos

- Las nubes de alta densidad pesan más de 1 GB: cargan bien en una máquina con
  GPU dedicada, pero en un equipo modesto conviene la densidad base.
- El pointer lock (mirar con el mouse) necesita un clic real; con el control de
  Xbox no hace falta, el stick derecho funciona igual.
- Sólo lee archivos ya exportados: si acabás de generar uno, usá "Refrescar".
