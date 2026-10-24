# Contratos espaciales: poses, identificadores, versiones y dependencias

Este documento fija qué pose usa cada etapa, cómo se identifica cada artefacto y qué hay que rehacer cuando algo cambia. Rige para ParaLingbot (rama `Vizuallizador`) y para el gemelo de Pruebas_Macondo (rama `estado-dinov2-mujoco-paralingbot`).

Arquitectura general en [ARQUITECTURA_GRAFOS_PARALINGBOT.md](ARQUITECTURA_GRAFOS_PARALINGBOT.md).

## 1. Qué pose usa cada etapa

| Etapa | Pose | Dónde se fija |
|---|---|---|
| LingBot-Map (vivo y offline) | BASIC: cabeza de cámara del modelo. `extrinsic` del `.npz` es w2c. | `lingbot_map/` (sin tocar) |
| Registro de la geometría en vivo | BASIC (`pose_source = 0` en todos los frames medidos) | `src/vivo/live_server.py` |
| STELLA, HYBRID | Solo referencias: `pose_ref_stella`, `pose_ref_hybrid`. Nunca registran geometría. | `src/vivo/tracking.py` |
| Filtro geométrico (`geo_filter.py`) | BASIC, o las realineadas por planos (`--align`, apagado) → `c2w_fix` | `src/mapas/geo_filter.py` |
| Estabilización Open3D (`o3d_estable.py`) | Entra BASIC; sale `c2w` estabilizada, con versión. Las originales se conservan. | `src/mapas/o3d_estable.py` |
| TSDF | BASIC (`malla`, `malla_f`) o estabilizada (`--stable`, `malla_e`) | `src/mapas/tsdf_mesh.py` |
| Gemelo MuJoCo (Macondo) | `--poses basic`, `estable`, `hybrid` o `stella`. El filtro y la estructura solo se usan si sus poses coinciden con las pedidas. | `vision/lingbot.py` |
| Memoria semántica (Macondo) | La de la escena; se puede reubicar con otra versión de la misma sesión | `vision/memoria_semantica.py` |

**Regla.** Ningún producto se arma combinando artefactos calculados con poses distintas. Las comprobaciones que lo impiden:

- **`lingbot.load` (Macondo):** compara `c2w_fix` del filtro con las poses pedidas, con tolerancias de 0.5° y del 1 % de la profundidad mediana. Si difieren, corta con `IncompatibleArtifacts`.
- **`o3d_estable.py`:** se niega a usar un filtro de otras poses o de otros frames.
- **`tsdf_mesh.py --stable`:** se niega a usar una estabilización de otra reconstrucción (otras poses de entrada u otros frames).
- **`lingbot.load_stable` (Macondo):** hace la misma comprobación.
- **Memoria semántica:** `reubicar` exige la misma sesión, y `ASOCIADO_A_ESTRUCTURA` solo se calcula si la estructura es de la misma versión de poses.

## 2. Marcos y unidades

| Marco | Definición | Unidades |
|---|---|---|
| Cámara | OpenCV: x a la derecha, y hacia abajo, z hacia adelante | del modelo |
| Mundo LingBot | El de las poses `c2w = inv(extrinsic)`. Stella y HYBRID vienen alineadas a él (`COORDINATE_FRAMES.md`). | del modelo, sin escala métrica |
| Mundo estabilizado | Mundo LingBot con la corrección del grafo aplicada a cada fragmento | del modelo |
| Nivelado (Macondo y MuJoCo) | `p = R·p·escala + t`: z vertical, x el rumbo de la primera cámara, origen en el piso bajo ella | metros **estimados** (Depth Anything V2, sin calibrar) |

La escala métrica es una estimación monocular: la mediana del cociente entre las profundidades de Depth Anything V2 y LingBot en 40 frames. Se guarda con su método y su dispersión en `escena.json → procedencia.escala`, con `calibrada: false`. Ningún módulo la reaplica.

**Convención de rayos de Open3D.** `create_rays_pinhole` da direcciones sin normalizar, con z = 1 en la cámara, por el centro del píxel (u + 0.5). Por eso `t_hit` ya es la profundidad z. `tsdf_mesh.py` dividía por la norma del rayo hasta el 2026-10-14; ver [PIPELINE_OPEN3D_TSDF.md](PIPELINE_OPEN3D_TSDF.md).

## 3. Identificadores

Ningún identificador depende del orden en que se recorren los archivos.

| Entidad | Identificador | Estable ante |
|---|---|---|
| Sesión (reconstrucción) | `<prueba>/<npz>/<sha1(extrinsic)[:8]>` | rutas; cambia si cambian las poses de entrada |
| Frame | `source_index` (el frame de `frames/` o del video); si no está, `frame_ids` | reproceso windowed de la misma sesión |
| Fragmento (nodo A) | `<sesión>/frag/<frame inicial>-<frame final>` | parámetros iguales |
| Arista (A) | `<origen>-><destino>`; prior: `<origen>~<destino>` | |
| Entidad estructural (B) | `<sesión>/estructura/<versión de poses>/<tipo>-<sha1(normal, centro cuantizados)>` | orden del RANSAC |
| Objeto (C) | `<sesión>/obj/<clase>/<sha1(frames y celdas de píxel de sus observaciones)>` | **cambios de poses** |
| Entidad física (D) | nombre del geom en la escena (`pared_lingbot_k`, `soporte_k`, `<clase>_k`) | |

## 4. Versiones y procedencia

- **Versión de poses:** `sha1(c2w float32)[:16]`.
  - `o3d_estable.py` la escribe como `version_poses_entrada` y `version_poses`.
  - `tsdf_mesh.py` la guarda en `version_poses` de `_info.json`.
  - Macondo la guarda en `procedencia.poses_sha1` y `memoria_semantica.json → version_poses`.
- **Huella de la máscara:** `sha1(packbits(keep))`.
- **Huella de entradas del TSDF:** cubre el `.npz`, sus poses, la versión de poses y la máscara de la estabilización, los parámetros y el filtro. Con `--skip_if_same`, si coincide con la malla guardada, no se reintegra.

Tipos de geometría, siempre explícitos:

| Tipo | Dónde |
|---|---|
| observada | profundidad de LingBot por frame, `*_nube_original.ply` |
| registrada y filtrada | `*_nube_estable.ply`, máscara de `*_estable.npz` |
| estructural (diagnóstico) | `*_estructura_grafo.json` (no reemplaza observaciones) |
| integrada TSDF | `*_malla.*`, `*_filtrado_malla.*`, `*_estable_malla.*` (`tipo_geometria` en `_info.json`) |
| inferida (GenPC, futuro) | solo por `genpc_interfaz.py validar`, marcada como inferida |
| simplificada para simulación | `escena.mjb` de Macondo, `escena.json → capas.fisico` |

## 5. Dependencias: qué se rehace cuando algo cambia

`o3d_estable.py` escribe `<name>_dependencias.json` con la versión de poses y la huella de la máscara.

| Cambia | Se invalida | Se conserva |
|---|---|---|
| Poses (versión) | TSDF estabilizada, nube estable, grafo estructural, escena MuJoCo, posiciones de la memoria | etiquetas DINOv2, identidad de los objetos y sus observaciones (`reubicar` recalcula posiciones sin DINOv2), nube original |
| Máscara de Open3D | TSDF estabilizada | todo lo demás |
| Etiquetas DINOv2 | escena MuJoCo y memoria semántica | TSDF, grafo de poses, estructura |
| Simplificación física | escena MuJoCo | nube, TSDF, memoria |
| Salida de GenPC | nada: queda aparte, validada y marcada como inferida | todo |

Hoy la TSDF se reintegra entera cuando cambian las poses: `ScalableTSDFVolume` no permite quitar un frame. Lo que sí se evita es reintegrar cuando las entradas no cambiaron, gracias a la huella. Una TSDF por fragmento permitiría reintegrar solo lo afectado y queda pendiente.
