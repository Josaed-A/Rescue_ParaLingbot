# Marcos de coordenadas de ParaLingbot

Referencia única de convenciones, marcos y transformaciones (etapa 8 de la integración con
Stella-VSLAM). El código que las implementa está en `src/vivo/frames.py`. Las pruebas
están en `test/test_frames_etapa8.py`, y la verificación de punta a punta en
`src/ros/check_frames.py`.

## Convenciones

| | OpenCV / CV (interno del repo) | ROS (REP-103), lo que sale por ROS2 |
|---|---|---|
| eje x | derecha | adelante |
| eje y | abajo | izquierda |
| eje z | adelante (eje óptico) | arriba |
| quiralidad | dextrógira | dextrógira |
| se usa en | LingBot, `.npz` (`extrinsic`, `pose_basic`, `pose_ref_*`), selector, `PoseBuffer`, Stella interno | TF2, `/paralingbot/tracking/*_pose`, `/stella/camera_pose` |

Un vector pasa de CV a ROS con `A = [[0,0,1],[-1,0,0],[0,-1,0]]`, es decir
`x_ros = z_cv`, `y_ros = −x_cv`, `z_ros = −y_cv`. Es la misma matriz que `stella_vslam_ros` llama
`rot_ros_to_cv_map_frame`.

**Poses.** Dentro del repo, una pose es `c2w` (cámara → mundo), 4×4, con la cámara óptica (CV).
Atención: lo que el `.npz` guarda en `extrinsic` es w2c, y `c2w = inv(extrinsic)`. Los nombres de
`demo.py` y `live_server.py` dicen lo contrario; ver la auditoría, § 3.1. Una pose de
`camera_link` en `paralingbot_map` es `A · c2w · Aᵀ`.

## Unidades y escala

Todas las distancias están en **unidades del modelo**, no en metros. La profundidad monocular de
LingBot no tiene escala métrica, y la escala de cada sesión la fija su fase inicial. Stella
monocular tiene **otra** escala, propia de cada mapa suyo, y en la etapa 4 no resultó constante:
varió un 12-21% en ventanas de 5 s. La relación entre los dos mundos es una **Sim(3)**. Llevar todo
a metros necesita una referencia externa (IMU, odometría, LiDAR o una distancia conocida). Es
**UNKNOWN** hasta que el sistema corra en el robot.

## Marcos

```
paralingbot_map ──(estático, identidad)──> paralingbot_odom ──(dinámico, pose de referencia)──> paralingbot_camera_link ──(estático)──> paralingbot_camera_optical
       │
       └──(estático, A)──> paralingbot_map_cv            (mundo interno, ejes CV)

stella_map ── Sim(3) ──> paralingbot_map               NO por TF: /paralingbot/alignment/stella
```

| Marco | Ejes | Origen | Quién lo publica |
|---|---|---|---|
| `paralingbot_map` | ROS, alineados con la cámara del **primer frame** de la sesión | centro óptico del primer frame | estático: el puente ROS2 |
| `paralingbot_odom` | = map | = map | estático identidad: no hay odometría propia; en el robot lo reemplaza el `odom` del robot (REP-105) |
| `paralingbot_camera_link` | ROS (cuerpo de la cámara) | centro óptico | dinámico `odom → camera_link`, uno por frame del modelo, **sellado con el stamp de adquisición**; pose = referencia del selector (`tracking_mode`) |
| `paralingbot_camera_optical` | CV | centro óptico | estático `camera_link → optical` = `A` (cuaternión x, y, z, w = −0.5, 0.5, −0.5, 0.5); `frame_id` de las imágenes |
| `paralingbot_map_cv` | CV del primer frame | = map | estático `map → map_cv` = `A`; es el mundo de las poses `c2w` del repo |
| `stella_map` | ROS de la cámara con que **Stella** inicializó cada mapa | esa cámara | nodo de Stella (`map_frame:=stella_map`); `publish_tf` apagado |
| `stella_camera_link` | ROS | — | `child_frame_id` de `/stella/camera_pose` |

**`stella_map` → `paralingbot_map` es una Sim(3) y va por un topic propio:**
`/paralingbot/alignment/stella`, de tipo `std_msgs/String` con JSON, latcheado (`TRANSIENT_LOCAL`).
El contenido es `{"parent", "child", "segmento", "escala", "R", "t", "formula": "p_parent = escala * R @ p_child + t"}`.
TF2 es rígido: publicarla como TF escondería la escala, y el plan lo prohíbe (regla 9). Se publica
una por cada mapa de Stella anclado por el modo STELLA.

**Nombres propios para no colisionar con GARDIAN.** El robot usa `map`, `odom`, `base_link`,
`camera_link` y `camera_optical_link`. ParaLingbot usa siempre el prefijo `paralingbot_`, y Stella
el prefijo `stella_`.

## Convenciones de cada componente

- **Cámara.** Imagen ya rotada como la ve LingBot; el celular en vertical se gira 90° en la fuente. Recorte de LingBot: 518 de ancho, centrado en alto, que conserva el eje óptico. Las imágenes que recibe Stella son la imagen completa rotada, en `paralingbot_camera_optical`.
- **LingBot.** Cámara OpenCV; mundo = cámara del primer frame de escala; `extrinsic` = w2c. Lo verificado:
  - el frame 0 es la identidad;
  - avanzar por el eje óptico es +z en CV y +x en el mapa;
  - el "abajo" de la cámara es +y en CV.
- **Stella.**
  - Internamente usa la convención CV, con mundo = cámara de inicialización de cada mapa.
  - Su nodo publica `T_ros = A · T_cv · Aᵀ`, la misma conversión que `frames.c2w_cv_to_map_link`; verificado en las pruebas.
  - Etapa 4: tras alinear el mundo, la orientación de Stella difiere de la de LingBot solo 3.4° de mediana. Las convenciones de cámara coinciden.
- **Selector (etapa 6).** Las poses de referencia están en el mundo y la escala de BASIC: `paralingbot_map_cv` y, por TF, `paralingbot_map`.

## Tiempo en ROS2

- Las cámaras en vivo usan la hora del reloj al recibir el frame.
- Carpetas y videos usan stamps que empiezan en 0. Video: el PTS del contenedor. Carpeta: índice / fps.
- **En tf2, el tiempo 0 significa "la transformación más reciente".** El primer frame se buscaba mal; encontrado al verificar la etapa 8. Por eso el puente corre todo lo que publica de esas fuentes con la hora de arranque, `stamp_offset_ros`, guardada en `info.json`. A lo que vuelve de Stella se lo resta. Los stamps internos del repo y del `.npz` no cambian.

## Verificado (etapa 8)

| Comprobación | Resultado |
|---|---|
| Pruebas unitarias: ejes dextrógiros; adelante = +x; girar a la izquierda = +yaw; primer frame = identidad; cadena `map → camera_link → optical` = `A · c2w`; ida y vuelta; conversión de Stella; Sim(3) CV ↔ ROS | 7 pruebas pasan |
| Sesión en vivo (60 frames, BASIC): TF `map ← optical` buscada en el instante de cada frame frente a la pose del `.npz` | 60/60 frames, error máx. 8.7e-8 (precisión float32) |
| TF frente a `/paralingbot/tracking/reference_pose` | error máx. 3e-16 |
| Estáticos `map → odom`, `map → map_cv`, `camera_link → optical` | correctos |
| Sesión en vivo con Stella, modo STELLA: cadena TF frente a `pose_ref_stella` | 86/86 frames, error máx. 2.9e-7 |
| La Sim(3) publicada en `/paralingbot/alignment/stella` aplicada a las poses de Stella reproduce la referencia del modo STELLA | 9/9 frames, error máx. 5e-8 |

## Limitaciones conocidas

- **El mapa no está nivelado con la gravedad.** Su "arriba" es el de la cámara del primer frame. Inclinación medida entre el primer frame y el "abajo" medio del recorrido:

  | Sesión | Inclinación |
  |---|---|
  | fablab | 4.5° |
  | escaleras | 22.9° |
  | pasillos | 32.9° (el primer frame apuntaba al piso) |

  `frames.gravity_tilt_deg` lo mide. Nivelar necesita una IMU, o estimar la vertical, como ya hace `compare_route.py` con el "abajo" medio. Queda para cuando el sistema corra en el robot: un marco `paralingbot_map_level`.
- Las pruebas físicas de la etapa 8 (cámara estática, movimiento lineal, rotación, circuito, regreso al punto inicial) quedan para la matriz de pruebas de la etapa 18, por decisión del usuario. Lo verificado aquí es la consistencia interna de los marcos y su publicación.
