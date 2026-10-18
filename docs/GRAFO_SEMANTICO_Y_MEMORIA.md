# Grafo semántico y memoria espacial (nivel C)

**Dónde está:** en el gemelo de Pruebas_Macondo, [`vision/memoria_semantica.py`](https://github.com/Josaed-A/Pruebas_Macondo/blob/estado-dinov2-mujoco-paralingbot/vision/memoria_semantica.py).

**Qué hace:** mantiene la identidad de los objetos que DINOv2 reconoce sobre la geometría de ParaLingbot, separada de la optimización geométrica. La semántica nunca mueve la geometría ni entra al grafo de poses.

## Qué es un objeto

**Qué es.** Un componente conexo de vóxeles de la misma clase de DINOv2 (`mapa_mujoco.extract_objects`), sobre la nube que el gemelo levanta con la profundidad de LingBot y las poses de la escena. DINOv2 se usa como fuente de rasgos y de clases: no hay un detector ni un seguidor de instancias.

**Qué persiste.** Las **observaciones** de cada objeto, no su posición: los píxeles (frame, y, x) de los que salen sus puntos. Hay hasta 3000 por objeto, en `memoria_obs.npz`. De ellas salen:

| Atributo | Cómo | Depende de las poses |
|---|---|---|
| Identidad | `sha1` de los frames (`source_index`) y celdas de 16 px de sus observaciones | no |
| Posición y extensión | levantar la profundidad de sus píxeles con una versión de poses y llevarla al marco nivelado | sí: hay una por versión (`posiciones[versión]`) |
| Descriptor visual | rasgos de DINOv2 base (última capa) promediados sobre sus píxeles en sus 3 frames con más observaciones | no |
| Clase | la votada por vóxel entre vistas | no (cambia si cambian las etiquetas) |

Un mismo objeto visto en muchos frames es **un** nodo con muchas observaciones, no un objeto por frame.

## Relaciones

**Dentro de una memoria:**

| Relación | Criterio |
|---|---|
| `OBSERVADO_EN` | objeto → fragmento del grafo de poses (nivel A) cuyos frames caen en el intervalo del objeto |
| `CERCA_DE` | centros a menos de 1 m |
| `SOBRE` | se solapan en planta y la base de uno está a menos de 15 cm del tope del otro |
| `ASOCIADO_A_ESTRUCTURA` | el objeto está sobre el plano de una entidad estructural (nivel B), dentro de su extensión. Solo si la estructura es de la misma versión de poses; si no, se anota un `AVISO`. |
| `CONTRADICCION` | dos clases distintas en el mismo lugar (IoU de cajas > 0.5) |

**Entre dos memorias** (`asociar`):

| Relación | Criterio |
|---|---|
| `MISMO_OBJETO_QUE` | misma clase, posición compatible en un marco común **y** similitud visual ≥ 0.75. Uno a uno, por mejor evidencia combinada. |
| `ALTERNATIVA` | también compatible, pero el otro lado ya tiene una pareja mejor: la identidad queda incierta, no se fuerza |
| `CANDIDATO_GEOMETRICO` | mismo lugar y clase, sin confirmación visual |
| `PENDIENTE` | parecido visual sin un marco común para comprobar la geometría |
| `CONTRADICCION` | se parecen pero están lejos, aun con la tolerancia del registro: otro objeto igual (puertas) o un registro malo |

**La similitud de DINOv2 nunca es razón suficiente para unir dos objetos.** La tolerancia de distancia es `max(0.5 m, 0.5 × tamaño) + 2 × incertidumbre del registro`. Con `--alinear-camaras`, la incertidumbre es el residuo de la alineación Sim(3) de las cámaras que vieron cada objeto, en los frames compartidos.

## Propagación de cambios

| Cambia | Qué hace la memoria |
|---|---|
| Poses (otra versión de la misma sesión) | `reubicar` recalcula las posiciones desde las observaciones. Identidad, observaciones y descriptores no cambian, y no se corre DINOv2. |
| Etiquetas DINOv2 | `construir` de nuevo (la TSDF no se toca) |
| Otra sesión | `asociar`, con un marco común explícito o con candidatos pendientes |

## Resultados (sesión 1 del desnivel, 14/10)

| Prueba | Resultado |
|---|---|
| Construir: streaming / windowed, `--poses estable` | 31 / 52 objetos, iguales a los de la escena (reconstrucción determinista), en 7–8 s cada una con DINOv2 en GPU |
| Relaciones (windowed) | 168 `OBSERVADO_EN`, 51 `CERCA_DE`, 26 `ASOCIADO_A_ESTRUCTURA`, 8 `SOBRE` |
| Reubicar: BASIC → HYBRID | 28 objetos con la misma identidad, desplazados 0.44 m de mediana y 0.82 m como máximo: la magnitud de la mezcla de poses medida (0.87 m) |
| Asociar streaming ↔ windowed sin marco común | 44 `PENDIENTE`, ninguno unido solo por parecido visual |
| Asociar con las cámaras alineadas (135 frames, residuo 0.96 m mediano, 1.66 m p90) | 19 `MISMO_OBJETO_QUE`, 15 `ALTERNATIVA`, 10 `CONTRADICCION`, 67 `CANDIDATO_GEOMETRICO` |

Las dos reconstrucciones del mismo recorrido difieren cerca de 1 m entre sí. Con esa incertidumbre, la identidad de los objetos repetidos (escaleras, puertas) suele ser ambigua, y la memoria lo deja así en lugar de fundirlos.

Pruebas: `tests/test_memoria_semantica.py`. Cubren:
- identidad independiente del orden y de las poses;
- posiciones que siguen un movimiento rígido;
- asociación que exige geometría y visión;
- `PENDIENTE` sin marco común.

## Limitaciones

- No hay seguimiento de instancias. Dos objetos iguales que se tocan forman un solo componente conexo, y un objeto partido por una oclusión pueden ser dos.
- El descriptor promedia rasgos de parches de 14 px. Sirve para descartar o confirmar, no para reidentificar con precisión.
- La identidad depende de las observaciones: si cambian las etiquetas, los objetos cambian y hay que volver a asociar.
- `ASOCIADO_A_ESTRUCTURA` depende de que el nivel B tenga la estructura en la misma versión de poses.
