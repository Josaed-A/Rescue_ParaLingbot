# Escena física para MuJoCo (nivel D) y su validación

**Dónde está:** en Pruebas_Macondo.
- Construcción: [`vision/mapa_mujoco.py`](https://github.com/Josaed-A/Pruebas_Macondo/blob/estado-dinov2-mujoco-paralingbot/vision/mapa_mujoco.py).
- Evaluación: [`vision/evaluar_mapeo.py`](https://github.com/Josaed-A/Pruebas_Macondo/blob/estado-dinov2-mujoco-paralingbot/vision/evaluar_mapeo.py).
- Estado previo y su evaluación: `docs/evaluacion_mapeo.md` de esa rama.

## Qué colisiona y qué no

El modelo físico se simplifica aparte de la nube observada y de la malla TSDF. Cada entidad física queda registrada en `escena.json → capas.fisico`, con sus geoms, si colisiona, su origen, su estado y su simplificación.

| Entidad | Colisión | Origen | Simplificación |
|---|---|---|---|
| Suelo | sí | nivel de piso más bajo | plano |
| Paredes de LingBot | sí | estructura del filtro geométrico (solo si es de las mismas poses) | cajas de 6 cm |
| Losas de piso de LingBot | sí | estructura del filtro | losas de 4 cm |
| **Superficies de apoyo** (nuevo) | sí | vóxeles de piso y escalera de DINOv2 | losas de 4 cm por celda de 10 cm y nivel (niveles a más de 30 cm son distintos), fundidas en rectángulos |
| Obstáculos (muebles, electrodomésticos…) | sí | componentes conexos de vóxeles de DINOv2 | envolventes convexas por piezas de 0.5 m |
| **Escaleras** (cambio) | no como envolvente | — | van en las superficies de apoyo, peldaño por peldaño |
| **Puertas** (cambio) | no | — | visual: el paso queda abierto, porque si estaba abierta o cerrada no se observa con certeza |
| **Personas** (cambio) | no | — | visual: agente dinámico |
| Vóxeles semánticos y de color | no | — | visual |

Antes, la envolvente convexa de una escalera rellenaba el espacio sobre los peldaños por donde se camina. Era la colisión espuria más frecuente.

**La X2 en el recorrido (`x2_recorrido.py`)** sigue caminando sobre un plano mocap a la altura del piso de DINOv2, y el mapa es decorado para ella. Las cero caídas no validan esta geometría: la validan las métricas de abajo.

## Cómo se valida

El recorrido de la cámara es espacio libre conocido: quien grababa pasó por ahí. El evaluador mide:

- **Colisión espuria:** toda geometría de colisión que ocupe el cuerpo de quien graba (un eje vertical 25 cm detrás de la cámara, de 0.2 a 1.1 m bajo ella). Para las superficies de apoyo, solo hasta 0.5 m bajo la cámara, porque se pisan.
- **Recorrido sin piso de colisión a la altura de una mano** (0.9–1.9 m bajo la cámara).
- **Huella en planta de las envolventes** frente a la de sus vóxeles.
- **Coherencia con la estructura:** paredes, inclinación del piso y deriva vertical al volver.

## Resultados (sesión 1 del desnivel)

| Escena | Geoms con colisión (paredes / losas / soporte / envolventes) | Recorrido con colisión espuria | Recorrido sin piso de colisión |
|---|---|---|---|
| Streaming, modelo anterior | 8 / 4 / 0 / 107 | 2.8 % (escaleras 2.4 %) | 57 % |
| Streaming, modelo nuevo | 8 / 4 / 1067 / 25 | **1.0 %** (soporte 0.8 %, losas 0.4 %) | 54 % |
| Streaming, modelo nuevo, `--poses estable` | 8 / 4 / 1062 / 25 | **1.0 %** | 54 % |
| Windowed, modelo anterior | 8 / 4 / 0 / 182 | 5.9 % (escaleras 4.2 %, paredes 1.2 %) | 42 % |
| Windowed, modelo nuevo, `--poses estable` | 8 / 4 / 1549 / 57 | **1.8 %** (paredes 1.2 %, losas 0.5 %) | **30 %** |

- **Escaleras:** desaparece su obstrucción.
- **Pared en el camino (windowed):** sigue la pared del filtro de LingBot que cae sobre el recorrido.
- **Piso faltante:** el que queda (30–54 %) son tramos donde nadie vio el piso bajo la cámara, porque se subía la escalera mirando hacia arriba. Es una región desconocida y no se rellena; hacerlo sería inventar geometría.

Pruebas: `tests/test_memoria_semantica.py`. Comprueban que las losas de apoyo conservan dos pisos superpuestos en planta y cada peldaño, que no rellenan el aire, y que escaleras, puertas y personas quedan sin colisión.

## Prueba de punta a punta (fablab, video nuevo)

El grafo corrigió las poses. Los gemelos se construyeron con las poses estabilizadas y sin el filtro ni la estructura de ParaLingbot, que son de las poses BASIC.

| | Streaming | Windowed |
|---|---|---|
| Recorrido con colisión espuria | 0 % | 0 % |
| Recorrido sin piso de colisión | 2 % | 1 % |
| Inclinación del piso | 0.6° | 1.4° |
| Tramo caminable para el X2 | 13.6 m | 13.0 m |

**Advertencias:**
- **Sin paredes:** no hay paredes de colisión, porque la estructura es de otras poses. El 0 % no prueba que las paredes estén bien: no hay paredes.
- **Losa espuria:** una losa de apoyo a 0.85 m bajo el recorrido (probablemente una mesa etiquetada como piso).

## Desnivel en vivo, pasillo y desnivel original (2026-10-15)

| Escena | Paredes con colisión | Recorrido con colisión espuria | Sin piso de colisión | X2 caminable |
|---|---|---|---|---|
| Desnivel en vivo, streaming (134 frames) | 12 | 2.6 % (soporte 1.7 %) | 51 % | 29.5 m |
| Desnivel en vivo, windowed | 10 | **11.3 %** (soporte 10.8 %) | 58 % | 20.1 m |
| Pasillo (prueba 1), streaming (106 frames) | 0 (el filtro no halló paredes) | 2.7 % (losa de LingBot) | 25 % | 65.7 m |
| Pasillo, windowed | 0 | 0.2 % | 15 % | 64.6 m |
| Desnivel original (735 frames), modelo anterior | 12 | 9.1 % (pared 6.3 %, escaleras 5.8 %) | 50 % | 31.1 m |
| Desnivel original, `--poses estable` y modelo nuevo | 12 | **6.7 %** (pared 6.2 %, soporte 1.3 %) | **44 %** | 31.4 m |

- **Desnivel original:** el grafo rechazó la corrección (deformaría la odometría), así que el filtro y la estructura siguen siendo compatibles. Lo que queda de colisión espuria es la pared de LingBot en la zona de la escalera.
- **Desnivel windowed en vivo:** las losas de apoyo cruzan el cuerpo de quien graba. Falta averiguar si vienen de etiquetas de DINOv2 o de niveles mal ubicados. Validarlas contra la geometría (horizontales, área continua, altura coherente con el recorrido) es el siguiente paso, junto con las paredes del nivel B.

## Qué falta para que el entorno sea apto para entrenar locomoción

1. **Escala calibrada.** La de hoy es Depth Anything V2, sin calibrar.
2. **Paredes desde la nube**, por ejemplo las del nivel B, que están en las poses estabilizadas. Hoy solo colisionan las 8–16 paredes del filtro, el 70–80 % de la superficie de pared vista no colisiona, y si el grafo corrige las poses no colisiona ninguna.
3. **Piso en las regiones no observadas,** con incertidumbre explícita y sin inventarlo.
4. **La X2 chocando con el mapa** en las pruebas de recorrido.
