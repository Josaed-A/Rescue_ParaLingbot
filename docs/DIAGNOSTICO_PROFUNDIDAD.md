# Diagnóstico: profundidad, poses o intrínsecos (2026-10-15)

**Pregunta.** ¿De dónde viene el error espacial entre visitas: de la profundidad monocular, de las poses, de la escala, de un desfase o de los intrínsecos? Y en particular, ¿por qué el piso sale en capas?

**Antecedente.** La alineación del piso (`--alinear_piso`, `PIPELINE_OPEN3D_TSDF.md` §7) mejoró el piso, pero la mayoría de los frames tocó el tope de inclinación de 3°. Por eso se sospechó de la profundidad.

**Resumen.**
- **Las poses no son la causa principal.** Una corrección de pose por frame solo «mejora» con giros implausibles (más de 5°) en la mayoría de los frames.
- **La focal variable tampoco:** LingBot-Map predice una focal distinta por frame, con un coeficiente de variación de hasta 8 %, pero corregirla no reduce el error.
- **El error es sobre todo de profundidad, y no es uniforme dentro del frame.** Varía con la distancia y con la fila de la imagen.
- **Fuera del piso, una escala por frame con gradiente por fila (E), o una escala sola (C), explica entre el 15 y el 30 % de la inconsistencia** en píxeles no usados para ajustar.
- **El piso tiene un desfase propio respecto del resto de su frame,** de 1–4 % de la profundidad en la mediana y 5–10 % en el p90. Ningún modelo plausible aprendido fuera del piso lo explica.
- **La alineación del piso existente mejora el piso, pero mueve el frame entero y puede dañar las escaleras:** hasta +83 % de inconsistencia en la windowed.

## 1. Auditoría

| Pregunta | Respuesta |
|---|---|
| ¿Dónde nace la profundidad? | LingBot-Map (`lingbot_map/models/gct_stream*.py`), la cabeza de profundidad, en la misma pasada que la pose y los intrínsecos (`pose_enc` → `pose_encoding_to_extri_intri`). En vivo la guarda `live_server.py`; en windowed, `windowed_lean.py`. |
| ¿Se conserva por frame? | Sí. `eval/<name>.npz` guarda `depth` (N×518×518, float16), `depth_conf`, `intrinsic` y `extrinsic` (w2c) por frame, con `ds` = 1 en las tres reconstrucciones medidas. |
| ¿Cómo se convierte en XYZ? | Cámara estenopeica con la K del propio frame: `x = (u − cx) / fx · z`. Ver `o3d_estable.lift`, `export_dense_cloud.py` y `tsdf_mesh.py` (TSDF de Open3D con la profundidad original). Luego `c2w = inv(extrinsic)`. |
| ¿Cómo se estiman las poses? | Las predice LingBot-Map (BASIC). Stella y HYBRID son solo de referencia. El grafo de poses (`grafo_poses.py`) las conserva salvo corrección verificada. |
| ¿Cómo se corrige el piso? | `o3d_estable.align_floor` modifica la pose (c2w) de cada frame: giro alrededor de los ejes horizontales y desplazamiento vertical. Topes: 3° por eje y 0.3 × profundidad mediana en total. El desplazamiento vertical se informa en el centro del piso del frame. |
| ¿TSDF? | `tsdf_mesh.py --stable`: profundidad original, poses y máscaras de la estabilización. |
| ¿Profundidad, pose y frame coinciden? | Sí. Salen de la misma predicción y del mismo índice, y `version_poses` y `huella_mascara` lo controlan. |
| ¿Intrínsecos? | **No son constantes.** `fx` varía por frame: cv 4.4 % (streaming de hoy), 8.2 % (windowed) y 8.3 % (original; de 520 a 866). La cámara es una sola y sin zoom. |
| ¿Algún problema latente? | Con `ds` > 1 (posible en `windowed_lean.py`), `lift` no divide la K por `ds`. No afecta a ninguna de las reconstrucciones medidas (`ds` = 1). |

## 2. Método

**Herramienta:** `src/mapas/diagnostico_profundidad.py`. Solo lee; no modifica nada.

**Referencia de otros momentos.**
- Cada frame se compara con una superficie hecha con los demás frames: se excluye su bloque temporal (1/8 del recorrido) más un margen de 8 frames, porque sus vecinos comparten sus errores.
- La referencia son vóxeles de profundidad mediana / 150, vistos por al menos 2 frames, con normales.

**Modelos ajustados por frame** (punto a plano, recortado, con correspondencias de umbral decreciente):

| Modelo | Corrección |
|---|---|
| A | ninguna |
| C | escala: `X = c + s (X0 − c)` |
| D | escala y desfase: `z' = s z + t` |
| E | escala que varía con la fila: `z' = (1 + a + b (v − cy) / fy) z`. Se agregó solo porque A–D no explican el piso. |
| F | focal |
| P | pose rígida de 6 grados de libertad |

**Validación.**
- Se ajusta con la mitad de los píxeles (damero de bloques de 32 px) y se mide en la otra mitad.
- Tres variantes: ajustando con todo, sin el piso y solo con el piso, para separar lo que depende del suelo.
- Se mide en los píxeles de prueba que, sin corregir, tienen referencia cerca. El conjunto es el mismo para todos los modelos.
- Un frame sin correspondencias con otros momentos es **inconcluso**.

**Grafo de observaciones** (sin otro optimizador; `<out>.json` → `observaciones`). Cada frame registra:
- el frame de origen (`source_index`) y su focal;
- la versión de las poses usadas y la de las originales;
- la profundidad del `.npz` sin modificar;
- los píxeles por grupo;
- la referencia (bloque y frames excluidos);
- las correspondencias;
- los parámetros de cada modelo, con error estándar y colinealidad;
- el residuo de prueba por grupo (validación);
- la forma del error sin corregir: signo por distancia, por fila y piso contra resto.

Las mediciones y las correcciones quedan separadas.

## 3. Validación con escenas sintéticas (profundidad conocida)

`test/test_diagnostico_profundidad.py` usa render exacto con un solo error inyectado. Residuo relativo en píxeles de prueba:

| Escena | A | C | D | F | P | Gana |
|---|---|---|---|---|---|---|
| Escala de la profundidad por frame (±5 %) | 0.0215 | 0.0095 | **0.0086** | 0.0136 | 0.0120 | D/C; parámetro correlacionado −0.82 con lo inyectado |
| Pose por frame (±2°, unos 5 cm) | 0.0192 | 0.0172 | 0.0171 | 0.0189 | **0.0139** | P |
| Focal por frame (±8 %) | 0.0164 | 0.0131 | 0.0125 | **0.0113** | 0.0126 | F; correlación 0.83 |
| Limpia | 0.0036 | 0.0035 | 0.0029 | 0.0032 | 0.0014 | — |

- **La herramienta atribuye bien la causa,** pero los márgenes son moderados: los demás modelos absorben parte del error. Hay que mirar quién gana y por cuánto.
- **En la escena limpia, P baja unos 0.002** (sesgo de la referencia: cuantización y esquinas). Mejoras de ese orden no significan nada, y P tiene ventaja por sus 6 grados de libertad.
- **Una corrección de profundidad aprendida sin el piso mejora el piso** si el error es de escala de la profundidad. Si el error es de pose, solo lo hace la pose. Esa transferencia es la prueba que separa las dos causas.

## 4. Atribución en los datos reales (desnivel)

Frames concluyentes (con correspondencias con otros momentos del recorrido):

| Reconstrucción | Frames concluyentes | Resultado |
|---|---|---|
| Original | 229 de 317 medidos (uno de cada 2) | concluyente |
| Windowed de hoy | 62 de 115 | concluyente |
| Streaming de hoy | 17 de 125 | **inconcluso**: el vivo graba menos de 2 frames por segundo y el desnivel se recorre casi siempre hacia adelante |

**Cociente contra A** (mediana por frame del residuo de prueba; menos de 1 = mejora; entre paréntesis, los frames). Solo cuentan los ajustes concluyentes y plausibles: escala ±20 %, desfase ≤ 0.2 × profundidad mediana y giro ≤ 5°. Sin ese filtro, D y P «mejoran» con correcciones absurdas: escalas negativas del piso y giros de 10–17°.

Original (desnivel de 735 frames):

| Ajuste | Modelo | Total | Piso | Pared | Escalera | Otro |
|---|---|---|---|---|---|---|
| con todo | C (escala) | 0.81 (199) | 0.91 | 0.88 | 0.85 | 0.90 |
| | D (escala + desfase) | 0.73 (117) | 0.87 | 0.84 | 0.90 | 0.81 |
| | E (escala por fila) | 0.70 (182) | 0.81 | 0.77 | 0.87 | 0.78 |
| | F (focal) | 0.97 (156) | 0.99 | 0.99 | 1.00 | 0.99 |
| | P (pose) | 0.72 (**64**) | 0.72 | 0.88 | 0.88 | 0.88 |
| **sin el piso** | C | 0.81 (182) | **1.00** | 0.84 | 0.88 | 0.77 |
| | E | 0.78 (163) | **1.02** | 0.77 | 0.81 | 0.70 |
| | P | 0.87 (43) | **1.09** | 0.85 | 0.82 | 0.85 |
| solo el piso | E | 0.82 (102) | 0.78 | 0.88 | 1.01 | 0.90 |
| | P | 0.68 (61) | 0.67 | 0.96 | 0.96 | 0.92 |

**Windowed:** la misma tendencia, más débil.
- E sin el piso: pared 0.89, escalera 0.89, otro 0.83, piso 0.94.
- C, alrededor de 0.94.
- P: solo 5 frames con un ajuste plausible.

**Forma del error sin corregir** (residuo con signo / profundidad, frames concluyentes del original):
- **Dentro de cada frame,** la diferencia entre lejos y cerca (mediana de |·|) es 0.023 y entre abajo y arriba, 0.020. Son del orden del residuo medio (0.012–0.015). En la windowed, 0.043 y 0.046. **El error no es una escala única por frame:** cambia con la distancia y con la fila.
- **El piso contra el resto del mismo frame:** |·| mediano 0.009 y p90 0.047 en el original; 0.043 y 0.098 en la windowed. El signo varía entre frames: es un **desfase propio del piso, distinto en cada frame**.

**Qué se atribuye a qué:**

| Factor | Veredicto |
|---|---|
| Intrínsecos (focal variable) | **Descartado.** F no reduce el error (0.92–0.99), y la escala estimada no correlaciona con la focal del frame (|r| < 0.2). |
| Poses | **No son la causa principal.** Solo 64 de 229 frames admiten una corrección de pose plausible. Ajustada sin el piso, empeora el piso (1.09–1.25), y no se comporta como el error de pose sintético. |
| Profundidad: escala o gradiente por frame | **Parcial.** Explica 15–30 % del residuo fuera del piso. Se identifica sin el piso y no daña el piso. |
| Profundidad: desfase (D) | **No identificable en la mayoría de los frames.** Escala y desfase son casi colineales (mediana 0.997) porque el rango de profundidades de un frame es estrecho. |
| Piso | **Error propio del piso** (desfase e inclinación por frame). Ninguna corrección aprendida fuera del piso lo explica. Solo lo reduce algo que mire el piso: P o E solo con el piso, o la alineación del piso. Los ajustes de profundidad del piso degeneran (escalas negativas): **no se corrige el piso por profundidad**. |
| Resto, no explicado por modelos globales | La mayor parte: errores locales dentro del frame. No se modelan sin más evidencia. |

## 5. Evaluación de los métodos (mismos frames, residuo entre visitas de la versión resultante)

**Alineación del piso existente (B)**, con el modelo A medido con las poses de B y comparado con las originales:

| Reconstrucción | Total | Piso | Pared | Escalera | Otro |
|---|---|---|---|---|---|
| Original (225 frames) | 0.83 | **0.58** | 0.87 | 1.06 | 0.92 |
| Windowed (62) | 0.89 | **0.47** | 0.89 | **1.83** | 0.96 |
| Streaming (17, inconcluso) | 1.09 | 1.12 | 0.90 | 0.92 | – |
| Multivía con guardas, streaming (17, inconcluso) | 1.24 | 1.00 | – | 1.56 | – |

B mejora mucho el piso, pero, como gira el frame entero, **puede dañar las escaleras**: +83 % en la windowed.

**Los cinco métodos sobre las mismas sesiones** (cociente contra el original del residuo entre visitas en píxeles de prueba; espesor local de las superficies y piso, de `o3d_estable`):

| Reconstrucción | Método | Total | Piso | Pared | Escalera | Otro | Espesor local de superficies | Piso: celdas separadas, espesor |
|---|---|---|---|---|---|---|---|---|
| Original (735 frames, 226 medidos) | 1. original | 1 | 1 | 1 | 1 | 1 | 0.00417 | 81.5 %, 0.063 |
| | 2. piso alineado (B) | 0.83 | **0.58** | 0.87 | 1.06 | 0.92 | 0.00436 | 40.1 %, 0.027 |
| | 3. multivía con guardas | rechazado (se aparta 13.6° de LingBot) = original | | | | | | |
| | 4. **profundidad corregida** | 0.80 | 0.96 | **0.75** | **0.66** | **0.71** | **0.00360** | 81.5 %, 0.064 |
| | 5. **profundidad corregida + B** | **0.65** | **0.55** | **0.74** | 0.78 | 0.85 | **0.00336** | **40.9 %, 0.027** |
| Windowed (62) | 2. B | 0.89 | **0.47** | 0.89 | **1.83** | 0.96 | 0.00280 | 66.4 %, 0.042 |
| | 3. multivía | rechazado (4.1°) = original | | | | | | |
| | 4. **profundidad corregida** | 0.95 | 1.00 | **0.88** | **0.61** | 0.94 | **0.00263** | 97.6 %, 0.073 |
| | 5. profundidad corregida + B | **0.82** | 0.55 | 0.94 | 1.21 | 1.15 | 0.00274 | 67.0 %, 0.044 |

En el método 1, el espesor local es 0.00273 en la windowed, y el piso tiene 97.7 % de celdas separadas y espesor 0.072. El espesor local está en unidades × profundidad mediana: menos es más fino.

**Dónde actúa la corrección de profundidad:**

| | Total | Piso | Pared | Escalera | Otro |
|---|---|---|---|---|---|
| Original, frames corregidos (154) | 0.74 | 0.95 | **0.62** | **0.50** | 0.68 |
| Original, frames no corregidos (76) | 0.94 | 0.97 | 0.94 | 0.85 | 0.89 |
| Windowed, frames corregidos (26) | 0.89 | 1.01 | **0.79** | **0.51** | 0.91 |

Los frames no corregidos también mejoran un poco, porque su referencia es más coherente.

**Cuánto cambia la profundidad** en los frames corregidos: |Δz|/z mediana 5.5–6 %, p90 9–16 %. Se corrigen 298 de 735 frames en el original (241 con E y 57 con C) y 27 de 136 en la windowed (11 con E y 16 con C).

**Costo** (CPU, sin GPU):

| | Diagnóstico y corrección | Flujo completo con la profundidad corregida |
|---|---|---|
| Original | 14 min, 4.9 GB de RAM | filtro, estable, piso, nubes y mallas: 29 min, 9.8 GB de pico (el filtro) |
| Windowed | 45 s, 1.2 GB | 7 min |

**Variante descartada (resultado negativo): D por clase, incluido el piso.**
- El ajuste D del piso aceptó escalas negativas (1 + s ≈ −0.1 a −0.4) que aplanan el piso a una profundidad constante.
- Sobre la windowed, el p90 de la diferencia entre visitas del piso empeoró: 0.39 → 0.56.
- Sus informes están en `exports/descartado_soloD/`.
- De ahí salieron los filtros de plausibilidad, y la decisión de no corregir el piso por profundidad.

## 6. Qué se implementó y por qué

- **`diagnostico_profundidad.py`.** Herramienta de diagnóstico y grafo de observaciones; solo lee.
- **`--corregir <npz>` y tarea `prof` de `build_maps`.** Corrección de la profundidad por frame, **solo fuera del piso**: el primer modelo aceptado entre E y C, ajustado sin el piso contra otros momentos del recorrido. Se acepta solo si:
  - tiene correspondencias suficientes;
  - es identificable: colinealidad ≤ 0.995, error estándar ≤ 0.02;
  - es plausible: escala ±20 %;
  - baja al menos un 5 % el residuo de **sus propios píxeles de prueba**.
  
  Los frames sin otro momento con que compararse **no se corrigen**.
- **El resultado se escribe como un `.npz` nuevo** (`eval/<name>_prof.npz`): el original no se toca. Lleva `correccion_profundidad`, con el modelo, los parámetros y el motivo de aceptación o rechazo de cada frame, el `.npz` de origen, la versión de las poses y los criterios.
- **Se usa con las tareas de siempre:** `--npz eval/<name>_prof.npz --tasks filtro,estable,...`. Los artefactos quedan aparte (`<name>_prof_*`), y la tarea `prof` se rehace si el `.npz` de origen es más nuevo.
- **En el visor,** las capas llevan «· profundidad corregida».
- **Lo que no cambia:**
  - `estable` sigue siendo el modo por defecto;
  - el gemelo de MuJoCo sigue usando las mismas poses (esta corrección no cambia poses);
  - la TSDF sigue siendo posterior a la estabilización geométrica;
  - no hay dependencias nuevas.
- **Pruebas:** `test/test_diagnostico_profundidad.py`.
  - Atribución: un error de escala de profundidad lo explica la profundidad, y un error de pose, la pose.
  - El `.npz` corregido: el original intacto, el piso sin corregir, E o C, y un ajuste implausible rechazado.

**Recomendación:**
- **Profundidad corregida sola (método 4):** mejora paredes, escaleras y objetos en las dos reconstrucciones, sin empeorar el piso ni nada más. Es la corrección mínima respaldada.
- **Combinada con la alineación del piso (método 5):** es la mejor en el original, pero en la windowed todavía empeora escaleras y objetos. La alineación del piso conviene **solo si el diagnóstico confirma, en esa reconstrucción, que no empeora ningún grupo**. Agregar esa validación automática a la tarea `piso` es el siguiente paso.

## 7. Limitaciones

- **Escala sin calibrar:**
  - todo se mide relativo a la profundidad y sin centímetros como referencia absoluta;
  - una escala global errónea no se detecta: solo se ven las inconsistencias entre frames;
  - la escala por frame es relativa a la referencia, que tiene sus propios errores.
- **Cobertura:** la corrección solo es posible donde hay revisitas. En el vivo (menos de 2 frames por segundo) hay pocas: el streaming de hoy es **inconcluso**, con 17 frames.
- **Error no modelado:** la mayor parte del error es local dentro de cada frame. E y C explican entre el 15 y el 30 %.
- **D no es identificable en la mayoría de los frames.** Un modelo más rico (por zonas de la imagen) necesitaría más evidencia y más correspondencias.
- **El piso sigue dependiendo de la alineación del piso,** que mueve frames enteros y cuyo efecto sobre las escaleras depende de la reconstrucción.
- **Una sola pasada:** la referencia de la corrección es la de la profundidad sin corregir; no se itera.
- **El p90 de la diferencia entre visitas del piso incluye escalones reales,** así que se privilegian la mediana, las celdas separadas y el residuo por grupo.

## 8. Criterios de aceptación

| Criterio | Cómo se cumple |
|---|---|
| 1. Atribución | §4: intrínsecos descartados; poses no principales; profundidad parcial y no uniforme; piso con error propio. |
| 2. Experimentos reproducibles | Herramienta y pruebas sintéticas con error conocido; mismas sesiones; comandos en §9. |
| 3. No depende solo del suelo | La corrección se ajusta **sin** el piso. |
| 4. Se conservan paredes, puertas, escaleras y obstáculos | Mejoran (0.62–0.88), y la escalera más (0.50–0.66). Puertas: sin píxeles de prueba suficientes, **no medido**. |
| 5. Datos no usados | Se mide en la otra mitad de los píxeles (damero) y por grupo, ajustando sin ese grupo. |
| 6. Limitaciones de escala | §7. |
| 7. El flujo sigue funcionando | 64 pruebas de ParaLingbot y 58 de Pruebas_Macondo; `estable` sin cambios. |
| 8. Resultados negativos | El multivía rechazado o dañino; la alineación del piso dañando escaleras; D del piso degenerado; la focal descartada; el streaming inconcluso. |

## 9. Reproducir

```bash
# diagnóstico (solo lee): con las poses originales, o con las de una estabilización
python3 src/mapas/diagnostico_profundidad.py <prueba>/eval/<name>.npz --filter <prueba>/exports/estructura/<name>_filtro.npz \
    --out <prueba>/exports/diagnostico/<name>_original [--poses <..>_estable.npz] [--every 2]
# corrección + flujo de siempre sobre el .npz corregido
python3 src/mapas/build_maps.py <prueba> --npz eval/<name>.npz --tasks filtro,prof
python3 src/mapas/build_maps.py <prueba> --npz eval/<name>_prof.npz --tasks filtro,estable,piso,nube_e,malla_e,nube_p,malla_p
```
