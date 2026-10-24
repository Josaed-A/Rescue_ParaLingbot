# Superficies duplicadas y grafo estructural (nivel B)

**Código:** [`src/mapas/grafo_estructura.py`](../src/mapas/grafo_estructura.py), llamado por `o3d_estable.py`.

**Salida:**
- `<name>_estructura_grafo.json`: entidades y relaciones;
- `<name>_estable.json → diagnostico.{original,estable}`: pares y pilas, espesor local;
- `<name>_duplicadas.ply`: los pares duplicados en rojo y azul, para mirarlos.

**Nada de esto modifica la geometría.** Una superficie duplicada se marca, no se funde ni se borra. Para corregir un duplicado hay que corregir las poses (nivel A).

## Entidades

**Cómo salen.** Planos dominantes por RANSAC iterativo sobre los vóxeles (umbral de 1.5 vóxeles, hasta 40 planos). De los inliers de cada plano queda **la componente conexa más grande**, medida en el plano.

Esto evita un error real que tuvo la primera versión. RANSAC armó un «plano» a 0.24 de altura, inclinado 2°, que juntaba la cara de arriba de un escalón con franjas sueltas de otras superficies que cruzaban esa altura. Medía 5 × 7 m y se superponía al piso, así que el escalón salió como «duplicado» del piso.

**Qué guarda cada entidad:**
- identificador persistente y versión de poses;
- tipo: pared, piso, techo, escalera, puerta o plano, según su orientación y el grupo semántico mayoritario del filtro;
- normal orientada hacia las cámaras que la vieron, centro, extensión y área;
- incertidumbre (RMS de los inliers) y confianza;
- frames que la respaldan y procedencia;
- estado: `valida`, `posible_duplicado` o `ambigua`.

## Cómo se clasifica un par de superficies paralelas

Un par es candidato si se cumple todo esto:
- normales a menos de 10°;
- separación entre 3 vóxeles y 0.3 × la profundidad mediana;
- solape en planta de al menos 20 % de la menor.

| Clase | Evidencia | Relación |
|---|---|---|
| `duplicada` | misma cara (las cámaras que las vieron están del mismo lado) **y** sus puntos vienen de momentos distintos del recorrido (intersección de los histogramas de frames < 0.5) | `POSIBLE_DUPLICADO_DE` |
| `paralelas_fisicas` | caras opuestas que se miran: las cámaras estuvieron entre las dos | `PARALELA_A` |
| `dos_caras` | caras opuestas de espaldas: hubo cámaras a los dos lados | `DOS_CARAS_DE` |
| `objetos_distintos` | grupos semánticos distintos y confiables (> 60 % cada uno) | `CERCA_DE` |
| `ambigua` | misma cara vista a la vez: un escalón real, ruido de profundidad o un objeto delante | `AMBIGUA_CON` |

Ninguna superficie se une solo por estar cerca.
- **Dos paredes reales de un pasillo:** suelen quedar fuera por separación. Si entran, son `paralelas_fisicas`.
- **El marco de una puerta y la pared:** quedan como pares distintos.
- **Los pares ambiguos:** conservan su incertidumbre.

Los planos unidos por pares forman **pilas**: una superficie partida en capas. La clase de una pila es la más fuerte de sus pares; las cifras de superficies que se reportan son pilas, no pares.

Otras relaciones:
- `ADYACENTE_A`: planos no paralelos que se tocan a menos de 2.5 vóxeles.
- `SOPORTA`: un piso bajo una pared que llega hasta él.
- `OBSERVADO_DESDE`: fragmentos del grafo de poses que aportan al menos el 5 % de los frames de la entidad.

## Pruebas sintéticas

| Caso | Resultado |
|---|---|
| Tabique de 0.15 visto de los dos lados | `dos_caras`, separación medida 0.142 |
| Escalón real de 0.25 | no aparece como duplicado (después de la corrección de componentes conexas) |
| Vuelta con deriva (circuito) | 5 superficies `duplicada` en la nube original → 0 en la estabilizada |
| Habitación limpia | ningún duplicado |

## Sesiones reales

Ver [PIPELINE_OPEN3D_TSDF.md](PIPELINE_OPEN3D_TSDF.md#resultados-sobre-sesiones-reales).

**Lectura general.** En las sesiones reales del desnivel el grafo no encontró cierres verificables y conservó las poses. Por eso las superficies `duplicada` de la original siguen en la estabilizada. Están detectadas y marcadas, pero no se corrigen sin evidencia. La bitácora del 2026-10-04 ya había visto que las paredes dobles de estos videos son frames enteros corridos; quitar puntos no las resuelve.
