# Matemática del proyecto

Este documento reúne las fórmulas que usa el repositorio, de punta a punta: qué predice el modelo y cómo, cómo se pasa de esas predicciones a nubes, mallas y splats, qué hace cada filtro y con qué se mide cada cosa. Cada sección dice **para qué** se usa y en qué archivo está.

Los resultados medidos (números, comparaciones, decisiones) no están aquí: están en la [bitácora del README](README.md#bitácora-técnica-de-la-investigación).

**Índice**

1. [Notación y convenciones de cámara](#1-notación-y-convenciones-de-cámara)
2. [El modelo: LingBot-Map](#2-el-modelo-lingbot-map)
3. [De la profundidad a los puntos](#3-de-la-profundidad-a-los-puntos)
4. [Nube fusionada por vóxel](#4-nube-fusionada-por-vóxel)
5. [Malla por fusión TSDF](#5-malla-por-fusión-tsdf)
6. [Gaussian Splatting](#6-gaussian-splatting)
7. [Análisis de frames y context-to-image](#7-análisis-de-frames-y-context-to-image)
8. [Filtro geométrico previo a la malla y al splat](#8-filtro-geométrico-previo-a-la-malla-y-al-splat)
9. [Métricas de evaluación](#9-métricas-de-evaluación)
10. [Resumen: qué modelo o método, para qué](#10-resumen-qué-modelo-o-método-para-qué)

---

## 1. Notación y convenciones de cámara

**Imagen.** Un frame es $I:\Omega\to[0,1]^3$, con $\Omega=\{0,\dots,W-1\}\times\{0,\dots,H-1\}$ y $W=H=518$ después del recorte del modelo. Un píxel es $\mathbf{u}=(u,v)$.

**Cámara pinhole.** Intrínsecos

$$
K=\begin{pmatrix} f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0&0&1\end{pmatrix},
\qquad c_x=\tfrac{W}{2},\; c_y=\tfrac{H}{2}.
$$

**Convención OpenCV:** $x$ a la derecha, $y$ hacia abajo, $z$ hacia adelante. La consecuencia práctica es que el eje $+Y$ de la cámara, llevado al mundo, apunta al suelo. Varias herramientas lo usan para saber dónde está "abajo" (secciones 8.4 y 9).

**Pose.** Una transformación rígida $T=\begin{pmatrix}R&\mathbf t\\0&1\end{pmatrix}\in SE(3)$. Hay dos sentidos:

- $T_{wc}$ (cámara → mundo, *c2w*): su columna $\mathbf t$ es el centro óptico $\mathbf c$ en el mundo.
- $T_{cw}=T_{wc}^{-1}$ (mundo → cámara, *w2c*).

**Convención del repositorio.** Lo que se guarda en `extrinsic` de cada `.npz` se trata como *w2c*, y el mundo se obtiene invirtiéndolo (`c2w = inv(E)`). Es la misma convención que usan el visor de `lingbot_map` y `demo_render`. Usarla al revés da trayectorias sin sentido (bitácora del 2026-09-17).

**Proyección** de un punto del mundo $\mathbf X$ en la cámara $j$:

$$
\mathbf X_c = R_j\mathbf X+\mathbf t_j,\qquad
u=f_x\frac{X_c}{Z_c}+c_x,\quad v=f_y\frac{Y_c}{Z_c}+c_y,\quad z=Z_c .
$$

**Retroproyección** de un píxel con profundidad $D(\mathbf u)$ (distancia sobre el eje óptico, no a lo largo del rayo):

$$
\mathbf X_c=D(\mathbf u)\,K^{-1}\begin{pmatrix}u\\v\\1\end{pmatrix}
=D\begin{pmatrix}(u-c_x)/f_x\\ (v-c_y)/f_y\\ 1\end{pmatrix},
\qquad \mathbf X=R_{wc}\mathbf X_c+\mathbf c .
$$

**Escala.** La profundidad monocular es ambigua en escala: todas las magnitudes del mapa están en "unidades del modelo", no en metros. Por eso los umbrales geométricos del repositorio son **relativos**: a la profundidad mediana $\tilde D$, a la diagonal de la escena, o al largo del recorrido.

---

## 2. El modelo: LingBot-Map

**Para qué:** es el único modelo que mira la geometría. A partir de video RGB estima, para cada frame, la **pose de la cámara**, sus **intrínsecos**, la **profundidad por píxel** y una **confianza por píxel**. Todo lo demás del repositorio trabaja sobre esas cuatro salidas. Código: `lingbot_map/` (upstream, sin cambios de arquitectura).

### 2.1 Entrada y tokens

Cada frame se reescala a 518 px de ancho y se recorta centrado a $518\times518$ (modo `crop` de `load_and_preprocess_images`). Un video vertical $1080\times1920$ pierde así $1-518/924\approx44\%$ de su alto. Con parches de $14\times14$ quedan

$$
\frac{518}{14}\times\frac{518}{14}=37\times37=1369 \text{ tokens por frame},
$$

contra $37\times28=1036$ en un frame horizontal $518\times392$: un 32% más de memoria de atención por frame.

### 2.2 Codificador: DINOv2 ViT-L/14

Cada parche $\mathbf p_k\in\mathbb R^{14\cdot14\cdot3}$ se proyecta linealmente, $\mathbf x_k=E\,\mathbf p_k+\mathbf e^{pos}_k$, con $d=1024$. Después vienen 24 bloques transformer de atención multi-cabeza:

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V,\qquad Q=XW_Q,\;K=XW_K,\;V=XW_V,
$$

con $d_h=64$ y 16 cabezas. Cada bloque es $X\leftarrow X+\mathrm{Attn}(\mathrm{LN}(X))$, seguido de $X\leftarrow X+\mathrm{MLP}(\mathrm{LN}(X))$. El codificador da descriptores visuales robustos, preentrenados de forma auto-supervisada. Es el 26% de los parámetros del modelo (bitácora del 2026-08-23).

### 2.3 Agregador GCT: atención por frame y global, con RoPE 3D

Sobre los tokens de todos los frames se alternan dos tipos de bloque (24 de cada uno, el 52% de los parámetros):

- **frame blocks:** atención solo entre los tokens de un mismo frame;
- **global blocks:** atención entre los tokens de frames distintos, que es donde se relacionan las vistas.

Se agregan tokens especiales por frame: cámara, registro y escala.

**RoPE (rotary position embedding).** En vez de sumar una posición, rota pares de coordenadas de $Q$ y $K$ con un ángulo proporcional a la posición. Para una posición $m$ y el par $(q_{2i},q_{2i+1})$:

$$
\begin{pmatrix}q'_{2i}\\q'_{2i+1}\end{pmatrix}=
\begin{pmatrix}\cos m\theta_i&-\sin m\theta_i\\ \sin m\theta_i&\cos m\theta_i\end{pmatrix}
\begin{pmatrix}q_{2i}\\q_{2i+1}\end{pmatrix},\qquad \theta_i=b^{-2i/d_h}.
$$

Así el producto $\langle q'_m,k'_n\rangle$ depende solo de $m-n$: la atención ve posiciones **relativas**. En la versión 3D las dimensiones se reparten entre tres ejes: fila, columna e **índice de frame en la secuencia**.

Una consecuencia que importa en este proyecto: el eje temporal es el **índice** del frame, no el tiempo real. Si los frames llegan con espaciado desigual (curación por movimiento), el modelo ve un "tiempo" distorsionado. Es la hipótesis de por qué la curación adaptativa empeoró la deriva (bitácora del 2026-09-19).

### 2.4 Cabeza de cámara

Produce un vector de 9 números por frame (`absT_quaR_FoV`): traslación $\mathbf t\in\mathbb R^3$, cuaternión $\mathbf q\in\mathbb R^4$ y dos campos de visión $(\phi_h,\phi_w)$. A partir de ellos (`pose_encoding_to_extri_intri`):

$$
R=R(\mathbf q),\qquad
f_y=\frac{H/2}{\tan(\phi_h/2)},\quad f_x=\frac{W/2}{\tan(\phi_w/2)},\quad c_x=\tfrac W2,\;c_y=\tfrac H2,
$$

con

$$
R(\mathbf q)=\begin{pmatrix}
1-2(y^2+z^2)&2(xy-wz)&2(xz+wy)\\
2(xy+wz)&1-2(x^2+z^2)&2(yz-wx)\\
2(xz-wy)&2(yz+wx)&1-2(x^2+y^2)\end{pmatrix}
$$

para $\mathbf q=(x,y,z,w)$ normalizado. La cabeza se aplica `camera_num_iterations` veces, cada una refinando la estimación anterior. Pasar de 1 a 4 iteraciones subió la coherencia a 1 s de 68% a 75% (bitácora del 2026-09-17).

### 2.5 Cabeza de profundidad (DPT)

Una cabeza tipo DPT junta tokens de varias capas, los vuelve a llevar a la grilla de la imagen y sobremuestrea hasta $518\times518$. Saca 2 canales por píxel, $(x_d,x_c)$, con las activaciones de `lingbot_map/heads/head_act.py`:

$$
D(\mathbf u)=e^{x_d(\mathbf u)}>0,\qquad C(\mathbf u)=1+e^{x_c(\mathbf u)}\ge1 .
$$

La confianza $C$ no es una probabilidad. Se usa como ranking: los filtros del repositorio descartan los píxeles por debajo de un **percentil** de $C$ (p. ej. p35), nunca por un valor absoluto. Hay **una profundidad por píxel** ($518^2=268\,324$ por frame): el modelo no detecta puntos de interés.

### 2.6 Modo streaming: atención causal con caché KV deslizante

Los frames llegan de a uno. Los primeros $n_s$ (frames de escala, `num_scale_frames`) se procesan juntos. Después, cada frame nuevo atiende a los tokens guardados de frames anteriores:

$$
\mathrm{Attn}\big(Q_t,\;[K_{\mathcal S};K_{t-w:t}],\;[V_{\mathcal S};V_{t-w:t}]\big),
$$

donde $\mathcal S$ son los frames de escala y $w$ es `kv_cache_sliding_window`. La memoria queda acotada por $O\big((n_s+w)\cdot1369\cdot d\cdot L\big)$ y el costo por frame deja de crecer una vez que la ventana se llena. Es lo que se midió como saturación de RAM y de tiempo por frame (campaña de 2026-08-24).

El costo: un frame solo ve el pasado, y solo $w$ frames hacia atrás. Los errores de pose se acumulan sin corrección, que es la **deriva**. En las muestras reales el recorrido llegó a medir 2.6 veces su largo real.

Es el único modo posible en vivo (`scripts_stream/live_server.py`), porque en vivo no hay futuro.

### 2.7 Modo windowed: ventanas solapadas y encadenadas por similaridad

**Para qué:** es lo que corrige la deriva en los videos grabados, y la configuración recomendada.

La secuencia se corta en ventanas de $w$ keyframes con solape. Dentro de cada ventana la atención es **bidireccional** (cada frame ve a todos los de su ventana). Cada ventana $b$ queda en su propio sistema de coordenadas y se lleva al de la anterior $a$ con una similaridad $(s,R,\mathbf t)$, estimada en los frames del solape (`_pairwise_alignment` en `gct_stream_window.py`):

$$
R_{ab}=R_a R_b^\top,\qquad
s_{ab}=\operatorname{mediana}_{\mathbf u,\,k\in\text{solape}}\frac{D^{(a)}_k(\mathbf u)}{D^{(b)}_k(\mathbf u)},\qquad
\mathbf t_{ab}=\mathbf c_a-s_{ab}R_{ab}\mathbf c_b .
$$

Aquí $(R_a,\mathbf c_a)$ y $(R_b,\mathbf c_b)$ son la rotación y el centro del mismo frame "ancla" visto desde cada ventana. Después se transforma toda la ventana:

$$
R_k\leftarrow R_{ab}R_k,\qquad \mathbf c_k\leftarrow s_{ab}R_{ab}\mathbf c_k+\mathbf t_{ab},\qquad D_k\leftarrow s_{ab}D_k .
$$

La escala se estima con la **mediana** del cociente de profundidades (robusta a píxeles malos) y la rotación con un solo par de cámaras. Si un tramo difícil (una escalera, una pared blanca) queda partido entre dos ventanas, esa estimación puntual puede equivocarse. Es lo que se vio con la ventana de 16 en la escalera, que la de 24 resolvió (bitácora del 2026-10-01).

---

## 3. De la profundidad a los puntos

**Para qué:** todas las nubes, la malla y la inicialización del splat parten de acá. El modelo funciona como un **sensor RGB-D virtual**. Para el frame $i$, cada píxel $\mathbf u$ válido da

$$
\mathbf X_i(\mathbf u)=R_{wc,i}\,D_i(\mathbf u)K_i^{-1}\tilde{\mathbf u}+\mathbf c_i,\qquad
\text{color}=I_i(\mathbf u),
$$

con $\tilde{\mathbf u}=(u,v,1)^\top$. "Válido" significa $D_i(\mathbf u)>0$ y $C_i(\mathbf u)\ge \mathrm{percentil}_p(C)$. Código: `export_dense_cloud.py`, `npz_to_webgl.py`, `tsdf_mesh.py`, `gsplat_train.py` y `geo_filter.py`.

---

## 4. Nube fusionada por vóxel

**Para qué:** frames vecinos ven casi lo mismo, así que la nube cruda repite cada superficie decenas de veces (114.6 M de puntos crudos contra 10.3 M únicos en la muestra 1). Código: `export_dense_cloud.py`.

El tamaño del vóxel es relativo a la escena: $\ell=\rho\cdot\|\mathbf X_{p98}-\mathbf X_{p2}\|$ (por ejemplo $\rho=0.0004$). Cada punto cae en una celda entera

$$
\mathbf g=\left\lfloor\frac{\mathbf X-\mathbf X_{\min}}{\ell}\right\rfloor\in\mathbb Z^3,
\qquad \kappa(\mathbf g)=(g_x\ll42)\;|\;(g_y\ll21)\;|\;g_z ,
$$

es decir, una clave de 64 bits (21 bits por eje, hasta $2^{21}$ celdas por lado). La fusión agrupa por clave y promedia:

$$
\bar{\mathbf X}_\kappa=\frac{1}{n_\kappa}\sum_{\kappa(\mathbf X)=\kappa}\mathbf X,\qquad
\bar{\mathbf c}_\kappa=\frac{1}{n_\kappa}\sum \mathbf c .
$$

Se hace con `np.unique(return_inverse)` y `np.bincount`, por bloques de frames, para acotar la RAM. Esa reimplementación bajó el pico de memoria de 13-17 GB a 3 GB.

---

## 5. Malla por fusión TSDF

**Para qué:** sacar una **superficie** continua con color a partir de las profundidades. Código: `tsdf_mesh.py`, con `ScalableTSDFVolume` de Open3D.

**Función de distancia con signo truncada.** Para un vóxel de centro $\mathbf x$ y un frame $i$ que lo ve en el píxel $\mathbf u=\pi_i(\mathbf x)$:

$$
\mathrm{sdf}_i(\mathbf x)=D_i(\mathbf u)-z_i(\mathbf x),\qquad
\psi_i(\mathbf x)=\min\!\left(1,\frac{\mathrm{sdf}_i(\mathbf x)}{\mu}\right)\quad\text{si } \mathrm{sdf}_i>-\mu ,
$$

con truncamiento $\mu=4\ell$ y vóxel $\ell=\tilde D/150$. Se acumula un promedio ponderado (Curless y Levoy, 1996):

$$
F(\mathbf x)=\frac{\sum_i w_i(\mathbf x)\,\psi_i(\mathbf x)}{\sum_i w_i(\mathbf x)},\qquad
\mathbf c(\mathbf x)=\frac{\sum_i w_i\, I_i(\mathbf u)}{\sum_i w_i},
$$

con $w_i=1$. La superficie es el conjunto de nivel $F=0$, que se extrae como triángulos por *marching cubes* (interpolación lineal del cruce por cero en cada arista del cubo). Después se descartan los componentes conexos con menos del 0.2% de los triángulos.

**Consistencia (`inlier5`).** Desde cada cámara se lanza un rayo por píxel contra la malla (raycasting). La distancia de impacto $t$ es a lo largo del rayo, así que se pasa a profundidad con

$$
z_{\text{malla}}=\frac{t}{\big\|K^{-1}\tilde{\mathbf u}\big\|}=\frac{t}{\sqrt{\left(\frac{u-c_x}{f_x}\right)^2+\left(\frac{v-c_y}{f_y}\right)^2+1}} ,
$$

y se cuenta la fracción de píxeles confiables con $|z_{\text{malla}}-D|/D<0.05$.

**Color en vistas apartadas.** En el impacto, Open3D da el triángulo $(a,b,c)$ y las coordenadas baricéntricas $(\beta_1,\beta_2)$. El color es $(1-\beta_1-\beta_2)\mathbf c_a+\beta_1\mathbf c_b+\beta_2\mathbf c_c$, y con eso se calculan PSNR y SSIM contra la foto real (sección 9).

**Realineación ICP con escala (`--refine`).** Para un frame rechazado se buscan $(s,R,\mathbf t)$ que minimicen $\sum_k\|sR\mathbf p_k+\mathbf t-\mathbf q_{\nu(k)}\|^2$, donde $\mathbf q_{\nu(k)}$ es el vecino más cercano en la malla de consenso. Se itera entre buscar correspondencias y resolver la similaridad en forma cerrada (Umeyama, sección 9.1).

---

## 6. Gaussian Splatting

**Para qué:** una representación que se entrena **contra las fotos** y no solo contra las profundidades, así que corrige parte de los errores del modelo y da las mejores vistas nuevas del repositorio. Código: `gsplat_train.py` (biblioteca gsplat 1.5.3).

### 6.1 La representación

Una escena es un conjunto de gaussianas 3D. Cada una tiene centro $\boldsymbol\mu\in\mathbb R^3$, covarianza $\Sigma$, opacidad $\alpha\in(0,1)$ y color dependiente de la dirección. La covarianza se parametriza para que siempre sea definida positiva:

$$
\Sigma=R(\mathbf q)\,S\,S^\top R(\mathbf q)^\top,\qquad S=\mathrm{diag}(e^{s_1},e^{s_2},e^{s_3}),\qquad \alpha=\sigma(o)=\frac{1}{1+e^{-o}} .
$$

**Color por armónicos esféricos** de grado $\ell_{\max}=1$:

$$
\mathbf c(\mathbf d)=\tfrac12+\sum_{\ell=0}^{\ell_{\max}}\sum_{m=-\ell}^{\ell}\mathbf k_{\ell m}Y_{\ell m}(\mathbf d),\qquad Y_{00}=\tfrac{1}{2\sqrt\pi}\approx0.2821 .
$$

Para empezar con el color $\mathbf c_0$ de un punto se fija $\mathbf k_{00}=(\mathbf c_0-\tfrac12)/Y_{00}$.

### 6.2 Proyección (EWA) y composición

Con $J$ el jacobiano de la proyección perspectiva en $\mathbf t=W\boldsymbol\mu$ (coordenadas de cámara) y $W$ la rotación de la vista, la gaussiana proyectada en la imagen tiene

$$
\boldsymbol\mu'=\pi(\boldsymbol\mu),\qquad
\Sigma'=J\,W\,\Sigma\,W^\top J^\top,\qquad
J=\begin{pmatrix}f_x/t_z&0&-f_xt_x/t_z^2\\0&f_y/t_z&-f_yt_y/t_z^2\end{pmatrix}.
$$

En cada píxel, las gaussianas ordenadas de adelante hacia atrás se componen con

$$
\alpha_k(\mathbf u)=\alpha_k\exp\!\left(-\tfrac12(\mathbf u-\boldsymbol\mu'_k)^\top\Sigma_k'^{-1}(\mathbf u-\boldsymbol\mu'_k)\right),\qquad
T_k=\prod_{j<k}(1-\alpha_j(\mathbf u)),
$$

$$
\hat I(\mathbf u)=\sum_k \mathbf c_k\,\alpha_k(\mathbf u)\,T_k,\qquad
\hat D(\mathbf u)=\frac{\sum_k z_k\,\alpha_k(\mathbf u)T_k}{\sum_k\alpha_k(\mathbf u)T_k}\quad(\text{modo "RGB+ED"}).
$$

### 6.3 Pérdida y optimización

Para un frame de entrenamiento $i$:

$$
\mathcal L=(1-\lambda)\,\|\hat I-I_i\|_1+\lambda\,(1-\mathrm{SSIM}(\hat I,I_i))+w(t)\,\frac{1}{|M|}\sum_{\mathbf u\in M}\frac{|\hat D(\mathbf u)-D_i(\mathbf u)|}{D_i(\mathbf u)},
$$

con $\lambda=0.2$ y $M$ el conjunto de píxeles confiables y cubiertos ($\sum_k\alpha_kT_k>0.5$). La guía de profundidad decae linealmente, $w(t)=0.1\,(1-t/t_{\max})$: al principio sostiene la geometría del modelo y al final deja que manden las fotos.

Se optimiza con Adam, una tasa por grupo de parámetros. La de los centros es proporcional a la escala de la escena y decae exponencialmente hasta 1% de la inicial. La **densificación adaptativa** (`DefaultStrategy`) clona las gaussianas chicas y divide las grandes cuando el gradiente medio de su posición en pantalla supera un umbral, y poda las casi transparentes. Se detiene al 60% de las iteraciones o al llegar a 3 M de gaussianas (tope de VRAM). La opacidad se reinicia cada 3000 iteraciones.

**Inicialización:** los puntos de la sección 3 cada 4 píxeles, fusionados por vóxel. La escala inicial es $\log$ de la distancia media a los 3 vecinos más cercanos.

### 6.4 Con el filtro geométrico (sección 8)

Con `--filter`:

- la nube inicial y la guía de profundidad usan la profundidad filtrada $D^{f}$ (sin los píxeles descartados y llevada a los planos);
- la parte fotométrica de la pérdida se promedia solo sobre los píxeles sin personas:

$$
\mathcal L_1=\frac{\sum_{\mathbf u}m(\mathbf u)\,\overline{|\hat I-I|}(\mathbf u)}{\sum_{\mathbf u}m(\mathbf u)},\qquad
\mathcal L_{\mathrm{SSIM}}=1-\frac{\sum_{\mathbf u}m(\mathbf u)\,\mathrm{ssim}(\mathbf u)}{\sum_{\mathbf u}m(\mathbf u)} ,
$$

donde $m$ es la máscara de píxeles estáticos y $\mathrm{ssim}(\mathbf u)$ es el mapa local de SSIM (sección 9.4).

---

## 7. Análisis de frames y context-to-image

**Para qué:** decidir qué frames de un video le llegan al modelo, y rellenar saltos grandes entre frames con frames intermedios generados, como "amortiguador visual". Es la idea de Paragraphica (computar según el contexto) llevada a este problema. **No hay generación de imagen por IA**: lo que se genera es una interpolación por flujo óptico. Código: `analyze_frames.py` y `curate_and_synthesize.py` (videos grabados), `scripts_stream/context_gate.py` (en vivo).

### 7.1 Nitidez

Varianza del laplaciano de la imagen en gris (Pech-Pacheco et al., 2000):

$$
\nabla^2 G=\frac{\partial^2G}{\partial x^2}+\frac{\partial^2G}{\partial y^2}\;\approx\;G*\begin{pmatrix}0&1&0\\1&-4&1\\0&1&0\end{pmatrix},\qquad
\mathrm{nitidez}=\operatorname{Var}_{\mathbf u}\big(\nabla^2G(\mathbf u)\big).
$$

Un frame movido tiene pocos bordes finos, así que su laplaciano es casi plano y la varianza es baja. Se compara contra el percentil 75 de una ventana de frames vecinos, porque la escena también cambia.

### 7.2 Movimiento: flujo óptico

El flujo $\mathbf F_{0\to1}(\mathbf u)$ es el desplazamiento que lleva cada píxel del frame 0 al 1, bajo la hipótesis de brillo constante $I_1(\mathbf u+\mathbf F(\mathbf u))\approx I_0(\mathbf u)$. Se usan dos métodos:

- **RAFT** (red neuronal, Teed y Deng 2020), fuera de línea. Construye un volumen de correlación entre todos los pares de píxeles de los mapas de rasgos, $C(\mathbf u,\mathbf u')=\langle g_0(\mathbf u),g_1(\mathbf u')\rangle$, y refina $\mathbf F$ iterativamente con una GRU. El volumen crece como $(H/8\cdot W/8)^2$: a $1080\times1920$ pide 3.9 GiB, por eso se calcula sobre una copia reducida (`--flow_max_side`).
- **DIS** (Kroeger et al. 2016), en vivo. Busca parches por descenso de gradiente inverso en una pirámide de resoluciones y densifica con un promedio ponderado. Corre en CPU en unos 2 ms a 256 px.

El movimiento de un frame es la mediana de $\|\mathbf F(\mathbf u)\|$, llevada a píxeles de la imagen de 518 px.

### 7.3 Presupuesto adaptativo de frames

Sea $m_k$ el movimiento entre los frames $k-1$ y $k$. Desde el último frame enviado se acumula $A=\sum m_k$, y se envía un frame cuando $A\ge\delta$ (`step_px`, 36 px por defecto) o cuando ya se saltaron `max_skip` frames seguidos. Del tramo acumulado se elige el más nítido de la segunda mitad, la parte con $A_k\ge\delta/2$. En los videos grabados la regla es la misma, sobre todos los frames del video (`curate_and_synthesize.py`).

### 7.4 Frames intermedios por flujo bidireccional

Si entre dos frames enviados el movimiento supera $1.5\,\delta$ (y no pasa de 240 px, donde la interpolación se desarma), se generan $n=\lceil A/\delta\rceil-1$ frames intermedios en $t=k/(n+1)$. Con los flujos de ida y vuelta $\mathbf F_{01},\mathbf F_{10}$, los flujos hacia el instante $t$ se aproximan como en Super-SloMo (Jiang et al. 2018):

$$
\mathbf F_{t\to0}=-(1-t)\,t\,\mathbf F_{01}+t^2\,\mathbf F_{10},\qquad
\mathbf F_{t\to1}=(1-t)^2\,\mathbf F_{01}-t(1-t)\,\mathbf F_{10},
$$

$$
\hat I_t(\mathbf u)=(1-t)\,I_0\big(\mathbf u+\mathbf F_{t\to0}(\mathbf u)\big)+t\,I_1\big(\mathbf u+\mathbf F_{t\to1}(\mathbf u)\big),
$$

con muestreo bilineal (`cv2.remap`). La versión fuera de línea agrega máscaras de oclusión por consistencia ida-vuelta: un píxel es poco confiable si $\|\mathbf F_{01}(\mathbf u)+\mathbf F_{10}(\mathbf u+\mathbf F_{01}(\mathbf u))\|$ es grande.

**Intensidad** $k$ (`synth_strength`): se sintetiza en todo salto mayor que $1.5\,\delta/k$ y se pone un intermedio cada $\delta/k$, es decir, $n=\min\big(8,\lceil kA/\delta\rceil-1\big)$. Con $k=1$ es la regla de arriba. Medido el 2026-10-04: $k=2$ y $k=3$ no mejoran la forma del recorrido en windowed y la empeoran mucho en streaming (bitácora).

Los frames sintéticos entran al modelo **solo como contexto temporal** (pose y caché KV). Nunca se dibujan ni se agregan a ningún mapa.

---

## 8. Filtro geométrico previo a la malla y al splat

**Para qué:** las predicciones por frame no son del todo consistentes entre sí. La misma pared vista en dos momentos del recorrido queda en dos lugares un poco distintos (**paredes dobles**), las personas que se mueven quedan pegadas al mapa, y en los bordes de los objetos aparecen puntos flotando. El filtro depura eso antes de construir la malla y el splat, busca la **estructura** del lugar (paredes, piso, esquinas) y arma con ella una malla simple. Código: `scripts_context/geo_filter.py`, que se aplica con `--filter` en `tsdf_mesh.py` y `gsplat_train.py`.

**No modifica las predicciones del modelo ni la nube fusionada.** Escribe un archivo aparte (`<name>_filtro.npz`) que solo usan la malla y el splat filtrados.

### 8.1 Interpretación semántica del video (SegFormer)

**Modelo:** SegFormer-B0 entrenado en ADE20K (150 clases), unos 3.7 M de parámetros. Un codificador transformer jerárquico (*Mix Transformer*) produce rasgos a 1/4, 1/8, 1/16 y 1/32 de la resolución. La atención usa *reducción espacial*: $K$ y $V$ se calculan sobre una versión de la imagen reducida por un factor $R$, así que el costo baja de $O(N^2)$ a $O(N^2/R)$. Un decodificador MLP lleva los cuatro niveles a 1/4, los concatena y clasifica cada píxel:

$$
p(c\mid\mathbf u)=\frac{e^{\ell_c(\mathbf u)}}{\sum_{c'}e^{\ell_{c'}(\mathbf u)}},\qquad \hat c(\mathbf u)=\arg\max_c\ell_c(\mathbf u).
$$

La entrada se normaliza con la media y el desvío de ImageNet. Los logits salen a 1/4 de resolución ($128\times128$ para una entrada de 512) y se interpolan bilinealmente a $518\times518$ **antes** del argmax.

Las clases se agrupan así:

- **quitar:** persona, animal, cielo. Se mueven entre frames o no tienen profundidad real, así que se sacan de la profundidad. Las personas y animales también se sacan de la pérdida fotométrica del splat.
- **pared:** pared, puerta, ventana, cuadro, espejo, cartelera, persiana, cortina, póster, pantalla.
- **piso:** piso, alfombra, calle, andén, sendero.
- **techo.**

### 8.2 Puntos flotantes en los bordes

En un borde de profundidad, los píxeles intermedios quedan "colgando" entre el objeto y el fondo. Con máximos y mínimos en una vecindad $3\times3$:

$$
\mathrm{borde}(\mathbf u)=\frac{\max_{\mathcal N(\mathbf u)}D-\min_{\mathcal N(\mathbf u)}D}{D(\mathbf u)}>\tau_e,\qquad \tau_e=0.08 .
$$

### 8.3 Consistencia multivista: apoyos y violaciones de espacio libre

**Vecinos.** Primero se calcula una matriz de solape entre todos los pares de frames. Para cada frame $i$ se toma una grilla gruesa de puntos (1 de cada $16\times16$ píxeles), se proyectan en cada frame $j$ y se cuenta qué fracción cae dentro de la imagen, delante de la cámara y sin quedar tapada por lo que $j$ ve:

$$
O_{ij}=\frac{1}{|P_i|}\sum_{\mathbf X\in P_i}\mathbb 1\big[\pi_j(\mathbf X)\in\Omega,\;z_j(\mathbf X)>0,\;z_j(\mathbf X)<1.15\,D_j(\pi_j(\mathbf X))\big].
$$

El puntaje de un par es $\min(O_{ij},O_{ji})$. Cada frame toma 4 vecinos **cercanos** en el tiempo ($|i-j|\le24$) y 8 **lejanos**. Los lejanos son las revisitas del mismo lugar, que es donde aparecen las paredes dobles. Los cercanos casi no sirven para detectarlas: en modo windowed salen de la misma ventana, así que son consistentes entre sí por construcción.

**Votación por píxel.** Cada punto $\mathbf X=\mathbf X_i(\mathbf u)$ se proyecta en cada vecino $j$, con $z=z_j(\mathbf X)$ y $d=D_j(\pi_j(\mathbf X))$ (vecino más cercano), y se clasifica:

$$
\text{observado: } z<(1+\tau_s)\,d,\qquad
\text{apoyo: } |z-d|<\tau_s\,d,\qquad
\text{violación: } z<(1-\tau_v)\,d .
$$

Por defecto $\tau_s=0.06$ y $\tau_v=0.15$. Si $z$ está bastante más allá de $d$, el punto queda tapado en $j$ y ese vecino no opina.

La **violación de espacio libre** es la clave: el vecino $j$ ve una superficie más lejos que el punto, a lo largo del mismo rayo. O sea, mira *a través* del punto, y ese punto está flotando en un espacio que otra cámara ve vacío. En una pared doble, la capa de adelante queda atravesada por las cámaras que vieron la capa de atrás. La de atrás, en cambio, queda tapada (no opina), así que sobrevive. El resultado es una sola capa.

Un píxel se conserva si

$$
\#\text{apoyos}\ge1\quad\text{y no}\quad\big(\#\text{viol}\ge2\;\wedge\;\#\text{viol}>0.34\,\#\text{obs}\big).
$$

Exigir al menos dos violaciones y más de un tercio de las observaciones protege las estructuras finas y los errores sueltos de un solo vecino.

### 8.4 La vertical

Al caminar, el teléfono casi no gira sobre su eje óptico. Entonces el eje $x$ ("derecha") de cada cámara, $\mathbf x_i=R_{wc,i}\,\mathbf e_1$, es casi horizontal, y la vertical es la dirección más perpendicular a todos:

$$
\mathbf n_{\uparrow}=\arg\min_{\|\mathbf n\|=1}\sum_i(\mathbf n^\top\mathbf x_i)^2=\text{autovector del menor autovalor de }\textstyle\sum_i\mathbf x_i\mathbf x_i^\top .
$$

El signo se elige para que $\mathbf n_\uparrow$ apunte al revés que el eje $+Y$ medio de las cámaras, que en OpenCV apunta al suelo. No se usa el eje $Y$ promediado, porque en una escalera el teléfono se inclina de forma sistemática: con ese criterio la vertical se desvía entre 4° y 11° (bitácora del 2026-10-01).

### 8.5 Nube de trabajo y normales

Los píxeles conservados (1 de cada $3\times3$) se fusionan por vóxel ($\ell=\tilde D/120$, sección 4). Cada vóxel guarda la posición y el color medios, el **grupo semántico por mayoría** y una cámara que lo vio. Se descartan los vóxeles con un solo punto.

La normal de cada punto es el autovector del menor autovalor de la covarianza de sus 20 vecinos más cercanos, orientada hacia la cámara que lo vio ($\mathbf n\cdot(\mathbf c-\mathbf X)>0$).

### 8.6 Paredes: RANSAC vertical con muestra de 2 puntos

Un plano vertical es $\{\mathbf X:\mathbf n^\top\mathbf X=d\}$ con $\mathbf n\perp\mathbf n_\uparrow$. Como la vertical ya está fija, dos puntos alcanzan para definirlo:

$$
\mathbf n=\frac{(\mathbf X_2-\mathbf X_1)\times\mathbf n_\uparrow}{\|(\mathbf X_2-\mathbf X_1)\times\mathbf n_\uparrow\|},\qquad d=\mathbf n^\top\mathbf X_1 .
$$

Un punto de "pared" con normal casi horizontal ($|\mathbf n_p\cdot\mathbf n_\uparrow|<0.35$) es **inlier** del plano si

$$
|\mathbf n^\top\mathbf X-d|<\epsilon\quad\text{y}\quad|\mathbf n_p^\top\mathbf n|>\cos30^\circ,\qquad \epsilon=0.02\,\tilde D .
$$

Por qué alcanza con 600 hipótesis: si una pared tiene una fracción $r$ de los puntos, una muestra de 2 cae entera en ella con probabilidad $r^2$, y la probabilidad de encontrarla al menos una vez en $N$ intentos es

$$
P=1-(1-r^2)^N .
$$

Con $r=0.15$ y $N=600$ da $P\approx1-10^{-6}$. Con una muestra mínima de 3 puntos (RANSAC de planos sin la vertical) haría falta $r^3$: unas $6.7$ veces más hipótesis para la misma $P$.

El mejor plano se reajusta por mínimos cuadrados con la vertical fija. Los puntos se proyectan al plano horizontal, en una base $(\mathbf e_1,\mathbf e_2)\perp\mathbf n_\uparrow$, y la normal es la dirección horizontal de **menor varianza**: el autovector del menor autovalor de la covarianza $2\times2$ de esas coordenadas. Sus inliers se sacan y se repite, hasta 60 planos o hasta que no quede un plano con suficientes puntos. La normal se orienta hacia las cámaras que vieron sus puntos.

### 8.7 Regularización de Manhattan

En interiores casi todas las paredes son paralelas o perpendiculares entre sí. Si $\theta_p$ es el azimut de la normal de la pared $p$ en la base $(\mathbf e_1,\mathbf e_2)$, el eje dominante módulo $90^\circ$ se estima con una **media circular** de $4\theta$, pesada por la cantidad $n_p$ de puntos de cada pared:

$$
\theta_0=\frac14\arg\sum_p n_p\,e^{\,i\,4\theta_p} .
$$

Multiplicar por 4 hace que $\theta$, $\theta+90^\circ$, $\theta+180^\circ$ y $\theta+270^\circ$ cuenten como la misma dirección. Cada pared con $|\theta_p-(\theta_0+k\cdot90^\circ)|<25^\circ$ se gira exactamente a ese eje, y su desplazamiento se reestima como $d=\operatorname{mediana}(\mathbf n^\top\mathbf X)$. Las paredes oblicuas solo se conservan si son grandes (al menos el 15% de la mayor). Esto es lo que convierte la nube en **rectángulos**.

### 8.8 Fusión de paredes dobles

Dos planos $p$ y $q$ son la misma pared si:

- son paralelos y con la misma orientación, $\mathbf n_p^\top\mathbf n_q>\cos12^\circ$. Como las normales apuntan a las cámaras, esto también exige que las **vean desde el mismo lado**: las dos caras de un tabique fino no se fusionan;
- se superponen a lo largo de la pared. Con $u=\mathbf t^\top\mathbf X$ y $\mathbf t=\mathbf n_\uparrow\times\mathbf n_p$, el solape de sus rangos $[p_5,p_{95}]$ supera el 30% del más corto;
- en esa zona de solape, la separación $|\operatorname{mediana}(\mathbf n_p^\top\mathbf X_q)-d_p|$ es menor que $0.3\,\tilde D$.

Los puntos de $q$ pasan a $p$, cuya orientación se mantiene (manda la pared más grande), y $d_p$ se reestima con la mediana de todos. Se repite hasta que no haya más fusiones.

### 8.9 Pisos y techos

Son planos horizontales, así que basta la altura $h=\mathbf n_\uparrow^\top\mathbf X$ de los puntos de "piso" (o de "techo") con normal casi vertical. Se buscan picos sucesivos del histograma de $h$ (bins de $1.5\,\epsilon$), refinando cada pico con la mediana de su banda. Los niveles más cercanos que $0.12\,\tilde D$ se fusionan, con promedio pesado por puntos, y se descartan los de menos del 10% del nivel mayor. Varios niveles de piso corresponden a pisos distintos (escaleras, desniveles).

### 8.10 Tramos rectangulares y esquinas

Cada plano de pared se parte en **tramos conexos**: sus puntos en coordenadas $(u,h)$ del plano se rasterizan en una grilla de celda $\tilde D/20$, con dilatación de 2 celdas, y se separan por componentes conexas. Un tramo es un rectángulo

$$
[u_0,u_1]\times[h_0,h_1]=[p_2(u),p_{98}(u)]\times[p_2(h),p_{98}(h)],
$$

que se acepta si cumple todo esto:

- tiene alto $h_1-h_0\ge0.55\,H_{\text{amb}}$, con $H_{\text{amb}}$ = techo − piso. Esto descarta escritorios y muebles que el segmentador confunde con pared;
- tiene largo $u_1-u_0\ge0.25\,\tilde D$;
- su rectángulo está cubierto en al menos un 20% (celdas ocupadas sobre celdas totales);
- tiene al menos el 2% de los puntos de la pared más grande.

Los bordes que quedan cerca del piso o del techo se extienden hasta ellos.

Sus 4 esquinas, con $\mathbf o=d\,\mathbf n$ el punto del plano más cercano al origen, son

$$
\mathbf o+u_a\mathbf t+h_b\mathbf n_\uparrow,\qquad a,b\in\{0,1\}.
$$

**Esquinas entre paredes.** Para dos tramos no paralelos ($|\mathbf n_s^\top\mathbf n_r|<\cos30^\circ$), su arista común es la recta vertical que cumple las dos ecuaciones de plano. Se calcula el punto de esa recta a altura 0 resolviendo un sistema $3\times3$; con el piso y el techo es la intersección de **tres planos**:

$$
\begin{pmatrix}\mathbf n_s^\top\\ \mathbf n_r^\top\\ \mathbf n_\uparrow^\top\end{pmatrix}\mathbf p=\begin{pmatrix}d_s\\d_r\\0\end{pmatrix},
\qquad
\text{esquina inferior}=\mathbf p+h_{\text{piso}}\,\mathbf n_\uparrow,\quad \text{superior}=\mathbf p+h_{\text{techo}}\,\mathbf n_\uparrow .
$$

Si el punto cae dentro de los dos tramos (con tolerancia $0.25\,\tilde D$), es una esquina: en L, una unión en T o un cruce. Los extremos de tramo a menos de esa tolerancia se mueven exactamente a la esquina, y las paredes quedan cerradas.

### 8.11 Polígono del piso y malla simple

Los puntos de cada nivel de piso se rasterizan en el plano horizontal y se cierran morfológicamente (dilatación seguida de erosión, núcleo $5\times5$). Se extrae el contorno exterior y se simplifica con Douglas-Peucker: se conserva el vértice más alejado de la cuerda mientras su distancia supere 1.5 celdas. El polígono simple resultante se triangula por **recorte de orejas**. Un vértice $b$, con sus vecinos $a$ y $c$, es una oreja si el giro es convexo,

$$
(\mathbf b-\mathbf a)\times(\mathbf c-\mathbf a)>0 ,
$$

y ningún otro vértice cae dentro del triángulo $abc$. Se corta la oreja y se repite; un polígono de $n$ vértices da $n-2$ triángulos.

La **malla estructural** (`<name>_estructura.glb`) es la unión de un cuadrilátero (2 triángulos) por tramo de pared y los polígonos de piso, con caras dobles para que se vean de los dos lados. El color de cada cara es el color medio de sus puntos. Son unos pocos cientos de vértices, contra millones de la malla TSDF.

### 8.12 Realineación suave por planos (opcional, `--align`)

La idea es corregir **frames enteros** en vez de píxeles: si un tramo del recorrido quedó corrido respecto de la estructura, se corrige la pose de esos frames. Para cada frame $i$ se buscan una rotación $R$ y una traslación $\mathbf v$ (escala fija) que lleven sus puntos de pared y piso $\mathbf X_k$ a los planos más cercanos $(\mathbf n_k,d_k)$ de su misma clase. La rotación es alrededor del centroide $\boldsymbol\mu$ de esos puntos:

$$
\min_{R,\mathbf v}\;\sum_k \rho\Big(\mathbf n_k^\top\big(\boldsymbol\mu+R(\mathbf X_k-\boldsymbol\mu)+\mathbf v\big)-d_k\Big),\qquad
\rho(r)=\tfrac{c^2}{2}\log\!\big(1+(r/c)^2\big),\quad c=0.05\,\tilde D .
$$

La pérdida de Cauchy $\rho$ le quita peso a los puntos mal asignados. Linealizando alrededor de la estimación actual, con $R\approx I+[\boldsymbol\omega]_\times$ e $\mathbf Y_k=\mathbf X_k-\boldsymbol\mu$, el residuo es lineal en $\boldsymbol\xi=(\boldsymbol\omega,\mathbf v)$:

$$
r_k(\boldsymbol\xi)\approx r_k+\boldsymbol\omega^\top(\mathbf Y_k\times\mathbf n_k)+\mathbf n_k^\top\mathbf v,
\qquad J_k=\big[(\mathbf Y_k\times\mathbf n_k)^\top,\ \mathbf n_k^\top\big],
$$

y cada iteración resuelve un Gauss-Newton amortiguado (Levenberg-Marquardt), con $w_k=1/(1+(r_k/c)^2)$:

$$
\Big(\textstyle\sum_k w_kJ_k^\top J_k+\lambda M I\Big)\boldsymbol\xi=-\sum_k w_kJ_k^\top r_k ,\qquad \lambda=0.1 .
$$

Aquí $M$ es la cantidad de residuos. Después se compone $R\leftarrow\exp([\boldsymbol\omega]_\times)R$, usando Rodrigues para pasar del vector de rotación a la matriz. Se hacen 5 iteraciones, reasignando los planos en cada una. Se descartan los frames con menos de dos planos a la vista (mal condicionados) y las correcciones de más de $3^\circ$ o de $0.1\,\tilde D$.

**Suavizado temporal.** La corrección de cada frame se expresa como transformación del mundo: el vector de rotación $\boldsymbol\theta_i$ y la traslación $\mathbf b_i=\boldsymbol\mu-R\boldsymbol\mu+\mathbf v$. Se promedia en el tiempo con una gaussiana de $\sigma=15$ frames, pesada por la validez $a_i\in\{0,1\}$:

$$
\bar{\boldsymbol\theta}_i=\frac{\sum_j G_\sigma(i-j)\,a_j\boldsymbol\theta_j}{\sum_jG_\sigma(i-j)\,a_j},\qquad
R_{wc,i}\leftarrow R(\bar{\boldsymbol\theta}_i)R_{wc,i},\quad \mathbf c_i\leftarrow R(\bar{\boldsymbol\theta}_i)\mathbf c_i+\bar{\mathbf b}_i .
$$

Sin este paso, la corrección independiente de cada frame mete un zigzag de frame a frame que multiplica el largo del recorrido. La deriva real entre revisitas es lenta, así que se corrige solo la parte de baja frecuencia.

### 8.13 Ajuste a planos (*depth snap*)

Cada píxel conservado de clase "pared" o "piso" cuyo punto $\mathbf X$ cae cerca de un tramo, $|\mathbf n^\top\mathbf X-d|<0.12\,\tilde D$ y dentro de su rectángulo, se lleva sobre el plano a lo largo **de su propio rayo**. Con $\mathbf r=R_{wc}K^{-1}\tilde{\mathbf u}$ (de modo que $\mathbf X=\mathbf c+D\,\mathbf r$):

$$
\mathbf n^\top(\mathbf c+D'\mathbf r)=d\quad\Longrightarrow\quad D'=\frac{d-\mathbf n^\top\mathbf c}{\mathbf n^\top\mathbf r}.
$$

Moverlo a lo largo del rayo, y no en línea recta hacia el plano, mantiene el píxel donde está en la imagen. Si cerca hay más de un plano, gana el más cercano. Se descartan los rayos rasantes, donde $D'$ cambiaría más de un 30%.

Las dos capas de una pared doble terminan sobre la misma superficie, y la malla y el splat parten de paredes planas.

### 8.14 Selección de frames

Se recorren los frames en orden. Se conserva el frame $i$ cuando su solape con el último conservado $\ell$ baja de un umbral, $\min(O_{i\ell},O_{\ell i})<0.7$, o cuando ya se saltaron 6. Si el umbral se cruza en $i$, entre $i-2$, $i-1$ e $i$ se elige el más nítido (sección 7.1). Así se limita cuántos frames construyen la malla y el splat sin dejar huecos de cobertura. Los frames apartados para evaluar no se tocan.

---

## 9. Métricas de evaluación

No hay ground truth métrico. Cada métrica mide una cosa distinta, y la bitácora muestra que mirar una sola puede engañar: la coherencia local subió mientras el recorrido empeoraba.

### 9.1 Forma del recorrido contra un croquis (`compare_route.py`)

El croquis a mano se binariza por color, se adelgaza a un píxel (Zhang-Suen) y se recorre el camino más largo del esqueleto. La trayectoria estimada se proyecta desde arriba usando el eje "abajo" medio de las cámaras. Las dos curvas se remuestrean a $n=300$ puntos equiespaciados por longitud de arco y se alinean con una **similaridad de Umeyama** (1991): con $\boldsymbol\mu_x,\boldsymbol\mu_y$ las medias, $\Sigma_{xy}=\frac1n\sum(\mathbf y_k-\boldsymbol\mu_y)(\mathbf x_k-\boldsymbol\mu_x)^\top=UDV^\top$ y $\sigma_x^2$ la varianza de $\mathbf x$,

$$
R=U\,\mathrm{diag}(1,\det(UV^\top))\,V^\top,\qquad s=\frac{\operatorname{tr}(D\,S)}{\sigma_x^2},\qquad \mathbf t=\boldsymbol\mu_y-sR\boldsymbol\mu_x ,
$$

donde $S=\mathrm{diag}(1,\det(UV^\top))$ evita reflexiones. Se prueban también el espejo y el recorrido invertido, y se queda la mejor alineación. Métricas:

$$
\text{error de forma}=\frac{\sqrt{\frac1n\sum_k\|sR\mathbf x_k+\mathbf t-\mathbf y_k\|^2}}{L_{\text{croquis}}},\qquad
\text{rectitud}=\frac{\|\mathbf x_n-\mathbf x_1\|}{\sum_k\|\mathbf x_{k+1}-\mathbf x_k\|},\qquad
\text{largo}=\frac{L(sR\mathbf x+\mathbf t)}{L_{\text{croquis}}}.
$$

### 9.2 Perfil de altura (`height_profile.py`)

Con la vertical de la sección 8.4, la altura de cada cámara es $h_i=\mathbf n_\uparrow^\top\mathbf c_i$. Sin escala métrica, se reporta

$$
\texttt{end\_over\_max}=\frac{h_{\text{fin}}-h_{\text{inicio}}}{\max_i h_i-\min_i h_i}.
$$

Da cerca de 0 si el recorrido vuelve a la altura de partida, y cerca de 1 si termina arriba.

### 9.3 Autoconsistencia entre frames (`evaluate_consistency.py`)

Para pares de frames reales $(a,b)$ separados por un tiempo fijo del video (0.5 s o 1 s), la profundidad de $a$ se proyecta en $b$ y se mide la fracción de puntos co-visibles con error relativo $<5\%$. Mide **coherencia local**. Es ciega a la deriva acumulada y satura en 1.0 en modo windowed.

### 9.4 Vistas nuevas: PSNR y SSIM sobre frames apartados

Uno de cada 8 frames no se usa para construir. Desde su cámara se renderiza la malla o el splat y se compara con la foto real:

$$
\mathrm{PSNR}=-10\log_{10}\Big(\frac{1}{3|\Omega|}\sum_{\mathbf u,c}(\hat I-I)^2\Big)\ \text{dB}.
$$

El SSIM (Wang et al. 2004) usa medias, varianzas y covarianza locales con ventana gaussiana ($11\times11$, $\sigma=1.5$):

$$
\mathrm{ssim}(\mathbf u)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)},\qquad C_1=0.01^2,\;C_2=0.03^2 .
$$

El SSIM global es el promedio del mapa. **PSNR estático** es el mismo PSNR promediado solo sobre los píxeles sin personas ni animales: lo que se mueve no se puede reconstruir, y mide lo mismo con y sin filtro. También se mide `inlier5` (sección 5) entre la profundidad renderizada y la del modelo.

### 9.5 Espesor de pared (`wall_thickness.py`)

Para cada tramo de pared de la estructura se toman los puntos de una reconstrucción que caen dentro de su rectángulo y a menos de $0.15\,\tilde D$ de su plano, con distancia con signo $\delta=\mathbf n^\top\mathbf X-d$. Se reporta

$$
\text{espesor}=\frac{p_{90}(\delta)-p_{10}(\delta)}{\tilde D},\qquad
\text{fuera de la lámina}=\Pr\big(|\delta-\operatorname{mediana}\delta|>0.05\,\tilde D\big),
$$

promediados por tramo y pesados por cantidad de puntos. Una pared limpia es una lámina fina; una pared doble son dos láminas, así que da más espesor y más puntos fuera. En las variantes con ajuste a planos la métrica es en parte circular, porque la profundidad se llevó justo a esos planos. Con solo máscaras no lo es.

---

## 10. Resumen: qué modelo o método, para qué

| Modelo o método | Tipo | Para qué | Dónde |
|---|---|---|---|
| LingBot-Map (DINOv2 ViT-L/14 + GCT + cabezas de cámara y DPT) | red neuronal, ~1158 M parámetros | pose, intrínsecos, profundidad y confianza por píxel a partir de video RGB | `lingbot_map/`, `demo.py`, `process_and_view.py` |
| Streaming con caché KV deslizante | modo de inferencia | mapeo en vivo, memoria acotada (deriva) | `live_server.py` |
| Windowed con similaridad entre ventanas | modo de inferencia | videos grabados, corrige la deriva | `process_and_view.py --mode windowed` |
| SegFormer-B0 (ADE20K) | red neuronal, ~3.7 M parámetros | interpretar el video: quitar personas y cielo, encontrar pared, piso y techo | `geo_filter.py` |
| RAFT | red neuronal (flujo óptico) | movimiento entre frames e interpolación, fuera de línea | `analyze_frames.py`, `curate_and_synthesize.py` |
| DIS | flujo óptico clásico | movimiento e interpolación en vivo | `context_gate.py` |
| Varianza del laplaciano | filtro clásico | nitidez de cada frame | los tres anteriores |
| Fusión por vóxel | geometría | nube sin repetidos | `export_dense_cloud.py` |
| TSDF + marching cubes | geometría | malla con color | `tsdf_mesh.py` |
| Gaussian Splatting (gsplat) | optimización diferenciable | reconstrucción fotorrealista | `gsplat_train.py` |
| Consistencia multivista, RANSAC, Manhattan, esquinas, recorte de orejas | geometría | filtro previo y malla estructural | `geo_filter.py` |
| Umeyama | geometría | alinear trayectoria y croquis; ICP con escala | `compare_route.py`, `tsdf_mesh.py` |
