# Metodologia completa y registro de corridas

Este documento describe de forma exhaustiva como se generaron los datos, como se
entrenaron las PINNs y como se produjeron las figuras/tablas del trabajo
alternativo. Todo el codigo vive en esta carpeta (`ger/`) y los resultados
generados quedan en `ger/outputs/`.

## 1. Objetivo del experimento

La consigna alternativa pide reproducir la metodologia central del paper:

Sifan Wang, Yujun Teng, Paris Perdikaris, "Understanding and mitigating
gradient pathologies in physics-informed neural networks", arXiv:2001.04536.

En este trabajo no se replica el mismo problema fisico del paper y no se usa la
arquitectura nueva que proponen los autores. Se replica el nucleo metodologico:

- PINN clasica como baseline (`M1`).
- PINN con balance adaptativo de terminos de perdida por estadisticas de
  gradientes (`M2`, Algoritmo 1 del paper).
- Analisis de gradientes por capa.
- Evolucion de perdidas y pesos adaptativos.
- Comparacion contra solucion analitica.
- Busqueda bayesiana de hiperparametros.
- Comparacion de optimizadores con presupuesto fijo.

El problema elegido es la difusion de calor 1D en una barra metalica.

## 2. Problema fisico

### 2.1 Sistema dimensional

Se considera una barra metalica recta de longitud `L = 1 m`, seccion constante y
sin fuentes internas. La temperatura dimensional es `T(X,t)`, donde:

- `X in [0,L]` es la posicion dimensional.
- `t >= 0` es el tiempo dimensional.
- `alpha = 1e-4 m^2/s` es la difusividad termica.
- `T_amb = 20 C` es la temperatura de los extremos.
- `DeltaT = 80 C` es la escala de temperatura usada para adimensionalizar.

La ecuacion de calor dimensional es:

```text
dT/dt = alpha d2T/dX2,      0 < X < L,  t > 0
T(0,t) = T(L,t) = T_amb
T(X,0) = T_amb + DeltaT * u0(X/L)
```

### 2.2 Adimensionalizacion

Se definen:

```text
x   = X / L
tau = alpha t / L^2
u   = (T - T_amb) / DeltaT
```

Con estas variables:

```text
u_tau - u_xx = 0,       0 < x < 1,  0 < tau < tau_f
u(0,tau) = u(1,tau) = 0
u(x,0) = u0(x)
```

Parametros usados:

- `tau_f = 0.25`.
- `t_f = tau_f * L^2 / alpha = 2500 s`.

### 2.3 Condicion inicial y solucion analitica

La condicion inicial se eligio como una combinacion de modos senoidales
compatibles con bordes de Dirichlet homogeneos:

```text
u0(x) = sin(pi x) + 0.5 sin(3 pi x) + 0.25 sin(5 pi x)
```

La solucion analitica por separacion de variables es:

```text
u(x,tau) = sum_n a_n sin(n pi x) exp(-(n pi)^2 tau)
```

con:

```text
(n, a_n) in {(1, 1.0), (3, 0.5), (5, 0.25)}
```

Esta solucion se usa solo para evaluacion y graficos, no como dato interior de
entrenamiento.

## 3. Generacion de datos de entrenamiento

Los datos se generan de forma sintetica en cada iteracion de entrenamiento. No
hay dataset fijo para la PINN; en cada paso se muestrean minibatches nuevos.

### 3.1 Puntos de residuo PDE

Para imponer la ecuacion `u_tau - u_xx = 0`, se muestrean puntos interiores:

```text
x_r   ~ Uniform(0,1)
tau_r ~ Uniform(0,tau_f)
```

En la corrida final:

- `batch_residual = 256`.
- Puntos nuevos en cada iteracion.
- La perdida es `L_res = mean((u_tau - u_xx)^2)`.

Las derivadas `u_tau`, `u_x` y `u_xx` se calculan con autograd de PyTorch,
usando `create_graph=True` para permitir derivadas segundas.

### 3.2 Puntos de condicion inicial

Para imponer `u(x,0)=u0(x)`:

```text
x_0   ~ Uniform(0,1)
tau_0 = 0
target = u0(x_0)
```

En la corrida final:

- `batch_initial = 256`.
- Puntos nuevos en cada iteracion.
- La perdida es `L_ic = mean((u_theta(x,0)-u0(x))^2)`.

### 3.3 Puntos de borde

Para imponer `u(0,tau)=u(1,tau)=0`:

```text
tau_b ~ Uniform(0,tau_f)
x_b = 0 para la mitad del batch
x_b = 1 para la otra mitad
target = 0
```

En la corrida final:

- `batch_boundary = 256`.
- `128` puntos en `x=0`.
- `128` puntos en `x=1`.
- La perdida es `L_bc = mean(u_theta(x_b,tau_b)^2)`.

### 3.4 Grilla de evaluacion

La evaluacion final no usa minibatches aleatorios. Se arma una grilla regular:

```text
nx = 180
nt = 180
x in linspace(0,1,nx)
tau in linspace(0,tau_f,nt)
```

El error reportado es:

```text
error_L2_rel = ||u_pred - u_exact||_2 / ||u_exact||_2
```

La solucion exacta se evalua en toda la grilla `180 x 180`.

## 4. Arquitectura de red

La red es una MLP totalmente conectada:

```text
entrada: [x, tau]
capas ocultas: tanh
salida: u_theta(x,tau)
```

### 4.1 Escalado de entrada

Antes de pasar por la MLP se escala cada coordenada al intervalo `[-1,1]`:

```text
z = 2 * ([x,tau] - [0,0]) / ([1,tau_f] - [0,0]) - 1
```

Este escalado ocurre dentro del grafo de PyTorch, por lo que autograd incorpora
correctamente las constantes de derivacion para `tau`.

### 4.2 Inicializacion

Para todas las capas lineales:

- Pesos: `torch.nn.init.xavier_normal_`.
- Sesgos: `0`.
- Activacion: `tanh` en todas las capas ocultas.
- Ultima capa: lineal sin activacion.

Esto replica el criterio general del paper y del repositorio de los autores,
donde se usa inicializacion de Glorot/Xavier y activaciones hiperbolicas.

### 4.3 Arquitectura final principal

La corrida final principal usa:

```text
layers = [2, 50, 50, 50, 1]
```

Es decir:

- 2 entradas (`x`, `tau`).
- 3 capas ocultas de 50 neuronas.
- 1 salida escalar (`u`).

La busqueda bayesiana tambien probo otras arquitecturas; ver seccion 8.

## 5. Funcion de perdida

La perdida general es:

```text
L = L_res + lambda_ic L_ic + lambda_bc L_bc
```

con:

```text
L_res = mean((u_tau - u_xx)^2)
L_ic  = mean((u_theta(x,0)-u0(x))^2)
L_bc  = mean(u_theta(0,tau)^2 y u_theta(1,tau)^2)
```

## 6. Modelos comparados

### 6.1 M1: PINN clasica

La PINN clasica usa:

```text
lambda_ic = 1
lambda_bc = 1
```

No hay balance adaptativo. La perdida es:

```text
L_M1 = L_res + L_ic + L_bc
```

### 6.2 M2: PINN con correccion del paper

La correccion usada es el Algoritmo 1 del paper: aprendizaje/adaptacion de pesos
de perdida usando estadisticas de gradientes. No se usa la arquitectura nueva
del paper.

Para cada termino de dato `i in {ic, bc}` se calcula:

```text
lambda_hat_i = max_theta |grad_theta L_res| / mean_theta |grad_theta L_i|
```

Detalles de implementacion:

- Para replicar el repositorio de autores, las estadisticas se calculan sobre
  los tensores de pesos de cada capa lineal, no sobre sesgos.
- `max_theta |grad L_res|`: maximo global entre capas.
- `mean_theta |grad L_i|`: media de las medias absolutas por capa.
- `eps = 1e-12` para evitar division por cero.
- `lambda_hat` se recorta a `[1e-6, 1e6]` por estabilidad numerica.

Actualizacion por media movil:

```text
lambda_i <- beta * lambda_i + (1-beta) * lambda_hat_i
```

Corrida final:

- `beta = 0.9`.
- Actualizacion cada `10` iteraciones.
- Valores iniciales: `lambda_ic = 1`, `lambda_bc = 1`.

## 7. Optimizacion y entrenamiento

### 7.1 Semilla

La semilla final principal es:

```text
seed = 0
```

Se fija con:

```text
numpy.random.seed(seed)
torch.manual_seed(seed)
```

En la busqueda bayesiana se usa `seed + trial.number` para que cada trial tenga
una inicializacion y muestreo reproducibles pero no identicos.

### 7.2 Optimizador principal

Corrida final principal:

```text
optimizer = Adam
lr inicial = 1e-3
weight_decay = 0
```

Scheduler:

```text
ExponentialLR(gamma = 0.9 ** (1/1000))
```

Equivale a decaer el learning rate por un factor `0.9` cada `1000` pasos, sin
escalones.

### 7.3 Threads y hardware

Se configuro PyTorch para usar todos los cores logicos disponibles:

```text
threads = 16
device = cpu
```

CPU detectada:

```text
12th Gen Intel(R) Core(TM) i5-1240P
16 CPUs logicos
```

### 7.4 Presupuesto final principal

Corrida final M1/M2:

```text
iterations = 3000
batch_residual = 256
batch_initial = 256
batch_boundary = 256
log_every = 500
nx = 180
nt = 180
```

Comando usado:

```bash
conda run --no-capture-output -n tp-optimizacion-ger \
  python experimento.py todos \
  --iters 3000 --batch 256 --width 50 --depth 3 \
  --lr 0.001 --adaptive-beta 0.9 \
  --log-every 500 --nx 180 --nt 180 --threads 16
```

## 8. Busqueda bayesiana de hiperparametros

Se uso Optuna con sampler TPE:

```text
sampler = TPESampler(seed=0, multivariate=True)
direction = minimize
objective = error_L2_rel en grilla 90 x 90
modelo evaluado = M2
```

Espacio de busqueda:

```text
width          in {24, 32, 48, 64, 80}
depth          in [2, 5] entero
lr             log-uniforme en [1e-4, 3e-3]
batch          in {128, 256, 384}
adaptive_beta  uniforme en [0.75, 0.97]
```

Presupuesto usado:

```text
trials = 8
iterations por trial = 800
adaptive_every = 10
threads = 16
```

Comando usado:

```bash
conda run --no-capture-output -n tp-optimizacion-ger \
  python experimento.py bayes \
  --trials 8 --search-iters 800 \
  --adaptive-every 10 --threads 16
```

Mejor trial corto:

```text
trial = 3
error_L2_rel corto = 0.1413831514
width = 48
depth = 5
lr = 0.0010165558775187275
batch = 384
adaptive_beta = 0.7632496037584393
lambda_ic_final corto = 1088.77
lambda_bc_final corto = 2945.52
```

Observacion importante: este mejor trial fue seleccionado con solo `800`
iteraciones. Al validarlo durante `3000` iteraciones, el `beta` bajo hizo crecer
los pesos adaptativos hasta orden `1e6`, y el error final fue peor que el
regimen conservador basado en `beta=0.9`.

La figura `outputs/sensibilidad_regimenes.png` documenta esta diferencia.

## 9. Comparacion de optimizadores

Se comparo el modo `M2` con presupuesto fijo, no una busqueda completa por
optimizador. El objetivo fue medir sensibilidad al algoritmo de optimizacion.

Configuracion:

```text
iterations = 1000
batch = 128
width = 32
depth = 3
lr = 1e-3
adaptive_beta = 0.9
seed = 0
threads = 16
```

Optimizadores:

- `SGD` con momentum `0.9`.
- `Adam`.
- `AdamW`.
- `Adam + LBFGS` con `10` pasos de refinamiento.

Comando usado:

```bash
conda run --no-capture-output -n tp-optimizacion-ger \
  python experimento.py optimizadores \
  --iters 1000 --batch 128 --width 32 --depth 3 \
  --lr 0.001 --adaptive-beta 0.9 \
  --log-every 500 --nx 120 --nt 120 \
  --lbfgs-steps 10 --threads 16
```

Resultado:

```text
Adam + LBFGS(10): error_L2 = 0.166899
Adam:             error_L2 = 0.176340
AdamW:            error_L2 = 0.176340
SGD:              NaN con este learning rate y pesos adaptativos
```

La divergencia de SGD esta documentada como resultado experimental: bajo el
mismo `lr=1e-3` y con el balance adaptativo activo, la dinamica produjo valores
`NaN`. No se ajusto un learning rate especial para SGD en esta comparacion.

## 10. Resultados principales

Corrida final conservadora (`3000` iteraciones, arquitectura `[2,50,50,50,1]`,
Adam, `beta=0.9`):

```text
M1 error_L2_rel = 0.2931340748
M2 error_L2_rel = 0.1029122186
mejora M1 -> M2 = 2.8483894213x
```

Pesos finales:

```text
M1: lambda_ic = 1, lambda_bc = 1
M2: lambda_ic = 219148.36, lambda_bc = 276036.68
```

Tiempo de entrenamiento:

```text
M1: 134.47 s
M2: 421.23 s
```

La diferencia de tiempo se debe a que M2 calcula gradientes adicionales cada
`10` iteraciones para estimar `lambda_hat_ic` y `lambda_hat_bc`.

## 11. Figuras y archivos generados

Archivos principales:

- `outputs/M1_solucion.png`: exacta, prediccion M1 y error absoluto.
- `outputs/M2_solucion.png`: exacta, prediccion M2 y error absoluto.
- `outputs/M1_historia.png`: perdidas y lambdas de M1.
- `outputs/M2_historia.png`: perdidas y lambdas de M2.
- `outputs/M1_gradientes.png`: histogramas de gradientes por capa para M1.
- `outputs/M2_gradientes.png`: histogramas de gradientes por capa para M2.
- `outputs/comparacion_modelos.png`: barras de error L2 para M1/M2.
- `outputs/cortes_temporales.png`: perfiles espaciales en tiempos fijos.
- `outputs/bayes_historia.png`: evolucion de trials y mejor acumulado.
- `outputs/bayes_parametros.png`: error vs parametros de busqueda.
- `outputs/bayes_heatmap_arquitectura.png`: mejor error por ancho/profundidad.
- `outputs/sensibilidad_regimenes.png`: comparacion de regimen bayesiano corto
  validado a 3000 iteraciones vs regimen conservador.
- `outputs/comparacion_optimizadores.png`: comparacion de optimizadores.

Tablas:

- `outputs/metricas_modelos.csv`
- `outputs/resumen.json`
- `outputs/bayes_trials.csv`
- `outputs/bayes_mejor.json`
- `outputs/sensibilidad_regimenes.csv`
- `outputs/optimizadores.csv`

## 12. Reproducibilidad desde cero

Crear entorno:

```bash
cd /home/ger/Desktop/tp_optimizacion_udesa/Optimizacion-tp-final/ger
conda env create -f environment.yml
conda activate tp-optimizacion-ger
```

Si el entorno ya existe:

```bash
conda activate tp-optimizacion-ger
python -m pip install -r requirements.txt
```

Ejecutar busqueda bayesiana:

```bash
python experimento.py bayes --trials 8 --search-iters 800 --threads 16
```

Ejecutar comparacion principal:

```bash
python experimento.py todos \
  --iters 3000 --batch 256 --width 50 --depth 3 \
  --lr 0.001 --adaptive-beta 0.9 \
  --log-every 500 --nx 180 --nt 180 --threads 16
```

Ejecutar comparacion de optimizadores:

```bash
python experimento.py optimizadores \
  --iters 1000 --batch 128 --width 32 --depth 3 \
  --lr 0.001 --adaptive-beta 0.9 \
  --log-every 500 --nx 120 --nt 120 \
  --lbfgs-steps 10 --threads 16
```

Generar sensibilidad de regimenes:

```bash
python analisis_sensibilidad.py
```

Compilar reporte LaTeX:

```bash
tectonic reporte.tex
```

## 13. Archivos de codigo

- `problema.py`: define el problema fisico, solucion analitica, condicion
  inicial, escala dimensional y grillas.
- `pinn.py`: define la MLP, escalado de entrada e inicializacion Xavier.
- `entrenamiento.py`: calcula perdidas, derivadas autograd, M1/M2, optimizadores,
  scheduler, snapshots de gradientes y evaluacion.
- `graficos.py`: genera figuras de solucion, historia, gradientes, bayes,
  optimizadores y cortes temporales.
- `experimento.py`: CLI principal.
- `analisis_sensibilidad.py`: postproceso de estabilidad de regimenes.
- `reporte.tex`: reporte LaTeX compilable con Tectonic.
- `reporte.md`: resumen narrativo en Markdown.
