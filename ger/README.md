# Reproduccion alternativa: PINNs y difusion de calor 1D

Este directorio contiene el trabajo desarrollado para la consigna alternativa.
El objetivo es replicar el nucleo metodologico de Wang, Teng y Perdikaris
(`Understanding and mitigating gradient pathologies in physics-informed neural
networks`) sobre un problema distinto al del paper: difusion de calor 1D en una
barra metalica.

La reproduccion es alternativa por diseno:

- No se reproducen los mismos problemas del paper.
- No se implementa la arquitectura nueva de los autores (`M3/M4`).
- Si se replica el diagnostico central de patologias de gradiente y la
  correccion por pesos adaptativos de perdida (`M2`).

## Resumen ejecutivo

Problema fisico:

```text
u_tau - u_xx = 0,        0 < x < 1, 0 < tau < 0.25
u(0,tau) = u(1,tau) = 0
u(x,0) = sin(pi x) + 0.5 sin(3 pi x) + 0.25 sin(5 pi x)
```

Modelos comparados:

- `M1`: PINN clasica con perdida no ponderada.
- `M2`: misma PINN, pero con pesos adaptativos para condicion inicial y borde
  calculados con estadisticas de gradientes, siguiendo el Algoritmo 1 del
  paper.
- `M1B`: M1 optimizada con una busqueda bayesiana propia, sin pesos adaptativos.
  Es una ablacion extra para separar tuning de hiperparametros y mejora del
  paper.
- Solucion analitica por separacion de variables.

Resultado principal:

```text
M1 error_L2_rel = 0.2643327358
M1B error_L2_rel = 0.1821036631
M2 error_L2_rel = 0.0285351083
mejora M1 -> M2 = 9.26x
mejora M1 -> M1B = 1.45x
brecha M1B -> M2 = 6.38x
```

Configuracion final:

```text
layers = [2, 80, 80, 80, 1]
activation = tanh
initialization = Xavier normal, bias cero
optimizer = Adam
lr = 0.0020667030862179673
iterations = 3000
batch = 256 para residuo, condicion inicial y borde
adaptive_beta = 0.9609402971992413
adaptive_every = 10
seed = 0
threads = 16
```

## Reportes

Los documentos principales ya estan compilados:

- `reporte.pdf`: reporte completo del experimento de calor 1D.
- `reporte.tex`: fuente LaTeX del reporte experimental.
- `reporte_paper_core.pdf`: resumen del paper, ideas centrales y aplicacion al
  proyecto.
- `reporte_paper_core.tex`: fuente LaTeX del reporte sobre el paper.
- `reporte.md`: resumen corto en Markdown.
- `METODOLOGIA.md`: bitacora tecnica exhaustiva, con generacion de datos,
  hiperparametros, comandos y artefactos.

## Instalacion

Desde esta carpeta:

```bash
conda env create -f environment.yml
conda activate tp-optimizacion-ger
```

Si el entorno ya existe:

```bash
conda activate tp-optimizacion-ger
python -m pip install -r requirements.txt
```

El entorno incluye PyTorch CPU, Optuna, Matplotlib, Pandas, SciPy y Tectonic.
El experimento se ejecuto en CPU usando todos los cores logicos disponibles.

## Reproduccion rapida

Los resultados ya estan en `outputs/`. Para regenerar solo reportes y figuras
de postproceso, sin reentrenar:

```bash
python analisis_sensibilidad.py
python plots_examen.py
tectonic reporte.tex
tectonic reporte_paper_core.tex
```

## Reproduccion completa

La reproduccion completa implica entrenamiento y busqueda bayesiana. Puede tomar
varias horas en CPU.

```bash
python experimento.py bayes \
  --trials 100 --target-trials --search-iters 800 \
  --storage sqlite:///outputs/optuna_calor_1d.db \
  --study-name calor_1d_m2_100 \
  --threads 16

python validar_regimenes.py --top-k 6 --iters 3000 --threads 16

python experimento.py bayes-m1 \
  --trials 100 --target-trials --search-iters 800 \
  --storage sqlite:///outputs/optuna_calor_1d_m1.db \
  --study-name calor_1d_m1_100 \
  --threads 16

python validar_m1b.py --top-k 6 --iters 3000 --threads 16

python experimento.py todos \
  --iters 3000 --batch 256 --width 80 --depth 3 \
  --lr 0.0020667030862179673 \
  --adaptive-beta 0.9609402971992413 \
  --log-every 500 --nx 180 --nt 180 --threads 16

python experimento.py m1b \
  --iters 3000 --log-every 500 --nx 180 --nt 180 --threads 16

python experimento.py semillas \
  --seeds 0 1 2 \
  --iters 3000 --batch 256 --width 80 --depth 3 \
  --lr 0.0020667030862179673 \
  --adaptive-beta 0.9609402971992413 \
  --log-every 1000 --nx 120 --nt 120 --threads 16

python experimento.py optimizadores \
  --iters 1000 --batch 128 --width 32 --depth 3 \
  --adaptive-beta 0.9 \
  --log-every 500 --nx 120 --nt 120 \
  --lbfgs-steps 10 --threads 16

python analisis_sensibilidad.py
python plots_examen.py
tectonic reporte.tex
tectonic reporte_paper_core.tex
```

## Problema fisico

La barra tiene longitud `L = 1 m`, difusividad termica
`alpha = 1e-4 m^2/s`, extremos a temperatura ambiente y condicion inicial
compatible con los bordes. Se usan variables adimensionales:

```text
x   = X / L
tau = alpha t / L^2
u   = (T - T_amb) / DeltaT
```

El tiempo final adimensional es `tau_f = 0.25`, equivalente a `2500 s`.

La solucion analitica usada para evaluar es:

```text
u(x,tau) = sum_n a_n sin(n pi x) exp(-(n pi)^2 tau)
(n,a_n) in {(1,1.0), (3,0.5), (5,0.25)}
```

La solucion analitica no se usa como dato interior de entrenamiento. Solo se
usa para evaluar error y construir figuras.

## Datos generados

No hay dataset fijo de entrenamiento. En cada iteracion se remuestrean puntos:

- Residuo PDE: `x_r ~ U(0,1)`, `tau_r ~ U(0,tau_f)`.
- Condicion inicial: `x_0 ~ U(0,1)`, `tau_0 = 0`.
- Bordes: `tau_b ~ U(0,tau_f)`, mitad de puntos en `x=0` y mitad en `x=1`.

Las derivadas `u_tau` y `u_xx` se calculan con autograd de PyTorch. La
evaluacion final usa una grilla regular `180 x 180` y reporta error relativo
`L2` contra la solucion analitica.

## Metodologia replicada del paper

El paper identifica que una PINN puede reducir el residuo de la PDE mientras
ignora condiciones iniciales o de borde, porque los gradientes de los terminos
de perdida quedan desbalanceados. La correccion implementada en este trabajo es:

```text
L_M1 = L_res + L_ic + L_bc

L_M2 = L_res + lambda_ic L_ic + lambda_bc L_bc
```

con pesos adaptativos:

```text
lambda_hat_i = max_theta |grad L_res| / mean_theta |grad L_i|
lambda_i <- beta lambda_i + (1 - beta) lambda_hat_i
```

Se calculan estadisticas sobre los pesos de las capas lineales, como en el
repositorio de los autores. La actualizacion ocurre cada 10 iteraciones.

## Busqueda bayesiana

La busqueda principal de hiperparametros uso Optuna/TPE sobre `M2`:

```text
trials = 100
iterations por trial = 800
objective = error_L2_rel de M2 en grilla 90 x 90
sampler = TPESampler(seed=0, multivariate=True)
```

Espacio de busqueda:

```text
width          in {24, 32, 48, 64, 80}
depth          in [2, 5]
lr             log-uniforme en [1e-4, 3e-3]
batch          in {128, 256, 384}
adaptive_beta  uniforme en [0.75, 0.97]
```

El mejor trial corto fue el `91`, con error `0.0319695`. La validacion larga de
los mejores regimenes eligio el trial `82`:

```text
width = 80
depth = 3
lr = 0.0020667030862179673
batch = 256
adaptive_beta = 0.9609402971992413
error validado = 0.0284171861
```

Esta distincion es importante: el mejor presupuesto corto no necesariamente es
el regimen mas estable al entrenar 3000 iteraciones.

Como extra, se hizo una segunda busqueda bayesiana para `M1`, sin
`adaptive_beta`, para construir `M1B`. Esta busqueda no pertenece al metodo del
paper; sirve para verificar si el tuning de la red clasica alcanza a explicar la
mejora de `M2`.

```text
objective = error_L2_rel de M1 en grilla 90 x 90
width      in {24, 32, 48, 64, 80}
depth      in [2, 5]
lr         log-uniforme en [1e-4, 3e-3]
batch      in {128, 256, 384}
```

El mejor trial corto de M1 fue el `91`, con error `0.2629275`, pero la
validacion larga eligio el trial `83`:

```text
width = 48
depth = 5
lr = 0.0020996352226311
batch = 128
error validado = 0.1845979044
error final 180 x 180 = 0.1821036631
```

## Resultados adicionales

Robustez con tres semillas:

```text
M1 error_L2_rel = 0.240136 +/- 0.022498
M2 error_L2_rel = 0.055138 +/- 0.023188
```

Comparacion de optimizadores con presupuesto fijo:

```text
Adam lr=5e-4:              error_L2 = 0.167167
AdamW lr=5e-4:             error_L2 = 0.167167
Adam lr=5e-4 + LBFGS(10):  error_L2 = 0.167244
Adam lr=1e-3:              error_L2 = 0.176340
SGD lr=1e-4, momentum=0.9: error_L2 = 0.220924
```

La decision final fue usar Adam para las corridas principales.

Comparacion extra con M1 optimizada:

```text
M1  error_L2_rel = 0.264333
M1B error_L2_rel = 0.182104
M2  error_L2_rel = 0.028535
```

M1B mejora a M1 en `1.45x`, pero queda `6.38x` por encima del error de M2. Esto
apoya que el balance adaptativo del paper aporta mas que solo el tuning de
hiperparametros de una PINN clasica.

## Figuras principales

Figuras tipo paper:

- `outputs/paper_comparacion_soluciones.png`: solucion exacta, M1, M2 y mapas
  de error.
- `outputs/paper_balance_gradientes.png`: magnitudes medias de gradientes por
  capa y termino de perdida.
- `outputs/paper_lambda_evolucion.png`: evolucion de `lambda_ic` y `lambda_bc`.

Figuras de solucion:

- `outputs/M1_solucion.png`
- `outputs/M1B_solucion.png`
- `outputs/M2_solucion.png`
- `outputs/cortes_temporales.png`
- `outputs/comparacion_modelos.png`
- `outputs/comparacion_modelos_m1b.png`

Figuras de diagnostico y busqueda:

- `outputs/M1_historia.png`
- `outputs/M1B_historia.png`
- `outputs/M2_historia.png`
- `outputs/M1_gradientes.png`
- `outputs/M1B_gradientes.png`
- `outputs/M2_gradientes.png`
- `outputs/bayes_historia.png`
- `outputs/bayes_parametros.png`
- `outputs/bayes_heatmap_arquitectura.png`
- `outputs/bayes_m1_historia.png`
- `outputs/bayes_m1_parametros.png`
- `outputs/bayes_m1_heatmap_arquitectura.png`
- `outputs/validacion_regimenes.png`
- `outputs/validacion_m1b.png`
- `outputs/sensibilidad_regimenes.png`
- `outputs/comparacion_semillas.png`
- `outputs/comparacion_optimizadores.png`

Figuras inspiradas en el notebook de examen:

- `outputs/examen_bayes_convergencia.png`
- `outputs/examen_bayes_trayectoria_lr_beta.png`
- `outputs/examen_bayes_m1_convergencia.png`
- `outputs/examen_bayes_m1_trayectoria_lr_width.png`
- `outputs/examen_optimizadores_lr_error.png`

El notebook de examen revisado fue
`/home/ger/Desktop/ejemplos_figuras_examen/jurado_german_parcial.ipynb`. Sus
graficos de `f(x_k)` y trayectorias 2D se trasladaron al caso PINN como curvas
de objetivo de Optuna y trayectoria en el plano `log10(lr)`-`beta`. No se
grafico una trayectoria 2D en el espacio de pesos porque la red tiene miles de
parametros y esa proyeccion no seria interpretable.

## Tablas y archivos de salida

- `outputs/metricas_modelos.csv`: errores y tiempos de M1/M2.
- `outputs/metricas_modelos_m1b.csv`: comparacion M1/M2/M1B.
- `outputs/metricas_m1b.csv`: corrida final M1B.
- `outputs/resumen.json`: configuracion y resumen principal.
- `outputs/bayes_trials.csv`: todos los trials de Optuna.
- `outputs/bayes_mejor.json`: mejor trial corto.
- `outputs/bayes_m1_trials.csv`: todos los trials de Optuna para M1B.
- `outputs/bayes_m1_mejor.json`: mejor trial corto de M1B.
- `outputs/validacion_regimenes.csv`: validacion larga de mejores trials.
- `outputs/validacion_regimenes_mejor.json`: mejor regimen validado.
- `outputs/validacion_m1b.csv`: validacion larga de mejores trials M1.
- `outputs/validacion_m1b_mejor.json`: regimen M1B elegido.
- `outputs/resumen_m1b.json`: configuracion y resumen de M1B.
- `outputs/metricas_semillas.csv`: corridas por semilla.
- `outputs/metricas_semillas_resumen.csv`: media y dispersion por modelo.
- `outputs/optimizadores.csv`: comparacion de optimizadores.
- `outputs/sensibilidad_regimenes.csv`: relacion error corto/error largo.
- `outputs/optuna_calor_1d.db`: base SQLite de Optuna.
- `outputs/optuna_calor_1d_m1.db`: base SQLite de Optuna para M1B.

## Estructura del codigo

- `problema.py`: parametros fisicos, condicion inicial, solucion analitica y
  grillas.
- `pinn.py`: MLP, escalado de entrada e inicializacion.
- `entrenamiento.py`: perdidas, autograd, M1/M2, optimizadores, scheduler,
  evaluacion y snapshots de gradientes.
- `graficos.py`: funciones de graficado para soluciones, historiales,
  gradientes, Optuna y optimizadores.
- `experimento.py`: CLI principal.
- `validar_regimenes.py`: reentrena y valida los mejores trials de Optuna.
- `validar_m1b.py`: reentrena y valida los mejores trials de Optuna para M1.
- `analisis_sensibilidad.py`: postproceso de estabilidad de regimenes.
- `plots_examen.py`: figuras analogas al notebook de examen.
- `environment.yml` y `requirements.txt`: entorno reproducible.

## Lectura recomendada

Para evaluar rapidamente la entrega:

1. Leer `reporte_paper_core.pdf` para entender el paper y la motivacion.
2. Leer `reporte.pdf` para el experimento completo de calor 1D.
3. Revisar `METODOLOGIA.md` para detalles de datos, entrenamiento y comandos.
4. Mirar `outputs/paper_comparacion_soluciones.png`,
   `outputs/paper_balance_gradientes.png` y
   `outputs/paper_lambda_evolucion.png` para la evidencia principal.

## Limitaciones declaradas

- No se implementa la arquitectura nueva del paper.
- No se replican los ejemplos originales de Helmholtz, Klein-Gordon o
  Navier-Stokes.
- No se guarda la norma global de gradiente en cada iteracion; el diagnostico
  de gradientes sigue el enfoque del paper usando histogramas y magnitudes por
  capa al final del entrenamiento.
- La busqueda bayesiana se hizo con presupuesto corto y luego se valido con
  corridas largas para evitar elegir un regimen inestable.
- La optimizacion `M1B` es un extra de ablacion propio del proyecto, no una
  propuesta de los autores del paper.
