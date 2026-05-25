# TP alternativa - PINN para difusion de calor 1D

Este directorio replica el nucleo metodologico de Wang, Teng y Perdikaris
(`GradientPathologiesPINNs`) sobre otro problema: difusion de calor 1D en una
barra metalica. Se comparan:

- `M1`: PINN clasica.
- `M2`: PINN clasica con balance adaptativo de terminos de perdida usando las
  estadisticas de gradientes del Algoritmo 1 del paper.
- Solucion analitica por separacion de variables.

No se implementa la arquitectura nueva del paper (`M3/M4`).

## Entorno

```bash
conda env create -f environment.yml
conda activate tp-optimizacion-ger
```

Si el entorno ya existe:

```bash
conda activate tp-optimizacion-ger
python -m pip install -r requirements.txt
```

## Corridas principales

Desde esta carpeta:

```bash
python experimento.py bayes --trials 100 --target-trials --search-iters 800 \
  --storage sqlite:///outputs/optuna_calor_1d.db --study-name calor_1d_m2_100
python validar_regimenes.py --top-k 6 --iters 3000
python experimento.py todos --iters 3000 --batch 256 --width 80 --depth 3 \
  --lr 0.0020667030862179673 --adaptive-beta 0.9609402971992413
python experimento.py semillas --seeds 0 1 2 --iters 3000 --batch 256 \
  --width 80 --depth 3 --lr 0.0020667030862179673 \
  --adaptive-beta 0.9609402971992413
python experimento.py optimizadores --iters 1000 --batch 128 --width 32 --depth 3
python plots_examen.py
```

Tambien se reviso el notebook de examen en
`/home/ger/Desktop/ejemplos_figuras_examen/jurado_german_parcial.ipynb`. Sus
graficos de `f(x_k)` y trayectorias 2D se trasladaron al caso PINN como curvas
de objetivo de Optuna y trayectoria sobre el plano `log10(lr)`-`beta`. La
trayectoria 2D de pesos no se grafica literalmente porque la red tiene miles de
parametros y ese contorno no seria interpretable.

Los resultados quedan en `outputs/`. Los plots principales son:

- `M1_solucion.png`, `M2_solucion.png`: solucion exacta, prediccion y error.
- `M1_gradientes.png`, `M2_gradientes.png`: histogramas de gradientes por capa.
- `M1_historia.png`, `M2_historia.png`: perdidas y evolucion de lambdas.
- `comparacion_modelos.png`: error relativo contra la solucion analitica.
- `cortes_temporales.png`: perfiles espaciales en tiempos fijos.
- `bayes_*.png`: diagnosticos de la busqueda bayesiana.
- `validacion_regimenes.png`: validacion larga de los mejores trials.
- `comparacion_semillas.png`: variabilidad con tres semillas.
- `paper_*.png`: figuras de comunicacion inspiradas en el paper.
- `examen_*.png`: analogos de los graficos del notebook de examen.

## Problema fisico

Variables dimensionales: posicion `X in [0,L]`, tiempo `t`, temperatura `T`.
Con `x=X/L`, `tau=alpha t/L^2` y `u=(T-T_amb)/DeltaT`, el modelo queda

```text
u_tau - u_xx = 0,        x in (0,1), tau in (0,tau_f)
u(0,tau) = u(1,tau) = 0
u(x,0) = sin(pi x) + 0.5 sin(3 pi x) + 0.25 sin(5 pi x)
```

La solucion analitica usada para evaluar es

```text
u(x,tau) = sum_n a_n sin(n pi x) exp(-(n pi)^2 tau)
```

con modos `(n,a_n) = (1,1), (3,0.5), (5,0.25)`.
