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
python experimento.py todos --iters 10000 --batch 256 --width 50 --depth 3
python experimento.py bayes --trials 12 --search-iters 2500
python experimento.py optimizadores --iters 4000 --batch 256
```

Los resultados quedan en `outputs/`. Los plots principales son:

- `M1_solucion.png`, `M2_solucion.png`: solucion exacta, prediccion y error.
- `M1_gradientes.png`, `M2_gradientes.png`: histogramas de gradientes por capa.
- `M1_historia.png`, `M2_historia.png`: perdidas y evolucion de lambdas.
- `comparacion_modelos.png`: error relativo contra la solucion analitica.
- `cortes_temporales.png`: perfiles espaciales en tiempos fijos.
- `bayes_*.png`: diagnosticos de la busqueda bayesiana.

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
