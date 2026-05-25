# Reproduccion alternativa: difusion de calor 1D

Este archivo resume el reporte. La documentacion exhaustiva esta en
`METODOLOGIA.md`; el reporte formal compilable esta en `reporte.tex`.

## Problema

Se modela una barra metalica 1D de longitud `L=1 m`, difusividad
`alpha=1e-4 m^2/s`, extremos a temperatura ambiente y condicion inicial
compatible con los bordes. Con `x=X/L`, `tau=alpha t/L^2` y
`u=(T-T_amb)/DeltaT`:

```text
u_tau - u_xx = 0,  0 < x < 1,  0 < tau < 0.25
u(0,tau) = u(1,tau) = 0
u(x,0) = sin(pi x) + 0.5 sin(3 pi x) + 0.25 sin(5 pi x)
```

La solucion analitica usada para evaluar es:

```text
u(x,tau) = sum_n a_n sin(n pi x) exp(-(n pi)^2 tau),
(n,a_n) = (1,1), (3,0.5), (5,0.25).
```

## Metodologia

Se comparan:

- `M1`: PINN clasica con `L = L_res + L_ic + L_bc`.
- `M2`: misma PINN con pesos adaptativos para `L_ic` y `L_bc` segun el
  Algoritmo 1 del paper. No se usa la arquitectura nueva `M3/M4`.

La red final principal es `[2, 80, 80, 80, 1]`, activacion `tanh`,
inicializacion Xavier normal, sesgos cero, Adam con `lr=0.0020667`, scheduler
exponencial `0.9` cada `1000` pasos, `3000` iteraciones, `batch=256`, semilla
`0`, CPU con `16` threads. Estos hiperparametros fueron elegidos tras 100
trials de Optuna y validacion larga de los mejores regimenes.

## Resultados principales

Corrida final validada:

```text
M1 error_L2_rel = 0.2643327358
M2 error_L2_rel = 0.0285351083
mejora M1 -> M2 = 9.26x
```

La busqueda bayesiana uso `100` trials. El mejor error corto fue el trial `91`
(`0.03197`), pero la validacion larga de los mejores trials eligio el trial
`82`: `width=80`, `depth=3`, `lr=0.0020667`, `batch=256`,
`beta=0.96094`. Con tres semillas, M2 obtuvo `0.0551 ± 0.0232` contra
`0.2401 ± 0.0225` de M1.

Ademas se incorporaron graficos analogos al notebook de examen de la materia:
objetivo por iteracion/trial, convergencia del mejor acumulado, trayectoria
sobre contornos en el plano `log10(lr)`-`beta` y sensibilidad de optimizadores
al learning rate. El contorno de trayectoria no se hizo en el espacio de pesos
de la PINN porque ese espacio es de alta dimension y no produce una figura 2D
interpretable.

## Artefactos

Figuras y tablas quedan en `outputs/`. Las mas importantes son:

- `comparacion_modelos.png`
- `M1_solucion.png`, `M2_solucion.png`
- `M1_gradientes.png`, `M2_gradientes.png`
- `bayes_historia.png`, `bayes_parametros.png`
- `validacion_regimenes.png`, `comparacion_semillas.png`
- `paper_comparacion_soluciones.png`, `paper_balance_gradientes.png`
- `sensibilidad_regimenes.png`
- `comparacion_optimizadores.png`
- `examen_bayes_convergencia.png`, `examen_bayes_trayectoria_lr_beta.png`
- `examen_optimizadores_lr_error.png`
