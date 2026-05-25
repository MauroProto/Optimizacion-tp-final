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

La red final principal es `[2, 50, 50, 50, 1]`, activacion `tanh`,
inicializacion Xavier normal, sesgos cero, Adam con `lr=1e-3`, scheduler
exponencial `0.9` cada `1000` pasos, `3000` iteraciones, `batch=256`, semilla
`0`, CPU con `16` threads.

## Resultados principales

Corrida final conservadora:

```text
M1 error_L2_rel = 0.2931340748
M2 error_L2_rel = 0.1029122186
mejora M1 -> M2 = 2.8483894213x
```

La busqueda bayesiana corta encontro un trial con `width=48`, `depth=5`,
`lr=0.0010166`, `batch=384`, `beta=0.76325`, error corto `0.14138`. Al
validarlo a `3000` iteraciones, los pesos adaptativos crecieron hasta orden
`1e6`; esa sensibilidad queda documentada en `sensibilidad_regimenes.png`.

## Artefactos

Figuras y tablas quedan en `outputs/`. Las mas importantes son:

- `comparacion_modelos.png`
- `M1_solucion.png`, `M2_solucion.png`
- `M1_gradientes.png`, `M2_gradientes.png`
- `bayes_historia.png`, `bayes_parametros.png`
- `sensibilidad_regimenes.png`
- `comparacion_optimizadores.png`
