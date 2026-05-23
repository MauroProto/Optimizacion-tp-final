# TP Final - Replica de PINN para Helmholtz 2D

Implementacion en PyTorch del caso Helmholtz 2D del paper:
*"Understanding and Mitigating Gradient Pathologies in Physics-Informed Neural Networks"*
(Wang, Teng, Perdikaris, 2021).

## Que se hizo

El paper propone 2 mejoras a las PINN. **Solo replicamos la primera**: el balanceo
adaptativo de gradientes durante el entrenamiento (`learning rate annealing`).
La segunda mejora (arquitectura nueva con encoders) no la usamos.

Entonces se comparan 3 cosas:

1. **M1**: PINN baseline (sin la mejora). La perdida es `L_res + L_bc`, sin balance.
2. **M2**: PINN con la mejora del paper. La perdida es `L_res + lambda_bc * L_bc`,
   donde `lambda_bc` se ajusta solo para balancear los gradientes.
3. **FD**: solver clasico por diferencias finitas. Sirve como control: si FD da
   bien la solucion, entonces el problema esta bien planteado.

Tambien hay una comparacion adicional de **optimizadores**: Adam puro vs Adam + LBFGS.

## El problema

Resolvemos la ecuacion de Helmholtz:

    Δu + λu = q(x, y)     en [-1, 1] x [-1, 1]
    u = 0                 en los bordes

Con `λ = 1` y la solucion fabricada `u(x, y) = sin(π·x) · sin(4π·y)`.

## Estructura de los archivos

Solo 5 archivos de codigo:

    problema.py             definicion de la PDE, solucion exacta, samplers
    pinn.py                 la red neuronal (MLP comun con tanh)
    entrenar.py             el loop de entrenamiento (Adam + opcional LBFGS)
    diferencias_finitas.py  el solver clasico FD
    experimento.py          el script principal que corre todo

## Como correr

Primera vez (instalar dependencias):

    cd Mauro
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Despues, los experimentos:

    # Corre todo: M1, M2, FD. Hace plots y un resumen.
    python experimento.py todos

    # Solo M1 (baseline, sin balance de gradientes)
    python experimento.py m1

    # Solo M2 (con la mejora del paper)
    python experimento.py m2

    # Solo el solver clasico FD
    python experimento.py fd

    # Comparacion Adam vs Adam+LBFGS
    python experimento.py optimizadores

    # Grid search sobre M2 (12 combinaciones: anchos {30, 50, 100} x profund {3, 5} x lr {1e-3, 5e-4})
    # Los anchos y profundidades coinciden con los que prueba la Tabla 2 del paper.
    python experimento.py grid

### Jugar con los hiperparametros

    # Mas iteraciones (default es 40001 - como el paper)
    python experimento.py todos --iters 60000

    # Cambiar el tamano de la red (default 3 capas ocultas de 50)
    python experimento.py todos --capas 2 100 100 100 1

    # Cambiar batch y learning rate
    python experimento.py todos --batch 256 --lr 5e-4

    # Para pruebas rapidas, pocas iteraciones
    python experimento.py todos --iters 2000

    # Grid search con mas iters por celda (mas lento, mas confiable)
    python experimento.py grid --grid-iters 10000

### Donde quedan los resultados

En `outputs/`:
- `M1_solucion.png`, `M1_historia.png`: plots de M1
- `M2_solucion.png`, `M2_historia.png`: plots de M2
- `FD_solucion.png`: plot del solver clasico
- `comparacion_optimizadores.png`: barras Adam vs Adam+LBFGS
- `grid.png`, `grid.csv`: resultados del grid search
- `resumen.txt`: tabla con los errores

## Que esperar

Con las 40001 iteraciones que usa el paper, sobre 1 seed:

| Modo                          | Error L2 relativo esperado |
|-------------------------------|----------------------------|
| M1 (PINN sin balance)         | ~ 1e-1 (malo en los bordes) |
| M2 (PINN con balance)         | ~ 1e-2 (mejora ~10x)        |
| FD con n=200                  | ~ 1e-3 (referencia clasica) |

La mejora de M1 a M2 es la prueba experimental de que el paper tiene razon:
solo balanceando los gradientes ya bajamos el error 10 veces.
