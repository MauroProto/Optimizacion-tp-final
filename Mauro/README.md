# TP Final - PINN para Helmholtz 2D

Replica simplificada en PyTorch del caso Helmholtz del paper de Wang, Teng y
Perdikaris (2021), aplicando solo la primera mejora propuesta: el balanceo
adaptativo de gradientes (`learning rate annealing`).

## Instalacion

```bash
cd Mauro
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como correr

```bash
python experimento.py todos           # M1 (baseline) + M2 (mejora) + FD
python experimento.py m1              # solo M1
python experimento.py m2              # solo M2
python experimento.py fd              # solo diferencias finitas
python experimento.py optimizadores   # Adam vs Adam + LBFGS
python experimento.py grid            # grid search de hiperparametros
```

Flags utiles:

```bash
--iters 2000               # menos iteraciones (mas rapido)
--capas 2 50 50 50 1       # arquitectura de la red
--batch 128                # puntos por iteracion
--lr 1e-3                  # learning rate
```

Los resultados (plots + tabla resumen) quedan en `outputs/`.

## Archivos

    problema.py             la PDE, solucion exacta, samplers
    pinn.py                 la red neuronal
    entrenar.py             el loop de entrenamiento
    diferencias_finitas.py  solver clasico FD
    experimento.py          script principal
