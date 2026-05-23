"""
Entrenamiento de la PINN.

Modos:
  - "M1": baseline. La perdida total es L_res + L_bc, sin balance.
  - "M2": con la mejora del paper. Multiplicamos L_bc por un lambda que se
          ajusta solo para balancear los gradientes (asi el borde no se "pisa"
          al optimizar el residuo).

En cada iteracion:
  1) Muestreamos puntos del interior y del borde.
  2) Pedimos a PyTorch las derivadas u_xx y u_yy con autograd.
  3) Calculamos:
        L_res = (u_xx + u_yy + λu - q)²        en el interior
        L_bc  = (u - 0)²                       en el borde
  4) Sumamos: perdida = L_res + lambda_bc · L_bc
  5) Si es M2, actualizamos lambda_bc cada 10 iters segun la formula del paper.
  6) Backprop + paso de Adam.
"""

import numpy as np
import torch

from pinn import PINN
from problema import LAMBDA, forcing, muestrear_borde, muestrear_interior


def derivada(salida, entrada):
    """Calcula d(salida)/d(entrada) con autograd.

    Necesitamos create_graph=True para poder derivar otra vez despues
    (queremos derivadas segundas).
    """
    return torch.autograd.grad(
        salida, entrada,
        grad_outputs=torch.ones_like(salida),
        create_graph=True,
    )[0]


def calcular_residuo(red, puntos):
    """Devuelve u_xx + u_yy + λ·u evaluado en los puntos dados.

    Esta es la cuenta del lado izquierdo de la ecuacion de Helmholtz.
    Como PyTorch puede derivar automaticamente, sale solo.
    """
    # Separamos x e y, y le pedimos a PyTorch que guarde como hacer las derivadas.
    x = puntos[:, 0:1].clone().detach().requires_grad_(True)
    y = puntos[:, 1:2].clone().detach().requires_grad_(True)

    # u(x, y) en estos puntos.
    u = red(torch.cat([x, y], dim=1))

    # Primeras derivadas.
    u_x = derivada(u, x)
    u_y = derivada(u, y)

    # Segundas derivadas.
    u_xx = derivada(u_x, x)
    u_yy = derivada(u_y, y)

    return u_xx + u_yy + LAMBDA * u


def calcular_lambda_sugerido(red, L_res, L_bc_ponderada):
    """Calcula que tan grande deberia ser lambda_bc para balancear gradientes.

    Idea del paper:
      - Miramos los gradientes de L_res respecto a los pesos de cada capa,
        y nos quedamos con el MAXIMO en valor absoluto.
      - Hacemos lo mismo con L_bc, pero con el PROMEDIO.
      - Si max(grad_res) >> promedio(grad_bc), significa que el residuo "pisa"
        al borde. Entonces lambda_bc tiene que ser grande para compensar.

    Devuelve: lambda_bc sugerido (un numero).
    """
    pesos = [c.weight for c in red.capas]

    maxs_grad_res = []
    medias_grad_bc = []

    for w in pesos:
        # Calculamos d(L)/d(w) sin destruir el grafo (lo vamos a necesitar despues).
        grad_res = torch.autograd.grad(L_res, w, retain_graph=True)[0]
        grad_bc = torch.autograd.grad(L_bc_ponderada, w, retain_graph=True)[0]
        maxs_grad_res.append(grad_res.abs().max())
        medias_grad_bc.append(grad_bc.abs().mean())

    max_global = torch.stack(maxs_grad_res).max()
    promedio_global = torch.stack(medias_grad_bc).mean()

    return (max_global / promedio_global).item()


def entrenar(modo, capas, n_iter=40001, batch=128, lr=1e-3, lbfgs_pasos=0):
    """Entrena una PINN.

    Argumentos:
        modo: "M1" (baseline) o "M2" (con balance de gradientes del paper).
        capas: lista de tamanos, ej. [2, 50, 50, 50, 1].
        n_iter: cuantas iteraciones de Adam.
        batch: cuantos puntos por iteracion (en interior y en borde).
        lr: learning rate inicial de Adam.
        lbfgs_pasos: si > 0, refina con LBFGS al final (segundo optimizador).

    Devuelve:
        red entrenada, historia (dict con listas de losses, lambda_bc, etc.).
    """
    red = PINN(capas)

    # Optimizador principal: Adam con learning rate que decae 0.9 cada 1000 pasos.
    optimizador = torch.optim.Adam(red.parameters(), lr=lr)
    factor_decay = 0.9 ** (1.0 / 1000)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizador, gamma=factor_decay)

    # En M2 usamos lambda_bc adaptativo. En M1 queda fijo en 1 (sin balance).
    usa_balance = (modo == "M2")
    lambda_bc = 1.0

    historia = {"iter": [], "L_res": [], "L_bc": [], "lambda_bc": []}

    print(f"Entrenando {modo} ({'con balance' if usa_balance else 'sin balance'})...")

    for it in range(n_iter):
        # Puntos nuevos en cada iteracion (mas variedad = mejor generalizacion).
        pts_int = torch.tensor(muestrear_interior(batch), dtype=torch.float32)
        pts_bnd = torch.tensor(muestrear_borde(batch), dtype=torch.float32)

        # L_res: que se cumpla u_xx + u_yy + λu = q en el interior.
        q_obj = torch.tensor(forcing(pts_int.numpy()), dtype=torch.float32)
        r_pred = calcular_residuo(red, pts_int)
        L_res = ((r_pred - q_obj) ** 2).mean()

        # L_bc: que u sea 0 en el borde.
        u_pred_borde = red(pts_bnd)
        L_bc = (u_pred_borde ** 2).mean()

        # Si es M2, actualizamos lambda_bc cada 10 iters.
        # No lo hacemos cada iter porque es caro (necesita gradientes extra).
        if usa_balance and it % 10 == 0 and it > 0:
            lambda_sugerido = calcular_lambda_sugerido(red, L_res, lambda_bc * L_bc)
            # EMA: mezclamos el valor viejo con el nuevo para no oscilar.
            lambda_bc = 0.9 * lambda_bc + 0.1 * lambda_sugerido

        # Perdida total y paso de optimizacion.
        perdida = L_res + lambda_bc * L_bc
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        scheduler.step()

        # Guardamos historia cada 10 iters.
        if it % 10 == 0:
            historia["iter"].append(it)
            historia["L_res"].append(L_res.item())
            historia["L_bc"].append(L_bc.item())
            historia["lambda_bc"].append(lambda_bc)

        if it % 1000 == 0:
            print(f"  iter {it:5d} | L_res={L_res.item():.3e} "
                  f"L_bc={L_bc.item():.3e} | lambda_bc={lambda_bc:.2f}")

    # Fase 2 opcional: LBFGS para refinar.
    # LBFGS es un optimizador de segundo orden (usa info del Hessiano).
    # Converge muy rapido cuando ya estamos cerca del minimo.
    if lbfgs_pasos > 0:
        print(f"Refinando con LBFGS ({lbfgs_pasos} pasos)...")
        red = refinar_con_lbfgs(red, lambda_bc, lbfgs_pasos)

    return red, historia


def refinar_con_lbfgs(red, lambda_bc, pasos):
    """Segunda fase con LBFGS.

    LBFGS necesita una "closure" que recalcule la perdida varias veces por paso
    (porque hace busqueda de linea internamente). Usamos un batch FIJO durante
    toda esta fase para que sea estable.
    """
    optimizador = torch.optim.LBFGS(
        red.parameters(),
        lr=1.0, max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    # Un batch grande y fijo.
    pts_int = torch.tensor(muestrear_interior(256), dtype=torch.float32)
    pts_bnd = torch.tensor(muestrear_borde(256), dtype=torch.float32)
    q_obj = torch.tensor(forcing(pts_int.numpy()), dtype=torch.float32)

    for paso in range(pasos):
        def closure():
            optimizador.zero_grad()
            r = calcular_residuo(red, pts_int)
            L_res = ((r - q_obj) ** 2).mean()
            L_bc = (red(pts_bnd) ** 2).mean()
            perdida = L_res + lambda_bc * L_bc
            perdida.backward()
            return perdida
        optimizador.step(closure)

    return red
