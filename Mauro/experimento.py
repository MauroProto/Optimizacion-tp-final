"""
Script principal del proyecto.

Corre los experimentos y guarda resultados + plots en outputs/.

Uso (desde la carpeta Mauro/):

    python experimento.py m1            # entrena solo M1 (baseline)
    python experimento.py m2            # entrena solo M2 (con balance)
    python experimento.py todos         # entrena M1, M2, compara FD, hace plots
    python experimento.py fd            # solo solver clasico FD
    python experimento.py optimizadores # compara Adam vs Adam+LBFGS en M2
    python experimento.py grid          # grid search chico de hiperparametros (M2)

Opcionales:
    --iters 40001     numero de iteraciones (default 40001)
    --batch 128       puntos por iteracion (default 128)
    --lr 1e-3         learning rate inicial (default 1e-3)
    --capas 2 50 50 50 1   tamano de la red

Ejemplo: probar con menos iters para que sea rapido:
    python experimento.py todos --iters 5000
"""

import argparse
import csv
import itertools
import os

import matplotlib
matplotlib.use("Agg")  # asi no abre ventanas, guarda directo a archivo
import matplotlib.pyplot as plt
import numpy as np
import torch

from diferencias_finitas import resolver_fd
from entrenar import entrenar
from problema import grilla_test, u_exacta


# Carpeta donde guardamos plots y resultados.
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers de evaluacion y plots
# -----------------------------------------------------------------------------

def evaluar(red):
    """Mide el error de la red sobre una grilla 100x100.

    Devuelve un diccionario con los valores que despues vamos a plotear.
    """
    X1, X2, pts = grilla_test(100)
    pts_torch = torch.tensor(pts, dtype=torch.float32)

    with torch.no_grad():
        u_pred = red(pts_torch).numpy()

    u_real = u_exacta(pts)

    # Error relativo L2: ||real - predicho|| / ||real||
    error_l2 = np.linalg.norm(u_real - u_pred) / np.linalg.norm(u_real)

    return {
        "X1": X1, "X2": X2,
        "u_real": u_real.reshape(100, 100),
        "u_pred": u_pred.reshape(100, 100),
        "error_l2": error_l2,
    }


def plot_solucion(resultado, modo, ruta):
    """3 paneles: solucion real, prediccion, error absoluto."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    cosas = [
        (resultado["u_real"], f"Solucion exacta"),
        (resultado["u_pred"], f"Prediccion ({modo})"),
        (np.abs(resultado["u_real"] - resultado["u_pred"]), "Error absoluto"),
    ]

    for ax, (datos, titulo) in zip(axes, cosas):
        im = ax.pcolormesh(resultado["X1"], resultado["X2"], datos, cmap="jet", shading="auto")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(titulo)
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_historia(historia, modo, ruta):
    """Curvas de L_res y L_bc en escala log, mas la curva de lambda_bc."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(historia["iter"], historia["L_res"], label="L_res (residuo)")
    axes[0].plot(historia["iter"], historia["L_bc"], label="L_bc (borde)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteraciones"); axes[0].set_ylabel("perdida")
    axes[0].set_title(f"Perdidas durante entrenamiento ({modo})")
    axes[0].legend()

    axes[1].plot(historia["iter"], historia["lambda_bc"])
    axes[1].set_xlabel("iteraciones"); axes[1].set_ylabel("lambda_bc")
    axes[1].set_title(f"Lambda_bc adaptativo ({modo})")

    fig.tight_layout()
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Experimentos
# -----------------------------------------------------------------------------

def correr_pinn(modo, args):
    """Entrena una PINN en el modo dado, mide error, hace los plots."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    red, historia = entrenar(
        modo=modo,
        capas=args.capas,
        n_iter=args.iters,
        batch=args.batch,
        lr=args.lr,
    )

    resultado = evaluar(red)
    print(f"\n>> {modo}: error L2 relativo = {resultado['error_l2']:.3e}")

    plot_solucion(resultado, modo, os.path.join(OUT, f"{modo}_solucion.png"))
    plot_historia(historia, modo, os.path.join(OUT, f"{modo}_historia.png"))

    return resultado["error_l2"]


def correr_fd(args):
    """Solver clasico por diferencias finitas (extension propia)."""
    print(f"\nResolviendo con diferencias finitas (n={args.n_fd})...")
    res = resolver_fd(n=args.n_fd)
    print(f">> FD n={res['n']}: error L2 relativo = {res['error_rel_L2']:.3e}")

    # Plot.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cosas = [
        (res["u_exacta"], "Solucion exacta"),
        (res["u_fd"], f"FD (n={res['n']})"),
        (np.abs(res["u_exacta"] - res["u_fd"]), "Error absoluto"),
    ]
    for ax, (datos, titulo) in zip(axes, cosas):
        im = ax.pcolormesh(res["X1"], res["X2"], datos, cmap="jet", shading="auto")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(titulo)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"FD_solucion.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    return res["error_rel_L2"]


def correr_optimizadores(args):
    """Compara Adam puro vs Adam + LBFGS sobre M2.

    Usamos el mismo seed asi parten del mismo punto, y vemos si LBFGS al
    final realmente ayuda.
    """
    print("\n=== Adam puro vs Adam + LBFGS (modo M2) ===")

    # Adam puro
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    red_a, _ = entrenar("M2", args.capas, n_iter=args.iters, batch=args.batch, lr=args.lr)
    err_a = evaluar(red_a)["error_l2"]
    print(f">> Adam solo: error L2 = {err_a:.3e}")

    # Adam + LBFGS
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    red_l, _ = entrenar("M2", args.capas, n_iter=args.iters, batch=args.batch,
                        lr=args.lr, lbfgs_pasos=args.lbfgs_pasos)
    err_l = evaluar(red_l)["error_l2"]
    print(f">> Adam + LBFGS ({args.lbfgs_pasos} pasos): error L2 = {err_l:.3e}")
    print(f">> Mejora: {err_a / err_l:.2f}x")

    # Plot de barras comparativo.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Adam", f"Adam + LBFGS\n({args.lbfgs_pasos} pasos)"], [err_a, err_l])
    ax.set_yscale("log"); ax.set_ylabel("error L2 relativo")
    ax.set_title("Comparacion de optimizadores (M2)")
    for x, v in zip(range(2), [err_a, err_l]):
        ax.text(x, v, f"{v:.2e}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "comparacion_optimizadores.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def correr_grid(args):
    """Grid search de hiperparametros sobre M2.

    El paper en la Tabla 2 prueba estas arquitecturas para Helmholtz:
        ancho        : 30, 50, 100 neuronas por capa
        profundidad  : 3, 5, 7 hidden layers

    Aca usamos ese mismo espacio + variamos el learning rate.
    Total = 2 lr x 3 anchos x 2 profundidades = 12 celdas.

    Para cada celda entrenamos M2 con `--grid-iters` iters (default 3000, rapido).
    """
    # Espacio de busqueda: alineado con la Tabla 2 del paper.
    lrs = [1e-3, 5e-4]
    anchos = [30, 50, 100]
    profundidades = [3, 5]

    total = len(lrs) * len(anchos) * len(profundidades)
    print("\n=== Grid search sobre M2 (anchos y profundidades de la Tabla 2 del paper) ===")
    print(f"lr        : {lrs}")
    print(f"ancho     : {anchos}      (paper Tabla 2: 30, 50, 100)")
    print(f"profund.  : {profundidades}         (paper Tabla 2: 3, 5, 7)")
    print(f"iters/celda: {args.grid_iters}")
    print(f"Total: {total} celdas\n")

    resultados = []

    # itertools.product nos da todas las combinaciones (lr, ancho, prof).
    for idx, (lr, ancho, prof) in enumerate(itertools.product(lrs, anchos, profundidades), 1):
        # Armamos las capas: entrada (2) -> [ancho] * profundidad -> salida (1)
        capas = [2] + [ancho] * prof + [1]
        print(f"[{idx}/{total}] lr={lr}, ancho={ancho}, prof={prof}, capas={capas}")

        # Reseteamos seeds para que todas partan igual.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        red, _ = entrenar("M2", capas, n_iter=args.grid_iters, batch=args.batch, lr=lr)
        error = evaluar(red)["error_l2"]
        print(f"     -> error L2 = {error:.3e}\n")
        resultados.append({"lr": lr, "ancho": ancho, "prof": prof,
                           "capas": str(capas), "error_l2": error})

    # Ordenamos del mejor al peor.
    resultados.sort(key=lambda r: r["error_l2"])

    # Imprimimos la tabla.
    print("=" * 60)
    print("RESULTADO DEL GRID (de mejor a peor)")
    print("=" * 60)
    print(f"{'lr':>8} | {'ancho':>5} | {'prof':>4} | {'error L2':>12}")
    print("-" * 45)
    for r in resultados:
        print(f"{r['lr']:>8.0e} | {r['ancho']:>5d} | {r['prof']:>4d} | {r['error_l2']:>12.3e}")

    # Guardamos en un CSV.
    csv_path = os.path.join(OUT, "grid.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lr", "ancho", "prof", "capas", "error_l2"])
        writer.writeheader()
        for r in resultados:
            writer.writerow(r)

    # Plot de barras simple.
    fig, ax = plt.subplots(figsize=(10, 4))
    etiquetas = [f"lr={r['lr']:.0e}\nw={r['ancho']}, d={r['prof']}" for r in resultados]
    errores = [r["error_l2"] for r in resultados]
    ax.bar(range(len(resultados)), errores)
    ax.set_xticks(range(len(resultados)))
    ax.set_xticklabels(etiquetas, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("error L2 relativo")
    ax.set_title(f"Grid search (M2, {args.grid_iters} iters por celda)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grid.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nMejor combinacion: lr={resultados[0]['lr']}, "
          f"ancho={resultados[0]['ancho']}, prof={resultados[0]['prof']}")
    print(f"CSV guardado en: {csv_path}")


def correr_todos(args):
    """Corre el experimento completo y arma una tabla resumen."""
    print("=" * 60)
    print("CORRIDA COMPLETA: M1 (baseline) + M2 (con balance) + FD")
    print("=" * 60)

    err_m1 = correr_pinn("M1", args)
    err_m2 = correr_pinn("M2", args)
    err_fd = correr_fd(args)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  M1 (PINN sin balance):    error L2 = {err_m1:.3e}")
    print(f"  M2 (PINN con balance):    error L2 = {err_m2:.3e}")
    print(f"  FD (diferencias finitas): error L2 = {err_fd:.3e}")
    print(f"\n  Mejora M1 -> M2: {err_m1 / err_m2:.1f}x")

    # Guardamos la tabla en un txt para tenerla a mano.
    with open(os.path.join(OUT, "resumen.txt"), "w") as f:
        f.write(f"Iteraciones: {args.iters}\n")
        f.write(f"Capas: {args.capas}\n")
        f.write(f"Batch: {args.batch}, lr: {args.lr}, seed: {args.seed}\n\n")
        f.write(f"M1 (PINN sin balance):    error L2 = {err_m1:.3e}\n")
        f.write(f"M2 (PINN con balance):    error L2 = {err_m2:.3e}\n")
        f.write(f"FD (diferencias finitas): error L2 = {err_fd:.3e}\n")
        f.write(f"\nMejora M1 -> M2: {err_m1 / err_m2:.1f}x\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modo",
                        choices=["m1", "m2", "todos", "fd", "optimizadores", "grid"],
                        help="que experimento correr")
    parser.add_argument("--iters", type=int, default=40001, help="iteraciones de Adam")
    parser.add_argument("--batch", type=int, default=128, help="puntos por iter")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate inicial")
    parser.add_argument("--capas", type=int, nargs="+", default=[2, 50, 50, 50, 1],
                        help="tamano de las capas, ej: 2 50 50 50 1")
    parser.add_argument("--n-fd", type=int, default=200, help="resolucion del FD")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lbfgs-pasos", type=int, default=200,
                        help="cantidad de pasos LBFGS (solo en modo 'optimizadores')")
    parser.add_argument("--grid-iters", type=int, default=3000,
                        help="iteraciones por celda del grid (solo en modo 'grid')")
    args = parser.parse_args()

    if args.modo == "m1":
        correr_pinn("M1", args)
    elif args.modo == "m2":
        correr_pinn("M2", args)
    elif args.modo == "todos":
        correr_todos(args)
    elif args.modo == "fd":
        correr_fd(args)
    elif args.modo == "optimizadores":
        correr_optimizadores(args)
    elif args.modo == "grid":
        correr_grid(args)

    print(f"\nListo. Resultados en: {OUT}/")


if __name__ == "__main__":
    main()
