"""Graficos inspirados en el notebook de examen.

El notebook de referencia compara optimizadores con:
  1. valor objetivo por iteracion,
  2. norma de gradiente por iteracion,
  3. trayectorias sobre contornos en un espacio 2D.

Para una PINN el espacio de pesos es demasiado grande para dibujar contornos
utiles. El analogo informativo es el espacio de hiperparametros explorado por
Optuna, donde cada trial es un punto de la trayectoria de busqueda.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


OUT = Path("outputs")


def _complete_trials(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "state" in df.columns:
        df = df[df["state"] == "COMPLETE"].copy()
    df = df[np.isfinite(df["value"])].copy()
    return df.sort_values("number")


def plot_bayes_convergence(df: pd.DataFrame, path: Path) -> None:
    """Objetivo de Optuna y mejor acumulado, estilo f(x_k)."""

    ordered = df.sort_values("number")
    trials = ordered["number"].to_numpy()
    values = ordered["value"].to_numpy()
    best_so_far = np.minimum.accumulate(values)
    global_best = float(np.min(values))
    gap = np.maximum(best_so_far - global_best, 1.0e-12)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    axes[0].plot(trials, values, marker=".", markersize=4, linewidth=0.9, label="trial")
    axes[0].plot(trials, best_so_far, color="black", linewidth=1.8, label="mejor acumulado")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("trial de Optuna")
    axes[0].set_ylabel("error relativo L2 corto")
    axes[0].set_title("Valor objetivo por trial")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].semilogy(trials, gap, marker=".", markersize=4, linewidth=0.9, color="#f58518")
    axes[1].set_xlabel("trial de Optuna")
    axes[1].set_ylabel("brecha al mejor observado")
    axes[1].set_title("Convergencia del mejor acumulado")
    axes[1].grid(alpha=0.25)

    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_bayes_trajectory(df: pd.DataFrame, path: Path) -> None:
    """Trayectoria de busqueda sobre contornos interpolados de error."""

    needed = {"params_lr", "params_adaptive_beta", "value", "number"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Faltan columnas para graficar trayectoria: {needed - set(df.columns)}")

    plot_df = df.dropna(subset=list(needed)).sort_values("number").copy()
    x = np.log10(plot_df["params_lr"].to_numpy(dtype=float))
    y = plot_df["params_adaptive_beta"].to_numpy(dtype=float)
    z = plot_df["value"].to_numpy(dtype=float)
    log_z = np.log10(np.maximum(z, 1.0e-12))

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    if len(plot_df) >= 6 and len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3:
        gx, gy = np.meshgrid(
            np.linspace(float(x.min()), float(x.max()), 160),
            np.linspace(float(y.min()), float(y.max()), 160),
        )
        gz = griddata(np.column_stack([x, y]), log_z, (gx, gy), method="linear")
        if np.isfinite(gz).any():
            levels = np.linspace(float(np.nanmin(log_z)), float(np.nanmax(log_z)), 16)
            contour = ax.contourf(gx, gy, gz, levels=levels, cmap="viridis_r", alpha=0.85)
            fig.colorbar(contour, ax=ax, label="log10(error L2 corto)")

    scatter = ax.scatter(
        x,
        y,
        c=log_z,
        cmap="viridis_r",
        s=44,
        edgecolor="black",
        linewidth=0.35,
        zorder=3,
    )
    ax.plot(x, y, color="white", linewidth=1.0, alpha=0.55, zorder=2)
    ax.plot(x, y, color="black", linewidth=0.7, alpha=0.35, zorder=2)

    best_idx = int(np.argmin(z))
    ax.scatter(
        x[best_idx],
        y[best_idx],
        marker="*",
        s=260,
        color="#d62728",
        edgecolor="white",
        linewidth=1.0,
        label=f"mejor trial {int(plot_df.iloc[best_idx]['number'])}",
        zorder=4,
    )
    ax.scatter(x[0], y[0], marker="s", s=70, color="white", edgecolor="black", label="inicio", zorder=4)
    ax.scatter(x[-1], y[-1], marker="X", s=80, color="black", edgecolor="white", label="fin", zorder=4)

    if not ax.collections or len(ax.figure.axes) == 1:
        fig.colorbar(scatter, ax=ax, label="log10(error L2 corto)")
    ax.set_xlabel("log10(learning rate)")
    ax.set_ylabel("beta adaptativo")
    ax.set_title("Trayectoria de Optuna sobre espacio 2D")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_bayes_m1_trajectory(df: pd.DataFrame, path: Path) -> None:
    """Trayectoria de M1 en el plano learning-rate/ancho."""

    needed = {"params_lr", "params_width", "value", "number"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Faltan columnas para graficar trayectoria M1: {needed - set(df.columns)}")

    plot_df = df.dropna(subset=list(needed)).sort_values("number").copy()
    x = np.log10(plot_df["params_lr"].to_numpy(dtype=float))
    y = plot_df["params_width"].to_numpy(dtype=float)
    z = plot_df["value"].to_numpy(dtype=float)
    log_z = np.log10(np.maximum(z, 1.0e-12))

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    scatter = ax.scatter(
        x,
        y,
        c=log_z,
        cmap="viridis_r",
        s=48,
        edgecolor="black",
        linewidth=0.35,
        zorder=3,
    )
    ax.plot(x, y, color="black", linewidth=0.7, alpha=0.25, zorder=2)

    best_idx = int(np.argmin(z))
    ax.scatter(
        x[best_idx],
        y[best_idx],
        marker="*",
        s=260,
        color="#d62728",
        edgecolor="white",
        linewidth=1.0,
        label=f"mejor trial {int(plot_df.iloc[best_idx]['number'])}",
        zorder=4,
    )
    ax.scatter(x[0], y[0], marker="s", s=70, color="white", edgecolor="black", label="inicio", zorder=4)
    ax.scatter(x[-1], y[-1], marker="X", s=80, color="black", edgecolor="white", label="fin", zorder=4)

    fig.colorbar(scatter, ax=ax, label="log10(error L2 corto)")
    ax.set_xlabel("log10(learning rate)")
    ax.set_ylabel("ancho")
    ax.set_title("Trayectoria M1 de Optuna")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_optimizer_lr_error(df: pd.DataFrame, path: Path) -> None:
    """Comparacion por familia: error final contra learning rate."""

    plot_df = df[np.isfinite(df["error_l2"])].copy()
    families = list(plot_df["familia"].drop_duplicates())
    markers = {"adam": "o", "adamw": "s", "sgd": "^", "adam+lbfgs": "*"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    for family in families:
        sub = plot_df[plot_df["familia"] == family]
        axes[0].scatter(
            sub["lr"],
            sub["error_l2"],
            s=85 if family != "adam+lbfgs" else 150,
            marker=markers.get(family, "o"),
            label=family,
            edgecolor="black",
            linewidth=0.4,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("learning rate")
    axes[0].set_ylabel("error relativo L2")
    axes[0].set_title("Sensibilidad al learning rate")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    ordered = plot_df.sort_values("error_l2").reset_index(drop=True)
    colors = ["#4c78a8" if fam in {"adam", "adamw"} else "#f58518" for fam in ordered["familia"]]
    axes[1].bar(ordered["optimizador"], ordered["error_l2"], color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("error relativo L2")
    axes[1].set_title("Ranking con presupuesto fijo")
    axes[1].tick_params(axis="x", rotation=25)
    for i, row in ordered.iterrows():
        axes[1].text(i, row["error_l2"], f"{row['error_l2']:.2e}", ha="center", va="bottom", fontsize=8)

    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    bayes_path = OUT / "bayes_trials.csv"
    if bayes_path.exists():
        trials = _complete_trials(bayes_path)
        plot_bayes_convergence(trials, OUT / "examen_bayes_convergencia.png")
        plot_bayes_trajectory(trials, OUT / "examen_bayes_trayectoria_lr_beta.png")

    bayes_m1_path = OUT / "bayes_m1_trials.csv"
    if bayes_m1_path.exists():
        trials_m1 = _complete_trials(bayes_m1_path)
        plot_bayes_convergence(trials_m1, OUT / "examen_bayes_m1_convergencia.png")
        plot_bayes_m1_trajectory(trials_m1, OUT / "examen_bayes_m1_trayectoria_lr_width.png")

    optimizers_path = OUT / "optimizadores.csv"
    if optimizers_path.exists():
        optimizers = pd.read_csv(optimizers_path)
        plot_optimizer_lr_error(optimizers, OUT / "examen_optimizadores_lr_error.png")


if __name__ == "__main__":
    main()
