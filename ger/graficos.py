"""Graficos del experimento."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_solution(result: dict, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    panels = [
        (result["u_exact"], "Solucion analitica"),
        (result["u_pred"], title),
        (result["abs_error"], "Error absoluto"),
    ]
    for ax, (data, label) in zip(axes, panels):
        im = ax.pcolormesh(result["x"], result["tau"], data, shading="auto", cmap="viridis")
        ax.set_xlabel("x / L")
        ax.set_ylabel("tau = alpha t / L^2")
        ax.set_title(label)
        fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(history: dict[str, list[float]], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(history["iter"], history["loss_res"], label="L_res")
    axes[0].plot(history["iter"], history["loss_ic"], label="L_ic")
    axes[0].plot(history["iter"], history["loss_bc"], label="L_bc")
    axes[0].plot(history["iter"], history["loss_total"], label="L_total", linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteracion")
    axes[0].set_ylabel("perdida")
    axes[0].set_title(f"Perdidas - {title}")
    axes[0].legend()

    axes[1].plot(history["iter"], history["lambda_ic"], label="lambda_ic")
    axes[1].plot(history["iter"], history["lambda_bc"], label="lambda_bc")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("iteracion")
    axes[1].set_ylabel("peso adaptativo")
    axes[1].set_title(f"Pesos de perdida - {title}")
    axes[1].legend()

    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_error_bars(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(metrics["modelo"], metrics["error_l2"], color=["#4c78a8", "#f58518", "#54a24b"][: len(metrics)])
    ax.set_yscale("log")
    ax.set_ylabel("error relativo L2")
    ax.set_title("Comparacion contra solucion analitica")
    for i, row in metrics.iterrows():
        ax.text(i, row["error_l2"], f"{row['error_l2']:.2e}", ha="center", va="bottom")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_seed_summary(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    summary = df.groupby("modelo")["error_l2"].agg(["mean", "std"]).reset_index()
    ax.bar(summary["modelo"], summary["mean"], yerr=summary["std"].fillna(0.0), capsize=5)
    ax.scatter(df["modelo"], df["error_l2"], color="black", zorder=3, label="semillas")
    ax.set_yscale("log")
    ax.set_ylabel("error relativo L2")
    ax.set_title("Robustez por semillas")
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_paper_solution_comparison(results: dict[str, dict], path: Path) -> None:
    m1 = results["M1"]
    m2 = results["M2"]
    panels = [
        (m1["u_exact"], "Solucion analitica"),
        (m1["u_pred"], "PINN M1"),
        (m2["u_pred"], "PINN M2"),
        (m1["abs_error"], "Error absoluto M1"),
        (m2["abs_error"], "Error absoluto M2"),
        (m1["abs_error"] - m2["abs_error"], "Reduccion de error M1-M2"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    for ax, (data, title) in zip(axes.ravel(), panels):
        cmap = "coolwarm" if "Reduccion" in title else "viridis"
        im = ax.pcolormesh(m1["x"], m1["tau"], data, shading="auto", cmap=cmap)
        ax.set_xlabel("x / L")
        ax.set_ylabel("tau")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_lambda_evolution(history: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(history["iter"], history["lambda_ic"], label="lambda_ic")
    ax.plot(history["iter"], history["lambda_bc"], label="lambda_bc")
    ax.set_yscale("log")
    ax.set_xlabel("iteracion")
    ax.set_ylabel("peso adaptativo")
    ax.set_title("Evolucion de pesos adaptativos en M2")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gradient_norms(snapshots: dict[str, dict[str, list[np.ndarray]]], path: Path) -> None:
    rows = []
    for model, snapshot in snapshots.items():
        for term, layers in snapshot.items():
            for layer_idx, values in enumerate(layers, start=1):
                abs_values = np.abs(values[np.isfinite(values)])
                rows.append(
                    {
                        "modelo": model,
                        "termino": term,
                        "capa": layer_idx,
                        "mean_abs_grad": float(abs_values.mean()) if abs_values.size else np.nan,
                        "max_abs_grad": float(abs_values.max()) if abs_values.size else np.nan,
                    }
                )
    df = pd.DataFrame(rows)
    terms = [("res", "L_res"), ("ic", "L_ic"), ("bc", "L_bc")]
    models = list(snapshots.keys())
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(6 * len(models), 4),
        sharey=True,
        constrained_layout=True,
    )
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = df[df["modelo"] == model]
        for term, label in terms:
            tsub = sub[sub["termino"] == term]
            ax.plot(tsub["capa"], tsub["mean_abs_grad"], marker="o", label=label)
        ax.set_yscale("log")
        ax.set_xlabel("capa")
        ax.set_title(f"Gradientes medios por capa - {model}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean(|grad|)")
    axes[-1].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_time_slices(results: dict[str, dict], tau_values: list[float], path: Path) -> None:
    first = next(iter(results.values()))
    x_grid = first["x"][0, :]
    tau_grid = first["tau"][:, 0]
    fig, axes = plt.subplots(1, len(tau_values), figsize=(4 * len(tau_values), 3.5), constrained_layout=True)
    if len(tau_values) == 1:
        axes = [axes]

    for ax, tau in zip(axes, tau_values):
        idx = int(np.argmin(np.abs(tau_grid - tau)))
        ax.plot(x_grid, first["u_exact"][idx], color="black", linewidth=2, label="analitica")
        for name, res in results.items():
            ax.plot(x_grid, res["u_pred"][idx], linestyle="--", label=name)
        ax.set_title(f"tau={tau_grid[idx]:.3f}")
        ax.set_xlabel("x / L")
        ax.set_ylabel("u")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gradient_histograms(snapshot: dict[str, list[np.ndarray]], title: str, path: Path) -> None:
    labels = [("res", "L_res", "#4c78a8"), ("ic", "L_ic", "#f58518"), ("bc", "L_bc", "#54a24b")]
    n_layers = len(snapshot["res"])
    fig, axes = plt.subplots(n_layers, 1, figsize=(8, max(2.2 * n_layers, 3)), constrained_layout=True)
    if n_layers == 1:
        axes = [axes]

    for layer_idx, ax in enumerate(axes):
        all_vals = np.concatenate([snapshot[key][layer_idx] for key, _, _ in labels])
        finite = all_vals[np.isfinite(all_vals)]
        scale = np.percentile(np.abs(finite), 99.0) if finite.size else 1.0
        if scale <= 0:
            scale = 1.0
        bins = np.linspace(-scale, scale, 80)
        for key, label, color in labels:
            vals = snapshot[key][layer_idx]
            ax.hist(vals, bins=bins, alpha=0.5, density=True, label=label, color=color)
        ax.set_yscale("log")
        ax.set_title(f"{title} - capa {layer_idx + 1}")
        ax.set_xlabel("gradiente")
        ax.set_ylabel("densidad")
        ax.legend(loc="upper right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_optimizer_comparison(df: pd.DataFrame, path: Path) -> None:
    plot_df = df[np.isfinite(df["error_l2"])].copy()
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(plot_df["optimizador"], plot_df["error_l2"], color="#4c78a8")
    ax.set_yscale("log")
    ax.set_ylabel("error relativo L2")
    ax.set_title("Comparacion de optimizadores en M2")
    ax.tick_params(axis="x", rotation=15)
    for i, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(i, row["error_l2"], f"{row['error_l2']:.2e}", ha="center", va="bottom")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bayes_trials(df: pd.DataFrame, out_dir: Path, stem: str = "bayes", title: str = "Busqueda bayesiana") -> None:
    ensure_dir(out_dir)
    ordered = df.sort_values("number")
    best_so_far = ordered["value"].cummin()

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.scatter(ordered["number"], ordered["value"], label="trial")
    ax.plot(ordered["number"], best_so_far, color="black", label="mejor acumulado")
    ax.set_yscale("log")
    ax.set_xlabel("trial")
    ax.set_ylabel("error relativo L2")
    ax.set_title(f"{title} - historia")
    ax.legend()
    fig.savefig(out_dir / f"{stem}_historia.png", dpi=160)
    plt.close(fig)

    param_cols = [c for c in df.columns if c.startswith("params_")]
    fig, axes = plt.subplots(1, len(param_cols), figsize=(4 * len(param_cols), 3.5), constrained_layout=True)
    if len(param_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, param_cols):
        x = df[col]
        if col.endswith("lr"):
            ax.set_xscale("log")
        ax.scatter(x, df["value"], alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(col.replace("params_", ""))
        ax.set_ylabel("error L2")
        ax.grid(alpha=0.25)
    fig.savefig(out_dir / f"{stem}_parametros.png", dpi=160)
    plt.close(fig)

    if {"params_width", "params_depth"}.issubset(df.columns):
        pivot = df.pivot_table(
            index="params_depth",
            columns="params_width",
            values="value",
            aggfunc="min",
        )
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        im = ax.imshow(pivot.values, aspect="auto", cmap="magma_r")
        ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), labels=[str(c) for c in pivot.index])
        ax.set_xlabel("ancho")
        ax.set_ylabel("profundidad")
        ax.set_title("Mejor error por arquitectura")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1e}", ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, label="error L2")
        fig.savefig(out_dir / f"{stem}_heatmap_arquitectura.png", dpi=160)
        plt.close(fig)
