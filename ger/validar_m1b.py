"""Valida a mas iteraciones los mejores regimenes de la busqueda M1.

La busqueda corta de Optuna usa 800 iteraciones por trial. Este script reentrena
los mejores candidatos como M1 clasica, sin pesos adaptativos, para elegir la
M1B final con el mismo criterio de validacion larga usado para M2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from entrenamiento import TrainConfig, evaluate_model, train_pinn


OUT = Path("outputs")


def layers(depth: int, width: int) -> tuple[int, ...]:
    return tuple([2] + [width] * depth + [1])


def make_candidates(trials: pd.DataFrame, top_k: int) -> list[dict]:
    complete = trials[np.isfinite(trials["value"])].copy()
    if "state" in complete.columns:
        complete = complete[complete["state"] == "COMPLETE"].copy()
    complete = complete.sort_values("value").head(top_k)
    candidates = []
    for _, row in complete.iterrows():
        candidates.append(
            {
                "regimen": f"m1_optuna_trial_{int(row['number'])}",
                "trial": int(row["number"]),
                "short_error_l2": float(row["value"]),
                "width": int(row["params_width"]),
                "depth": int(row["params_depth"]),
                "lr": float(row["params_lr"]),
                "batch": int(row["params_batch"]),
            }
        )
    return candidates


def plot_validation(df: pd.DataFrame, path: Path) -> None:
    ordered = df.sort_values("long_error_l2")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].bar(ordered["regimen"], ordered["long_error_l2"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("error relativo L2 a validacion larga")
    axes[0].set_title("Validacion larga M1B")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].scatter(ordered["short_error_l2"], ordered["long_error_l2"], label="trials M1")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("error corto usado por Optuna")
    axes[1].set_ylabel("error largo validado")
    axes[1].set_title("M1: error corto vs validacion")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nx", type=int, default=120)
    parser.add_argument("--nt", type=int, default=120)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    trials = pd.read_csv(OUT / "bayes_m1_trials.csv")
    candidates = make_candidates(trials, args.top_k)
    rows = []
    for cand in candidates:
        cfg = TrainConfig(
            mode="M1",
            layers=layers(cand["depth"], cand["width"]),
            iterations=args.iters,
            batch_residual=cand["batch"],
            batch_initial=cand["batch"],
            batch_boundary=cand["batch"],
            lr=cand["lr"],
            optimizer="adam",
            seed=args.seed,
            adaptive_every=10,
            adaptive_beta=1.0,
            log_every=max(args.iters // 3, 1),
            threads=args.threads,
        )
        result = train_pinn(cfg, verbose=True)
        eval_res = evaluate_model(result.model, nx=args.nx, nt=args.nt)
        rows.append(
            {
                **cand,
                "seed": args.seed,
                "iters": args.iters,
                "long_error_l2": eval_res["rel_l2"],
                "lambda_ic_final": result.lambdas["ic"],
                "lambda_bc_final": result.lambdas["bc"],
                "tiempo_s": result.elapsed_s,
            }
        )

    df = pd.DataFrame(rows).sort_values("long_error_l2")
    df.to_csv(OUT / "validacion_m1b.csv", index=False)
    plot_validation(df, OUT / "validacion_m1b.png")
    best = df.iloc[0].to_dict()
    (OUT / "validacion_m1b_mejor.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(df)
    print("Mejor regimen M1B validado:", best)


if __name__ == "__main__":
    main()
