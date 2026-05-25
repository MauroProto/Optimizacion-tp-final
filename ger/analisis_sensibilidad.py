"""Postproceso de sensibilidad usando resultados actuales.

El objetivo es separar dos preguntas:

1. Que hiperparametros minimizan el error corto de Optuna.
2. Que regimen se sostiene cuando se valida con mas iteraciones.

El script lee `bayes_trials.csv`, `validacion_regimenes.csv` y
`metricas_modelos.csv`; no contiene resultados hardcodeados.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("outputs")


def main() -> None:
    trials = pd.read_csv(OUT / "bayes_trials.csv")
    validation = pd.read_csv(OUT / "validacion_regimenes.csv")
    final_metrics = pd.read_csv(OUT / "metricas_modelos.csv")

    final_m2 = final_metrics[final_metrics["modelo"] == "M2"].iloc[0]
    final_row = {
        "regimen": "final_M2",
        "trial": validation.iloc[0]["trial"],
        "short_error_l2": validation.iloc[0]["short_error_l2"],
        "width": validation.iloc[0]["width"],
        "depth": validation.iloc[0]["depth"],
        "lr": validation.iloc[0]["lr"],
        "batch": validation.iloc[0]["batch"],
        "adaptive_beta": validation.iloc[0]["adaptive_beta"],
        "seed": 0,
        "iters": 3000,
        "long_error_l2": final_m2["error_l2"],
        "lambda_ic_final": final_m2["lambda_ic_final"],
        "lambda_bc_final": final_m2["lambda_bc_final"],
        "tiempo_s": final_m2["tiempo_s"],
    }
    summary = pd.concat([validation, pd.DataFrame([final_row])], ignore_index=True)
    summary.to_csv(OUT / "sensibilidad_regimenes.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    ordered_trials = trials.sort_values("number")
    axes[0].scatter(ordered_trials["number"], ordered_trials["value"], s=22, alpha=0.75)
    axes[0].plot(ordered_trials["number"], ordered_trials["value"].cummin(), color="black")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("trial Optuna")
    axes[0].set_ylabel("error corto")
    axes[0].set_title("100 trials bayesianos")
    axes[0].grid(alpha=0.25)

    valid = validation.sort_values("long_error_l2")
    axes[1].bar(valid["regimen"], valid["long_error_l2"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("error validado")
    axes[1].set_title("Top trials validados a 3000 iters")
    axes[1].tick_params(axis="x", rotation=35)

    optuna_valid = validation[validation["trial"] >= 0]
    axes[2].scatter(
        optuna_valid["short_error_l2"],
        optuna_valid["long_error_l2"],
        s=70,
        label="top trials",
    )
    axes[2].scatter(
        [validation.iloc[0]["short_error_l2"]],
        [final_m2["error_l2"]],
        marker="*",
        s=160,
        label="regimen final",
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("error corto Optuna")
    axes[2].set_ylabel("error largo")
    axes[2].set_title("Corto no implica largo")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.savefig(OUT / "sensibilidad_regimenes.png", dpi=160)
    plt.close(fig)

    print(summary.sort_values("long_error_l2"))


if __name__ == "__main__":
    main()
