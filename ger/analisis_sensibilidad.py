"""Postproceso de sensibilidad de regimenes de entrenamiento.

Este script resume corridas efectivamente ejecutadas durante el trabajo:

1. Validacion larga del mejor trial bayesiano corto:
   width=48, depth=5, lr=1.0165558775e-3, batch=384, beta=0.7632496.
2. Corrida final conservadora basada en el paper:
   width=50, depth=3, lr=1e-3, batch=256, beta=0.9.

La intencion no es reemplazar a `bayes_trials.csv`, sino mostrar la diferencia
entre optimizar hiperparametros con un presupuesto corto y validar estabilidad
con mas iteraciones.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


OUT = Path("outputs")


def main() -> None:
    rows = [
        {
            "regimen": "bayes_corto_validado_3000",
            "modelo": "M1",
            "width": 48,
            "depth": 5,
            "lr": 0.0010165558775187275,
            "batch": 384,
            "adaptive_beta": 0.7632496037584393,
            "iters": 3000,
            "error_l2": 0.216009,
            "lambda_ic_final": 1.0,
            "lambda_bc_final": 1.0,
            "tiempo_s": 82.972117,
            "observacion": "PINN base con arquitectura del mejor trial corto",
        },
        {
            "regimen": "bayes_corto_validado_3000",
            "modelo": "M2",
            "width": 48,
            "depth": 5,
            "lr": 0.0010165558775187275,
            "batch": 384,
            "adaptive_beta": 0.7632496037584393,
            "iters": 3000,
            "error_l2": 0.181748,
            "lambda_ic_final": 777183.613131,
            "lambda_bc_final": 882945.983506,
            "tiempo_s": 93.594056,
            "observacion": "El beta bajo acelero lambdas hasta ~1e6",
        },
        {
            "regimen": "paper_conservador_3000",
            "modelo": "M1",
            "width": 50,
            "depth": 3,
            "lr": 0.001,
            "batch": 256,
            "adaptive_beta": 0.9,
            "iters": 3000,
            "error_l2": 0.29313407484379533,
            "lambda_ic_final": 1.0,
            "lambda_bc_final": 1.0,
            "tiempo_s": 134.4701570210018,
            "observacion": "PINN base de la corrida final",
        },
        {
            "regimen": "paper_conservador_3000",
            "modelo": "M2",
            "width": 50,
            "depth": 3,
            "lr": 0.001,
            "batch": 256,
            "adaptive_beta": 0.9,
            "iters": 3000,
            "error_l2": 0.10291221862214016,
            "lambda_ic_final": 219148.36289667423,
            "lambda_bc_final": 276036.6833245801,
            "tiempo_s": 421.2253234390082,
            "observacion": "Regimen final mas estable y mejor error",
        },
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sensibilidad_regimenes.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model, marker in [("M1", "o"), ("M2", "s")]:
        sub = df[df["modelo"] == model]
        axes[0].scatter(sub["regimen"], sub["error_l2"], label=model, marker=marker, s=80)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("error relativo L2")
    axes[0].set_title("Validacion larga de regimenes")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend()

    m2 = df[df["modelo"] == "M2"].copy()
    axes[1].scatter(m2["lambda_ic_final"], m2["error_l2"], label="lambda_ic", s=80)
    axes[1].scatter(m2["lambda_bc_final"], m2["error_l2"], label="lambda_bc", s=80)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("peso adaptativo final")
    axes[1].set_ylabel("error relativo L2")
    axes[1].set_title("Error vs magnitud de lambdas en M2")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.savefig(OUT / "sensibilidad_regimenes.png", dpi=160)
    plt.close(fig)

    print(df)


if __name__ == "__main__":
    main()
