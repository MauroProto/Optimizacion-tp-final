"""CLI para reproducir el estudio en difusion de calor 1D."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from entrenamiento import (
    TrainConfig,
    evaluate_model,
    gradient_snapshot,
    refine_with_lbfgs,
    train_pinn,
)
from graficos import (
    ensure_dir,
    plot_bayes_trials,
    plot_error_bars,
    plot_gradient_histograms,
    plot_history,
    plot_optimizer_comparison,
    plot_solution,
    plot_time_slices,
)
from problema import DEFAULT_PROBLEM


OUT = Path("outputs")


def layers_from_args(depth: int, width: int) -> tuple[int, ...]:
    return tuple([2] + [width] * depth + [1])


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_model(mode: str, args: argparse.Namespace, layers: tuple[int, ...] | None = None) -> dict:
    cfg = TrainConfig(
        mode=mode,
        layers=layers or layers_from_args(args.depth, args.width),
        iterations=args.iters,
        batch_residual=args.batch,
        batch_initial=args.batch,
        batch_boundary=args.batch,
        lr=args.lr,
        optimizer=args.optimizer,
        seed=args.seed,
        adaptive_every=args.adaptive_every,
        adaptive_beta=args.adaptive_beta,
        log_every=args.log_every,
        threads=args.threads,
    )
    result = train_pinn(cfg, problem=DEFAULT_PROBLEM, verbose=True)
    eval_res = evaluate_model(result.model, problem=DEFAULT_PROBLEM, nx=args.nx, nt=args.nt, device=cfg.device)
    snap = gradient_snapshot(result.model, cfg, problem=DEFAULT_PROBLEM, batch=max(args.batch, 512))
    return {"train": result, "eval": eval_res, "snapshot": snap}


def command_todos(args: argparse.Namespace) -> None:
    ensure_dir(OUT)
    results = {}
    rows = []
    for mode in ["M1", "M2"]:
        data = run_model(mode, args)
        results[mode] = data
        rows.append(
            {
                "modelo": mode,
                "error_l2": data["eval"]["rel_l2"],
                "tiempo_s": data["train"].elapsed_s,
                "lambda_ic_final": data["train"].lambdas["ic"],
                "lambda_bc_final": data["train"].lambdas["bc"],
            }
        )
        plot_solution(data["eval"], f"Prediccion {mode}", OUT / f"{mode}_solucion.png")
        plot_history(data["train"].history, mode, OUT / f"{mode}_historia.png")
        plot_gradient_histograms(data["snapshot"], mode, OUT / f"{mode}_gradientes.png")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metricas_modelos.csv", index=False)
    plot_error_bars(metrics, OUT / "comparacion_modelos.png")
    plot_time_slices(
        {name: data["eval"] for name, data in results.items()},
        tau_values=[0.0, DEFAULT_PROBLEM.tau_final / 8, DEFAULT_PROBLEM.tau_final / 2, DEFAULT_PROBLEM.tau_final],
        path=OUT / "cortes_temporales.png",
    )

    summary = {
        "problema": {
            "length_m": DEFAULT_PROBLEM.length_m,
            "diffusivity_m2_s": DEFAULT_PROBLEM.diffusivity_m2_s,
            "tau_final": DEFAULT_PROBLEM.tau_final,
            "t_final_s": DEFAULT_PROBLEM.t_final_s,
            "modes": list(DEFAULT_PROBLEM.modes),
        },
        "config": asdict(results["M1"]["train"].config),
        "metricas": rows,
        "mejora_M1_a_M2": rows[0]["error_l2"] / rows[1]["error_l2"],
    }
    write_json(OUT / "resumen.json", summary)
    print(metrics)
    print(f"Mejora M1 -> M2: {summary['mejora_M1_a_M2']:.2f}x")


def command_optimizadores(args: argparse.Namespace) -> None:
    ensure_dir(OUT)
    rows = []
    for opt in ["sgd", "adam", "adamw"]:
        opt_args = argparse.Namespace(**vars(args))
        opt_args.optimizer = opt
        data = run_model("M2", opt_args)
        rows.append(
            {
                "optimizador": opt,
                "error_l2": data["eval"]["rel_l2"],
                "tiempo_s": data["train"].elapsed_s,
                "lambda_ic_final": data["train"].lambdas["ic"],
                "lambda_bc_final": data["train"].lambdas["bc"],
            }
        )

    if args.lbfgs_steps > 0:
        opt_args = argparse.Namespace(**vars(args))
        opt_args.optimizer = "adam"
        data = run_model("M2", opt_args)
        refined = refine_with_lbfgs(data["train"], steps=args.lbfgs_steps, batch=max(args.batch, 512))
        eval_refined = evaluate_model(refined.model, problem=DEFAULT_PROBLEM, nx=args.nx, nt=args.nt)
        rows.append(
            {
                "optimizador": f"adam+lbfgs_{args.lbfgs_steps}",
                "error_l2": eval_refined["rel_l2"],
                "tiempo_s": refined.elapsed_s,
                "lambda_ic_final": refined.lambdas["ic"],
                "lambda_bc_final": refined.lambdas["bc"],
            }
        )

    df = pd.DataFrame(rows).sort_values("error_l2")
    df.to_csv(OUT / "optimizadores.csv", index=False)
    plot_optimizer_comparison(df, OUT / "comparacion_optimizadores.png")
    print(df)


def command_bayes(args: argparse.Namespace) -> None:
    import optuna

    ensure_dir(OUT)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        width = trial.suggest_categorical("width", [24, 32, 48, 64, 80])
        depth = trial.suggest_int("depth", 2, 5)
        lr = trial.suggest_float("lr", 1.0e-4, 3.0e-3, log=True)
        batch = trial.suggest_categorical("batch", [128, 256, 384])
        adaptive_beta = trial.suggest_float("adaptive_beta", 0.75, 0.97)
        cfg = TrainConfig(
            mode="M2",
            layers=layers_from_args(depth, width),
            iterations=args.search_iters,
            batch_residual=batch,
            batch_initial=batch,
            batch_boundary=batch,
            lr=lr,
            optimizer="adam",
            seed=args.seed + trial.number,
            adaptive_every=args.adaptive_every,
            adaptive_beta=adaptive_beta,
            log_every=max(args.search_iters // 5, 1),
            threads=args.threads,
        )
        result = train_pinn(cfg, problem=DEFAULT_PROBLEM, verbose=False)
        eval_res = evaluate_model(result.model, problem=DEFAULT_PROBLEM, nx=90, nt=90, device=cfg.device)
        trial.set_user_attr("lambda_ic_final", result.lambdas["ic"])
        trial.set_user_attr("lambda_bc_final", result.lambdas["bc"])
        trial.set_user_attr("elapsed_s", result.elapsed_s)
        return float(eval_res["rel_l2"])

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    df = study.trials_dataframe()
    df.to_csv(OUT / "bayes_trials.csv", index=False)
    plot_bayes_trials(df, OUT)

    best = {
        "value": study.best_value,
        "params": study.best_params,
        "trial": study.best_trial.number,
    }
    write_json(OUT / "bayes_mejor.json", best)
    print("Mejor trial:", best)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comando", choices=["todos", "m1", "m2", "optimizadores", "bayes"])
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--search-iters", type=int, default=2500)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adaptive-every", type=int, default=10)
    parser.add_argument("--adaptive-beta", type=float, default=0.9)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--nx", type=int, default=180)
    parser.add_argument("--nt", type=int, default=180)
    parser.add_argument("--lbfgs-steps", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    np.set_printoptions(precision=4, suppress=True)
    if args.comando == "todos":
        command_todos(args)
    elif args.comando == "m1":
        ensure_dir(OUT)
        data = run_model("M1", args)
        plot_solution(data["eval"], "Prediccion M1", OUT / "M1_solucion.png")
        plot_history(data["train"].history, "M1", OUT / "M1_historia.png")
        plot_gradient_histograms(data["snapshot"], "M1", OUT / "M1_gradientes.png")
    elif args.comando == "m2":
        ensure_dir(OUT)
        data = run_model("M2", args)
        plot_solution(data["eval"], "Prediccion M2", OUT / "M2_solucion.png")
        plot_history(data["train"].history, "M2", OUT / "M2_historia.png")
        plot_gradient_histograms(data["snapshot"], "M2", OUT / "M2_gradientes.png")
    elif args.comando == "optimizadores":
        command_optimizadores(args)
    elif args.comando == "bayes":
        command_bayes(args)


if __name__ == "__main__":
    main()
