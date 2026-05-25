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
    plot_gradient_norms,
    plot_gradient_histograms,
    plot_history,
    plot_lambda_evolution,
    plot_optimizer_comparison,
    plot_paper_solution_comparison,
    plot_seed_summary,
    plot_solution,
    plot_time_slices,
)
from problema import DEFAULT_PROBLEM


OUT = Path("outputs")


def layers_from_args(depth: int, width: int) -> tuple[int, ...]:
    return tuple([2] + [width] * depth + [1])


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        momentum=args.momentum,
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
    plot_paper_solution_comparison(
        {name: data["eval"] for name, data in results.items()},
        OUT / "paper_comparacion_soluciones.png",
    )
    plot_gradient_norms(
        {name: data["snapshot"] for name, data in results.items()},
        OUT / "paper_balance_gradientes.png",
    )
    plot_lambda_evolution(results["M2"]["train"].history, OUT / "paper_lambda_evolucion.png")

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


def command_semillas(args: argparse.Namespace) -> None:
    ensure_dir(OUT)
    rows = []
    for seed in args.seeds:
        for mode in ["M1", "M2"]:
            seed_args = argparse.Namespace(**vars(args))
            seed_args.seed = seed
            data = run_model(mode, seed_args)
            rows.append(
                {
                    "seed": seed,
                    "modelo": mode,
                    "error_l2": data["eval"]["rel_l2"],
                    "tiempo_s": data["train"].elapsed_s,
                    "lambda_ic_final": data["train"].lambdas["ic"],
                    "lambda_bc_final": data["train"].lambdas["bc"],
                    "iters": args.iters,
                    "width": args.width,
                    "depth": args.depth,
                    "lr": args.lr,
                    "batch": args.batch,
                    "adaptive_beta": args.adaptive_beta,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metricas_semillas.csv", index=False)
    summary = (
        df.groupby("modelo")["error_l2"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.to_csv(OUT / "metricas_semillas_resumen.csv", index=False)
    plot_seed_summary(df, OUT / "comparacion_semillas.png")
    print(df)
    print(summary)


def finite_or_inf(value: float) -> float:
    return float(value) if np.isfinite(value) else float("inf")


def command_optimizadores(args: argparse.Namespace) -> None:
    ensure_dir(OUT)
    candidates = [
        {"name": "sgd_lr1e-5_m0.0", "optimizer": "sgd", "lr": 1.0e-5, "momentum": 0.0},
        {"name": "sgd_lr3e-5_m0.5", "optimizer": "sgd", "lr": 3.0e-5, "momentum": 0.5},
        {"name": "sgd_lr1e-4_m0.9", "optimizer": "sgd", "lr": 1.0e-4, "momentum": 0.9},
        {"name": "adam_lr5e-4", "optimizer": "adam", "lr": 5.0e-4, "momentum": args.momentum},
        {"name": "adam_lr1e-3", "optimizer": "adam", "lr": 1.0e-3, "momentum": args.momentum},
        {"name": "adam_lr2e-3", "optimizer": "adam", "lr": 2.0e-3, "momentum": args.momentum},
        {"name": "adamw_lr5e-4", "optimizer": "adamw", "lr": 5.0e-4, "momentum": args.momentum},
        {"name": "adamw_lr1e-3", "optimizer": "adamw", "lr": 1.0e-3, "momentum": args.momentum},
        {"name": "adamw_lr2e-3", "optimizer": "adamw", "lr": 2.0e-3, "momentum": args.momentum},
    ]
    rows = []
    trained = {}
    for cand in candidates:
        opt_args = argparse.Namespace(**vars(args))
        opt_args.optimizer = cand["optimizer"]
        opt_args.lr = cand["lr"]
        opt_args.momentum = cand["momentum"]
        data = run_model("M2", opt_args)
        err = finite_or_inf(data["eval"]["rel_l2"])
        rows.append(
            {
                "optimizador": cand["name"],
                "familia": cand["optimizer"],
                "lr": cand["lr"],
                "momentum": cand["momentum"],
                "error_l2": data["eval"]["rel_l2"],
                "tiempo_s": data["train"].elapsed_s,
                "lambda_ic_final": data["train"].lambdas["ic"],
                "lambda_bc_final": data["train"].lambdas["bc"],
            }
        )
        trained[cand["name"]] = (data, err)

    finite_rows = [row for row in rows if np.isfinite(row["error_l2"])]
    best_adam = min(
        (row for row in finite_rows if row["familia"] == "adam"),
        key=lambda row: row["error_l2"],
        default=None,
    )
    if best_adam and args.lbfgs_steps > 0:
        data, _ = trained[best_adam["optimizador"]]
        refined = refine_with_lbfgs(data["train"], steps=args.lbfgs_steps, batch=max(args.batch, 512))
        eval_refined = evaluate_model(refined.model, problem=DEFAULT_PROBLEM, nx=args.nx, nt=args.nt)
        rows.append(
            {
                "optimizador": f"{best_adam['optimizador']}+lbfgs_{args.lbfgs_steps}",
                "familia": "adam+lbfgs",
                "lr": best_adam["lr"],
                "momentum": best_adam["momentum"],
                "error_l2": eval_refined["rel_l2"],
                "tiempo_s": refined.elapsed_s,
                "lambda_ic_final": refined.lambdas["ic"],
                "lambda_bc_final": refined.lambdas["bc"],
            }
        )

    df = pd.DataFrame(rows).sort_values("error_l2", na_position="last")
    df.to_csv(OUT / "optimizadores.csv", index=False)
    plot_optimizer_comparison(df, OUT / "comparacion_optimizadores.png")
    print(df)


def run_bayes_search(
    args: argparse.Namespace,
    mode: str,
    trials_name: str,
    best_name: str,
    plot_stem: str,
    plot_title: str,
) -> None:
    import optuna

    ensure_dir(OUT)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    storage = args.storage
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=args.study_name,
        load_if_exists=storage is not None,
    )

    def objective(trial: optuna.Trial) -> float:
        width = trial.suggest_categorical("width", [24, 32, 48, 64, 80])
        depth = trial.suggest_int("depth", 2, 5)
        lr = trial.suggest_float("lr", 1.0e-4, 3.0e-3, log=True)
        batch = trial.suggest_categorical("batch", [128, 256, 384])
        adaptive_beta = (
            trial.suggest_float("adaptive_beta", 0.75, 0.97)
            if mode.upper() == "M2"
            else args.adaptive_beta
        )
        cfg = TrainConfig(
            mode=mode,
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

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    remaining = max(args.trials - len(complete), 0) if args.target_trials else args.trials
    study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    df = study.trials_dataframe()
    df.to_csv(OUT / trials_name, index=False)
    plot_bayes_trials(df, OUT, stem=plot_stem, title=plot_title)

    best = {
        "value": study.best_value,
        "params": study.best_params,
        "trial": study.best_trial.number,
    }
    write_json(OUT / best_name, best)
    print("Mejor trial:", best)


def command_bayes(args: argparse.Namespace) -> None:
    run_bayes_search(
        args,
        mode="M2",
        trials_name="bayes_trials.csv",
        best_name="bayes_mejor.json",
        plot_stem="bayes",
        plot_title="Busqueda bayesiana M2",
    )


def command_bayes_m1(args: argparse.Namespace) -> None:
    run_bayes_search(
        args,
        mode="M1",
        trials_name="bayes_m1_trials.csv",
        best_name="bayes_m1_mejor.json",
        plot_stem="bayes_m1",
        plot_title="Busqueda bayesiana M1",
    )


def load_m1b_regimen() -> dict:
    validation_path = OUT / "validacion_m1b_mejor.json"
    if validation_path.exists():
        data = read_json(validation_path)
        return {
            "source": str(validation_path),
            "trial": int(data["trial"]),
            "short_error_l2": float(data["short_error_l2"]),
            "validated_error_l2": float(data["long_error_l2"]),
            "width": int(data["width"]),
            "depth": int(data["depth"]),
            "lr": float(data["lr"]),
            "batch": int(data["batch"]),
        }

    best_path = OUT / "bayes_m1_mejor.json"
    data = read_json(best_path)
    params = data["params"]
    return {
        "source": str(best_path),
        "trial": int(data["trial"]),
        "short_error_l2": float(data["value"]),
        "validated_error_l2": np.nan,
        "width": int(params["width"]),
        "depth": int(params["depth"]),
        "lr": float(params["lr"]),
        "batch": int(params["batch"]),
    }


def command_m1b(args: argparse.Namespace) -> None:
    ensure_dir(OUT)
    regimen = load_m1b_regimen()
    m1b_args = argparse.Namespace(**vars(args))
    m1b_args.width = regimen["width"]
    m1b_args.depth = regimen["depth"]
    m1b_args.lr = regimen["lr"]
    m1b_args.batch = regimen["batch"]
    m1b_args.optimizer = "adam"
    m1b_args.adaptive_beta = 1.0

    data = run_model("M1", m1b_args)
    plot_solution(data["eval"], "Prediccion M1B", OUT / "M1B_solucion.png")
    plot_history(data["train"].history, "M1B", OUT / "M1B_historia.png")
    plot_gradient_histograms(data["snapshot"], "M1B", OUT / "M1B_gradientes.png")

    row = {
        "modelo": "M1B",
        "error_l2": data["eval"]["rel_l2"],
        "tiempo_s": data["train"].elapsed_s,
        "lambda_ic_final": data["train"].lambdas["ic"],
        "lambda_bc_final": data["train"].lambdas["bc"],
        "trial": regimen["trial"],
        "short_error_l2": regimen["short_error_l2"],
        "validated_error_l2": regimen["validated_error_l2"],
        "width": regimen["width"],
        "depth": regimen["depth"],
        "lr": regimen["lr"],
        "batch": regimen["batch"],
        "source": regimen["source"],
    }
    m1b_df = pd.DataFrame([row])
    m1b_df.to_csv(OUT / "metricas_m1b.csv", index=False)

    metrics_path = OUT / "metricas_modelos.csv"
    if metrics_path.exists():
        base = pd.read_csv(metrics_path)
        comparison = pd.concat(
            [base, m1b_df[["modelo", "error_l2", "tiempo_s", "lambda_ic_final", "lambda_bc_final"]]],
            ignore_index=True,
        )
    else:
        comparison = m1b_df[["modelo", "error_l2", "tiempo_s", "lambda_ic_final", "lambda_bc_final"]]
    comparison.to_csv(OUT / "metricas_modelos_m1b.csv", index=False)
    plot_error_bars(comparison, OUT / "comparacion_modelos_m1b.png")

    write_json(
        OUT / "resumen_m1b.json",
        {
            "regimen": regimen,
            "config": asdict(data["train"].config),
            "metricas": row,
        },
    )
    print(m1b_df)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comando",
        choices=["todos", "m1", "m2", "m1b", "optimizadores", "bayes", "bayes-m1", "semillas"],
    )
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--search-iters", type=int, default=800)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--adaptive-every", type=int, default=10)
    parser.add_argument("--adaptive-beta", type=float, default=0.9)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--nx", type=int, default=180)
    parser.add_argument("--nt", type=int, default=180)
    parser.add_argument("--lbfgs-steps", type=int, default=30)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--study-name", default="calor_1d_m2")
    parser.add_argument("--target-trials", action="store_true")
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
    elif args.comando == "m1b":
        command_m1b(args)
    elif args.comando == "optimizadores":
        command_optimizadores(args)
    elif args.comando == "bayes":
        command_bayes(args)
    elif args.comando == "bayes-m1":
        command_bayes_m1(args)
    elif args.comando == "semillas":
        command_semillas(args)


if __name__ == "__main__":
    main()
