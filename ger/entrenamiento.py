"""Entrenamiento de PINNs para calor 1D.

Modelos:
  M1: PINN clasica, L = L_r + L_ic + L_bc.
  M2: misma arquitectura, pero con pesos adaptativos para L_ic y L_bc.

El modo M2 implementa la correccion central del paper de Wang, Teng y
Perdikaris: balancear terminos de la perdida usando estadisticas de gradientes.
No se usa la arquitectura nueva del paper.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch

from pinn import PINN
from problema import DEFAULT_PROBLEM, HeatProblem, analytic_solution_np


OptimizerName = Literal["adam", "adamw", "sgd"]


@dataclass
class TrainConfig:
    mode: Literal["M1", "M2"] = "M1"
    layers: tuple[int, ...] = (2, 50, 50, 50, 1)
    iterations: int = 10000
    batch_residual: int = 256
    batch_initial: int = 256
    batch_boundary: int = 256
    lr: float = 1.0e-3
    optimizer: OptimizerName = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    seed: int = 0
    adaptive_every: int = 10
    adaptive_beta: float = 0.9
    log_every: int = 100
    decay_rate: float = 0.9
    decay_steps: int = 1000
    device: str = "cpu"
    threads: int | None = None


@dataclass
class TrainResult:
    model: PINN
    history: dict[str, list[float]]
    config: TrainConfig
    lambdas: dict[str, float]
    elapsed_s: float


def configure_torch(threads: int | None = None) -> int:
    """Fija threads de CPU para usar los cores disponibles."""

    n_threads = int(threads or os.cpu_count() or 1)
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, n_threads // 2)))
    except RuntimeError:
        pass
    return n_threads


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def initial_condition_torch(x: torch.Tensor, problem: HeatProblem) -> torch.Tensor:
    u = torch.zeros_like(x)
    for n, coef in problem.modes:
        u = u + coef * torch.sin(n * torch.pi * x)
    return u


def derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def make_optimizer(model: PINN, cfg: TrainConfig) -> torch.optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Optimizador no soportado: {cfg.optimizer}")


def sample_losses(
    model: PINN,
    cfg: TrainConfig,
    problem: HeatProblem,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Calcula L_res, L_ic y L_bc sobre minibatches nuevos."""

    # Residuo PDE: u_tau - u_xx = 0.
    x_r = torch.rand(cfg.batch_residual, 1, device=device, requires_grad=True)
    tau_r = problem.tau_final * torch.rand(
        cfg.batch_residual, 1, device=device, requires_grad=True
    )
    u_r = model(torch.cat([x_r, tau_r], dim=1))
    u_x = derivative(u_r, x_r)
    u_tau = derivative(u_r, tau_r)
    u_xx = derivative(u_x, x_r)
    residual = u_tau - u_xx
    loss_res = torch.mean(residual**2)

    # Condicion inicial.
    x_0 = torch.rand(cfg.batch_initial, 1, device=device)
    tau_0 = torch.zeros_like(x_0)
    u_0 = model(torch.cat([x_0, tau_0], dim=1))
    target_0 = initial_condition_torch(x_0, problem)
    loss_ic = torch.mean((u_0 - target_0) ** 2)

    # Condiciones de borde u(0,tau)=u(1,tau)=0.
    n_left = cfg.batch_boundary // 2
    n_right = cfg.batch_boundary - n_left
    tau_left = problem.tau_final * torch.rand(n_left, 1, device=device)
    tau_right = problem.tau_final * torch.rand(n_right, 1, device=device)
    x_left = torch.zeros(n_left, 1, device=device)
    x_right = torch.ones(n_right, 1, device=device)
    pts_b = torch.cat(
        [
            torch.cat([x_left, tau_left], dim=1),
            torch.cat([x_right, tau_right], dim=1),
        ],
        dim=0,
    )
    loss_bc = torch.mean(model(pts_b) ** 2)

    return {"res": loss_res, "ic": loss_ic, "bc": loss_bc}


def layer_gradients(
    model: PINN,
    loss: torch.Tensor,
    retain_graph: bool = True,
) -> list[torch.Tensor]:
    grads = torch.autograd.grad(
        loss,
        model.weight_tensors(),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=False,
    )
    return [g.detach() for g in grads]


def suggested_lambdas(
    model: PINN,
    loss_res: torch.Tensor,
    data_losses: dict[str, torch.Tensor],
    eps: float = 1.0e-12,
) -> dict[str, float]:
    """Implementa el estimador lambda_hat del Algoritmo 1.

    Se usa L_ic y L_bc sin ponderar en el denominador, tal como esta escrito
    en la ecuacion (40) del paper. La media movil de M2 replica el suavizado
    beta=0.9 del codigo publicado por los autores.
    """

    grad_res = layer_gradients(model, loss_res, retain_graph=True)
    max_res = torch.stack([g.abs().max() for g in grad_res]).max()

    out: dict[str, float] = {}
    for name, loss in data_losses.items():
        grad_data = layer_gradients(model, loss, retain_graph=True)
        mean_data = torch.stack([g.abs().mean() for g in grad_data]).mean()
        out[name] = float((max_res / (mean_data + eps)).clamp(1.0e-6, 1.0e6).cpu())
    return out


def total_loss(
    losses: dict[str, torch.Tensor],
    lambda_ic: float,
    lambda_bc: float,
) -> torch.Tensor:
    return losses["res"] + lambda_ic * losses["ic"] + lambda_bc * losses["bc"]


def train_pinn(
    cfg: TrainConfig,
    problem: HeatProblem = DEFAULT_PROBLEM,
    verbose: bool = True,
) -> TrainResult:
    threads = configure_torch(cfg.threads)
    _ = set_seed(cfg.seed)
    device = torch.device(cfg.device)
    bounds = torch.tensor(problem.bounds, dtype=torch.float32, device=device)
    model = PINN(cfg.layers, bounds=bounds).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg.decay_rate ** (1.0 / cfg.decay_steps)
    )

    lambda_ic = 1.0
    lambda_bc = 1.0
    use_adaptive = cfg.mode.upper() == "M2"

    history: dict[str, list[float]] = {
        "iter": [],
        "loss_total": [],
        "loss_res": [],
        "loss_ic": [],
        "loss_bc": [],
        "lambda_ic": [],
        "lambda_bc": [],
        "lr": [],
    }

    start = time.perf_counter()
    if verbose:
        print(
            f"Entrenando {cfg.mode} con {cfg.optimizer}, layers={list(cfg.layers)}, "
            f"iters={cfg.iterations}, threads={threads}"
        )

    for it in range(cfg.iterations + 1):
        losses = sample_losses(model, cfg, problem, device)

        if use_adaptive and it > 0 and it % cfg.adaptive_every == 0:
            suggested = suggested_lambdas(
                model,
                losses["res"],
                {"ic": losses["ic"], "bc": losses["bc"]},
            )
            lambda_ic = cfg.adaptive_beta * lambda_ic + (1.0 - cfg.adaptive_beta) * suggested["ic"]
            lambda_bc = cfg.adaptive_beta * lambda_bc + (1.0 - cfg.adaptive_beta) * suggested["bc"]

        loss = total_loss(losses, lambda_ic, lambda_bc)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if it % cfg.log_every == 0 or it == cfg.iterations:
            history["iter"].append(float(it))
            history["loss_total"].append(float(loss.detach().cpu()))
            history["loss_res"].append(float(losses["res"].detach().cpu()))
            history["loss_ic"].append(float(losses["ic"].detach().cpu()))
            history["loss_bc"].append(float(losses["bc"].detach().cpu()))
            history["lambda_ic"].append(float(lambda_ic))
            history["lambda_bc"].append(float(lambda_bc))
            history["lr"].append(float(scheduler.get_last_lr()[0]))
            if verbose:
                print(
                    f"  it={it:6d} loss={history['loss_total'][-1]:.3e} "
                    f"Lr={history['loss_res'][-1]:.3e} "
                    f"Lic={history['loss_ic'][-1]:.3e} "
                    f"Lbc={history['loss_bc'][-1]:.3e} "
                    f"lam_ic={lambda_ic:.2f} lam_bc={lambda_bc:.2f}"
                )

    elapsed = time.perf_counter() - start
    return TrainResult(
        model=model,
        history=history,
        config=cfg,
        lambdas={"ic": lambda_ic, "bc": lambda_bc},
        elapsed_s=elapsed,
    )


def evaluate_model(
    model: PINN,
    problem: HeatProblem = DEFAULT_PROBLEM,
    nx: int = 180,
    nt: int = 180,
    device: str = "cpu",
) -> dict[str, np.ndarray | float]:
    from problema import make_grid

    xx, tt, pts = make_grid(nx=nx, nt=nt, problem=problem)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()
    exact = analytic_solution_np(pts[:, 0:1], pts[:, 1:2], problem=problem)
    error = np.linalg.norm(pred - exact) / np.linalg.norm(exact)
    return {
        "x": xx,
        "tau": tt,
        "points": pts,
        "u_pred": pred.reshape(nt, nx),
        "u_exact": exact.reshape(nt, nx),
        "abs_error": np.abs(pred - exact).reshape(nt, nx),
        "rel_l2": float(error),
    }


def gradient_snapshot(
    model: PINN,
    cfg: TrainConfig,
    problem: HeatProblem = DEFAULT_PROBLEM,
    batch: int = 1024,
) -> dict[str, list[np.ndarray]]:
    """Gradientes por capa para graficos tipo Fig. 2/Fig. 7 del paper."""

    device = torch.device(cfg.device)
    snap_cfg = TrainConfig(**asdict(cfg))
    snap_cfg.batch_residual = batch
    snap_cfg.batch_initial = batch
    snap_cfg.batch_boundary = batch
    losses = sample_losses(model, snap_cfg, problem, device)
    snapshot: dict[str, list[np.ndarray]] = {}
    for key in ["res", "ic", "bc"]:
        grads = layer_gradients(model, losses[key], retain_graph=True)
        snapshot[key] = [g.cpu().numpy().ravel() for g in grads]
    return snapshot


def refine_with_lbfgs(
    result: TrainResult,
    steps: int = 80,
    batch: int = 1024,
    problem: HeatProblem = DEFAULT_PROBLEM,
) -> TrainResult:
    """Refinamiento opcional con LBFGS despues de Adam/AdamW/SGD."""

    cfg = TrainConfig(**asdict(result.config))
    cfg.batch_residual = batch
    cfg.batch_initial = batch
    cfg.batch_boundary = batch
    device = torch.device(cfg.device)
    model = result.model
    model.train()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    lambda_ic = result.lambdas["ic"]
    lambda_bc = result.lambdas["bc"]
    start = time.perf_counter()

    for _ in range(steps):
        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            losses = sample_losses(model, cfg, problem, device)
            loss = total_loss(losses, lambda_ic, lambda_bc)
            loss.backward()
            return loss

        optimizer.step(closure)

    result.elapsed_s += time.perf_counter() - start
    return result
