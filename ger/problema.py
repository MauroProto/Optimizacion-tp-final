"""Problema fisico: difusion de calor 1D en una barra metalica.

Se trabaja con variables adimensionales:

    x = X / L,  tau = alpha * t / L^2,  u = (T - T_amb) / DeltaT

La ecuacion queda

    u_tau - u_xx = 0,  x in (0, 1), tau in (0, tau_f)

con extremos mantenidos a temperatura ambiente:

    u(0, tau) = u(1, tau) = 0

y condicion inicial compatible con los bordes. Al elegir una suma de senos,
la solucion analitica se obtiene por separacion de variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


ModeSpec = Tuple[Tuple[int, float], ...]


@dataclass(frozen=True)
class HeatProblem:
    """Parametros fisicos y numericos del problema."""

    length_m: float = 1.0
    diffusivity_m2_s: float = 1.0e-4
    ambient_c: float = 20.0
    delta_temp_c: float = 80.0
    tau_final: float = 0.25
    modes: ModeSpec = ((1, 1.0), (3, 0.5), (5, 0.25))

    @property
    def t_final_s(self) -> float:
        return self.tau_final * self.length_m**2 / self.diffusivity_m2_s

    @property
    def bounds(self) -> np.ndarray:
        return np.array([[0.0, 0.0], [1.0, self.tau_final]], dtype=np.float32)


DEFAULT_PROBLEM = HeatProblem()


def initial_condition_np(x: np.ndarray, problem: HeatProblem = DEFAULT_PROBLEM) -> np.ndarray:
    """u(x, 0) para x con shape (N, 1) o (N,)."""

    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    u = np.zeros_like(x)
    for n, coef in problem.modes:
        u += coef * np.sin(n * np.pi * x)
    return u


def analytic_solution_np(
    x: np.ndarray,
    tau: np.ndarray,
    problem: HeatProblem = DEFAULT_PROBLEM,
) -> np.ndarray:
    """Solucion analitica adimensional u(x, tau)."""

    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    tau = np.asarray(tau, dtype=np.float64).reshape(-1, 1)
    u = np.zeros_like(x)
    for n, coef in problem.modes:
        decay = np.exp(-((n * np.pi) ** 2) * tau)
        u += coef * np.sin(n * np.pi * x) * decay
    return u


def dimensional_temperature_c(
    u: np.ndarray,
    problem: HeatProblem = DEFAULT_PROBLEM,
) -> np.ndarray:
    """Convierte u adimensional a temperatura en Celsius."""

    return problem.ambient_c + problem.delta_temp_c * np.asarray(u)


def make_grid(
    nx: int = 160,
    nt: int = 160,
    problem: HeatProblem = DEFAULT_PROBLEM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grilla regular para evaluar y graficar.

    Devuelve X, T y puntos planos con columnas [x, tau].
    """

    x = np.linspace(0.0, 1.0, nx)
    tau = np.linspace(0.0, problem.tau_final, nt)
    tt, xx = np.meshgrid(tau, x, indexing="ij")
    pts = np.column_stack([xx.ravel(), tt.ravel()])
    return xx, tt, pts


def sample_residual(
    n: int,
    rng: np.random.Generator,
    problem: HeatProblem = DEFAULT_PROBLEM,
) -> np.ndarray:
    x = rng.uniform(0.0, 1.0, size=(n, 1))
    tau = rng.uniform(0.0, problem.tau_final, size=(n, 1))
    return np.column_stack([x[:, 0], tau[:, 0]]).astype(np.float32)


def sample_initial(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(0.0, 1.0, size=(n, 1))
    tau = np.zeros_like(x)
    return np.column_stack([x[:, 0], tau[:, 0]]).astype(np.float32)


def sample_boundary(
    n: int,
    rng: np.random.Generator,
    problem: HeatProblem = DEFAULT_PROBLEM,
) -> np.ndarray:
    n_left = n // 2
    n_right = n - n_left
    tau_left = rng.uniform(0.0, problem.tau_final, size=(n_left, 1))
    tau_right = rng.uniform(0.0, problem.tau_final, size=(n_right, 1))
    left = np.column_stack([np.zeros(n_left), tau_left[:, 0]])
    right = np.column_stack([np.ones(n_right), tau_right[:, 0]])
    return np.vstack([left, right]).astype(np.float32)
