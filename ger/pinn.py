"""Red neuronal base para la PINN."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class PINN(nn.Module):
    """MLP totalmente conectada con activacion tanh.

    La entrada [x, tau] se escala internamente a [-1, 1]^2. Como el escalado
    ocurre dentro del grafo de PyTorch, autograd aplica correctamente las
    constantes de derivacion al calcular u_tau y u_xx.
    """

    def __init__(self, layers: Sequence[int], bounds: torch.Tensor):
        super().__init__()
        if len(layers) < 3:
            raise ValueError("layers debe tener entrada, al menos una capa oculta y salida")

        self.layers = nn.ModuleList(
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)
        )
        self.register_buffer("lb", bounds[0].clone().detach().float())
        self.register_buffer("ub", bounds[1].clone().detach().float())

        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)

    def weight_tensors(self) -> list[torch.Tensor]:
        """Pesos usados para medir gradientes, como en el repo de los autores."""

        return [layer.weight for layer in self.layers]
