"""
La red neuronal de la PINN.

Es un MLP comun: varias capas lineales con tanh entre medio.
Recibe un punto (x, y) y devuelve un numero u(x, y).
"""

import torch
import torch.nn as nn


class PINN(nn.Module):
    """Red neuronal totalmente conectada con activacion tanh.

    Ejemplo de uso:
        red = PINN(capas=[2, 50, 50, 50, 1])
        # 2 entradas (x, y) -> 50 -> 50 -> 50 -> 1 salida (u)
    """

    def __init__(self, capas):
        super().__init__()
        # Armamos las capas lineales en orden.
        self.capas = nn.ModuleList()
        for i in range(len(capas) - 1):
            self.capas.append(nn.Linear(capas[i], capas[i + 1]))

        # Inicializacion Xavier: arranca con pesos balanceados.
        # Sin esto, las redes profundas suelen empezar mal.
        for capa in self.capas:
            nn.init.xavier_normal_(capa.weight)
            nn.init.zeros_(capa.bias)

    def forward(self, x):
        # Aplicamos tanh despues de cada capa, menos la ultima.
        for capa in self.capas[:-1]:
            x = torch.tanh(capa(x))
        # La ultima capa es lineal (sin tanh) para no acotar la salida.
        return self.capas[-1](x)
