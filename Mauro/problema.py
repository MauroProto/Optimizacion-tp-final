"""
Definicion del problema Helmholtz 2D.

Resolvemos:    Δu + λu = q     en el cuadrado [-1, 1] x [-1, 1]
Con borde:     u = 0           en los 4 lados

Para tener la respuesta correcta y poder medir el error, "fabricamos" la solucion:
    u_exacta(x, y) = sin(π·x) · sin(4π·y)

y calculamos a mano que forcing 'q' necesitamos. Sale derivando dos veces:
    u_xx = -π²    · sin(π·x) · sin(4π·y)
    u_yy = -(4π)² · sin(π·x) · sin(4π·y)
Entonces:
    q = u_xx + u_yy + λu = [ -π² - (4π)² + λ ] · u_exacta

Asi sabemos que valor de 'q' usar y, despues de entrenar, comparamos la
prediccion de la red con u_exacta para medir cuanto error tiene.
"""

import numpy as np


# Constantes del problema (no se tocan, salen del paper)
LAMBDA = 1.0          # el "λ" de la ecuacion de Helmholtz
A1 = 1.0              # frecuencia en x
A2 = 4.0              # frecuencia en y (mas alta -> 4 oscilaciones en y, mas dificil)


def u_exacta(x):
    """Solucion exacta del problema, evaluada en los puntos x.

    x tiene shape (N, 2): cada fila es un punto (x, y) del cuadrado.
    Devuelve un array (N, 1) con el valor de u en cada punto.
    """
    return np.sin(A1 * np.pi * x[:, 0:1]) * np.sin(A2 * np.pi * x[:, 1:2])


def forcing(x):
    """El lado derecho de la PDE: q(x, y). Lo calculamos analiticamente."""
    factor = -(A1 * np.pi) ** 2 - (A2 * np.pi) ** 2 + LAMBDA
    return factor * u_exacta(x)


def muestrear_interior(n):
    """Saca n puntos al azar del interior del cuadrado [-1, 1]^2.

    Estos puntos los usamos para forzar que la red cumpla la ecuacion (residuo).
    """
    return np.random.uniform(-1, 1, size=(n, 2))


def muestrear_borde(n):
    """Saca n puntos al azar de los 4 bordes del cuadrado (juntos).

    Estos puntos los usamos para forzar que la red valga 0 en el borde.
    Cada borde tiene n/4 puntos.
    """
    m = n // 4
    bordes = []
    # Borde inferior: y = -1
    bordes.append(np.column_stack([np.random.uniform(-1, 1, m), -np.ones(m)]))
    # Borde superior: y = +1
    bordes.append(np.column_stack([np.random.uniform(-1, 1, m), np.ones(m)]))
    # Borde izquierdo: x = -1
    bordes.append(np.column_stack([-np.ones(m), np.random.uniform(-1, 1, m)]))
    # Borde derecho: x = +1
    bordes.append(np.column_stack([np.ones(m), np.random.uniform(-1, 1, m)]))
    return np.vstack(bordes)


def grilla_test(n=100):
    """Arma una grilla regular de n x n puntos sobre el cuadrado para evaluar.

    Devuelve:
      X1, X2  -> matrices (n, n) para hacer pcolor / contour
      puntos  -> array (n*n, 2) plano, listo para meter en la red
    """
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X1, X2 = np.meshgrid(x, y)
    puntos = np.column_stack([X1.flatten(), X2.flatten()])
    return X1, X2, puntos
