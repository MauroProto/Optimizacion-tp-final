"""
Solver clasico por diferencias finitas (FD). Es el "control" del proyecto.

Por que sirve:
  - Es un metodo numerico estandar, no usa redes ni nada raro.
  - Si FD da una solucion muy parecida a u_exacta, sabemos que el problema esta
    bien planteado. Entonces, si la PINN falla, es por la optimizacion, no por
    un error en el setup.
  - Tambien es un baseline para comparar: ¿que tan rapido / preciso es FD vs PINN?

Como funciona:
  - Cubrimos el cuadrado [-1, 1]^2 con una grilla de n x n puntos.
  - En cada punto interior, reemplazamos la derivada segunda por su aproximacion
    discreta usando los 4 vecinos (el "stencil de 5 puntos"):

        u_xx ≈ (u_derecha + u_izquierda - 2·u_centro) / h²
        u_yy ≈ (u_arriba + u_abajo     - 2·u_centro) / h²

  - Eso convierte la PDE en un sistema lineal grande pero disperso: A·u = b.
  - Lo resolvemos con scipy (rapido para sistemas dispersos).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from problema import LAMBDA, forcing, u_exacta


def resolver_fd(n=200):
    """Resuelve la PDE en una grilla n x n y devuelve la solucion + el error.

    Recomendacion: n >= 100. Como la solucion tiene 4 oscilaciones en y
    (porque a2=4), si la grilla es muy gruesa no las captura.
    """
    # 1) Grilla.
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    h = x[1] - x[0]
    X1, X2 = np.meshgrid(x, y)
    puntos = np.column_stack([X1.flatten(), X2.flatten()])

    # 2) Identificamos los puntos del borde y del interior.
    es_borde = (
        (X1 == -1) | (X1 == 1) | (X2 == -1) | (X2 == 1)
    ).flatten()
    idx_interior = np.where(~es_borde)[0]
    N_int = len(idx_interior)

    # Mapa: indice global del punto -> indice 0..N_int-1 dentro de la matriz A.
    global_a_interior = -np.ones(n * n, dtype=int)
    global_a_interior[idx_interior] = np.arange(N_int)

    # 3) Forcing en todos los puntos, y borde (que vale 0 pero lo dejamos general).
    q = forcing(puntos).flatten()
    u_borde = u_exacta(puntos).flatten()

    # 4) Armamos la matriz A y el vector b del sistema A·u = b.
    # Cada fila k corresponde a un punto interior y dice:
    #     (u_der + u_izq + u_arr + u_aba - 4·u_k)/h² + λ·u_k = q_k
    diag_coef = -4.0 / h ** 2 + LAMBDA
    off_coef = 1.0 / h ** 2

    filas, cols, valores = [], [], []
    b = np.zeros(N_int)

    for k, g in enumerate(idx_interior):
        i, j = g // n, g % n  # fila y columna del punto en la grilla
        # Diagonal
        filas.append(k); cols.append(k); valores.append(diag_coef)

        # Vecinos: arriba, abajo, izquierda, derecha
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n:
                g_vecino = ni * n + nj
                if es_borde[g_vecino]:
                    # Si el vecino es del borde, su valor es conocido,
                    # lo pasamos al lado derecho con signo cambiado.
                    b[k] -= off_coef * u_borde[g_vecino]
                else:
                    filas.append(k)
                    cols.append(global_a_interior[g_vecino])
                    valores.append(off_coef)
        b[k] += q[g]

    A = sp.csr_matrix((valores, (filas, cols)), shape=(N_int, N_int))

    # 5) Resolvemos el sistema (scipy hace LU disperso por dentro).
    u_int = spla.spsolve(A, b)

    # 6) Armamos la solucion completa (interior + borde) en forma de matriz n x n.
    u_full = np.empty(n * n)
    u_full[idx_interior] = u_int
    u_full[es_borde] = u_borde[es_borde]
    u_fd = u_full.reshape((n, n))

    # 7) Error vs la solucion exacta.
    u_real = u_exacta(puntos).reshape((n, n))
    error_l2 = np.linalg.norm((u_real - u_fd).flatten()) / np.linalg.norm(u_real.flatten())

    return {
        "n": n, "h": h,
        "X1": X1, "X2": X2,
        "u_fd": u_fd,
        "u_exacta": u_real,
        "error_rel_L2": error_l2,
    }
