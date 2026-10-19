"""Pérdida logística + ranking por pares + regularización L2."""
import numpy as np
from scipy.special import expit


def generar_pares(y, cantidad=20000, semilla=42):
    """Muestreo uniforme por fila, con reemplazo; sólo datos de ajuste."""
    y = np.asarray(y)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("El objetivo debe ser binario 0/1.")
    positivos = np.flatnonzero(y == 1)
    negativos = np.flatnonzero(y == 0)
    if not len(positivos) or not len(negativos):
        raise ValueError("Se necesitan ambas clases.")
    if cantidad < 1:
        raise ValueError("La cantidad de pares debe ser positiva.")
    rng = np.random.default_rng(semilla)
    return (
        rng.choice(positivos, size=cantidad, replace=True),
        rng.choice(negativos, size=cantidad, replace=True)
    )


def perdida_gradiente(theta, X, y, positivos, negativos, alpha=0.25, l2=0.01):
    """theta[0] es el intercepto, excluido de la penalización.

    X ya debe estar transformada usando exclusivamente el ajuste.
    Los pares deben permanecer fijos durante una optimización.
    No se construye una matriz de diferencias de características por par.
    """
    if not 0 <= alpha <= 1 or l2 < 0:
        raise ValueError("Se requiere 0 <= alpha <= 1 y l2 >= 0.")

    intercepto, coef = theta[0], theta[1:]
    score = X @ coef + intercepto
    n = len(y)

    bce = np.mean(np.logaddexp(0.0, score) - y * score)
    derivada_bce = (expit(score) - y) / n
    derivada_score = (1.0 - alpha) * derivada_bce
    ranking = 0.0

    if alpha:
        if len(positivos) == 0 or len(positivos) != len(negativos):
            raise ValueError("Pares vacíos o de tamaños diferentes.")
        diferencias = score[positivos] - score[negativos]
        ranking = np.mean(np.logaddexp(0.0, -diferencias))
        q = expit(-diferencias) / len(diferencias)
        derivada_ranking = (
            np.bincount(negativos, weights=q, minlength=n)
            - np.bincount(positivos, weights=q, minlength=n)
        )
        derivada_score = derivada_score + alpha * derivada_ranking

    valor = (
        (1.0 - alpha) * bce
        + alpha * ranking
        + 0.5 * l2 * np.dot(coef, coef)
    )
    gradiente = np.empty_like(theta)
    # El ranking es invariante a un desplazamiento común del intercepto.
    gradiente[0] = (1.0 - alpha) * derivada_bce.sum()
    gradiente[1:] = X.T @ derivada_score + l2 * coef
    return float(valor), gradiente
