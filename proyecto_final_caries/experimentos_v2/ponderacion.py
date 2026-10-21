
"""Ponderación calculada exclusivamente con las etiquetas de cada ajuste."""
import inspect
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

def ajustar_ponderado(modelo, X, y):
    clases = np.unique(y)
    if set(clases) != {0, 1}:
        raise ValueError("Se requieren ambas clases binarias.")
    parametros = modelo.get_params()
    if "class_weight" in parametros:
        # No añadir sample_weight: evitar ponderar dos veces.
        modelo.set_params(class_weight="balanced")
        return modelo.fit(X, y)

    nombre = type(modelo).__name__
    if nombre not in {"GaussianNB", "XGBClassifier"}:
        raise TypeError(f"Ponderación no implementada para {nombre}.")
    if "sample_weight" not in inspect.signature(modelo.fit).parameters:
        raise TypeError(f"{nombre}.fit no admite sample_weight.")

    if nombre == "XGBClassifier":
        valor = parametros.get("scale_pos_weight")
        if valor not in (None, 1, 1.0):
            raise ValueError("No combinar scale_pos_weight con estos pesos.")

    pesos = compute_sample_weight(class_weight="balanced", y=y)
    assert np.isfinite(pesos).all() and (pesos > 0).all()
    assert np.isclose(pesos.sum(), len(y))
    return modelo.fit(X, y, sample_weight=pesos)
