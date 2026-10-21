
"""Transformaciones aprendidas exclusivamente en cada ajuste interno."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CAT = ["gender", "hearing(left)", "hearing(right)", "Urine protein", "smoking"]

def antes_balanceo(features):
    """Crear imputación y escalado numérico conservando categorías.

    Parameters
    ----------
    features : list of str
        Predictores originales; no debe incluir objetivo ni identificadores.

    Returns
    -------
    ColumnTransformer
        Transformador sin ajustar, con salida DataFrame.
    """
    if not set(CAT).issubset(features):
        raise ValueError("Faltan variables categóricas.")
    if set(features) & {"ID", "oral", "tartar", "dental caries"}:
        raise ValueError("Hay variables excluidas entre los predictores.")
    num = [c for c in features if c not in CAT]
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler())
        ]), num),
        ("cat", SimpleImputer(
            strategy="most_frequent", keep_empty_features=True
        ), CAT)
    ], verbose_feature_names_out=False).set_output(transform="pandas")

def despues_balanceo(features):
    """Crear codificador sin ajustar para la matriz destinada al clasificador.

    Parameters
    ----------
    features : list of str
        Columnas de la representación mixta.

    Returns
    -------
    ColumnTransformer
        Numéricas conservadas y categóricas codificadas one-hot.
        Ajustar solo con el entrenamiento, nunca con validación.
    """
    num = [c for c in features if c not in CAT]
    return ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ), CAT)
    ])
