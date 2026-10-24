"""Modelo aditivo con ranking e interacciones numéricas limitadas."""
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted
from proyecto_final_caries.objetivo import generar_pares, perdida_gradiente

CAT = ["gender", "smoking", "hearing(left)", "hearing(right)", "Urine protein"]


class ModeloCaries(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=0.25, l2=0.01, n_knots=5,
                 max_interactions=5, stability_repeats=5,
                 stability_threshold=0.6, n_pairs=20000,
                 max_iter=300, tol=1e-6, random_state=42):
        self.alpha = alpha
        self.l2 = l2
        self.n_knots = n_knots
        self.max_interactions = max_interactions
        self.stability_repeats = stability_repeats
        self.stability_threshold = stability_threshold
        self.n_pairs = n_pairs
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _optimizar(self, Z, y, alpha, pares):
        inicial = np.zeros(Z.shape[1] + 1)
        prevalencia = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        inicial[0] = np.log(prevalencia / (1 - prevalencia))
        resultado = minimize(
            perdida_gradiente, inicial,
            args=(Z, y, pares[0], pares[1], alpha, self.l2),
            method="L-BFGS-B", jac=True,
            options={"maxiter": self.max_iter,
                     "gtol": self.tol, "ftol": 1e-10}
        )
        registro = {
            "success": bool(resultado.success),
            "iterations": int(resultado.nit),
            "message": str(resultado.message),
            "loss": float(resultado.fun),
            "gradient_max": float(np.max(np.abs(resultado.jac)))
        }
        if not resultado.success:
            warnings.warn(str(registro), ConvergenceWarning)
        return resultado.x, registro

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un DataFrame con nombres de variables.")
        if set(X.columns) & {"ID", "oral", "tartar", "dental caries"}:
            raise ValueError("X contiene variables excluidas u objetivo.")
        if X.columns.duplicated().any() or not set(CAT).issubset(X.columns):
            raise ValueError("Esquema de predictores incorrecto.")
        if not 0 <= self.alpha < 1 or self.l2 <= 0:
            raise ValueError("Se requiere 0 <= alpha < 1 y l2 > 0.")
        if self.max_interactions < 0 or self.stability_repeats < 1:
            raise ValueError("Configuración de interacciones incorrecta.")

        y = np.asarray(y, dtype=float)
        if len(y) != len(X) or set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError("Se necesitan ambas clases 0/1 y longitudes iguales.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.numeric_ = [c for c in X.columns if c not in CAT]
        self.classes_ = np.array([0, 1])
        self.n_fit_rows_ = len(X)

        numerico = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("splines", SplineTransformer(
                n_knots=self.n_knots, degree=3, knots="quantile",
                include_bias=False, extrapolation="constant"))
        ])
        categorico = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        self.preprocessor_ = ColumnTransformer([
            ("numeric", numerico, self.numeric_),
            ("categorical", categorico, CAT)
        ])
        A = self.preprocessor_.fit_transform(X)
        self.main_scaler_ = StandardScaler().fit(A)
        A = self.main_scaler_.transform(A)
        if not np.isfinite(A).all():
            raise ValueError("Transformaciones no finitas.")

        pares = generar_pares(y, self.n_pairs, self.random_state)
        self.selected_pairs_ = []
        self.interaction_selection_ = []
        self.preliminary_optimizer_ = None

        if self.max_interactions:
            preliminar, self.preliminary_optimizer_ = self._optimizar(
                A, y, 0.0, pares
            )
            residuo = y - expit(A @ preliminar[1:] + preliminar[0])
            self.product_preprocessor_ = Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler())
            ])
            N = self.product_preprocessor_.fit_transform(X[self.numeric_])
            candidatos = list(combinations(range(N.shape[1]), 2))
            P = np.column_stack([N[:, i] * N[:, j] for i, j in candidatos])

            grupos = pd.util.hash_pandas_object(X, index=False).to_numpy()
            unicos = np.unique(grupos)
            rng = np.random.default_rng(self.random_state)
            frecuencias = np.zeros(len(candidatos))
            fortalezas = np.zeros(len(candidatos))

            for _ in range(self.stability_repeats):
                elegidos = rng.choice(
                    unicos, size=max(1, int(0.7 * len(unicos))), replace=False
                )
                mascara = np.isin(grupos, elegidos)
                Pc = P[mascara] - P[mascara].mean(axis=0)
                rc = residuo[mascara] - residuo[mascara].mean()
                denominador = np.sqrt((Pc * Pc).sum(axis=0) * np.dot(rc, rc))
                asociacion = np.divide(
                    np.abs(Pc.T @ rc), denominador,
                    out=np.zeros(len(candidatos)), where=denominador > 1e-12
                )
                orden = np.argsort(-asociacion, kind="stable")
                mejores = orden[:min(10, len(orden))]
                mejores = mejores[asociacion[mejores] > 1e-12]
                frecuencias[mejores] += 1
                fortalezas += asociacion

            frecuencias /= self.stability_repeats
            fortalezas /= self.stability_repeats
            orden = sorted(
                range(len(candidatos)),
                key=lambda k: (-frecuencias[k], -fortalezas[k], k)
            )
            seleccion = [
                k for k in orden
                if frecuencias[k] >= self.stability_threshold
                and fortalezas[k] > 1e-12
            ][:self.max_interactions]
            self.selected_pairs_ = [candidatos[k] for k in seleccion]
            self.interaction_selection_ = [
                {"variable_1": self.numeric_[candidatos[k][0]],
                 "variable_2": self.numeric_[candidatos[k][1]],
                 "frequency": float(frecuencias[k]),
                 "residual_association": float(fortalezas[k]),
                 "selected": k in seleccion}
                for k in orden
            ]
            if seleccion:
                self.interaction_scaler_ = StandardScaler().fit(P[:, seleccion])
                A = np.column_stack([
                    A, self.interaction_scaler_.transform(P[:, seleccion])
                ])

        theta, self.optimizer_ = self._optimizar(A, y, self.alpha, pares)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        self.converged_ = self.optimizer_["success"] and (
            self.preliminary_optimizer_ is None
            or self.preliminary_optimizer_["success"]
        )
        return self

    def _transform(self, X):
        check_is_fitted(self, "coef_")
        X = X.loc[:, self.feature_names_in_]
        A = self.main_scaler_.transform(self.preprocessor_.transform(X))
        if self.selected_pairs_:
            N = self.product_preprocessor_.transform(X[self.numeric_])
            P = np.column_stack([
                N[:, i] * N[:, j] for i, j in self.selected_pairs_
            ])
            A = np.column_stack([A, self.interaction_scaler_.transform(P)])
        return A

    def decision_function(self, X):
        return self._transform(X) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        # Probabilidades sin calibración adicional; deben evaluarse.
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)
