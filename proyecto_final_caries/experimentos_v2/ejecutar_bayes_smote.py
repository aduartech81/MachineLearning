
"""Búsqueda TPE interna, secuencial y reanudable; con SMOTE dentro del entrenamiento interno."""
from pathlib import Path
from importlib.metadata import version
import argparse
import hashlib
import json
import time
import warnings
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import optuna
from sklearn import config_context
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from proyecto_final_caries.experimentos_v2 import ejecutar_grid as g

BASE = Path(__file__).resolve().parent
PROYECTO = BASE.parent
RAIZ = PROYECTO.parent

def verificar_o_guardar(path, contenido):
    if path.exists():
        if json.loads(path.read_text()) != contenido:
            raise RuntimeError(f"Cambió la configuración o la propuesta: {path}")
    else:
        g.guardar(path, contenido)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("familia")
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    fuentes = {
        "ejecutor": Path(__file__),
        "auxiliar": Path(g.__file__),
        "preprocesamiento": BASE / "preprocesamiento.py",
        "espacios": BASE / "espacios_busqueda_v2.json",
        "train": RAIZ / "entregable_2/results/train.csv",
        "summary": RAIZ / "entregable_2/results/summary.json",
        "externos": PROYECTO / "evaluacion/particiones_desarrollo.csv",
        "internos": PROYECTO / "evaluacion/particiones_internas.csv"
    }
    espacio = json.loads(fuentes["espacios"].read_text())["familias"][args.familia]
    clases = {
        "KNN": g.KNeighborsClassifier,
        "GaussianNB": g.GaussianNB,
        "Logistica_L1_L2": g.LogisticRegression,
        "DecisionTree": g.DecisionTreeClassifier,
        "RandomForest": g.RandomForestClassifier,
        "XGBoost": g.XGBClassifier,
        "SVM": g.LinearSVC
    }
    out = BASE / "resultados_bayes_smote" / args.familia / f"fold_{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    manifiesto = {
        "familia": args.familia, "outer_fold": args.fold,
        "optimizador": "TPE", "balanceo": "SMOTE", "semilla_balanceo": 42, "balanceo_espacio": "Numericas estandarizadas y categorias one-hot", "limitacion_balanceo": "Puede producir categorias fraccionarias",
        "semilla": 42 + args.fold, "propuestas_maximas": 16,
        "n_startup_trials": 4, "n_ei_candidates": 24,
        "multivariate": False, "constant_liar": False,
        "duplicados": "Consumen una propuesta; reutilizan resultado sin ajustar",
        "huellas": {
            k: hashlib.sha256(p.read_bytes()).hexdigest()
            for k, p in fuentes.items()
        },
        "versiones": {
            p: version(p) for p in
            ["imbalanced-learn", "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "optuna"]
        }
    }
    verificar_o_guardar(out / "manifiesto.json", manifiesto)

    train = pd.read_csv(fuentes["train"])
    features = json.loads(fuentes["summary"].read_text())["features"]
    externos = pd.read_csv(fuentes["externos"])
    internos = pd.read_csv(fuentes["internos"])
    X = train[features].copy()
    y = train["dental caries"].copy()
    grupos = pd.util.hash_pandas_object(X, index=False)
    assert externos.row_id.is_unique
    assert np.array_equal(grupos.to_numpy(), externos.perfil.to_numpy())
    assert np.array_equal(y.to_numpy(), externos.objetivo.to_numpy())
    X.index = externos.row_id.to_numpy()
    y.index = X.index
    grupos.index = X.index
    sub = internos.loc[internos.outer_fold == args.fold]
    ie = externos.loc[externos.outer_validation_fold == args.fold, "row_id"]
    assert sub.row_id.is_unique
    assert set(sub.row_id) == set(X.index) - set(ie)
    assert not set(grupos.loc[sub.row_id]).intersection(grupos.loc[ie])
    assert set(sub.inner_validation_fold) == {0, 1, 2}

    matrices = []
    for interno in range(3):
        ia = sub.loc[sub.inner_validation_fold != interno, "row_id"]
        iv = sub.loc[sub.inner_validation_fold == interno, "row_id"]
        assert not set(grupos.loc[ia]).intersection(grupos.loc[iv])
        assert y.loc[ia].nunique() == y.loc[iv].nunique() == 2
        t = time.perf_counter()
        antes = g.antes_balanceo(features)
        despues = g.despues_balanceo(features)
        A = despues.fit_transform(antes.fit_transform(X.loc[ia]))
        V = despues.transform(antes.transform(X.loc[iv]))
        y_balanceado = y.loc[ia].to_numpy().copy()
        balanceador = SMOTE(
            random_state=42, k_neighbors=5
        )
        A, y_balanceado = balanceador.fit_resample(A, y_balanceado)
        assert len(A) == len(y_balanceado)
        assert set(np.unique(y_balanceado)) == {0, 1}
        segundos = time.perf_counter() - t
        assert np.isfinite(A).all() and np.isfinite(V).all()
        matrices.append((A, V, y_balanceado, y.loc[iv]))
        p = out / f"preprocesamiento_inner_{interno}.json"
        if not p.exists():
            g.guardar(p, {
                "segundos": segundos, "filas_ajuste": len(ia), "filas_tras_balanceo": len(A), "clases_tras_balanceo": {str(int(c)): int(n) for c, n in zip(  *np.unique(y_balanceado, return_counts=True))},
                "filas_validacion": len(iv), "columnas": A.shape[1],
                "alcance": "Transformaciones compartidas sólo dentro del mismo fold"
            })

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42 + args.fold, n_startup_trials=4,
            n_ei_candidates=24, multivariate=False, constant_liar=False
        )
    )
    vistos = {}
    resultados = []
    for numero in range(16):
        # Reproducir ask/tell permite recuperar también la secuencia del muestreador.
        trial = estudio.ask()
        t = time.perf_counter()
        parametros = {}
        for nombre, dominio in espacio["dominios"].items():
            if dominio["tipo"] == "categorico":
                parametros[nombre] = trial.suggest_categorical(nombre, dominio["valores"])
            elif dominio["tipo"] == "real_log":
                parametros[nombre] = trial.suggest_float(
                    nombre, dominio["min"], dominio["max"], log=True
                )
            else:
                raise ValueError(dominio)
        segundos_propuesta = time.perf_counter() - t
        verificar_o_guardar(
            out / f"propuesta_{numero:02d}.json",
            {"numero": numero, "parametros": parametros}
        )
        clave = json.dumps(parametros, sort_keys=True)
        ruta = out / f"trial_{numero:02d}.json"

        if ruta.exists():
            registro = json.loads(ruta.read_text())
            assert registro["parametros"] == parametros
        elif clave in vistos:
            anterior = vistos[clave]
            registro = {
                "trial": numero, "parametros": parametros,
                "auc_medio": anterior["auc_medio"],
                "reutilizado_de": anterior["trial"],
                "segundos_propuesta": segundos_propuesta
            }
            g.guardar(ruta, registro)
        else:
            aucs = []
            for interno, (A, V, ya, yv) in enumerate(matrices):
                p = out / f"trial_{numero:02d}_inner_{interno}.json"
                if p.exists():
                    r = json.loads(p.read_text())
                    assert r["parametros"] == parametros
                else:
                    modelo = clases[args.familia](**espacio["fijos"], **parametros)
                    with warnings.catch_warnings(record=True) as avisos:
                        warnings.simplefilter("always")
                        with config_context(working_memory=64):
                            t = time.perf_counter()
                            modelo.fit(A, ya)
                            ajuste = time.perf_counter() - t
                            t = time.perf_counter()
                            score = (
                                modelo.predict_proba(V)[:, 1]
                                if hasattr(modelo, "predict_proba")
                                else modelo.decision_function(V)
                            )
                            prediccion = time.perf_counter() - t
                    assert np.isfinite(score).all()
                    fallo = any(
                        issubclass(a.category, ConvergenceWarning) for a in avisos
                    )
                    r = {
                        "parametros": parametros,
                        "auc": float(roc_auc_score(yv, score)),
                        "segundos_ajuste": ajuste,
                        "segundos_prediccion": prediccion,
                        "advertencias": [str(a.message) for a in avisos],
                        "estado": "REVISAR_CONVERGENCIA" if fallo else "OK"
                    }
                    g.guardar(p, r)
                if r["estado"] != "OK":
                    raise RuntimeError(f"Revisar convergencia: {p}")
                aucs.append(r["auc"])
            registro = {
                "trial": numero, "parametros": parametros,
                "auc_medio": float(np.mean(aucs)),
                "auc_por_fold": aucs, "reutilizado_de": None,
                "segundos_propuesta": segundos_propuesta
            }
            g.guardar(ruta, registro)

        vistos.setdefault(clave, registro)
        resultados.append(registro)
        estudio.tell(trial, registro["auc_medio"])
        print(
            f"TPE {args.familia} | externo {args.fold+1}/5 "
            f"| propuesta {numero+1}/16 | AUC interno={registro['auc_medio']:.6f}",
            flush=True
        )

    mejor = max(resultados, key=lambda r: r["auc_medio"])
    seleccion = {
        "familia": args.familia, "outer_fold": args.fold,
        "optimizador": "TPE", "trial": mejor["trial"],
        "parametros": mejor["parametros"], "auc_interno": mejor["auc_medio"],
        "propuestas": 16, "configuraciones_unicas": len(vistos),
        "evaluacion_externa": "NO REALIZADA"
    }
    verificar_o_guardar(out / "seleccion_interna.json", seleccion)
    print("\nSELECCIÓN INTERNA:", json.dumps(seleccion, ensure_ascii=False))
    print("RESULTADOS:", out)
    print("Sin evaluación externa ni modificación de resultados anteriores.")

if __name__ == "__main__":
    main()
