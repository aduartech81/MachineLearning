
"""Grid interno reanudable, sin balanceo ni evaluación externa."""
from pathlib import Path
from importlib.metadata import version
import argparse
import hashlib
import json
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from proyecto_final_caries.experimentos_v2.preprocesamiento import (
    antes_balanceo, despues_balanceo
)

BASE = Path(__file__).resolve().parent
PROYECTO = BASE.parent
RAIZ = PROYECTO.parent

def guardar(path, contenido):
    """Guardar un resultado completo mediante sustitución atómica."""
    if path.exists():
        raise FileExistsError(path)
    temporal = path.with_suffix(path.suffix + ".tmp")
    with temporal.open("w", encoding="utf-8") as f:
        json.dump(contenido, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporal, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("familia")
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()

    fuentes = {
        "espacios": BASE / "espacios_busqueda_v2.json",
        "ejecutor": Path(__file__),
        "preprocesamiento": BASE / "preprocesamiento.py",
        "train": RAIZ / "entregable_2/results/train.csv",
        "summary": RAIZ / "entregable_2/results/summary.json",
        "externos": PROYECTO / "evaluacion/particiones_desarrollo.csv",
        "internos": PROYECTO / "evaluacion/particiones_internas.csv"
    }
    espacios = json.loads(fuentes["espacios"].read_text())["familias"]
    espacio = espacios[args.familia]
    clases = {
        "KNN": KNeighborsClassifier,
        "GaussianNB": GaussianNB,
        "Logistica_L1_L2": LogisticRegression,
        "DecisionTree": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
        "XGBoost": XGBClassifier,
        "SVM": LinearSVC
    }
    out = (BASE / "resultados_grid_sin_balanceo" /
           args.familia / f"fold_{args.fold}")
    out.mkdir(parents=True, exist_ok=True)

    manifiesto = {
        "familia": args.familia,
        "fold": args.fold,
        "balanceo": "Sin_balanceo",
        "huellas": {
            k: hashlib.sha256(p.read_bytes()).hexdigest()
            for k, p in fuentes.items()
        },
        "versiones": {
            p: version(p) for p in
            ["numpy", "pandas", "scipy", "scikit-learn", "xgboost"]
        }
    }
    ruta_manifest = out / "manifiesto.json"
    if ruta_manifest.exists():
        if json.loads(ruta_manifest.read_text()) != manifiesto:
            raise RuntimeError("Cambió código, entorno o datos; no mezclar ejecuciones.")
    else:
        guardar(ruta_manifest, manifiesto)

    train = pd.read_csv(fuentes["train"])
    summary = json.loads(fuentes["summary"].read_text())
    externos = pd.read_csv(fuentes["externos"])
    internos = pd.read_csv(fuentes["internos"])
    X = train[summary["features"]].copy()
    y = train["dental caries"].copy()
    grupos = pd.util.hash_pandas_object(X, index=False)
    assert externos.row_id.is_unique
    assert np.array_equal(grupos.to_numpy(), externos.perfil.to_numpy())
    assert np.array_equal(y.to_numpy(), externos.objetivo.to_numpy())
    X.index = externos.row_id.to_numpy()
    y.index = X.index
    grupos.index = X.index

    sub = internos.loc[internos.outer_fold == args.fold]
    ie = externos.loc[
        externos.outer_validation_fold == args.fold, "row_id"
    ]
    assert sub.row_id.is_unique
    assert set(sub.row_id) == set(X.index) - set(ie)
    assert not set(grupos.loc[sub.row_id]).intersection(grupos.loc[ie])
    assert set(sub.inner_validation_fold) == {0, 1, 2}

    registros = []
    for interno in range(3):
        ia = sub.loc[sub.inner_validation_fold != interno, "row_id"]
        iv = sub.loc[sub.inner_validation_fold == interno, "row_id"]
        assert not set(grupos.loc[ia]).intersection(grupos.loc[iv])
        assert y.loc[ia].nunique() == y.loc[iv].nunique() == 2

        # Transformaciones compartidas sólo entre candidatos del mismo fold.
        t = time.perf_counter()
        antes = antes_balanceo(list(X.columns))
        despues = despues_balanceo(list(X.columns))
        A = despues.fit_transform(antes.fit_transform(X.loc[ia]))
        V = despues.transform(antes.transform(X.loc[iv]))
        segundos_pre = time.perf_counter() - t
        assert np.isfinite(A).all() and np.isfinite(V).all()

        ruta_pre = out / f"preprocesamiento_inner_{interno}.json"
        if not ruta_pre.exists():
            guardar(ruta_pre, {
                "segundos": segundos_pre,
                "filas_ajuste": len(ia), "filas_validacion": len(iv),
                "columnas": A.shape[1],
                "alcance": "Una transformación por fold, compartida por candidatos"
            })

        for candidato, parametros in enumerate(espacio["grid"]):
            ruta = out / f"candidato_{candidato:02d}_inner_{interno}.json"
            if ruta.exists():
                registro = json.loads(ruta.read_text())
                if registro["estado"] != "OK":
                    raise RuntimeError(f"Ajuste pendiente de revisión: {ruta}")
                registros.append(registro)
                print(f"CONSERVADO: candidato {candidato+1}, interno {interno+1}",
                      flush=True)
                continue

            modelo = clases[args.familia](**espacio["fijos"], **parametros)
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always")
                with config_context(working_memory=64):
                    t = time.perf_counter()
                    modelo.fit(A, y.loc[ia])
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
            registro = {
                "candidato": candidato,
                "inner_fold": interno,
                "parametros": parametros,
                "auc": float(roc_auc_score(y.loc[iv], score)),
                "segundos_ajuste": ajuste,
                "segundos_prediccion": prediccion,
                "advertencias": [str(a.message) for a in avisos],
                "estado": "REVISAR_CONVERGENCIA" if fallo else "OK"
            }
            guardar(ruta, registro)
            if fallo:
                raise RuntimeError(f"No convergió; evidencia conservada: {ruta}")
            registros.append(registro)
            print(
                f"{args.familia} | interno {interno+1}/3 | "
                f"candidato {candidato+1}/16 | AUC={registro['auc']:.5f}",
                flush=True
            )

    tabla = pd.DataFrame(registros)
    assert len(tabla) == 48
    resumen = tabla.groupby("candidato").agg(
        auc_medio=("auc", "mean"),
        sd_folds=("auc", "std"),
        segundos_ajuste=("segundos_ajuste", "sum"),
        segundos_prediccion=("segundos_prediccion", "sum")
    ).reset_index()
    mejor = int(resumen.loc[resumen.auc_medio.idxmax(), "candidato"])
    seleccion = {
        "familia": args.familia,
        "outer_fold": args.fold,
        "candidato": mejor,
        "parametros": espacio["grid"][mejor],
        "auc_interno": float(resumen.auc_medio.max()),
        "evaluacion_externa": "NO REALIZADA",
        "nota_sd": "Dispersión entre folds; no es un intervalo de confianza"
    }
    ruta_seleccion = out / "seleccion_interna.json"
    if not ruta_seleccion.exists():
        guardar(ruta_seleccion, seleccion)
    print("\n", resumen.to_string(index=False))
    print("\nSELECCIÓN INTERNA:", json.dumps(seleccion, ensure_ascii=False))
    print("RESULTADOS:", out)
    print("Sin evaluación externa ni modificación de V1.")

if __name__ == "__main__":
    main()
