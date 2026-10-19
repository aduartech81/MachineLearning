"""Evaluación anidada de una familia; ejecución secuencial."""
from pathlib import Path
import hashlib
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from ml_common import preprocessor
from proyecto_final_caries.modelo import ModeloCaries, CAT

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "entregable_2/results"


def construir(nombre, parametros, X):
    if nombre == "Logistica":
        est = LogisticRegression(max_iter=2000, **parametros)
    elif nombre == "SVM_lineal":
        est = LinearSVC(
            loss="squared_hinge", dual=False, max_iter=3000,
            random_state=42, **parametros
        )
    elif nombre == "Boosting":
        est = HistGradientBoostingClassifier(
            early_stopping=False, learning_rate=0.1,
            l2_regularization=1.0, random_state=42, **parametros
        )
    else:
        return ModeloCaries(**parametros)
    return Pipeline([
        ("preprocess", preprocessor(X, CAT)),
        ("model", est)
    ])


def ajustar(modelo, X, y):
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo.fit(X, y)
    if any(issubclass(a.category, ConvergenceWarning) for a in avisos):
        raise RuntimeError("No convergió; detener y revisar sin cambiar la grilla.")
    if hasattr(modelo, "converged_") and not modelo.converged_:
        raise RuntimeError("El modelo propio no convergió.")
    return [str(a.message) for a in avisos]


def puntuar(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def main():
    nombre = sys.argv[1]
    plan = json.loads((ROOT / "plan_comparacion_v1.json").read_text())
    grilla = plan["grillas"][nombre]
    summary = json.loads((BASE / "summary.json").read_text())
    train = pd.read_csv(BASE / "train.csv")
    externos = pd.read_csv(ROOT / "evaluacion/particiones_desarrollo.csv")
    internos = pd.read_csv(ROOT / "evaluacion/particiones_internas.csv")
    X = train[summary["features"]].copy()
    y = train["dental caries"].copy()
    grupos = pd.util.hash_pandas_object(X, index=False)
    assert len(X) == len(externos)
    assert np.array_equal(grupos.to_numpy(), externos.perfil.to_numpy())
    assert np.array_equal(y.to_numpy(), externos.objetivo.to_numpy())
    X.index = externos.row_id.to_numpy()
    y.index = X.index
    grupos.index = X.index

    out = ROOT / "resultados/comparacion_v1" / nombre
    out.mkdir(parents=True, exist_ok=True)
    entradas = [
        ROOT / "plan_comparacion_v1.json", ROOT / "evaluar.py",
        ROOT / "modelo.py", ROOT / "objetivo.py",
        ROOT.parent / "ml_common.py", BASE / "train.csv",
        ROOT / "evaluacion/particiones_desarrollo.csv",
        ROOT / "evaluacion/particiones_internas.csv"
    ]
    huellas = {
        str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in entradas
    }
    manifest = out / "huellas.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == huellas, "Cambió código o datos."
    else:
        manifest.write_text(json.dumps(huellas, indent=2))

    for fold in range(5):
        fd = out / f"fold_{fold}"
        fd.mkdir(exist_ok=True)
        if (fd / "resultado.json").exists():
            print("YA COMPLETADO:", nombre, fold, flush=True)
            continue
        inicio = time.perf_counter()
        ids_ext = externos.loc[externos.outer_validation_fold == fold, "row_id"]
        sub = internos.loc[internos.outer_fold == fold]
        assert sub.row_id.is_unique
        assert set(sub.row_id) == set(X.index) - set(ids_ext)
        assert not set(grupos.loc[sub.row_id]).intersection(grupos.loc[ids_ext])
        registros = []
        medias = []

        for candidato, parametros in enumerate(grilla):
            aucs = []
            for interno in range(3):
                ia = sub.loc[sub.inner_validation_fold != interno, "row_id"]
                iv = sub.loc[sub.inner_validation_fold == interno, "row_id"]
                assert not set(grupos.loc[ia]).intersection(grupos.loc[iv])
                modelo = construir(nombre, parametros, X.loc[ia])
                t = time.perf_counter()
                avisos = ajustar(modelo, X.loc[ia], y.loc[ia])
                auc = float(roc_auc_score(y.loc[iv], puntuar(modelo, X.loc[iv])))
                aucs.append(auc)
                registros.append({
                    "candidato": candidato, "inner_fold": interno,
                    "parametros": json.dumps(parametros),
                    "auc": auc, "segundos_ajuste_y_prediccion": time.perf_counter() - t,
                    "advertencias": json.dumps(avisos)
                })
                pd.DataFrame(registros).to_csv(fd / "busqueda.csv", index=False)
            medias.append(float(np.mean(aucs)))
            print(
                f"{nombre} | externo {fold+1}/5 | candidato {candidato+1}/4 "
                f"| AUC interno={medias[-1]:.5f}", flush=True
            )

        mejor = int(np.argmax(medias))
        modelo = construir(nombre, grilla[mejor], X.loc[sub.row_id])
        avisos = ajustar(modelo, X.loc[sub.row_id], y.loc[sub.row_id])
        score = puntuar(modelo, X.loc[ids_ext])
        prob = score if hasattr(modelo, "predict_proba") else np.full(len(score), np.nan)
        pred = modelo.predict(X.loc[ids_ext])
        assert np.isfinite(score).all()
        pd.DataFrame({
            "row_id": ids_ext.to_numpy(),
            "actual": y.loc[ids_ext].to_numpy(),
            "score": score, "probability": prob, "prediction": pred,
            "dummy_probability": float(y.loc[sub.row_id].mean())
        }).to_csv(fd / "predicciones.csv", index=False)
        joblib.dump(modelo, fd / "modelo.joblib", compress=3)
        resultado = {
            "familia": nombre, "outer_fold": fold,
            "parametros": grilla[mejor], "auc_interno": medias[mejor],
            "auc_externo": float(roc_auc_score(y.loc[ids_ext], score)),
            "filas_externas": len(ids_ext),
            "segundos_totales": time.perf_counter() - inicio,
            "advertencias": avisos
        }
        (fd / "resultado.json").write_text(json.dumps(resultado, indent=2))
        print("FOLD GUARDADO:", fold+1, flush=True)

    print("FAMILIA COMPLETADA:", nombre)
    print("RESULTADOS:", out)


if __name__ == "__main__":
    main()
