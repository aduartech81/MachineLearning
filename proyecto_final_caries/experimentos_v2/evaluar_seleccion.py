
"""Evaluación de selecciones existentes; sin nuevas búsquedas."""
from pathlib import Path
from datetime import datetime
from importlib.metadata import version
import argparse
import hashlib
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    brier_score_loss, log_loss, confusion_matrix,
    precision_score, f1_score
)
from imblearn.over_sampling import SMOTE, ADASYN
from proyecto_final_caries.experimentos_v2 import ejecutar_grid as g
from proyecto_final_caries.experimentos_v2.preprocesamiento import (
    antes_balanceo, despues_balanceo
)
from proyecto_final_caries.experimentos_v2.ponderacion import ajustar_ponderado

BASE = Path(__file__).resolve().parent
PROYECTO = BASE.parent
RAIZ = PROYECTO.parent

def leer(p):
    return json.loads(p.read_text(encoding="utf-8"))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def guardar(p, contenido):
    with p.open("x", encoding="utf-8") as f:
        json.dump(contenido, f, ensure_ascii=False, indent=2, allow_nan=False)

def ajustar(X, y, familia, espacio, parametros, balanceo):
    antes = antes_balanceo(list(X.columns))
    despues = despues_balanceo(list(X.columns))
    A = despues.fit_transform(antes.fit_transform(X))
    yy = np.asarray(y)
    if balanceo == "smote":
        A, yy = SMOTE(random_state=42, k_neighbors=5).fit_resample(A, yy)
    elif balanceo == "adasyn":
        A, yy = ADASYN(random_state=42, n_neighbors=5).fit_resample(A, yy)

    clases = {
        "KNN": g.KNeighborsClassifier,
        "GaussianNB": g.GaussianNB,
        "Logistica_L1_L2": g.LogisticRegression,
        "DecisionTree": g.DecisionTreeClassifier,
        "RandomForest": g.RandomForestClassifier,
        "XGBoost": g.XGBClassifier,
        "SVM": g.LinearSVC
    }
    modelo = clases[familia](**espacio["fijos"], **parametros)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        with config_context(working_memory=64):
            if balanceo == "pesos":
                ajustar_ponderado(modelo, A, yy)
            else:
                modelo.fit(A, yy)
    if any(issubclass(a.category, ConvergenceWarning) for a in avisos):
        raise RuntimeError("No convergió; evaluación detenida.")
    assert np.array_equal(modelo.classes_, [0, 1])
    return {
        "antes": antes, "despues": despues, "modelo": modelo,
        "advertencias": [str(a.message) for a in avisos]
    }

def puntuar(bundle, X):
    V = bundle["despues"].transform(bundle["antes"].transform(X))
    m = bundle["modelo"]
    with config_context(working_memory=64):
        prob = m.predict_proba(V)[:, 1] if hasattr(m, "predict_proba") else None
        score = m.decision_function(V) if hasattr(m, "decision_function") else prob
    assert np.isfinite(score).all()
    if prob is not None:
        assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    return np.asarray(score), prob

def metricas(y, prob, corte, nombre):
    pred = (prob >= corte).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        variante=nombre, umbral=float(corte),
        auc=float(roc_auc_score(y, prob)),
        average_precision=float(average_precision_score(y, prob)),
        brier=float(brier_score_loss(y, prob)),
        log_loss=float(log_loss(y, prob, labels=[0, 1])),
        accuracy=float((pred == y).mean()),
        sensibilidad=float(tp / (tp + fn)),
        especificidad=float(tn / (tn + fp)),
        precision=float(precision_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("familia")
    parser.add_argument("--optimizador", choices=["grid","random","bayes","genetica"],
                        required=True)
    parser.add_argument("--balanceo", choices=["sin_balanceo","smote","adasyn","pesos"],
                        required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    if args.familia == "KNN" and args.balanceo == "pesos":
        raise SystemExit("KNN ponderado: no aplicable.")

    origen = (BASE / f"resultados_{args.optimizador}_{args.balanceo}"
              / args.familia / f"fold_{args.fold}")
    seleccion = leer(origen / "seleccion_interna.json")
    manifest_busqueda = leer(origen / "manifiesto.json")
    assert seleccion["familia"] == args.familia
    assert seleccion["outer_fold"] == args.fold

    fuentes = {
        "espacios": BASE / "espacios_busqueda_v2.json",
        "preprocesamiento": BASE / "preprocesamiento.py",
        "train": RAIZ / "entregable_2/results/train.csv",
        "summary": RAIZ / "entregable_2/results/summary.json",
        "externos": PROYECTO / "evaluacion/particiones_desarrollo.csv",
        "internos": PROYECTO / "evaluacion/particiones_internas.csv"
    }
    for k, p in fuentes.items():
        assert sha(p) == manifest_busqueda["huellas"][k], f"Cambió: {k}"
    for paquete, esperado in manifest_busqueda["versiones"].items():
        assert version(paquete) == esperado, f"Cambió versión: {paquete}"

    huellas = {k: sha(p) for k, p in fuentes.items()}
    huellas.update({
        "evaluador": sha(Path(__file__)),
        "auxiliar": sha(Path(g.__file__)),
        "ponderacion": sha(BASE / "ponderacion.py"),
        "seleccion": sha(origen / "seleccion_interna.json"),
        "manifiesto_busqueda": sha(origen / "manifiesto.json")
    })
    dest = (BASE / "evaluacion_v2" / f"{args.optimizador}_{args.balanceo}"
            / args.familia / f"fold_{args.fold}")
    if (dest / "resultado.json").exists():
        anterior = leer(dest / "resultado.json")
        assert anterior["huellas"] == huellas, "Cambió la procedencia."
        for nombre, valor in anterior["artefactos_sha256"].items():
            assert sha(dest / nombre) == valor, f"Cambió artefacto: {nombre}"
        print("YA COMPLETADO Y CONSERVADO:", dest, flush=True)
        return
    if dest.exists():
        raise RuntimeError(f"Destino incompleto; revisar: {dest}")

    train = pd.read_csv(fuentes["train"])
    ext = pd.read_csv(fuentes["externos"])
    inter = pd.read_csv(fuentes["internos"])
    features = leer(fuentes["summary"])["features"]
    espacio = leer(fuentes["espacios"])["familias"][args.familia]
    X = train[features].copy()
    y = train["dental caries"].copy()
    grupos = pd.util.hash_pandas_object(X, index=False)
    assert ext.row_id.is_unique
    assert np.array_equal(grupos.to_numpy(), ext.perfil.to_numpy())
    assert np.array_equal(y.to_numpy(), ext.objetivo.to_numpy())
    X.index = ext.row_id.to_numpy()
    y.index = grupos.index = X.index
    sub = inter.loc[inter.outer_fold == args.fold]
    ie = ext.loc[ext.outer_validation_fold == args.fold, "row_id"]
    assert sub.row_id.is_unique
    assert set(sub.row_id) == set(X.index) - set(ie)
    assert set(sub.inner_validation_fold) == {0, 1, 2}
    assert not set(grupos.loc[sub.row_id]).intersection(grupos.loc[ie])

    trabajo = BASE / "evaluacion_v2_intentos" / (
        f"{args.familia}_{args.optimizador}_{args.balanceo}_fold{args.fold}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    trabajo.mkdir(parents=True)
    inicio = time.perf_counter()
    internos, aucs, avisos = [], [], []

    for k in range(3):
        ia = sub.loc[sub.inner_validation_fold != k, "row_id"]
        iv = sub.loc[sub.inner_validation_fold == k, "row_id"]
        assert not set(grupos.loc[ia]).intersection(grupos.loc[iv])
        print(f"{args.familia} | externo {args.fold+1}/5 | predicciones internas {k+1}/3",
              flush=True)
        bundle = ajustar(X.loc[ia], y.loc[ia], args.familia, espacio,
                         seleccion["parametros"], args.balanceo)
        s, p = puntuar(bundle, X.loc[iv])
        aucs.append(float(roc_auc_score(y.loc[iv], p if p is not None else s)))
        avisos.extend(bundle["advertencias"])
        internos.append(pd.DataFrame({
            "row_id": iv.to_numpy(), "perfil": grupos.loc[iv].to_numpy(),
            "actual": y.loc[iv].to_numpy(), "inner_fold": k, "score": s
        }))
        del bundle

    assert abs(np.mean(aucs) - seleccion["auc_interno"]) <= 1e-8, (
        "AUC interno no reproducido; revisar antes de evaluar."
    )
    oof = pd.concat(internos, ignore_index=True).sort_values("row_id")
    assert oof.row_id.is_unique and set(oof.row_id) == set(sub.row_id)
    assert oof.groupby("perfil").inner_fold.nunique().max() == 1
    oof.to_csv(trabajo / "predicciones_internas.csv", index=False)
    z = oof.score.to_numpy().reshape(-1, 1)
    yi = oof.actual.to_numpy()

    sig = make_pipeline(StandardScaler(), LogisticRegression(
        C=1e6, solver="lbfgs", max_iter=2000, tol=1e-8, random_state=42))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sig.fit(z, yi)
    if any(issubclass(a.category, ConvergenceWarning) for a in w):
        raise RuntimeError("Calibrador no convergió.")
    avisos.extend(str(a.message) for a in w)
    if sig.steps[-1][1].coef_[0, 0] <= 0:
        raise RuntimeError("Pendiente del calibrador no positiva; revisar.")
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip",
                             y_min=0, y_max=1).fit(z.ravel(), yi)
    pi = sig.predict_proba(z)[:, 1]
    fpr, tpr, cortes = roc_curve(yi, pi, drop_intermediate=False)
    validos = np.flatnonzero(np.isfinite(cortes))
    j = tpr - fpr
    empatados = validos[np.isclose(j[validos], j[validos].max(),
                                  rtol=0, atol=1e-12)]
    umbral = float(cortes[empatados[np.argmax(cortes[empatados])]])
    guardar(trabajo / "decision_interna.json", {
        "principal": "Sigmoide logística univariada",
        "secundario": "Isotónica creciente",
        "entrada": "decision_function si disponible; si no, predict_proba clase 1",
        "umbral": umbral, "regla": "Youden interno; empate: mayor umbral",
        "auc_interno_reproducido": float(np.mean(aucs)),
        "alcance": "Desarrollo; no evaluación independiente del calibrador."
    })

    print("AJUSTE FINAL Y EVALUACIÓN DEL FOLD EXTERNO", flush=True)
    t = time.perf_counter()
    bundle = ajustar(X.loc[sub.row_id], y.loc[sub.row_id], args.familia,
                     espacio, seleccion["parametros"], args.balanceo)
    tiempo_final = time.perf_counter() - t
    avisos.extend(bundle["advertencias"])
    t = time.perf_counter()
    s, p = puntuar(bundle, X.loc[ie])
    ps = sig.predict_proba(s.reshape(-1, 1))[:, 1]
    pis = iso.predict(s)
    tiempo_pred = time.perf_counter() - t
    yt = y.loc[ie].to_numpy()

    filas = [
        metricas(yt, ps, 0.5, "Sigmoide_0.5"),
        metricas(yt, ps, umbral, "Sigmoide_Youden_interno"),
        metricas(yt, pis, 0.5, "Isotonica_0.5_secundaria")
    ]
    if p is not None:
        filas.insert(0, metricas(yt, p, 0.5, "Original_0.5"))

    predicciones = pd.DataFrame({
        "row_id": ie.to_numpy(), "perfil": grupos.loc[ie].to_numpy(),
        "actual": yt, "outer_fold": args.fold, "score": s,
        "probability_sigmoide": ps, "probability_isotonica": pis,
        "prediction_sigmoide_youden": (ps >= umbral).astype(int),
        "dummy_probability": float(y.loc[sub.row_id].mean())
    })
    if p is not None:
        predicciones["probability_original"] = p
    else:
        predicciones["prediction_original"] = (s > 0).astype(int)
    predicciones.to_csv(trabajo / "predicciones.csv", index=False)
    pd.DataFrame(filas).to_csv(trabajo / "metricas.csv", index=False)
    joblib.dump({
        **bundle, "sigmoide": sig, "isotonica": iso,
        "umbral_sigmoide": umbral, "features": features
    }, trabajo / "modelo_completo.joblib", compress=3)

    resultado = {
        "familia": args.familia, "optimizador": args.optimizador,
        "balanceo": args.balanceo, "outer_fold": args.fold,
        "parametros": seleccion["parametros"],
        "filas_ajuste": len(sub), "filas_externas": len(ie),
        "auc_score_original": float(roc_auc_score(yt, s)),
        "segundos_ajuste_final_con_preprocesamiento": tiempo_final,
        "segundos_prediccion_con_calibracion": tiempo_pred,
        "segundos_totales": time.perf_counter() - inicio,
        "huellas": huellas, "versiones": manifest_busqueda["versiones"],
        "advertencias": avisos,
        "alcance": "Evaluación anidada interna; V2 exploratoria, no validación clínica externa.",
        "artefactos_sha256": {
            p.name: sha(p) for p in trabajo.iterdir() if p.is_file()
        }
    }
    guardar(trabajo / "resultado.json", resultado)
    dest.parent.mkdir(parents=True, exist_ok=True)
    trabajo.rename(dest)
    print(pd.DataFrame(filas)[[
        "variante","auc","brier","sensibilidad","especificidad","f1"
    ]].to_string(index=False))
    print("FOLD COMPLETADO:", dest, flush=True)

if __name__ == "__main__":
    main()
