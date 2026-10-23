"""Benchmark nuevo, aislado por proceso; nunca modifica resultados V1/V2.

Compara parámetros por defecto y selección interna V2 con idénticos datos,
preprocesamiento y balanceo por pareja. No incluye el coste de búsqueda ni
calibración. RSS es memoria total del proceso, no tamaño exclusivo del modelo.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DECISIONES = ROOT / 'proyecto_final_caries/presentacion/evidencias/revision_v2_20261020_154421_004423/comparacion/decisiones_solo_validacion_interna.csv'


def huella(path):
    """Devolver SHA-256 de un archivo de entrada."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guardar(path, value):
    """Crear JSON sin sobrescribir resultados existentes."""
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def worker(familia, fold, variante, salida):
    """Ajustar un modelo y evaluar exclusivamente el fold externo reservado."""
    import resource
    import warnings
    import numpy as np
    import pandas as pd
    import psutil
    from sklearn import config_context
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from threadpoolctl import threadpool_limits
    from imblearn.over_sampling import SMOTE, ADASYN
    from proyecto_final_caries.experimentos_v2 import ejecutar_grid as g
    from proyecto_final_caries.experimentos_v2.preprocesamiento import antes_balanceo, despues_balanceo
    from proyecto_final_caries.experimentos_v2.ponderacion import ajustar_ponderado

    clases = dict(KNN=g.KNeighborsClassifier, GaussianNB=g.GaussianNB,
                  Logistica_L1_L2=g.LogisticRegression, DecisionTree=g.DecisionTreeClassifier,
                  RandomForest=g.RandomForestClassifier, XGBoost=g.XGBClassifier, SVM=g.LinearSVC)
    train = pd.read_csv(ROOT / 'entregable_2/results/train.csv')
    ext = pd.read_csv(ROOT / 'proyecto_final_caries/evaluacion/particiones_desarrollo.csv')
    features = json.loads((ROOT / 'entregable_2/results/summary.json').read_text())['features']
    X, y = train[features].copy(), train['dental caries'].copy()
    assert ext.row_id.is_unique and len(ext) == len(X)
    assert np.array_equal(ext.perfil.to_numpy(), pd.util.hash_pandas_object(X, index=False).to_numpy())
    assert np.array_equal(ext.objetivo.to_numpy(), y.to_numpy())
    val = ext.outer_validation_fold == fold
    assert not set(ext.loc[val, 'perfil']) & set(ext.loc[~val, 'perfil'])
    X.index = y.index = ext.row_id.to_numpy()
    internos = pd.read_csv(ROOT / 'proyecto_final_caries/evaluacion/particiones_internas.csv')
    ia = internos.loc[internos.outer_fold == fold, 'row_id']
    iv = ext.loc[val, 'row_id']
    assert ia.is_unique and set(ia) == set(ext.loc[~val, 'row_id'])
    seleccion = pd.read_csv(DECISIONES).query('familia == @familia and fold == @fold')
    assert len(seleccion) == 1
    row = seleccion.iloc[0]
    selected_path = ROOT / row.seleccion_origen
    assert huella(selected_path) == row.seleccion_sha256
    espacio = json.loads((ROOT / 'proyecto_final_caries/experimentos_v2/espacios_busqueda_v2.json').read_text())['familias'][familia]
    if variante == 'optimizado':
        parametros = dict(espacio['fijos'], **json.loads(row.parametros))
    else:
        parametros = {}
        default = clases[familia]().get_params()
        if 'random_state' in default:
            parametros['random_state'] = 42
        if 'n_jobs' in default:
            parametros['n_jobs'] = 1
    model = clases[familia](**parametros)
    # Misma restricción de CPU en ambas variantes; el resto queda registrado.
    if 'n_jobs' in model.get_params():
        model.set_params(n_jobs=1)
    antes, despues = antes_balanceo(features), despues_balanceo(features)
    rss_base = psutil.Process().memory_info().rss
    t0 = time.perf_counter()
    with threadpool_limits(limits=1), config_context(working_memory=64), warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        A = despues.fit_transform(antes.fit_transform(X.loc[ia]))
        yy = y.loc[ia].to_numpy()
        if row.balanceo == 'smote':
            A, yy = SMOTE(random_state=42, k_neighbors=5).fit_resample(A, yy)
        elif row.balanceo == 'adasyn':
            A, yy = ADASYN(random_state=42, n_neighbors=5).fit_resample(A, yy)
        tpre = time.perf_counter() - t0
        tf = time.perf_counter()
        if row.balanceo == 'pesos':
            ajustar_ponderado(model, A, yy)
        else:
            model.fit(A, yy)
        tfit = time.perf_counter() - tf
        ti = time.perf_counter()
        V = despues.transform(antes.transform(X.loc[iv]))
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(V)
            pred = (scores >= 0).astype(int)
        else:
            scores = model.predict_proba(V)[:, 1]
            pred = (scores >= .5).astype(int)
        tinf = time.perf_counter() - ti
    assert np.isfinite(scores).all()
    # getrusage usa bytes en macOS y KiB en Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == 'darwin' else peak * 1024)
    result = dict(familia=familia, fold=fold, variante=variante, balanceo=row.balanceo,
                  optimizador_seleccionado=row.optimizador, seleccion_sha256=row.seleccion_sha256,
                  n_train=int((~val).sum()), n_val=int(val.sum()), n_tras_balanceo=len(yy),
                  p_transformadas=A.shape[1], preprocesamiento_s=tpre, ajuste_s=tfit,
                  entrenamiento_total_s=tpre+tfit, inferencia_total_s=tinf,
                  rss_base_bytes=rss_base, rss_max_proceso_bytes=peak_bytes,
                  auc=float(roc_auc_score(y.loc[iv], scores)),
                  accuracy=float(accuracy_score(y.loc[iv], pred)), f1=float(f1_score(y.loc[iv], pred)),
                  convergio=not any(issubclass(a.category, ConvergenceWarning) for a in avisos),
                  advertencias=[str(a.message) for a in avisos],
                  parametros_efectivos=model.get_params(),
                  fuente_train_sha256=huella(ROOT / 'entregable_2/results/train.csv'))
    # Valores None son válidos; versiones XGBoost contienen NaN en missing.
    for k, v in result['parametros_efectivos'].items():
        if isinstance(v, float) and not np.isfinite(v):
            result['parametros_efectivos'][k] = str(v)
    guardar(salida, result)


def main():
    """Ejecutar parejas en procesos frescos con orden alternado y tres repeticiones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, required=True)
    parser.add_argument('--repeticiones', type=int, default=3)
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--familias', nargs='+')
    parser.add_argument('--worker', nargs=3, metavar=('FAMILIA', 'FOLD', 'VARIANTE'))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), args.worker[2], args.salida)
        return
    import importlib.metadata as metadata
    import pandas as pd
    args.salida.mkdir(parents=True, exist_ok=False)
    familias = args.familias or list(pd.read_csv(DECISIONES).familia.unique())
    guardar(args.salida / 'protocolo.json', dict(
        alcance='Benchmark nuevo: parámetros por defecto vs seleccionados por validación interna; mismo balanceo por pareja. Sin búsqueda ni calibración.',
        limitaciones=['RSS máximo del proceso completo incluye importaciones y datos; no es RAM histórica ni memoria exclusiva del modelo.',
                      'El benchmark no demuestra por sí solo optimización algorítmica/computacional de la sección 4.',
                      'No usa ni reevalúa el test histórico. Los folds se solapan en entrenamiento.',
                      'Accuracy/F1 usan corte fijo 0 o 0.5; no son las métricas del calibrador/Youden publicado.'],
        familias=familias, folds=args.folds, repeticiones=args.repeticiones,
        cpu_limite=1, plataforma=platform.platform(), python=sys.version,
        versiones={x: metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','imbalanced-learn','xgboost','psutil','threadpoolctl']},
        script_sha256=huella(__file__), decisiones_sha256=huella(DECISIONES)))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for fold in args.folds:
        for familia in familias:
            for rep in range(args.repeticiones):
                orden = ['defecto', 'optimizado'] if (rep + fold) % 2 == 0 else ['optimizado', 'defecto']
                for variante in orden:
                    dest = args.salida / f'{familia}_fold{fold}_rep{rep}_{variante}.json'
                    with dest.with_suffix('.log').open('x') as log:
                        proc = subprocess.run([sys.executable, '-m', 'proyecto_final_caries.analisis.benchmark_recursos',
                            '--worker', familia, str(fold), variante, '--salida', str(dest.resolve())],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if proc.returncode:
                        raise RuntimeError(f'Falló {dest.name}; revisar log. No declarar benchmark completo.')
                    print(dest.name, flush=True)
    guardar(args.salida / 'COMPLETADO.json', {'ajustes':len(familias)*len(args.folds)*args.repeticiones*2})


if __name__ == '__main__':
    main()
