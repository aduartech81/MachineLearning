"""Ejecutar desde la raíz: python -m entregable_2.train."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from ml_common import preprocessor, evaluate

ROOT = Path(__file__).parent
SEED = 42


def prepare():
    path = ROOT / 'data/smoking.csv'
    df = pd.read_csv(path)
    if len(df) != 55692 or 'dental caries' not in df:
        raise ValueError('La base no corresponde al esquema esperado.')
    # Hash sólo de predictores: evita compartir perfiles repetidos entre splits.
    raw = df.drop(columns=['ID', 'dental caries'])
    groups = pd.util.hash_pandas_object(raw, index=False)
    if df.groupby(groups)['dental caries'].nunique().gt(1).any():
        raise ValueError('Perfiles idénticos con etiquetas discordantes: revisar.')
    unique = df.loc[~groups.duplicated()].copy()
    unique_groups = groups.loc[unique.index]
    train_groups, test_groups = train_test_split(
        unique_groups, test_size=0.2, stratify=unique['dental caries'],
        random_state=SEED)
    train_mask = groups.isin(train_groups)
    train = df.loc[train_mask].copy()
    test = df.loc[~train_mask].copy()
    # oral es constante; tartar procede del examen oral y se excluye del escenario
    # previo a la exploración. No se presume su disponibilidad temporal.
    drop = ['ID', 'dental caries', 'oral', 'tartar']
    return df, train, test, train.drop(columns=drop), test.drop(columns=drop)


def main():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    df, train, test, X_train, X_test = prepare()
    y_train, y_test = train['dental caries'], test['dental caries']
    # CV sobre perfiles únicos: las copias tampoco cruzan pliegues internos.
    unique_mask = ~pd.util.hash_pandas_object(X_train, index=False).duplicated()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    model = Pipeline([('preprocess', preprocessor(X_train)),
                      ('model', LogisticRegression(max_iter=2000))])
    scores = cross_validate(model, X_train.loc[unique_mask], y_train.loc[unique_mask],
                            cv=cv, scoring=['roc_auc', 'average_precision'],
                            return_train_score=True, n_jobs=1)
    model.fit(X_train, y_train)
    dummy = DummyClassifier(strategy='prior').fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    baseline = dummy.predict_proba(X_test)[:, 1]
    summary = {'seed': SEED, 'rows': len(df), 'train_rows': len(train),
               'test_rows': len(test), 'positive_rows': int(df['dental caries'].sum()),
               'prevalence': float(df['dental caries'].mean()),
               'duplicates_without_ID': int(df.drop(columns='ID').duplicated().sum()),
               'missing_values': int(df.isna().sum().sum()),
               'test_prevalence': float(y_test.mean()),
               'features': X_train.columns.tolist(),
               'excluded': ['ID', 'oral', 'tartar'],
               'logistic': evaluate(y_test, probs), 'dummy': evaluate(y_test, baseline),
               'cv_auc_mean': float(scores['test_roc_auc'].mean()),
               'cv_auc_sd': float(scores['test_roc_auc'].std()),
               'sha256': hashlib.sha256((ROOT/'data/smoking.csv').read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    pd.DataFrame(scores).to_csv(out/'cv.csv', index=False)
    pd.DataFrame({'index': test.index, 'actual': y_test, 'probability': probs,
                  'dummy_probability': baseline}).to_csv(out/'predictions.csv', index=False)
    train.to_csv(out/'train.csv', index=False)
    joblib.dump(model, out/'model.joblib')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
