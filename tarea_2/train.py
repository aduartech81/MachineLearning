"""Entrenamiento seguro y demostración explícita de fuga del target."""
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from ml_common import preprocessor, evaluate

ROOT = Path(__file__).parent
SEED = 42


def load_split():
    df = pd.read_csv(ROOT/'data/heart.csv')
    if len(df) != 918 or 'HeartDisease' not in df:
        raise ValueError('Se espera Heart Failure Prediction: 918 filas y HeartDisease.')
    X = df.drop(columns='HeartDisease').copy()
    # Regla clínica fija; no estima parámetros con datos de test.
    X[['RestingBP', 'Cholesterol']] = X[['RestingBP', 'Cholesterol']].replace(0, np.nan)
    return df, train_test_split(X, df['HeartDisease'], stratify=df['HeartDisease'],
                               test_size=0.2, random_state=SEED)


def train_pipeline(X, y, model, grid):
    pipe = Pipeline([('preprocess', preprocessor(X)), ('model', model)])
    search = GridSearchCV(pipe, grid,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                          scoring='roc_auc', n_jobs=1, return_train_score=True)
    search.fit(X, y)
    return search


def leakage_demo(X_train, X_test, y_train, y_test):
    a, b = X_train.copy(), X_test.copy()
    rng = np.random.default_rng(SEED)
    a['leaky_feature'] = y_train + rng.normal(0, 0.01, len(a))
    b['leaky_feature'] = y_test + rng.normal(0, 0.01, len(b))
    # Un Pipeline NO elimina una variable derivada del target.
    fit = train_pipeline(a, y_train, LogisticRegression(max_iter=2000),
                         {'model__C': [0.1, 1]})
    return evaluate(y_test, fit.predict_proba(b)[:, 1])


def main():
    df, (X_train, X_test, y_train, y_test) = load_split()
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    candidates = {
        'LogisticRegression': (LogisticRegression(max_iter=2000), {'model__C': [0.1, 1, 10]}),
        'SVC': (SVC(probability=True, random_state=SEED),
                {'model__C': [0.1, 1, 10], 'model__gamma': ['scale', 0.01]}),
        'RandomForest': (RandomForestClassifier(random_state=SEED, n_jobs=1),
                         {'model__n_estimators': [100, 200], 'model__max_depth': [4, None],
                          'model__min_samples_leaf': [1, 5]}),
        'KNN': (KNeighborsClassifier(), {'model__n_neighbors': [5, 11, 21],
                                        'model__weights': ['uniform', 'distance']}),
        'GradientBoosting': (GradientBoostingClassifier(random_state=SEED),
                             {'model__n_estimators': [100, 150], 'model__max_depth': [1, 3]}),
        'GaussianNB': (GaussianNB(), {'model__var_smoothing': [1e-9, 1e-7]})}
    ranked = []
    searches = {}
    for name, (estimator, grid) in candidates.items():
        start = time.perf_counter()
        search = train_pipeline(X_train, y_train, estimator, grid)
        searches[name] = search
        ranked.append({'model': name, 'cv_auc': float(search.best_score_),
                       'seconds': time.perf_counter()-start,
                       'parameters': json.dumps(search.best_params_)})
        pd.DataFrame(search.cv_results_).to_csv(out/f'cv_{name}.csv', index=False)
    # Selección exclusivamente por CV de entrenamiento; test no elige modelo.
    winner = max(searches, key=lambda n: searches[n].best_score_)
    predictions = pd.DataFrame({'actual': y_test})
    for row in ranked:
        proba = searches[row['model']].predict_proba(X_test)[:, 1]
        row.update(evaluate(y_test, proba))
        predictions[row['model']] = proba
    pd.DataFrame(ranked).sort_values('cv_auc', ascending=False).to_csv(out/'ranking.csv', index=False)
    predictions.to_csv(out/'predictions.csv', index=False)
    joblib.dump(searches[winner].best_estimator_, out/'model.joblib')
    summary = {'winner_by_cv': winner, 'rows': len(df), 'train_rows': len(X_train),
               'test_rows': len(X_test), 'features': X_train.columns.tolist(),
               'leakage_demo': leakage_demo(X_train, X_test, y_train, y_test),
               'zero_cholesterol': int((df['Cholesterol'] == 0).sum()),
               'zero_resting_bp': int((df['RestingBP'] == 0).sum()),
               'sha256': hashlib.sha256((ROOT/'data/heart.csv').read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    X_train.to_csv(out/'reference.csv', index=False)
    X_test.to_csv(out/'current.csv', index=False)
    print(pd.DataFrame(ranked).sort_values('cv_auc', ascending=False).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
