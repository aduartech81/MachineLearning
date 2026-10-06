import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from entregable_2.train import prepare
from tarea_2.api import app, model


def test_duplicates_do_not_cross_split():
    _, train, test, _, _ = prepare()
    a = set(pd.util.hash_pandas_object(train.drop(columns=['ID', 'dental caries']), index=False))
    b = set(pd.util.hash_pandas_object(test.drop(columns=['ID', 'dental caries']), index=False))
    assert not a.intersection(b)


def test_api_matches_saved_pipeline():
    frame = pd.read_csv('tarea_2/results/current.csv').iloc[[0]]
    payload = frame.iloc[0].to_dict()
    payload = {k: (None if pd.isna(v) else v) for k, v in payload.items()}
    client = TestClient(app)
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    expected = model.predict_proba(frame)[0, 1]
    assert np.isclose(response.json()['heart_disease_probability'], expected)
    assert client.get('/health').json()['model_loaded']


def test_api_rejects_invalid_categories_and_extra_features():
    client = TestClient(app)
    payload = pd.read_csv('tarea_2/results/current.csv').iloc[0].to_dict()
    payload = {k: (None if pd.isna(v) else v) for k, v in payload.items()}
    payload['Sex'] = 'categoría inexistente'
    payload['HeartDisease'] = 1
    assert client.post('/predict', json=payload).status_code == 422


def test_deployed_model_excludes_target_and_demo_leak():
    assert 'HeartDisease' not in model.feature_names_in_
    assert 'leaky_feature' not in model.feature_names_in_
    assert len(model.feature_names_in_) == 11


def test_dashboard_tabs_and_callbacks():
    from entregable_2.app import app, update_eda, update_threshold
    client = app.server.test_client()
    assert client.get('/').status_code == 200
    assert client.get('/_dash-layout').status_code == 200
    for gender in ['Todos', 'F', 'M']:
        figures = update_eda('age', gender)
        assert len(figures[0].data) == 2
    fig, cards = update_threshold(0.5)
    assert len(cards) == 4 and len(fig.data) == 1
