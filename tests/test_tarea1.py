import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tarea_1.common import FoldClassifier,NUM,CAT
from tarea_1.statistics import validate

def test_delong_matches_direct_algorithm_with_ties():
    assert all(validate().values())

def test_fold_classifier_learns_transformations_only_on_fit_data():
    train=pd.DataFrame({c:[1.,2.,np.nan,4.,5.,6.,7.,8.] for c in NUM})
    for c in CAT:train[c]=['a','b']*4
    model=FoldClassifier(LogisticRegression(),scale=True).fit(train,[0,1]*4)
    before=model.imputer_.statistics_.copy()
    test=train.iloc[:2].copy();test.loc[:,NUM]=1e12;test.loc[:,CAT]='new'
    prediction=model.predict_proba(test)
    assert np.array_equal(model.imputer_.statistics_,before)
    assert not any('new' in c for c in model.encoder_.categories_)
    assert np.isfinite(prediction).all()
