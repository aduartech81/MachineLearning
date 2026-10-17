"""Definiciones compartidas para la evaluación completa de Lending Club."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,ClassifierMixin,clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
ROOT=Path(__file__).parent
OUT=ROOT/'results'
NUM=['loan_amnt','int_rate','annual_inc','dti','fico_range_high','open_acc','revol_bal','total_acc']
CAT=['term','grade','home_ownership','verification_status','purpose']
FEATURES=NUM+CAT
NAMES=['GaussianNB','LogisticRegression','DecisionTree','HistGradientBoosting','LinearSVC','RandomForest']

def metrics(y,score,pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y,score)),accuracy=float(accuracy_score(y,pred)),
        precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred)),
        f1=float(f1_score(y,pred)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

class FoldClassifier(ClassifierMixin,BaseEstimator):
    """GridSearchCV sin Pipeline: ajusta transformaciones dentro de cada fit."""
    def __init__(self,estimator=None,scale=False,reg=None):
        self.estimator=estimator;self.scale=scale;self.reg=reg
    def fit(self,X,y):
        start=time.perf_counter()
        self.imputer_=SimpleImputer(strategy='median').fit(X[NUM])
        nums=self.imputer_.transform(X[NUM])
        self.scaler_=StandardScaler().fit(nums) if self.scale else None
        self.encoder_=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(X[CAT])
        data=self.transform(X)
        self.preprocess_seconds_=time.perf_counter()-start
        self.model_=clone(self.estimator)
        if self.reg is not None:self.model_.set_params(C=1/(self.reg*len(y)))
        self.model_.fit(data,y)
        if type(self.model_).__name__=='LinearSVC':
            event=dict(rows=len(y),reg=self.reg,C=self.model_.C,iterations=int(self.model_.n_iter_),
                       max_iter=self.model_.max_iter,tol=self.model_.tol,
                       reached_limit=bool(self.model_.n_iter_>=self.model_.max_iter))
            with (OUT/'svc_convergence.jsonl').open('a') as log:log.write(json.dumps(event)+'\n')
        self.classes_=self.model_.classes_
        self.n_features_in_=X.shape[1]
        return self
    def transform(self,X):
        a=self.imputer_.transform(X[NUM])
        if self.scaler_ is not None:a=self.scaler_.transform(a)
        return np.column_stack([a,self.encoder_.transform(X[CAT])]).astype(np.float32)
    def predict(self,X):return self.model_.predict(self.transform(X))
    def predict_proba(self,X):return self.model_.predict_proba(self.transform(X))
    def decision_function(self,X):
        x=self.transform(X)
        if hasattr(self.model_,'decision_function'):return self.model_.decision_function(x)
        return self.model_.predict_proba(x)[:,1]


def prediction_path(framework,name):
    plain=OUT/f'{framework}_{name}_predictions.csv'
    return plain if plain.exists() else OUT/f'{framework}_{name}_predictions.csv.gz'
