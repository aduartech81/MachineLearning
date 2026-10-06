"""Preprocesamiento y evaluación binaria reproducibles."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def preprocessor(frame):
    cat = frame.select_dtypes(include=['object', 'category']).columns.tolist()
    num = [c for c in frame if c not in cat]
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), num),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('encoder', OneHotEncoder(handle_unknown='ignore',
                                                    sparse_output=False))]), cat)
    ])


def evaluate(y, probability, threshold=0.5):
    pred = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y, probability)),
                average_precision=float(average_precision_score(y, probability)),
                accuracy=float(accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                specificity=float(tn / (tn + fp)),
                f1=float(f1_score(y, pred, zero_division=0)),
                brier=float(brier_score_loss(y, probability)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
