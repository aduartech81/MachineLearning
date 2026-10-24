"""Pruebas de propiedades del modelo propio, sin tocar los datos del estudio."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from proyecto_final_caries.modelo import ModeloCaries,CAT
from proyecto_final_caries.objetivo import generar_pares,perdida_gradiente

def datos(n=240):
    """Problema sintético separable por edad; otras columnas son ruido."""
    rng=np.random.default_rng(42)
    features=json.loads((Path(__file__).resolve().parents[2]/'entregable_2/results/summary.json').read_text())['features']
    X=pd.DataFrame({k:rng.normal(size=n) for k in features if k not in CAT})
    X['gender']=rng.choice(['F','M'],n);X['smoking']=rng.choice(['N','Y'],n)
    for k in ['hearing(left)','hearing(right)','Urine protein']:X[k]=rng.choice([1,2],n)
    return X[features],(X.age>0).astype(int)

def modelo():
    return ModeloCaries(n_knots=3,max_interactions=0,n_pairs=500,max_iter=300,random_state=42)

def test_aprende_regla_conocida_fuera_de_ajuste():
    X,y=datos();m=modelo().fit(X.iloc[:180],y.iloc[:180]);p=m.predict_proba(X.iloc[180:])
    assert m.converged_
    assert roc_auc_score(y.iloc[180:],p[:,1])>.9
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)

def test_reproducibilidad_misma_semilla():
    X,y=datos(140);a=modelo().fit(X,y);b=modelo().fit(X,y)
    np.testing.assert_allclose(a.predict_proba(X),b.predict_proba(X),atol=1e-12,rtol=0)

def test_clone_sin_estado_ajustado():
    X,y=datos(140);a=modelo().fit(X,y);b=clone(a)
    assert b.get_params()==a.get_params() and not hasattr(b,'coef_')

def test_rechaza_objetivo_identificador_y_clase_unica():
    X,y=datos(140)
    for k in ['ID','dental caries','tartar','oral']:
        with pytest.raises(ValueError):modelo().fit(X.assign(**{k:0}),y)
    with pytest.raises(ValueError):modelo().fit(X,np.zeros(len(X)))

def test_gradiente_coincide_con_diferencias_finitas():
    rng=np.random.default_rng(10);X=rng.normal(size=(12,4));y=np.array([0,1]*6)
    p,n=generar_pares(y,40,42);theta=rng.normal(size=5)*.1
    fun=lambda t:perdida_gradiente(t,X,y,p,n,.3,.1)
    _,jac=fun(theta);eps=1e-6;num=np.empty_like(theta)
    for k in range(len(theta)):
        plus=theta.copy();minus=theta.copy();plus[k]+=eps;minus[k]-=eps
        num[k]=(fun(plus)[0]-fun(minus)[0])/(2*eps)
    np.testing.assert_allclose(jac,num,atol=1e-6,rtol=1e-5)
