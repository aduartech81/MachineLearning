import numpy as np
from sklearn.metrics import roc_auc_score
from proyecto_final_caries.cierre_e3.bootstrap_perfiles import preparar,auc_ponderada,jackknife_perfiles

def fixture():
 y=np.array([0,0,1,1,0,1,0,1]);s=np.array([.2,.2,.8,.5,.4,.4,.7,.9]);g=np.array([0,0,1,2,3,3,4,5]);return y,s,g

def test_auc_pesos_y_empates_coincide_sklearn():
 y,s,g=fixture();p=preparar(y,s,g);w=np.array([[1,1,1,1,1,1],[2,0,3,1,2,1]])
 for weights,value in zip(w,auc_ponderada(p,w)):assert np.isclose(value,roc_auc_score(y,s,sample_weight=weights[g]),atol=1e-14)

def test_jackknife_exactamente_igual_a_eliminar_perfil():
 y,s,g=fixture();p=preparar(y,s,g);actual=jackknife_perfiles(p)
 for k,v in enumerate(actual):assert np.isclose(v,roc_auc_score(y[g!=k],s[g!=k]),atol=1e-14)
