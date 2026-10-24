"""Validación suplementaria limitada: tres bases, tres semillas, 3×2 CV agrupada.
Dos candidatos por familia, mismas particiones y filtro no supervisado para todos.
No sustituye comparación con estado del arte reciente ni prueba superioridad.
"""
from pathlib import Path
import json,hashlib,urllib.request,zipfile,io,time,warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits
from proyecto_final_caries.modelo import ModeloCaries,CAT
BASE=Path(__file__).resolve().parent
class SoporteCuantiles(TransformerMixin,BaseEstimator):
 def fit(self,X,y=None):
  q=X.quantile(np.linspace(0,1,5));self.columns_=[c for c in X if (np.diff(q[c])>0).all()];assert self.columns_;return self
 def transform(self,X):return X[self.columns_].copy()
def adaptar(X):
 X=X.copy()
 for c in CAT:X[c]=0
 return X

def crear(f,p,seed):
 if f=='Propio':return Pipeline([('soporte',SoporteCuantiles()),('adaptar',FunctionTransformer(adaptar)),('modelo',ModeloCaries(**p,max_iter=1000,random_state=seed))])
 models={'Logistica':lambda:LogisticRegression(**p,max_iter=5000,random_state=seed),'SVM':lambda:LinearSVC(**p,dual=False,max_iter=5000,random_state=seed),'KNN':lambda:KNeighborsClassifier(**p,n_jobs=1),'GaussianNB':lambda:GaussianNB(**p),'DecisionTree':lambda:DecisionTreeClassifier(**p,random_state=seed),'RandomForest':lambda:RandomForestClassifier(**p,n_estimators=100,n_jobs=1,random_state=seed),'XGBoost':lambda:XGBClassifier(**p,n_estimators=100,n_jobs=1,tree_method='hist',random_state=seed)}
 return Pipeline([('soporte',SoporteCuantiles()),('imputar',SimpleImputer(strategy='median')),('escalar',StandardScaler()),('modelo',models[f]())])
GRID={'Logistica':[{'C':.1},{'C':1.}],'SVM':[{'C':.1},{'C':1.}],'KNN':[{'n_neighbors':5},{'n_neighbors':25}],'GaussianNB':[{'var_smoothing':1e-9},{'var_smoothing':1e-7}],'DecisionTree':[{'max_depth':5},{'max_depth':10}],'RandomForest':[{'max_depth':5},{'max_depth':None}],'XGBoost':[{'max_depth':2},{'max_depth':4}],'Propio':[{'alpha':0.,'l2':.01},{'alpha':.25,'l2':.01}]}
def score(m,X):return m.predict_proba(X)[:,1] if hasattr(m,'predict_proba') else m.decision_function(X)
def fit(m,X,y):
 with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=1):warnings.simplefilter('always');m.fit(X,y)
 assert not any(issubclass(w.category,ConvergenceWarning) for w in ww),str([str(w.message) for w in ww]);return m

def datasets():
 raw=BASE/'datos';raw.mkdir(exist_ok=True);b=load_breast_cancer();yield 'WDBC',pd.DataFrame(b.data,columns=[f'x_{i:02}' for i in range(b.data.shape[1])]),(b.target==0).astype(int),'https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic'
 for name,id_,file in [('Banknote',267,'data_banknote_authentication.txt'),('Spambase',94,'spambase.data')]:
  url=f'https://archive.ics.uci.edu/static/public/{id_}/'+('banknote+authentication.zip' if id_==267 else 'spambase.zip');path=raw/(name+'.zip')
  if not path.exists():path.write_bytes(urllib.request.urlopen(url,timeout=40).read())
  with zipfile.ZipFile(path) as z:
   match=next(n for n in z.namelist() if n.endswith(file));d=pd.read_csv(io.BytesIO(z.read(match)),header=None)
  X=d.iloc[:,:-1];X.columns=[f'x_{i:02}' for i in range(X.shape[1])];yield name,X,d.iloc[:,-1].to_numpy(int),url

def main():
 out=BASE/'resultados';out.mkdir(exist_ok=False);(out/'protocolo.json').write_text(json.dumps(dict(semillas=[42,43,44],outer=3,inner=2,grid=GRID,metrica='AUC',presupuesto='2 candidatos × 2 folds internos por familia y fold externo',filtro='Conservar columnas cuyos cinco cuantiles del ajuste sean estrictamente crecientes. Igual filtro para todos, aprendido dentro de cada ajuste.',adaptador='El propio requiere cinco columnas categóricas de caries: se añaden constantes cero sin información de etiqueta; su núcleo se conserva.',limites=['Filtro elimina variables dispersas de Spambase; evaluación de representación restringida.','No se añade estado del arte reciente, por lo que este ejercicio no cierra toda la evaluación amplia E3.','Tres semillas y folds reutilizan datos; no son réplicas independientes para inferencia.','No transferencia de un modelo de caries: cada base se entrena por separado.']),indent=2,ensure_ascii=False));records=[];hist=[];sources=[]
 for name,X,y,url in datasets():
  assert np.isfinite(X).all().all();groups=pd.util.hash_pandas_object(X,index=False).to_numpy();data=X.copy();data['target']=y;data.to_csv(BASE/'datos'/(name+'.csv.gz'),index=False);sources.append(dict(nombre=name,url=url,n=len(y),predictores=X.shape[1],sha256=hashlib.sha256((BASE/'datos'/(name+'.csv.gz')).read_bytes()).hexdigest()))
  for seed in [42,43,44]:
   outer=StratifiedGroupKFold(3,shuffle=True,random_state=seed)
   for k,(ia,iv) in enumerate(outer.split(X,y,groups)):
    assert not set(groups[ia])&set(groups[iv]);inners=list(StratifiedGroupKFold(2,shuffle=True,random_state=seed+k).split(X.iloc[ia],y[ia],groups[ia]));pd.DataFrame({'row':np.arange(len(y)),'train':np.isin(np.arange(len(y)),ia),'valid':np.isin(np.arange(len(y)),iv)}).to_csv(out/f'particion_{name}_{seed}_{k}.csv.gz',index=False)
    for family,grid in GRID.items():
     best=-np.inf;chosen=None
     for p in grid:
      vals=[]
      for ti,vi in inners:
       assert not set(groups[ia[ti]])&set(groups[ia[vi]]);m=fit(crear(family,p,seed),X.iloc[ia[ti]],y[ia[ti]]);vals.append(roc_auc_score(y[ia[vi]],score(m,X.iloc[ia[vi]])))
      value=float(np.mean(vals));hist.append(dict(dataset=name,semilla=seed,fold=k,familia=family,parametros=json.dumps(p),auc_interno=value))
      if value>best:best=value;chosen=p
     t=time.perf_counter();m=fit(crear(family,chosen,seed),X.iloc[ia],y[ia]);s=score(m,X.iloc[iv]);yp=m.predict(X.iloc[iv]);records.append(dict(dataset=name,semilla=seed,fold=k,familia=family,auc=float(roc_auc_score(y[iv],s)),ap=float(average_precision_score(y[iv],s)),balanced_accuracy=float(balanced_accuracy_score(y[iv],yp)),parametros=json.dumps(chosen),auc_interno=best,ajuste_y_prediccion_s=time.perf_counter()-t,variables_conservadas=len(m.named_steps['soporte'].columns_)))
     pd.DataFrame({'row':iv,'actual':y[iv],'score':s,'pred':yp}).to_csv(out/f'pred_{name}_{seed}_{k}_{family}.csv.gz',index=False);pd.DataFrame(records).to_csv(out/'metricas.csv',index=False);pd.DataFrame(hist).to_csv(out/'busqueda.csv',index=False)
    print(name,'semilla',seed,'fold',k,'completo',len(records),'/216',flush=True)
  (out/'fuentes.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False))
 r=pd.DataFrame(records);r.groupby(['dataset','familia'])[['auc','ap','balanced_accuracy']].agg(['mean','std']).to_csv(out/'resumen_descriptivo.csv');assert len(r)==216;(out/'COMPLETADO.json').write_text(json.dumps(dict(evaluaciones_externas=216,ajustes_internos=864,semillas=3,datasets=3,limite='Resumen descriptivo con datos repetidos; no prueba superioridad ni cierra estado del arte reciente.')));print('VALIDACION ADICIONAL COMPLETADA',flush=True)
if __name__=='__main__':main()
