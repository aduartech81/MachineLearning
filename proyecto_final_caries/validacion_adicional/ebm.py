"""Referencia EBM actual, con las particiones y presupuesto del suplemento original."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from interpret.glassbox import ExplainableBoostingClassifier
from proyecto_final_caries.validacion_adicional.ejecutar import SoporteCuantiles,fit,score,BASE,datasets
GRID=[{'learning_rate':.02,'interactions':0},{'learning_rate':.02,'interactions':5}]
def create(p,seed):return Pipeline([('soporte',SoporteCuantiles()),('imputar',SimpleImputer(strategy='median')),('escalar',StandardScaler()),('modelo',ExplainableBoostingClassifier(**p,max_rounds=500,outer_bags=1,validation_size=0,early_stopping_rounds=0,n_jobs=1,random_state=seed))])
def main():
 out=BASE/'resultados_ebm';out.mkdir(exist_ok=False);rows=[];history=[];(out/'protocolo.json').write_text(json.dumps(dict(grids=GRID,version_interpret='0.7.8',max_rounds=500,outer_bags=1,validation_size=0,presupuesto='Dos candidatos, dos folds internos; mismos tres folds externos y tres semillas que las otras ocho familias',limite='Referencia moderna acotada, no búsqueda exhaustiva ni prueba de mejor estado del arte.'),indent=2,ensure_ascii=False))
 for name,X,y,url in datasets():
  groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
  for seed in [42,43,44]:
   for k in range(3):
    part=pd.read_csv(BASE/'resultados'/f'particion_{name}_{seed}_{k}.csv.gz');ia=part.loc[part.train,'row'].to_numpy();iv=part.loc[part.valid,'row'].to_numpy();assert not set(groups[ia])&set(groups[iv]);inner=list(StratifiedGroupKFold(2,shuffle=True,random_state=seed+k).split(X.iloc[ia],y[ia],groups[ia]));chosen=None;best=-np.inf
    for p in GRID:
     vals=[]
     for ti,vi in inner:
      m=fit(create(p,seed),X.iloc[ia[ti]],y[ia[ti]]);vals.append(roc_auc_score(y[ia[vi]],score(m,X.iloc[ia[vi]])))
     value=float(np.mean(vals));history.append(dict(dataset=name,semilla=seed,fold=k,parametros=json.dumps(p),auc_interno=value))
     if value>best:best=value;chosen=p
    t=time.perf_counter();m=fit(create(chosen,seed),X.iloc[ia],y[ia]);s=score(m,X.iloc[iv]);pred=m.predict(X.iloc[iv]);rows.append(dict(dataset=name,semilla=seed,fold=k,familia='EBM',auc=float(roc_auc_score(y[iv],s)),ap=float(average_precision_score(y[iv],s)),balanced_accuracy=float(balanced_accuracy_score(y[iv],pred)),parametros=json.dumps(chosen),auc_interno=best,ajuste_y_prediccion_s=time.perf_counter()-t,variables_conservadas=len(m.named_steps['soporte'].columns_)))
    pd.DataFrame({'row':iv,'actual':y[iv],'score':s,'pred':pred}).to_csv(out/f'pred_{name}_{seed}_{k}_EBM.csv.gz',index=False);pd.DataFrame(rows).to_csv(out/'metricas.csv',index=False);pd.DataFrame(history).to_csv(out/'busqueda.csv',index=False);print('EBM adicional',len(rows),'/27',flush=True)
 assert len(rows)==27;(out/'COMPLETADO.json').write_text(json.dumps(dict(evaluaciones_externas=27,ajustes_internos=108)))
if __name__=='__main__':main()
