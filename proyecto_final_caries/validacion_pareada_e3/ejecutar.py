"""Comparación suplementaria pareada, presupuesto común y sin balanceo.
Conserva particiones externas/internas históricas. No sustituye V1/V2 ni prueba R3.
"""
from pathlib import Path
import argparse,json,hashlib,time,warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn import config_context
from threadpoolctl import threadpool_limits
from proyecto_final_caries.experimentos_v2 import ejecutar_grid as g
from proyecto_final_caries.experimentos_v2.preprocesamiento import antes_balanceo,despues_balanceo
from proyecto_final_caries.modelo import ModeloCaries
from interpret.glassbox import ExplainableBoostingClassifier
ROOT=Path(__file__).resolve().parents[2]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--salida',type=Path,required=True);a=ap.parse_args();out=a.salida;out.mkdir(parents=True,exist_ok=False)
 spaces=json.loads((ROOT/'proyecto_final_caries/experimentos_v2/espacios_busqueda_v2.json').read_text())['familias'];grids={f:[s['grid'][i] for i in [0,len(s['grid'])//3,2*len(s['grid'])//3,len(s['grid'])-1]] for f,s in spaces.items()};grids['Propio']=json.loads((ROOT/'proyecto_final_caries/plan_comparacion_v1.json').read_text())['grillas']['Propio'];grids['EBM']=[{'learning_rate':lr,'interactions':it} for lr in [.02,.05] for it in [0,5]]
 classes=dict(KNN=g.KNeighborsClassifier,GaussianNB=g.GaussianNB,Logistica_L1_L2=g.LogisticRegression,DecisionTree=g.DecisionTreeClassifier,RandomForest=g.RandomForestClassifier,XGBoost=g.XGBClassifier,SVM=g.LinearSVC)
 def make(f,p):
  if f=='Propio':return ModeloCaries(**p,max_iter=1000,random_state=42)
  if f=='EBM':return ExplainableBoostingClassifier(**p,max_rounds=500,outer_bags=1,validation_size=0,early_stopping_rounds=0,n_jobs=1,random_state=42)
  fixed=spaces[f]['fijos'].copy()
  if 'n_jobs' in fixed:fixed['n_jobs']=1
  return classes[f](**fixed,**p)
 df=pd.read_csv(ROOT/'entregable_2/results/train.csv');features=json.loads((ROOT/'entregable_2/results/summary.json').read_text())['features'];X=df[features];y=df['dental caries'].to_numpy();outer=pd.read_csv(ROOT/'proyecto_final_caries/evaluacion/particiones_desarrollo.csv');inner=pd.read_csv(ROOT/'proyecto_final_caries/evaluacion/particiones_internas.csv');assert np.array_equal(outer.perfil,pd.util.hash_pandas_object(X,index=False));X.index=outer.row_id
 protocol=dict(semilla=42,grids=grids,n_candidatos_por_familia=4,outer=5,inner=3,balanceo='Ninguno en las nueve familias',upstream='Imputación y escalado común, ajustado dentro de cada entrenamiento; one-hot común para referencias. Splines/interacciones pertenecen al núcleo del propio.',ebm='InterpretML 0.7.8, 500 rondas fijas, una bolsa y sin parada temprana; referencia moderna restringida, no búsqueda exhaustiva.',seleccion='AUC interno medio; desempate primer candidato',datos_sha256=hashlib.sha256((ROOT/'entregable_2/results/train.csv').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limites=['Nuevo protocolo exploratorio de presupuesto común; no reemplaza las 108 configuraciones E2 ni los artefactos de producción.','Representación spline del propio es parte de su algoritmo; no se afirma igualdad de representaciones finales.','Folds históricos ya consultados; sin test independiente nuevo.','Una semilla; cinco folds comparten entrenamiento.'])
 (out/'protocolo.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False));records=[];history=[]
 # El orden de y se toma por row_id explícito, no mediante una máscara con orden potencialmente distinto.
 ys=pd.Series(y,index=X.index)
 def data(ia,iv):
  before=antes_balanceo(features);after=despues_balanceo(features);A=before.fit_transform(X.loc[ia]);V=before.transform(X.loc[iv]);return A,V,after.fit_transform(A),after.transform(V),ys.loc[ia].to_numpy(),ys.loc[iv].to_numpy()
 def run(f,p,D):
  A,V,Z,Q,yt,yv=D;m=make(f,p);a_,v_=(A,V) if f=='Propio' else (Z,Q)
  with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=1),config_context(working_memory=64):
   warnings.simplefilter('always');t=time.perf_counter();m.fit(a_,yt);fit=time.perf_counter()-t;t=time.perf_counter();score=m.decision_function(v_) if hasattr(m,'decision_function') else m.predict_proba(v_)[:,1];pred=m.predict(v_);inf=time.perf_counter()-t
  assert not any(issubclass(w.category,ConvergenceWarning) for w in ww),(f,p,[str(w.message) for w in ww]);assert np.isfinite(score).all()
  return np.asarray(score),np.asarray(pred),float(roc_auc_score(yv,score)),fit,inf
 for k in range(5):
  iv=outer.loc[outer.outer_validation_fold==k,'row_id'].to_numpy();ia=inner.loc[inner.outer_fold==k,'row_id'].to_numpy();assert len(np.unique(ia))==len(ia);assert not set(outer.set_index('row_id').loc[ia,'perfil'])&set(outer.set_index('row_id').loc[iv,'perfil']);sub=inner[inner.outer_fold==k];cached=[]
  for j in range(3):cached.append(data(sub.loc[sub.inner_validation_fold!=j,'row_id'].to_numpy(),sub.loc[sub.inner_validation_fold==j,'row_id'].to_numpy()))
  ext=data(ia,iv)
  for f,grid in grids.items():
   best=-np.inf;chosen=None;t0=time.perf_counter()
   for p in grid:
    vals=[]
    for D in cached:vals.append(run(f,p,D)[2])
    val=float(np.mean(vals));history.append(dict(familia=f,fold=k,parametros=json.dumps(p,sort_keys=True),auc_interno=val))
    if val>best:best=val;chosen=p
   s,pred,auc,fit,inf=run(f,chosen,ext);records.append(dict(familia=f,fold=k,auc=auc,average_precision=float(average_precision_score(ext[-1],s)),accuracy=float(accuracy_score(ext[-1],pred)),balanced_accuracy=float(balanced_accuracy_score(ext[-1],pred)),parametros=json.dumps(chosen,sort_keys=True),auc_interno=best,ajuste_s=fit,inferencia_s=inf,total_s=time.perf_counter()-t0,n_train=len(ia),n_valid=len(iv),semilla=42))
   pd.DataFrame({'row_id':iv,'perfil':outer.set_index('row_id').loc[iv,'perfil'].to_numpy(),'actual':ext[-1],'score':s,'prediccion':pred}).to_csv(out/f'pred_{f}_fold{k}.csv.gz',index=False);pd.DataFrame(records).to_csv(out/'metricas.csv',index=False);pd.DataFrame(history).to_csv(out/'busqueda.csv',index=False);print(len(records),'/45',f,'fold',k,'AUC',round(auc,6),flush=True)
 assert len(records)==45 and len(history)==180;(out/'COMPLETADO.json').write_text(json.dumps(dict(evaluaciones_externas=45,ajustes_internos=540,total_ajustes=585)));print('COMPARACIÓN PAREADA COMPLETADA',flush=True)
if __name__=='__main__':main()
