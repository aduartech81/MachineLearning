"""Reajuste final para despliegue E3, separado de las evaluaciones históricas.
Regla fijada: mejor configuración por AUC externo medio; parámetros modales de
sus cinco selecciones internas, empate lexicográfico. No añade estimación de desempeño.
"""
from pathlib import Path
import argparse,json,hashlib,time,warnings
from collections import Counter
from importlib.metadata import version
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from sklearn.exceptions import ConvergenceWarning
from proyecto_final_caries.experimentos_v2.evaluar_seleccion import ajustar,puntuar
from proyecto_final_caries.modelo import ModeloCaries

def moda(items):
 c=Counter(json.dumps(x,sort_keys=True) for x in items);return json.loads(sorted(c,key=lambda x:(-c[x],x))[0])
def main():
 p=argparse.ArgumentParser();p.add_argument('--salida',type=Path,required=True);a=p.parse_args();a.salida.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
 Xall=pd.read_csv(root/'entregable_2/results/train.csv');features=json.loads((root/'entregable_2/results/summary.json').read_text())['features'];X=Xall[features];y=Xall['dental caries'];probe=X.iloc[[0,1,2,50,100]].copy()
 tab=pd.read_csv(root/'_static/cierre_entregable2/resumen_108_media_de.csv');raw=pd.read_csv(root/'_static/cierre_entregable2/tabla_maestra_540_folds.csv');spaces=json.loads((root/'proyecto_final_caries/experimentos_v2/espacios_busqueda_v2.json').read_text())['familias'];best=tab.loc[tab.groupby('familia').auc_mean.idxmax()].sort_values('auc_mean',ascending=False);records=[]
 for row in best.itertuples():
  selected=raw[(raw.familia==row.familia)&(raw.optimizador==row.optimizador)&(raw.balanceo==row.balanceo)];assert len(selected)==5;params=moda([json.loads(s) for s in selected.parametros]);t=time.perf_counter()
  if 'n_jobs' in spaces[row.familia]['fijos']:spaces[row.familia]['fijos']['n_jobs']=1
  with threadpool_limits(limits=1):bundle=ajustar(X,y,row.familia,spaces[row.familia],params,row.balanceo);before,_=puntuar(bundle,probe)
  path=a.salida/(row.familia+'.joblib');joblib.dump(bundle,path,compress=3);after,_=puntuar(joblib.load(path),probe);assert np.array_equal(before,after)
  records.append(dict(familia=row.familia,optimizador=row.optimizador,balanceo=row.balanceo,auc_externo_historico_medio=row.auc_mean,parametros=params,n=len(X),ajuste_s=time.perf_counter()-t,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),roundtrip_identico=True,escala='Score original sin calibración nueva'))
  print('FINAL SERIALIZADO',row.familia,flush=True)
 items=[json.loads((root/f'proyecto_final_caries/resultados/comparacion_v1/Propio/fold_{k}/resultado.json').read_text()) for k in range(5)];params=moda([r['parametros'] for r in items]);m=ModeloCaries(**params)
 with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=1):warnings.simplefilter('always');m.fit(X,y)
 assert m.converged_ and not any(issubclass(w.category,ConvergenceWarning) for w in ww);before=m.decision_function(probe);path=a.salida/'Propio.joblib';joblib.dump(m,path,compress=3);assert np.array_equal(before,joblib.load(path).decision_function(probe));records.append(dict(familia='Propio',parametros=params,n=len(X),roundtrip_identico=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),escala='Margen y sigmoide sin calibración nueva'))
 schema={}
 for c in features:
  numeric=pd.api.types.is_numeric_dtype(X[c]);schema[c]=dict(tipo='numero' if numeric else 'categoria',ejemplo=float(X[c].median()) if numeric else str(X[c].mode().iloc[0]),min=float(X[c].min()) if numeric else None,max=float(X[c].max()) if numeric else None,valores=[v.item() if hasattr(v,'item') else v for v in sorted(X[c].dropna().unique())] if X[c].nunique()<10 else None)

 manifest=dict(modelos=records,features=features,esquema=schema,datos_sha256=hashlib.sha256((root/'entregable_2/results/train.csv').read_bytes()).hexdigest(),versiones={p:version(p) for p in ['numpy','pandas','scikit-learn','scipy','xgboost','joblib','imbalanced-learn']},regla='Mejor configuración por media externa; parámetros modales de los cinco folds internos, empate lexicográfico. Propio: parámetros modales V1.',limites=['La selección por CV externa está expuesta a sesgo de selección; no es una nueva prueba independiente.','Reajuste sobre todo desarrollo; test histórico no consultado.','Resultados de evaluación son históricos; artefactos finales no tienen evaluación externa nueva.','Score/sigmoide originales; calibración y umbral de despliegue clínico no establecidos.'])
 (a.salida/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False));print('OCHO FINALES Y ROUNDTRIP COMPLETADOS',flush=True)
if __name__=='__main__':main()
