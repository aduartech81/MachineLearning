"""Sensibilidad descriptiva con hiperparámetros internos fijados, sin retuning del test."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from proyecto_final_caries.validacion_adicional.ejecutar import datasets,crear,fit,score,BASE
from proyecto_final_caries.validacion_adicional.ebm import create

def main():
 out=BASE/'robustez';out.mkdir(exist_ok=False);ref=pd.concat([pd.read_csv(BASE/'resultados/metricas.csv'),pd.read_csv(BASE/'resultados_ebm/metricas.csv')]);rows=[];(out/'protocolo.json').write_text(json.dumps(dict(modelos=['Propio','XGBoost','EBM'],semillas=[42,43,44],fold_externo=0,escenarios=['Original','50% perfiles de entrenamiento','10% perfiles con etiqueta de entrenamiento invertida'],seleccion='Hiperparámetros ya elegidos por CV interna en cada base/semilla/fold 0. No se reoptimizan después de perturbar.',limites=['Análisis descriptivo y posterior al experimento principal.','No es un test de generalización clínica del modelo de caries.','Solo un fold externo por semilla; no muestra exhaustivamente todo ruido o desbalance posible.']),indent=2,ensure_ascii=False))
 for name,X,y,url in datasets():
  groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
  for seed in [42,43,44]:
   part=pd.read_csv(BASE/'resultados'/f'particion_{name}_{seed}_0.csv.gz');ia=part.loc[part.train,'row'].to_numpy();iv=part.loc[part.valid,'row'].to_numpy();unique=np.unique(groups[ia]);rng=np.random.default_rng(seed+200);half=rng.choice(unique,len(unique)//2,replace=False);small=ia[np.isin(groups[ia],half)];rng=np.random.default_rng(seed+100);flips=rng.choice(unique,max(1,int(.1*len(unique))),replace=False);yn=y.copy();mask=np.isin(groups,flips)&np.isin(np.arange(len(y)),ia);yn[mask]=1-yn[mask]
   for scenario,train,target in [('Original',ia,y),('50% perfiles',small,y),('10% etiquetas',ia,yn)]:
    assert not set(groups[train])&set(groups[iv]);assert set(np.unique(target[train]))=={0,1}
    for family in ['Propio','XGBoost','EBM']:
     selected=ref[(ref.dataset==name)&(ref.semilla==seed)&(ref.fold==0)&(ref.familia==family)];assert len(selected)==1;p=json.loads(selected.parametros.iloc[0]);m=create(p,seed) if family=='EBM' else crear(family,p,seed);t=time.perf_counter();m=fit(m,X.iloc[train],target[train]);s=score(m,X.iloc[iv]);auc=float(roc_auc_score(y[iv],s))
     if scenario=='Original':assert np.isclose(auc,selected.auc.iloc[0],atol=1e-12)
     rows.append(dict(dataset=name,semilla=seed,escenario=scenario,familia=family,auc=auc,ap=float(average_precision_score(y[iv],s)),n_train=len(train),n_valid=len(iv),n_etiquetas_invertidas=int(mask.sum()) if scenario=='10% etiquetas' else 0,segundos=time.perf_counter()-t,parametros=json.dumps(p)));pd.DataFrame(rows).to_csv(out/'metricas.csv',index=False)
   print('Robustez',name,seed,len(rows),'/81',flush=True)
 assert len(rows)==81;(out/'COMPLETADO.json').write_text(json.dumps(dict(evaluaciones=81,retuning=False)))
if __name__=='__main__':main()
