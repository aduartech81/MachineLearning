"""Descomposición aditiva exacta de modelos finales, no evaluación externa ni causalidad."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from proyecto_final_caries.modelo import CAT
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'_static/entregable3';ART=ROOT/'proyecto_final_caries/artefactos_finales_20261023_v3'
def groups(names,features):
 result=[]
 for term in names:
  label=None
  for c in features:
   if term.startswith('numeric__'+c+'_sp_') or term=='num__'+c:label=c;break
   if c in CAT and (term.startswith('categorical__'+c+'_') or term.startswith('cat__'+c+'_')):label=c;break
  result.append(label or term)
 return result
def aggregate(contrib,labels):
 labels=np.asarray(labels);return pd.DataFrame({k:contrib[:,labels==k].sum(1) for k in dict.fromkeys(labels)})
def main():
 OUT.mkdir(parents=True,exist_ok=True);meta=json.loads((ART/'manifest.json').read_text());f=meta['features'];df=pd.read_csv(ROOT/'entregable_2/results/train.csv');X=df[f];ids=X.loc[~pd.util.hash_pandas_object(X,index=False).duplicated()].sample(500,random_state=42).index;sample=X.loc[ids];case=pd.DataFrame([{c:s['ejemplo'] for c,s in meta['esquema'].items()}],columns=f)
 own=joblib.load(ART/'Propio.joblib');names=list(own.preprocessor_.get_feature_names_out())+['Interacción: '+own.numeric_[i]+' × '+own.numeric_[j] for i,j in own.selected_pairs_];labels=groups(names,f);co=aggregate(own._transform(sample)*own.coef_,labels);local=aggregate(own._transform(case)*own.coef_,labels).iloc[0];margin=float(own.decision_function(case)[0]);err=float(abs(own.intercept_+local.sum()-margin));assert err<1e-12
 log=joblib.load(ART/'Logistica_L1_L2.joblib');Z=log['despues'].transform(log['antes'].transform(sample));terms=list(log['despues'].get_feature_names_out());cl=aggregate(Z*log['modelo'].coef_[0],groups(terms,f));assert np.max(np.abs(cl.sum(1).to_numpy()+log['modelo'].intercept_[0]-log['modelo'].decision_function(Z)))<1e-12
 fig,axs=plt.subplots(1,2,figsize=(13,5))
 for ax,title,contrib in zip(axs,['Propio final','Logística final (mejor AUC medio E2)'],[co,cl]):
  m=contrib.abs().mean().sort_values().tail(12);ax.barh(m.index,m,color='#197c88');ax.set(title=title,xlabel='Media de |contribución| al margen bruto')
 fig.suptitle('Explicación exacta: 500 perfiles de desarrollo, modelos ya reentrenados',fontsize=11);fig.tight_layout();fig.savefig(OUT/'explicacion_global_finales.png',dpi=150);plt.close(fig)
 order=local.abs().sort_values(ascending=False).head(8).index;values=[float(local[k]) for k in order]+[float(local.drop(order).sum())];labels_local=list(order)+['Otros términos'];fig,ax=plt.subplots(figsize=(10,6));start=float(own.intercept_);ends=[start]
 for i,(v,label) in enumerate(zip(values,labels_local)):
  end=start+v;ax.barh(i,v,left=start,color='#197c88' if v>=0 else '#bd743f');ax.text(end+(.008 if v>=0 else -.008),i,f'{v:+.3f}',va='center',ha='left' if v>=0 else 'right',color='#173747',fontsize=10);start=end;ends.append(end)
 ax.axvline(own.intercept_,color='#82949b',linestyle=':',label=f'Base {own.intercept_:.4f}');ax.axvline(margin,color='#173747',linestyle='--',label=f'Margen total {margin:.4f}');ax.set_xlim(min(ends)-.06,max(ends)+.07);ax.set_yticks(range(len(values)),labels_local);ax.invert_yaxis();ax.set(xlabel='Margen bruto: se parte del intercepto y se suman los términos',title=f'Perfil sintético: sigmoide original {100*expit(margin):.2f}% (sin calibración nueva)');ax.legend(loc='lower right');fig.tight_layout();fig.savefig(OUT/'explicacion_local_propio.png',dpi=150);plt.close(fig)
 co.abs().mean().rename('media_abs_contribucion').to_csv(OUT/'explicacion_global_propio.csv');local.rename('contribucion').to_csv(OUT/'explicacion_local_propio.csv');(OUT/'explicacion_metodo.json').write_text(json.dumps(dict(muestra=500,unidad='Perfiles distintos de predictores del desarrollo, no pacientes acreditados',semilla=42,caso='Perfil sintético de medianas/modas; no participante real',escala='Margen bruto antes de sigmoide; bases centradas y estandarizadas',error_aditividad_caso=err,probabilidad_original_caso=float(expit(margin)),limites=['Modelos finales ajustados con todo el desarrollo; estas figuras no son validación externa.','Las magnitudes de márgenes de dos modelos no tienen una escala común para comparar importancia absoluta.','Las bases se suman por variable y las interacciones se mantienen aparte.','Alternativa exacta justificada a SHAP para una función aditiva; no atribución causal ni estabilidad inferencial.']),indent=2,ensure_ascii=False));print('Explicaciones finales verificadas; error del propio',err)
if __name__=='__main__':main()
