"""Comprueba y resume el suplemento multi-base y las perturbaciones controladas."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from proyecto_final_caries.validacion_adicional.ejecutar import BASE,datasets
ROOT=Path(__file__).resolve().parents[2]
def main():
 out=ROOT/'_static/entregable3';out.mkdir(parents=True,exist_ok=True);assert (BASE/'resultados/COMPLETADO.json').exists() and (BASE/'resultados_ebm/COMPLETADO.json').exists();raw=pd.concat([pd.read_csv(BASE/'resultados/metricas.csv'),pd.read_csv(BASE/'resultados_ebm/metricas.csv')],ignore_index=True);assert len(raw)==243 and not raw.duplicated(['dataset','semilla','fold','familia']).any();assert (raw.groupby(['dataset','familia']).size()==9).all()
 for name,X,y,url in datasets():
  groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
  for seed in [42,43,44]:
   expected=list(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(X,y,groups));seen=[]
   for k,(train,valid) in enumerate(expected):
    p=pd.read_csv(BASE/'resultados'/f'particion_{name}_{seed}_{k}.csv.gz');assert np.array_equal(p.loc[p.train,'row'],train) and np.array_equal(p.loc[p.valid,'row'],valid);assert not set(groups[train])&set(groups[valid]);seen.extend(valid)
    for family in raw.familia.unique():
     folder='resultados_ebm' if family=='EBM' else 'resultados';q=pd.read_csv(BASE/folder/f'pred_{name}_{seed}_{k}_{family}.csv.gz',float_precision='round_trip');r=raw[(raw.dataset==name)&(raw.semilla==seed)&(raw.fold==k)&(raw.familia==family)].iloc[0];assert np.array_equal(q.row,valid) and np.array_equal(q.actual,y[valid]);assert np.isclose(roc_auc_score(q.actual,q.score),r.auc,atol=1e-12);assert np.isclose(average_precision_score(q.actual,q.score),r.ap,atol=1e-12);assert np.isclose(balanced_accuracy_score(q.actual,q.pred),r.balanced_accuracy,atol=1e-12)
   assert sorted(seen)==list(range(len(y)))
 raw.to_csv(out/'adicionales_243_folds.csv',index=False);byseed=raw.groupby(['dataset','familia','semilla'])[['auc','ap','balanced_accuracy']].mean().reset_index();byseed.to_csv(out/'adicionales_medias_semillas.csv',index=False);summary=byseed.groupby(['dataset','familia'])[['auc','ap','balanced_accuracy']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'adicionales_resumen.csv',index=False);p=summary.auc_mean.unstack(0);fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(p.to_numpy(),cmap='YlGnBu',vmin=.5,vmax=1,aspect='auto');ax.set_xticks(np.arange(len(p.columns)),p.columns);ax.set_yticks(np.arange(len(p)),p.index)
 for i in range(len(p)):
  for j in range(len(p.columns)):ax.text(j,i,f'{p.iloc[i,j]:.3f}',ha='center',va='center',color='white' if p.iloc[i,j]>.9 else '#163747')
 fig.colorbar(im,ax=ax,label='AUC medio (medias de tres semillas)');ax.set_title('Tres bases adicionales: comparación con presupuesto común');fig.tight_layout();fig.savefig(out/'adicionales_heatmap.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(14,4),sharey=True)
 for ax,(name,g) in zip(axs,byseed.groupby('dataset')):
  for f in ['Propio','XGBoost','EBM']:
   z=g[g.familia==f];ax.plot(z.semilla,z.auc,'o-',label=f)
  ax.set(title=name,xlabel='Semilla de partición y modelo',ylim=(.5,1));ax.set_xticks([42,43,44])
 axs[0].set_ylabel('Media AUC de tres folds externos');axs[-1].legend();fig.tight_layout();fig.savefig(out/'adicionales_semillas.png',dpi=150);plt.close(fig)
 note=dict(verificacion='243 evaluaciones y sus AUC/AP/balanced accuracy, filas, objetivos y grupos comprobados; particiones reconstruidas desde los datos fuente originales.',datasets=3,semillas=[42,43,44],modelos=9,presupuesto='Dos candidatos, dos folds internos y tres externos por semilla. EBM: 500 rondas fijas, una bolsa.',limites=['Desviación de tres medias por semilla; no intervalo de confianza ni réplicas independientes.','Filtro común de cuantiles limita la representación, especialmente en Spambase; se reporta el número de variables conservadas.','El adaptador añade cinco columnas categóricas constantes al propio por su esquema específico; no aportan señal de etiqueta.','Comparación de EBM restringida; no acredita dominio de todo el estado del arte.','Cada base se entrena por separado; no demuestra que un modelo de caries generalice a otras poblaciones clínicas.']);(out/'adicionales_metodo.json').write_text(json.dumps(note,indent=2,ensure_ascii=False));print(note['verificacion'])
 if (BASE/'robustez/COMPLETADO.json').exists():
  r=pd.read_csv(BASE/'robustez/metricas.csv');assert len(r)==81;r.to_csv(out/'robustez_81.csv',index=False);summ=r.groupby(['dataset','familia','escenario']).auc.agg(['mean','std']);summ.to_csv(out/'robustez_resumen.csv');fig,axs=plt.subplots(1,3,figsize=(14,4),sharey=True);scenarios=['Original','50% perfiles','10% etiquetas']
  for ax,(name,g) in zip(axs,r.groupby('dataset')):
   for family in ['Propio','XGBoost','EBM']:
    z=g[g.familia==family].groupby('escenario').auc.mean().reindex(scenarios);ax.plot(range(3),z,'o-',label=family)
   ax.set(title=name,ylim=(.5,1));ax.set_xticks(range(3),scenarios,rotation=15)
  axs[0].set_ylabel('AUC externo medio: fold 0 en tres semillas');axs[-1].legend();fig.tight_layout();fig.savefig(out/'robustez_escenarios.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
