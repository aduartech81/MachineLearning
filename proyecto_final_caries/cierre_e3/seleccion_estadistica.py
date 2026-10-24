"""Finalistas de las siete configuraciones elegidas por media externa E2."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'_static/entregable3'
def main():
 chosen=pd.read_csv(OUT/'seleccion_7_referencias.csv');raw=pd.read_csv(ROOT/'_static/cierre_entregable2/tabla_maestra_540_folds.csv');pieces=[]
 for row in chosen.itertuples():
  q=raw[(raw.familia==row.familia)&(raw.optimizador==row.optimizador)&(raw.balanceo==row.balanceo)];assert len(q)==5;pieces.append(q[['fold','auc']].set_index('fold').rename(columns={'auc':row.familia}))
 p=pd.concat(pieces,axis=1);ranks=p.rank(axis=1,ascending=False).mean();N,k=p.shape;fr=stats.friedmanchisquare(*[p[c] for c in p]);best=chosen.iloc[0].familia;se=np.sqrt(k*(k+1)/(6*N));cd=float(stats.studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*se);rows=[]
 for f in p.columns:
  difference=float(abs(ranks[best]-ranks[f]));pv=float(stats.studentized_range.sf(difference/se*np.sqrt(2),k,np.inf)) if f!=best else 1.;rows.append(dict(familia=f,rango_medio=float(ranks[f]),referencia_global=best,p_nemenyi_frente_global=pv,finalista_no_rechazado=pv>=.05))
 pd.DataFrame(rows).to_csv(OUT/'finalistas_nemenyi_seleccion_externa.csv',index=False);(OUT/'seleccion_estadistica.json').write_text(json.dumps(dict(mejor_global=best,auc_medio=float(chosen.iloc[0].auc_mean),friedman_chi2=float(fr.statistic),p_friedman=float(fr.pvalue),N=N,k=k,cd_nemenyi=cd,limites=['Configuraciones elegidas por su media externa sobre estos mismos cinco folds: selección posterior a resultados.','Conjunto de finalistas significa no rechazo frente al mejor observado, no equivalencia ni probabilidad de ser el mejor.','Folds con entrenamiento solapado; inferencia exploratoria y aproximación asintótica débil con N=5.','Este análisis usa las configuraciones fijas elegidas para E3; no se confunde con la selección interna cambiante por fold del informe V2 anterior.']),indent=2,ensure_ascii=False));print('Selección externa y finalistas E3 comprobados')
if __name__=='__main__':main()
