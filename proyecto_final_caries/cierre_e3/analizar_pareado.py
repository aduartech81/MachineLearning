"""Análisis pareado exploratorio con límites explícitos por CV reutilizada."""
from pathlib import Path
import argparse,json,itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from proyecto_final_caries.cierre_e3.bootstrap_perfiles import preparar,auc_ponderada,jackknife_perfiles,intervalo_bca
from proyecto_final_caries.analisis.contrastes_sensibilidad import holm

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--entrada',type=Path,required=True);ap.add_argument('--salida',type=Path,required=True);a=ap.parse_args();out=a.salida;out.mkdir(parents=True,exist_ok=True);src=a.entrada
 assert (src/'COMPLETADO.json').exists();r=pd.read_csv(src/'metricas.csv');assert len(r)==45 and not r.duplicated(['familia','fold']).any();piv=r.pivot(index='fold',columns='familia',values='auc');assert piv.shape==(5,9);summ=r.groupby('familia')[['auc','average_precision','accuracy','balanced_accuracy','ajuste_s','inferencia_s']].agg(['mean','std']);summ.columns=['_'.join(c) for c in summ.columns];summ=summ.sort_values('auc_mean',ascending=False)
 frames={}
 for f in piv.columns:
  frames[f]=pd.concat([pd.read_csv(src/f'pred_{f}_fold{k}.csv.gz',float_precision='round_trip') for k in range(5)]).sort_values('row_id').reset_index(drop=True)
 ref=frames['Propio'];assert ref.row_id.is_unique;y=ref.actual.to_numpy();groups=ref.perfil.to_numpy();prepared={}
 for f,q in frames.items():
  assert np.array_equal(q.row_id,ref.row_id) and np.array_equal(q.actual,y) and np.array_equal(q.perfil,groups)
  assert np.isfinite(q.score).all();prepared[f]=preparar(y,q.score.to_numpy(),groups)
  for k in range(5):
   qk=pd.read_csv(src/f'pred_{f}_fold{k}.csv.gz');value=r[(r.familia==f)&(r.fold==k)].auc.iloc[0];assert np.isclose(roc_auc_score(qk.actual,qk.score),value)
 # Condicional a estas predicciones: remuestreo pareado de perfiles completos, nunca de filas sueltas.
 rng=np.random.default_rng(42);B=2000;K=len(prepared['Propio']['groups']);boots={f:[] for f in prepared};prob=np.full(K,1/K)
 for start in range(0,B,16):
  weights=rng.multinomial(K,prob,size=min(16,B-start))
  for f,p in prepared.items():boots[f].extend(auc_ponderada(p,weights).tolist())
 jack={f:jackknife_perfiles(p) for f,p in prepared.items()};oof={f:float(roc_auc_score(y,q.score)) for f,q in frames.items()};ci=[]
 for f in piv.columns:
  item=dict(familia=f,auc_oof=oof[f],**intervalo_bca(oof[f],boots[f],jack[f]));ci.append(item)
 pd.DataFrame(ci).to_csv(out/'auc_oof_bca_condicional.csv',index=False)
 test=[];ratio=float((r.n_valid/r.n_train).mean());fr=stats.friedmanchisquare(*[piv[c] for c in piv]);N,k=piv.shape;cd=float(stats.studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*N)));ranks=piv.rank(axis=1,ascending=False).mean();ranks.to_csv(out/'rangos.csv',header=['rango_medio'])
 for f in piv.columns:
  if f=='Propio':continue
  d=(piv.Propio-piv[f]).to_numpy();mean=float(d.mean());sd=float(d.std(ddof=1));dz=mean/sd if sd else 0.;cliff=float(np.sign(piv.Propio.to_numpy()[:,None]-piv[f].to_numpy()).mean());w=stats.wilcoxon(d,method='exact' if np.all(d!=0) else 'auto');t=mean/np.sqrt((1/N+ratio)*sd*sd) if sd else 0.;pt=float(2*stats.t.sf(abs(t),df=N-1));delta=oof['Propio']-oof[f];bb=np.asarray(boots['Propio'])-np.asarray(boots[f]);bci=intervalo_bca(delta,bb,jack['Propio']-jack[f]);centered=bb-delta;pb=float((1+np.count_nonzero(np.abs(centered)>=abs(delta)))/(B+1))
  for name,stat,pval,hyp,limits in [
   ('Wilcoxon pareado',float(w.statistic),float(w.pvalue),'Mediana simétrica de diferencias AUC por fold igual a cero','Cinco folds con entrenamiento solapado; baja potencia; simetría/independencia no certificadas.'),
   ('t CV corregida (Nadeau–Bengio)',float(t),pt,'Media de diferencias AUC igual a cero','Corrección aproximada de varianza: (1/N + nvalid/ntrain)s². Una repetición de cinco folds; normalidad y corrección no garantizadas.'),
   ('Bootstrap de perfiles centrado',delta,pb,'Diferencia AUC OOF condicional igual a cero','Predicciones fijas; remuestrea perfiles, no pacientes verificados ni reentrenamientos. Distribución centrada bajo nula aproximada.')]:
   test.append(dict(prueba=name,comparacion='Propio − '+f,estadistico=stat,p_crudo=pval,hipotesis_nula=hyp,supuestos_y_limites=limits,delta_auc_media_folds=mean,delta_auc_oof=delta,bca95_inferior=bci['inferior'],bca95_superior=bci['superior'],cohen_dz_descriptivo=dz,cliff_marginal_descriptivo=cliff,direccion='Favorece propio' if mean>0 else 'Favorece referencia'))
 tests=pd.DataFrame(test);tests['p_holm_24']=holm(tests.p_crudo.to_numpy());tests['conclusion_exploratoria']=np.where((tests.p_holm_24<.05)&(tests.delta_auc_media_folds>0),'Diferencia favorable dentro de este protocolo; no acredita R3 general',np.where(tests.p_holm_24<.05,'Diferencia desfavorable al propio','No se rechaza igualdad; no demuestra equivalencia'));tests.to_csv(out/'pruebas_pareadas.csv',index=False);summ.to_csv(out/'resumen_9_modelos.csv')
 fig,ax=plt.subplots(figsize=(10,5));colors=['#be733f' if f=='Propio' else '#197c88' for f in summ.index];ax.barh(summ.index,summ.auc_mean,xerr=summ.auc_std,color=colors);ax.invert_yaxis();ax.set(xlim=(.5,.65),xlabel='AUC externo medio ± DE entre cinco folds',title='Comparación suplementaria: mismo presupuesto y particiones');fig.tight_layout();fig.savefig(out/'comparacion_pareada.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5));rr=ranks.sort_values();ax.scatter(rr.values,np.arange(k),color=['#be733f' if f=='Propio' else '#197c88' for f in rr.index]);ax.set_yticks(np.arange(k),rr.index);ax.invert_yaxis();ax.set(xlim=(.5,k+.5),xlabel='Rango medio (1 = mejor AUC)',title=f'Friedman exploratorio: p={fr.pvalue:.4g}; CD Nemenyi={cd:.2f}');ax.plot([1,1+cd],[k-.35,k-.35],color='#bc7442',lw=3);ax.text(1,k-.8,'Longitud de la diferencia crítica',fontsize=9);fig.tight_layout();fig.savefig(out/'rangos_cd.png',dpi=150);plt.close(fig)
 method=dict(verificacion='45 evaluaciones, nueve familias y alineación de 44.582 filas OOF verificadas',familias=list(piv.columns),friedman=dict(chi2=float(fr.statistic),p=float(fr.pvalue),N=N,k=k),nemenyi_cd=cd,comparaciones_pareadas=len(tests),multiplicidad='Holm conjunto de los 24 contrastes pareados mostrados; no corrige todos los análisis históricos del proyecto.',bootstrap=dict(B=B,semilla=42,perfiles=K,intervalo='BCa, delete-one-profile exacto si el score es constante dentro del perfil'),limites=['Alcance exploratorio y condicional; no nuevos pacientes ni test independiente.','Se muestran tres procedimientos pareados además de un procedimiento global; no son confirmaciones independientes.','Nemenyi es un diagnóstico de rangos de muestra pequeña; una conexión/no rechazo no implica equivalencia.','Las representaciones difieren en el núcleo spline del propio; el preprocesamiento previo y las particiones son comunes.','No sustituye la comparación con los mejores de las 108 configuraciones E2 ni demuestra R3 por contar pruebas.'],R3='No acreditado por este análisis; interpretar junto a resultados V1/V2 y referencias seleccionadas E2.')
 (out/'metodo.json').write_text(json.dumps(method,indent=2,ensure_ascii=False));print(json.dumps(method,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
