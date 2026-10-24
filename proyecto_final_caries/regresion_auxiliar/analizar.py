"""Verifica la corrida completa y produce análisis transversal sin pruebas temporales."""
from pathlib import Path
import argparse,json,hashlib,itertools,shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--entrada',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);a=p.parse_args();src=a.entrada;out=a.salida;out.mkdir(parents=True,exist_ok=True)
 done=json.loads((src/'COMPLETADO.json').read_text());r=pd.read_csv(src/'metricas_por_fold.csv');h=pd.read_csv(src/'historial_propuestas.csv.gz');proto=json.loads((src/'protocolo.json').read_text())
 assert len(r)==140 and len(h)==1120 and r.convergio.all()
 assert not r.duplicated(['familia','optimizador','fold']).any()
 assert (r.groupby(['familia','optimizador']).size()==5).all()
 assert np.isfinite(r[['rmse','mae','r2','rmse_interno']]).all().all()
 script=Path(__file__).with_name('ejecutar.py');assert hashlib.sha256(script.read_bytes()).hexdigest()==proto['script_sha256']
 parts=pd.read_csv(src/'particiones_externas.csv');assert parts.row_id.is_unique and len(parts)==proto['n'];assert parts.groupby('perfil').outer_fold.nunique().max()==1
 inn=pd.read_csv(src/'particiones_internas.csv').merge(parts[['row_id','perfil','outer_fold']],on='row_id',suffixes=('','_externo'))
 assert (inn.outer_fold!=inn.outer_fold_externo).all();assert inn.groupby(['outer_fold','perfil']).inner_fold.nunique().max()==1
 for x in r.itertuples():
  check=pd.read_csv(src/f'{x.familia}_{x.optimizador}_fold{x.fold}_pred.csv.gz')
  assert len(check)==x.n_validacion and check.row_id.is_unique
  expected=parts.loc[parts.outer_fold==x.fold,'row_id'].to_numpy()
  assert np.array_equal(check.row_id.to_numpy(),expected)
  assert np.isfinite(check[['actual','prediccion']]).all().all()
  assert np.isclose(np.sqrt(np.mean((check.actual-check.prediccion)**2)),x.rmse)
  assert np.isclose(np.mean(np.abs(check.actual-check.prediccion)),x.mae)
 summary=pd.read_csv(src/'resumen_28.csv').sort_values('rmse_mean');summary.to_csv(out/'resumen_28.csv',index=False)
 # Los finalistas se eligen por CV interna en cada fold, sin consultar su RMSE externo.
 finalists=r.loc[r.groupby(['familia','fold']).rmse_interno.idxmin()].sort_values(['familia','fold']);finalists.to_csv(out/'finalistas_35_seleccion_interna.csv',index=False)
 fm=finalists.groupby('familia')[['rmse','mae','r2','rmse_ajuste','dummy_rmse','dummy_mae']].agg(['mean','std']);fm.columns=['_'.join(c) for c in fm.columns];fm=fm.sort_values('rmse_mean');fm.to_csv(out/'resumen_finalistas_7.csv')
 predictions={}
 for fam,g in finalists.groupby('familia'):
  pp=[]
  for x in g.itertuples():
   q=pd.read_csv(src/f'{fam}_{x.optimizador}_fold{x.fold}_pred.csv.gz');assert len(q)==x.n_validacion
   assert np.isclose(np.sqrt(np.mean((q.actual-q.prediccion)**2)),x.rmse)
   pp.append(q)
  q=pd.concat(pp).sort_values('row_id');assert q.row_id.is_unique and np.array_equal(q.row_id,np.sort(parts.row_id));q['residuo']=q.actual-q.prediccion;predictions[fam]=q
 best=str(fm.index[0]);q=predictions[best];q.to_csv(out/'predicciones_finalista_descriptivo.csv.gz',index=False)
 # Comparaciones de sensibilidad: cinco folds con entrenamientos solapados; Holm sobre seis contrastes.
 pivot=finalists.pivot(index='fold',columns='familia',values='rmse');fr=stats.friedmanchisquare(*[pivot[c] for c in pivot]);compar=[]
 for fam in fm.index[1:]:
  d=(pivot[best]-pivot[fam]).to_numpy();w=stats.wilcoxon(d,method='exact') if np.all(d!=0) else stats.wilcoxon(d,method='auto')
  null=np.array([np.mean(d*np.array(s)) for s in itertools.product([-1,1],repeat=len(d))]);per=float(np.mean(np.abs(null)>=abs(d.mean())-1e-12))
  compar.append(dict(modelo_a=best,modelo_b=fam,delta_rmse_media=float(d.mean()),p_wilcoxon=float(w.pvalue),p_permutacion_signos=per))
 c=pd.DataFrame(compar)
 for col in ['p_wilcoxon','p_permutacion_signos']:
  order=np.argsort(c[col]);adj=np.empty(len(c));running=0
  for j,i in enumerate(order):running=max(running,(len(c)-j)*c.iloc[i][col]);adj[i]=min(1,running)
  c[col+'_holm']=adj
 c.to_csv(out/'contrastes_sensibilidad.csv',index=False)
 # Diagnóstico de forma/heterocedasticidad descriptivo; no se presupone orden temporal.
 diag=[]
 for fam,pred in predictions.items():
  e=pred.residuo.to_numpy();rho,pval=stats.spearmanr(np.abs(e),pred.prediccion)
  diag.append(dict(familia=fam,rmse_oof=float(np.sqrt(np.mean(e**2))),mae_oof=float(np.mean(abs(e))),residuo_medio=float(e.mean()),mediana=float(np.median(e)),q01=float(np.quantile(e,.01)),q99=float(np.quantile(e,.99)),asimetria=float(stats.skew(e)),curtosis_exceso=float(stats.kurtosis(e)),rho_abs_residuo_pred=float(rho),p_nominal_spearman=float(pval)))
 pd.DataFrame(diag).to_csv(out/'diagnostico_residuos.csv',index=False)
 fig,ax=plt.subplots(2,2,figsize=(11,8));sample=q.sample(min(6000,len(q)),random_state=42)
 ax[0,0].scatter(sample.actual,sample.prediccion,s=5,alpha=.2);lim=[q.actual.min(),q.actual.max()];ax[0,0].plot(lim,lim,'k--');ax[0,0].set(xlabel='Glucemia registrada (unidades originales)',ylabel='Predicción OOF',title=f'{best}: estimación contemporánea')
 ax[0,1].scatter(sample.prediccion,sample.residuo,s=5,alpha=.2);ax[0,1].axhline(0,color='k');ax[0,1].set(xlabel='Predicción OOF',ylabel='Residuo: observado − predicho',title='Dispersión y heterocedasticidad')
 ax[1,0].hist(q.residuo,bins=80);ax[1,0].set(xlabel='Residuo',ylabel='Filas',title='Distribución de errores (sin recortar extremos)')
 stats.probplot(q.residuo,dist='norm',plot=ax[1,1]);ax[1,1].set_title('Q-Q normal: diagnóstico, no supuesto de RMSE')
 fig.tight_layout();fig.savefig(out/'residuos.png',dpi=140);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(fm));ax.bar(x-.2,fm.rmse_ajuste_mean,.4,label='Ajuste');ax.bar(x+.2,fm.rmse_mean,.4,yerr=fm.rmse_std,label='Validación externa ± DE');ax.axhline(fm.dummy_rmse_mean.iloc[0],color='k',linestyle='--',label='Dummy media, RMSE externo medio');ax.set_xticks(x,fm.index,rotation=20);ax.set_ylabel('RMSE (unidades originales)');ax.legend();fig.tight_layout();fig.savefig(out/'ajuste_validacion_dummy.png',dpi=140);plt.close(fig)
 fig,axs=plt.subplots(2,4,figsize=(15,7),sharex=True)
 for fam,ax in zip(sorted(r.familia.unique()),axs.flat):
  for opt,g in h[h.familia==fam].groupby('optimizador'):
   m=g.groupby('propuesta').mejor_rmse.mean();ax.plot(m.index,m,label=opt)
  ax.set(title=fam,xlabel='Propuesta',ylabel='Mejor RMSE interno medio')
 axs.flat[-1].axis('off');axs.flat[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'convergencia.png',dpi=140);plt.close(fig)
 note=dict(**done,verificacion='140 resultados, 1120 propuestas, 28 configuraciones, no solapamiento de perfiles y coincidencia de RMSE con predicciones verificados',mejor_familia_descriptiva=best,mejor_configuracion_externa=summary.iloc[0].to_dict(),friedman_sensibilidad=dict(estadistico=float(fr.statistic),p=float(fr.pvalue)),limites=['Selección interna por fold para comparar familias; ranking externo de 28 configuraciones descriptivo y sujeto a selección.','Cinco folds con entrenamiento solapado; Friedman, Wilcoxon y permutación son sensibilidades exploratorias, no tres confirmaciones independientes.','Sin orden temporal ni identificador de paciente validado: no se ejecutan BDS, Ljung-Box, DM, GW ni MCS estacionario sobre row_id.','Correlación de |residuo| con predicción solo orienta heterocedasticidad; p nominal no incorpora dependencia por perfiles/entrenamientos.','Una semilla, presupuesto reducido y una sola base; no valida generalización externa ni uso clínico.'])
 (out/'analisis.json').write_text(json.dumps(note,indent=2,ensure_ascii=False))
 for name in ['metricas_por_fold.csv','historial_propuestas.csv.gz','protocolo.json','particiones_externas.csv','particiones_internas.csv','COMPLETADO.json']:shutil.copy2(src/name,out/name)
 print(json.dumps(note,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
