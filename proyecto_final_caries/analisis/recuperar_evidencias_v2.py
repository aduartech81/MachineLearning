from pathlib import Path
import argparse, hashlib, json, re, zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--fuente',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);a=p.parse_args()
 src=a.fuente;repo=a.repo;out=a.salida;out.mkdir(parents=True,exist_ok=True)
 read=lambda p:json.loads(p.read_text())
 manifest=read(src/'MANIFIESTO_RECOPILACION.json')
 for f in manifest['archivos']:
  b=(src/f['ruta']).read_bytes();assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256'],f['ruta']
 base=src/'proyecto_final_caries/experimentos_v2'
 rows=[];history=[];warn={};records=[]
 for sel in sorted(base.glob('resultados_*/*/*/seleccion_interna.json')):
  folder=sel.parent;block=folder.parent.parent.name.removeprefix('resultados_');opt,bal=block.split('_',1);s=read(sel)
  key=dict(optimizador=opt,balanceo=bal,familia=folder.parent.name,fold=int(folder.name.split('_')[1]))
  fits=[]
  for f in sorted(folder.glob('*_inner_*.json')):
   if f.name.startswith('preprocesamiento'):continue
   r=read(f);assert r['estado']=='OK';fits.append((f,r))
   for w in r['advertencias']:warn[w]=warn.get(w,0)+1
   records.append({**key,'archivo':f.name,**{k:r[k] for k in ('auc','segundos_ajuste','segundos_prediccion')},'parametros':json.dumps(r['parametros'],sort_keys=True)})
  pre=sum(read(f)['segundos'] for f in folder.glob('preprocesamiento_inner_*.json'))
  unique=set(json.dumps(r['parametros'],sort_keys=True) for _,r in fits)
  assert len(fits)==len(unique)*3
  cum=pre;best=-np.inf
  for i in range(16):
   if opt in ('grid','random'):
    fs=[r for f,r in fits if f.name.startswith(f'candidato_{i:02d}_')];assert len(fs)==3
    auc=np.mean([r['auc'] for r in fs]);params=fs[0]['parametros'];reused=False;proposal=0.
   else:
    trial=read(folder/f'trial_{i:02d}.json');auc=trial['auc_medio'];params=trial['parametros'];reused=trial['reutilizado_de'] is not None
    fs=[r for f,r in fits if f.name.startswith(f'trial_{i:02d}_')];assert len(fs)==(0 if reused else 3)
    proposal=trial['segundos_propuesta']
   cost=sum(r['segundos_ajuste']+r['segundos_prediccion'] for r in fs);cum+=cost+proposal;best=max(best,auc)
   history.append({**key,'propuesta':i+1,'reutilizada':reused,'auc_interno':auc,'mejor_auc_hasta_propuesta':best,'segundos_componentes_acumulados':cum,'parametros':json.dumps(params,sort_keys=True)})
  assert np.isclose(best,s['auc_interno'],rtol=0,atol=1e-12)
  rows.append({**key,'propuestas':16,'configuraciones_unicas':len(unique),'ajustes_internos_registrados':len(fits),'segundos_preprocesamiento':pre,'segundos_ajuste':sum(r['segundos_ajuste'] for _,r in fits),'segundos_prediccion':sum(r['segundos_prediccion'] for _,r in fits),'segundos_componentes_busqueda':cum,'auc_interno_seleccionado':best})
 t=pd.DataFrame(rows);assert len(t)==540
 t.to_csv(out/'tiempos_busquedas_540.csv',index=False)
 pd.DataFrame(history).to_csv(out/'historial_propuestas.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 pd.DataFrame(records).to_csv(out/'ajustes_internos.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 t.groupby(['familia','optimizador','balanceo']).agg(folds=('fold','count'),unicas_media=('configuraciones_unicas','mean'),ajustes=('ajustes_internos_registrados','sum'),segundos_busqueda_componentes=('segundos_componentes_busqueda','sum'),auc_interno_medio=('auc_interno_seleccionado','mean')).reset_index().to_csv(out/'costes_108_configuraciones.csv',index=False)
 byopt=t.groupby('optimizador').agg(busquedas=('fold','size'),propuestas=('propuestas','sum'),configuraciones_unicas=('configuraciones_unicas','sum'),ajustes=('ajustes_internos_registrados','sum'),segundos=('segundos_componentes_busqueda','sum')).reset_index();byopt.to_csv(out/'presupuesto_efectivo.csv',index=False)
 # Costes de evaluación externa; el piloto conserva un esquema de tiempos distinto.
 external=[]
 for pth in sorted(base.glob('evaluacion_v2/*/*/*/resultado.json'))+sorted(base.glob('evaluacion_externa_piloto/*/*/resultado.json')):
  rr=read(pth);external.append({k:v for k,v in rr.items() if k in ('familia','optimizador','balanceo','outer_fold','auc_score_original') or k.startswith('segundos_')})
 assert len(external)==540
 pd.DataFrame(external).to_csv(out/'tiempos_evaluaciones_540.csv',index=False)
 # Verificación de las 35 decisiones finales; sólo las predicciones externas.
 evidence=repo/'proyecto_final_caries/presentacion/evidencias/revision_v2_20261020_154421_004423'
 decisions=pd.read_csv(evidence/'comparacion/decisiones_solo_validacion_interna.csv')
 partition=pd.read_csv(repo/'proyecto_final_caries/evaluacion/particiones_desarrollo.csv').set_index('row_id')
 arrays={};cal=[];bins=[];ext_times=[]
 for r in decisions.itertuples():
  f=src/r.seleccion_origen;assert hashlib.sha256(f.read_bytes()).hexdigest()==r.seleccion_sha256
  folder=base/'evaluacion_v2'/f'{r.optimizador}_{r.balanceo}'/r.familia/f'fold_{r.fold}'
  df=pd.read_csv(folder/'predicciones.csv',float_precision='round_trip').sort_values('row_id')
  ids=partition.index[partition.outer_validation_fold==r.fold];assert set(df.row_id)==set(ids) and df.row_id.is_unique
  expected=partition.loc[df.row_id];assert np.array_equal(df.actual,expected.objetivo) and np.array_equal(df.perfil,expected.perfil)
  internal=pd.read_csv(folder/'predicciones_internas.csv',usecols=['row_id','perfil'])
  assert set(internal.row_id)==set(partition.index)-set(df.row_id)
  assert not set(internal.perfil).intersection(df.perfil)
  assert np.isfinite(df.probability_sigmoide).all()
  arrays.setdefault(r.familia,[]).append(df)
  result=read(folder/'resultado.json');ext_times.append(dict(familia=r.familia,fold=r.fold,**{k:v for k,v in result.items() if k.startswith('segundos_')}))
 for fam,dfs in arrays.items():
  df=pd.concat(dfs).sort_values('row_id');assert len(df)==44582 and df.row_id.is_unique
  for column,label in [('probability_original','Original'),('probability_sigmoide','Sigmoide'),('probability_isotonica','Isotonica'),('dummy_probability','Dummy_entrenamiento')]:
   if column not in df:continue
   y=df.actual.to_numpy();pr=df[column].to_numpy();assert np.isfinite(pr).all() and ((pr>=0)&(pr<=1)).all()
   idx=np.minimum((pr*10).astype(int),9);ece=0.
   for k in range(10):
    mask=idx==k
    if mask.any():
     mean=float(pr[mask].mean());obs=float(y[mask].mean());ece+=mask.mean()*abs(mean-obs)
     bins.append(dict(familia=fam,variante=label,bin=k,n=int(mask.sum()),probabilidad_media=mean,fraccion_positiva=obs))
   cal.append(dict(familia=fam,variante=label,n=len(y),auc_oof=roc_auc_score(y,pr),brier=brier_score_loss(y,pr),log_loss=log_loss(y,pr),ece_10_bins=ece))
 calibration=pd.DataFrame(cal);calibration.to_csv(out/'calibracion_antes_despues.csv',index=False);pd.DataFrame(bins).to_csv(out/'calibracion_bins.csv',index=False);pd.DataFrame(ext_times).to_csv(out/'tiempos_finalistas.csv',index=False)
 fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for label,color in [('Original','#b14c36'),('Sigmoide','#147d92'),('Isotonica','#7353a6')]:
  sub=calibration[calibration.variante==label].set_index('familia');fams=sorted(arrays)
  for ax,metric in zip(axs,['brier','log_loss']):ax.plot([sub.loc[f,metric] if f in sub.index else np.nan for f in fams],range(len(fams)),marker='o',linestyle='none',label=label,color=color)
 for ax,metric in zip(axs,['Brier (menor es mejor)','Log-loss (menor es mejor)']):
  ax.set_yticks(range(len(fams)),fams);ax.set_xlabel(metric);ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Calibración: predicciones de los cinco folds externos\nSVM: original no disponible como probabilidad',fontsize=12);fig.savefig(out/'calibracion_antes_despues.png',dpi=150);plt.close(fig)
 method={'archivos_sha256_verificados':len(manifest['archivos']),'busquedas':len(t),'propuestas_nominales':int(t.propuestas.sum()),'ajustes_internos_registrados':int(t.ajustes_internos_registrados.sum()),'configuraciones_unicas_por_busqueda_sumadas':int(t.configuraciones_unicas.sum()),'predicciones_externas':35,'predicciones_internas':35,'advertencias_busqueda':warn,'limitaciones':['Tiempos medidos por componentes; no equivalen a tiempo de pared ni benchmark aislado.','Grid/Random ejecutan por fold interno: la curva por propuesta se reconstruye con sus tres ajustes y no refleja la cronología real.','El tiempo de propuesta genética registrado no incluye toda la generación de descendencia; no permite comparar overhead total de optimizadores.','Memoria máxima histórica no medida; no se infiere a partir de duración ni tamaño del archivo.','Calibradores ajustados en OOF interno, métricas calculadas en folds externos; folds comparten entrenamiento.','Los 35 finalistas resultan de selección interna entre métodos; V2 sigue siendo exploratoria.']}
 (out/'metodo.json').write_text(json.dumps(method,ensure_ascii=False,indent=2))
 print(byopt.to_string(index=False));print(calibration.to_string(index=False));print(json.dumps({k:v for k,v in method.items() if k not in ['advertencias_busqueda','limitaciones']}))
if __name__=='__main__':main()
