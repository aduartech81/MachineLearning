"""Recuperar métricas existentes para E2; no entrena ni cambia V1/V2."""
from pathlib import Path
import argparse,hashlib,io,json,shutil,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--zip-evidencias',type=Path,required=True);a=p.parse_args()
 out=ROOT/'_static/cierre_entregable2';out.mkdir(exist_ok=True);records=[]
 with zipfile.ZipFile(a.zip_evidencias) as z:
  known={r['ruta']:r for r in json.loads(z.read('MANIFIESTO_RECOPILACION.json'))['archivos']}
  sources=[n for n in z.namelist() if n.endswith('/resultado.json') and ('/evaluacion_v2/' in n or '/evaluacion_externa_piloto/' in n)]
  assert len(sources)==540
  for name in sources:
   metric=name.replace('resultado.json','metricas.csv')
   for n in [name,metric]:
    data=z.read(n);assert len(data)==known[n]['bytes'] and hashlib.sha256(data).hexdigest()==known[n]['sha256']
   r=json.loads(z.read(name));m=pd.read_csv(io.BytesIO(z.read(metric)));selected=m.loc[m.variante=='Sigmoide_Youden_interno'];assert len(selected)==1
   key={k:r[k] for k in ['familia','optimizador','balanceo']};key['fold']=r['outer_fold']
   records.append(dict(**key,run_id=f"{r['familia']}__{r['optimizador']}__{r['balanceo']}__fold{r['outer_fold']}",**selected.iloc[0].to_dict(),semilla_global=42,semilla_busqueda=str(42+r['outer_fold']) if r['optimizador'] in ['bayes','genetica'] else 'propuestas_prefijadas_ver_espacios',parametros=json.dumps(r['parametros'],sort_keys=True),filas_ajuste=r['filas_ajuste'],filas_externas=r['filas_externas'],ajuste_final_s=r.get('segundos_ajuste_final_con_preprocesamiento',r.get('segundos_ajuste')),inferencia_con_calibracion_s=r.get('segundos_prediccion_con_calibracion',r.get('segundos_prediccion')),evaluacion_total_s=r.get('segundos_totales'),fuente=name,fuente_sha256=known[name]['sha256'],metricas_sha256=known[metric]['sha256']))
 df=pd.DataFrame(records).sort_values(['familia','optimizador','balanceo','fold']);keys=['familia','optimizador','balanceo']
 assert len(df)==540 and not df.duplicated(keys+['fold']).any();assert len(df.groupby(keys))==108 and (df.groupby(keys).size()==5).all()
 df=df.merge(pd.read_csv(ROOT/'_static/recuperacion_v2/tiempos_busquedas_540.csv'),on=keys+['fold'],validate='one_to_one')
 metrics=['auc','average_precision','accuracy','precision','sensibilidad','especificidad','f1','brier','log_loss']
 agg=df.groupby(keys)[metrics].agg(['mean','std']).reset_index();agg.columns=['_'.join(x).strip('_') for x in agg.columns]
 old=pd.read_csv(ROOT/'_static/proyecto_final_caries_v2/cobertura/resumen_108_configuraciones.csv');check=agg.merge(old,on=keys,validate='one_to_one')
 delta=float(np.max(np.abs(check.auc_mean-check.auc_medio)));assert delta<1e-12
 df.to_csv(out/'tabla_maestra_540_folds.csv',index=False);agg.to_csv(out/'resumen_108_media_de.csv',index=False)
 h=pd.read_csv(ROOT/'_static/recuperacion_v2/historial_propuestas.csv.gz');assert len(h)==8640
 fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
 for ax,opt in zip(axes.ravel(),['grid','random','bayes','genetica']):
  data=h.loc[h.optimizador==opt];means=data.groupby(['familia','propuesta']).mejor_auc_hasta_propuesta.mean()
  for fam in sorted(data.familia.unique()):
   s=means.loc[fam];ax.plot(s.index,s.values,label=fam,linewidth=1.5)
  ax.set_title(opt);ax.set_xlabel('Propuesta lógica (1–16)');ax.set_ylabel('Mejor AUC interno acumulado');ax.grid(alpha=.2)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
 fig.suptitle('Convergencia de búsqueda: media de ejecuciones aplicables por familia\nGrid/Random: orden lógico reconstruido, no cronología real; no es AUC externo',fontsize=12);fig.savefig(out/'convergencia_busquedas.png',dpi=150);plt.close(fig)
 bench=json.loads((ROOT/'proyecto_final_caries/resultados/benchmark_recursos_20261023.json').read_text());(out/'benchmark_recursos.json').write_text(json.dumps(bench,indent=2,ensure_ascii=False))
 method=dict(evaluaciones=540,configuraciones=108,folds_por_configuracion=5,variante='Sigmoide_Youden_interno; calibrador y umbral ajustados con OOF interno',archivo_origen=a.zip_evidencias.name,archivos_fuente_verificados=1080,diferencia_maxima_auc_resumen_previo=delta,limites=['Media y DE por cinco folds; no intervalos de confianza ni observaciones independientes.','Piloto logística Grid fold 0 conserva tiempos originales; evaluación total ausente permanece vacía.','Suma de componentes no equivale a tiempo de pared.','Grid/Random: orden lógico de propuestas, no cronología real.','108 combinaciones de clasificación; no se añadieron regresiones.','RAM de siete referencias finalistas en procesos nuevos; no del modelo propio ni de búsquedas históricas.'])
 (out/'metodo.json').write_text(json.dumps(method,indent=2,ensure_ascii=False));shutil.copytree(out,ROOT/'proyecto_final_caries/presentacion/evidencias/cierre_entregable2',dirs_exist_ok=True);print(json.dumps(method,ensure_ascii=False))
if __name__=='__main__':main()
