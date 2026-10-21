"""DeLong por filas y efectos descriptivos: complemento, no sustituto del bootstrap agrupado."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import norm,rankdata,false_discovery_control
from sklearn.metrics import roc_auc_score

def componentes(y,s):
 pos=s[y==1];neg=s[y==0];m=len(pos);n=len(neg)
 q=np.sort(neg);a=(np.searchsorted(q,pos,'left')+np.searchsorted(q,pos,'right'))/(2*n)
 q=np.sort(pos);b=1-(np.searchsorted(q,neg,'left')+np.searchsorted(q,neg,'right'))/(2*m)
 return a,b

def holm(p):
 order=np.argsort(p);v=np.minimum(1,np.maximum.accumulate(np.asarray(p)[order]*(len(p)-np.arange(len(p)))));out=np.empty(len(p));out[order]=v;return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--fuente',type=Path,required=True);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--salida',type=Path,required=True);a=parser.parse_args();a.salida.mkdir(parents=True,exist_ok=True)
 ev=a.repo/'proyecto_final_caries/presentacion/evidencias/revision_v2_20261020_154421_004423'
 decisions=pd.read_csv(ev/'comparacion/decisiones_solo_validacion_interna.csv');frames={}
 for r in decisions.itertuples():
  p=a.fuente/'proyecto_final_caries/experimentos_v2/evaluacion_v2'/f'{r.optimizador}_{r.balanceo}'/r.familia/f'fold_{r.fold}'/'predicciones.csv'
  f=pd.read_csv(p,float_precision='round_trip');f['score']=f.probability_sigmoide;frames.setdefault(r.familia,[]).append(f)
 frames={k:pd.concat(v).set_index('row_id').sort_index() for k,v in frames.items()}
 for k in ('Propio','Sin_interacciones','Sin_ranking','Aditivo'):
  frames[k]=pd.read_csv(a.repo/f'proyecto_final_caries/resultados/analisis_20261018_204847_425225/oof_{k}.csv',float_precision='round_trip').set_index('row_id').sort_index()
 ref=frames['Propio'];y=ref.actual.to_numpy();s=ref.score.to_numpy();ap,bp=componentes(y,s)
 # Comprobación independiente con empates y comparación explícita de pares.
 yt=np.array([1,1,1,0,0,0]);st=np.array([.2,.5,.5,.2,.4,.8]);x,z=componentes(yt,st);kernel=(st[yt==1,None]>st[yt==0]).astype(float)+.5*(st[yt==1,None]==st[yt==0]);assert np.allclose(x,kernel.mean(1)) and np.allclose(z,kernel.mean(0))
 folds=pd.read_csv(ev/'contrastes/auc_por_fold.csv').set_index('fold');rows=[]
 for name,f in frames.items():
  if name=='Propio':continue
  assert f.index.equals(ref.index) and np.array_equal(f.actual,y)
  score=f.score.to_numpy();aq,bq=componentes(y,score);delta=ap.mean()-aq.mean();assert np.isclose(delta,roc_auc_score(y,s)-roc_auc_score(y,score),atol=1e-12)
  variance=np.var(ap-aq,ddof=1)/len(ap)+np.var(bp-bq,ddof=1)/len(bp);se=np.sqrt(variance);pv=2*norm.sf(abs(delta)/se)
  v=folds.Propio.to_numpy();w=folds[name].to_numpy();diff=v-w;nonzero=diff!=0;ranks=rankdata(abs(diff[nonzero]));rr=float(np.sum(ranks*np.sign(diff[nonzero]))/ranks.sum()) if ranks.size else 0.
  rows.append(dict(comparacion='Propio - '+name,delta_auc_oof=delta,ic95_filas_inferior=delta-1.95996398454*se,ic95_filas_superior=delta+1.95996398454*se,p_delong_filas=pv,delta_auc_medio_folds=diff.mean(),cliff_marginal_5x5=np.sign(v[:,None]-w).mean(),biserial_rangos_pareados=rr,cohen_dz_descriptivo=diff.mean()/diff.std(ddof=1)))
 result=pd.DataFrame(rows);result['p_holm_10_filas']=holm(result.p_delong_filas);result['q_bh_10_filas']=false_discovery_control(result.p_delong_filas.to_numpy())
 result.to_csv(a.salida/'delong_y_efectos_sensibilidad.csv',index=False)
 (a.salida/'metodo_contrastes.json').write_text(json.dumps({'alcance':'Sensibilidad exploratoria. DeLong clásico trata filas como independientes: supuesto no satisfecho por perfiles repetidos y entrenamiento solapado. No usar sus p como confirmación.','estimando_delong':'AUC OOF agrupado, no promedio de AUC de folds del análisis principal.','comparaciones':'Los mismos diez contrastes del bootstrap previo; selección ya exploratoria.','multiplicidad':'Holm de diez como referencia; BH de diez secundario. No intercambiar selectivamente.','efectos':'Cliff describe dominancia marginal 5x5 y no respeta pareamiento; se acompaña de biserial de rangos pareados y dz. Sólo cinco folds dependientes: no se calculan IC ni etiquetas de magnitud universales.','conclusion':'El bootstrap agrupado de perfiles continúa siendo el análisis principal condicional. Ningún análisis incorpora toda la variabilidad de entrenamiento.','referencias':['https://pubmed.ncbi.nlm.nih.gov/3203132/','https://doi.org/10.1037/0033-2909.114.3.494']},ensure_ascii=False,indent=2))
 print(result.to_string(index=False))
if __name__=='__main__':main()
