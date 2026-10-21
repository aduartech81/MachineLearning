"""Reproduce la auditoría descriptiva de desarrollo sin ajustar modelos."""
from pathlib import Path
import json,hashlib,re,html
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[2];b=r/'proyecto_final_caries/presentacion';out=b/'evidencias/cierre_auditoria';out.mkdir(exist_ok=True)
summary=json.loads((r/'entregable_2/results/summary.json').read_text());d=pd.read_csv(r/'entregable_2/results/train.csv');f=summary['features'];y=d['dental caries'];g=pd.util.hash_pandas_object(d[f],index=False);parts=pd.read_csv(r/'proyecto_final_caries/evaluacion/particiones_desarrollo.csv')
assert np.array_equal(g.to_numpy(),parts.perfil.to_numpy());assert np.array_equal(y,parts.objetivo)
assert parts.groupby('perfil').outer_validation_fold.nunique().max()==1
z=pd.DataFrame({'perfil':g,'y':y});discord=int((z.groupby('perfil').y.nunique()>1).sum())
rows=[]
for k in range(5):
 q=parts[parts.outer_validation_fold==k];rows.append({'fold_externo':k,'filas':len(q),'perfiles':q.perfil.nunique(),'positivos':int(q.objetivo.sum()),'prevalencia':q.objetivo.mean(),'filas_repetidas_adicionales':len(q)-q.perfil.nunique()})
t=pd.DataFrame(rows);t.to_csv(out/'particiones_resumen.csv',index=False)
num=[c for c in f if c not in summary['categorical_features']]
desc=d.groupby('dental caries')[num].agg(['count','mean','median','std','min','max']).T.reset_index();desc.columns=['variable','estadistico','sin_caries','con_caries'];desc.to_csv(out/'descriptivos_por_clase.csv',index=False)
meta={'filas_desarrollo':len(d),'perfiles':int(g.nunique()),'perfiles_con_etiquetas_discordantes':discord,'filas_repetidas_adicionales':int(len(d)-g.nunique()),'positivos':int(y.sum()),'faltantes_predictores':int(d[f].isna().sum().sum()),'perfil':'Hash pandas de los 23 predictores finales en el orden de summary.json; excluye ID y objetivo','comprobacion':'Hashes y etiquetas coinciden con particiones conservadas; cada perfil aparece en un único fold externo','datos_sha256':hashlib.sha256((r/'entregable_2/results/train.csv').read_bytes()).hexdigest(),'alcance':'Descriptivo sobre desarrollo; no implica independencia por paciente; no se consultaron nuevos resultados de test'}
(out/'auditoria_datos.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,c in zip(axes,['age','hemoglobin','waist(cm)']):
 edges=np.histogram_bin_edges(d[c].dropna(),bins=20)
 for label,color in [(0,'#147d92'),(1,'#d48632')]:
  a=d.loc[y==label,c];ax.hist(a,bins=edges,density=True,histtype='step',linewidth=2,color=color,label='Sin caries' if label==0 else 'Con caries')
 ax.set_title(c);ax.set_ylabel('Densidad por clase');ax.legend(fontsize=8)
fig.suptitle('Distribuciones en desarrollo · todas las filas, sin eliminar extremos');fig.tight_layout();fig.savefig(out/'eda_desarrollo.png',dpi=150);plt.close(fig)
