"""Tablas maestras, metadatos por artefacto y figuras esquemáticas del cierre."""
from pathlib import Path
import json,hashlib,shutil,sys,tempfile,math
import numpy as np
import pandas as pd
import joblib
from proyecto_final_caries.experimentos_v2.evaluar_seleccion import puntuar
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'_static/entregable3'
def clean_json(x):
 if isinstance(x,dict):return {str(k):clean_json(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean_json(v) for v in x]
 if hasattr(x,'item'):return clean_json(x.item())
 if isinstance(x,float) and not math.isfinite(x):return str(x)
 return x

def svg_flow(filename,title,labels,footer):
 width=1100;boxw=235;gap=33;x0=20;y=75;height=210;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{title}"><rect width="1100" height="210" fill="#f3f8f8" rx="16"/><text x="24" y="34" font-family="Arial" font-size="22" fill="#173747" font-weight="bold">{title}</text><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#147c86"/></marker></defs>']
 for i,(head,line1,line2) in enumerate(labels):
  x=x0+i*(boxw+gap);parts.append(f'<rect x="{x}" y="{y}" width="{boxw}" height="88" rx="12" fill="white" stroke="#bfd5dc"/><text x="{x+13}" y="{y+25}" font-family="Arial" font-size="17" font-weight="bold" fill="#0e7882">{head}</text><text x="{x+13}" y="{y+49}" font-family="Arial" font-size="14" fill="#173747">{line1}</text><text x="{x+13}" y="{y+69}" font-family="Arial" font-size="14" fill="#173747">{line2}</text>')
  if i<3:parts.append(f'<path d="M{x+boxw+3},{y+44} H{x+boxw+gap-7}" stroke="#147c86" stroke-width="2" marker-end="url(#arrow)"/>')
 parts.append(f'<text x="24" y="193" font-family="Arial" font-size="13" fill="#4b6875">{footer}</text></svg>');(OUT/filename).write_text(''.join(parts))

def main():
 OUT.mkdir(parents=True,exist_ok=True);summary=pd.read_csv(ROOT/'_static/cierre_entregable2/resumen_108_media_de.csv');chosen=summary.loc[summary.groupby('familia').auc_mean.idxmax()].sort_values('auc_mean',ascending=False);chosen.to_csv(OUT/'seleccion_7_referencias.csv',index=False)
 c=pd.DataFrame(dict(tarea='Clasificación',familia=summary.familia,optimizador=summary.optimizador,balanceo=summary.balanceo,estado='Evaluada',metrica='AUC',media=summary.auc_mean,desviacion=summary.auc_std,folds=5,semilla=42,version='V2'))
 reg=pd.read_csv(ROOT/'_static/cierre_entregable2/reg_resumen_28.csv');r=pd.DataFrame(dict(tarea='Regresión auxiliar',familia=reg.familia,optimizador=reg.optimizador,balanceo='No aplica',estado='Evaluada',metrica='RMSE',media=reg.rmse_mean,desviacion=reg.rmse_std,folds=5,semilla=42,version='Regresión 23/10'))
 na=pd.DataFrame([dict(tarea='Clasificación',familia='KNN',optimizador=o,balanceo='pesos',estado='Inaplicable: class_weight no disponible; distance no lo sustituye',metrica='AUC',media=None,desviacion=None,folds=0,semilla=42,version='V2') for o in ['grid','random','bayes','genetica']])
 ownvalues=[json.loads((ROOT/f'proyecto_final_caries/resultados/comparacion_v1/Propio/fold_{k}/resultado.json').read_text())['auc_externo'] for k in range(5)];own=pd.DataFrame([dict(tarea='Clasificación',familia='Propio',optimizador='grid',balanceo='sin_balanceo',estado='Evaluada',metrica='AUC',media=float(np.mean(ownvalues)),desviacion=float(np.std(ownvalues,ddof=1)),folds=5,semilla=42,version='V1')]);master=pd.concat([c,na,r,own],ignore_index=True);assert len(master)==141;master.to_csv(OUT/'tabla_maestra_140_nominales_y_propio.csv',index=False)
 src=ROOT/'proyecto_final_caries/artefactos_finales_20261023_v3';manifest=json.loads((src/'manifest.json').read_text());df=pd.read_csv(ROOT/'entregable_2/results/train.csv');rng=np.random.default_rng(42);caseids=np.sort(rng.choice(len(df),128,replace=False));probe=df.iloc[caseids][manifest['features']];technical=[]
 for record in manifest['modelos']:
  name=record['familia'];file=src/(name+'.joblib');assert hashlib.sha256(file.read_bytes()).hexdigest()==record['sha256'];m=joblib.load(file)
  before=m.decision_function(probe) if name=='Propio' else puntuar(m,probe)[0]
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'modelo.joblib';joblib.dump(m,path,compress=3);loaded=joblib.load(path);after=loaded.decision_function(probe) if name=='Propio' else puntuar(loaded,probe)[0];assert np.array_equal(before,after),name
  target=ROOT/'proyecto_final_caries/artifacts'/name;target.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target/'model.joblib');recordcv=chosen[chosen.familia==name];metrics=recordcv.iloc[0].to_dict() if len(recordcv) else dict(auc_mean=float(np.mean(ownvalues)),auc_std=float(np.std(ownvalues,ddof=1)),version='V1')
  params=m.get_params() if name=='Propio' else m['modelo'].get_params();meta=dict(familia=name,semilla=42,hiperparametros_finales=params,esquema=manifest['esquema'],columnas=manifest['features'],versiones=manifest['versiones'],metricas_externas_historicas=metrics,artefacto_sha256=record['sha256'],n_ajuste=44582,regla_consolidacion=manifest['regla'],roundtrip=dict(casos=128,ids_desarrollo=caseids.tolist(),exactamente_iguales=True,alcance='Verificación técnica de serialización sobre un conjunto fijo de desarrollo; no estimación de generalización.'),calibracion='No se ajustó calibrador final nuevo; salida original de cada algoritmo.',limites=manifest['limites']);(target/'metadata.json').write_text(json.dumps(clean_json(meta),indent=2,ensure_ascii=False,allow_nan=False));technical.append(dict(familia=name,casos=128,roundtrip_identico=True,sha256=record['sha256'],bytes=file.stat().st_size))
 pd.DataFrame(technical).to_csv(OUT/'roundtrip_8_modelos_128_casos.csv',index=False)
 svg_flow('flujo_validacion.svg','Evaluación sin aprender del pliegue externo', [('Datos de desarrollo','44.582 filas del chequeo','23 predictores en clasificación'),('Agrupar perfiles','Copias de entrada juntas','Perfil no identifica paciente'),('Selección interna','Tres folds en entrenamiento','Transformaciones dentro del ajuste'),('Evaluación externa','Cinco folds de validación','Sin nuevo test independiente')],'Las mismas particiones se conservan; las representaciones propias de cada algoritmo se documentan.')
 svg_flow('flujo_modelo.svg','Modelo aditivo implementado: del perfil al score', [('Entrada','23 predictores permitidos','Sin ID, oral, tartar ni objetivo'),('Representación','Splines y categorías','Hasta cinco interacciones'),('Aprendizaje','BCE + ranking + L2','L-BFGS-B y gradiente propio'),('Salida','Score y sigmoide original','Explicación aditiva del margen')],'La convexidad se demuestra con representación y pares fijos; no para la selección previa de interacciones.')
 svg_flow('arquitectura_web.svg','Separación entre entrenamiento y aplicación pública', [('Fuera del servidor web','Selección y ajuste final','Guardar modelos y metadatos'),('Repositorio','Código y artefactos','Versiones y hashes verificables'),('Render + Dash','Carga de modelos guardados','Validación de las entradas'),('Persona usuaria','Exploración y predicción','Sin reentrenamiento en línea')],'El libro fechado conserva la entrega anterior; el portal presenta el estado actual del proyecto.')
 shutil.copy2(ROOT/'proyecto_final_caries/presentacion/assets/portada_caries.png',OUT/'portada_caries.png');print('141 filas nominales + propio, ocho metadatos y roundtrip de 128 casos verificados.')
if __name__=='__main__':main()
