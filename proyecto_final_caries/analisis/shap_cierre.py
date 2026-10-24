"""Explica una referencia XGBoost; no el modelo propio."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import pandas as pd
import joblib,shap,xgboost
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--modelo',type=Path,required=True);p.add_argument('--salida',type=Path,required=True);a=p.parse_args();out=a.salida;out.mkdir(parents=True,exist_ok=True)
digest=hashlib.sha256(a.modelo.read_bytes()).hexdigest();assert digest==json.loads(a.modelo.with_name('resultado.json').read_text())['artefactos_sha256'][a.modelo.name]
b=joblib.load(a.modelo);root=Path(__file__).resolve().parents[2];df=pd.read_csv(root/'entregable_2/results/train.csv');parts=pd.read_csv(root/'proyecto_final_caries/evaluacion/particiones_desarrollo.csv');df.index=parts.row_id
ids=parts.loc[parts.outer_validation_fold==0,'row_id'].sample(500,random_state=42).to_numpy();X=df.loc[ids,b['features']];V=b['despues'].transform(b['antes'].transform(X));V=V.toarray() if hasattr(V,'toarray') else V
names=list(b['despues'].get_feature_names_out());e=shap.TreeExplainer(b['modelo'])(V);e.feature_names=names
margin=b['modelo'].get_booster().predict(xgboost.DMatrix(V),output_margin=True);error=float(np.max(np.abs(e.values.sum(1)+e.base_values-margin)));assert error<1e-4
shap.plots.beeswarm(e,max_display=16,show=False);plt.title('XGBoost de referencia, fold 0: SHAP en margen bruto');plt.tight_layout();plt.savefig(out/'shap_beeswarm_xgb.png',dpi=140,bbox_inches='tight');plt.close()
i=int(np.argmax(margin));shap.plots.waterfall(e[i],max_display=13,show=False);plt.title(f'Caso externo row_id={ids[i]}: margen bruto');plt.tight_layout();plt.savefig(out/'shap_waterfall_xgb.png',dpi=140,bbox_inches='tight');plt.close()
(out/'shap_metodo.json').write_text(json.dumps(dict(modelo='XGBoost grid pesos fold 0',sha256=digest,n=500,semilla=42,caso_waterfall=int(ids[i]),criterio_caso='Máximo margen en muestra aleatoria de validación externa; ilustrativo',escala='Margen bruto antes de sigmoide y calibración',error_aditividad_max=error,limites=['No explica al modelo propio','Columnas transformadas; no causalidad','Un fold y una muestra; no estudio de estabilidad']),indent=2,ensure_ascii=False))
print('SHAP verificado, error máximo',error)
