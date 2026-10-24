"""Demostración real con artefactos finales; no entrena ni usa datos de prueba."""
from pathlib import Path
import json,hashlib,html
from functools import lru_cache
from flask import request,jsonify
BASE=Path(__file__).resolve().parents[1]/'artefactos_finales_20261023_v3'
LABELS={'gender':'Sexo (código del archivo)','age':'Edad','height(cm)':'Estatura (cm)','weight(kg)':'Peso (kg)','waist(cm)':'Cintura (cm)','eyesight(left)':'Medición visual izquierda','eyesight(right)':'Medición visual derecha','hearing(left)':'Audición izquierda (código)','hearing(right)':'Audición derecha (código)','systolic':'Presión sistólica registrada','relaxation':'Presión de relajación registrada','fasting blood sugar':'Glucemia registrada en ayunas','Cholesterol':'Colesterol registrado','triglyceride':'Triglicéridos registrados','HDL':'HDL registrado','LDL':'LDL registrado','hemoglobin':'Hemoglobina registrada','Urine protein':'Proteína en orina (código)','serum creatinine':'Creatinina sérica registrada','AST':'AST registrado','ALT':'ALT registrado','Gtp':'GTP registrado','smoking':'Tabaquismo (código N/Y)'}

def integrar(server):
 @lru_cache(maxsize=8)
 def cargar(nombre):
  import joblib
  meta=json.loads((BASE/'manifest.json').read_text());record=next(r for r in meta['modelos'] if r['familia']==nombre);path=BASE/(nombre+'.joblib');assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'];return joblib.load(path)
 @server.route('/predecir/',methods=['GET'])
 def formulario():
  if not (BASE/'manifest.json').exists():return 'Artefactos finales aún no disponibles',503
  meta=json.loads((BASE/'manifest.json').read_text());fields=[]
  for name,s in meta['esquema'].items():
   label=html.escape(name);display=html.escape(LABELS.get(name,name));val=html.escape(str(s['ejemplo']))
   if s['valores'] is not None:
    options=''.join(f'<option value="{html.escape(str(v))}"'+(' selected' if str(v)==str(s['ejemplo']) else '')+'>'+html.escape(str(v))+'</option>' for v in s['valores']);input_=f'<select name="{label}">{options}</select>'
   else:input_=f'<input name="{label}" type="number" step="any" min="{s["min"]}" max="{s["max"]}" value="{val}" required>'
   fields.append(f'<label title="{label}">{display}{input_}</label>')
  models=''.join('<option>'+r['familia']+'</option>' for r in meta['modelos'])
  return '''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Predicción académica de caries</title><style>body{font:17px/1.5 system-ui;max-width:1000px;margin:30px auto;padding:20px;color:#163246}label{display:flex;flex-direction:column;gap:4px}input,select,button{font:inherit;padding:8px}#campos{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}button{margin:20px 0;background:#086b7b;color:white;border:0;border-radius:5px}pre{white-space:pre-wrap;background:#eef4f7;padding:20px}</style><a href="/" target="_top">Volver al proyecto</a><h1>Explorar una predicción</h1><p>Demostración académica de clasificación contemporánea. No diagnostica ni predice caries futura. Los valores iniciales son un perfil sintético con medianas/modas del desarrollo; no representan un paciente.</p><p>Se calcula la salida original del modelo final. No se ha establecido calibración nueva ni umbral clínico. Los rangos permitidos corresponden al desarrollo y no garantizan validez clínica.</p><form id="form"><label>Modelo<select name="modelo">'''+models+'''</select></label><div id="campos">'''+''.join(fields)+'''</div><button>Calcular con el modelo</button></form><div id="resultado" role="status"></div><script>document.getElementById('form').onsubmit=async function(ev){ev.preventDefault();let d=Object.fromEntries(new FormData(this));let modelo=d.modelo;delete d.modelo;let el=document.getElementById('resultado');el.textContent='Calculando…';try{let response=await fetch('/api/prediccion-final',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({modelo,datos:d})});let r=await response.json();el.textContent='';let pre=document.createElement('pre');let lines=[];if(r.error){lines.push(r.error)}else{lines.push('Modelo: '+r.modelo);if(r.probabilidad_original_sin_calibracion_nueva!==null){lines.push('Salida original: '+(100*r.probabilidad_original_sin_calibracion_nueva).toFixed(2)+' % (sin calibración nueva)')}else{lines.push('Score original: '+r.score_original.toFixed(5)+' (no es una probabilidad)')}let e=r.explicacion_local;if(e){lines.push('','Explicación del modelo propio — margen aditivo:');lines.push('Intercepto: '+e.intercepto.toFixed(5));for(let t of e.principales_terminos){lines.push(t.termino+': '+(t.contribucion>=0?'+':'')+t.contribucion.toFixed(5))}lines.push('','Suma de todos los términos + intercepto: '+e.margen_reconstruido.toFixed(5));lines.push('Las contribuciones explican la salida matemática; no establecen causas.')}lines.push('',r.advertencia)}pre.textContent=lines.join('\\n');el.appendChild(pre)}catch(e){el.textContent='No se pudo calcular. Revisa la conexión y el servidor.'}};</script></html>'''
 @server.route('/api/prediccion-final',methods=['POST'])
 def calcular():
  import numpy as np
  import pandas as pd
  try:
   meta=json.loads((BASE/'manifest.json').read_text());payload=request.get_json(silent=True) or {};name=payload.get('modelo');valid=[r['familia'] for r in meta['modelos']]
   if name not in valid:return jsonify(error='Selecciona un modelo disponible'),400
   data=payload.get('datos',{})
   if set(data)!=set(meta['features']):return jsonify(error='Debes aportar exactamente los 23 predictores, sin ID ni variable objetivo'),400
   row={}
   for c,s in meta['esquema'].items():
    v=float(data[c]) if s['tipo']=='numero' else str(data[c])
    if s['tipo']=='numero' and (not np.isfinite(v) or v<s['min'] or v>s['max']):return jsonify(error=f'{c}: valor fuera del rango de desarrollo'),400
    if s['valores'] is not None and v not in s['valores']:return jsonify(error=f'{c}: categoría no disponible'),400
    row[c]=v
   X=pd.DataFrame([row],columns=meta['features']);m=cargar(name);explanation=None
   if name=='Propio':
    z=m._transform(X)[0];contrib=z*m.coef_;score=float(m.decision_function(X)[0]);prob=float(m.predict_proba(X)[0,1]);indices=np.argsort(-np.abs(contrib))[:8]
    base_names=list(m.preprocessor_.get_feature_names_out());names=base_names+['interaccion_'+m.numeric_[i]+'×'+m.numeric_[j] for i,j in m.selected_pairs_]
    explanation={'escala':'Margen bruto aditivo antes de sigmoide','intercepto':float(m.intercept_),'suma_todas_contribuciones':float(contrib.sum()),'margen_reconstruido':float(m.intercept_+contrib.sum()),'error_aditividad':float(abs(score-m.intercept_-contrib.sum())),'principales_terminos':[dict(termino=names[i],contribucion=float(contrib[i])) for i in indices],'limite':'Términos transformados; no causalidad. Se muestran ocho, la suma usa todos.'}
   else:
    from proyecto_final_caries.experimentos_v2.evaluar_seleccion import puntuar
    values,p=puntuar(m,X);score=float(values[0]);prob=float(p[0]) if p is not None else None
   return jsonify(modelo=name,score_original=score,probabilidad_original_sin_calibracion_nueva=prob,explicacion_local=explanation,advertencia='Uso académico. La salida no establece diagnóstico ni utilidad clínica; no se fija umbral clínico.')
  except (ValueError,TypeError,KeyError):return jsonify(error='Entradas inválidas; revisa tipos y categorías'),400
  except ImportError:return jsonify(error='El servidor necesita las dependencias de inferencia del proyecto final'),503
