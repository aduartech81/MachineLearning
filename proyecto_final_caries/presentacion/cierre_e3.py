"""Cierre visual E3 y consulta de evidencias ya calculadas; sin entrenamiento."""
from pathlib import Path
import json,hashlib,mimetypes,zipfile
from io import BytesIO
import pandas as pd
from dash import html
from flask import abort,send_from_directory,send_file

def integrar(app,secciones):
 root=Path(__file__).resolve().parent;repo=root.parents[1];data=root/'evidencias/entregable3';book=root/'documentacion/entregable3_20261024';mapping=json.loads((book/'recursos.json').read_text())
 with zipfile.ZipFile(book/'estructura.zip') as z:embedded=set(z.namelist())
 allowed=set(mapping)|embedded
 @app.server.route('/evidencias-e3/<nombre>')
 def evidencia_cierre_e3(nombre):
  if nombre not in {p.name for p in data.iterdir() if p.is_file()}:abort(404)
  return send_from_directory(data,nombre,as_attachment=not nombre.endswith(('.png','.svg')))
 @app.server.route('/entregas/entregable-3-20261024/',defaults={'nombre':'index.html'})
 @app.server.route('/entregas/entregable-3-20261024/<path:nombre>')
 def libro_tercera_entrega(nombre):
  if nombre not in allowed:abort(404)
  if nombre in mapping:
   entry=mapping[nombre];payload=(repo/entry['archivo']).read_bytes()
   if hashlib.sha256(payload).hexdigest()!=entry['sha256']:abort(409)
  else:
   with zipfile.ZipFile(book/'estructura.zip') as z:payload=z.read(nombre)
  return send_file(BytesIO(payload),download_name=Path(nombre).name,mimetype=mimetypes.guess_type(nombre)[0] or 'application/octet-stream')
 previous=secciones['modelo'];context=secciones['problema']
 def modelo():
  selected=pd.read_csv(data/'seleccion_7_referencias.csv')[['familia','optimizador','balanceo','auc_mean','auc_std']];names=['Familia','Búsqueda','Balanceo','AUC externo medio','DE entre folds'];tab=html.Table([html.Thead(html.Tr([html.Th(n) for n in names])),html.Tbody([html.Tr([html.Td(f'{v:.6f}' if isinstance(v,float) else str(v)) for v in row]) for row in selected.itertuples(index=False,name=None)])])
  return html.Div([html.Section([html.P('CIERRE DEL PROYECTO',className='pf-kicker'),html.H2('Modelos finales y predicción'),html.P('Siete referencias seleccionadas por media externa del E2 y el modelo propio. Los artefactos están reajustados con el desarrollo, comprimidos y verificados al guardar/cargar; las métricas de esta tabla siguen siendo históricas.'),html.Div(tab,className='pf-table'),html.P('No se estableció una calibración final nueva ni un umbral clínico. Las diferencias observadas no acreditan superioridad del propio.'),html.A('Abrir la predicción en una página independiente',href='/predecir/',className='pf-button'),html.A('Leer el libro del entregable 3',href='/entregas/entregable-3-20261024/',className='pf-button'),html.Iframe(src='/predecir/',title='Predicción académica y explicación local con modelos finales',style={'width':'100%','height':'1280px','border':'1px solid #dbe7e9','borderRadius':'12px','marginTop':'18px'})],className='pf-section'),previous(),html.Section([html.H2('Cómo se calcula el modelo propio'),html.Img(src='/evidencias-e3/flujo_modelo.svg',alt='Entrada, bases spline e interacciones, pérdida y score del modelo propio'),html.P('La explicación suma contribuciones del margen. No representa causas ni un diagnóstico.')],className='pf-section')])
 def problema():return html.Div([context(),html.Section([html.H2('Del dato a la evaluación'),html.Img(src='/evidencias-e3/flujo_validacion.svg',alt='Esquema de validación agrupada externa e interna'),html.P('Las copias de un perfil permanecen juntas. No se acredita identidad de pacientes ni un nuevo test independiente.')],className='pf-section')])
 secciones['modelo']=modelo;secciones['problema']=problema
 app.layout.children[0].children.append(html.Div([html.A('Entregable 3 · libro por capítulos',href='/entregas/entregable-3-20261024/',className='pf-button'),html.A('Entregable 2 · versión enviada',href='/entregas/entregable-2-20261023/',className='pf-button')],className='pf-links e3-release'))
