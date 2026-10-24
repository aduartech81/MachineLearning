"""Índice del portafolio y versión fechada de la entrega, sin reentrenamientos."""
from pathlib import Path
from io import BytesIO
import json,zipfile,mimetypes,hashlib
from flask import send_file,send_from_directory,abort

def integrar(server):
 base=Path(__file__).parent/'documentacion';repo=Path(__file__).resolve().parents[2];snapshot=base/'entregable2_20261023';mapping=json.loads((snapshot/'recursos.json').read_text());structure=snapshot/'estructura.zip'
 with zipfile.ZipFile(structure) as z:embedded=set(z.namelist())
 allowed=set(mapping)|embedded
 @server.route('/portafolio/')
 def indice_portafolio():return send_from_directory(base,'portafolio.html')
 @server.route('/entregas/entregable-2-20261023/',defaults={'nombre':'index.html'})
 @server.route('/entregas/entregable-2-20261023/<path:nombre>')
 def entrega_fechada(nombre):
  if nombre not in allowed:abort(404)
  if nombre in mapping:
   info=mapping[nombre];data=(repo/info['archivo']).read_bytes()
   if hashlib.sha256(data).hexdigest()!=info['sha256']:abort(409,description='El recurso ya no coincide con la versión archivada')
  else:
   with zipfile.ZipFile(structure) as z:data=z.read(nombre)
  return send_file(BytesIO(data),download_name=Path(nombre).name,mimetype=mimetypes.guess_type(nombre)[0] or 'application/octet-stream')
