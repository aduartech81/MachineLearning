"""Servir un Jupyter Book ya construido y su respaldo; sin compilar ni entrenar."""
from pathlib import Path
from io import BytesIO
import json,mimetypes,zipfile
from flask import abort,send_file

def integrar(server):
    repo=Path(__file__).resolve().parents[2]
    base=Path(__file__).parent/'documentacion/libro'
    mapping=json.loads((base/'recursos.json').read_text())
    structure=base/'estructura.zip'
    with zipfile.ZipFile(structure) as z:embedded=set(z.namelist())
    allowed=set(mapping)|embedded
    def read(name):
        if name in mapping:
            path=repo/mapping[name]['archivo']
            return path.read_bytes()
        with zipfile.ZipFile(structure) as z:return z.read(name)
    @server.route('/libro-final/',defaults={'nombre':'index.html'})
    @server.route('/libro-final/<path:nombre>')
    def libro_final(nombre):
        if nombre not in allowed:abort(404)
        return send_file(BytesIO(read(nombre)),download_name=Path(nombre).name,
                         mimetype=mimetypes.guess_type(nombre)[0] or 'application/octet-stream')
    @server.route('/descargar-libro-final.zip')
    def descargar_libro_final():
        result=BytesIO()
        with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as z:
            for name in sorted(allowed):z.writestr(name,read(name))
            z.writestr('LEEME.txt','Abrir index.html. Jupyter Book construido sin ejecutar entrenamientos. Evidencias locales incluidas; algunos recursos externos como MathJax pueden necesitar conexión.\n')
        result.seek(0)
        return send_file(result,download_name='jupyter_book_caries.zip',as_attachment=True,mimetype='application/zip')
