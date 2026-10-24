"""Construir respaldo Jupyter Book del proyecto final sin ejecutar cuadernos."""
from pathlib import Path
import argparse,shutil,subprocess,zipfile,json,re

def main():
    p=argparse.ArgumentParser();p.add_argument('--salida',type=Path,required=True);p.add_argument('--jupyter-book',default='jupyter-book');a=p.parse_args()
    root=Path(__file__).resolve().parents[2];source=a.salida/'fuente';source.mkdir(parents=True,exist_ok=False)
    notebook=root/'proyecto_final_caries_v2.ipynb';shutil.copy2(notebook,source/notebook.name)
    nb=json.loads(notebook.read_text());text='\n'.join(''.join(c.get('source',[])) for c in nb['cells'])
    folders=set(re.findall(r'_static/([A-Za-z0-9_-]+)',text))
    for folder in folders:
        shutil.copytree(root/'_static'/folder,source/'_static'/folder)
    (source/'_config.yml').write_text('title: "Proyecto final de caries"\nauthor: "Andrés Duarte Chaves"\nexecute:\n  execute_notebooks: "off"\nonly_build_toc_files: true\nexclude_patterns: ["_build"]\n')
    (source/'_toc.yml').write_text('format: jb-book\nroot: proyecto_final_caries_v2\n')
    subprocess.run([a.jupyter_book,'build',str(source),'--warningiserror'],check=True)
    target=a.salida/'jupyter_book_caries.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        html=source/'_build/html'
        for f in sorted(html.rglob('*')):
            if f.is_file():z.write(f,str(f.relative_to(html)))
        z.writestr('LEEME.txt','Abrir index.html. Libro construido sin ejecutar entrenamientos. Las evidencias locales están incluidas. Algunos recursos externos, como fuentes o MathJax, pueden necesitar conexión.\n')
    print('LIBRO:',target)
if __name__=='__main__':main()
