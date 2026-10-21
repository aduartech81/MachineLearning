"""Empaqueta fuentes, registros y predicciones seleccionadas V2; no entrena ni publica."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import csv
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from importlib.metadata import version, PackageNotFoundError

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'proyecto_final_caries/experimentos_v2'
DECISIONS = ROOT / 'proyecto_final_caries/presentacion/evidencias/revision_v2_20261020_154421_004423/comparacion/decisiones_solo_validacion_interna.csv'


def main():
    if not BASE.is_dir():
        raise SystemExit(f'No se encontró la carpeta de ejecución local: {BASE}')
    with DECISIONS.open(newline='',encoding='utf-8') as handle:
        decisions=list(csv.DictReader(handle))
    selected=set()
    missing=[]
    for row in decisions:
        folder=BASE/'evaluacion_v2'/f"{row['optimizador']}_{row['balanceo']}"/row['familia']/f"fold_{row['fold']}"
        if not folder.is_dir() and row['familia']=='Logistica_L1_L2' and row['optimizador']=='grid' and row['balanceo']=='sin_balanceo' and row['fold']=='0':
            folder=BASE/'evaluacion_externa_piloto/Logistica_L1_L2_grid_sin_balanceo/fold_0'
        found=[]
        if folder.is_dir():
            for p in folder.glob('*.csv'):
                with p.open(encoding='utf-8-sig',newline='') as handle: header=next(csv.reader(handle),[])
                if {'row_id','actual','score'}.issubset(header):found.append(p.resolve())
        selected.update(found)
        if not found:missing.append(str(folder.relative_to(ROOT)))
    files=[];skipped=[]
    for p in sorted(BASE.rglob('*')):
        if not p.is_file() or p.is_symlink() or any(v.startswith('.') or v=='__pycache__' for v in p.relative_to(BASE).parts):continue
        suffix=p.suffix.lower()
        if suffix not in {'.py','.json','.jsonl','.csv','.log','.md','.txt','.yml','.yaml'}:continue
        if p.name.lower() in {'train.csv','test.csv','smoking.csv','canonical.csv'}:continue
        if suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='',errors='replace') as handle:header=next(csv.reader(handle),[])
            # Únicamente las predicciones de las 35 elecciones finales; no se
            # empaquetan cientos de matrices de predicción ni los predictores crudos.
            if ('row_id' in header or 'perfil' in header or 'actual' in header) and p.resolve() not in selected:continue
        if p.stat().st_size>20*1024**2:skipped.append(str(p.relative_to(ROOT)));continue
        files.append(p)
    counts=Counter(p.suffix for p in files)
    total=sum(p.stat().st_size for p in files)
    if total>300*1024**2:raise SystemExit(f'Paquete estimado {total/1024**2:.1f} MiB: se detuvo para revisar su contenido.')
    out=ROOT/'resultados_tarea1_mac';out.mkdir(exist_ok=True)
    dest=out/('evidencias_v2_para_revision_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
    packages={}
    for name in ['numpy','pandas','scipy','scikit-learn','imbalanced-learn','xgboost','optuna','deap','shap','lime','psutil']:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]='no instalado en el entorno activo'
    manifest={'alcance':'Recopilación local, sin entrenamiento ni publicación','entorno_observado_ahora':{'python':sys.version,'executable':sys.executable,'plataforma':platform.platform(),'paquetes':packages},'advertencia_entorno':'Describe el entorno activo ahora; los registros originales conservan las versiones de ejecución cuando existen.','commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'predicciones_finalistas_encontradas':len(selected),'carpetas_sin_predicciones':missing,'omitidos_por_tamano':skipped,'archivos':[]}
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            data=p.read_bytes();rel=str(p.relative_to(ROOT));archive.writestr(rel,data)
            manifest['archivos'].append({'ruta':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        archive.writestr('MANIFIESTO_RECOPILACION.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    print('PAQUETE:',dest)
    print('Archivos:',len(files),'| Tipos:',dict(counts))
    print('Tamaño ZIP MiB:',round(dest.stat().st_size/1024**2,2))
    print('Predicciones seleccionadas encontradas:',len(selected),'| Carpetas pendientes:',len(missing))
    print('Adjunta este ZIP al chat. No se subió automáticamente a ningún servicio.')
    print('Datos originales, modelos, entrenamientos y Render intactos.')


if __name__=='__main__':main()
