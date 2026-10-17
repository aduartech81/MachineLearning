"""Publica predicciones completas con compresión sin pérdida y hashes."""
import gzip,hashlib,json,shutil
from tarea_1.common import OUT,NAMES

def main():
    manifest=[]
    for engine in ['sklearn','spark']:
        for name in NAMES:
            source=OUT/f'{engine}_{name}_predictions.csv'
            if not (OUT/f'{engine}_{name}.json').exists() or not source.exists():continue
            target=source.with_suffix('.csv.gz')
            if not target.exists() or target.stat().st_mtime<source.stat().st_mtime:
                with source.open('rb') as src,target.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as dst:shutil.copyfileobj(src,dst)
            with gzip.open(target,'rb') as f:restored=f.read()
            original=source.read_bytes();assert restored==original
            manifest.append(dict(file=target.name,raw_sha256=hashlib.sha256(original).hexdigest(),
                gzip_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),raw_bytes=len(original),gzip_bytes=target.stat().st_size,rows=original.count(b'\n')-1))
    (OUT/'predictions_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Series comprimidas y verificadas:',len(manifest),flush=True)
if __name__=='__main__':main()
