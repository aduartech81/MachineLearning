from pathlib import Path
import hashlib,json,zipfile
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--html',type=Path,required=True);parser.add_argument('--destino',type=Path);args=parser.parse_args()
repo=Path(__file__).resolve().parents[2];build=args.html;out=args.destino if args.destino else repo/'proyecto_final_caries/presentacion/documentacion/libro';out.mkdir(parents=True,exist_ok=True)
known={}
for f in (repo/'_static').rglob('*'):
 if f.is_file():known[hashlib.sha256(f.read_bytes()).hexdigest()]=str(f.relative_to(repo))
mapping={}
with zipfile.ZipFile(out/'estructura.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(build.rglob('*')):
  if not f.is_file():continue
  name=str(f.relative_to(build));digest=hashlib.sha256(f.read_bytes()).hexdigest()
  if digest in known:mapping[name]={'archivo':known[digest],'sha256':digest}
  else:z.write(f,name)
(out/'recursos.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2))
print('estructura bytes',(out/'estructura.zip').stat().st_size,'recursos reutilizados',len(mapping))
