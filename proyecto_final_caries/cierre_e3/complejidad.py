"""Medición nueva en procesos independientes: tiempo y RSS total observado."""
from pathlib import Path
import argparse,subprocess,sys,json,itertools,time,threading,os
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from proyecto_final_caries.modelo import ModeloCaries,CAT
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--salida',type=Path,required=True);ap.add_argument('--n',type=int);ap.add_argument('--p',type=int);ap.add_argument('--rep',type=int);a=ap.parse_args()
 if a.n is None:
  a.salida.mkdir(parents=True,exist_ok=False);rows=[]
  for n,p,rep in itertools.product([2000,8000,20000],[11,17,23],range(3)):
   target=a.salida/f'n{n}_p{p}_r{rep}.json';subprocess.run([sys.executable,'-m','proyecto_final_caries.cierre_e3.complejidad','--salida',str(target),'--n',str(n),'--p',str(p),'--rep',str(rep)],check=True);rows.append(json.loads(target.read_text()));pd.DataFrame(rows).to_csv(a.salida/'mediciones.csv',index=False);print('Complejidad',len(rows),'/27',flush=True)
  (a.salida/'COMPLETADO.json').write_text(json.dumps(dict(ajustes=27,limites=['RSS total del proceso observado cada 10 ms, no memoria exclusiva del modelo ni máximo continuo exacto.','Las bases y productos previos son dependientes de n y p; tres repeticiones con igual semilla miden variación de ejecución, no robustez estadística.','Mediciones en la Mac durante otros trabajos; no comparación de hardware ni medición aislada sin carga concurrente.']),indent=2,ensure_ascii=False));return
 df=pd.read_csv(ROOT/'entregable_2/results/train.csv',nrows=a.n);features=json.loads((ROOT/'entregable_2/results/summary.json').read_text())['features'];num=[c for c in features if c not in CAT];chosen=num[:a.p-len(CAT)]+CAT;X=df[chosen].copy();y=df['dental caries'];process=psutil.Process();rss=[process.memory_info().rss];stop=threading.Event()
 def observe():
  while not stop.wait(.01):rss.append(process.memory_info().rss)
 monitor=threading.Thread(target=observe,daemon=True);monitor.start();m=ModeloCaries(alpha=.25,l2=.01,max_interactions=5,max_iter=1000,random_state=42)
 with threadpool_limits(limits=1):t=time.perf_counter();m.fit(X,y);fit=time.perf_counter()-t;t=time.perf_counter();m.predict_proba(X.iloc[:20]);infer=time.perf_counter()-t
 stop.set();monitor.join();rss.append(process.memory_info().rss);assert m.converged_;a.salida.write_text(json.dumps(dict(n=a.n,p=a.p,repeticion=a.rep,semilla=42,ajuste_s=fit,inferencia_20_filas_s=infer,rss_max_MiB=max(rss)/1048576,rss_inicial_MiB=rss[0]/1048576,convergio=bool(m.converged_),iteraciones=m.optimizer_['iterations'],columnas_transformadas=len(m.coef_)),indent=2))
if __name__=='__main__':main()
