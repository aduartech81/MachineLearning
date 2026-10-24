"""Comparación de la misma pérdida: matriz M×d explícita frente a bincount.
La variante optimizada es la que ya usa objetivo.py. No modifica ningún modelo.
"""
from pathlib import Path
import argparse,subprocess,sys,json,time,threading
import numpy as np
import psutil
from scipy.special import expit
from threadpoolctl import threadpool_limits
from proyecto_final_caries.objetivo import perdida_gradiente,generar_pares
ROOT=Path(__file__).resolve().parents[2]
def datos(M):
 rng=np.random.default_rng(42);X=rng.normal(size=(5000,128));y=rng.integers(0,2,5000);t=rng.normal(scale=.1,size=129);p,n=generar_pares(y,M,42);return t,X,y,p,n
def matriz(theta,X,y,p,n,alpha=.25,l2=.01):
 w=theta[1:];s=X@w+theta[0];bce=np.mean(np.logaddexp(0,s)-y*s);e=(expit(s)-y)/len(y);D=X[p]-X[n];d=D@w;q=expit(-d)/len(d);loss=(1-alpha)*bce+alpha*np.mean(np.logaddexp(0,-d))+.5*l2*(w@w);grad=np.r_[(1-alpha)*e.sum(),(1-alpha)*(X.T@e)-alpha*(D.T@q)+l2*w];return float(loss),grad
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--salida',type=Path,required=True);ap.add_argument('--M',type=int);ap.add_argument('--variante',choices=['matriz','bincount']);ap.add_argument('--rep',type=int);a=ap.parse_args()
 if a.M is None:
  import pandas as pd
  a.salida.mkdir(parents=True,exist_ok=False);rows=[]
  for M in [5000,20000,80000]:
   with threadpool_limits(limits=1):inputs=datos(M);expected=matriz(*inputs);opt=perdida_gradiente(*inputs);assert np.allclose(expected[1],opt[1],atol=1e-10,rtol=1e-11) and np.isclose(expected[0],opt[0],atol=1e-12)
   for rep in range(3):
    for variant in (['matriz','bincount'] if rep%2==0 else ['bincount','matriz']):
     f=a.salida/f'M{M}_{variant}_{rep}.json';subprocess.run([sys.executable,'-m','proyecto_final_caries.cierre_e3.vectorizacion','--salida',str(f),'--M',str(M),'--variante',variant,'--rep',str(rep)],check=True);r=json.loads(f.read_text());assert np.allclose(r['gradiente'],expected[1],atol=1e-10,rtol=1e-11) and np.isclose(r['perdida'],expected[0],atol=1e-12);r.pop('gradiente');rows.append(r)
  out=ROOT/'_static/entregable3';pd.DataFrame(rows).to_csv(out/'vectorizacion_mediciones.csv',index=False);summary=pd.DataFrame(rows).groupby(['M','variante'])[['segundos_por_evaluacion','rss_max_MiB']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'vectorizacion_resumen.csv',index=False);(out/'vectorizacion_metodo.json').write_text(json.dumps(dict(n=5000,d=128,semilla=42,ajustes_modelo=0,procesos=18,repeticiones_internas=5,igualdad_numerica='Pérdida y gradiente coinciden en las dos implementaciones para tres tamaños; atol gradiente 1e-10, rtol 1e-11.',limites=['Benchmark de evaluación de pérdida/gradiente, no entrenamiento completo ni cambio de AUC.','RSS del proceso muestreado cada 10 ms; no memoria exclusiva del modelo.','Carga concurrente de la Mac; tiempos descriptivos.','La variante bincount evita una matriz M×d y ya formaba parte del algoritmo.']),indent=2,ensure_ascii=False));print('18 procesos de vectorización verificados',flush=True);return
 inputs=datos(a.M);fn=matriz if a.variante=='matriz' else perdida_gradiente;process=psutil.Process();rss=[process.memory_info().rss];stop=threading.Event()
 def observe():
  while not stop.wait(.01):rss.append(process.memory_info().rss)
 mon=threading.Thread(target=observe,daemon=True);mon.start()
 with threadpool_limits(limits=1):t=time.perf_counter();values=[fn(*inputs) for _ in range(5)];seconds=(time.perf_counter()-t)/5
 stop.set();mon.join();rss.append(process.memory_info().rss);loss,grad=values[-1];a.salida.write_text(json.dumps(dict(M=a.M,variante=a.variante,repeticion=a.rep,segundos_por_evaluacion=seconds,rss_max_MiB=max(rss)/1048576,perdida=loss,gradiente=grad.tolist())))
if __name__=='__main__':main()
