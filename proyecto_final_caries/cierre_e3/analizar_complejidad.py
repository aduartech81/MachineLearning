from pathlib import Path
import argparse,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--entrada',type=Path,required=True);a=p.parse_args();assert(a.entrada/'COMPLETADO.json').exists();r=pd.read_csv(a.entrada/'mediciones.csv');assert len(r)==27 and r.convergio.all();out=ROOT/'_static/entregable3';r.to_csv(out/'complejidad_27_ajustes.csv',index=False);s=r.groupby(['n','p'])[['ajuste_s','inferencia_20_filas_s','rss_max_MiB','columnas_transformadas']].agg(['mean','std']);s.columns=['_'.join(c) for c in s.columns];s.reset_index().to_csv(out/'complejidad_resumen.csv',index=False);fig,axs=plt.subplots(1,2,figsize=(11,4))
 for p,g in r.groupby('p'):
  m=g.groupby('n').agg(t=('ajuste_s','mean'),rss=('rss_max_MiB','mean'));axs[0].plot(m.index,m.t,'o-',label=f'p={p}');axs[1].plot(m.index,m.rss,'o-',label=f'p={p}')
 axs[0].set(xlabel='Filas de ajuste n',ylabel='Tiempo medio de ajuste (s)',title='Modelo propio: ajuste completo');axs[1].set(xlabel='Filas de ajuste n',ylabel='RSS máximo observado del proceso (MiB)',title='Memoria del proceso, no del modelo aislado');axs[0].legend();axs[1].legend();fig.tight_layout();fig.savefig(out/'complejidad_np.png',dpi=150);plt.close(fig);(out/'complejidad_metodo.json').write_text((a.entrada/'COMPLETADO.json').read_text());print('27 mediciones de complejidad verificadas')
if __name__=='__main__':main()
