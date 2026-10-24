"""AUC ponderada y jackknife de perfiles con score constante dentro del perfil.
La incertidumbre resultante es condicional a predicciones fijas, no a reentrenar.
"""
import numpy as np
from scipy.stats import norm
from proyecto_final_caries.analisis.contrastes_sensibilidad import componentes

def preparar(y,s,grupos):
 y=np.asarray(y,int);s=np.asarray(s,float);groups,codes=np.unique(grupos,return_inverse=True);order=np.argsort(s,kind='stable');ss=s[order];starts=np.r_[0,np.flatnonzero(np.diff(ss))+1]
 return dict(y=y,s=s,codes=codes,groups=groups,order=order,starts=starts)
def auc_ponderada(p,w):
 """Una fila de pesos por bootstrap, una columna por perfil; promedio por filas."""
 w=np.atleast_2d(w);weights=w[:,p['codes'][p['order']]];yy=p['y'][p['order']];wp=np.add.reduceat(weights*yy,p['starts'],axis=1);wn=np.add.reduceat(weights*(1-yy),p['starts'],axis=1);den=wp.sum(1)*wn.sum(1)
 return np.sum(wp*(np.cumsum(wn,axis=1)-.5*wn),axis=1)/den

def jackknife_perfiles(p):
 """AUC al retirar cada perfil, con corrección exacta de pares eliminados."""
 y,s,codes=p['y'],p['s'],p['codes'];k=len(p['groups']);low=np.full(k,np.inf);high=np.full(k,-np.inf);np.minimum.at(low,codes,s);np.maximum.at(high,codes,s);assert np.allclose(low,high,atol=1e-12,rtol=0),'Scores distintos en un perfil'
 a,b=componentes(y,s);m=int(y.sum());n=len(y)-m;mg=np.bincount(codes,weights=y,minlength=k);ng=np.bincount(codes,weights=1-y,minlength=k);pa=np.bincount(codes[y==1],weights=a,minlength=k);nb=np.bincount(codes[y==0],weights=b,minlength=k);U=float(a.mean()*m*n)
 return (U-n*pa-m*nb+.5*mg*ng)/((m-mg)*(n-ng))
def intervalo_bca(observado,boots,jack,alpha=.05):
 boots=np.asarray(boots);jack=np.asarray(jack);z0=norm.ppf(np.clip((np.sum(boots<observado)+.5*np.sum(boots==observado))/len(boots),1/(2*len(boots)),1-1/(2*len(boots))));d=jack.mean()-jack;den=6*np.sum(d*d)**1.5;acc=float(np.sum(d**3)/den) if den else 0.;z=norm.ppf([alpha/2,1-alpha/2]);q=norm.cdf(z0+(z0+z)/(1-acc*(z0+z)));q=np.clip(q,0,1);lo,hi=np.quantile(boots,q)
 return dict(inferior=float(lo),superior=float(hi),aceleracion=acc,z0=float(z0))
