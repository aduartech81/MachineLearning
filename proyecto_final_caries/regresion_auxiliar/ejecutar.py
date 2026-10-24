"""Regresión auxiliar prospectiva de glucemia: 7 familias × 4 búsquedas, CV 5×3.
No convierte caries en un objetivo continuo ni utiliza el test histórico.
"""
from pathlib import Path
import argparse,hashlib,itertools,json,os,time,warnings
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import optuna
from sklearn import config_context
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits
from proyecto_final_caries.experimentos_v2.preprocesamiento import antes_balanceo,despues_balanceo
from proyecto_final_caries.experimentos_v2.motor_genetico import EstudioGenetico
ROOT=Path(__file__).resolve().parents[2]
FAMILIAS=['Ridge','Lasso','DecisionTree','XGBoost','SVR_lineal','RandomForest','KNN']
def domcat(v):return {'tipo':'categorico','valores':v}
def domlog(a,b):return {'tipo':'real_log','min':a,'max':b}
DOM={
 'Ridge':{'alpha':domlog(.001,10000)},'Lasso':{'alpha':domlog(.0001,1)},
 'DecisionTree':{'max_depth':domcat([4,10]),'min_samples_leaf':domcat([10,60]),'max_features':domcat([.7,1.])},
 'RandomForest':{'max_depth':domcat([5,10]),'min_samples_leaf':domcat([20,100]),'max_features':domcat([.7,1.])},
 'XGBoost':{'max_depth':domcat([2,4]),'learning_rate':domcat([.03,.1]),'subsample':domcat([.7,1.])},
 'SVR_lineal':{'C':domlog(.001,1),'epsilon':domcat([.05,.2])},
 'KNN':{'n_neighbors':domcat([25,100]),'weights':domcat(['uniform','distance']),'p':domcat([1,2])}}
GRID={f:list(dict(zip(DOM[f],v)) for v in itertools.product(*[d['valores'] if d['tipo']=='categorico' else np.geomspace(d['min'],d['max'],8 if len(DOM[f])==1 else 4).tolist() for d in DOM[f].values()])) for f in FAMILIAS}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def modelo(f,p):
 if f=='Ridge':return Ridge(**p)
 if f=='Lasso':return Lasso(**p,max_iter=5000,tol=1e-4,random_state=42)
 if f=='DecisionTree':return DecisionTreeRegressor(**p,random_state=42)
 if f=='RandomForest':return RandomForestRegressor(**p,n_estimators=60,n_jobs=2,random_state=42)
 if f=='XGBoost':return XGBRegressor(**p,n_estimators=60,tree_method='hist',n_jobs=2,random_state=42,objective='reg:squarederror',colsample_bytree=.8)
 if f=='SVR_lineal':return LinearSVR(**p,loss='squared_epsilon_insensitive',dual='auto',random_state=42,max_iter=5000,tol=1e-3)
 return KNeighborsRegressor(**p,algorithm='brute',n_jobs=2)
def transformar(X,ia,iv,y):
 a=antes_balanceo(list(X.columns));b=despues_balanceo(list(X.columns))
 A=b.fit_transform(a.fit_transform(X.iloc[ia])).astype(np.float32);V=b.transform(a.transform(X.iloc[iv])).astype(np.float32)
 mu=float(y[ia].mean());sd=float(y[ia].std());assert sd>0
 return A,V,(y[ia]-mu)/sd,y[iv],mu,sd

def ajustar(f,p,data):
 A,V,yy,yv,mu,sd=data;m=modelo(f,p)
 with warnings.catch_warnings(record=True) as ww,threadpool_limits(limits=1),config_context(working_memory=64):
  warnings.simplefilter('always');t=time.perf_counter();m.fit(A,yy);fit=time.perf_counter()-t;t=time.perf_counter();pred=m.predict(V)*sd+mu;inf=time.perf_counter()-t
 conv=not any(issubclass(w.category,ConvergenceWarning) for w in ww)
 return m,pred,float(np.sqrt(mean_squared_error(yv,pred))),fit,inf,conv,[str(w.message) for w in ww]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--salida',type=Path,required=True);args=ap.parse_args();out=args.salida.resolve();out.mkdir(parents=True,exist_ok=False)
 train=ROOT/'entregable_2/results/train.csv';df=pd.read_csv(train);features=json.loads((ROOT/'entregable_2/results/summary.json').read_text())['features'];target='fasting blood sugar';features=[c for c in features if c!=target]
 X=df[features].copy();y=df[target].to_numpy(float);assert np.isfinite(y).all() and len(np.unique(y))>100
 groups=pd.util.hash_pandas_object(X,index=False).to_numpy();rowids=pd.read_csv(ROOT/'proyecto_final_caries/evaluacion/particiones_desarrollo.csv').row_id.to_numpy();assert len(rowids)==len(X)
 proto=dict(objetivo=target,unidades='Unidades originales del archivo, no confirmadas clínicamente',alcance='Estimación contemporánea auxiliar de glucemia. No severidad de caries ni pronóstico clínico.',predictores=features,exclusiones=['ID','oral','tartar','dental caries',target],semilla=42,folds_externos=5,folds_internos=3,presupuesto_propuestas=8,dominios=DOM,grids=GRID,datos_sha256=sha(train),script_sha256=sha(__file__),n=len(X),perfiles=int(len(np.unique(groups))),familias=FAMILIAS,advertencias=['CV nueva agrupada por los 22 predictores; perfil no acredita paciente.','Objetivo escalado con media/DE del ajuste, predicciones invertidas antes de evaluar.','SVR lineal con pérdida epsilon-insensible cuadrática via LinearSVR por coste; no SVR RBF.','Presupuesto acotado a 8 propuestas; no comparable sin matices a las 16 de clasificación.','Se reutilizan evaluaciones idénticas entre optimizadores con semilla de modelo fija; el registro separa propuestas de ajustes nuevos.','Sin test histórico, sin balanceo de clases; no existe orden temporal verificado.'])
 dump(out/'protocolo.json',proto);outer=list(GroupKFold(n_splits=5,shuffle=True,random_state=42).split(X,y,groups));assign=np.full(len(X),-1);results=[];hist=[];innerassign=[]
 optuna.logging.set_verbosity(optuna.logging.WARNING)
 for k,(ia,iv) in enumerate(outer):
  assign[iv]=k;assert not set(groups[ia])&set(groups[iv]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=42+k).split(X.iloc[ia],y[ia],groups[ia]));cached=[]
  for j,(ta,tv) in enumerate(inner):
   assert not set(groups[ia[ta]])&set(groups[ia[tv]]);cached.append(transformar(X,ia[ta],ia[tv],y));innerassign.extend({'row_id':int(rowids[r]),'outer_fold':k,'inner_fold':j} for r in ia[tv])
  ext=transformar(X,ia,iv,y)
  for f in FAMILIAS:
   evaluation_cache={}
   for opt in ['grid','random','bayes','genetica']:
    seed=42+k;rng=np.random.default_rng(seed);study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=4)) if opt=='bayes' else None;ga=EstudioGenetico(DOM[f],seed) if opt=='genetica' else None;best=None;best_loss=np.inf;div=[];fresh=0;t0=time.perf_counter()
    for trial in range(8):
     handle=study.ask() if study is not None else ga.ask() if ga is not None else None
     if opt=='grid':p=GRID[f][trial]
     elif opt=='random':p={n:(d['valores'][int(rng.integers(len(d['valores'])))] if d['tipo']=='categorico' else float(np.exp(rng.uniform(np.log(d['min']),np.log(d['max']))))) for n,d in DOM[f].items()}
     else:p={n:(handle.suggest_categorical(n,d['valores']) if d['tipo']=='categorico' else handle.suggest_float(n,d['min'],d['max'],log=True)) for n,d in DOM[f].items()}
     key=json.dumps(p,sort_keys=True);reused=key in evaluation_cache
     if not reused:
      vals=[];warn=[];ok=True;fits=preds=0.
      for j,data in enumerate(cached):
       _,yp,rmse,fit,inf,conv,w=ajustar(f,p,data);vals.append(rmse);fits+=fit;preds+=inf;ok=ok and conv;warn+=w
      evaluation_cache[key]=(float(np.mean(vals)),ok,vals,fits,preds,warn);fresh+=3
     loss,ok,vals,fits,preds,warn=evaluation_cache[key];fitness=-loss if ok else -1e12
     if study is not None:study.tell(handle,fitness)
     if ga is not None:ga.tell(handle,fitness);div.append(tuple(handle.genes))
     if ok and loss<best_loss:best_loss=loss;best=p
     hist.append(dict(familia=f,fold=k,optimizador=opt,propuesta=trial+1,parametros=key,reutilizada=reused,rmse_interno=loss,convergio=ok,mejor_rmse=best_loss if np.isfinite(best_loss) else None,ajuste_s_nuevo=0. if reused else fits,inferencia_s_nueva=0. if reused else preds,advertencias=json.dumps(warn),diversidad_genetica=len(set(div[-4:])) if ga is not None else None))
    assert best is not None,f'No convergió ninguna propuesta: {f}, {opt}, {k}'
    m,pred,rmse,fit,inf,conv,w=ajustar(f,best,ext);assert conv,f'No convergió ajuste externo: {f}';A,V,yy,yv,mu,sd=ext
    with threadpool_limits(limits=1),config_context(working_memory=64):yptrain=m.predict(A)*sd+mu
    key=dict(familia=f,optimizador=opt,fold=k);result=dict(**key,parametros=json.dumps(best,sort_keys=True),rmse_interno=best_loss,rmse=rmse,mae=float(mean_absolute_error(yv,pred)),r2=float(r2_score(yv,pred)),rmse_ajuste=float(np.sqrt(mean_squared_error(y[ia],yptrain))),mae_ajuste=float(mean_absolute_error(y[ia],yptrain)),dummy_rmse=float(np.sqrt(mean_squared_error(yv,np.full(len(iv),y[ia].mean())))),dummy_mae=float(mean_absolute_error(yv,np.full(len(iv),np.median(y[ia])))),ajuste_s=fit,inferencia_s=inf,busqueda_s=time.perf_counter()-t0-fit-inf,ajustes_internos_nuevos=fresh,semilla=42,semilla_busqueda=seed,n_ajuste=len(ia),n_validacion=len(iv),convergio=conv)
    results.append(result);dump(out/f'{f}_{opt}_fold{k}.json',result)
    pd.DataFrame({'row_id':rowids[iv],'perfil':groups[iv],'actual':yv,'prediccion':pred,'fold':k}).to_csv(out/f'{f}_{opt}_fold{k}_pred.csv.gz',index=False)
    pd.DataFrame(results).to_csv(out/'metricas_por_fold.csv',index=False);pd.DataFrame(hist).to_csv(out/'historial_propuestas.csv.gz',index=False)
    print(f'{len(results)}/140 | {f} {opt} fold {k} | RMSE {rmse:.4f}',flush=True)
 pd.DataFrame({'row_id':rowids,'perfil':groups,'outer_fold':assign,'objetivo':y}).to_csv(out/'particiones_externas.csv',index=False);pd.DataFrame(innerassign).to_csv(out/'particiones_internas.csv',index=False)
 r=pd.DataFrame(results);summary=r.groupby(['familia','optimizador'])[['rmse','mae','r2','rmse_ajuste','dummy_rmse','dummy_mae','ajuste_s','inferencia_s']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'resumen_28.csv',index=False)
 assert len(results)==140 and len(summary)==28;dump(out/'COMPLETADO.json',{'configuraciones':28,'evaluaciones_externas':140,'propuestas':len(hist),'ajustes_internos_nuevos':sum(r.ajustes_internos_nuevos),'alcance':'Regresión auxiliar; clasificación original conservada.'});print('REGRESIÓN COMPLETADA',out,flush=True)
if __name__=='__main__':main()
