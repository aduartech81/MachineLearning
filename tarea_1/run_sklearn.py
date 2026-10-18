"""Ejecutar: python -m tarea_1.run_sklearn [nombre de modelo]."""
import sys,time,json,platform,os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from tarea_1.common import OUT,FEATURES,FoldClassifier,metrics

def main():
    t=time.perf_counter();df=pd.read_csv(OUT/'canonical.csv')
    train=df.loc[df.split=='train'];test=df.loc[df.split=='test']
    read_seconds=time.perf_counter()-t
    X=train[FEATURES];y=train['default'];Xt=test[FEATURES];yt=test['default']
    fold=train.fold.to_numpy();cv=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(3)]
    configs={
     'GaussianNB':(GaussianNB(),{},False),
     'LogisticRegression':(LogisticRegression(max_iter=2000),{'reg':[1e-6,1e-5,1e-4]},True),
     'DecisionTree':(DecisionTreeClassifier(random_state=42),{'estimator__max_depth':[5,10,15]},False),
     'HistGradientBoosting':(HistGradientBoostingClassifier(learning_rate=.1,early_stopping=False,max_leaf_nodes=None,random_state=42),{'estimator__max_iter':[50,100],'estimator__max_depth':[3,5]},False),
     'LinearSVC':(LinearSVC(loss='hinge',max_iter=2000,tol=1e-3,random_state=42),{'reg':[1e-6,1e-5,1e-4]},True),
     'RandomForest':(RandomForestClassifier(random_state=42,n_jobs=4),{'estimator__n_estimators':[10,50,100],'estimator__max_depth':[5,10,15]},False)}
    selected=sys.argv[1:]
    for name,(est,grid,scale) in configs.items():
        if selected and name not in selected:continue
        dest=OUT/f'sklearn_{name}.json'
        if dest.exists():print('Ya calculado',name,flush=True);continue
        print('INICIO sklearn',name,flush=True);t=time.perf_counter()
        # Una búsqueda simultánea evita multiplicar matrices de >1 millón de filas en 8 GiB.
        search=GridSearchCV(FoldClassifier(est,scale),grid,cv=cv,scoring='roc_auc',n_jobs=1,error_score='raise',return_train_score=False,verbose=3)
        with joblib.parallel_backend('threading'):
            search.fit(X,y)
        fit_time=time.perf_counter()-t
        model=search.best_estimator_;t=time.perf_counter()
        z=model.transform(Xt)
        score=model.model_.decision_function(z) if name=='LinearSVC' else model.model_.predict_proba(z)[:,1]
        pred=model.model_.predict(z);prediction_seconds=time.perf_counter()-t
        result=dict(framework='sklearn',model=name,rows=len(df),train=len(train),test=len(test),
                    cv_auc=float(search.best_score_),cv_sd=float(search.cv_results_['std_test_score'][search.best_index_]),
                    fit_cv_seconds=fit_time,predict_seconds=prediction_seconds,read_seconds=read_seconds,
                    final_preprocess_seconds=model.preprocess_seconds_,parameters=search.best_params_,iterations=(int(np.max(search.best_estimator_.model_.n_iter_)) if hasattr(search.best_estimator_.model_,'n_iter_') else None),**metrics(yt,score,pred))
        pd.DataFrame(search.cv_results_).to_csv(OUT/f'sklearn_{name}_cv.csv',index=False)
        pd.DataFrame(dict(row_id=test.row_id,actual=yt,score=score,prediction=pred)).to_csv(OUT/f'sklearn_{name}_predictions.csv',index=False)
        joblib.dump(model,OUT/f'sklearn_{name}.joblib',compress=3)
        dest.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
