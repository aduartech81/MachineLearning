"""Reconstruye modelos con parámetros seleccionados ya guardados; no repite ni reemplaza CV."""
import json,time
import joblib,numpy as np,pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier
from tarea_1.common import OUT,FEATURES,FoldClassifier,metrics

def main():
 df=pd.read_csv(OUT/'canonical.csv');train=df[df.split=='train'];test=df[df.split=='test']
 configs={'GaussianNB':GaussianNB(),'LogisticRegression':LogisticRegression(max_iter=2000),'DecisionTree':DecisionTreeClassifier(random_state=42),'HistGradientBoosting':HistGradientBoostingClassifier(learning_rate=.1,early_stopping=False,max_leaf_nodes=None,random_state=42),'LinearSVC':LinearSVC(loss='hinge',max_iter=2000,tol=1e-3,random_state=42),'RandomForest':RandomForestClassifier(random_state=42,n_jobs=4)}
 audit=[]
 for name,est in configs.items():
  original=json.loads((OUT/f'sklearn_{name}.json').read_text());start=time.perf_counter()
  model=FoldClassifier(est,name in ['LogisticRegression','LinearSVC'],normalize_rows=name=='LinearSVC').set_params(**original['parameters'])
  model.fit(train[FEATURES],train['default']);z=model.transform(test[FEATURES]);score=model.model_.decision_function(z) if name=='LinearSVC' else model.model_.predict_proba(z)[:,1];pred=model.model_.predict(z)
  result=metrics(test['default'],score,pred)
  for k in ['roc_auc','accuracy','precision','recall','f1']:assert np.isclose(result[k],original[k],atol=1e-6), (name,k,result[k],original[k])
  for k in ['tn','fp','fn','tp']:assert result[k]==original[k],(name,k)
  joblib.dump(model,OUT/f'sklearn_{name}.joblib',compress=3)
  pd.DataFrame(dict(row_id=test.row_id,actual=test['default'],score=score,prediction=pred)).to_csv(OUT/f'sklearn_{name}_predictions.csv',index=False)
  audit.append(dict(model=name,seconds=time.perf_counter()-start,metrics_match=True));print(audit[-1],flush=True)
  (OUT/'recovery_sklearn.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
