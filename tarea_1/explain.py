from tarea_1.common import prediction_path
"""LIME sobre variables originales: categorías perturbadas como categorías válidas."""
import json
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
from tarea_1.common import OUT,NUM,CAT,FEATURES,NAMES

def main():
    df=pd.read_csv(OUT/'canonical.csv');train=df[df.split=='train'];test=df[df.split=='test'].set_index('row_id')
    category_names={len(NUM)+i:sorted(train[c].fillna('Missing').unique().tolist()) for i,c in enumerate(CAT)}
    medians=train[NUM].median()
    def encode(frame):
        a=frame[NUM].fillna(medians).to_numpy()
        cats=[]
        for i,c in enumerate(CAT):
            levels=category_names[len(NUM)+i];mapping={v:k for k,v in enumerate(levels)}
            encoded=frame[c].map(mapping)
            if encoded.isna().any():raise ValueError('Categoría desconocida en caso LIME; no sustituir silenciosamente')
            cats.append(encoded.to_numpy())
        return np.column_stack([a,*cats])
    def decode(a):
        frame=pd.DataFrame(a[:,:len(NUM)],columns=NUM)
        for i,c in enumerate(CAT):
            levels=category_names[len(NUM)+i]
            frame[c]=[levels[int(np.clip(round(v),0,len(levels)-1))] for v in a[:,len(NUM)+i]]
        return frame
    background=encode(train.sample(n=5000,random_state=42))
    explainer=LimeTabularExplainer(background,feature_names=FEATURES,class_names=['Fully Paid','Charged Off'],
        categorical_features=list(category_names),categorical_names=category_names,random_state=42,mode='classification')
    rows=[];spark=None
    for framework in ['sklearn','spark']:
        candidates=[json.loads((OUT/f'{framework}_{name}.json').read_text()) for name in NAMES if name!='LinearSVC']
        winner=max(candidates,key=lambda r:r['cv_auc'])['model']
        prediction=pd.read_csv(prediction_path(framework,winner))
        errors=prediction[prediction.actual!=prediction.prediction].head(2)
        if framework=='sklearn':
            model=joblib.load(OUT/f'{framework}_{winner}.joblib')
            def predict(a):return model.predict_proba(decode(a))
        else:
            from pyspark.sql import SparkSession,functions as F
            from pyspark.ml import PipelineModel
            from pyspark.ml.functions import vector_to_array
            from tarea_1.spark_components import CacheVectors
            spark=(SparkSession.builder.master('local[4]').appName('Tarea1_LIME').config('spark.driver.memory','4g')
                   .config('spark.driver.host','127.0.0.1').config('spark.driver.bindAddress','127.0.0.1').config('spark.ui.enabled','false').getOrCreate())
            spark.sparkContext.setLogLevel('ERROR');model=PipelineModel.load(str((OUT/f'spark_{winner}_model').resolve()))
            def predict(a):
                frame=decode(a);frame['lime_id']=np.arange(len(frame))
                source=spark.createDataFrame(frame)
                output=model.transform(source).select('lime_id',vector_to_array('probability').alias('p')).orderBy('lime_id').collect()
                result=np.array([row.p for row in output]);spark.catalog.clearCache();return result
        for case in errors.itertuples():
            original=test.loc[[case.row_id]]
            # Missing numeric values get train medians for local perturbation; disclosed in report.
            exp=explainer.explain_instance(encode(original)[0],predict,num_features=8,num_samples=2000)
            for feature,weight in exp.as_list():
                rows.append(dict(framework=framework,model=winner,row_id=case.row_id,actual=case.actual,
                    prediction=case.prediction,score=case.score,feature=feature,weight=weight,
                    local_r2=exp.score,local_prediction=float(exp.local_pred[0])))
            exp.save_to_file(str(OUT/f'lime_{framework}_{int(case.row_id)}.html'))
        print('LIME completado',framework,winner,flush=True)
    if spark:spark.stop()
    pd.DataFrame(rows).to_csv(OUT/'lime_explanations.csv',index=False)
if __name__=='__main__':main()
