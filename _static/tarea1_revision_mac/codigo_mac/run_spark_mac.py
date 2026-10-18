"""CV nativa de Spark sobre los mismos IDs y folds que scikit-learn."""
import sys,time,json
from pathlib import Path
from pyspark.sql import SparkSession,functions as F,types as T
from pyspark.ml import Pipeline,Transformer
from pyspark.ml.util import DefaultParamsReadable,DefaultParamsWritable
from pyspark.ml.feature import Imputer,StringIndexer,OneHotEncoder,VectorAssembler,StandardScaler
from pyspark.ml.classification import LogisticRegression,DecisionTreeClassifier,RandomForestClassifier,GBTClassifier,LinearSVC,NaiveBayes
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.storagelevel import StorageLevel
import numpy as np
import pandas as pd
from tarea_1.common import OUT,NUM,CAT,metrics

from tarea_1.spark_components import CacheVectors

def main():
    spark=(SparkSession.builder.master('local[2]').appName('Tarea1_LendingClub')
      .config('spark.driver.memory','3g').config('spark.driver.host','127.0.0.1')
      .config('spark.driver.bindAddress','127.0.0.1').config('spark.ui.enabled','false').config('spark.ui.showConsoleProgress','false')
      .config('spark.sql.shuffle.partitions','16').config('spark.default.parallelism','16')
      .config('spark.memory.fraction','0.65').config('spark.memory.storageFraction','0.3').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    fields=[T.StructField(c,T.DoubleType()) for c in NUM]+[T.StructField(c,T.StringType()) for c in CAT]
    # canonical.csv conserva el orden del CSV fuente: usar cabecera para construir esquema exacto.
    columns=pd.read_csv(OUT/'canonical.csv',nrows=0).columns
    schema=T.StructType([T.StructField(c,T.StringType() if c in CAT+['split'] else T.DoubleType()) for c in columns])
    t=time.perf_counter()
    data=spark.read.csv(str(OUT/'canonical.csv'),header=True,schema=schema).fillna('Missing',subset=CAT)
    data=data.withColumn('row_id',F.col('row_id').cast('long')).withColumn('fold',F.col('fold').cast('int')).withColumn('label',F.col('default'))
    train=data.filter(F.col('split')=='train');test=data.filter(F.col('split')=='test')
    ntrain,ntest=train.count(),test.count();read_time=time.perf_counter()-t
    assert ntrain==1076248 and ntest==269062
    selected=sys.argv[1:]
    for name in ['GaussianNB','LogisticRegression','DecisionTree','HistGradientBoosting','LinearSVC','RandomForest']:
        if selected and name not in selected:continue
        dest=OUT/f'spark_{name}.json'
        if dest.exists():print('Ya calculado',name,flush=True);continue
        print('INICIO Spark',name,flush=True)
        spark.catalog.clearCache()
        stages=[Imputer(inputCols=NUM,outputCols=[c+'_imp' for c in NUM],strategy='median',relativeError=0.0)]
        for c in CAT:
            stages += [StringIndexer(inputCol=c,outputCol=c+'_idx',handleInvalid='keep',stringOrderType='alphabetAsc'),
                       OneHotEncoder(inputCol=c+'_idx',outputCol=c+'_vec',dropLast=False)]
        scale=name in ['LogisticRegression','LinearSVC']
        if scale:
            stages += [VectorAssembler(inputCols=[c+'_imp' for c in NUM],outputCol='numeric_raw'),
                       StandardScaler(inputCol='numeric_raw',outputCol='numeric_scaled',withMean=True,withStd=True)]
        inputs=(['numeric_scaled'] if scale else [c+'_imp' for c in NUM])+[c+'_vec' for c in CAT]
        stages += [VectorAssembler(inputCols=inputs,outputCol='features')]
        feature_col='features'
        common=dict(labelCol='label',featuresCol=feature_col)
        grid=ParamGridBuilder()
        if name=='GaussianNB':model=NaiveBayes(modelType='gaussian',**common)
        elif name=='LogisticRegression':
            model=LogisticRegression(elasticNetParam=0.,standardization=False,maxIter=1000,tol=1e-6,**common)
            grid.addGrid(model.regParam,[1e-6,1e-5,1e-4])
        elif name=='LinearSVC':
            model=LinearSVC(standardization=False,maxIter=1000,tol=1e-6,**common)
            grid.addGrid(model.regParam,[1e-6,1e-5,1e-4])
        elif name=='DecisionTree':
            model=DecisionTreeClassifier(seed=42,maxBins=64,**common);grid.addGrid(model.maxDepth,[5,10,15])
        elif name=='RandomForest':
            model=RandomForestClassifier(seed=42,maxBins=64,**common)
            grid.addGrid(model.numTrees,[10,50,100]).addGrid(model.maxDepth,[5,10,15])
        else:
            model=GBTClassifier(seed=42,maxBins=64,stepSize=.1,**common)
            grid.addGrid(model.maxIter,[50,100]).addGrid(model.maxDepth,[3,5])
        stages += [model]
        params=grid.build()
        evaluator=BinaryClassificationEvaluator(labelCol='label',metricName='areaUnderROC',numBins=0,rawPredictionCol=('rawPrediction' if name=='LinearSVC' else 'probability'))
        cv=CrossValidator(estimator=Pipeline(stages=stages),estimatorParamMaps=params,evaluator=evaluator,
                          numFolds=3,foldCol='fold',parallelism=1,seed=42)
        t=time.perf_counter();fit=cv.fit(train);fit_time=time.perf_counter()-t
        t=time.perf_counter()
        col='rawPrediction' if name=='LinearSVC' else 'probability'
        pred=fit.transform(test).select('row_id','label','prediction',vector_to_array(col)[1].alias('score')).persist(StorageLevel.MEMORY_AND_DISK)
        assert pred.count()==ntest
        predict_time=time.perf_counter()-t
        t=time.perf_counter()
        # Transferencia limitada a IDs, etiquetas y predicciones DESPUÉS del ajuste y predicción.
        frame=pred.toPandas().rename(columns={'label':'actual'})
        transfer_time=time.perf_counter()-t
        frame.to_csv(OUT/f'spark_{name}_predictions.csv',index=False)
        avg=np.asarray(fit.avgMetrics);best=int(avg.argmax())
        param_rows=[{**{p.name:v for p,v in item.items()},'cv_auc':float(avg[i])} for i,item in enumerate(params)]
        pd.DataFrame(param_rows).to_csv(OUT/f'spark_{name}_cv.csv',index=False)
        result=dict(framework='spark',model=name,implementation=type(model).__name__,rows=ntrain+ntest,
                    train=ntrain,test=ntest,cv_auc=float(avg[best]),
                    cv_sd=float(fit.stdMetrics[best]) if hasattr(fit,'stdMetrics') else None,
                    read_seconds=read_time,fit_cv_seconds=fit_time,predict_seconds=predict_time,
                    transfer_seconds=transfer_time,parameters=param_rows[best],maxBins=64,
                    **metrics(frame.actual,frame.score,frame.prediction))
        fit.bestModel.write().save(str((OUT/f'spark_{name}_model').resolve()))
        dest.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    spark.stop()
if __name__=='__main__':main()
