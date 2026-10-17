from tarea_1.common import prediction_path
"""Informe de resultados ejecutados; muestra explícitamente cualquier pendiente."""
from pathlib import Path
import json
import nbformat as nb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from tarea_1.common import ROOT,OUT,NAMES

def table(df):return df.to_markdown(index=False,floatfmt='.5f')
def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    rows=[];pending=[]
    for framework in ['sklearn','spark']:
        for name in NAMES:
            path=OUT/f'{framework}_{name}.json'
            if path.exists():rows.append(json.loads(path.read_text()))
            else:pending.append(f'{framework}: {name}')
    ranking=pd.DataFrame(rows);ranking.to_csv(OUT/'ranking.csv',index=False)
    stats_done=(OUT/'paired_tests.csv').exists() and (OUT/'statistics_validation.json').exists()
    lime_done=(OUT/'lime_explanations.csv').exists()
    complete=not pending and stats_done and lime_done
    status='Ejecución completa de modelos, contrastes y explicaciones.' if complete else 'EN PROCESO: no se afirma cumplimiento completo.'
    text=f'''# Tarea 1 — Evaluación de modelos y riesgo crediticio

Andrés Duarte Chaves · Maestría en Ingeniería Biomédica · Dr. Lihki Rubio.
Revisión del 17 de octubre de 2026. **{status}**

Esta versión sustituye la tabla manual y la ROC aproximada de Spark de la entrega anterior.
Cada resultado mostrado procede de una ejecución guardada. El historial Git conserva la versión previa.

## Datos y objetivo

CSV real completo recuperado de ML_PROYECTOS en Archivo.zip, de {protocol['bytes']:,} bytes.
SHA-256: `{protocol['sha256']}`. No se usa el CSV sintético de la otra carpeta.
Se incluyen los {protocol['eligible_rows']:,} préstamos con estado terminal Fully Paid (0) o Charged Off (1).
Los demás estados no se etiquetan como pagos exitosos: todavía no representan desenlaces terminales.
No se realiza muestreo del entrenamiento supervisado. Se leen las 13 variables preespecificadas,
además del estado; la proyección de columnas no reduce el número de filas.

La tarea clasifica el desenlace observado entre préstamos aprobados. No representa todas las
solicitudes de crédito ni prueba causalidad. Tasa y grado de préstamo ya incorporan decisiones
de originación, por lo que el escenario es posterior a su asignación, no anterior a la oferta.
El identificador es de fila, no confirma independencia por prestatario. Una partición aleatoria
no evalúa cambios temporales; esa limitación afecta también los contrastes de independencia.

## Partición y reproducibilidad

Train: {protocol['train_rows']:,}; test: {protocol['test_rows']:,}. Semilla 42, estratificación 80/20.
Los IDs y los tres folds se guardan antes del EDA; ambos motores usan exactamente esa asignación.
El archivo canónico resuelve la discrepancia previa de una fila entre lectores. Las transformaciones
se aprenden de nuevo en cada fold y en el ajuste final sólo con entrenamiento.
Mediana numérica, Missing para categorías ausentes y one-hot con categorías de entrenamiento.
En los modelos lineales se estandarizan sólo las variables numéricas; las categorías mantienen
codificación binaria. sklearn usa varianza poblacional y Spark muestral al escalar: diferencia pequeña,
pero no se presentan transformaciones idénticas bit a bit.

## Hardware y costos

Límite efectivo 8 GiB, cuota de 8 CPU; modelos con hasta 4 hilos. Spark local[4], driver 4 GiB y
16 particiones. Es una ejecución local, no una prueba de escalabilidad multinodo.
Spark se inició mientras scikit-learn continuaba con SVM y RF, con hasta cuatro hilos por motor.
Los tiempos concurrentes pueden verse afectados por memoria y carga compartida: **no se concluye
superioridad de velocidad entre bibliotecas**. Las métricas predictivas usan los mismos IDs y folds.
GridSearchCV usa un estimador compuesto sin Pipeline que aprende transformaciones en cada fit.
La búsqueda usa una tarea simultánea salvo SVM (dos hilos de búsqueda compartiendo memoria); RF usa 4 hilos.
SVM usa hinge, max_iter=2000 y tol=1e-3. Se registra cada ajuste en svc_convergence.jsonl.
La primera búsqueda con 10000 iteraciones se interrumpió por costo antes de obtener resultados;
su tiempo no se atribuye a la ejecución completada. Un ajuste que llegue al límite no demuestra convergencia.
Spark utiliza ParamGridBuilder y CrossValidator de tres folds, con vectores materializados y
persistidos dentro de cada fold. Los tiempos CV incluyen transformaciones y ajuste final;
lectura, inferencia y transferencia de predicciones se reportan por separado cuando aplican.
No deben sumarse de nuevo los tiempos parciales de preprocesamiento que ya están incluidos en fit.

HistGradientBoosting sustituye a GradientBoosting de sklearn por costo de cómputo, como permite
la guía actual. Usa 50/100 iteraciones, profundidad 3/5, tasa 0,1, sin parada temprana ni límite de
hojas adicional. Spark usa GBT. HistGradientBoosting usa 255 bins y Spark 64: los algoritmos no son
idénticos. RF y árboles también difieren en cortes y convenciones internas. GaussianNB usa
estabilización propia de cada biblioteca. La correspondencia C–regParam no demuestra equivalencia
numérica exacta; se calcula C con el número de filas del fit de cada fold.

## Resultados calculados

'''
    if len(ranking):
        text+=table(ranking[['framework','model','cv_auc','cv_sd','roc_auc','accuracy','precision','recall','f1','fit_cv_seconds','predict_seconds']])+'\n'
        text+='\n### Matrices de confusión a corte fijo\n\n'+table(ranking[['framework','model','tn','fp','fn','tp']])+'\n'
    if pending:text+='\n**Pendientes:** '+', '.join(pending)+'.\n'
    text+='''
Los clasificadores probabilísticos usan corte 0,5; SVM utiliza margen cero. La selección de modelos
se basa en AUC de validación, sin escoger por test. Exactitud elevada puede ocultar baja sensibilidad
ante la clase minoritaria. AUC no determina por sí sola costos o utilidad para conceder crédito.

## EDA exclusivo de entrenamiento

'''
    for filename,title in [('descriptive.csv','Descriptivos y Tukey'),('missing.csv','Faltantes'),('associations.csv','Asociaciones y tamaños de efecto'),('categories.csv','Categorías'),('categorical_pairs.csv','Asociaciones entre categorías'),('missing_associations.csv','Faltantes frente al desenlace')]:
        path=OUT/filename
        if path.exists():text+='\n### '+title+'\n\n'+table(pd.read_csv(path))+'\n'
    text+='''
Los contrastes de asociación usan ajuste Benjamini–Hochberg. Un tamaño de muestra grande puede
producir valores p muy pequeños para efectos modestos. El efecto biserial tiene signo; V de Cramer
no lo tiene. Una asociación entre faltantes y desenlace cuestiona MCAR, pero su ausencia no prueba
MCAR ni justifica asumir MAR. Los extremos se conservan; Tukey no diagnostica errores.

'''
    for name in ['histograms','boxplots','correlation','categories']:
        if (ROOT/'figures'/f'{name}.png').exists():text+=f'![{name}](tarea_1/figures/{name}.png)\n\n'
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    for framework in ['sklearn','spark']:
        fig,ax=plt.subplots(figsize=(8,6));count=0
        for name in NAMES:
            path=prediction_path(framework,name)
            if not path.exists():continue
            pred=pd.read_csv(path);fpr,tpr,_=roc_curve(pred.actual,pred.score)
            ax.plot(fpr,tpr,label=name);count+=1
        if count:
            ax.plot([0,1],[0,1],'k--');ax.set(xlabel='FPR',ylabel='Sensibilidad',title=f'ROC reales — {framework}');ax.legend(fontsize=8)
            fig.tight_layout();fig.savefig(figdir/f'roc_{framework}.png',dpi=140)
            text+=f'![ROC {framework}](tarea_1/figures/roc_{framework}.png)\n\n'
        plt.close(fig)
    convergence=OUT/'svc_convergence.jsonl'
    if convergence.exists():
        text+='\n### Convergencia de SVM\n\n'+table(pd.read_json(convergence,lines=True))+'\n'
    text+='''## Comparación estadística

DeLong se implementa con rangos, O(n log n), y se verifica frente al cálculo directo en una muestra
pequeña con empates. Compara AUC correlacionadas sobre los mismos IDs. Las familias son seis pares
entre motores y quince pares dentro de cada motor. Holm se aplica por familia. McNemar compara
decisiones a corte fijo; no es otra prueba de AUC. El bootstrap es pareado y estratificado por clase,
con 1.000 remuestras de IDs y semilla 42. Los intervalos son percentiles.
El margen descriptivo de magnitud |ΔAUC| ≥ 0,005 se especificó en el protocolo antes de comparar
motores; no define utilidad económica ni una prueba de equivalencia. Estos análisis capturan
variación del test, no la del reentrenamiento, ni corrigen dependencia por prestatario desconocida.

'''
    if stats_done:
        comparisons=pd.read_csv(OUT/'paired_tests.csv')
        text+=table(comparisons)+'\n'
        for framework in ['sklearn','spark']:
            sub=comparisons[comparisons.family==framework]
            for field in ['delta_auc','p_holm']:
                matrix=np.full((6,6),np.nan);np.fill_diagonal(matrix,0 if field=='delta_auc' else 1)
                for row in sub.itertuples():
                    a=NAMES.index(row.a.removeprefix(framework+'_'));b=NAMES.index(row.b.removeprefix(framework+'_'))
                    value=getattr(row,field);matrix[a,b]=value;matrix[b,a]=-value if field=='delta_auc' else value
                fig,ax=plt.subplots(figsize=(9,7));im=ax.imshow(matrix,cmap='RdBu_r' if field=='delta_auc' else 'viridis')
                ax.set_xticks(range(6),NAMES,rotation=45,ha='right');ax.set_yticks(range(6),NAMES);fig.colorbar(im,ax=ax)
                ax.set_title(f'{framework}: {field}');fig.tight_layout();filename=f'{framework}_{field}.png';fig.savefig(figdir/filename,dpi=140);plt.close(fig)
                text+=f'![{framework} {field}](tarea_1/figures/{filename})\n\n'
    else:text+='**Pendiente:** no se generan contrastes hasta completar las doce series de predicciones.\n'
    text+='\n## Interpretabilidad local\n\n'
    if lime_done:text+=table(pd.read_csv(OUT/'lime_explanations.csv'))+'\n'
    else:text+='LIME está pendiente de terminar los modelos y escoger el ganador probabilístico por CV en cada entorno.\n'
    text+='''
LIME utiliza fondo de 5.000 registros de entrenamiento y 2.000 perturbaciones por caso; este
muestreo sólo corresponde a la explicación, no al entrenamiento supervisado. Las categorías se
perturban como categorías y los faltantes numéricos del caso se representan por la mediana train.
Se explican hasta dos errores por motor. R² local informa fidelidad del sustituto; pesos locales
no son efectos causales ni una explicación global. Correlaciones y perturbaciones poco plausibles
limitan la interpretación. Las explicaciones no justifican decisiones financieras reales.

## Reproducción

```bash
pip install -r tarea_1/requirements.txt
python -m tarea_1.prepare
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m tarea_1.run_sklearn
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m tarea_1.run_spark
python -m tarea_1.eda
python -m tarea_1.statistics
python -m tarea_1.explain
python -m tarea_1.report
```

Se requiere Java 17 y el CSV real en la raíz. No se sintetizan sustitutos. Cada familia guarda
resultados al completar el ajuste; los archivos existentes se omiten al reanudar. Para cambiar
el protocolo debe usarse una carpeta nueva de resultados, sin mezclar ejecuciones.

## Fuentes

- [Guía docente de evaluación de modelos](https://lihkir.github.io/MachineLearning/model_evaluation.html)
- DeLong, DeLong y Clarke-Pearson (1988). Comparing the areas under two or more correlated ROC curves. Biometrics 44:837–845.
- Sun y Xu (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated ROC curves. IEEE Signal Processing Letters 21:1389–1393.
'''
    (ROOT.parent/'REVISION_TAREA_1.md').write_text(text)
    n=nb.v4.new_notebook();n.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    n.cells=[nb.v4.new_markdown_cell(text),nb.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'tarea_1/results/ranking.csv').exists())\ndisplay(pd.read_csv(root/'tarea_1/results/ranking.csv'))")]
    nb.write(n,ROOT.parent/'09_evaluacion_modelos.ipynb')
    print(status,flush=True)
if __name__=='__main__':main()
