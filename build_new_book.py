"""Construye informes a partir de artefactos calculados, sin reentrenar."""
from pathlib import Path
import json
import nbformat as nb
import pandas as pd
from entregable_2.app import context, models, eda, update_eda
ROOT=Path(__file__).parent
BOOK=ROOT/'jbook_entregas_2'
STATIC=BOOK/'_static'
D=json.loads((ROOT/'entregable_2/results/summary.json').read_text())
H=json.loads((ROOT/'tarea_2/results/summary.json').read_text())
R=pd.read_csv(ROOT/'tarea_2/results/ranking.csv')
A=pd.read_csv(ROOT/'entregable_2/results/associations.csv')
C=pd.read_csv(ROOT/'entregable_2/results/categorical_eda.csv')
B=pd.read_csv(ROOT/'entregable_2/results/coefficients.csv')
O=json.loads((ROOT/'entregable_2/results/eda_summary.json').read_text())
def table(x): return x.to_markdown(index=False,floatfmt='.5f')
def figures(c):
    if hasattr(c,'figure'): yield c.figure
    children=getattr(c,'children',[])
    if not isinstance(children,(list,tuple)): children=[children]
    for child in children:
        if child is not None and not isinstance(child,(str,int,float)): yield from figures(child)
def embed(fig,name):
    fig.write_html(STATIC/f'{name}.html',include_plotlyjs='directory')
    return f'```{{raw}} html\n<iframe src="_static/{name}.html" width="100%" height="550" style="border:0" title="{name}"></iframe>\n```\n'
metrics=pd.DataFrame([{'modelo':k,**D[k]} for k in ['logistic','dummy']])
revision=f'''# Caries dental: revisión metodológica del 17 de octubre de 2026

Autor: Andrés Duarte Chaves · Docente: Dr. Lihki Rubio.

Esta revisión responde a la retroalimentación del primer entregable y actualiza el modelo base
del segundo. No sustituye el proyecto final ni añade otras familias de modelos al dashboard.
La versión originalmente evaluada permanece en el historial Git. Las cifras siguientes proceden
de la nueva ejecución; no se presentan como resultados de aquella entrega.

## Problema, procedencia y escenario

Se clasifica caries **registrada** en un chequeo; no incidencia futura ni diagnóstico clínico.
Se conservan {D['rows']:,} registros y {D['positive_rows']:,} positivos ({D['prevalence']:.2%}).
El archivo original adjunto y la copia usada coinciden en valores y orden de filas después de
normalizar smoking 0/1 a N/Y y las representaciones numéricas. Sus hashes de bytes son distintos.
Esto no verifica licencia, protocolo clínico ni independencia por paciente.

SHA-256 de la copia utilizada: `{D['sha256']}`.
Ver `entregable_2/results/comparacion_csv_original.json` para la comparación documentada.
Fuente: [Body Signal of Smoking](https://www.kaggle.com/datasets/kucuro3/body-signal-of-smoking).

## Respuesta a las observaciones del profesor

| Observación | Corrección verificable |
|---|---|
| Faltaban categóricas | Conteos, proporciones, chi cuadrado y V de Cramer en entrenamiento |
| Sin corrección múltiple | Benjamini–Hochberg sobre los {len(A)} contrastes de asociación |
| Sin atípicos multivariados | Isolation Forest exploratorio en entrenamiento; sin eliminar filas |
| Tártaro no disponible antes del examen | Exclusión explícita de tartar, además de ID y oral constante |
| Mecanismos clínicos atribuidos a efectos pequeños | Se reportan asociaciones y magnitudes, sin explicaciones causales |
| Escalado y MAR incoherentes | StandardScaler realmente ejecutado; no se presume MAR |
| Sin coeficientes ni CV | Coeficientes regularizados y cinco folds agrupados sin copias compartidas |

## Partición y preprocesamiento

Se agrupan perfiles de los **23 predictores finales**, excluyendo ID, oral, tartar y objetivo.
Un mismo perfil no cruza entrenamiento/prueba ni folds. Hay {D['train_rows']:,} filas train y
{D['test_rows']:,} test. El EDA inferencial usa {O['profiles']:,} perfiles únicos de entrenamiento
para no contabilizar copias como observaciones independientes; esto no demuestra independencia clínica.
Se conservan todas las filas en el ajuste y CV, con ponderación por fila consistente.

Género, tabaquismo, audición izquierda/derecha y proteína urinaria se codifican como categorías.
No se impone una distancia lineal entre códigos ordinales. El resto usa imputación por mediana
y StandardScaler; las categorías usan moda y OneHotEncoder. Cada fold ajusta sus transformaciones
sólo en entrenamiento. Hay {D['missing_values']} celdas vacías en el archivo: no se deduce mecanismo MAR.
Los extremos se conservan; el uso de StandardScaler no se denomina escalado robusto.

## EDA categórico

{table(C)}

Las proporciones dependen del tamaño de cada categoría. Una diferencia observada no implica causa.
Para celdas esperadas pequeñas, chi cuadrado es aproximado; no se usa como prueba clínica confirmatoria.

## Asociaciones y multiplicidad

{table(A)}

Para continuas, el efecto es correlación biserial de rangos (positivo: valores mayores en positivos).
Para categorías se informa V de Cramer, sin signo. No son magnitudes intercambiables.
`q_bh` controla FDR bajo los supuestos del procedimiento; no es probabilidad de que una hipótesis sea cierta.
Un p o q pequeño con efecto pequeño no demuestra relevancia clínica. Las variables no se seleccionan
por estos contrastes ni se atribuyen mecanismos fisiológicos a asociaciones transversales.

## Atípicos multivariados

Isolation Forest identifica {O['flagged']:,} de {O['profiles']:,} perfiles de entrenamiento usando
contaminación prefijada del 5%. Este porcentaje es una decisión exploratoria, **no una estimación de
prevalencia de errores o enfermedad**. Se usa preprocesamiento ajustado sólo en entrenamiento.
No se elimina ninguna fila ni se usa este diagnóstico para modificar el test. Los scores e indicadores
están en `entregable_2/results/outliers.csv`. Requieren revisión de origen antes de cualquier exclusión.

## Modelo base y validación

Regresión logística L2, sin búsqueda de hiperparámetros. Cinco folds StratifiedGroupKFold:
AUC **{D['cv_auc_mean']:.5f} ± {D['cv_auc_sd']:.5f}** (media ± desviación estándar poblacional de folds,
no intervalo de confianza). Tanto CV como ajuste final usan todas las filas y la misma ponderación.
La CV de una familia fija no implica comparación exhaustiva ni validación externa.

{table(metrics)}

AUC test {D['logistic']['roc_auc']:.5f} y AP {D['logistic']['average_precision']:.5f} muestran una
señal limitada. La referencia AP es la prevalencia de prueba {D['test_prevalence']:.5f}.
A umbral 0,5: TP={D['logistic']['tp']}, FN={D['logistic']['fn']}, TN={D['logistic']['tn']}, FP={D['logistic']['fp']}.
Por tanto, la exactitud elevada oculta que casi todos los positivos se pierden. No se recomienda
este umbral para un uso clínico. El slider es retrospectivo y no selecciona cortes en test.
El Brier describe error probabilístico; la curva de calibración no demuestra transportabilidad.

## Coeficientes e interpretación

{table(B)}

Los coeficientes numéricos corresponden a una desviación estándar de entrenamiento, manteniendo
los demás términos fijos. Son asociaciones ajustadas de un modelo penalizado, no efectos causales.
Para una variable continua, exp(coeficiente) es el multiplicador de odds por esa unidad estandarizada.
OneHotEncoder conserva todas las categorías: **exp(coeficiente) de una categoría no es por sí solo
un odds ratio frente a una categoría de referencia**. La comparación A vs B usa exp(beta_A-beta_B).
No se reportan intervalos ni significación inferencial de coeficientes regularizados.
'''
# Descriptivos calculados sólo en perfiles de entrenamiento.
train = pd.read_csv(ROOT/'entregable_2/results/train.csv').drop_duplicates(subset=D['features'])
num = [c for c in D['features'] if c not in D['categorical_features']]
stats = train[num].describe().T.reset_index(names='variable')
stats['IQR'] = stats['75%']-stats['25%']
stats['skew'] = [train[c].skew() for c in num]
stats['tukey_count'] = [int(((train[c]<train[c].quantile(.25)-1.5*train[c].quantile(.75)+1.5*train[c].quantile(.25)) | (train[c]>train[c].quantile(.75)+1.5*(train[c].quantile(.75)-train[c].quantile(.25)))).sum()) for c in num]
stats.to_csv(ROOT/'entregable_2/results/descriptive.csv',index=False)
revision += '\n## Descriptivos univariados y extremos\n\n'+table(stats)+'\n\nTukey es una regla descriptiva; sus extremos no se eliminan automáticamente. Asimetría no demuestra enfermedad ni mecanismo clínico.\n'
# Computed categorical contrast, not a mechanistic interpretation.
for prefix in ['cat__gender_','cat__smoking_']:
    v=B.loc[B.feature.str.startswith(prefix)]
    if len(v)==2:
        import numpy as np
        a,b=v.iloc[0],v.iloc[1]
        revision+=f'\nContraste {a.feature} frente a {b.feature}: OR ajustado descriptivo **{np.exp(a.coefficient-b.coefficient):.3f}**.\n'
revision+='''
## Límites y comparación histórica

La cifra anterior AUC 0,6441 procede de otro escenario y partición; no se interpreta como
reproducción de este experimento ni se atribuye la diferencia únicamente a excluir tártaro.
Un AUC univariado bajo no descarta fuga: importan origen y disponibilidad temporal de cada predictor.
El test es interno y la base se ha usado anteriormente. No existe validación externa ni confirmación
de representatividad colombiana, independencia por persona o protocolo diagnóstico del origen.

## Reproducción

Desde la raíz del repositorio:

```bash
pip install -r requirements_nuevas.txt
python -m entregable_2.train
python -m entregable_2.audit
python build_new_book.py
python -m pytest -q
```
'''
# Teacher-facing notebook: complete computed report plus executable data inspection.
n=nb.v4.new_notebook();n.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
n.cells=[nb.v4.new_markdown_cell(revision),nb.v4.new_code_cell('''from pathlib import Path
import os, sys, json
import pandas as pd
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'ml_common.py').exists())
os.chdir(root)
sys.path.insert(0, str(root))
from entregable_2.train import prepare
_, train, test, X_train, X_test = prepare()
a = set(pd.util.hash_pandas_object(X_train,index=False))
b = set(pd.util.hash_pandas_object(X_test,index=False))
assert not a.intersection(b)
display(pd.read_csv('entregable_2/results/cv.csv'))
display(pd.read_csv('entregable_2/results/coefficients.csv'))
''')]
nb.write(n,ROOT/'entregable_1_clasificacion.ipynb')
(ROOT/'REVISION_ENTREGABLE_1.md').write_text(revision)
visual='\n## Gráficos del dashboard\n'
for group,component in [('contexto',context()),('modelo',models()),('eda',eda())]:
    for i,fig in enumerate(figures(component)): visual+=embed(fig,f'revision_{group}_{i}')
for i,fig in enumerate(update_eda('age','Todos')[:3]): visual+=embed(fig,f'revision_edad_{i}')
BOOK.joinpath('entregable_2.md').write_text(revision+visual)
BOOK.joinpath('tarea_2.md').write_text(f'''# Tarea 2 — Revisión del 17 de octubre de 2026

Se clasifica HeartDisease registrada en Heart Failure Prediction, sin equiparar la etiqueta a
incidencia futura o insuficiencia cardíaca. Hay {H['rows']} filas, {H['train_rows']} train y {H['test_rows']} test.
La partición es estratificada, semilla 42. Los ceros de Cholesterol y RestingBP se convierten a
faltantes por una regla fija, antes de imputar dentro de Pipeline. La variable artificial de fuga
se usa sólo en la demostración y no forma parte del modelo desplegado.

## Selección e incertidumbre

{table(R[['model','cv_auc','cv_auc_sd','roc_auc','average_precision','recall','specificity','f1']])}

`cv_auc_sd` procede de la misma fila ganadora de los archivos cv_*.csv guardados, sin inventar
ni recalcular entrenamientos. Es dispersión entre cinco folds, no intervalo de confianza.
El ganador **{H['winner_by_cv']}** se seleccionó por CV exclusivamente. No se cambia por el ranking
de test. La CV después de búsqueda puede ser optimista y no es anidada. Las diferencias pequeñas
no establecen superioridad definitiva. Los datos no tienen validación externa.

## Errores a umbral 0,5

{table(R[['model','tn','fp','fn','tp']])}

Los falsos negativos corresponden a registros positivos que el modelo no detecta; los falsos
positivos a negativos clasificados como positivos. Ni exactitud ni AUC sustituyen este análisis.
El umbral se mantiene en 0,5; no se optimiza con el test.

## API, Docker y Kubernetes

Se conserva el pipeline completo. La API valida campos, categorías y entradas adicionales;
se verifica equivalencia numérica entre `/predict` y predict_proba. La probabilidad es académica.
La evidencia guardada el 6 de octubre de 2026 en `tarea_2/results/verificacion_despliegue/respuestas.json`
registra `/health` y `/predict` para Docker y Kubernetes: ambos respondieron probabilidad
0,7890747996765479 para la misma entrada. Esta revisión no constituye un nuevo despliegue.

## Monitoreo ilustrativo

`drift_report.html` compara train y test. `drift_simulated.html` añade artificialmente 20 años a Age.
La simulación es un control de funcionamiento, no un evento observado en producción. Cambios de
variables no demuestran caída de desempeño; para ello se necesitan resultados etiquetados posteriores.
Los reportes completos se conservan en `tarea_2/results/` y el resumen se presenta a continuación.

## Reproducción

```bash
python -m tarea_2.train
python -m tarea_2.drift
python -m pytest -q
```

La ejecución Docker/Kubernetes se documenta en `ejecucion.md`; el workflow remoto verifica el código.
''')
# Preserve existing operational guide, which includes actual deployment evidence.
BOOK.joinpath('intro.md').write_text('''# Revisión de las entregas

Andrés Duarte Chaves · Dr. Lihki Rubio · Actualización: 17 de octubre de 2026.

Se actualizan Entregable 1 (retroalimentación), Entregable 2 (dashboard) y Tarea 2 (pipelines).
Tarea 1 incluye once modelos realmente ejecutados; siguen pendientes Random Forest de Spark,
los contrastes completos y LIME del ganador Spark. El historial conserva las entregas originales.

[Dashboard de caries](https://machinelearning-caries-andres-duarte.onrender.com)

[Repositorio y registro de correcciones](https://github.com/aduartech81/MachineLearning)

El dashboard conserva exactamente tres pestañas. Estos ejercicios no se mezclan con el proyecto final.
''')
# Drift summary extraction: scalar metrics only, no speculative interpretation.
for filename in ['drift_report.json','drift_simulated.json']:
    data=json.loads((ROOT/'tarea_2/results'/filename).read_text())
    rows=[]
    count=data['metrics'][0]['value']
    with (BOOK/'tarea_2.md').open('a') as f:
        f.write(f"\n{filename}: {int(count['count'])}/11 variables con alerta ({count['share']:.1%}). La regla global requiere al menos 50%: no se activa en ninguno de los dos escenarios. Age sí activa una alerta individual en la simulación; los contrastes de monitoreo son exploratorios sin ajuste múltiple.\n")
    for m in data.get('metrics',[]):
        value=m.get('value')
        if isinstance(value,(int,float,str,bool)):
            rows.append({'metric':m.get('metric_name',m.get('metric_id',m.get('id',''))),'value':value})
    if rows:
        with (BOOK/'tarea_2.md').open('a') as f:f.write('\n### '+filename+'\n\n'+table(pd.DataFrame(rows))+'\n')
print('Informes y notebook corregidos generados desde artefactos calculados.')
