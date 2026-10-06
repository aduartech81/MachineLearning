"""Genera informe y notebooks a partir de resultados realmente calculados."""
import json
from pathlib import Path
import nbformat as nbf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve
from entregable_2.app import context, models, update_eda

ROOT = Path(__file__).parent
BOOK = ROOT/'jbook_entregas_2'
STATIC = BOOK/'_static'
STATIC.mkdir(parents=True, exist_ok=True)
D = json.loads((ROOT/'entregable_2/results/summary.json').read_text())
H = json.loads((ROOT/'tarea_2/results/summary.json').read_text())
R = pd.read_csv(ROOT/'tarea_2/results/ranking.csv')


def collect_figures(component):
    if hasattr(component, 'figure'):
        yield component.figure
    children = getattr(component, 'children', [])
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is not None and not isinstance(child, (str, int, float)):
            yield from collect_figures(child)


def save_figure(fig, name):
    fig.write_html(STATIC/f'{name}.html', include_plotlyjs='directory')
    return f'```{{raw}} html\n<iframe src="_static/{name}.html" width="100%" height="540" style="border:0" title="{name}"></iframe>\n```\n'


def table(df):
    return df.to_markdown(index=False, floatfmt='.4f')


def notebook(path, title, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(title), nbf.v4.new_code_cell(
        'from pathlib import Path\nimport os, sys\n'
        '# Puede abrirse desde la raíz o desde la carpeta de notebooks.\n'
        'root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/"ml_common.py").exists())\n'
        'os.chdir(root)\nsys.path.insert(0, str(root))\n')]
    for kind, source in cells:
        n.cells.append(nbf.v4.new_code_cell(source) if kind == 'code' else nbf.v4.new_markdown_cell(source))
    n.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(n, path)


BOOK.joinpath('_config.yml').write_text('''title: "Machine Learning — Tarea 2 y Entregable 2"
author: "Andrés Duarte Chaves"
execute:
  execute_notebooks: "off"
html:
  use_repository_button: true
repository:
  url: "https://github.com/aduartech81/MachineLearning"
''')
BOOK.joinpath('_toc.yml').write_text('''format: jb-book
root: intro
chapters:
  - file: entregable_2
  - file: tarea_2
  - file: ejecucion
''')
BOOK.joinpath('intro.md').write_text('''# Tarea 2 y Entregable 2

**Autor:** Andrés Duarte Chaves · **Docente:** Dr. Lihki Rubio
**Programa:** Maestría en Ingeniería Biomédica · **Fecha de ejecución:** 4 de octubre de 2026

Este informe acompaña dos productos: un dashboard Dash Plotly sobre caries dental y un proyecto
integrador de clasificación de enfermedad cardíaca mediante pipelines y herramientas de MLOps.
Los productos conservan objetivos y datasets distintos. Los resultados provienen de código ejecutado;
los archivos anteriores entregados al docente no se modificaron.

El dashboard contiene exactamente tres pestañas. El informe interpreta los gráficos y presenta
las decisiones metodológicas, limitaciones, procedencia y detalles de reproducción adicionales.
El despliegue público de Dash y la ejecución de Docker/Minikube siguen pendientes; no se presentan
como servicios publicados ni pruebas ya realizadas.
''')

charts = list(collect_figures(context()))
eda = update_eda('age', 'Todos')
model_figs = list(collect_figures(models()))
metrics = pd.DataFrame([{'Modelo': name, **D[key]} for name, key in [('Logística', 'logistic'), ('Dummy', 'dummy')]])
fields = ['Modelo', 'roc_auc', 'average_precision', 'accuracy', 'recall', 'specificity', 'f1', 'brier']
text = f'''# Entregable 2 — Dashboard de caries dental

## 1. Contexto del problema

Se clasifica la presencia de caries **registrada en un chequeo**, no el riesgo de desarrollarla en el
futuro. La tarea es binaria: `dental caries=1` frente a `0`. Una etiqueta negativa no significa que
el individuo esté sano respecto de otras enfermedades. La motivación es explorar cuánto discrimina
un modelo lineal a partir de variables generales de chequeo. No se propone un dispositivo diagnóstico.

La copia utilizada contiene **{D['rows']:,} filas**, **{D['positive_rows']:,} positivas** y una proporción
positiva de **{D['prevalence']:.2%}**. Estos conteos coinciden con el primer entregable. La fuente inmediata
es [la copia pública en GitHub](https://github.com/aliabdallah7/smoking-detection/blob/main/smoking.csv),
asociada al dataset [Body Signal of Smoking](https://www.kaggle.com/datasets/kucuro3/body-signal-of-smoking).
No se ha comprobado su identidad byte a byte con el archivo original del estudiante; tampoco se
verificaron de forma independiente los supuestos de edad, independencia por persona o licencia del origen.
SHA-256 del archivo utilizado: `{D['sha256']}`.

### Interpretación del gráfico de clases

La clase negativa domina la base. Un clasificador mayoritario alcanza una exactitud elevada sin
detectar positivos. Por ello se presentan discriminación, precision–recall y sensibilidad además de exactitud.

{save_figure(charts[0], 'caries_clases')}

## 2. Auditoría y partición

El archivo tiene **{D['missing_values']} faltantes** y **{D['duplicates_without_ID']:,} filas duplicadas**
cuando se elimina `ID`. No se presupone MAR ni se confirma independencia individual a partir de un ID
de fila. Se construyen grupos mediante hash de los predictores originales; los perfiles únicos se
dividen de forma estratificada, con semilla 42, y cada copia se conserva en el grupo de su perfil.
Quedan **{D['train_rows']:,} filas de entrenamiento** y **{D['test_rows']:,} de prueba**. Las proporciones
de filas no son exactamente 80/20 porque la unidad de asignación es el perfil, no cada copia.

El modelo utiliza **{len(D['features'])} predictores**. Se excluyen `ID`, `oral` (constante) y `tartar`.
La exclusión de tártaro define un escenario previo al examen oral: su disponibilidad temporal antes
del registro de caries no está demostrada. La codificación categórica aplica a género y tabaquismo;
los restantes valores numéricos se tratan como numéricos, incluidas variables ordinales codificadas.

Un AUC univariado bajo no demuestra ausencia de fuga. La revisión de fuga exige verificar el origen
de cada variable, disponibilidad en el escenario propuesto, separación de grupos y ajuste de
transformaciones sólo con entrenamiento. La protección frente a duplicados no sustituye una
partición por paciente cuando existan identificadores clínicos fiables.

## 3. EDA del entrenamiento

El dashboard permite seleccionar variable numérica y sexo registrado. Frecuencias, mediana,
dispersión y proporción de caries describen el conjunto seleccionado; no se interpretan como
efectos causales. El informe muestra la selección inicial (edad, ambos sexos).

### Distribución y dispersión de la edad

Los histogramas usan conteos: una barra mayor puede reflejar más registros en una clase. El boxplot
permite comparar posición y dispersión. La edad está registrada en valores discretizados, por lo que
no debe interpretarse como una medida continua de resolución anual en todos los casos.

{save_figure(eda[0], 'caries_histograma')}
{save_figure(eda[1], 'caries_boxplot')}

### Proporción de caries por edad

Las barras presentan positivos sobre el total del grupo de edad. El tamaño de cada grupo aparece
en el cursor del gráfico. Un grupo pequeño ofrece una estimación menos estable. Diferencias entre
edades pueden involucrar factores de selección y confusión; no representan una trayectoria individual.

{save_figure(eda[2], 'caries_edad')}

### Correlaciones

La matriz corresponde al entrenamiento completo y no cambia con los filtros. Las correlaciones
entre variables de tamaño corporal y laboratorio señalan redundancia potencial. El escalado no
elimina colinealidad; la regresión logística emplea regularización L2 por defecto. Valores extremos
se conservan y no se clasifican automáticamente como errores o enfermedad específica.

{save_figure(list(collect_figures(__import__('entregable_2.app', fromlist=['eda']).eda()))[0], 'caries_correlacion')}

## 4. Modelo base y resultados

El Pipeline contiene `ColumnTransformer`, imputación por mediana/moda, `StandardScaler`,
`OneHotEncoder(handle_unknown='ignore')` y `LogisticRegression(max_iter=2000)`.
Las transformaciones se aprenden exclusivamente en entrenamiento. El imputador permite manejar
faltantes futuros aunque este archivo no tenga celdas vacías.

La validación cruzada usa cinco pliegues estratificados sobre perfiles únicos de los predictores
finales de entrenamiento: AUC **{D['cv_auc_mean']:.4f} ± {D['cv_auc_sd']:.4f}** (media ± desviación estándar,
no intervalo de confianza). El modelo final usa todas las filas de entrenamiento; por tanto las copias
ponderan algunos perfiles más que otros. Esa diferencia de ponderación limita la comparación directa
entre el estimador de CV y el de prueba. El modelo base no incluye búsqueda de hiperparámetros.

### Tabla comparativa, umbral 0,5

{table(metrics[fields])}

La AUC de prueba de **{D['logistic']['roc_auc']:.4f}** indica discriminación modesta. AP
**{D['logistic']['average_precision']:.4f}** supera la referencia de prevalencia de prueba
**{D['test_prevalence']:.4f}**. AP es un resumen ponderado de precision–recall y no debe confundirse con
integración trapezoidal. Brier menor sugiere menor error cuadrático probabilístico frente a Dummy.
No se concluye utilidad clínica a partir de una mejora estadística aislada.

{save_figure(model_figs[0], 'caries_roc')}
{save_figure(model_figs[1], 'caries_pr')}

### Matriz de confusión y control de umbral

Con corte 0,5 se observan TN={D['logistic']['tn']}, FP={D['logistic']['fp']},
FN={D['logistic']['fn']} y TP={D['logistic']['tp']}. La sensibilidad resulta casi nula, aunque la
exactitud se aproxime a la del clasificador mayoritario. El control de umbral del dashboard muestra
el intercambio entre sensibilidad y especificidad sobre las mismas predicciones. Es una herramienta
descriptiva, **no un procedimiento para elegir el mejor corte en test**. Un corte operativo debe fijarse
en validación de entrenamiento con un criterio de costos explícito y después evaluarse en prueba.

### Calibración

Cada punto compara probabilidad media y frecuencia positiva en un decil. La cercanía a la diagonal
es evidencia descriptiva de calibración en esta muestra, sin demostrar equivalencia al riesgo clínico
ni transportabilidad. No se ha realizado calibración adicional o validación externa.

{save_figure(model_figs[2], 'caries_calibracion')}

## 5. Comparación con el primer entregable y límites

El primer informe reportó AUC 0,6441 y AP 0,3156. Aquí se usan una partición que agrupa copias y
un escenario sin tártaro; por ello **no son reproducciones directas del mismo experimento**.
No puede atribuirse la diferencia a un único cambio. También falta comprobar la identidad del CSV
original. No se reutiliza la cifra anterior como resultado nuevo.

El test pertenece a la misma base pública que se exploró anteriormente: no se presenta como una
validación externa ni totalmente inédita. No hay confirmación de independencia por paciente,
representatividad colombiana, protocolo diagnóstico de caries o secuencia temporal de las variables.
La magnitud de n/p no garantiza ausencia de sobreajuste y una curva de aprendizaje no demuestra
que aumentar la muestra nunca pueda mejorar el modelo. Las conclusiones se restringen al experimento.
'''
BOOK.joinpath('entregable_2.md').write_text(text)

heart_preds = pd.read_csv(ROOT/'tarea_2/results/predictions.csv')
fig = go.Figure()
for name in R.model:
    fpr, tpr, _ = roc_curve(heart_preds.actual, heart_preds[name])
    fig.add_scatter(x=fpr, y=tpr, name=name)
fig.add_scatter(x=[0, 1], y=[0, 1], name='Azar', line_dash='dash')
fig.update_layout(title='ROC: predicciones reales de prueba', xaxis_title='FPR', yaxis_title='Sensibilidad')
BOOK.joinpath('tarea_2.md').write_text(f'''# Tarea 2 — Proyecto integrador con pipelines

## 1. Lineamientos y objetivo

El enlace docente corresponde actualmente a **10.20 Proyecto Integrador**, aunque la consigna
lo llama 10.10. Se implementan estructura modular, demostración de fuga, pipelines con búsqueda,
API FastAPI, Docker, manifiestos Kubernetes, GitHub Actions y reportes Evidently.

Se utiliza Heart Failure Prediction (918 registros, 11 predictores y `HeartDisease` binaria).
El nombre comercial del dataset **no permite equiparar la etiqueta HeartDisease con insuficiencia
cardíaca ni con incidencia futura**. Se clasifica el estado registrado de enfermedad cardíaca.
Fuente original: [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
Archivo obtenido de [copia pública en GitHub](https://github.com/amankataria100/Data-Visualization-on-Heart-Disease-Dataset/blob/main/heart.csv).
SHA-256: `{H['sha256']}`. La copia tiene 918 filas y 508 positivos. No se sustituye silenciosamente
por la base UCI de diferente esquema ni se cambia el target a `target`.

## 2. Preprocesamiento y fuga

Se reservan {H['test_rows']} filas para prueba mediante partición estratificada, semilla 42;
las {H['train_rows']} restantes se usan para entrenamiento y validación cruzada.
Se interpretan los ceros de colesterol ({H['zero_cholesterol']}) y presión en reposo
({H['zero_resting_bp']}) como valores no plausibles para estas mediciones, se reemplazan por NaN
y se imputan dentro del pipeline. La regla es fija y se documenta; no se aprende del test.

El ejemplo docente añade `leaky_feature=y+ruido` y no la elimina en el supuesto flujo correcto.
Se corrige: la variable sólo existe en el experimento contaminado y **nunca se incorpora al modelo
válido**. La demostración contaminada obtiene AUC {H['leakage_demo']['roc_auc']:.4f}; un rendimiento
perfecto es aquí producto de acceso artificial a la respuesta, no de descubrimiento clínico.
Un Pipeline evita fuga por ajuste de transformaciones, pero no elimina automáticamente proxies del target.

## 3. Entrenamiento y comparación

Se comparan seis clasificadores: regresión logística, SVC, Random Forest, KNN, Gradient Boosting
y Gaussian Naive Bayes. Cada uno se integra con el mismo preprocesamiento dentro de `Pipeline`
y se optimiza con `GridSearchCV`, cinco pliegues estratificados y ROC AUC.
El ranking se ordena por AUC de validación cruzada, no por AUC de prueba. La selección de
hiperparámetros y modelo queda dentro de entrenamiento. El ranking de CV tras selección puede
ser optimista; no se presenta como una estimación de validación cruzada anidada.

{table(R[['model', 'cv_auc', 'roc_auc', 'accuracy', 'recall', 'f1', 'seconds']])}

El modelo seleccionado por CV es **{H['winner_by_cv']}**. Otro modelo puede alcanzar mayor AUC
en test sin que eso autorice a cambiar el ganador usando ese mismo test. Las diferencias pequenas
no permiten afirmar superioridad definitiva sin análisis de incertidumbre o validación externa.
Los tiempos son mediciones de esta ejecución local y no una comparación entre plataformas.

{save_figure(fig, 'heart_roc')}

## 4. API, contenedor y Kubernetes

Se guarda el pipeline completo como `model.joblib`. La API acepta campos nombrados, limita categorías,
rechaza campos extra y comprueba rangos básicos. Los campos numéricos nulos se imputan con parámetros
de entrenamiento. El endpoint `/health` comprueba la carga del modelo. `/predict` devuelve probabilidad,
clase a umbral 0,5 y alcance académico. La probabilidad no está calibrada para atención clínica.

Docker incluye las dependencias y el artefacto. Kubernetes configura Deployment, Service NodePort,
readiness probe y límites de recursos. **Docker y Kubernetes no se ejecutaron en este entorno**, que
no dispone de sus comandos. Sus archivos se entregan para validación local con las instrucciones del
capítulo siguiente; no se afirma que la etapa de orquestación esté desplegada.

## 5. Integración continua

El workflow de GitHub Actions instala dependencias, verifica errores esenciales de sintaxis/nombres
y ejecuta pruebas de separación de perfiles, equivalencia entre API y pipeline, rechazo de entradas
inválidas y callbacks del dashboard. Las pruebas se ejecutaron localmente. El workflow remoto
queda pendiente hasta publicar los archivos y observar una ejecución satisfactoria en GitHub.

## 6. Monitoreo ilustrativo con Evidently

Se generan `drift_report.html` y `drift_simulated.html`. El primero compara train contra test y
el segundo agrega **20 años artificiales a Age** como control positivo de cambio. La alteración
es simulada y se declara explícitamente; no representa pacientes nuevos ni un evento real.
Los reportes son archivos exportados, no un servicio continuo de monitoreo en producción.
La deriva de distribuciones no implica por sí sola caída de desempeño ni cambio causal.

## 7. Fuentes metodológicas

- [Consigna de pipelines del docente](https://lihkir.github.io/MachineLearning/chains_pipelines.html#proyecto-integrador-de-aprendizaje-automatico)
- [Prevención de fuga: scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html)
- [Reportes: Evidently](https://docs.evidentlyai.com/docs/library/report)
- [Publicación de Dash](https://dash.plotly.com/deployment)
''')

BOOK.joinpath('ejecucion.md').write_text('''# Ejecución y estado de entrega

## Preparar el entorno

Desde la raíz del repositorio, con Python 3.11 o 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_nuevas.txt
python -m entregable_2.train
python -m tarea_2.train
python -m tarea_2.drift
python build_new_book.py
python -m pytest -q
jupyter-book build jbook_entregas_2 --warningiserror
```

Los entrenamientos requieren los CSV incluidos y nunca generan sustitutos sintéticos cuando falta un archivo.
Los notebooks también se pueden abrir en Jupyter; localizan la raíz del proyecto y usan sus módulos.
`requirements_lock.txt` registra las versiones de los paquetes principales que se usaron en esta ejecución.

## Abrir dashboard y API

```bash
python -m entregable_2.app
```

Abrir `http://127.0.0.1:8050`. Para la API en otra terminal:

```bash
python -m uvicorn tarea_2.api:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000/docs` para probar `/predict` con el esquema mostrado.

## Docker y Kubernetes local

```bash
docker build -t heart-api:local -f tarea_2/docker/Dockerfile .
docker run --rm -p 8000:8000 heart-api:local
minikube start
minikube image load heart-api:local
kubectl apply -f tarea_2/k8s/deployment.yaml
kubectl apply -f tarea_2/k8s/service.yaml
kubectl rollout status deployment/heart-model
minikube service heart-service --url
```

Verificar `/health` y `/predict` en la URL obtenida. No se dieron por realizadas estas pruebas.

## Publicar Dash con Render

La referencia del docente usa Render. Desde Render, conectar este repositorio y crear el servicio
usando `render.yaml`, o configurar:

- Build: `pip install -r requirements_dashboard.txt`
- Start: `gunicorn entregable_2.app:server --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- Root Directory: raíz del repositorio.

Los resultados deben estar versionados antes del despliegue. Después verificar las tres pestañas,
filtros y control de umbral en la URL pública. No se ha creado un servicio Render ni una URL pública.
GitHub Pages puede alojar el JBook estático, pero no ejecutar el servidor Python de Dash.

## Estado comprobado

| Elemento | Estado |
|---|---|
| Entrenamiento de ambos productos | Ejecutado |
| Datos, hashes y predicciones | Guardados |
| Dashboard: endpoints y callbacks | Probados localmente |
| API: predicción y validación | Probadas localmente |
| Evidently: dos reportes | Generados |
| Docker / Minikube | Archivos preparados; ejecución pendiente |
| GitHub Actions | Workflow preparado; ejecución remota pendiente |
| Render y publicación JBook | Configuración preparada; publicación pendiente |
''')

notebook(ROOT/'entregable_2/entregable_2_caries.ipynb', '# Entregable 2: Caries dental', [
    ('markdown', 'El código completo vive en `entregable_2/train.py` y `ml_common.py`; este notebook lo ejecuta sin depender de variables ocultas. Datos, semilla y split se documentan en el JBook.'),
    ('code', 'from entregable_2.train import main\nmain()'),
    ('code', 'import json, pandas as pd\nfrom IPython.display import display\ns = json.loads(Path("entregable_2/results/summary.json").read_text())\ndisplay(pd.DataFrame([s["logistic"], s["dummy"]], index=["Logística", "Dummy"]))'),
    ('code', 'from entregable_2.app import update_eda\nfor fig in update_eda("age", "Todos")[:3]:\n    display(fig)'),
    ('markdown', 'Con umbral 0,5 la sensibilidad es casi nula: la exactitud no describe por sí sola la detección de positivos. El JBook interpreta las figuras, documenta duplicados y compara con el primer entregable.')])
notebook(ROOT/'tarea_2/notebooks/1_model_leakage_demo.ipynb', '# Tarea 2 — Demostración de fuga', [
    ('code', 'from tarea_2.train import load_split, leakage_demo\ndf, (X_train, X_test, y_train, y_test) = load_split()\nleakage_demo(X_train, X_test, y_train, y_test)'),
    ('markdown', 'La característica contaminada usa la respuesta y produce AUC artificialmente alta incluso dentro de un pipeline. El entrenamiento seguro elimina completamente esa característica.'),
    ('code', 'from tarea_2.train import train_pipeline\nfrom sklearn.linear_model import LogisticRegression\nfrom ml_common import evaluate\nsafe = train_pipeline(X_train, y_train, LogisticRegression(max_iter=2000), {"model__C": [0.1, 1]})\nevaluate(y_test, safe.predict_proba(X_test)[:, 1])')])
notebook(ROOT/'tarea_2/notebooks/2_model_pipeline_cv.ipynb', '# Tarea 2 — Modelado seguro', [
    ('code', 'from tarea_2.train import main\nmain()'),
    ('code', 'import pandas as pd\nfrom IPython.display import display\ndisplay(pd.read_csv("tarea_2/results/ranking.csv"))'),
    ('markdown', 'El ganador se selecciona con CV de entrenamiento. La tabla de prueba evalúa modelos ya fijados y no se usa para elegir nuevamente el ganador.'),
    ('code', 'import json, plotly.express as px, plotly.graph_objects as go\nfrom sklearn.metrics import confusion_matrix, roc_curve\ns = json.loads(Path("tarea_2/results/summary.json").read_text())\np = pd.read_csv("tarea_2/results/predictions.csv")\nname = s["winner_by_cv"]\ncm = confusion_matrix(p.actual, (p[name] >= 0.5).astype(int))\ndisplay(px.imshow(cm, text_auto=True, x=["Predicción 0", "Predicción 1"], y=["Real 0", "Real 1"], title=f"Matriz de confusión: {name}"))\nfpr, tpr, _ = roc_curve(p.actual, p[name])\nfig = go.Figure(go.Scatter(x=fpr, y=tpr, name=name))\nfig.add_scatter(x=[0,1], y=[0,1], name="Azar", line_dash="dash")\nfig.update_layout(title="ROC de prueba", xaxis_title="FPR", yaxis_title="Sensibilidad")\ndisplay(fig)'),
    ('code', 'from tarea_2.drift import main as drift\ndrift()')])
print('Informe, figuras y tres notebooks generados.')
