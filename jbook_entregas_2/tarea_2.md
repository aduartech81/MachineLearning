# Tarea 2 — Proyecto integrador con pipelines

## 1. Lineamientos y objetivo

El enlace docente corresponde actualmente a **10.20 Proyecto Integrador**, aunque la consigna
lo llama 10.10. Se implementan estructura modular, demostración de fuga, pipelines con búsqueda,
API FastAPI, Docker, manifiestos Kubernetes, GitHub Actions y reportes Evidently.

Se utiliza Heart Failure Prediction (918 registros, 11 predictores y `HeartDisease` binaria).
El nombre comercial del dataset **no permite equiparar la etiqueta HeartDisease con insuficiencia
cardíaca ni con incidencia futura**. Se clasifica el estado registrado de enfermedad cardíaca.
Fuente original: [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
Archivo obtenido de [copia pública en GitHub](https://github.com/amankataria100/Data-Visualization-on-Heart-Disease-Dataset/blob/main/heart.csv).
SHA-256: `948420b084d8a3a0ca42b8419fce9aee175879e43f8aedf712377899a67aa49b`. La copia tiene 918 filas y 508 positivos. No se sustituye silenciosamente
por la base UCI de diferente esquema ni se cambia el target a `target`.

## 2. Preprocesamiento y fuga

Se reservan 184 filas para prueba mediante partición estratificada, semilla 42;
las 734 restantes se usan para entrenamiento y validación cruzada.
Se interpretan los ceros de colesterol (172) y presión en reposo
(1) como valores no plausibles para estas mediciones, se reemplazan por NaN
y se imputan dentro del pipeline. La regla es fija y se documenta; no se aprende del test.

El ejemplo docente añade `leaky_feature=y+ruido` y no la elimina en el supuesto flujo correcto.
Se corrige: la variable sólo existe en el experimento contaminado y **nunca se incorpora al modelo
válido**. La demostración contaminada obtiene AUC 1.0000; un rendimiento
perfecto es aquí producto de acceso artificial a la respuesta, no de descubrimiento clínico.
Un Pipeline evita fuga por ajuste de transformaciones, pero no elimina automáticamente proxies del target.

## 3. Entrenamiento y comparación

Se comparan seis clasificadores: regresión logística, SVC, Random Forest, KNN, Gradient Boosting
y Gaussian Naive Bayes. Cada uno se integra con el mismo preprocesamiento dentro de `Pipeline`
y se optimiza con `GridSearchCV`, cinco pliegues estratificados y ROC AUC.
El ranking se ordena por AUC de validación cruzada, no por AUC de prueba. La selección de
hiperparámetros y modelo queda dentro de entrenamiento. El ranking de CV tras selección puede
ser optimista; no se presenta como una estimación de validación cruzada anidada.

| model              |   cv_auc |   roc_auc |   accuracy |   recall |     f1 |   seconds |
|:-------------------|---------:|----------:|-----------:|---------:|-------:|----------:|
| RandomForest       |   0.9307 |    0.9299 |     0.8750 |   0.9216 | 0.8910 |    6.5991 |
| GradientBoosting   |   0.9237 |    0.9258 |     0.8967 |   0.9020 | 0.9064 |    2.2238 |
| LogisticRegression |   0.9220 |    0.9329 |     0.8859 |   0.9118 | 0.8986 |    0.2278 |
| SVC                |   0.9198 |    0.9376 |     0.8804 |   0.9118 | 0.8942 |    1.6070 |
| KNN                |   0.9143 |    0.9368 |     0.8967 |   0.9314 | 0.9091 |    0.5267 |
| GaussianNB         |   0.9087 |    0.9069 |     0.8696 |   0.8725 | 0.8812 |    0.1475 |

El modelo seleccionado por CV es **RandomForest**. Otro modelo puede alcanzar mayor AUC
en test sin que eso autorice a cambiar el ganador usando ese mismo test. Las diferencias pequenas
no permiten afirmar superioridad definitiva sin análisis de incertidumbre o validación externa.
Los tiempos son mediciones de esta ejecución local y no una comparación entre plataformas.

```{raw} html
<iframe src="_static/heart_roc.html" width="100%" height="540" style="border:0" title="heart_roc"></iframe>
```


## 4. API, contenedor y Kubernetes

Se guarda el pipeline completo como `model.joblib`. La API acepta campos nombrados, limita categorías,
rechaza campos extra y comprueba rangos básicos. Los campos numéricos nulos se imputan con parámetros
de entrenamiento. El endpoint `/health` comprueba la carga del modelo. `/predict` devuelve probabilidad,
clase a umbral 0,5 y alcance académico. La probabilidad no está calibrada para atención clínica.

Docker incluye las dependencias y el artefacto. Kubernetes configura Deployment, Service NodePort,
readiness probe y límites de recursos. Docker y Kubernetes se ejecutaron en el Mac del autor. Minikube completó el rollout y el pod quedó Running, READY 1/1. Las pruebas mediante port-forward del servicio devolvieron el modelo cargado y la misma predicción que Docker. Evidencia: `tarea_2/results/verificacion_despliegue/respuestas.json`.

## 5. Integración continua

El workflow de GitHub Actions instala dependencias, verifica errores esenciales de sintaxis/nombres
y ejecuta pruebas de separación de perfiles, equivalencia entre API y pipeline, rechazo de entradas
inválidas y callbacks del dashboard. Las pruebas se ejecutaron localmente. La validación remota de GitHub Actions se completó con éxito.

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
