# Tarea 2 — Revisión del 17 de octubre de 2026

Se clasifica HeartDisease registrada en Heart Failure Prediction, sin equiparar la etiqueta a
incidencia futura o insuficiencia cardíaca. Hay 918 filas, 734 train y 184 test.
La partición es estratificada, semilla 42. Los ceros de Cholesterol y RestingBP se convierten a
faltantes por una regla fija, antes de imputar dentro de Pipeline. La variable artificial de fuga
se usa sólo en la demostración y no forma parte del modelo desplegado.

## Selección e incertidumbre

| model              |   cv_auc |   cv_auc_sd |   roc_auc |   average_precision |   recall |   specificity |      f1 |
|:-------------------|---------:|------------:|----------:|--------------------:|---------:|--------------:|--------:|
| RandomForest       |  0.93072 |     0.02766 |   0.92994 |             0.93176 |  0.92157 |       0.81707 | 0.89100 |
| GradientBoosting   |  0.92368 |     0.03722 |   0.92581 |             0.93617 |  0.90196 |       0.89024 | 0.90640 |
| LogisticRegression |  0.92199 |     0.03372 |   0.93293 |             0.94401 |  0.91176 |       0.85366 | 0.89855 |
| SVC                |  0.91979 |     0.03445 |   0.93759 |             0.94508 |  0.91176 |       0.84146 | 0.89423 |
| KNN                |  0.91429 |     0.03059 |   0.93675 |             0.94077 |  0.93137 |       0.85366 | 0.90909 |
| GaussianNB         |  0.90869 |     0.04403 |   0.90686 |             0.89262 |  0.87255 |       0.86585 | 0.88119 |

`cv_auc_sd` procede de la misma fila ganadora de los archivos cv_*.csv guardados, sin inventar
ni recalcular entrenamientos. Es dispersión entre cinco folds, no intervalo de confianza.
El ganador **RandomForest** se seleccionó por CV exclusivamente. No se cambia por el ranking
de test. La CV después de búsqueda puede ser optimista y no es anidada. Las diferencias pequeñas
no establecen superioridad definitiva. Los datos no tienen validación externa.

## Errores a umbral 0,5

| model              |   tn |   fp |   fn |   tp |
|:-------------------|-----:|-----:|-----:|-----:|
| RandomForest       |   67 |   15 |    8 |   94 |
| GradientBoosting   |   73 |    9 |   10 |   92 |
| LogisticRegression |   70 |   12 |    9 |   93 |
| SVC                |   69 |   13 |    9 |   93 |
| KNN                |   70 |   12 |    7 |   95 |
| GaussianNB         |   71 |   11 |   13 |   89 |

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

drift_report.json: 0/11 variables con alerta (0.0%). La regla global requiere al menos 50%: no se activa en ninguno de los dos escenarios. Age sí activa una alerta individual en la simulación; los contrastes de monitoreo son exploratorios sin ajuste múltiple.

### drift_report.json

| metric                                                                    |   value |
|:--------------------------------------------------------------------------|--------:|
| ValueDrift(column=Age,method=K-S p_value,threshold=0.05)                  | 0.07214 |
| ValueDrift(column=RestingBP,method=K-S p_value,threshold=0.05)            | 0.57696 |
| ValueDrift(column=Cholesterol,method=K-S p_value,threshold=0.05)          | 0.17155 |
| ValueDrift(column=MaxHR,method=K-S p_value,threshold=0.05)                | 0.60824 |
| ValueDrift(column=Oldpeak,method=K-S p_value,threshold=0.05)              | 0.32288 |
| ValueDrift(column=Sex,method=Z-test p_value,threshold=0.05)               | 0.88991 |
| ValueDrift(column=ChestPainType,method=chi-square p_value,threshold=0.05) | 0.19888 |
| ValueDrift(column=FastingBS,method=Z-test p_value,threshold=0.05)         | 0.54466 |
| ValueDrift(column=RestingECG,method=chi-square p_value,threshold=0.05)    | 0.51726 |
| ValueDrift(column=ExerciseAngina,method=Z-test p_value,threshold=0.05)    | 0.57222 |
| ValueDrift(column=ST_Slope,method=chi-square p_value,threshold=0.05)      | 0.31492 |

drift_simulated.json: 1/11 variables con alerta (9.1%). La regla global requiere al menos 50%: no se activa en ninguno de los dos escenarios. Age sí activa una alerta individual en la simulación; los contrastes de monitoreo son exploratorios sin ajuste múltiple.

### drift_simulated.json

| metric                                                                    |   value |
|:--------------------------------------------------------------------------|--------:|
| ValueDrift(column=Age,method=K-S p_value,threshold=0.05)                  | 0.00000 |
| ValueDrift(column=RestingBP,method=K-S p_value,threshold=0.05)            | 0.57696 |
| ValueDrift(column=Cholesterol,method=K-S p_value,threshold=0.05)          | 0.17155 |
| ValueDrift(column=MaxHR,method=K-S p_value,threshold=0.05)                | 0.60824 |
| ValueDrift(column=Oldpeak,method=K-S p_value,threshold=0.05)              | 0.32288 |
| ValueDrift(column=Sex,method=Z-test p_value,threshold=0.05)               | 0.88991 |
| ValueDrift(column=ChestPainType,method=chi-square p_value,threshold=0.05) | 0.19888 |
| ValueDrift(column=FastingBS,method=Z-test p_value,threshold=0.05)         | 0.54466 |
| ValueDrift(column=RestingECG,method=chi-square p_value,threshold=0.05)    | 0.51726 |
| ValueDrift(column=ExerciseAngina,method=Z-test p_value,threshold=0.05)    | 0.57222 |
| ValueDrift(column=ST_Slope,method=chi-square p_value,threshold=0.05)      | 0.31492 |
