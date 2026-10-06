# Entregable 2 — Dashboard de caries dental

## 1. Contexto del problema

Se clasifica la presencia de caries **registrada en un chequeo**, no el riesgo de desarrollarla en el
futuro. La tarea es binaria: `dental caries=1` frente a `0`. Una etiqueta negativa no significa que
el individuo esté sano respecto de otras enfermedades. La motivación es explorar cuánto discrimina
un modelo lineal a partir de variables generales de chequeo. No se propone un dispositivo diagnóstico.

La copia utilizada contiene **55,692 filas**, **11,881 positivas** y una proporción
positiva de **21.33%**. Estos conteos coinciden con el primer entregable. La fuente inmediata
es [la copia pública en GitHub](https://github.com/aliabdallah7/smoking-detection/blob/main/smoking.csv),
asociada al dataset [Body Signal of Smoking](https://www.kaggle.com/datasets/kucuro3/body-signal-of-smoking).
No se ha comprobado su identidad byte a byte con el archivo original del estudiante; tampoco se
verificaron de forma independiente los supuestos de edad, independencia por persona o licencia del origen.
SHA-256 del archivo utilizado: `da697e1d194e6eeedbc643057a4e84f058496db61d0ee5574e212ce4aa102637`.

### Interpretación del gráfico de clases

La clase negativa domina la base. Un clasificador mayoritario alcanza una exactitud elevada sin
detectar positivos. Por ello se presentan discriminación, precision–recall y sensibilidad además de exactitud.

```{raw} html
<iframe src="_static/caries_clases.html" width="100%" height="540" style="border:0" title="caries_clases"></iframe>
```


## 2. Auditoría y partición

El archivo tiene **0 faltantes** y **11,140 filas duplicadas**
cuando se elimina `ID`. No se presupone MAR ni se confirma independencia individual a partir de un ID
de fila. Se construyen grupos mediante hash de los predictores originales; los perfiles únicos se
dividen de forma estratificada, con semilla 42, y cada copia se conserva en el grupo de su perfil.
Quedan **44,582 filas de entrenamiento** y **11,110 de prueba**. Las proporciones
de filas no son exactamente 80/20 porque la unidad de asignación es el perfil, no cada copia.

El modelo utiliza **23 predictores**. Se excluyen `ID`, `oral` (constante) y `tartar`.
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

```{raw} html
<iframe src="_static/caries_histograma.html" width="100%" height="540" style="border:0" title="caries_histograma"></iframe>
```

```{raw} html
<iframe src="_static/caries_boxplot.html" width="100%" height="540" style="border:0" title="caries_boxplot"></iframe>
```


### Proporción de caries por edad

Las barras presentan positivos sobre el total del grupo de edad. El tamaño de cada grupo aparece
en el cursor del gráfico. Un grupo pequeño ofrece una estimación menos estable. Diferencias entre
edades pueden involucrar factores de selección y confusión; no representan una trayectoria individual.

```{raw} html
<iframe src="_static/caries_edad.html" width="100%" height="540" style="border:0" title="caries_edad"></iframe>
```


### Correlaciones

La matriz corresponde al entrenamiento completo y no cambia con los filtros. Las correlaciones
entre variables de tamaño corporal y laboratorio señalan redundancia potencial. El escalado no
elimina colinealidad; la regresión logística emplea regularización L2 por defecto. Valores extremos
se conservan y no se clasifican automáticamente como errores o enfermedad específica.

```{raw} html
<iframe src="_static/caries_correlacion.html" width="100%" height="540" style="border:0" title="caries_correlacion"></iframe>
```


## 4. Modelo base y resultados

El Pipeline contiene `ColumnTransformer`, imputación por mediana/moda, `StandardScaler`,
`OneHotEncoder(handle_unknown='ignore')` y `LogisticRegression(max_iter=2000)`.
Las transformaciones se aprenden exclusivamente en entrenamiento. El imputador permite manejar
faltantes futuros aunque este archivo no tenga celdas vacías.

La validación cruzada usa cinco pliegues estratificados sobre perfiles únicos de los predictores
finales de entrenamiento: AUC **0.6058 ± 0.0062** (media ± desviación estándar,
no intervalo de confianza). El modelo final usa todas las filas de entrenamiento; por tanto las copias
ponderan algunos perfiles más que otros. Esa diferencia de ponderación limita la comparación directa
entre el estimador de CV y el de prueba. El modelo base no incluye búsqueda de hiperparámetros.

### Tabla comparativa, umbral 0,5

| Modelo    |   roc_auc |   average_precision |   accuracy |   recall |   specificity |     f1 |   brier |
|:----------|----------:|--------------------:|-----------:|---------:|--------------:|-------:|--------:|
| Logística |    0.6018 |              0.2844 |     0.7837 |   0.0004 |        0.9999 | 0.0008 |  0.1660 |
| Dummy     |    0.5000 |              0.2163 |     0.7837 |   0.0000 |        1.0000 | 0.0000 |  0.1695 |

La AUC de prueba de **0.6018** indica discriminación modesta. AP
**0.2844** supera la referencia de prevalencia de prueba
**0.2163**. AP es un resumen ponderado de precision–recall y no debe confundirse con
integración trapezoidal. Brier menor sugiere menor error cuadrático probabilístico frente a Dummy.
No se concluye utilidad clínica a partir de una mejora estadística aislada.

```{raw} html
<iframe src="_static/caries_roc.html" width="100%" height="540" style="border:0" title="caries_roc"></iframe>
```

```{raw} html
<iframe src="_static/caries_pr.html" width="100%" height="540" style="border:0" title="caries_pr"></iframe>
```


### Matriz de confusión y control de umbral

Con corte 0,5 se observan TN=8706, FP=1,
FN=2402 y TP=1. La sensibilidad resulta casi nula, aunque la
exactitud se aproxime a la del clasificador mayoritario. El control de umbral del dashboard muestra
el intercambio entre sensibilidad y especificidad sobre las mismas predicciones. Es una herramienta
descriptiva, **no un procedimiento para elegir el mejor corte en test**. Un corte operativo debe fijarse
en validación de entrenamiento con un criterio de costos explícito y después evaluarse en prueba.

### Calibración

Cada punto compara probabilidad media y frecuencia positiva en un decil. La cercanía a la diagonal
es evidencia descriptiva de calibración en esta muestra, sin demostrar equivalencia al riesgo clínico
ni transportabilidad. No se ha realizado calibración adicional o validación externa.

```{raw} html
<iframe src="_static/caries_calibracion.html" width="100%" height="540" style="border:0" title="caries_calibracion"></iframe>
```


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
