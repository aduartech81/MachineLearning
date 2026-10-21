## Datos, exploración y trazabilidad del desarrollo

Describí las 44,582 filas de desarrollo y sus 35,641 perfiles de predictores. Hay 8,941 filas adicionales que repiten un perfil; este conteo no equivale a personas repetidas. Los perfiles se construyen con los 23 predictores, sin ID ni objetivo. Comprobé que sus hashes y etiquetas coinciden con las particiones conservadas y que ningún perfil cruza folds externos. Encontré **0 perfiles con etiquetas discordantes** en desarrollo. No se acredita identidad ni independencia por paciente.

Excluí **ID** por ser identificador, **oral** por ser constante y **tartar** por su disponibilidad ligada al examen odontológico. Estas son razones diferentes. No interpreto los extremos ni los códigos como errores clínicos sin un diccionario de origen que lo respalde. Los predictores de desarrollo presentan 0 valores faltantes en esta copia.

La exploración por filas y las tablas históricas por perfiles únicos tienen denominadores distintos. Las figuras siguientes usan todas las filas y densidad por clase para evitar que el tamaño de una clase domine la comparación.

![Distribuciones por clase en desarrollo.](/cierre-auditoria/eda_desarrollo.png)

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>fold_externo</th>
      <th>filas</th>
      <th>perfiles</th>
      <th>positivos</th>
      <th>prevalencia</th>
      <th>filas_repetidas_adicionales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>8917</td>
      <td>7128</td>
      <td>1896</td>
      <td>0.2126</td>
      <td>1789</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8916</td>
      <td>7128</td>
      <td>1895</td>
      <td>0.2125</td>
      <td>1788</td>
    </tr>
    <tr>
      <td>2</td>
      <td>8917</td>
      <td>7129</td>
      <td>1896</td>
      <td>0.2126</td>
      <td>1788</td>
    </tr>
    <tr>
      <td>3</td>
      <td>8916</td>
      <td>7128</td>
      <td>1896</td>
      <td>0.2127</td>
      <td>1788</td>
    </tr>
    <tr>
      <td>4</td>
      <td>8916</td>
      <td>7128</td>
      <td>1895</td>
      <td>0.2125</td>
      <td>1788</td>
    </tr>
  </tbody>
</table>

[Descriptivos completos por clase](/cierre-auditoria/descriptivos_por_clase.csv) · [Auditoría de agrupación](/cierre-auditoria/auditoria_datos.json) · [Diccionario de variables](/evidencias/diccionario.csv)

### Procedimiento de preparación

Importación del archivo → exclusiones justificadas → construcción de perfiles → particiones agrupadas → ajuste de imputación, escalado y codificación dentro del entrenamiento → búsqueda interna → evaluación del fold externo. El balanceo, cuando se aplica, afecta únicamente al entrenamiento interno o al reajuste, nunca al fold de evaluación.

### Referencia trivial para interpretar las métricas

En desarrollo hay 9.478 positivos y 35.104 negativos: predecir siempre ausencia alcanza 78,74 % de exactitud, pero sensibilidad y F1 iguales a cero. Por eso no seleccioné modelos solo por exactitud. Esta referencia describe las etiquetas de desarrollo; no sustituye al Dummy ajustado con prevalencia de entrenamiento en cada fold.

### Qué conservé y qué amplié

Conservé el modelo logístico inicial y su test histórico como antecedentes. La ampliación incorporó el diccionario de 23 predictores, curvas de aprendizaje agrupadas, evaluación anidada, búsqueda de siete referencias, calibración y umbral internos, contrastes y explicaciones locales. Esta lista documenta cambios; no certifica que todas las observaciones originales del docente estén resueltas sin disponer de su lista completa.

### Cómo leer V1 y V2

V1 es un antecedente experimental conservado. Su «Boosting» corresponde a HistGradientBoostingClassifier; XGBoost es una referencia distinta de V2. También difieren los espacios de búsqueda y las familias comparadas. Por ello cambian AUC, desviaciones y número de contrastes. Las tablas V1 usan umbral 0,5 o puntuación cero en SVM; un Brier vacío en SVM significa que no se proporcionaron probabilidades, no un valor cero. Una precisión alta basada en muy pocos positivos predichos no indica alta sensibilidad.

### Excepción de KNN y presupuesto

Las 112 combinaciones nominales de clasificación se reducen a 108 aplicables porque KNN no implementa el mismo mecanismo de ponderación de clases utilizado en las otras familias. `weights="distance"` pondera vecinos por distancia y no equivale a `class_weight`. Las 540 evaluaciones externas son 108 combinaciones por cinco folds; no son 540 entrenamientos totales ni observaciones independientes. El número efectivo de candidatos, reajustes, fallos y reutilizaciones debe recuperarse de los historiales V2.
