# Correcciones del 17 de octubre de 2026

Se preserva la entrega previa en el commit cb97ee4a99f1b0143aa7bea4cbb683962b1d9df8.
Esta actualización es posterior a la entrega: no modifica archivos adjuntos del aula ni presupone
que el profesor acepte una recalificación.

- **Entregable 1:** notebook autocontenido revisado y REVISION_ENTREGABLE_1.md responden a las
  observaciones sobre categóricas, multiplicidad, atípicos, tártaro, efectos, escalado, coeficientes y CV.
- **Entregable 2:** ajuste realmente ejecutado, codificación semántica, CV agrupada con ponderación
  consistente, perfiles definidos por predictores finales, EDA y dashboard de tres pestañas.
- **Tarea 2:** desviación de CV recuperada de los ajustes guardados, matrices de error por modelo,
  interpretación de alertas de deriva y evidencia original de Docker/Kubernetes conservada.
- **Tarea 1:** once modelos ejecutados sobre 1.345.310 préstamos del CSV real: seis familias
  en scikit-learn y cinco en Spark, con IDs y folds compartidos, EDA y resultados de CV guardados.
  El informe distingue resultados calculados y pendientes. Random Forest de Spark se interrumpió
  antes de guardar resultados; faltan ese ajuste, los contrastes completos y LIME del ganador Spark.
  No se presenta esta tarea como completa. Se preservan código, métricas y predicciones verificadas.
  [Informe de estado](https://aduartech81.github.io/MachineLearning/09_evaluacion_modelos.html).


Los resultados de caries son internos y exploratorios. No se optimizó un umbral en test.
La corrección metodológica no implica mejores métricas ni garantiza aumento de nota.

Reproducir con `python -m entregable_2.train`, `python -m entregable_2.audit`,
`python build_new_book.py`, `python -m pytest -q` y Jupyter Book.
