# Reproducción de Tarea 1 — ejecución parcial

Desde la raíz, instalar `pip install -r tarea_1/requirements.txt` y Java 17.
El CSV real `accepted_2007_to_2018Q4.csv` (1.675.133.810 bytes) debe estar en la raíz.
El protocolo fija 1.345.310 préstamos terminales, 80/20 estratificado y tres folds compartidos.
No se usan datos sintéticos ni submuestras para el entrenamiento.

Ejecutar en orden:

```bash
python -m tarea_1.prepare
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m tarea_1.run_sklearn
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m tarea_1.run_spark
python -m tarea_1.eda
python -m tarea_1.statistics
python -m tarea_1.explain
python -m tarea_1.report
```

Los dos entrenadores aceptan nombres de familias como argumentos para reanudar una fase.
Un JSON de resultados se guarda sólo después de obtener métricas y guardar predicciones/modelo.
Una familia que ya tenga su JSON no se recalcula. Si se cambia el protocolo, crear otra carpeta
para no mezclar datos ni resultados. Los modelos binarios grandes y el dataset no se versionan.
Las predicciones pueden publicarse en CSV gzip sin pérdida; los lectores aceptan ambas variantes.

El script `statistics.py` verifica la correspondencia de IDs y etiquetas antes de comparar.
Las pruebas de `tests/test_tarea1.py` verifican DeLong con empates frente al cálculo directo
así como el ajuste de transformaciones limitado a los datos recibidos por fit.

La guía actual se consulta en https://lihkir.github.io/MachineLearning/model_evaluation.html.
El historial de Git conserva las versiones de la entrega previa.

Estado guardado: once modelos completados. Spark RandomForest y los contrastes completos siguen pendientes.
Si se pierden los binarios, `python -m tarea_1.restore_sklearn` y `python -m tarea_1.run_spark --restore` reconstruyen los modelos completados usando los parámetros de CV guardados y verifican sus métricas. No sustituyen la CV original.
