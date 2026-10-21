# Reproducción de la clasificación V2

Los 20 ejecutores recuperados conservan los bytes de la ejecución original. No los modifiqué para eliminar advertencias de obsolescencia: cambiar el estimador exigiría otra versión experimental. Python observado al recopilar: 3.11.16; el archivo de requisitos fija las versiones guardadas en los manifiestos de búsqueda. El sistema observado fue macOS arm64; no se guardó el modelo exacto de CPU ni una medición histórica de RAM.

Desde la raíz de una copia del repositorio, instalar los requisitos en un entorno independiente. Se necesitan `entregable_2/results/train.csv`, `summary.json`, las dos particiones de `proyecto_final_caries/evaluacion/` y los espacios publicados. No sustituirlos ni reconstruir folds al azar.

Ejemplo de una búsqueda y su evaluación (estos comandos SÍ entrenan):

```bash
python -m proyecto_final_caries.experimentos_v2.ejecutar_grid Logistica_L1_L2 --fold 0
python -m proyecto_final_caries.experimentos_v2.evaluar_seleccion Logistica_L1_L2 --optimizador grid --balanceo sin_balanceo --fold 0
```

Para los otros bloques, sustituir `grid` por `random`, `bayes` o `genetica`, y añadir al ejecutor el sufijo `_smote`, `_adasyn` o `_pesos`. Repetir folds 0–4. KNN con pesos está excluido. Los 16 scripts de búsqueda suman cuatro optimizadores por cuatro balanceos; cuatro módulos auxiliares completan el código.

Los resultados se guardan en `experimentos_v2/resultados_*` y `evaluacion_v2`. Los scripts son reanudables y comprueban huellas; una configuración diferente exige otra carpeta/copia del proyecto. El piloto original de logística Grid sin balanceo, fold 0, quedó en `evaluacion_externa_piloto`; sus tiempos tienen un esquema diferente y no se inventó un total de evaluación.

Para reconstruir sólo el resumen desde el ZIP extraído, sin entrenar:

```bash
python -m proyecto_final_caries.analisis.recuperar_evidencias_v2 --fuente /ruta/zip_extraido --repo . --salida /ruta/resumen_nuevo
python -m proyecto_final_caries.analisis.contrastes_sensibilidad --fuente /ruta/zip_extraido --repo . --salida /ruta/resumen_nuevo
```

El ZIP fuente contiene registros y predicciones; no se confunde con el ZIP descargable de código. No contiene los modelos binarios ni los datos crudos. La aplicación Render sirve evidencias estáticas y no ejecuta búsquedas al iniciarse.
