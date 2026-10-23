# Medición nueva de recursos de clasificación

Este ejecutor compara las siete familias finalistas con parámetros de librería por defecto y con las selecciones internas V2 ya guardadas. Conserva preprocesamiento, balanceo y folds por pareja. No usa el test histórico. Tres repeticiones por variante en cinco folds producen 210 ajustes nuevos, sin repetir búsquedas.

Requiere Python 3.11 y `requirements_benchmark.txt`. Desde la raíz del repositorio, en un entorno independiente con esos requisitos:

```bash
python -m proyecto_final_caries.analisis.preparar_benchmark /ruta/al/evidencias_v2_para_revision_20261021_055250.zip
python -m proyecto_final_caries.analisis.benchmark_recursos --salida /ruta/nueva/benchmark --repeticiones 3
```

El primer comando usa el ZIP existente, verifica las 35 huellas de selección y no repite la recopilación. El segundo entrena y requiere una carpeta de salida nueva. Cada ajuste se ejecuta en un proceso independiente con un hilo, semilla 42 donde el estimador la admite y orden alternado entre variantes. Los parámetros efectivos completos quedan en cada JSON. Los parámetros operativos de semilla y CPU son excepciones explícitas a los valores de librería por defecto.

La comparación aplica el mismo balanceo seleccionado internamente a ambas variantes. No debe describirse como comparación de un procedimiento completamente sin balanceo frente al finalista. Los parámetros optimizados incluyen los parámetros fijos del protocolo V2.

Entrenamiento total incluye transformaciones y balanceo. Inferencia incluye transformación y puntuación de todo el fold, sin calibración ni elección de Youden. AUC se calcula con puntuaciones originales. Accuracy y F1 usan corte fijo (0 para margen, 0.5 para probabilidad). Las advertencias de convergencia se registran y no se ocultan.

RSS máximo se obtiene del máximo histórico del proceso independiente (`getrusage`), convertido de bytes en macOS y KiB en Linux. Incluye importaciones, datos y ejecución; no equivale a memoria exclusiva del modelo ni reconstruye la RAM de las búsquedas originales. El RSS de base también queda registrado, pero no se resta como si el máximo hubiese ocurrido necesariamente después de tomar esa base.

El archivo `COMPLETADO.json` solo se escribe después de terminar todos los ajustes solicitados. Un piloto con menos familias/folds/repeticiones no acredita el benchmark completo. No se deben tratar las repeticiones como datasets independientes para contrastes de rendimiento.

Esta comparación de hiperparámetros no cierra por sí sola la sección 4 de la guía: la comparación computacional estándar/optimizado debe justificar cambios algorítmicos o de implementación, incluir complejidad e interpretar tiempo, memoria y rendimiento. Tampoco acredita reproducción completa de las 540 búsquedas V2.
