# Modelo propio de caries: especificación inicial

## Estado
Contribución candidata de integración e implementación.
No se afirma novedad científica, superioridad ni utilidad clínica validada.

## Problema
Clasificar presencia de caries registrada mediante los 23 predictores
conservados. No predecir incidencia futura.
Se mantienen excluidos ID, oral y tartar.

## Modelo
Puntuación:
s(x) = intercepto + suma de funciones univariadas
       + suma de un número limitado de interacciones.

Las funciones numéricas se representan mediante bases spline.
Las categorías usan one-hot; no se presuponen distancias entre códigos.
Imputación, escalado, bases y selección se aprenden dentro de cada ajuste.

## Objetivo de entrenamiento
L = (1-alpha) * BCE
    + alpha * media[log(1 + exp(-(s_positivo - s_negativo)))]
    + lambda * penalización L2.

BCE es la pérdida logística de clasificación.
La segunda pérdida favorece una puntuación mayor para casos positivos.
Las dos pérdidas se promedian por separado para controlar su escala.
Se muestrean pares reproducibles; no se enumeran todos los pares.
El intercepto no recibe penalización L2.

La combinación de pérdidas no garantiza probabilidades calibradas.
La calibración se evalúa y, si se aplica un calibrador, se aprende
exclusivamente con predicciones internas fuera de muestra.

## Interacciones
Seleccionar dentro de fit mediante remuestreo de perfiles completos.
Limitar el número de interacciones y registrar su frecuencia de selección.
El algoritmo exacto, umbrales y límites se fijarán antes de comparar AUC.
No utilizar folds externos ni test histórico para seleccionar interacciones.

## Comparadores y ablaciones
- Dummy.
- Regresión logística con búsqueda interna.
- SVM con búsqueda interna.
- Gradient boosting con búsqueda interna.
- Modelo aditivo con las mismas bases, sin interacciones ni ranking.
- Modelo propio sin interacciones.
- Modelo propio sin pérdida de ranking.
- Modelo propio completo.

La comparación con la misma representación y pérdida BCE permite
distinguir el aporte de la función de entrenamiento del aporte de las bases.

## Evaluación
Usar particiones externas e internas ya guardadas.
Seleccionar hiperparámetros sólo con folds internos.
Mismos predictores, observaciones y folds para todos los modelos.
Registrar presupuesto de búsqueda, convergencia, tiempos y memoria.
Ejecutar secuencialmente en el Mac; guardar resultados por ajuste.

## Inferencia
Reportar AUC, diferencia pareada, intervalos y comparaciones múltiples.
El procedimiento debe respetar perfiles repetidos y dependencia de CV.
Un bootstrap de predicciones OOF fijas no incorpora incertidumbre de
reentrenamiento: no se presentará como prueba confirmatoria completa.
No tratar los cinco folds como cinco experimentos independientes.

## Umbral y utilidad
Fijar el criterio operativo antes de examinar resultados externos.
Reportar sensibilidad, especificidad, precisión, F1, matriz, AP y Brier.
No elegir el umbral por rendimiento sobre el test histórico.
Uso potencial: apoyar priorización de evaluación odontológica cuando
los predictores estén disponibles; requiere validación adicional.

## Límites
El test histórico ya fue consultado.
La evaluación anidada sigue siendo interna y no elimina decisiones previas.
No hay identidad longitudinal verificada por paciente.
No se trasladan AUC publicados en otras poblaciones a este experimento.

## Antes del primer entrenamiento
Fijar algoritmo de selección de interacciones, espacio de búsqueda,
optimizador y criterio de convergencia; verificar gradientes y aislamiento
de datos con pruebas pequeñas. No ajustar estos detalles contra el test.

## Criterio de aporte
La implementación propia debe acompañarse de justificación y ablaciones.
Si no mejora significativamente, se informa sin buscar significancia
mediante consultas repetidas al conjunto de evaluación.
