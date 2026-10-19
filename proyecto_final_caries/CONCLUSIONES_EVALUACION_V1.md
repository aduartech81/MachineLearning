# Conclusiones de la evaluación interna — versión 1

## Resultado principal
El modelo propio obtuvo un AUC medio externo de aproximadamente 0,6051.
No presentó mejora numérica frente a regresión logística o gradient boosting.
Los intervalos de las seis diferencias incluyen cero y los valores p
ajustados por Holm son 1,000. No se encontró evidencia de ventaja en el
análisis exploratorio realizado. Esto no demuestra equivalencia.

| Comparación | Diferencia de AUC | IC 95 % | p Holm exploratorio |
|---|---:|---|---:|
| Propio - Logistica | -0.000659 | [-0.003919, 0.002825] | 1.0000 |
| Propio - SVM_lineal | -0.000182 | [-0.003383, 0.003241] | 1.0000 |
| Propio - Boosting | -0.000656 | [-0.003888, 0.002420] | 1.0000 |
| Propio - Sin_interacciones | -0.000030 | [-0.001491, 0.001450] | 1.0000 |
| Propio - Sin_ranking | -0.000902 | [-0.002759, 0.000892] | 1.0000 |
| Propio - Aditivo | -0.001089 | [-0.003209, 0.001007] | 1.0000 |


## Aporte de los componentes
Las ablaciones no aportan evidencia de beneficio de las interacciones
seleccionadas ni de la pérdida de ranking. La variante aditiva tiene
el mayor AUC medio observado, pero esa posición descriptiva no demuestra
superioridad estadística ni justifica seleccionarla retrospectivamente
como ganadora confirmada.

Las ablaciones comparan procedimientos con ajuste interno de sus
hiperparámetros; no mantienen necesariamente idéntica regularización.

## Alcance de los intervalos y contrastes
Se remuestrearon perfiles completos dentro de cada fold y clase,
con los mismos pesos para todos los modelos. El estadístico es
la media no ponderada de los cinco AUC externos.

Los intervalos son percentiles individuales del 95 %, no simultáneos.
El contraste bootstrap centrado es aproximado y exploratorio.
Los modelos y sus predicciones permanecen fijos: el procedimiento
no incorpora variabilidad de reentrenamiento ni toda la dependencia
inducida por conjuntos de entrenamiento solapados.
La independencia entre perfiles no está verificada por paciente.
No se trata de una evaluación independiente confirmatoria.

## Utilidad y umbral
Con probabilidad 0,5 —o puntuación 0 para SVM— la sensibilidad es
prácticamente nula. La exactitud y especificidad elevadas no compensan
la falta de identificación de casos positivos.
Estos resultados no respaldan uso clínico de detección de caries.

Un umbral alternativo requiere un criterio operativo explícito y
selección exclusivamente dentro del desarrollo interno. No se elegirá
un corte examinando el mejor resultado de las predicciones externas.
Cambiar el umbral no mejora el AUC del mismo conjunto de puntuaciones.

## Originalidad
Se implementó un estimador con bases spline, selección exploratoria
de interacciones y pérdida combinada de clasificación y ranking.
Sus componentes tienen antecedentes. La originalidad científica de
la integración no está demostrada por haber escrito código propio.
No se afirma superioridad ni se garantiza una calificación académica.

## Alcance del problema
El objetivo es caries registrada, no aparición futura.
El test histórico ya fue consultado y no se presenta como nueva
validación independiente. Los resultados actuales proceden de
evaluación interna mediante validación cruzada anidada agrupada.

## Próximos productos
- Figuras ROC, precisión-recall y calibración con límites explícitos.
- Métricas por fold y matrices de confusión.
- Interpretación de funciones e interacciones sin atribución causal.
- Integración del proyecto final en Jupyter Book.
- Síntesis de antecedentes y respuestas para la defensa.
- Mantener pendientes documentales de procedencia y codificación visibles.
