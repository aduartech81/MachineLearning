# Guion de exposición · ocho minutos

Andrés Duarte Chaves

Este recorrido resume el proyecto. Las tablas completas y las evidencias quedan disponibles en el informe y en los selectores del sitio.

## 0:00–0:45 · La pregunta

¿Hasta dónde puedo clasificar la caries registrada con variables de un chequeo general? Evalué esa pregunta y desarrollé una propuesta propia. Mi objetivo fue medir su aporte frente a modelos de referencia, no asumir que una mayor complejidad garantizaría mejores resultados.

**Mostrar:** portada y pregunta de investigación.

## 0:45–1:45 · Datos y prevención de fugas

Trabajé con 23 predictores y 44.582 filas de desarrollo. Excluí el identificador, una variable constante y el indicador de sarro por su relación con el examen odontológico. Conservé juntos los perfiles idénticos en las particiones. Esto evita que esos duplicados crucen entre ajuste y validación, pero no demuestra independencia por paciente.

**Mostrar:** explorador de datos, una sola variable. El objetivo es caries registrada, no aparición futura.

## 1:45–3:00 · Mi propuesta

Implementé un modelo con funciones spline para relaciones no lineales, selección limitada de interacciones y una pérdida que combina clasificación y ordenación. Comprobé el gradiente numéricamente y la convergencia de los cinco ajustes. También evalué versiones sin interacciones, sin ranking y una variante aditiva para examinar el aporte de sus componentes.

**Mostrar:** esquema del modelo y las ablaciones. La implementación es propia; no afirmo que sus componentes ni su combinación sean una novedad científica establecida.

## 3:00–4:00 · Comparación experimental

Usé cinco folds externos y tres internos, conservando los grupos. La ampliación incluyó siete familias, cuatro optimizadores y alternativas de balanceo: 108 configuraciones aplicables y 540 evaluaciones de folds. Los hiperparámetros, la calibración y el umbral se eligieron dentro del entrenamiento externo. Como esta ampliación se planteó después de conocer resultados previos, la presento como exploratoria.

**Mostrar:** diseño anidado. Las 540 evaluaciones no son 540 muestras independientes.

## 4:00–5:30 · Qué encontré

Mi modelo obtuvo un AUC medio de 0,6051, frente a 0,6059 de logística y 0,6055 de XGBoost. No encontré evidencia de mejora frente a las principales referencias. Los contrastes exploratorios ajustados sí favorecieron mi modelo frente a KNN y GaussianNB. No confundí ausencia de significación con equivalencia.

El umbral interno permitió a las referencias detectar aproximadamente la mitad de los positivos, con especificidad moderada. Cambiar el umbral modifica ese intercambio, pero no aumenta por sí mismo el AUC.

**Mostrar:** tabla principal y una curva ROC. Distinguir AUC medio entre folds del AUC calculado con todas las predicciones fuera de muestra reunidas.

## 5:30–6:30 · Interpretación

En mi modelo, edad–ALT y edad–HDL se seleccionaron en cuatro de cinco folds. Esa repetición no prueba causalidad. Para XGBoost, usé SHAP y LIME como análisis complementarios: SHAP explica el margen original y LIME aproxima la probabilidad calibrada. En algunos casos la fidelidad local fue limitada, especialmente en el verdadero negativo; por eso no presento todas las explicaciones como igualmente fiables.

**Mostrar:** un falso positivo y su explicación. XGBoost es una referencia, no mi modelo propio.

## 6:30–8:00 · Conclusión y límites

Desarrollé y evalué una propuesta reproducible, pero su mayor complejidad no produjo una ventaja demostrada sobre las referencias principales. La discriminación cercana a 0,61 es limitada y no respalda usar este sistema para tomar decisiones clínicas.

El bootstrap cuantifica incertidumbre condicionada a las predicciones guardadas; no incorpora toda la variabilidad del entrenamiento. Además, los perfiles no equivalen necesariamente a pacientes y falta verificar metadatos clínicos. El siguiente paso sería obtener datos independientes y confirmar la disponibilidad de las variables en el momento de uso.

## Preguntas que debo poder responder

- **¿Por qué no ganó el modelo propio?** La evaluación no mostró una mejora; añadir componentes puede incorporar complejidad sin aportar información útil en estos datos.
- **¿El balanceo solucionó el problema?** Fue una alternativa evaluada, no una garantía. SMOTE y ADASYN generaron categorías fraccionarias en el espacio codificado; documenté esa limitación.
- **¿Por qué no reutilizar el test histórico como prueba nueva?** Ya fue consultado y no constituye una nueva evaluación independiente.
- **¿Por qué los intervalos son condicionales?** Remuestreé perfiles manteniendo fijas las predicciones; no repetí todo el ajuste en cada réplica.
- **¿Cuál es el aporte?** La implementación y evaluación de una combinación explícita, sus ablaciones y una comparación trazable. No afirmo utilidad clínica ni originalidad científica no demostradas.
