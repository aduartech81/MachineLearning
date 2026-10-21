# Revisión de cumplimiento del proyecto final

Fecha: 21 de octubre de 2026. Esta revisión diferencia lo implementado de lo pendiente. No constituye una certificación de cumplimiento total ni una estimación de calificación.

Fuente: guía compartida «Proyecto Final: Segundo Entregable — Modelos de Machine Learning para Clasificación y Regresión» y dos auditorías aportadas por Andrés. La aplicabilidad de adaptaciones debe contrastarse con las indicaciones específicas del docente.

| Requisito | Evidencia disponible | Estado y acción |
|---|---|---|
| EDA y diccionario | Explorador de 23 variables y evidencia agregada de desarrollo | Integrado en portal e informe: descriptivos por clase, figura, diccionario y comprobación de perfiles |
| Clasificación combinatoria | 108 configuraciones, 540 evaluaciones; KNN ponderado no aplicable | Explicar las cuatro excepciones frente a las 112 nominales |
| Regresión: 7 familias × 4 optimizadores | No hay rama de regresión | Pendiente; definir objetivo continuo y pregunta justificada antes de entrenar |
| Validación anidada | Cinco folds externos, tres internos, agrupación de perfiles | Implementada; conservar advertencia de que perfil no acredita identidad de paciente |
| Grid, Random, TPE y genética | Resultados y elecciones internas publicados | Publicados los 20 módulos originales, dominios, versiones, operadores y presupuesto efectivo: 21.417 ajustes internos registrados |
| Espacios y coincidencias entre optimizadores | Tabla de 108 configuraciones y elecciones por fold | Auditar espacios y configuraciones únicas; coincidencia de métricas no demuestra error ni selección en un borde demuestra óptimo fuera de rango |
| Calibración y umbral | Sigmoide principal; isotónica secundaria; Youden interno | Comparación original/sigmoide/isotónica y dummy de entrenamiento integrada sobre 35 predicciones externas; SVM original no aplica |
| Balanceo mixto | SMOTE/ADASYN auditados; SMOTENC en piloto | Documentar categorías fraccionarias y evaluar alternativa mixta si se amplía el protocolo; no sustituir silenciosamente resultados |
| Complejidad y recursos | Tiempos V1; registros de búsqueda V2 en el Mac | Recuperados tiempos de 540 búsquedas y 540 evaluaciones; siguen pendientes RAM medida y benchmark por defecto frente a optimizado |
| Friedman, Nemenyi y CD | Integrados: siete referencias, cinco folds; script, rangos, 21 pares, figura y método | Alcance exploratorio documentado; aproximación asintótica y sensibilidad por permutación no resuelven entrenamiento solapado |
| Comparaciones dirigidas y efectos | Bootstrap agrupado y Holm de diez contrastes | Añadidos DeLong/Holm/BH de sensibilidad y Cliff/biserial/dz descriptivos. Sus supuestos de independencia no se satisfacen; bootstrap agrupado sigue como análisis principal |
| SHAP y LIME | XGBoost: global y cuatro casos; tres semillas LIME | Implementados; son explicaciones de una referencia, no del modelo propio |
| Modelo propio y ablaciones | Código, gradiente comprobado, convergencia y resultados V1 | Implementado; no hay superioridad demostrada frente a las principales referencias |
| Reproducibilidad | Cuaderno, evidencias con hashes y código parcial publicado | Paquete recibido y 35.585 hashes verificados; publicados código, requisitos, historiales agregados y comandos de reproducción |
| Jupyter Book | Cuaderno V2 y construcción verificada | Libro independiente construido con --warningiserror y ejecución off; ruta /libro-final/ y ZIP descargable. No contiene guion. |
| Exposición | Guion de ocho minutos conservado en repositorio | Retirado del portal público; ajustar duración y comparación de optimizadores al tiempo asignado |

## Orden de cierre

1. Recuperar los ejecutores y registros V2 existentes; no repetir búsquedas para obtener información ya registrada.
2. Completar la comparación estadística de clasificación y la documentación computacional.
3. Definir y ejecutar la regresión requerida con una pregunta propia y sin reutilizar el objetivo binario como si fuese continuo.
4. Integrar EDA, métodos, resultados y limitaciones en un único informe autocontenido; conservar V1 como histórico claramente identificado.
5. Verificar rutas, navegación, respaldo estático y trazabilidad de los archivos publicados.

## Límites de las auditorías

Las estimaciones de nota no son decisiones del docente. Boosting V1 y XGBoost V2 son modelos diferentes. No corresponde afirmar que falta todo Jupyter Book: existe una construcción, aunque su acceso público debe mejorarse. Tampoco corresponde tratar las 540 evaluaciones como muestras independientes. El test histórico ya consultado no puede convertirse en una nueva validación independiente.
