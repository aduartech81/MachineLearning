# Evidencias V2 del proyecto final de caries

## Alcance
540 evaluaciones de folds auditadas; 108 configuraciones aplicables.
Se conservaron siete referencias seleccionadas por validación interna
en cada fold y el modelo propio V1 con sus tres ablaciones.
V2 es exploratoria: se consultaron resultados previos del desarrollo.

## Resultado principal
El modelo propio obtuvo AUC medio por fold de 0.605062.
No se demostró una mejora frente a logística, SVM, XGBoost ni Random Forest.
En el contraste condicional exploratorio, las diferencias frente a KNN
y GaussianNB superaron el ajuste Holm de diez comparaciones.
Esto no acredita superioridad frente a las mejores referencias ni
establece novedad científica.

## Cómo leer las métricas
La media de AUC por fold y el AUC de predicciones OOF concatenadas son
estimaciones distintas y se presentan con sus nombres correspondientes.
Los intervalos bootstrap son condicionales a modelos y predicciones fijos.
No capturan toda la incertidumbre de reentrenamiento ni la dependencia
de los entrenamientos solapados.
Los intervalos individuales no son simultáneos.
Un ECE pequeño no demuestra buena discriminación ni utilidad clínica.

## Explicaciones
SHAP describe el margen original de un XGBoost del fold externo 0.
Su resumen global usa una muestra reproducible de 500 filas.
Las contribuciones corresponden a columnas transformadas.
LIME explica la probabilidad calibrada de cuatro casos del mismo modelo.
Se analizaron tres semillas y se conservó la semilla 42 para las figuras.
La fidelidad local es limitada en los casos negativos, especialmente
en el verdadero negativo ID 2.
Las explicaciones no son efectos causales.

## Estado
Paquete preparado para integración.
No modifica modelos, entrenamientos ni entregables publicados.
No implica despliegue en Render ni actualización de Jupyter Book.
