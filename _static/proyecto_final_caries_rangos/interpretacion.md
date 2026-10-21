## Comparación global de las siete referencias

Comparé los rangos del AUC externo de las siete referencias V2 en los cinco folds compartidos. Un rango menor indica mayor AUC dentro del fold. El modelo propio y sus ablaciones conservan sus contrastes específicos: no los incorporé como si tuvieran el mismo presupuesto de búsqueda V2.

Friedman obtuvo χ²(6) = 21,4286 y p asintótico = 0,001536. La diferencia crítica de Nemenyi fue 4,0282 (α = 0,05). Solo logística–GaussianNB (p = 0,021791) y XGBoost–GaussianNB (p = 0,004720) superaron esa diferencia. No detecté diferencias por Nemenyi entre las referencias principales; esto no demuestra equivalencia.

![Rangos medios y diferencia crítica de Nemenyi.](/estadistica-rangos/diferencia_critica.png)

**Alcance exploratorio.** Hay solo cinco bloques y los entrenamientos se solapan. La aproximación chi-cuadrado de Friedman es poco fiable con tan pocos bloques. Como sensibilidad, 100.000 permutaciones de etiquetas dentro de fold dieron p = 0,000070, pero tampoco corrigen el solapamiento del entrenamiento ni convierten el análisis en confirmatorio. No interpreto estos resultados como evidencia de utilidad clínica.

Nemenyi controla conjuntamente sus 21 pares bajo sus supuestos; esos p-valores no son los p-valores Holm de los diez contrastes bootstrap del modelo propio. Los dos análisis responden a preguntas diferentes.

Fuentes metodológicas: [SciPy, Friedman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html) y [Demšar, 2006](https://www.jmlr.org/papers/v7/demsar06a.html). La aplicación clásica compara datasets independientes; aquí su uso sobre folds es una aproximación exploratoria solicitada por la guía.

[Descargar rangos medios](/estadistica-rangos/rangos_medios.csv) · [Descargar los 21 pares de Nemenyi](/estadistica-rangos/nemenyi.csv) · [Método y huella de la fuente](/estadistica-rangos/metodo.json)
