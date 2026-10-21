## Búsquedas V2: presupuesto efectivo, cómputo y reproducción

Recuperé y comprobé por SHA-256 los 35.585 archivos del paquete de ejecución. Separé las 35 predicciones externas de las 35 internas: las internas se usaron para calibrar y fijar el umbral, no para estimar el desempeño externo. Las decisiones finales coinciden con las huellas publicadas y cada familia conserva las mismas 44.582 filas externas, una vez por fila, sin cruce de perfiles entre entrenamiento interno y validación externa.

### Qué ejecuté y qué significa el presupuesto

Completé 540 búsquedas correspondientes a 108 combinaciones y cinco folds externos. Hubo 8.640 propuestas nominales, 7.139 configuraciones distintas al sumar las búsquedas y 21.417 ajustes internos registrados. Una configuración repetida en búsquedas diferentes se cuenta en cada búsqueda. Estos ajustes no incluyen los reajustes para calibración y evaluación final. TPE y genética reutilizaron resultados de propuestas repetidas; las repeticiones consumieron presupuesto, sin un nuevo ajuste. Por ello, igualdad de propuestas no significa igualdad de evaluaciones únicas.

### TPE y búsqueda genética

En TPE utilicé Optuna 5.0.0, 16 propuestas, cuatro ensayos iniciales aleatorios, 24 candidatos de adquisición, `multivariate=False`, `constant_liar=False` y semilla 42 + fold externo. TPE estima una densidad l(x) para configuraciones con mejores resultados y otra g(x) para el resto. Selecciona candidatos con mayor razón l(x)/g(x); bajo su formulación, este criterio está relacionado con maximizar la mejora esperada. Es un estimador de densidades, no un proceso gaussiano. No cambié la regla al consultar el fold externo.

En genética utilicé DEAP: población de cuatro, cuatro descendientes, inicialización y tres generaciones de descendencia hasta 16 propuestas. Conservé los cuatro mejores del conjunto de padres y descendientes; seleccioné progenitores por torneo de tamaño dos. El cruce uniforme tiene probabilidad 0,8 y probabilidad por gen 0,5. La mutación se aplica a cada hijo con probabilidad 0,5 y a cada gen con probabilidad 1/d. Los genes categóricos se remuestrean; los reales se perturban en escala logarítmica con desviación 0,2 del ancho del dominio y se acotan al intervalo permitido. La semilla es 42 + fold. No hice múltiples reinicios por búsqueda: estos resultados no prueban estabilidad general del optimizador.

Grid y Random evaluaron 16 configuraciones distintas por búsqueda. Los dominios y parámetros fijos completos están en el paquete de código. Grid recorre los puntos publicados; Random y los métodos adaptativos pueden usar valores intermedios en dominios logarítmicos. Los dominios son comparables, pero las mallas no son idénticas.

### Cómo interpretar los tiempos

Los tiempos provienen de los registros originales, no de estimaciones retrospectivas. La tabla suma preprocesamiento, ajuste, predicción y tiempos de propuesta registrados. Son componentes observados, no duración de pared de toda la ejecución ni benchmark aislado. En Grid/Random se procesaron candidatos dentro de cada fold interno; el historial por propuesta reorganiza sus tres ajustes y no representa la cronología real. El tiempo de propuesta genética no incluye toda la generación de descendencia. No atribuyo la menor suma temporal de genética a mayor eficiencia intrínseca: evaluó menos configuraciones únicas.

Conservé los tiempos de las 540 evaluaciones externas en otra tabla. El piloto de logística tiene campos diferentes; las celdas no registradas permanecen vacías. No medí RAM máxima durante las ejecuciones originales. Tampoco dispongo todavía de un benchmark emparejado de parámetros por defecto frente a optimizados; estos tiempos no lo sustituyen.

### SVM y coste teórico

Usé LinearSVC con pérdida squared_hinge, dual=False, máximo 3.000 iteraciones y tolerancia 0,0001. Esta elección controla el coste en 44.582 filas y 32 columnas codificadas; no constituye una búsqueda de kernels ni demuestra que RBF sea inferior. Los parámetros fijos y dominios de C quedan publicados.

Como orientación algorítmica, para n filas, p columnas, I iteraciones, T árboles, profundidad h y B bins: GaussianNB recorre O(np); logística y SVM lineal suelen requerir trabajo del orden O(Inp), dependiente del optimizador y su convergencia. KNN por fuerza bruta cuesta O(np) por consulta y almacena O(np). Un árbol balanceado con búsqueda de cortes tiene un coste típico O(p n log n), dependiente de implementación; un bosque lo multiplica por T y las variables examinadas. Boosting por histogramas depende de T, n, p, h y B; no lo reduzco a una constante independiente del número de árboles. Son órdenes orientativos, no mediciones de memoria ni garantías universales del ejecutor. El balanceo aumenta n y puede aumentar tanto tiempo como almacenamiento.

### Calibración antes y después

Comparé las probabilidades originales, la sigmoide principal y la isotónica secundaria sobre las mismas predicciones externas. Los modelos balanceados o ponderados pueden producir probabilidades originales alejadas de la prevalencia real. En logística, Brier pasó de 0,241400 a 0,163357 con sigmoide; en XGBoost, de 0,178529 a 0,163067. KNN cambió poco: 0,164020 a 0,163884. SVM no produce una probabilidad original, por lo que ese valor no aplica. La referencia constante se calculó con la prevalencia del entrenamiento de cada fold.

Una sigmoide creciente conserva el orden dentro de cada fold. Sin embargo, cada fold tiene un calibrador diferente: al reunir sus predicciones, pueden cambiar los ordenamientos entre folds y el AUC OOF agrupado. No interpreto ese cambio como una mejora del AUC dentro de los folds. Un ECE bajo tampoco demuestra buena discriminación. La sigmoide se mantuvo como método principal; la isotónica es una comparación secundaria, no una selección posterior por desempeño externo.

### DeLong y tamaños de efecto: análisis de sensibilidad

Añadí los diez contrastes dirigidos mediante DeLong clásico sobre AUC OOF agrupado, con Holm y BH, y tamaños de efecto descriptivos sobre los cinco AUC de folds. Este DeLong trata las filas como independientes: perfiles repetidos y entrenamiento solapado incumplen ese supuesto. Sus p e intervalos son una sensibilidad metodológica, no evidencia confirmatoria ni sustituto del bootstrap por perfiles. Además, el AUC agrupado y el promedio de AUC por fold son estimandos distintos.

Cliff describe la dominancia marginal entre las dos series de cinco valores y no conserva el pareamiento; por ello también presenté la correlación biserial de rangos pareados y Cohen dz. Con sólo cinco folds dependientes no asigné categorías universales de magnitud ni intervalos inferenciales a estos efectos. La conclusión principal permanece: no hay mejora demostrada del modelo propio frente a logística, SVM o XGBoost en el bootstrap agrupado exploratorio.

### Fuentes y archivos reproducibles

La documentación de TPE se contrastó con [Optuna](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html). DeLong se basa en [DeLong, DeLong y Clarke-Pearson, 1988](https://pubmed.ncbi.nlm.nih.gov/3203132/); la dominancia ordinal se refiere a [Cliff, 1993](https://doi.org/10.1037/0033-2909.114.3.494). Las advertencias de obsolescencia de penalty en scikit-learn 1.8 se conservaron en los registros; no se reetiquetaron como fallos de convergencia ni se modificaron los resultados históricos.
