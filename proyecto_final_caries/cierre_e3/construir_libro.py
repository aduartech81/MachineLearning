"""Construye las fuentes del libro E3; nunca entrena modelos al generar capítulos."""
from pathlib import Path
import argparse,json,re,shutil,html,zipfile,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];STATIC=ROOT/'_static/entregable3'
PUBLIC='https://caries-proyecto-final-andres-duarte.onrender.com'
def table(name,cols=None):
 d=pd.read_csv(STATIC/name);d=d[cols] if cols else d;return d.to_html(index=False,border=0,float_format=lambda x:f'{x:.6f}')
def md(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
def code(text):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(keepends=True)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pareado',type=Path,required=True);ap.add_argument('--ejecucion',type=Path,required=True);a=ap.parse_args();book=ROOT/'book_proyecto_final_e3';book.mkdir(exist_ok=True)
 for f in a.pareado.iterdir():
  if f.is_file():shutil.copy2(f,STATIC/('pareado_'+f.name))
 for name in ['metricas.csv','busqueda.csv','protocolo.json']:
  shutil.copy2(a.ejecucion/name,STATIC/('pareado_'+name))
 n=json.loads((ROOT/'proyecto_final_caries_v2.ipynb').read_text());sources=[''.join(c.get('source',[])) for c in n['cells'] if c['cell_type']=='markdown']
 def old(prefix):return next(t for t in sources if t.startswith(prefix))
 chosen=pd.read_csv(STATIC/'seleccion_7_referencias.csv');sel=json.loads((STATIC/'seleccion_estadistica.json').read_text());pm=json.loads((STATIC/'pareado_metodo.json').read_text());comp=pd.read_csv(STATIC/'pareado_resumen_9_modelos.csv');own=float(comp.loc[comp.familia=='Propio','auc_mean'].iloc[0]);best=comp.sort_values('auc_mean',ascending=False).iloc[0]
 trace=json.loads((ROOT/'_static/cierre_entregable2/trazabilidad_auditoria.json').read_text());updates={2:('Ejecutado','Rama auxiliar real: 28 configuraciones y 140 evaluaciones; ver Apéndice A3.'),13:('Ampliado con mediciones','Complejidad n,p: 27 ajustes y teoría condicional; Apéndice A2.'),14:('Ampliado con mediciones','RSS del propio y comparación de dos implementaciones del mismo gradiente en 18 procesos; memoria de proceso, no modelo aislado.'),25:('Ampliado','Descomposición aditiva exacta del propio y de la mejor referencia final; capítulo 5.'),26:('Ampliado','Beeswarm/waterfall de XGBoost en E2 y explicación local exacta del propio final en E3.'),29:('Parcial con evidencia nueva','Tres semillas en tres bases adicionales y perturbaciones controladas. No se declara sensibilidad completa a semillas en caries.'),33:('Adaptación transversal','Contrastes apropiados al diseño; no se inventa orden temporal para justificar DM/GW/MCS.'),35:('Libro E3 por capítulos','Fuentes, código y resultados precalculados; ejecución ligera verificada aparte. Reentrenamiento histórico integral no repetido.'),37:('Ampliado','Requisitos fijados, metadatos por modelo y hashes. Entorno CPU del servidor separado de investigación.'),39:('Siete pruebas pasaron','Cinco del modelo y dos de AUC ponderada/jackknife por perfiles.'),40:('Guion privado preparado','Rúbrica de 20 minutos con 18 de exposición y 2 de preguntas; no se publica el guion.'),41:('Conservado','Discriminación limitada, R1 no acreditado y R3 no demostrado; no se ocultan resultados desfavorables.')}
 for t in trace:
  if t['id'] in updates:t['estado'],t['accion_y_evidencia']=updates[t['id']]
  t['ubicacion']='Cap. 1; detalles en capítulos 2–9 y apéndices'
 pd.DataFrame(trace).to_csv(STATIC/'trazabilidad_42_e3.csv',index=False)
 chapters={}
 chapters['intro']=[md(f'''# 0. Caries dental: modelar, contrastar y explicar

<div class="e3-hero"><div><p class="e3-kicker">ANDRÉS DUARTE CHAVES · MACHINE LEARNING</p><h2>Del chequeo general a una evaluación verificable.</h2><p>Proyecto final · Maestría en Ingeniería Biomédica. Cierre preparado el 23 de octubre para la entrega y sustentación del 24 de octubre de 2026.</p><p class="e3-actions"><a class="e3-button" href="{PUBLIC}/">Explorar el dashboard</a> <a class="e3-button e3-secondary" href="{PUBLIC}/predecir/">Probar una predicción</a></p></div><figure><img src="_static/entregable3/portada_caries.png" alt="Ilustración del proyecto de caries y análisis de datos"><figcaption>Ilustración generada con IA. No muestra participantes ni es una entrada del modelo.</figcaption></figure></div>

## Resumen ejecutivo

Se estudia qué información aportan 23 variables del chequeo general para identificar **caries ya registrada**. El objetivo es binario, $y_i\\in\\{{0,1\\}}$. Es clasificación contemporánea; no se pronostica aparición futura. Las 44.582 filas de desarrollo se agrupan por perfiles idénticos de predictores para evitar repartir sus copias entre folds. Ese agrupamiento no identifica pacientes.

La comparación V2 del segundo entregable incluye 108 configuraciones aplicables; las cuatro KNN con ponderación de clases no son implementables mediante `class_weight`. La regresión auxiliar tiene 28 configuraciones y otro objetivo: glucemia registrada. Para E3 se selecciona **clasificación** como tarea de despliegue.

**Hallazgo central:** en la evaluación pareada suplementaria, el propio alcanza AUC medio **{own:.6f}** y la mejor referencia observada, **{best.familia}**, **{best.auc_mean:.6f}**. La discriminación sigue siendo limitada. No se acredita superioridad del propio ni utilidad clínica. Esta comparación tiene presupuesto común y es distinta del ranking V2 utilizado para seleccionar artefactos.

**Producto de ingeniería:** siete referencias y el propio guardados, metadatos por modelo, verificación exacta de guardar/cargar en 128 casos técnicos, dashboard público y explicación aditiva del propio. Hay siete pruebas unitarias ejecutadas.

**Resultado de investigación:** R1 no está acreditado; hay antecedentes de GAM en caries y de modelos aditivos con interacciones/ranking en otras tareas. R2 cuenta con núcleo algorítmico y pruebas, pero el dominio individual debe demostrarse en la sustentación. R3 no está demostrado. Las limitaciones no se reemplazan por una afirmación de novedad o significancia.

## Objetivos y recorrido

1. Documentar correcciones y distinguir auditoría de retroalimentación docente.
2. Seleccionar y serializar las siete mejores configuraciones de clasificación del E2.
3. Explicar el algoritmo propio, sus supuestos, gradiente, propiedad condicional y ablaciones.
4. Contrastar resultados mediante métodos adecuados al diseño transversal y control múltiple.
5. Publicar una interfaz verificable con predicción y explicación local, fuentes y resultados.

![Esquema de datos y validación](_static/entregable3/flujo_validacion.svg)

El esquema resume el entrenamiento dentro de cada partición. No implica independencia clínica ni un nuevo conjunto de prueba no consultado.

## Mapa del libro

| Capítulo | Pregunta que responde |
|---|---|
| 1. Correcciones | ¿Qué se corrigió y qué sigue parcial? |
| 2. Datos y EDA | ¿Qué representan las filas y cómo se transforman? |
| 3. Benchmarks | ¿Por qué se seleccionó cada referencia? |
| 4. Literatura | ¿Qué antecedentes existen y qué novedad no está probada? |
| 5. Modelo propio | ¿Qué se calcula, cómo aprende y cómo se explica? |
| 6. Validación | ¿Qué muestran las comparaciones y sus supuestos? |
| 7. Dashboard | ¿Cómo se publican los modelos sin entrenar en línea? |
| 8. Conclusiones | ¿Qué puede y qué no puede concluirse? |
| 9. Reproducibilidad | ¿Cómo se reconstruyen tablas, código y evidencias? |
| Apéndices | Tres bases adicionales, robustez/cómputo y regresión auxiliar |

La entrega anterior permanece en su [enlace fechado]({PUBLIC}/entregas/entregable-2-20261023/). El [portafolio]({PUBLIC}/portafolio/) conecta las etapas del trabajo.
''')]
 chapters['01_correcciones']=[md('''# 1. Trazabilidad de correcciones

La tabla siguiente procede de la **auditoría integral de 42 controles**, no de observaciones textuales atribuidas al docente. No se han aportado nuevas observaciones docentes posteriores al envío del E2 del 23 de octubre. No se afirma que ese ciclo de retroalimentación esté cerrado.

| Observación del docente | Acción | Ubicación | Evidencia |
|---|---|---|---|
| No disponible en la documentación recibida tras el envío del E2 | Conservar este registro y añadir las observaciones literales si se reciben | Este capítulo | Estado explícito; no se inventan correcciones |

## Matriz de la auditoría

Los estados parciales se mantienen cuando hay una limitación real. Los resultados V1/V2 y la entrega enviada no se reescriben para que coincidan con los experimentos suplementarios.

'''+pd.DataFrame(trace)[['id','observacion','estado','accion_y_evidencia','ubicacion']].to_html(index=False,border=0)+'''

[Descargar matriz de 42 controles](_static/entregable3/trazabilidad_42_e3.csv)

## Cambios que deben interpretarse por versión

La selección de siete configuraciones por media externa E2 no es el mismo procedimiento que elegir internamente una configuración distinta en cada fold. La comparación suplementaria de presupuesto común introduce otro protocolo identificado. Sus métricas no sustituyen tablas V1/V2. Las explicaciones de modelos finales pertenecen a objetos reajustados con todo el desarrollo y no se usan como evaluación externa.
'''),code("# Comprobar que ningún control se pierde al actualizar la matriz.\nimport pandas as pd\nauditoria = pd.read_csv('_static/entregable3/trazabilidad_42_e3.csv')\nassert len(auditoria) == 42 and auditoria.id.is_unique\nprint(auditoria[['id', 'estado']].to_string(index=False))\n")]
 chapters['02_datos']=[md('# 2. Datos, ETL y EDA\n\n'+old('## 2. Datos y diseño experimental')+'\n\n'+old('## Datos, exploración y trazabilidad del desarrollo')+'''

## Interpretación para este cierre

Las 55.692 filas del archivo original no son una población clínica validada por este proyecto. Se conservan 44.582 para desarrollo; el conjunto histórico de 11.110 ya fue consultado en etapas anteriores y no se presenta como un nuevo test independiente. En clasificación hay 35.641 perfiles de predictores distintos en desarrollo. Los códigos, la licencia y el protocolo diagnóstico original requieren verificación adicional; no se completan con supuestos.

ID se excluye por ser identificador, oral por ser constante en este archivo y tartar por su disponibilidad ligada al examen oral. Son motivos diferentes. El balanceo se aplica solo al entrenamiento correspondiente. SMOTE/ADASYN sobre categorías codificadas pueden generar valores fraccionarios: esa limitación no desaparece por usar CV.

![Flujo de validación agrupada](_static/entregable3/flujo_validacion.svg)

Agrupar perfiles evita compartir entradas idénticas entre entrenamiento y validación, pero no garantiza ausencia de otros vínculos entre personas. La selección y las transformaciones deben respetar ese alcance.
''')]
 chapters['03_benchmarks']=[md('''# 3. Referencias seleccionadas y modelos finales

## Regla de selección E3

Para cada familia se escoge la combinación de optimizador y balanceo con mayor AUC medio externo de las 108 configuraciones aplicables E2. La tabla muestra media y DE de cinco folds; **la DE no es un intervalo de confianza**. Es una selección sobre resultados ya observados, por lo que el AUC del ganador puede ser optimista.

'''+table('seleccion_7_referencias.csv',['familia','optimizador','balanceo','auc_mean','auc_std','average_precision_mean','accuracy_mean'])+f'''

La mejor configuración global observada pertenece a **{sel['mejor_global']}**, AUC medio **{sel['auc_medio']:.6f}**. El conjunto exploratorio de no rechazo frente a ella se calcula con las siete configuraciones fijas seleccionadas para E3, no con selecciones cambiantes de otro análisis.

'''+table('finalistas_nemenyi_seleccion_externa.csv')+'''

No rechazar diferencias no demuestra equivalencia. La selección posterior y el solapamiento de entrenamientos limitan la inferencia. El detalle de esta selección está en [metodología](_static/entregable3/seleccion_estadistica.json).

## Consolidación y artefactos

La familia y su combinación proceden de la media externa. Para fijar un único vector de hiperparámetros se toma el modo de las cinco selecciones internas de esa configuración; el desempate es lexicográfico. Es una regla de consolidación documentada, no una búsqueda nueva en el test. El propio usa el modo de sus parámetros V1.

Los pipelines se reajustaron con las 44.582 filas de desarrollo. Los objetos finales de clasificación están en `proyecto_final_caries/artifacts/<familia>/model.joblib`, junto a `metadata.json`: versiones, semilla, hiperparámetros completos, esquema, rangos observados, métricas históricas y huella. La aplicación carga la copia fechada equivalente de esos artefactos.

'''+table('roundtrip_8_modelos_128_casos.csv')+'''

La prueba técnica tomó 128 posiciones fijas del CSV de desarrollo (semilla 42), serializó cada objeto y comprobó igualdad exacta de sus scores después de cargarlo. No estima generalización: los casos provienen de datos usados al ajustar. Los artefactos comprimidos ocupan en conjunto unos pocos MiB. La salida original de los modelos finales no tiene calibración final nueva ni umbral clínico establecido.

[Selección de siete referencias](_static/entregable3/seleccion_7_referencias.csv) · [Prueba técnica](_static/entregable3/roundtrip_8_modelos_128_casos.csv)

## Presupuesto y resultados E2 conservados

'''+old('## Búsquedas V2: presupuesto efectivo')+'''

Estos resultados históricos conservan su versión. La comparación de presupuesto común del capítulo 6 no los sustituye.
'''),code("# Reproducir la regla de selección desde la tabla de 108 configuraciones.\nimport pandas as pd\ntabla = pd.read_csv('_static/cierre_entregable2/resumen_108_media_de.csv')\nmejores = tabla.loc[tabla.groupby('familia').auc_mean.idxmax()]\nassert len(mejores) == 7\nprint(mejores[['familia', 'optimizador', 'balanceo', 'auc_mean', 'auc_std']].to_string(index=False))\n")]
 chapters['04_literatura']=[md('''# 4. Literatura y evaluación de novedad

## Protocolo y alcance

Se registraron cinco cadenas en inglés/español en tres fuentes: PubMed, Crossref y arXiv. **Crossref es un registro bibliográfico**, no una base equivalente a Scopus. Hay 15 intentos y 14 respuestas; un acceso fallido se conserva. Las cadenas amplias de PubMed/Crossref pueden devolver resultados poco pertinentes, mientras las frases estrictas en arXiv no recuperaron entradas. Esos conteos no son conteos de artículos relevantes.

Esta es una búsqueda estructurada preliminar más lectura focalizada, no una revisión sistemática exhaustiva. No se inventan capturas de búsquedas, acceso a bases no consultadas, número de resúmenes cribados ni ausencia universal de antecedentes.

'''+table('matriz_busquedas.csv',['base','consulta','cadena','fecha_utc','estado','recuperados'])+'''

[Registro tabular](_static/entregable3/matriz_busquedas.csv) · [Veredicto y límites](_static/entregable3/veredicto_novedad.json)

## Ocho métodos afines y aplicación

'''+table('matriz_novedad.csv',['familia_afin','anio','lectura_efectuada','mecanismo_cercano','que_se_conserva','diferencia_y_limite'])+'''

Los GAM ya se aplicaron al estudio de caries en dientes primarios {cite:p}`helf1997`. Esa aplicación no acredita identidad de población y etiqueta con nuestro experimento binario, pero impide presentar el uso general de GAM en caries como una idea inédita. Las funciones aditivas y la selección de pocas interacciones tienen antecedentes explícitos {cite:p}`lou2012,lou2013`. La pérdida logística por pares y su relación con AUC también son conocidas {cite:p}`gao2015,agarwal2014`.

EBM y sus variantes de sparsity son referencias cercanas {cite:p}`nori2019,greenwell2023`; SpAM muestra que regularizar modelos aditivos tampoco es una novedad por sí sola {cite:p}`ravikumar2007`. La evaluación añade una implementación actual de EBM con presupuesto acotado; no se la presenta como cobertura de todo el estado del arte.

## Declaración de novedad

**R1 no acreditado.** La evidencia disponible permite describir una implementación propia de un objetivo compuesto y una heurística de interacción, pero no demostrar un mecanismo algorítmico científicamente nuevo ni la ausencia de su aplicación exacta a caries. No se usa la frase “nadie lo ha aplicado” como sustituto de una búsqueda y diferenciación suficientes.

La matriz indica qué partes de cada fuente se consultaron; una lectura de resumen no se presenta como revisión de todo el texto. El registro bruto y las respuestas están incluidos en el código/evidencias del cierre.
''')]
 chapters['05_modelo_original']=[md('''# 5. Modelo propio: algoritmo, propiedad y explicación

## Idea y alcance

Se busca una función aditiva que permita inspeccionar términos univariados y pocas interacciones. El ranking intenta favorecer ordenamiento positivo–negativo; L2 limita coeficientes. La combinación no garantiza calibración, estabilidad de las interacciones ni mejora predictiva.

![Esquema del algoritmo propio](_static/entregable3/flujo_modelo.svg)

'''+old('## Fundamento del modelo propio: alcance verificable')+r'''

## Formulación numerada y gradiente

Para $z_i=\phi(x_i)$ y parámetros $\theta=(b,w)$:

$$
s_i=b+z_i^\top w,\qquad p_i=\sigma(s_i)=\frac{1}{1+\exp(-s_i)}. \tag{5.1}
$$

Con $\mathcal P$ fijo de $M$ pares $(p,q)$, positivos y negativos:

$$
L(\theta)=\frac{1-\alpha}{n}\sum_{i=1}^n[\log(1+e^{s_i})-y_i s_i]+\frac{\alpha}{M}\sum_{(p,q)\in\mathcal P}\log(1+e^{-(s_p-s_q)})+\frac{\lambda}{2}\Vert w\Vert_2^2. \tag{5.2}
$$

Si $r_{pq}=\sigma(-(s_p-s_q))/M$, el ranking aporta $-r_{pq}$ al score positivo y $+r_{pq}$ al negativo. Agrupar esos aportes por índice produce un vector $g_s$, sin construir diferencias de características para cada par. El gradiente es:

$$
\nabla_w L=Z^\top g_s+\lambda w,\qquad
\partial_b L=\frac{1-\alpha}{n}\sum_i(\sigma(s_i)-y_i). \tag{5.3}
$$

El ranking cancela el intercepto porque solo depende de diferencias de scores. En la implementación, `logaddexp` evita exponentes inestables y `bincount` acumula gradientes por fila.

## Pseudocódigo formal

**Entrada:** $X,y$, $0\le\alpha<1$, $\lambda>0$, número de nudos, máximo de interacciones, número de pares, semilla, tolerancia y límite de iteraciones. **Salida:** transformaciones ajustadas, pares de variables seleccionados, coeficientes, intercepto y diagnóstico de convergencia.

1. Validar esquema, columnas excluidas, tamaños y presencia de ambas clases.
2. Aprender imputación, bases spline, one-hot y escalas solo en el ajuste actual.
3. Muestrear con reemplazo $M$ pares positivo–negativo y mantenerlos fijos.
4. Ajustar el aditivo preliminar y calcular residuos internos.
5. Explorar productos numéricos mediante submuestras de perfiles del ajuste; ordenar por frecuencia/asociación y conservar hasta cinco.
6. Optimizar (5.2) con gradiente (5.3), L-BFGS-B y máximo de iteraciones.
7. Registrar éxito, iteraciones, pérdida y gradiente. Un fallo de convergencia detiene la evaluación.
8. Para nuevas filas, aplicar las transformaciones guardadas y calcular (5.1).

La parada es la condición numérica de L-BFGS-B o el límite de iteraciones; no una prueba de solución exacta. La representación y los pares fijos permiten la propiedad convexa demostrada arriba, pero la selección de interacciones queda fuera de ella.

## Pruebas y ablaciones

Siete pruebas pasaron: regla sintética conocida, repetibilidad, clone sin estado, entradas inválidas/clase única, gradiente por diferencias finitas, AUC ponderada con empates frente a sklearn y jackknife exacto frente a retirar cada perfil. No son una certificación clínica.

'''+old('## 3. Contribución implementada y ablaciones')+'''

Las ablaciones V1 se mantienen: el aditivo sin ranking ni interacciones alcanzó AUC medio aproximadamente 0,606151, frente a 0,605062 del propio completo. No prueban que añadir ambos componentes ayude. Los hiperparámetros V1 se optimizaron por grilla dentro de la CV anidada; el suplemento pareado conserva cuatro candidatos propios y cuatro por referencia.

## Interpretación propia y mejor referencia final

![Importancia aditiva descriptiva de dos modelos finales](_static/entregable3/explicacion_global_finales.png)

Se suman las contribuciones de bases por variable, manteniendo las interacciones aparte. Son 500 perfiles de desarrollo y objetos ya reajustados; no observaciones externas. Las escalas de margen de modelos distintos no permiten comparar magnitudes de importancia absoluta entre paneles. La interpretación es predictiva, no causal.

![Explicación exacta de un perfil sintético](_static/entregable3/explicacion_local_propio.png)

Valor base más todas las contribuciones reconstruye el margen con error inferior a $10^{-12}$. La sigmoide original no se presenta como riesgo clínico calibrado. Esta descomposición es una alternativa justificada a SHAP para una función aditiva explícita; no es una estimación de efectos de intervenir sobre una persona.

[Método de explicación](_static/entregable3/explicacion_metodo.json) · [Contribuciones del caso](_static/entregable3/explicacion_local_propio.csv)

## Código del núcleo

La copia siguiente coincide con `proyecto_final_caries/modelo.py`. `objetivo.py` contiene la pérdida y su gradiente. No se oculta el uso de transformadores y optimización SciPy; la lógica propia está identificada, y el autor debe demostrar comprensión y capacidad de modificarla durante la defensa.

````{dropdown} Ver la implementación de la pérdida y el gradiente
```{literalinclude} src/objetivo.py
:language: python
:linenos:
```
````
''')]
 chapters['06_validacion']=[md('''# 6. Evaluación crítica y validación estadística

## Separar protocolos y preguntas

- **V1/V2:** resultados históricos, calibración y comparaciones del E2, con sus presupuestos distintos y límites conservados.
- **Selección E3:** siete configuraciones fijas según media externa E2; no un nuevo test.
- **Suplemento pareado E3:** nueve familias, sin balanceo, mismas particiones externas e internas y cuatro candidatos por familia. Preprocesamiento previo común; los splines/interacciones son representación interna del propio. No reemplaza las 108 configuraciones E2.

No se presenta una diferencia entre estos protocolos como una mejora causal del algoritmo. En todos los casos se reutilizan datos de desarrollo ya consultados.

## Comparación pareada suplementaria

'''+table('pareado_resumen_9_modelos.csv',['familia','auc_mean','auc_std','average_precision_mean','balanced_accuracy_mean','ajuste_s_mean','inferencia_s_mean'])+'''

![AUC y variabilidad en el protocolo pareado](_static/entregable3/pareado_comparacion_pareada.png)

La gráfica presenta media y DE de cinco folds. Las barras no son intervalos de confianza ni evidencia de utilidad clínica. La discriminación modesta es común a varias familias; los datos no justifican prometer detección clínica.

## Procedimientos, hipótesis y control múltiple

Friedman con rangos globales y Nemenyi constituye el procedimiento global exploratorio {cite:p}`demsar2006`. Se acompaña de Wilcoxon pareado, una prueba t con corrección de varianza CV de Nadeau–Bengio y bootstrap pareado de perfiles centrado {cite:p}`nadeau2003`. Se corrigen conjuntamente con Holm los **24 contrastes pareados** mostrados (ocho referencias por tres procedimientos). Esa corrección no abarca todo el historial exploratorio del proyecto.

'''+table('pareado_pruebas_pareadas.csv',['prueba','comparacion','estadistico','p_crudo','p_holm_24','delta_auc_media_folds','delta_auc_oof','bca95_inferior','bca95_superior','cohen_dz_descriptivo','conclusion_exploratoria'])+'''

[Tabla con hipótesis y supuestos por contraste](_static/entregable3/pareado_pruebas_pareadas.csv) · [Protocolo estadístico](_static/entregable3/pareado_metodo.json)

### Límites de las pruebas

Hay cinco folds con entrenamiento solapado, una sola partición externa en el problema principal y selección ya exploratoria. La corrección t es aproximada; no certifica normalidad ni independencia. Wilcoxon tiene baja potencia y supuestos no certificados. Los resultados de distintos procedimientos comparten datos y no son confirmaciones independientes.

El bootstrap remuestrea perfiles completos y calcula intervalos BCa con jackknife delete-one-profile exacto, verificado mediante pruebas. Es **condicional a predicciones fijas**: no incluye reentrenamiento ni identidad clínica verificada. Los IC 95 % son puntuales, no simultáneos. AUC OOF y promedio de AUC por fold son estimandos diferentes; además, los scores de modelos ajustados en folds distintos pueden tener escalas diferentes. No se sustituyen selectivamente.

![Rangos medios y longitud de diferencia crítica](_static/entregable3/pareado_rangos_cd.png)

La diferencia crítica es un diagnóstico asintótico frágil con cinco bloques. No rechazar igualdad no demuestra equivalencia. Cohen dz es descriptivo sobre diferencias de folds dependientes; Cliff marginal también se conserva como descriptor, no como prueba de relevancia clínica.

## Inferencia histórica y resultados desfavorables

'''+old('## 4. Evaluación e incertidumbre')+'''

DeLong clásico supone independencia por observación y no corrige perfiles repetidos o entrenamiento compartido {cite:p}`delong1988`. Su sensibilidad histórica no se convierte en confirmación. McNemar no se aplica a un supuesto test nuevo porque no existe tal conjunto independiente disponible en este cierre.

## Veredicto R3

**No demostrado.** El propio no supera a todas las mejores referencias de E2 y los análisis no acreditan las tres confirmaciones ajustadas y un efecto relevante que exige la guía. Añadir pruebas, semillas o bases no cambia automáticamente ese veredicto. Las pruebas temporales DM, GW, Clark–West y MCS estacionario no se justifican creando un orden con `row_id` en datos transversales.

La evaluación amplia, las semillas y las perturbaciones se presentan en {doc}`10_adicionales` y {doc}`11_robustez_computo`.
'''),code("# Reconstruir media y DE del protocolo pareado desde sus 45 resultados.\nimport pandas as pd\nfolds = pd.read_csv('_static/entregable3/pareado_metricas.csv')\nassert len(folds) == 45 and not folds.duplicated(['familia', 'fold']).any()\nprint(folds.groupby('familia').auc.agg(['mean', 'std']).to_string())\n")]
 chapters['07_dashboard']=[md(f'''# 7. Dashboard, serialización y despliegue

## Interfaz pública

[Dashboard Dash]({PUBLIC}/) · [Predicción y explicación local]({PUBLIC}/predecir/) · [Portafolio]({PUBLIC}/portafolio/) · [Libro E3 fechado]({PUBLIC}/entregas/entregable-3-20261024/)

El dashboard organiza contexto, datos/EDA, modelos y aporte, evaluación, resultados, conclusiones y regresión auxiliar. La entrada a predicción permite escoger las siete referencias o el propio. Valida esquema exacto, tipos, categorías, finitud y rangos observados de desarrollo. Esos rangos no son límites clínicos. El perfil inicial es sintético (medianas/modas), no un paciente real.

![Arquitectura del despliegue](_static/entregable3/arquitectura_web.svg)

La aplicación carga objetos ya entrenados y cachea hasta ocho. No ajusta modelos ni busca hiperparámetros en un callback. La explicación del propio suma términos del margen, y las otras referencias muestran score/probabilidad original según su capacidad. No se fija un umbral clínico ni se finge una probabilidad calibrada para SVM.

## Entorno del servicio independiente

El servicio se conecta a la rama `proyecto-final-independiente-20261018-213747`. Se conserva `main` y el servicio anterior. Comando de construcción:

```text
pip install -r proyecto_final_caries/requirements_inferencia.txt
```

Comando de inicio:

```text
CARIES_HABILITAR_PREDICCION=1 gunicorn proyecto_final_caries.presentacion.app:server --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

El servidor usa la distribución CPU de XGBoost 3.2.0; el entorno de entrenamiento Mac usa la distribución XGBoost 3.2.0. Las versiones se documentan por entorno; no se afirma igualdad de binarios ni scores bit a bit entre plataformas.

## Verificaciones ejecutadas

- Ocho modelos responden con entradas sintéticas válidas; la explicación propia verifica aditividad.
- Entradas faltantes y fuera de rango se rechazan.
- Libro, rutas de evidencias, imágenes y enlaces públicos se comprueban antes de cerrar el paquete.
- `/exposicion/` sigue retirado. El guion oral permanece privado.

El plan gratuito puede tardar en arrancar tras inactividad. Se entrega copia HTML del libro; abrir el portal antes de la sustentación permite verificar su disponibilidad. Esto no garantiza ausencia de caídas futuras.

## Captura de la interfaz

![Dashboard publicado con la regresión auxiliar](_static/entregable3/dashboard_publicado.png)

La captura es evidencia de una versión consultada; la comprobación de la URL pública y los artefactos es la verificación funcional, no la imagen por sí sola.
''')]
 chapters['08_conclusiones']=[md(f'''# 8. Conclusiones, limitaciones y veredicto

## Hallazgos sustentados

1. Los predictores del chequeo general tienen discriminación limitada para caries registrada. En el protocolo pareado el propio alcanza AUC medio {own:.6f}; no se acredita uso clínico.
2. El propio combina componentes conocidos y tiene una propiedad de optimización condicional demostrable. Las ablaciones no demostraron que ranking e interacciones mejoren el aditivo simple.
3. La selección E3 y el reajuste permiten una demostración real con ocho artefactos, metadatos y verificación de serialización. Desplegar no convierte el modelo en un sistema clínicamente validado.
4. Tres bases adicionales y perturbaciones muestran comportamiento fuera del único problema principal, pero no transferencia clínica del modelo de caries ni dominio de todo el estado del arte.

## R1, R2 y R3

| Condición | Veredicto defendible |
|---|---|
| R1: novedad | No acreditada. Hay antecedentes cercanos y la búsqueda es preliminar. |
| R2: implementación algorítmica | Hay clase, función de pérdida, gradiente, selección y pruebas; el dominio individual y la responsabilidad del autor deben demostrarse. |
| R3: superioridad | No demostrada frente a todas las mejores referencias con las garantías requeridas. |

No se promete una calificación ni se convierte un resultado negativo en una afirmación de superioridad.

## Límites teóricos y empíricos

La convexidad es condicional a la representación y pares fijados; no acredita estabilidad de selección ni generalización. El muestreo de pares por fila puede sobreponderar perfiles repetidos dentro del ajuste. La evaluación agrupa entradas idénticas, pero no pacientes identificados. La fuente y los códigos diagnósticos no tienen verificación clínica completa.

Las cinco particiones principales se reutilizaron durante el desarrollo y comparten entrenamiento. Las pruebas y los intervalos no incorporan todas las fuentes de incertidumbre. Los hiperparámetros modales de despliegue son una regla de consolidación; sus métricas siguen siendo las históricas, no un nuevo test del objeto final.

## Mejoras futuras concretas

Verificar procedencia y definición de caries; conseguir una cohorte independiente con disponibilidad temporal acreditada; comparar bajo presupuesto más amplio y CV repetida; estudiar un selector de interacciones con garantías; evaluar calibración y utilidad según costes establecidos con expertos. No se promete que una de esas mejoras vaya a producir significancia.

## Uso de IA y revisión del autor

Se utilizó ChatGPT/Codex para revisión de documentos y referencias, recuperación de evidencias, implementación asistida, pruebas, análisis, documentación, diseño visual y preparación de la sustentación. Los resultados nuevos corresponden a ejecuciones registradas. La intervención de IA no se oculta ni equivale a comprensión demostrada por el autor. El autor debe revisar, comprender y asumir el contenido, y completar cualquier declaración firmada exigida por el docente. No se incorpora una firma automática.
''')]
 chapters['09_reproducibilidad']=[md('''# 9. Reproducibilidad y bibliografía

## Qué se puede reconstruir

La guía E3 admite resultados costosos precalculados y versionados si se entrega el código generador. El libro compila con ejecución de entrenamientos desactivada; sus celdas ligeras de agregación se verifican aparte. No se afirma que se haya repetido integralmente toda la historia V1/V2 en un entorno limpio.

La tabla siguiente incluye **140 configuraciones nominales de E2 más el propio V1**. De las nominales, 136 son aplicables y cuatro KNN con pesos quedan sin métricas. AUC y RMSE responden a tareas distintas y no se mezclan en un ranking único.

'''+table('tabla_maestra_140_nominales_y_propio.csv')+'''

[Tabla maestra de 141 filas](_static/entregable3/tabla_maestra_140_nominales_y_propio.csv)

## Ejecución desde el repositorio

El archivo de requisitos de inferencia contiene solo lo necesario para el servidor. Para investigación se usan los requisitos V2 y el complemento `proyecto_final_caries/requirements_investigacion_e3.txt`. InterpretML 0.7.8 es una referencia externa, no el algoritmo propio.

```text
python -m pytest -q proyecto_final_caries/tests
python -m proyecto_final_caries.validacion_pareada_e3.ejecutar --salida NUEVA_CARPETA
python -m proyecto_final_caries.cierre_e3.analizar_pareado --entrada NUEVA_CARPETA --salida NUEVO_ANALISIS
python -m proyecto_final_caries.validacion_adicional.ejecutar
python -m proyecto_final_caries.validacion_adicional.ebm
python -m proyecto_final_caries.cierre_e3.robustez
python -m proyecto_final_caries.cierre_e3.complejidad --salida NUEVA_CARPETA_COMPLEJIDAD
python -m proyecto_final_caries.cierre_e3.vectorizacion --salida NUEVA_CARPETA_VECTORIZACION
jupyter-book build book_proyecto_final_e3 --warningiserror
```

Los ejecutores de experimentos costosos protegen las carpetas existentes; para repetir los suplementos adicionales debe conservarse la evidencia previa fuera de las carpetas de salida fijas o usar una copia nueva del repositorio. La selección, semillas, grids y limitaciones están en los protocolos JSON. Los módulos generadores del cierre construyen tablas/figuras desde esos archivos; no ejecutan entrenamientos al navegar por el libro.

Se entregan datos públicos, instrucciones, versiones y hashes. Los metadatos codifican sentinelas no finitos de hiperparámetros como cadenas para mantener JSON estricto. Los archivos precalculados no suplen una dependencia que falte; no se generan sustitutos silenciosos.

## Control técnico

Siete pruebas pasaron. Los ocho artefactos se verificaron por SHA256 y con un conjunto fijo de 128 casos de serialización. Los resultados de las 243 evaluaciones adicionales se cotejan con sus predicciones y se reconstruyen sus particiones desde los datos fuente. Los resultados principales suplementarios verifican las 45 evaluaciones y la alineación OOF.

## Bibliografía

```{bibliography}
:all:
```
'''),code("# Auditoría de cobertura nominal sin inventar métricas en combinaciones inaplicables.\nimport pandas as pd\nmaestra = pd.read_csv('_static/entregable3/tabla_maestra_140_nominales_y_propio.csv')\nassert len(maestra) == 141\nassert (maestra.folds == 0).sum() == 4\nassert maestra.loc[maestra.folds == 0, 'media'].isna().all()\nprint(maestra.groupby(['tarea', 'estado']).size().to_string())\n")]
 chapters['10_adicionales']=[md('''# A1. Tres bases adicionales y referencia moderna

## Muestras y tareas

Se evalúa clasificación binaria por separado en WDBC (569 filas y 30 predictores, tumor maligno como positivo), Banknote (1.372 filas y cuatro medidas de imágenes) y Spambase (4.601 filas y 57 atributos de correo, spam como positivo) {cite:p}`uci_wdbc,uci_banknote,uci_spam`. Son tres ámbitos distintos, no una validación externa de caries.

Las ocho familias originales y EBM usan las mismas particiones, dos candidatos por familia, dos folds internos, tres externos y semillas 42, 43 y 44. Hay **243 evaluaciones externas** en total. El filtro no supervisado conserva variables cuyos cinco cuantiles sean estrictamente crecientes y se ajusta solo con entrenamiento, por igual para todos. Esto limita especialmente la representación dispersa de Spambase.

El propio exige cinco columnas categóricas del esquema de caries; un adaptador añade constantes cero sin señal de etiqueta. Es una limitación de su interfaz general, no una propiedad de los nuevos dominios. No se oculta la adaptación ni se presenta como un modelo general listo sin cambios.

![Comparación en tres bases](_static/entregable3/adicionales_heatmap.png)

La figura muestra medias de AUC por base y algoritmo. No se comparan esos valores como si fueran dificultad clínica equivalente. Un resultado favorable en una base no implica superioridad universal.

![Sensibilidad a semillas](_static/entregable3/adicionales_semillas.png)

Son tres medias de folds con datos reutilizados, no tres cohortes independientes. La tabla informa media y DE entre esas medias; no intervalos de confianza.

'''+table('adicionales_resumen.csv',['dataset','familia','auc_mean','auc_std','ap_mean','balanced_accuracy_mean'])+'''

[243 resultados por fold](_static/entregable3/adicionales_243_folds.csv) · [Método y verificación](_static/entregable3/adicionales_metodo.json)

La versión EBM es actual, con 500 rondas y una bolsa fijadas por coste; no es una optimización exhaustiva ni cubre todas las variantes recientes. La comparación moderna añade un control útil, no una acreditación de estado del arte.
'''),code("import pandas as pd\nresultados = pd.read_csv('_static/entregable3/adicionales_243_folds.csv')\nassert len(resultados) == 243\nassert (resultados.groupby(['dataset', 'familia']).size() == 9).all()\nprint(resultados.groupby(['dataset', 'familia']).auc.mean().unstack().round(4).to_string())\n")]
 chapters['11_robustez_computo']=[md('''# A2. Robustez y recursos del algoritmo propio

## Perturbaciones controladas

Se fijan los hiperparámetros elegidos internamente para el fold externo 0 de cada base y semilla, sin reajustarlos con los resultados perturbados. Se comparan propio, XGBoost y EBM en entrenamiento original, 50 % de perfiles de entrenamiento y 10 % de perfiles con etiqueta invertida. La validación queda limpia y fija. Son 81 ajustes; las tres semillas reutilizan los mismos datos.

![Escenarios de robustez](_static/entregable3/robustez_escenarios.png)

Las perturbaciones son escenarios acotados y posteriores al experimento. No cubren todo el ruido, desbalance, tamaño de muestra ni cambio de población posible, y no acreditan robustez clínica en caries.

[81 resultados de robustez](_static/entregable3/robustez_81.csv)

## Escalamiento empírico con n y p

Se entrenó el propio en 27 procesos nuevos: n de 2.000, 8.000 y 20.000; p de 11, 17 y 23; tres repeticiones con semilla 42. La configuración es fija y la variación entre repeticiones mide ejecución, no sensibilidad estadística al azar de entrenamiento.

![Tiempo y memoria con n y p](_static/entregable3/complejidad_np.png)

El tiempo incluye el ajuste y sus transformaciones. RSS es el máximo observado del proceso muestreado cada 10 ms, con importaciones y datos; no memoria exclusiva del modelo ni máximo continuo exacto. La Mac ejecutaba otros trabajos: no es un experimento de hardware aislado. Las mediciones complementan, pero no prueban por sí solas, los órdenes asintóticos del capítulo 5.

'''+table('complejidad_resumen.csv')+'''

## Optimización computacional distinta de cambiar hiperparámetros

Se comparó evaluar exactamente la misma pérdida/gradiente construyendo una matriz explícita de diferencias por par, frente a acumular aportes por fila con `bincount`. La segunda variante ya es la implementación de `objetivo.py`; el benchmark no modifica modelos ni sus AUC.

'''+table('vectorizacion_resumen.csv')+'''

El ensayo usa n=5.000, d=128 y tres tamaños de pares M, con 18 procesos nuevos y cinco evaluaciones por proceso. Pérdida y gradiente coinciden con tolerancia de redondeo verificada. Evitar la matriz M×d cambia la memoria de ese componente de O(Md) a acumuladores O(n+M); el producto Zᵀ por el gradiente sigue costando O(nd). La pérdida total mantiene el término de regularización original.

[Mediciones de vectorización](_static/entregable3/vectorizacion_mediciones.csv) · [Método y límites](_static/entregable3/vectorizacion_metodo.json) · [27 ajustes del propio](_static/entregable3/complejidad_27_ajustes.csv)
''')]
 chapters['12_regresion_auxiliar']=[md('''# A3. Regresión auxiliar conservada del E2

La tarea auxiliar estima `fasting blood sugar` con 22 predictores del mismo chequeo, excluyendo objetivo y derivados, ID, caries, oral y tartar. Es otra pregunta contemporánea; no convierte la etiqueta binaria de caries en una variable continua ni mide su gravedad.

Se completaron 28 configuraciones y 140 evaluaciones externas, con CV agrupada 5×3, ocho propuestas por búsqueda, 1.120 propuestas y 1.821 ajustes internos nuevos al reutilizar candidatos idénticos. No se consultó el test histórico. El objetivo se escala con media/DE del entrenamiento y las predicciones se devuelven a unidades originales antes de evaluar; la unidad clínica no se confirmó.

XGBoost obtuvo RMSE externo medio de aproximadamente 19,125 frente a 20,722 del predictor de media. Esa mejora no basta para un uso clínico. Los errores grandes de valores extremos y la variación de residuos son visibles.

![Diagnóstico de regresión](_static/cierre_entregable2/reg_residuos.png)

Las sensibilidades principales se justifican como transversales. No se usa un orden artificial de fila para confirmar rendimiento temporal. Las limitaciones y los resultados completos del E2 permanecen en su enlace fechado; el despliegue E3 selecciona clasificación como tarea principal.
''')]
 # Fuentes, código y activos estrictamente necesarios. La entrega E2 no se modifica.
 for name,cells in chapters.items():
  for i,cell in enumerate(cells):cell['id']=name+'-'+str(i)
  nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':cells};(book/(name+'.ipynb')).write_text(json.dumps(nb,indent=1,ensure_ascii=False))
 (book/'_toc.yml').write_text('format: jb-book\nroot: intro\nchapters:\n'+''.join('  - file: '+name+'\n' for name in chapters if name!='intro'))
 (book/'_config.yml').write_text('title: "Caries dental: evaluación y despliegue"\nauthor: "Andrés Duarte Chaves"\ncopyright: "2026"\nexecute:\n  execute_notebooks: "off"\nonly_build_toc_files: true\nexclude_patterns: ["_build", "src/**"]\nbibtex_bibfiles: ["references.bib"]\nhtml:\n  use_repository_button: true\nrepository:\n  url: "https://github.com/aduartech81/MachineLearning"\n  branch: "proyecto-final-independiente-20261018-213747"\nsphinx:\n  config:\n    html_css_files: ["e3.css"]\n')
 shutil.copy2(ROOT/'proyecto_final_caries/cierre_e3/references.bib',book/'references.bib');(book/'src').mkdir(exist_ok=True)
 for file in ['modelo.py','objetivo.py']:shutil.copy2(ROOT/'proyecto_final_caries'/file,book/'src'/file)
 shutil.copy2(ROOT/'proyecto_final_caries/modelo.py',book/'src/original_model.py')
 # Captura propia ya disponible, sin contenidos de otras aplicaciones.
 capture=ROOT.parent.parent/'outputs/dash_regresion_publicado.png'
 if not capture.exists():capture=Path('/Users/andresduarte/Documents/Codex/2026-10-21/continuemos-el-proyecto-final-de-machine/outputs/dash_regresion_publicado.png')
 shutil.copy2(capture,STATIC/'dashboard_publicado.png')
 refs=set()
 for cells in chapters.values():
  for cell in cells:refs.update(re.findall(r'_static/[A-Za-z0-9_./-]+',''.join(cell['source'])))
 for relative in refs:
  src=ROOT/relative
  if src.is_file():dest=book/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
  elif src.is_dir():continue
  else:raise FileNotFoundError(relative)
 shutil.copytree(STATIC,book/'_static/entregable3',dirs_exist_ok=True)
 # Las celdas también leen este CSV para reconstruir la selección.
 dest=book/'_static/cierre_entregable2';dest.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'_static/cierre_entregable2/resumen_108_media_de.csv',dest)
 (book/'_static/e3.css').write_text(':root{--pst-color-primary:#147c86;--pst-color-link:#0c7582}body{color:#193948}.bd-article-container{max-width:1050px!important}h1,h2{font-family:Georgia,serif}h1{line-height:1.15!important}h2{margin-top:2em!important}.e3-hero{display:grid;grid-template-columns:1.1fr .9fr;gap:24px;align-items:center;margin:24px 0;background:#f1f7f7;padding:24px;border-radius:14px}.e3-hero h2{margin-top:0!important}.e3-hero figure{margin:0}.e3-hero img{width:100%;border-radius:12px;aspect-ratio:3/2;object-fit:cover}.e3-hero figcaption{font-size:12px;color:#4b6875}.e3-kicker{font-size:12px;letter-spacing:.1em;font-weight:bold;color:#0e7c86}.e3-button{display:inline-block;background:#0d7783;color:white!important;padding:9px 13px;border-radius:7px;text-decoration:none}.e3-actions{display:flex;flex-wrap:wrap;gap:10px}.e3-button{white-space:nowrap}.e3-secondary{background:white;color:#0d7783!important;border:1px solid #0d7783}table{font-size:13px}td,th{vertical-align:top!important;padding:9px!important}table.dataframe{display:block;overflow-x:auto;width:100%;font-variant-numeric:tabular-nums}p{line-height:1.7}figure figcaption{color:#4b6875}@media(max-width:760px){.e3-hero{grid-template-columns:1fr;padding:16px}.bd-article-container{max-width:100%!important}h1{font-size:2rem!important}}')
 (STATIC/'estado_libro.json').write_text(json.dumps(dict(capitulaciones=len(chapters),fuentes='book_proyecto_final_e3',computo_costoso='Resultados precalculados y código incluido',R1='No acreditado',R3='No demostrado'),indent=2,ensure_ascii=False));print('Fuentes E3 generadas:',len(chapters),'capítulos; activos seleccionados sin duplicar archivo masivo de E2.')
if __name__=='__main__':main()
