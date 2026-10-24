# Proyecto final: clasificación de caries registrada

Andrés Duarte Chaves · Maestría en Ingeniería Biomédica · Entregable 3, 24 de octubre de 2026.

Libro: https://caries-proyecto-final-andres-duarte.onrender.com/entregas/entregable-3-20261024/

Dashboard: https://caries-proyecto-final-andres-duarte.onrender.com/

Portafolio: https://caries-proyecto-final-andres-duarte.onrender.com/portafolio/

## Alcance y resultados

Clasificación contemporánea de caries registrada con 23 predictores de chequeo. No se estima aparición futura ni se acredita utilidad clínica. El estudio usa agrupación de perfiles idénticos, que no demuestra identidad de pacientes. La comparación pareada suplementaria E3 no acredita superioridad del algoritmo propio (AUC medio 0.605079) frente a las referencias. La novedad científica tampoco está acreditada. Las limitaciones, antecedentes y supuestos estadísticos se documentan en el libro.

E2 conserva 108 configuraciones aplicables de clasificación y 28 de regresión auxiliar. E3 añade comparación pareada de nueve familias, tres bases públicas adicionales, pruebas de robustez y recursos, y ocho artefactos finales con metadatos. No se mezclan las métricas de ambos objetivos.

## Reproducción

Desde la raíz del repositorio, Python 3.11. Instalar `proyecto_final_caries/requirements_investigacion_e3.txt` en un entorno independiente. Los CSV, parámetros, semillas, predicciones y métodos acompañan los resultados; véanse los capítulos 6 y 9 del libro para el diseño y comandos de cada experimento. Las ejecuciones completas pueden tardar horas. Los notebooks del libro no vuelven a entrenar automáticamente los experimentos históricos.

Los artefactos finales utilizados por el servicio están en `artefactos_finales_20261023_v3`; `artifacts/<familia>` contiene copias verificadas y metadatos detallados. Las dependencias exclusivas de inferencia están fijadas en `requirements_inferencia.txt`.

Construir el libro con:

```sh
jupyter-book build book_proyecto_final_e3 --warningiserror
```

Preparar sus rutas para el servidor con:

```sh
python -m proyecto_final_caries.utilidades.empaquetar_libro_render --html book_proyecto_final_e3/_build/html --destino proyecto_final_caries/presentacion/documentacion/entregable3_20261024
```

## Servicio web

La publicación existente usa la rama `proyecto-final-independiente-20261018-213747`. Instalación: `pip install -r proyecto_final_caries/requirements_inferencia.txt`. Inicio:

```sh
CARIES_HABILITAR_PREDICCION=1 gunicorn proyecto_final_caries.presentacion.app:server --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

La inferencia no entrena. `/predecir/` valida categorías y rangos observados. Sus valores iniciales son un perfil sintético. No hay umbral clínico ni nueva calibración final. El enlace fechado del E2 conserva la versión enviada. La preparación privada para la sustentación no forma parte del sitio ni del repositorio.
