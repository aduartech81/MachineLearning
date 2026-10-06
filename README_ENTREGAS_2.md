# Tarea 2 y Entregable 2 — Andrés Duarte Chaves

## Productos

| Consigna | Implementación |
|---|---|
| Entregable 2: Contexto | Primera pestaña, distribución del target, fuente y diseño |
| EDA | Segunda pestaña, filtros, histogramas, boxplots, asociación por edad y correlaciones |
| Modelo base | Tercera pestaña: regresión logística, Dummy, ROC, PR, calibración y matriz de confusión |
| JBook | `jbook_entregas_2`: interpretaciones y detalles fuera del dashboard |
| Tarea 2: prevención de fuga | Notebook de demostración separado del entrenamiento válido |
| Pipeline + GridSearchCV | Seis clasificadores, ranking por CV, métricas reales de prueba |
| API + Docker | `tarea_2/api.py`, Dockerfile y dependencias |
| Kubernetes | Deployment y Service para Minikube |
| CI | `.github/workflows/nuevas-entregas.yml` |
| Evidently | Reportes train-test y control de deriva explícitamente simulado |

Los notebooks anteriores se conservan sin modificaciones. La Tarea 2 sigue el ejemplo cardíaco del
docente; el Entregable 2 continúa el problema de caries dental del primer entregable. Por ser
clasificación, corresponde regresión logística como modelo base, no SVR de regresión.

## Uso

Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_nuevas.txt
python -m entregable_2.train
python -m tarea_2.train
python -m tarea_2.drift
python build_new_book.py
python -m pytest -q
jupyter-book build jbook_entregas_2 --warningiserror
python -m entregable_2.app
```

Dashboard local: http://127.0.0.1:8050. JBook: `jbook_entregas_2/_build/html/index.html`.
API: `python -m uvicorn tarea_2.api:app --port 8000`; documentación en `/docs`.

Las fuentes públicas, hashes, decisiones y limitaciones se documentan en el JBook. Nunca se generan
datos sustitutos silenciosamente. La copia pública de smoking.csv coincide en conteos con el primer
entregable, pero no se ha demostrado identidad con el CSV original del estudiante.

Las nuevas métricas de caries no reproducen el experimento previo: agrupan duplicados y excluyen
tártaro para el escenario previo al examen oral. No deben sustituirse sin explicación las cifras anteriores.

## Pendientes de publicación y orquestación

- Código publicado y validación remota completada.
- Dashboard: https://machinelearning-caries-andres-duarte.onrender.com
- Informe: https://aduartech81.github.io/MachineLearning/entregas-2/intro.html
- Dashboard desplegado; callbacks de las tres pestañas verificados. Falta revisión visual de filtros y umbral.
- Docker y Minikube ejecutados en el Mac del autor. Respuestas verificadas y guardadas en `tarea_2/results/verificacion_despliegue/respuestas.json`.
- JBook publicado en GitHub Pages.

La sección `ejecucion` del JBook incluye los comandos de Docker/Minikube y configuración de Render.
El archivo de ejemplo README del docente menciona otro proyecto de forecasting; aquí la documentación
se ajusta al proyecto cardíaco real solicitado.
