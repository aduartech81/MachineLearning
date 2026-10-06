# Ejecución y estado de entrega

## Preparar el entorno

Desde la raíz del repositorio, con Python 3.11 o 3.12:

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
```

Los entrenamientos requieren los CSV incluidos y nunca generan sustitutos sintéticos cuando falta un archivo.
Los notebooks también se pueden abrir en Jupyter; localizan la raíz del proyecto y usan sus módulos.
`requirements_lock.txt` registra las versiones de los paquetes principales que se usaron en esta ejecución.

## Abrir dashboard y API

```bash
python -m entregable_2.app
```

Abrir `http://127.0.0.1:8050`. Para la API en otra terminal:

```bash
python -m uvicorn tarea_2.api:app --host 127.0.0.1 --port 8000
```

Abrir `http://127.0.0.1:8000/docs` para probar `/predict` con el esquema mostrado.

## Docker y Kubernetes local

```bash
docker build -t heart-api:local -f tarea_2/docker/Dockerfile .
docker run --rm -p 8000:8000 heart-api:local
minikube start
minikube image load heart-api:local
kubectl apply -f tarea_2/k8s/deployment.yaml
kubectl apply -f tarea_2/k8s/service.yaml
kubectl rollout status deployment/heart-model
minikube service heart-service --url
```

Verificar `/health` y `/predict` en la URL obtenida. No se dieron por realizadas estas pruebas.

## Publicar Dash con Render

La referencia del docente usa Render. Desde Render, conectar este repositorio y crear el servicio
usando `render.yaml`, o configurar:

- Build: `pip install -r requirements_dashboard.txt`
- Start: `gunicorn entregable_2.app:server --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
- Root Directory: raíz del repositorio.

Los resultados deben estar versionados antes del despliegue. Después verificar las tres pestañas,
filtros y control de umbral en la URL pública. Dashboard publicado: https://machinelearning-caries-andres-duarte.onrender.com. Los endpoints y callbacks de las tres pestañas respondieron correctamente; falta comprobar visualmente filtros y umbral.
GitHub Pages puede alojar el JBook estático, pero no ejecutar el servidor Python de Dash.

## Estado comprobado

| Elemento | Estado |
|---|---|
| Entrenamiento de ambos productos | Ejecutado |
| Datos, hashes y predicciones | Guardados |
| Dashboard: endpoints y callbacks | Probados localmente |
| API: predicción y validación | Probadas localmente |
| Evidently: dos reportes | Generados |
| Docker / Minikube | Archivos preparados; ejecución pendiente |
| GitHub Actions | Validación remota completada con éxito |
| Render y publicación JBook | Dashboard y JBook publicados |

## Verificación de notebooks y límite de vista previa

Se verificaron las celdas de los tres notebooks en namespaces Python nuevos, ejecutando su código y guardando salidas. No se usó un kernel Jupyter porque este entorno bloqueó sus sockets. Las pruebas de Flask/Dash y FastAPI pasaron; no se completó una inspección visual en navegador porque el navegador requerido no pudo descargarse.
