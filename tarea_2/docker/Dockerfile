FROM python:3.11-slim
WORKDIR /project
COPY tarea_2/docker/requirements.txt /project/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ml_common.py /project/ml_common.py
COPY tarea_2 /project/tarea_2
EXPOSE 8000
CMD ["uvicorn", "tarea_2.api:app", "--host", "0.0.0.0", "--port", "8000"]
