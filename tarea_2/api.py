"""API con esquema explícito, modelo completo y validación de entrada."""
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).parent/'results/model.joblib'
model = joblib.load(MODEL_PATH)
app = FastAPI(title='Enfermedad cardíaca — Tarea 2', version='1.0')


class Patient(BaseModel):
    model_config = ConfigDict(extra='forbid')
    Age: int = Field(ge=18, le=120)
    Sex: Literal['M', 'F']
    ChestPainType: Literal['ATA', 'NAP', 'ASY', 'TA']
    RestingBP: float | None = Field(default=None, gt=0, le=300)
    Cholesterol: float | None = Field(default=None, gt=0, le=1000)
    FastingBS: Literal[0, 1]
    RestingECG: Literal['Normal', 'ST', 'LVH']
    MaxHR: float = Field(gt=0, le=250)
    ExerciseAngina: Literal['Y', 'N']
    Oldpeak: float = Field(ge=-5, le=10)
    ST_Slope: Literal['Up', 'Flat', 'Down']


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': True}


@app.post('/predict')
def predict(patient: Patient):
    row = pd.DataFrame([patient.model_dump()]).reindex(columns=model.feature_names_in_)
    for c in ['RestingBP', 'Cholesterol']:
        row[c] = pd.to_numeric(row[c], errors='coerce')
    probability = float(model.predict_proba(row)[0, 1])
    return {'heart_disease_probability': probability,
            'prediction': int(probability >= 0.5), 'threshold': 0.5,
            'scope': 'Clasificación académica de enfermedad cardíaca registrada; no diagnóstico clínico.'}
