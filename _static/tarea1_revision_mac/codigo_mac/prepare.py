import os,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold
from tarea_1.common import OUT
data_path = os.environ.get("TAREA1_CSV", "accepted_2007_to_2018Q4.csv")
if any((OUT / n).exists() for n in ("canonical.csv", "splits.csv")):
    raise FileExistsError("Ya existen datos preparados; revisar antes de repetir.")
if not os.path.exists(data_path):
    raise FileNotFoundError("Se requiere el CSV real completo de Lending Club; no se generan sustitutos sintéticos.")
num_cols = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_high", "open_acc", "revol_bal", "total_acc"]
cat_cols = ["term", "grade", "home_ownership", "verification_status", "purpose"]
target = "default"
t0 = time.perf_counter()
full = pd.read_csv(data_path, usecols=num_cols+cat_cols+["loan_status"], low_memory=False)
if len(full) < 1_300_000:
    raise ValueError("El archivo no corresponde al dataset completo requerido; revisar procedencia.")
full["row_id"] = np.arange(len(full), dtype=np.int64)
full = full.loc[full.loan_status.isin(["Fully Paid", "Charged Off"])].copy()
full[target] = (full.loan_status == "Charged Off").astype(int)
for col in num_cols:
    full[col] = pd.to_numeric(full[col], errors="coerce")
for col in cat_cols:
    full[col] = full[col].fillna("Missing").astype(str)
train_ids, test_ids = train_test_split(full.index, test_size=.2, stratify=full[target], random_state=42)
full["split"] = np.where(full.index.isin(train_ids), "train", "test")
full["fold"] = -1
train_frame = full.loc[train_ids]
folds = StratifiedKFold(3, shuffle=True, random_state=42)
for k, (_, valid) in enumerate(folds.split(train_frame, train_frame[target])):
    full.loc[train_frame.index[valid], "fold"] = k
OUT.mkdir(parents=True, exist_ok=True)
full[["row_id","split","fold",target]].to_csv(OUT/"splits.csv", index=False)
# Archivo canónico sin transformaciones aprendidas: evita diferencias de parsing entre motores.
full.drop(columns="loan_status").to_csv(OUT/"canonical.csv", index=False)
t_load_pandas = time.perf_counter()-t0
# EDA sólo sobre train; no se submuestrea el entrenamiento supervisado.
df = full.loc[train_ids].copy()
print({"terminal_rows": len(full), "train": len(train_ids), "test": len(test_ids)})
