"""EDA en perfiles únicos de entrenamiento; ejecutar después de train.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, false_discovery_control
from sklearn.ensemble import IsolationForest
from ml_common import preprocessor
from entregable_2.train import CATEGORICAL

ROOT = Path(__file__).parent / 'results'

def main():
    train = pd.read_csv(ROOT/'train.csv')
    features = json.loads((ROOT/'summary.json').read_text())['features']
    data = train.drop_duplicates(subset=features)
    rows, counts = [], []
    y = data['dental caries']
    for col in features:
        if col in CATEGORICAL:
            table = pd.crosstab(data[col], y).reindex(columns=[0, 1], fill_value=0)
            chi, p, _, expected = chi2_contingency(table, correction=False)
            effect = np.sqrt(chi/(table.values.sum()*min(table.shape[0]-1, 1)))
            rows.append(dict(variable=col, test='Chi cuadrado', p=p,
                             effect=effect, effect_name='V de Cramer',
                             cells_expected_lt5=int((expected<5).sum())))
            for level, row in table.iterrows():
                counts.append(dict(variable=col, category=level, n=int(row.sum()),
                                   positive=int(row[1]), prevalence=float(row[1]/row.sum())))
        else:
            a,b=data.loc[y==1,col].dropna(), data.loc[y==0,col].dropna()
            u,p=mannwhitneyu(a,b,alternative='two-sided',method='asymptotic')
            rows.append(dict(variable=col,test='Mann Whitney',p=p,
                             effect=2*u/(len(a)*len(b))-1,effect_name='Correlación biserial de rangos',
                             cells_expected_lt5=0))
    result=pd.DataFrame(rows)
    result['q_bh']=false_discovery_control(result.p.to_numpy(), method='bh')
    result.to_csv(ROOT/'associations.csv',index=False)
    pd.DataFrame(counts).to_csv(ROOT/'categorical_eda.csv',index=False)
    transformed=preprocessor(data[features],CATEGORICAL).fit_transform(data[features])
    isolation=IsolationForest(contamination=0.05,random_state=42,n_jobs=1)
    flags=isolation.fit_predict(transformed)==-1
    pd.DataFrame({'ID':data.ID,'flag':flags,'score':isolation.decision_function(transformed)}).to_csv(ROOT/'outliers.csv',index=False)
    (ROOT/'eda_summary.json').write_text(json.dumps(dict(profiles=len(data),
        flagged=int(flags.sum()),contamination=0.05,
        action='Se conservan todos los registros; el 5% es una decisión exploratoria, no una prevalencia clínica.'),indent=2))

if __name__=='__main__':
    main()
