"""Recrea ROC/PR desde predicciones y curvas conservadas; no entrena modelos."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'proyecto_final_caries/presentacion/evidencias'
PACK = DATA / 'revision_v2_20261020_154421_004423'
source = ROOT / 'proyecto_final_caries/resultados/analisis_20261018_204847_425225/oof_Propio.csv'

def main():
    d = pd.read_csv(source, float_precision='round_trip')
    assert len(d) == 44582 and d.row_id.is_unique
    assert np.isfinite(d.score).all() and set(d.actual.unique()) == {0, 1}
    curves = pd.read_csv(PACK / 'evaluacion/curvas.csv')
    metrics = pd.read_csv(PACK / 'evaluacion/metricas_oof_agrupadas.csv').drop_duplicates('familia')
    labels = {'Logistica_L1_L2':'Regresión logística','SVM':'SVM lineal','XGBoost':'XGBoost',
              'RandomForest':'Bosque aleatorio','DecisionTree':'Árbol de decisión','KNN':'KNN','GaussianNB':'Naive Bayes gaussiano'}
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    own_auc = roc_auc_score(d.actual,d.score)
    own_ap = average_precision_score(d.actual,d.score)
    for kind in ['ROC','PR']:
        fig, ax = plt.subplots(figsize=(10,6.3), layout='constrained')
        for r in metrics.itertuples():
            c = curves.loc[(curves.familia == r.familia) & (curves.curva == kind)]
            assert len(c)>0, (kind,r.familia)
            metric = r.auc_oof_agrupado if kind=='ROC' else r.average_precision_oof
            ax.plot(c.x,c.y,lw=1.2,alpha=.65,label=f'{labels[r.familia]} · {metric:.3f}')
        if kind=='ROC':
            x,y,_=roc_curve(d.actual,d.score)
            ax.plot([0,1],[0,1],ls='--',color='#777',lw=1)
            ax.set(xlabel='1 − especificidad',ylabel='Sensibilidad',title='ROC · modelo propio y referencias')
            metric=own_auc
        else:
            y,x,_=precision_recall_curve(d.actual,d.score)
            ax.axhline(d.actual.mean(),ls='--',color='#777',lw=1,label=f'Prevalencia · {d.actual.mean():.3f}')
            ax.set(xlabel='Sensibilidad',ylabel='Valor predictivo positivo',title='Precisión–recall · modelo propio y referencias')
            metric=own_ap
        ax.plot(x,y,color='#b76524',lw=2.8,label=f'Modelo propio (V1 conservado) · {metric:.3f}',zorder=10)
        ax.set(xlim=(0,1),ylim=(0,1));ax.grid(alpha=.15)
        ax.legend(fontsize=8,loc='lower right' if kind=='ROC' else 'upper right',title='AUC OOF' if kind=='ROC' else 'Precisión promedio (AP)')
        fig.savefig(DATA / f'{kind.lower()}_propio_referencias.png',dpi=160)
        plt.close(fig)
    paths=[source, PACK/'evaluacion/curvas.csv',PACK/'evaluacion/metricas_oof_agrupadas.csv']
    (DATA/'metodo_figuras_comparacion.json').write_text(json.dumps({
        'fuentes_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'auc_oof_propio':own_auc,'ap_oof_propio':own_ap,'filas':len(d),
        'alcance':'Figuras descriptivas de predicciones OOF conservadas. Modelo propio V1 y referencias V2; no se recalibró el propio. Inferencia: AUC medio por fold en los contrastes existentes.',
        'entrenamiento':False
    },ensure_ascii=False,indent=2))
    print('ROC y PR generadas sin entrenamiento:',own_auc,own_ap)

if __name__=='__main__':main()
