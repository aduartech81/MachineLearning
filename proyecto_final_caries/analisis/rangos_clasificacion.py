"""Comparación exploratoria de siete referencias, sin volver a ajustar modelos."""
from pathlib import Path
import hashlib
import json
import itertools
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, friedmanchisquare, studentized_range
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'proyecto_final_caries/presentacion/evidencias'
SOURCE = BASE / 'revision_v2_20261020_154421_004423/contrastes/auc_por_fold.csv'
NAMES = ['Logistica_L1_L2', 'SVM', 'XGBoost', 'RandomForest', 'DecisionTree', 'KNN', 'GaussianNB']
LABELS = ['Logística L1/L2', 'SVM lineal', 'XGBoost', 'Random Forest', 'Árbol de decisión', 'KNN', 'GaussianNB']


def main():
    df = pd.read_csv(SOURCE, float_precision='round_trip')
    assert df.fold.is_unique and sorted(df.fold.tolist()) == list(range(5))
    x = df[NAMES].to_numpy()
    assert np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all()
    n, k = x.shape
    ranks = rankdata(-x, axis=1, method='average')
    mean = ranks.mean(axis=0)
    result = friedmanchisquare(*x.T)
    tie_correction = 1 - sum(sum(v**3-v for v in np.unique(row, return_counts=True)[1]) for row in x)/(n*(k**3-k))
    q_direct = (12*n/(k*(k+1))*np.sum((mean-(k+1)/2)**2))/tie_correction
    assert np.isclose(q_direct, result.statistic)
    se = np.sqrt(k*(k+1)/(6*n))
    cd = studentized_range.ppf(.95, k, np.inf)/np.sqrt(2)*se
    # Sensibilidad Monte Carlo: permutaciones independientes de etiquetas dentro
    # de cada bloque. No resuelve la dependencia entre folds por ajuste solapado.
    rng = np.random.default_rng(42)
    exceed = 0
    B = 100000
    for _ in range(B//1000):
        perm = np.argsort(rng.random((1000,n,k)),axis=2)
        sums = np.take_along_axis(np.broadcast_to(ranks,(1000,n,k)),perm,axis=2).mean(axis=1)
        q = 12*n/(k*(k+1))*np.sum((sums-(k+1)/2)**2,axis=1)/tie_correction
        exceed += int((q >= result.statistic-1e-12).sum())
    pmc = (exceed+1)/(B+1)
    posthoc = result.pvalue < .05
    out = BASE/'rangos_clasificacion';out.mkdir(exist_ok=True)
    pd.DataFrame(ranks,columns=NAMES).assign(fold=df.fold).to_csv(out/'rangos_por_fold.csv',index=False)
    summary = pd.DataFrame({'modelo':NAMES,'etiqueta':LABELS,'auc_medio':x.mean(axis=0),'rango_medio':mean}).sort_values('rango_medio')
    summary.to_csv(out/'rangos_medios.csv',index=False)
    pairs=[]
    for i,j in itertools.combinations(range(k),2):
        diff=abs(mean[i]-mean[j])
        p=float(studentized_range.sf(diff/se*np.sqrt(2),k,np.inf))
        pairs.append({'modelo_a':NAMES[i],'modelo_b':NAMES[j],'diferencia_rangos':diff,'p_nemenyi_asintotico':p,'supera_cd':bool(diff>cd),'posthoc_habilitado':bool(posthoc)})
    pd.DataFrame(pairs).to_csv(out/'nemenyi.csv',index=False)
    ordered=np.argsort(mean)
    cliques=[]
    for a in range(k):
        for b in range(a+1,k):
            group=set(ordered[a:b+1])
            if mean[ordered[b]]-mean[ordered[a]] <= cd:
                cliques.append(group)
    cliques=[g for g in cliques if not any(g < h for h in cliques)]
    fig,ax=plt.subplots(figsize=(10,6))
    y=np.arange(k)
    ax.scatter(mean[ordered],y,s=90,color='#147d92',zorder=3)
    for rank,yi in zip(mean[ordered],y):ax.annotate(f'{rank:.2f}',(rank,yi),xytext=(8,8),textcoords='offset points',fontsize=10)
    ax.set_yticks(y,[LABELS[i] for i in ordered]);ax.invert_yaxis()
    ax.set_ylim(6.6,-1.8);ax.set_xlim(.7,7.5);ax.set_xticks(range(1,8));ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Rango medio (1 = mayor AUC dentro del fold)')
    ax.set_title('Comparación exploratoria de rangos · siete referencias')
    # Segmentos verticales unen grupos máximos sin diferencias detectadas.
    for c,g in enumerate(cliques):
        ix=[list(ordered).index(i) for i in g]
        ax.plot([7.05+c*.09]*2,[min(ix),max(ix)],color='#d48632',lw=3)
    ax.plot([1,1+cd],[-1.1,-1.1],color='#17384a',lw=2,clip_on=False)
    ax.text(1,-1.35,f'CD de Nemenyi = {cd:.3f} · α = 0,05',fontsize=11)
    fig.text(.02,.015,'Líneas naranjas: grupos sin diferencia detectada; no equivalencia.\nCinco folds con entrenamientos solapados: inferencia exploratoria, no confirmatoria.',fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.94));fig.savefig(out/'diferencia_critica.png',dpi=160);plt.close(fig)
    meta={'familias':NAMES,'n_bloques':n,'k_modelos':k,'metrica':'AUC externo por fold; rango descendente','friedman_chi2':float(result.statistic),'gl':k-1,'p_asintotico':float(result.pvalue),'p_permutacion_sensibilidad':pmc,'permutaciones':B,'semilla':42,'nemenyi_cd':float(cd),'alpha':.05,'posthoc_habilitado':bool(posthoc),'scipy':scipy.__version__,'fuente':str(SOURCE.relative_to(ROOT)),'sha256_fuente':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'limites':['Solo cinco bloques: aproximación chi-cuadrado poco fiable.','Los folds comparten datos de entrenamiento: no son datasets independientes.','La permutación supone intercambiabilidad dentro de bloque y no corrige el entrenamiento solapado.','Selección del análisis posterior a resultados: alcance exploratorio.','Las siete referencias tienen selección interna V2; modelo propio y ablaciones mantienen sus contrastes separados.'],'referencias':['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html','https://www.jmlr.org/papers/v7/demsar06a.html']}
    (out/'metodo.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(json.dumps(meta,ensure_ascii=False,indent=2));print(pd.DataFrame(pairs).query('supera_cd').to_string(index=False))


if __name__ == '__main__':main()
