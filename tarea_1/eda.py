"""EDA de entrenamiento completo, sin usar test para tomar decisiones."""
import json
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu,chi2_contingency,false_discovery_control
from tarea_1.common import OUT,ROOT,NUM,CAT

def main():
    df=pd.read_csv(OUT/'canonical.csv');train=df.loc[df.split=='train'].copy()
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    stats=train[NUM].describe().T
    stats['median']=train[NUM].median();stats['IQR']=stats['75%']-stats['25%'];stats['skew']=train[NUM].skew()
    stats['tukey_count']=[int(((train[c]<stats.loc[c,'25%']-1.5*stats.loc[c,'IQR']) | (train[c]>stats.loc[c,'75%']+1.5*stats.loc[c,'IQR'])).sum()) for c in NUM]
    stats.to_csv(OUT/'descriptive.csv')
    null=train[NUM+CAT].isna().sum().rename('missing').to_frame();null['percent']=null['missing']/len(train)*100
    null.to_csv(OUT/'missing.csv')
    rows=[];counts=[]
    y=train['default']
    for c in NUM:
        a=train.loc[y==1,c].dropna();b=train.loc[y==0,c].dropna()
        u,p=mannwhitneyu(a,b,method='asymptotic')
        rows.append(dict(variable=c,test='Mann Whitney',p=p,effect=2*u/(len(a)*len(b))-1,effect_type='biserial rangos'))
    for c in CAT:
        tab=pd.crosstab(train[c],y)
        chi,p,_,expected=chi2_contingency(tab,correction=False)
        rows.append(dict(variable=c,test='Chi cuadrado',p=p,effect=np.sqrt(chi/(len(train)*min(tab.shape[0]-1,1))),effect_type='V de Cramer'))
        for level,row in tab.iterrows():counts.append(dict(variable=c,category=level,n=int(row.sum()),positive=int(row[1]),rate=row[1]/row.sum()))
    associations=pd.DataFrame(rows);associations['q_bh']=false_discovery_control(associations.p.to_numpy())
    associations.to_csv(OUT/'associations.csv',index=False);pd.DataFrame(counts).to_csv(OUT/'categories.csv',index=False)
    cat_pairs=[]
    for a,b in combinations(CAT,2):
        tab=pd.crosstab(train[a],train[b]);chi,p,_,_=chi2_contingency(tab,correction=False)
        cat_pairs.append(dict(a=a,b=b,p=p,cramer_v=np.sqrt(chi/(len(train)*min(tab.shape[0]-1,tab.shape[1]-1)))))
    pair_table=pd.DataFrame(cat_pairs);pair_table['q_bh']=false_discovery_control(pair_table.p.to_numpy())
    pair_table.to_csv(OUT/'categorical_pairs.csv',index=False)
    missing_assoc=[]
    for c in NUM:
        flag=train[c].isna()
        if flag.any() and (~flag).any():
            tab=pd.crosstab(flag,y);chi,p,_,_=chi2_contingency(tab,correction=False)
            missing_assoc.append(dict(variable=c,p=p,rate_missing=float(y[flag].mean()),rate_observed=float(y[~flag].mean())))
    pd.DataFrame(missing_assoc).to_csv(OUT/'missing_associations.csv',index=False)
    rates=train[NUM].isna().groupby(train['default']).mean().T*100
    fig,ax=plt.subplots(figsize=(6,5))
    im=ax.imshow(rates.values,aspect='auto',cmap='Blues')
    ax.set_xticks([0,1],['Fully Paid','Charged Off']);ax.set_yticks(range(len(NUM)),NUM)
    ax.set_title('Porcentaje de faltantes por clase · entrenamiento')
    for i in range(len(NUM)):
        for j in range(2):ax.text(j,i,f'{rates.iloc[i,j]:.3f}%',ha='center',va='center')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figdir/'missing_by_class.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(4,2,figsize=(12,15))
    for c,ax in zip(NUM,axs.flat):
        for label,color in [(0,'#237d9b'),(1,'#ce763c')]:
            ax.hist(train.loc[y==label,c].dropna(),bins=60,histtype='step',label=str(label),color=color)
        ax.set_title(c);ax.set_ylabel('Conteo');ax.legend()
    fig.tight_layout();fig.savefig(figdir/'histograms.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(4,2,figsize=(12,15))
    for c,ax in zip(NUM,axs.flat):
        ax.boxplot([train.loc[y==i,c].dropna() for i in [0,1]],tick_labels=['Fully Paid','Charged Off'],showfliers=False)
        ax.set_title(c)
    fig.suptitle('Boxplots: extremos omitidos sólo en la figura; filas conservadas')
    fig.tight_layout();fig.savefig(figdir/'boxplots.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,7));corr=train[NUM].corr()
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(len(NUM)),NUM,rotation=65,ha='right');ax.set_yticks(range(len(NUM)),NUM)
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figdir/'correlation.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(13,13))
    cdf=pd.DataFrame(counts)
    for c,ax in zip(CAT,axs.flat):
        sub=cdf[cdf.variable==c];ax.bar(sub.category.astype(str),sub.rate,color='#237d9b');ax.tick_params(axis='x',rotation=70);ax.set_title(c);ax.set_ylabel('Proporción Charged Off')
    axs.flat[-1].axis('off');fig.tight_layout();fig.savefig(figdir/'categories.png',dpi=140);plt.close(fig)
    print('EDA completo de train guardado',len(train),flush=True)
if __name__=='__main__':main()
