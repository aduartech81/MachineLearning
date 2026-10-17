from tarea_1.common import prediction_path
"""DeLong por rangos O(n log n), McNemar y bootstrap pareado estratificado."""
from itertools import combinations
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata,norm,binomtest
from sklearn.metrics import roc_auc_score
from tarea_1.common import OUT,NAMES

def delong(y,scores):
    y=np.asarray(y);pos=y==1; m=pos.sum();n=(~pos).sum()
    positive=[];negative=[];aucs=[]
    for score in np.asarray(scores):
        total=rankdata(score,method='average')
        v1=(total[pos]-rankdata(score[pos],method='average'))/n
        v0=1-(total[~pos]-rankdata(score[~pos],method='average'))/m
        positive.append(v1);negative.append(v0);aucs.append(v1.mean())
    cov=np.atleast_2d(np.cov(positive,ddof=1))/m+np.atleast_2d(np.cov(negative,ddof=1))/n
    return np.array(aucs),cov

def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty_like(p)
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def validate():
    y=np.array([1,0,1,0,1,0,0]);s=np.array([[.2,.2,.7,.4,.8,.9,.1],[.3,.4,.4,.1,.8,.9,.1]])
    auc,cov=delong(y,s)
    vp=[];vn=[]
    for score in s:
        a=score[y==1,None];b=score[y==0][None,:]
        h=(a>b).astype(float)+.5*(a==b)
        vp.append(h.mean(1));vn.append(h.mean(0))
    direct=np.cov(vp)/3+np.cov(vn)/4
    assert np.allclose(cov,direct)
    assert np.allclose(auc,[roc_auc_score(y,v) for v in s])
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])
    return dict(delong_direct_covariance_match=True,auc_sklearn_match=True,holm_known_example=True)

def main():
    validation=validate()
    names=[];frames=[]
    for framework in ['sklearn','spark']:
        for model in NAMES:
            path=prediction_path(framework,model)
            if not path.exists():raise FileNotFoundError(path)
            frames.append(pd.read_csv(path).sort_values('row_id'));names.append(f'{framework}_{model}')
    base=frames[0];y=base.actual.to_numpy().astype(int)
    for frame in frames:
        assert frame.row_id.is_unique
        assert np.array_equal(frame.row_id.to_numpy(),base.row_id.to_numpy())
        assert np.array_equal(frame.actual.to_numpy(),y)
    scores=np.array([f.score.to_numpy() for f in frames]);preds=np.array([f.prediction.to_numpy() for f in frames])
    auc,cov=delong(y,scores)
    assert np.allclose(auc,[roc_auc_score(y,s) for s in scores])
    pairs=[('entre_entornos',i,i+6) for i in range(6)]
    pairs += [(framework,i,j) for framework,indices in [('sklearn',range(6)),('spark',range(6,12))] for i,j in combinations(indices,2)]
    rows=[]
    for family,i,j in pairs:
        diff=auc[i]-auc[j];se=np.sqrt(max(0,cov[i,i]+cov[j,j]-2*cov[i,j]))
        z=diff/se if se else (0 if diff==0 else np.sign(diff)*np.inf)
        b=int(((preds[i]==y)&(preds[j]!=y)).sum());c=int(((preds[i]!=y)&(preds[j]==y)).sum())
        rows.append(dict(family=family,a=names[i],b=names[j],auc_a=auc[i],auc_b=auc[j],
            auc_a_low=max(0,auc[i]-1.96*np.sqrt(cov[i,i])),auc_a_high=min(1,auc[i]+1.96*np.sqrt(cov[i,i])),
            auc_b_low=max(0,auc[j]-1.96*np.sqrt(cov[j,j])),auc_b_high=min(1,auc[j]+1.96*np.sqrt(cov[j,j])),
            delta_auc=diff,ci_low=diff-1.96*se,ci_high=diff+1.96*se,z=z,p=2*norm.sf(abs(z)),
            practical_margin_met=bool(abs(diff)>=.005),a_correct_b_wrong=b,a_wrong_b_correct=c,
            mcnemar_p=binomtest(b,b+c,.5).pvalue if b+c else 1.))
    result=pd.DataFrame(rows)
    for _,idx in result.groupby('family').groups.items():
        result.loc[idx,'p_holm']=holm(result.loc[idx,'p'])
        result.loc[idx,'mcnemar_p_holm']=holm(result.loc[idx,'mcnemar_p'])
    result.to_csv(OUT/'paired_tests.csv',index=False)
    # Exact paired stratified empirical bootstrap: same weight per ID for every model.
    repetitions=1000;rng=np.random.default_rng(42);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    sorted_info=[]
    for s in scores:
        order=np.argsort(s,kind='stable');values=s[order]
        starts=np.r_[0,np.flatnonzero(np.diff(values)!=0)+1]
        sorted_info.append((order,starts,y[order]))
    boots=np.empty((repetitions,len(scores)))
    for k in range(repetitions):
        w=np.zeros(len(y),dtype=np.int64)
        w[pos]=rng.multinomial(len(pos),np.full(len(pos),1/len(pos)))
        w[neg]=rng.multinomial(len(neg),np.full(len(neg),1/len(neg)))
        for i,(order,starts,ordered_y) in enumerate(sorted_info):
            weight=w[order]
            positive=np.add.reduceat(weight*ordered_y,starts)
            negative=np.add.reduceat(weight*(1-ordered_y),starts)
            boots[k,i]=np.dot(positive,np.cumsum(negative)-.5*negative)/(len(pos)*len(neg))
        if k%100==0:print('Bootstrap',k,'/',repetitions,flush=True)
    for index,(_,i,j) in enumerate(pairs):
        delta=boots[:,i]-boots[:,j]
        result.loc[index,'bootstrap_low'],result.loc[index,'bootstrap_high']=np.quantile(delta,[.025,.975])
    result.to_csv(OUT/'paired_tests.csv',index=False)
    pd.DataFrame(boots,columns=names).to_csv(OUT/'bootstrap_auc.csv',index=False)
    (OUT/'statistics_validation.json').write_text(json.dumps(dict(**validation,paired_test_ids_verified=True,repetitions=repetitions,seed=42),indent=2))
if __name__=='__main__':main()
