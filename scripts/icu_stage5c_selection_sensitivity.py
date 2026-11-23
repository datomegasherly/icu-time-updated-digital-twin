#!/usr/bin/env python3
"""Read-only cohort-selection sensitivity: 3 no-temperature models x 2 training cohorts.
Same deterministic group split for all MIMIC cohorts. Evaluates each trained model on
both original restricted and expanded MIMIC holdout and eICU external sets.
This is exploratory, not publication-ready; no patient-level data is exported.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

VITAL=['hr','rr','spo2','map'] # No temperature: eICU units uncertain
LAB=['creatinine','sodium','potassium','bun','glucose','lactate']
STATS=['mean','min','max','last','n']
ARM={'vital':VITAL,'lab':LAB,'integrated':VITAL+LAB}

def hash_holdout(x,seed):
    key=f'{seed}|{x}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'big')%5==0

def sex_numeric(s):
    t=s.astype('string').str.strip().str.lower()
    return t.map({'m':1.,'male':1.,'f':0.,'female':0.}).astype(float)

def design(df,arm):
    columns=['age']+[f'{v}_{s}' for v in ARM[arm] for s in STATS]
    x=df[columns].apply(pd.to_numeric,errors='coerce').copy()
    x['sex_numeric']=sex_numeric(df['sex'])
    return x.replace([np.inf,-np.inf],np.nan)

def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if not np.isfinite(p).all() or not ((0<=p)&(p<=1)).all():raise ValueError('Invalid probabilities')
    edges=np.linspace(0,1,11)
    bins=np.minimum(np.searchsorted(edges,p,side='right')-1,9)
    ece=sum(float((bins==k).mean()*abs(y[bins==k].mean()-p[bins==k].mean())) for k in range(10) if (bins==k).any())
    return {'n':len(y),'events':int(y.sum()),'prevalence':float(y.mean()),'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,'auprc':float(average_precision_score(y,p)) if y.sum() else None,'brier':float(brier_score_loss(y,p)),'ece10':ece,'mean_predicted':float(p.mean())}

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser()
    for tag in ('restricted_mimic','restricted_eicu','expanded_mimic','expanded_eicu'):
        a.add_argument('--'+tag.replace('_','-'),required=True,type=Path)
    a.add_argument('--out',required=True,type=Path)
    a.add_argument('--seed',type=int,default=2026)
    args=a.parse_args()
    inp={tag:pd.read_csv(getattr(args,tag),low_memory=False) for tag in ('restricted_mimic','restricted_eicu','expanded_mimic','expanded_eicu')}
    required={'stay_id','group_id','age','sex','y_24h'}|{f'{v}_{s}' for v in set(VITAL+LAB) for s in STATS}
    for key,d in inp.items():
        if required-set(d.columns):raise ValueError(f'{key} columns missing: {required-set(d.columns)}')
        if d.stay_id.isna().any() or d.stay_id.duplicated().any():raise ValueError(f'{key} duplicate or missing stay ids')
        if d.group_id.isna().any():raise ValueError(f'{key} missing grouping ids')
        if not d.y_24h.isin([0,1]).all():raise ValueError(f'{key} invalid labels')
        if (d.y_24h==1).sum()==0:raise ValueError(f'{key} no outcomes')
    comparison={}
    for source in ('mimic','eicu'):
        small=inp[f'restricted_{source}'].set_index('stay_id')
        big=inp[f'expanded_{source}'].set_index('stay_id')
        if not small.index.isin(big.index).all():raise ValueError(f'{source}: baseline not subset of expanded')
        both=big.reindex(small.index)
        if not (small.y_24h.to_numpy()==both.y_24h.to_numpy()).all():raise ValueError(f'{source}: shared labels disagree')
        if not (small.group_id.astype(str).to_numpy()==both.group_id.astype(str).to_numpy()).all():raise ValueError(f'{source}: shared group ids disagree')
        added=big.loc[~big.index.isin(small.index)]
        if added.y_24h.sum():raise ValueError(f'{source}: newly included positives, inspect before proceeding')
        comparison[source]={'restricted':len(small),'expanded':len(big),'added_negative':len(added),'events':int(big.y_24h.sum()),'same_shared_labels':True}
    # Stable group hash across both cohorts keeps the same individuals in holdout.
    hold={key:np.array([hash_holdout(g,args.seed) for g in d.group_id],dtype=bool) for key,d in inp.items() if key.endswith('mimic')}
    if not set(inp['restricted_mimic'].loc[hold['restricted_mimic'],'stay_id']).issubset(set(inp['expanded_mimic'].loc[hold['expanded_mimic'],'stay_id'])):
        raise ValueError('Holdout membership not stable for shared patients')
    args.out.mkdir(parents=True,exist_ok=True)
    paths={x:args.out/x for x in ('stage5c_metrics.csv','stage5c_delta_on_shared.csv','stage5c_manifest.json')}
    if any(p.exists() for p in paths.values()):raise FileExistsError('Output already exists; choose a new output folder')
    pred={};rows=[]
    for training in ('restricted','expanded'):
        dtrain=inp[training+'_mimic'];idx=~hold[training+'_mimic']
        for arm in ARM:
            fit_x=design(dtrain,arm)
            fit_y=dtrain.y_24h.to_numpy(dtype=int)
            model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler()),('lr',LogisticRegression(C=1.0,solver='liblinear',max_iter=2000,random_state=args.seed))])
            model.fit(fit_x.loc[idx],fit_y[idx])
            for eval_set in ('restricted_mimic','expanded_mimic','restricted_eicu','expanded_eicu'):
                d=inp[eval_set]
                sel=hold[eval_set] if eval_set.endswith('mimic') else np.ones(len(d),dtype=bool)
                p=model.predict_proba(design(d,arm).loc[sel])[:,1]
                y=d.loc[sel,'y_24h'].to_numpy(dtype=int)
                rows.append({'trained_on':training,'model':arm,'evaluated_on':eval_set,'evaluation':'holdout' if eval_set.endswith('mimic') else 'external',**metrics(y,p)})
                if eval_set in ('restricted_eicu','restricted_mimic'):
                    pred[(training,arm,eval_set)]=p
            print('Fitted',training,arm,flush=True)
    # All evaluations on same shared subjects and same y: clean training-definition comparison.
    delta=[]
    for es in ('restricted_mimic','restricted_eicu'):
        yy=inp[es].loc[hold[es] if es.endswith('mimic') else np.ones(len(inp[es]),dtype=bool),'y_24h'].to_numpy(dtype=int)
        for arm in ARM:
            a=metrics(yy,pred[('restricted',arm,es)])
            b=metrics(yy,pred[('expanded',arm,es)])
            delta.append({'evaluated_on':es,'model':arm,'delta_expanded_training_minus_restricted_training_auroc':b['auroc']-a['auroc'],'delta_auprc':b['auprc']-a['auprc'],'delta_brier':b['brier']-a['brier']})
    pd.DataFrame(rows).to_csv(paths['stage5c_metrics.csv'],index=False)
    pd.DataFrame(delta).to_csv(paths['stage5c_delta_on_shared.csv'],index=False)
    manifest={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED','models':'3 logistic regressions without temperature, same preprocessing and deterministic group split','seed':args.seed,'cohort_comparison':comparison,'input_sha256':{k:digest(getattr(args,k)) for k in inp},'train_holdout_counts':{k:{'train_n':int((~hold[k]).sum()),'holdout_n':int(hold[k].sum()),'holdout_events':int(inp[k].loc[hold[k],'y_24h'].sum())} for k in hold},'external_models_never_retrained_on_eicu':True,'warnings':['eICU ICU discharge status/offset is a proxy for death time','eICU group_id is not necessarily a person identifier','No bootstrap confidence intervals in this exploratory phase','No temperature: eICU temperature unit provenance not verified','Do not compare AUPRC/Brier across changing prevalence as causal improvements']}
    paths['stage5c_manifest.json'].write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('STAGE 5C COMPLETED',args.out,flush=True)
    print(json.dumps({'cohorts':comparison,'train_holdout_counts':manifest['train_holdout_counts']},indent=2))

if __name__=='__main__':main()
