#!/usr/bin/env python3
"""Exploratory, read-only 3-arm × temperature sensitivity study.
Training on MIMIC Stage 3C; independent holdout and frozen eICU external prediction.
No raw records or patient-level predictions are exported.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

VITAL=['hr','rr','spo2','map','temp']
LAB=['creatinine','sodium','potassium','bun','glucose','lactate']
STAT=['mean','min','max','last','n']

def sex_numeric(x):
    s=x.astype('string').str.strip().str.casefold()
    return s.map({'m':1.0,'male':1.0,'f':0.0,'female':0.0}).astype(float)

def prep(df,columns):
    x=df[[c for c in columns if c != 'sex_numeric']].copy()
    x['sex_numeric']=sex_numeric(df['sex'])
    x=x[columns]
    for col in columns:
        if col!='sex_numeric': x[col]=pd.to_numeric(x[col],errors='coerce')
    return x

def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1)
    b=np.minimum(np.searchsorted(edges,p,side='right')-1,bins-1)
    return float(sum((b==k).mean()*abs(y[b==k].mean()-p[b==k].mean()) for k in range(bins) if (b==k).any()))

def calc(y,p):
    return {'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'auprc':float(average_precision_score(y,p)) if y.sum()>0 else None,
            'brier':float(brier_score_loss(y,p)), 'ece10':ece(y,p),
            'mean_predicted':float(np.mean(p)), 'observed_prevalence':float(np.mean(y)),
            'risk_over_50pct':int((p>0.5).sum())}

def ci_by_bootstrap(y, preds, n, seed):
    rng=np.random.default_rng(seed)
    names=list(preds)
    store={k:{'auroc':[],'auprc':[],'brier':[]} for k in names}
    nrows=len(y)
    for _ in range(n):
        ix=rng.integers(0,nrows,nrows)
        yy=y[ix]
        if yy.sum()==0 or yy.sum()==len(yy):continue
        for k in names:
            pp=preds[k][ix]
            store[k]['auroc'].append(roc_auc_score(yy,pp))
            store[k]['auprc'].append(average_precision_score(yy,pp))
            store[k]['brier'].append(brier_score_loss(yy,pp))
    out={}
    for k in names:
        out[k]={metric: {'lower':float(np.percentile(v,2.5)),'upper':float(np.percentile(v,97.5))}
                for metric,v in store[k].items() if v}
    # Paired delta bootstrap: without temperature minus with temperature.
    changes={arm:{'auroc':[],'auprc':[],'brier':[]} for arm in ('vital','integrated')}
    rng=np.random.default_rng(seed)  # reset to identical resample indices as individual CIs
    for _ in range(n):
        ix=rng.integers(0,nrows,nrows)
        yy=y[ix]
        if yy.sum()==0 or yy.sum()==len(yy):continue
        for arm in changes:
            a=preds[f'with_temp:{arm}'][ix]
            b=preds[f'without_temp:{arm}'][ix]
            changes[arm]['auroc'].append(roc_auc_score(yy,b)-roc_auc_score(yy,a))
            changes[arm]['auprc'].append(average_precision_score(yy,b)-average_precision_score(yy,a))
            changes[arm]['brier'].append(brier_score_loss(yy,b)-brier_score_loss(yy,a))
    out['paired_differences_without_minus_with']={arm:{metric:{'lower':float(np.percentile(values,2.5)),'upper':float(np.percentile(values,97.5))}
        for metric,values in metric_vals.items() if values} for arm,metric_vals in changes.items()}
    return out

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(2**20),b''):h.update(buf)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mimic',type=Path,required=True)
    ap.add_argument('--eicu',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--bootstrap',type=int,default=300)
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    if args.bootstrap<0:ap.error('--bootstrap cannot be negative')
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    for fname in ['stage4_metrics.csv','stage4_bootstrap_ci.json','stage4_run_manifest.json','stage4_temperature_effect.csv']:
        if (out/fname).exists():raise FileExistsError(f'Refusing to overwrite {out/fname}')
    m=pd.read_csv(args.mimic,low_memory=False)
    e=pd.read_csv(args.eicu,low_memory=False)
    needed={'stay_id','group_id','age','sex','y_24h'}|{f'{v}_{s}' for v in VITAL+LAB for s in STAT}
    for label,d in [('mimic',m),('eicu',e)]:
        if needed-set(d.columns):raise ValueError(f'{label} missing columns: {sorted(needed-set(d.columns))}')
        if d.stay_id.duplicated().any():raise ValueError(f'{label}: duplicate stay_id')
        if d.y_24h.isna().any() or not d.y_24h.isin([0,1]).all():raise ValueError(f'{label}: invalid outcomes')
    m_y=m.y_24h.to_numpy(dtype=int); e_y=e.y_24h.to_numpy(dtype=int)
    train_ix,hold_ix=next(GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=args.seed).split(m,m_y,groups=m.group_id))
    if np.unique(m_y[train_ix]).size<2 or np.unique(m_y[hold_ix]).size<2:raise ValueError('One-class train/holdout split')
    preds={'internal':{},'external':{}}
    rows=[]
    for vname,vt in [('with_temp',VITAL),('without_temp',[x for x in VITAL if x!='temp'])]:
        for arm,vars in [('vital',vt),('lab',LAB),('integrated',vt+LAB)]:
            cols=['age','sex_numeric']+[f'{v}_{s}' for v in vars for s in STAT]
            Xm=prep(m,cols); Xe=prep(e,cols)
            # Indicator constructed exclusively from training missingness; frozen transformer reused externally.
            pipe=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                ('scaler',StandardScaler()),
                ('model',LogisticRegression(C=1.0,solver='liblinear',max_iter=2000,random_state=args.seed))])
            pipe.fit(Xm.iloc[train_ix],m_y[train_ix])
            for cohort, X, yy in [('internal',Xm.iloc[hold_ix],m_y[hold_ix]),('external',Xe,e_y)]:
                prob=pipe.predict_proba(X)[:,1]
                key=f'{vname}:{arm}'
                preds[cohort][key]=prob
                rows.append({'cohort':cohort,'temperature_arm':vname,'model':arm,'n':int(len(yy)),'events':int(yy.sum()),**calc(yy,prob)})
            print(f'Finished {vname} / {arm}',flush=True)
    pd.DataFrame(rows).to_csv(out/'stage4_metrics.csv',index=False)
    differences=[]
    for cohort in ('internal','external'):
        for arm in ('vital','lab','integrated'):
            a=next(r for r in rows if r['cohort']==cohort and r['temperature_arm']=='with_temp' and r['model']==arm)
            b=next(r for r in rows if r['cohort']==cohort and r['temperature_arm']=='without_temp' and r['model']==arm)
            differences.append({'cohort':cohort,'model':arm,**{f'delta_without_minus_with_{metric}':b[metric]-a[metric] for metric in ['auroc','auprc','brier','ece10']}})
    pd.DataFrame(differences).to_csv(out/'stage4_temperature_effect.csv',index=False)
    intervals={}
    if args.bootstrap:
        for c,yy in [('internal',m_y[hold_ix]),('external',e_y)]:
            print(f'Paired bootstrap: {c}, {args.bootstrap} samples',flush=True)
            intervals[c]=ci_by_bootstrap(yy,preds[c],args.bootstrap,args.seed+({'internal':1,'external':2}[c]))
    (out/'stage4_bootstrap_ci.json').write_text(json.dumps(intervals,indent=2),encoding='utf-8')
    manifest={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED',
        'comparison':'6 fixed designs: vital, lab, integrated × with/without temperature',
        'note':'Lab-only models are identical in both temperature arms by design',
        'outcome_warning':'eICU ICU discharge Expired/offset is a proxy, not independently verified mortality time',
        'unit_warning':'eICU temperature uses numerical band heuristic, not source-confirmed units',
        'selection_warning':'Cohort requires final clinical eligibility and censoring review',
        'mimic_sha256':sha(args.mimic),'eicu_sha256':sha(args.eicu),
        'seed':args.seed,'bootstrap':args.bootstrap,'train_n':int(len(train_ix)),
        'holdout_n':int(len(hold_ix)),'holdout_events':int(m_y[hold_ix].sum()),
        'external_n':int(len(e_y)),'external_events':int(e_y.sum()),
        'same_split_for_all_arms':True,'no_patient_level_export':True,'external_retraining':False}
    (out/'stage4_run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('STAGE 4 EXPLORATORY COMPLETED:',out,flush=True)
    print(json.dumps({k:manifest[k] for k in ('train_n','holdout_n','holdout_events','external_n','external_events')},indent=2))

if __name__=='__main__':main()
