#!/usr/bin/env python3
"""Stage 6C exploratory patient-cluster paired count-feature ablation.

Reads expanded MIMIC/eICU CSVs, trains on MIMIC only, validates unchanged external
cohort, and reports aggregate metrics/paired bootstrap uncertainty. Does not edit inputs.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from stage5c_model_helpers import ARM,design,hash_holdout,metrics,digest

OUT=['stage6c_metrics.csv','stage6c_paired_effect.csv','stage6c_cluster_ci.json','stage6c_manifest.json']

def get_groups(db,stay_ids):
    try: import duckdb
    except ImportError as e: raise RuntimeError('Install duckdb: python -m pip install duckdb') from e
    con=duckdb.connect(str(db),read_only=True)
    try:
        raw=con.execute('SELECT patientunitstayid, uniquepid FROM eicu_patient').fetchdf()
    finally: con.close()
    if raw.patientunitstayid.isna().any() or raw.patientunitstayid.duplicated().any():
        raise ValueError('eICU raw stay IDs duplicated/null')
    src=pd.DataFrame({'stay_id':pd.to_numeric(stay_ids,errors='raise').astype('int64')})
    raw['patientunitstayid']=pd.to_numeric(raw.patientunitstayid,errors='raise').astype('int64')
    link=src.merge(raw,left_on='stay_id',right_on='patientunitstayid',how='left',validate='one_to_one',sort=False,indicator=True)
    groups=link.uniquepid.astype('string').str.strip()
    if not link._merge.eq('both').all() or groups.isna().any() or groups.eq('').any():
        raise ValueError('Missing patient cluster linkage')
    return groups.to_numpy(dtype=str)

def design_variant(data,arm,variant):
    x=design(data,arm)
    if variant=='drop_all_counts':
        x=x.drop(columns=[c for c in x.columns if c.endswith('_n')])
    return x

def sampling_weighted_metric(y,p,w,m):
    if m=='auroc': return float(roc_auc_score(y,p,sample_weight=w))
    if m=='auprc': return float(average_precision_score(y,p,sample_weight=w))
    if m=='brier': return float(brier_score_loss(y,p,sample_weight=w))
    raise ValueError(m)

def paired_ci(y,preds,groups,b,seed):
    # Vectorize group membership; share cluster draws across all model arms and ablations.
    _,inverse=np.unique(np.asarray(groups,dtype=str),return_inverse=True)
    ncluster=int(inverse.max())+1
    rng=np.random.default_rng(seed)
    contrasts=[(f'{arm}:drop_all_counts_minus_baseline',f'{arm}:drop_all_counts',f'{arm}:baseline') for arm in ARM]
    vals={label:{m:[] for m in ('auroc','auprc','brier')} for label,_,_ in contrasts}
    skipped=0
    for i in range(b):
        draws=rng.integers(0,ncluster,size=ncluster)
        freqs=np.bincount(draws,minlength=ncluster)
        weight=freqs[inverse].astype(float)
        if not np.any(weight[y==1]) or not np.any(weight[y==0]):
            skipped+=1; continue
        cache={}
        for _,a,baseline in contrasts:
            for name in (a,baseline):
                if name not in cache:
                    cache[name]={m:sampling_weighted_metric(y,preds[name],weight,m) for m in ('auroc','auprc','brier')}
        for label,a,baseline in contrasts:
            for m in ('auroc','auprc','brier'):
                vals[label][m].append(cache[a][m]-cache[baseline][m])
        if (i+1)%100==0: print(f'  cluster bootstrap {i+1}/{b}',flush=True)
    if b-skipped < max(30,int(.90*b)): raise RuntimeError('Too many invalid resamples')
    ci={label:{m:{'lower':float(np.quantile(v,.025)), 'upper':float(np.quantile(v,.975))} for m,v in sub.items()} for label,sub in vals.items()}
    return {'n_clusters':ncluster,'iterations_requested':b,'iterations_valid':b-skipped,'skipped':skipped,'paired_delta_ci':ci}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for a in ('expanded_mimic','expanded_eicu','db','stage6a_metrics','out'):
        ap.add_argument('--'+a.replace('_','-'),required=True,type=Path)
    ap.add_argument('--bootstrap',default=300,type=int)
    ap.add_argument('--seed',default=2026,type=int)
    a=ap.parse_args()
    if a.bootstrap<100: ap.error('--bootstrap must be >= 100')
    for x in (a.expanded_mimic,a.expanded_eicu,a.db,a.stage6a_metrics):
        if not x.is_file(): raise FileNotFoundError(x)
    if any((a.out/f).exists() for f in OUT):raise FileExistsError('Choose an unused --out')
    mimic=pd.read_csv(a.expanded_mimic,low_memory=False)
    eicu=pd.read_csv(a.expanded_eicu,low_memory=False)
    for name,d in [('mimic',mimic),('eicu',eicu)]:
        required={'stay_id','group_id','sex','age','y_24h'}
        if required-set(d.columns):raise ValueError(f'{name} missing columns {required-set(d.columns)}')
        if d.stay_id.isna().any() or d.stay_id.duplicated().any():raise ValueError(f'{name} invalid stay IDs')
        if d.group_id.isna().any() or not d.y_24h.isin((0,1)).all():raise ValueError(f'{name} missing groups/invalid outcome')
    hold=np.array([hash_holdout(z,a.seed) for z in mimic.group_id],dtype=bool)
    train=mimic.loc[~hold].reset_index(drop=True)
    internal=mimic.loc[hold].reset_index(drop=True)
    if train.y_24h.nunique()!=2 or internal.y_24h.nunique()!=2 or eicu.y_24h.nunique()!=2:
        raise ValueError('Need both classes in all cohorts')
    if pd.DataFrame({'group':mimic.group_id,'hold':hold}).groupby('group').hold.nunique().gt(1).any():
        raise ValueError('Patient leakage across MIMIC split')
    reference=pd.read_csv(a.stage6a_metrics)
    if not {'cohort','model','auroc','auprc','brier'}.issubset(reference.columns):
        raise ValueError('Invalid Stage6A metrics reference')
    cohorts={'internal':internal,'external':eicu}
    predictions={name:{} for name in cohorts}
    output=[];checks=[]
    for arm in ARM:
        for variant in ('baseline','drop_all_counts'):
            x=design_variant(train,arm,variant)
            xx={cohort:design_variant(d,arm,variant) for cohort,d in cohorts.items()}
            if not x.columns.tolist()==xx['external'].columns.tolist():raise ValueError('Input column mismatch')
            model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                            ('scale',StandardScaler()),
                            ('lr',LogisticRegression(C=1.0,solver='liblinear',max_iter=2000,random_state=a.seed))])
            model.fit(x,train.y_24h.to_numpy(dtype=int))
            for cohort,d in cohorts.items():
                y=d.y_24h.to_numpy(dtype=int)
                p=model.predict_proba(xx[cohort])[:,1]
                predictions[cohort][f'{arm}:{variant}']=p
                met=metrics(y,p)
                output.append({'cohort':cohort,'model':arm,'variant':variant,'n_features_raw':x.shape[1],**met,
                               'n_pred_gt50':int((p>.5).sum()),'pct_pred_gt50':float((p>.5).mean()*100),
                               'observed_event_rate_gt50':float(y[p>.5].mean()) if (p>.5).any() else None})
                if variant=='baseline':
                    ref=reference[(reference.cohort==cohort)&(reference.model==arm)]
                    if len(ref)!=1:raise ValueError(f'Missing/duplicate Stage6A reference {cohort}/{arm}')
                    mx=max(abs(met[m]-float(ref.iloc[0][m])) for m in ('auroc','auprc','brier'))
                    if mx>1e-5:raise RuntimeError(f'FAIL CLOSED: Stage6A mismatch {cohort}/{arm}: {mx}')
                    checks.append({'cohort':cohort,'model':arm,'max_diff':mx})
            print('Fitted and validated',arm,variant,flush=True)
    effect=[]
    for cohort in cohorts:
        d=cohorts[cohort]
        y=d.y_24h.to_numpy(dtype=int)
        for arm in ARM:
            p0=predictions[cohort][f'{arm}:baseline'];p1=predictions[cohort][f'{arm}:drop_all_counts']
            m0=metrics(y,p0);m1=metrics(y,p1)
            effect.append({'cohort':cohort,'model':arm,'comparison':'drop_all_counts_minus_baseline',
                           **{f'delta_{m}':m1[m]-m0[m] for m in ('auroc','auprc','brier','ece10','mean_predicted')},
                           'delta_n_pred_gt50':int((p1>.5).sum()-(p0>.5).sum())})
    eicu_groups=get_groups(a.db,eicu.stay_id)
    ci={}
    for name,groups in [('internal',internal.group_id.astype('string').to_numpy(dtype=str)),('external',eicu_groups)]:
        print('Paired cluster bootstrap',name,flush=True)
        ci[name]=paired_ci(cohorts[name].y_24h.to_numpy(dtype=int),predictions[name],groups,a.bootstrap,a.seed+(10 if name=='internal' else 20))
    a.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(output).to_csv(a.out/OUT[0],index=False)
    pd.DataFrame(effect).to_csv(a.out/OUT[1],index=False)
    (a.out/OUT[2]).write_text(json.dumps(ci,indent=2),encoding='utf-8')
    manifest={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED',
       'hypothesis':'Removing measurement counts changes transfer performance in a fixed cohort',
       'variants':{'baseline':'All non-temperature feature summaries including _n',
                   'drop_all_counts':'Remove every raw feature ending _n before preprocessing; retain other statistics and covariates'},
       'seed':a.seed,'bootstrap':a.bootstrap,
       'model':'liblinear logistic regression C=1, 2000 iterations, train-median impute+missing indicators+scaling',
       'model_refits_between_variants':True,'no_external_training_or_recalibration':True,
       'patient_cluster_bootstrap':'Fixed fitted predictions, pairs use same sampled patient groups',
       'input_sha256':{'mimic':digest(a.expanded_mimic),'eicu':digest(a.expanded_eicu),'db':None,'reference':digest(a.stage6a_metrics)},
       'counts':{'train':len(train),'holdout':len(internal),'holdout_events':int(internal.y_24h.sum()),
                 'external':len(eicu),'external_events':int(eicu.y_24h.sum()),'external_unique_patients':len(np.unique(eicu_groups))},
       'baseline_reproduction':checks,
       'limitations':['Nonrandomized ablation is not proof of causal recording effects',
                      'Bootstrap is conditional on fitted models and not training variability',
                      'eICU event time uses ICU discharge proxy',
                      'Hospital-level clustering not included; do not infer deployment readiness',
                      'This is exploratory; do not replace manuscript numbers']}
    (a.out/OUT[3]).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('STAGE 6C COMPLETE — exploratory aggregate outputs only',a.out,flush=True)
    print(json.dumps(manifest['counts'],indent=2),flush=True)

if __name__=='__main__': main()
