#!/usr/bin/env python3
"""Stage 5G: read-only, patient-cluster paired bootstrap for ICU model evaluation.

Fits three prespecified logistic regressions on expanded MIMIC training set only.
No eICU fitting/recalibration. Evaluates internal expanded holdout and external
expanded eICU, with paired bootstrap resampling of whole patient clusters.
Writes aggregate metrics/CIs only; no patient/stay identifiers or predictions.
Status: protocol candidate, not manuscript-approved.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from stage5c_model_helpers import ARM, design, hash_holdout, metrics, digest
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

METRICS=('auroc','auprc','brier')
OUTFILES=('stage5g_metrics.csv','stage5g_paired_deltas.csv','stage5g_cluster_ci.json','stage5g_manifest.json')

def get_eicu_id_map(db):
    try: import duckdb
    except ImportError as exc: raise RuntimeError('Install duckdb with python -m pip install duckdb') from exc
    con=duckdb.connect(str(db),read_only=True)
    try:
        cols={r[0].lower() for r in con.execute('DESCRIBE SELECT * FROM eicu_patient').fetchall()}
        if not {'patientunitstayid','uniquepid'}.issubset(cols):
            raise ValueError('eicu_patient requires patientunitstayid and uniquepid')
        df=con.execute('SELECT patientunitstayid, uniquepid FROM eicu_patient').fetchdf()
    finally: con.close()
    if df.patientunitstayid.isna().any() or df.patientunitstayid.duplicated().any():
        raise ValueError('Invalid duplicate/missing patientunitstayid in raw eICU')
    return df

def cluster_ci(y, pred, group, B, seed):
    """Sample distinct groups with replacement, carrying all group rows together.

    Paired model comparisons use the exact same resampled groups each iteration.
    This quantifies evaluation sampling uncertainty, not training uncertainty.
    """
    y=np.asarray(y,dtype=np.int8)
    group=pd.Series(group,dtype='string').reset_index(drop=True)
    if group.isna().any() or (group.str.strip()=='').any():
        raise ValueError('Missing patient cluster IDs: fail closed')
    labels,codes=np.unique(group.to_numpy(dtype=str),return_inverse=True)
    names=list(pred)
    rg=np.random.default_rng(seed)
    vals={name:{m:[] for m in METRICS} for name in names}
    pairs={'lab_minus_integrated':('lab','integrated'),
           'integrated_minus_vital':('integrated','vital'),
           'lab_minus_vital':('lab','vital')}
    diffs={name:{m:[] for m in METRICS} for name in pairs}
    skipped=0
    for j in range(B):
        drawn=rg.integers(0,len(labels),size=len(labels))
        freq=np.bincount(drawn,minlength=len(labels))
        weight=freq[codes].astype(float)
        if np.sum(weight[y==1])==0 or np.sum(weight[y==0])==0:
            skipped+=1
            continue
        point={}
        for name in names:
            p=pred[name]
            q={'auroc':float(roc_auc_score(y,p,sample_weight=weight)),
               'auprc':float(average_precision_score(y,p,sample_weight=weight)),
               'brier':float(brier_score_loss(y,p,sample_weight=weight))}
            point[name]=q
            for m in METRICS: vals[name][m].append(q[m])
        for label,(a,b) in pairs.items():
            for m in METRICS: diffs[label][m].append(point[a][m]-point[b][m])
        if (j+1)%100==0: print(f'Cluster bootstrap {j+1}/{B}',flush=True)
    valid=B-skipped
    if valid<max(50,int(.90*B)):
        raise ValueError(f'Too many one-class samples: valid {valid}/{B}')
    def summarize(items):
        a=np.asarray(items,dtype=float)
        return {'lower':float(np.quantile(a,.025)),'upper':float(np.quantile(a,.975))}
    return {'resampling_unit':'patient','n_groups':int(len(labels)),
            'n_rows':int(len(y)), 'resamples_requested':B,'resamples_valid':valid,
            'resamples_one_class_skipped':skipped,
            'metrics_ci':{name:{m:summarize(v) for m,v in sub.items()} for name,sub in vals.items()},
            'paired_delta_ci':{name:{m:summarize(v) for m,v in sub.items()} for name,sub in diffs.items()}}

def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--expanded-mimic',type=Path,required=True)
    pa.add_argument('--expanded-eicu',type=Path,required=True)
    pa.add_argument('--db',type=Path,required=True,help='Stage3 DuckDB read-only, provides uniquepid')
    pa.add_argument('--out',type=Path,required=True)
    pa.add_argument('--bootstrap',type=int,default=1000)
    pa.add_argument('--seed',type=int,default=2026)
    args=pa.parse_args()
    if args.bootstrap<100: pa.error('--bootstrap must be >=100')
    for p in (args.expanded_mimic,args.expanded_eicu,args.db):
        if not p.is_file():raise FileNotFoundError(p)
    output=args.out
    if any((output/f).exists() for f in OUTFILES):
        raise FileExistsError('Some output files already exist; choose a fresh --out folder')
    mimic=pd.read_csv(args.expanded_mimic,low_memory=False)
    eicu=pd.read_csv(args.expanded_eicu,low_memory=False)
    for name,d in [('MIMIC',mimic),('eICU',eicu)]:
        required={'stay_id','group_id','age','sex','y_24h'}
        if required-set(d.columns):raise ValueError(f'{name} missing {required-set(d.columns)}')
        if d.stay_id.isna().any() or d.stay_id.duplicated().any():raise ValueError(f'{name} invalid stay ids')
        if d.group_id.isna().any():raise ValueError(f'{name} missing group ids')
        if not d.y_24h.isin([0,1]).all():raise ValueError(f'{name} invalid labels')
    hold=np.array([hash_holdout(z,args.seed) for z in mimic.group_id],dtype=bool)
    if mimic.loc[hold,'y_24h'].nunique()!=2: raise ValueError('MIMIC holdout requires both classes')
    # MIMIC group_id should be a true patient identifier.
    if pd.DataFrame({'group_id':mimic.group_id.to_numpy(), 'hold':hold}).groupby('group_id')['hold'].nunique().gt(1).any():
        raise ValueError('MIMIC patient groups leak across split')
    raw=get_eicu_id_map(args.db)
    # numeric stays are stable in these ICU csvs, but explicit merge validation is required.
    eicu['stay_id_num']=pd.to_numeric(eicu.stay_id,errors='raise').astype('int64')
    raw['patientunitstayid']=pd.to_numeric(raw.patientunitstayid,errors='raise').astype('int64')
    joined=eicu[['stay_id_num']].merge(raw,left_on='stay_id_num',right_on='patientunitstayid',how='left',validate='one_to_one',indicator=True,sort=False)
    if not joined._merge.eq('both').all():raise ValueError('Some eICU stays have no raw patient linkage')
    gid=joined.uniquepid.astype('string').str.strip()
    if gid.isna().any() or gid.eq('').any() or gid.str.lower().isin(['nan','null','none','unknown']).any():
        raise ValueError('Some eICU stays have invalid uniquepid; cannot cluster-bootstrap')
    internal=mimic.loc[hold].reset_index(drop=True)
    external=eicu.reset_index(drop=True)
    predictions={'internal':{},'external':{}}
    for name in ARM:
        model=Pipeline([
            ('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
            ('scale',StandardScaler()),
            ('lr',LogisticRegression(C=1.,solver='liblinear',max_iter=2000,random_state=args.seed)),
        ])
        model.fit(design(mimic,name).loc[~hold],mimic.loc[~hold,'y_24h'].to_numpy(dtype=int))
        predictions['internal'][name]=model.predict_proba(design(internal,name))[:,1]
        predictions['external'][name]=model.predict_proba(design(external,name))[:,1]
        print('Fitted',name,flush=True)
    rows=[]; delta_rows=[]; ci={}
    for ii,(ev,d,clusters) in enumerate([('internal',internal,internal.group_id.astype('string')),
                                          ('external',external,gid)]):
        y=d.y_24h.to_numpy(dtype=int)
        vals={name:metrics(y,p) for name,p in predictions[ev].items()}
        for name,stats in vals.items():
            rows.append({'cohort':ev,'model':name,**stats,
                         'prevalence_brier_reference':float(y.mean()*(1-y.mean())),
                         'brier_skill_vs_prevalence_reference':float(1-stats['brier']/(y.mean()*(1-y.mean())))})
        for label,a,b in [('lab_minus_integrated','lab','integrated'),
                          ('integrated_minus_vital','integrated','vital'),
                          ('lab_minus_vital','lab','vital')]:
            delta_rows.append({'cohort':ev,'comparison':label,**{f'delta_{m}':float(vals[a][m]-vals[b][m]) for m in METRICS}})
        print('Bootstrapping',ev,'clusters',clusters.nunique(),flush=True)
        ci[ev]=cluster_ci(y,predictions[ev],clusters,args.bootstrap,args.seed+ii+300)
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/OUTFILES[0],index=False)
    pd.DataFrame(delta_rows).to_csv(output/OUTFILES[1],index=False)
    (output/OUTFILES[2]).write_text(json.dumps(ci,indent=2),encoding='utf-8')
    manifest={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED',
        'protocol':'Expanded ICU cohort; no-temperature logistic models trained on MIMIC only',
        'clinical_outcome_limitation':'eICU expired unit discharge offset is not independent death timestamp',
        'bootstrap_method':'paired patient-cluster percentile resampling; fitted models fixed',
        'no_external_training_or_recalibration':True,
        'seed':args.seed,'bootstrap':args.bootstrap,
        'mimic_split':'deterministic hash of group_id, stable split (Stage 5C/5D)',
        'input_sha256':{'mimic':digest(args.expanded_mimic),'eicu':digest(args.expanded_eicu)},
        'counts':{'internal_n':len(internal),'internal_events':int(internal.y_24h.sum()),
                  'external_n':len(external),'external_events':int(external.y_24h.sum()),
                  'external_unique_patients':int(gid.nunique())},
        'warnings':['External death timing is proxy-based','No-temperature baseline avoids unverified eICU temperature units',
                    'Cluster bootstrap assesses evaluation uncertainty, not model training uncertainty',
                    'No hospital-level clustering; center heterogeneity requires separate sensitivity analysis',
                    'Exploratory stage; not approved for manuscript use']}
    (output/OUTFILES[3]).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('STAGE 5G COMPLETED — AGGREGATE OUTPUTS ONLY',str(output),flush=True)
    print(json.dumps(manifest['counts'],indent=2))

if __name__=='__main__':main()
