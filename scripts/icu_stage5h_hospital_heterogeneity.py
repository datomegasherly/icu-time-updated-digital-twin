#!/usr/bin/env python3
"""Exploratory hospital heterogeneity audit of frozen Stage 5G design; read-only sources.
No patient-level or actual hospital identifier export. This is NOT publication approval.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from stage5c_model_helpers import ARM, design, hash_holdout, metrics, digest
OUT=('stage5h_summary.json','stage5h_hospital_distribution.csv','stage5h_hospital_performance.csv','stage5h_manifest.json')

def obtain_ids(db):
    import duckdb
    con=duckdb.connect(str(db),read_only=True)
    try:
        cols={c[0].lower() for c in con.execute('DESCRIBE SELECT * FROM eicu_patient').fetchall()}
        needed={'patientunitstayid','uniquepid','hospitalid'}
        if not needed.issubset(cols): raise ValueError('Missing eICU patient columns: '+str(needed-cols))
        x=con.execute('SELECT patientunitstayid, uniquepid, hospitalid FROM eicu_patient').fetchdf()
    finally:con.close()
    if x.patientunitstayid.isna().any() or x.patientunitstayid.duplicated().any():raise ValueError('Missing/duplicate raw stay IDs')
    return x

def calculate(mimic,eicu,raw,seed,min_stays,min_events):
    for name,d in [('MIMIC',mimic),('eICU',eicu)]:
        req={'stay_id','group_id','age','sex','y_24h'}
        if req-set(d.columns):raise ValueError(name+' missing '+str(req-set(d.columns)))
        if d.stay_id.isna().any() or d.stay_id.duplicated().any() or not d.y_24h.isin([0,1]).all(): raise ValueError(name+' invalid stay IDs or outcomes')
    hold=np.asarray([hash_holdout(z,seed) for z in mimic.group_id],dtype=bool)
    if (mimic.loc[hold,'y_24h'].sum()==0 or mimic.loc[~hold,'y_24h'].sum()==0):raise ValueError('No events in MIMIC split')
    eicu=eicu.copy();eicu['_id']=pd.to_numeric(eicu.stay_id,errors='raise').astype('int64')
    raw=raw.copy();raw['patientunitstayid']=pd.to_numeric(raw.patientunitstayid,errors='raise').astype('int64')
    z=eicu[['_id']].merge(raw,left_on='_id',right_on='patientunitstayid',how='left',validate='one_to_one',indicator=True,sort=False)
    if not z['_merge'].eq('both').all():raise ValueError('Unmatched stays in eICU')
    if z.uniquepid.isna().any(): raise ValueError('Missing patient IDs')
    hospitals=z.hospitalid.astype('string').str.strip().fillna('__missing__').replace('', '__missing__')
    ys=eicu.y_24h.to_numpy(dtype=int)
    pred={}
    for name in ARM:
        model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler()),('lr',LogisticRegression(C=1.,solver='liblinear',max_iter=2000,random_state=seed))])
        model.fit(design(mimic,name).loc[~hold],mimic.loc[~hold,'y_24h'].to_numpy(dtype=int))
        pred[name]=model.predict_proba(design(eicu,name))[:,1]
        print('Fitted',name,flush=True)
    overall={name:{k:float(v) for k,v in metrics(ys,p).items()} for name,p in pred.items()}
    # Stable site numbering has no correspondence to real hospitalid in the exported data.
    # Output distribution is bucketed by size only; per-hospital anonymous rows require thresholds.
    counts=pd.DataFrame({'hospital':hospitals,'y':ys}).groupby('hospital',dropna=False).agg(stays=('y','size'),events=('y','sum'))
    records=[]
    for size,a,b in [('under_100',0,100),('100_to_499',100,500),('500_to_999',500,1000),('1000_plus',1000,10**12)]:
        ss=counts[(counts.stays>=a)&(counts.stays<b)]
        records.append({'hospital_size_bucket':size,'hospitals':int(len(ss)),'stays':int(ss.stays.sum()),'events':int(ss.events.sum())})
    eligible=counts[(counts.stays>=min_stays)&(counts.events>=min_events)&((counts.stays-counts.events)>=min_events)]
    hospital_results=[]
    # only anonymized labels; no raw hospital ID and no stay-level output.
    for i,(hospital,row) in enumerate(eligible.iterrows(),1):
        mask=hospitals.eq(hospital).to_numpy(dtype=bool)
        item={'site_label':f'Site_{i:03d}', 'stays':int(mask.sum()),'events':int(ys[mask].sum()),'prevalence':float(ys[mask].mean())}
        for name in ARM:
            p=pred[name][mask]
            item.update({f'{name}_auroc':float(roc_auc_score(ys[mask],p)),f'{name}_auprc':float(average_precision_score(ys[mask],p)),f'{name}_brier':float(brier_score_loss(ys[mask],p))})
        item['lab_minus_integrated_auroc']=item['lab_auroc']-item['integrated_auroc']
        hospital_results.append(item)
    # Reorder by descending stays: anonymous labels not stable across reruns; do not infer hospital identity.
    site=pd.DataFrame(hospital_results)
    if not site.empty:
        site=site.sort_values(['stays','events'],ascending=False).reset_index(drop=True)
        site['site_label']=[f'Site_{i:03d}' for i in range(1,len(site)+1)]
    summary={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED','external_stays':int(len(eicu)),'external_events':int(ys.sum()),'unique_patients':int(z.uniquepid.astype('string').nunique()),'hospital_count':int(len(counts)),'hospitals_missing_id':int((hospitals=='__missing__').sum()),'eligible_hospitals_reporting':int(len(eligible)),'hospitals_suppressed':int(len(counts)-len(eligible)),'stays_covered_by_reporting':int(eligible.stays.sum()),'events_covered_by_reporting':int(eligible.events.sum()),'overall_metrics':overall,'per_hospital_thresholds':{'min_stays':min_stays,'min_events':min_events,'min_nonevents':min_events},'notes':['Unweighted site-wise performance is not equal to pooled external performance','Site IDs are anonymized and do not map to actual hospitals','Patient-cluster bootstrap remains primary uncertainty method; no hospital bootstrap here','eICU endpoint uses discharge proxy and needs clinical caveat']}
    if not site.empty:
        delta=site.lab_minus_integrated_auroc
        summary['hospital_direction']={'lab_better_n':int((delta>0).sum()),'integrated_better_n':int((delta<0).sum()),'equal_n':int((delta==0).sum()),'median_delta_auroc':float(delta.median()),'q25_delta_auroc':float(delta.quantile(.25)),'q75_delta_auroc':float(delta.quantile(.75))}
    return summary,pd.DataFrame(records),site

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('expanded-mimic','expanded-eicu','db','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--min-stays',type=int,default=100);ap.add_argument('--min-events',type=int,default=10)
    a=ap.parse_args()
    if a.min_stays<50 or a.min_events<5:ap.error('Privacy/reliability thresholds too low')
    for p in (a.expanded_mimic,a.expanded_eicu,a.db):
        if not p.is_file():raise FileNotFoundError(p)
    if any((a.out/f).exists() for f in OUT):raise FileExistsError('Choose fresh output directory')
    mimic=pd.read_csv(a.expanded_mimic,low_memory=False);eicu=pd.read_csv(a.expanded_eicu,low_memory=False)
    summary,dist,performance=calculate(mimic,eicu,obtain_ids(a.db),a.seed,a.min_stays,a.min_events)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/OUT[0]).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    dist.to_csv(a.out/OUT[1],index=False);performance.to_csv(a.out/OUT[2],index=False)
    manifest={'input_sha256':{'mimic':digest(a.expanded_mimic),'eicu':digest(a.expanded_eicu)},'source_db_read_only':True,'seed':a.seed,'no_patient_or_real_hospital_ids_exported':True,'no_external_retraining':True,'selection_thresholds':{'min_stays':a.min_stays,'min_events':a.min_events}}
    (a.out/OUT[3]).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('STAGE 5H COMPLETE — EXPLORATORY',json.dumps({k:summary[k] for k in ('hospital_count','eligible_hospitals_reporting','stays_covered_by_reporting','events_covered_by_reporting')}),flush=True)
if __name__=='__main__': main()
