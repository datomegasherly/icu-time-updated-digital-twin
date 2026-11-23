#!/usr/bin/env python3
"""Stage 8A: descriptive audit of laboratory timestamp availability for ICU landmarks.

Read-only MIMIC-IV/eICU DuckDB audit; exports only aggregate summaries.
NO patient-level output, model training, clinical validation, or leakage-free claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

HOURS=(6,12,18,24)
LABS=('creatinine','sodium','potassium','bun','glucose','lactate')
BOUNDS={'creatinine':(.1,30),'sodium':(90,190),'potassium':(1,12),
        'bun':(1,250),'glucose':(10,1500),'lactate':(0,40)}

def sqlquote(s):
    return "'"+str(s).replace("'","''")+"'"

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def columns(con, table):
    return {str(r[0]).lower():r[0] for r in con.execute(f'DESCRIBE {table}').fetchall()}

def audit_sql(source,feature,ids):
    lo,hi=BOUNDS[feature]
    if source=='mimic':
        itemids=','.join(str(int(x)) for x in ids)
        raw=f"""SELECT c.stay_id, c.intime,
           TRY_CAST(x.charttime AS TIMESTAMP) AS measured,
           TRY_CAST(x.storetime AS TIMESTAMP) AS available,
           TRY_CAST(x.valuenum AS DOUBLE) AS val
        FROM m_cohort c JOIN mimic_labevents x ON x.hadm_id=c.hadm_id
           AND TRY_CAST(x.itemid AS BIGINT) IN ({itemids})
           AND TRY_CAST(x.charttime AS TIMESTAMP)>=c.intime
           AND TRY_CAST(x.charttime AS TIMESTAMP)<c.intime+INTERVAL 1440 MINUTE"""
        def before(h):return f'measured<intime+INTERVAL {60*h} MINUTE'
        def within(h):return f'available<intime+INTERVAL {60*h} MINUTE'
    else:
        names=','.join(sqlquote(x.lower()) for x in ids)
        raw=f"""SELECT c.stay_id, NULL AS intime,
           TRY_CAST(x.labresultoffset AS DOUBLE) AS measured,
           TRY_CAST(x.labresultrevisedoffset AS DOUBLE) AS available,
           TRY_CAST(x.labresult AS DOUBLE) AS val
        FROM e_cohort c JOIN eicu_lab x ON TRY_CAST(x.patientunitstayid AS BIGINT)=c.stay_id
           AND lower(trim(CAST(x.labname AS VARCHAR))) IN ({names})
           AND TRY_CAST(x.labresultoffset AS DOUBLE)>=0
           AND TRY_CAST(x.labresultoffset AS DOUBLE)<1440"""
        def before(h):return f'measured<{60*h}'
        def within(h):return f'available<{60*h}'
    parts=[]
    for h in HOURS:
        t=before(h); a=within(h)
        # "available" is a STORETIME proxy for MIMIC and revised-result-entry proxy for eICU.
        valid=f'({t} AND val IS NOT NULL)'
        observed=f'({valid} AND available IS NOT NULL AND available>=measured AND {a})'
        late=f'({valid} AND available IS NOT NULL AND available>=measured AND NOT ({a}))'
        missing=f'({valid} AND available IS NULL)'
        discordant=f'({valid} AND available IS NOT NULL AND available<measured)'
        for name,expr in [('measured_n',valid),('proxy_available_n',observed),
                          ('late_n',late),('missing_proxy_n',missing),('reverse_time_n',discordant)]:
            parts.append(f'count(*) FILTER (WHERE {expr}) AS {name}_{h}')
        for name,expr in [('mean','avg'),('min','min'),('max','max')]:
            parts.append(f'{expr}(val) FILTER (WHERE {valid}) AS baseline_{name}_{h}')
            parts.append(f'{expr}(val) FILTER (WHERE {observed}) AS proxy_{name}_{h}')
        parts.append(f'arg_max(val,measured ORDER BY val DESC) FILTER (WHERE {valid}) AS baseline_last_{h}')
        parts.append(f'arg_max(val,measured ORDER BY val DESC) FILTER (WHERE {observed}) AS proxy_last_{h}')
    return ('WITH raw_input AS ('+raw+'), raw AS (SELECT stay_id,intime,measured,available,'
            +f' CASE WHEN val BETWEEN {lo} AND {hi} THEN val ELSE NULL END AS val FROM raw_input) '
            +'SELECT stay_id, '+', '.join(parts)+' FROM raw GROUP BY stay_id')

def summarize(source,feature,h,cohort,features):
    # All rows of cohort are eligible at this landmark. Missing lab rows are zeros/nans.
    values=cohort[['stay_id']].merge(features,on='stay_id',how='left',validate='one_to_one',sort=False)
    n=len(values)
    measured=pd.to_numeric(values[f'measured_n_{h}'],errors='coerce').fillna(0)
    available=pd.to_numeric(values[f'proxy_available_n_{h}'],errors='coerce').fillna(0)
    late=pd.to_numeric(values[f'late_n_{h}'],errors='coerce').fillna(0)
    missing=pd.to_numeric(values[f'missing_proxy_n_{h}'],errors='coerce').fillna(0)
    reverse=pd.to_numeric(values[f'reverse_time_n_{h}'],errors='coerce').fillna(0)
    if not np.array_equal((available+late+missing+reverse).to_numpy(), measured.to_numpy()):
        raise RuntimeError(f'{source}/{feature}/{h}: timestamp partition does not sum to measured rows')
    has=(measured>0)
    proxy=(available>0)
    # Any differences between numerical aggregates, including value becoming missing.
    feature_disagreement={}
    for stat in ('mean','min','max','last'):
        a=pd.to_numeric(values[f'baseline_{stat}_{h}'],errors='coerce')
        b=pd.to_numeric(values[f'proxy_{stat}_{h}'],errors='coerce')
        differing=has & (~((a.isna()&b.isna()) | (a.notna()&b.notna()&np.isclose(a,b,atol=1e-8,rtol=1e-8))))
        feature_disagreement[f'{stat}_changed_stays']=int(differing.sum())
    return {'source':source,'feature':feature,'landmark_h':h,'at_risk_stays':n,
       'stays_with_early_measured_lab':int(has.sum()),
       'stays_with_proxy_available_lab':int(proxy.sum()),
       'stays_losing_all_lab_under_proxy':int((has&~proxy).sum()),
       'stays_with_any_late_result':int((late>0).sum()),
       'stays_with_unknown_availability':int((missing>0).sum()),
       'stays_with_reverse_time':int((reverse>0).sum()),
       'valid_measured_rows':int(measured.sum()),
       'proxy_available_rows':int(available.sum()),
       'later_recorded_rows':int(late.sum()),
       'missing_availability_rows':int(missing.sum()),
       'reverse_timestamp_rows':int(reverse.sum()),
       'fraction_measured_rows_not_proxy_available':float(1-available.sum()/measured.sum()) if measured.sum() else None,
       **feature_disagreement}

def main():
    parser=argparse.ArgumentParser(description='Stage8A lab availability/timing audit, aggregate-only')
    for key in ('db','config','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--landmark-hours',default='6,12,18,24',help='Fixed at 6,12,18,24 for this protocol')
    args=parser.parse_args()
    if args.landmark_hours!='6,12,18,24':raise ValueError('Only 6,12,18,24 are supported')
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Use a new/empty --out directory')
    try:
        import duckdb
        from icu_stage7c_prospective_landmarks import cohorts,risk_labels
    except ImportError as exc:
        raise SystemExit('Required: duckdb, and existing Stage7C + Stage7B Python modules in same folder: '+str(exc)) from exc
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    for source in ('mimic','eicu'):
        for lab in LABS:
            if lab not in cfg[source]['labs']:raise ValueError(f'No configured {source}/{lab}')
    args.out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(args.db),read_only=True)
    outputs=[];manifest={}
    try:
        req={'mimic_labevents':['hadm_id','itemid','charttime','storetime','valuenum'],
             'eicu_lab':['patientunitstayid','labname','labresult','labresultoffset','labresultrevisedoffset']}
        for table,cols in req.items():
            found=columns(con,table)
            missing=[c for c in cols if c not in found]
            if missing:raise RuntimeError(f'{table} missing columns {missing}. Database needs raw columns imported; do not infer availability from draw time.')
        m,e=cohorts(con)
        # Stage7C eligibility is applied at each landmark, without future eligibility selection.
        for source,df in (('mimic',m),('eicu',e)):
            masks={h:risk_labels(df,source,h)[0] for h in HOURS}
            print(f'{source}: 6h eligible {int(masks[6].sum()):,}; 24h eligible {int(masks[24].sum()):,}',flush=True)
            manifest[f'{source}_eligible']={str(h):int(v.sum()) for h,v in masks.items()}
            if df.stay_id.isna().any() or df.stay_id.duplicated().any():raise RuntimeError('Invalid stays')
            c=df[['stay_id']].copy()
            if source=='mimic':c=df[['stay_id','hadm_id','intime']].copy()
            con.register(f'{source}_registered',c)
            con.execute(f'CREATE TEMP TABLE {"m_cohort" if source=="mimic" else "e_cohort"} AS SELECT * FROM {source}_registered')
            for lab in LABS:
                query=audit_sql(source,lab,cfg[source]['labs'][lab])
                feature_rows=con.execute(query).df()
                if feature_rows.stay_id.duplicated().any():raise RuntimeError(f'Duplicate aggregation IDs: {source}/{lab}')
                for h in HOURS:
                    eligible=df.loc[masks[h],['stay_id']].copy()
                    outputs.append(summarize(source,lab,h,eligible,feature_rows))
                print('  audited',source,lab,flush=True)
    finally:con.close()
    result=pd.DataFrame(outputs).sort_values(['source','landmark_h','feature'])
    result.to_csv(args.out/'stage8a_lab_availability_summary.csv',index=False)
    overview=result.groupby(['source','landmark_h'],as_index=False).agg(
        features=('feature','count'),total_measured_rows=('valid_measured_rows','sum'),
        proxy_available_rows=('proxy_available_rows','sum'),
        later_recorded_rows=('later_recorded_rows','sum'),
        missing_availability_rows=('missing_availability_rows','sum'),
        reverse_timestamp_rows=('reverse_timestamp_rows','sum'),
        feature_stay_pairs_losing_all=('stays_losing_all_lab_under_proxy','sum'),
        feature_stay_pairs_mean_changed=('mean_changed_stays','sum'),
        feature_stay_pairs_last_changed=('last_changed_stays','sum'))
    overview.to_csv(args.out/'stage8a_aggregate_overview.csv',index=False)
    manifest.update({'status':'DESCRIPTIVE_TIMING_AUDIT_NOT_MODEL_VALIDATION',
        'mimic_proxy':'labevents.storetime; not guaranteed clinical visibility',
        'eicu_proxy':'labresultrevisedoffset; revised value entry, not guaranteed first availability',
        'time_boundaries':'[ICU admission, landmark), strict upper bound',
        'values':'same item mappings, bounds and landmark selection; no retraining',
        'privacy':'aggregate-only no patient IDs in outputs',
        'db_path':str(args.db),'config_sha256':sha256(args.config),
        'scripts':'Stage7C cohort and label helpers used for selection; this script does not modify them',
        'warning':'Cannot conclude leakage-free features from revised-offset or storetime alone',
        'overall_rows':len(result)})
    (args.out/'stage8a_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('STAGE 8A COMPLETE — DESCRIPTIVE AUDIT ONLY',str(args.out),flush=True)
    print(overview.to_string(index=False),flush=True)

if __name__=='__main__': main()
