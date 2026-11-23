#!/usr/bin/env python3
"""Stage 8B — paired, retrospective availability-proxy sensitivity (aggregate exports only).

Runs measured-time Stage7E comparator and availability-aware 6h-trained models in
ONE execution, on identical patient risk sets. Never writes row-level predictions.
This does not establish clinical first-result visibility or an operational twin.
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from icu_stage7c_prospective_landmarks import cohorts, risk_labels, assess, qc_24
from icu_stage7b_frozen_model import (VITAL, LAB, ARM, STATS, features_sql, design,
                                       hash_holdout, make_model, digest)
from icu_stage7e_unified_dynamic_audit import paired_bootstrap, calib

HOURS=(6,12,18,24)
SCENARIOS=('exclude','unknown','sensitivity')


def available_sql(ds,feature,ids,scenario):
    """Construct lab features using the exact Stage7B aggregation and tie policy.

    exclude: require availability>=sample and availability<landmark.
    unknown: same conservative eligibility, separate reverse/unknown reporting.
    sensitivity: admit reverse records at measured time (optimistic upper bound).
    Invalid/missing availability remains excluded in ALL scenarios.
    """
    from icu_stage7b_frozen_model import BOUNDS, strsql
    lo,hi=BOUNDS[feature]
    if ds=='mimic':
        ids_sql=','.join(str(int(z)) for z in ids)
        raw=f'''SELECT c.stay_id, i.intime AS anchor,
         TRY_CAST(x.charttime AS TIMESTAMP) AS ts,
         TRY_CAST(x.storetime AS TIMESTAMP) AS availability,
         CASE WHEN TRY_CAST(x.valuenum AS DOUBLE) BETWEEN {lo} AND {hi}
         THEN TRY_CAST(x.valuenum AS DOUBLE) ELSE NULL END AS val
         FROM m_cohort c JOIN mimic_icustays i ON i.stay_id=c.stay_id
         JOIN mimic_labevents x ON x.hadm_id=i.hadm_id
         AND TRY_CAST(x.itemid AS BIGINT) IN ({ids_sql})
         WHERE TRY_CAST(x.charttime AS TIMESTAMP)>=i.intime
          AND TRY_CAST(x.charttime AS TIMESTAMP)<i.intime+INTERVAL 1440 MINUTE'''
        for_h=lambda h: (f'ts < anchor+INTERVAL {60*h} MINUTE',
                          f'availability < anchor+INTERVAL {60*h} MINUTE')
    else:
        ids_sql=','.join(strsql(str(z).lower()) for z in ids)
        raw=f'''SELECT c.stay_id, TRY_CAST(x.labresultoffset AS DOUBLE) AS ts,
         TRY_CAST(x.labresultrevisedoffset AS DOUBLE) AS availability,
         CASE WHEN TRY_CAST(x.labresult AS DOUBLE) BETWEEN {lo} AND {hi}
         THEN TRY_CAST(x.labresult AS DOUBLE) ELSE NULL END AS val
         FROM e_cohort c JOIN eicu_lab x ON
         TRY_CAST(x.patientunitstayid AS BIGINT)=c.stay_id
         AND lower(CAST(x.labname AS VARCHAR)) IN ({ids_sql})
         WHERE TRY_CAST(x.labresultoffset AS DOUBLE)>=0
          AND TRY_CAST(x.labresultoffset AS DOUBLE)<1440'''
        for_h=lambda h:(f'ts<{60*h}',f'availability<{60*h}')
    expressions=[]
    for h in HOURS:
        observed,available=for_h(h)
        valid=f'({observed} AND val IS NOT NULL)'
        reverse=f'(availability IS NOT NULL AND availability<ts)'
        strict=f'({valid} AND availability IS NOT NULL AND availability>=ts AND {available})'
        if scenario=='sensitivity':
            passed=f'({strict} OR ({valid} AND {reverse}))'
        else:passed=strict
        suffix=f't{h}'
        for stat,fn in (('mean','avg'),('min','min'),('max','max')):
            expressions.append(f'{fn}(val) FILTER (WHERE {passed}) AS {feature}_{stat}_{suffix}')
        expressions.append(f'arg_max(val,ts ORDER BY val DESC) FILTER (WHERE {passed}) AS {feature}_last_{suffix}')
        expressions.extend((
            f'count(*) FILTER (WHERE {valid}) AS {feature}_measured_n_{suffix}',
            f'count(*) FILTER (WHERE {strict}) AS {feature}_available_n_{suffix}',
            f'count(*) FILTER (WHERE {valid} AND {reverse}) AS {feature}_reverse_n_{suffix}',
            f'count(*) FILTER (WHERE {valid} AND availability IS NULL) AS {feature}_unknown_n_{suffix}',
            f'count(*) FILTER (WHERE {valid} AND availability IS NOT NULL AND availability>=ts AND NOT ({available})) AS {feature}_late_n_{suffix}'
        ))
    return f'WITH raw AS ({raw}) SELECT stay_id, '+', '.join(expressions)+' FROM raw GROUP BY stay_id'


def attach(d,f):
    if f.stay_id.duplicated().any():raise RuntimeError('Duplicate feature stay_id')
    merged=d[['stay_id']].merge(f,on='stay_id',how='left',sort=False,validate='one_to_one')
    if not merged.stay_id.equals(d.stay_id):raise RuntimeError('Feature merge order drift')
    return pd.concat([d.reset_index(drop=True),merged.drop(columns=['stay_id'])],axis=1)


def build_scenario(baseline, lab_frame):
    d=baseline.copy(deep=False).copy()
    for lab in LAB:
        for stat in STATS:
            for h in HOURS:
                col=f'{lab}_{stat}_t{h}'
                d[col]=lab_frame[col].to_numpy()
    return d


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('db','config','expanded-mimic','expanded-eicu','stage7e-metrics','out'):
        ap.add_argument('--'+k,required=True,type=Path)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--bootstrap',type=int,default=300)
    a=ap.parse_args()
    if a.bootstrap<100:raise ValueError('bootstrap must be >=100')
    for path in (a.db,a.config,a.expanded_mimic,a.expanded_eicu,a.stage7e_metrics):
        if not path.is_file():raise FileNotFoundError(path)
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError('Use an empty/new output directory')
    import duckdb
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    a.out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(a.db),read_only=True)
    baseline={}; alt={}; recon=[]
    try:
        con.execute('SET threads=1')
        m,e=cohorts(con)
        recon=[qc_24(m,'mimic',a.expanded_mimic),qc_24(e,'eicu',a.expanded_eicu)]
        pd.DataFrame(recon).to_csv(a.out/'stage8b_cohort_reconciliation.csv',index=False)
        if any(r['shared_label_mismatch'] or r['new_only'] for r in recon):
            raise RuntimeError('Cohort/label reconciliation failed')
        for tag,d in (('mimic',m),('eicu',e)):
            if tag=='mimic':
                reg=d[['stay_id']]
            else:reg=d[['stay_id']]
            con.register('ids_'+tag,reg)
            con.execute(f'CREATE TEMP TABLE {"m" if tag=="mimic" else "e"}_cohort AS SELECT * FROM ids_{tag}')
            chunks=[]; scenario_chunks={s:[] for s in SCENARIOS}
            for v in VITAL+LAB:
                ids=cfg[tag]['vitals' if v in VITAL else 'labs'][v]
                chunks.append(con.execute(features_sql(tag,v,ids)).df())
                if v in LAB:
                    for s in ('exclude','sensitivity'):
                        scenario_chunks[s].append(con.execute(available_sql(tag,v,ids,s)).df())
                print(f'Extracted {tag}/{v}',flush=True)
            b=d.reset_index(drop=True)
            for f in chunks:b=attach(b,f)
            baseline[tag]=b
            for s in ('exclude','sensitivity'):
                labs=d[['stay_id']].reset_index(drop=True)
                for f in scenario_chunks[s]:labs=attach(labs,f)
                alt[tag,s]=labs
            alt[tag,'unknown']=alt[tag,'exclude']
    finally:con.close()

    hold=np.asarray([hash_holdout(x,a.seed) for x in baseline['mimic'].group_id],bool)
    eligible={}; outcomes={}; groups={}
    for tag,d in baseline.items():
        for h in HOURS:
            valid,y=risk_labels(d,tag,h)
            mask=valid & (hold if tag=='mimic' else np.ones(len(d),bool))
            eligible[tag,h]=mask
            outcomes[tag,h]=y[mask]
            groups[tag,h]=(d.loc[mask,'group_id'] if tag=='mimic' else d.loc[mask,'person']).astype(str).to_numpy()
    m6,y6=risk_labels(baseline['mimic'],'mimic',6)
    train=m6 & ~hold
    preds={}; metrics=[]; calibration=[]; changes=[]; coverage=[]; same_run=[]
    modes=('measured',)+SCENARIOS
    for tag in baseline:
        for s in SCENARIOS:
            d=build_scenario(baseline[tag],alt[tag,s]);
            # Compare baseline vs new feature arrays without exporting patient records.
            for h in HOURS:
                x=d.loc[eligible[tag,h]]
                for lab in LAB:
                    for stat in STATS:
                        a0=pd.to_numeric(baseline[tag].loc[eligible[tag,h],f'{lab}_{stat}_t{h}'],errors='coerce')
                        b0=pd.to_numeric(x[f'{lab}_{stat}_t{h}'],errors='coerce')
                        eq=(a0.isna()&b0.isna()) | (a0.notna()&b0.notna()&np.isclose(a0,b0,atol=1e-8,rtol=1e-8))
                        changes.append(dict(cohort=tag,scenario=s,landmark_h=h,feature=lab,stat=stat,
                            n_stays=len(x),changed=int((~eq).sum()),original_nonmissing=int(a0.notna().sum()),
                            revised_nonmissing=int(b0.notna().sum()),original_mean=float(a0.mean()) if a0.notna().any() else None,
                            revised_mean=float(b0.mean()) if b0.notna().any() else None))
            # measured timestamp partitions only on same eligible patient risk sets
            for h in HOURS:
                x=alt[tag,s].loc[eligible[tag,h]]
                for lab in LAB:
                    values={name:pd.to_numeric(x[f'{lab}_{name}_n_t{h}'],errors='coerce').fillna(0) for name in ('measured','available','reverse','unknown','late')}
                    if not np.array_equal(values['measured'].to_numpy(),sum(values[z] for z in ('available','reverse','unknown','late')).to_numpy()):
                        raise RuntimeError(f'Timestamp partition mismatch {tag}/{lab}/{h}')
                    n=values['available']+(values['reverse'] if s=='sensitivity' else 0)
                    coverage.append(dict(cohort=tag,scenario=s,landmark_h=h,feature=lab,n_stays=len(x),
                        measured_rows=int(values['measured'].sum()),available_rows=int(values['available'].sum()),
                        reverse_rows=int(values['reverse'].sum()),unknown_rows=int(values['unknown'].sum()),
                        late_rows=int(values['late'].sum()),stays_measured=int((values['measured']>0).sum()),
                        stays_included=int((n>0).sum()),stays_losing_all=int(((values['measured']>0)&(n==0)).sum())))
            alt[tag,s]=d
    for mode in modes:
        for arm in ARM:
            model=make_model(a.seed)
            dm=baseline['mimic'] if mode=='measured' else alt['mimic',mode]
            model.fit(design(dm,arm,360).loc[train],y6[train])
            print('Fitted',mode,arm,flush=True)
            for tag in baseline:
                d=baseline[tag] if mode=='measured' else alt[tag,mode]
                for h in HOURS:
                    mask=eligible[tag,h]; y=outcomes[tag,h]
                    p=model.predict_proba(design(d,arm,h*60).loc[mask])[:,1]
                    preds[mode,tag,arm,h]=p
                    rows=assess(y,p)
                    metrics.append(dict(scenario=mode,cohort='internal' if tag=='mimic' else 'external',
                                        model=arm,landmark_h=h,**rows))
                    calibration.append(dict(scenario=mode,cohort='internal' if tag=='mimic' else 'external',
                                            model=arm,landmark_h=h,**calib(y,p)))
    frame=pd.DataFrame(metrics)
    ref=pd.read_csv(a.stage7e_metrics)
    current=frame[frame.scenario=='measured'].drop(columns='scenario')
    joined=current.merge(ref,on=['model','cohort','landmark_h'],validate='one_to_one',suffixes=('_new','_archived'))
    if len(joined)!=24 or len(ref)!=24:raise RuntimeError('Stage7E reference grid mismatch')
    for field in ('n','events','auroc','auprc','brier','mean_predicted_risk','pct_risk_gt_50'):
        delta=(joined[field+'_new']-joined[field+'_archived']).abs()
        for i in range(len(joined)):
            same_run.append(dict(model=joined.iloc[i].model,cohort=joined.iloc[i].cohort,
                landmark_h=int(joined.iloc[i].landmark_h),metric=field,absolute_difference=float(delta.iloc[i])))
        if (field in ('n','events') and delta.max()!=0) or (field not in ('n','events') and delta.max()>1e-4):
            pd.DataFrame(same_run).to_csv(a.out/'stage8b_reference_drift.csv',index=False)
            raise RuntimeError(f'Fail closed: Stage7E drift in {field}: {delta.max()}')
    pd.DataFrame(same_run).to_csv(a.out/'stage8b_reference_drift.csv',index=False)
    frame.to_csv(a.out/'stage8b_metrics.csv',index=False)
    pd.DataFrame(calibration).to_csv(a.out/'stage8b_calibration.csv',index=False)
    pd.DataFrame(coverage).to_csv(a.out/'stage8b_availability_coverage.csv',index=False)
    pd.DataFrame(changes).to_csv(a.out/'stage8b_feature_shift.csv',index=False)
    paired=[]
    # Paired across timestamp strategies, and between models within a strategy.
    for tag in ('eicu',):
        cohort='external'
        for h in HOURS:
            y=outcomes[tag,h]; g=groups[tag,h]
            contrasts=[]
            for s in SCENARIOS:
                for arm in ARM:contrasts.append((s,arm,'measured',arm))
                contrasts.extend(((s,'integrated',s,'lab'),(s,'integrated',s,'vital')))
            for i,(sa,arm_a,sb,arm_b) in enumerate(contrasts):
                p=preds[sa,tag,arm_a,h]; q=preds[sb,tag,arm_b,h]
                if np.array_equal(p,q):
                    results=[dict(metric=metric,delta=0.,ci_low=0.,ci_high=0.,valid_replicates=a.bootstrap)
                             for metric in ('auroc','auprc','brier')]
                else:
                    print('Bootstrap',cohort,h,sa,arm_a,'vs',sb,arm_b,flush=True)
                    results=paired_bootstrap(y,p,q,g,a.bootstrap,a.seed+(100000 if tag=='eicu' else 0)+h*100+i)
                for r in results:
                    paired.append(dict(cohort=cohort,landmark_h=h,scenario_a=sa,model_a=arm_a,
                                       scenario_b=sb,model_b=arm_b,**r))
    pd.DataFrame(paired).to_csv(a.out/'stage8b_paired_patient_cluster_ci.csv',index=False)
    manifest=dict(status='EXPLORATORY_AVAILABILITY_PROXY_SENSITIVITY_NOT_LEAKAGE_FREE_PROOF',
       measured_comparator='Stage7E recreated in this execution; drift checked against archived Stage7E',
       train='Independent frozen 6h models per scenario; each scenario keeps its own unchanged 6h-trained model at later landmarks',
       proxy_mimic='labevents.storetime',proxy_eicu='labresultrevisedoffset',
       scenarios={'exclude':'Only availability >= measured and availability < landmark',
           'unknown':'Numerically identical conservative predictions to exclude; reverse timestamps explicitly unknown, not classified as leakage',
           'sensitivity':'Optimistic upper bound: accept reverse timestamps if sampled before landmark (not an assertion of clinical availability)'},
       time_boundary='[admission, landmark)',
       limitations=['Result proxy is not proven first clinician visibility','eICU reverse timestamps cannot be called leakage',
           'Original external cohort used for prior model selection; intervals do not correct for selection bias',
           'Retrospective risk-set predictor, not clinical operational twin',
           'eICU uses ICU-discharge offset for death timing',
           'Each scenario refits on MIMIC 6h; differences include learned coefficients and changes in features'],
       baseline_drift_max=float(pd.DataFrame(same_run).absolute_difference.max()),
       cohort_qc=recon,bootstrap=a.bootstrap,seed=a.seed,read_only_database=True,
       patient_level_export=False,config_sha256=digest(a.config),
       input_hashes={name:digest(path) for name,path in [('stage7e_metrics',a.stage7e_metrics),
           ('expanded_mimic',a.expanded_mimic),('expanded_eicu',a.expanded_eicu)]},
       scripts_sha256={name:digest(Path(__file__).with_name(name)) for name in
           ('icu_stage8b_availability_aware.py','icu_stage7b_frozen_model.py',
            'icu_stage7c_prospective_landmarks.py','icu_stage7e_unified_dynamic_audit.py')})
    (a.out/'stage8b_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('STAGE 8B COMPLETE — exploratory availability-proxy sensitivity only',a.out,flush=True)

if __name__=='__main__':main()
