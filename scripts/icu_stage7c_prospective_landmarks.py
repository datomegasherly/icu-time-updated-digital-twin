#!/usr/bin/env python3
"""Stage7C: at-risk-at-landmark 24h-ahead ICU death, frozen 6h model.
Retrospective evaluation; not prospective validation or an operational digital twin.
Read-only DuckDB. Outputs are aggregate only. Requires Stage7B module beside script.
"""
import argparse, json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from icu_stage7b_frozen_model import (WINDOWS, ARM, VITAL, LAB, STATS, features_sql, design,
                                      hash_holdout, make_model, digest)

HORIZON=1440

def cohorts(con):
    # First ICU stay by subject, consistent with original MIMIC index stay selection.
    m=con.execute('''WITH idx AS (
      SELECT i.stay_id,i.subject_id AS group_id,i.hadm_id,i.intime,i.outtime,
        a.deathtime AS death_ts,p.gender AS sex,
        TRY_CAST(p.anchor_age AS DOUBLE) + year(i.intime)-TRY_CAST(p.anchor_year AS DOUBLE) AS age,
        row_number() OVER(PARTITION BY i.subject_id ORDER BY i.intime,i.stay_id) rn
      FROM mimic_icustays i JOIN mimic_admissions a ON i.hadm_id=a.hadm_id AND i.subject_id=a.subject_id
      JOIN mimic_patients p ON i.subject_id=p.subject_id)
      SELECT * FROM idx WHERE rn=1 AND age>=18
        AND outtime>intime+INTERVAL 360 MINUTE
        AND (death_ts IS NULL OR death_ts>intime+INTERVAL 360 MINUTE)''').df()
    # First ICU index stay per health-system stay, using Stage5F ordering.
    e=con.execute('''WITH idx AS (
       SELECT TRY_CAST(p.patientunitstayid AS BIGINT) stay_id,
       CAST(p.uniquepid AS VARCHAR) AS person,
       TRY_CAST(p.patienthealthsystemstayid AS BIGINT) AS group_id,
       CASE WHEN TRY_CAST(p.age AS DOUBLE) IS NOT NULL THEN TRY_CAST(p.age AS DOUBLE)
            WHEN trim(CAST(p.age AS VARCHAR)) LIKE '>%' THEN 90.0 ELSE NULL END AS age,
       p.gender AS sex,
       TRY_CAST(p.unitdischargeoffset AS DOUBLE) AS discharge_min,
       lower(trim(CAST(p.unitdischargestatus AS VARCHAR))) AS discharge_status,
       row_number() OVER(PARTITION BY p.patienthealthsystemstayid ORDER BY
         TRY_CAST(p.hospitaladmitoffset AS DOUBLE) DESC NULLS LAST,p.patientunitstayid) rn
       FROM eicu_patient p)
       SELECT * FROM idx WHERE rn=1 AND age>=18 AND discharge_min>360''').df()
    for tag,d in [('mimic',m),('eicu',e)]:
        if d.stay_id.isna().any() or d.stay_id.duplicated().any():
            raise RuntimeError(f'{tag}: nonunique / missing index ICU stay ID')
    if e.person.isna().any() or e.person.astype(str).str.strip().eq('').any():
        raise RuntimeError('eICU patient IDs missing: patient-cluster validation unavailable')
    return m,e

def risk_labels(d,source,h):
    minute=h*60
    if source=='mimic':
        start=d.intime+pd.to_timedelta(minute,unit='m')
        end=d.intime+pd.to_timedelta(minute+HORIZON,unit='m')
        out=pd.to_datetime(d.outtime,errors='coerce')
        death=pd.to_datetime(d.death_ts,errors='coerce')
        # Still in ICU and not already dead AT the landmark.
        at_risk=(out>start)&(death.isna()|(death>start))
        event=death.notna()&(death>start)&(death<=end)&(death<=out)
        # Death within current ICU stay only. Discharged alive before horizon is a
        # negative for ICU-specific endpoint; hospital death after ICU is not ICU death.
        usable=at_risk & out.notna()
    else:
        t=pd.to_numeric(d.discharge_min,errors='coerce')
        status=d.discharge_status.astype('string').str.lower().str.strip()
        at_risk=t>minute
        event=(status=='expired').fillna(False)&(t>minute)&(t<=minute+HORIZON)
        # Unknown status after leaving the unit before horizon is unascertainable.
        negative=(t>=minute+HORIZON)|((status=='alive').fillna(False)&(t>minute)&(t<minute+HORIZON))
        usable=at_risk&(event|negative)
    return np.asarray(usable,dtype=bool),np.asarray(event,dtype=np.int8)

def qc_24(d,source,reference):
    eligible,y=risk_labels(d,source,24)
    new=pd.DataFrame({'stay_id':d.loc[eligible,'stay_id'].astype('int64'),
                      'label':y[eligible]})
    ref=pd.read_csv(reference,usecols=['stay_id','y_24h'])
    ref['stay_id']=pd.to_numeric(ref.stay_id,errors='raise').astype('int64')
    merged=new.merge(ref,on='stay_id',how='outer',indicator=True,validate='one_to_one')
    both=merged['_merge']=='both'
    mismatch=int((merged.loc[both,'label']!=merged.loc[both,'y_24h']).sum())
    result={'source':source,'new_stays':len(new),'archived_stays':len(ref),
            'new_events':int(new.label.sum()),'archived_events':int(ref.y_24h.sum()),
            'new_only':int((merged['_merge']=='left_only').sum()),
            'archived_only':int((merged['_merge']=='right_only').sum()),'shared_label_mismatch':mismatch}
    return result

def bootstrap_ci(y,p,groups,replicates,seed):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    _,inv=np.unique(np.asarray(groups,dtype=str),return_inverse=True)
    nclust=inv.max()+1
    rng=np.random.default_rng(seed)
    samples={'auroc':[],'auprc':[],'brier':[]}
    for _ in range(replicates):
        counts=np.bincount(rng.integers(0,nclust,size=nclust),minlength=nclust)
        wt=counts[inv]
        if not (wt[y==1].sum()>0 and wt[y==0].sum()>0):continue
        samples['auroc'].append(roc_auc_score(y,p,sample_weight=wt))
        samples['auprc'].append(average_precision_score(y,p,sample_weight=wt))
        samples['brier'].append(np.average((y-p)**2,weights=wt))
    if min(map(len,samples.values()))<.9*replicates:
        raise RuntimeError('Too many bootstrap replicates without both outcomes')
    return {k:{'low':float(np.quantile(v,.025)),'high':float(np.quantile(v,.975)),
               'valid_replicates':len(v)} for k,v in samples.items()}

def assess(y,p):
    if len(np.unique(y))<2:raise RuntimeError('Only one outcome class in evaluation')
    return {'n':len(y),'events':int(y.sum()),'prevalence':float(y.mean()),
            'auroc':float(roc_auc_score(y,p)),
            'auprc':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)),
            'mean_predicted_risk':float(p.mean()),
            'pct_risk_gt_50':float((p>.5).mean())}

def main():
    p=argparse.ArgumentParser()
    for k in ('db','config','expanded-mimic','expanded-eicu','out'):
        p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--bootstrap',type=int,default=200)
    args=p.parse_args()
    if args.bootstrap<50:raise ValueError('Bootstrap must be >=50')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('Output folder not empty; use a new folder')
    args.out.mkdir(parents=True,exist_ok=True)
    import duckdb
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    con=duckdb.connect(str(args.db),read_only=True)
    try:
        m,e=cohorts(con)
        print(f'6h-at-risk cohorts: MIMIC {len(m):,}; eICU {len(e):,}',flush=True)
        qa=[qc_24(m,'mimic',args.expanded_mimic),qc_24(e,'eicu',args.expanded_eicu)]
        pd.DataFrame(qa).to_csv(args.out/'stage7c_24h_cohort_reconciliation.csv',index=False)
        print('Stage5B 24h cohort reconciliation:',qa,flush=True)
        # To prevent unacknowledged outcome drift, stop on a label mismatch.
        if any(x['shared_label_mismatch'] for x in qa):
            raise RuntimeError('Stage5B shared 24h labels differ. Review cohort reconciliation, no models fit')
        if any(x['new_only']+x['archived_only']>max(10,.001*x['archived_stays']) for x in qa):
            raise RuntimeError('Stage5B 24h cohort set differs materially; review QA before model fitting')
        for tag,d in [('mimic',m),('eicu',e)]:
            con.register('ids_'+tag,pd.DataFrame({'stay_id':pd.to_numeric(d.stay_id).astype('int64')}))
            con.execute(f'CREATE TEMP TABLE {"m" if tag=="mimic" else "e"}_cohort AS SELECT * FROM ids_{tag}')
            additions=[]
            for feature in VITAL+LAB:
                ids=cfg[tag]['vitals'][feature] if feature in VITAL else cfg[tag]['labs'][feature]
                f=con.execute(features_sql(tag,feature,ids)).df()
                if f.stay_id.duplicated().any():raise RuntimeError('Duplicate feature rows')
                d0=d[['stay_id']].merge(f,on='stay_id',how='left',validate='one_to_one',sort=False)
                if not d0.stay_id.equals(d.stay_id):raise RuntimeError('Feature merge order changed')
                additions.append(d0.drop(columns='stay_id'))
                print('  extracted',tag,feature,flush=True)
            d=pd.concat([d.reset_index(drop=True)]+additions,axis=1)
            if tag=='mimic':m=d
            else:e=d
    finally:
        con.close()
    # Identical training patient group split at all time points; fit at 6h only.
    hold=np.array([hash_holdout(v,args.seed) for v in m.group_id],dtype=bool)
    train_ok,y_m6=risk_labels(m,'mimic',6)
    train=train_ok&~hold
    if not (train.sum()>100 and y_m6[train].sum()>10):
        raise RuntimeError('Insufficient 6h MIMIC training outcomes')
    rows=[];counts=[];cis={}
    for source,d in [('internal',m),('external',e)]:
        for h in (6,12,18,24):
            ok,y=risk_labels(d,'mimic' if source=='internal' else 'eicu',h)
            mask=ok&(hold if source=='internal' else np.ones(len(d),bool))
            counts.append({'cohort':source,'landmark_h':h,'at_risk_n':int(ok.sum()),
                           'evaluated_n':int(mask.sum()),'evaluated_events':int(y[mask].sum()),
                           'competing_discharge_included':True})
    for arm in ARM:
        model=make_model(args.seed)
        x6=design(m,arm,360)
        model.fit(x6.loc[train],y_m6[train])
        print('Frozen at 6h:',arm,flush=True)
        for source,d in [('internal',m),('external',e)]:
            for h in (6,12,18,24):
                ok,y=risk_labels(d,'mimic' if source=='internal' else 'eicu',h)
                mask=ok&(hold if source=='internal' else np.ones(len(d),bool))
                x=design(d,arm,h*60)
                pred=model.predict_proba(x.loc[mask])[:,1]
                result={'model':arm,'cohort':source,'train_h':6,'landmark_h':h,
                        'target':'icu_death_within_next_24h',**assess(y[mask],pred)}
                rows.append(result)
                print(f'  {arm} {source} {h}h: n={result["n"]:,} events={result["events"]} AUROC={result["auroc"]:.4f}',flush=True)
                if source=='external' and h in (6,24):
                    groups=d.loc[mask,'person'].astype(str).to_numpy()
                    cis[f'{arm}_external_{h}h']=bootstrap_ci(y[mask],pred,groups,args.bootstrap,args.seed+100*h+len(arm))
    pd.DataFrame(rows).to_csv(args.out/'stage7c_rolling_metrics.csv',index=False)
    pd.DataFrame(counts).to_csv(args.out/'stage7c_landmark_cohorts.csv',index=False)
    (args.out/'stage7c_cluster_ci.json').write_text(json.dumps(cis,indent=2),encoding='utf8')
    manifest={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED','label':'ICU death in next 24 hours',
              'design':'Landmark-specific risk sets, fixed model trained at 6h on MIMIC, updated features but NO model refit',
              'inclusion':'Adult first-index ICU stays, present and alive strictly beyond each landmark',
              'outcome_mimic':'deathtime in current ICU stay within (landmark,landmark+24h]',
              'outcome_eicu':'expired unitdischarge status and discharge offset in (landmark,landmark+24h] (proxy)',
              'early_alive_discharge':'negative ICU death',
              'unknown_discharge_before_horizon':'excluded eICU',
              'source_hashes':{name:digest(path) for name,path in {'config':args.config,'stage5b_mimic':args.expanded_mimic,'stage5b_eicu':args.expanded_eicu}.items()},
              'db_read_only':True,'db_file_size_bytes':args.db.stat().st_size,
              'seed':args.seed,'bootstrap':args.bootstrap,'no_patient_level_exports':True,
              'limitations':['Retrospective repeated snapshots, not an operational/live twin',
                 'At-risk cohorts and outcome prevalence CHANGE across landmarks; do not read cross-landmark AUROC as paired improvement',
                 'Frozen 6h model has a moving target; later risk sets differ from training distribution',
                 'eICU discharge offset is a proxy for ICU death timestamp',
                 'Models and feature choices informed by prior external eICU exploration; eICU not fresh independent test',
                 'Bootstrap covers patient sampling uncertainty, not feature/model selection uncertainty']}
    (args.out/'stage7c_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('STAGE 7C COMPLETE — RETROSPECTIVE MOVING 24H LANDMARKS',flush=True)
    print(json.dumps({'cohort_reconciliation':qa,'landmarks':counts},indent=2),flush=True)

if __name__=='__main__': main()
