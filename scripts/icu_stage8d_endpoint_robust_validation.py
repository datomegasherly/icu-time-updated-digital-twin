#!/usr/bin/env python3
"""Stage 8D: frozen 6-hour models under Stage8C endpoint perturbations.

Only aggregate results are exported. Neither archived outputs nor DuckDB are mutated.
These are hypothetical endpoint sensitivities, NOT independently adjudicated death times.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from icu_stage7c_prospective_landmarks import cohorts, risk_labels, assess
from icu_stage7b_frozen_model import VITAL, LAB, STATS, features_sql, design, hash_holdout, make_model, digest
from icu_stage7e_unified_dynamic_audit import paired_bootstrap, calib
from icu_stage8b_availability_aware import available_sql, attach, build_scenario
from icu_stage8c_endpoint_adjudication import adjudicate

HOURS=(6,12,18,24)
MODES=('measured','exclude')
ARMS=('lab','integrated')


def endpoint_alternatives(df,source,h,ok,event,aux):
    """Mirror Stage8C scenario denominators and event counts exactly."""
    result=[('stage7e_primary',ok.copy(),event.copy())]
    if source=='eicu':
        unknown=aux['early_unknown']
        result += [('unknown_early_as_negative_sensitivity',ok|unknown,event.copy()),
                   ('unknown_early_as_event_sensitivity',ok|unknown,event|unknown)]
        los=aux['los']; eligible=ok & aux['expired'] & aux['at_risk_raw']
        for shift in (-60,-30,30,60):
            hypothetical=los+shift
            excluded=eligible & (hypothetical<=h*60)
            mask=ok & ~excluded
            outcome=eligible & (hypothetical>h*60) & (hypothetical<=h*60+1440)
            result.append((f'expired_proxy_shift_{shift:+d}min',mask,outcome))
    else:
        start=aux['start']; end=aux['end']; death=aux['death']; discharge=aux['discharge']
        for shift in (-60,60):
            shifted=death+pd.to_timedelta(shift,unit='m')
            outcome=ok & (shifted>start).fillna(False).to_numpy() & (shifted<=end).fillna(False).to_numpy() & (shifted<=discharge).fillna(False).to_numpy()
            result.append((f'artificial_deathtime_shift_{shift:+d}min',ok.copy(),outcome))
    return result


def validate_stage8c(rows, reference):
    actual=pd.DataFrame(rows)
    ref=reference[['source','landmark_h','scenario','n','events']].copy()
    pair=actual.merge(ref,on=['source','landmark_h','scenario'],how='outer',indicator=True,
                      suffixes=('_new','_archive'),validate='one_to_one')
    if len(pair)!=len(ref) or not pair._merge.eq('both').all():
        raise RuntimeError('FAIL_CLOSED Stage8C scenario grid differs')
    if not (pair.n_new.eq(pair.n_archive)&pair.events_new.eq(pair.events_archive)).all():
        raise RuntimeError('FAIL_CLOSED Stage8C scenarios n/events differ')
    return pair.drop(columns='_merge')


def validate_stage8b(rows, reference, tolerance):
    frame=pd.DataFrame(rows)
    ref=reference[(reference.scenario.isin(MODES)) & (reference.model.isin(ARMS))].copy()
    q=frame.merge(ref,on=['scenario','cohort','model','landmark_h'],how='outer',indicator=True,
                  suffixes=('_new','_archive'),validate='one_to_one')
    if len(q)!=len(ref) or not q._merge.eq('both').all():
        raise RuntimeError('FAIL_CLOSED Stage8B reference grid differs')
    for field in ('n','events','auroc','auprc','brier','mean_predicted_risk','pct_risk_gt_50'):
        delta=(q[field+'_new']-q[field+'_archive']).abs()
        q[field+'_abs_drift']=delta
        if (field in ('n','events') and delta.max()!=0) or (field not in ('n','events') and delta.max()>tolerance):
            raise RuntimeError(f'FAIL_CLOSED Stage8B {field} drift {delta.max()}')
    return q


def metrics(y,p):
    if not 0 < int(np.sum(y)) < len(y):
        return dict(n=len(y),events=int(np.sum(y)),auroc=np.nan,auprc=np.nan,brier=np.nan,
                    mean_predicted_risk=float(np.mean(p)),pct_risk_gt_50=float(np.mean(p>.5)))
    return assess(y,p)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    for name in ('db','config','stage8b-metrics','stage8c-sensitivity','out'):
        a.add_argument('--'+name,required=True,type=Path)
    a.add_argument('--seed',type=int,default=2026)
    a.add_argument('--bootstrap',type=int,default=300)
    a.add_argument('--reference-tolerance',type=float,default=1e-4)
    opt=a.parse_args()
    if opt.bootstrap<100:raise ValueError('Require >=100 bootstrap iterations')
    for p in (opt.db,opt.config,opt.stage8b_metrics,opt.stage8c_sensitivity):
        if not p.is_file():raise FileNotFoundError(p)
    if opt.out.exists() and any(opt.out.iterdir()):raise FileExistsError('FAIL_CLOSED: output directory must be empty')
    cfg=json.loads(opt.config.read_text(encoding='utf-8-sig'))
    import duckdb
    con=duckdb.connect(str(opt.db),read_only=True)
    try:
        con.execute('SET threads=1')
        mimic,eicu=cohorts(con)
        original={'mimic':mimic,'eicu':eicu}; datasets={}; scenarios={}
        for source,raw in original.items():
            con.register('ids_'+source,raw[['stay_id']])
            con.execute(f'CREATE TEMP TABLE {"m" if source=="mimic" else "e"}_cohort AS SELECT * FROM ids_{source}')
            base=raw.reset_index(drop=True)
            alt=raw[['stay_id']].reset_index(drop=True)
            for v in VITAL+LAB:
                ids=cfg[source]['vitals' if v in VITAL else 'labs'][v]
                base=attach(base,con.execute(features_sql(source,v,ids)).df())
                if v in LAB:
                    alt=attach(alt,con.execute(available_sql(source,v,ids,'exclude')).df())
                print(f'Extracted {source}/{v}',flush=True)
            datasets[source,'measured']=base
            datasets[source,'exclude']=build_scenario(base,alt)
    finally:
        con.close()
    hold=np.asarray([hash_holdout(x,opt.seed) for x in mimic.group_id],bool)
    m6,y6=risk_labels(mimic,'mimic',6)
    train=m6 & ~hold
    checks=[]; primary_reference=[]; frames={}; pred={}; current_metrics=[]
    stage8c_ref=pd.read_csv(opt.stage8c_sensitivity)
    for source,d in original.items():
        for h in HOURS:
            _,ok,event,aux=adjudicate(d,source,h)
            ss=endpoint_alternatives(d,source,h,ok,event,aux)
            for name,mask,y in ss:
                checks.append(dict(source=source,landmark_h=h,scenario=name,n=int(mask.sum()),events=int(y[mask].sum())))
                if source=='mimic':mask=mask & hold
                frames[source,h,name]=(mask,y[mask].astype('int8'),
                        (d.loc[mask,'person'] if source=='eicu' else d.loc[mask,'group_id']).astype(str).to_numpy())
            primary_reference.append((source,h,ok))
    stage8c_qc=validate_stage8c(checks,stage8c_ref)
    print('Stage8C complete endpoint scenario grid MATCHED',flush=True)
    # Every model is trained ONLY in the original MIMIC 6h training partition.
    # The same unchanged model evaluates every alternative external label.
    models={}
    for mode in MODES:
        for arm in ARMS:
            mod=make_model(opt.seed)
            mod.fit(design(datasets['mimic',mode],arm,360).loc[train],y6[train])
            models[mode,arm]=mod
            print('Fitted MIMIC 6h',mode,arm,flush=True)
    # IMPORTANT: For each scenario, predict on the exact eligible subset,
    # preserving reference inference batch composition for Stage8B primary gate.
    for source in ('mimic','eicu'):
        for h in HOURS:
            names=[r['scenario'] for r in checks if r['source']==source and r['landmark_h']==h]
            for mode in MODES:
                for arm in ARMS:
                    x=design(datasets[source,mode],arm,h*60)
                    model=models[mode,arm]
                    for scenario in names:
                        mask,y,g=frames[source,h,scenario]
                        p=model.predict_proba(x.loc[mask])[:,1]
                        pred[source,h,scenario,mode,arm]=p
                        stat=metrics(y,p)
                        cohort='internal' if source=='mimic' else 'external'
                        record=dict(source=source,cohort=cohort,landmark_h=h,scenario=scenario,
                                    feature_timing=mode,model=arm,**stat,unique_clusters=len(np.unique(g)))
                        if int(y.sum())>=2 and len(y)-int(y.sum())>=2:
                            record.update(calib(y,p))
                        current_metrics.append(record)
                        if scenario=='stage7e_primary':
                            current_metrics[-1]['is_reference_primary']=True
    frame=pd.DataFrame(current_metrics)
    base=frame.loc[frame.scenario.eq('stage7e_primary')].drop(columns=['scenario']).rename(columns={'feature_timing':'scenario'})
    qc=validate_stage8b(base,pd.read_csv(opt.stage8b_metrics),opt.reference_tolerance)
    print('Stage8B frozen predictions & metric grid MATCHED',flush=True)
    paired=[]; paired_endpoint=[]; summary=[]
    # Bootstrap within endpoint scenario; patient-unit stays clustered by uniquepid.
    for h in HOURS:
        for scenario in [r['scenario'] for r in checks if r['source']=='eicu' and r['landmark_h']==h]:
            mask,y,g=frames['eicu',h,scenario]
            summary.append(dict(landmark_h=h,scenario=scenario,n=len(y),events=int(y.sum()),
                                clusters=len(np.unique(g))))
            for mi,mode in enumerate(MODES):
                pa=pred['eicu',h,scenario,mode,'integrated']
                pb=pred['eicu',h,scenario,mode,'lab']
                print('Patient-cluster paired bootstrap',h,scenario,mode,flush=True)
                values=paired_bootstrap(y,pa,pb,g,opt.bootstrap,opt.seed+17000+h*200+mi*11+len(paired))
                paired.extend(dict(landmark_h=h,scenario=scenario,feature_timing=mode,
                                   contrast='integrated_minus_lab',**val) for val in values)
            # Endpoint sensitivity with common stay population, same predictions,
            # changed label only (and report changed cohort separately).
            if scenario!='stage7e_primary':
                p_mask,_,_=frames['eicu',h,'stage7e_primary']
                overlap=mask & p_mask
                primary_y=risk_labels(original['eicu'],'eicu',h)[1][overlap]
                # align scenario outcomes on the original d row positions
                # Avoid inferred alignment: derive labels by subset positions below.
                pos=np.flatnonzero(mask)
                sub= np.searchsorted(pos,np.flatnonzero(overlap))
                if not np.array_equal(pos[sub],np.flatnonzero(overlap)):
                    raise RuntimeError('FAIL_CLOSED common-cohort positional indexing')
                alt_y=y[sub]
                # Recompute probabilities on overlap: compare common-cohort outcomes
                for mode in MODES:
                    for arm in ARMS:
                        x=design(datasets['eicu',mode],arm,h*60)
                        p=models[mode,arm].predict_proba(x.loc[overlap])[:,1]
                        record=dict(landmark_h=h,scenario=scenario,feature_timing=mode,model=arm,
                                    common_n=int(overlap.sum()),common_changed_labels=int((alt_y!=primary_y).sum()),
                                    removed_primary_n=int((p_mask&~mask).sum()),added_scenario_n=int((mask&~p_mask).sum()))
                        for name,metric in [('auroc',roc_auc_score),('auprc',average_precision_score),('brier',brier_score_loss)]:
                            record['primary_common_'+name]=float(metric(primary_y,p))
                            record['alternative_common_'+name]=float(metric(alt_y,p))
                            record['alternative_minus_primary_'+name]=record['alternative_common_'+name]-record['primary_common_'+name]
                        paired_endpoint.append(record)
        print('Finished',h,'h',flush=True)
    opt.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(checks).to_csv(opt.out/'stage8d_endpoint_reconciliation.csv',index=False)
    qc.to_csv(opt.out/'stage8d_stage8b_reference_gate.csv',index=False)
    stage8c_qc.to_csv(opt.out/'stage8d_stage8c_reference_gate.csv',index=False)
    frame.to_csv(opt.out/'stage8d_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(opt.out/'stage8d_paired_patient_cluster_ci.csv',index=False)
    pd.DataFrame(paired_endpoint).to_csv(opt.out/'stage8d_common_cohort_endpoint_effect.csv',index=False)
    pd.DataFrame(summary).to_csv(opt.out/'stage8d_external_endpoint_summary.csv',index=False)
    manifest=dict(status='EXPLORATORY_ENDPOINT_ROBUSTNESS_NOT_ADJUDICATED_DEATH_TIME',
        stage8b_gate='PASS',stage8c_gate='PASS',seed=opt.seed,bootstrap=opt.bootstrap,
        feature_timing_scenarios=list(MODES),model_arms=list(ARMS),landmarks=list(HOURS),
        models='Fitted only at MIMIC 6h; no retraining or recalibration in eICU',
        no_patient_level_export=True,database_read_only=True,
        interpretation='Hypothetical outcome time perturbations, not observed patient death times',
        ci='Patient cluster bootstrap on each scenario for within-scenario Integrated vs Lab, not adjusted for post-hoc model choice',
        common_cohort='Paired descriptive performance difference induced by changing labels only on intersection; full scenario metrics may also reflect changed eligibility',
        limitations=['eICU death-time uses ICU discharge proxy',
                     'No independent adjudication of exact death time',
                     'eICU was already examined during feature/model selection',
                     'Bootstrap intervals describe conditional sampling uncertainty, not individualized risk intervals',
                     'Not prospective, not a live operational digital twin'],
        inputs_sha256={k:digest(p) for k,p in [('config',opt.config),('stage8b_metrics',opt.stage8b_metrics),
                                             ('stage8c_sensitivity',opt.stage8c_sensitivity)]})
    (opt.out/'stage8d_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('STAGE 8D COMPLETE — exploratory endpoint robustness; reference gates PASS',opt.out,flush=True)

if __name__=='__main__':main()
