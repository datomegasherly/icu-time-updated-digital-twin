#!/usr/bin/env python3
"""Stage 7D. Exploratory paired comparisons at each rolling risk-set and patient-state changes.
Read-only original DB, aggregate outputs. Models fixed from the 6h MIMIC training set.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from icu_stage7c_prospective_landmarks import cohorts, risk_labels, assess, qc_24
from icu_stage7b_frozen_model import VITAL, LAB, ARM, features_sql, design, hash_holdout, make_model, digest

PAIR=[('integrated','lab'),('integrated','vital')]
HOURS=(6,12,18,24)
METRICS=('auroc','auprc','brier')

def paired_bootstrap(y,pa,pb,groups,n,seed):
    y=np.asarray(y,dtype=np.int8);pa=np.asarray(pa,float);pb=np.asarray(pb,float)
    _,idx=np.unique(np.asarray(groups,dtype=str),return_inverse=True)
    nc=int(idx.max())+1; rng=np.random.default_rng(seed)
    obs={'auroc':roc_auc_score(y,pa)-roc_auc_score(y,pb),
         'auprc':average_precision_score(y,pa)-average_precision_score(y,pb),
         'brier':brier_score_loss(y,pa)-brier_score_loss(y,pb)}
    b={m:[] for m in METRICS}
    for k in range(n):
        w=np.bincount(rng.integers(nc,size=nc),minlength=nc)[idx]
        if w[y==1].sum()==0 or w[y==0].sum()==0:continue
        b['auroc'].append(roc_auc_score(y,pa,sample_weight=w)-roc_auc_score(y,pb,sample_weight=w))
        b['auprc'].append(average_precision_score(y,pa,sample_weight=w)-average_precision_score(y,pb,sample_weight=w))
        b['brier'].append(np.average((y-pa)**2-(y-pb)**2,weights=w))
        if (k+1)%100==0:print('  paired bootstrap',k+1,'/',n,flush=True)
    if min(map(len,b.values()))<.9*n:raise RuntimeError('Too many bootstrap replications invalid')
    return [{'metric':m,'delta':float(obs[m]),'ci_low':float(np.quantile(b[m],.025)),
             'ci_high':float(np.quantile(b[m],.975)),'valid_replicates':len(b[m])} for m in METRICS]

def calib(y,p):
    # Strictly diagnostic calibration (no recalibration of predicted risks).
    pp=np.clip(np.asarray(p,dtype=float),1e-8,1-1e-8)
    logit=np.log(pp/(1-pp))
    y=np.asarray(y,dtype=np.int8)
    try:
        # sklearn uses penalized fitting by default; use unpenalized diagnostic fit.
        mdl=LogisticRegression(penalty=None,solver='lbfgs',max_iter=3000,tol=1e-9)
        mdl.fit(logit[:,None],y)
        slope=float(mdl.coef_[0,0]);intercept=float(mdl.intercept_[0])
    except Exception as exc:
        raise RuntimeError(f'Diagnostic calibration fit failed: {exc}') from exc
    return {'cal_slope':slope,'cal_intercept':intercept,
            'mean_pred_risk':float(pp.mean()),'observed_event_rate':float(y.mean()),
            'risk_gt50_n':int((pp>.5).sum())}

def boundary_audit(d,source,reference):
    ok,_=risk_labels(d,source,24)
    current=set(pd.to_numeric(d.loc[ok,'stay_id']).astype('int64'))
    ref=pd.read_csv(reference,usecols=['stay_id','y_24h'])
    archive=set(pd.to_numeric(ref.stay_id).astype('int64'))
    old_only=d[d.stay_id.astype('int64').isin(archive-current)].copy()
    entries=[]
    if source=='mimic':
        # Stage5B used date_diff minute for LOS >= 1440; Stage7C requires exact outtime > 24h.
        delta=(pd.to_datetime(old_only.outtime)-pd.to_datetime(old_only.intime)).dt.total_seconds()
        at24=1440*60
        entries.extend([
            {'source':source,'reason':'actual ICU discharge at or before 24h','n':int((delta<=at24).sum())},
            {'source':source,'reason':'actual ICU discharge after 24h','n':int((delta>at24).sum())},
            {'source':source,'reason':'discharge in final minute before 24h','n':int(((delta>at24-60)&(delta<=at24)).sum())},
            {'source':source,'reason':'missing discharge timestamp','n':int(delta.isna().sum())}])
    else:
        t=pd.to_numeric(old_only.discharge_min,errors='coerce')
        s=old_only.discharge_status.astype('string').str.lower().str.strip()
        entries.extend([
            {'source':source,'reason':'ICU discharge exactly at 24h','n':int((t==1440).sum())},
            {'source':source,'reason':'ICU discharge before 24h','n':int((t<1440).sum())},
            {'source':source,'reason':'ICU discharge after 24h','n':int((t>1440).sum())},
            {'source':source,'reason':'missing discharge offset','n':int(t.isna().sum())},
            {'source':source,'reason':'alive status among archive only','n':int((s=='alive').sum())},
            {'source':source,'reason':'expired status among archive only','n':int((s=='expired').sum())}])
    for x in entries:x['archived_only_total']=len(archive-current)
    return entries

def trajectory(d,source,preds,mask_by_hour):
    # Compare predictions on identical stays present at 6 AND later hour. These
    # are descriptive risk-score changes, NOT performance improvements.
    res=[]
    for arm in ARM:
        p6=preds[(source,arm,6)]
        for h in (12,18,24):
            mask=mask_by_hour[(source,6)] & mask_by_hour[(source,h)]
            v6=p6[mask]; vh=preds[(source,arm,h)][mask]
            diff=vh-v6
            res.append({'source':source,'model':arm,'from_h':6,'to_h':h,
                'common_stays_n':int(mask.sum()),'mean_risk_6h':float(v6.mean()),
                'mean_risk_later':float(vh.mean()),'mean_delta':float(diff.mean()),
                'median_delta':float(np.median(diff)),
                'pct_increased':float((diff>1e-12).mean()),
                'pct_decreased':float((diff< -1e-12).mean()),
                'pct_abs_delta_gt_0p01':float((np.abs(diff)>.01).mean())})
    return res

def main():
    ap=argparse.ArgumentParser()
    for k in ('db','config','expanded-mimic','expanded-eicu','out','stage7c-metrics'):
        ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--bootstrap',type=int,default=300)
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    if args.bootstrap<100:raise ValueError('--bootstrap must be >=100')
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Use a new/empty output folder')
    args.out.mkdir(parents=True,exist_ok=True)
    import duckdb
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    con=duckdb.connect(str(args.db),read_only=True)
    try:
        m,e=cohorts(con)
        qc=[qc_24(m,'mimic',args.expanded_mimic),qc_24(e,'eicu',args.expanded_eicu)]
        pd.DataFrame(qc).to_csv(args.out/'stage7d_reconciliation.csv',index=False)
        if any(x['shared_label_mismatch'] or x['new_only'] for x in qc):
            raise RuntimeError('Unexpected 24h label/entry drift; inspect reconciliation')
        boundaries=boundary_audit(m,'mimic',args.expanded_mimic)+boundary_audit(e,'eicu',args.expanded_eicu)
        pd.DataFrame(boundaries).to_csv(args.out/'stage7d_boundary_audit.csv',index=False)
        print('24h boundary categories:',boundaries,flush=True)
        for tag,d in [('mimic',m),('eicu',e)]:
            con.register('ids_'+tag,pd.DataFrame({'stay_id':pd.to_numeric(d.stay_id).astype('int64')}))
            con.execute(f'CREATE TEMP TABLE {"m" if tag=="mimic" else "e"}_cohort AS SELECT * FROM ids_{tag}')
            additions=[]
            for feature in VITAL+LAB:
                ids=cfg[tag]['vitals'][feature] if feature in VITAL else cfg[tag]['labs'][feature]
                f=con.execute(features_sql(tag,feature,ids)).df()
                if f.stay_id.duplicated().any():raise RuntimeError('Duplicate features')
                merged=d[['stay_id']].merge(f,on='stay_id',how='left',sort=False,validate='one_to_one')
                if not merged.stay_id.equals(d.stay_id):raise RuntimeError('Feature merge order changed')
                additions.append(merged.drop(columns='stay_id'))
                print('Extracted',tag,feature,flush=True)
            d=pd.concat([d.reset_index(drop=True)]+additions,axis=1)
            if tag=='mimic':m=d
            else:e=d
    finally:
        con.close()
    hold=np.asarray([hash_holdout(v,args.seed) for v in m.group_id],dtype=bool)
    ok6,y6=risk_labels(m,'mimic',6)
    train=ok6 & ~hold
    pred={}; mask_by_hour={}; y_by_hour={}; rows=[]; cals=[]
    for source,d in [('internal',m),('external',e)]:
        for h in HOURS:
            good,y=risk_labels(d,'mimic' if source=='internal' else 'eicu',h)
            mask=good & (hold if source=='internal' else np.ones(len(d),bool))
            mask_by_hour[(source,h)]=mask
            y_by_hour[(source,h)]=y[mask]
    for arm in ARM:
        mod=make_model(args.seed)
        x6=design(m,arm,360)
        mod.fit(x6.loc[train],y6[train])
        print('Fitted fixed 6h model:',arm,flush=True)
        for source,d in [('internal',m),('external',e)]:
            for h in HOURS:
                # IMPORTANT: Stage 7C predicts on the eligible subset directly.
                # Batch size can change floating-point accumulation in sklearn/BLAS.
                # Predict on the identical ordered rows to reproduce Stage 7C,
                # rather than predicting for all stays and filtering afterward.
                mask=mask_by_hour[(source,h)]
                y=y_by_hour[(source,h)]
                x=design(d,arm,h*60)
                p_eval=mod.predict_proba(x.loc[mask])[:,1]
                p=np.full(len(d),np.nan,dtype=float)
                p[mask]=p_eval
                pred[(source,arm,h)]=p
                val=assess(y,p_eval); rows.append({'model':arm,'cohort':source,'landmark_h':h,**val})
                cals.append({'model':arm,'cohort':source,'landmark_h':h,**calib(y,p_eval)})
                print('  assessed',arm,source,h,round(val['auroc'],6),flush=True)
    metrics=pd.DataFrame(rows)
    # Stage7C values must reproduce before further inference.
    ref=pd.read_csv(args.stage7c_metrics)
    key=['model','cohort','landmark_h']
    joined=metrics.merge(ref,on=key,suffixes=('_new','_reference'),validate='one_to_one')
    if len(joined)!=24:raise RuntimeError('Stage7C evaluation grid does not match')
    fields=['n','events','auroc','auprc','brier','mean_predicted_risk','pct_risk_gt_50']
    worst=0.
    for name in fields:
        dif=np.abs(joined[name+'_new'].to_numpy()-joined[name+'_reference'].to_numpy())
        worst=max(worst,float(dif.max()))
    if worst>1e-6:raise RuntimeError(f'Stage7C metrics not reproduced: max absolute diff={worst}')
    print('Stage7C metrics validated; worst error',worst,flush=True)
    metrics.to_csv(args.out/'stage7d_metrics.csv',index=False)
    pd.DataFrame(cals).to_csv(args.out/'stage7d_calibration.csv',index=False)
    # External comparisons paired at same hour, same patient risk set and outcomes.
    pairs=[]
    for h in HOURS:
        source='external';mask=mask_by_hour[(source,h)];y=y_by_hour[(source,h)]
        groups=e.loc[mask,'person'].astype(str).to_numpy()
        for ai,bi in PAIR:
            print('External paired contrast',h,ai,bi,flush=True)
            aa=pred[(source,ai,h)][mask]; bb=pred[(source,bi,h)][mask]
            out=paired_bootstrap(y,aa,bb,groups,args.bootstrap,args.seed+h*1000+len(ai)+len(bi))
            for r in out:pairs.append({'landmark_h':h,'cohort':source,'model_a':ai,'model_b':bi,**r})
    pd.DataFrame(pairs).to_csv(args.out/'stage7d_external_paired_ci.csv',index=False)
    trends=trajectory(e,'external',pred,mask_by_hour)+trajectory(m,'internal',pred,mask_by_hour)
    pd.DataFrame(trends).to_csv(args.out/'stage7d_risk_trajectory.csv',index=False)
    manifest={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED',
        'description':'Fixed 6h models applied at rolling landmarks; same-time paired arm comparisons and patient-score changes',
        'n_bootstrap':args.bootstrap,'seed':args.seed,'reproduction_max_absolute_error':worst,
        'risk_trajectory_note':'Common surviving/observable stays only, no fixed target labels; do not interpret changes as outcome or longitudinal performance effects',
        'strict_cohort_note':'24h boundary differences retained and disclosed; not forcibly matched to Stage5B',
        'no_patient_level_export':True,'read_only_db':True,'post_hoc_external_selection':True,
        'limitations':['Retrospective predictive surrogate, not a synchronized clinical digital twin',
          'eICU discharge time is death-time proxy','Site and patient overlap may affect uncertainty',
          'Models selected after prior eICU exploratory results; confidence intervals not adjusted for selection',
          'Rolling landmark cohorts and 24h targets change over time'],
        'source_hashes':{k:digest(p) for k,p in {'db':args.db,'config':args.config,'stage7c_metrics':args.stage7c_metrics,'stage5b_mimic':args.expanded_mimic,'stage5b_eicu':args.expanded_eicu}.items() if k!='db'},
        'db_size_bytes':args.db.stat().st_size}
    (args.out/'stage7d_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('STAGE 7D COMPLETE — EXPLORATORY DIAGNOSTICS ONLY',flush=True)

if __name__=='__main__':main()
