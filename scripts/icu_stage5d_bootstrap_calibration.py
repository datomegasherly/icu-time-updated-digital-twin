#!/usr/bin/env python3
"""Stage 5D: reproducible paired bootstrap & calibration diagnostics (read-only).
Recreates Stage 5C fitting on the exact provided restricted/expanded cohorts.
No patient-level predictions exported. The eICU group ID may not uniquely identify a patient.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from stage5c_model_helpers import ARM, design, hash_holdout, metrics, digest


def calibration(y, p):
    """Logistic recalibration: logit(Y) = intercept + slope*logit(pred)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p/(1-p)).reshape(-1,1)
    y = np.asarray(y,dtype=int)
    if len(np.unique(y))<2:
        return {'intercept': None, 'slope': None}
    try:
        from sklearn.linear_model import LogisticRegression as LR
        # A practically unpenalized logistic calibration model (not clinical recalibration).
        reg=LR(C=1e9, solver='lbfgs',max_iter=300,tol=1e-8)
        reg.fit(z,y)
        return {'intercept':float(reg.intercept_[0]), 'slope':float(reg.coef_[0,0])}
    except Exception as ex:
        return {'intercept':None,'slope':None,'warning':str(ex)}


def paired_bootstrap(y,preds,pairs,B,seed):
    y=np.asarray(y,dtype=np.int8)
    N=len(y)
    rg=np.random.default_rng(seed)
    retained={label:{m:[] for m in ('auroc','auprc','brier')} for label in preds}
    changes={label:{m:[] for m in ('auroc','auprc','brier')} for label in pairs}
    valid=0
    for b in range(B):
        sel=rg.integers(0,N,N)
        yy=y[sel]
        if yy.min()==yy.max(): continue
        valid+=1
        vals={}
        for label,p in preds.items():
            pp=p[sel]
            vals[label]={'auroc':roc_auc_score(yy,pp),'auprc':average_precision_score(yy,pp),'brier':brier_score_loss(yy,pp)}
            for k,v in vals[label].items():retained[label][k].append(float(v))
        for label,(a,c) in pairs.items():
            for k in ('auroc','auprc','brier'):
                changes[label][k].append(float(vals[a][k]-vals[c][k]))
        if (b+1)%100==0:print('  bootstrap',b+1,'/',B,flush=True)
    if valid<max(50,int(B*.90)):raise ValueError('Too many bootstrap replicates have one outcome class')
    def summarize(values):
        a=np.asarray(values)
        return {'lower':float(np.quantile(a,.025)),'upper':float(np.quantile(a,.975))}
    return {'valid_resamples':valid,'metrics_ci':{k:{m:summarize(v) for m,v in d.items()} for k,d in retained.items()},'paired_delta_ci':{k:{m:summarize(v) for m,v in d.items()} for k,d in changes.items()}}


def run(args):
    keys=('restricted_mimic','restricted_eicu','expanded_mimic','expanded_eicu')
    inp={k:pd.read_csv(getattr(args,k),low_memory=False) for k in keys}
    for k,d in inp.items():
        required={'stay_id','group_id','y_24h','age','sex'}
        if required-set(d.columns):raise ValueError(f'{k}: missing fields {required-set(d.columns)}')
        if d.stay_id.isna().any() or d.stay_id.duplicated().any():raise ValueError(f'{k}: stay_id not unique')
        if d.group_id.isna().any():raise ValueError(f'{k}: missing group ids')
        if not d.y_24h.isin([0,1]).all():raise ValueError(f'{k}: invalid outcomes')
    for source in ('mimic','eicu'):
        r=inp['restricted_'+source].set_index('stay_id')
        e=inp['expanded_'+source].set_index('stay_id')
        if not r.index.isin(e.index).all():raise ValueError('Restricted cohort not contained in expanded cohort')
        if not (r.y_24h.values == e.reindex(r.index).y_24h.values).all():raise ValueError('Shared label mismatch')
        if e.loc[~e.index.isin(r.index),'y_24h'].sum():raise ValueError('Unexpected added positive event')
    hold={k:np.asarray([hash_holdout(v,args.seed) for v in inp[k].group_id],dtype=bool) for k in ('restricted_mimic','expanded_mimic')}
    evals=['restricted_mimic','expanded_mimic','restricted_eicu','expanded_eicu']
    outcomes={}
    predictions={k:{} for k in evals}
    for ev in evals:
        dd=inp[ev]
        mask=hold[ev] if ev.endswith('mimic') else np.ones(len(dd),dtype=bool)
        outcomes[ev]=dd.loc[mask,'y_24h'].to_numpy(dtype=int)
    allrows=[]
    for tr in ('restricted','expanded'):
        data=inp[tr+'_mimic']; mask=~hold[tr+'_mimic']
        for arm in ARM:
            model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler()),('lr',LogisticRegression(C=1.,solver='liblinear',max_iter=2000,random_state=args.seed))])
            model.fit(design(data,arm).loc[mask],data.loc[mask,'y_24h'].to_numpy(dtype=int))
            for ev in evals:
                d=inp[ev]
                test=hold[ev] if ev.endswith('mimic') else np.ones(len(d),dtype=bool)
                pp=model.predict_proba(design(d,arm).loc[test])[:,1]
                label=tr+'_'+arm
                predictions[ev][label]=pp
                y=outcomes[ev]
                stat=metrics(y,pp)
                baseline=float(y.mean()*(1-y.mean()))
                cal=calibration(y,pp)
                allrows.append({'trained_on':tr,'model':arm,'evaluated_on':ev,**stat,'prevalence_brier_reference':baseline,'brier_skill_vs_observed_prevalence':1-stat['brier']/baseline,'calibration_intercept':cal['intercept'],'calibration_slope':cal['slope']})
            print('Trained',tr,arm,flush=True)
    # In each evaluation set, compare models on precisely the same labels/subjects.
    pairs={}
    for ev in evals:
        spec={}
        for arm in ARM:
            spec['training_expanded_minus_restricted_'+arm]=('expanded_'+arm,'restricted_'+arm)
        for tr in ('restricted','expanded'):
            spec[tr+'_lab_minus_integrated']=(tr+'_lab',tr+'_integrated')
            spec[tr+'_integrated_minus_vital']=(tr+'_integrated',tr+'_vital')
        pairs[ev]=spec
    delta=[]; bootstrap={}
    for i,ev in enumerate(evals):
        p=predictions[ev]; y=outcomes[ev]
        for label,(one,two) in pairs[ev].items():
            m1=metrics(y,p[one]);m2=metrics(y,p[two])
            delta.append({'evaluated_on':ev,'comparison':label,'delta_auroc':m1['auroc']-m2['auroc'],'delta_auprc':m1['auprc']-m2['auprc'],'delta_brier':m1['brier']-m2['brier']})
        print('Paired bootstrap for',ev,flush=True)
        bootstrap[ev]=paired_bootstrap(y,p,pairs[ev],args.bootstrap,args.seed+100+i)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    paths=[out/'stage5d_metrics.csv',out/'stage5d_paired_deltas.csv',out/'stage5d_bootstrap_ci.json',out/'stage5d_manifest.json']
    if any(p.exists() for p in paths):raise FileExistsError('Choose a fresh output folder, files already exist')
    pd.DataFrame(allrows).to_csv(paths[0],index=False)
    pd.DataFrame(delta).to_csv(paths[1],index=False)
    paths[2].write_text(json.dumps(bootstrap,indent=2),encoding='utf-8')
    manifest={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED','seed':args.seed,'bootstrap':args.bootstrap,'input_sha256':{k:digest(getattr(args,k)) for k in keys},'counts':{ev:{'n':len(outcomes[ev]),'events':int(outcomes[ev].sum())} for ev in evals},'design':'same Stage5C grouping, preprocessing, 3 arms without temperature; both training cohorts x both evaluation cohorts','bootstrap_unit':'evaluation rows; eICU patient-level uniqueness not established','method':'paired percentile bootstrap; all model probabilities are held fixed (no training resampling)','warnings':['eICU death time remains proxy','eICU person-level linkage unverified','calibration slope/intercept descriptive, not applied to predictions','observed-prevalence reference is descriptive, not deployable','bootstrap across different cohort memberships is not a paired causal comparison']}
    paths[3].write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('STAGE 5D COMPLETE',out,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('restricted_mimic','restricted_eicu','expanded_mimic','expanded_eicu'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=500)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    if args.bootstrap < 100: parser.error('Use at least 100 bootstrap repetitions')
    run(args)
