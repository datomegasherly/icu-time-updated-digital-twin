#!/usr/bin/env python3
"""Stage 6D: Post-hoc no-count vs count comparisons; read-only, aggregate only.

Reproduces Stage6C models exactly; paired patient-cluster bootstrap for no-count
Integrated vs Lab, diagnostic calibration and site-level heterogeneity.
NOT independent external validation: eICU has already informed the ablation choice.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from stage5c_model_helpers import ARM, design, hash_holdout, metrics, digest

OUT = ['stage6d_metrics.csv','stage6d_paired_comparisons.csv',
       'stage6d_cluster_ci.json','stage6d_calibration_bins.csv',
       'stage6d_hospital_summary.json','stage6d_hospital_distribution.csv',
       'stage6d_hospital_performance.csv','stage6d_manifest.json']


def links(db, stay_ids):
    try: import duckdb
    except ImportError as exc: raise RuntimeError('Install duckdb with python -m pip install duckdb') from exc
    con=duckdb.connect(str(db),read_only=True)
    try:
        # Raw table has one row per unit stay. Stay ordering is restored in the merge.
        raw=con.execute('SELECT patientunitstayid, uniquepid, hospitalid FROM eicu_patient').fetchdf()
    finally: con.close()
    if raw.patientunitstayid.isna().any() or raw.patientunitstayid.duplicated().any():
        raise ValueError('Raw eICU has null/duplicate stay IDs')
    ids=pd.DataFrame({'stay_id':pd.to_numeric(stay_ids,errors='raise').astype('int64')})
    raw['patientunitstayid']=pd.to_numeric(raw.patientunitstayid,errors='raise').astype('int64')
    result=ids.merge(raw,left_on='stay_id',right_on='patientunitstayid',how='left',sort=False,
                     validate='one_to_one',indicator=True)
    pid=result.uniquepid.astype('string').str.strip()
    if not result['_merge'].eq('both').all() or pid.isna().any() or pid.eq('').any():
        raise ValueError('FAIL CLOSED: missing patient linkage')
    hosp=result.hospitalid.astype('string').str.strip().replace('',pd.NA)
    if hosp.isna().any():
        raise ValueError('FAIL CLOSED: missing hospital identifiers; inspect source')
    return pid.to_numpy(dtype=str),hosp.to_numpy(dtype=str)


def make_model(seed):
    return Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                     ('scale',StandardScaler()),
                     ('lr',LogisticRegression(C=1.,solver='liblinear',max_iter=2000,random_state=seed))])


def design_v(d,arm,variant):
    x=design(d,arm)
    if variant=='drop_all_counts':
        x=x.drop(columns=[c for c in x.columns if c.endswith('_n')])
    return x


def calibrate(y,p):
    eps=1e-12
    q=logit(np.clip(np.asarray(p,float),eps,1-eps))
    yy=np.asarray(y,dtype=float)
    prevalence=float(yy.mean())
    # Logistic recalibration: observed logit = intercept + slope * predicted logit
    def objective(theta):
        eta=theta[0]+theta[1]*q
        return float(np.sum(np.logaddexp(0,eta)-yy*eta))
    def gradient(theta):
        resid=expit(theta[0]+theta[1]*q)-yy
        return np.array([resid.sum(),(resid*q).sum()])
    x0=[float(logit(np.clip(prevalence,eps,1-eps))),0.0]
    fit=minimize(objective,x0,jac=gradient,method='L-BFGS-B',options={'maxiter':500})
    if not fit.success and np.linalg.norm(gradient(fit.x))>max(1e-4,len(y)*1e-7):
        raise RuntimeError('Calibration logistic regression failed: '+fit.message)
    # Intercept-in-the-large is an offset logistic fit with slope fixed at 1.
    cal_large=float(brentq(lambda v: np.mean(expit(q+v))-prevalence,-100,100))
    return {'calibration_intercept':float(fit.x[0]),'calibration_slope':float(fit.x[1]),
            'calibration_in_the_large':cal_large}


def weighted_metric(y,p,w,metric):
    if metric=='auroc':return float(roc_auc_score(y,p,sample_weight=w))
    if metric=='auprc':return float(average_precision_score(y,p,sample_weight=w))
    if metric=='brier':return float(brier_score_loss(y,p,sample_weight=w))
    raise ValueError(metric)


def contrasts():
    return {
      'integrated_no_counts_minus_lab_no_counts':('integrated:drop_all_counts','lab:drop_all_counts'),
      'integrated_no_counts_minus_vital_no_counts':('integrated:drop_all_counts','vital:drop_all_counts'),
      'integrated_baseline_minus_lab_baseline':('integrated:baseline','lab:baseline'),
      'integrated_no_counts_minus_integrated_baseline':('integrated:drop_all_counts','integrated:baseline'),
    }


def cluster_ci(y,preds,group_ids,b,seed):
    # Draw patient clusters, not independent ICU stays; all contrasts use same sample.
    _,inverse=np.unique(group_ids.astype(str),return_inverse=True)
    ng=int(inverse.max())+1
    rng=np.random.default_rng(seed)
    cs=contrasts()
    vals={c:{m:[] for m in ('auroc','auprc','brier')} for c in cs}
    skipped=0
    for i in range(b):
        counts=np.bincount(rng.integers(ng,size=ng),minlength=ng)
        w=counts[inverse].astype(float)
        if w[y==1].sum()==0 or w[y==0].sum()==0:
            skipped+=1;continue
        computed={name:{m:weighted_metric(y,preds[name],w,m) for m in ('auroc','auprc','brier')}
                  for name in sorted({k for pair in cs.values() for k in pair})}
        for name,(a,bline) in cs.items():
            for m in ('auroc','auprc','brier'):
                vals[name][m].append(computed[a][m]-computed[bline][m])
        if (i+1)%100==0:print(f'  cluster bootstrap {i+1}/{b}',flush=True)
    if b-skipped<max(30,int(0.9*b)):
        raise RuntimeError('Too many bootstrap draws with only one outcome class')
    return {'clusters':ng,'requested':b,'valid':b-skipped,'skipped':skipped,
       'paired_difference_ci':{name:{m:{'lower':float(np.quantile(v,.025)),'upper':float(np.quantile(v,.975))}
                                      for m,v in mm.items()} for name,mm in vals.items()}}


def risk_bins(y,p,cohort,arm,variant):
    # Fixed clinically readable bands. No event counts below per-site threshold are exported.
    boundaries=[0,.001,.005,.01,.02,.05,.10,.20,.50,1.0000001]
    rows=[]
    for low,high in zip(boundaries[:-1],boundaries[1:]):
        mask=(p>=low)&(p<high)
        if not mask.any(): continue
        rows.append({'cohort':cohort,'model':arm,'variant':variant,
                     'risk_from':low,'risk_to':min(high,1.),'n':int(mask.sum()),
                     'events':int(y[mask].sum()),'mean_predicted':float(p[mask].mean()),
                     'observed_rate':float(y[mask].mean())})
    return rows


def hospital_results(y,preds,hosp,min_stays,min_events):
    names=np.asarray(hosp,str)
    sizes=pd.DataFrame({'site':names,'y':y}).groupby('site',sort=True).y.agg(stays='size',events='sum')
    size_summary=[]
    for name,low,high in [('under_100',0,100),('100_to_499',100,500),('500_to_999',500,1000),('1000_plus',1000,10**10)]:
        sample=sizes.loc[(sizes.stays>=low)&(sizes.stays<high)]
        size_summary.append({'size_bucket':name,'hospitals':int(len(sample)),
                             'stays':int(sample.stays.sum()),'events':int(sample.events.sum())})
    eligible=sizes.loc[(sizes.stays>=min_stays)&(sizes.events>=min_events)&
                       ((sizes.stays-sizes.events)>=min_events)]
    rows=[]
    for site,stats in eligible.iterrows():
        mask=names==site
        vals={}
        for arm in ARM:
            p=preds[f'{arm}:drop_all_counts'][mask]
            vals[f'{arm}_auroc']=float(roc_auc_score(y[mask],p))
            vals[f'{arm}_auprc']=float(average_precision_score(y[mask],p))
            vals[f'{arm}_brier']=float(brier_score_loss(y[mask],p))
        rows.append({'stays':int(stats.stays),'events':int(stats.events),
                     'integrated_minus_lab_auroc':vals['integrated_auroc']-vals['lab_auroc'],**vals})
    per_site=pd.DataFrame(rows)
    if not per_site.empty:
        per_site=per_site.sort_values(['stays','events'],ascending=False).reset_index(drop=True)
        per_site.insert(0,'site_label',[f'Site_{i:03d}' for i in range(1,len(per_site)+1)])
    base={'total_hospitals':int(len(sizes)),'eligible_hospitals':int(len(eligible)),
          'suppressed_hospitals':int(len(sizes)-len(eligible)),
          'reporting_stays':int(eligible.stays.sum()),'reporting_events':int(eligible.events.sum()),
          'thresholds':{'min_stays':min_stays,'min_events':min_events,'min_nonevents':min_events},
          'site_ids_anonymized':True}
    if not per_site.empty:
        d=per_site.integrated_minus_lab_auroc
        base['integrated_vs_lab']={'integrated_better':int((d>0).sum()),
            'lab_better':int((d<0).sum()),'tie':int((d==0).sum()),
            'median_delta':float(d.median()),'q25_delta':float(d.quantile(.25)),
            'q75_delta':float(d.quantile(.75)),
            'integrated_lower_brier_sites':int((per_site.integrated_brier<per_site.lab_brier).sum())}
    return base,pd.DataFrame(size_summary),per_site


def main():
    a=argparse.ArgumentParser(description=__doc__)
    for name in ('expanded-mimic','expanded-eicu','db','stage6c-metrics','out'):
        a.add_argument('--'+name,required=True,type=Path)
    a.add_argument('--seed',default=2026,type=int)
    a.add_argument('--bootstrap',default=1000,type=int)
    a.add_argument('--min-stays',default=100,type=int)
    a.add_argument('--min-events',default=10,type=int)
    args=a.parse_args()
    if args.bootstrap<100:a.error('--bootstrap must be >=100')
    if args.min_stays<100 or args.min_events<10:a.error('Hospital reporting thresholds must be >=100 stays and >=10 events')
    for p in (args.expanded_mimic,args.expanded_eicu,args.db,args.stage6c_metrics):
        if not p.is_file():raise FileNotFoundError(p)
    if any((args.out/name).exists() for name in OUT):
        raise FileExistsError('Output already exists: choose a fresh --out path')
    mimic=pd.read_csv(args.expanded_mimic,low_memory=False)
    eicu=pd.read_csv(args.expanded_eicu,low_memory=False)
    for name,d in (('mimic',mimic),('eicu',eicu)):
        needed={'stay_id','group_id','sex','age','y_24h'}
        if needed-set(d.columns):raise ValueError(name+': missing '+str(needed-set(d.columns)))
        if d.stay_id.isna().any() or d.stay_id.duplicated().any():raise ValueError(name+': duplicate/null stay IDs')
        if not d.y_24h.isin([0,1]).all() or d.group_id.isna().any():raise ValueError(name+': invalid label/group')
    hold=np.array([hash_holdout(g,args.seed) for g in mimic.group_id],dtype=bool)
    train=mimic.loc[~hold].reset_index(drop=True)
    internal=mimic.loc[hold].reset_index(drop=True)
    if train.y_24h.nunique()!=2 or internal.y_24h.nunique()!=2 or eicu.y_24h.nunique()!=2:
        raise ValueError('All cohorts must have positive and negative outcomes')
    source=pd.read_csv(args.stage6c_metrics)
    if not {'cohort','model','variant','auroc','auprc','brier'}.issubset(source.columns):
        raise ValueError('Stage6C reference is malformed')
    pid,hosp=links(args.db,eicu.stay_id)
    cohorts={'internal':internal,'external':eicu}
    predictions={c:{} for c in cohorts}
    rows=[];bins=[];verification=[]
    for arm in ARM:
        for variant in ('baseline','drop_all_counts'):
            xx=design_v(train,arm,variant)
            model=make_model(args.seed)
            model.fit(xx,train.y_24h.to_numpy(dtype=int))
            for cname,d in cohorts.items():
                x=design_v(d,arm,variant)
                if list(x.columns)!=list(xx.columns):raise ValueError('Feature mismatch')
                y=d.y_24h.to_numpy(dtype=int)
                p=model.predict_proba(x)[:,1]
                preds=predictions[cname]
                preds[f'{arm}:{variant}']=p
                met=metrics(y,p)
                original=source.loc[(source.cohort==cname)&(source.model==arm)&(source.variant==variant)]
                if len(original)!=1:raise ValueError('Missing reference for '+cname+':'+arm+':'+variant)
                delta=max(abs(met[k]-float(original.iloc[0][k])) for k in ('auroc','auprc','brier'))
                if delta>1e-5:raise RuntimeError(f'FAIL CLOSED: Stage6C mismatch {cname}/{arm}/{variant}: {delta}')
                verification.append({'cohort':cname,'model':arm,'variant':variant,'maximum_abs_delta':delta})
                cal=calibrate(y,p)
                rows.append({'cohort':cname,'model':arm,'variant':variant,**met,**cal,
                             'n_pred_gt50':int((p>.5).sum()),'pct_pred_gt50':float((p>.5).mean()*100),
                             'p99_predicted':float(np.quantile(p,.99))})
                bins.extend(risk_bins(y,p,cname,arm,variant))
            print('Validated Stage6C and calibrated',arm,variant,flush=True)
    comparisons=[];cis={}
    cs=contrasts()
    for cname,d in cohorts.items():
        y=d.y_24h.to_numpy(dtype=int);preds=predictions[cname]
        for name,(a,b) in cs.items():
            first=metrics(y,preds[a]);second=metrics(y,preds[b])
            comparisons.append({'cohort':cname,'contrast':name,**{f'delta_{m}':first[m]-second[m] for m in ('auroc','auprc','brier')}})
        grp=internal.group_id.astype('string').to_numpy(dtype=str) if cname=='internal' else pid
        print('Patient-cluster bootstrap',cname,flush=True)
        cis[cname]=cluster_ci(y,preds,grp,args.bootstrap,args.seed+(50 if cname=='internal' else 60))
    summary,dist,site=hospital_results(eicu.y_24h.to_numpy(dtype=int),predictions['external'],hosp,args.min_stays,args.min_events)
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out/OUT[0],index=False)
    pd.DataFrame(comparisons).to_csv(args.out/OUT[1],index=False)
    (args.out/OUT[2]).write_text(json.dumps(cis,indent=2),encoding='utf-8')
    pd.DataFrame(bins).to_csv(args.out/OUT[3],index=False)
    (args.out/OUT[4]).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    dist.to_csv(args.out/OUT[5],index=False);site.to_csv(args.out/OUT[6],index=False)
    manifest={'status':'POST_HOC_EXPLORATORY_NOT_SUBMISSION_APPROVED',
              'warning':'eICU has been inspected repeatedly; no-count variant is post-hoc selected on external results. These estimates are not independent validation of this choice.',
              'assumption':'ICU death in (24h,48h]; eICU uses discharge status/offset as proxy',
              'model':'Exact Stage 6C design / logistic liblinear C=1 / no external fitting',
              'seed':args.seed,'bootstrap':args.bootstrap,'patient_cluster_external':True,
              'count_ablation_only':'No temperature in any arm; baseline retains _n, ablated removes all _n',
              'input_sha256':{'mimic':digest(args.expanded_mimic),'eicu':digest(args.expanded_eicu),
                              'stage6c_reference':digest(args.stage6c_metrics)},
              'n':{'train':int(len(train)),'internal':int(len(internal)),
                   'internal_events':int(internal.y_24h.sum()),'external':int(len(eicu)),
                   'external_events':int(eicu.y_24h.sum()),'external_unique_patients':int(len(np.unique(pid)))},
              'stage6c_reproduction':verification,
              'limits':['Calibration fits on external labels are descriptive only; not deployed recalibration',
                        'No model refitting within bootstrap; intervals conditional on trained model',
                        'Hospital reporting restricted by sample/event thresholds',
                        'Site-level differences descriptive, not significance tests',
                        'External endpoint timestamp is proxy, not verified death time',
                        'No independent external cohort for ablation-selected model']}
    (args.out/OUT[7]).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('STAGE 6D COMPLETE (post-hoc aggregate diagnostics only):',args.out,flush=True)
    print(json.dumps({'n':manifest['n'],'hospital_summary':summary},indent=2),flush=True)

if __name__=='__main__':main()
