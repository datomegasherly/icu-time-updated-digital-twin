#!/usr/bin/env python3
"""Stage 6A: read-only validation of absolute risk calibration against Stage 5G.

Re-fits exactly the Stage 5G prespecified no-temperature pipelines on expanded MIMIC.
Writes only aggregate tables, plots, and run manifest; no row-level exports.
External eICU values never used for fitting or recalibration.
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import brentq
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage5c_model_helpers import ARM, design, hash_holdout, metrics, digest

OUTPUTS=['stage6a_calibration_metrics.csv','stage6a_risk_bins.csv','stage6a_summary.json','stage6a_manifest.json', 'stage6a_reliability.png','stage6a_risk_distribution.png']

def fit_calibration(y,p):
    """Logistic calibration: y ~ intercept + slope*logit(p).
    Calibration-in-the-large: solve intercept with fixed logit(p) coefficient 1.
    These are diagnostics; no recalibration is applied to predictions.
    """
    eps=1e-7; lp=logit(np.clip(np.asarray(p,float),eps,1-eps)); y=np.asarray(y,int)
    prevalence=float(np.mean(y))
    if not (0<prevalence<1):return {'calibration_intercept':None,'calibration_slope':None,'calibration_in_large':None}
    cal_in_large=float(brentq(lambda a:float(np.mean(expit(a+lp)))-prevalence,-100,100))
    x=np.column_stack((np.ones(len(lp)),lp))
    # A logistic recalibration diagnostic; no regularization in this diagnostic fit.
    from scipy.optimize import minimize
    from scipy.special import log_expit
    def nll_grad(z):
        s=z[0]+z[1]*lp
        loss=float(np.sum(np.logaddexp(0,s)-y*s))
        grad=np.array([np.sum(expit(s)-y),np.dot(expit(s)-y,lp)],dtype=float)
        return loss,grad
    opt=minimize(nll_grad, np.array([0.,1.]), jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12})
    if not opt.success and np.max(np.abs(opt.jac))>1e-3:raise RuntimeError('Calibration diagnostic regression did not converge: '+str(opt.message))
    return {'calibration_intercept':float(opt.x[0]),'calibration_slope':float(opt.x[1]),'calibration_in_large':cal_in_large}

def make_bins(y,p,cohort,arm,scheme):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if scheme=='fixed':
        edges=np.linspace(0,1,11)
        codes=np.clip(np.searchsorted(edges,p,side='right')-1,0,9)
        label=lambda k:f'[{edges[k]:.1f},{edges[k+1]:.1f}{"]" if k==9 else ")"}'
    else:
        # Rank-based equal-frequency groups, robust to tied probabilities.
        ranks=pd.Series(p).rank(method='first').to_numpy()-1
        codes=np.floor(ranks*10/len(p)).astype(int).clip(0,9)
        label=lambda k:f'Decile {k+1}'
    out=[]
    for k in range(10):
        take=codes==k
        if not np.any(take): continue
        out.append({'cohort':cohort,'model':arm,'scheme':scheme,'bin':k+1,'bin_label':label(k),
                    'n':int(take.sum()),'events':int(y[take].sum()),'observed_rate':float(y[take].mean()),
                    'mean_prediction':float(p[take].mean()),'min_prediction':float(p[take].min()),
                    'max_prediction':float(p[take].max())})
    return out

def compare_reference(actual,ref,cohort,arm,atol):
    r=ref[(ref.cohort==cohort)&(ref.model==arm)]
    if len(r)!=1:raise ValueError(f'Stage 5G reference missing/duplicate: {cohort}/{arm}')
    for key in ('n','events','prevalence','auroc','auprc','brier','ece10','mean_predicted'):
        expected=float(r.iloc[0][key]);got=float(actual[key])
        if not np.isclose(expected,got,rtol=0,atol=atol):
            raise ValueError(f'Stage 5G mismatch {cohort}/{arm}/{key}: Stage 5G {expected}, Stage 6A {got} (tolerance {atol})')

def run(mimic_path,eicu_path,ref_path,out,seed=2026,atol=1e-5,expected_manifest=None):
    for path in (mimic_path,eicu_path,ref_path):
        if not Path(path).is_file():raise FileNotFoundError(path)
    if expected_manifest:
        reference=json.loads(Path(expected_manifest).read_text(encoding='utf-8'))
        sha=reference.get('input_sha256',{})
        for name,path in [('mimic',mimic_path),('eicu',eicu_path)]:
            if sha.get(name)!=digest(path):raise ValueError(f'{name} input SHA256 mismatches Stage 5G manifest')
        if int(reference['seed'])!=seed:raise ValueError('Seed mismatches Stage 5G')
    out=Path(out)
    if any((out/x).exists() for x in OUTPUTS):raise FileExistsError('Choose a fresh output directory; files already exist')
    m=pd.read_csv(mimic_path,low_memory=False);e=pd.read_csv(eicu_path,low_memory=False)
    ref=pd.read_csv(ref_path)
    for name,d in [('MIMIC',m),('eICU',e)]:
        required={'stay_id','group_id','age','sex','y_24h'}
        if required-set(d.columns):raise ValueError(f'{name} required columns missing: {required-set(d.columns)}')
        if d.stay_id.isna().any() or d.stay_id.duplicated().any():raise ValueError(f'{name} stay_id missing/duplicate')
        if d.group_id.isna().any():raise ValueError(f'{name} group_id missing')
        if not d.y_24h.isin([0,1]).all():raise ValueError(f'{name} nonbinary outcome')
    hold=np.array([hash_holdout(g,seed) for g in m.group_id],bool)
    subsets={'internal':m.loc[hold].reset_index(drop=True),'external':e.reset_index(drop=True)}
    rows=[];bins=[];pred={}
    for arm in ARM:
        model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                        ('scale',StandardScaler()),('lr',LogisticRegression(C=1.,solver='liblinear',max_iter=2000,random_state=seed))])
        model.fit(design(m,arm).loc[~hold],m.loc[~hold,'y_24h'].to_numpy(dtype=int))
        for cohort,d in subsets.items():
            y=d.y_24h.to_numpy(dtype=int);p=model.predict_proba(design(d,arm))[:,1]
            actual=metrics(y,p);compare_reference(actual,ref,cohort,arm,atol)
            pred[(cohort,arm)]=p
            rows.append({'cohort':cohort,'model':arm,**actual,**fit_calibration(y,p),
                         'median_predicted':float(np.median(p)), 'p90_predicted':float(np.quantile(p,.9)),
                         'p99_predicted':float(np.quantile(p,.99)),
                         'fraction_risk_gt_10pct':float((p>0.1).mean()),'fraction_risk_gt_50pct':float((p>0.5).mean()),
                         'prevalence_brier_reference':float(y.mean()*(1-y.mean()))})
            bins.extend(make_bins(y,p,cohort,arm,'fixed'))
            bins.extend(make_bins(y,p,cohort,arm,'quantile'))
        print('Validated against Stage 5G:',arm,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/OUTPUTS[0],index=False)
    pd.DataFrame(bins).to_csv(out/OUTPUTS[1],index=False)
    # Fixed clinical reference line and same axes for visual comparability.
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for ax,cohort in zip(axes,('internal','external')):
        ax.plot([0,1],[0,1],ls='--',c='0.5',label='Perfect calibration')
        for arm in ARM:
            sub=[b for b in bins if b['cohort']==cohort and b['model']==arm and b['scheme']=='quantile']
            ax.plot([b['mean_prediction'] for b in sub],[b['observed_rate'] for b in sub],marker='o',ms=3,label=arm)
        ax.set(xlabel='Mean predicted risk',ylabel='Observed event rate',title=cohort+' (quantile bins)',xlim=(0,0.3 if cohort=='internal' else 0.8),ylim=(0,0.3 if cohort=='internal' else 0.8))
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/OUTPUTS[4],dpi=220);plt.close(fig)
    # Risk histograms with log y-axis avoid visually hiding rare high-risk predictions.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,cohort in zip(axes,('internal','external')):
        for arm in ARM:
            ax.hist(pred[(cohort,arm)],bins=np.linspace(0,1,41),histtype='step',linewidth=1.5,label=arm)
        ax.set(xlabel='Predicted event probability',ylabel='Count (log scale)',yscale='log',title=cohort)
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/OUTPUTS[5],dpi=220);plt.close(fig)
    summary={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED','stage5g_reproduction':'PASS_WITH_NUMERIC_TOLERANCE',
             'warning':'Calibration diagnostics are descriptive fits to evaluation outcomes, not a deployed recalibration.',
             'clinical_warning':'eICU unit-discharge proxy does not verify exact death time.',
             'rows':[{k:r[k] for k in ('cohort','model','calibration_intercept','calibration_slope','calibration_in_large','mean_predicted','prevalence','brier','ece10')} for r in rows]}
    (out/OUTPUTS[2]).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    manifest={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED','method':'Stage 5G equivalent training; floating point implementations compared with explicit tolerance; no external model refit or recalibration',
              'input_sha256':{'mimic':digest(mimic_path),'eicu':digest(eicu_path),'reference':digest(ref_path)},
              'seed':seed,'metrics_tolerance':atol,'temperature':'Excluded (Stage 5G primary)',
              'binning':['fixed 10 equal-width bins','rank-based 10 equal-size bins (ties broken by row order)'],
              'calibration':'logistic intercept/slope and offset-based calibration-in-the-large; probability clipping 1e-7 for logits',
              'nonidentifying_outputs_only':True,'n_internal':len(subsets['internal']),'n_external':len(subsets['external']),
              'known_limitations':['eICU discharge-time proxy','hospital heterogeneity','evaluation calibration diagnostics are not independent validation of a recalibrated model']}
    (out/OUTPUTS[3]).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('STAGE 6A COMPLETE — all Stage 5G metrics reproduced. Output:',out)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    for name in ('expanded-mimic','expanded-eicu','stage5g-metrics','out'):
        a.add_argument('--'+name,required=True,type=Path)
    a.add_argument('--stage5g-manifest',type=Path)
    a.add_argument('--seed',type=int,default=2026)
    a.add_argument('--tolerance',type=float,default=1e-5)
    z=a.parse_args()
    run(z.expanded_mimic,z.expanded_eicu,z.stage5g_metrics,z.out,z.seed,z.tolerance,z.stage5g_manifest)
