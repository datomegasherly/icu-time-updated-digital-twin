#!/usr/bin/env python3
"""Read-only Stage 6B: cross-database feature shift and extreme-risk attribution.

No patient-level outputs. No recalibration or external training. Descriptive diagnostics,
not a causal explanation or manuscript-approved findings.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from stage5c_model_helpers import ARM, design, hash_holdout, metrics, digest

OUTFILES=['stage6b_feature_shift.csv','stage6b_extreme_prediction_summary.csv',
          'stage6b_logit_contributors.csv','stage6b_manifest.json']

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--expanded-mimic',required=True,type=Path)
    ap.add_argument('--expanded-eicu',required=True,type=Path)
    ap.add_argument('--stage6a-metrics',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    for p in (args.expanded_mimic,args.expanded_eicu,args.stage6a_metrics):
        if not p.is_file():raise FileNotFoundError(p)
    if any((args.out/f).exists() for f in OUTFILES):raise FileExistsError('Choose an unused --out folder')
    mimic=pd.read_csv(args.expanded_mimic,low_memory=False)
    eicu=pd.read_csv(args.expanded_eicu,low_memory=False)
    for name,d in [('MIMIC',mimic),('eICU',eicu)]:
        needed={'stay_id','group_id','age','sex','y_24h'}
        if needed-set(d.columns):raise ValueError(f'{name}: missing columns {needed-set(d.columns)}')
        if d.stay_id.isna().any() or d.stay_id.duplicated().any():raise ValueError(f'{name}: bad stay IDs')
        if not d.y_24h.isin([0,1]).all():raise ValueError(f'{name}: invalid endpoint')
    hold=np.array([hash_holdout(g,args.seed) for g in mimic.group_id],dtype=bool)
    training=mimic.loc[~hold].reset_index(drop=True)
    internal=mimic.loc[hold].reset_index(drop=True)
    ref=pd.read_csv(args.stage6a_metrics)
    if not {'cohort','model','auroc','auprc','brier'}.issubset(ref.columns):raise ValueError('Stage 6A reference schema invalid')
    shifts=[]; extremes=[]; contrib=[]; comparisons=[]
    for arm in ARM:
        train_x=design(training,arm)
        external_x=design(eicu,arm)
        model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                        ('scale',StandardScaler()),
                        ('lr',LogisticRegression(C=1.,solver='liblinear',max_iter=2000,random_state=args.seed))])
        model.fit(train_x,training.y_24h.to_numpy(dtype=int))
        for cohort,d in [('internal',internal),('external',eicu)]:
            p=model.predict_proba(design(d,arm))[:,1]
            met=metrics(d.y_24h.to_numpy(dtype=int),p)
            row=ref[(ref.cohort==cohort)&(ref.model==arm)]
            if len(row)!=1:raise ValueError('Missing/duplicate Stage6A reference row')
            delta={k:abs(float(met[k])-float(row.iloc[0][k])) for k in ('auroc','auprc','brier')}
            if max(delta.values())>1e-5:raise ValueError(f'Model mismatch for {cohort}/{arm}: {delta}')
            comparisons.append({'cohort':cohort,'model':arm,'max_absolute_metric_difference':max(delta.values())})
            for cutoff in (0.1,0.5,0.9,0.99):
                mask=p>cutoff
                y=d.y_24h.to_numpy(dtype=int)
                extremes.append({'cohort':cohort,'model':arm,'risk_threshold':cutoff,
                                 'n_above':int(mask.sum()),'pct_above':float(mask.mean()*100),
                                 'observed_events_above':int(y[mask].sum()),
                                 'observed_event_rate_above':float(y[mask].mean()) if mask.any() else None,
                                 'mean_predicted_above':float(p[mask].mean()) if mask.any() else None})
        # Distribution shift at original column level, using same filtered/model inputs.
        # SMD uses training SD; zero variance is reported separately and not divided by zero.
        for col in train_x.columns:
            a=pd.to_numeric(train_x[col],errors='coerce').to_numpy(dtype=float)
            b=pd.to_numeric(external_x[col],errors='coerce').to_numpy(dtype=float)
            af=a[np.isfinite(a)];bf=b[np.isfinite(b)]
            ma=float(np.mean(af)) if len(af) else None
            mb=float(np.mean(bf)) if len(bf) else None
            sd=float(np.std(af)) if len(af) else None
            smd=float((mb-ma)/sd) if sd is not None and sd>1e-12 and mb is not None else None
            shifts.append({'model':arm,'feature':col,'train_n_observed':len(af),'external_n_observed':len(bf),
                           'train_missing_pct':100*(1-len(af)/len(a)),
                           'external_missing_pct':100*(1-len(bf)/len(b)),
                           'train_mean':ma,'external_mean':mb,'train_sd':sd,
                           'standardized_mean_difference_vs_train_sd':smd})
        # Decompose logistic logit into transformed feature contributions.
        # Report aggregate means of absolute contributions for high-risk and other stays only.
        # Attributions are model-space arithmetic, not biological causality.
        imp=model.named_steps['imputer'];scale=model.named_steps['scale'];lr=model.named_steps['lr']
        z=scale.transform(imp.transform(external_x))
        if hasattr(imp,'get_feature_names_out'):
            feat=list(imp.get_feature_names_out(external_x.columns))
        else:
            feat=[f'transformed_{i}' for i in range(z.shape[1])]
        if len(feat)!=z.shape[1]:raise ValueError('Transformed feature names/dimensions mismatch')
        p=model.predict_proba(external_x)[:,1]
        group=p>0.5
        for label,mask in [('extreme_gt_50pct',group),('remaining_le_50pct',~group)]:
            if not mask.any():continue
            # Feature-wise signed and absolute logit contributions; downstream coefficient signs intact.
            signed=np.mean(z[mask],axis=0)*lr.coef_[0]
            absolute=np.mean(np.abs(z[mask]*lr.coef_[0]),axis=0)
            for f,sv,av in zip(feat,signed,absolute):
                contrib.append({'model':arm,'group':label,'feature':f,'n_group':int(mask.sum()),
                                'mean_signed_logit_contribution':float(sv),
                                'mean_absolute_logit_contribution':float(av)})
        print('Reproduced Stage 6A and audited',arm,flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(shifts).to_csv(args.out/OUTFILES[0],index=False)
    pd.DataFrame(extremes).to_csv(args.out/OUTFILES[1],index=False)
    pd.DataFrame(contrib).to_csv(args.out/OUTFILES[2],index=False)
    manifest={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED','method':'Stage5G-equivalent fixed logistic regressions; no temperature; no eICU training',
              'max_metric_deviation_tolerance':1e-5,'reproduction':comparisons,
              'n_mimic_train':len(training),'n_mimic_holdout':len(internal),'n_eicu':len(eicu),
              'sha256':{'mimic':digest(args.expanded_mimic),'eicu':digest(args.expanded_eicu),'reference':digest(args.stage6a_metrics)},
              'interpretation':['SMDs are descriptive under source case-mix and missingness shift.',
                  'Logit contributions are algebraic components of a fitted predictive model, not causal feature effects.',
                  'Extreme-risk comparisons may reflect correlated covariates and preprocessing; no clinical mechanism established.',
                  'No patient-level export; evaluation fits fixed; no external recalibration.'],
              'known_limitations':['eICU discharge-time mortality proxy','possible site-specific data recording differences']}
    (args.out/OUTFILES[3]).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('STAGE 6B COMPLETED — aggregate diagnostics only:',args.out,flush=True)

if __name__=='__main__':main()
