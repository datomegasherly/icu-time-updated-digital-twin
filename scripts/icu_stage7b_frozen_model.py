#!/usr/bin/env python3
"""Stage 7B: Frozen-model repeated patient-state updating feasibility (not a live digital twin).

Same 24h-eligible cohort and fixed ICU death (24,48] target at all 6/12/18/24h
snapshots. This is retrospective conditional analysis (selection at 24h), not a
prospective 6h mortality forecast. No counts, no temperature; raw data read-only.
No patient-level export. Reuses Stage7A timestamp extraction and Stage5G split/model. A fixed 6h model is applied to repeated patient-state snapshots without refitting.
"""
import argparse, json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

WINDOWS=[360,720,1080,1440]
VITAL=['hr','rr','spo2','map']
LAB=['creatinine','sodium','potassium','bun','glucose','lactate']
BOUNDS={'hr':(20,300),'rr':(0,90),'spo2':(40,100),'map':(10,250),
'creatinine':(.1,30),'sodium':(90,190),'potassium':(1,12),'bun':(1,250),'glucose':(10,1500),'lactate':(0,40)}
STATS=['mean','min','max','last']
ARM={'vital':VITAL,'lab':LAB,'integrated':VITAL+LAB}

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()

def hash_holdout(x,seed):
 b=(f'{seed}|{x}').encode('utf8')
 return int.from_bytes(hashlib.sha256(b).digest()[:8],'big')%5==0

def strsql(x):return "'"+str(x).replace("'","''")+"'"

def numerics(s):
 return pd.to_numeric(s,errors='coerce')

def features_sql(ds,feature,ids):
 lo,hi=BOUNDS[feature]
 if ds=='mimic':
  if feature in VITAL:
   table='mimic_chartevents'; alias='x';link='x.stay_id = c.stay_id';idcol='itemid'; t="TRY_CAST(x.charttime AS TIMESTAMP)";start='i.intime'
  else:
   table='mimic_labevents'; alias='x';link='x.hadm_id = i.hadm_id';idcol='itemid';t="TRY_CAST(x.charttime AS TIMESTAMP)";start='i.intime'
  cond='TRY_CAST(x.itemid AS BIGINT) IN ('+','.join(str(int(z)) for z in ids)+')'
  v='TRY_CAST(x.valuenum AS DOUBLE)'
  # IMPORTANT: MIMIC ICU admission timestamps may contain seconds. DuckDB's
  # date_diff('minute', intime, charttime) counts clock-minute boundaries,
  # which is NOT equivalent to testing charttime < intime + INTERVAL n MINUTE.
  # It can wrongly exclude readings in the final minute of each snapshot.
  # Preserve the exact timestamp boundary used in the archived Stage 5B extraction.
  raw=f"SELECT c.stay_id, i.intime AS anchor, {t} AS ts, CASE WHEN {v} BETWEEN {lo} AND {hi} THEN {v} ELSE NULL END AS val FROM m_cohort c JOIN mimic_icustays i ON i.stay_id=c.stay_id JOIN {table} x ON {link} AND {cond} WHERE {t}>=i.intime AND {t}<i.intime+INTERVAL 1440 MINUTE"
 else:
  if feature in VITAL:
   table='eicu_vital';col={'hr':'heartrate','rr':'respiration','spo2':'sao2','map':'systemicmean'}[feature]
   t='TRY_CAST(x.observationoffset AS DOUBLE)';link='x.patientunitstayid=c.stay_id';cond='TRUE';v=f'TRY_CAST(x.{col} AS DOUBLE)'
  else:
   table='eicu_lab';t='TRY_CAST(x.labresultoffset AS DOUBLE)';link='x.patientunitstayid=c.stay_id';
   cond='lower(CAST(x.labname AS VARCHAR)) IN ('+','.join(strsql(z.lower()) for z in ids)+')';v='TRY_CAST(x.labresult AS DOUBLE)'
  raw=f"SELECT c.stay_id, {t} AS min_offset, {t} AS ts, CASE WHEN {v} BETWEEN {lo} AND {hi} THEN {v} ELSE NULL END AS val FROM e_cohort c JOIN {table} x ON {link} AND {cond} WHERE {t}>=0 AND {t}<1440"
 expr=[]
 for w in WINDOWS:
  suffix=f't{w//60}'
  # Same left-open observation window as baseline ([0,t)).
  filt=(f"ts < anchor + INTERVAL {w} MINUTE" if ds=='mimic' else f'min_offset < {w}')
  expr += [f'avg(val) FILTER (WHERE {filt}) AS {feature}_mean_{suffix}',
           f'min(val) FILTER (WHERE {filt}) AS {feature}_min_{suffix}',
           f'max(val) FILTER (WHERE {filt}) AS {feature}_max_{suffix}',
           f'arg_max(val,ts ORDER BY val DESC) FILTER (WHERE {filt} AND val IS NOT NULL) AS {feature}_last_{suffix}']
 return f"WITH raw AS ({raw}) SELECT stay_id, "+', '.join(expr)+" FROM raw GROUP BY stay_id"

def design(df,arm,w):
 cols=[f'{v}_{s}_t{w//60}' for v in ARM[arm] for s in STATS]
 x=df[cols].apply(pd.to_numeric,errors='coerce').copy()
 x.columns=[c.rsplit('_t',1)[0] for c in x.columns]
 x.insert(0,'age',numerics(df.age).to_numpy())
 x['sex_numeric']=df.sex.astype('string').str.strip().str.lower().map({'m':1.,'male':1.,'f':0.,'female':0.}).astype(float)
 return x.replace([np.inf,-np.inf],np.nan)

def eval_stats(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
 if not np.isfinite(p).all():raise ValueError('invalid predictions')
 return {'n':len(y),'events':int(y.sum()),'auroc':float(roc_auc_score(y,p)),
  'auprc':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),
  'mean_risk':float(p.mean())}

def make_model(seed):
 return Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                  ('scale',StandardScaler()),
                  ('lr',LogisticRegression(solver='liblinear',C=1.,max_iter=2000,random_state=seed))])

def get_external_patient_groups(con, stay_ids):
 raw=con.execute('SELECT patientunitstayid, uniquepid FROM eicu_patient').df()
 if raw.patientunitstayid.isna().any() or raw.patientunitstayid.duplicated().any():
  raise RuntimeError('eICU patient stay IDs missing or duplicated')
 raw['patientunitstayid']=pd.to_numeric(raw.patientunitstayid,errors='raise').astype('int64')
 ids=pd.DataFrame({'stay_id':pd.to_numeric(stay_ids,errors='raise').astype('int64')})
 joined=ids.merge(raw,left_on='stay_id',right_on='patientunitstayid',how='left',sort=False,validate='one_to_one',indicator=True)
 patients=joined.uniquepid.astype('string').str.strip()
 if not joined['_merge'].eq('both').all() or patients.isna().any() or patients.eq('').any():
  raise RuntimeError('Missing eICU person linkage; cannot patient-cluster bootstrap')
 return patients.to_numpy(dtype=str)

def safe_metrics(y,p):
 return eval_stats(y,p)

def fit_and_evaluate(m,e,seed):
 # Fit ONCE on 6h states and apply unchanged pipeline across all 4 snapshots.
 hold=np.asarray([hash_holdout(g,seed) for g in m.group_id],bool)
 ym=m.y_24h.to_numpy(dtype=int); ye=e.y_24h.to_numpy(dtype=int)
 rows=[]; preds={'internal':{},'external':{}}
 for arm in ARM:
  model=make_model(seed)
  x6=design(m,arm,360)
  model.fit(x6.loc[~hold],ym[~hold])
  for w in WINDOWS:
   for cohort,d,x,mask,y in [('internal',m,design(m,arm,w),hold,ym[hold]),
                               ('external',e,design(e,arm,w),np.ones(len(e),dtype=bool),ye)]:
    p=model.predict_proba(x.loc[mask])[:,1]
    preds[cohort][(arm,w//60)]=p
    rows.append({'cohort':cohort,'training_landmark_h':6,'evaluation_landmark_h':w//60,
                 'model':arm,**safe_metrics(y,p),
                 'pct_risk_gt_50':float(np.mean(p>0.5))})
   print(f'Frozen 6h model evaluated: {arm} at {w//60}h',flush=True)
 return pd.DataFrame(rows),preds,hold

def validate_6h_against_stage7a(met, path, tolerance=1e-5):
 reference=pd.read_csv(path)
 qc=[]
 for cohort in ('internal','external'):
  for arm in ARM:
   x=met[(met.cohort==cohort)&(met.model==arm)&(met.evaluation_landmark_h==6)].iloc[0]
   r=reference[(reference.cohort==cohort)&(reference.model==arm)&(reference.landmark_h==6)]
   if len(r)!=1:raise RuntimeError(f'Stage7A reference row missing {cohort}/{arm}')
   r=r.iloc[0]
   for metric in ('auroc','auprc','brier','mean_risk'):
    delta=float(x[metric]-r[metric]);qc.append({'cohort':cohort,'model':arm,'metric':metric,'delta':delta})
    if abs(delta)>tolerance:raise RuntimeError(f'Fail-closed: 6h frozen training differs from Stage7A {cohort}/{arm}/{metric}: {delta}')
 return pd.DataFrame(qc)

def cluster_bootstrap_change(y,p6,p24,groups,n_iter,seed):
 # Paired percentile CI of 24h minus 6h, WITH the same frozen 6h model.
 # Patient-cluster draws retain all ICU stays for a selected patient.
 _, inv=np.unique(np.asarray(groups,dtype=str),return_inverse=True)
 nclust=int(inv.max())+1;rng=np.random.default_rng(seed)
 vals={m:[] for m in ('auroc','auprc','brier')}
 skipped=0
 for i in range(n_iter):
  counts=np.bincount(rng.integers(0,nclust,size=nclust),minlength=nclust)
  weights=counts[inv].astype(float)
  if weights[y==1].sum()==0 or weights[y==0].sum()==0:
   skipped+=1;continue
  vals['auroc'].append(float(roc_auc_score(y,p24,sample_weight=weights)-roc_auc_score(y,p6,sample_weight=weights)))
  vals['auprc'].append(float(average_precision_score(y,p24,sample_weight=weights)-average_precision_score(y,p6,sample_weight=weights)))
  vals['brier'].append(float(np.average((y-p24)**2,weights=weights)-np.average((y-p6)**2,weights=weights)))
  if (i+1)%100==0:print(f'  Bootstrap {i+1}/{n_iter}',flush=True)
 if n_iter-skipped<max(30,int(.9*n_iter)):
  raise RuntimeError('Insufficient valid bootstrap draws')
 return {'clusters':nclust,'requested':n_iter,'valid':n_iter-skipped,'skipped':skipped,
         'delta_24_minus_6':{k:{'estimate':float(np.mean(v)),
                                  'ci_lower':float(np.quantile(v,.025)),
                                  'ci_upper':float(np.quantile(v,.975))} for k,v in vals.items()}}

def check_baseline(d,stem):
 # Compare temporal re-extraction with Stage5B baseline 24h input, fail for substantive disagreement.
 vals=[]
 for v in VITAL+LAB:
  for s in STATS:
   a=numerics(d[f'{v}_{s}']); b=numerics(d[f'{v}_{s}_t24'])
   ok=(a.isna()&b.isna()) | (a.notna()&b.notna() & np.isclose(a,b,rtol=1e-5,atol=1e-5))
   vals.append({'dataset':stem,'feature':f'{v}_{s}','n':len(d),'mismatched':int((~ok).sum()),'mismatch_fraction':float((~ok).mean())})
 return vals

def main():
 a=argparse.ArgumentParser()
 a.add_argument('--db',required=True,type=Path)
 a.add_argument('--config',required=True,type=Path)
 a.add_argument('--expanded-mimic',required=True,type=Path)
 a.add_argument('--expanded-eicu',required=True,type=Path)
 a.add_argument('--out',required=True,type=Path)
 a.add_argument('--seed',type=int,default=2026)
 a.add_argument('--stage7a-metrics',required=True,type=Path)
 a.add_argument('--bootstrap',type=int,default=500)
 a.add_argument('--skip-baseline-assert',action='store_true',help='Diagnostic only; not for manuscript')
 args=a.parse_args()
 try:import duckdb
 except ImportError as exc:raise SystemExit('Install duckdb: python -m pip install duckdb') from exc
 cfg=json.loads(args.config.read_text(encoding='utf8'))
 m=pd.read_csv(args.expanded_mimic,low_memory=False)
 e=pd.read_csv(args.expanded_eicu,low_memory=False)
 for tag,d in [('mimic',m),('eicu',e)]:
  if d.stay_id.isna().any() or d.stay_id.duplicated().any():raise ValueError(f'{tag}: duplicated/null stay_id')
  if not set(d.y_24h.unique()).issubset({0,1}):raise ValueError(f'{tag}: invalid labels')
  if len(d)==0:raise ValueError(f'{tag}: empty cohort')
  print(f'{tag}: {len(d):,} stays, {int(d.y_24h.sum()):,} events',flush=True)
 if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Output directory not empty; use a new directory')
 args.out.mkdir(parents=True,exist_ok=True)
 con=duckdb.connect(str(args.db),read_only=True)
 try:
  con.register('m_ids',pd.DataFrame({'stay_id':pd.to_numeric(m.stay_id).astype('int64')}))
  con.register('e_ids',pd.DataFrame({'stay_id':pd.to_numeric(e.stay_id).astype('int64')}))
  con.execute('CREATE TEMP TABLE m_cohort AS SELECT * FROM m_ids')
  con.execute('CREATE TEMP TABLE e_cohort AS SELECT * FROM e_ids')
  ext_patient_groups=get_external_patient_groups(con,e.stay_id)
  for tag,d in [('mimic',m),('eicu',e)]:
   for v in VITAL+LAB:
    ids=cfg[tag]['vitals'][v] if v in VITAL else cfg[tag]['labs'][v]
    sql=features_sql(tag,v,ids)
    res=con.execute(sql).df()
    if res.stay_id.duplicated().any():raise RuntimeError(f'non-unique {tag} feature {v}')
    d0=d.merge(res,on='stay_id',how='left',sort=False,validate='one_to_one')
    # Preserve row order/identity and attach new feature cols without duplicating baseline.
    if not d0.stay_id.equals(d.stay_id):raise RuntimeError('stay ordering changed')
    for c in res.columns:
     if c!='stay_id':d[c]=d0[c].to_numpy()
    print(f'Extracted {tag} {v}: raw timeline to 6/12/18/24h',flush=True)
 finally:con.close()
 qc=pd.DataFrame(check_baseline(m,'mimic')+check_baseline(e,'eicu'))
 qc.to_csv(args.out/'stage7a_baseline_reconciliation.csv',index=False)
 worst=float(qc.mismatch_fraction.max())
 # Tied-time 'last' may differ due nondeterministic ordering. Fail closed at >0.5%; never bypass this gate for publishable results.
 if worst>0.005 and not args.skip_baseline_assert:
  raise RuntimeError(f'24h features do not reproduce archived Stage5B! largest mismatch={worst:.3%}. Review stage7a_baseline_reconciliation.csv BEFORE model results. No manuscript numbers generated.')
 if args.bootstrap<50:raise ValueError('--bootstrap must be >= 50')
 metrics,preds,hold=fit_and_evaluate(m,e,args.seed)
 qc6=validate_6h_against_stage7a(metrics,args.stage7a_metrics)
 metrics.to_csv(args.out/'stage7b_frozen_metrics.csv',index=False)
 qc6.to_csv(args.out/'stage7b_stage7a_reconciliation.csv',index=False)
 y_internal=m.loc[hold,'y_24h'].to_numpy(dtype=int)
 y_external=e.y_24h.to_numpy(dtype=int)
 internal_groups=m.loc[hold,'group_id'].astype(str).to_numpy()
 external_groups=ext_patient_groups
 summary={'status':'EXPLORATORY_RETROSPECTIVE_FIXED_COHORT','training_landmark_h':6,
          'description':'Frozen pipeline trained at 6h, repeatedly applied to 6/12/18/24h patient-state snapshots; same 24h-survivor cohort and fixed 24-48h mortality endpoint.',
          'comparisons':{},'n':{'mimic':len(m),'eicu':len(e),'eicu_unique_patients':len(np.unique(external_groups))},
          'reconciliation_24h_max_mismatch':worst,'seed':args.seed,'bootstrap':args.bootstrap,
          'limitations':['24h survivor selection: retrospective conditioning prevents genuine 6h prospective prediction',
                         'Fixed endpoint ICU death 24-48h, not a moving 24h risk window',
                         'This tests a frozen predictive model with updated feature snapshots, not an operational digital twin',
                         'eICU death time is approximated by discharge offset',
                         'Post-hoc design; eICU already used in model-development decisions',
                         'Bootstrap quantifies sampling uncertainty, not model-development/selection uncertainty']}
 for cohort,y,groups in [('internal',y_internal,internal_groups),('external',y_external,external_groups)]:
  summary['comparisons'][cohort]={}
  for arm in ARM:
   p6=preds[cohort][(arm,6)];p24=preds[cohort][(arm,24)]
   point={k:float(safe_metrics(y,p24)[k]-safe_metrics(y,p6)[k]) for k in ('auroc','auprc','brier')}
   print('Patient-cluster bootstrap',cohort,arm,flush=True)
   ci=cluster_bootstrap_change(y,p6,p24,groups,args.bootstrap,args.seed+len(arm)+(0 if cohort=='internal' else 10000))
   summary['comparisons'][cohort][arm]={'point_delta_24_minus_6':point,'bootstrap':ci}
 (args.out/'stage7b_cluster_ci.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
 manifest={'status':'EXPLORATORY_NOT_MANUSCRIPT_APPROVED',
           'reconciliation_24h_max_mismatch':worst,
           'reconciliation_6h_max_absolute_delta':float(qc6.delta.abs().max()),
           'source_hashes_sha256':{key:digest(val) for key,val in {'config':args.config,'mimic':args.expanded_mimic,'eicu':args.expanded_eicu,'stage7a_metrics':args.stage7a_metrics}.items()},
           'n':summary['n'],'read_only_db':True,'bootstrap':args.bootstrap,
           'limitations':summary['limitations']}
 (args.out/'stage7b_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 print('STAGE 7B COMPLETE — FROZEN 6H MODEL, RETROSPECTIVE ONLY',args.out,flush=True)
 print(json.dumps({'n':summary['n'],'reconciliation_24h':worst,'reconciliation_6h_max':manifest['reconciliation_6h_max_absolute_delta']},indent=2))

if __name__=='__main__':main()
