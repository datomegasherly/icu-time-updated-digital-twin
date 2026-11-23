#!/usr/bin/env python3
"""Stage 7A: retrospective rolling patient-state feasibility (not a live digital twin).

Same 24h-eligible cohort and fixed ICU death (24,48] target at all 6/12/18/24h
snapshots. This is retrospective conditional analysis (selection at 24h), not a
prospective 6h mortality forecast. No counts, no temperature; raw data read-only.
No patient-level export. Reuses exact Stage5B feature clipping and 5G split/model.
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
           f'arg_max(val,ts) FILTER (WHERE {filt} AND val IS NOT NULL) AS {feature}_last_{suffix}']
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

def evaluate(m,e,seed):
 hold=np.asarray([hash_holdout(g,seed) for g in m.group_id],bool)
 if sum(m.loc[hold,'y_24h'])<2 or sum(m.loc[~hold,'y_24h'])<2:raise ValueError('too few outcome events')
 rows=[];coverage=[]
 for w in WINDOWS:
  for arm in ARM:
   xx=design(m,arm,w);xe=design(e,arm,w)
   model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                   ('scale',StandardScaler()),('lr',LogisticRegression(solver='liblinear',C=1.,max_iter=2000,random_state=seed))])
   model.fit(xx.loc[~hold],m.loc[~hold,'y_24h'].to_numpy(dtype=int))
   for ds,d,x,mask in [('internal',m,xx,hold),('external',e,xe,np.ones(len(e),dtype=bool))]:
    p=model.predict_proba(x.loc[mask])[:,1]
    row={'cohort':ds,'landmark_h':w//60,'model':arm,**eval_stats(d.loc[mask,'y_24h'],p)}
    rows.append(row)
   for ds,x in [('mimic',xx),('eicu',xe)]:
    coverage.append({'dataset':ds,'landmark_h':w//60,'model':arm,
       'feature_nonmissing_fraction':float(x.notna().to_numpy().mean()),
       'rows_with_any_feature':int(x.drop(columns=['age','sex_numeric']).notna().any(axis=1).sum())})
   print(f'Fitted and evaluated landmark={w//60} model={arm}',flush=True)
 return pd.DataFrame(rows),pd.DataFrame(coverage)

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
 a.add_argument('--skip-baseline-assert',action='store_true',help='Diagnostic only; default is fail-closed')
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
 metrics,coverage=evaluate(m,e,args.seed)
 metrics.to_csv(args.out/'stage7a_landmark_metrics.csv',index=False)
 coverage.to_csv(args.out/'stage7a_observation_coverage.csv',index=False)
 manifest={'status':'EXPLORATORY_DYNAMIC_STATE_FEASIBILITY','description':'4 retrospective snapshots of fixed 24h-survivor cohorts, with fixed 24-48h label','landmarks_h':[6,12,18,24],
  'frozen_cohort_at_24h':True,'no_data_after_snapshot_in_features':True,'same_endpoint_all_snapshots':'ICU death (24,48] h; eICU discharge proxy',
  'not_valid_for_prospective_early_prediction':True,'no_live_update_or_counterfactual_simulation':True,
  'no_temperature_or_measurement_counts':True,'baseline_max_mismatch_fraction':worst,'seed':args.seed,
  'inputs_sha256':{key:digest(val) for key,val in {'config':args.config,'mimic':args.expanded_mimic,'eicu':args.expanded_eicu}.items()},
  'db_read_only':True,'db_path':str(args.db),'db_size_bytes':args.db.stat().st_size,
  'n':{'mimic':len(m),'mimic_events':int(m.y_24h.sum()),'eicu':len(e),'eicu_events':int(e.y_24h.sum())},
  'warnings':['Fixed 24h cohort creates retrospective survivor/selection conditioning for earlier 6/12/18h snapshots','All models re-trained separately at each snapshot; no demonstration of updating a fixed model','External eICU endpoint time is a discharge proxy','eICU has repeated patients and point estimates here lack cluster bootstrap','Post-hoc design motivated by prior external evaluation; needs independent confirmation']}
 (args.out/'stage7a_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 print('STAGE 7A COMPLETE — retrospective feasibility ONLY',args.out,flush=True)
 print(json.dumps({'n':manifest['n'],'worst_24h_reconciliation':worst},indent=2))

if __name__=='__main__':main()
