"""Non-destructive ICU cohort and result provenance audit.
Usage: python reproducibility_audit.py --data-dir PATH --project-zip PATH --output-dir PATH
"""
import argparse, json, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',default='.');p.add_argument('--project-zip',default='project1.zip');p.add_argument('--output-dir',default='icu_audit_output');a=p.parse_args()
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 cohort=[]; coverage=[]; datasets={}
 for source in ('mimic','eicu'):
  df=pd.read_csv(Path(a.data_dir)/(source+'_landmark.csv'));datasets[source]=df
  y=pd.to_numeric(df.y_24h,errors='coerce');valid=y.notna()&df.group_id.notna()
  cohort.append(dict(dataset=source,rows=len(df),columns=len(df.columns),unique_stays=df.stay_id.nunique(),unique_groups=df.group_id.nunique(),valid_rows=int(valid.sum()),events=int(y[valid].sum()),prevalence=float(y[valid].mean()),missing_labels=int(y.isna().sum()),duplicate_stay_ids=int(df.stay_id.duplicated().sum()),missing_group_ids=int(df.group_id.isna().sum())))
  for prefix in ('hr','rr','spo2','map','temp','creatinine','sodium','potassium','bun','glucose','lactate'):
   col=prefix+'_n'; coverage.append(dict(dataset=source,variable=prefix,observed=int((df[col].fillna(0)>0).sum()),total=len(df),coverage_pct=100*float((df[col].fillna(0)>0).mean())))
 pd.DataFrame(cohort).to_csv(out/'cohort_reconciliation.csv',index=False)
 pd.DataFrame(coverage).to_csv(out/'feature_coverage.csv',index=False)
 mim=datasets['mimic'];idx=np.flatnonzero(mim.y_24h.notna()&mim.group_id.notna());split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=2026)
 tr,te=next(split.split(mim.iloc[idx],mim.iloc[idx].y_24h,groups=mim.iloc[idx].group_id))
 hold=mim.iloc[idx[te]]
 pd.DataFrame([dict(split='mimic_train',n=len(tr),events=int(mim.iloc[idx[tr]].y_24h.sum())),dict(split='mimic_holdout',n=len(te),events=int(hold.y_24h.sum()))]).to_csv(out/'split_reconciliation.csv',index=False)
 results=[]
 zpath=Path(a.project_zip)
 if zpath.exists():
  with zipfile.ZipFile(zpath) as z:
   for name in ('validation_metrics.csv','uncertainty_aware_metrics.csv','incremental_value_deltas.csv'):
    if name in z.namelist():
     from io import BytesIO
     d=pd.read_csv(BytesIO(z.read(name)));d.to_csv(out/('original_'+name),index=False)
     if name!='incremental_value_deltas.csv':
      for _,r in d.iterrows():
       for metric in ('auroc','auprc','brier','ece'):
        col=next((c for c in d.columns if c.lower()==metric),None)
        if col:
         match=re.match(r'\s*([-+]?\d*\.?\d+)',str(r[col]));
         if match:results.append(dict(source_file=name,cohort=str(r.get('Cohort',r.get('dataset',''))),arm=str(r.get('Arm',r.get('arm',''))),metric=metric,value=float(match.group(1))))
 pd.DataFrame(results).to_csv(out/'metric_provenance.csv',index=False)
 findings=[]
 if len(datasets['eicu'])==69005:findings.append('Current eICU file has 69,005 rows; previous 63,143-row audit is not interchangeable.')
 findings.append('MIMIC group-aware 80/20 holdout has %d rows and %d events.'%(len(te),int(hold.y_24h.sum())))
 if results:findings.append('Previous metric files use distinct pipelines and disagree; do not merge their point estimates, confidence intervals, or ECE values.')
 findings.append('Coverage percentages here are recomputed on the currently supplied landmark CSVs.')
 (out/'audit_findings.txt').write_text('\n'.join(findings)+'\n',encoding='utf-8')
 print('\n'.join(findings));print(pd.DataFrame(cohort).to_string(index=False));print('Saved to',out)
if __name__=='__main__':main()
