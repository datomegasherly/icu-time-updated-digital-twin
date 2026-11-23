#!/usr/bin/env python3
"""Read-only landmark feature quality audit; no model training or data correction."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES=['hr','rr','spo2','map','temp','creatinine','sodium','potassium','bun','glucose','lactate']
# Conservative broad plausibility bounds; flags for manual review, not automatic deletion.
BOUNDS={'hr':(20,250),'rr':(3,80),'spo2':(40,100),'map':(20,200),
        'creatinine':(0.1,25),'sodium':(100,180),'potassium':(1.5,9),
        'bun':(1,250),'glucose':(20,1000),'lactate':(0.1,30)}
STATS=['mean','min','max','last','n']

def safeint(x): return int(x)
def classify_temp(s):
    v=pd.to_numeric(s,errors='coerce')
    return {'missing':int(v.isna().sum()),'celsius_20_50':int(v.between(20,50).sum()),
            'fahrenheit_68_122':int(v.between(68,122).sum()),
            'other':int((v.notna() & ~v.between(20,50) & ~v.between(68,122)).sum())}

def scan(path,source):
    df=pd.read_csv(path,low_memory=False)
    rows=[]; anomalies=[]; summary={'dataset':source,'n':len(df),'events':int(pd.to_numeric(df.y_24h,errors='coerce').eq(1).sum()),
                    'temperature_last_categories':classify_temp(df.temp_last)}
    for f in FEATURES:
        counts=pd.to_numeric(df[f+'_n'],errors='coerce')
        last=pd.to_numeric(df[f+'_last'],errors='coerce')
        # n>0 marks at least one measured value; distinguish from missing summarized values.
        rows.append({'dataset':source,'feature':f,'n':len(df),'with_measurement':int(counts.gt(0).sum()),
                     'with_last':int(last.notna().sum()),'count_zero':int(counts.fillna(0).eq(0).sum()),
                     'median_last':None if last.notna().sum()==0 else float(last.median()),
                     'p01_last':None if last.notna().sum()==0 else float(last.quantile(.01)),
                     'p99_last':None if last.notna().sum()==0 else float(last.quantile(.99))})
        for stat in ['min','max','last']:
            v=pd.to_numeric(df[f+'_'+stat],errors='coerce')
            if f=='temp':
                codes=classify_temp(v)
                for category,count in codes.items():
                    anomalies.append({'dataset':source,'feature':f,'stat':stat,'flag':category,'count':count})
            else:
                low,high=BOUNDS[f]
                for flag,mask in [('below_review_bound',v.lt(low)),('above_review_bound',v.gt(high)),('negative',v.lt(0))]:
                    anomalies.append({'dataset':source,'feature':f,'stat':stat,'flag':flag,'count':int(mask.sum())})
    # Data-based inconsistencies that should not occur for legitimate aggregations
    for f in FEATURES:
        lo=pd.to_numeric(df[f+'_min'],errors='coerce'); hi=pd.to_numeric(df[f+'_max'],errors='coerce')
        n=pd.to_numeric(df[f+'_n'],errors='coerce')
        anomalies.append({'dataset':source,'feature':f,'stat':'aggregate','flag':'min_greater_than_max','count':int((lo>hi).sum())})
        anomalies.append({'dataset':source,'feature':f,'stat':'aggregate','flag':'zero_count_but_latest_present','count':int((n.fillna(0).eq(0)&df[f+'_last'].notna()).sum())})
    return pd.DataFrame(rows),pd.DataFrame(anomalies),summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mimic',required=True);ap.add_argument('--eicu',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    x1,y1,s1=scan(a.mimic,'MIMIC-IV');x2,y2,s2=scan(a.eicu,'eICU-CRD')
    pd.concat([x1,x2]).to_csv(out/'feature_coverage_distribution.csv',index=False)
    pd.concat([y1,y2]).to_csv(out/'quality_flags.csv',index=False)
    (out/'audit_summary.json').write_text(json.dumps({'mimic':s1,'eicu':s2,'notes':['Review thresholds are conservative flags only.','Do not convert aggregated temperature statistics blindly if units are mixed within a stay.','Re-extraction from raw measurement level is required to harmonize mixed units properly.','Original files and manuscript results were not changed.']},indent=2),encoding='utf-8')
    print('STAGE 2 FEATURE AUDIT COMPLETED (read-only):',out)
    print('MIMIC:',s1)
    print('eICU:',s2)
if __name__=='__main__':main()
