#!/usr/bin/env python3
"""Non-destructive audit of existing ICU landmark CSVs and optional raw eICU patient data."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ['hr','rr','spo2','map','temp','creatinine','sodium','potassium','bun','glucose','lactate']
# Wide, conservative plausibility limits: audit flags are not automatic deletion rules.
LIMITS = {'hr': (20, 250), 'rr': (3, 80), 'spo2': (50, 100), 'map': (20, 200),
          'creatinine': (0.1, 25), 'sodium': (100, 180), 'potassium': (1.5, 9),
          'bun': (1, 250), 'glucose': (20, 1000), 'lactate': (0.1, 40)}

def iv(x): return int(x)
def audit_dataset(path, name, out, landmark, horizon):
    df = pd.read_csv(path, low_memory=False)
    expected = ['stay_id', 'group_id','los_min','death_t','event_t','y_24h']
    missing = [x for x in expected if x not in df]
    if missing: raise ValueError(f'{path} missing columns: {missing}')
    y = pd.to_numeric(df.y_24h, errors='coerce')
    los = pd.to_numeric(df.los_min, errors='coerce')
    death = pd.to_numeric(df.death_t, errors='coerce')
    event = pd.to_numeric(df.event_t, errors='coerce')
    positive = y.eq(1)
    death_inside = death.gt(landmark) & death.le(landmark+horizon)
    event_inside = event.gt(landmark) & event.le(landmark+horizon)
    summary = {
        'dataset':name,'n':len(df),'positive_labels':iv(positive.sum()),
        'prevalence':float(positive.mean()),
        'unique_stay_ids':iv(df.stay_id.nunique()),
        'duplicate_stay_rows':iv(df.stay_id.duplicated().sum()),
        'unique_group_ids':iv(df.group_id.nunique()),
        'invalid_or_missing_y':iv((~y.isin([0,1])).sum()),
        'los_below_landmark':iv(los.lt(landmark).sum()),
        'positive_event_outside_window':iv((positive & ~event_inside).sum()),
        'positive_without_death_in_window':iv((positive & ~death_inside).sum()),
        'positive_event_after_unit_discharge':iv((positive & event.gt(los)).sum()),
        'positive_death_after_unit_discharge':iv((positive & death.gt(los)).sum()),
        'any_death_before_or_at_landmark':iv(death.le(landmark).sum()),
        'non_event_without_48h_observability':iv(((~positive)&los.lt(landmark+horizon)).sum()),
        'positive_with_los_below_horizon':iv((positive&los.lt(landmark+horizon)).sum()),
        'death_time_present':iv(death.notna().sum()),
        'event_time_present':iv(event.notna().sum()),
        'event_vs_death_differ_where_both_present':iv((event.notna() & death.notna() & ~event.eq(death)).sum()),
    }
    pd.DataFrame([summary]).to_csv(out/f'{name}_cohort_summary.csv',index=False)
    bad = df.loc[(positive & event.gt(los)) | death.le(landmark) | (positive & ~death_inside),expected].copy()
    bad['reason'] = np.select([positive.loc[bad.index]&event.loc[bad.index].gt(los.loc[bad.index]), death.loc[bad.index].le(landmark), positive.loc[bad.index]&~death_inside.loc[bad.index]], ['event_after_icu_discharge','death_at_or_before_landmark','positive_not_supported_by_death_window'], default='other')
    # These are patient/stay identifiers; export aggregate only by default, not row-level identifiers.
    pd.DataFrame([{'dataset':name, 'reason':k,'count':int(v)} for k,v in bad.reason.value_counts().items()]).to_csv(out/f'{name}_outcome_flags.csv',index=False)
    features=[]
    for f in FEATURES:
        last = f+'_last'; count=f+'_n'
        if last not in df: continue
        x=pd.to_numeric(df[last],errors='coerce')
        count_x=pd.to_numeric(df[count],errors='coerce') if count in df else pd.Series(np.nan,index=df.index)
        lower,upper = LIMITS.get(f,(None,None))
        invalid = ((x<lower)|(x>upper)) if lower is not None else pd.Series(False,index=df.index)
        features.append({'dataset':name,'feature':f,'observed_last':iv(x.notna().sum()),'coverage':float(x.notna().mean()),
          'median':float(x.median()) if x.notna().any() else None,
          'p01':float(x.quantile(.01)) if x.notna().any() else None,
          'p99':float(x.quantile(.99)) if x.notna().any() else None,
          'negative_last':iv((x<0).sum()),'outside_wide_plausibility_range_last':iv(invalid.sum()),
          'count_nonpositive_with_last_present':iv((x.notna()&count_x.le(0)).sum()),
          'last_missing_with_positive_count':iv((x.isna()&count_x.gt(0)).sum()),
          'temp_last_below_60':iv((x<=60).sum()) if f=='temp' else None,
          'temp_last_above_60':iv((x>60).sum()) if f=='temp' else None})
    pd.DataFrame(features).to_csv(out/f'{name}_feature_audit.csv',index=False)
    return summary,features

def raw_eicu_check(path,out):
    # Only patient table is needed for cross-checking event timestamp semantics.
    use=['patientunitstayid','patienthealthsystemstayid','unitdischargeoffset','hospitaldischargeoffset','unitdischargestatus','hospitaldischargestatus','hospitaladmitoffset','age']
    d = pd.read_csv(path,compression='infer',usecols=lambda c: c.lower() in use, low_memory=False)
    d.columns=[c.lower() for c in d.columns]
    for c in ('unitdischargeoffset','hospitaldischargeoffset'):
        d[c]=pd.to_numeric(d[c],errors='coerce')
    expired=d.hospitaldischargestatus.astype(str).str.casefold().eq('expired')
    result={'raw_rows':len(d),'hospital_expired':iv(expired.sum()),
      'hospital_expired_discharged_from_icu_before_hospital_discharge':iv((expired & d.hospitaldischargeoffset.gt(d.unitdischargeoffset)).sum()),
      'unit_discharge_expired':iv(d.unitdischargestatus.astype(str).str.casefold().eq('expired').sum()),
      'hospital_expired_and_unit_discharge_expired':iv((expired & d.unitdischargestatus.astype(str).str.casefold().eq('expired')).sum())}
    pd.DataFrame([result]).to_csv(out/'eicu_raw_patient_semantics.csv',index=False)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mimic',required=True,type=Path,help='Existing mimic_landmark.csv')
    p.add_argument('--eicu',required=True,type=Path,help='Existing eicu_landmark.csv')
    p.add_argument('--out',required=True,type=Path,help='New audit output directory')
    p.add_argument('--raw-eicu-patient',type=Path,default=None,help='Optional eICU/patient.csv.gz')
    p.add_argument('--landmark',type=int,default=1440)
    p.add_argument('--horizon',type=int,default=1440)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    data={}
    for name,path in [('mimic',a.mimic),('eicu',a.eicu)]:
        s,features=audit_dataset(path,name,a.out,a.landmark,a.horizon)
        data[name]={'cohort':s,'features':features}
    if a.raw_eicu_patient: data['raw_eicu_patient']=raw_eicu_check(a.raw_eicu_patient,a.out)
    (a.out/'audit_report.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('AUDIT COMPLETED (read-only inputs). Output:',a.out)
    for name in ('mimic','eicu'):
        s=data[name]['cohort'];print(name,'n=',s['n'],'events=',s['positive_labels'],'after_icu_discharge=',s['positive_event_after_unit_discharge'])

if __name__=='__main__': main()
