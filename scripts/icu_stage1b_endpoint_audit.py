#!/usr/bin/env python3
"""Read-only endpoint semantics audit; DOES NOT modify original datasets/models."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def norm(s): return s.astype('string').str.strip().str.casefold()
def count(x):
    """Count True values in a NumPy array or pandas nullable boolean object."""
    return int(pd.Series(x).fillna(False).astype(bool).sum())
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--eicu',required=True,help='Existing eicu_landmark.csv')
    p.add_argument('--raw-eicu-patient',required=True,help='Raw eICU patient.csv.gz')
    p.add_argument('--mimic',help='Existing mimic_landmark.csv, optional')
    p.add_argument('--out',required=True)
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prior=pd.read_csv(a.eicu, usecols=['stay_id','y_24h','los_min','event_t','death_t'])
    prior['stay_id']=pd.to_numeric(prior['stay_id'], errors='coerce')
    prior=prior.set_index('stay_id')
    if not prior.index.is_unique: raise ValueError('Duplicate stay_id in landmark CSV')
    needed=['patientunitstayid','patienthealthsystemstayid','unitdischargeoffset','unitdischargestatus','hospitaldischargeoffset','hospitaldischargestatus']
    frame=[]
    for chunk in pd.read_csv(a.raw_eicu_patient,compression='infer',usecols=lambda c:c.lower() in needed,chunksize=100000,low_memory=False):
        chunk.columns=chunk.columns.str.lower()
        chunk['patientunitstayid']=pd.to_numeric(chunk['patientunitstayid'], errors='coerce')
        chunk=chunk.loc[chunk['patientunitstayid'].isin(prior.index)]
        if len(chunk): frame.append(chunk)
    if not frame: raise ValueError('No matching eICU stay IDs found in raw patient file')
    raw=pd.concat(frame,ignore_index=True).set_index('patientunitstayid')
    if not raw.index.is_unique: raise ValueError('Duplicate patientunitstayid in raw patient CSV')
    dat=prior.merge(raw,how='left',left_index=True,right_index=True,validate='one_to_one',indicator=True)
    if count(dat['_merge']=='left_only'): raise ValueError('Missing selected stays from raw patient file')
    for c in ['unitdischargeoffset','hospitaldischargeoffset']:
        dat[c]=pd.to_numeric(dat[c],errors='coerce')
    unit_exp=norm(dat['unitdischargestatus']).eq('expired').fillna(False)
    hosp_exp=norm(dat['hospitaldischargestatus']).eq('expired').fillna(False)
    current=dat['y_24h'].eq(1)
    # Unit discharge time is an *operational proxy* for death time if unit discharge status is Expired.
    unit_window=unit_exp & dat['unitdischargeoffset'].gt(1440) & dat['unitdischargeoffset'].le(2880)
    hosp_window=hosp_exp & dat['hospitaldischargeoffset'].gt(1440) & dat['hospitaldischargeoffset'].le(2880)
    # Conservative comparability proxy: ICU-death at unit discharge during (24,48], or no ICU-death and observed ICU through hour 48.
    eligible=unit_window | dat['unitdischargeoffset'].ge(2880)
    summary={
      'raw_selected_rows':len(dat), 'original_positive':count(current),
      'raw_hospital_expired':count(hosp_exp),'raw_unit_expired':count(unit_exp),
      'current_positive_with_unit_alive':count(current & ~unit_exp),
      'current_positive_with_unit_expired':count(current & unit_exp),
      'current_positive_after_icu_discharge':count(current & dat['hospitaldischargeoffset'].gt(dat['unitdischargeoffset'])),
      'raw_hospital_expired_24_48_proxy':count(hosp_window),
      'raw_unit_expired_24_48_proxy':count(unit_window),
      'new_unit_proxy_eligible_n':count(eligible),
      'new_unit_proxy_positive_n':count(unit_window & eligible),
      'original_positive_excluded_by_unit_proxy_eligibility':count(current & ~eligible),
      'original_positive_but_negative_in_unit_proxy':count(current & eligible & ~unit_window),
      'original_negative_but_positive_in_unit_proxy':count(~current & eligible & unit_window),
      'current_positive_with_icu_stay_under_48h':count(current & dat['unitdischargeoffset'].lt(2880)),
      'current_positive_with_no_icu_expired_flag':count(current & ~unit_exp),
      'hospital_offset_before_unit_offset':count(dat['hospitaldischargeoffset'].lt(dat['unitdischargeoffset'])),
      'unit_offset_disagreement_with_landmark':count(~np.isclose(dat['unitdischargeoffset'].to_numpy(dtype=float),dat['los_min'].to_numpy(dtype=float),equal_nan=True)),
    }
    rows=[{'measure':k,'value':v} for k,v in summary.items()]
    pd.DataFrame(rows).to_csv(out/'endpoint_comparison_summary.csv',index=False)
    table=pd.crosstab(pd.Series(np.where(current,'current_event','current_non_event'),index=dat.index,name='original_label'),pd.Series(np.select([unit_window,eligible],['unit_proxy_event','unit_proxy_non_event'],default='not_eligible'),index=dat.index,name='unit_proxy_class'))
    table.to_csv(out/'endpoint_cross_tab.csv')
    # Thresholds and mapping check, only aggregate data, no patient-level export
    bad=dat['hospitaldischargeoffset'].lt(dat['unitdischargeoffset'])
    note=[
        'These are aggregate read-only diagnostics, not corrected labels for publication.',
        'A hospital Expired status + hospital discharge offset is not an exact death timestamp.',
        'A unit Expired status + unit discharge offset is also an operational proxy, not independently adjudicated death time.',
        'Restricting to unit-death or ICU stay >=48 hours selects a different cohort and may create selection bias.',
        'Do not infer that the new endpoint is clinically equivalent to the previous one without scientific review.',
        'Do not overwrite original cohort CSV or manuscript results.'
    ]
    (out/'interpretation_notes.txt').write_text('\n'.join(note)+'\n',encoding='utf-8')
    (out/'endpoint_comparison_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('STAGE 1B COMPLETED (read-only):',out)
    print(json.dumps(summary,indent=2))
    if a.mimic:
        m=pd.read_csv(a.mimic,usecols=['stay_id','y_24h','los_min','death_t','event_t'])
        q={'n':len(m),'positive':count(m.y_24h.eq(1)),'death_at_or_before_landmark':count(m.death_t.le(1440)),'positive_after_icu_discharge':count(m.y_24h.eq(1)&m.death_t.gt(m.los_min)),'positive_while_in_icu':count(m.y_24h.eq(1)&m.death_t.le(m.los_min))}
        pd.DataFrame([{'measure':k,'value':v} for k,v in q.items()]).to_csv(out/'mimic_baseline_check.csv',index=False)
        print('MIMIC baseline:',q)
if __name__=='__main__': main()
