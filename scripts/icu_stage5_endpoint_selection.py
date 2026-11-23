#!/usr/bin/env python3
"""Read-only ICU endpoint/selection audit. Outputs aggregate counts only.

Sources: stage3c MIMIC/eICU CSV and eICU patient.csv.gz. Nothing is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

LM, END = 1440, 2880

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def to_num(s):return pd.to_numeric(s,errors='coerce')
def count(x):return int(np.asarray(x,dtype=bool).sum())

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw-eicu-patient',type=Path,required=True)
    p.add_argument('--mimic',type=Path,required=True)
    p.add_argument('--eicu',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    for q in (a.raw_eicu_patient,a.mimic,a.eicu):
        if not q.is_file():raise FileNotFoundError(q)
    a.out.mkdir(parents=True,exist_ok=True)
    if any((a.out/x).exists() for x in ('stage5_summary.json','stage5_flow.csv','stage5_contingency.csv')):
        raise FileExistsError('Refusing to overwrite existing Stage5 outputs; use a new --out directory')
    m=pd.read_csv(a.mimic,low_memory=False)
    e=pd.read_csv(a.eicu,low_memory=False)
    required_lm={'stay_id','y_24h','los_min','death_t'}
    for name,df in [('MIMIC',m),('eICU',e)]:
        missing=required_lm-set(df.columns)
        if missing:raise ValueError(f'{name} landmark lacks {sorted(missing)}')
        if df.stay_id.isna().any() or df.stay_id.duplicated().any():raise ValueError(f'{name} stay IDs null or duplicate')
        if not df.y_24h.isin([0,1]).all():raise ValueError(f'{name} invalid label')
    cols=['patientunitstayid','patienthealthsystemstayid','hospitaladmitoffset','age',
          'unitdischargeoffset','unitdischargestatus','hospitaldischargeoffset','hospitaldischargestatus']
    raw=pd.read_csv(a.raw_eicu_patient,compression='infer',usecols=cols,low_memory=False)
    if raw.patientunitstayid.duplicated().any():raise ValueError('Raw eICU duplicated patientunitstayid; stop')
    raw['health_group']=raw.patienthealthsystemstayid
    raw['hospital_offset_num']=to_num(raw.hospitaladmitoffset)
    raw['unit_los']=to_num(raw.unitdischargeoffset)
    raw['hospital_offset']=to_num(raw.hospitaldischargeoffset)
    raw['age_num']=to_num(raw.age.astype(str).str.strip().replace({'> 89':'90','>89':'90'}))
    # Exact equivalent of Stage3's ORDER BY hospitaladmitoffset DESC NULLS LAST, patientunitstayid.
    raw=raw.sort_values(['health_group','hospital_offset_num','patientunitstayid'],
       ascending=[True,False,True],na_position='last',kind='mergesort')
    first=raw.drop_duplicates(subset='health_group',keep='first').copy()
    selected=first[(first.age_num>=18)&(first.unit_los>=LM)].copy()
    unit=selected.unitdischargestatus.fillna('').astype(str).str.strip().str.casefold()
    hosp=selected.hospitaldischargestatus.fillna('').astype(str).str.strip().str.casefold()
    icu_death=(unit=='expired')
    hosp_death=(hosp=='expired')
    during=icu_death & selected.unit_los.gt(LM)&selected.unit_los.le(END)
    after=icu_death & selected.unit_los.gt(END)
    boundary=selected.unit_los.eq(END)
    early_alive=(~icu_death)&selected.unit_los.lt(END)
    early_expired=during
    eligible=during|selected.unit_los.ge(END)
    neg=eligible&~during
    selected_ids=set(to_num(selected.loc[eligible,'patientunitstayid']).astype('int64'))
    csv_ids=set(to_num(e.stay_id).astype('int64'))
    expected_positive=set(to_num(selected.loc[during,'patientunitstayid']).astype('int64'))
    actual_positive=set(to_num(e.loc[e.y_24h.eq(1),'stay_id']).astype('int64'))
    cmp={'expected_eligible_n':len(selected_ids),'observed_stage3c_n':len(csv_ids),
         'missing_from_stage3c':len(selected_ids-csv_ids),'extra_in_stage3c':len(csv_ids-selected_ids),
         'expected_positive_n':len(expected_positive),'observed_positive_n':len(actual_positive),
         'positive_disagreement_count':len(expected_positive.symmetric_difference(actual_positive))}
    flow=[
        ('raw_patient_unit_stays',len(raw)),('first_unit_stay_per_health_system_stay',len(first)),
        ('adult_and_unit_los_at_least_24h',len(selected)),
        ('icu_expired_in_24_to_48h',count(during)),
        ('discharge_alive_before_48h_excluded',count(early_alive)),
        ('icu_expired_after_48h',count(after)),
        ('still_in_icu_at_48h_by_offset',count(selected.unit_los.ge(END))),
        ('endpoint_eligible_selected',count(eligible)),('endpoint_negative',count(neg)),
        ('unit_discharge_exactly_48h',count(boundary)),
        ('unit_discharge_status_missing',count(unit.eq(''))),
        ('hospital_expired_among_early_alive_icu_discharge',count(hosp_death&early_alive)),
        ('hospital_expired_among_icu_deaths_24_to_48',count(hosp_death&during)),
        ('hospital_offset_less_than_unit_offset',count(selected.hospital_offset.lt(selected.unit_los))),
        ('icu_expired_but_hospital_status_not_expired',count(icu_death&~hosp_death)),
        ('hospital_expired_but_unit_not_expired',count(hosp_death&~icu_death)),
    ]
    pd.DataFrame(flow,columns=['measure','n']).to_csv(a.out/'stage5_flow.csv',index=False)
    contingency=pd.crosstab(
        np.where(unit.eq('expired'),'unit_expired',np.where(unit.eq(''),'unit_status_missing','unit_not_expired')),
        np.where(hosp.eq('expired'),'hospital_expired',np.where(hosp.eq(''),'hospital_status_missing','hospital_not_expired')),
        dropna=False)
    contingency.rename_axis(index='icu_discharge_status',columns='hospital_discharge_status').to_csv(a.out/'stage5_contingency.csv')
    def label_check(df):
        los=to_num(df.los_min);dt=to_num(df.death_t);y=df.y_24h.eq(1)
        return {'n':int(len(df)),'positive':count(y),'prevalence':float(y.mean()),
                'positive_death_time_outside_24_48':count(y&~(dt.gt(LM)&dt.le(END))),
                'positive_death_time_after_unit_discharge':count(y&dt.gt(los)),
                'negative_unit_stay_shorter_than_48h':count(~y&los.lt(END))}
    summary={'status':'READ_ONLY_PROTOCOL_AUDIT_NOT_MANUSCRIPT_APPROVED',
       'endpoint':'ICU death in (24h,48h] conditional on ICU stay at hour 24; live discharges before 48h excluded',
       'clinical_warning':'eICU unit discharge Expired and unitdischargeoffset are proxies, not timestamp-confirmed mortality',
       'mimic_csv':label_check(m),'eicu_csv':label_check(e),'raw_selected_eicu_vs_stage3c':cmp,
       'source_sha256':{'raw_eicu_patient':sha(a.raw_eicu_patient),'mimic_landmark':sha(a.mimic),'eicu_landmark':sha(a.eicu)},
       'protocol_flags':{'selection':'first selected unit stay per health-system stay; not one-per-person',
           'discharged_alive_before_48':'excluded, not labeled negative; selection bias risk',
           'status_at_48':'ICU stay duration is proxy for continued observation; no evidence of exact death timestamp in eICU',
           'boundary':'Discharge at 48 hours included; ICU event at exactly 48 hours counts positive'},
       'no_patient_level_export':True}
    (a.out/'stage5_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('STAGE 5 ENDPOINT/SELECTION AUDIT COMPLETE — read-only')
    print(json.dumps({'mimic':summary['mimic_csv'],'eicu':summary['eicu_csv'],
          'selected_eicu':len(selected),'eligible':count(eligible),'early_alive_excluded':count(early_alive),
          'crosscheck':cmp},indent=2))
    if any(cmp[k]!=0 for k in ('missing_from_stage3c','extra_in_stage3c','positive_disagreement_count')):
        print('WARNING: Raw eICU reconstruction differs from Stage3C. Investigate before any final fitting.')

if __name__=='__main__': main()
