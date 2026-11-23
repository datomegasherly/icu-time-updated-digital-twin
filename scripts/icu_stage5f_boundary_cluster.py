#!/usr/bin/env python3
"""Stage 5F: independent, read-only eICU endpoint boundary and identity audit.
No patient-level identifiers are exported. No fitting or modification of sources.
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

LM=1440.0
END=2880.0

def load_raw(db):
    import duckdb
    con=duckdb.connect(str(db),read_only=True)
    try:
        need=['patientunitstayid','patienthealthsystemstayid','hospitaladmitoffset',
              'unitdischargeoffset','unitdischargestatus','age','uniquepid']
        available=set(con.execute('DESCRIBE SELECT * FROM eicu_patient').fetchdf().column_name.str.lower())
        absent=set(need)-available
        if absent: raise ValueError(f'Missing eicu_patient fields: {sorted(absent)}')
        return con.execute('SELECT '+','.join('"'+c+'"' for c in need)+' FROM eicu_patient').fetchdf()
    finally:con.close()

def audit(df,expanded):
    d=df.copy()
    for c in ['patientunitstayid','patienthealthsystemstayid','hospitaladmitoffset','unitdischargeoffset']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d['age_num']=pd.to_numeric(d.age.astype('string').str.strip().replace({' > 89':'90','>89':'90','> 89':'90'}),errors='coerce')
    d['status']=d.unitdischargestatus.astype('string').str.strip().str.casefold()
    if d.patientunitstayid.isna().any() or d.patientunitstayid.duplicated().any():
        raise ValueError('Raw stay IDs must be unique and nonmissing')
    if d.patienthealthsystemstayid.isna().any():raise ValueError('Missing health-system stay ID')
    d=d.sort_values(['patienthealthsystemstayid','hospitaladmitoffset','patientunitstayid'],ascending=[True,False,True],na_position='last',kind='mergesort').drop_duplicates('patienthealthsystemstayid')
    eligible=d.loc[d.age_num.ge(18)&d.unitdischargeoffset.ge(LM)].copy()
    los=eligible.unitdischargeoffset
    st=eligible.status
    expired=st.eq('expired').fillna(False)
    alive=st.eq('alive').fillna(False)
    exact=los.eq(LM)
    short=los.lt(END)
    event=expired&los.gt(LM)&los.le(END)
    cases={
      'positive_icu_expired_after_24_to_48':event,
      'negative_stay_at_least_48':(~event)&los.ge(END),
      'negative_alive_discharge_24_to_under_48':alive&short,
      'excluded_expired_exactly_24':expired&exact,
      'excluded_expired_between_24_and_48_unaccounted':expired&los.gt(LM)&short&(~event),
      'excluded_unknown_status_24_to_under_48':(~alive)&(~expired)&short,
    }
    counted=np.zeros(len(eligible),dtype=int)
    for v in cases.values(): counted+=v.to_numpy(dtype=int)
    if (counted>1).any():raise AssertionError('Non-disjoint category definitions')
    if (counted==0).any():
        # Outcome classification must fail closed rather than silently lose rows.
        raise AssertionError(f'{int((counted==0).sum())} rows unclassified')
    categories={k:int(v.sum()) for k,v in cases.items()}
    expected_include=event|los.ge(END)|(alive&short)
    expanded_ids=set(pd.to_numeric(expanded.stay_id,errors='raise').astype('int64'))
    expected_ids=set(eligible.loc[expected_include,'patientunitstayid'].astype('int64'))
    if expected_ids!=expanded_ids:raise AssertionError(f'Expanded mismatch: missing={len(expected_ids-expanded_ids)},extra={len(expanded_ids-expected_ids)}')
    selected=eligible.loc[eligible.patientunitstayid.astype('int64').isin(expanded_ids)].copy()
    selected['person']=selected.uniquepid.astype('string').str.strip()
    person_counts=selected.person.value_counts(dropna=False)
    if selected.person.isna().any() or selected.person.eq('').any():raise AssertionError('Missing patient person identifier')
    duplicated=person_counts[person_counts>1]
    event_ids=set(eligible.loc[event,'patientunitstayid'].astype('int64'))
    selected['event']=selected.patientunitstayid.astype('int64').isin(event_ids)
    out={
      'status':'READ_ONLY_NOT_MANUSCRIPT_APPROVED',
      'raw_unit_stays':len(df),
      'index_stays':len(d),
      'eligible_24h':len(eligible),
      'categories_disjoint':categories,
      'sum_categories':sum(categories.values()),
      'eligible_minus_expanded':len(eligible)-len(expanded_ids),
      'expanded_reconciled':True,
      'expanded_stays':len(selected),
      'expanded_unique_patients':int(person_counts.size),
      'patients_multiple_stays':len(duplicated),
      'stays_from_repeated_patients':int(duplicated.sum()),
      'maximum_stays_one_patient':int(person_counts.max()),
      'events_in_repeated_patients':int(selected.loc[selected.person.isin(duplicated.index),'event'].sum()),
      'events_total':int(selected.event.sum()),
      'warning':'Use uniquepid grouping or patient-level clustering in external uncertainty estimation; eICU event time remains unit-discharge proxy.'
    }
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--db',type=Path,required=True)
    p.add_argument('--expanded-eicu',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    raw=load_raw(a.db)
    expanded=pd.read_csv(a.expanded_eicu,usecols=['stay_id'])
    result=audit(raw,expanded)
    (a.out/'stage5f_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame([{'category':k,'n':v} for k,v in result['categories_disjoint'].items()]).to_csv(a.out/'stage5f_boundary_categories.csv',index=False)
    print('STAGE 5F COMPLETE — READ ONLY')
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
