#!/usr/bin/env python3
"""Stage 5E: read-only eICU patient identity and 24h boundary audit.

Aggregate outputs only; no patient-, stay-, or hospital-level identifiers are exported.
The source DuckDB is opened READ ONLY. Alternative --raw-eicu-patient is for testing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

LM, END = 1440.0, 2880.0
OUTPUTS = ['stage5e_summary.json', 'stage5e_selection_boundary.csv',
           'stage5e_identity_groups.csv', 'stage5e_reconciliation.csv']

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def numeric(s):
    return pd.to_numeric(s, errors='coerce')

def norm(s):
    return s.astype('string').str.strip().str.casefold()

def n(mask):
    """Count True values in numpy/pandas masks, treating nullable NA as False."""
    return int(pd.Series(mask, dtype="boolean").fillna(False).sum())

def get_raw(args):
    fields = ['patientunitstayid', 'patienthealthsystemstayid', 'uniquepid',
              'hospitaladmitoffset', 'unitdischargeoffset', 'unitdischargestatus',
              'age', 'hospitalid']
    if args.db:
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError('Install duckdb: python -m pip install duckdb') from exc
        con = duckdb.connect(str(args.db), read_only=True)
        try:
            info = con.execute('DESCRIBE SELECT * FROM eicu_patient').fetchdf()
            available = {str(c).casefold(): str(c) for c in info['column_name']}
            required = ['patientunitstayid', 'patienthealthsystemstayid',
                        'hospitaladmitoffset', 'unitdischargeoffset', 'unitdischargestatus', 'age']
            missing = [c for c in required if c not in available]
            if missing:
                raise ValueError('eicu_patient missing fields: ' + ', '.join(missing))
            selected = [available[c] for c in fields if c in available]
            if 'uniquepid' not in available:
                print('WARNING: uniquepid column unavailable: patient-level leakage cannot be assessed.')
            identifiers = ', '.join('"'+col.replace('"','""')+'"' for col in selected)
            df = con.execute('SELECT ' + identifiers + ' FROM eicu_patient').fetchdf()
            return df, {'source': 'duckdb', 'db': str(args.db),
                        'available_fields': selected,
                        'identifier_column_available': 'uniquepid' in available}
        finally:
            con.close()
    # Raw CSV alternative (no patient-level export).
    columns = pd.read_csv(args.raw_eicu_patient, nrows=0).columns.tolist()
    selected = [c for c in fields if c in columns]
    required = ['patientunitstayid', 'patienthealthsystemstayid',
                'hospitaladmitoffset', 'unitdischargeoffset', 'unitdischargestatus', 'age']
    missing = [c for c in required if c not in selected]
    if missing:
        raise ValueError('raw patient file missing fields: ' + ', '.join(missing))
    df = pd.read_csv(args.raw_eicu_patient, usecols=selected, low_memory=False)
    return df, {'source': 'csv', 'raw': str(args.raw_eicu_patient),
                'available_fields': selected,
                'identifier_column_available': 'uniquepid' in columns}

def check_landmark(path, tag):
    df = pd.read_csv(path, usecols=['stay_id', 'group_id', 'los_min', 'y_24h'], low_memory=False)
    if df.stay_id.isna().any() or df.stay_id.duplicated().any():
        raise ValueError(f'{tag}: stay_id missing or repeated')
    if not df.y_24h.isin([0, 1]).all():
        raise ValueError(f'{tag}: invalid labels')
    df['stay_id'] = numeric(df.stay_id)
    if df.stay_id.isna().any():
        raise ValueError(f'{tag}: stay_id not numeric')
    return df

def analyze(raw, restricted, expanded, metadata):
    raw = raw.copy()
    if raw['patientunitstayid'].isna().any() or raw['patientunitstayid'].duplicated().any():
        raise ValueError('Raw eICU patientunitstayid missing or duplicate; fail closed.')
    raw['patientunitstayid'] = numeric(raw['patientunitstayid'])
    raw['patienthealthsystemstayid'] = numeric(raw['patienthealthsystemstayid'])
    if raw['patienthealthsystemstayid'].isna().any():
        raise ValueError('patienthealthsystemstayid is null; selection cannot be reconstructed without a policy.')
    raw['hospitaladmitoffset'] = numeric(raw['hospitaladmitoffset'])
    raw['unitdischargeoffset'] = numeric(raw['unitdischargeoffset'])
    raw['age_num'] = numeric(raw['age'].astype('string').str.strip().replace({'> 89':'90', '>89':'90'}))
    status = norm(raw['unitdischargestatus'])
    raw['unit_status'] = status
    # Matches Stage3C/5B ORDER BY hospitaladmitoffset DESC NULLS LAST, patientunitstayid ASC.
    selected = raw.sort_values(['patienthealthsystemstayid', 'hospitaladmitoffset', 'patientunitstayid'],
                                ascending=[True, False, True], na_position='last', kind='mergesort')
    selected = selected.drop_duplicates('patienthealthsystemstayid', keep='first')
    eligible = selected[(selected.age_num >= 18) & (selected.unitdischargeoffset >= LM)].copy()
    los = eligible.unitdischargeoffset
    status = eligible.unit_status
    positive = status.eq('expired') & los.gt(LM) & los.le(END)
    restricted_expected = positive | los.ge(END)
    expanded_expected = restricted_expected | (status.eq('alive') & los.ge(LM) & los.lt(END))
    early_other = status.ne('alive') & status.ne('expired') & los.lt(END)
    # Summaries with disjoint classification of unselected stays
    boundary_rows = [
        ('raw_unit_stays', len(raw)),
        ('first_unit_stay_per_health_system_stay', len(selected)),
        ('adult_and_at_least_24h_icu', len(eligible)),
        ('unit_los_exactly_24h', n(los.eq(LM))),
        ('unit_los_exactly_48h', n(los.eq(END))),
        ('early_alive_24_to_under_48', n(status.eq('alive') & los.ge(LM) & los.lt(END))),
        ('early_expired_over_24_to_48', n(positive)),
        ('early_expired_exactly_24', n(status.eq('expired') & los.eq(LM))),
        ('early_unknown_24_to_under_48', n(early_other)),
        ('early_not_alive_or_expired', n(early_other)),
        ('stay_at_least_48h', n(los.ge(END))),
        ('expected_restricted', n(restricted_expected)),
        ('expected_expanded', n(expanded_expected)),
        ('positive_events', n(positive)),
        ('gap_expanded_selection', len(eligible)-n(expanded_expected)),
    ]
    r_ids = set(restricted.stay_id.astype('int64'))
    x_ids = set(expanded.stay_id.astype('int64'))
    exp_r = set(eligible.loc[restricted_expected, 'patientunitstayid'].astype('int64'))
    exp_x = set(eligible.loc[expanded_expected, 'patientunitstayid'].astype('int64'))
    positive_ids = set(eligible.loc[positive, 'patientunitstayid'].astype('int64'))
    observed_positive_r = set(restricted.loc[restricted.y_24h.eq(1), 'stay_id'].astype('int64'))
    observed_positive_x = set(expanded.loc[expanded.y_24h.eq(1), 'stay_id'].astype('int64'))
    reconcile = {
        'restricted_expected': len(exp_r), 'restricted_observed':len(r_ids),
        'restricted_missing':len(exp_r-r_ids), 'restricted_extra':len(r_ids-exp_r),
        'expanded_expected':len(exp_x), 'expanded_observed':len(x_ids),
        'expanded_missing':len(exp_x-x_ids), 'expanded_extra':len(x_ids-exp_x),
        'restricted_not_in_expanded': len(r_ids-x_ids),
        'positive_mismatch_restricted':len(positive_ids.symmetric_difference(observed_positive_r)),
        'positive_mismatch_expanded':len(positive_ids.symmetric_difference(observed_positive_x)),
        'added_rows':len(x_ids-r_ids),
        'added_positive':n(expanded.stay_id.isin(x_ids-r_ids) & expanded.y_24h.eq(1)),
        'eicu_group_id_equals_stay_id_restricted':n(numeric(restricted.group_id).eq(restricted.stay_id)),
        'eicu_group_id_equals_stay_id_expanded':n(numeric(expanded.group_id).eq(expanded.stay_id)),
    }
    raw_ids = eligible[['patientunitstayid','patienthealthsystemstayid']].copy()
    if 'uniquepid' in eligible:
        raw_ids['uniquepid'] = eligible['uniquepid'].astype('string').str.strip()
        raw_ids.loc[raw_ids.uniquepid.isna() | raw_ids.uniquepid.eq('') |
                    raw_ids.uniquepid.str.casefold().isin(['nan', 'null', 'none', 'unknown']), 'uniquepid'] = pd.NA
        patient_col = 'uniquepid'
    else:
        patient_col = None
    identity_rows = []
    identity = {}
    for name,ids in [('restricted',r_ids),('expanded',x_ids)]:
        subset = raw_ids[raw_ids.patientunitstayid.isin(ids)]
        observed = subset[patient_col].dropna() if patient_col else pd.Series(dtype='string')
        counts = observed.value_counts(dropna=True)
        repeated = counts[counts > 1]
        identity[name] = {
            'stays':len(subset), 'unique_health_system_stays':int(subset.patienthealthsystemstayid.nunique()),
            'nonmissing_patient_ids':len(observed), 'missing_patient_ids':len(subset)-len(observed),
            'distinct_patient_ids':int(counts.size),
            'patient_ids_with_multiple_stays':int(len(repeated)),
            'stays_belonging_to_repeated_patient_ids':int(repeated.sum()),
            'maximum_stays_per_patient_id':int(counts.max()) if len(counts) else 0,
            'patient_identifier_available':bool(patient_col),
        }
        for metric,val in identity[name].items():
            identity_rows.append({'cohort':name,'metric':metric,'value':val})
    # Quantify whether same patient is represented in both train and holdout based on Stage5C
    # Cannot infer exact split from the CSVs without Stage5C hashed split code; this is a risk audit.
    split_status = 'not_assessed_exact_stage5c_split; identify patient-level grouping before final split'
    summary = {
        'status':'READ_ONLY_AUDIT_NOT_MANUSCRIPT_APPROVED',
        'source':metadata,
        'selection_counts':dict(boundary_rows),
        'reconciliation':reconcile,
        'identity':identity,
        'risk_note':'Stage5B eICU group_id is unit stay ID. Distinct units from same uniquepid may share a train/holdout split or cross split; not guaranteed patient-disjoint.',
        'split_check':split_status,
        'scientific_note':'unitdischargestatus Expired and unitdischargeoffset are a proxy for ICU mortality timing; no true death timestamp inferred.',
        'no_patient_identifiers_exported':True,
    }
    warnings=[]
    if not patient_col:
        warnings.append('No uniquepid in source. Patient-level independence cannot be verified.')
    if identity['expanded']['patient_ids_with_multiple_stays']:
        warnings.append('Repeated uniquepid detected in expanded cohort: patient-level grouping required for future train/test splits and interval estimation.')
    if any(reconcile[k] for k in ['restricted_missing','restricted_extra','expanded_missing','expanded_extra','positive_mismatch_restricted','positive_mismatch_expanded']):
        warnings.append('Selected cohort/labels diverge from reconstructed selection. Investigate before using results.')
    if reconcile['restricted_not_in_expanded']:
        warnings.append('Expanded cohort unexpectedly lost restricted stays.')
    summary['warnings']=warnings
    return summary,pd.DataFrame(boundary_rows,columns=['measure','n']),pd.DataFrame(identity_rows),pd.DataFrame([{'metric':k,'value':v} for k,v in reconcile.items()])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--db',type=Path)
    src.add_argument('--raw-eicu-patient',type=Path)
    p.add_argument('--restricted-eicu',type=Path,required=True)
    p.add_argument('--expanded-eicu',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    for path in [a.db or a.raw_eicu_patient,a.restricted_eicu,a.expanded_eicu]:
        if not path.is_file():raise FileNotFoundError(path)
    a.out.mkdir(parents=True,exist_ok=True)
    for item in OUTPUTS:
        if (a.out/item).exists():raise FileExistsError(f'Refusing overwrite: {a.out/item}')
    restricted=check_landmark(a.restricted_eicu,'restricted')
    expanded=check_landmark(a.expanded_eicu,'expanded')
    raw,meta=get_raw(a)
    summary,boundary,identity,reconciliation=analyze(raw,restricted,expanded,meta)
    summary['sha256_landmarks']={'restricted':sha256(a.restricted_eicu),'expanded':sha256(a.expanded_eicu)}
    boundary.to_csv(a.out/OUTPUTS[1],index=False)
    identity.to_csv(a.out/OUTPUTS[2],index=False)
    reconciliation.to_csv(a.out/OUTPUTS[3],index=False)
    (a.out/OUTPUTS[0]).write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print('STAGE 5E COMPLETE (read-only, aggregate only)')
    print(json.dumps({'selection':summary['selection_counts'], 'identity':summary['identity'],
                      'reconciliation':summary['reconciliation'], 'warnings':summary['warnings']},
                     indent=2,ensure_ascii=False))
if __name__=='__main__':main()
