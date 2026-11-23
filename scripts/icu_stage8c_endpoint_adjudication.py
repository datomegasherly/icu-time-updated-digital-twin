#!/usr/bin/env python3
"""Stage 8C: aggregate-only read-only audit of rolling ICU death endpoint.
Uses EXACT Stage7C cohort selection and risk_labels, fails closed on mismatch to
archived Stage7E n/events. No model refitting, no data mutation, no patient export.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from icu_stage7c_prospective_landmarks import cohorts, risk_labels

LANDMARKS = (6, 12, 18, 24)
HORIZON = 1440.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def adjudicate(d, source, h):
    """Mutually exclusive categories for every selected adult index stay."""
    t = float(h * 60)
    u = t + HORIZON
    if source == 'eicu':
        los = pd.to_numeric(d.discharge_min, errors='coerce').to_numpy(dtype=float)
        status = d.discharge_status.astype('string').str.strip().str.lower()
        expired = status.eq('expired').fillna(False).to_numpy(dtype=bool)
        alive = status.eq('alive').fillna(False).to_numpy(dtype=bool)
        nonstandard = ~(expired | alive)
        invalid = ~np.isfinite(los)
        before = np.isfinite(los) & (los < t)
        exact = np.isfinite(los) & (los == t)
        after = np.isfinite(los) & (los > t)
        early = after & (los < u)
        at_end = after & (los == u)
        late = after & (los > u)
        classes = np.select([
            invalid,
            before,
            exact & expired,
            exact & alive,
            exact & nonstandard,
            early & expired,
            early & alive,
            early & nonstandard,
            at_end & expired,
            at_end & alive,
            at_end & nonstandard,
            late & expired,
            late & alive,
            late & nonstandard,
        ], [
            'missing_discharge_offset',
            'discharged_before_landmark',
            'expired_exact_landmark_not_at_risk',
            'alive_exact_landmark_not_at_risk',
            'unknown_exact_landmark_not_at_risk',
            'event_expired_before_horizon',
            'competing_alive_discharge_before_horizon',
            'unknown_status_early_discharge_excluded',
            'event_expired_exact_horizon',
            'observed_alive_exact_horizon',
            'observed_unknown_exact_horizon',
            'expired_after_horizon_no_event',
            'alive_after_horizon_no_event',
            'unknown_after_horizon_no_event',
        ], default='UNCLASSIFIED')
        if np.any(classes == 'UNCLASSIFIED'):
            raise AssertionError('Unclassified eICU outcome')
        original_ok, original_y = risk_labels(d, 'eicu', h)
        # Stage7C: any known ICU residence to horizon is a negative, even
        # when discharge status is unknown/expired later than the horizon.
        event = after & expired & (los <= u)
        neg = late | at_end & ~expired | early & alive
        ok = event | neg
        if not (np.array_equal(original_ok, ok) and
                np.array_equal(original_y[ok], event[ok].astype('int8'))):
            raise AssertionError(f'eICU endpoint mismatch at {h}h')
        return classes, ok, event, {'los': los, 'expired': expired,
                                    'alive': alive, 'unknown': nonstandard,
                                    'at_risk_raw': after, 'early_unknown': early & nonstandard}

    start = pd.to_datetime(d.intime, errors='coerce') + pd.to_timedelta(t, unit='m')
    end = start + pd.to_timedelta(HORIZON, unit='m')
    discharge = pd.to_datetime(d.outtime, errors='coerce')
    death = pd.to_datetime(d.death_ts, errors='coerce')
    missing = discharge.isna().to_numpy()
    out_pre = (discharge < start).fillna(False).to_numpy()
    out_exact = (discharge == start).fillna(False).to_numpy()
    out_after = (discharge > start).fillna(False).to_numpy()
    dead_pre = (death < start).fillna(False).to_numpy()
    dead_exact = (death == start).fillna(False).to_numpy()
    prior = dead_pre | dead_exact
    at_risk = out_after & ~prior
    death_in = (death > start).fillna(False).to_numpy() & (death <= end).fillna(False).to_numpy()
    death_in_unit = (death <= discharge).fillna(False).to_numpy()
    event = at_risk & death_in & death_in_unit
    early_alive = at_risk & (discharge < end).fillna(False).to_numpy() & ~event
    late_observed = at_risk & ~early_alive & ~event
    classes = np.select([
        missing,
        out_pre,
        out_exact,
        prior & out_after,
        event & (death == end).fillna(False).to_numpy(),
        event,
        early_alive & death_in & ~death_in_unit,
        early_alive,
        late_observed,
    ], [
        'missing_icu_discharge',
        'discharged_before_landmark',
        'discharged_exact_landmark_not_at_risk',
        'death_at_or_before_landmark',
        'icu_death_exact_horizon',
        'icu_death_before_horizon',
        'post_icu_death_within_horizon_not_icu_event',
        'competing_icu_discharge_before_horizon',
        'observed_to_horizon_no_icu_death',
    ], default='UNCLASSIFIED')
    if np.any(classes == 'UNCLASSIFIED'):
        raise AssertionError(f'Unclassified MIMIC outcome at {h}h')
    original_ok, original_y = risk_labels(d, 'mimic', h)
    if not (np.array_equal(original_ok, at_risk) and
            np.array_equal(original_y[at_risk], event[at_risk].astype('int8'))):
        raise AssertionError(f'MIMIC endpoint mismatch at {h}h')
    return classes, at_risk, event, {'start': start, 'end': end,
                                    'discharge': discharge, 'death': death,
                                    'at_risk_raw': at_risk}


def scenarios(d, source, h, ok, event, aux):
    """Counts are bounded sensitivity, NOT revised mortality ground truth."""
    base = {'source': source, 'landmark_h': h, 'scenario': 'stage7e_primary',
            'n': int(ok.sum()), 'events': int(event[ok].sum())}
    rows = [base]
    if source == 'eicu':
        early_unknown = aux['early_unknown']
        for name, flags in (
            ('unknown_early_as_negative_sensitivity', np.zeros(len(d), bool)),
            ('unknown_early_as_event_sensitivity', early_unknown),
        ):
            mask = ok | early_unknown
            y = event | flags
            rows.append({'source': source, 'landmark_h': h, 'scenario': name,
                         'n': int(mask.sum()), 'events': int(y[mask].sum())})
        # Shift the *event proxy* only; do NOT shift ICU at-risk/discharge
        # eligibility. For known expired patients eligible at current landmark,
        # vary the hypothetical death event offset near the horizon.
        los = aux['los']
        expired = aux['expired']
        eligible = ok & expired & aux['at_risk_raw']
        for delta in (-60, -30, 30, 60):
            # An offset shifted to/before landmark is not interpretable as
            # a valid future ICU death: mark its label unknown, not negative.
            hypothetical = los + delta
            unknown_eventtime = eligible & (hypothetical <= h * 60)
            mask = ok & ~unknown_eventtime
            y = eligible & (hypothetical > h * 60) & (hypothetical <= h * 60 + HORIZON)
            rows.append({'source': source, 'landmark_h': h,
                         'scenario': f'expired_proxy_shift_{delta:+d}min',
                         'n': int(mask.sum()), 'events': int(y[mask].sum()),
                         'excluded_shifted_before_landmark': int(unknown_eventtime.sum())})
    else:
        # MIMIC observes deathtime; only report an artificial endpoint boundary
        # perturbation as an audit, never as a replacement for primary labels.
        death = aux['death']
        start = aux['start']
        end = aux['end']
        discharge = aux['discharge']
        for delta in (-60, 60):
            shifted = death + pd.to_timedelta(delta, unit='m')
            event_shifted = ok & (shifted > start).fillna(False).to_numpy() & (
                shifted <= end).fillna(False).to_numpy() & (shifted <= discharge).fillna(False).to_numpy()
            rows.append({'source': source, 'landmark_h': h,
                         'scenario': f'artificial_deathtime_shift_{delta:+d}min',
                         'n': int(ok.sum()), 'events': int(event_shifted.sum())})
    for row in rows:
        row['prevalence'] = row['events'] / row['n'] if row['n'] else None
        row['delta_n_vs_primary'] = row['n'] - base['n']
        row['delta_events_vs_primary'] = row['events'] - base['events']
    return rows


def check_reference(frame, reference):
    req = {'cohort', 'landmark_h', 'n', 'events'}
    if not req.issubset(reference.columns):
        raise ValueError(f'Stage7E reference requires columns {sorted(req)}')
    ref = reference.loc[reference.cohort.eq('external') | reference.cohort.eq('internal'),
                        ['cohort', 'landmark_h', 'n', 'events']].drop_duplicates()
    if len(ref) != 8:
        raise AssertionError(f'Expected 8 unique Stage7E cohort/horizon records, got {len(ref)}')
    current = frame.copy()
    current['cohort'] = current.source.map({'mimic': 'internal', 'eicu': 'external'})
    # Internal Stage7E is MIMIC holdout, while current audit is the ENTIRE
    # MIMIC at-risk cohort. External eICU full population is comparable.
    current = current[current.cohort.eq('external')][['cohort', 'landmark_h', 'n', 'events']]
    ref = ref[ref.cohort.eq('external')]
    merged = current.merge(ref, on=['cohort', 'landmark_h'], how='outer',
                           suffixes=('_audit', '_stage7e'), validate='one_to_one', indicator=True)
    if len(merged) != 4 or (merged._merge != 'both').any():
        raise AssertionError('External landmark reference keys differ')
    for col in ['n', 'events']:
        merged[f'{col}_match'] = merged[f'{col}_audit'].eq(merged[f'{col}_stage7e'])
    if not (merged.n_match & merged.events_match).all():
        raise AssertionError('FAIL_CLOSED: external Stage7E cohort/event mismatch')
    return merged.drop(columns='_merge')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', required=True, type=Path)
    p.add_argument('--stage7e-metrics', required=True, type=Path)
    p.add_argument('--expanded-mimic', type=Path)
    p.add_argument('--expanded-eicu', type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    for key in ['db', 'stage7e_metrics']:
        if not getattr(args, key).is_file():
            raise FileNotFoundError(getattr(args, key))
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('FAIL_CLOSED: output directory is not empty')
    import duckdb
    con = duckdb.connect(str(args.db), read_only=True)
    try:
        m, e = cohorts(con)
    finally:
        con.close()
    categories, primary, alternatives, audits = [], [], [], []
    for source, d in [('mimic', m), ('eicu', e)]:
        for h in LANDMARKS:
            cls, ok, y, aux = adjudicate(d, source, h)
            for category, n in zip(*np.unique(cls, return_counts=True)):
                categories.append({'source': source, 'landmark_h': h, 'category': category, 'n': int(n)})
            primary.append({'source': source, 'landmark_h': h, 'n': int(ok.sum()),
                            'events': int(y[ok].sum()), 'index_adult_stays': len(d),
                            'excluded_n': int((~ok).sum())})
            alternatives.extend(scenarios(d, source, h, ok, y, aux))
            if source == 'eicu':
                patient = d.loc[ok, 'person'].astype(str)
                audits.append({'source': source, 'landmark_h': h,
                               'eligible_stays': int(ok.sum()),
                               'unique_patient_clusters': int(patient.nunique()),
                               'repeated_stay_count': int(patient.duplicated(keep=False).sum()),
                               'early_unknown_status_excluded': int(aux['early_unknown'].sum())})
            print(f'{source} {h}h: n={ok.sum():,} events={y[ok].sum():,}', flush=True)
    qc = check_reference(pd.DataFrame(primary), pd.read_csv(args.stage7e_metrics))
    if args.expanded_mimic or args.expanded_eicu:
        # Optional 24h cross-check, matching Stage7C reconciliation logic.
        from icu_stage7c_prospective_landmarks import qc_24
        for source, d, path in [('mimic', m, args.expanded_mimic), ('eicu', e, args.expanded_eicu)]:
            if path:
                if not path.is_file():
                    raise FileNotFoundError(path)
                check = qc_24(d, source, path)
                if check['shared_label_mismatch']:
                    raise AssertionError(f'FAIL_CLOSED: {source} 24h archived label mismatch')
                if source == 'mimic' and (check['new_only'], check['archived_only']) != (0, 8):
                    raise AssertionError('FAIL_CLOSED: MIMIC 24h Stage5B boundary changed')
                if source == 'eicu' and (check['new_only'], check['archived_only']) != (0, 93):
                    raise AssertionError('FAIL_CLOSED: eICU 24h Stage5B boundary changed')
    args.out.mkdir(parents=True, exist_ok=False) if not args.out.exists() else None
    pd.DataFrame(primary).to_csv(args.out / 'stage8c_primary_counts.csv', index=False)
    pd.DataFrame(categories).to_csv(args.out / 'stage8c_endpoint_categories.csv', index=False)
    pd.DataFrame(alternatives).to_csv(args.out / 'stage8c_endpoint_sensitivity.csv', index=False)
    pd.DataFrame(audits).to_csv(args.out / 'stage8c_patient_cluster_audit.csv', index=False)
    qc.to_csv(args.out / 'stage8c_stage7e_reference_gate.csv', index=False)
    manifest = {
        'status': 'EXPLORATORY_ENDPOINT_ADJUDICATION_NO_NEW_MODEL_PERFORMANCE',
        'design': 'read-only retrospective adjudication from Stage7C cohort and risk_labels',
        'external_reference_gate': 'PASS_EXACT_COUNTS_AND_EVENTS',
        'mimic_endpoint': 'deathtime after landmark, within next 24h, and no later than ICU outtime',
        'eicu_endpoint': 'Expired unitdischargestatus with unitdischargeoffset proxy; not verified death timestamp',
        'negative': 'observed to horizon or documented alive discharge before horizon',
        'unknown_early_eicu_status': 'excluded from primary; scenarios represent hypothetical classification bounds',
        'offset_perturbations': 'hypothetical proxy timing sensitivity, not proven death times',
        'internal_reference': 'Stage7E internal is holdout only; entire MIMIC at-risk count is NOT comparable',
        'source_sha256': {'stage7e_metrics': sha256(args.stage7e_metrics)},
        'no_patient_level_export': True,
        'no_external_recalibration_or_retraining': True,
        'db_read_only': True,
        'limitations': ['No independently confirmed eICU death time',
                        'Outcome misclassification remains unquantified; perturbations are sensitivity assumptions',
                        'Cannot equate ICU death with hospital death',
                        'No new AUROC, AUPRC, Brier or individual uncertainty is estimated',
                        'eICU already used for post-hoc choices; not untouched confirmation'],
    }
    (args.out / 'stage8c_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('STAGE 8C COMPLETE — ENDPOINT AUDIT ONLY; STAGE7E EXTERNAL GATE PASSED', flush=True)


if __name__ == '__main__':
    main()
