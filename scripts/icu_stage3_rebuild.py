#!/usr/bin/env python3
"""STAGE 3 DRAFT rebuild, independent DB; raw measurement QC and ICU mortality proxy.
Requires scientific review of unit-heuristics and cutoffs before manuscript use.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import duckdb
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_ROOT = Path(r"F:\Articles\1\data")
DEFAULT_DB = Path(r"F:\Articles\1\project\icu_landmark.duckdb")
FEATURES = {
    "vitals": ["hr", "rr", "spo2", "map", "temp"],
    "labs": ["creatinine", "sodium", "potassium", "bun", "glucose", "lactate"],
}
RAW = {
    "mimic_patients": ("MIMIC/hosp/patients.csv.gz", "mimic_patients"),
    "mimic_admissions": ("MIMIC/hosp/admissions.csv.gz", "mimic_admissions"),
    "mimic_icustays": ("MIMIC/icu/icustays.csv.gz", "mimic_icustays"),
    "mimic_labevents": ("MIMIC/hosp/labevents.csv.gz", "mimic_labevents"),
    "mimic_chartevents": ("MIMIC/icu/chartevents.csv.gz", "mimic_chartevents"),
    "mimic_d_items": ("MIMIC/icu/d_items.csv.gz", "mimic_d_items"),
    "mimic_d_labitems": ("MIMIC/hosp/d_labitems.csv.gz", "mimic_d_labitems"),
    "eicu_patient": ("eICU/patient.csv.gz", "eicu_patient"),
    "eicu_lab": ("eICU/lab.csv.gz", "eicu_lab"),
    "eicu_vital": ("eICU/vitalPeriodic.csv.gz", "eicu_vital"),
}

# Conservative initial range rules for protocol review; not clinically validated cut-offs.
# Temperature: numeric heuristics (20-50 C or 68-122 F) are mutually exclusive.
# RR=0 retained intentionally. All invalid values become SQL NULL before aggregation.
BOUNDS = {
    'hr': (20, 300), 'rr': (0, 90), 'spo2': (40, 100),
    'map': (10, 250), 'creatinine': (0.1, 30),
    'sodium': (90, 190), 'potassium': (1, 12), 'bun': (1, 250),
    'glucose': (10, 1500), 'lactate': (0, 40),
}

def cleaned_value(feature, expr):
    if feature == 'temp':
        return (f"CASE WHEN {expr} BETWEEN 20 AND 50 THEN {expr} "
                f"WHEN {expr} BETWEEN 68 AND 122 THEN ({expr}-32)*5/9 "
                f"ELSE NULL END")
    lo, hi = BOUNDS[feature]
    return f"CASE WHEN {expr} BETWEEN {lo} AND {hi} THEN {expr} ELSE NULL END"

def qid(s): return '"' + str(s).replace('"', '""') + '"'
def sqlstr(s): return "'" + str(s).replace("'", "''") + "'"
def say(s): print(s, flush=True)

def validate(cfg, path):
    errors = []
    for k in ("landmark_minute", "horizon_minute"):
        if not isinstance(cfg.get(k), int) or cfg[k] <= 0:
            errors.append(f"{k} must be a positive integer")
    for ds in ("mimic", "eicu"):
        x = cfg.get(ds, {})
        for typ, feats in FEATURES.items():
            m = x.get(typ, {})
            for f in feats:
                v = m.get(f)
                if ds == "mimic":
                    if not isinstance(v, list) or not v or any(not isinstance(i, int) for i in v):
                        errors.append(f"{ds}.{typ}.{f}: provide integer itemids")
                elif typ == "vitals":
                    if not isinstance(v, dict) or not v.get("column") or not v.get("offset_column"):
                        errors.append(f"eicu.vitals.{f}: provide column and offset_column")
                elif not isinstance(v, list) or not v or any(not isinstance(i, str) for i in v):
                    errors.append(f"eicu.labs.{f}: provide exact labName strings")
        evs = x.get("event_components", [])
        if not isinstance(evs, list) or not evs:
            errors.append(f"{ds}.event_components: define at least one event")
        if ds == "eicu":
            for idx, ev in enumerate(evs):
                if not ev.get("offset_column"):
                    errors.append(f"eicu.event_components[{idx}]: offset_column is required")
        if not isinstance(x.get("death_component"), str) or not x["death_component"]:
            errors.append(f"{ds}.death_component must name a component")
    if errors:
        raise ValueError("Configuration validation failed:\n- " + "\n- ".join(errors))

def preflight(root, config):
    say(f"1/7: Preflight | config: {config}")
    if not config.exists():
        raise FileNotFoundError(f"Config file not found: {config}")
    cfg = json.loads(config.read_text(encoding="utf-8-sig"))
    validate(cfg, config)
    say("  ok  Configuration validated successfully.")
    return cfg

def _column_map(con, table):
    return {str(row[0]).casefold(): str(row[0]) for row in con.execute(f"DESCRIBE {qid(table)}").fetchall()}

def _resolve_column(columns, requested, table):
    actual = columns.get(str(requested).casefold())
    if actual is None:
        raise ValueError(f"{table} lacks required column (case-insensitive): {requested}")
    return actual

def import_raw(con, root, cfg, force):
    say("2/7: Loading/reusing raw tables")
    ids = sorted(set(i for f in FEATURES['vitals'] for i in cfg['mimic']['vitals'][f]) | set(cfg['mimic']['exclude_dnr_itemids']) | set(cfg['mimic']['exclude_terminal_itemids']))
    for key, (rel_path, table) in RAW.items():
        exists = con.execute(f"SELECT count(*) FROM information_schema.tables WHERE table_name = {sqlstr(table)}").fetchone()[0]
        if exists and not force:
            raise RuntimeError(f"Raw table {table} already exists in database. Use a NEW Stage3 DB, or --force-reload intentionally. Existing cached tables must not be silently reused.")
        p = root / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
        psql = sqlstr(str(p))
        read = f"read_csv_auto({psql}, header=true, compression='gzip', sample_size=-1, ignore_errors=true)"
        if table == "mimic_chartevents":
            read = f"(SELECT * FROM {read} WHERE TRY_CAST(itemid AS BIGINT) IN ({','.join(map(str, ids))}))"
        say(f"  importing {table} from {p.name} ...")
        con.execute(f"CREATE OR REPLACE TABLE {qid(table)} AS SELECT * FROM {read}")

def build_mimic(con, cfg):
    say("3/7: Building MIMIC landmark cohort")
    con.execute("""CREATE OR REPLACE TEMP TABLE mc AS
      SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
       p.gender AS sex,
       TRY_CAST(p.anchor_age AS DOUBLE) + year(i.intime) - TRY_CAST(p.anchor_year AS DOUBLE) AS age,
       date_diff('minute', i.intime, i.outtime) AS los_min,
       a.deathtime AS death_ts,
       row_number() OVER(PARTITION BY i.subject_id ORDER BY i.intime, i.stay_id) rn
      FROM mimic_icustays i
      JOIN mimic_admissions a ON i.hadm_id = a.hadm_id AND i.subject_id = a.subject_id
      JOIN mimic_patients p ON i.subject_id = p.subject_id""")
    lm = cfg["landmark_minute"]
    con.execute(f"CREATE OR REPLACE TEMP TABLE mc0 AS SELECT * FROM mc WHERE rn=1 AND age>=18 AND los_min>={lm} AND (death_ts IS NULL OR death_ts > intime + INTERVAL {lm} MINUTE)")
    n0 = con.execute("SELECT count(*) FROM mc0").fetchone()[0]

    dnr = cfg['mimic']['exclude_dnr_itemids']
    term = cfg['mimic']['exclude_terminal_itemids']
    con.execute(f"""CREATE OR REPLACE TEMP TABLE mc1 AS SELECT c.* FROM mc0 c WHERE NOT EXISTS (
      SELECT 1 FROM mimic_chartevents x WHERE x.stay_id=c.stay_id
      AND TRY_CAST(x.itemid AS BIGINT) IN ({','.join(map(str, dnr+term))})
      AND TRY_CAST(x.charttime AS TIMESTAMP)>=c.intime
      AND TRY_CAST(x.charttime AS TIMESTAMP)<c.intime+INTERVAL 6 HOUR)""")
    n1 = con.execute("SELECT count(*) FROM mc1").fetchone()[0]
    say(f"  MIMIC early-chart exclusions: {n0-n1} (verify mapped item meanings against d_items)")

    measures = []
    lm = cfg["landmark_minute"]
    for f in FEATURES['vitals']:
        ids_str = ','.join(map(str, cfg['mimic']['vitals'][f]))
        val = cleaned_value(f, "TRY_CAST(x.valuenum AS DOUBLE)")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE mv_{f} AS
          WITH cleaned AS (SELECT c.stay_id, TRY_CAST(x.charttime AS TIMESTAMP) t, {val} AS v
          FROM mc1 c LEFT JOIN mimic_chartevents x ON c.stay_id=x.stay_id
            AND TRY_CAST(x.itemid AS BIGINT) IN ({ids_str})
            AND TRY_CAST(x.charttime AS TIMESTAMP) >= c.intime
            AND TRY_CAST(x.charttime AS TIMESTAMP) < c.intime + INTERVAL {lm} MINUTE)
          SELECT stay_id, avg(v) {f}_mean, min(v) {f}_min, max(v) {f}_max,
            arg_max(v, t) FILTER (WHERE v IS NOT NULL) {f}_last, count(v) {f}_n
          FROM cleaned GROUP BY stay_id""")
        measures.append(f"mv_{f}")

    for f in FEATURES['labs']:
        ids_str = ','.join(map(str, cfg['mimic']['labs'][f]))
        val = cleaned_value(f, "TRY_CAST(x.valuenum AS DOUBLE)")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE ml_{f} AS
          WITH cleaned AS (SELECT c.stay_id, TRY_CAST(x.charttime AS TIMESTAMP) t, {val} AS v
          FROM mc1 c LEFT JOIN mimic_labevents x ON c.hadm_id=x.hadm_id
            AND TRY_CAST(x.itemid AS BIGINT) IN ({ids_str})
            AND TRY_CAST(x.charttime AS TIMESTAMP) >= c.intime
            AND TRY_CAST(x.charttime AS TIMESTAMP) < c.intime + INTERVAL {lm} MINUTE)
          SELECT stay_id, avg(v) {f}_mean, min(v) {f}_min, max(v) {f}_max,
            arg_max(v, t) FILTER (WHERE v IS NOT NULL) {f}_last, count(v) {f}_n
          FROM cleaned GROUP BY stay_id""")
        measures.append(f"ml_{f}")

    join_parts = " ".join([f"LEFT JOIN {m} USING(stay_id)" for m in measures])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE mf AS
      SELECT c.stay_id, c.subject_id AS group_id, c.age, c.sex, c.los_min,
        {", ".join(f"{f}_mean, {f}_min, {f}_max, {f}_last, {f}_n" for f in FEATURES['vitals'] + FEATURES['labs'])}
      FROM mc1 c {join_parts}""")
    return n0, n1

def build_eicu(con, cfg):
    say("4/7: Building eICU landmark cohort")
    pc = _column_map(con, "eicu_patient")
    p_unit = _resolve_column(pc, "patientunitstayid", "eicu_patient")
    p_health = _resolve_column(pc, "patienthealthsystemstayid", "eicu_patient")
    p_age = _resolve_column(pc, "age", "eicu_patient")
    p_gender = _resolve_column(pc, "gender", "eicu_patient")
    p_unit_discharge = _resolve_column(pc, "unitdischargeoffset", "eicu_patient")
    p_hosp_admit = pc.get("hospitaladmitoffset")

    admit_order = (f"TRY_CAST(p.{qid(p_hosp_admit)} AS DOUBLE) DESC NULLS LAST, p.{qid(p_unit)}"
                   if p_hosp_admit else f"p.{qid(p_unit)}")

    con.execute(f"""CREATE OR REPLACE TEMP TABLE ec AS SELECT p.*,
      p.{qid(p_unit)} AS patient_unit_stay_id,
      p.{qid(p_health)} AS patient_healthsystem_stay_id,
      CASE WHEN TRY_CAST(p.{qid(p_age)} AS DOUBLE) IS NOT NULL THEN TRY_CAST(p.{qid(p_age)} AS DOUBLE)
           WHEN TRY_CAST(p.{qid(p_age)} AS VARCHAR) LIKE '>%' THEN 90.0 ELSE NULL END age_num,
      TRY_CAST(p.{qid(p_unit_discharge)} AS DOUBLE) los_min,
      row_number() OVER(PARTITION BY p.{qid(p_health)} ORDER BY {admit_order}) rn
      FROM eicu_patient p""")

    dnr_col = _resolve_column(pc, cfg['eicu']['dnr_column'], "eicu_patient")
    term_col = _resolve_column(pc, cfg['eicu']['terminal_column'], "eicu_patient")
    def vals(col, values):
        clean_v = [v for v in values if v != "__NONE__"]
        if not clean_v:
            return "FALSE"
        return f"TRY_CAST({qid(col)} AS VARCHAR) IN ({','.join(sqlstr(v) for v in clean_v)})"

    con.execute(f"""CREATE OR REPLACE TEMP TABLE ec1 AS SELECT * FROM ec
      WHERE rn=1 AND age_num>=18 AND los_min>={cfg['landmark_minute']}
      AND NOT coalesce(({vals(dnr_col, cfg['eicu']['dnr_values'])}), false)
      AND NOT coalesce(({vals(term_col, cfg['eicu']['terminal_values'])}), false)""")
    n1 = con.execute("SELECT count(*) FROM ec1").fetchone()[0]

    lm = cfg['landmark_minute']
    feats = FEATURES['vitals'] + FEATURES['labs']
    
    for f in FEATURES['vitals']:
        v_cfg = cfg['eicu']['vitals'][f]
        col = v_cfg['column']
        off = v_cfg['offset_column']
        v_unit = _column_map(con, "eicu_vital").get(str(p_unit).casefold(), "patientunitstayid")
        val = cleaned_value(f, f"TRY_CAST(v.{qid(col)} AS DOUBLE)")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE ea_{f} AS
          WITH cleaned AS (SELECT p.patient_unit_stay_id stay_id, TRY_CAST(v.{qid(off)} AS DOUBLE) t, {val} v
          FROM ec1 p LEFT JOIN eicu_vital v ON v.{qid(v_unit)}=p.patient_unit_stay_id
          AND TRY_CAST(v.{qid(off)} AS DOUBLE)>=0 AND TRY_CAST(v.{qid(off)} AS DOUBLE)<{lm})
          SELECT stay_id, avg(v) {f}_mean, min(v) {f}_min, max(v) {f}_max,
            arg_max(v,t) FILTER (WHERE v IS NOT NULL) {f}_last, count(v) {f}_n
          FROM cleaned GROUP BY stay_id""")

    for f in FEATURES['labs']:
        names_sql = ','.join(sqlstr(n) for n in cfg['eicu']['labs'][f])
        val = cleaned_value(f, "TRY_CAST(l.labresult AS DOUBLE)")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE ea_{f} AS
          WITH cleaned AS (SELECT p.patient_unit_stay_id stay_id, TRY_CAST(l.labresultoffset AS DOUBLE) t, {val} v
          FROM ec1 p LEFT JOIN eicu_lab l ON l.patientunitstayid=p.patient_unit_stay_id
          AND lower(TRY_CAST(l.labname AS VARCHAR)) IN ({names_sql.lower()})
          AND TRY_CAST(l.labresultoffset AS DOUBLE)>=0 AND TRY_CAST(l.labresultoffset AS DOUBLE)<{lm})
          SELECT stay_id, avg(v) {f}_mean, min(v) {f}_min, max(v) {f}_max,
            arg_max(v,t) FILTER (WHERE v IS NOT NULL) {f}_last, count(v) {f}_n
          FROM cleaned GROUP BY stay_id""")

    select = []
    for f in feats:
        select.extend(f"a_{f}.{f}_{s} AS {f}_{s}" for s in ('mean', 'min', 'max', 'last', 'n'))
    joins = "".join(f" LEFT JOIN ea_{f} a_{f} ON a_{f}.stay_id=p.patient_unit_stay_id" for f in feats)
    con.execute(f"CREATE OR REPLACE TEMP TABLE ef AS SELECT p.patient_unit_stay_id AS stay_id, p.patient_unit_stay_id AS group_id, p.age_num AS age, p.{qid(p_gender)} AS sex, p.los_min, " + ", ".join(select) + " FROM ec1 p" + joins)
    return n1

def event_sql_mimic(cfg):
    parts = []
    for x in cfg['mimic']['event_components']:
        col = x['column']
        parts.append(f"SELECT c.stay_id, {sqlstr(x['name'])} AS component, TRY_CAST(a.{qid(col)} AS TIMESTAMP) AS ts FROM mc1 c JOIN mimic_admissions a USING(hadm_id) WHERE TRY_CAST(a.{qid(col)} AS TIMESTAMP) IS NOT NULL")
    return " UNION ALL ".join(parts)

def event_sql_eicu(con, cfg):
    pc = _column_map(con, "eicu_patient")
    parts = []
    for x in cfg['eicu']['event_components']:
        name = x['name']
        offset_col = _resolve_column(pc, x['offset_column'], "eicu_patient")
        off = f"TRY_CAST(p.{qid(offset_col)} AS DOUBLE)"
        where = ""
        if x.get("flag_column"):
            flag_col = _resolve_column(pc, x['flag_column'], "eicu_patient")
            vals = x.get("flag_values", [])
            where = f" AND lower(TRY_CAST(p.{qid(flag_col)} AS VARCHAR)) IN ({','.join(sqlstr(str(v).lower()) for v in vals)})"
        parts.append(f"SELECT p.patient_unit_stay_id AS stay_id, {sqlstr(name)} AS component, {off} AS event_min FROM ec1 p WHERE {off} IS NOT NULL{where}")
    return " UNION ALL ".join(parts)

def add_labels(con, cfg, ds):
    """Target: ICU death in (24,48] h, conditional on survival to 24h.
    eICU unit discharge Expired is an outcome-time proxy; NOT an exact death timestamp.
    """
    lm = cfg['landmark_minute']
    end = lm + cfg['horizon_minute']
    if ds == 'mimic':
        con.execute(f"""CREATE OR REPLACE TEMP TABLE labeled_mimic AS
          SELECT f.*, date_diff('minute',c.intime,c.death_ts) death_t,
            CASE WHEN c.death_ts IS NOT NULL AND c.death_ts <= c.outtime
                 AND c.death_ts > c.intime+INTERVAL {lm} MINUTE
                 AND c.death_ts <= c.intime+INTERVAL {end} MINUTE
                 THEN date_diff('minute',c.intime,c.death_ts) END event_t,
            CAST(CASE WHEN c.death_ts IS NOT NULL AND c.death_ts <= c.outtime
                 AND c.death_ts > c.intime+INTERVAL {lm} MINUTE
                 AND c.death_ts <= c.intime+INTERVAL {end} MINUTE
                 THEN 1 ELSE 0 END AS INTEGER) y_24h
          FROM mf f JOIN mc1 c USING(stay_id)
          WHERE (c.death_ts IS NOT NULL AND c.death_ts<=c.outtime
                AND c.death_ts>c.intime+INTERVAL {lm} MINUTE
                AND c.death_ts<=c.intime+INTERVAL {end} MINUTE)
             OR c.los_min>={end}""")
    else:
        pc = _column_map(con, 'eicu_patient')
        col = _resolve_column(pc, 'unitdischargestatus', 'eicu_patient')
        con.execute(f"""CREATE OR REPLACE TEMP TABLE labeled_eicu AS
          SELECT f.*, CASE WHEN lower(trim(CAST(p.{qid(col)} AS VARCHAR)))='expired'
                    THEN p.los_min END death_t,
            CASE WHEN lower(trim(CAST(p.{qid(col)} AS VARCHAR)))='expired'
                    AND p.los_min>{lm} AND p.los_min<={end}
                    THEN p.los_min END event_t,
            CAST(CASE WHEN lower(trim(CAST(p.{qid(col)} AS VARCHAR)))='expired'
                    AND p.los_min>{lm} AND p.los_min<={end} THEN 1 ELSE 0 END AS INTEGER) y_24h
          FROM ef f JOIN ec1 p ON f.stay_id=p.patient_unit_stay_id
          WHERE (lower(trim(CAST(p.{qid(col)} AS VARCHAR)))='expired'
                 AND p.los_min>{lm} AND p.los_min<={end}) OR p.los_min>={end}""")


def finalize_csv(con, cfg, ds, outdir):
    source = f"labeled_{ds}"
    feats = FEATURES['vitals'] + FEATURES['labs']
    fields = "stay_id, group_id, age, sex, los_min, death_t, event_t, y_24h, " + ", ".join(f"{f}_{s}" for f in feats for s in ("mean","min","max","last","n"))
    con.execute(f"CREATE OR REPLACE TEMP TABLE out_{ds} AS SELECT {fields} FROM {source}")
    p = outdir / f"{ds}_landmark.csv"
    con.execute(f"COPY (SELECT * FROM out_{ds}) TO {sqlstr(str(p))} (HEADER, DELIMITER ',')")
    n = con.execute(f"SELECT count(*) FROM out_{ds}").fetchone()[0]
    prev = con.execute(f"SELECT coalesce(avg(y_24h),0), min(y_24h), max(y_24h) FROM out_{ds}").fetchone()
    return p, n, prev

def validate_models(outdir, seed=2026, test_size=0.20):
    df_mimic = pd.read_csv(outdir / "mimic_landmark.csv")
    df_eicu = pd.read_csv(outdir / "eicu_landmark.csv")

    arm_features = {
        "M_vital": [f"{f}_{s}" for f in FEATURES['vitals'] for s in ("mean","min","max","last","n")] + ["age"],
        "M_lab": [f"{f}_{s}" for f in FEATURES['labs'] for s in ("mean","min","max","last","n")] + ["age"],
        "M_integrated": [f"{f}_{s}" for f in FEATURES['vitals']+FEATURES['labs'] for s in ("mean","min","max","last","n")] + ["age"]
    }

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df_mimic, groups=df_mimic['group_id']))
    df_train = df_mimic.iloc[train_idx]
    df_test = df_mimic.iloc[test_idx]

    rows = []
    for arm, cols in arm_features.items():
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
        ])
        pipe.fit(df_train[cols], df_train['y_24h'])

        # Internal Holdout
        prob_test = pipe.predict_proba(df_test[cols])[:, 1]
        rows.append({
            "dataset": "MIMIC", "arm": arm, "split": "internal_holdout",
            "n": len(df_test), "events": int(df_test['y_24h'].sum()), "prevalence": float(df_test['y_24h'].mean()),
            "auroc": float(roc_auc_score(df_test['y_24h'], prob_test)),
            "auprc": float(average_precision_score(df_test['y_24h'], prob_test)),
            "brier": float(brier_score_loss(df_test['y_24h'], prob_test))
        })

        # External Validation
        prob_eicu = pipe.predict_proba(df_eicu[cols])[:, 1]
        rows.append({
            "dataset": "eICU", "arm": arm, "split": "external_validation",
            "n": len(df_eicu), "events": int(df_eicu['y_24h'].sum()), "prevalence": float(df_eicu['y_24h'].mean()),
            "auroc": float(roc_auc_score(df_eicu['y_24h'], prob_eicu)),
            "auprc": float(average_precision_score(df_eicu['y_24h'], prob_eicu)),
            "brier": float(brier_score_loss(df_eicu['y_24h'], prob_eicu))
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(outdir / "validation_metrics.csv", index=False)
    (outdir / "validation_metrics.json").write_text(metrics_df.to_json(orient='records', indent=2), encoding='utf-8')
    say("  Validation metrics exported successfully.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--db', type=Path, default=DEFAULT_DB)
    ap.add_argument('--config', type=Path, default=None)
    ap.add_argument('--output-dir', type=Path, default=None)
    ap.add_argument('--force-reload', action='store_true')
    ap.add_argument('--train-models', action='store_true', help="Optional; not recommended until cohort audit accepted")
    args = ap.parse_args()

    cfg_path = args.config or Path(__file__).resolve().with_name("item_mapping.json")
    cfg = preflight(args.data_root, cfg_path)
    outdir = args.output_dir or args.db.parent
    outdir.mkdir(parents=True, exist_ok=True)
    if args.db.resolve() == DEFAULT_DB.resolve():
        raise ValueError("Refusing to open the legacy/default DB. Specify --db for a NEW stage3 database.")
    for ds in ("mimic", "eicu"):
        if (outdir / f"{ds}_landmark.csv").exists():
            raise FileExistsError(f"Refusing overwrite: {outdir / (ds+'_landmark.csv')}")

    con = duckdb.connect(str(args.db))
    try:
        import_raw(con, args.data_root, cfg, args.force_reload)
        n0, n1 = build_mimic(con, cfg)
        ne = build_eicu(con, cfg)
        add_labels(con, cfg, 'mimic')
        add_labels(con, cfg, 'eicu')
        finalize_csv(con, cfg, 'mimic', outdir)
        finalize_csv(con, cfg, 'eicu', outdir)
        # Export basic counts, distribution summaries and source-level audit before model fitting.
        report = {'protocol':'ICU mortality (24,48] h conditional on ICU presence at 24h',
                  'eicu_timing':'unitdischargestatus=Expired; unitdischargeoffset proxy',
                  'mimic_n_after_minimum_age_los_survival':n0,
                  'mimic_n_after_configured_early_chart_exclusions':n1,
                  'eicu_n_before_outcome_filter':ne,
                  'bounds':BOUNDS,
                  'temperature':'numeric heuristic 20..50 C, 68..122 F; otherwise missing',
                  'window':'[0,1440) minutes (prediction starts >1440)'}
        for ds in ('mimic','eicu'):
            n, y, min_los, max_los = con.execute(f"SELECT count(*), sum(y_24h), min(los_min), max(los_min) FROM out_{ds}").fetchone()
            report[ds]={'n':n, 'events':y, 'min_icu_los':min_los, 'max_icu_los':max_los}
        (outdir/'cohort_rebuild_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str), encoding='utf-8')
        if args.train_models:
            validate_models(outdir)
        say("Stage3 rebuild finished. Outputs are NOT yet approved for manuscript use.")
    finally:
        con.close()

if __name__ == "__main__":
    main()
