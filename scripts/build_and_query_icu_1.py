#!/usr/bin/env python3
"""Build auditable, config-driven ICU landmark datasets (DuckDB; no inferred labels)."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import duckdb

DEFAULT_ROOT = Path(r"F:\Articles\1\data")
DEFAULT_DB = Path(r"F:\Articles\1\project\icu_landmark.duckdb")
FEATURES = {"vitals": ["hr", "rr", "spo2", "map", "temp"],
            "labs": ["creatinine", "sodium", "potassium", "bun", "glucose", "lactate"]}
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

def qid(s): return '"' + str(s).replace('"', '""') + '"'
def sqlstr(s): return "'" + str(s).replace("'", "''") + "'"
def say(s): print(s, flush=True)

def template():
    return {
      "landmark_minute": None, "horizon_minute": None,
      "mimic": {
        "vitals": {k: [] for k in FEATURES["vitals"]},
        "labs": {k: [] for k in FEATURES["labs"]},
        "event_components": [], "death_component": None,
        "exclude_dnr_itemids": [], "exclude_terminal_itemids": []
      },
      "eicu": {
        "vitals": {k: {"column": None, "offset_column": None} for k in FEATURES["vitals"]},
        "labs": {k: [] for k in FEATURES["labs"]},
        "event_components": [], "death_component": None,
        "dnr_column": None, "dnr_values": [], "terminal_column": None, "terminal_values": []
      }
    }

def ensure_template(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(template(), indent=2), encoding="utf-8")

def validate(cfg, path):
    errors=[]
    for k in ("landmark_minute", "horizon_minute"):
        if not isinstance(cfg.get(k), int) or cfg[k] <= 0: errors.append(f"{k} must be a positive integer")
    for ds in ("mimic", "eicu"):
        x=cfg.get(ds, {})
        for typ, feats in FEATURES.items():
            m=x.get(typ, {})
            for f in feats:
                v=m.get(f)
                if ds == "mimic":
                    if not isinstance(v, list) or not v or any(not isinstance(i,int) for i in v): errors.append(f"{ds}.{typ}.{f}: provide one or more integer itemids")
                elif typ == "vitals":
                    if not isinstance(v,dict) or not v.get("column") or not v.get("offset_column"): errors.append(f"eicu.vitals.{f}: provide column and offset_column")
                elif not isinstance(v,list) or not v or any(not isinstance(i,str) for i in v): errors.append(f"eicu.labs.{f}: provide exact labName strings")
        if not isinstance(x.get("event_components"),list) or not x["event_components"]: errors.append(f"{ds}.event_components: define at least one event; labels will not be inferred")
        if not isinstance(x.get("death_component"),str) or not x["death_component"]: errors.append(f"{ds}.death_component: name one configured event component or explicitly use 'none'")
    m=cfg.get("mimic",{}); e=cfg.get("eicu",{})
    if not m.get("exclude_dnr_itemids") or not m.get("exclude_terminal_itemids"): errors.append("mimic: configure exclude_dnr_itemids and exclude_terminal_itemids")
    for kind in ("dnr","terminal"):
        if not e.get(kind+"_column") or not e.get(kind+"_values"): errors.append(f"eicu: configure {kind}_column and {kind}_values")
    if errors:
        ensure_template(path)
        raise ValueError("Configuration missing/incomplete; no cohort/labels were built.\nTemplate written/available at: %s\n- %s"%(path,"\n- ".join(errors)))

def preflight(root, config):
    say(f"1/6: preflight | data root: {root}")
    if not config.exists():
        ensure_template(config)
        raise ValueError(f"Config required; fill the generated template: {config}")
    cfg=json.loads(config.read_text(encoding="utf-8-sig"))
    validate(cfg,config)
    say("  ok  configuration")
    return cfg

def import_raw(con, root, cfg, force):
    say("3/6: raw tables | existing tables are reused; --force-reload rereads all sources")
    ids=sorted(set(i for t in FEATURES.values() for f in t for i in cfg['mimic'][('vitals' if f in FEATURES['vitals'] else 'labs')][f]))
    ids += sorted(set(cfg['mimic'].get('exclude_dnr_itemids',[])+cfg['mimic'].get('exclude_terminal_itemids',[])))
    for name,(rel,table) in RAW.items():
        exists=con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=? AND table_type='BASE TABLE'",[table]).fetchone()[0] > 0
        if exists and not force:
            old=con.execute(f"SELECT count(*) FROM {qid(table)}").fetchone()[0]
            say(f"  reuse {table}: {old:,} rows")
            continue
        path=root/rel
        if not path.is_file():
            raise FileNotFoundError(f"Cannot import {table}: source file is missing: {path}")
        psql=sqlstr(str(path))
        read=f"read_csv_auto({psql}, header=true, compression='gzip', sample_size=-1, ignore_errors=true)"
        if name=="mimic_chartevents":
            if not ids: raise ValueError("No configured MIMIC itemids")
            read=f"(SELECT * FROM {read} WHERE TRY_CAST(itemid AS BIGINT) IN ({','.join(map(str,ids))}))"
        say(f"  import {table} <- {path}")
        con.execute(f"CREATE OR REPLACE TABLE {qid(table)} AS SELECT * FROM {read}")
        say(f"    loaded {con.execute(f'SELECT count(*) FROM {qid(table)}').fetchone()[0]:,} rows")

def build_mimic(con,cfg):
    say("4/6: MIMIC cohort + 24h features")
    lm=cfg['landmark_minute']
    con.execute("""CREATE OR REPLACE TEMP TABLE mc AS
    SELECT 
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,
        a.admittime,
        a.dischtime,
        a.deathtime,
        p.gender,
        p.anchor_age AS age,
        date_diff('minute', i.intime, i.outtime) AS los_min,
        row_number() OVER(PARTITION BY i.subject_id ORDER BY i.intime) AS rn
    FROM mimic_icustays i 
    JOIN mimic_admissions a ON i.hadm_id = a.hadm_id AND i.subject_id = a.subject_id
    JOIN mimic_patients p ON i.subject_id = p.subject_id""")
    
    con.execute("CREATE OR REPLACE TEMP TABLE mc0 AS SELECT * FROM mc WHERE rn=1 AND age>=18 AND los_min>=1440")
    n0=con.execute("select count(*) from mc0").fetchone()[0]
    
    dnr=cfg['mimic']['exclude_dnr_itemids']; term=cfg['mimic']['exclude_terminal_itemids']
    con.execute(f"""CREATE OR REPLACE TEMP TABLE mc1 AS SELECT c.* FROM mc0 c WHERE NOT EXISTS (
      SELECT 1 FROM mimic_chartevents x WHERE x.stay_id=c.stay_id
      AND TRY_CAST(x.itemid AS BIGINT) IN ({','.join(map(str,dnr+term))})
      AND TRY_CAST(x.charttime AS TIMESTAMP)>=c.intime
      AND TRY_CAST(x.charttime AS TIMESTAMP)<c.intime+INTERVAL 6 HOUR)""")
    n1=con.execute("select count(*) from mc1").fetchone()[0]
    
    for typ,feats in FEATURES.items():
      table="mimic_chartevents" if typ=="vitals" else "mimic_labevents"
      itcol="itemid"; tcol="charttime"; valcol="valuenum"
      for f in feats:
        itemids=cfg['mimic'][typ][f]
        key="stay_id" if typ=="vitals" else "hadm_id"
        con.execute(f"""CREATE OR REPLACE TEMP TABLE z AS
          SELECT c.stay_id, TRY_CAST(x.{valcol} AS DOUBLE) val, TRY_CAST(x.{tcol} AS TIMESTAMP) tm
          FROM mc1 c JOIN {table} x ON x.{key}=c.{key}
          WHERE TRY_CAST(x.{itcol} AS BIGINT) IN ({','.join(map(str,itemids))})
            AND TRY_CAST(x.{tcol} AS TIMESTAMP)>=c.intime
            AND TRY_CAST(x.{tcol} AS TIMESTAMP)<c.intime+INTERVAL {lm} MINUTE
            AND TRY_CAST(x.{valcol} AS DOUBLE) IS NOT NULL""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE agg_{f} AS SELECT stay_id,
          avg(val) {f}_mean, min(val) {f}_min, max(val) {f}_max,
          arg_max(val,tm) {f}_last, count(*) {f}_n FROM z GROUP BY stay_id""")
          
    select=[f"a_{f}.{f}_{s} AS {f}_{s}" for f in FEATURES['vitals']+FEATURES['labs'] for s in ('mean','min','max','last','n')]
    joins="".join(f" LEFT JOIN agg_{f} a_{f} USING(stay_id)" for f in FEATURES['vitals']+FEATURES['labs'])
    con.execute("CREATE OR REPLACE TEMP TABLE mf AS SELECT c.*,"+",".join(select)+" FROM mc1 c"+joins)
    say(f"  MIMIC filters: raw ICU={con.execute('select count(*) from mc').fetchone()[0]:,}; index/adult/LOS24h={n0:,}; after first-6h DNR/terminal exclusion={n1:,}")
    return n0,n1

def build_eicu(con,cfg):
    say("5/6: eICU cohort + 24h features")
    con.execute("""CREATE OR REPLACE TEMP TABLE ec AS SELECT p.*,
      CASE WHEN TRY_CAST(p.age AS DOUBLE) IS NOT NULL THEN TRY_CAST(p.age AS DOUBLE)
           WHEN TRY_CAST(p.age AS VARCHAR) LIKE '>%' THEN 90.0 ELSE NULL END age_num,
      CASE WHEN TRY_CAST(p.age AS VARCHAR) LIKE '>%' THEN 1 ELSE 0 END age_over89,
      TRY_CAST(p.unitDischargeOffset AS DOUBLE) los_min,
      row_number() OVER(PARTITION BY p.patientHealthSystemStayID ORDER BY TRY_CAST(p.unitDischargeOffset AS DOUBLE), p.patientUnitStayID) rn
      FROM eicu_patient p""")
    dnr=cfg['eicu']['dnr_column']; term=cfg['eicu']['terminal_column']
    def vals(col,values): return f"TRY_CAST({qid(col)} AS VARCHAR) IN ({','.join(sqlstr(v) for v in values)})"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE ec1 AS SELECT * FROM ec WHERE rn=1 AND age_num>=18 AND los_min>=1440
      AND NOT coalesce(({vals(dnr,cfg['eicu']['dnr_values'])}),false) AND NOT coalesce(({vals(term,cfg['eicu']['terminal_values'])}),false)""")
    n1=con.execute("select count(*) from ec1").fetchone()[0]
    lm=cfg['landmark_minute']; feats=FEATURES['vitals']+FEATURES['labs']
    for f in FEATURES['vitals']:
        spec=cfg['eicu']['vitals'][f]; col=spec['column']; off=spec['offset_column']
        con.execute(f"""CREATE OR REPLACE TEMP TABLE ea_{f} AS SELECT p.patientUnitStayID stay_id,
          avg(TRY_CAST(v.{qid(col)} AS DOUBLE)) {f}_mean, min(TRY_CAST(v.{qid(col)} AS DOUBLE)) {f}_min,
          max(TRY_CAST(v.{qid(col)} AS DOUBLE)) {f}_max,
          arg_max(TRY_CAST(v.{qid(col)} AS DOUBLE),TRY_CAST(v.{qid(off)} AS DOUBLE)) {f}_last,
          count(TRY_CAST(v.{qid(col)} AS DOUBLE)) {f}_n FROM ec1 p LEFT JOIN eicu_vital v
          ON v.patientUnitStayID=p.patientUnitStayID AND TRY_CAST(v.{qid(off)} AS DOUBLE)>=0
          AND TRY_CAST(v.{qid(off)} AS DOUBLE)<{lm} GROUP BY p.patientUnitStayID""")
    for f in FEATURES['labs']:
        names=cfg['eicu']['labs'][f]
        con.execute(f"""CREATE OR REPLACE TEMP TABLE ea_{f} AS SELECT p.patientUnitStayID stay_id,
          avg(TRY_CAST(l.labresult AS DOUBLE)) {f}_mean, min(TRY_CAST(l.labresult AS DOUBLE)) {f}_min,
          max(TRY_CAST(l.labresult AS DOUBLE)) {f}_max,
          arg_max(TRY_CAST(l.labresult AS DOUBLE),TRY_CAST(l.labresultoffset AS DOUBLE)) {f}_last,
          count(TRY_CAST(l.labresult AS DOUBLE)) {f}_n FROM ec1 p LEFT JOIN eicu_lab l
          ON l.patientUnitStayID=p.patientUnitStayID AND lower(TRY_CAST(l.labname AS VARCHAR)) IN ({','.join(sqlstr(s.lower()) for s in names)})
          AND TRY_CAST(l.labresultoffset AS DOUBLE)>=0 AND TRY_CAST(l.labresultoffset AS DOUBLE)<{lm}
          GROUP BY p.patientUnitStayID""")
    select=[f"a_{f}.{f}_{s} AS {f}_{s}" for f in feats for s in ('mean','min','max','last','n')]
    joins="".join(f" LEFT JOIN ea_{f} a_{f} ON a_{f}.stay_id=p.patientUnitStayID" for f in feats)
    con.execute("CREATE OR REPLACE TEMP TABLE ef AS SELECT p.patientUnitStayID stay_id, p.patientUnitStayID group_id, p.age_num age, p.gender sex, p.los_min, p.age_over89, "+",".join(select)+" FROM ec1 p"+joins)
    say(f"  eICU filters: input={con.execute('select count(*) from ec').fetchone()[0]:,}; index/adult/LOS24h/DNR-terminal={n1:,}")
    return n1

def event_sql_mimic(cfg):
    parts=[]
    for x in cfg['mimic']['event_components']:
        col=x['column']
        parts.append(f"SELECT c.stay_id,{sqlstr(x['name'])} component,TRY_CAST(a.{qid(col)} AS TIMESTAMP) ts FROM mc1 c JOIN mimic_admissions a USING(hadm_id) WHERE TRY_CAST(a.{qid(col)} AS TIMESTAMP) IS NOT NULL")
    return " UNION ALL ".join(parts)

def event_sql_eicu(cfg):
    parts=[]
    for x in cfg['eicu']['event_components']:
        off_col = x.get('offset_column', 'unitdischargeoffset')
        off = f"TRY_CAST(p.{qid(off_col)} AS DOUBLE)"
        where = ""
        if x.get('flag_column'):
            vals = x.get('flag_values', [])
            where = f" AND lower(TRY_CAST(p.{qid(x['flag_column'])} AS VARCHAR)) IN ({','.join(sqlstr(v.lower()) for v in vals)})"
        parts.append(f"SELECT p.patientUnitStayID stay_id, {sqlstr(x['name'])} component, {off} event_min FROM ec1 p WHERE {off} IS NOT NULL{where}")
    return " UNION ALL ".join(parts)

def add_labels(con,cfg,ds):
    lm=cfg['landmark_minute']; horizon=cfg['horizon_minute']; x=cfg[ds]
    con.execute(f"CREATE OR REPLACE TEMP TABLE ev_{ds} AS {event_sql_mimic(cfg) if ds=='mimic' else event_sql_eicu(cfg)}")
    death=x['death_component']
    death_expr="NULL::DOUBLE" if death=="none" else f"min(CASE WHEN component={sqlstr(death)} THEN event_min END)"
    if ds=='mimic':
        con.execute(f"""CREATE OR REPLACE TEMP TABLE el_{ds} AS SELECT stay_id,
          min(CASE WHEN event_min>{lm} THEN event_min END) event_t,{death_expr} death_t
          FROM (SELECT stay_id,component,date_diff('minute',c.intime,ts) event_min FROM ev_{ds} e JOIN mc1 c USING(stay_id)) GROUP BY stay_id""")
        base="mf"
    else:
        con.execute(f"""CREATE OR REPLACE TEMP TABLE el_{ds} AS SELECT stay_id,
          min(CASE WHEN event_min>{lm} THEN event_min END) event_t,{death_expr} death_t
          FROM (SELECT stay_id,component,event_min FROM ev_{ds}) GROUP BY stay_id""")
        base="ef"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE labeled_{ds} AS SELECT b.*,
      e.death_t,e.event_t,CAST(CASE WHEN e.event_t>{lm} AND e.event_t<={lm+horizon} THEN 1 ELSE 0 END AS INTEGER) y_24h
      FROM {base} b LEFT JOIN el_{ds} e USING(stay_id)""")

def finalize_csv(con,cfg,ds,outdir):
    feats=FEATURES['vitals']+FEATURES['labs']
    table='labeled_mimic' if ds=='mimic' else 'labeled_eicu'
    if ds=='mimic':
        con.execute("CREATE OR REPLACE TEMP TABLE export_m AS SELECT l.*, l.subject_id AS group_id FROM labeled_mimic l")
        table='export_m'

    n_v=" + ".join(f"coalesce({f}_n,0)" for f in FEATURES['vitals'])
    n_l=" + ".join(f"coalesce({f}_n,0)" for f in FEATURES['labs'])
    cols=[", ".join([
        "*",
        "map_last AS map_t",
        "spo2_last AS spo2_t",
        f"({n_v}) AS n_v",
        f"({n_l}) AS n_l",
    ])]
    con.execute(f"CREATE OR REPLACE TEMP TABLE out_{ds} AS SELECT {cols[0]} FROM {table}")

    p=outdir/f"{ds}_landmark.csv"
    con.execute(f"COPY (SELECT * FROM out_{ds}) TO {sqlstr(str(p))} (HEADER, DELIMITER ',')")
    
    n_res = con.execute(f"SELECT count(*) FROM out_{ds}").fetchone()
    n = n_res[0] if n_res and n_res[0] is not None else 0
    
    prev_res = con.execute(f"SELECT coalesce(avg(y_24h),0), min(y_24h), max(y_24h) FROM out_{ds}").fetchone()
    if prev_res and prev_res[0] is not None:
        prev = prev_res
    else:
        prev = (0.0, 0, 0)
        
    return p,n,prev

def audit(con,cfg,paths,summary,outdir):
    rows=[]
    for ds in ('mimic','eicu'):
        t='out_'+ds
        n=con.execute(f"select count(*) from {t}").fetchone()[0]
        for f in FEATURES['vitals']+FEATURES['labs']:
            r=con.execute(f"select count(*) filter(where {f}_n>0),count(*) filter(where {f}_last is null),count(*) from {t}").fetchone()
            rows.append({"dataset":ds,"metric":f+"_coverage","value":r[0],"missing_last":r[1],"n":n})
        prevalence=con.execute(f"select avg(y_24h) from {t}").fetchone()[0]
        rows.append({"dataset":ds,"metric":"y_24h_prevalence","value":prevalence,"n":n})
        if ds == 'eicu':
            rows.append({"dataset":ds,"metric":"age_gt_89_flagged","value":con.execute("select count(*) from ef where age_over89=1").fetchone()[0],"n":n})
    audit_csv=outdir/'data_quality_audit.csv'
    with audit_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','metric','value','missing_last','n']); w.writeheader(); w.writerows(rows)
    audit_json=outdir/'data_quality_audit.json'
    audit_json.write_text(json.dumps({"config":str(paths['config']),"outputs":{k:str(v) for k,v in paths.items()},"summary":summary,"metrics":rows},indent=2),encoding='utf-8')
    return audit_csv,audit_json

def main(argv=None):
    ap=argparse.ArgumentParser(description="Config-driven MIMIC-IV demo/eICU ICU landmark builder")
    ap.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--db',type=Path,default=DEFAULT_DB)
    ap.add_argument('--config',type=Path,default=None)
    ap.add_argument('--output-dir',type=Path,default=None)
    ap.add_argument('--force-reload',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(argv)
    config=a.config or Path(__file__).resolve().with_name('item_mapping.json')
    cfg=preflight(a.data_root,config)
    outdir=a.output_dir or a.db.parent
    if a.dry_run: return 0
    a.db.parent.mkdir(parents=True,exist_ok=True); outdir.mkdir(parents=True,exist_ok=True)
    say(f"2/6: connect -> {a.db}")
    con=duckdb.connect(str(a.db))
    try:
        import_raw(con,a.data_root,cfg,a.force_reload)
        n0,n1=build_mimic(con,cfg)
        ne=build_eicu(con,cfg)
        say("6/6: configured events/labels and CSV exports")
        add_labels(con,cfg,'mimic'); add_labels(con,cfg,'eicu')
        p1,c1,prev1=finalize_csv(con,cfg,'mimic',outdir)
        p2,c2,prev2=finalize_csv(con,cfg,'eicu',outdir)
        summary={"mimic":{"index_adult_los24h":n0,"after_exclusion":n1,"export_rows":c1,"y_24h_prevalence":prev1[0]},
                 "eicu":{"after_cohort_filters":ne,"export_rows":c2,"y_24h_prevalence":prev2[0]}}
        audit_paths=audit(con,cfg,{"config":config,"mimic_csv":p1,"eicu_csv":p2},summary,outdir)
        say(f"  CSV: {p1} | {p2}\n  Audit: {audit_paths[0]} | {audit_paths[1]}")
    finally: con.close()
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR / خطا: {exc}",file=sys.stderr); raise
