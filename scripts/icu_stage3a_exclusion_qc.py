#!/usr/bin/env python3
"""Read-only audit of Stage3 MIMIC early-chart exclusions; no patient-level export."""
import argparse,csv,json
from pathlib import Path
import duckdb
IDS=(223875,223876,225792)
def write_csv(path, rows, cols):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(cols);w.writerows(rows)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--db',required=True,help='Existing Stage3 DuckDB file; opened read-only')
    ap.add_argument('--out',required=True,help='Separate new output directory')
    args=ap.parse_args()
    path=Path(args.db);out=Path(args.out)
    if not path.exists():raise FileNotFoundError(path)
    out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(path),read_only=True)
    needed=['mimic_patients','mimic_admissions','mimic_icustays','mimic_chartevents','mimic_d_items']
    names={r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    missing=[x for x in needed if x not in names]
    if missing:raise RuntimeError(f'Missing raw tables in Stage3 DB: {missing}')
    # DuckDB TEMP tables are in memory, main DB always read-only.
    con.execute("""CREATE TEMP TABLE audit_base AS
      WITH ranked AS (
        SELECT i.stay_id,i.intime,i.outtime,i.subject_id,
           TRY_CAST(p.anchor_age AS DOUBLE) + year(i.intime)-TRY_CAST(p.anchor_year AS DOUBLE) AS age,
           date_diff('minute',i.intime,i.outtime) AS los_min,
           a.deathtime,
           row_number() OVER(PARTITION BY i.subject_id ORDER BY i.intime,i.stay_id) rn
        FROM mimic_icustays i JOIN mimic_admissions a
          ON i.hadm_id=a.hadm_id AND i.subject_id=a.subject_id
        JOIN mimic_patients p ON p.subject_id=i.subject_id)
      SELECT stay_id,intime FROM ranked
      WHERE rn=1 AND age>=18 AND los_min>=1440
       AND (deathtime IS NULL OR deathtime>intime+INTERVAL 1440 MINUTE)
    """)
    con.execute("""CREATE TEMP TABLE audit_hit AS
      SELECT DISTINCT b.stay_id,TRY_CAST(x.itemid AS BIGINT) AS itemid,
             COALESCE(NULLIF(TRIM(TRY_CAST(x.value AS VARCHAR)),''),'[BLANK]') AS val
      FROM audit_base b JOIN mimic_chartevents x USING(stay_id)
      WHERE TRY_CAST(x.itemid AS BIGINT) IN (223875,223876,225792)
       AND TRY_CAST(x.charttime AS TIMESTAMP)>=b.intime
       AND TRY_CAST(x.charttime AS TIMESTAMP)<b.intime+INTERVAL 6 HOUR
    """)
    nbase=con.execute('SELECT count(*) FROM audit_base').fetchone()[0]
    nany=con.execute('SELECT count(DISTINCT stay_id) FROM audit_hit').fetchone()[0]
    descriptions=con.execute('''SELECT TRY_CAST(itemid AS BIGINT),
      TRY_CAST(label AS VARCHAR),TRY_CAST(category AS VARCHAR),TRY_CAST(linksto AS VARCHAR)
      FROM mimic_d_items WHERE TRY_CAST(itemid AS BIGINT) IN (223875,223876,225792)
      ORDER BY TRY_CAST(itemid AS BIGINT)''').fetchall()
    counts=con.execute('''SELECT itemid, val, count(DISTINCT stay_id) AS n_stays
       FROM audit_hit GROUP BY itemid,val ORDER BY itemid,n_stays DESC,val''').fetchall()
    totals=con.execute('''SELECT itemid,count(DISTINCT stay_id) FROM audit_hit GROUP BY itemid ORDER BY itemid''').fetchall()
    combos=con.execute('''SELECT flags,count(*) FROM (
      SELECT stay_id,string_agg(CAST(itemid AS VARCHAR),'+ ' ORDER BY itemid) flags
      FROM (SELECT DISTINCT stay_id,itemid FROM audit_hit)
      GROUP BY stay_id) GROUP BY flags ORDER BY count(*) DESC''').fetchall()
    temp_desc=con.execute('''SELECT TRY_CAST(itemid AS BIGINT),TRY_CAST(label AS VARCHAR),TRY_CAST(unitname AS VARCHAR)
       FROM mimic_d_items WHERE TRY_CAST(itemid AS BIGINT) IN (676,678,223761,223762)
       ORDER BY TRY_CAST(itemid AS BIGINT)''').fetchall()
    write_csv(out/'item_descriptions.csv',descriptions,['itemid','label','category','linksto'])
    write_csv(out/'early_chart_value_counts.csv',counts,['itemid','value','distinct_eligible_stays'])
    write_csv(out/'early_chart_item_totals.csv',totals,['itemid','distinct_eligible_stays'])
    write_csv(out/'early_chart_item_combinations.csv',combos,['itemid_combination','distinct_eligible_stays'])
    write_csv(out/'temperature_item_metadata.csv',temp_desc,['itemid','label','unitname'])
    result={'eligible_before_early_chart_filter':nbase,'excluded_by_any_configured_itemid':nany,
         'remaining_after_filter':nbase-nany,'configured_itemids':list(IDS),
         'warning':'Item presence alone does not establish DNR or terminal status. Review actual item labels and values before selecting exclusion rules.',
         'source_database_opened_read_only':True,'patient_level_export':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('STAGE 3A QC COMPLETE (read-only):',out)
    print(json.dumps(result,indent=2))
    con.close()
if __name__=='__main__':main()
