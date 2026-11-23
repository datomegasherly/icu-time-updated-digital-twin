#!/usr/bin/env python3
"""Read-only ICU raw measurement audit. Outputs aggregated counts only.
Requires: pip install duckdb pandas
"""
import argparse, json
from pathlib import Path
import duckdb

VARS={'hr':'heartrate','rr':'respiration','spo2':'sao2','map':'systemicmean','temp':'temperature'}
MIMIC={'hr':[220045,211], 'rr':[220210,618], 'spo2':[220277,646], 'map':[220052,225312], 'temp':[223761,678]}

def s(p):return "'"+str(p).replace("'","''")+"'"
def run(con,query,path):
    con.execute(f"COPY ({query}) TO {s(path)} (HEADER, DELIMITER ',')")

def main():
    p=argparse.ArgumentParser(description='Aggregate-only raw ICU vitals audit (no writes to input)')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--mimic-landmark',type=Path,required=True)
    p.add_argument('--eicu-landmark',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--skip-mimic',action='store_true',help='Skip large MIMIC chartevents scan')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(':memory:')
    con.execute('SET threads=4')
    def scan(path):
        if not path.exists():raise FileNotFoundError(path)
        return f"read_csv_auto({s(path)}, header=true, compression='gzip', sample_size=10000, ignore_errors=false)"
    eicu=args.data_root/'eICU/vitalPeriodic.csv.gz'
    mimic=args.data_root/'MIMIC/icu/chartevents.csv.gz'
    micu=args.data_root/'MIMIC/icu/icustays.csv.gz'
    for x in (eicu,args.eicu_landmark,args.mimic_landmark):
        if not x.exists():raise FileNotFoundError(x)
    con.execute(f'CREATE VIEW ee AS SELECT TRY_CAST(stay_id AS BIGINT) stay_id FROM read_csv_auto({s(args.eicu_landmark)})')
    con.execute(f'''CREATE VIEW ev0 AS SELECT TRY_CAST(v.patientunitstayid AS BIGINT) stay_id,
       TRY_CAST(v.observationoffset AS DOUBLE) off,
       {','.join(f'TRY_CAST(v.{col} AS DOUBLE) AS {name}' for name,col in VARS.items())}
       FROM {scan(eicu)} v INNER JOIN ee e ON TRY_CAST(v.patientunitstayid AS BIGINT)=e.stay_id
       WHERE TRY_CAST(v.observationoffset AS DOUBLE) BETWEEN 0 AND 1440''')
    union=' UNION ALL '.join(f"SELECT stay_id, '{name}' feature, {name} val FROM ev0 WHERE {name} IS NOT NULL" for name in VARS)
    con.execute(f'CREATE VIEW ev AS {union}')
    def tempcat(v):
        return f"CASE WHEN {v} BETWEEN 20 AND 50 THEN 'celsius_20_50' WHEN {v} BETWEEN 68 AND 122 THEN 'fahrenheit_68_122' ELSE 'other' END"
    def create_report(prefix,view,with_item=False):
        item='CAST(itemid AS VARCHAR)' if with_item else "'vitalPeriodic'"
        run(con,f'''SELECT feature, {item} source, COUNT(*) raw_measurements,
          SUM(CASE WHEN val<0 THEN 1 ELSE 0 END) negative_n,
          SUM(CASE WHEN feature='map' AND val<=0 THEN 1 ELSE 0 END) nonpositive_map_n,
          SUM(CASE WHEN feature='rr' AND val=0 THEN 1 ELSE 0 END) rr_zero_n,
          MIN(val) min_value, MAX(val) max_value, MEDIAN(val) median_value
          FROM {view} GROUP BY 1,2 ORDER BY 1,2''',args.out/f'{prefix}_raw_distribution.csv')
        run(con,f'''SELECT {item} source, {tempcat('val')} temp_category, COUNT(*) raw_measurements,
          COUNT(DISTINCT stay_id) stays FROM {view} WHERE feature='temp'
          GROUP BY 1,2 ORDER BY 1,2''',args.out/f'{prefix}_temperature_units.csv')
        run(con,f'''WITH st AS (SELECT stay_id,
          MAX(CASE WHEN val BETWEEN 20 AND 50 THEN 1 ELSE 0 END) c,
          MAX(CASE WHEN val BETWEEN 68 AND 122 THEN 1 ELSE 0 END) f,
          MAX(CASE WHEN val NOT BETWEEN 20 AND 50 AND val NOT BETWEEN 68 AND 122 THEN 1 ELSE 0 END) other
          FROM {view} WHERE feature='temp' GROUP BY stay_id)
          SELECT COUNT(*) stays_with_temp, SUM(CASE WHEN c=1 AND f=1 THEN 1 ELSE 0 END) mixed_c_and_f_stays,
          SUM(CASE WHEN other=1 THEN 1 ELSE 0 END) stays_with_other_values FROM st''',args.out/f'{prefix}_temperature_mixed_stays.csv')
    print('Auditing eICU raw measurements...',flush=True)
    create_report('eicu','ev')
    if not args.skip_mimic:
        if not mimic.exists() or not micu.exists():raise FileNotFoundError(f'MIMIC raw paths missing: {mimic}, {micu}')
        con.execute(f'''CREATE VIEW mi AS SELECT TRY_CAST(i.stay_id AS BIGINT) stay_id,
           TRY_CAST(i.intime AS TIMESTAMP) intime
           FROM {scan(micu)} i INNER JOIN read_csv_auto({s(args.mimic_landmark)}) l
           ON TRY_CAST(i.stay_id AS BIGINT)=TRY_CAST(l.stay_id AS BIGINT)''')
        ids=sorted({v for a in MIMIC.values() for v in a})
        con.execute(f'''CREATE VIEW mv0 AS SELECT TRY_CAST(v.stay_id AS BIGINT) stay_id,
         TRY_CAST(v.itemid AS BIGINT) itemid, TRY_CAST(v.valuenum AS DOUBLE) val
         FROM {scan(mimic)} v INNER JOIN mi i ON TRY_CAST(v.stay_id AS BIGINT)=i.stay_id
         WHERE TRY_CAST(v.itemid AS BIGINT) IN ({','.join(map(str,ids))})
         AND TRY_CAST(v.charttime AS TIMESTAMP)>=i.intime
         AND TRY_CAST(v.charttime AS TIMESTAMP)<=i.intime+INTERVAL 24 HOUR
         AND TRY_CAST(v.valuenum AS DOUBLE) IS NOT NULL''')
        case='CASE '+' '.join(f"WHEN itemid IN ({','.join(map(str,vals))}) THEN '{k}'" for k,vals in MIMIC.items())+' END'
        con.execute(f'CREATE VIEW mv AS SELECT stay_id,itemid,val,{case} feature FROM mv0')
        print('Auditing MIMIC raw measurements (large scan)...',flush=True)
        create_report('mimic','mv',with_item=True)
    readme={"purpose":"Read-only source measurement audit; categories are heuristic and not validated units.","window":"0 to 1440 minutes after ICU admission (inclusive)","note":"Counts are per measurement except mixed_stays report. MIMIC item IDs use the supplied project mapping. No raw patient rows are exported.","mimic_skipped":args.skip_mimic}
    (args.out/'audit_metadata.json').write_text(json.dumps(readme,indent=2),encoding='utf-8')
    print('STAGE 2B COMPLETE:',args.out,flush=True)
if __name__=='__main__': main()
