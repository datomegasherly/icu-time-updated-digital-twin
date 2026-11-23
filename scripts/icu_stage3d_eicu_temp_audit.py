#!/usr/bin/env python3
"""Read-only audit of eICU vitalPeriodic temperature in the selected Stage 3C cohort.
Does not infer/convert units, change cohort, or export patient records.
"""
import argparse, json
from pathlib import Path
import duckdb
import pandas as pd

def qi(x): return '"' + str(x).replace('"','""') + '"'
def qs(x): return "'" + str(x).replace("'","''") + "'"

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--db', required=True)
    p.add_argument('--eicu', required=True, help='Stage 3C eicu_landmark.csv')
    p.add_argument('--out', required=True)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if not Path(a.db).is_file() or not Path(a.eicu).is_file():
        raise FileNotFoundError('Check --db and --eicu paths')
    con=duckdb.connect(a.db,read_only=True)
    try:
        for table, needed in [('eicu_vital',{'patientunitstayid','observationoffset','temperature'}),('eicu_patient',{'patientunitstayid'})]:
            cols={str(r[0]).casefold():str(r[0]) for r in con.execute(f'DESCRIBE {qi(table)}').fetchall()}
            missing=needed-set(cols)
            if missing:raise ValueError(f'{table}: missing {sorted(missing)}')
        con.execute(f'''CREATE TEMP VIEW selected_stays AS SELECT DISTINCT TRY_CAST(stay_id AS BIGINT) AS stay_id FROM read_csv_auto({qs(str(Path(a.eicu).resolve()))},header=true) WHERE TRY_CAST(stay_id AS BIGINT) IS NOT NULL''')
        con.execute('''CREATE TEMP VIEW temp_samples AS
           SELECT s.stay_id, TRY_CAST(v.observationoffset AS DOUBLE) AS offset_min,
             TRY_CAST(v.temperature AS DOUBLE) AS t,
             TRY_CAST(p.hospitalid AS VARCHAR) AS hospital_id
           FROM selected_stays s
           INNER JOIN eicu_vital v ON TRY_CAST(v.patientunitstayid AS BIGINT)=s.stay_id
           LEFT JOIN eicu_patient p ON TRY_CAST(p.patientunitstayid AS BIGINT)=s.stay_id
           WHERE TRY_CAST(v.observationoffset AS DOUBLE) >=0 AND TRY_CAST(v.observationoffset AS DOUBLE)<1440
             AND TRY_CAST(v.temperature AS DOUBLE) IS NOT NULL''')
        # These are numerical bands only, NOT verified measurement units.
        con.execute('''CREATE TEMP VIEW classified AS
          SELECT *, CASE WHEN t BETWEEN 20 AND 50 THEN 'C_range'
          WHEN t BETWEEN 68 AND 122 THEN 'F_range' ELSE 'other' END AS band
          FROM temp_samples''')
        bands=con.execute('''SELECT band, COUNT(*) n_rows,COUNT(DISTINCT stay_id) n_stays,
                     MIN(t) min_value,MAX(t) max_value FROM classified GROUP BY band ORDER BY band''').df()
        bands.to_csv(out/'temperature_numeric_bands.csv',index=False)
        mixed=con.execute('''WITH s AS (SELECT stay_id,
                    MAX(CASE WHEN band='C_range' THEN 1 ELSE 0 END) has_c,
                    MAX(CASE WHEN band='F_range' THEN 1 ELSE 0 END) has_f,
                    MAX(CASE WHEN band='other' THEN 1 ELSE 0 END) has_other
                    FROM classified GROUP BY stay_id)
                    SELECT COUNT(*) total_with_temp, SUM(CASE WHEN has_c=1 AND has_f=1 THEN 1 ELSE 0 END) mixed_c_f_stays,
                     SUM(has_other) stays_with_other FROM s''').df().iloc[0].to_dict()
        hosp=con.execute('''WITH ss AS (SELECT p.hospitalid::VARCHAR hospital_id, COUNT(*) selected_stays
                    FROM selected_stays s JOIN eicu_patient p ON TRY_CAST(p.patientunitstayid AS BIGINT)=s.stay_id GROUP BY 1),
                   agg AS (SELECT hospital_id,COUNT(*) measurements,COUNT(DISTINCT stay_id) stays_with_temp,
                         SUM(CASE WHEN band='C_range' THEN 1 ELSE 0 END) c_rows,
                         SUM(CASE WHEN band='F_range' THEN 1 ELSE 0 END) f_rows,
                         SUM(CASE WHEN band='other' THEN 1 ELSE 0 END) other_rows
                         FROM classified GROUP BY 1)
                   SELECT ss.hospital_id,ss.selected_stays,COALESCE(agg.stays_with_temp,0) stays_with_temp,
                     COALESCE(agg.measurements,0) measurements,
                     COALESCE(agg.c_rows,0) c_rows,COALESCE(agg.f_rows,0) f_rows,COALESCE(agg.other_rows,0) other_rows
                   FROM ss LEFT JOIN agg USING(hospital_id) WHERE ss.selected_stays>=20
                   ORDER BY ss.selected_stays DESC''').df()
        hosp.to_csv(out/'hospital_temperature_coverage.csv',index=False)
        n_selected=int(con.execute('SELECT COUNT(*) FROM selected_stays').fetchone()[0]); n_raw=int(con.execute('SELECT COUNT(*) FROM classified').fetchone()[0]);
        report={'selected_stays':n_selected,'first_24h_temperature_readings':n_raw,
                'stays_with_temperature':int(mixed.get('total_with_temp') or 0),
                'mixed_numeric_range_stays':int(mixed.get('mixed_c_f_stays') or 0),
                'stays_with_out_of_band_temperature':int(mixed.get('stays_with_other') or 0),
                'hospitals_reported_min_20_stays':int(len(hosp)),
                'interpretation':'Numerical band audit only. No raw units in vitalPeriodic temperature; do not equate numeric band to verified unit.',
                'source_db_read_only':True, 'changes_to_existing_outputs':False}
        (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('STAGE 3D COMPLETE',json.dumps(report,indent=2))
    finally:
        con.close()
if __name__=='__main__':main()
