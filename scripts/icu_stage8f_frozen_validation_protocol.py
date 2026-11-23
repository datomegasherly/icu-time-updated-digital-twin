#!/usr/bin/env python3
"""Stage 8F: audit-preserving FUTURE validation protocol lock (no patient data)."""
import argparse, hashlib, json, csv, sys
from pathlib import Path
from datetime import datetime, timezone

EXPECTED='PASS_EVIDENCE_INTEGRITY_EXPLORATORY_NOT_SUBMISSION_READY'
LANDMARKS=[6,12,18,24]
REQUIRED=['stage8e_manifest.json','stage8e_integrity_gates.csv','stage8e_evidence_summary.csv','stage8e_claim_matrix.csv','stage8e_scientific_report_FA.md','stage8e_scientific_report_EN.md']

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--s8e',required=True,help='Stage 8E output folder, read-only')
    ap.add_argument('--out',required=True,help='New empty output folder')
    ap.add_argument('--project',help='Project root for optional source-code and mapping fingerprints')
    ap.add_argument('--db',help='Optional local database for SHA256 fingerprint; large and slow')
    ap.add_argument('--mode',choices=['create','verify'],default='create')
    a=ap.parse_args();src=Path(a.s8e);out=Path(a.out)
    src_files={n:src/n for n in REQUIRED}; errors=[]
    for n,p in src_files.items():
        if not p.is_file():errors.append('MISSING_SOURCE '+n)
    if errors:raise SystemExit('\n'.join(errors))
    if out.resolve()==src.resolve():raise SystemExit('FAIL: output cannot equal input')
    src_hash={n:digest(p) for n,p in src_files.items()}
    manifest=json.loads(src_files['stage8e_manifest.json'].read_text(encoding='utf8'))
    checks=manifest.get('checks',[])
    if manifest.get('status')!=EXPECTED:errors.append('FAIL: Stage8E integrity status')
    # Stage8E has stored outcomes as dictionaries. Gate report is the authoritative readable output.
    gates=readcsv(src_files['stage8e_integrity_gates.csv'])
    if len(gates)!=16:errors.append(f'FAIL: expected 16 integrity gates, found {len(gates)}')
    if any(r.get('status')!='PASS' for r in gates):errors.append('FAIL: not all gates passed')
    ev=readcsv(src_files['stage8e_evidence_summary.csv'])
    combos={(r.get('landmark_h'),r.get('feature_timing')) for r in ev}
    if combos != {(str(h),m) for h in LANDMARKS for m in ['measured','exclude']}:
        errors.append('FAIL: evidence grid expected 4 landmarks × measured/exclude')
    if any(float(r['delta_auroc']) <= 0 for r in ev):errors.append('FAIL: unexpected nonpositive archived AUROC difference')
    for r in ev:
        if int(float(r['n_shift_scenarios']))!=4:errors.append('FAIL: 8D shift grid mismatch')
    if not manifest.get('not_independent_confirmation',False):errors.append('FAIL: independent validation limitation missing')
    if manifest.get('no_refit') is not True:errors.append('FAIL: 8E refit gate not stated')
    if errors:raise SystemExit('\n'.join(errors))
    project_hash={}
    if a.project:
        folder=Path(a.project)
        for n in ['item_mapping_stage3c.json','icu_stage7e_unified_dynamic_audit.py','icu_stage8a_lab_availability.py','icu_stage8b_availability_aware.py','icu_stage8c_endpoint_adjudication.py','icu_stage8d_endpoint_robust_validation.py','icu_stage8e_scientific_evidence_lock.py']:
            p=folder/n
            project_hash[n]=digest(p) if p.is_file() else 'NOT_PRESENT'
    db_hash=None
    if a.db:
        if not Path(a.db).is_file():raise SystemExit('FAIL: database does not exist')
        db_hash=digest(Path(a.db))
    protocol={
      'name':'ICU Digital Twin Stage 8F — prospective-from-now validation protocol',
      'version':'1.0.0','status':'PROTOCOL_LOCKED_FOR_FUTURE_EVALUATION_ONLY',
      'scope':'Retrospective patient-specific time-updated prediction surrogate; not clinical or counterfactual twin',
      'legacy_evidence':'ALL PRE-8F RESULTS EXPLORATORY; external eICU was consulted for post-hoc feature decisions',
      'locked_comparator':{'source':'MIMIC-IV 6h training','models':['integrated','lab-only'],
       'features':'no temperature; no measurement-count (_n) columns; Stage8B conservative available-by-landmark lab timing (exclude)',
       'landmarks_hours':LANDMARKS,'horizon':'ICU death in [t,t+24h) among alive and in ICU at t; competing live ICU discharge non-event',
       'eicu_death_timestamp':'unitdischargeoffset is proxy, not verified event time',
       'no_external_retraining_or_recalibration':True},
      'primary_estimand':'External patient-cluster paired ΔAUROC = AUROC(integrated)-AUROC(lab-only) at 24h, full strict at-risk cohort',
      'secondary_estimands':['ΔAUROC at 6/12/18h','AUPRC','Brier','calibration slope/intercept','hospital-level heterogeneity','endpoint proxy ±30/60min sensitivity','feature availability quality'],
      'inference':{'unit':'unique patient cluster, not ICU stay','method':'paired percentile bootstrap','replicates':1000,'seed':2026,
        'confidence':'95% two-sided','multiplicity':'one designated primary comparison; secondary comparisons descriptive, not confirmatory'},
      'future_sample_rule':'New genuinely untouched site/time cohort with no role in feature/model/threshold choice; if unavailable, NO independent confirmation claim',
      'future_success_rule':'Primary two-sided 95% cluster-bootstrap CI lower bound >0, with calibration and endpoint quality reported; not a clinical efficacy criterion',
      'future_failure_rule':'Primary CI includes 0 or worse, data quality gate fails, inaccessible labels, or no untouched cohort -> do not claim confirmatory transport',
      'future_quality_gates':['Lock executable/container+dependency versions and SHA256 before seeing new outcomes',
          'Verify cohort inclusion and patient overlap','Verify lab first availability or disclose proxy and unknown/reverse records',
          'Audit endpoint and competing discharge, independently adjudicate when possible','Record n, events, prevalence, missingness, center counts and site eligibility',
          'No retrospective threshold optimization based on future test outcomes'],
      'explicit_prohibitions':['Do not rename older external results confirmatory','Do not assert calibrated individual uncertainty from population bootstrap',
          'Do not claim real-time bidirectional synchronization or therapy simulation','Do not assert external eICU death time is true death timestamp'],
      'locked_input_manifest_sha256':src_hash['stage8e_manifest.json'],
      'analysis_selections_made_at':'Stage8F lock after Stage7E-8E exploratory analyses'
    }
    protocol_bytes=canonical(protocol).encode('utf-8');protocol_sha=hashlib.sha256(protocol_bytes).hexdigest()
    lock={'status':'STAGE8F_FUTURE_PROTOCOL_LOCKED_NOT_INDEPENDENT_VALIDATION',
          'stage8e_status':manifest['status'],'stage8e_file_sha256':src_hash,
          'stage8e_upstream_file_sha256':manifest.get('input_file_sha256',{}),
          'project_code_sha256':project_hash,'db_sha256':db_hash,
          'protocol_sha256':protocol_sha,'stage8e_gate_count':len(gates),'stage8e_evidence_rows':len(ev),
          'privacy':'No patient-level data exported; no model trained; no upstream file changed',
          'known_limitations':['New validation data not supplied','Protocol is post-exploration, not historical preregistration',
                                'Source code/environment fingerprints only as complete as files supplied']}
    if a.mode=='create':
        if out.exists() and any(out.iterdir()):raise SystemExit('FAIL: output directory must be empty. Never overwrite a lock.')
        out.mkdir(parents=True,exist_ok=True)
        (out/'stage8f_protocol.json').write_bytes(protocol_bytes)
        (out/'stage8f_lock_manifest.json').write_text(canonical(lock),encoding='utf8')
        writecsv(out/'stage8f_freeze_inventory.csv',[{'file':k,'sha256':v,'role':'STAGE8E_ARCHIVED_INPUT'} for k,v in src_hash.items()]+[{'file':k,'sha256':v,'role':'PROJECT_SOURCE_OPTIONAL'} for k,v in project_hash.items()])
        (out/'stage8f_protocol_FA.md').write_text(f"""# Stage 8F — قفل پروتکل اعتبارسنجی آینده\n\n**وضعیت:** پروتکل آینده قفل شد؛ هیچ تأیید مستقل جدید انجام نشده است.\n\n- شواهد Stage 7E تا 8E اکتشافی و پسینی هستند؛ eICU در تصمیم حذف شمارش‌ها استفاده شده است.\n- مقایسه اصلی برای مجموعه جدید و دست‌نخورده: ΔAUROC مدل Integrated منهای Lab-only در ساعت ۲۴.\n- معیار موفقیت پژوهشی آینده: کران پایین ۹۵٪ bootstrap زوجی خوشه‌ای بیمار بالاتر از صفر، همراه با گزارش اجباری کالیبراسیون، کیفیت endpoint و محدودیت‌های انتقال.\n- ویژگی‌های مدل: بدون دما و بدون counts، با proxy محافظه‌کارانه دسترسی نتایج آزمایش Stage 8B.\n- مدل فقط در ۶ ساعت MIMIC آموزش می‌بیند و روی ۶/۱۲/۱۸/۲۴ ساعت به‌صورت frozen ارزیابی می‌شود.\n- اگر داده جدیدِ واقعاً دست‌نخورده در دسترس نباشد، ادعای independent confirmation ممنوع است.\n- Digital Twin به‌معنی retrospective predictive surrogate است، نه سیستم همگام و نه شبیه‌ساز مداخله.\n\n**SHA256 پروتکل:** `{protocol_sha}`\n\nاین قفل بر اساس پرونده تاریخی ایجاد شد و معادل preregistration قبل از تمام تحلیل‌های گذشته نیست.\n""",encoding='utf8')
        print('STAGE 8F COMPLETE — FUTURE PROTOCOL LOCKED; NO INDEPENDENT VALIDATION',out)
    else:
        if not out.is_dir():raise SystemExit('FAIL: lock directory absent')
        prior=json.loads((out/'stage8f_lock_manifest.json').read_text(encoding='utf8'))
        stored=(out/'stage8f_protocol.json').read_bytes()
        if digest(out/'stage8f_protocol.json')!=prior['protocol_sha256']:errors.append('PROTOCOL MODIFIED')
        if prior['stage8e_file_sha256']!=src_hash:errors.append('STAGE8E INPUT CHANGED')
        if prior['stage8e_upstream_file_sha256']!=manifest.get('input_file_sha256',{}):errors.append('UPSTREAM DIGEST CHAIN CHANGED')
        if prior['project_code_sha256']!=project_hash:errors.append('PROJECT SOURCE DIGEST CHANGED OR OPTIONAL PROJECT PATH OMITTED')
        if prior['db_sha256']!=db_hash:errors.append('DB DIGEST CHANGED OR DB PATH OMITTED')
        if errors:raise SystemExit('STAGE 8F VERIFY FAIL: '+'; '.join(errors))
        print('STAGE 8F VERIFY PASS — protocol and available source fingerprints unchanged')

if __name__=='__main__':main()
