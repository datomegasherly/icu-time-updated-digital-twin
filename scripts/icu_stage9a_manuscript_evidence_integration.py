#!/usr/bin/env python3
"""Stage 9A — non-destructive archived evidence/manuscript gap audit.
Only reads Stage 8E/8F/8G and an existing Word manuscript; writes aggregate reports.
"""
import argparse,csv,hashlib,json,re,sys
from datetime import datetime,timezone
from pathlib import Path
from docx import Document

EXPECTED_E='PASS_EVIDENCE_INTEGRITY_EXPLORATORY_NOT_SUBMISSION_READY'
EXPECTED_G='INDEPENDENT_VALIDATION_NOT_AVAILABLE'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,data,cols):
 with open(p,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(data)
def gate(all_g,name,ok,detail):
 all_g.append(dict(check=name,status='PASS' if ok else 'FAIL',detail=str(detail)))
def fmt(v):return f'{float(v):.4f}'

def main():
 ap=argparse.ArgumentParser()
 for flag in ['s8e','s8f','s8g','manuscript','out']:ap.add_argument('--'+flag,required=True)
 a=ap.parse_args(); E,F,G,M,O=map(Path,[a.s8e,a.s8f,a.s8g,a.manuscript,a.out]);O.mkdir(parents=True,exist_ok=True)
 required={E:['stage8e_manifest.json','stage8e_integrity_gates.csv','stage8e_evidence_summary.csv','stage8e_claim_matrix.csv'],F:['stage8f_lock_manifest.json','stage8f_protocol.json'],G:['stage8g_manifest.json','stage8g_claim_audit.csv','stage8g_integrity_gates.csv']}
 for folder,names in required.items():
  for name in names:
   if not (folder/name).is_file():raise FileNotFoundError(folder/name)
 if not M.is_file():raise FileNotFoundError(M)
 em=load(E/'stage8e_manifest.json');fm=load(F/'stage8f_lock_manifest.json');pr=load(F/'stage8f_protocol.json');gm=load(G/'stage8g_manifest.json')
 e_g=rows(E/'stage8e_integrity_gates.csv'); evidence=rows(E/'stage8e_evidence_summary.csv');claims=rows(E/'stage8e_claim_matrix.csv');g_g=rows(G/'stage8g_integrity_gates.csv')
 checks=[]
 gate(checks,'stage8e_status',em.get('status')==EXPECTED_E,em.get('status'))
 gate(checks,'stage8e_16_checks',len(e_g)==16 and all(x['status']=='PASS' for x in e_g),f'{len(e_g)} checks')
 gate(checks,'stage8f_protocol_hash',sha(F/'stage8f_protocol.json')==fm.get('protocol_sha256'), 'protocol SHA256')
 gate(checks,'stage8g_protocol_link',gm.get('stage8f_protocol_sha256')==fm.get('protocol_sha256'),'8G points to locked 8F')
 gate(checks,'stage8g_status',gm.get('status')==EXPECTED_G and gm.get('independent_validation_performed') is False,gm.get('status'))
 gate(checks,'stage8g_integrity',all(x['status']=='PASS' for x in g_g if x['check']!='candidate_documentation'),'existing source checks')
 # Check 8E live file bytes against frozen 8F inventory fingerprints.
 e_hashes=fm.get('stage8e_file_sha256',{})
 gate(checks,'stage8e_frozen_file_hashes',bool(e_hashes) and all((E/k).exists() and sha(E/k)==v for k,v in e_hashes.items()),f'{len(e_hashes)} files')
 combos={(int(r['landmark_h']),r['feature_timing']) for r in evidence}
 gate(checks,'evidence_grid',len(evidence)==8 and combos=={(h,t) for h in (6,12,18,24) for t in ('measured','exclude')},'4 landmarks × 2 timing modes')
 gate(checks,'external_auroc_contrasts',all(float(r['delta_auroc'])>0 and float(r['delta_auroc_ci_low'])>0 for r in evidence),'all 8 exploratory contrasts positive')
 gate(checks,'calibration_tradeoff_present',any(float(r['integrated_brier'])>float(r['lab_brier']) for r in evidence),'Integrated Brier not universally best')
 doc=Document(M)
 paragraphs=[p.text.strip() for p in doc.paragraphs if p.text.strip()]
 full='\n'.join(paragraphs)+'\n'+'\n'.join(c.text for t in doc.tables for row in t.rows for c in row.cells)
 headings=[{'heading':p.text.strip(),'style':p.style.name} for p in doc.paragraphs if p.text.strip() and p.style.name.startswith('Heading')]
 gate(checks,'manuscript_readable',len(paragraphs)>100 and len(doc.tables)>0,f'{len(paragraphs)} paragraphs; {len(doc.tables)} tables')
 old_markers=[('obsolete MIMIC cohort','17,260'),('obsolete eICU cohort','69,005'),('obsolete external performance','0.631'),('obsolete fixed landmark','hours 24–48'),('obsolete original DNR eligibility','Early DNR')]
 gaps=[]
 for title,needle in old_markers:
  hits=full.count(needle)
  gaps.append({'finding':title,'evidence_in_manuscript':needle,'occurrences':hits,'severity':'CRITICAL' if hits else 'REVIEW','required_action':'Reconcile against Stage 7E/8B baseline; do not blindly replace numbers across different cohorts'})
 for title,needle,reason in [('frozen 6h updates','frozen model','Clarify one model trained at 6h and reused at later landmarks'),('availability proxy','labresultrevisedoffset','Explain eICU lab timing proxy and uncertainty'),('endpoint proxy','unitdischargestatus','Report ICU discharge proxy rather than hospital discharge'),('post-hoc external selection','post-hoc','Disclose feature selection after viewing eICU'),('strict rolling risk set','at-risk','Clarify landmark-specific eligibility, moving horizon'),('no true individual intervals','patient-specific intervals','Avoid calling bootstrap CI an individual uncertainty interval')]:
  gaps.append({'finding':title,'evidence_in_manuscript':needle,'occurrences':len(re.findall(re.escape(needle),full,re.I)),'severity':'REVIEW','required_action':reason})
 # Only write derived evidence, never alter Word or any input.
 summary=[]
 for r in sorted(evidence,key=lambda x:(int(x['landmark_h']),0 if x['feature_timing']=='exclude' else 1)):
  summary.append({'landmark_h':r['landmark_h'],'timing':r['feature_timing'],'eicu_stays':r['n'],'events':r['events'],'integrated_auroc':r['integrated_auroc'],'lab_auroc':r['lab_auroc'],'delta_auroc':r['delta_auroc'],'ci_low':r['delta_auroc_ci_low'],'ci_high':r['delta_auroc_ci_high'],'integrated_auprc':r['integrated_auprc'],'lab_auprc':r['lab_auprc'],'integrated_brier':r['integrated_brier'],'lab_brier':r['lab_brier'],'integrated_shifted_auroc_min':r['integrated_shifted_auroc_min'],'integrated_shifted_auroc_max':r['integrated_shifted_auroc_max']})
 write_csv(O/'stage9a_integrity_gates.csv',checks,['check','status','detail'])
 write_csv(O/'stage9a_evidence_table.csv',summary,list(summary[0]))
 write_csv(O/'stage9a_manuscript_gap_audit.csv',gaps,list(gaps[0]))
 write_csv(O/'stage9a_claim_matrix.csv',claims,list(claims[0]))
 (O/'stage9a_existing_headings.json').write_text(json.dumps(headings,ensure_ascii=False,indent=2),encoding='utf-8')
 conservative=[r for r in summary if r['timing']=='exclude']
 lines=['# Stage 9A — manuscript evidence integration (DRAFT, not submission ready)','',
  '## Evidence scope','This document is an evidence-grounded *draft*, not a new experimental result. The model is a retrospective, patient-specific, landmark-updated predictive surrogate, trained at 6h in MIMIC-IV, evaluated at 6/12/18/24h with rolling 24h ICU mortality and dynamic at-risk cohorts. No prospective/live/causal/clinical operational twin was evaluated.','',
  '## Main candidate table — external eICU, availability-aware conservative proxy','',
  '| Landmark (h) | Stays | Events | Integrated AUROC | Lab AUROC | ΔAUROC [95% patient-cluster CI] | Integrated AUPRC | Lab AUPRC | Integrated Brier | Lab Brier |','|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|']
 for r in conservative:
  lines.append(f"| {r['landmark_h']} | {int(r['eicu_stays']):,} | {int(r['events']):,} | {fmt(r['integrated_auroc'])} | {fmt(r['lab_auroc'])} | {fmt(r['delta_auroc'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {fmt(r['integrated_auprc'])} | {fmt(r['lab_auprc'])} | {fmt(r['integrated_brier'])} | {fmt(r['lab_brier'])} |")
 lines+=['','The external comparison is **exploratory**: eICU was inspected during post-hoc feature selection (not untouched independent confirmation). The conservative proxies (`storetime`, `labresultrevisedoffset`) are **not verified first clinician visibility**. Endpoint timing in eICU uses ICU-discharge information, not verified death timestamps. The frozen model is re-fitted on MIMIC per feature-timing scenario; do not attribute differences solely to a timestamp filter.','',
  '## Suggested manuscript structure','1. Introduction: predictive digital twin surrogate, question, cross-database measurement-process shift.','2. Related work: ICU prognosis, data/measurement shift, patient digital twins (no claim of intervention simulation).','3. Methods: sources/cohort index; dynamic at-risk landmark sets and rolling endpoint; strict interval boundaries; MIMIC/eICU endpoint differences; lab availability proxies; predictor selection without temperature/counts; 6h-frozen train; preprocessing; bootstrap and calibration; post-hoc disclosures.','4. Results: cohort/endpoint checks; 7E main; 8A descriptive availability; 8B conservative primary sensitivity; 8C endpoint audit; 8D perturbation; calibration tradeoffs; hospital heterogeneity as exploratory.','5. Discussion: what is a predictive twin surrogate vs operational twin; sensitivity interpretation; non-independent external evaluation; next independent test.','6. Conclusion: qualified, retrospective exploratory findings only.','',
  '## Claims to avoid','- Independent external validation confirmed after post-hoc eICU-guided feature selection.','- Laboratory inputs were guaranteed available at decision time.','- eICU recorded true time of death.','- Individual uncertainty coverage or calibrated patient risk intervals.','- Continuous bidirectional, counterfactual or treatment-optimizing ICU digital twin.','- Integrated model uniformly outperforms Lab-only in calibration or Brier.','',
  '## Old manuscript audit','Existing Word file is read-only input. The old single-24h fixed-horizon results and cohort figures must be explicitly reconciled, *not* changed by global find/replace.','']
 for g in gaps:lines.append(f"- {g['finding']}: marker `{g['evidence_in_manuscript']}` occurs {g['occurrences']} time(s). {g['required_action']}.")
 lines+=['','## Outstanding submission blockers','- True untouched new external validation dataset not available (Stage 8G).','- Journal requirements, article type, ethics/data/code availability, and bibliographic DOI verification still require separate current checks.','- Final authorized Word manuscript version must be confirmed before editing.','- Formal design of patient-level uncertainty coverage would require additional validation.']
 (O/'stage9a_manuscript_integration_EN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 fa=['# Stage 9A — گزارش تطبیق شواهد و مقاله','', 'این گزارش و جدول‌ها **اکتشافی** هستند و مقاله را آماده ارسال اعلام نمی‌کنند. فایل Word اصلی تغییر نکرده است.','', '## نتیجه کنترل‌ها',f"{sum(x['status']=='PASS' for x in checks)} از {len(checks)} کنترل موفق.",'', '## یافته محوری','مدل Integrated در هر چهار Landmark از نظر AUROC خارجی نسبت به Lab-only برتر است؛ اما برتری یکسانی در Brier و کالیبراسیون ندارد. داده‌های eICU در انتخاب پسینی ویژگی‌ها استفاده شده‌اند، بنابراین نتیجه تأیید کاملاً مستقل نیست.','', '## محدودیت‌های غیرقابل حذف','- برچسب eICU بر اساس زمان ترخیص ICU به‌عنوان جانشین رخداد است.','- زمان ثبت/بازبینی نتیجه، معادل اثبات زمان اولین دسترسی پزشک نیست.','- فاصله اطمینان Bootstrap مربوط به عملکرد در سطح جمعیت است، نه عدم‌قطعیت پیش‌بینی یک بیمار.','- Digital Twin در این مطالعه، جانشین پیش‌بینی گذشته‌نگر است، نه سامانه زنده یا شبیه‌ساز درمان.','- نسخه Word حاضر شامل جدول‌ها و جمعیت‌های قدیمی‌تر است و نباید با جایگزینی سراسری اعداد اصلاح شود.','', '## اقدامات پیش از ارسال','تطبیق Methods/Results با Stage 7E تا 8G؛ بررسی مستقل مراجع و راهنمای مجله؛ تأیید آخرین نسخه Word؛ انتخاب عنوان متناسب با ادعای محدود.']
 (O/'stage9a_scientific_report_FA.md').write_text('\n'.join(fa)+'\n',encoding='utf-8')
 status='PASS_ARCHIVED_EVIDENCE_MANUSCRIPT_GAPS_OPEN' if all(x['status']=='PASS' for x in checks) else 'FAIL_CLOSED_SOURCE_INTEGRITY'
 man={'status':status,'generated_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':{str(p):sha(p) for p in [M,E/'stage8e_manifest.json',E/'stage8e_evidence_summary.csv',E/'stage8e_claim_matrix.csv',F/'stage8f_protocol.json',F/'stage8f_lock_manifest.json',G/'stage8g_manifest.json']},'checks':checks,'source_exploratory':True,'independent_validation_performed':False,'manuscript_modified':False}
 man['output_sha256']={p.name:sha(p) for p in sorted(O.iterdir()) if p.is_file()}
 (O/'stage9a_manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False),encoding='utf-8')
 for c in checks:print(c['status'],c['check'],c['detail'])
 print('STAGE 9A COMPLETE —',status,'out=',O)
 if not all(x['status']=='PASS' for x in checks):sys.exit(2)
if __name__=='__main__':main()
