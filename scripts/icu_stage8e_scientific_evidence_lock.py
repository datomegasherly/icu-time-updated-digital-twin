#!/usr/bin/env python3
"""Stage 8E: aggregate-only, read-only scientific evidence audit (no model fitting)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

LANDMARKS = (6, 12, 18, 24)
REQUIRED = {
 's7e': ['stage7e_metrics.csv', 'stage7e_calibration.csv', 'stage7e_external_paired_ci.csv', 'stage7e_manifest.json'],
 's8a': ['stage8a_aggregate_overview.csv', 'stage8a_manifest.json'],
 's8b': ['stage8b_metrics.csv', 'stage8b_calibration.csv', 'stage8b_reference_drift.csv', 'stage8b_cohort_reconciliation.csv', 'stage8b_paired_patient_cluster_ci.csv', 'stage8b_manifest.json'],
 's8c': ['stage8c_primary_counts.csv', 'stage8c_endpoint_sensitivity.csv', 'stage8c_stage7e_reference_gate.csv', 'stage8c_manifest.json'],
 's8d': ['stage8d_metrics.csv', 'stage8d_paired_patient_cluster_ci.csv', 'stage8d_stage8b_reference_gate.csv', 'stage8d_stage8c_reference_gate.csv', 'stage8d_common_cohort_endpoint_effect.csv', 'stage8d_manifest.json'],
}

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for piece in iter(lambda:f.read(1024*1024),b''): h.update(piece)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in REQUIRED: p.add_argument('--'+k,required=True, type=Path, help='Existing stage output directory')
 p.add_argument('--out',required=True,type=Path)
 p.add_argument('--tolerance',type=float,default=1e-8)
 a=p.parse_args(); roots={k:getattr(a,k) for k in REQUIRED}
 if a.tolerance<=0: p.error('tolerance must be positive')
 output=a.out.resolve()
 if output.exists() and any(output.iterdir()): p.error('Output must be a NEW or EMPTY directory; existing evidence cannot be overwritten')
 for key,root in roots.items():
  if not root.is_dir(): p.error('Missing directory '+str(root))
  if output==root.resolve() or root.resolve() in output.parents: p.error('Output cannot be located inside an input evidence directory')
  for fn in REQUIRED[key]:
   if not (root/fn).is_file(): p.error('Missing '+str(root/fn))
 raw={(k,f):(root/f) for k,root in roots.items() for f in REQUIRED[k]}
 hashes={f'{k}/{name}':sha256(path) for (k,name),path in raw.items()}
 def csv(k,f): return pd.read_csv(raw[(k,f)])
 def manifest(k): return json.loads(raw[(k,REQUIRED[k][-1])].read_text(encoding='utf-8'))
 e=csv('s7e','stage7e_metrics.csv'); ec=csv('s7e','stage7e_calibration.csv')
 a8=csv('s8a','stage8a_aggregate_overview.csv'); b=csv('s8b','stage8b_metrics.csv')
 bc=csv('s8b','stage8b_calibration.csv'); bd=csv('s8b','stage8b_reference_drift.csv')
 br=csv('s8b','stage8b_cohort_reconciliation.csv'); c=csv('s8c','stage8c_primary_counts.csv')
 cs=csv('s8c','stage8c_endpoint_sensitivity.csv'); cg=csv('s8c','stage8c_stage7e_reference_gate.csv')
 d=csv('s8d','stage8d_metrics.csv'); dg=csv('s8d','stage8d_stage8b_reference_gate.csv')
 dc=csv('s8d','stage8d_stage8c_reference_gate.csv'); dci=csv('s8d','stage8d_paired_patient_cluster_ci.csv')
 de=csv('s8d','stage8d_common_cohort_endpoint_effect.csv')
 checks=[]
 def check(code, condition, details):
  result=bool(condition);checks.append(dict(check=code,status='PASS' if result else 'FAIL',detail=str(details)))
  print(('PASS' if result else 'FAIL'),code,details,flush=True)
 def keys_ok(df,cols): return all(z in df for z in cols) and not df.duplicated(cols).any()
 def allclose(x,y):
  u=pd.to_numeric(x,errors='coerce');v=pd.to_numeric(y,errors='coerce')
  return bool(np.isfinite(u).all() and np.isfinite(v).all() and np.allclose(u,v,rtol=0,atol=a.tolerance))
 def compare(left,right,keys,cols):
  if not keys_ok(left,keys) or not keys_ok(right,keys):return False,'duplicate or missing keys'
  left=left[keys+cols];right=right[keys+cols]
  if set(map(tuple,left[keys].itertuples(index=False,name=None)))!=set(map(tuple,right[keys].itertuples(index=False,name=None))): return False,'key coverage mismatch'
  m=left.merge(right,on=keys,validate='one_to_one',suffixes=('_l','_r'))
  bad=[k for k in cols if not allclose(m[k+'_l'],m[k+'_r'])]
  return not bad, ('numerically matched' if not bad else 'mismatch '+','.join(bad))
 check('stage7e_shape',len(e)==24 and keys_ok(e,['cohort','model','landmark_h']) and set(e.landmark_h)==set(LANDMARKS),'24 unique model×cohort×landmark records')
 check('stage8a_shape',len(a8)==8 and keys_ok(a8,['source','landmark_h']) and (a8.total_measured_rows>=a8.proxy_available_rows).all(),'8 source×landmark aggregate records')
 check('stage8b_drift',len(bd)>0 and np.isfinite(bd.absolute_difference).all() and float(bd.absolute_difference.max())<=a.tolerance,'max drift '+str(bd.absolute_difference.max()))
 check('stage8b_reconciliation',len(br)==2 and (br.shared_label_mismatch==0).all() and (br.new_only==0).all(),'Stage 5B vs strict at-risk counts have no shared-label mismatch')
 eb=b[b.scenario=='measured']; ok,msg=compare(e,eb,['cohort','model','landmark_h'],['n','events','auroc','auprc','brier','mean_predicted_risk'])
 check('stage7e_to_stage8b_measured',ok,msg)
 check('stage8c_reference_gate',len(cg)==4 and cg.n_match.fillna(False).astype(str).str.lower().eq('true').all() and cg.events_match.fillna(False).astype(str).str.lower().eq('true').all(),'4 external landmarks')
 exc=b[(b.cohort=='external') & (b.scenario=='exclude') & (b.model.isin(['lab','integrated']))]
 dref=d[(d.cohort=='external') & (d.scenario=='stage7e_primary') & (d.feature_timing=='exclude')]
 ok,msg=compare(exc,dref,['model','landmark_h'],['n','events','auroc','auprc','brier','mean_predicted_risk'])
 check('stage8b_to_stage8d_primary',ok,msg)
 check('stage8d_stage8b_gate',len(dg)==32 and (dg['_merge']=='both').all() and all((pd.to_numeric(dg[col],errors='coerce')<=a.tolerance).all() for col in dg if col.endswith('_abs_drift')),'32 rows; exact recorded numeric drift')
 check('stage8d_stage8c_gate',len(dc)==40 and (dc.n_new==dc.n_archive).all() and (dc.events_new==dc.events_archive).all(),'40 rows matched')
 check('stage8d_ci',len(dci)>0 and (dci.valid_replicates==300).all() and (dci.ci_low<=dci.ci_high).all(),'300 valid patient-cluster replicates in every reported comparison')
 check('stage8d_common_cohort',len(de)>0 and (de.common_n>0).all() and (de.common_changed_labels>=0).all(),'common cohort label effect table present')
 check('stage8d_scope',set(d.feature_timing)=={'measured','exclude'} and set(d.model)=={'lab','integrated'} and set(d.landmark_h)==set(LANDMARKS),'2 timing modes × 2 model arms × 4 landmarks')
 mans={k:manifest(k) for k in REQUIRED}
 check('status_disclosures',all(('EXPLORATORY' in str(mans[k].get('status',''))) for k in ['s8b','s8c','s8d']),'Stage 8B–8D retain exploratory designation')
 # Descriptive evidence: all numerical entries are taken solely from archived aggregates.
 evid=[]
 for h in LANDMARKS:
  for timing in ['measured','exclude']:
   bas=d[(d.cohort=='external')&(d.landmark_h==h)&(d.scenario=='stage7e_primary')&(d.feature_timing==timing)]
   if len(bas)!=2: continue
   lab=bas[bas.model=='lab'].iloc[0];integ=bas[bas.model=='integrated'].iloc[0]
   cis=dci[(dci.landmark_h==h)&(dci.scenario=='stage7e_primary')&(dci.feature_timing==timing)&(dci.metric=='auroc')]
   ci=cis.iloc[0] if len(cis)==1 else None
   shifts=d[(d.cohort=='external')&(d.landmark_h==h)&(d.feature_timing==timing)&(d.model=='integrated')&(d.scenario.str.contains('expired_proxy_shift_',regex=False))]
   evid.append(dict(landmark_h=h,feature_timing=timing,n=int(integ.n),events=int(integ.events),integrated_auroc=float(integ.auroc),lab_auroc=float(lab.auroc),delta_auroc=float(integ.auroc-lab.auroc),delta_auroc_ci_low=float(ci.ci_low) if ci is not None else np.nan,delta_auroc_ci_high=float(ci.ci_high) if ci is not None else np.nan,integrated_auprc=float(integ.auprc),lab_auprc=float(lab.auprc),integrated_brier=float(integ.brier),lab_brier=float(lab.brier),integrated_shifted_auroc_min=float(shifts.auroc.min()),integrated_shifted_auroc_max=float(shifts.auroc.max()),n_shift_scenarios=int(len(shifts))))
 evidence=pd.DataFrame(evid)
 check('evidence_rows',len(evidence)==8 and (evidence.n_shift_scenarios==4).all(),'4 landmarks × 2 timing modes with 4 shifted endpoint scenarios')
 check('integrated_auroc_sensitivity',bool((evidence.delta_auroc>0).all()) and bool((evidence.delta_auroc_ci_low>0).all()),'positive paired AUROC differences in observed 8 baseline comparisons')
 check('brier_not_universally_superior',bool((evidence.integrated_brier>evidence.lab_brier).any()),'discrimination != calibration / probabilistic error')
 # Lock is an evidence-integrity lock, NOT a preregistered or independent-validation claim.
 passed=all(x['status']=='PASS' for x in checks)
 output.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(checks).to_csv(output/'stage8e_integrity_gates.csv',index=False)
 evidence.to_csv(output/'stage8e_evidence_summary.csv',index=False,float_format='%.12g')
 claim_rows=[
 ('Time-updated retrospective patient-specific risk state','SUPPORTED_WITH_QUALIFICATION','Frozen MIMIC 6h model receives updated landmark features; rolling 24h endpoint','Not bidirectional or live; future landmark risk sets differ'),
 ('External discrimination of integrated vs lab-only','EXPLORATORY_SUPPORTED','Stage 7E/8B/8D external AUROC contrasts and patient-cluster CIs','External eICU examined during post-hoc count ablation/model choice'),
 ('Lab result availability accounted for','PROXY_BASED_SENSITIVITY','MIMIC storetime; eICU labresultrevisedoffset in Stage 8B','Neither is verified first clinician visibility; unknown/reverse timestamps'),
 ('Endpoint robustness','SCENARIO_CONDITIONAL','Stage 8C/8D label and at-risk set perturbations ±30/60 min','eICU exact death time unconfirmed; shifts hypothetical'),
 ('Individual predictive uncertainty intervals','NOT_SUPPORTED','Bootstrap CIs quantify aggregate model-performance sampling uncertainty','No individual interval coverage/calibration validation'),
 ('Clinical operational or causal treatment digital twin','NOT_SUPPORTED','No prospective, live synchronization or therapy/counterfactual experiment','Do not claim clinical efficacy or treatment guidance'),
 ('Independent untouched external confirmation','NOT_SUPPORTED','Post-hoc count choices inspected eICU','Requires newly held-out cohort or genuinely locked validation'),
 ('Universal probability calibration superiority','NOT_SUPPORTED','Integrated may outperform AUROC yet have worse Brier/calibration','Report each metric without conflating'),
 ('Submission-ready evidence lock','NOT_YET','Stage 8E audits integrity and claims only','Protocol, independent validation, ethics/code/data and latest manuscript remain pending'),
 ]
 claims=pd.DataFrame(claim_rows,columns=['claim','status','evidence','critical_limitation']);claims.to_csv(output/'stage8e_claim_matrix.csv',index=False)
 final_status='PASS_EVIDENCE_INTEGRITY_EXPLORATORY_NOT_SUBMISSION_READY' if passed else 'FAIL_EVIDENCE_INTEGRITY_DO_NOT_LOCK'
 meta={'status':final_status,'input_file_sha256':hashes,'tolerance':a.tolerance,'input_manifests':mans,'checks':{x['check']:x['status'] for x in checks},'no_raw_patient_data':True,'no_refit':True,'no_external_recalibration':True,'no_input_modifications':True,'not_independent_confirmation':True}
 (output/'stage8e_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 def fmt(row):return f"| {int(row.landmark_h)} | {row.feature_timing} | {row.integrated_auroc:.4f} | {row.lab_auroc:.4f} | {row.delta_auroc:+.4f} [{row.delta_auroc_ci_low:.4f}, {row.delta_auroc_ci_high:.4f}] | {row.integrated_brier:.5f} | {row.lab_brier:.5f} |"
 table='\n'.join(fmt(row) for row in evidence.itertuples())
 report=f'''# Stage 8E — Scientific Evidence Integrity Audit\n\n**Status:** `{final_status}`\n\nThis is a retrospective exploratory, aggregate-only evidence integrity audit, **not** a prospective or independent validation and **not** a clinical digital twin certification. No original metrics or model weights were changed.\n\n## Numerical checks\n\n'''+ '\n'.join(f"- {x['status']}: `{x['check']}` — {x['detail']}" for x in checks)+f'''\n\n## External discrimination and Brier (original archived estimates)\n\n| Landmark (h) | Feature time | Integrated AUROC | Lab AUROC | Paired ΔAUROC [95% cluster CI] | Integrated Brier | Lab Brier |\n|---|---|---:|---:|---|---:|---:|\n{table}\n\n## Claim restrictions\n\n- The model produces updated retrospective landmark risks, not a live bidirectionally connected digital twin.\n- The external dataset was inspected during post-hoc feature selection: bootstrap intervals do **not** restore independence.\n- Availability proxies are not verified first clinician access.\n- eICU ICU discharge time is a proxy for death timing, not adjudicated patient death time.\n- Patient-cluster bootstrap is uncertainty about *aggregate performance*, not patient-specific uncertainty intervals.\n- Integrated discrimination may exceed Lab-only while Brier/calibration remain worse.\n- A successful audit permits descriptive evidence freezing but **not** declaring the manuscript submission-ready.\n\n## Next publication safeguards\n\n1. Freeze code/config hashes and a non-post-hoc protocol for any subsequent validation.\n2. Secure genuinely independent external or temporal/site holdout validation.\n3. Audit study ethics, license, data/software statements and manuscript version against reported populations.\n4. Match each claim to its table, limitation and a verified bibliographic reference.\n'''
 (output/'stage8e_scientific_report_EN.md').write_text(report,encoding='utf-8')
 fa=f'''# مرحله ۸E — ممیزی و قفل شواهد علمی\n\n**وضعیت:** `{final_status}`\n\nاین قفل صرفاً سلامت تطبیق شواهد موجود را می‌سنجد؛ نه تأیید مستقل، نه اثبات کاربرد بالینی و نه قفل پروتکل پیشینی. هیچ خروجی گذشته بازنویسی نشده است.\n\n## کنترل‌ها\n\n'''+ '\n'.join(f"- {x['status']} — {x['check']}: {x['detail']}" for x in checks)+'''\n\n## تفسیر\n\n- مدل در ساعت ۶ آموزش می‌بیند و ریسک گذشته‌نگر بیمار را با داده‌های جدید در ساعت‌های ۱۲، ۱۸ و ۲۴ به‌روزرسانی می‌کند. مدل عملیاتی متصل و شبیه‌ساز مداخله نیست.\n- داده eICU قبلاً برای انتخاب پسینی حذف شمارش‌های اندازه‌گیری استفاده شده است؛ CIهای Bootstrap این مسئله را حل نمی‌کنند.\n- زمان نتیجه آزمایش با proxy سنجیده شده و زمان مرگ eICU مستقل تأیید نشده است.\n- عدم‌قطعیت عملکرد جمعیتی معادل بازه عدم‌قطعیت ریسک فردی نیست.\n- Integrated از نظر AUROC برتر است، نه لزوماً Brier یا کالیبراسیون.\n\n## خروجی‌ها\n\n`stage8e_integrity_gates.csv`، `stage8e_evidence_summary.csv`، `stage8e_claim_matrix.csv`، دو گزارش فارسی و انگلیسی و `stage8e_manifest.json` شامل SHA256 تمام ورودی‌ها.\n\n## ادامه ضروری\n\nاعتبارسنجی مستقل با پروتکل قفل‌شده، کنترل الزامات ژورنال، منابع واقعی و آخرین نسخه مقاله قبل از submit لازم است.\n'''
 (output/'stage8e_scientific_report_FA.md').write_text(fa,encoding='utf-8')
 print('STAGE 8E COMPLETE —',final_status,'out=',output)
 return 0 if passed else 2
if __name__=='__main__':sys.exit(main())
