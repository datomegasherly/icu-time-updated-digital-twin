#!/usr/bin/env python3
"""Stage 8G: read-only prospective validation feasibility & claim audit.
No access to patient rows; no train/test fitting; no independent validation claim.
"""
import argparse, csv, hashlib, json, pathlib, sys
from datetime import datetime, timezone

CLAIMS = [
 ('retrospective_patient_specific_state','SUPPORTED_WITH_QUALIFICATION','6h-trained frozen model receives individual landmark feature summaries','Not continuous or physiologically simulated state'),
 ('time_updated_rolling_24h_risk','SUPPORTED_WITH_QUALIFICATION','At-risk cohorts and rolling 24h endpoint at 6/12/18/24h','Different risk sets and horizons; not direct individual disease progress'),
 ('external_auroc_integrated_advantage','EXPLORATORY_ONLY','Stage 7E/8B/8D external paired metrics','eICU informed post-hoc drop-count choice'),
 ('availability_aware_labs','PROXY_SENSITIVITY_ONLY','MIMIC storetime; eICU labresultrevisedoffset','First clinical availability not proven'),
 ('endpoint_robustness','HYPOTHETICAL_SCENARIOS_ONLY','8C/8D shift proxy time by ±30/60 minutes','True eICU death time unavailable'),
 ('individual_uncertainty_guarantees','NOT_SUPPORTED','Patient-cluster CIs concern population performance','No patient-level risk interval coverage study'),
 ('bidirectional_realtime_twin','NOT_SUPPORTED','Retrospective offline landmarks','No live linkage, synchronization, or prospective deployment'),
 ('causal_treatment_simulation','NOT_SUPPORTED','No intervention simulator','No treatment-effect or counterfactual evaluation'),
 ('independent_external_confirmation','NOT_SUPPORTED_UNTIL_NEW_DATA','Stage 8F locks future validation only','Past eICU already examined'),
 ('superior_calibration_for_integrated','NOT_SUPPORTED','Lab-only lower Brier at some landmarks','AUROC != probability calibration'),
]
REQUIRED_CANDIDATE_FIELDS = ['dataset_id','data_custodian','collection_start','collection_end','sites','newly_unseen_by_team','model_selection_used_dataset','patient_overlap_checked','outcome_mapping_verified','lab_availability_verified','access_approved','sample_size','event_count']

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()

def emit_csv(path,head,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(head);w.writerows(rows)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--s8f',required=True,help='Stage 8F output folder')
 ap.add_argument('--s8e',help='Stage 8E folder for live integrity check')
 ap.add_argument('--project',help='Project root for available code hash comparison')
 ap.add_argument('--candidate',help='Optional non-identifying JSON describing a truly new validation dataset')
 ap.add_argument('--out',required=True)
 a=ap.parse_args()
 source=pathlib.Path(a.s8f);out=pathlib.Path(a.out)
 for n in ['stage8f_protocol.json','stage8f_lock_manifest.json','stage8f_freeze_inventory.csv']:
  if not (source/n).is_file():raise FileNotFoundError(source/n)
 protocol_bytes=(source/'stage8f_protocol.json').read_bytes()
 protocol=json.loads(protocol_bytes);manifest=json.loads((source/'stage8f_lock_manifest.json').read_text(encoding='utf-8-sig'))
 checks=[]
 def gate(name,status,note):checks.append((name,status,note))
 if sha(source/'stage8f_protocol.json')!=manifest['protocol_sha256']:
  raise RuntimeError('FAIL CLOSED: Stage8F protocol SHA256 mismatch')
 gate('protocol_sha256','PASS','Protocol matches Stage8F lock manifest')
 if protocol.get('status')!='PROTOCOL_LOCKED_FOR_FUTURE_EVALUATION_ONLY':raise RuntimeError('Protocol status not frozen')
 gate('protocol_scope','PASS','Forward-looking protocol; earlier data exploratory')
 if manifest.get('stage8e_gate_count')!=16 or manifest.get('stage8e_status')!='PASS_EVIDENCE_INTEGRITY_EXPLORATORY_NOT_SUBMISSION_READY':
  raise RuntimeError('Stage8E evidence status unexpected')
 gate('stage8e_recorded_status','PASS','16 archived integrity gates, not submission readiness')
 if a.s8e:
  e=pathlib.Path(a.s8e)
  for name,expected in manifest.get('stage8e_file_sha256',{}).items():
   p=e/name
   if not p.is_file() or sha(p)!=expected:raise RuntimeError('FAIL CLOSED: Stage8E mismatch '+str(p))
  gate('stage8e_live_files','PASS','Stage8E files match Stage8F frozen SHA256')
 else:gate('stage8e_live_files','NOT_CHECKED','Optional --s8e not provided')
 if a.project:
  p=pathlib.Path(a.project);missing=[];changed=[];matched=0
  for name,expected in manifest.get('project_code_sha256',{}).items():
   f=p/name
   if not f.is_file():missing.append(name)
   elif sha(f)!=expected:changed.append(name)
   else:matched+=1
  gate('project_source_fingerprints','PASS' if not (missing or changed) else 'REVIEW_REQUIRED',f'matched={matched},missing={missing},changed={changed}')
 else:gate('project_source_fingerprints','NOT_CHECKED','Optional --project not provided')
 candidate=None;missing_fields=[];reasons=[];ready=False
 if a.candidate:
  candidate=json.loads(pathlib.Path(a.candidate).read_text(encoding='utf-8-sig'))
  missing_fields=[k for k in REQUIRED_CANDIDATE_FIELDS if k not in candidate]
  if missing_fields:reasons.append('Missing evidence fields: '+', '.join(missing_fields))
  for key in ('newly_unseen_by_team','patient_overlap_checked','outcome_mapping_verified','lab_availability_verified','access_approved'):
   if candidate.get(key) is not True:reasons.append(key+' must be documented true')
  if candidate.get('model_selection_used_dataset') is not False:reasons.append('candidate already used for model selection or use unknown')
  if not isinstance(candidate.get('sites'),list) or not candidate.get('sites'):reasons.append('At least one site identifier required (non-personal)')
  if not isinstance(candidate.get('event_count'),int) or candidate['event_count']<=0:reasons.append('Observed event count required for feasibility; no minimum power proven')
  if not isinstance(candidate.get('sample_size'),int) or candidate['sample_size']<=0:reasons.append('Observed sample size required')
  if not reasons:ready=True
  gate('candidate_documentation','DOCUMENTARY_ELIGIBLE_PENDING_AUDIT' if ready else 'BLOCKED', '; '.join(reasons) if reasons else 'Declared fields meet documentary checks; not independently verified')
 else:
  reasons=['No new validation dataset supplied']
  gate('candidate_documentation','NOT_AVAILABLE',reasons[0])
 # Fail closed if pre-existing archival record was changed.
 if a.out and out.resolve()==source.resolve():raise RuntimeError('Output must not overwrite locked Stage8F folder')
 out.mkdir(parents=True,exist_ok=True)
 emit_csv(out/'stage8g_integrity_gates.csv',['check','status','explanation'],checks)
 emit_csv(out/'stage8g_claim_audit.csv',['claim','permitted_status','support','limitation'],CLAIMS)
 emit_csv(out/'stage8g_candidate_readiness.csv',['field','value_or_status'],[(k,('UNSUPPLIED' if candidate is None else json.dumps(candidate.get(k,'MISSING'),ensure_ascii=False))) for k in REQUIRED_CANDIDATE_FIELDS])
 status='DOCUMENTARY_CANDIDATE_ELIGIBLE_NOT_VALIDATED' if ready else 'INDEPENDENT_VALIDATION_NOT_AVAILABLE'
 report=f'''# Stage 8G — Independent validation feasibility and Digital Twin claim audit\n\n**Status:** {status}\n\n**Protocol SHA256:** `{manifest['protocol_sha256']}`\n\n**Primary future estimand:** {protocol['primary_estimand']}\n\nStage 8F was frozen after analyses 7E–8E. Neither a historical preregistration nor a new independent validation has occurred.\n\n## Data readiness\n\n{chr(10).join('- '+r for r in reasons) if reasons else '- Candidate information is documentary-eligible only; external provenance, overlap, access, and endpoint quality require independent verification.'}\n\n## Claim boundaries\n\nRetrospective patient-specific state updates and rolling horizon prediction are supported with qualification. Patient-specific uncertainty guarantees, real-time/bidirectional synchronization, causal therapy simulation and independent confirmatory transport are not supported. The eICU outcome timing is a proxy and lab availability uses source-specific timestamp proxies.\n\n## Next steps\n\nDo not run a new confirmatory test unless a genuinely untouched dataset is documented; independently verify patient overlap and endpoint mapping; freeze executable model artifacts and environment fingerprints before viewing candidate outcomes; report calibration and competing discharge. If unavailable, write the paper as exploratory and disclose the selection history.\n'''
 (out/'stage8g_report_EN.md').write_text(report,encoding='utf-8')
 (out/'stage8g_candidate_template.json').write_text(json.dumps({k:([] if k=='sites' else None) for k in REQUIRED_CANDIDATE_FIELDS},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 result={'status':status,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'stage8f_protocol_sha256':manifest['protocol_sha256'],'stage8f_evidence_is_exploratory':True,'candidate_documentary_eligible':ready,'candidate_evaluated':candidate is not None,'independent_validation_performed':False,'stage8g_checks':dict((row[0],row[1]) for row in checks),'limitations':reasons,'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
 (out/'stage8g_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print('STAGE 8G COMPLETE —',status)
 for x in checks:print(*x[:2])

if __name__=='__main__':main()
