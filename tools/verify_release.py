#!/usr/bin/env python3
"""Standard-library-only code release audit. No input patient data loaded."""
from pathlib import Path
import hashlib,json,re,sys

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'release_manifest.json').read_text(encoding='utf-8'))
expected=manifest['source_hashes_sha256']
allowed_suffix={'.py','.md','.json','.txt','.cff',''}
prohibited_suffix={'.csv','.tsv','.parquet','.arrow','.feather','.duckdb','.db','.sqlite','.gz','.zip','.7z','.rar','.xlsx','.xls','.pkl','.pickle','.joblib','.npy','.npz','.onnx','.pt','.pth','.h5','.bin','.pdf','.docx'}
prohibited_names={'.env','credentials.json','service-account.json','secrets.json'}
errors=[]

actual={f.relative_to(root).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'scripts').rglob('*') if f.is_file()}
for name,sha in expected.items():
    if actual.get(name)!=sha:errors.append('SOURCE_HASH_MISMATCH:'+name)
for name in actual:
    if name not in expected:errors.append('UNEXPECTED_SOURCE_FILE:'+name)

for p in root.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(root).as_posix()
    if p.name.lower() in prohibited_names or p.suffix.lower() in prohibited_suffix:
        errors.append('FORBIDDEN_FILE:'+rel)
    elif p.suffix.lower() not in allowed_suffix:
        errors.append('UNREVIEWED_EXTENSION:'+rel)
    if p.stat().st_size>1_500_000:errors.append('OVERSIZE_FILE:'+rel)
    if p.suffix.lower() in {'.py','.md','.json','.txt'}:
        t=p.read_text(errors='replace')
        # Require expert review; these are high-confidence token-looking strings.
        for pattern in [r'(?i)-----BEGIN (RSA |EC |OPENSSH |)PRIVATE KEY-----',
                        r'(?i)(?:AKIA|ASIA)[A-Z0-9]{16}',
                        r'(?i)gh[pousr]_[A-Za-z0-9_]{30,}',
                        r'(?i)sk-[A-Za-z0-9]{35,}']:
            if re.search(pattern,t):errors.append('SECRET_LIKE_STRING:'+rel)

source_keys=manifest['stage8f_locked_source_files']
for row in source_keys:
    name='scripts/'+row['file']
    if actual.get(name)!=row['sha256']:errors.append('STAGE8F_SOURCE_GATE_FAILED:'+name)

if errors:
    print('RELEASE AUDIT FAILED:')
    for e in errors: print('  '+e)
    sys.exit(1)
print(f'RELEASE AUDIT PASS: {len(actual)} source files; {len(source_keys)} Stage8F hash gates; no forbidden files or known secret patterns')
print('NOTE: static scanning cannot prove patient privacy or complete end-to-end reproducibility; author review remains essential.')
