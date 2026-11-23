#!/usr/bin/env python3
"""Patient-free syntax and synthetic checks; explicitly not a dataset replication."""
import pathlib, subprocess,sys,py_compile
root=pathlib.Path(__file__).resolve().parents[1]
files=sorted((root/'scripts').rglob('*.py'))
for f in files:
    compile(f.read_text(encoding='utf-8-sig'),str(f),'exec')
print('SYNTAX PASS:',len(files),'Python files')
for name in ('test_stage8c_synthetic.py','test_stage8d_synthetic.py'):
    script=root/'scripts'/name
    if not script.is_file():raise FileNotFoundError(script)
    r=subprocess.run([sys.executable,str(script)],cwd=str(root/'scripts'))
    if r.returncode:raise RuntimeError(name+' failed; install requirements and re-run')
print('SYNTHETIC SELFTEST PASS; no clinical data tested')
