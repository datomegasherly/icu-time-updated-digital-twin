"""Static no-DB source validation; does not touch datasets."""
import ast, json
from pathlib import Path
root=Path(__file__).parent
src=(root/'icu_stage3_rebuild.py').read_text(encoding='utf8')
ast.parse(src)
cfg=json.loads((root/'item_mapping_stage3.json').read_text(encoding='utf8'))
assert cfg['landmark_minute']==1440
assert cfg['horizon_minute']==1440
assert {223761,223762,678,676}.issubset(set(cfg['mimic']['vitals']['temp']))
assert "--train-models" in src
assert "Stage3 rebuild finished" in src
assert "p.los_min>{lm} AND p.los_min<={end}" in src
assert "Refusing overwrite" in src
print('STATIC PREFLIGHT PASSED (not an end-to-end data execution).')
