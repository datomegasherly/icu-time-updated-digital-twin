#!/usr/bin/env python3
"""No patient data or DuckDB needed. Checks schema gates and hypothetical labels."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from icu_stage8d_endpoint_robust_validation import (endpoint_alternatives,
                                                      validate_stage8b, validate_stage8c)


def main():
    frame=pd.DataFrame({'discharge_min':[1000.,1800.,2000.,1200.,2200.,3000.,999.],
                        'discharge_status':['expired','alive','expired','unknown','alive','expired','alive']})
    aux={'los':frame.discharge_min.to_numpy(),
         'expired':np.array([1,0,1,0,0,1,0],bool),
         'at_risk_raw':np.ones(len(frame),bool),
         'early_unknown':np.array([0,0,0,1,0,0,0],bool)}
    ok=np.array([1,1,1,0,1,1,1],bool)
    event=np.array([1,0,1,0,0,0,0],bool)
    rows=endpoint_alternatives(frame,'eicu',6,ok,event,aux)
    assert len(rows)==7
    assert [(name,int(mask.sum()),int(y[mask].sum())) for name,mask,y in rows[:3]]==[
        ('stage7e_primary',6,2),('unknown_early_as_negative_sensitivity',7,2),
        ('unknown_early_as_event_sensitivity',7,3)]
    for name,mask,y in rows:
        assert len(mask)==len(frame) and len(y)==len(frame)
    counts=[{'source':'eicu','landmark_h':6,'scenario':n,
             'n':int(m.sum()),'events':int(y[m].sum())} for n,m,y in rows]
    ref=pd.DataFrame(counts)
    assert len(validate_stage8c(counts,ref))==7
    bad=ref.copy();bad.loc[0,'events']+=1
    try:validate_stage8c(counts,bad)
    except RuntimeError:pass
    else:raise AssertionError('Stage8C fail-closed missing')
    main_data=[]
    for mode in ('measured','exclude'):
        for arm in ('lab','integrated'):
            main_data.append(dict(scenario=mode,cohort='external',model=arm,landmark_h=6,
                                  n=6,events=2,auroc=.8,auprc=.4,brier=.1,
                                  mean_predicted_risk=.2,pct_risk_gt_50=.01))
    r=pd.DataFrame(main_data)
    assert len(validate_stage8b(r,r,1e-4))==4
    r2=r.copy();r2.loc[0,'auroc']+=.001
    try:validate_stage8b(r2,r,1e-4)
    except RuntimeError:pass
    else:raise AssertionError('Stage8B fail-closed missing')
    print('STAGE 8D SYNTHETIC SELFTEST PASS — reference gates, sensitivity labels, fail-closed')

if __name__=='__main__':main()
