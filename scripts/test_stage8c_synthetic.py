import tempfile
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
from icu_stage8c_endpoint_adjudication import adjudicate, scenarios, check_reference


class EndpointTests(unittest.TestCase):
    def test_eicu_edges_and_unknown(self):
        d = pd.DataFrame({
            'discharge_min': [359, 360, 361, 1799, 1800, 1801, 800, 800, 1800, np.nan],
            'discharge_status': ['alive','expired','expired','expired','expired','expired','alive','other','other','alive'],
        })
        cats, ok, event, aux = adjudicate(d, 'eicu', 6)
        self.assertEqual(int(ok.sum()), 6)
        self.assertEqual(int(event.sum()), 3)
        self.assertEqual(int(aux['early_unknown'].sum()), 1)
        self.assertEqual(cats[1], 'expired_exact_landmark_not_at_risk')
        self.assertEqual(cats[4], 'event_expired_exact_horizon')
        ss = scenarios(d, 'eicu', 6, ok, event, aux)
        self.assertEqual(ss[0]['events'], 3)
        self.assertEqual(ss[1]['n'], 7)
        self.assertEqual(ss[2]['events'], 4)

    def test_mimic_edges_and_post_icu_death(self):
        start = pd.Timestamp('2020-01-01')
        def t(m): return start + pd.Timedelta(minutes=m)
        d = pd.DataFrame({
            'intime': [start]*8,
            'outtime': [t(x) if x is not None else pd.NaT for x in [359,360,600,1800,2400,1000,2000,None]],
            'death_ts': [pd.NaT,pd.NaT,t(550),t(1800),t(360),t(1500),pd.NaT,pd.NaT],
        })
        cats, ok, event, aux = adjudicate(d, 'mimic', 6)
        self.assertEqual(int(ok.sum()), 4)
        self.assertEqual(int(event.sum()), 2)
        self.assertEqual(cats[5], 'post_icu_death_within_horizon_not_icu_event')
        self.assertEqual(cats[3], 'icu_death_exact_horizon')

    def test_reference_gate(self):
        q = pd.DataFrame([{'source': 'eicu', 'landmark_h': h, 'n':100+h, 'events':h} for h in (6,12,18,24)])
        ref = pd.DataFrame([{'cohort': x, 'landmark_h': h, 'n':100+h if x=='external' else 2,
                             'events':h if x=='external' else 1} for x in ('external','internal') for h in (6,12,18,24)])
        self.assertEqual(len(check_reference(q, ref)), 4)
        ref.loc[ref.cohort.eq('external') & ref.landmark_h.eq(12), 'events'] = 999
        with self.assertRaises(AssertionError): check_reference(q, ref)

if __name__ == '__main__': unittest.main()
