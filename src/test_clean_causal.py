"""Leakage regression checks using synthetic data, without evaluation answers."""
import unittest

import numpy as np
import pandas as pd

import clean_causal as c
from clean_causal_stable import choose


def synthetic(n=180):
    rng = np.random.default_rng(42)
    raw = pd.DataFrame({'Time_step':np.arange(1,n+1), 'Weekday':np.arange(n)%7+1})
    for name in ['FI_Sp500','FI_Nasdaq','FI_Dow','FI_FX_USDKRW','FI_Gold','FI_WTI',
                 'FI_FX_CNYKRW','FI_FX_JPYKRW','FI_Corp3yr','FI_CD91','FI_Guri',
                 'KE_마스크','KE_코로나','SH1_쇼핑1_식품','SH1_쇼핑1_생활건강']:
        raw[name] = rng.normal(size=n).cumsum()+100
    raw['Target_1'] = np.arange(n)%2
    raw['Target_2'] = np.arange(n)%3==0
    return raw


class LeakageTests(unittest.TestCase):
    def test_future_and_label_changes_do_not_change_features(self):
        raw = synthetic()
        x,_ = c.feature_frames(raw)
        changed = raw.copy()
        columns = [s for s in raw if s not in ['Time_step','Weekday',*c.TARGETS]]
        changed.loc[100:,columns] = changed.loc[100:,columns]*20+1000
        changed[c.TARGETS] = 1
        other,_ = c.feature_frames(changed)
        pd.testing.assert_frame_equal(x.iloc[:100],other.iloc[:100])
        pd.testing.assert_frame_equal(x.iloc[:100],c.feature_frames(raw.iloc[:100])[0])

    def test_training_purges_immature_labels_and_future_rows(self):
        raw = synthetic()
        x,groups = c.feature_frames(raw)
        spec = c.specifications('Target_1')[0]
        b = c.fit_one(x,groups,raw.Target_1,spec,120)
        changed = raw.Target_1.copy()
        changed.iloc[117:] = 1-changed.iloc[117:]
        other = c.fit_one(x,groups,changed,spec,120)
        np.testing.assert_allclose(c.predict_one(b,x.iloc[120:]),c.predict_one(other,x.iloc[120:]))
        self.assertEqual(b['fit_last_time'],117)
        self.assertEqual(b['label_maturity_last_time'],120)

    def test_evaluation_answers_cannot_be_opened(self):
        with self.assertRaises(AssertionError):
            c.read_data(c.ROOT/'evaluation/public_truth_user_provided.csv')

    def test_selection_rejects_labels_that_have_not_matured(self):
        raw = synthetic()
        with self.assertRaises(AssertionError):
            c.selection(raw,'Target_1',[(100,80)],{})

    def test_followup_requires_improvement_at_both_window_sizes(self):
        baseline = dict(mean150=.7,mean100=.6,mean=.65)
        unstable = dict(mean150=.8,mean100=.59,mean=.7)
        stable = dict(mean150=.71,mean100=.61,mean=.66)
        self.assertEqual(choose({'baseline':baseline,'unstable':unstable}), 'baseline')
        self.assertEqual(choose({'baseline':baseline,'unstable':unstable,'stable':stable}), 'stable')

    def test_followup_does_not_claim_ties_as_improvement(self):
        baseline = dict(mean150=.7,mean100=.6,mean=.65)
        self.assertEqual(choose({'baseline':baseline,'same':baseline.copy()}),'baseline')


if __name__ == '__main__':
    unittest.main()
