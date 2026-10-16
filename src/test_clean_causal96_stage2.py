"""Nonlinear financial models must keep causal prediction inputs."""
import unittest

import numpy as np

import clean_causal96_stage2 as stage
from test_clean_causal import synthetic


class NonlinearTests(unittest.TestCase):
    def test_read_guard_rejects_evaluation_answers_and_legacy_models(self):
        read,load=stage.pd.read_csv,stage.joblib.load
        try:
            stage.guard_reads()
            with self.assertRaises(AssertionError):
                stage.pd.read_csv(stage.ROOT/'evaluation/public_truth_user_provided.csv')
            with self.assertRaises(AssertionError):
                stage.joblib.load(stage.ROOT/'models/financial_forward_batch/Target_1_lr.joblib')
        finally:
            stage.pd.read_csv,stage.joblib.load=read,load

    def test_all_model_families_ignore_later_observations_and_targets(self):
        raw=synthetic()
        selected=[stage.specs()[0],stage.specs()[6],stage.specs()[8]]
        for spec in selected:
            with self.subTest(kind=spec['kind']):
                bundle=stage.fit(raw.iloc[:140],spec)
                full=stage.predict_model(bundle,raw)
                short=stage.predict_model(bundle,raw.iloc[:150])
                np.testing.assert_allclose(full[:150],short,rtol=0,atol=1e-12)
                changed=raw.copy();changed[['Target_1','Target_2']]=1
                cols=[c for c in raw if c.startswith(('FI_','SH1_','KE_'))]
                changed.loc[150:,cols]=changed.loc[150:,cols]*7+4
                np.testing.assert_allclose(full[:150],stage.predict_model(bundle,changed)[:150],rtol=0,atol=1e-12)
                self.assertTrue(np.isfinite(full).all())

    def test_immature_labels_do_not_change_fit(self):
        raw=synthetic(140);spec=stage.specs()[0]
        bundle=stage.fit(raw,spec)
        changed=raw.copy();changed.loc[137:,'Target_1']=1-changed.loc[137:,'Target_1']
        other=stage.fit(changed,spec)
        np.testing.assert_allclose(stage.predict_model(bundle,raw),stage.predict_model(other,raw),rtol=0,atol=1e-12)
        self.assertEqual(bundle['fit_last_time'],137)


if __name__=='__main__':
    unittest.main()
