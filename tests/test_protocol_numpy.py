"""Data-free numerical checks; runnable without a deep-learning installation."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libs.calibration import select_calibration, load_calibration
from libs.survival_metrics import summarize_survival_query, compute_survival_metrics, _binary_auc
from tools.augment_validation import augment_validation
from tools.prepare_hi_annotations import validate_official_annotations


class ProtocolTests(unittest.TestCase):
    def prediction(self, end=None):
        h = np.full(5, .2)
        return dict(mask=np.ones(5, dtype=bool), start_hazard=h, end_hazard=h if end is None else end,
                    start_survival=np.cumprod(1-h),end_survival=np.cumprod(1-h),
                    cure=np.full(5,.1),state=np.full((5,3),1/3),boundary_objective='hazard')

    def test_joint_nll_excludes_cure(self):
        s = summarize_survival_query(self.prediction(),[1.,3.],True)
        result = compute_survival_metrics([dict(survival_summary=s)])
        self.assertAlmostEqual(result['survival_nll'], s['start_nll']+s['end_nll'])
        self.assertGreater(s['cure_nll'],0)

    def test_end_survival_ignores_pre_start_hazard(self):
        a = self.prediction(np.array([.01,.01,.2,.2,.2]))
        b = self.prediction(np.array([.99,.99,.2,.2,.2]))
        x = summarize_survival_query(a,[2.,4.],True)
        y = summarize_survival_query(b,[2.,4.],True)
        self.assertAlmostEqual(x['end_brier'],y['end_brier'])
        self.assertAlmostEqual(x['end_nll'],y['end_nll'])

    def test_pre_event_has_no_end_risk(self):
        s = summarize_survival_query(self.prediction(),[8.,10.],True)
        self.assertIsNone(s['end_nll'])
        self.assertIsNone(s['end_brier'])

    def test_irrelevant_excludes_end_and_state(self):
        s = summarize_survival_query(self.prediction(),[-1.,-1.],False)
        self.assertIsNone(s['end_nll'])
        self.assertEqual(s['state_count'],0)
        self.assertEqual(s['cure_label'],1)

    def test_left_censored_start_and_ended_crop(self):
        s = summarize_survival_query(self.prediction(),[-2.,3.],True)
        self.assertIsNone(s['start_nll'])
        self.assertIsNotNone(s['end_nll'])
        s = summarize_survival_query(self.prediction(),[-3.,-1.],True)
        self.assertIsNone(s['end_nll'])

    def test_auc_ties_and_undefined(self):
        self.assertEqual(_binary_auc([1,0],[.5,.5]),.5)
        self.assertEqual(_binary_auc([1,0],[.8,.2]),1.)
        self.assertTrue(math.isnan(_binary_auc([1,1],[.8,.2])))

    def test_hi_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            select_calibration(dict(benchmark='hi_activitynet',split='hi_test'))

    def test_calibration_roundtrip_and_variant_guard(self):
        payload=dict(benchmark='standard',split='val_1',model_variant='survvtg',predictions=[
            dict(cure_probability=.1,is_relevant=True),dict(cure_probability=.8,is_relevant=False)])
        selection=select_calibration(payload)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cal.json';path.write_text(json.dumps(selection))
            self.assertEqual(load_calibration(path,'survvtg')['threshold'],.8)
            with self.assertRaises(ValueError):load_calibration(path,'direct_mt')

    def test_generated_negatives_are_not_official_benchmark(self):
        source=dict(val_1={f'v{i}':dict(fps=1,num_frames=10,num_clips=10,annotations=[
            dict(sentence=f'person opens door {i}',segment=[1,3])]) for i in range(2)})
        result=augment_validation(source)
        self.assertEqual(result['benchmark'],'ActivityNet-validation-augmentation')
        self.assertTrue(all(any(not p.get('is_relevant', True) for p in v['annotations']) for v in result['val_1'].values()))
        with self.assertRaises(ValueError):validate_official_annotations(result)

    def test_empty_metrics_are_undefined(self):
        self.assertTrue(math.isnan(compute_survival_metrics([])['survival_nll']))


if __name__ == '__main__':
    unittest.main()
