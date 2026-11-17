import csv
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
from metrics import ece, reliability
from train import split_indices


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(ece([0, 0, 1, 1], [0, 0, 1, 1]), 0)

    def test_wrong_extremes(self):
        self.assertEqual(ece([1, 0], [0, 1]), 1)

    def test_weighted_bins(self):
        self.assertAlmostEqual(ece([0, 0, 1, 1], [.1, .1, .1, .9], 2), .2)

    def test_empty_bins_and_boundaries(self):
        rows = reliability([0, 1, 1], [0, .1, 1])
        self.assertEqual(rows[0]['count'], 1)
        self.assertEqual(rows[1]['count'], 1)
        self.assertEqual(rows[9]['count'], 1)
        self.assertIsNone(rows[2]['mean_probability'])
        self.assertEqual(sum(r['count'] for r in rows), 3)

    def test_invalid_inputs(self):
        for y, p in [([], []), ([0], [1.1]), ([2], [.5]), ([0], [float('nan')]), ([0, 1], [.5])]:
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                ece(y, p)

    def test_splits_disjoint_complete(self):
        splits = split_indices(np.tile([0, 1], 100))
        merged = np.concatenate(list(splits.values()))
        self.assertEqual(len(np.unique(merged)), 200)
        self.assertEqual(sorted(merged.tolist()), list(range(200)))
        self.assertEqual([len(v) for v in splits.values()], [100, 40, 30, 30])


@unittest.skipUnless(Path('outputs/manifest.json').exists(), 'Run train.py for artifact integration tests')
class ArtifactTests(unittest.TestCase):
    def test_selection_uses_validation(self):
        with open('outputs/validation_metrics.csv') as file:
            rows = list(csv.DictReader(file))
        expected = min(rows, key=lambda r: float(r['brier']))['model']
        manifest = json.loads(Path('outputs/manifest.json').read_text())
        self.assertEqual(manifest['selected_model'], expected)
        ids = [i for values in manifest['splits'].values() for i in values]
        self.assertEqual(len(set(ids)), 10000)

    def test_probabilities_and_metric_reproduction(self):
        with open('outputs/test_predictions.csv') as file:
            rows = list(csv.DictReader(file))
        with open('outputs/metrics.csv') as file:
            metrics = list(csv.DictReader(file))
        y = [int(r['actual']) for r in rows]
        self.assertEqual(len(metrics), 6)
        self.assertEqual(len(y), 1500)
        for model in metrics:
            p = np.array([float(r[model['model']]) for r in rows])
            self.assertTrue(np.all((p >= 0) & (p <= 1)))
            self.assertAlmostEqual(ece(y, p), float(model['ece_10']))
            self.assertAlmostEqual(np.mean((p - y) ** 2), float(model['brier']))

    def test_saved_scorer_reproduces_predictions(self):
        bundle = joblib.load('outputs/scorer.joblib')
        X = np.loadtxt('outputs/example_features.csv', delimiter=',', skiprows=1)
        with open('outputs/test_predictions.csv') as file:
            rows = list(csv.DictReader(file))[:5]
        expected = [float(r[bundle['name']]) for r in rows]
        np.testing.assert_allclose(bundle['model'].predict_proba(X)[:, 1], expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
