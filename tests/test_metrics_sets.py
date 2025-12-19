import unittest
import numpy as np

from src.metrics import compute_last_per_unit_metrics, compute_all_samples_metrics


class TestMetricsSets(unittest.TestCase):
    def test_last_per_unit_uses_stable_last_occurrence(self):
        # unit 1 appears last at idx=3, unit 2 at idx=4
        unit_ids = np.array([1, 1, 2, 1, 2], dtype=np.int64)
        y_true = np.array([10.0, 9.0, 8.0, 7.0, 6.0], dtype=float)
        y_pred = np.array([10.0, 9.0, 0.0, 5.0, 9.0], dtype=float)

        m = compute_last_per_unit_metrics(unit_ids, y_true, y_pred, clip=None)
        self.assertEqual(m["n_units"], 2)

        # last points are (unit1: true=7 pred=5) and (unit2: true=6 pred=9)
        errors = np.array([5.0 - 7.0, 9.0 - 6.0], dtype=float)
        rmse_expected = float(np.sqrt(np.mean(errors ** 2)))
        mae_expected = float(np.mean(np.abs(errors)))
        bias_expected = float(np.mean(errors))

        self.assertAlmostEqual(m["rmse_last"], rmse_expected, places=10)
        self.assertAlmostEqual(m["mae_last"], mae_expected, places=10)
        self.assertAlmostEqual(m["bias_last"], bias_expected, places=10)
        self.assertIn("nasa_last_sum", m)
        self.assertIn("nasa_last_mean", m)
        self.assertEqual(m["note_last_definition"], "LAST_AVAILABLE_PER_UNIT (truncated-aware)")

    def test_all_samples_metrics_counts_and_clips(self):
        y_true = np.array([0.0, 5.0, 20.0], dtype=float)
        y_pred = np.array([-5.0, 6.0, 1000.0], dtype=float)
        unit_ids = np.array([1, 1, 2], dtype=np.int64)

        m = compute_all_samples_metrics(y_true, y_pred, unit_ids=unit_ids, clip=(0.0, 10.0))
        self.assertEqual(m["n_samples_all"], 3)
        self.assertEqual(m["n_units"], 2)
        self.assertEqual(m["max_rul_used"], 10.0)

        # After clipping: y_true=[0,5,10], y_pred=[0,6,10]
        yt = np.array([0.0, 5.0, 10.0], dtype=float)
        yp = np.array([0.0, 6.0, 10.0], dtype=float)
        errors = yp - yt
        rmse_expected = float(np.sqrt(np.mean(errors ** 2)))
        self.assertAlmostEqual(m["rmse_all"], rmse_expected, places=10)
        self.assertIn("nasa_all_sum", m)
        self.assertIn("nasa_all_mean", m)


if __name__ == "__main__":
    unittest.main()

