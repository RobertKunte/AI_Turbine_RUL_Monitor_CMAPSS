import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.quantile_calibration import _compute_metrics_from_arrays

class TestQuantileCalibration(unittest.TestCase):
    def test_metrics_computation(self):
        """
        Test that metrics (coverage, interval, overconfidence) are computed correctly
        using synthetic arrays.
        """
        # Create synthetic data
        # 10 samples
        n = 10
        unit_ids = np.arange(n)
        cond_ids = np.array([0]*5 + [1]*5) # Two conditions
        unique_conds = np.array([0, 1])
        
        # y_true: [10, 10, ... 10]
        y_true = np.zeros(n) + 10.0
        
        # Scenario:
        # q50 is perfect (10.0)
        # Interval is width 2.0 (q10=9.0, q90=11.0)
        q10 = np.zeros(n) + 9.0
        q50 = np.zeros(n) + 10.0
        q90 = np.zeros(n) + 11.0
        
        # Modify a few to test coverage
        # Sample 0: true=10, q10=11 (true < q10) -> Covered by q10? No, y <= q10 check: 10 <= 11 True.
        # Wait, q10 is the lower bound. If we want coverage `y <= q`, then for q10 coverage means "y is below the 10th percentile"? 
        # Standard def: q_tau is value s.t. P(Y <= q_tau) = tau.
        # So "coverage of q10" usually means proportion of samples where y_true <= q10. Ideally 0.1.
        
        # Let's adjust to verify calculations:
        
        # Sample 0: y=10, q10=9. y <= q10 (10<=9) False.
        # Sample 0: y=10, q90=11. y <= q90 (10<=11) True.
        
        # Case A: Perfect calibration around 10
        # q10=9, q50=10, q90=11. y=10.
        # y <= q10: False
        # y <= q50: True (equal)
        # y <= q90: True
        
        # Case B: y=10, q10=12 (Way too high predictions). y <= q10 True.
        # Case C: y=10, q90=8 (Way too low predictions). y <= q90 False.
        
        # Let's set up specific values for the 10 samples
        
        # 0-4 (Cond 0): Perfect center
        # y=10, q=[9, 10, 11]
        # cov10=0, cov50=1, cov90=1
        
        # 5-9 (Cond 1): Overconfident logic test
        # Overconfidence def: abs_err >= 25 AND interval <= 10
        # Let's make:
        # Sample 5: y=100, q50=10 (Err=90), q10=9, q90=11 (Int=2). -> Overconfident
        # Sample 6: y=100, q50=10 (Err=90), q10=0, q90=20 (Int=20). -> Not Overconf (Int too big)
        # Sample 7: y=15,  q50=10 (Err=5),  q10=9, q90=11 (Int=2). -> Not Overconf (Err too small)
        
        y_true[5] = 100.0
        q10[5] = 9.0; q50[5] = 10.0; q90[5] = 11.0
        
        y_true[6] = 100.0
        q10[6] = 0.0; q50[6] = 10.0; q90[6] = 20.0
        
        y_true[7] = 15.0
        q10[7] = 9.0; q50[7] = 10.0; q90[7] = 11.0
        
        # Expected metrics:
        
        metrics = _compute_metrics_from_arrays(
            y_true, q10, q50, q90, unit_ids, cond_ids, unique_conds
        )
        
        # Global Checks
        glob = metrics["global"]
        self.assertEqual(glob["n_units"], 10)
        
        # Overconfidence check
        # Only Sample 5 matches both conditions.
        # Sample 5: Err=|100-10|=90>=25. Int=11-9=2<=10. -> True
        # Sample 6: Err=90. Int=20>10. -> False
        # Sample 7: Err=5<25. Int=2. -> False
        # Others (0-4, 8, 9): y=10, q50=10, Err=0. Int=2. -> False
        # Total overconfident: 1/10 = 0.1
        self.assertAlmostEqual(glob["frac_overconfident"], 0.1)
        
        # Check Interval Width
        # 0-4: 2.0
        # 5: 2.0
        # 6: 20.0
        # 7: 2.0
        # 8,9: 2.0
        # Sum = 2*5 + 2 + 20 + 2 + 2*2 = 10 + 2 + 20 + 2 + 4 = 38
        # Mean = 3.8
        self.assertAlmostEqual(glob["mean_interval_q90_q10"], 3.8)
        
        # Per Condition
        per_cond = metrics["per_condition"]
        self.assertIn(0, per_cond)
        self.assertIn(1, per_cond)
        
        # Cond 0 (indices 0-4): All "perfect" center
        # y=10, q=[9, 10, 11]
        c0 = per_cond[0]
        self.assertEqual(c0["n_units"], 5)
        self.assertAlmostEqual(c0["mean_abs_err_last_q50"], 0.0)
        self.assertEqual(c0["frac_overconfident"], 0.0)
        # Cov10: 10 <= 9 False (0.0)
        # Cov50: 10 <= 10 True (1.0)
        # Cov90: 10 <= 11 True (1.0)
        self.assertAlmostEqual(c0["coverage_q10"], 0.0)
        self.assertAlmostEqual(c0["coverage_q50"], 1.0)
        self.assertAlmostEqual(c0["coverage_q90"], 1.0)
        
        # Cond 1 (indices 5-9)
        # 5: Overconf
        # 6: Wide int
        # 7: Small err
        # 8,9: Perfect (copied from initialization)
        
        c1 = per_cond[1]
        self.assertEqual(c1["n_units"], 5)
        # Overconfidence: Only sample 5. 1/5 = 0.2
        self.assertAlmostEqual(c1["frac_overconfident"], 0.2) 

if __name__ == '__main__':
    unittest.main()
