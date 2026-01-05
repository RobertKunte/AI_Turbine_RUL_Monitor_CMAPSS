import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.failure_case_analysis import select_groups_from_last_error

class TestFailureSelection(unittest.TestCase):
    def test_selection_logic(self):
        # Create dummy data: 100 units
        # Unit i has abs_err = i
        n = 100
        data = []
        for i in range(n):
            data.append({
                "unit_id": i,
                "abs_err_last": float(i),
                "signed_err_last": float(i)
            })
        
        df = pd.DataFrame(data)
        # Sort descending as required by function contract
        df = df.sort_values("abs_err_last", ascending=False)
        
        K = 10
        worst, best, mid = select_groups_from_last_error(df, K)
        
        # Worst 10 should be 99 down to 90
        # Expected: [99, 98, ..., 90]
        # (Though unit_id might be sorted ascending if error is tie, here error is unique)
        self.assertEqual(len(worst), K)
        self.assertEqual(worst[0], 99)
        self.assertEqual(worst[-1], 90)
        
        # Best 10 should be 0 to 9 (smallest error)
        # Expected: [9, 8, ..., 0] or similar depending on slicing reverse
        # The logic is: best_20 = all_units[-K:]
        # df is sorted DESC error. So end of list is lowest error.
        # unit 0 is at end. unit 1 is before it.
        # So last K will be [9, 8, ..., 0] reversed? No, the list is [99, ..., 0]
        # So last 10 is [9, 8, 7, ..., 0]
        # Wait, if list is [99, 98, ..., 0]
        # last 10 is [9, 8, ..., 0]
        self.assertEqual(len(best), K)
        self.assertIn(0, best) # Lowest error
        self.assertIn(9, best) # 10th lowest
        self.assertNotIn(10, best)
        
        # Mid 10
        # Median of 100 is 50.
        # Start should be around 50 - 5 = 45.
        # End 55.
        # Values should be around 50.
        self.assertEqual(len(mid), K)
        # Median value in [0..99] is 49.5
        # We expect a slice around the middle of the sorted array
        # Sorted array indices: 0 (val 99) ... 49 (val 50) ... 99 (val 0)
        # Middle index 50 corresponds to val 49.
        # Slice [45:55]
        # Values: 54, 53, ..., 45
        sample_mid = mid[0]
        self.assertTrue(40 <= sample_mid <= 60, f"Mid value {sample_mid} not in expected range")

if __name__ == '__main__':
    unittest.main()
