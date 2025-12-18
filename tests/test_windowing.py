import unittest
import numpy as np
import pandas as pd

from src.data.windowing import WindowConfig, TargetConfig, build_sliding_windows


class TestWindowing(unittest.TestCase):
    def test_end_padding_clamp_and_eol_modes(self):
        # Mini engine with 5 cycles, strictly decreasing RUL to 0
        df = pd.DataFrame(
            {
                "UnitNumber": [1, 1, 1, 1, 1],
                "TimeInCycles": [1, 2, 3, 4, 5],
                "ConditionID": [0, 0, 0, 0, 0],
                "f1": [10, 11, 12, 13, 14],
                "RUL": [4, 3, 2, 1, 0],
            }
        )

        win = WindowConfig(past_len=3, horizon=4, stride=1, require_full_horizon=False, pad_mode="clamp")
        tgt = TargetConfig(max_rul=125, cap_targets=True, eol_target_mode="future0", clip_eval_y_true=False)
        out = build_sliding_windows(
            df,
            feature_cols=["f1"],
            target_col="RUL",
            unit_col="UnitNumber",
            time_col="TimeInCycles",
            cond_col="ConditionID",
            window_cfg=win,
            target_cfg=tgt,
            return_mask=True,
        )

        # We should get windows for t_end positions 2,3,4 (0-based) => 3 samples
        self.assertEqual(out["X"].shape, (3, 3, 1))
        self.assertEqual(out["Y_seq"].shape, (3, 4, 1))
        self.assertEqual(out["mask"].shape, (3, 4, 1))

        # For t_end=2 (cycle 3), future RUL is [1,0] then clamp-pad with last (0)
        y0 = out["Y_seq"][0, :, 0]
        m0 = out["mask"][0, :, 0]
        np.testing.assert_allclose(y0, np.array([1, 0, 0, 0], dtype=np.float32))
        np.testing.assert_allclose(m0, np.array([1, 1, 0, 0], dtype=np.float32))

        # eol_target_mode=future0 => scalar should be first future step
        np.testing.assert_allclose(out["Y_eol"], np.array([1, 0, 0], dtype=np.float32))

        # Check current_from_df mode matches current RUL at window end
        tgt2 = TargetConfig(max_rul=125, cap_targets=True, eol_target_mode="current_from_df", clip_eval_y_true=False)
        out2 = build_sliding_windows(
            df,
            feature_cols=["f1"],
            target_col="RUL",
            unit_col="UnitNumber",
            time_col="TimeInCycles",
            cond_col="ConditionID",
            window_cfg=win,
            target_cfg=tgt2,
            return_mask=True,
        )
        # current RUL at t_end=2,3,4 is [2,1,0]
        np.testing.assert_allclose(out2["Y_eol"], np.array([2, 1, 0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

