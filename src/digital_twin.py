from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class HealthyTwinRegressor:
    """
    Global 'digital twin light' that predicts healthy sensor values
    from a continuous condition vector using a simple linear model.
    """

    def __init__(self) -> None:
        self.model: LinearRegression | None = None
        self.sensor_cols: List[str] = []
        self.condition_cols: List[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        *,
        unit_col: str,
        cycle_col: str,
        condition_cols: List[str],
        sensor_cols: List[str],
        baseline_len: int = 30,
    ) -> "HealthyTwinRegressor":
        """
        Fit a multi-output linear regression using early cycles of all engines
        as 'healthy' reference data.

        - For each unit, use the first `baseline_len` cycles.
        - Stack all units to form a global healthy dataset.
        """
        self.condition_cols = list(condition_cols)
        self.sensor_cols = list(sensor_cols)

        # Sort and select early cycles per engine
        df_sorted = df.sort_values([unit_col, cycle_col])
        ranks = df_sorted.groupby(unit_col)[cycle_col].rank(method="first")
        mask_healthy = ranks.le(baseline_len)
        df_healthy = df_sorted.loc[mask_healthy].copy()

        if df_healthy.empty:
            raise ValueError(
                "HealthyTwinRegressor.fit: no healthy samples selected. "
                "Check baseline_len, unit_col and cycle_col."
            )

        # Build design matrices
        X = df_healthy[condition_cols].to_numpy(dtype=float)
        y = df_healthy[sensor_cols].to_numpy(dtype=float)

        model = LinearRegression()
        model.fit(X, y)

        self.model = model
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict healthy sensor values for all rows in `df`.
        """
        if self.model is None:
            raise RuntimeError("HealthyTwinRegressor.predict: model is not fitted")
        X = df[self.condition_cols].to_numpy(dtype=float)
        return self.model.predict(X)

    def add_twin_and_residuals(
        self,
        df: pd.DataFrame,
        *,
        prefix_twin: str = "Twin_",
        prefix_resid: str = "Resid_",
    ) -> pd.DataFrame:
        """
        Add twin prediction and residual columns to df:

        - Twin_<sensor>
        - Resid_<sensor>
        """
        twin_pred = self.predict(df)
        if twin_pred.shape[1] != len(self.sensor_cols):
            raise RuntimeError(
                "HealthyTwinRegressor.add_twin_and_residuals: prediction dimension "
                "does not match number of sensor_cols"
            )

        for i, col in enumerate(self.sensor_cols):
            twin_col = f"{prefix_twin}{col}"
            resid_col = f"{prefix_resid}{col}"
            df[twin_col] = twin_pred[:, i]
            df[resid_col] = df[col] - df[twin_col]

        return df


