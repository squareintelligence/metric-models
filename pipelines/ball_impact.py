"""Example pipeline: replace feature logic and estimator with your metric."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class Pipeline:
    """Training pipeline with ``compute_features`` + ``train``."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=["number"]).copy()
        return numeric.fillna(0.0)

    def train(self, features: pd.DataFrame) -> RandomForestRegressor:
        if features.shape[1] < 2:
            raise ValueError("Expected at least two numeric columns (features + target).")
        x = features.iloc[:, :-1].to_numpy(dtype=float)
        y = features.iloc[:, -1].to_numpy(dtype=float)
        model = RandomForestRegressor(n_estimators=20, random_state=0, n_jobs=-1)
        model.fit(x, y)
        return model
